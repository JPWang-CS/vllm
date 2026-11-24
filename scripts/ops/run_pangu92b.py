"""Real 92B native deployment with quality-first acceptance."""
import argparse,copy,json,os,re,shutil,subprocess,sys
from pathlib import Path
from fullnet.analysis import report
from fullnet.execution import plan,run_experiment
from fullnet.export import export_run
from fullnet.measurements import collect_performance,collect_profile,pd_mixed_probe,pd_split_probe
from fullnet.store import read_json,rows,write_json


def apply_reference_env_overrides(env, overrides):
    for key,value in overrides.items():
        if key.startswith(('HCCL_','HYBRID_','OMNI_','PYTORCH_NPU_')):
            env[key]=str(value)


def apply_aligned_decode_plan(env, *, disable=False, enable=False):
    if disable and enable:
        raise ValueError('aligned decode cannot be both enabled and disabled')
    key='OMNI_HYBRID_ALIGNED_DECODE'
    # Do not inherit a benchmark-only setting from the invoking shell or reference.
    env.pop(key,None)
    if disable:env[key]='0'
    elif enable:env[key]='1'
    return env.get(key)


def assert_aligned_decode_plan(env, frozen_env, expected):
    key='OMNI_HYBRID_ALIGNED_DECODE'
    if env.get(key)!=expected or frozen_env.get(key)!=expected:
        raise AssertionError('aligned decode environment differs from the selected experiment plan')


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',required=True);p.add_argument('--npu-ids',type=int,nargs='+',required=True)
    p.add_argument('--tp',type=int,default=2);p.add_argument('--dp',type=int,default=1);p.add_argument('--pp',type=int,default=1)
    p.add_argument('--graph',choices=('eager','full_decode_only'),default='eager');p.add_argument('--mtp',type=int,choices=(0,3),default=0)
    p.add_argument('--pd-mode',choices=('','mixed','split'),default='')
    p.add_argument('--task',required=True);p.add_argument('--port',type=int,default=18322)
    p.add_argument('--dataset',type=Path);p.add_argument('--keep-ready',action='store_true')
    p.add_argument('--quality-only',action='store_true');p.add_argument('--requests',type=int,default=32)
    p.add_argument('--max-batched-tokens',type=int,default=4096)
    p.add_argument('--max-seqs',type=int,default=8)
    p.add_argument('--env-override-file',type=Path)
    aligned=p.add_mutually_exclusive_group()
    aligned.add_argument('--disable-aligned-decode',action='store_true')
    aligned.add_argument('--enable-aligned-decode',action='store_true')
    a=p.parse_args();root=a.project.resolve(strict=True);task=(root/'runs'/a.task).resolve(strict=True)
    split=a.pd_mode=='split'
    workers_needed=a.tp*a.dp*a.pp*(2 if split else 1)
    if not task.is_relative_to(root) or len(a.devices)!=workers_needed or len(a.npu_ids)!=len(a.devices):raise ValueError('project or worker count mismatch')
    if a.max_batched_tokens<1 or a.max_seqs<1:raise ValueError('max batched tokens and max seqs must be positive')
    ref=read_json(a.reference);b=ref['bootstrap'];python=b['python'];weights=a.weights.resolve(strict=True)
    hf=read_json(weights/'config.json')
    if hf.get('architectures')!=['PanguUltraMoEForCausalLM'] or hf.get('num_hidden_layers')!=46 or hf.get('hidden_size')!=2560:
        raise ValueError('expected actual Pangu Ultra MoE 92B checkpoint')
    index=read_json(weights/'model.safetensors.index.json');shards=set(index['weight_map'].values())
    if not shards or any(not (weights/v).is_file() for v in shards):raise ValueError('missing 92B weight shards')
    def guard(label):
        result=subprocess.run([str(root/'tools/npu-smi'),'info'],capture_output=True,text=True,timeout=35)
        (task/(label+'.txt')).write_text(result.stdout+result.stderr)
        for i in a.npu_ids:
            if result.returncode or not re.search(r'\|\s*'+str(i)+r'\s*\|\s*Ascend\w*\s*\|\s*OK\b',result.stdout) or 'No running processes found in NPU '+str(i)+' ' not in result.stdout:
                raise RuntimeError('selected device is not confirmed healthy and empty: '+str(i))
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env['PYTHONPATH']=os.pathsep.join([str(root/'src'),str(root/'envs/evaluation-deps'),*b['source_roots'],str(Path(b['cann_prefix'])/'python/site-packages')])
    vendors=b['vendor_roots'];libraries=[str(Path(v)/'op_api/lib/libcust_opapi.so') for v in vendors]
    if any(not Path(v).is_file() for v in libraries):raise ValueError('baseline model operator library missing')
    env.update(ASCEND_CUSTOM_OPP_PATH=os.pathsep.join(vendors),FULLNET_BASELINE_VENDOR_LIBS=json.dumps(libraries),
        LD_LIBRARY_PATH=os.pathsep.join([b['native_root'],*[str(Path(v)/'op_api/lib') for v in vendors],*b.get('driver_library_roots',[]),env.get('LD_LIBRARY_PATH','')]),
        VLLM_PLUGINS='omni-npu,omni_custom_models,omni_npu_patches',VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        OMNI_NPU_VLLM_PATCHES='ALL',OMNI_NPU_PATCHES_DIR='pangu_v2_benchmark,pangu_v2_base,pangu_sink_swa_mla,pangu_v2_hybrid',
        ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,a.devices)),ASCEND_VISIBLE_DEVICES=','.join(map(str,a.devices)),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_PROXY='localhost,127.0.0.1,::1',GLOO_SOCKET_IFNAME='lo')
    apply_reference_env_overrides(env,ref.get('env_overrides',{}))
    if a.env_override_file:apply_reference_env_overrides(env,read_json(a.env_override_file))
    aligned_decode_env=apply_aligned_decode_plan(env,disable=a.disable_aligned_decode,enable=a.enable_aligned_decode)
    aligned_decode_effective=('disabled' if aligned_decode_env=='0' else
                              'enabled' if aligned_decode_env=='1' else 'runtime_default')
    iface=b.get('network_interface','lo')
    env['HCCL_SOCKET_IFNAME']=iface if (Path('/sys/class/net')/iface).exists() else 'lo'
    extra=copy.deepcopy(read_json(Path(b['model_extra_source'])))
    extra.setdefault('model_parallel_config',{})['enable_aicpu_dp_sync']=a.dp>1
    extra_path=task/'model-extra.json';write_json(extra_path,extra)
    env['OMNI_CUSTOM_MODEL_CONFIG_PATH']=str(extra_path)
    for key,suffix in {'HOME':'home','TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf',
          'HF_MODULES_CACHE':'cache/hf-modules','ASCEND_PROCESS_LOG_PATH':'logs/ascend','ASCEND_WORK_PATH':'tmp/ascend'}.items():
        path=task/suffix;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    if env.get('FULLNET_CA_BUNDLE'):
        env['SSL_CERT_FILE']=env['FULLNET_CA_BUNDLE'];env['REQUESTS_CA_BUNDLE']=env['FULLNET_CA_BUNDLE']
    def call(label,args,timeout=900):
        with (task/(label+'.log')).open('w') as log:r=subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        write_json(task/(label+'-exit.json'),{'argv':args,'exit':r.returncode})
        if r.returncode:raise RuntimeError(label+' failed')
    status={'model':'pangu-92b','model_requests':'not_run','profile':'not_run','performance':'not_run','quality_only':a.quality_only,
            'stage':'device-pre','pd_mode':a.pd_mode or 'off',
            'pd_mixed_probe':'not_run','max_batched_tokens':a.max_batched_tokens,'max_seqs':a.max_seqs,
            'aligned_decode':aligned_decode_effective,'aligned_decode_env':aligned_decode_env}
    run=None;owns_run=False;exit_code=0;failure=None
    try:
        guard('device-pre')
        status['stage']='evaluation-data'
        ruler=root/'cache/RULER'
        call('ruler-source',[sys.executable,'-B',str(root/'scripts/evaluation/get_ruler.py'),'--output',str(ruler)])
        if a.dataset:dataset=a.dataset.resolve(strict=True)
        else:
            data=task/'data'
            call('ruler-data',[python,'-B',str(root/'scripts/evaluation/prepare_ruler_subset.py'),'--ruler-root',str(ruler),
                '--model',str(weights),'--output',str(data),'--lengths','4096','8192','16384','--samples','8','--seed','42',
                '--model-template-token','64','--chat-template-kwargs','{"thinking":false}'])
            dataset=data/'all.jsonl';dataset.write_text(''.join(v.read_text() for v in sorted(data.glob('*-*.jsonl'))))
        if len(rows(dataset))!=48:raise ValueError('48 fixed samples required')
        args=[python,'-B',str(root/'models/pangu-92b/adapters/serve.py'),'--model',str(weights),'--served-model-name','pangu-92b',
              '--trust-remote-code','--host','127.0.0.1','--port',str(a.port),'--dtype','bfloat16','--max-model-len','32768',
              '--tensor-parallel-size',str(a.tp),'--data-parallel-size',str(a.dp),'--pipeline-parallel-size',str(a.pp),
              '--enable-expert-parallel','--distributed-executor-backend','mp','--max-num-batched-tokens',str(a.max_batched_tokens),
              '--max-num-seqs',str(a.max_seqs),'--gpu-memory-utilization','0.8','--block-size','128','--kv-cache-dtype','hif8_ds_mla',
              '--load-format','safetensors','--enable-chunked-prefill','--no-disable-hybrid-kv-cache-manager','--seed','0']
        if a.graph=='eager':args+=['--enforce-eager']
        else:args+=['--compilation-config',json.dumps({'level':3,'backend':'eager','cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[32]})]
        if a.mtp:args+=['--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':a.mtp})]
        if split:
            producer_devices=','.join(map(str,a.devices[:a.tp*a.dp]));consumer_devices=','.join(map(str,a.devices[a.tp*a.dp:]))
            args=[python,'-B',str(root/'models/pangu-92b/adapters/serve_pd_split.py'),
                '--serve-adapter',str(root/'models/pangu-92b/adapters/serve.py'),'--model',str(weights),
                '--proxy-script',str(root/'models/pangu-92b/adapters/pd_relay_proxy.py'),
                '--producer-devices',producer_devices,'--consumer-devices',consumer_devices,
                '--producer-port',str(a.port+1),'--consumer-port',str(a.port+2),
                '--proxy-port',str(a.port+3),'--front-port',str(a.port),
                '--kv-port-base','7100','--gpu-share','0.8','--ready-timeout','3600',
                '--max-batched-tokens',str(a.max_batched_tokens),'--max-seqs',str(a.max_seqs),
                # pd_split_probe reads pd-split-<role>.log from the experiment
                # run dir; plan() creates this directory before the service starts.
                '--logs',str(root/'models/pangu-92b/experiments'/a.task/'logs')]
        target=root/'models/pangu-92b/experiments'/a.task
        if not a.quality_only:
            env.update(VLLM_TORCH_PROFILER_DIR=str(target/'profiles'),VLLM_TORCH_PROFILER_RECORD_SHAPES='1',
                       VLLM_TORCH_PROFILER_WITH_STACK='0',VLLM_TORCH_PROFILER_WITH_FLOPS='0',VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY='0')
        else:
            for key in ('VLLM_TORCH_PROFILER_DIR','VLLM_TORCH_PROFILER_RECORD_SHAPES',
                        'VLLM_TORCH_PROFILER_WITH_STACK','VLLM_TORCH_PROFILER_WITH_FLOPS',
                        'VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY'):
                env.pop(key,None)
        selected={k:v for k,v in env.items() if k.startswith(('VLLM_','OMNI_','HCCL_','HYBRID_','PYTORCH_NPU_','ASCEND_')) or k in
                  ('PYTHONPATH','LD_LIBRARY_PATH','FULLNET_BASELINE_VENDOR_LIBS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TORCH_DEVICE_BACKEND_AUTOLOAD','GLOO_SOCKET_IFNAME')}
        c={'schema_version':1,'model':'pangu-92b','platform':'a5','devices':a.devices,'changes':[],'resolved_changes':[],
           'identity':{'architecture':hf['architectures'][0],'weights':str(weights),'quantization':'mxfp8','dtype':'bfloat16',
             'kv_cache_dtype':'hif8_ds_mla','tp':a.tp,'dp':a.dp,'pp':a.pp,'ep':a.tp*a.dp,'graph':a.graph,'mtp':a.mtp,'npu_ids':a.npu_ids,
             'max_batched_tokens':a.max_batched_tokens,'max_seqs':a.max_seqs,'pd_mode':a.pd_mode or 'off',
             'aligned_decode':aligned_decode_effective,'aligned_decode_env':aligned_decode_env},
           'env':selected,'service':{'argv':args,'base_url':f'http://127.0.0.1:{a.port}/v1','served_model':'pangu-92b',
              # A degraded host can spend several minutes per shard loading;
              # the longer readiness allowance does not change frozen identity,
              # protocol or workload fields.
              'ready_timeout':3600,'request_timeout':300,'shutdown_timeout':45,
              'smoke':{'enabled':True,'prompt':'Reply with OK.','max_tokens':8,'timeout':60}},
           'evaluation':{'dataset':str(dataset),'scope':'RULER two-task subset, 48 samples','metric':'ruler_official',
              'official_metrics':str(ruler/'scripts/eval/synthetic/constants.py'),'context_limit':32768-a.mtp*3,
              'seed':42,'temperature':0,'top_p':1,'chat_template_kwargs':{'thinking':False},'add_special_tokens':False,'minimum_score':None},
           'measurements':{'requests':a.requests,'warmups':2,
              'deferred_stages':['profile','performance'] if a.quality_only else [],
              # Orchestration ceiling only: single-engine decode (~5 tok/s observed) makes the
              # 1024/4096-output groups run 2-4h and leaves headroom if c4 aggregate decode
              # lands low; the previous 3600s default killed case-03 mid-run in 06-01.
              # Request counts, lengths and concurrency budgets stay frozen.
              'timeout_per_group':21600,'workloads':[
              {'input':128,'output':128,'concurrency':1},{'input':4096,'output':128,'concurrency':1},
              {'input':15999,'output':128,'concurrency':1},{'input':128,'output':1024,'concurrency':1},
              {'input':4096,'output':1024,'concurrency':4},{'input':128,'output':4096,'concurrency':4}]}}
        status['stage']='device-before-launch'
        guard('device-before-launch')
        assert_aligned_decode_plan(env,c['env'],aligned_decode_env)
        os.environ.update(env);run=plan(root,c,a.task);owns_run=True
        frozen=read_json(run/'config.json')
        assert_aligned_decode_plan(env,frozen['env'],aligned_decode_env)
        if frozen['identity'].get('aligned_decode_env')!=aligned_decode_env:
            raise AssertionError('frozen aligned decode identity differs from effective environment')
        status['stage']='model-startup-and-evaluation'
        def finished(path,config,result):
            status.update(model_requests='completed',score=result['score'],quality_status=result['quality_status'],protocol_pass=result['protocol_pass'],
                          attempted=result['attempted'],succeeded=result['succeeded'],failed=result['failed'],skipped=result['skipped'],timeout=result['timeout'])
            write_json(task/'validation.json',status)
            if a.pd_mode=='mixed':
                # P/D is a functional gate even when profile/performance are deferred.
                status['stage']='pd-mixed-probe'
                pd_mixed_probe(path,config)
                status['pd_mixed_probe']='passed'
                write_json(task/'validation.json',status)
            if a.pd_mode=='split':
                # Split P/D must prove real KV transfer before later stages.
                status['stage']='pd-split-probe'
                pd_split_probe(path,config)
                status['pd_split_probe']='passed'
                write_json(task/'validation.json',status)
            if not a.quality_only:
                # Device profile first: a real trace must exist before any multi-hour
                # decode group can fail; the six-group contract itself is unchanged.
                status['stage']='device-profile'
                collect_profile(path,config)
                status['profile']='passed' if read_json(path/'state.json')['phases'].get('profile',{}).get('status')=='passed' else 'failed'
                write_json(task/'validation.json',status)
                status['stage']='measurements'
                collect_performance(path,config)
            else:
                status['profile']='deferred_by_user'
                status['performance']='deferred_by_user'
            status['stage']='evaluated'
            report(path);write_json(task/'validation.json',status)
        result=run_experiment(root,'pangu-92b',a.task,after_evaluation=finished,keep_ready=a.keep_ready)
        if not result['protocol_pass']:raise RuntimeError('quality request protocol failed')
    except Exception as exc:
        exit_code=1;failure=exc;status.update(error=type(exc).__name__,detail=str(exc),exit_code=1)
    finally:
        if owns_run and run:
            try:
                log=run/'logs/service.log'
                if log.exists():(task/'service.log').write_text(log.read_text(errors='replace'))
                progress=run/'evaluation-progress.json'
                if progress.exists():
                    counts=read_json(progress);attempted=counts.get('attempted',0);total=counts.get('total',48)
                    status.update(attempted=attempted,succeeded=counts.get('succeeded',0),failed=counts.get('failed',0),
                                  skipped=counts.get('skipped',0),timeout=counts.get('timeout',0),total=total,
                                  model_requests='completed' if attempted==total else ('partial' if attempted else 'not_run'))
                phases=read_json(run/'state.json').get('phases',{})
                profile_status=phases.get('profile',{}).get('status')
                if profile_status in ('passed','failed'):status['profile']=profile_status
                try:
                    r=subprocess.run([str(root/'tools/npu-smi'),'info'],capture_output=True,text=True,timeout=35)
                    (task/'device-post.txt').write_text(r.stdout+r.stderr)
                    status['device_empty_after_exit']=r.returncode==0 and all('No running processes found in NPU '+str(i)+' ' in r.stdout for i in a.npu_ids)
                except Exception as exc:status['device_post_error']=type(exc).__name__+': '+str(exc)
            except Exception as exc:
                status['finalization_error']=type(exc).__name__+': '+str(exc)
                exit_code=1
        status['exit_code']=exit_code
        def render_report():
            lines=['# Pangu 92B task report','',f"- Exit code: {status.get('exit_code')}",f"- Stage: {status.get('stage')}",
                   f"- Model requests: {status.get('model_requests')}",
                   '- Attempted/succeeded/failed/skipped/timeout: '+
                   '/'.join(str(status.get(k,'unknown')) for k in ('attempted','succeeded','failed','skipped','timeout')),
                   f"- Profile: {status.get('profile')}",f"- Failure: {status.get('error','none')} {status.get('detail','')}",
                   f"- Finalization error: {status.get('finalization_error','none')}", '']
            (task/'task-report.md').write_text('\n'.join(lines),encoding='utf-8')
            return '\n'.join(lines)
        write_json(task/'validation.json',status)
        task_report=render_report()
        if owns_run and run:
            try:report(run)
            except Exception as exc:status['analysis_report_error']=type(exc).__name__+': '+str(exc)
            try:
                write_json(run/'validation.json',status)
                (run/'final-report.md').write_text(task_report,encoding='utf-8')
                export_run(root,run,task/'export')
                status['exported']=True
            except Exception as exc:
                status['exported']=False;status['export_error']=type(exc).__name__+': '+str(exc)
                exit_code=1;status['exit_code']=exit_code
        else:
            try:
                destination=task/'export';destination.mkdir(mode=0o700)
                for name in ('validation.json','task-report.md','device-pre.txt','device-before-launch.txt','device-post.txt','service.log'):
                    source=task/name
                    if source.is_file():shutil.copy2(source,destination/name)
                status['exported']=True
            except Exception as exc:
                status['exported']=False;status['export_error']=type(exc).__name__+': '+str(exc)
                exit_code=1;status['exit_code']=exit_code
            write_json(task/'validation.json',status)
            task_report=render_report()
            if owns_run and run:
                try:
                    write_json(run/'validation.json',status)
                    (run/'final-report.md').write_text(task_report,encoding='utf-8')
                    export_validation=task/'export'/'validation.json'
                    export_report=task/'export'/'final-report.md'
                    if status.get('exported') and export_validation.parent.is_dir():
                        write_json(export_validation,status)
                        export_report.write_text(task_report,encoding='utf-8')
                except Exception as exc:
                    status['final_status_sync_error']=type(exc).__name__+': '+str(exc)
                    status['exit_code']=1;exit_code=1;write_json(task/'validation.json',status)
            elif status.get('exported'):
                try:
                    write_json(task/'export'/'validation.json',status)
                    (task/'export'/'task-report.md').write_text(task_report,encoding='utf-8')
                except Exception as exc:
                    status['final_status_sync_error']=type(exc).__name__+': '+str(exc)
                    status['exit_code']=1;exit_code=1;write_json(task/'validation.json',status)
            if status.get('exported') is False:
                (task/'export-error.txt').write_text(status['export_error']+'\n',encoding='utf-8')
        if failure is not None:print(f"FAILED: {type(failure).__name__}: {failure}",file=sys.stderr)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
