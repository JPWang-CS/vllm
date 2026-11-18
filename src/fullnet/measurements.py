"""Official service benchmark and server-side NPU profiling after quality requests."""
import csv,json,os,subprocess,urllib.request
from pathlib import Path
from .execution import private_environment
from .store import state,write_json


def _trace_device_kernel_events(path):
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError,UnicodeError):
        return 0
    events=data.get('traceEvents',data.get('events',[])) if isinstance(data,dict) else data
    if not isinstance(events,list):return 0
    process_names={}
    for event in events:
        if isinstance(event,dict) and event.get('name') in ('process_name','process_labels'):
            args=event.get('args') or {};process_names[event.get('pid')]=str(args.get('name',''))
    count=0
    for event in events:
        if not isinstance(event,dict) or not event.get('name'):continue
        cat=str(event.get('cat','')).lower();name=str(event.get('name','')).lower()
        args=event.get('args') or {};process=str(process_names.get(event.get('pid'),'' )).lower()
        if 'cpu' in cat:continue
        try:duration=float(event.get('dur',event.get('duration',0)))
        except (TypeError,ValueError):continue
        if duration<=0:continue
        task_values=' '.join(str(v).upper() for k,v in args.items() if 'task' in str(k).lower() or 'type' in str(k).lower())
        task_device=any(tag in task_values for tag in ('AI_CORE','AICORE','AIVEC','AI_VEC','AI_CUBE','AICUBE'))
        keys={str(k).strip().lower().replace(' ','_') for k in args}
        device_id=any(k in keys for k in ('device_id','deviceid','device'))
        process_device=any(tag in process for tag in ('ascend hardware','ascend device','device'))
        structural_device=task_device or device_id or process_device
        if not structural_device:continue
        kernel='kernel' in cat or 'kernel' in name or any(tag in cat for tag in ('device','npu','ascend hardware','aicore','aiv'))
        if kernel:count+=1
    return count


def _csv_device_kernel_rows(path):
    count=0
    try:
        with path.open('r',encoding='utf-8-sig',newline='',errors='replace') as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames:return 0
            fields={str(k).strip().lower().replace(' ','_'):k for k in reader.fieldnames}
            kernel_key=next((v for k,v in fields.items() if 'kernel' in k or k in ('op_name','task_name')),None)
            duration_key=next((v for k,v in fields.items() if 'duration' in k or 'execution_time' in k),None)
            task_key=next((v for k,v in fields.items() if 'task_type' in k or 'tasktype' in k),None)
            device_key=next((v for k,v in fields.items() if k in ('device_id','deviceid','device')),None)
            if not kernel_key or not duration_key or not (task_key or device_key):return 0
            for row in reader:
                kernel=str(row.get(kernel_key,'')).strip()
                if not kernel or 'cpu' in kernel.lower():continue
                task=str(row.get(task_key,'')).upper() if task_key else ''
                device=str(row.get(device_key,'')).strip() if device_key else ''
                if not (any(tag in task for tag in ('AI_CORE','AICORE','AIVEC','AI_VEC','AI_CUBE','AICUBE')) or device):continue
                try:duration=float(str(row.get(duration_key,'')).replace(',',''))
                except (TypeError,ValueError):continue
                if duration>0:count+=1
    except OSError:
        return 0
    return count


def profile_artifact_index(run):
    root=Path(run);profiles=root/'profiles';items=[]
    candidates=[]
    if profiles.exists():
        candidates.extend((p,'trace') for p in profiles.rglob('trace_view.json'))
        candidates.extend((p,'kernel_csv') for p in profiles.rglob('kernel_details.csv'))
    for path,kind in candidates:
        exists=path.is_file();size=path.stat().st_size if exists else 0
        events=_trace_device_kernel_events(path) if exists and kind=='trace' else 0
        kernels=_csv_device_kernel_rows(path) if exists and kind=='kernel_csv' else 0
        items.append({'kind':kind,'path':str(path.resolve()),'exists':exists,'bytes':size,
                      'device_kernel_events':events,'device_kernel_rows':kernels,
                      'substantive':bool(size and (events or kernels))})
    return {'artifacts':items,'has_device_kernel_evidence':any(v['substantive'] for v in items),
            'acceptance':'accepted' if any(v['substantive'] for v in items) else 'not_accepted',
            'source_directory':str(profiles.resolve()) if profiles.exists() else str(profiles)}


def _bench_environment(run,c,root):
    env=private_environment(run,c,root)
    env.update(VLLM_PLUGINS='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('VLLM_TORCH_PROFILER_DIR',None)
    return env


def collect_performance(run,c):
    """Six-group official benchmark; frozen request counts, lengths, concurrency."""
    root=Path(run).parents[3];run=Path(run);settings=c['measurements']
    out=run/'performance';out.mkdir(exist_ok=True);env=_bench_environment(run,c,root)
    if not (out/'profile-artifacts.json').is_file():
        write_json(out/'profile-artifacts.json',{'acceptance':'not_run','artifacts':[],
                    'has_device_kernel_evidence':False,'source_directory':str((run/'profiles').resolve())})
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    cli=[python,'-B','-m','vllm.entrypoints.cli.main','bench','serve']
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    body={'chat_template_kwargs':kwargs}
    if 'add_special_tokens' in c['evaluation']:body['add_special_tokens']=c['evaluation']['add_special_tokens']
    commands=[]
    def call(label,argv,folder,timeout=1800):
        record={'label':label,'argv':argv,'status':'running'};commands.append(record);write_json(out/'commands.json',commands)
        try:
            with (folder/(label+'.log')).open('w') as log:
                r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            record.update(exit=r.returncode,status='passed' if r.returncode==0 else 'failed')
            write_json(out/'commands.json',commands)
            if r.returncode:raise RuntimeError('measurement failed: '+label)
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__)
            if isinstance(exc,subprocess.TimeoutExpired):record['timeout']=True
            write_json(out/'commands.json',commands)
            raise
    call('official-help',cli+['--help'],out,120)
    def one(spec,folder,requests,warmups):
        folder.mkdir(parents=True,exist_ok=True);dataset=folder/'input.jsonl'
        call('prepare',[python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',c['identity']['weights'],
             '--tokens',str(spec['input']),'--output',str(dataset),'--kwargs',json.dumps(kwargs)],folder,180)
        args=cli+['--backend','openai-chat','--base-url',base,'--endpoint','/v1/chat/completions',
             '--model',c['service']['served_model'],'--tokenizer',c['identity']['weights'],'--trust-remote-code',
             '--dataset-name','custom','--dataset-path',str(dataset),'--skip-chat-template','--disable-shuffle',
             '--custom-output-len',str(spec['output']),'--num-prompts',str(requests),'--num-warmups',str(warmups),
             '--max-concurrency',str(spec['concurrency']),'--request-rate','inf','--ignore-eos','--temperature','0',
             '--extra-body',json.dumps(body),'--save-result','--save-detailed','--result-dir',str(folder),'--result-filename','official.json']
        call('benchmark',args,folder,settings.get('timeout_per_group',3600))
        raw=json.loads((folder/'official.json').read_text())
        if raw.get('completed')!=requests or raw.get('failed',0)!=0:raise RuntimeError('benchmark request count or failure check failed')
        return raw
    state(run,'performance','running')
    summary=[]
    for i,spec in enumerate(settings['workloads']):
        raw=one(spec,out/f'case-{i:02d}',settings.get('requests',32),settings.get('warmups',2))
        summary.append({'workload':spec,'completed':raw['completed'],**{k:raw.get(k) for k in
            ('output_throughput','request_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms','mean_e2el_ms')}})
        write_json(out/'summary.json',summary)
    state(run,'performance','passed',groups=len(summary))
    return summary


def collect_profile(run,c):
    """Server-side NPU profile capture with real device kernel evidence gate.

    Kept independent from the benchmark so a device trace exists before any
    long decode group can fail; the six-group contract stays untouched.
    """
    root=Path(run).parents[3];run=Path(run);settings=c['measurements']
    out=run/'performance';out.mkdir(exist_ok=True);env=_bench_environment(run,c,root)
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    cli=[python,'-B','-m','vllm.entrypoints.cli.main','bench','serve']
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    body={'chat_template_kwargs':kwargs}
    if 'add_special_tokens' in c['evaluation']:body['add_special_tokens']=c['evaluation']['add_special_tokens']
    commands=[]
    def call(label,argv,folder,timeout=1800):
        record={'label':label,'argv':argv,'status':'running'};commands.append(record);write_json(out/'profile-commands.json',commands)
        try:
            with (folder/(label+'.log')).open('w') as log:
                r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            record.update(exit=r.returncode,status='passed' if r.returncode==0 else 'failed')
            write_json(out/'profile-commands.json',commands)
            if r.returncode:raise RuntimeError('measurement failed: '+label)
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__)
            if isinstance(exc,subprocess.TimeoutExpired):record['timeout']=True
            write_json(out/'profile-commands.json',commands)
            raise
    def one(spec,folder,requests,warmups):
        folder.mkdir(parents=True,exist_ok=True);dataset=folder/'input.jsonl'
        call('prepare',[python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',c['identity']['weights'],
             '--tokens',str(spec['input']),'--output',str(dataset),'--kwargs',json.dumps(kwargs)],folder,180)
        args=cli+['--backend','openai-chat','--base-url',base,'--endpoint','/v1/chat/completions',
             '--model',c['service']['served_model'],'--tokenizer',c['identity']['weights'],'--trust-remote-code',
             '--dataset-name','custom','--dataset-path',str(dataset),'--skip-chat-template','--disable-shuffle',
             '--custom-output-len',str(spec['output']),'--num-prompts',str(requests),'--num-warmups',str(warmups),
             '--max-concurrency',str(spec['concurrency']),'--request-rate','inf','--ignore-eos','--temperature','0',
             '--extra-body',json.dumps(body),'--save-result','--save-detailed','--result-dir',str(folder),'--result-filename','official.json']
        call('benchmark',args,folder,settings.get('timeout_per_group',3600))
        raw=json.loads((folder/'official.json').read_text())
        if raw.get('completed')!=requests or raw.get('failed',0)!=0:raise RuntimeError('benchmark request count or failure check failed')
        return raw
    def control(action):
        req=urllib.request.Request(base+'/'+action+'_profile',method='POST')
        with urllib.request.urlopen(req,timeout=120) as response:
            write_json(out/(action+'-profile.json'),{'http_status':response.status,'body':response.read(8192).decode(errors='replace')})
    state(run,'profile','running')
    profile_error=None
    try:
        control('start')
        one(settings.get('profile_workload',{'input':4096,'output':128,'concurrency':1}),out/'profile-request',1,0)
    except BaseException as exc:
        profile_error=exc
    finally:
        try:control('stop')
        except BaseException as exc:
            if profile_error is None:profile_error=exc
        index=profile_artifact_index(run)
        write_json(out/'profile-artifacts.json',index)
    if profile_error is not None:
        state(run,'profile','failed',error=type(profile_error).__name__,detail=str(profile_error)[:1000],
              artifact_count=len(index['artifacts']),acceptance=index['acceptance'])
        raise profile_error
    if not index['has_device_kernel_evidence']:
        state(run,'profile','failed',reason='no substantive NPU kernel events or kernel CSV rows',artifact_count=len(index['artifacts']),
              acceptance=index['acceptance'])
        raise RuntimeError('profile contains no substantive device kernel events; CPU-only trace is insufficient')
    state(run,'profile','passed',artifact_count=len(index['artifacts']),
          device_kernel_events=sum(v['device_kernel_events'] for v in index['artifacts']),
          device_kernel_rows=sum(v['device_kernel_rows'] for v in index['artifacts']))
    return index


def collect(run,c,result):
    """Legacy order (performance then profile); new callers use the split stages."""
    collect_performance(run,c)
    collect_profile(run,c)
