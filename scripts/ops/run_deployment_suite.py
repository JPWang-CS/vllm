"""Run the declared native deployment cases sequentially; retain the last service."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--devices',type=int,nargs='+',required=True)
    p.add_argument('--requests',type=int,default=32);a=p.parse_args()
    root=a.project.resolve(strict=True);task=(root/'runs'/a.task).resolve(strict=True)
    if not task.is_relative_to(root):raise ValueError('task outside project')
    variants=[(2,1),(4,1),(2,2),(1,4)]
    cases=[]
    for tp,dp in variants:
        for graph,mtp in [('eager',0),('full_decode_only',0),('eager',3),('full_decode_only',3)]:
            cases.append({'tp':tp,'dp':dp,'ep':tp*dp,'graph':graph,'mtp':mtp,'status':'planned'})
    state={'scope':'A5 native single-host; TP2, TP4, TP2DP2, TP1DP4; eager/graph x MTP0/3',
           'cases':cases,'other_required_modes':{
               'same_host_pd':'pending_adapter_validation','cross_host_pd':'pending_network_and_adapter_validation',
               'cross_host_mixed':'pending_network_validation','a3':'pending_model_runtime_validation'}}
    def save():
        path=task/'suite.json';tmp=task/'suite.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(path)
    dataset=None;save()
    for i,case in enumerate(cases):
        if len(a.devices)<case['tp']*case['dp']:
            case.update(status='blocked_resource',reason='insufficient selected devices');save();continue
        ident=a.task+f'-{i:02d}';case['run']=ident;run=root/'runs'/ident
        if run.exists():raise RuntimeError('existing case must be inspected before resuming')
        # The container executes as root; the host reviewer must still read results.
        run.mkdir(mode=0o755)
        if os.name=='posix' and os.geteuid()==0:
            owner=task.stat();os.chown(run,owner.st_uid,owner.st_gid)
        chosen=a.devices[:case['tp']*case['dp']]
        args=[sys.executable,'-B',str(root/'scripts/ops/run_pangu92b.py'),'--project',str(root),'--reference',str(a.reference),
              '--weights',str(a.weights),'--devices',*map(str,chosen),'--npu-ids',*map(str,chosen),'--tp',str(case['tp']),
              '--dp',str(case['dp']),'--graph',case['graph'],'--mtp',str(case['mtp']),'--task',ident,'--requests',str(a.requests)]
        if dataset:args+=['--dataset',str(dataset)]
        if i==len(cases)-1:args+=['--keep-ready']
        case.update(status='running',argv=args,started=time.time());save()
        with (run/'suite-case.log').open('x') as log:
            child=subprocess.Popen(args,cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            case['supervisor_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        case.update(status='passed' if code==0 else 'failed',exit=code,ended=time.time());save()
        validation=run/'validation.json'
        detail=json.loads(validation.read_text()) if validation.is_file() else {}
        if detail.get('stage') in ('device-pre','device-before-launch') and code:
            state['blocked_reason']=detail.get('detail','device preflight failed')
            for remaining in cases[i+1:]:
                remaining.update(status='blocked_resource',reason=state['blocked_reason'])
            break
        if detail.get('device_empty_after_exit') is False:
            state['blocked_reason']='previous model task did not release selected devices'
            for remaining in cases[i+1:]:
                remaining.update(status='blocked_cleanup',reason=state['blocked_reason'])
            break
        if i==0 and code:
            state['blocked_reason']='native baseline failed; inspect its first error before expanding the matrix'
            for remaining in cases[i+1:]:
                remaining.update(status='blocked_baseline',reason=state['blocked_reason'])
            break
        candidate=run/'data/all.jsonl'
        if candidate.is_file():
            rows=[json.loads(v) for v in candidate.read_text().splitlines() if v.strip()]
            if len(rows)==48:dataset=candidate
        # One failed config remains a separate result; proceed to other declared configs.
    state['finished']=time.time()
    state['passed']=sum(c['status']=='passed' for c in cases)
    state['failed']=sum(c['status']=='failed' for c in cases)
    state['status']='passed' if state['passed']==len(cases) else 'incomplete'
    save()
    return 0 if state['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
