"""Read registered container metadata and live device mapping without changing it."""
import argparse,json,subprocess,shlex
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--management',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--container',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);run=(root/'runs'/a.task).resolve(strict=True)
    if not run.is_relative_to(root):raise ValueError('invalid task')
    path=a.management/'tools/containers/release-peer-profile.json'
    profile=json.loads(path.read_text())
    safe={k:profile.get(k) for k in ('image','privileged','runtime','user','ipc','network','devices','pid_mode')}
    safe['visibility']=[v for v in profile.get('env',[]) if v.startswith(('ASCEND_RT_VISIBLE_DEVICES=','ASCEND_VISIBLE_DEVICES='))]
    safe['mounts']=[{'source':m.get('source'),'target':m.get('target'),'writable':m.get('writable')} for m in profile.get('mounts',[])]
    alternate=a.management/'tools/containers/namespace-profile.json'
    if alternate.is_file():
        other=json.loads(alternate.read_text())
        safe['namespace_profile']={k:other.get(k) for k in ('image','privileged','runtime','devices')}
        safe['namespace_profile']['visibility']=[v for v in other.get('env',[]) if v.startswith(('ASCEND_RT_VISIBLE_DEVICES=','ASCEND_VISIBLE_DEVICES='))]
    for label,args in [('mapping',['info','-m']),('status',['info'])]:
        command='set -e; source '+shlex.quote(str(a.management/'source-cann-9.2.0.sh'))+'; exec '+shlex.join([str(root/'tools/npu-smi'),*args])
        result=subprocess.run(['docker','exec','-u','0','-w',str(root),a.container,'bash','--noprofile','--norc','-c',command],capture_output=True,text=True,timeout=35)
        safe[label]={'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    (run/'container-plan.json').write_text(json.dumps(safe,ensure_ascii=False,indent=2));print(json.dumps(safe,ensure_ascii=False))


if __name__=='__main__':main()
