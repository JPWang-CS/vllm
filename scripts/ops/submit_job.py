"""Submit an immutable argv job to a project-private tmux socket."""
import argparse,json,re,shlex,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--tmux',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.task):raise ValueError('invalid task id')
    root=a.project.resolve(strict=True);run=(root/'runs'/a.task).resolve(strict=True);config=a.config.resolve(strict=True)
    if not run.is_relative_to(root) or not config.is_relative_to(root):raise ValueError('job path outside project')
    settings=json.loads(config.read_text());argv=settings.get('argv')
    if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv):raise ValueError('argv list required')
    runtime=root/'.runtime';runtime.mkdir(exist_ok=True);socket=runtime/'jobs.sock'
    if len(str(socket).encode())>=104:raise ValueError('project socket path too long')
    if subprocess.run([str(a.tmux),'-S',str(socket),'has-session','-t',a.task],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('job already running')
    target=run/'job-config.json'
    with target.open('x') as output:json.dump(settings,output,ensure_ascii=False,indent=2)
    command=shlex.join([sys.executable,'-B',str(root/'scripts/ops/job_runner.py'),'--project',str(root),'--task',a.task])+' > '+shlex.quote(str(run/'runner.log'))+' 2>&1'
    subprocess.run([str(a.tmux),'-S',str(socket),'new-session','-d','-s',a.task,'-c',str(root),command],check=True)
    print(json.dumps({'submitted':True,'task':a.task,'socket':str(socket),'state':str(run/'job-state.json')}))


if __name__=='__main__':main()
