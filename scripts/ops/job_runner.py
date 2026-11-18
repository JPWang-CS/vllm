"""A business task supervised by project tmux, independent of agent tool timeouts."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--task',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);run=(root/'runs'/a.task).resolve(strict=True)
    if not run.is_relative_to(root):raise ValueError('invalid task')
    config=json.loads((run/'job-config.json').read_text())
    env=os.environ.copy();env.update(config.get('env',{}))
    process=None;code=1;started=time.time()
    try:
        with (run/'job.log').open('x') as output:
            process=subprocess.Popen(config['argv'],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            (run/'job-state.json').write_text(json.dumps({'status':'running','pid':process.pid,'runner_pid':os.getpid(),'started':started,'argv':config['argv']}))
            code=process.wait()
    except BaseException:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:
        (run/'job-exit.txt').write_text(str(code)+'\n')
        (run/'job-state.json').write_text(json.dumps({'status':'passed' if code==0 else 'failed','exit':code,'started':started,'ended':time.time(),'pid':None if process is None else process.pid}))
    return code


if __name__=='__main__':raise SystemExit(main())
