"""Submit the reviewed, project-local model tasks without an agent-tool deadline."""
import argparse,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--tmux',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);config=a.config.resolve(strict=True)
    if not config.is_relative_to(root):raise ValueError('config outside project')
    for item in json.loads(config.read_text()):
        task=item['task'];run=root/'runs'/task
        if '/' in task or '\\' in task or task in {'.','..'}:raise ValueError('invalid task')
        run.mkdir(mode=0o700)
        path=run/'requested-job.json';path.write_text(json.dumps({'argv':item['argv']}))
        subprocess.run([sys.executable,'-B',str(root/'scripts/ops/submit_job.py'),'--project',str(root),
            '--task',task,'--config',str(path),'--tmux',a.tmux],check=True)


if __name__=='__main__':main()
