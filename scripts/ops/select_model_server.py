"""Collect the target container's device status and existing model interpreter."""
import argparse,json,os,shutil,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--task',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);run=(root/'runs'/a.task).resolve(strict=True)
    if not run.is_relative_to(root):raise ValueError('invalid task')
    result={'disk_free':shutil.disk_usage(root).free,'mapped_devices':[], 'runtime_python':str(a.runtime_root/'envs/pangu7b-image/bin/python')}
    for item in Path('/dev').glob('davinci[0-9]*'):
        s=item.stat();result['mapped_devices'].append({'path':str(item),'major':os.major(s.st_rdev),'minor':os.minor(s.st_rdev)})
    result['runtime_exists']=Path(result['runtime_python']).is_file()
    try:
        r=subprocess.run([str(root/'tools/npu-smi'),'info'],capture_output=True,text=True,timeout=30)
        (run/'device-status.txt').write_text(r.stdout+r.stderr)
        result['device_status']={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc:result['device_status']={'error':type(exc).__name__}
    if result['runtime_exists']:
        code='import json,importlib.metadata as m; out={}; exec("for n in [\\\"torch\\\",\\\"torch-npu\\\",\\\"vllm\\\",\\\"omni-npu\\\",\\\"transformers\\\"]:\\n try:out[n]=m.version(n)\\n except m.PackageNotFoundError:out[n]=None");print(json.dumps(out))'
        code='import json,importlib.metadata as m\nout={}\nfor n in ["torch","torch-npu","vllm","omni-npu","transformers"]:\n try:out[n]=m.version(n)\n except m.PackageNotFoundError:out[n]=None\nprint(json.dumps(out))'
        r=subprocess.run([result['runtime_python'],'-B','-c',code],capture_output=True,text=True,timeout=30)
        result['runtime_metadata']={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    (run/'server-selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
