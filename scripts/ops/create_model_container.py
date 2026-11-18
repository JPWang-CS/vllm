"""Create a project container from the registered template with read-only assets."""
import argparse,json,re,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--task',required=True)
    p.add_argument('--resume-task',help='Resume only a stopped container created by this exact prior task');a=p.parse_args()
    root=a.project.resolve(strict=True);task=(root/'runs'/a.task).resolve(strict=True)
    if not task.is_relative_to(root) or not re.fullmatch(r'[A-Za-z0-9_.-]+-vllm',a.name):raise ValueError('invalid project or name')
    weights=a.weights.resolve(strict=True)
    if not (weights/'config.json').is_file() or not (weights/'model.safetensors.index.json').is_file():
        raise ValueError('real indexed checkpoint required')
    profile=json.loads(a.template.read_text())
    if profile['runtime']!='runc' or not profile['privileged']:raise ValueError('unaccepted runtime template')
    existing=None
    inspected=subprocess.run(['docker','inspect',a.name],capture_output=True,text=True)
    if inspected.returncode==0:
        existing=json.loads(inspected.stdout)[0]
        labels=existing['Config'].get('Labels') or {}
        if not a.resume_task or labels.get('fullnet.project')!=str(root) or labels.get('fullnet.task')!=a.resume_task or existing['State']['Running']:
            raise RuntimeError('existing container preserved; ownership or stopped-state check failed')
        if existing['Config']['Image']!=profile['image']:raise ValueError('container image differs from registered template')
    argv=['docker','run','-d','--name',a.name,'--label','fullnet.project='+str(root),
          '--label','fullnet.task='+a.task,'--restart','no','--pid','host','--runtime','runc',
          '--privileged','--user','0:0','--workdir',str(root),'--network',profile['network'],'--ipc',profile['ipc']]
    if profile.get('shm_size'):argv+=['--shm-size',str(profile['shm_size'])]
    for key,flag in [('groups','--group-add'),('cap_add','--cap-add'),('cap_drop','--cap-drop'),('security_opt','--security-opt')]:
        for value in profile.get(key) or []:argv += [flag,str(value)]
    for value in profile.get('env',[]):argv+=['--env',value]
    mount_records=[]
    for item in profile['mounts']:
        if item['type']!='bind':continue
        source=Path(item['source'])
        if not source.exists():raise ValueError('registered bind source is absent: '+str(source))
        target=item['target']
        # All inherited files and environments are read-only in this container.
        argv+=['--mount',f'type=bind,src={source},dst={target},readonly']
        mount_records.append({'source':str(source),'target':target,'writable':False})
    for target in (str(root),str(root).replace('/data2/','/data/docker/',1)):
        if any(v['target']==target for v in mount_records):raise ValueError('project mount conflicts with template')
        argv+=['--mount',f'type=bind,src={root},dst={target}']
        mount_records.append({'source':str(root),'target':target,'writable':True})
    target=root/'assets/weights/pangu-92b'
    target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise ValueError('project weight mount target must be empty')
    argv+=['--mount',f'type=bind,src={weights},dst={target},readonly']
    mount_records.append({'source':str(weights),'target':str(target),'writable':False})
    for device in profile.get('devices') or []:
        argv+=['--device',device['PathOnHost']+':'+device['PathInContainer']+':'+device['CgroupPermissions']]
    for item in profile.get('ulimits') or []:argv+=['--ulimit',f"{item['Name']}={item['Soft']}:{item['Hard']}"]
    import shlex
    boot='python3 '+shlex.quote(str(root/'scripts/ops/container_boot.py'))+' && exec sleep infinity'
    argv+=['--entrypoint','/bin/bash',profile['image'],'--noprofile','--norc','-c',boot]
    (task/'container-plan.json').write_text(json.dumps({'name':a.name,'image':profile['image'],'mounts':mount_records,
        'runtime':'runc','pid_mode':'host','npu_initialized':False},ensure_ascii=False,indent=2))
    if existing:
        old_mounts={m['Destination']:m for m in existing['Mounts']}
        for m in mount_records:
            old=old_mounts.get(m['target'],{})
            if old.get('Source')!=m['source'] or old.get('RW')!=m['writable']:raise ValueError('existing mount differs')
        subprocess.run(['docker','start',a.name],capture_output=True,text=True,check=True,timeout=45)
    else:
        subprocess.run(argv,capture_output=True,text=True,check=True,timeout=90)
    try:
        time.sleep(2)
        state=json.loads(subprocess.check_output(['docker','inspect',a.name],text=True))[0]
        if not state['State']['Running']:raise RuntimeError('container boot failed')
        actual={m['Destination']:m for m in state['Mounts']}
        for m in mount_records:
            if actual[m['target']]['RW']!=m['writable']:raise RuntimeError('mount mode mismatch')
        probe=('import json,pathlib\np=pathlib.Path('+repr(str(target))+')\n'
               'files=list(p.glob("*.safetensors"))\n'
               'out=dict(architectures=json.loads((p/"config.json").read_text())["architectures"],shards=len(files),bytes=sum(f.stat().st_size for f in files))\n'
               'print(json.dumps(out))\n')
        compile(probe,'weight-probe','exec')
        r=subprocess.run(['docker','exec','-u','0','-w',str(root),a.name,'python3','-B','-c',probe],capture_output=True,text=True,timeout=30)
        (task/'weight-probe.stdout').write_text(r.stdout);(task/'weight-probe.stderr').write_text(r.stderr)
        if r.returncode:raise RuntimeError('weight probe failed: '+r.stderr[-1000:])
        result={'container':a.name,'running':True,'weights':json.loads(r.stdout),'model_started':False}
        (task/'container-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (root/'.runtime/container.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False))
    except BaseException:
        # Only the container created by this invocation is stopped on failed setup.
        subprocess.run(['docker','stop','--time','15',a.name],capture_output=True,timeout=25)
        raise


if __name__=='__main__':main()
