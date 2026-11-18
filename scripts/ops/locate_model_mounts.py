"""Check registered weight paths on the host and our existing containers, read-only."""
import argparse,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--task',required=True)
    p.add_argument('--container-prefix',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);task=(root/'runs'/a.task).resolve(strict=True)
    if not task.is_relative_to(root):raise ValueError('task outside project')
    inv=json.loads(a.inventory.read_text());paths=set()
    for item in inv.get('deployment_pointers',[]):
        for key,value in item['paths'].items():
            if key.endswith(('weight_path','model_path')) and isinstance(value,str) and value.startswith('/'):
                paths.add(value)
    result={'weight_paths':sorted(paths),'host':[],'containers':[]}
    for value in sorted(paths):
        path=Path(value)
        result['host'].append({'path':value,'exists':path.is_dir(),'config':(path/'config.json').is_file()})
    ps=subprocess.run(['docker','ps','-a','--format','{{.Names}}'],capture_output=True,text=True,check=True,timeout=15)
    for name in ps.stdout.splitlines():
        if name!=a.container_prefix and not name.startswith(a.container_prefix+'-'):continue
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
        if inspected.returncode:continue
        item=json.loads(inspected.stdout)[0]
        record={'name':name,'running':item['State']['Running'],'pid':item['State']['Pid'],
                'image':item['Config']['Image'],'model_mounts':[],'weights':[]}
        for m in item.get('Mounts',[]):
            if any(v in (m['Source']+' '+m['Destination']).lower() for v in ['sfs','model','weight','pangu']):
                record['model_mounts'].append({k:m.get(k) for k in ('Source','Destination','RW','Type')})
        if item['State']['Running']:
            # Inspect the container mount namespace through its root without executing its environment.
            base=Path('/proc')/str(item['State']['Pid'])/'root'
            for value in sorted(paths):
                path=base/value.lstrip('/')
                try:
                    config=json.loads((path/'config.json').read_text())
                    files=list(path.glob('*.safetensors'))
                    record['weights'].append({'path':value,'access_path':str(path),'architectures':config.get('architectures'),
                        'layers':config.get('num_hidden_layers'),'hidden':config.get('hidden_size'),
                        'weight_files':len(files),'weight_bytes':sum(v.stat().st_size for v in files)})
                except (OSError,ValueError):record['weights'].append({'path':value,'accessible':False})
        result['containers'].append(record)
    (task/'model-mounts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
