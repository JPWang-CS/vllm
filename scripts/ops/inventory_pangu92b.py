"""Read deployment pointers and real model assets before any 92B device work."""
import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


PATH_KEYS = {'weight_path', 'model_path', 'model_dir', 'model', 'python', 'cann_prefix',
             'native_root', 'model_extra_source', 'effective_path', 'source_roots', 'vendor_roots',
             'driver_library_roots', 'wrapper'}
CONFIG_KEYS = {'architectures', 'model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
               'num_key_value_heads', 'n_routed_experts', 'num_experts_per_tok', 'max_position_embeddings',
               'quantization_config', 'torch_dtype', 'num_nextn_predict_layers'}


def pointers(value, prefix=''):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            label = prefix+'.'+key
            if key in PATH_KEYS and isinstance(item, (str, list)):
                found[label] = item
            elif isinstance(item, (dict, list)):
                found.update(pointers(item, label))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                found.update(pointers(item, prefix+f'[{i}]'))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--task',required=True)
    a=p.parse_args();root=a.project.resolve(strict=True);task=(root/'runs'/a.task).resolve(strict=True)
    runtime=a.runtime_root.resolve(strict=True)
    if not task.is_relative_to(root):raise ValueError('task outside project')
    result={'runtime_exists':runtime.is_dir(),'deployment_pointers':[], 'model_assets':[],
            'device_computation':'not_started','source_roots':[], 'interpreters':[]}
    candidates=set()
    if runtime.is_dir():
        for base in [runtime/'r',runtime/'runs',runtime/'config',runtime/'configs']:
            if not base.is_dir():continue
            files=[]
            for current,dirs,names in os.walk(base):
                here=Path(current);depth=len(here.relative_to(base).parts)
                dirs[:]=[d for d in dirs if d not in {'cache','logs','raw','profiles','source','source-bound','node_modules','.git'} and depth<4]
                for name in names:
                    if name.endswith('.json') and ('config' in name.lower() or name in {'pair.json','bootstrap.json','profile.json'}):
                        path=here/name
                        try:
                            if path.stat().st_size<256*1024:files.append(path)
                        except OSError:pass
                if len(files)>1500:break
            files.sort(key=lambda v:v.stat().st_mtime,reverse=True)
            for path in files[:500]:
                try:
                    content=json.loads(path.read_text());refs=pointers(content)
                    if refs:
                        result['deployment_pointers'].append({'path':str(path),'paths':refs})
                        for key,value in refs.items():
                            if key.rsplit('.',1)[-1] in {'model','model_path','model_dir','weight_path'} and isinstance(value,str) and value.startswith('/'):
                                candidates.add(value)
                except (OSError,ValueError):pass
        for name in ('models','model','weights'):
            base=runtime/name
            if base.is_dir():
                candidates.add(str(base))
                try:candidates.update(str(v) for v in base.iterdir() if v.is_dir())
                except OSError:pass
        for source in (runtime/'source/image/opt/vllm',runtime/'source/image/workspace/omniinfer/components/omni-npu/src'):
            if source.is_dir():result['source_roots'].append(str(source.resolve()))
        envs=runtime/'envs'
        if envs.is_dir():
            result['interpreters']=[str(v/'bin/python') for v in envs.iterdir() if (v/'bin/python').is_file()]
    for value in sorted(candidates):
        directory=Path(value)
        try:
            config=json.loads((directory/'config.json').read_text())
            if not config.get('architectures'):continue
            weights=list(directory.glob('*.safetensors'))+list(directory.glob('*.bin'))
            result['model_assets'].append({'path':str(directory),'realpath':str(directory.resolve()),
                'config':{k:v for k,v in config.items() if k in CONFIG_KEYS},
                'weight_files':len(weights),'weight_bytes':sum(v.stat().st_size for v in weights),
                'index_present':(directory/'model.safetensors.index.json').is_file()})
        except (OSError,ValueError):pass
    for label,args in [('device_status',['info']),('device_mapping',['info','-m'])]:
        try:
            r=subprocess.run([str(root/'tools/npu-smi'),*args],capture_output=True,text=True,timeout=35)
            (task/(label+'.txt')).write_text(r.stdout+r.stderr)
            result[label]={'exit':r.returncode,'file':label+'.txt'}
        except (OSError,subprocess.TimeoutExpired) as exc:result[label]={'error':str(exc)}
    result['profilers']={}
    for name in ('msprof','msprof-analyze'):
        import shutil
        result['profilers'][name]=shutil.which(name)
    (task/'pangu92b-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'models':result['model_assets'],'device_status':result['device_status'],
                      'profilers':result['profilers'],'config_count':len(result['deployment_pointers'])},ensure_ascii=False))


if __name__=='__main__':main()
