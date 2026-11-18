"""Check a tiny public weight range over available download routes."""
import argparse,json,ssl,time,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--server-proxy',required=True);a=p.parse_args()
    root=a.project.resolve(strict=True);run=(root/'runs'/a.task).resolve(strict=True)
    if not run.is_relative_to(root):raise ValueError('invalid task')
    context=ssl.create_default_context(cafile=str(root/'cache/trust/workstation-ca.pem'))
    url='https://huggingface.co/openpangu/openPangu-Embedded-7B-V1.1/resolve/main/model-00004-of-00004.safetensors'
    results=[]
    for name,proxy in [('direct',{}),('server_proxy',{'http':a.server_proxy,'https':a.server_proxy})]:
        started=time.monotonic()
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler(proxy),urllib.request.HTTPSHandler(context=context))
            request=urllib.request.Request(url,headers={'Range':'bytes=0-15'})
            with opener.open(request,timeout=15) as response:
                size=len(response.read(16));status=response.status
            item={'route':name,'status':status,'bytes':size,'seconds':time.monotonic()-started}
        except Exception as exc:item={'route':name,'error':type(exc).__name__,'seconds':time.monotonic()-started}
        results.append(item);print(json.dumps(item),flush=True)
        (run/'download-routes.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
