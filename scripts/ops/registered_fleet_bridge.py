"""Workstation-owned SSH broker for task-scoped multi-host GLM jobs.

The server agent writes structured requests, never receives SSH credentials,
and uses the same private project on each registered target. Device jobs are
gated immediately before their script executes inside the personal container.
"""
from __future__ import annotations

import argparse
import os
import importlib.util
import json
import re
import shlex
import time
from pathlib import Path, PurePosixPath


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("invalid task, request or job identifier")
    return value


def validate_request(value, targets):
    if set(value) - {"request_id", "action", "host", "job", "script", "devices", "args", "asset"}:
        raise ValueError("unexpected fleet request fields")
    identifier(value.get("request_id"))
    identifier(value.get("job", value["request_id"]))
    if value.get("host") not in targets:
        raise ValueError("target is not registered for this task")
    if value.get("action") not in {"inventory", "run", "status", "cancel", "copy_asset"}:
        raise ValueError("unsupported action")
    if value.get("action") == "copy_asset" and value.get("asset") not in {"pangu-7b", "pangu-92b"}:
        raise ValueError("only the two registered model assets may be transferred")
    devices = value.get("devices", [])
    if not isinstance(devices, list) or len(set(devices)) != len(devices) or any(type(x) is not int or not 0 <= x < 32 for x in devices):
        raise ValueError("invalid device selection")
    args = value.get("args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) or len(x) > 4096 for x in args):
        raise ValueError("arguments must be bounded strings")
    if value["action"] == "run":
        script = PurePosixPath(value.get("script", ""))
        if script.is_absolute() or ".." in script.parts or script.suffix not in {".py", ".sh"} or script.parts[0] != "repair":
            raise ValueError("script must be a task repair Python or Bash file")
    return value


# Runs inside the selected personal container, under the target job directory.
RUNNER = r'''
import json,os,pathlib,signal,subprocess,sys,time
j=pathlib.Path(sys.argv[1]).resolve(strict=True); c=json.loads((j/'config.json').read_text())
root=pathlib.Path(c['project']).resolve(strict=True)
if not j.is_relative_to(root):raise ValueError('job outside project')
os.chdir(root)
for k,s in {'HOME':'home','TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_EXTENSIONS_DIR':'cache/torch','TRITON_CACHE_DIR':'cache/triton','ASCEND_PROCESS_LOG_PATH':'logs/ascend','ASCEND_WORK_PATH':'tmp/ascend'}.items():
 p=j/s;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',FULLNET_REMOTE_PROJECT=str(root),FULLNET_FLEET_JOB_ROOT=str(j),FULLNET_SERVER_ENV_ROOT=c['management'],FULLNET_FLEET_DEVICES=json.dumps(c['devices']),PYTHONPATH=str(root/'src'))
os.environ.pop('ASCEND_RT_VISIBLE_DEVICES',None);os.environ.pop('ASCEND_VISIBLE_DEVICES',None)
# Host-matched npu-smi: the broker stages the host ELF into <job>/tools/npu-smi;
# containers without it fall back to the shared wrapper (which needs a real
# in-container npu-smi). Driver libs are prepended for either candidate.
dlenv=dict(os.environ);dlenv['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common'+((':'+dlenv['LD_LIBRARY_PATH']) if dlenv.get('LD_LIBRARY_PATH') else '')
tool=None
for cand in (j/'tools/npu-smi',pathlib.Path(c['management'])/'tools/bin/npu-smi'):
 try:
  if cand.is_file() and cand.stat().st_size>4096:tool=cand;break
 except OSError:pass
def write(name,value):
 p=j/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
child=None;code=1
try:
 if c['devices']:
  if tool is None:raise RuntimeError('no host-matched npu-smi tool staged or shared for this container')
  mapping=subprocess.run([str(tool),'info','-m'],capture_output=True,text=True,timeout=30,env=dlenv)
  (j/'device-mapping.txt').write_text(mapping.stdout+mapping.stderr)
  if mapping.returncode:raise RuntimeError('device mapping query failed')
  for d in c['devices']:
   h=subprocess.run([str(tool),'info','-t','health','-i',str(d)],capture_output=True,text=True,timeout=30,env=dlenv)
   p=subprocess.run([str(tool),'info','-t','proc-mem','-i',str(d)],capture_output=True,text=True,timeout=30,env=dlenv)
   (j/f'device-{d}-pre.txt').write_text(h.stdout+h.stderr+p.stdout+p.stderr)
   import re
   if h.returncode or p.returncode or not re.search(r'Health\s+Status\s*:\s*OK\b',h.stdout) or 'No process in device' not in p.stdout:raise RuntimeError(f'device {d} is not confirmed healthy and empty')
  # Scripts derive logical/physical runtime IDs from the saved official mapping;
  # host NPU IDs are deliberately not reused as runtime device IDs here.
 write('state.json',{'status':'running','runner_pid':os.getpid(),'started_at':time.time(),'devices':c['devices']})
 argv=['bash',str(j/c['script'])] if c['script'].endswith('.sh') else ['python3','-B',str(j/c['script'])]
 with (j/'workload.log').open('x') as log:
  child=subprocess.Popen(argv+c['args'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write('process.json',{'pid':child.pid,'pgid':child.pid,'argv':argv+c['args'],'job':c['job'],'project':str(root)})
  while child.poll() is None:
   if (j/'CANCEL').exists():
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(2)
  code=child.wait()
except BaseException as e:
 write('failure.json',{'error':type(e).__name__,'detail':str(e)[:2000]})
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=45)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
 (j/'exit.txt').write_text(str(code)+'\n')
 write('state.json',{'status':'exited','exit':code,'finished_at':time.time(),'runner_pid':os.getpid()})
 # Device post-state is retained independently from the script exit status.
 for d in c['devices']:
  try:
   p=subprocess.run([str(tool or pathlib.Path(c['management'])/'tools/bin/npu-smi'),'info','-t','proc-mem','-i',str(d)],capture_output=True,text=True,timeout=30,env=dlenv)
   (j/f'device-{d}-post.txt').write_text(p.stdout+p.stderr)
  except Exception as e:(j/f'device-{d}-post.txt').write_text(str(e))
sys.exit(code)
'''


class Broker:
    def __init__(self, helper, control, project, task, targets, local):
        self.helper, self.control, self.project = helper, control, project
        self.task, self.targets, self.local = identifier(task), targets, Path(local)
        self.run = PurePosixPath(project) / "runs" / task
        self.local.mkdir(parents=True, exist_ok=True)

    def call(self, client, cmd, body=None, timeout=45):
        rc, out, err = self.helper.call(client, cmd, body, timeout=timeout)
        if rc:
            raise RuntimeError(f"remote command exit {rc}: {err[-1500:]} {out[-1000:]}")
        return out

    def atomic(self, sftp, name, value):
        temp = name + ".tmp"
        with sftp.open(temp, "w") as out:
            out.write(json.dumps(value, ensure_ascii=False).encode())
        sftp.posix_rename(temp, name)

    def execute(self, request, source):
        value = validate_request(request, self.targets)
        target = self.targets[value["host"]]
        root, management, con = target["project"], target["management"], target["container"]
        if set(value.get("devices", [])) & set(target.get("protected_devices", [])):
            raise ValueError("selected device is protected by an existing service")
        client = self.helper.connect(value["host"])
        try:
            job = str(PurePosixPath(root) / "runs" / self.task / "fleet-jobs" / value.get("job", value["request_id"]))
            if value["action"] == "copy_asset":
                if value["host"] == self.control_host:
                    raise ValueError("asset already belongs to the control project")
                asset = value["asset"]
                origin = self.project + "/assets/weights/" + asset
                code = "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]).resolve(strict=True); print(json.dumps({'root':str(p),'files':[{'path':str(f.relative_to(p)),'size':f.stat().st_size} for f in p.rglob('*') if f.is_file() and f.suffix in {'.safetensors','.json','.txt','.model','.tiktoken','.py','.jinja'} and f.resolve().is_relative_to(p)]}))"
                listing = json.loads(self.call(self.control, shlex.join(["docker", "exec", "-i", "--user", "0", self.targets[self.control_host]["container"], "python3", "-B", "-c", code, origin]), timeout=60))
                if not any(x["path"].endswith(".safetensors") for x in listing["files"]):
                    raise ValueError("source has no real model weight shards")
                source_sf = self.control.open_sftp(); target_sf = client.open_sftp()
                transferred = 0
                try:
                    for entry in listing["files"]:
                        rel = PurePosixPath(entry["path"])
                        if rel.is_absolute() or ".." in rel.parts:raise ValueError("asset path outside model")
                        dest = root + "/assets/weights/" + asset + "/" + rel.as_posix()
                        self.helper.mkdirs(target_sf, str(PurePosixPath(dest).parent), 0o700)
                        actual = self.call(client, shlex.join(["realpath", "-e", str(PurePosixPath(dest).parent)])).strip()
                        if not PurePosixPath(actual).is_relative_to(PurePosixPath(root)):
                            raise ValueError("asset destination resolves outside target project")
                        try:
                            attrs = target_sf.lstat(dest)
                            import stat
                            if not stat.S_ISREG(attrs.st_mode):raise ValueError("existing target asset is not regular")
                            if attrs.st_size == entry["size"]:continue
                            raise ValueError("existing asset size differs; preserve it and inspect")
                        except FileNotFoundError:pass
                        partial = dest + "." + self.task + ".partial"
                        with source_sf.open(listing["root"] + "/" + rel.as_posix(), "rb") as inp, target_sf.open(partial, "w") as out:
                            while True:
                                data = inp.read(4 * 1024 * 1024)
                                if not data:break
                                out.write(data);transferred += len(data)
                        if target_sf.stat(partial).st_size != entry["size"]:raise RuntimeError("incomplete asset transfer")
                        target_sf.rename(partial, dest)
                finally:
                    source_sf.close();target_sf.close()
                return {"asset":asset,"transferred_bytes":transferred,"files":len(listing["files"]),"target":root+"/assets/weights/"+asset,"acceptance":"files_transferred_model_not_validated"}
            if value["action"] == "inventory":
                code = r'''
import json,pathlib,subprocess,sys
m=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);o={'management':str(m),'project':str(r),'project_exists':r.exists(),'project_realpath':str(r.resolve()),'container':sys.argv[3]}
def call(args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=20);return {'exit':p.returncode,'stdout':p.stdout[-16000:],'stderr':p.stderr[-1500:]}
 except Exception as e:return {'error':type(e).__name__,'detail':str(e)}
o['container_info']=call(['docker','inspect','--format','{{json .Mounts}}',sys.argv[3]])
for label,args in [('mapping',[str(m/'tools/bin/npu-smi'),'info','-m']),('devices',[str(m/'tools/bin/npu-smi'),'info']),('addresses',['ip','-br','address']),('routes',['ip','route'])]:o[label]=call(args)
for label,base in [('model_runtime',m.parents[1]/'pangu-deploy'),('project_runtime',r/'envs'),('weights',r/'assets/weights')]:
 o[label]={'path':str(base),'exists':base.exists(),'entries':sorted(x.name for x in base.iterdir())[:40] if base.is_dir() else []}
print(json.dumps(o,ensure_ascii=False))
'''
                return json.loads(self.call(client, shlex.join(["python3", "-B", "-", management, root, con]), code, 100))
            if value["action"] in {"status", "cancel"}:
                code = r'''
import pathlib,json,sys
j=pathlib.Path(sys.argv[1]).resolve(strict=True);r=pathlib.Path(sys.argv[2]).resolve(strict=True)
if not j.is_relative_to(r/'runs'/sys.argv[3]/'fleet-jobs'):raise ValueError('job outside task')
if sys.argv[4]=='cancel':(j/'CANCEL').touch(exist_ok=True)
o={'job':str(j)}
for n in ['state.json','process.json','failure.json','exit.txt','workload.log','runner.log','result.json','validation.json','report.md']:
 p=j/n
 if p.is_file():o[n]=p.read_text(errors='replace')[-14000:]
print(json.dumps(o,ensure_ascii=False))
'''
                return json.loads(self.call(client, shlex.join(["docker", "exec", "-i", "--user", "0", con, "python3", "-B", "-", job, root, self.task, value["action"]]), code))
            # Realpaths and container mount visibility are checked before writing.
            check = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); assert str(p.resolve())==str(p), 'project symlink outside registered root'"
            self.call(client, shlex.join(["python3", "-B", "-c", check, root]))
            self.call(client, shlex.join(["docker", "exec", con, "test", "-d", root]))
            sf = client.open_sftp()
            try:
                try:
                    sf.stat(job)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("job already exists; inspect instead of duplicating")
                self.helper.mkdirs(sf, job, 0o700)
                source_root = str(self.run)
                # Copy only task-authored repair scripts; never credentials or environments.
                def copy_scripts(relative):
                    folder = source_root + "/" + relative
                    for ent in source.listdir_attr(folder):
                        import stat
                        rel = relative + "/" + ent.filename
                        if stat.S_ISDIR(ent.st_mode):
                            copy_scripts(rel)
                        elif stat.S_ISREG(ent.st_mode) and PurePosixPath(rel).suffix in {".py", ".sh"}:
                            if ent.st_size > 2_000_000:
                                raise ValueError("repair script exceeds transfer limit")
                            with source.open(source_root + "/" + rel, "rb") as inp:
                                data = inp.read()
                            self.helper.mkdirs(sf, str(PurePosixPath(job + "/" + rel).parent), 0o700)
                            with sf.open(job + "/" + rel, "wx") as out:
                                out.write(data.replace(b"\r\n", b"\n"))
                copy_scripts("repair")
                # Stage the current framework, retaining overwritten target sources.
                # This is a project transfer, not an environment/model installation.
                if value["host"] != self.control_host:
                    def copy_project(relative):
                        import stat
                        try:entries = source.listdir_attr(self.project + "/" + relative)
                        except FileNotFoundError:return
                        for ent in entries:
                            if ent.filename in {"experiments", "__pycache__", "weights", "cache", "envs", ".git"}:
                                continue
                            rel = relative + "/" + ent.filename
                            if stat.S_ISDIR(ent.st_mode):
                                copy_project(rel)
                            elif stat.S_ISREG(ent.st_mode) and PurePosixPath(rel).suffix in {".py", ".sh", ".md", ".toml", ".json", ".txt"} and ".local." not in ent.filename:
                                if ent.st_size > 2_000_000:continue
                                with source.open(self.project + "/" + rel, "rb") as inp:data = inp.read()
                                dest = root + "/" + rel
                                self.helper.mkdirs(sf, str(PurePosixPath(dest).parent), 0o700)
                                actual_parent = self.call(client, shlex.join(["realpath", "-e", str(PurePosixPath(dest).parent)])).strip()
                                if not PurePosixPath(actual_parent).is_relative_to(PurePosixPath(root)):
                                    raise ValueError("target source directory resolves outside project")
                                try:
                                    attrs = sf.lstat(dest)
                                    if not stat.S_ISREG(attrs.st_mode):raise ValueError("target source is not a regular file")
                                    with sf.open(dest, "rb") as old:old_data = old.read()
                                    if old_data == data:continue
                                    backup = job + "/source-before/" + rel
                                    self.helper.mkdirs(sf, str(PurePosixPath(backup).parent), 0o700)
                                    with sf.open(backup, "wx") as out:out.write(old_data)
                                except FileNotFoundError:pass
                                with sf.open(dest, "w") as out:out.write(data)
                    for directory in ("scripts", "src", "models", ".agents", "requirements"):
                        copy_project(directory)
                # Confirm the selected source is a regular file after canonicalization.
                actual = self.call(self.control, shlex.join(["realpath", "-e", str(self.run / value["script"])] )).strip()
                if not PurePosixPath(actual).is_relative_to(self.run / "repair"):
                    raise ValueError("source script resolves outside task repair")
                # Stage the host-matched npu-smi ELF into the job directory so the
                # RUNNER device gate works in containers that ship no real npu-smi
                # (e.g. only a zero-byte /usr/local/bin/npu-smi stub).
                staged_tool = ""
                if value.get("devices"):
                    stage = ("mkdir -p " + shlex.quote(job + "/tools") +
                             " && for t in /usr/local/sbin/npu-smi /usr/local/bin/npu-smi; do "
                             "if [ -f \"$t\" ] && [ -x \"$t\" ] && [ \"$(stat -c %s \"$t\")\" -gt 4096 ]; then "
                             "cp \"$t\" " + shlex.quote(job + "/tools/npu-smi") +
                             " && chmod 755 " + shlex.quote(job + "/tools/npu-smi") +
                             " && echo \"$t\"; break; fi; done")
                    staged_tool = self.call(client, shlex.join(["bash", "-c", stage]), timeout=30).strip()
                config = {"project": root, "management": management, "container": con,
                          "job": value.get("job", value["request_id"]), "devices": value.get("devices", []),
                          "script": value["script"], "args": value.get("args", [])}
                for name, content in [("runner.py", RUNNER), ("config.json", json.dumps(config, ensure_ascii=False))]:
                    with sf.open(job + "/" + name, "wx") as out:
                        out.write(content.encode())
            finally:
                sf.close()
            sock = str(PurePosixPath(root) / ".runtime/fleet.sock")
            if len(sock.encode()) >= 104:
                raise ValueError("project tmux socket path is too long")
            self.call(client, shlex.join(["mkdir", "-p", str(PurePosixPath(sock).parent)]))
            argv = ["docker", "exec", "--user", "0", "-w", root, con, "python3", "-B", job + "/runner.py", job]
            command = shlex.join(argv) + " > " + shlex.quote(job + "/runner.log") + " 2>&1"
            session = "fleet-" + self.task[-24:] + "-" + config["job"][-24:]
            self.call(client, shlex.join([management + "/tools/bin/tmux", "-S", sock, "new-session", "-d", "-s", session, "-c", root, command]))
            return {"submitted": True, "host": value["host"], "project": root, "container": con,
                    "job": config["job"], "directory": job, "socket": sock, "session": session,
                    "staged_npu_smi": staged_tool or None,
                    "acceptance": "submitted_not_validated"}
        finally:
            client.close()

    def poll(self):
        sf = self.control.open_sftp()
        count = 0
        try:
            requests = str(self.run / "fleet/requests")
            responses = str(self.run / "fleet/responses")
            self.helper.mkdirs(sf, requests, 0o700)
            self.helper.mkdirs(sf, responses, 0o700)
            for name in sorted(sf.listdir(requests)):
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.json", name):
                    continue
                try:
                    sf.stat(responses + "/" + name)
                    continue
                except FileNotFoundError:
                    pass
                with sf.open(requests + "/" + name, "rb") as f:
                    raw = f.read(131073)
                if len(raw) > 131072:
                    result = {"error": "request too large"}
                else:
                    try:
                        request = json.loads(raw)
                        if request.get("request_id") != name[:-5]:
                            raise ValueError("request filename differs from request_id")
                        result = self.execute(request, sf)
                    except Exception as exc:
                        result = {"error": type(exc).__name__, "detail": str(exc)[:2500]}
                result["broker_checked_at"] = time.time()
                self.atomic(sf, responses + "/" + name, result)
                (self.local / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1
        finally:
            sf.close()
        return count


def terminal_controller_state(value):
    """Only the controller's completed schema ends an attempt."""
    return (isinstance(value, dict) and type(value.get('exit')) is int
            and value.get('model') == 'glm-5.3-flash'
            and value.get('child_alive') is False
            and value.get('fifo_removed') is True
            and value.get('credentials_persisted') is False)


def reconcile_status(broker):
    """Refresh business cases separately from GLM/controller liveness."""
    code = r'''import json,pathlib,sys,time
r=pathlib.Path(sys.argv[1]);task=sys.argv[2];sys.path.insert(0,str(r))
from scripts.ops import deployment_matrix as matrix
p=r/'runs'/task;out={'at':time.time(),'task':task}
if (p/'deployment-matrix.json').is_file():
 # A stale job-state file is not evidence of a live job.
 ledger=p/'ledger.json'
 if ledger.is_file():
  for item in json.loads(ledger.read_text()).get('cases',[]):
   ident=item.get('task','');j=r/'runs'/ident;f=j/'job-state.json'
   if not ident or not j.resolve().is_relative_to(r/'runs') or not f.is_file():continue
   value=json.loads(f.read_text());pid=value.get('pid')
   if value.get('status')!='running' or type(pid) is not int:continue
   try:
    raw=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes()
    alive=str(r).encode() in raw and ident.encode() in raw
   except OSError:alive=False
   if not alive:
    value.update(status='interrupted',reconciled_at=time.time(),reason='recorded business process is no longer present')
    tmp=f.with_suffix('.reconcile.tmp');tmp.write_text(json.dumps(value));tmp.replace(f)
 out['matrix']=matrix.refresh_matrix(r,task)
for name in ('STATE.json','ACTIVE.json'):
 f=p/name
 if f.is_file():
  try:out[name]=json.loads(f.read_text())
  except ValueError:out[name]={'status':'incomplete_json'}
# These counts describe the controller, never model acceptance.
c=out.get('STATE.json',{});active=out.get('ACTIVE.json',{});pid=active.get('controller_pid')
alive=False
if type(pid) is int:
 try:
  text=pathlib.Path('/proc')/str(pid)/'cmdline'
  alive=str(p).encode() in text.read_bytes()
 except OSError:pass
out['controller_status']='running' if alive else ('exited' if type(c.get('exit')) is int else 'not_running')
f=p/'BACKGROUND_STATUS.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(out,ensure_ascii=False));t.replace(f)
print(json.dumps({'task':task,'controller_status':out['controller_status'],'at':out['at']}))
'''
    return json.loads(broker.call(broker.control, shlex.join(['python3','-B','-',broker.project,broker.task]),code,90))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--management", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("registered_ssh", a.management / "runs/remote-model-agents-20261027/install_binary.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    registered = json.loads((a.management / "tools/ssh/server-inventory.json").read_text(encoding="utf-8"))["servers"]
    permitted = {x["ip"]: x for x in registered if not x["maintenance_forbidden"] and x["server_ai_required"]}
    for host, target in c["targets"].items():
        if host not in permitted or target["management"] != permitted[host]["project_root"]:
            raise ValueError("target differs from registered inventory")
        if target["project"] != target["management"] + "/projects/vllm":
            raise ValueError("fleet writes must stay in the registered vllm project")
    receipt_root = Path(c['local_receipts'])
    receipt_root.mkdir(parents=True, exist_ok=True)
    heartbeat = receipt_root / 'broker-heartbeat.json'
    lock = receipt_root / 'broker.lock'
    # OS lock is released after crashes; the file is diagnostic, not a stale PID gate.
    lock_stream = lock.open('a+b')
    if os.name == 'nt':
        import msvcrt
        if lock.stat().st_size == 0:lock_stream.write(b'0');lock_stream.flush()
        lock_stream.seek(0)
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_refresh = 0
    while True:
        client = None
        try:
            client = helper.connect(c["control_host"])
            client.get_transport().set_keepalive(30)
            broker = Broker(helper, client, c["control_project"], c["task"], c["targets"], c["local_receipts"])
            broker.control_host = c["control_host"]
            while True:
                broker.poll()
                if a.once:
                    return 0
                sf = client.open_sftp()
                reconnect = False
                try:
                    now = time.time()
                    value = {'status':'connected','at':now,'pid':os.getpid(),'task':c['task']}
                    broker.atomic(sf, str(broker.run/'fleet/broker-heartbeat.json'), value)
                    heartbeat.write_text(json.dumps(value),encoding='utf-8')
                    try:
                        with sf.open(str(broker.run/'fleet/broker-control.json')) as stream:control=json.loads(stream.read())
                        if control.get('action') == 'reconnect':
                            sf.remove(str(broker.run/'fleet/broker-control.json'));reconnect=True
                    except (FileNotFoundError, ValueError):pass
                    try:sf.stat(str(broker.run/'fleet/STOP'));stopping=True
                    except FileNotFoundError:stopping=False
                finally:sf.close()
                if now-last_refresh >= 30:
                    try:
                        state=reconcile_status(broker)
                        (receipt_root/'background-status.json').write_text(json.dumps(state),encoding='utf-8')
                    except Exception as exc:
                        (receipt_root/'refresh-error.json').write_text(json.dumps({'error':type(exc).__name__,'detail':str(exc)[:1500],'at':now}),encoding='utf-8')
                    last_refresh=now
                if stopping:return 0
                if reconnect:break
                time.sleep(5)
        except (OSError, EOFError, helper.paramiko.SSHException) as exc:
            (Path(c["local_receipts"]) / "connection-error.json").write_text(json.dumps({"error":type(exc).__name__,"at":time.time()}), encoding="utf-8")
            heartbeat.write_text(json.dumps({'status':'reconnecting','at':time.time(),'pid':os.getpid(),'task':c['task'],'error':type(exc).__name__}),encoding='utf-8')
            if a.once:raise
            time.sleep(15)
        finally:
            if client:client.close()


if __name__ == "__main__":
    raise SystemExit(main())
