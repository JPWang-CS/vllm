"""Bounded tools for GLM to inspect and resume model experiments in our container.

The agent chooses a JSON plan in its repair directory. This entrypoint accepts
model configuration values, never arbitrary shell commands or process IDs.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def identifier(value):
    """Validate a request/job identifier before using it in task-owned paths."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("invalid request or job identifier")
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def owned(root, path):
    target = Path(path).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("path outside project")
    return target


def standards(root, task):
    """Expose registered read-only rules inside the restricted agent workspace."""
    management = root.parents[1]
    folder = task / 'standards'
    folder.mkdir(exist_ok=True)
    paths = []
    for name in ('SERVER_STANDARDS.md', 'SERVERS.md', 'OPERATIONS.md'):
        source = management / name
        if source.is_file():
            target = folder / name
            target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
            paths.append(str(target))
    for target in (folder / 'server-use/SKILL.md', task / 'repair/MAIN_GUIDANCE.md'):
        if target.is_file():
            paths.append(str(target))
    return paths


def container(root, argv, *, source=False, timeout=45):
    config = read(root / "scripts/ops/framework-recovery.local.json")
    body = "exec " + shlex.join(argv)
    if source:
        body = "set -e; source " + shlex.quote(config["cann_source"]) + "; " + body
    command = ["docker", "exec", "--user", "0", "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONPATH=" + str(root / "src"), "-e", "FULLNET_CA_BUNDLE=" + str(root / "cache/trust/workstation-ca.pem"),
               config["container"], "bash", "--noprofile", "--norc", "-c", body]
    return subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)


def inventory(root, task):
    result = container(root, [str(root / "tools/npu-smi"), "info"], source=True)
    (task / "device-latest.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("device query failed or returned no device table")
    healthy = {int(v) for v in re.findall(r"\|\s*(\d+)\s*\|\s*Ascend\w*\s*\|\s*OK\b", result.stdout)}
    empty = {int(v) for v in re.findall(r"No running processes found in NPU\s+(\d+)\s", result.stdout)}
    config = read(root / "scripts/ops/framework-recovery.local.json")
    protected = set(config.get("protected_devices", []))
    available = sorted((healthy & empty) - protected)
    value = {"at": time.time(), "healthy": sorted(healthy), "available": available,
             "protected_devices": sorted(protected), "raw": str(task / "device-latest.txt")}
    write(task / "preflight.json", value)
    return value


def snapshot(root, task):
    ledger = read(task / "ledger.json") if (task / "ledger.json").exists() else {"cases": []}
    ids = [v["task"] for v in ledger["cases"]]
    config = read(root / "scripts/ops/framework-recovery.local.json")
    code = '''import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); ids=json.loads(sys.argv[2]); out=[]
for ident in ids:
 t=r/'runs'/ident; e=r/'models/pangu-92b/experiments'/ident
 v={'task':ident}
 for base,name in [(t,'job-state.json'),(t,'validation.json'),(e,'config.json'),(e,'state.json'),(e,'results.json'),(e,'smoke.json'),(e/'performance','summary.json'),(e/'performance','profile-artifacts.json')]:
  p=base/name
  if p.is_file():
   try:v[name]=json.loads(p.read_text())
   except ValueError:v[name]={'error':'incomplete_json'}
 for name,p in [('service_tail',e/'logs/service.log'),('job_tail',t/'job.log')]:
  if p.is_file():v[name]=p.read_text(errors='replace')[-12000:]
 out.append(v)
print(json.dumps(out,ensure_ascii=False))
'''
    result = container(root, [config["python"], "-B", "-c", code, str(root), json.dumps(ids)])
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    value = {"at": time.time(), "cases": json.loads(result.stdout)}
    write(task / "progress.json", value)
    return value


def inspect_previous(root, task):
    config = read(root / "scripts/ops/framework-recovery.local.json")
    code = '''import json,pathlib,sys
c=json.loads(sys.argv[1]);r=pathlib.Path(c['previous_run']);ref=json.loads(pathlib.Path(c['reference']).read_text())
value={'reference':{k:ref[k] for k in ('model','parallel','graph','speculative_config') if k in ref},
       'source_roots':ref['bootstrap']['source_roots'],'previous':{}}
if isinstance(value['reference'].get('model'),dict):
 value['reference']['model']={k:v for k,v in value['reference']['model'].items() if k in ('block_size','dtype','kv_cache_dtype','max_model_len','served_model_name','weight_path')}
for name in ('config.json','state.json','results.json'):
 p=r/name
 if p.is_file():value['previous'][name]=json.loads(p.read_text())
p=r/'logs/service.log'
if p.is_file():
 lines=p.read_text(errors='replace').splitlines()
 value['previous']['service_tail']='\\n'.join(lines[-70:])
 value['previous']['first_request_events']=[s for s in lines if any(x in s for x in ('10:40:5','10:41:','10:45:'))][:100]
 value['previous']['aligned_decode_events']=[s for s in lines if 'hybrid-aligned-decode' in s][:20]
print(json.dumps(value,ensure_ascii=False))
'''
    result = container(root, [config['python'], '-B', '-c', code, json.dumps(config)])
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    value = json.loads(result.stdout)
    write(task / 'previous-evidence.json', value)
    return {'readable_evidence': str(task / 'previous-evidence.json'), 'source_roots': value['source_roots'],
            'guidance': 'Root-owned old experiment files are exposed in readable_evidence. For read-only platform source use --action source --path ABSOLUTE_FILE_OR_DIRECTORY; do not read credential/configuration stores.'}


def source_read(root, path, offset):
    config = read(root / 'scripts/ops/framework-recovery.local.json')
    code = '''import json,pathlib,sys
c=json.loads(sys.argv[1]);ref=json.loads(pathlib.Path(c['reference']).read_text());p=pathlib.Path(sys.argv[2]).resolve(strict=True)
roots=[pathlib.Path(v).resolve(strict=True) for v in ref['bootstrap']['source_roots']]
if not any(p.is_relative_to(r) for r in roots):raise ValueError('source path outside registered read-only runtime')
if p.is_dir():print(json.dumps({'path':str(p),'entries':sorted(v.name for v in p.iterdir())[:200]}))
elif p.suffix=='.py':
 lines=p.read_text(errors='replace').splitlines();offset=int(sys.argv[3]);print(json.dumps({'path':str(p),'line':offset,'total_lines':len(lines),'source':'\\n'.join(lines[offset:offset+200])},ensure_ascii=False))
else:raise ValueError('only Python source files and directory listings are allowed')
'''
    result = container(root, [config['python'], '-B', '-c', code, json.dumps(config), path, str(max(0, offset))])
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)


def start(root, task, plan):
    ledger = read(task / "ledger.json") if (task / "ledger.json").exists() else {"cases": []}
    current = snapshot(root, task)
    if any(v.get("job-state.json", {}).get("status") == "running" for v in current["cases"]):
        raise ValueError("existing recovery case is running; wait before starting another")
    failed = sum(v.get("job-state.json", {}).get("status") == "failed" for v in current["cases"])
    if failed >= 3 or len(ledger["cases"]) >= 20:
        raise ValueError("recovery budget exhausted; retain evidence for main-agent review")
    allowed = {"tp", "dp", "graph", "mtp", "max_batched_tokens", "max_seqs", "disable_aligned_decode", "reason"}
    if set(plan) - allowed:
        raise ValueError("unexpected plan fields")
    tp, dp = plan.get("tp", 2), plan.get("dp", 1)
    if (tp, dp) not in {(2, 1), (4, 1), (2, 2), (1, 4)}:
        raise ValueError("unsupported parallel plan")
    graph, mtp = plan.get("graph", "eager"), plan.get("mtp", 0)
    batched, seqs = plan.get("max_batched_tokens", 4096), plan.get("max_seqs", 8)
    if graph not in {"eager", "full_decode_only"} or mtp not in {0, 3}:
        raise ValueError("unsupported graph or MTP plan")
    if type(batched) is not int or batched not in {512, 4096, 10240, 16384} or type(seqs) is not int or seqs not in {4, 8, 32, 56}:
        raise ValueError("unsupported scheduling limits")
    if type(plan.get("disable_aligned_decode", False)) is not bool or not isinstance(plan.get("reason"), str) or not plan["reason"].strip():
        raise ValueError("configuration change needs a reason and a boolean alignment setting")
    cards = inventory(root, task)["available"]
    if len(cards) < tp * dp:
        write(task / "blocked.json", {"reason": "insufficient healthy empty devices", "required": tp * dp, "available": cards})
        raise RuntimeError("insufficient healthy empty devices; no process was stopped")
    cards = cards[:tp * dp]
    mapping = container(root, [str(root / 'tools/npu-smi'), 'info', '-m'], source=True)
    (task / 'device-mapping.txt').write_text(mapping.stdout + mapping.stderr, encoding='utf-8')
    physical = {}
    for line in mapping.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5 and all(v.isdigit() for v in fields[:4]):
            physical[int(fields[0])] = int(fields[3])
    if mapping.returncode or any(card not in physical for card in cards):
        raise RuntimeError('device mapping is unavailable; no runtime IDs inferred')
    runtime_cards = [physical[card] for card in cards]
    config = read(root / "scripts/ops/framework-recovery.local.json")
    ident = task.name + "-" + str(len(ledger["cases"]) + 1).zfill(2)
    if len(ident) > 63:
        raise ValueError("task identifier too long")
    run = root / "runs" / ident
    run.mkdir(mode=0o755)
    args = [config["python"], "-B", str(root / "scripts/ops/run_pangu92b.py"), "--project", str(root),
            "--reference", config["reference"], "--weights", config["weights"], "--task", ident,
            "--dataset", config["dataset"], "--devices", *map(str, runtime_cards), "--npu-ids", *map(str, cards),
            "--tp", str(tp), "--dp", str(dp), "--graph", graph, "--mtp", str(mtp), "--requests", "32",
            "--max-batched-tokens", str(batched), "--max-seqs", str(seqs)]
    if plan.get("disable_aligned_decode", False):
        args.append("--disable-aligned-decode")
    body = "set -e; source " + shlex.quote(config["cann_source"]) + "; exec " + shlex.join(args)
    command = ["docker", "exec", "--user", "0", "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONPATH=" + str(root / "src"), "-e", "FULLNET_CA_BUNDLE=" + str(root / "cache/trust/workstation-ca.pem"),
               config["container"], "bash", "--noprofile", "--norc", "-c", body]
    write(run / "requested-job.json", {"argv": command})
    write(run / "recovery-plan.json", {"plan": plan, "devices": cards, "runtime_physical_ids": runtime_cards})
    receipt = subprocess.run([sys.executable, "-B", str(root / "scripts/ops/submit_job.py"), "--project", str(root),
                              "--task", ident, "--config", str(run / "requested-job.json"), "--tmux", config["tmux"]],
                             cwd=root, capture_output=True, text=True, timeout=30)
    ledger["cases"].append({"task": ident, "plan": plan, "devices": cards, "submitted_at": time.time(), "submit_exit": receipt.returncode})
    write(task / "ledger.json", ledger)
    if receipt.returncode:
        raise RuntimeError(receipt.stderr[-1000:])
    return {"submitted": ident, "devices": cards, "receipt": receipt.stdout}


def run_cpu_script(root, task, script):
    path = owned(task / 'repair', script)
    if path.suffix not in {'.py', '.sh'}:
        raise ValueError('CPU repair script must be Python or Bash')
    config = read(root / 'scripts/ops/framework-recovery.local.json')
    ident = task.name + '-cpu-' + str(time.time_ns())[-10:]
    directory = root / 'runs' / ident
    directory.mkdir(mode=0o700)
    argv = [config['python'], '-B', str(path)] if path.suffix == '.py' else ['bash', str(path)]
    # CPU scripts run as the project owner, without device visibility.
    uid, gid = str(task.stat().st_uid), str(task.stat().st_gid)
    private = str(directory)
    command = ['docker', 'exec', '--user', uid + ':' + gid, '-w', str(root),
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=' + str(root / 'src'),
               '-e', 'ASCEND_RT_VISIBLE_DEVICES=', '-e', 'ASCEND_VISIBLE_DEVICES=',
               '-e', 'HOME=' + private, '-e', 'TMPDIR=' + private,
               '-e', 'XDG_CACHE_HOME=' + private, config['container'], *argv]
    write(directory / 'requested-job.json', {'argv': command})
    result = subprocess.run([sys.executable, '-B', str(root / 'scripts/ops/submit_job.py'),
        '--project', str(root), '--task', ident, '--config', str(directory / 'requested-job.json'),
        '--tmux', config['tmux']], cwd=root, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return {'cpu_job': ident, 'output': result.stdout, 'log': str(directory / 'job.log')}


def fleet_submit(root, task, request_id, action, host, job, script=None, devices=None, args=None, asset=None):
    """Write a broker request only; this process never opens SSH or reads credentials."""
    request_id = identifier(request_id)
    job = identifier(job)
    if action not in {'inventory', 'run', 'status', 'cancel', 'copy_asset'}:
        raise ValueError('unsupported fleet action')
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError('fleet host must be a registered IPv4 address') from exc
    if address.version != 4 or str(address) != host:
        raise ValueError('fleet host must be a canonical IPv4 address')
    devices = [] if devices is None else devices
    if (not isinstance(devices, list) or len(set(devices)) != len(devices) or
            any(type(value) is not int or not 0 <= value < 32 for value in devices)):
        raise ValueError('invalid fleet NPU ID list')
    args = [] if args is None else args
    if not isinstance(args, list) or any(not isinstance(value, str) or '\x00' in value or len(value) > 4096 for value in args):
        raise ValueError('fleet args must be bounded strings')
    script_name = None
    if action == 'run':
        if not devices:
            raise ValueError('fleet run requires explicit host NPU IDs')
        if not isinstance(script, str) or '\\' in script:
            raise ValueError('fleet run script must be relative to task repair/')
        from pathlib import PurePosixPath
        relative = PurePosixPath(script)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'repair' or relative.suffix not in {'.py', '.sh'}:
            raise ValueError('fleet run script must be relative to task repair/ and be a Python or Bash file')
        repair = owned(root, task / 'repair')
        source = owned(repair, repair.joinpath(*relative.parts[1:]))
        if not source.is_file():
            raise ValueError('fleet repair script must be a regular file')
        script_name = relative.as_posix()
    elif script is not None:
        raise ValueError('only fleet run accepts a script')
    if action == 'copy_asset':
        if asset not in {'pangu-7b', 'pangu-92b'}:
            raise ValueError('copy_asset must select one of the registered model assets')
    elif asset is not None:
        raise ValueError('asset is only valid for copy_asset')

    record = {'request_id': request_id, 'action': action, 'host': str(address), 'job': job,
              'script': script_name, 'devices': devices, 'args': args}
    if asset is not None:
        record['asset'] = asset
    directory = task / 'fleet' / 'requests'
    directory.mkdir(parents=True, exist_ok=True)
    directory = owned(root, directory)
    target = directory / (request_id + '.json')
    if target.exists():
        raise FileExistsError('fleet request already exists; inspect its response instead of overwriting')
    claim = directory / ('.' + request_id + '.lock')
    try:
        claim.open('x', encoding='utf-8').close()
    except FileExistsError as exc:
        raise RuntimeError('another writer owns this fleet request ID; do not replace it') from exc
    temporary = directory / ('.' + request_id + '-' + str(os.getpid()) + '.tmp')
    try:
        if target.exists():
            raise FileExistsError('fleet request already exists; inspect its response instead of overwriting')
        with temporary.open('x', encoding='utf-8') as output:
            json.dump(record, output, ensure_ascii=False, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        claim.unlink(missing_ok=True)
    return {'status': 'submitted_to_broker_queue', 'request': str(target), 'request_id': request_id,
            'host': record['host'], 'action': action, 'job': job}


def fleet_status(root, task, request_id, wait_seconds=0):
    request_id = identifier(request_id)
    if type(wait_seconds) not in (int, float) or not 0 <= wait_seconds <= 300:
        raise ValueError('fleet status wait must be between 0 and 300 seconds')
    requests = owned(root, task / 'fleet' / 'requests')
    request_path = owned(root, requests / (request_id + '.json'))
    request = read(request_path)
    response_path = requests.parent / 'responses' / (request_id + '.json')
    response_dir = response_path.parent
    response_dir.mkdir(parents=True, exist_ok=True)
    response_dir = owned(root, response_dir)
    response_path = response_dir / response_path.name
    deadline = time.monotonic() + wait_seconds
    while not response_path.is_file() and time.monotonic() < deadline:
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    if not response_path.is_file():
        return {'request_id': request_id, 'job': request.get('job'), 'action': request.get('action'),
                'status': 'pending', 'request': str(request_path), 'response': str(response_path)}
    if response_path.stat().st_size > 1_000_000:
        raise ValueError('fleet response exceeds read limit')
    response = read(response_path)
    status = 'failed' if 'error' in response else 'received'
    return {'request_id': request_id, 'job': request.get('job'), 'action': request.get('action'),
            'status': status, 'request': str(request_path), 'response_path': str(response_path),
            'response': response}


def fleet_inventory_request(root, task, request_id, host, job):
    return fleet_submit(root, task, request_id, 'inventory', host, job, devices=[], args=[])


def fleet_broker(root, task, broker_action):
    """Inspect the workstation broker or request a reconnect without ending GLM."""
    folder = task / 'fleet'
    folder.mkdir(exist_ok=True)
    if broker_action == 'status':
        heartbeat = folder / 'broker-heartbeat.json'
        value = read(heartbeat) if heartbeat.is_file() else {'status': 'not_observed'}
        if 'at' in value:
            value['age_seconds'] = round(max(0, time.time() - value['at']), 1)
            value['alive'] = value['age_seconds'] < 120
        return value
    if broker_action not in ('start', 'restart'):
        raise ValueError('unsupported broker action')
    value = {'action': 'reconnect', 'requested_at': time.time(), 'task': task.name}
    write(folder / 'broker-control.json', value)
    return {'status': 'reconnect_requested', 'controller_state_unchanged': True,
            'guidance': 'Check broker-heartbeat.json and submit a fresh inventory request. Workstation supervisor owns broker startup.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--action", choices=("inspect", "source", "preflight", "start", "exec", "wait", "finish", "fleet-submit", "fleet-status", "fleet-inventory", "fleet-broker", "matrix"), required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--path")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument('--script', type=Path)
    p.add_argument('--request-id')
    p.add_argument('--fleet-action', choices=('inventory', 'run', 'status', 'cancel', 'copy_asset'))
    p.add_argument('--asset', choices=('pangu-7b', 'pangu-92b'))
    p.add_argument('--host')
    p.add_argument('--job')
    p.add_argument('--fleet-script', help='task repair relative .py or .sh path')
    p.add_argument('--devices', type=int, nargs='*')
    p.add_argument('--args-json', default='[]', help='JSON string array forwarded to the reviewed repair script')
    p.add_argument('--wait-seconds', type=float, default=0)
    p.add_argument('--fleet-hosts', nargs='*', help='registered IPv4 target hosts for inventory requests')
    p.add_argument('--broker-action', choices=('status', 'start', 'restart'))
    p.add_argument('--matrix-action', choices=('plan', 'status', 'submit', 'refresh', 'fleet-inventory', 'fleet-asset', 'fleet-status'))
    p.add_argument('--case')
    p.add_argument('--retry', action='store_true')
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    if root != Path(__file__).resolve().parents[2] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", a.task):
        raise ValueError("invalid project or task")
    task = owned(root, root / "runs" / a.task)
    (task / "repair").mkdir(exist_ok=True)
    if a.action == "matrix":
        import sys as _sys
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from scripts.ops import deployment_matrix
        if not a.matrix_action:
            raise ValueError("matrix action requires --matrix-action")
        if a.matrix_action == "plan":
            result = deployment_matrix.create_matrix(root, a.task)
        elif a.matrix_action == "submit":
            if not a.case:
                raise ValueError("matrix submit requires --case")
            result = deployment_matrix.submit_case(root, a.task, a.case, retry=a.retry)
        elif a.matrix_action == "refresh":
            result = deployment_matrix.refresh_matrix(root, a.task)
        elif a.matrix_action == "fleet-inventory":
            if not a.case or not a.fleet_hosts:
                raise ValueError("matrix fleet-inventory requires --case and --fleet-hosts")
            result = deployment_matrix.submit_fleet_inventory(root, a.task, a.case, a.fleet_hosts)
        elif a.matrix_action == "fleet-asset":
            if not a.case or not a.host or not a.asset:
                raise ValueError("matrix fleet-asset requires --case, --host and --asset")
            result = deployment_matrix.submit_fleet_asset(root, a.task, a.case, a.host, a.asset)
        elif a.matrix_action == "fleet-status":
            if not a.case:
                raise ValueError("matrix fleet-status requires --case")
            result = deployment_matrix.read_fleet_inventory(root, a.task, a.case, a.wait_seconds)
        else:
            matrix, _ = deployment_matrix.load_matrix(root, a.task)
            result = deployment_matrix._matrix_summary(matrix)
    elif a.action == "preflight":
        result = inventory(root, task)
        result.update(inspect_previous(root, task))
    elif a.action == 'fleet-submit':
        if not all((a.request_id, a.fleet_action, a.host, a.job)):
            raise ValueError('fleet-submit requires request-id, fleet-action, host and job')
        try:
            args = json.loads(a.args_json)
        except ValueError as exc:
            raise ValueError('--args-json must be a JSON string array') from exc
        result = fleet_submit(root, task, a.request_id, a.fleet_action, a.host, a.job,
                              str(a.fleet_script) if a.fleet_script is not None else None,
                              a.devices, args, a.asset)
    elif a.action == 'fleet-status':
        if not a.request_id:
            raise ValueError('fleet-status requires --request-id')
        result = fleet_status(root, task, a.request_id, a.wait_seconds)
    elif a.action == 'fleet-inventory':
        hosts = a.fleet_hosts or []
        if not hosts:
            raise ValueError('fleet-inventory requires --fleet-hosts')
        result = {'requests': [fleet_inventory_request(root, task,
                   f'{a.task[:20]}-inventory-{index:02d}', host, a.task)
                   for index, host in enumerate(hosts)]}
    elif a.action == 'fleet-broker':
        if not a.broker_action:
            raise ValueError('fleet-broker requires --broker-action')
        result = fleet_broker(root, task, a.broker_action)
    elif a.action == "source":
        if not a.path:
            raise ValueError('--path is required for source inspection')
        result = source_read(root, a.path, a.offset)
    elif a.action == "start":
        path = owned(task / "repair", a.plan)
        result = start(root, task, read(path))
    elif a.action == 'exec':
        if not a.script:
            raise ValueError('--script is required')
        result = run_cpu_script(root, task, a.script)
    elif a.action == "wait":
        time.sleep(40)
        result = snapshot(root, task)
        for case in result["cases"]:
            for name in ("service_tail", "job_tail"):
                if name in case:
                    case[name] = case[name][-1200:]
    elif a.action == "finish":
        result = snapshot(root, task)
        result["agent_finished_at"] = time.time()
        write(task / "handoff.json", result)
    else:
        result = snapshot(root, task)
        config = read(root / "scripts/ops/framework-recovery.local.json")
        result["baseline_failure"] = config["previous_run"]
        result["reference"] = config["reference"]
        result.update(inspect_previous(root, task))
    result['readable_server_standards'] = standards(root, task)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
