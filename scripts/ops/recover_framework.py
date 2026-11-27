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
from pathlib import Path, PurePosixPath


FRAMEWORK_SOURCE_CONTRACT = 1
FRAMEWORK_SOURCE_RECEIPT = ".runtime/framework-source.json"


def broker_heartbeat_compatible(value):
    """Allow new device runs only through the freshness-aware broker."""
    return (isinstance(value, dict)
            and value.get("framework_source_receipt_schema") == FRAMEWORK_SOURCE_CONTRACT
            and value.get("broker_source_file") == "scripts/ops/registered_fleet_bridge.py"
            and value.get("loaded_source_matches_disk") is True)


def framework_source_receipt(root, task):
    """Validate the task-local published tracked-snapshot receipt."""
    heartbeat = Path(task) / 'fleet/broker-heartbeat.json'
    if not heartbeat.is_file() or not broker_heartbeat_compatible(read(heartbeat)):
        raise RuntimeError('freshness-aware broker required; start the current broker before a new run')
    path = Path(root) / FRAMEWORK_SOURCE_RECEIPT
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("published framework snapshot receipt is missing; refusing new run")
    value = read(path)
    if not isinstance(value, dict) or value.get("schema") != FRAMEWORK_SOURCE_CONTRACT:
        raise RuntimeError("published framework snapshot receipt has an unsupported schema")
    if value.get("branch") != "main":
        raise RuntimeError("published framework snapshot receipt must declare branch main")
    files = value.get("files")
    if (not isinstance(files, list) or not files
            or any(not isinstance(item, str) or PurePosixPath(item).is_absolute()
                   or ".." in PurePosixPath(item).parts for item in files)):
        raise RuntimeError("published framework snapshot receipt has no valid file list")
    return {"schema": value["schema"], "branch": "main",
            "receipt_path": FRAMEWORK_SOURCE_RECEIPT}


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
 if not e.is_dir(): e=r/'models/pangu-7b/experiments'/ident
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
    value = {"at": time.time(), "cases": json.loads(result.stdout),
             "business": business_checkpoint_status(task)}
    write(task / "progress.json", value)
    return value


def _normalized_plan_key(plan):
    """Return the retry-budget identity; reasons and scheduling limits are excluded."""
    plan = plan or {}
    value = {
        "tp": plan.get("tp", 2),
        "dp": plan.get("dp", 1),
        "graph": plan.get("graph", "eager"),
        "mtp": plan.get("mtp", 0),
        "pd_mode": plan.get("pd_mode") or "off",
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _execution_policy(task):
    """Load an optional task-local budget without removing the hard upper bounds."""
    defaults = {
        "max_total_jobs": 20, "max_failures_per_plan": 3, "quality_only": False,
        "max_controller_retries": 2, "retry_backoff_seconds": 30,
        "max_retry_backoff_seconds": 300, "controller_retry_window_seconds": 3600,
    }
    path = Path(task) / "repair" / "EXECUTION_POLICY.json"
    if not path.is_file():
        return {**defaults, "per_plan": False}
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError("execution policy must be an object")
    policy = {}
    for key, hard_max in (("max_total_jobs", 24), ("max_failures_per_plan", 3)):
        candidate = value.get(key, defaults[key])
        if type(candidate) is not int or candidate < 1 or candidate > hard_max:
            raise ValueError(f"execution policy {key} must be an integer in 1..{hard_max}")
        policy[key] = candidate
    for key, lower, upper in (("max_controller_retries", 0, 3),
                              ("retry_backoff_seconds", 1, 300),
                              ("max_retry_backoff_seconds", 1, 1800),
                              ("controller_retry_window_seconds", 60, 86400)):
        candidate = value.get(key, defaults[key])
        if type(candidate) is not int or not lower <= candidate <= upper:
            raise ValueError(f"execution policy {key} must be an integer in {lower}..{upper}")
        policy[key] = candidate
    if "quality_only" in value and type(value["quality_only"]) is not bool:
        raise ValueError("execution policy quality_only must be a boolean")
    policy["quality_only"] = value.get("quality_only", defaults["quality_only"])
    policy["per_plan"] = True
    return policy


def business_checkpoint_status(task, expected_cases=()):
    """Read business progress independently of controller or broker heartbeat."""
    task = Path(task)
    checkpoint_path = next((p for p in (task / "BUSINESS_CHECKPOINT.json", task / "CHECKPOINT.json",
                                        task / "repair/BUSINESS_CHECKPOINT.json") if p.is_file()), None)
    raw = read(checkpoint_path) if checkpoint_path else None
    raw_cases, checkpoint_field = checkpoint_cases(raw) if checkpoint_path else ({}, None)
    aliases = {"92b-mixed": "mixed", "same-host-pd-split": "same_host_pd_split",
               "multi-host-tp": "multi_host_tp", "cross-host-pd-split": "cross_host_pd_split"}
    raw_cases = {aliases.get(str(name), str(name)): value for name, value in raw_cases.items()}
    cases = {}
    submitted_request_ids = set()
    request_dir = task / "fleet/requests"
    if request_dir.is_dir():
        for path in request_dir.glob("*.json"):
            try:
                request = read(path)
            except (OSError, ValueError):
                continue
            if request.get("action") == "run":
                submitted_request_ids.add(request.get("request_id") or path.stem)
    names = sorted(set(expected_cases) | set(raw_cases))
    completed = {"completed", "passed", "accepted", "done"}
    running = {"running", "starting", "active"}
    pending = {"pending", "queued", "waiting", "deferred", "submitted"}
    failed = {"failed", "error", "aborted", "cancelled"}
    blocked = {"blocked", "resource_blocked", "waiting_resource"}
    for name in names:
        entry = raw_cases.get(name) or {}
        if not isinstance(entry, dict):
            entry = {"status": entry}
        status = str(entry.get("status", entry.get("phase", "pending"))).lower()
        request_id = entry.get("request_id") or entry.get("fleet_request_id")
        job_id = entry.get("job_id") or entry.get("job")
        submitted = bool(request_id or job_id or status in running)
        if status in completed:
            state = "completed"
        elif status in running:
            state = "running"
        elif status in failed:
            state = "failed"
        elif status in blocked:
            state = "blocked"
        elif status in pending:
            state = "pending"
        else:
            state = "unknown"
        cases[name] = {
            "state": state, "raw_status": status, "submitted": submitted,
            "request_id": request_id, "job_id": job_id,
            "experiment": entry.get("experiment") or entry.get("experiment_id"),
            "next_action": entry.get("next_action"),
            "resubmit_allowed": not submitted and state == "pending",
            "last_error": entry.get("last_error"),
        }
    return {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_field": checkpoint_field,
        "cases": cases,
        "submitted_request_ids": sorted(str(v) for v in submitted_request_ids),
        "controller_independent": True,
    }


def checkpoint_cases(raw):
    """Read stable ``cases`` and the narrow legacy/model ``four_types`` alias."""
    if not isinstance(raw, dict):
        raise ValueError("business checkpoint must be a JSON object")

    def normalize(value, field):
        if isinstance(value, list):
            normalized = {}
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"business checkpoint {field} list item must be an object")
                name = item.get("case") or item.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"business checkpoint {field} list item needs case or name")
                if name in normalized:
                    raise ValueError(f"business checkpoint {field} has duplicate case name: {name}")
                normalized[name] = item
            value = normalized
        if not isinstance(value, dict):
            raise ValueError(f"business checkpoint {field} must be an object or list")
        if any(not isinstance(name, str) or not name.strip() or not isinstance(item, dict)
               for name, item in value.items()):
            raise ValueError(f"business checkpoint {field} entries must be named objects")
        return value

    cases = normalize(raw["cases"], "cases") if "cases" in raw else None
    legacy = normalize(raw["four_types"], "four_types") if "four_types" in raw else None
    if cases:
        return cases, "cases"
    if legacy:
        return legacy, "four_types"
    if cases is not None:
        return cases, "cases"
    if legacy is not None:
        return legacy, "four_types"
    raise ValueError("business checkpoint must contain cases (or legacy four_types)")


def classify_controller_failure(error_type, detail=""):
    text = f"{error_type} {detail}".lower()
    if "timeout" in text or "timed out" in text or "readtimeout" in text:
        return "CONTROLLER_API_TIMEOUT"
    return "CONTROLLER_API_FAILURE"


def record_controller_failure(task, error_type, detail, controller_run=None, now=None,
                              controller_identity=None, resume_attempt=False,
                              resume_attempt_id=None):
    """Persist a bounded controller failure without touching business jobs."""
    task = Path(task)
    policy = _execution_policy(task)
    path = task / "repair/CONTROLLER_FAILURE.json"
    previous = read(path) if path.is_file() else {}
    observed = time.time() if now is None else float(now)
    failure_class = classify_controller_failure(error_type, detail)
    identity = controller_identity or controller_run
    same_observation = bool(identity and previous.get("controller_identity") == identity
                            and previous.get("failure_class") == failure_class)
    window_start = float(previous.get("retry_window_started_at", observed))
    expired = observed - window_start >= policy["controller_retry_window_seconds"]
    if expired:
        window_start = observed
    retry_count = 0 if expired else int(previous.get("retry_count", 0))
    resume_attempts = int(previous.get("resume_attempts", retry_count))
    counted_attempt = bool(resume_attempt and resume_attempt_id != previous.get("last_resume_attempt_id"))
    if counted_attempt:
        retry_count += 1
        resume_attempts += 1
    backoff = (min(policy["retry_backoff_seconds"] * (2 ** max(0, retry_count - 1)),
                   policy["max_retry_backoff_seconds"]) if counted_attempt
               else float(previous.get("backoff_seconds", 0)))
    value = {
        "status": "recorded",
        "failure_class": failure_class,
        "error_type": str(error_type), "detail": str(detail)[:3000],
        "controller_run": controller_run, "controller_identity": identity,
        "retry_count": retry_count, "resume_attempts": resume_attempts,
        "observation_count": int(previous.get("observation_count", 0)) + 1,
        "same_controller_observation": same_observation,
        "last_resume_attempt_id": (resume_attempt_id if counted_attempt
                                    else previous.get("last_resume_attempt_id")),
        "max_controller_retries": policy["max_controller_retries"],
        "retry_window_started_at": window_start,
        "controller_retry_window_seconds": policy["controller_retry_window_seconds"],
        "backoff_seconds": backoff,
        "next_retry_at": observed + backoff if counted_attempt else previous.get("next_retry_at", observed),
        "observed_at": observed,
    }
    write(path, value)
    return value


def resume_gate(task, controller_state, cleanup=None, now=None, business_observation=None):
    """Allow public run_task --resume only after strict controller cleanup."""
    task = Path(task)
    policy = _execution_policy(task)
    failure_path = task / "repair/CONTROLLER_FAILURE.json"
    failure = read(failure_path) if failure_path.is_file() else {}
    observed = time.time() if now is None else float(now)
    window_start = float(failure.get("retry_window_started_at", observed))
    retry_count = (0 if observed - window_start >= policy["controller_retry_window_seconds"]
                   else int(failure.get("retry_count", 0)))
    cleanup = cleanup or {}
    terminal = (isinstance(controller_state, dict)
                and type(controller_state.get("exit")) is int
                and controller_state.get("model") == "glm-5.3-flash"
                and controller_state.get("child_alive") is False
                and controller_state.get("fifo_removed") is True
                and controller_state.get("credentials_persisted") is False)
    remaining = cleanup.get("remaining_direct_children")
    clean = (controller_state.get("cleanup_complete") is True
             and isinstance(remaining, list) and remaining == []
             and cleanup.get("model_waited") is True)
    checkpoint = business_checkpoint_status(task)
    idle_verified = bool(business_observation) and all(
        v.get('observed') is True and not v.get('uncertain') and not v.get('active')
        for v in business_observation)
    if not terminal or not clean:
        status = "BLOCKED_CONTROLLER_NOT_STRICTLY_TERMINAL"
    elif not idle_verified and any(v["state"] == "running" for v in checkpoint["cases"].values()):
        status = "BLOCKED_BUSINESS_JOB_ACTIVE"
    elif not idle_verified and any(v["state"] in {"blocked", "unknown"} for v in checkpoint["cases"].values()):
        status = "BLOCKED_BUSINESS_STATE_UNCERTAIN"
    elif retry_count >= policy["max_controller_retries"]:
        status = "CONTROLLER_RETRY_BUDGET_EXHAUSTED"
    else:
        wait = max(0.0, float(failure.get("next_retry_at", 0)) - observed)
        status = "BACKOFF_ACTIVE" if wait else "RESUME_ALLOWED"
    return {
        "status": status, "resume_allowed": status == "RESUME_ALLOWED",
        "retry_count": retry_count,
        "max_controller_retries": policy["max_controller_retries"],
        "checkpoint": checkpoint, "failure": failure,
        "cleanup_verified": clean, "controller_terminal": terminal,
    }


def supervision_remote_jobs(management, task, plan):
    """Observe the registered personal containers, never local Docker on Windows."""
    import importlib.util
    config = read(task / 'fleet-config.local.json')
    if config['control_project'] != plan['remote_project']:
        raise ValueError('supervisor control project differs from the actual launch')
    registered = read(management / 'tools/ssh/server-inventory.json')['servers']
    allowed = {row['ip']: row for row in registered if not row.get('maintenance_forbidden')}
    spec = importlib.util.spec_from_file_location('supervisor_registered_ssh', management / 'runs/remote-model-agents-20261027/install_binary.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    views = []
    for host, target in config['targets'].items():
        if host not in allowed or target['management'] != allowed[host]['project_root']:
            raise ValueError('supervisor target is not registered')
        remote_root = PurePosixPath(target['project'])
        if '..' in remote_root.parts or not remote_root.is_relative_to(PurePosixPath(target['management']) / 'projects'):
            raise ValueError('supervisor target project is outside its registered personal projects')
        body = '''from pathlib import Path
import json
r=Path(ROOT);assert r.resolve(strict=False)==r,'project root is not canonical';b=r/'runs'/BATCH
if r.exists() and not r.is_dir():raise ValueError('project root is not a directory')
def load(p):
 return json.loads(p.read_text()) if p.is_file() else None
out={'observed':True,'project_exists':r.is_dir(),'jobs':[],'experiments':[],'active':False,'uncertain':False}
for j in sorted((b/'fleet-jobs').glob('*')):
 if not j.is_dir():continue
 state=load(j/'state.json');item={'job':j.name,'state':state}
 if state and state.get('status')=='running':
  pid=state.get('runner_pid');p=Path('/proc')/str(pid)
  try:
   argv=(p/'cmdline').read_bytes().split(b'\\0')
   alive=str(j/'runner.py').encode() in argv
   item['process_alive']=alive;out['active']|=alive
   if not alive:item['stale_running_record']=True
  except FileNotFoundError:item['process_alive']=False
  except OSError:out['uncertain']=True
 if state is None:out['uncertain']=True
 out['jobs'].append(item)
for e in sorted((r/'models').glob('*/experiments/'+BATCH+'*')):
 if not e.is_dir() or e.is_symlink() or not e.resolve().is_relative_to(r/'models'):
  out['uncertain']=True;continue
 p=load(e/'process.json');item={'model':e.parent.parent.name,'experiment':e.name,'state':load(e/'state.json')}
 if p and isinstance(p.get('pid'),int):
  proc=Path('/proc')/str(p['pid'])
  try:
   argv=(proc/'cmdline').read_bytes().split(b'\\0')
   expected=[str(x).encode() for x in p.get('argv',[])]
   live=bool(expected) and argv[:len(expected)]==expected
   item['service_alive']=live;out['active']|=live
   if not live and proc.exists() and (proc/'cmdline').read_bytes().strip(b'\\0'):out['uncertain']=True
  except FileNotFoundError:item['service_alive']=False
  except OSError:out['uncertain']=True
 out['experiments'].append(item)
print(json.dumps(out))
'''.replace('ROOT', repr(target['project'])).replace('BATCH', repr(task.name))
        try:
            client = helper.connect(allowed[host]['alias'])
            try:
                command = shlex.join(['docker', 'exec', '-i', target['container'], 'python3', '-B', '-'])
                rc, stdout, stderr = helper.call(client, command, body, timeout=45)
                if rc:
                    raise RuntimeError('registered container observation failed: ' + stderr[-500:])
                value = json.loads(stdout)
                value['host'] = host
                views.append(value)
            finally:
                client.close()
        except Exception as exc:
            views.append({'host': host, 'observed': False, 'uncertain': True,
                          'error': type(exc).__name__, 'detail': str(exc)[:500]})
    return views


def supervision_publish_context(management, task, plan, context):
    import importlib.util
    expected = str(PurePosixPath(plan['remote_project']) / 'runs' / task.name)
    if plan['business_run'] != expected:
        raise ValueError('supervisor context is outside the original remote business run')
    spec = importlib.util.spec_from_file_location('supervisor_context_ssh', management / 'runs/remote-model-agents-20261027/install_binary.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    client = helper.connect(plan['host'])
    try:
        with client.open_sftp() as sf:
            parent = expected + '/repair'
            if sf.normalize(parent) != parent:
                raise ValueError('supervisor context parent resolves outside the task')
            target = parent + '/SUPERVISOR_CONTEXT.json'
            temporary = target + '.tmp'
            with sf.open(temporary, 'w') as output:
                output.write(context.read_bytes())
            sf.posix_rename(temporary, target)
    finally:
        client.close()


def supervise_once(root, task, management, *, allow_resume=False,
                   call=None, observe=None, publish=None):
    """One hourly observation/recovery through the existing public launcher."""
    call = call or subprocess.run
    observe = observe or supervision_remote_jobs
    publish = publish or supervision_publish_context
    launcher = management / 'tools/claude/run_task.py'
    launch = read(task / 'GLM_LAUNCH.json')
    actual = read(task / 'glm-attempts' / launch['key'] / 'PLAN.json')
    result = call([sys.executable, '-B', '-X', 'utf8', str(launcher), '--status', str(task), '--refresh'],
                  cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=90)
    if result.returncode:
        raise RuntimeError('public status failed: ' + result.stderr[-700:])
    public = json.loads(result.stdout)
    snapshot = public.get('remote_snapshot', {})
    terminal, cleanup = snapshot.get('terminal'), snapshot.get('cleanup')
    # These files are copies of the live public observation, not invented defaults.
    if terminal is not None:
        write(task / 'repair/CONTROLLER_STATE.json', terminal)
    if cleanup is not None:
        write(task / 'repair/PROCESS_CLEANUP.json', cleanup)
    out = {'at': time.time(), 'controller': public, 'action': 'OBSERVED',
           'execution_surface': 'registered remote personal containers'}
    out['remote_business'] = observe(management, task, actual)
    if public.get('state') in {'TOOLS_RUNNING', 'SUBMITTED', 'DRIVER_STARTED_NOT_TOOL_VERIFIED', 'SUBMISSION_IN_PROGRESS'}:
        out['action'] = 'CONTROLLER_RUNNING'
    elif any(v.get('active') for v in out['remote_business']):
        out['action'] = 'BUSINESS_RUNNING'
    elif not out['remote_business'] or any(not v.get('observed') or v.get('uncertain') for v in out['remote_business']):
        out['action'] = 'BLOCKED_REMOTE_OBSERVATION'
    else:
        # An actual remote observation supersedes historical checkpoint status.
        failure = public.get('failure') or {}
        detail = failure.get('class', public.get('state', 'unknown'))
        completed_normally = (public.get('state') == 'MODEL_COMPLETED'
                              and isinstance(terminal, dict) and terminal.get('exit') == 0)
        if completed_normally:
            write(task / 'repair/CONTROLLER_COMPLETION.json', {
                'controller_run': launch['remote_run'], 'exit': 0,
                'business_acceptance': 'NOT_INFERRED_FROM_CONTROLLER',
                'observed_at': time.time(),
            })
        elif terminal:
            record_controller_failure(task, detail, detail, launch['remote_run'],
                                      controller_identity=launch['remote_run'])
        gate = resume_gate(task, terminal or {}, cleanup or {}, business_observation=out['remote_business'])
        out['gate'] = gate
        if gate['resume_allowed'] and allow_resume:
            base = task / 'repair/SUPERVISOR_BASE_PROMPT.txt'
            if not base.exists():
                base.write_text(owned(root, actual['prompt_file']).read_text(encoding='utf-8'), encoding='utf-8')
            context = task / 'repair/SUPERVISOR_CONTEXT.json'
            write(context, out)
            publish(management, task, actual, context)
            prompt = task / 'repair/SUPERVISOR_RESUME_PROMPT.txt'
            prefix = ('这是每小时自动续接。先读取本次上下文：' + str(actual['business_run']) +
                      '/repair/SUPERVISOR_CONTEXT.json。核对已有请求、真实job和实验后只继续未完成步骤。'
                      '不要照抄旧说明里的首次提交命令、request ID或旧空闲卡；旧407记录不代表当前网络，'
                      '本次公共启动器会做实时网络预检。已有任务与结果保留，不重复启动。\n\n')
            prompt.write_text(prefix + base.read_text(encoding='utf-8'), encoding='utf-8')
            argv = [sys.executable, '-B', '-X', 'utf8', str(launcher), '--host', actual['host'],
                    '--project', str(root), '--task', task.name, '--prompt-file', str(prompt),
                    '--remote-project', actual['remote_project'], '--resume']
            prior_argv = actual.get('argv', [])
            handoff = actual.get('handoff_file')
            if not handoff and '--handoff-file' in prior_argv:
                handoff = prior_argv[prior_argv.index('--handoff-file') + 1]
            if handoff:
                argv.extend(['--handoff-file', str(owned(root, handoff))])
            resumed = call(argv, cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=120)
            out['resume_exit'] = resumed.returncode
            out['resume_output'] = resumed.stdout[-4000:]
            out['action'] = 'RESUME_SUBMITTED' if resumed.returncode == 0 else 'RESUME_REJECTED'
            if resumed.returncode == 0 and not completed_normally:
                current = read(task / 'GLM_LAUNCH.json')
                record_controller_failure(task, detail, detail, launch['remote_run'],
                                          controller_identity=launch['remote_run'], resume_attempt=True,
                                          resume_attempt_id=current['key'])
            elif resumed.returncode != 0:
                out['resume_error'] = resumed.stderr[-1500:]
        elif gate['resume_allowed']:
            out['action'] = 'RESUME_AVAILABLE_OBSERVE_ONLY'
        else:
            out['action'] = gate['status']
    write(task / 'HOURLY_SUPERVISOR.json', out)
    return out


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
    policy = _execution_policy(task)
    records = {v.get("task"): v for v in current["cases"]}
    classifications = {}
    classification_path = task / "repair/RUN_CLASSIFICATIONS.json"
    if classification_path.exists():
        try:
            classifications = read(classification_path)
        except (ValueError, OSError):
            classifications = {}

    def runtime_failure(record):
        """Only explicit resource conflicts are excluded from retry budgets.
        Classifications live in RUN_CLASSIFICATIONS.json with their evidence;
        communication and execution failures remain counted."""
        if record.get("job-state.json", {}).get("status") != "failed":
            return False
        entry = classifications.get(record.get("task") or "") or {}
        if entry.get("classification") == "resource":
            return False
        detail = str(record.get("validation.json", {}).get("detail", ""))
        if any(tag in detail for tag in ("target device is occupied", "port already in use",
                                         "insufficient healthy empty devices")):
            return False
        return True
    allowed = {"tp", "dp", "graph", "mtp", "max_batched_tokens", "max_seqs", "disable_aligned_decode", "pd_mode", "reason", "quality_only"}
    if set(plan) - allowed:
        raise ValueError("unexpected plan fields")
    if "quality_only" in plan and type(plan["quality_only"]) is not bool:
        raise ValueError("quality_only must be a boolean")
    quality_only = bool(plan.get("quality_only", False) or policy.get("quality_only", False))
    tp, dp = plan.get("tp", 2), plan.get("dp", 1)
    if (tp, dp) not in {(2, 1), (4, 1), (2, 2), (1, 4)}:
        raise ValueError("unsupported parallel plan")
    graph, mtp = plan.get("graph", "eager"), plan.get("mtp", 0)
    batched, seqs = plan.get("max_batched_tokens", 4096), plan.get("max_seqs", 8)
    if graph not in {"eager", "full_decode_only"} or mtp not in {0, 3}:
        raise ValueError("unsupported graph or MTP plan")
    if type(batched) is not int or batched not in {512, 4096, 10240, 16384} or type(seqs) is not int or seqs not in {4, 8, 32, 56}:
        raise ValueError("unsupported scheduling limits")
    pd_mode = plan.get("pd_mode")
    if pd_mode not in {None, "mixed", "split"}:
        raise ValueError("unsupported pd_mode plan")
    if type(plan.get("disable_aligned_decode", False)) is not bool or not isinstance(plan.get("reason"), str) or not plan["reason"].strip():
        raise ValueError("configuration change needs a reason and a boolean alignment setting")
    if len(ledger["cases"]) >= policy["max_total_jobs"]:
        raise ValueError("recovery total-job budget exhausted; retain evidence for main-agent review")
    if policy["per_plan"]:
        plan_key = _normalized_plan_key(plan)
        failures_by_plan = {}
        for entry in ledger["cases"]:
            record = records.get(entry.get("task"))
            if record is not None and runtime_failure(record):
                key = _normalized_plan_key(entry.get("plan"))
                failures_by_plan[key] = failures_by_plan.get(key, 0) + 1
        if failures_by_plan.get(plan_key, 0) >= policy["max_failures_per_plan"]:
            raise ValueError("recovery plan failure budget exhausted; use a different normalized plan")
    else:
        failures = sum(runtime_failure(record) for record in current["cases"])
        if failures >= policy["max_failures_per_plan"]:
            raise ValueError("recovery budget exhausted; retain evidence for main-agent review")
    # Same-host PD split launches TWO independent instances (producer +
    # consumer), so the device budget doubles; this mirrors the
    # workers_needed check inside run_pangu92b.py, which otherwise rejects
    # the job at startup ("project or worker count mismatch").
    workers_needed = tp * dp * (2 if pd_mode == "split" else 1)
    cards = inventory(root, task)["available"]
    if len(cards) < workers_needed:
        write(task / "blocked.json", {"reason": "insufficient healthy empty devices", "required": workers_needed, "available": cards})
        raise RuntimeError("insufficient healthy empty devices; no process was stopped")
    cards = cards[:workers_needed]
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
    if quality_only:
        args.append("--quality-only")
    # Optional task-scoped HCCL/OMNI environment overrides: applied only when
    # this task owns repair/env-overrides.json; run_pangu92b filters the keys.
    override = task / "repair/env-overrides.json"
    if override.is_file():
        args += ["--env-override-file", str(override)]
    if pd_mode:
        args += ["--pd-mode", pd_mode]
    if plan.get("disable_aligned_decode", False):
        args.append("--disable-aligned-decode")
    body = "set -e; source " + shlex.quote(config["cann_source"]) + "; exec " + shlex.join(args)
    command = ["docker", "exec", "--user", "0", "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONPATH=" + str(root / "src"), "-e", "FULLNET_CA_BUNDLE=" + str(root / "cache/trust/workstation-ca.pem"),
               config["container"], "bash", "--noprofile", "--norc", "-c", body]
    write(run / "requested-job.json", {"argv": command})
    recorded_plan = dict(plan)
    recorded_plan["quality_only"] = quality_only
    write(run / "recovery-plan.json", {"plan": recorded_plan, "devices": cards, "runtime_physical_ids": runtime_cards})
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


def operator_update(task):
    """Surface a task-local steering note without restarting a running agent."""
    path = Path(task) / 'repair/OPERATOR_UPDATE.json'
    if not path.exists():
        return None
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(Path(task).resolve(strict=True)):
        raise ValueError('operator update must stay inside the task')
    if path.stat().st_size > 8192:
        raise ValueError('operator update exceeds 8192 bytes')
    value = read(path)
    if (not isinstance(value, dict) or set(value) - {'id', 'instructions', 'paths'}
            or not isinstance(value.get('id'), str) or not isinstance(value.get('instructions'), str)):
        raise ValueError('invalid operator update')
    return value


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
    # Preserve the distinction between an omitted --devices and an explicit
    # []: a run without a device contract must fail before an active-owner
    # check or request-file write can have a queueing side effect.
    if action == 'run' and devices is None:
        raise ValueError('fleet run requires an explicit host NPU ID list; pass [] only for a device-free script job')
    # A native executor may own an exclusive device case while the GLM is
    # still alive.  Keep status/inventory and other hosts usable, but stop a
    # stale controller from queueing a competing device run on that host.
    # This is intentionally opt-in: without explicit exclusive_active and
    # exclusive_host fields the existing fleet behavior is unchanged.
    if action == 'run':
        active_path = task / 'repair' / 'ACTIVE_BUSINESS.json'
        if active_path.is_file():
            active = read(active_path)
            if (isinstance(active, dict)
                    and active.get('exclusive_active') is True
                    and isinstance(active.get('owner'), str)
                    and bool(active.get('owner').strip())
                    and isinstance(active.get('exclusive_host'), str)
                    and active.get('exclusive_host') == host):
                existing = active.get('case_id') or active.get('job') or active.get('request_id') or 'recorded active case'
                raise RuntimeError(
                    'fleet run refused for exclusive host %s: owner=%s case/job=%s; '
                    'read %s and reuse the active case/job instead of submitting another run'
                    % (host, active['owner'], existing, active_path)
                )
    devices = [] if devices is None else devices
    if (not isinstance(devices, list) or len(set(devices)) != len(devices) or
            any(type(value) is not int or not 0 <= value < 32 for value in devices)):
        raise ValueError('invalid fleet NPU ID list')
    args = [] if args is None else args
    if not isinstance(args, list) or any(not isinstance(value, str) or '\x00' in value or len(value) > 4096 for value in args):
        raise ValueError('fleet args must be bounded strings')
    script_name = None
    if action == 'run':
        # An explicit [] is a declared device-free job (e.g. container network
        # probes); omission was rejected above before any queue side effect.
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
    if action == 'run':
        record['framework_source'] = framework_source_receipt(root, task)
    if asset is not None:
        record['asset'] = asset
    steering = operator_update(task)
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
            'host': record['host'], 'action': action, 'job': job,
            'response_path': str(task / 'fleet/responses' / (request_id + '.json')),
            'business_run': str(task), 'business_acceptance': 'NOT_VALIDATED',
            'operator_update': steering,
            'next_action': ('Read this request response, then submit a new status request for this host/job; '
                            'a run receipt is not a job result' if action == 'run'
                            else 'Read the absolute response_path; do not assume the shell working directory')}


def fleet_job_outcome(action, response):
    """Separate queueing, execution and business acceptance for AI callers."""
    if not isinstance(response, dict):
        return {'execution': 'unknown', 'business_acceptance': 'NOT_VALIDATED'}
    values = {}
    if response.get('error'):
        execution = 'broker_failed'
    elif action == 'run':
        execution = 'submitted_not_validated'
    elif action not in {'status', 'cancel'}:
        execution = 'not_a_job_status'
    else:
        for name in ('state.json', 'failure.json'):
            value = response.get(name)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = None
            if isinstance(value, dict) and not value.get('truncated'):
                values[name] = value
        state = values.get('state.json', {})
        exit_code = state.get('exit')
        raw_exit = response.get('exit.txt')
        if exit_code is None and isinstance(raw_exit, str):
            try:
                exit_code = int(raw_exit.strip())
            except ValueError:
                pass
        if values.get('failure.json') or (type(exit_code) is int and exit_code != 0):
            execution = 'failed'
        elif state.get('status') == 'exited' and type(exit_code) is int and exit_code == 0:
            execution = 'succeeded'
        elif state.get('status') == 'running':
            execution = 'running'
        else:
            execution = 'unknown'
    invalid = {name: meta.get('json_error', 'invalid JSON')
               for name, meta in response.get('file_metadata', {}).items()
               if isinstance(meta, dict) and meta.get('json_valid') is False}
    outcome = {'execution': execution, 'business_acceptance': 'NOT_VALIDATED',
            'artifact_validation': 'failed' if invalid else 'not_inferred',
            'invalid_json_files': invalid}
    failure = values.get('failure.json', {})
    detail = str(failure.get('detail', ''))
    if ('device lease busy:' in detail and not response.get('process.json')):
        outcome.update(failure_layer='device_resource_gate', workload_execution='not_started',
                       reason='device_lease_busy',
                       next_action='Wait for the current project lease holder to release the device or select another confirmed idle device; do not delete lease/lock files or kill another task')
    elif ('is not confirmed healthy and empty' in detail and not response.get('process.json')):
        outcome.update(failure_layer='device_resource_gate', workload_execution='not_started',
                       reason='healthy_empty_gate',
                       next_action='Refresh inventory and select other confirmed healthy idle devices or another registered host; do not retry the same stale selection')
    return outcome


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
                'status': 'pending', 'request': str(request_path),
                'response_path': str(response_path.resolve()), 'response': {},
                'job_outcome': {'execution': 'unknown', 'business_acceptance': 'NOT_VALIDATED'},
                'operator_update': operator_update(task)}
    if response_path.stat().st_size > 1_000_000:
        raise ValueError('fleet response exceeds read limit')
    response = read(response_path)
    status = 'failed' if 'error' in response else 'received'
    return {'request_id': request_id, 'job': request.get('job'), 'action': request.get('action'),
            'status': status, 'request': str(request_path), 'response_path': str(response_path),
            'response': response, 'job_outcome': fleet_job_outcome(request.get('action'), response),
            'operator_update': operator_update(task)}


def fleet_inventory_request(root, task, request_id, host, job):
    return fleet_submit(root, task, request_id, 'inventory', host, job, devices=[], args=[])


def fleet_job_status(root, task, host, job, wait_seconds=10):
    """Fetch actual job files through the broker with one explicit operation.

    Reuse an unanswered status request and a valid immutable terminal snapshot;
    running, failed-query, incomplete, or old-schema snapshots get a new
    request. Never read a run submission receipt as if it were the job state.
    """
    identifier(job)
    pending = []
    terminal = []
    directory = task / 'fleet/requests'
    if directory.is_dir():
        directory = owned(root, directory)
        responses = directory.parent / 'responses'
        for path in directory.glob('*.json'):
            if path.is_symlink() or path.stat().st_size > 131072:
                continue
            try:
                value = read(path)
            except (OSError, ValueError):
                continue
            if (value.get('action') == 'status' and value.get('host') == host
                    and value.get('job') == job):
                response_path = responses / path.name
                if not response_path.is_file():
                    pending.append((path.stat().st_mtime_ns, value['request_id']))
                    continue
                try:
                    if response_path.is_symlink() or response_path.stat().st_size > 1_000_000:
                        continue
                    response_path = owned(root, response_path)
                    if response_path.is_symlink() or response_path.stat().st_size > 1_000_000:
                        continue
                    response = read(response_path)
                    state_metadata = response.get('file_metadata', {}).get('state.json', {})
                    state_text = response.get('state.json')
                    state_value = json.loads(state_text) if isinstance(state_text, str) else None
                    terminal_state = (isinstance(state_metadata, dict)
                                      and state_metadata.get('json_valid') is True
                                      and state_metadata.get('truncated') is not True
                                      and isinstance(state_value, dict)
                                      and state_value.get('status') == 'exited'
                                      and type(state_value.get('exit')) is int)
                    if terminal_state:
                        terminal.append((response_path.stat().st_mtime_ns, path, response_path, response))
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
    if terminal:
        _, request_path, response_path, response = max(terminal, key=lambda value: value[0])
        result = {'request_id': request_path.stem, 'job': job, 'action': 'status',
                  'status': 'received', 'request': str(request_path.resolve()),
                  'response_path': str(response_path.resolve()), 'response': response,
                  'job_outcome': fleet_job_outcome('status', response),
                  'business_run': str(task), 'observation': 'cached_terminal_job_result',
                  'snapshot_at': response.get('broker_checked_at'),
                  'operator_update': operator_update(task)}
        result['observation'] += '; not a fresh device-idle proof'
        if result.get('operator_update'):
            result['job_outcome']['operator_update'] = result['operator_update']
        return result
    if pending:
        request_id = max(pending)[1]
    else:
        request_id = 'job-status-' + str(time.time_ns())
        fleet_submit(root, task, request_id, 'status', host, job, devices=[], args=[])
    result = fleet_status(root, task, request_id, wait_seconds)
    result['business_run'] = str(task)
    result['observation'] = ('pending_status_query; not a job-queued result; never a run receipt'
                             if result.get('status') == 'pending'
                             else 'fresh_job_status; never a run receipt')
    if result.get('operator_update'):
        # Compact callers often display only job_outcome. Keep live steering
        # visible there as well rather than losing it in a wrapper they omit.
        result.setdefault('job_outcome', {'execution': 'unknown',
                                          'business_acceptance': 'NOT_VALIDATED'})[
            'operator_update'] = result['operator_update']
    return result


def fleet_request_status(root, task, request_id, wait_seconds=10):
    """CLI status follows a successful run receipt to its actual job status."""
    result = fleet_status(root, task, request_id, min(wait_seconds, 60))
    response = result.get('response')
    if (result.get('action') == 'run' and isinstance(response, dict)
            and response.get('submitted') is True and not response.get('error')):
        request = read(owned(root, task / 'fleet/requests' / (identifier(request_id) + '.json')))
        observed = fleet_job_status(root, task, request['host'], request['job'], min(wait_seconds, 60))
        observed['run_request_id'] = request_id
        observed['submission_response_path'] = result['response_path']
        return observed
    return result


def fleet_inventory_batch(root, task, hosts, *, request_prefix=None):
    """A fresh scan must not reuse a previous scan's cached responses."""
    if not hosts:
        raise ValueError('fleet-inventory requires --fleet-hosts')
    prefix = request_prefix or f'{task.name[:20]}-inv-{time.time_ns()}'
    request_ids = [identifier(f'{prefix}-{index:02d}') for index in range(len(hosts))]
    return {'requests': [fleet_inventory_request(root, task, request_id, host, task.name)
                         for request_id, host in zip(request_ids, hosts)]}


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


def compact_job_status(result, host):
    """Render existing facts and pointers without logs or full job artifacts.

    Keep this separate from the legacy outcome/default renderer: an exit-zero
    snapshot with incomplete structured evidence must not look successful in
    the compact view. The original response remains untouched on disk.
    """
    response = result.get('response')
    response = response if isinstance(response, dict) else {}
    outcome = dict(result.get('job_outcome') or fleet_job_outcome('status', response))
    metadata = response.get('file_metadata', {})
    parsed, incomplete, invalid = {}, {}, dict(outcome.get('invalid_json_files', {}))
    for name in ('state.json', 'failure.json', 'result.json', 'validation.json', 'process.json'):
        value = response.get(name)
        info = metadata.get(name, {})
        if value is None:
            if name == 'state.json' or name in metadata:
                incomplete[name] = 'file missing from response'
            continue
        if info.get('truncated') or (isinstance(value, dict) and value.get('truncated')):
            incomplete[name] = 'truncated; read original file via response_path'
            continue
        try:
            value = json.loads(value) if isinstance(value, str) else value
        except ValueError as exc:
            invalid[name] = info.get('json_error', str(exc))
            continue
        if info.get('json_valid') is False or not isinstance(value, dict):
            invalid[name] = info.get('json_error', 'expected a JSON object')
            continue
        parsed[name] = value
    state = parsed.get('state.json', {})
    if state:
        outcome['process_status'] = state.get('status')
        outcome['process_exit'] = state.get('exit')
    verified_terminal = (state.get('status') == 'exited' and type(state.get('exit')) is int
                         and state['exit'] == 0
                         and metadata.get('state.json', {}).get('json_valid') is True)
    if outcome.get('execution') == 'succeeded' and (
            not verified_terminal or incomplete or invalid or result.get('status') == 'pending'):
        outcome['execution'] = 'unknown'
        outcome['evidence_status'] = 'incomplete_or_invalid; read response_path'
    if incomplete:
        outcome['incomplete_files'] = incomplete
    if invalid:
        outcome.update(artifact_validation='failed', invalid_json_files=invalid)
    if not outcome.get('failure_layer'):
        for name in ('failure.json', 'result.json', 'validation.json'):
            layer = parsed.get(name, {}).get('failure_layer')
            if isinstance(layer, str) and layer:
                outcome['failure_layer'] = layer
                break

    # Only report explicitly recorded errors; never infer a root cause from a
    # log tail or substitute a wrapper's exit code for the model's first error.
    if not outcome.get('first_error'):
        for name in ('result.json', 'failure.json', 'validation.json'):
            value = parsed.get(name, {})
            error = value.get('first_error') or value.get('raw_error')
            if not error and (value.get('error') or (name == 'failure.json' and value.get('detail'))):
                error = value
            if not error:
                continue
            if not isinstance(error, dict):
                error = {'detail': error}
            fact = {'source_file': name, 'read_path': result.get('response_path')}
            for key in ('error', 'type', 'detail', 'message', 'phase', 'path', 'line'):
                if key in error:
                    value = error[key]
                    if isinstance(value, str) and len(value) > 2048:
                        fact[key] = value[:2048]
                        fact.setdefault('truncated_fields', []).append(key)
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        fact[key] = value
            outcome['first_error'] = fact
            break
    if response.get('error') and not outcome.get('first_error'):
        detail = str(response['error'])
        outcome['first_error'] = {'source_file': 'broker_response',
                                 'read_path': result.get('response_path'),
                                 'detail': detail[:2048]}
        if len(detail) > 2048:
            outcome['first_error']['truncated_fields'] = ['detail']
    keys = ('status', 'observation', 'request_id', 'job', 'action', 'request',
            'response_path', 'business_run', 'run_request_id',
            'submission_response_path', 'snapshot_at', 'readable_server_standards')
    compact = {key: result[key] for key in keys if key in result}
    compact.update(job_outcome=outcome, host=host,
                   operator_update=result.get('operator_update', outcome.get('operator_update')),
                   response_omitted='Full original JSON is preserved at response_path')
    return compact


def order_cli_result(result):
    """Put compact status facts first while preserving every original value."""
    if not isinstance(result, dict):
        result = {'result': result}
    summary_keys = (
        'job_outcome', 'operator_update', 'status', 'observation', 'response_path',
        'request_id', 'job', 'action', 'business_run', 'run_request_id',
        'submission_response_path',
    )
    ordered = {key: result[key] for key in summary_keys if key in result}
    ordered.update((key, value) for key, value in result.items()
                   if key not in ordered and key != 'response')
    if 'response' in result:
        ordered['response'] = result['response']
    return ordered


def main():
    # CLI JSON is UTF-8 even when the parent shell selects a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, 'reconfigure', None)):
            stream.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--action", choices=("inspect", "source", "preflight", "start", "exec", "wait", "finish", "fleet-submit", "fleet-status", "fleet-job-status", "fleet-inventory", "fleet-broker", "controller-failure", "recovery-status", "supervise", "matrix"), required=True)
    p.add_argument('--management-root', type=Path)
    p.add_argument('--resume-controller', action='store_true')
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
    p.add_argument('--failure-type')
    p.add_argument('--detail')
    p.add_argument('--controller-run')
    p.add_argument('--controller-identity')
    p.add_argument('--resume-attempt', action='store_true')
    p.add_argument('--resume-attempt-id')
    p.add_argument('--controller-state', type=Path)
    p.add_argument('--cleanup-file', type=Path)
    p.add_argument('--retry', action='store_true')
    p.add_argument('--compact', action='store_true',
                   help='Only for fleet-job-status: print facts and original-file pointers, not full response')
    a = p.parse_args()
    if a.compact and a.action != 'fleet-job-status':
        p.error('--compact is only supported for --action fleet-job-status')
    expected_root = Path(__file__).resolve().parents[2]
    try:
        root = a.project.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ValueError(f"invalid --project: expected existing project root {expected_root}") from exc
    if root != expected_root:
        raise ValueError(f"invalid --project: expected project root {expected_root}; got {root}")
    task_identifier_pattern = r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}"
    if not re.fullmatch(task_identifier_pattern, a.task):
        candidate = str(a.task).replace("\\", "/").rstrip("/").split("/")[-1]
        suggestion = (f"--task {candidate}" if re.fullmatch(task_identifier_pattern, candidate)
                      else "--task <task-id>")
        raise ValueError(
            "invalid --task: expected a task identifier, not a path; "
            f"use {suggestion}"
        )
    task_path = root / "runs" / a.task
    if not task_path.is_dir():
        raise ValueError(
            f"invalid --task: task directory does not exist for identifier {a.task!r}; "
            "pass an existing task identifier"
        )
    task = owned(root, task_path)
    (task / "repair").mkdir(exist_ok=True)
    if a.action == 'supervise':
        if not a.management_root:
            raise ValueError('supervise requires --management-root for the existing registered launcher')
        result = supervise_once(root, task, a.management_root.resolve(strict=True), allow_resume=a.resume_controller)
    elif a.action == "matrix":
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
        result = fleet_request_status(root, task, a.request_id, a.wait_seconds)
    elif a.action == 'fleet-job-status':
        if not a.host or not a.job:
            raise ValueError('fleet-job-status requires --host and --job')
        result = fleet_job_status(root, task, a.host, a.job, min(a.wait_seconds, 60))
    elif a.action == 'fleet-inventory':
        result = fleet_inventory_batch(root, task, a.fleet_hosts or [], request_prefix=a.request_id)
    elif a.action == 'fleet-broker':
        if not a.broker_action:
            raise ValueError('fleet-broker requires --broker-action')
        result = fleet_broker(root, task, a.broker_action)
    elif a.action == 'controller-failure':
        if not a.failure_type or not a.detail:
            raise ValueError('controller-failure requires --failure-type and --detail')
        if a.resume_attempt and not a.resume_attempt_id:
            raise ValueError('actual resume attempt requires --resume-attempt-id')
        if a.resume_attempt_id:
            identifier(a.resume_attempt_id)
        result = record_controller_failure(task, a.failure_type, a.detail, a.controller_run,
                                          controller_identity=a.controller_identity,
                                          resume_attempt=a.resume_attempt,
                                          resume_attempt_id=a.resume_attempt_id)
    elif a.action == 'recovery-status':
        state_path = a.controller_state or (task / 'repair/CONTROLLER_STATE.json')
        cleanup_path = a.cleanup_file or (task / 'repair/PROCESS_CLEANUP.json')
        controller_state = read(owned(task, state_path)) if state_path.is_file() else {}
        cleanup = read(owned(task, cleanup_path)) if cleanup_path.is_file() else {}
        result = resume_gate(task, controller_state, cleanup)
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
    # matrix fleet-inventory/fleet-status return request lists; normalize so
    # the standards attachment never corrupts an already-successful submission.
    if not isinstance(result, dict):
        result = {"result": result}
    result['readable_server_standards'] = standards(root, task)
    if a.compact:
        result = compact_job_status(result, a.host)
    result = order_cli_result(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
