"""Foreground supervisor for a project tmux task; owns only its process group."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import threading
import tempfile
from contextlib import contextmanager
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import runtime
from .evaluation import evaluate, headers, smoke_request, validate_samples
from .store import RunLock, append_json, experiment, inside, read_json, rows, state, utc, write_json


def profiler_deferred(c):
    """Whether this acceptance stage intentionally defers profiler setup."""
    return "profile" in set(c.get("measurements", {}).get("deferred_stages", []))


@contextmanager
def graceful_termination():
    """Let the supervisor's finally block release its owned service on TERM."""
    installed = os.name == "posix" and threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if installed else None
    def terminate(signum, frame):
        raise InterruptedError("experiment supervisor received SIGTERM")
    if installed:
        signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


def plan(root, c, run_id):
    root = Path(root).resolve()
    run = experiment(root, c["model"], run_id)
    c = runtime(c, root, run)
    source = Path(c["evaluation"]["dataset"]).resolve(strict=True)
    samples = rows(source)
    validate_samples(samples, c["evaluation"])
    # Frozen configs/datasets are owned by this experiment. Existing runs never overwritten.
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()
    write_json(run / "config.json", c)
    write_json(run / "changes.json", c["resolved_changes"])
    with (run / "dataset.jsonl").open("x", encoding="utf-8") as output:
        for row in samples:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    state(run, "plan", "passed", model=c["model"], platform=c["platform"], samples=len(samples))
    return run


def private_environment(run, c, root=None):
    env = os.environ.copy()
    env.update(c.get("env", {}))
    if profiler_deferred(c):
        for key in (
            "VLLM_TORCH_PROFILER_DIR",
            "VLLM_TORCH_PROFILER_RECORD_SHAPES",
            "VLLM_TORCH_PROFILER_WITH_STACK",
            "VLLM_TORCH_PROFILER_WITH_FLOPS",
            "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY",
        ):
            env.pop(key, None)
    paths = {"HOME": "home", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
             "XDG_CONFIG_HOME": "home/config", "XDG_CACHE_HOME": "cache", "PIP_CACHE_DIR": "cache/pip",
             "TORCH_EXTENSIONS_DIR": "cache/torch-extensions", "TRITON_CACHE_DIR": "cache/triton",
             "CUDA_CACHE_PATH": "cache/cuda", "HF_HOME": "cache/huggingface", "HF_MODULES_CACHE": "cache/hf-modules",
             "HF_HUB_CACHE": "cache/huggingface/hub", "HF_XET_CACHE": "cache/huggingface/xet",
             "ASCEND_CACHE_PATH": "cache/ascend", "ASCEND_WORK_PATH": "tmp/ascend",
             "ASCEND_PROCESS_LOG_PATH": "logs/ascend", "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
             "VLLM_CACHE_ROOT": "cache/vllm", "PYTHONPYCACHEPREFIX": "cache/python"}
    for key, relative in paths.items():
        path = inside(run, Path(run) / relative)
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # Unix IPC has a byte-length limit. Deep experiment directories and Chinese
    # project names can exceed it even when normal file operations succeed.
    project = Path(root) if root is not None else Path(run).parents[3]
    ipc_parent = inside(project, project / '.i')
    if c['platform'] != 'cpu-test' and len(str(ipc_parent).encode('utf-8')) + 9 + 37 > 107:
        raise ValueError('project root is too long for Unix IPC; use a shorter project directory')
    ipc_parent.mkdir(parents=True, exist_ok=True)
    ipc = tempfile.mkdtemp(prefix='', dir=ipc_parent)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'VLLM_RPC_BASE_PATH'):
        env[key] = ipc
    if c["platform"] in ("a3", "a5"):
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(v) for v in c["devices"])
    env["FULLNET_RUN"] = str(run)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_port(c):
    url = urllib.parse.urlparse(c["service"]["base_url"])
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or not url.port:
        raise ValueError("owned service base_url must use explicit loopback http port")
    with socket.socket() as probe:
        if probe.connect_ex((url.hostname, url.port)) == 0:
            raise ValueError("port already in use; refusing to connect to an unrelated service")


def stop_owned(process, timeout):
    if process is None:
        return
    if os.name == "posix":
        # Child workers can outlive a service root; signal our created group even if root exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    process.wait(timeout=timeout)


def service_identity(c):
    """Optional non-root uid for the service process only.

    The CANN profiler analysis stage (msprof) refuses to run when its euid
    differs from the CANN file owner (a root service on a non-root-owned CANN
    tree can make the export stage produce a CPU-only trace; the export stage still
    produced a CPU-only trace). The supervisor stays root for device guards and
    cleanup, while the service and its profiler children run as the owner.
    """
    spec = c["service"].get("uid")
    if not spec or os.name != "posix" or not hasattr(os, "getuid") or os.getuid() != 0:
        return None
    import pwd
    record = pwd.getpwuid(int(spec))

    def apply():
        os.setgroups([record.pw_gid])
        os.setgid(record.pw_gid)
        os.setuid(record.pw_uid)
    return apply


def _yield_service_paths(run, c, env):
    """Hand the service-writable trees to the service uid before launch."""
    spec = c["service"].get("uid")
    if not spec or os.name != "posix" or not hasattr(os, "getuid") or os.getuid() != 0:
        return
    import pwd
    record = pwd.getpwuid(int(spec))
    run_root = Path(run).resolve(strict=True)
    project = run_root.parents[3]
    ipc_root = (project / ".i").resolve(strict=True)
    ipc = Path(env.get("VLLM_RPC_BASE_PATH", ""))
    roots = [(run_root, run_root)]
    if ipc:
        try:
            ipc_real = ipc.resolve(strict=True)
            if ipc_real.is_relative_to(ipc_root) and ipc_real.parent == ipc_root:
                roots.append((ipc_real, ipc_root))
        except (OSError, RuntimeError):
            pass
    for base, boundary in roots:
        stack = [base]
        while stack:
            path = stack.pop()
            try:
                if path.is_symlink() or not path.resolve(strict=True).is_relative_to(boundary):
                    continue
                os.chown(path, record.pw_uid, record.pw_gid, follow_symlinks=False)
                if path.is_dir():
                    stack.extend(child for child in path.iterdir() if not child.is_symlink())
            except (OSError, RuntimeError):
                continue


def run_experiment(root, model, run_id, *, after_evaluation=None, keep_ready=False):
    run = experiment(root, model, run_id)
    c = read_json(run / "config.json")
    current = read_json(run / "state.json")
    if current["phase"] != "plan":
        raise ValueError("this experiment has already started; inspect state and create a new run")
    process = None
    with graceful_termination(), RunLock(run):
        state(run, "execution", "running")
        try:
            check_port(c)
            env = private_environment(run, c, root)
            (run / "logs").mkdir(exist_ok=True)
            for change in c["resolved_changes"]:
                argv = change["spec"]["verify_argv"]
                append_json(run / "commands.jsonl", {"phase": "change_verify", "argv": argv, "at": utc()})
                with (run / "logs" / (change["name"] + "-verify.log")).open("w", encoding="utf-8") as output:
                    verified = subprocess.run(argv, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=120)
                if verified.returncode:
                    raise RuntimeError("change verification failed: " + change["name"])
            argv = c["service"]["argv"]
            append_json(run / "commands.jsonl", {"phase": "service", "argv": argv, "at": utc()})
            preexec = service_identity(c)
            _yield_service_paths(run, c, env)
            if profiler_deferred(c):
                tools = {"env_updates": {}}
                write_json(run / "performance" / "profiler-tools.json", {
                    "acceptance": "deferred_by_user",
                    "prepared_before_launch": False,
                    "reason": "quality-only stage defers profile and performance",
                    "env_updates": {},
                })
                state(run, "profile", "deferred_by_user", reason="quality-only stage")
                state(run, "performance", "deferred_by_user", reason="quality-only stage")
            else:
                # Deferred import: measurements imports private_environment from here.
                from .measurements import prepare_profiler_tools
                # Profiler children are spawned by the running service, so the
                # msprof toolchain (shadow PATH + sqlite compat lib) must already
                # be part of the launch environment before the service starts.
                tools = prepare_profiler_tools(run, c)
            env.update(tools.get("env_updates") or {})
            with (run / "logs" / "service.log").open("w", encoding="utf-8") as output:
                popen_kwargs = {"cwd": root, "env": env, "stdout": output,
                                "stderr": subprocess.STDOUT,
                                "start_new_session": os.name == "posix"}
                if preexec is not None:
                    popen_kwargs["preexec_fn"] = preexec
                process = subprocess.Popen(argv, **popen_kwargs)
                write_json(run / "process.json", {"pid": process.pid, "argv": argv, "at": utc(), "cwd": str(root),
                           "ipc_directory": env['VLLM_RPC_BASE_PATH'], "keep_ready": keep_ready,
                           "service_uid": c["service"].get("uid")})
                deadline = time.monotonic() + c["service"].get("ready_timeout", 1800)
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"service exited during load: {process.returncode}")
                    try:
                        req = urllib.request.Request(c["service"]["base_url"].rstrip("/") + "/models", headers=headers(c))
                        with urllib.request.urlopen(req, timeout=2) as reply:
                            models = json.loads(reply.read())["data"]
                        if c["service"]["served_model"] in {v["id"] for v in models}:
                            break
                    except (OSError, KeyError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("service readiness timed out")
                    time.sleep(.2)
                state(run, "startup", "passed", pid=process.pid)
                if (c["service"].get("smoke") or {}).get("enabled", False):
                    state(run, "smoke", "running", max_tokens=c["service"]["smoke"].get("max_tokens", 8))
                    try:
                        smoke_request(run, c)
                    except BaseException as exc:
                        state(run, "smoke", "failed", error=type(exc).__name__, detail=str(exc))
                        raise
                    state(run, "smoke", "passed", record="smoke.json")
                state(run, "evaluation", "running")
                result = evaluate(run, c)
                state(run, "evaluation", "passed" if result["protocol_pass"] else "failed",
                      quality_status=result["quality_status"], numerical_parity="not_run",
                      **{key: result[key] for key in ("attempted", "succeeded", "failed", "skipped", "timeout", "samples")})
                passed = result["protocol_pass"] and result["quality_status"] != "failed"
                state(run, "execution", "passed" if passed else "failed")
                if passed and after_evaluation is not None:
                    after_evaluation(run, c, result)
                if passed and keep_ready:
                    state(run, 'serving', 'ready', pid=process.pid, base_url=c['service']['base_url'])
                    print(json.dumps({'serving': 'ready', 'run': str(run), 'pid': process.pid,
                                      'base_url': c['service']['base_url']}), flush=True)
                    code = process.wait()
                    state(run, 'serving', 'stopped', service_exit=code)
                    if code:
                        raise RuntimeError(f'retained service exited: {code}')
                return result
        except BaseException as exc:
            state(run, "execution", "failed", error=type(exc).__name__, detail=str(exc))
            raise
        finally:
            stop_owned(process, c["service"].get("shutdown_timeout", 10))
            state(run, "cleanup", "process_group_signalled", service_exit=None if process is None else process.poll(),
                  device_release="requires_external_verification")
