"""Multi-host (cross-host rank) distributed adapter for the 92B matrix cases.

Covers the launch_contract mode "distributed_adapter_required" scenarios
(multi_host_tp: TP4 across 2 hosts x 2 NPUs; multi_host_dp: DP4 x TP1
across the same split) on the fork's RayDistributedExecutor surface
(vllm/v1/executor/ray_executor.py: VLLM_RAY_BUNDLE_INDICES /
VLLM_RAY_PER_WORKER_GPUS, placement groups formed PACK in ray_utils).

Contract implemented here:
  plan()             host-major rank/bundle map + cluster formation artifacts
  preflight()        TCP reachability gate; ANY unreachable -> blocked_external
                     (resource/infra verdict, never a runtime failure)
  cluster_scripts()  per-host ray start commands to run as fleet repair/*.sh jobs
  head_env()         env for the head-host serve process
  fixture_rendezvous()  CPU-only gloo proof of the rank map (fixture; per the
                     acceptance contract it does NOT count as device acceptance)
  rank_evidence()    parse service log + placement table for real cross-host
                     rank proof (>= 2 distinct hosts)
"""
import json
import os
import pathlib
import re
import shlex
import signal
import socket
import sys
import threading
import time

MAX_UNIX_SOCKET_BYTES = 107
RAY_SESSION_NAME_TEMPLATE = "session_YYYY-MM-DD_HH-MM-SS_ffffff_9999999"
RAY_SOCKET_NAMES = ("plasma_store", "raylet")
RAY_NODE_MAJOR_MINOR = (2, 48)


def path_byte_length(path):
    """Return the UTF-8 byte length used by Ray's Unix socket check."""
    return len(os.fspath(path).encode("utf-8"))


def ray_runtime_budget(path, session_name=None, max_bytes=MAX_UNIX_SOCKET_BYTES):
    """Check the complete Ray 2.48 head/worker path budget.

    Ray publishes ``temp_dir`` and ``session_dir`` through the head's internal
    KV; workers inherit those paths.  The socket paths are therefore under
    ``<temp-dir>/<session>/sockets`` rather than directly under temp-dir.
    The generated session name contains a timestamp, microseconds and PID; the
    template below is a conservative byte budget, not an invented environment
    variable or a claim that the fleet root is usable.
    """
    resolved = pathlib.Path(path).resolve(strict=False)
    session = session_name or RAY_SESSION_NAME_TEMPLATE
    session_dir = resolved / session
    sockets_dir = session_dir / "sockets"
    paths = {
        "temp_dir": resolved,
        "session_dir": session_dir,
        "session_latest": resolved / "session_latest",
        "sockets_dir": sockets_dir,
        "plasma_store": sockets_dir / "plasma_store",
        "raylet": sockets_dir / "raylet",
        "logs_dir": session_dir / "logs",
        "runtime_env_dir": session_dir / "runtime_resources",
    }
    lengths = {name: path_byte_length(value) for name, value in paths.items()}
    max_name = max(lengths, key=lengths.get)
    max_length = lengths[max_name]
    if max_length > max_bytes:
        raise ValueError(
            f"Ray 2.48 complete path budget exceeds {max_bytes} bytes: "
            f"{max_name}={max_length}; shorten temp-dir by at least "
            f"{max_length - max_bytes} bytes or use an authorized in-project "
            "short bind mount"
        )
    return {
        "path": str(resolved),
        "bytes": lengths["temp_dir"],
        "max_path": max_name,
        "max_path_bytes": max_length,
        "max_bytes": max_bytes,
        "session_name": session,
        "paths": {name: {"path": str(value), "bytes": lengths[name]}
                  for name, value in paths.items()},
        "worker_temp_dir": "inherited_from_head_internal_kv",
    }


def validate_runtime_path(path, project_root=None, max_bytes=MAX_UNIX_SOCKET_BYTES):
    """Validate a complete Ray runtime path inside the authorized project."""
    resolved = pathlib.Path(path).resolve(strict=False)
    if project_root is not None:
        root = pathlib.Path(project_root).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"runtime path escapes project root: {resolved}") from exc
    return ray_runtime_budget(resolved, max_bytes=max_bytes)


def short_runtime_path(project_root, leaf=".r", max_bytes=MAX_UNIX_SOCKET_BYTES):
    """Choose a short project-private runtime directory.

    The caller creates the directory in its own task setup.  This helper only
    chooses and validates a path, so CPU planning cannot write outside the
    project or silently fall back to /tmp.
    """
    root = pathlib.Path(project_root).resolve(strict=True)
    candidates = (root / leaf, root / ".runtime" / leaf)
    errors = []
    for candidate in candidates:
        try:
            return validate_runtime_path(candidate, root, max_bytes)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("no project-private short runtime path: " + "; ".join(errors))


def ray_project_paths(project_root, task_id, max_bytes=MAX_UNIX_SOCKET_BYTES):
    """Return project-private Ray 2.48 temp and explicit socket paths.

    ``p`` and ``r`` are passed as complete socket filenames to RayParams;
    Ray's own Node._prepare_socket_file performs the final validation.  The
    task id is deliberately short and cannot contain path separators.
    """
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", task_id):
        raise ValueError("Ray task_id must be 1-24 ASCII path-safe characters")
    root = pathlib.Path(project_root).resolve(strict=True)
    runtime = root / ".r" / task_id
    sockets = {"temp_dir": runtime, "plasma_store_socket": runtime / "p",
               "raylet_socket": runtime / "r"}
    resolved = {name: path.resolve(strict=False) for name, path in sockets.items()}
    for name, path in resolved.items():
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Ray path escapes project root: {path}") from exc
    lengths = {name: path_byte_length(path) for name, path in resolved.items()}
    over = {name: size for name, size in lengths.items() if size > max_bytes}
    if over:
        raise ValueError(
            "project-private Ray socket path exceeds AF_UNIX limit: "
            + ", ".join(f"{name}={size}" for name, size in over.items())
        )
    return {"root": str(root), "task_id": task_id,
            "paths": {name: str(path) for name, path in resolved.items()},
            "bytes": lengths, "max_bytes": max_bytes}


def ray_node_argv(python, adapter_script, project_root, task_id, role,
                  node_ip, ray_port=29600, head_address=None,
                  ray_gpus=0, ray_cpus=None, ray_resources=None,
                  visible_devices=None):
    """Build the Ray 2.48 Node/RayParams Python entry command."""
    if role not in {"head", "worker"}:
        raise ValueError("Ray role must be head or worker")
    if role == "worker" and not head_address:
        raise ValueError("worker requires head_address")
    if not os.path.isabs(os.fspath(adapter_script)):
        raise ValueError("adapter_script must be an absolute project path")
    paths = ray_project_paths(project_root, task_id)
    argv = [python, "-B", adapter_script, "ray-node", "--role", role,
            "--project-root", paths["root"], "--task-id", task_id,
            "--node-ip", node_ip, "--ray-port", str(ray_port)]
    if head_address:
        argv += ["--head-address", head_address]
    if ray_cpus is not None:
        argv += ["--num-cpus", str(ray_cpus)]
    argv += ["--num-gpus", str(ray_gpus), "--ray-resources-json",
             json.dumps(ray_resources or {}, separators=(",", ":"))]
    if visible_devices is not None:
        argv += ["--visible-devices", str(visible_devices)]
    return argv


def ray_node_main(argv):
    """Start one Ray 2.48 Node and scope cleanup to that Node object."""
    import argparse
    parser = argparse.ArgumentParser(prog="multinode_adapter.py ray-node")
    parser.add_argument("--role", choices=("head", "worker"), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--ray-port", type=int, default=29600)
    parser.add_argument("--head-address")
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--ray-resources-json", default="{}")
    parser.add_argument("--visible-devices")
    args = parser.parse_args(argv)

    import ray
    from ray._private.node import Node
    from ray._private.parameter import RayParams

    version = tuple(int(part) for part in ray.__version__.split(".")[:2])
    if version != RAY_NODE_MAJOR_MINOR:
        raise RuntimeError(
            f"ray-node entry requires Ray 2.48.x, got {ray.__version__}"
        )
    try:
        resources = json.loads(args.ray_resources_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--ray-resources-json must be a JSON object") from exc
    if not isinstance(resources, dict):
        raise ValueError("--ray-resources-json must be a JSON object")
    if args.visible_devices is not None:
        # These are the existing project runtime visibility knobs.  The exact
        # Ray Ascend resource name remains a GLM/runtime contract, not a
        # fabricated default in this adapter.
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
        os.environ["ASCEND_VISIBLE_DEVICES"] = args.visible_devices
    paths = ray_project_paths(args.project_root, args.task_id)
    status_path = pathlib.Path(paths["paths"]["temp_dir"]) / "node-status.json"

    def write_status(status, **extra):
        record = {"status": status, "pid": os.getpid(), "role": args.role,
                  "node_ip": args.node_ip, "task_id": args.task_id,
                  "resource_mode": "cpu" if args.num_gpus == 0 and not resources else "explicit",
                  "num_gpus": args.num_gpus, "resources": resources,
                  "visible_devices": args.visible_devices, "paths": paths}
        record.update(extra)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    params = RayParams(
        node_ip_address=args.node_ip,
        gcs_server_port=args.ray_port if args.role == "head" else None,
        gcs_address=args.head_address if args.role == "worker" else None,
        temp_dir=paths["paths"]["temp_dir"],
        plasma_store_socket_name=paths["paths"]["plasma_store_socket"],
        raylet_socket_name=paths["paths"]["raylet_socket"],
        include_dashboard=False,
        include_log_monitor=False,
        no_monitor=True,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        resources=resources,
    )
    node = None
    failure = None
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        write_status("stop_requested", signal=signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        node = Node(params, head=args.role == "head", shutdown_at_exit=True,
                    spawn_reaper=True)
        write_status("running", gcs_address=node.address,
                     process_snapshot=_ray_process_snapshot(node))
        print(json.dumps({"ray_node_started": True, "role": args.role,
                          "pid": os.getpid(), "gcs_address": node.address,
                          "paths": paths}, ensure_ascii=False), flush=True)
        while not stop_event.wait(1):
            write_status("running", gcs_address=node.address,
                         process_snapshot=_ray_process_snapshot(node))
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        write_status("failed", failure=failure,
                     process_snapshot=_ray_process_snapshot(node))
        raise
    finally:
        if node is not None:
            node.kill_all_processes(check_alive=False, allow_graceful=True)
        if failure is None:
            write_status("stopped", process_snapshot=(
                _ray_process_snapshot(node) if node is not None else {}))
        (status_path.parent / "node-exit.json").write_text(json.dumps(
            {"status": "failed" if failure is not None else "stopped",
             "pid": os.getpid(), "task_id": args.task_id, "role": args.role,
             **({"failure": failure} if failure is not None else {})},
            ensure_ascii=False, indent=2))
    return 0


def _ray_process_snapshot(node):
    """Serialize only child processes owned by this Ray Node object."""
    if node is None:
        return {}
    result = {}
    for kind, infos in getattr(node, "all_processes", {}).items():
        result[str(kind)] = []
        for info in infos:
            process = getattr(info, "process", info)
            result[str(kind)].append({
                "pid": getattr(process, "pid", None),
                "returncode": getattr(process, "returncode", None),
            })
    return result


def ray_node_commands(plan_value, project_root, task_id, adapter_script,
                      python_executable="python3", ray_gpus=0,
                      ray_resources=None, visible_devices_by_host=None):
    """Generate per-host 2.48 Python Node commands using project-local p/r."""
    head = plan_value["head"]
    advertise = plan_value.get("ray_advertise_address") or head
    bind = plan_value.get("ray_bind_address")
    commands = {}
    commands[head] = shlex.join(ray_node_argv(
        python_executable, adapter_script, project_root, task_id, "head",
        advertise, ray_port=plan_value["ray_port"], ray_gpus=ray_gpus,
        ray_resources=ray_resources,
        visible_devices=(visible_devices_by_host or {}).get(head)))
    for entry in plan_value["hosts"]:
        if entry["host"] == head:
            continue
        node_ip = bind or entry["host"]
        commands[entry["host"]] = shlex.join(ray_node_argv(
            python_executable, adapter_script, project_root, task_id, "worker",
            node_ip, ray_port=plan_value["ray_port"],
            head_address=f"{advertise}:{plan_value['ray_port']}", ray_gpus=ray_gpus,
            ray_resources=ray_resources,
            visible_devices=(visible_devices_by_host or {}).get(entry["host"])))
    return commands


def plan(hosts, devices_per_host, parallel, ray_port=29600, served_model="pangu-92b",
         bind_address=None, advertise_address=None, runtime_dir=None):
    """Build a host-major rank map for any explicit host/device split.

    The old two-host/two-device defaults remain valid, while 7B/A2/A3
    representatives can pass their own model, parallelism and host split.
    Ray bundles remain host-major; rank_evidence() must verify real placement.
    """
    hosts = list(hosts)
    devices_per_host = list(devices_per_host)
    if not hosts or len(hosts) != len(devices_per_host):
        raise ValueError("hosts and devices_per_host must have the same non-zero length")
    if any(not isinstance(count, int) or count <= 0 for count in devices_per_host):
        raise ValueError("devices_per_host must contain positive integers")
    world = sum(devices_per_host)
    tp, dp = parallel.get("tp", 1), parallel.get("dp", 1)
    pp = parallel.get("pp", 1)
    if any(not isinstance(value, int) or value <= 0 for value in (tp, dp, pp)):
        raise ValueError("tp, dp and pp must be positive integers")
    if pp != 1:
        raise ValueError("pipeline parallelism is not implemented by this rank map; pp must be 1")
    if tp * dp != world:
        raise ValueError(f"tp*dp={tp * dp} != world {world}")
    ranks = []
    base = 0
    for host, count in zip(hosts, devices_per_host):
        ranks.append({"host": host, "ranks": list(range(base, base + count)),
                      "bundles": list(range(base, base + count))})
        base += count
    return {
        "head": hosts[0], "ray_port": ray_port, "world": world,
        "model": served_model, "tp": tp, "dp": dp, "pp": pp,
        "hosts": [{"host": h, "devices_per_host": c} for h, c in zip(hosts, devices_per_host)],
        "rank_map": ranks,
        "vllm_ray_bundle_indices": ",".join(str(b) for h in ranks for b in h["bundles"]),
        "vllm_ray_per_worker_gpus": 1.0,
        "ray_bind_address": bind_address,
        "ray_advertise_address": advertise_address,
        "ray_runtime_dir": runtime_dir,
    }


def preflight(targets, timeout=5.0):
    """TCP reachability from the submit host. Evidence, not assumption: a
    failing probe writes a dated verdict that maps to blocked_external."""
    results = {}
    for host, port in targets:
        entry = {"host": host, "port": port}
        try:
            with socket.create_connection((host, port), timeout=timeout):
                entry["reachable"] = True
        except OSError as exc:
            entry.update(reachable=False, error=f"{type(exc).__name__}: {exc}")
        results[host] = entry
    return {"verdict": "reachable" if all(v["reachable"] for v in results.values()) else "unreachable",
            "probed_at": time.time(), "targets": results}


def cluster_scripts(plan_value, runtime_dir=None, project_root=None,
                    task_id="mnet", adapter_script=None,
                    python_executable="python3"):
    """Per-host ray formation commands (run as fleet repair/*.sh jobs, per
    the broker rule that forbids arbitrary shell in fleet requests)."""
    if adapter_script is not None:
        if project_root is None:
            raise ValueError("project_root is required for ray-node Python entry")
        scripts = ray_node_commands(
            plan_value, project_root, task_id, adapter_script, python_executable)
        scripts["_verify"] = (
            f"{python_executable} -c \"import ray; ray.init(address='auto'); "
            "print(len(ray.nodes()))\""
        )
        return scripts

    head = plan_value["head"]
    port = plan_value["ray_port"]
    runtime = runtime_dir or plan_value.get("ray_runtime_dir")
    if runtime is not None:
        runtime = validate_runtime_path(runtime)
    bind_address = plan_value.get("ray_bind_address")
    advertise_address = plan_value.get("ray_advertise_address")
    scripts = {}
    for entry in plan_value["hosts"]:
        host = entry["host"]
        if host == head:
            command = ["ray", "start", "--head", "--port", str(port), "--disable-usage-stats"]
            if advertise_address:
                command += ["--node-ip-address", advertise_address]
            if runtime is not None:
                command += ["--temp-dir", runtime["path"]]
        else:
            command = ["ray", "start", f"--address={head}:{port}", "--disable-usage-stats"]
            if bind_address:
                command += ["--node-ip-address", bind_address]
        scripts[host] = shlex.join(command)
    scripts["_verify"] = "python3 -c \"import ray; ray.init(address='auto'); print(len(ray.nodes()))\""
    return scripts


def pd_plan(hosts, devices_per_host, tp=None, kv_port_base=7100,
            served_model="pangu-92b", bind_addresses=None,
            advertise_addresses=None, relay_ports=None, device_ids=None):
    """Host-split PD rank map for the disaggregated contract: producer
    (prefill) TP group on hosts[0], consumer (decode) TP group on hosts[1].

    Mirrors the same-host split topology of run_pangu92b.py (producer devices
    = devices[:tp], consumer devices = devices[tp:]) with each role pinned to
    its own host. The KV endpoint is the piece the same-host adapter never
    exercised: producer and consumer now sit in DIFFERENT containers, so the
    connector address must be reachable cross-host; that is gated by the
    device-free container-network probes before any device phase starts.
    """
    hosts = list(hosts)
    devices_per_host = list(devices_per_host)
    if len(hosts) != 2 or len(devices_per_host) != 2:
        raise ValueError("PD requires exactly one prefill host and one decode host")
    if any(not isinstance(count, int) or count <= 0 for count in devices_per_host):
        raise ValueError("PD role device counts must be positive integers")
    if tp is None:
        tp = devices_per_host[0]
    if tp != devices_per_host[0] or tp != devices_per_host[1]:
        raise ValueError(f"pd tp={tp} does not match per-role devices {devices_per_host}")
    if device_ids is None:
        device_ids = [list(range(count)) for count in devices_per_host]
    else:
        device_ids = [list(values) for values in device_ids]
        if len(device_ids) != 2 or any(not values for values in device_ids):
            raise ValueError("device_ids must contain two non-empty role lists")
        if [len(values) for values in device_ids] != devices_per_host:
            raise ValueError("device_ids lengths must match devices_per_host")
    producer, consumer = hosts
    bind_addresses = list(bind_addresses or ["127.0.0.1", "127.0.0.1"])
    advertise_addresses = list(advertise_addresses or bind_addresses)
    if len(bind_addresses) != 2 or len(advertise_addresses) != 2:
        raise ValueError("PD bind and advertise addresses must contain two role addresses")
    relay_ports = dict(relay_ports or {"prefill": 18333, "decode": 18334, "proxy": 18335})
    required_ports = {"prefill", "decode", "proxy"}
    if set(relay_ports) != required_ports:
        raise ValueError("relay_ports must define prefill, decode and proxy")
    return {
        "head": producer,
        "model": served_model,
        "world": 2 * tp,
        "rank_map": [{"host": producer, "ranks": list(range(tp)), "role": "prefill",
                       "devices": devices_per_host[0], "device_ids": device_ids[0]},
                     {"host": consumer, "ranks": list(range(tp, 2 * tp)), "role": "decode",
                      "devices": devices_per_host[1], "device_ids": device_ids[1]}],
        "producer": {"host": producer, "role": "prefill", "tp": tp,
                      "devices": devices_per_host[0], "device_ids": device_ids[0],
                      "bind_address": bind_addresses[0],
                      "advertise_address": advertise_addresses[0]},
        "consumer": {"host": consumer, "role": "decode", "tp": tp,
                      "devices": devices_per_host[1], "device_ids": device_ids[1],
                      "bind_address": bind_addresses[1],
                      "advertise_address": advertise_addresses[1]},
        "kv_port_base": kv_port_base,
        "relay_ports": relay_ports,
        "endpoints": {
            "prefill_http": f"http://{advertise_addresses[0]}:{relay_ports['prefill']}",
            "decode_http": f"http://{advertise_addresses[1]}:{relay_ports['decode']}",
            "producer_kv": f"{advertise_addresses[0]}:{kv_port_base}",
            "consumer_kv": f"{advertise_addresses[1]}:{kv_port_base + 100}",
        },
        "cross_host_kv_endpoint": f"{advertise_addresses[0]}:{kv_port_base}",
        "endpoint_verified": False,
    }


def role_argv(python, serve_adapter, model_path, role, port, gpu_share,
              device_ids, kv_port, eager=True, max_batched_tokens=10240,
              max_seqs=56, served_model_name="pangu-92b",
              bind_address="127.0.0.1", tensor_parallel_size=None,
              data_parallel_size=1, pipeline_parallel_size=1,
              dtype="bfloat16", max_model_len=32768,
              kv_cache_dtype="hif8_ds_mla", block_size=128,
              expert_parallel=True):
    """Dependency-free command builder for one real PD role."""
    role_map = {"prefill": "kv_producer", "producer": "kv_producer",
                "decode": "kv_consumer", "consumer": "kv_consumer"}
    if role not in role_map:
        raise ValueError("role must be prefill/producer or decode/consumer")
    ids = list(device_ids) if not isinstance(device_ids, str) else [
        item for item in device_ids.split(",") if item
    ]
    if not ids:
        raise ValueError("device_ids must not be empty")
    tp = tensor_parallel_size if tensor_parallel_size is not None else len(ids)
    argv = [python, "-B", serve_adapter, "--model", model_path,
            "--served-model-name", served_model_name, "--trust-remote-code",
            "--host", bind_address, "--port", str(port), "--dtype", dtype,
            "--max-model-len", str(max_model_len), "--tensor-parallel-size", str(tp),
            "--data-parallel-size", str(data_parallel_size), "--pipeline-parallel-size",
            str(pipeline_parallel_size), "--distributed-executor-backend", "mp",
            "--max-num-batched-tokens", str(max_batched_tokens), "--max-num-seqs",
            str(max_seqs), "--gpu-memory-utilization", str(gpu_share),
            "--block-size", str(block_size), "--kv-cache-dtype", kv_cache_dtype,
            "--load-format", "safetensors", "--enable-chunked-prefill",
            "--no-disable-hybrid-kv-cache-manager", "--seed", "0",
            "--kv-transfer-config", json.dumps({
                "kv_connector": "LLMDataDistConnector",
                "kv_role": role_map[role], "kv_port": int(kv_port)})]
    if expert_parallel:
        argv.insert(argv.index("--distributed-executor-backend"), "--enable-expert-parallel")
    if eager:
        argv.append("--enforce-eager")
    return argv


def proxy_argv(python, proxy_script, model, prefill_endpoint, decode_endpoint,
               port, bind_address="0.0.0.0"):
    """Dependency-free command builder for the real HTTP relay."""
    return [python, "-B", proxy_script, "--model", model,
            "--prefill", prefill_endpoint, "--decode", decode_endpoint,
            "--bind-address", bind_address, "--port", str(port)]


def pd_role_launches(plan_value, python, serve_adapter, proxy_script, model_path,
                     gpu_share=0.8, eager=True, max_batched_tokens=10240,
                     max_seqs=56, proxy_host=None, dtype="bfloat16",
                     max_model_len=32768, kv_cache_dtype="hif8_ds_mla",
                     block_size=128, expert_parallel=True):
    """Generate independent role commands plus reachable relay endpoints.

    The returned argv is ready for the host-specific launcher.  The proxy
    forwards the producer's opaque runtime ``kv_transfer_params`` to the
    consumer; this function does not invent or serialize KV metadata.
    ``endpoint_verified`` remains false until a real two-host probe verifies
    the endpoints and the runtime reports a successful transfer.
    """
    relay_ports = plan_value["relay_ports"]
    producer = plan_value["producer"]
    consumer = plan_value["consumer"]
    model = plan_value["model"]
    producer_devices = ",".join(map(str, producer["device_ids"]))
    consumer_devices = ",".join(map(str, consumer["device_ids"]))
    roles = {
        "prefill": {
            "host": producer["host"],
            "argv": role_argv(
                python, serve_adapter, model_path, "prefill", relay_ports["prefill"],
                gpu_share, producer_devices,
                plan_value["kv_port_base"], eager=eager,
                max_batched_tokens=max_batched_tokens, max_seqs=max_seqs,
                served_model_name=model, bind_address=producer["bind_address"],
                tensor_parallel_size=producer["tp"], dtype=dtype,
                max_model_len=max_model_len, kv_cache_dtype=kv_cache_dtype,
                block_size=block_size, expert_parallel=expert_parallel),
            "endpoint": plan_value["endpoints"]["prefill_http"],
            "kv_endpoint": plan_value["endpoints"]["producer_kv"],
            "env": {"ASCEND_RT_VISIBLE_DEVICES": producer_devices,
                    "ASCEND_VISIBLE_DEVICES": producer_devices},
            "kv_config": {"kv_connector": "LLMDataDistConnector",
                          "kv_role": "kv_producer",
                          "kv_port": plan_value["kv_port_base"]},
        },
        "decode": {
            "host": consumer["host"],
            "argv": role_argv(
                python, serve_adapter, model_path, "decode", relay_ports["decode"],
                gpu_share, consumer_devices,
                plan_value["kv_port_base"] + 100, eager=eager,
                max_batched_tokens=max_batched_tokens, max_seqs=max_seqs,
                served_model_name=model, bind_address=consumer["bind_address"],
                tensor_parallel_size=consumer["tp"], dtype=dtype,
                max_model_len=max_model_len, kv_cache_dtype=kv_cache_dtype,
                block_size=block_size, expert_parallel=expert_parallel),
            "endpoint": plan_value["endpoints"]["decode_http"],
            "kv_endpoint": plan_value["endpoints"]["consumer_kv"],
            "env": {"ASCEND_RT_VISIBLE_DEVICES": consumer_devices,
                    "ASCEND_VISIBLE_DEVICES": consumer_devices},
            "kv_config": {"kv_connector": "LLMDataDistConnector",
                          "kv_role": "kv_consumer",
                          "kv_port": plan_value["kv_port_base"] + 100},
        },
    }
    relay_host = proxy_host or consumer["host"]
    relay = {
        "host": relay_host,
            "argv": proxy_argv(
            python, proxy_script, model,
            f"{producer['advertise_address']}:{relay_ports['prefill']}",
            f"{consumer['advertise_address']}:{relay_ports['decode']}",
            relay_ports["proxy"], bind_address="0.0.0.0"),
        "endpoint": f"http://{relay_host}:{relay_ports['proxy']}",
    }
    return {"model": model, "roles": roles, "relay": relay,
            "endpoint_verified": bool(plan_value.get("endpoint_verified", False))}


def head_env(plan_value):
    return {
        "VLLM_RAY_BUNDLE_INDICES": plan_value["vllm_ray_bundle_indices"],
        "VLLM_RAY_PER_WORKER_GPUS": str(plan_value["vllm_ray_per_worker_gpus"]),
        "RAY_USAGE_STATS_ENABLED": "0",
    }


def _rank_worker(rank, plan_value, port, out_dir):
    import datetime
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      WORLD_SIZE=str(plan_value["world"]), RANK=str(rank),
                      GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist
    dist.init_process_group("gloo")
    mine = next(h for h in plan_value["rank_map"] if rank in h["ranks"])
    tensor = torch.tensor([rank], dtype=torch.float)
    dist.all_reduce(tensor)
    ok = int(tensor.item()) == sum(range(plan_value["world"]))
    (out_dir / f"rank{rank}.json").write_text(json.dumps(
        {"rank": rank, "host_group": mine["host"], "all_reduce_ok": ok,
         "at": datetime.datetime.now().isoformat(timespec="seconds")}))
    dist.barrier()
    dist.destroy_process_group()


def fixture_rendezvous(plan_value, out_dir, port=29671):
    """CPU gloo fixture: the host-major map from plan() is exercised by 4 real
    processes rendezvousing over loopback; ranks learn their host group from
    the SAME map the real launch uses. FIXTURE ONLY - per acceptance contract
    (fixture_counts_as_device_acceptance: false) this proves adapter logic,
    never device acceptance."""
    import multiprocessing as mp
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_worker, args=(r, plan_value, port, out_dir))
             for r in range(plan_value["world"])]
    started = time.time()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    ranks = []
    for r in range(plan_value["world"]):
        path = out_dir / f"rank{r}.json"
        ranks.append(json.loads(path.read_text()) if path.is_file() else {"rank": r, "missing": True})
    ok = all(entry.get("all_reduce_ok") and entry.get("host_group") is not None for entry in ranks)
    verdict = {"fixture": "multinode_rendezvous", "passed": bool(ok),
               "ranks": ranks, "elapsed": round(time.time() - started, 1),
               "counts_as_device_acceptance": False}
    (out_dir / "fixture-verdict.json").write_text(json.dumps(verdict, ensure_ascii=False))
    return verdict


def rank_evidence(service_log_text, placement_table_text):
    """Real cross-host rank proof for acceptance: >= 2 distinct worker hosts in
    the service log AND a placement group whose bundles span 2 nodes."""
    hosts = sorted(set(_HOST_RE.findall(service_log_text)))
    nodes = sorted(set(_HOST_RE.findall(placement_table_text)))
    return {"worker_hosts": hosts, "distinct_hosts": len(hosts),
            "placement_nodes": nodes,
            "has_cross_host_ranks": len(hosts) >= 2 and len(nodes) >= 2}


_HOST_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|7\.\d{1,3}\.\d{1,3}\.\d{1,3}|host-[A-Za-z0-9_-]+)\b")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ray-node":
        sys.exit(ray_node_main(sys.argv[2:]))
    demo = plan(["host-a", "host-b"], [2, 2], {"tp": 4, "dp": 1})
    print(json.dumps(demo, ensure_ascii=False))
    sys.exit(0)
