"""Same-host P/D split service supervisor for the deployment matrix.

Launches three children and fronts them with one endpoint so the standard
experiment runner sees a single service:
  - producer: adapters/serve.py with LLMDataDistConnector kv_producer
  - consumer: adapters/serve.py with LLMDataDistConnector kv_consumer
  - proxy:    adapters/pd_relay_proxy.py (prefill -> kv_transfer_params
              handoff -> decode with real KV pull)

Front routes:
  GET  /v1/models, /health  200 only when BOTH instances report ready
  POST /v1/chat/completions -> proxy (bytes relay; the proxy splices P/D)
  POST /start_profile|/stop_profile -> fan out to BOTH instances
  GET  /metrics_snapshot    -> raw /metrics of both instances for the probe

KV-transfer evidence is collected by the matrix probe from /metrics_snapshot
and the engine logs, not from the front.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
import uvicorn

from multinode_adapter import proxy_argv as _shared_proxy_argv
from multinode_adapter import role_argv as _shared_role_argv

app = FastAPI()
STATE = {}
EXPECTED_MODEL = "pangu-92b"
CHILD_LABELS = ("producer", "consumer", "proxy")


def fetch(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except Exception as exc:
        return None, repr(exc).encode()


def post_bytes(url, data, content_type, timeout=6 * 60 * 60):
    request = urllib.request.Request(url, method="POST", data=data,
                                     headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type", content_type)
    except Exception as exc:
        return None, repr(exc).encode(), "text/plain"


def model_ids(payload, expected_model=EXPECTED_MODEL):
    try:
        value = json.loads(payload.decode() if isinstance(payload, bytes) else payload)
        data = value["data"]
        if not isinstance(data, list) or not data:
            return None
        ids = [item["id"] for item in data]
        if any(not isinstance(item, str) or not item for item in ids):
            return None
        if len(set(ids)) != len(ids) or expected_model not in ids:
            return None
        return ids
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def readiness_result(results, expected_model=EXPECTED_MODEL):
    parsed = {role: model_ids(body, expected_model) if status == 200 else None
              for role, (status, body) in results.items()}
    valid = all(parsed.values()) and len({frozenset(ids) for ids in parsed.values()}) == 1
    if not valid:
        return JSONResponse(status_code=503, content={
            "status": "waiting",
            "producer": {"http_status": results["producer"][0], "model_ids": parsed["producer"]},
            "consumer": {"http_status": results["consumer"][0], "model_ids": parsed["consumer"]},
        })
    return {"status": "ready", "producer": True, "consumer": True,
            "data": [{"id": model_id, "object": "model", "owned_by": "vllm"}
                     for model_id in parsed["producer"]]}


@app.get("/v1/models")
@app.get("/health")
async def readiness():
    results = {}
    for role in ("producer", "consumer"):
        results[role] = await run_in_threadpool(fetch, STATE["urls"][role] + "/v1/models", 30)
    return readiness_result(results, STATE.get("served_model_name", EXPECTED_MODEL))


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.body()
    status, payload, media_type = await run_in_threadpool(
        post_bytes, STATE["urls"]["proxy"] + "/v1/chat/completions", body,
        request.headers.get("Content-Type", "application/json"))
    return Response(content=payload, status_code=status or 502, media_type=media_type)


@app.post("/start_profile")
@app.post("/stop_profile")
async def profile_control(request: Request):
    action = request.url.path  # /start_profile or /stop_profile
    body = await request.body()
    content_type = request.headers.get("Content-Type", "application/json")
    results = {}
    for role in ("producer", "consumer"):
        status_code, payload, _ = await run_in_threadpool(
            post_bytes, STATE["urls"][role] + action, body, content_type, 120)
        results[role] = {"http_status": status_code, "body": payload.decode(errors="replace")[:512]}
    ok = all(v["http_status"] == 200 for v in results.values())
    return JSONResponse(status_code=200 if ok else 500, content=results)


@app.get("/metrics_snapshot")
async def metrics_snapshot():
    results = {}
    for role in ("producer", "consumer"):
        status_code, payload = fetch(STATE["urls"][role] + "/metrics", 60)
        results[role] = payload.decode(errors="replace") if status_code == 200 else "error:" + payload.decode(errors="replace")
    return results


def kv_transfer_config(role, kv_port):
    """Return only the connector keys supported by the frozen runtime."""
    if role not in {"kv_producer", "kv_consumer"}:
        raise ValueError("role must be kv_producer or kv_consumer")
    return {"kv_connector": "LLMDataDistConnector", "kv_role": role,
            "kv_port": int(kv_port)}


def serve_argv(python, adapter, model, port, gpu_share, devices, kv_config, eager,
               max_batched_tokens, max_seqs, served_model_name=EXPECTED_MODEL,
               bind_address="127.0.0.1", tensor_parallel_size=2,
               data_parallel_size=1, pipeline_parallel_size=1,
               dtype="bfloat16", max_model_len=32768,
               kv_cache_dtype="hif8_ds_mla", block_size=128,
               expert_parallel=True):
    role = {"kv_producer": "prefill", "kv_consumer": "decode"}[kv_config["kv_role"]]
    return _shared_role_argv(
        python, adapter, model, role, port, gpu_share, devices, kv_config["kv_port"],
        eager=eager, max_batched_tokens=max_batched_tokens, max_seqs=max_seqs,
        served_model_name=served_model_name, bind_address=bind_address,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
        dtype=dtype, max_model_len=max_model_len, kv_cache_dtype=kv_cache_dtype,
        block_size=block_size, expert_parallel=expert_parallel)


def role_argv(python, adapter, model, role, port, gpu_share, devices, kv_port,
              eager=True, max_batched_tokens=10240, max_seqs=56,
              served_model_name=EXPECTED_MODEL, bind_address="127.0.0.1",
              tensor_parallel_size=None, data_parallel_size=1,
              pipeline_parallel_size=1, dtype="bfloat16", max_model_len=32768,
              kv_cache_dtype="hif8_ds_mla", block_size=128,
              expert_parallel=True):
    """Build one executable producer/consumer command for a host/role.

    This is the cross-host launch seam.  The HTTP bind address is local to the
    role host; the remote endpoint is supplied separately to the relay.  KV
    transfer metadata is still produced by the runtime response and is never
    constructed here.
    """
    return _shared_role_argv(
        python, adapter, model, role, port, gpu_share, devices, kv_port,
        eager=eager, max_batched_tokens=max_batched_tokens, max_seqs=max_seqs,
        served_model_name=served_model_name, bind_address=bind_address,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
        dtype=dtype, max_model_len=max_model_len, kv_cache_dtype=kv_cache_dtype,
        block_size=block_size, expert_parallel=expert_parallel)


def proxy_argv(python, proxy_script, model, prefill_endpoint, decode_endpoint,
               port, bind_address="127.0.0.1"):
    """Build the relay command with explicit reachable role endpoints."""
    return _shared_proxy_argv(
        python, proxy_script, model, prefill_endpoint, decode_endpoint,
        port, bind_address=bind_address)


def service_urls(producer_advertise, consumer_advertise, proxy_bind,
                 ports, proxy_advertise=None):
    """Build supervisor URLs from the same advertised addresses used by argv."""
    proxy_host = proxy_advertise or (
        "127.0.0.1" if proxy_bind in {"0.0.0.0", "::"} else proxy_bind)
    return {
        "producer": "http://%s:%d" % (producer_advertise, ports["producer"]),
        "consumer": "http://%s:%d" % (consumer_advertise, ports["consumer"]),
        "proxy": "http://%s:%d" % (proxy_host, ports["proxy"]),
        "advertised": {"producer": producer_advertise,
                       "consumer": consumer_advertise, "proxy": proxy_host},
    }


def instance_env(base, devices):
    env = dict(base)
    env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    env["ASCEND_VISIBLE_DEVICES"] = devices
    return env


ROLE_ENV_KEYS = frozenset({"PYTHONPATH", "VLLM_PLUGINS", "PROMETHEUS_MULTIPROC_DIR",
                           "FULLNET_KV_OBSERVER_ENABLED", "FULLNET_KV_OBSERVER_ROLE",
                           "FULLNET_KV_OBSERVER_METRICS"})


def private_role_path(value, job):
    """Accept a real path under this broker job, with no symlink components."""
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("role environment paths must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(job):
        raise ValueError("role environment path escapes current fleet job")
    current = path
    while current != job:
        if current.is_symlink():
            raise ValueError("role environment symlink refused")
        if current == current.parent:
            raise ValueError("role environment path is not under canonical job")
        current = current.parent
    return resolved


def role_environments(base, sidecar=None):
    """Merge only explicit hook/registry settings; device selection stays last."""
    result = {role: dict(base) for role in ("producer", "consumer")}
    if sidecar is None:
        return result
    if not base.get("FULLNET_FLEET_JOB_ROOT"):
        raise ValueError("--role-env-file requires FULLNET_FLEET_JOB_ROOT")
    job = Path(base["FULLNET_FLEET_JOB_ROOT"]).resolve(strict=True)
    path = private_role_path(sidecar, job)
    if not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("role environment must be a small private JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"producer", "consumer"}:
        raise ValueError("role environment requires producer and consumer mappings")
    for role, overrides in data.items():
        if not isinstance(overrides, dict):
            raise ValueError("role environment entry must be a mapping")
        for key, value in overrides.items():
            reserve = bool(re.fullmatch(r"FULLNET_[A-Z0-9_]+_RESERVE_BYTES", key))
            if (key not in ROLE_ENV_KEYS and not reserve) or re.search(
                    r"TOKEN|PASSWORD|SECRET|CREDENTIAL|API_KEY|DEVICE|CANN|VISIBLE", key):
                raise ValueError("role environment key is not allowed: " + key)
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("role environment values must be strings without NUL")
            if reserve and (not value.isdecimal() or int(value) <= 0):
                raise ValueError("role reserve must be a positive integer string")
            if key == "FULLNET_KV_OBSERVER_ROLE" and value != role:
                raise ValueError("observer role differs from sidecar role")
            if key == "PROMETHEUS_MULTIPROC_DIR":
                if not private_role_path(value, job).is_dir():
                    raise ValueError("role metrics directory is missing")
            if key == "PYTHONPATH":
                baseline = [x for x in base.get(key, "").split(os.pathsep) if x]
                entries = [x for x in value.split(os.pathsep) if x]
                for entry in entries:
                    same_baseline = entry in baseline
                    if not same_baseline and Path(entry).is_dir():
                        same_baseline = any(Path(old).is_dir() and Path(entry).samefile(old) for old in baseline)
                    if not same_baseline and not private_role_path(entry, job).is_dir():
                        raise ValueError("new role Python paths must be job-private directories")
                value = os.pathsep.join(dict.fromkeys([*entries, *baseline]))
            result[role][key] = value
    directories = [value.get("PROMETHEUS_MULTIPROC_DIR") for value in result.values()]
    if all(directories) and Path(directories[0]).resolve() == Path(directories[1]).resolve():
        raise ValueError("producer and consumer metrics directories must be distinct")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-adapter", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--proxy-script", required=True)
    parser.add_argument("--producer-devices", required=True)
    parser.add_argument("--consumer-devices", required=True)
    parser.add_argument("--producer-port", type=int, required=True)
    parser.add_argument("--consumer-port", type=int, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--front-port", type=int, required=True)
    parser.add_argument("--served-model-name", default=EXPECTED_MODEL)
    parser.add_argument("--producer-bind-address", default="127.0.0.1")
    parser.add_argument("--consumer-bind-address", default="127.0.0.1")
    parser.add_argument("--producer-advertise-address")
    parser.add_argument("--consumer-advertise-address")
    parser.add_argument("--proxy-bind-address", default="127.0.0.1")
    parser.add_argument("--proxy-advertise-address")
    parser.add_argument("--front-bind-address", default="127.0.0.1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--kv-cache-dtype", default="hif8_ds_mla")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-port-base", type=int, default=7100)
    parser.add_argument("--gpu-share", type=float, default=0.8)
    parser.add_argument("--max-batched-tokens", type=int, default=10240)
    parser.add_argument("--max-seqs", type=int, default=56)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--role-env-file", help="explicit current-job private producer/consumer environment JSON")
    args = parser.parse_args()

    env = os.environ.copy()
    role_envs = role_environments(env, args.role_env_file)
    python = sys.executable
    producer_args = role_argv(
        python, args.serve_adapter, args.model, "prefill", args.producer_port,
        args.gpu_share, args.producer_devices, args.kv_port_base,
        max_batched_tokens=args.max_batched_tokens, max_seqs=args.max_seqs,
        served_model_name=args.served_model_name,
        bind_address=args.producer_bind_address,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size, dtype=args.dtype,
        max_model_len=args.max_model_len, kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size, expert_parallel=args.expert_parallel)
    consumer_args = role_argv(
        python, args.serve_adapter, args.model, "decode", args.consumer_port,
        args.gpu_share, args.consumer_devices, args.kv_port_base + 100,
        max_batched_tokens=args.max_batched_tokens, max_seqs=args.max_seqs,
        served_model_name=args.served_model_name,
        bind_address=args.consumer_bind_address,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size, dtype=args.dtype,
        max_model_len=args.max_model_len, kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size, expert_parallel=args.expert_parallel)
    producer_advertise = args.producer_advertise_address or (
        "127.0.0.1" if args.producer_bind_address in {"0.0.0.0", "::"}
        else args.producer_bind_address)
    consumer_advertise = args.consumer_advertise_address or (
        "127.0.0.1" if args.consumer_bind_address in {"0.0.0.0", "::"}
        else args.consumer_bind_address)
    proxy_args = proxy_argv(
        python, args.proxy_script, args.served_model_name,
        "%s:%d" % (producer_advertise, args.producer_port),
        "%s:%d" % (consumer_advertise, args.consumer_port),
        args.proxy_port, bind_address=args.proxy_bind_address)

    def spawn(label, argv, child_env):
        target = os.path.join(args.logs, "pd-split-%s.log" % label)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        log = open(target, "w")
        STATE[label] = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=child_env)

    STATE["served_model_name"] = args.served_model_name
    STATE["urls"] = service_urls(
        producer_advertise, consumer_advertise, args.proxy_bind_address,
        {"producer": args.producer_port, "consumer": args.consumer_port,
         "proxy": args.proxy_port}, args.proxy_advertise_address)
    spawn("producer", producer_args, instance_env(role_envs["producer"], args.producer_devices))
    spawn("consumer", consumer_args, instance_env(role_envs["consumer"], args.consumer_devices))
    spawn("proxy", proxy_args, env)

    deadline = time.time() + args.ready_timeout
    ready = {"producer": False, "consumer": False}
    while time.time() < deadline:
        if not all_children_alive():
            print(json.dumps({"pd_split": "child_exited_early",
                "codes": child_exit_codes()}), flush=True)
            shutdown(1)
            return
        for role in ("producer", "consumer"):
            if not ready[role]:
                status, body = fetch(STATE["urls"][role] + "/v1/models", 30)
                ready[role] = status == 200 and model_ids(
                    body, STATE.get("served_model_name", EXPECTED_MODEL)) is not None
        if ready["producer"] and ready["consumer"]:
            break
        time.sleep(5)
    if not (ready["producer"] and ready["consumer"]):
        print(json.dumps({"pd_split": "instances_not_ready", "ready": ready}), flush=True)
        shutdown(1)
        return
    print(json.dumps({"pd_split": "instances_ready",
        "producer_port": args.producer_port, "consumer_port": args.consumer_port,
        "proxy_port": args.proxy_port, "front_port": args.front_port}), flush=True)
    uvicorn.run(app, host=args.front_bind_address, port=args.front_port, log_level="warning")
    shutdown(0)


def child_exit_codes(state=None):
    """Poll only the three subprocess entries; STATE also carries metadata."""
    state = STATE if state is None else state
    return {label: state[label].poll() for label in CHILD_LABELS if label in state}


def all_children_alive(state=None):
    """Return true only while every tracked child has not exited.

    ``child_exit_codes`` returns a role-to-code mapping; metadata in STATE
    (URLs and served model name) is deliberately not part of this check.
    """
    return all(code is None for code in child_exit_codes(state).values())


def shutdown(code):
    for label in ("producer", "consumer", "proxy"):
        child = STATE.get(label)
        if child and child.poll() is None:
            child.terminate()
    for label in ("producer", "consumer", "proxy"):
        child = STATE.get(label)
        if child and child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
    sys.exit(code)


def on_signal(*_):
    shutdown(1)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    main()
