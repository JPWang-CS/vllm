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
import signal
import subprocess
import sys
import time
import urllib.request

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
import uvicorn

app = FastAPI()
STATE = {}


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


@app.get("/v1/models")
@app.get("/health")
async def readiness():
    producer_ok, _ = await run_in_threadpool(fetch, STATE["urls"]["producer"] + "/v1/models", 30)
    consumer_ok, _ = await run_in_threadpool(fetch, STATE["urls"]["consumer"] + "/v1/models", 30)
    if producer_ok == 200 and consumer_ok == 200:
        return {"status": "ready", "producer": True, "consumer": True}
    return JSONResponse(status_code=503, content={"status": "waiting",
        "producer": producer_ok, "consumer": consumer_ok})


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
    results = {}
    for role in ("producer", "consumer"):
        status_code, payload = await run_in_threadpool(fetch, STATE["urls"][role] + action, 120)
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


def serve_argv(python, adapter, model, port, gpu_share, devices, kv_config, eager,
               max_batched_tokens, max_seqs):
    argv = [python, "-B", adapter, "--model", model, "--served-model-name", "pangu-92b",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port),
            "--dtype", "bfloat16", "--max-model-len", "32768", "--tensor-parallel-size", "2",
            "--data-parallel-size", "1", "--pipeline-parallel-size", "1",
            "--enable-expert-parallel", "--distributed-executor-backend", "mp",
            "--max-num-batched-tokens", str(max_batched_tokens), "--max-num-seqs", str(max_seqs),
            "--gpu-memory-utilization", str(gpu_share), "--block-size", "128",
            "--kv-cache-dtype", "hif8_ds_mla", "--load-format", "safetensors",
            "--enable-chunked-prefill", "--no-disable-hybrid-kv-cache-manager", "--seed", "0",
            "--kv-transfer-config", json.dumps(kv_config)]
    if eager:
        argv.append("--enforce-eager")
    return argv


def instance_env(base, devices):
    env = dict(base)
    env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    env["ASCEND_VISIBLE_DEVICES"] = devices
    return env


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
    parser.add_argument("--kv-port-base", type=int, default=7100)
    parser.add_argument("--gpu-share", type=float, default=0.8)
    parser.add_argument("--max-batched-tokens", type=int, default=10240)
    parser.add_argument("--max-seqs", type=int, default=56)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    args = parser.parse_args()

    env = os.environ.copy()
    python = sys.executable
    producer_args = serve_argv(python, args.serve_adapter, args.model, args.producer_port, args.gpu_share,
                               args.producer_devices,
                               {"kv_connector": "LLMDataDistConnector", "kv_role": "kv_producer",
                                "kv_port": args.kv_port_base}, eager=True,
                               max_batched_tokens=args.max_batched_tokens, max_seqs=args.max_seqs)
    consumer_args = serve_argv(python, args.serve_adapter, args.model, args.consumer_port, args.gpu_share,
                               args.consumer_devices,
                               {"kv_connector": "LLMDataDistConnector", "kv_role": "kv_consumer",
                                "kv_port": args.kv_port_base + 100}, eager=True,
                               max_batched_tokens=args.max_batched_tokens, max_seqs=args.max_seqs)
    proxy_args = [python, "-B", args.proxy_script, "--model", "pangu-92b",
                  "--prefill", "127.0.0.1:%d" % args.producer_port,
                  "--decode", "127.0.0.1:%d" % args.consumer_port,
                  "--port", str(args.proxy_port)]

    def spawn(label, argv, child_env):
        target = os.path.join(args.logs, "pd-split-%s.log" % label)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        log = open(target, "w")
        STATE[label] = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=child_env)

    STATE["urls"] = {"producer": "http://127.0.0.1:%d" % args.producer_port,
                     "consumer": "http://127.0.0.1:%d" % args.consumer_port,
                     "proxy": "http://127.0.0.1:%d" % args.proxy_port}
    spawn("producer", producer_args, instance_env(env, args.producer_devices))
    spawn("consumer", consumer_args, instance_env(env, args.consumer_devices))
    spawn("proxy", proxy_args, env)

    def all_children_alive():
        return all(STATE[label].poll() is None for label in ("producer", "consumer", "proxy"))

    deadline = time.time() + args.ready_timeout
    ready = {"producer": False, "consumer": False}
    while time.time() < deadline:
        if not all_children_alive():
            print(json.dumps({"pd_split": "child_exited_early",
                "codes": {label: STATE[label].poll() for label in STATE if label != "urls"}}), flush=True)
            shutdown(1)
            return
        for role in ("producer", "consumer"):
            if not ready[role]:
                status, _ = fetch(STATE["urls"][role] + "/v1/models", 30)
                ready[role] = status == 200
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
    uvicorn.run(app, host="127.0.0.1", port=args.front_port, log_level="warning")
    shutdown(0)


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


