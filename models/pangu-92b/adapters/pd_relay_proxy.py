"""P/D relay proxy for the deployment matrix (same-host and cross-host).

The fork's disagg_proxy_demo.py forwards the decode request WITHOUT the
kv_transfer_params that LLMDataDistConnector's consumer side requires
(DecodeScheduler.get_num_new_matched_tokens returns 0 unless
request.kv_transfer_params is set), so it cannot drive real KV transfer.
This proxy implements the handoff explicitly:

  1. prefill  -> producer instance, max_tokens=1 so the producer finishes
                 FINISHED_LENGTH_CAPPED and hands its KV blocks off
                 (PrefillScheduler.request_finished returns the transfer
                 params, which serving_chat attaches to the response).
  2. decode   -> consumer instance, original payload + the producer's
                 kv_transfer_params; the consumer pulls KV from the producer
                 workers before decoding.

Evidence for the matrix probe: /pd_stats counters plus each instance's
/metrics (kv_transfer_failures stays 0 on success) and engine logs.
The matrix contract is non-streaming; streaming requests are rejected
explicitly rather than silently losing the handoff.

HTTP clients use urllib in worker threads: the runtime environment carries
fastapi/uvicorn but no aiohttp/httpx/requests.
"""
import json
import time
import urllib.error
import urllib.request
from collections import Counter

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
import uvicorn

UPSTREAM_TIMEOUT = 6 * 60 * 60
STATE = {"urls": {}, "stats": Counter(), "last": {}}
EXPECTED_MODEL = "pangu-92b"
app = FastAPI()


def note(event, detail=None):
    STATE["stats"][event] += 1
    if detail is not None:
        STATE["last"][event] = {"at": time.time(), "detail": str(detail)[:400]}


def _get(url, timeout=30):
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        return None, repr(exc).encode()


def _post_json(url, payload, timeout=UPSTREAM_TIMEOUT):
    request = urllib.request.Request(url, method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), "application/json"
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
            "prefill": {"http_status": results["prefill"][0], "model_ids": parsed["prefill"]},
            "decode": {"http_status": results["decode"][0], "model_ids": parsed["decode"]},
        })
    return {"status": "ready",
            "instances": {role: status for role, (status, _) in results.items()},
            "data": [{"id": model_id, "object": "model", "owned_by": "vllm"}
                     for model_id in parsed["prefill"]]}


@app.get("/v1/models")
@app.get("/health")
async def readiness():
    results = {}
    for role in ("prefill", "decode"):
        status, body = await run_in_threadpool(_get, STATE["urls"][role] + "/v1/models", 30)
        results[role] = (status, body)
    response = readiness_result(results, STATE.get("model", EXPECTED_MODEL))
    if isinstance(response, dict):
        response["instances"] = {role: status for role, (status, _) in results.items()}
    return response


@app.get("/pd_stats")
async def pd_stats():
    return {"stats": dict(STATE["stats"]), "last": STATE["last"],
            "instances": STATE["urls"]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "bad json: %r" % exc})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "json body must be an object"})
    if body.get("stream"):
        return JSONResponse(status_code=400, content={"error":
            "streaming unsupported: the P/D handoff contract is non-streaming"})

    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    if "max_completion_tokens" in prefill_body:
        prefill_body["max_completion_tokens"] = 1
    prefill_body.pop("kv_transfer_params", None)

    status, payload, _ = await run_in_threadpool(
        _post_json, STATE["urls"]["prefill"] + "/v1/chat/completions", prefill_body)
    if status is None:
        note("prefill_error", payload)
        return JSONResponse(status_code=502, content={"error": "prefill instance unreachable: %s" % payload.decode(errors="replace")[:300]})
    if status != 200:
        note("prefill_failed", "%s %s" % (status, payload[:400]))
        return JSONResponse(status_code=502, content={"error": "prefill stage failed",
            "upstream_status": status, "upstream_body": payload.decode(errors="replace")[:2000]})
    try:
        params = json.loads(payload).get("kv_transfer_params")
    except Exception:
        params = None
    if not isinstance(params, dict) or "prefill_req_id" not in params:
        note("handoff_missing", payload[:400])
        return JSONResponse(status_code=502, content={"error":
            "producer response carried no kv_transfer_params; KV handoff did not happen"})

    decode_body = dict(body)
    decode_body["kv_transfer_params"] = params
    status, payload, media_type = await run_in_threadpool(
        _post_json, STATE["urls"]["decode"] + "/v1/chat/completions", decode_body)
    if status is None:
        note("decode_error", payload)
        return JSONResponse(status_code=502, content={"error": "decode instance unreachable: %s" % payload.decode(errors="replace")[:300]})
    if status != 200:
        note("decode_failed", "%s %s" % (status, payload[:400]))
    else:
        note("handoff_ok", {"prefill_req_id": params.get("prefill_req_id")})
    return Response(content=payload, status_code=status, media_type=media_type)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill", required=True, help="host:port of the producer instance")
    parser.add_argument("--decode", required=True, help="host:port of the consumer instance")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    STATE["urls"] = {"prefill": "http://%s" % args.prefill, "decode": "http://%s" % args.decode}
    STATE["model"] = args.model
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
