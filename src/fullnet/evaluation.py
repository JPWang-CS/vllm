"""Fixed-sample requests and task scoring; failures remain in the denominator."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from .store import append_json, read_json, rows, state, utc, write_json


def validate_samples(samples, contract):
    for row in samples:
        tokens, budget = row.get("actual_chat_tokens"), row.get("output_tokens")
        if type(tokens) is not int or type(budget) is not int or min(tokens, budget) <= 0:
            raise ValueError(f"{row['id']}: positive actual_chat_tokens and output_tokens required")
        limit = min(contract["context_limit"], row.get("effective_context_limit", row.get("context_limit", contract["context_limit"])))
        if tokens + budget > limit:
            raise ValueError(f"{row['id']}: context budget exceeded; input is retained")
        if not row.get("messages") and not isinstance(row.get("prompt"), str):
            raise ValueError(f"{row['id']}: prompt or messages required")
        if not isinstance(row.get("outputs"), list) or not row["outputs"] or any(not isinstance(v, str) or not v for v in row["outputs"]):
            raise ValueError(f"{row['id']}: expected outputs must be nonempty strings")


def headers(c):
    value = {"Content-Type": "application/json"}
    key = c["service"].get("api_key_env")
    if key:
        token = os.environ.get(key)
        if not token:
            raise ValueError(f"missing runtime credential environment: {key}")
        value["Authorization"] = "Bearer " + token
    return value


def request(row, c):
    e = c["evaluation"]
    payload = {"model": c["service"]["served_model"],
               "messages": row.get("messages") or [{"role": "user", "content": row["prompt"]}],
               "temperature": e.get("temperature", 0), "top_p": e.get("top_p", 1),
               "seed": e.get("seed", 42), "max_tokens": row["output_tokens"], "stream": False,
               "chat_template_kwargs": e.get("chat_template_kwargs", {})}
    if "add_special_tokens" in e:
        payload["add_special_tokens"] = e["add_special_tokens"]
    record = {"id": row["id"], "sample": row, "payload": payload,
              "success": False, "request_attempted": True, "status": "attempted",
              "timeout": False, "communication_failure": False, "response": "", "start": utc()}
    started = time.monotonic()
    try:
        req = urllib.request.Request(c["service"]["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers(c))
        with urllib.request.urlopen(req, timeout=c["service"].get("request_timeout", 120)) as response:
            data = json.loads(response.read().decode("utf-8"))
            record["http_status"] = response.status
        record["raw_response"] = data
        choice = data["choices"][0]
        record["response"] = choice["message"].get("content") or ""
        record["finish_reason"] = choice.get("finish_reason")
        record["truncated"] = record["finish_reason"] == "length"
        usage = data.get("usage", {})
        errors = []
        if not record["response"]:
            errors.append("empty_response")
        if record["finish_reason"] not in ("stop", "length"):
            errors.append("finish_reason")
        if usage.get("prompt_tokens") != row["actual_chat_tokens"]:
            errors.append("prompt_tokens_mismatch")
        count = usage.get("completion_tokens")
        if type(count) is not int or not 0 < count <= row["output_tokens"]:
            errors.append("completion_tokens_mismatch")
        record["success"] = not errors
        record["errors"] = errors
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        record["raw_error_body"] = exc.read(16384).decode("utf-8", "replace")
        record["communication_failure"] = True
        record["errors"] = ["HTTPError"]
    except (TimeoutError, socket.timeout) as exc:
        record["timeout"] = True
        record["communication_failure"] = True
        record["errors"] = [type(exc).__name__]
    except urllib.error.URLError as exc:
        record["communication_failure"] = True
        reason = exc.reason
        record["timeout"] = isinstance(reason, (TimeoutError, socket.timeout))
        record["errors"] = ["TimeoutError" if record["timeout"] else "URLError"]
    except OSError as exc:
        record["communication_failure"] = True
        record["timeout"] = isinstance(exc, (TimeoutError, socket.timeout))
        record["errors"] = [type(exc).__name__]
    except Exception as exc:
        # Never persist request headers, authentication or arbitrary exception bodies.
        record["errors"] = [type(exc).__name__]
    record.update(elapsed_seconds=time.monotonic() - started, end=utc())
    record["status"] = "succeeded" if record["success"] else "failed"
    return record


def smoke_request(run, c):
    """One independent real generation after readiness; never checks fixed-set token accounting."""
    settings = c["service"].get("smoke")
    if not settings or not settings.get("enabled", False):
        return None
    started = time.monotonic()
    record = {"status": "running", "started_at": utc(), "max_tokens": int(settings.get("max_tokens", 8)),
              "timeout_seconds": float(settings.get("timeout", 60)), "prompt": settings.get("prompt", "Reply with OK."),
              "actual_chat_tokens": None}
    target = Path(run) / "smoke.json"
    write_json(target, record)
    evaluation = c.get("evaluation", {})
    payload = {"model": c["service"]["served_model"],
               "messages": settings.get("messages") or [{"role": "user", "content": record["prompt"]}],
               "temperature": evaluation.get("temperature", 0), "top_p": evaluation.get("top_p", 1),
               "seed": evaluation.get("seed", 42), "max_tokens": record["max_tokens"], "stream": False,
               "chat_template_kwargs": evaluation.get("chat_template_kwargs", {})}
    if "add_special_tokens" in evaluation:
        payload["add_special_tokens"] = evaluation["add_special_tokens"]
    record["payload"] = payload
    write_json(target, record)
    try:
        req = urllib.request.Request(c["service"]["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers(c))
        with urllib.request.urlopen(req, timeout=record["timeout_seconds"]) as response:
            data = json.loads(response.read().decode("utf-8"))
            record.update(http_status=response.status, raw_response=data)
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not text and not reasoning:
            raise ValueError("smoke returned an empty generation")
        record.update(status="passed", response=text, reasoning_content=reasoning,
                      generation_channel="content" if text else "reasoning_content",
                      finish_reason=choice.get("finish_reason"),
                      usage=data.get("usage", {}))
    except urllib.error.HTTPError as exc:
        record.update(status="failed", http_status=exc.code,
                      error="HTTPError", raw_error_body=exc.read(16384).decode("utf-8", "replace"))
    except Exception as exc:
        record.update(status="failed", error=type(exc).__name__, detail=str(exc)[:1000])
    record.update(elapsed_seconds=time.monotonic() - started, finished_at=utc())
    write_json(target, record)
    if record["status"] != "passed":
        raise RuntimeError("startup smoke failed: " + record.get("error", "empty generation"))
    return record


def score(samples, predictions, contract):
    identifiers = [v["id"] for v in predictions]
    expected = {v["id"]: v for v in samples}
    if len(set(identifiers)) != len(identifiers) or not set(identifiers).issubset(expected):
        raise ValueError("duplicate or unexpected prediction IDs")
    by_id = {v["id"]: v for v in predictions}
    for pred in predictions:
        if pred.get("sample") != expected[pred["id"]]:
            raise ValueError("prediction sample differs from frozen input")
    metric = None
    if contract["metric"] == "ruler_official":
        spec = importlib.util.spec_from_file_location("ruler_metrics", contract["official_metrics"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric = module.string_match_all
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample.get("task", "task"), sample.get("context_limit", contract["context_limit"]))].append(sample)
    groups = []
    for (task, length), selected in sorted(grouped.items()):
        actual = [by_id.get(v["id"], {}) for v in selected]
        texts = [p.get("response", "") if p.get("success") else "" for p in actual]
        refs = [v["outputs"] for v in selected]
        value = float(metric(texts, refs)) if metric else 100 * sum(t in ref for t, ref in zip(texts, refs)) / len(refs)
        if not math.isfinite(value):
            raise ValueError("nonfinite metric result")
        groups.append({"task": task, "context_limit": length, "samples": len(selected), "score": value,
                       "missing": sum(not p for p in actual), "protocol_failures": sum(not p.get("success") for p in actual),
                       "truncated": sum(bool(p.get("truncated")) for p in actual),
                       "skipped": sum(p.get("status") == "skipped" for p in actual),
                       "timeout": sum(bool(p.get("timeout")) for p in actual)})
    # Weights are the declared fixed sample counts, including unsuccessful samples.
    value = sum(v["score"] * v["samples"] for v in groups) / len(samples)
    threshold = contract.get("minimum_score")
    if threshold is not None and (type(threshold) not in (int, float) or not 0 <= threshold <= 100):
        raise ValueError("minimum_score must be in [0, 100]")
    clean = all(v["protocol_failures"] == 0 for v in groups)
    attempted = sum(p.get("request_attempted", True) for p in predictions)
    succeeded = sum(bool(p.get("success")) for p in predictions)
    skipped = sum(p.get("status") == "skipped" for p in predictions)
    timed_out = sum(bool(p.get("timeout")) for p in predictions)
    return {"scope": contract["scope"], "metric": contract["metric"], "samples": len(samples),
            "attempted": attempted, "succeeded": succeeded, "failed": attempted - succeeded,
            "skipped": skipped, "timeout": timed_out,
            "protocol_pass": clean, "score": value, "groups": groups, "minimum_score": threshold,
            "quality_status": "not_specified" if threshold is None else ("passed" if value >= threshold and clean else "failed"),
            "numerical_parity": "not_run"}


def evaluate(run, c):
    samples = rows(Path(run) / "dataset.jsonl")
    validate_samples(samples, c["evaluation"])
    target = Path(run) / "raw" / "predictions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("predictions already exist; retain the run and create a new experiment")
    target.touch()
    predictions = []
    threshold = int(c["evaluation"].get("consecutive_communication_failure_threshold", 3))
    if threshold < 1:
        raise ValueError("consecutive_communication_failure_threshold must be positive")
    consecutive = 0

    def progress(status):
        counts = {"attempted": sum(p.get("request_attempted", True) for p in predictions),
                  "succeeded": sum(bool(p.get("success")) for p in predictions),
                  "failed": sum(p.get("request_attempted", True) and not p.get("success") for p in predictions),
                  "skipped": sum(p.get("status") == "skipped" for p in predictions),
                  "timeout": sum(bool(p.get("timeout")) for p in predictions),
                  "total": len(samples), "consecutive_communication_failures": consecutive}
        write_json(Path(run) / "evaluation-progress.json", {**counts, "status": status, "updated_at": utc()})
        state(run, "evaluation", status, **counts)

    progress("running")
    for index, sample in enumerate(samples):
        if consecutive >= threshold:
            for remaining in samples[index:]:
                skipped = {"id": remaining["id"], "sample": remaining, "request_attempted": False,
                           "success": False, "status": "skipped", "timeout": False,
                           "errors": ["skipped_after_consecutive_communication_failures"],
                           "response": "", "start": utc(), "end": utc()}
                append_json(target, skipped)
                predictions.append(skipped)
                progress("stopping_after_communication_failures")
            break
        prediction = request(sample, c)
        append_json(target, prediction)
        predictions.append(prediction)
        consecutive = consecutive + 1 if prediction.get("communication_failure") else 0
        progress("running")
    result = score(samples, predictions, c["evaluation"])
    write_json(Path(run) / "results.json", result)
    progress("passed" if result["protocol_pass"] else "failed")
    return result
