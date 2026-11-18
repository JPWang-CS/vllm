import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from fullnet.evaluation import evaluate, smoke_request
from fullnet.measurements import _csv_device_kernel_rows, _trace_device_kernel_events
from fullnet.store import read_json, rows


def _samples(count=5):
    return [{"id": f"s{i}", "prompt": "hello", "outputs": ["ok"],
             "actual_chat_tokens": 2, "output_tokens": 8} for i in range(count)]


def test_communication_failure_threshold_skips_rest_and_keeps_denominator(tmp_path, monkeypatch):
    samples = _samples()
    (tmp_path / "dataset.jsonl").write_text("".join(json.dumps(v) + "\n" for v in samples))
    attempts = []

    def failed_request(sample, config):
        attempts.append(sample["id"])
        return {"id": sample["id"], "sample": sample, "request_attempted": True,
                "success": False, "status": "failed", "timeout": len(attempts) == 1,
                "communication_failure": True, "errors": ["TimeoutError" if len(attempts) == 1 else "HTTPError"],
                "response": ""}

    monkeypatch.setattr("fullnet.evaluation.request", failed_request)
    c = {"service": {}, "evaluation": {"context_limit": 128, "metric": "exact_match", "scope": "test",
                                             "minimum_score": None, "consecutive_communication_failure_threshold": 3}}
    result = evaluate(tmp_path, c)
    predictions = rows(tmp_path / "raw/predictions.jsonl")
    assert attempts == ["s0", "s1", "s2"]
    assert [v["status"] for v in predictions] == ["failed"] * 3 + ["skipped"] * 2
    assert (result["samples"], result["attempted"], result["failed"], result["skipped"], result["timeout"]) == (5, 3, 3, 2, 1)
    assert result["score"] == 0
    progress = read_json(tmp_path / "evaluation-progress.json")
    assert progress["status"] == "failed" and progress["total"] == 5


def test_smoke_uses_evaluation_template_and_accepts_reasoning_channel(tmp_path, monkeypatch):
    captured = {}

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "", "reasoning_content": "OK"},
                                              "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 9, "completion_tokens": 1}}).encode()

    def open_url(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("fullnet.evaluation.urllib.request.urlopen", open_url)
    c = {"service": {"served_model": "m", "base_url": "http://127.0.0.1:1234/v1",
                      "smoke": {"enabled": True, "prompt": "short", "max_tokens": 8, "timeout": 60}},
         "evaluation": {"seed": 42, "temperature": 0, "top_p": 1,
                        "chat_template_kwargs": {"thinking": False}, "add_special_tokens": False}}
    result = smoke_request(tmp_path, c)
    assert result["status"] == "passed" and result["generation_channel"] == "reasoning_content"
    assert result["actual_chat_tokens"] is None and result["usage"]["prompt_tokens"] == 9
    assert captured["payload"]["chat_template_kwargs"] == {"thinking": False}
    assert captured["payload"]["add_special_tokens"] is False and captured["payload"]["max_tokens"] == 8
    assert captured["timeout"] == 60


def test_profile_requires_structural_non_cpu_device_kernel_evidence(tmp_path):
    cpu = tmp_path / "cpu.json"
    cpu.write_text(json.dumps({"traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 3, "args": {"name": "python worker"}},
        {"name": "npu_kernel_fake", "cat": "cpu_op", "ph": "X", "ts": 1, "dur": 5,
         "pid": 3, "args": {"Device Id": 0, "Task Type": "AI_CORE"}},
        {"name": "npu_kernel_fake", "cat": "kernel", "ph": "X", "ts": 1, "dur": 5, "pid": 3},
    ]}))
    device = tmp_path / "device.json"
    device.write_text(json.dumps({"traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 4, "args": {"name": "Ascend Hardware"}},
        {"name": "rms_norm_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 8, "pid": 4},
        {"name": "rms_norm_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 0, "pid": 4},
    ]}))
    assert _trace_device_kernel_events(cpu) == 0
    assert _trace_device_kernel_events(device) == 1

    csv_path = tmp_path / "kernel_details.csv"
    csv_path.write_text("Kernel Name,Task Type,Duration(us)\n"
                        "fake_cpu_kernel,CPU,10\n"
                        "empty,AI_CORE,0\n"
                        "rms_norm,AI_CORE,12\n")
    assert _csv_device_kernel_rows(csv_path) == 1


def test_pangu_runner_exports_failure_report_before_run_exists(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/ops/run_pangu92b.py"
    spec = importlib.util.spec_from_file_location("run_pangu92b_failure_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project"
    task = root / "runs" / "failed-before-plan"
    task.mkdir(parents=True)
    reference = tmp_path / "reference.json"
    extra = tmp_path / "model-extra.json"
    extra.write_text("{}")
    reference.write_text(json.dumps({"bootstrap": {"python": sys.executable, "source_roots": [],
        "cann_prefix": str(tmp_path), "vendor_roots": [], "native_root": str(tmp_path),
        "model_extra_source": str(extra)}}))
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"architectures": ["PanguUltraMoEForCausalLM"],
                                                       "num_hidden_layers": 46, "hidden_size": 2560}))
    (weights / "shard.safetensors").write_bytes(b"x")
    (weights / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "shard.safetensors"}}))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="device query failed", stderr=""))
    monkeypatch.setattr(sys, "argv", [str(script), "--project", str(root), "--reference", str(reference),
        "--weights", str(weights), "--devices", "0", "--npu-ids", "1", "--tp", "1", "--task", task.name])
    assert module.main() == 1
    validation = read_json(task / "validation.json")
    assert validation["exit_code"] == 1 and validation["model_requests"] == "not_run"
    assert (task / "task-report.md").is_file()
    assert (task / "export" / "task-report.md").is_file()
    assert read_json(task / "export" / "validation.json")["exit_code"] == 1
