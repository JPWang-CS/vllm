import importlib.util
import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from fullnet.evaluation import evaluate, smoke_request
from fullnet.measurements import (_capture_parse_evidence, _csv_device_kernel_rows,
                                  _offline_parse_profiles, _trace_device_kernel_events,
                                  msprof_shadow_toolkit, profile_artifact_index)
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
        {"name": "process_name", "ph": "M", "pid": 4,
         "args": {"name": "Ascend Hardware", "Device Id": 0}},
        {"name": "rms_norm_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 8, "pid": 4,
         "args": {"Device Id": 0, "Task Type": "AI_CORE"}},
        {"name": "rms_norm_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 0, "pid": 4},
    ]}))
    assert _trace_device_kernel_events(cpu) == 0
    assert _trace_device_kernel_events(device) == 1

    csv_path = tmp_path / "kernel_details.csv"
    csv_path.write_text("Device_id,Kernel Name,Task Type,Duration(us)\n"
                        "0,fake_cpu_kernel,CPU,10\n"
                        "0,empty,AI_CORE,0\n"
                        "0,rms_norm,AI_CORE,12\n")
    assert _csv_device_kernel_rows(csv_path) == 1


def test_device_csv_requires_positive_duration_and_device_identity(tmp_path):
    csv_path = tmp_path / "Device.csv"
    csv_path.write_text("Device Id,Task Type,Duration(us)\n"
                        "0,AI_CORE,0\n"
                        "0,AICPU,15\n"
                        "0,AI_CORE,12\n")
    assert _csv_device_kernel_rows(csv_path) == 1


def test_compute_gate_accepts_cann_types_and_rejects_non_compute_evidence(tmp_path):
    trace_path = tmp_path / "compute-gate.json"
    events = [
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "HCCL Hardware", "Device Id": 0}},
        {"name": "hccl_kernel", "cat": "kernel", "ph": "X", "dur": 8, "pid": 1,
         "args": {"Device_id": 0}},
        {"name": "process_name", "ph": "M", "pid": 2,
         "args": {"name": "Ascend Hardware", "Device Id": 0}},
        {"name": "cpu_submit", "cat": "cpu_op", "ph": "X", "dur": 8, "pid": 2,
         "args": {"Device_id": 0, "Task Type": "AI_CORE"}},
        {"name": "device_only_comm", "cat": "kernel", "ph": "X", "dur": 8, "pid": 2,
         "args": {"Device_id": 0}},
        {"name": "aicpu", "cat": "aicore", "ph": "X", "dur": 8, "pid": 2,
         "args": {"Device_id": 0, "Task Type": "AICPU"}},
    ]
    for index, task_type in enumerate(("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")):
        events.append({"name": f"compute_{index}", "cat": "kernel", "ph": "X", "dur": 8,
                       "pid": 2, "args": {"Device_id": 0, "Task Type": task_type}})
    for value in (0, "nan", "inf"):
        events.append({"name": "bad_duration", "cat": "kernel", "ph": "X", "dur": value,
                       "pid": 2, "args": {"Device_id": 0, "Task Type": "AI_CORE"}})
    trace_path.write_text(json.dumps({"traceEvents": events}))
    assert _trace_device_kernel_events(trace_path) == 4

    csv_path = tmp_path / "op_summary.csv"
    rows = [
        ("1", "AI_CORE", "2"), ("1", "AI_VECTOR_CORE", "3"),
        ("1", "MIX_AIC", "4"), ("1", "MIX_AIV", "5"),
        ("1", "HCCL", "6"), ("1", "AICPU", "7"),
        ("1", "COMMUNICATION", "8"), ("1", "AI_CORE", "0"),
        ("1", "AI_CORE", "nan"), ("1", "AI_CORE", "inf"),
    ]
    csv_path.write_text("Device_id,Task Type,Task Duration(us),Op Name\n" +
                        "".join(f"{device},{task},{duration},op\n" for device, task, duration in rows))
    assert _csv_device_kernel_rows(csv_path) == 4

    device_only = tmp_path / "device-only-communication.csv"
    device_only.write_text("Device_id,Task Duration(us),Op Name\n1,8,communication\n")
    assert _csv_device_kernel_rows(device_only) == 0


def test_profile_artifact_index_discovers_extra_timeline_forms_deduplicates_and_rejects_cpu(tmp_path):
    run = tmp_path / "run"
    extra = run / "offline-export"
    extra.mkdir(parents=True)
    events = {"traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 7,
         "args": {"name": "NPU Device 1"}},
        {"name": "compute", "ph": "X", "cat": "kernel", "pid": 7,
         "args": {"Task Type": "AI_CORE"}, "dur": 2},
    ]}
    (extra / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
    (extra / "msprof_001.json").write_text(json.dumps(events), encoding="utf-8")
    with gzip.open(extra / "rank.pt.trace.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(events, handle)
    indexed = profile_artifact_index(run, extra_dirs=[extra, extra])
    traces = [item for item in indexed["artifacts"] if item["kind"] == "trace"]
    assert len(traces) == 3 and all(item["device_kernel_events"] == 1 for item in traces)
    assert indexed["has_device_kernel_evidence"] is True

    cpu = tmp_path / "cpu-run" / "offline-export"
    cpu.mkdir(parents=True)
    (cpu / "trace_view.json").write_text(json.dumps({"traceEvents": [
        {"name": "cpu", "ph": "X", "cat": "cpu_op", "pid": 1,
         "args": {"Task Type": "AI_CORE"}, "dur": 5},
    ]}), encoding="utf-8")
    cpu_result = profile_artifact_index(cpu.parent, extra_dirs=[cpu])
    assert cpu_result["has_device_kernel_evidence"] is False
    assert cpu_result["acceptance"] == "not_accepted"

    bad = tmp_path / "bad-run" / "offline-export"
    bad.mkdir(parents=True)
    (bad / "msprof_bad.json").write_text("{broken", encoding="utf-8")
    bad_result = profile_artifact_index(bad.parent, extra_dirs=[bad])
    assert bad_result["has_device_kernel_evidence"] is False
    assert bad_result["artifacts"][0]["substantive"] is False


def test_profiler_shadow_does_not_modify_source_sdk_and_keeps_executable(tmp_path):
    toolkit = tmp_path / "toolkit"
    binary = toolkit / "bin" / "msprof"
    profiler = toolkit / "tools" / "profiler" / "profiler_tool" / "analysis" / "msprof"
    binary.parent.mkdir(parents=True)
    profiler.mkdir(parents=True)
    binary.write_bytes(b"msprof")
    binary.chmod(0o755)
    script = profiler / "msprof.py"
    script.write_text("print('ok')\n")
    source_before = {str(p.relative_to(toolkit)): p.read_bytes() for p in toolkit.rglob("*") if p.is_file()}
    run = tmp_path / "run"
    run.mkdir()
    result = msprof_shadow_toolkit(toolkit, run)
    assert result["msprof_py_exists"] is True
    assert {str(p.relative_to(toolkit)): p.read_bytes() for p in toolkit.rglob("*") if p.is_file()} == source_before
    if os.name == "posix":
        assert (tmp_path / "run" / "performance" / "msprof-shadow" / "tools" / "profiler" / "bin" / "msprof").stat().st_mode & 0o111


def test_parse_evidence_preserves_error_tail_and_relax_keeps_exec_bit(tmp_path):
    run = tmp_path / "run"
    profiles = run / "profiles" / "rank0"
    profiles.mkdir(parents=True)
    log = profiles / "parse.log"
    log.write_text("first-error\nfinal-tail-error\n")
    executable = profiles / "msprof-helper"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    evidence = _capture_parse_evidence(run, run / "performance")
    assert evidence["logs"][0]["tail"].rstrip("\r\n").endswith("final-tail-error")
    if os.name == "posix":
        assert executable.stat().st_mode & 0o111


def test_offline_parse_keeps_first_error_and_does_not_accept_empty_trace(tmp_path, monkeypatch):
    run = tmp_path / "run"
    profiles = run / "profiles"
    (profiles / "rank0_ascend_pt" / "PROF_0").mkdir(parents=True)
    (profiles / "rank1_ascend_pt" / "PROF_1").mkdir(parents=True)
    for rank,device in ((0,4),(1,5)):
        data=profiles/f"rank{rank}_ascend_pt"/f"PROF_{rank}"/f"device_{device}"/"data"
        data.mkdir(parents=True)
        (data/"stars_soc").write_bytes(b"raw")
    (tmp_path/"toolkit").mkdir()
    shadow = run / "performance" / "msprof-shadow"
    shadow.mkdir(parents=True)
    msprof_py = shadow / "msprof.py"
    msprof_py.write_text("# fixture\n")
    (run / "performance" / "profiler-tools.json").parent.mkdir(parents=True, exist_ok=True)
    (run / "performance" / "profiler-tools.json").write_text(json.dumps({
        "home": str(tmp_path / "toolkit"),
        "env_updates": {},
        "shadow_toolkit": {"msprof_py": str(msprof_py)},
    }))
    (run / "config.json").write_text(json.dumps({"devices":[4,5],"service": {"argv": [sys.executable]}}))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="later error\n", stderr="first offline ERROR\n")

    monkeypatch.setattr("fullnet.measurements.subprocess.run", fake_run)
    result = _offline_parse_profiles(run)
    assert len(calls) == 1  # SQLite fails before any official CLI/import or raw copy.
    assert result["first_error"] == "first offline ERROR"
    assert result["acceptance"] == "not_accepted"


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
