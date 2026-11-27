import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

import pytest


spec = importlib.util.spec_from_file_location("fleet_bridge", Path(__file__).parents[1] / "scripts/ops/registered_fleet_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def request(**changes):
    value = {"request_id": "probe-01", "action": "run", "host": "registered",
             "job": "first", "script": "repair/probe.py", "devices": [1, 3], "args": [],
             "framework_source": {"schema": 1, "branch": "main", "receipt_path": ".runtime/framework-source.json"}}
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", [
    {"host": "other"}, {"script": "../secret.py"}, {"script": "/tmp/probe.py"},
    {"script": "repair/../../probe.py"}, {"script": "repair/probe.json"},
    {"request_id": "../../escape"}, {"devices": [1, 1]}, {"devices": [True]},
    {"devices": [-1]}, {"args": "shell command"}, {"action": "shell"},
    {"command": "arbitrary"},
])
def test_rejects_out_of_scope_request(changes):
    with pytest.raises((ValueError, TypeError)):
        bridge.validate_request(request(**changes), {"registered": {}})


def test_run_request_preserves_args_without_shell_interpolation():
    r = request(args=["literal $(command)", "space in argument"])
    assert bridge.validate_request(r, {"registered": {}})["args"] == r["args"]


def test_runtime_runner_compiles_and_records_job_exit():
    compile(bridge.RUNNER, "fleet_runner", "exec")
    assert "(j/'exit.txt').write_text" in bridge.RUNNER
    assert "No process in device" in bridge.RUNNER
    assert "os.killpg(child.pid,signal.SIGTERM)" in bridge.RUNNER


def _run_status_reader(tmp_path, files):
    root = tmp_path / "project"
    job = root / "runs" / "task" / "fleet-jobs" / "job"
    job.mkdir(parents=True)
    for name, value in files.items():
        (job / name).write_bytes(value)
    result = subprocess.run(
        [sys.executable, "-B", "-c", bridge.STATUS_READER, str(job), str(root), "task", "status"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), job


def test_status_reader_returns_complete_reasonable_json_over_old_tail_limit(tmp_path):
    document = json.dumps({"status": "completed", "padding": "x" * 18000})
    payload, _ = _run_status_reader(tmp_path, {"result.json": document.encode()})
    assert len(payload["result.json"]) > 14000
    assert json.loads(payload["result.json"])["status"] == "completed"


def test_status_reader_marks_oversized_json_as_truncated(tmp_path):
    document = json.dumps({"status": "completed", "padding": "x" * (256 * 1024)})
    payload, job = _run_status_reader(tmp_path, {"validation.json": document.encode()})
    value = payload["validation.json"]
    assert value["status"] == "truncated"
    assert value["truncated"] is True
    assert value["original_bytes"] == (job / "validation.json").stat().st_size
    assert value["limit_bytes"] == 256 * 1024
    assert value["path"].replace("\\", "/").endswith("/validation.json")


def test_status_reader_reads_logs_from_bounded_tail_with_byte_metadata(tmp_path):
    content = (b"early-line\n" + b"x" * 20000 + b"\nfinal-line\n")
    payload, job = _run_status_reader(tmp_path, {"workload.log": content})
    assert payload["workload.log"] == content[-14000:].decode("utf-8")
    value = payload["file_metadata"]["workload.log"]
    assert value["truncated"] is True
    assert value["original_bytes"] == len(content)
    assert value["snippet_start_byte"] == len(content) - 14000
    assert value["limit_bytes"] == 14000
    assert value["path"].replace("\\", "/").endswith("/workload.log")


def test_status_reader_keeps_legacy_exit_and_json_log_string_consumers(tmp_path):
    payload, _ = _run_status_reader(tmp_path, {
        "exit.txt": b"7\n",
        "workload.log": json.dumps({"phase": "done"}).encode(),
    })
    assert int(payload["exit.txt"].strip()) == 7
    assert json.loads(payload["workload.log"])["phase"] == "done"
    assert payload["file_metadata"]["exit.txt"]["truncated"] is False
    assert payload["file_metadata"]["workload.log"]["truncated"] is False


def test_status_reader_rejects_status_file_symlink_escape(tmp_path):
    root = tmp_path / "project"
    job = root / "runs" / "task" / "fleet-jobs" / "job"
    job.mkdir(parents=True)
    outside = tmp_path / "outside-state.json"
    outside.write_text('{"status":"outside"}', encoding="utf-8")
    try:
        (job / "state.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable in this Windows test environment")
    result = subprocess.run(
        [sys.executable, "-B", "-c", bridge.STATUS_READER, str(job), str(root), "task", "status"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "symlink or path escape" in result.stderr
