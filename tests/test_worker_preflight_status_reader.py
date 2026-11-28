"""Regression tests for current-job worker preflight evidence exposure."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "ops" / "registered_fleet_bridge.py"
SPEC = importlib.util.spec_from_file_location("fleet_status_reader_worker_evidence", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


def run_status(tmp_path, *, post=True, plan=False):
    project = tmp_path / "project"
    job = project / "runs" / "task" / "fleet-jobs" / "worker-job"
    job.mkdir(parents=True)
    if post:
        (job / "device-6-post.txt").write_bytes(
            b"not_run: workload did not start; post device query skipped\n")
    if plan:
        (job / "plan.json").write_text(json.dumps({"status": "reviewable", "case": "pd"}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-B", "-c", BRIDGE.STATUS_READER,
         str(job), str(project), "task", "status"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_status_reader_exposes_current_job_absence_and_post_bytes(tmp_path):
    value = run_status(tmp_path)

    assert value["file_presence"]["process.json"] is False
    assert value["file_presence"]["workload.log"] is False
    assert value["file_presence"]["device-6-post.txt"] is True
    assert value["device-6-post.txt"] == "not_run: workload did not start; post device query skipped\n"
    assert value["file_metadata"]["device-6-post.txt"]["original_bytes"] == 59
    assert value["file_metadata"]["device-6-post.txt"]["truncated"] is False


def test_status_reader_marks_post_absence_without_inference(tmp_path):
    value = run_status(tmp_path, post=False)

    assert value["file_presence"]["process.json"] is False
    assert value["file_presence"]["workload.log"] is False
    assert "device-6-post.txt" not in value["file_presence"]


def test_status_reader_returns_complete_current_job_plan_json(tmp_path):
    value = run_status(tmp_path, plan=True)

    assert json.loads(value["plan.json"]) == {"status": "reviewable", "case": "pd"}
    assert value["file_metadata"]["plan.json"]["json_valid"] is True
    assert value["file_metadata"]["plan.json"]["truncated"] is False
