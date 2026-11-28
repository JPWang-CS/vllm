"""CPU tests for the formal single-host actor evidence publisher."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cross_host_role_under_test", ROOT / "scripts/ops/cross_host_role.py")
ROLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROLE)


def setup_job(tmp_path, *, completion=True):
    project = tmp_path / "project"
    job = project / "runs" / "case-tp23" / "fleet-jobs" / "job-head"
    job.mkdir(parents=True)
    (job / "role-spec.json").write_text(json.dumps({
        "case_id": "case-tp23", "model": "pangu-7b", "role": "head",
        "host": "192.0.2.10", "job": "job-head", "request_id": "req-head",
        "expected_ranks": [0], "expected_world": 2,
    }), encoding="utf-8")
    (job / "config.json").write_text(json.dumps({
        "project": str(project.resolve()), "management": "/registered/management",
        "container": "current-container", "job": "job-head", "devices": [3],
        "script": "repair/role.py", "args": ["--role", "head"],
    }), encoding="utf-8")
    (job / "validation.json").write_text(json.dumps({
        "schema": "tp.runtime-progress.v1", "stage": "head_ray_running",
        "model_started": True,
    }), encoding="utf-8")
    logs = project / ".r" / "case-tp23-head" / "session_current" / "logs"
    logs.mkdir(parents=True)
    text = "world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://redacted\n"
    if completion:
        text += "Loading weights took 19.79 seconds\n"
        text += "Model loading took 7.51 GiB memory and 20.027972 seconds\n"
    (logs / "worker-head.out").write_text(text, encoding="utf-8")
    return project, job


def collect(project, job):
    return ROLE.collect_publish(
        project=project, job_root=job, role="head", host="192.0.2.10",
        task_id="case-tp23", case_id="case-tp23", model="pangu-7b",
        request_id="req-head", job="job-head")


def test_single_host_entry_collects_and_merges_adapter_envelope(tmp_path):
    project, job = setup_job(tmp_path)
    result = collect(project, job)
    validation = json.loads((job / "validation.json").read_text(encoding="utf-8"))
    envelope = validation["actor_evidence"]
    assert result["status"] == "ACTOR_EVIDENCE_PUBLISHED"
    assert validation["stage"] == "head_ray_running"
    assert envelope["schema"] == "fullnet.actor-evidence.v1"
    assert envelope["rank_facts"]["0"]["rank"] == 0
    assert envelope["completion_facts"]
    assert envelope["request_id"] == "req-head"
    assert envelope["job"] == "job-head"


def test_single_host_entry_rejects_rank_without_completion(tmp_path):
    project, job = setup_job(tmp_path, completion=False)
    with pytest.raises(ROLE.CASE.CaseContractError, match="completion"):
        collect(project, job)
    validation = json.loads((job / "validation.json").read_text(encoding="utf-8"))
    assert "actor_evidence" not in validation
    assert validation["stage"] == "head_ray_running"


def test_single_host_publish_requires_real_registered_runner_config(tmp_path):
    project, job = setup_job(tmp_path)
    (job / "config.json").unlink()
    with pytest.raises(ROLE.CASE.CaseContractError, match="config.json"):
        collect(project, job)


def test_single_host_entry_rejects_old_role_root_and_wrong_job_identity(tmp_path):
    project, job = setup_job(tmp_path)
    with pytest.raises(ROLE.CASE.CaseContractError, match="basename"):
        ROLE.collect_publish(
            project=project, job_root=job, role="head", host="192.0.2.10",
            task_id="case-tp23", case_id="case-tp23", model="pangu-7b",
            request_id="req-head", job="old-job")
    old_logs = project / ".r" / "old-case-head" / "session_old" / "logs"
    old_logs.mkdir(parents=True)
    (old_logs / "worker-old.out").write_text(
        "world_size=2 rank=0 local_rank=0\nLoading weights took 1.0 seconds\n",
        encoding="utf-8")
    (project / ".r" / "case-tp23-head").rename(project / ".r" / "case-tp23-head-missing")
    with pytest.raises(ROLE.CASE.CaseContractError):
        collect(project, job)
