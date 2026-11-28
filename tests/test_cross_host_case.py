"""CPU contract tests for the public cross-host coordinator and rank gate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CASE = load("cross_host_case_under_test", ROOT / "scripts/ops/cross_host_case.py")
ADAPTER = load("multinode_adapter_under_test", ROOT / "models/pangu-92b/adapters/multinode_adapter.py")


def manifest(tmp_path):
    return {
        "schema": CASE.SCHEMA,
        "case_id": "case-tp23",
        "model": "pangu-7b",
        "world_size": 2,
        "expected_ranks": [0, 1],
        "roles": {
            "head": {"host": "192.0.2.10", "task_id": "case-tp23",
                     "expected_ranks": [0], "expected_world": 2, "model": "pangu-7b"},
            "worker": {"host": "192.0.2.11", "task_id": "case-tp23",
                       "expected_ranks": [1], "expected_world": 2, "model": "pangu-7b"},
        },
        "candidates": [
            {"role": "head", "host": "192.0.2.10", "device_ids": [3],
             "prepare_request_id": "prep-head", "prepare_job": "prep-head-job",
             "cpu_prepared": True, "source_ready": True,
             "environment_ready": True, "network_ready": True},
            {"role": "worker", "host": "192.0.2.11", "device_ids": [4],
             "prepare_request_id": "prep-worker", "prepare_job": "prep-worker-job",
             "cpu_prepared": True, "source_ready": True,
             "environment_ready": True, "network_ready": True},
        ],
    }


def write_manifest(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(manifest(tmp_path)), encoding="utf-8")
    return path


def write_pd_manifest(tmp_path):
    value = {
        "schema": CASE.SCHEMA, "scenario": "cross_host_pd_split",
        "case_id": "case-pd03", "model": "pangu-92b",
        "roles": {
            "producer": {"host": "192.0.2.20", "task_id": "case-pd03",
                          "expected_ranks": [0], "expected_world": 1, "model": "pangu-92b"},
            "consumer": {"host": "192.0.2.21", "task_id": "case-pd03",
                          "expected_ranks": [0], "expected_world": 1, "model": "pangu-92b"},
        },
        "candidates": [
            {"role": "producer", "host": "192.0.2.20", "device_ids": [3],
             "cpu_prepared": True, "source_ready": True, "environment_ready": True, "network_ready": True},
            {"role": "consumer", "host": "192.0.2.21", "device_ids": [4],
             "cpu_prepared": True, "source_ready": True, "environment_ready": True, "network_ready": True},
        ],
    }
    path = tmp_path / "pd-case.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def write_actor_log(project, task_id, role, text, session="session_current"):
    folder = project / ".r" / f"{task_id}-{role}" / session / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "worker-actor.out"
    path.write_text(text, encoding="utf-8")
    return path


def valid_records(tmp_path):
    write_actor_log(tmp_path, "case-tp23", "head",
                    "rank 0 in world size 2; weights loaded 7.51GiB\n")
    write_actor_log(tmp_path, "case-tp23", "worker",
                    "rank 1 in world size 2; weights loaded 7.51GiB\n")
    spec = manifest(tmp_path)
    records = ADAPTER.collect_actor_rank_evidence(
        tmp_path, spec["roles"], case_id=spec["case_id"], model=spec["model"])
    rank_hosts = {0: "192.0.2.10", 1: "192.0.2.11"}
    return ADAPTER.rank_evidence(
        actor_records=records, expected_ranks=[0, 1], expected_world=2,
        expected_hosts={"192.0.2.10", "192.0.2.11"}, expected_model="pangu-7b",
        expected_rank_hosts=rank_hosts, case_id="case-tp23"), records


def test_rank_gate_accepts_current_actor_logs_and_preserves_sources(tmp_path):
    evidence, records = valid_records(tmp_path)
    assert evidence["has_cross_host_ranks"] is True
    assert evidence["actor_ranks"] == [0, 1]
    assert evidence["loaded_models"] == ["pangu-7b"]
    assert all(records[index]["log_paths"] for index in range(2))
    assert all(Path(path).is_file() for record in records for path in record["log_paths"])


def test_rank_parser_accepts_real_parallel_state_and_completed_loading_lines(tmp_path):
    # Extracted, privacy-sanitized forms of the completed TP23 actor lines;
    # no host, IP, or original path is part of the public fixture.
    write_actor_log(tmp_path, "case-tp23", "head",
                    "world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://redacted\n"
                    "Loading weights took 19.79 seconds\n"
                    "Model loading took 7.51 GiB memory and 20.027972 seconds\n")
    write_actor_log(tmp_path, "case-tp23", "worker",
                    "world_size=2 rank=1 local_rank=0 distributed_init_method=tcp://redacted\n"
                    "Loading weights took 32.93 seconds\n"
                    "Model loading took 7.51 GiB memory and 33.256264 seconds\n")
    spec = manifest(tmp_path)
    records = ADAPTER.collect_actor_rank_evidence(
        tmp_path, spec["roles"], case_id=spec["case_id"], model=spec["model"])
    evidence = ADAPTER.rank_evidence(
        actor_records=records, expected_ranks=[0, 1], expected_world=2,
        expected_hosts={"192.0.2.10", "192.0.2.11"}, expected_model="pangu-7b",
        expected_rank_hosts={0: "192.0.2.10", 1: "192.0.2.11"}, case_id="case-tp23")
    assert evidence["has_cross_host_ranks"] is True
    assert evidence["actor_ranks"] == [0, 1]
    assert CASE.ADAPTER._WEIGHTS_RE.search("weights loading") is None
    assert CASE.ADAPTER._WEIGHTS_RE.search("Loading weights took 1.0 seconds") is not None


@pytest.mark.parametrize("builder", [
    "placement_only", "missing_rank", "wrong_world", "old_case_session",
    "weights_not_loaded", "one_host_only",
])
def test_rank_gate_rejects_incomplete_or_stale_actor_proof(tmp_path, builder):
    spec = manifest(tmp_path)
    if builder == "placement_only":
        result = ADAPTER.rank_evidence(
            "host-192.0.2.10 host-192.0.2.11",
            "node 192.0.2.10 node 192.0.2.11",
            expected_ranks=[0, 1], expected_world=2,
            expected_hosts={"192.0.2.10", "192.0.2.11"},
            expected_model="pangu-7b", expected_rank_hosts={0: "192.0.2.10", 1: "192.0.2.11"},
            case_id="case-tp23")
    else:
        if builder == "missing_rank":
            write_actor_log(tmp_path, "case-tp23", "head",
                            "rank 0 in world size 2; weights loaded\n")
            write_actor_log(tmp_path, "case-tp23", "worker", "noise only\n")
        elif builder == "wrong_world":
            write_actor_log(tmp_path, "case-tp23", "head",
                            "rank 0 in world size 3; weights loaded\n")
            write_actor_log(tmp_path, "case-tp23", "worker",
                            "rank 1 in world size 3; weights loaded\n")
        elif builder == "old_case_session":
            write_actor_log(tmp_path, "case-tp23-old", "head",
                            "rank 0 in world size 2; weights loaded\n")
            write_actor_log(tmp_path, "case-tp23-old", "worker",
                            "rank 1 in world size 2; weights loaded\n")
        elif builder == "weights_not_loaded":
            write_actor_log(tmp_path, "case-tp23", "head",
                            "rank 0 in world size 2; weights loading\n")
            write_actor_log(tmp_path, "case-tp23", "worker",
                            "rank 1 in world size 2; weights loading\n")
        else:
            write_actor_log(tmp_path, "case-tp23", "head",
                            "rank 0 in world size 2; weights loaded\n")
            # The worker has no current actor log.  A role/host label alone is
            # deliberately not converted into an actor record.
        records = ADAPTER.collect_actor_rank_evidence(
            tmp_path, spec["roles"], case_id=spec["case_id"], model=spec["model"])
        result = ADAPTER.rank_evidence(
            actor_records=records, expected_ranks=[0, 1], expected_world=2,
            expected_hosts={"192.0.2.10", "192.0.2.11"}, expected_model="pangu-7b",
            expected_rank_hosts={0: "192.0.2.10", 1: "192.0.2.11"}, case_id="case-tp23")
    assert result["has_cross_host_ranks"] is False, (builder, result)
    assert result["evidence_reasons"]


def test_explicit_old_role_root_and_symlink_parent_are_rejected(tmp_path):
    spec = manifest(tmp_path)
    old = tmp_path / ".r" / "old-case-head" / "session_old" / "logs"
    old.mkdir(parents=True)
    (old / "worker.out").write_text("rank 0 in world size 2; weights loaded\n", encoding="utf-8")
    altered = dict(spec["roles"]["head"])
    altered["log_root"] = str(old)
    with pytest.raises(ValueError, match="current role task directory"):
        ADAPTER.collect_actor_rank_evidence(
            tmp_path, {"head": altered, "worker": spec["roles"]["worker"]},
            case_id=spec["case_id"], model=spec["model"])

    link_root = tmp_path / ".r" / "case-tp23-head" / "session_link"
    link_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_root.symlink_to(old, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows test account cannot create symlinks: {exc}")
    altered["log_root"] = str(link_root)
    with pytest.raises(ValueError, match="symlink"):
        ADAPTER.collect_actor_rank_evidence(
            tmp_path, {"head": altered, "worker": spec["roles"]["worker"]},
            case_id=spec["case_id"], model=spec["model"])


def _complete_response(role, *, truncated=False, model_started=False):
    return {
        "file_metadata": {
            "state.json": {"json_valid": True, "truncated": truncated},
            "failure.json": {"json_valid": True, "truncated": truncated},
            "exit.txt": {"truncated": truncated},
            "validation.json": {"json_valid": True, "truncated": truncated},
        },
        "state.json": {"status": "exited", "exit": 1},
        "failure.json": {"error": "RuntimeError",
                          "detail": "device 3 is not confirmed healthy and empty"},
        "exit.txt": "1",
        "file_presence": {"process.json": False, "workload.log": False},
        "validation.json": {"stage": role, "model_started": model_started},
    }


def test_never_queued_intent_is_discarded_and_archive_is_explicit(tmp_path):
    decision = CASE.safe_requeue_decision(
        intents=[{"role": "front", "request_id": "req-front"}],
        request_paths=set(), observations={}, cleanup=None,
        expected_roles={"front", "consumer"},
        unsubmitted_roles={"front": {"request_present": False, "business_started": False},
                           "consumer": {"request_present": False, "business_started": False}},
        model_started=False)
    assert decision["allowed"] is True
    assert decision["action"] == "discard_never_queued_intent"
    archive = CASE.archive_resource_requeue(
        tmp_path, {"case_id": "case-pd", "entries": [{"request_id": "req-front"}]}, decision)
    assert archive["entries"] == []
    assert Path(archive["archive"]).is_file()


def test_active_job_is_adopted_and_never_replayed():
    decision = CASE.safe_requeue_decision(
        intents=[{"role": "front", "request_id": "req-front"}],
        request_paths={"req-front"},
        observations={"front": {"state.json": {"status": "running"}}},
        cleanup=None, expected_roles={"front"}, model_started=False)
    assert decision == {"allowed": False, "action": "adopt_active_job",
                        "reason": "active job exists; do not submit another request",
                        "active_roles": ["front"]}


def test_submitted_requeue_requires_complete_two_sided_resource_proof():
    responses = {"front": _complete_response("front"),
                 "consumer": _complete_response("consumer")}
    responses["front"]["_request_binding"] = {"request_id": "req-front", "host": "host-front", "job": "job-front"}
    responses["consumer"]["_request_binding"] = {"request_id": "req-consumer", "host": "host-consumer", "job": "job-consumer"}
    cleanup = {"verified": True, "jobs": [
        {"role": "front", "host": "host-front", "job": "job-front", "request_id": "req-front", "verified": True},
        {"role": "consumer", "host": "host-consumer", "job": "job-consumer", "request_id": "req-consumer", "verified": True},
    ]}
    decision = CASE.safe_requeue_decision(
        intents=[{"role": "front", "host": "host-front", "job": "job-front", "request_id": "req-front"},
                 {"role": "consumer", "host": "host-consumer", "job": "job-consumer", "request_id": "req-consumer"}],
        request_paths={"req-front", "req-consumer"}, observations=responses,
        cleanup=cleanup, expected_roles={"front", "consumer"}, model_started=False)
    assert decision["allowed"] is True
    assert decision["action"] == "archive_resource_requeue"

    responses["consumer"]["file_metadata"]["state.json"]["truncated"] = True
    blocked = CASE.safe_requeue_decision(
        intents=[{"role": "front", "request_id": "req-front"},
                 {"role": "consumer", "request_id": "req-consumer"}],
        request_paths={"req-front", "req-consumer"}, observations=responses,
        cleanup=cleanup, expected_roles={"front", "consumer"}, model_started=False)
    assert blocked["allowed"] is False
    assert blocked["action"] == "stop"


def test_partial_pd_submission_can_requeue_only_with_unsubmitted_role_proof():
    response = _complete_response("front")
    response["_request_binding"] = {"request_id": "req-front", "host": "host-front", "job": "job-front"}
    cleanup = {"verified": True, "jobs": [
        {"role": "front", "host": "host-front", "job": "job-front",
         "request_id": "req-front", "verified": True},
    ]}
    result = CASE.safe_requeue_decision(
        intents=[{"role": "front", "host": "host-front", "job": "job-front",
                  "request_id": "req-front"}],
        request_paths={"req-front"}, observations={"front": response}, cleanup=cleanup,
        expected_roles={"front", "consumer"},
        unsubmitted_roles={"consumer": {"request_present": False, "business_started": False}},
        model_started=False)
    assert result["allowed"] is True
    assert result["action"] == "archive_resource_requeue"


def test_worker_resource_gate_with_head_waiting_is_requeueable_only_after_both_cleanup_bindings():
    head = {
        "_request_binding": {"request_id": "req-head", "host": "host-head", "job": "job-head"},
        "file_presence": {"process.json": False, "workload.log": False},
        "file_metadata": {"validation.json": {"json_valid": True, "truncated": False}},
        "validation.json": {"stage": "head_ray_running", "model_started": False},
        "state.json": {"status": "exited", "exit": 0},
    }
    worker = _complete_response("worker")
    worker["_request_binding"] = {"request_id": "req-worker", "host": "host-worker", "job": "job-worker"}
    cleanup = {"verified": True, "jobs": [
        {"role": "head", "host": "host-head", "job": "job-head", "request_id": "req-head", "verified": True},
        {"role": "worker", "host": "host-worker", "job": "job-worker", "request_id": "req-worker", "verified": True},
    ]}
    result = CASE.safe_requeue_decision(
        intents=[{"role": "head", "host": "host-head", "job": "job-head", "request_id": "req-head"},
                 {"role": "worker", "host": "host-worker", "job": "job-worker", "request_id": "req-worker"}],
        request_paths={"req-head", "req-worker"}, observations={"head": head, "worker": worker},
        cleanup=cleanup, expected_roles={"head", "worker"}, model_started=False)
    assert result["allowed"] is True
    assert result["evidence"]["worker"]["resource"]["reasons"] == []


def test_old_job_cleanup_or_response_binding_cannot_authorize_requeue():
    response = _complete_response("front")
    response["_request_binding"] = {"request_id": "old-req", "host": "host-front", "job": "old-job"}
    cleanup = {"verified": True, "jobs": [
        {"role": "front", "host": "host-front", "job": "old-job",
         "request_id": "old-req", "verified": True},
    ]}
    result = CASE.safe_requeue_decision(
        intents=[{"role": "front", "host": "host-front", "job": "job-front",
                  "request_id": "req-front"}],
        request_paths={"req-front"}, observations={"front": response}, cleanup=cleanup,
        expected_roles={"front"}, model_started=False)
    assert result["allowed"] is False
    assert "bound" in result["reason"] or "binding" in result["reason"]


def test_two_phase_candidate_gate_requires_fresh_container_proof_and_intersection():
    spec = manifest(Path("."))
    row = {"device_id": 3, "health": "OK", "idle": True, "process_list_clear": True}
    def prepared(host, request_id, job):
        return {"response_path": f"/task/{job}.json",
                "_request_binding": {"host": host, "request_id": request_id, "job": job},
                "response": {
                    "state.json": {"status": "exited", "exit": 0},
                    "result.json": {"status": "CPU_PREPARED", "no_model_started": True,
                                     "source_manifest": {"complete": True, "files": ["src"]},
                                     "network_probe": {"complete": True, "reachable": True}},
                    "file_metadata": {"state.json": {"json_valid": True, "truncated": False},
                                      "result.json": {"json_valid": True, "truncated": False}},
                }}
    good_head = {"broker_checked_at": 100.0,
                 "container_query": {"devices": {"exit": 0, "stdout": ""},
                                      "device_rows": [row]},
                 "cpu_prepare": prepared("192.0.2.10", "prep-head", "prep-head-job")}
    good_worker = {"broker_checked_at": 100.0,
                   "container_query": {"devices": {"exit": 0, "stdout": ""},
                                        "device_rows": [{**row, "device_id": 4}]},
                   "cpu_prepare": prepared("192.0.2.11", "prep-worker", "prep-worker-job")}
    first = CASE.eligible_candidates(spec, {"192.0.2.10": good_head, "192.0.2.11": good_worker}, now=110)
    second = CASE.eligible_candidates(spec, {"192.0.2.10": good_head, "192.0.2.11": good_worker}, now=111)
    selected = CASE.select_two_phase(first["candidates"], second["candidates"], ["head", "worker"])
    assert set(selected) == {"head", "worker"}
    stale = CASE.eligible_candidates(spec, {"192.0.2.10": good_head}, now=400)
    assert stale["candidates"] == {}

    booleans_only = {"broker_checked_at": 100.0,
                     "container_query": {"devices": {"exit": 0, "stdout": ""},
                                          "device_rows": [row]}}
    rejected = CASE.eligible_candidates(spec, {"192.0.2.10": booleans_only}, now=110)
    assert rejected["candidates"] == {}
    assert "CPU prepare/source/network gate not proved" in rejected["rejected"]["head"][0]["reason"]


def test_public_case_driver_calls_recover_and_adapter_entries(tmp_path, monkeypatch):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    manifest_path = write_manifest(project)
    case = CASE.CrossHostCase(project, task, manifest_path)
    inventory_calls = []
    monkeypatch.setattr(CASE.RECOVER, "fleet_inventory_batch",
                        lambda *args, **kwargs: inventory_calls.append((args, kwargs)) or {"requests": []})
    case.inventory_requests()
    assert inventory_calls and inventory_calls[0][0][0] == project
    assert case.state["attempts"][0]["kind"] == "inventory"

    (task / "fleet" / "requests").mkdir(parents=True)
    request_id = "req-head"
    (task / "fleet" / "requests" / f"{request_id}.json").write_text(json.dumps({
        "request_id": request_id, "host": "192.0.2.10", "job": "job-head",
        "script": "repair/role.py", "devices": [3], "args": ["--role", "head"]}),
        encoding="utf-8")
    adopted = case.submit_role("head", host="192.0.2.10", job="job-head",
                               request_id=request_id, script="repair/role.py",
                               devices=[3], args=["--role", "head"])
    assert adopted["status"] == "adopted_existing_request"
    assert case.state["entries"][0]["role"] == "head"

    fresh = {"head": [{"role": "head", "host": "192.0.2.10", "device_ids": [3],
                        "inventory_at": 100, "inventory_proof": {"source": "test"}}],
             "worker": [{"role": "worker", "host": "192.0.2.11", "device_ids": [4],
                          "inventory_at": 100, "inventory_proof": {"source": "test"}}]}
    case.record_selection(fresh, fresh)
    calls = []
    monkeypatch.setattr(CASE.RECOVER, "fleet_submit",
                        lambda *args: calls.append(args) or {"status": "submitted_to_broker_queue"})
    submitted = case.submit_role("worker", host="192.0.2.11", job="job-worker",
                                 request_id="req-worker", script="repair/role.py",
                                 devices=[4], args=["--role", "worker"])
    assert submitted["status"] == "submitted_to_broker_queue"
    assert calls[0][1] == task
    assert calls[0][4] == "192.0.2.11"


def test_public_cross_host_evidence_merges_complete_fleet_role_receipts(tmp_path):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    manifest_path = write_manifest(project)
    case = CASE.CrossHostCase(project, task, manifest_path)
    case.state["entries"] = [
        {"role": "head", "host": "192.0.2.10", "job": "job-head", "request_id": "req-head"},
        {"role": "worker", "host": "192.0.2.11", "job": "job-worker", "request_id": "req-worker"},
    ]
    def response(role, host, job, rank, path, rank_line, completion_line=None):
        envelope = {
            "schema": "fullnet.actor-evidence.v1", "case_id": "case-tp23",
            "model": "pangu-7b", "role": role, "host": host,
            "job": job, "request_id": "req-" + role, "log_paths": [path],
            "rank_facts": [{"rank": rank, "world_size": 2, "path": path,
                            "line": rank_line}],
            "completion_facts": ([] if completion_line is None else
                                 [{"rank": rank, "path": path, "lines": [completion_line]}]),
        }
        validation = {
            "schema": "tp.runtime-progress.v1",
            "model_started": True, "model_ready": True,
            "actor_evidence": envelope,
        }
        return {
            "response_path": str(task / "fleet" / "responses" / f"{job}.json"),
            "response": {
                "job": f"/remote/runs/case-tp23/fleet-jobs/{job}",
                "validation.json": validation,
                "file_metadata": {"validation.json": {"json_valid": True, "truncated": False}},
            },
        }
    case.state["observations"] = {
        "head": {"entry": case.state["entries"][0], "status": response(
            "head", "192.0.2.10", "job-head", 0,
            "/data2/project/.r/case-tp23-head/session_a/logs/worker-a.out",
            "world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://redacted",
            "Loading weights took 19.79 seconds")},
        "worker": {"entry": case.state["entries"][1], "status": response(
            "worker", "192.0.2.11", "job-worker", 1,
            "/data2/project/.r/case-tp23-worker/session_b/logs/worker-b.out",
            "world_size=2 rank=1 local_rank=0 distributed_init_method=tcp://redacted",
            "Model loading took 7.51 GiB memory and 33.256264 seconds")},
    }
    evidence = case.collect_rank_evidence()
    assert evidence["has_cross_host_ranks"] is True
    assert set(evidence["actor_log_paths"]) == {
        "/data2/project/.r/case-tp23-head/session_a/logs/worker-a.out",
        "/data2/project/.r/case-tp23-worker/session_b/logs/worker-b.out",
    }


def test_public_cross_host_evidence_rejects_rank_lines_without_completion_envelope(tmp_path):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    case = CASE.CrossHostCase(project, task, write_manifest(project))
    case.state["entries"] = [
        {"role": "head", "host": "192.0.2.10", "job": "job-head", "request_id": "req-head"},
        {"role": "worker", "host": "192.0.2.11", "job": "job-worker", "request_id": "req-worker"},
    ]
    observations = {}
    for role, host, rank in (("head", "192.0.2.10", 0), ("worker", "192.0.2.11", 1)):
        path = f"/data2/project/.r/case-tp23-{role}/session_x/logs/worker-x.out"
        envelope = {"schema": "fullnet.actor-evidence.v1", "case_id": "case-tp23",
                    "model": "pangu-7b", "role": role, "host": host,
                    "job": f"job-{role}", "request_id": f"req-{role}",
                    "log_paths": [path],
                    "rank_facts": [{"rank": rank, "world_size": 2, "path": path,
                                    "line": f"world_size=2 rank={rank} local_rank=0"}],
                    "completion_facts": []}
        observations[role] = {"entry": next(item for item in case.state["entries"] if item["role"] == role),
                              "status": {"response_path": f"/task/{role}.json",
                                         "response": {"job": f"/remote/{role}/job-{role}",
                                                      "validation.json": {"actor_evidence": envelope},
                                                      "file_metadata": {"validation.json": {"json_valid": True, "truncated": False}}}}}
    case.state["observations"] = observations
    with pytest.raises(CASE.CaseContractError, match="completion"):
        case.collect_rank_evidence()


def test_public_cross_host_evidence_rejects_missing_remote_role_receipt(tmp_path):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    case = CASE.CrossHostCase(project, task, write_manifest(project))
    case.state["entries"] = [{"role": "head", "host": "192.0.2.10",
                               "job": "job-head", "request_id": "req-head"}]
    case.state["observations"] = {}
    with pytest.raises(CASE.CaseContractError, match="current role"):
        case.collect_rank_evidence()


def test_public_cli_validate_uses_the_same_manifest_contract(tmp_path, capsys):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    result = CASE.main(["--project", str(project), "--task", "case-tp23",
                        "--manifest", str(write_manifest(project)), "--action", "validate"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "CPU_PREPARED"
    assert payload["no_model_started"] is True


def test_public_cli_select_and_requeue_read_task_local_originals(tmp_path, capsys):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    manifest_path = write_manifest(project)
    selected = {
        "head": [{"role": "head", "host": "192.0.2.10", "device_ids": [3],
                   "inventory_at": 100, "inventory_proof": {"source": "fleet"},
                   "cpu_prepare_proof": {"response_path": "/task/head.json"}}],
        "worker": [{"role": "worker", "host": "192.0.2.11", "device_ids": [4],
                     "inventory_at": 100, "inventory_proof": {"source": "fleet"},
                     "cpu_prepare_proof": {"response_path": "/task/worker.json"}}],
    }
    (task / "repair" / "CROSS_HOST_FIRST_SCAN.json").write_text(json.dumps(selected), encoding="utf-8")
    (task / "repair" / "CROSS_HOST_SECOND_SCAN.json").write_text(json.dumps(selected), encoding="utf-8")
    assert CASE.main(["--project", str(project), "--task", "case-tp23",
                      "--manifest", str(manifest_path), "--action", "select"]) == 0
    capsys.readouterr()
    state = json.loads((task / "repair" / "CROSS_HOST_STATE.json").read_text(encoding="utf-8"))
    assert state["selection"]["verified"] is True

    state["entries"] = [{"role": "head", "request_id": "req-head"}]
    (task / "repair" / "CROSS_HOST_STATE.json").write_text(json.dumps(state), encoding="utf-8")
    (task / "repair" / "CROSS_HOST_REQUEUE_EVIDENCE.json").write_text(json.dumps({
        "unsubmitted_roles": {
            "head": {"request_present": False, "business_started": False},
            "worker": {"request_present": False, "business_started": False},
        }
    }), encoding="utf-8")
    assert CASE.main(["--project", str(project), "--task", "case-tp23",
                      "--manifest", str(manifest_path), "--action", "requeue"]) == 0
    capsys.readouterr()
    final = json.loads((task / "repair" / "CROSS_HOST_STATE.json").read_text(encoding="utf-8"))
    assert final["status"] == CASE.RESOURCE_GATE_REQUEUE
    assert final["entries"] == []


def test_public_cli_rank_evidence_rejection_returns_nonzero(tmp_path, capsys):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "repair").mkdir(parents=True)
    manifest_path = write_manifest(project)
    entries = [
        {"role": "head", "host": "192.0.2.10", "job": "job-head", "request_id": "req-head"},
        {"role": "worker", "host": "192.0.2.11", "job": "job-worker", "request_id": "req-worker"},
    ]
    observations = {}
    for role, host, rank in (("head", "192.0.2.10", 0), ("worker", "192.0.2.11", 1)):
        job = f"job-{role}"
        request = f"req-{role}"
        path = f"/remote/.r/case-tp23-{role}/session_x/logs/worker-x.out"
        envelope = {"schema": "fullnet.actor-evidence.v1", "case_id": "case-tp23",
                    "model": "pangu-7b", "role": role, "host": host,
                    "job": job, "request_id": request, "log_paths": [path],
                    "rank_facts": [{"rank": rank, "world_size": 3, "path": path,
                                    "line": f"world_size=3 rank={rank} local_rank=0"}],
                    "completion_facts": [{"rank": rank, "path": path,
                                           "lines": ["Loading weights took 1.0 seconds"]}]}
        response = {"job": f"/remote/fleet-jobs/{job}",
                    "validation.json": {"actor_evidence": envelope},
                    "file_metadata": {"validation.json": {"json_valid": True, "truncated": False}}}
        observations[role] = {"entry": next(item for item in entries if item["role"] == role),
                              "status": {"response_path": f"/remote/{job}.json", "response": response}}
    (task / "repair" / "CROSS_HOST_STATE.json").write_text(json.dumps({
        "schema": CASE.STATE_SCHEMA, "case_id": "case-tp23", "status": "RUNNING",
        "entries": entries, "observations": observations,
    }), encoding="utf-8")
    result = CASE.main(["--project", str(project), "--task", "case-tp23",
                        "--manifest", str(manifest_path), "--action", "rank-evidence"])
    output = capsys.readouterr().out
    assert result == 1
    assert '"has_cross_host_ranks": false' in output


def test_public_cli_requeue_block_is_nonzero(tmp_path, capsys):
    project = tmp_path / "project"
    task = project / "runs" / "case-tp23"
    (task / "fleet" / "requests").mkdir(parents=True)
    (task / "repair").mkdir(parents=True)
    manifest_path = write_manifest(project)
    (task / "fleet" / "requests" / "req-head.json").write_text("{}", encoding="utf-8")
    (task / "repair" / "CROSS_HOST_STATE.json").write_text(json.dumps({
        "schema": CASE.STATE_SCHEMA, "case_id": "case-tp23", "status": "FAILED",
        "entries": [{"role": "head", "host": "192.0.2.10", "job": "job-head", "request_id": "req-head"}],
    }), encoding="utf-8")
    (task / "repair" / "CROSS_HOST_REQUEUE_EVIDENCE.json").write_text(json.dumps({
        "unsubmitted_roles": {"worker": {"request_present": False, "business_started": False}}
    }), encoding="utf-8")
    result = CASE.main(["--project", str(project), "--task", "case-tp23",
                        "--manifest", str(manifest_path), "--action", "requeue"])
    output = capsys.readouterr().out
    assert result == 1
    assert '"allowed": false' in output
    assert '"action": "stop"' in output


def test_pd_manifest_uses_independent_tp1_role_worlds_for_select_and_requeue(tmp_path):
    project = tmp_path / "project"
    task = project / "runs" / "case-pd03"
    (task / "repair").mkdir(parents=True)
    manifest_path = write_pd_manifest(project)
    spec = CASE.load_case_spec(manifest_path)
    assert spec["scenario"] == "cross_host_pd_split"
    assert spec["world_size"] is None
    assert spec["roles"]["producer"]["expected_world"] == 1
    assert spec["roles"]["consumer"]["expected_world"] == 1
    assert spec["roles"]["producer"]["expected_ranks"] == [0]
    assert spec["roles"]["consumer"]["expected_ranks"] == [0]

    case = CASE.CrossHostCase(project, task, manifest_path)
    selected = {
        "producer": [{"role": "producer", "host": "192.0.2.20", "device_ids": [3],
                      "inventory_at": 100, "inventory_proof": {"source": "fleet"},
                      "cpu_prepare_proof": {"response_path": "/task/producer.json"}}],
        "consumer": [{"role": "consumer", "host": "192.0.2.21", "device_ids": [4],
                       "inventory_at": 100, "inventory_proof": {"source": "fleet"},
                       "cpu_prepare_proof": {"response_path": "/task/consumer.json"}}],
    }
    assert set(case.record_selection(selected, selected)) == {"producer", "consumer"}
    case.state["entries"] = [{"role": "producer", "request_id": "req-producer"}]
    case.save()
    result = case.decide_requeue({"unsubmitted_roles": {
        "producer": {"request_present": False, "business_started": False},
        "consumer": {"request_present": False, "business_started": False},
    }})
    assert result["status"] == "WAITING_RESOURCES"


def test_pd_case_cannot_enter_tp_rank_evidence_gate(tmp_path):
    project = tmp_path / "project"
    task = project / "runs" / "case-pd03"
    (task / "repair").mkdir(parents=True)
    case = CASE.CrossHostCase(project, task, write_pd_manifest(project))
    with pytest.raises(CASE.CaseContractError, match="P/D KV-transfer"):
        case.collect_rank_evidence()
