import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ops/recover_framework.py"


def _module():
    spec = importlib.util.spec_from_file_location("recover_checkpoint", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy(task, **overrides):
    value = {
        "max_total_jobs": 20, "max_failures_per_plan": 3,
        "max_controller_retries": 2, "retry_backoff_seconds": 10,
        "max_retry_backoff_seconds": 30,
    }
    value.update(overrides)
    (task / "repair").mkdir(parents=True)
    (task / "repair/EXECUTION_POLICY.json").write_text(json.dumps(value), encoding="utf-8")


def test_timeout_failure_is_recorded_with_bounded_exponential_backoff(tmp_path):
    module = _module()
    task = tmp_path / "task"
    _policy(task)
    first = module.record_controller_failure(task, "RequestTimeout", "API request timed out", now=100,
                                             controller_identity="controller-1")
    second = module.record_controller_failure(task, "TimeoutError", "read timed out", now=200,
                                              controller_identity="controller-1")
    assert first["failure_class"] == "CONTROLLER_API_TIMEOUT"
    assert first["retry_count"] == 0 and first["observation_count"] == 1
    assert second["retry_count"] == 0 and second["observation_count"] == 2
    attempt = module.record_controller_failure(task, "RequestTimeout", "resume attempt", now=210,
                                               controller_identity="controller-1",
                                               resume_attempt=True, resume_attempt_id="resume-1")
    repeat = module.record_controller_failure(task, "RequestTimeout", "same attempt observed", now=220,
                                              controller_identity="controller-1",
                                              resume_attempt=True, resume_attempt_id="resume-1")
    assert attempt["retry_count"] == 1 and attempt["backoff_seconds"] == 10
    assert repeat["retry_count"] == 1 and repeat["observation_count"] == 4
    assert attempt["next_retry_at"] == 220


def test_business_checkpoint_exposes_completed_running_and_pending_without_heartbeat(tmp_path):
    module = _module()
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    (task / "repair/BUSINESS_CHECKPOINT.json").write_text(json.dumps({"cases": {
        "mixed": {"status": "completed"},
        "same_host_pd_split": {"status": "running", "job_id": "job-split"},
        "multi_host_tp": {"status": "pending"},
        "cross_host_pd_split": {"status": "failed", "job_id": "old-job", "last_error": "timeout"},
    }}), encoding="utf-8")
    (task / "fleet/requests").mkdir(parents=True)
    (task / "fleet/requests/job-split.json").write_text(json.dumps({
        "request_id": "job-split", "action": "run", "job": "job-split",
    }), encoding="utf-8")
    result = module.business_checkpoint_status(task, ("mixed", "same_host_pd_split", "multi_host_tp", "cross_host_pd_split"))
    assert result["controller_independent"] is True
    assert result["cases"]["mixed"]["state"] == "completed"
    assert result["cases"]["same_host_pd_split"]["state"] == "running"
    assert result["cases"]["multi_host_tp"]["state"] == "pending"
    assert result["cases"]["cross_host_pd_split"]["state"] == "failed"
    assert result["submitted_request_ids"] == ["job-split"]


def test_resume_gate_requires_strict_cleanup_and_does_not_treat_history_as_active(tmp_path):
    module = _module()
    task = tmp_path / "task"
    _policy(task)
    terminal = {"exit": 1, "model": "glm-5.3-flash", "child_alive": False,
                "fifo_removed": True, "credentials_persisted": False}
    assert module.resume_gate(task, {**terminal, "child_alive": True}, {}, now=100)["status"] == "BLOCKED_CONTROLLER_NOT_STRICTLY_TERMINAL"
    (task / "repair/BUSINESS_CHECKPOINT.json").write_text(json.dumps({"cases": {
        "mixed": {"status": "submitted", "request_id": "req-mixed"}
    }}), encoding="utf-8")
    blocked = module.resume_gate(task, {**terminal, "cleanup_complete": True}, {"remaining_direct_children": [], "model_waited": True}, now=100)
    assert blocked["status"] == "RESUME_ALLOWED"
    assert blocked["checkpoint"]["cases"]["mixed"]["submitted"] is True


def test_resume_gate_allows_only_after_backoff_and_budget(tmp_path):
    module = _module()
    task = tmp_path / "task"
    _policy(task)
    terminal = {"exit": 1, "model": "glm-5.3-flash", "child_alive": False,
                "fifo_removed": True, "credentials_persisted": False}
    cleanup = {"remaining_direct_children": [], "model_waited": True}
    terminal["cleanup_complete"] = True
    module.record_controller_failure(task, "RequestTimeout", "timed out", now=100,
                                     controller_identity="controller-1")
    module.record_controller_failure(task, "RequestTimeout", "resume attempt", now=100,
                                     controller_identity="controller-1", resume_attempt=True,
                                     resume_attempt_id="resume-1")
    assert module.resume_gate(task, terminal, cleanup, now=105)["status"] == "BACKOFF_ACTIVE"
    assert module.resume_gate(task, terminal, cleanup, now=110)["status"] == "RESUME_ALLOWED"
    module.record_controller_failure(task, "RequestTimeout", "resume attempt", now=110,
                                     controller_identity="controller-1", resume_attempt=True,
                                     resume_attempt_id="resume-1")
    module.record_controller_failure(task, "RequestTimeout", "resume attempt", now=130,
                                     controller_identity="controller-1", resume_attempt=True,
                                     resume_attempt_id="resume-2")
    exhausted = module.record_controller_failure(task, "RequestTimeout", "resume attempt", now=160,
                                                 controller_identity="controller-1", resume_attempt=True,
                                                 resume_attempt_id="resume-3")
    assert exhausted["retry_count"] == 3
    assert module.resume_gate(task, terminal, cleanup, now=1000)["status"] == "CONTROLLER_RETRY_BUDGET_EXHAUSTED"
    # Hourly user-authorized supervision opens a fresh bounded retry window.
    assert module.resume_gate(task, terminal, cleanup, now=4000)["status"] == "RESUME_ALLOWED"
    next_hour = module.record_controller_failure(task, "RequestTimeout", "hourly continuation", now=4000,
                                                 controller_identity="controller-2", resume_attempt=True,
                                                 resume_attempt_id="resume-next-hour")
    assert next_hour["retry_count"] == 1
    assert next_hour["resume_attempts"] == 4


def test_resume_gate_requires_explicit_cleanup_fields(tmp_path):
    module = _module()
    task = tmp_path / "task"
    _policy(task)
    terminal = {"exit": 1, "model": "glm-5.3-flash", "child_alive": False,
                "fifo_removed": True, "credentials_persisted": False,
                "cleanup_complete": True}
    assert module.resume_gate(task, terminal, {})["status"] == "BLOCKED_CONTROLLER_NOT_STRICTLY_TERMINAL"
    assert module.resume_gate(task, terminal, {"cleanup_complete": True, "model_waited": True,
                                               "remaining_direct_children": "[]"})["status"] == "BLOCKED_CONTROLLER_NOT_STRICTLY_TERMINAL"


def test_actual_glm_checkpoint_names_do_not_duplicate_the_four_cases(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "CHECKPOINT.json").write_text(json.dumps({"cases": {
        "92b-mixed": {"status": "failed", "job": "real-failed-job"},
        "same-host-pd-split": {"status": "pending"},
        "multi-host-tp": {"status": "pending"},
        "cross-host-pd-split": {"status": "pending"},
    }}), encoding="utf-8")
    expected = ("mixed", "same_host_pd_split", "multi_host_tp", "cross_host_pd_split")
    bridge_spec = importlib.util.spec_from_file_location("bridge_checkpoint", ROOT / "scripts/ops/registered_fleet_bridge.py")
    bridge = importlib.util.module_from_spec(bridge_spec)
    bridge_spec.loader.exec_module(bridge)
    for read_status in (_module().business_checkpoint_status, bridge.business_checkpoint_summary):
        result = read_status(task, expected)
        assert set(result["cases"]) == set(expected)
        assert result["cases"]["mixed"]["state"] == "failed"
        assert result["cases"]["mixed"]["resubmit_allowed"] is False
