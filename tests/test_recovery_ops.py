import importlib.util
import json
from pathlib import Path

import pytest
import subprocess
import sys


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts/ops" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_missing_file_never_creates_local_placeholder(tmp_path):
    module = load("artifact_transfer")

    class Missing:
        def stat(self, source):
            raise FileNotFoundError(source)

    target = tmp_path / "report.md"
    assert not module.fetch_if_present(Missing(), "/missing", target)
    assert list(tmp_path.iterdir()) == []


def test_failed_download_preserves_previous_complete_result(tmp_path):
    module = load("artifact_transfer")

    class Failing:
        def stat(self, source):
            return object()

        def get(self, source, destination):
            Path(destination).write_text("partial")
            raise PermissionError(source)

    target = tmp_path / "report.md"
    target.write_text("complete")
    assert not module.fetch_if_present(Failing(), "/remote", target)
    assert target.read_text() == "complete"
    assert not (tmp_path / "report.md.download").exists()


def test_recovery_rejects_arbitrary_plan_arguments_before_device_use(tmp_path, monkeypatch):
    module = load("recover_framework")
    monkeypatch.setattr(module, "snapshot", lambda root, task: {"cases": []})
    with pytest.raises(ValueError, match="unexpected plan fields"):
        module.start(tmp_path, tmp_path, {"argv": ["unauthorized"]})


def test_recovery_refuses_to_overlap_running_model_job(tmp_path, monkeypatch):
    module = load("recover_framework")
    monkeypatch.setattr(module, "snapshot", lambda root, task: {"cases": [{"job-state.json": {"status": "running"}}]})
    with pytest.raises(ValueError, match="running"):
        module.start(tmp_path, tmp_path, {"tp": 2, "dp": 1})


def test_task_policy_counts_failures_per_normalized_plan(tmp_path, monkeypatch):
    module = load("recover_framework")
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    (task / "repair" / "EXECUTION_POLICY.json").write_text(json.dumps({
        "max_total_jobs": 24, "max_failures_per_plan": 3
    }))
    plans = [
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "pd_mode": None},
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "pd_mode": None},
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "pd_mode": None},
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "pd_mode": "mixed"},
    ]
    ledger = {"cases": [{"task": f"failed-{i}", "plan": plan} for i, plan in enumerate(plans)]}
    (task / "ledger.json").write_text(json.dumps(ledger))
    monkeypatch.setattr(module, "snapshot", lambda root, current_task: {"cases": [
        {"task": f"failed-{i}", "job-state.json": {"status": "failed"},
         "validation.json": {"detail": "real communication failure"}}
        for i in range(4)
    ]})
    monkeypatch.setattr(module, "inventory", lambda root, current_task: {"available": []})
    with pytest.raises(RuntimeError, match="insufficient healthy empty devices"):
        module.start(tmp_path, task, {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0,
                                      "pd_mode": "split", "reason": "test different plan"})


def test_task_policy_blocks_same_plan_at_three_failures(tmp_path, monkeypatch):
    module = load("recover_framework")
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    (task / "repair" / "EXECUTION_POLICY.json").write_text(json.dumps({
        "max_total_jobs": 24, "max_failures_per_plan": 3
    }))
    plan = {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "pd_mode": None}
    (task / "ledger.json").write_text(json.dumps({"cases": [
        {"task": f"failed-{i}", "plan": plan} for i in range(3)
    ]}))
    monkeypatch.setattr(module, "snapshot", lambda root, current_task: {"cases": [
        {"task": f"failed-{i}", "job-state.json": {"status": "failed"},
         "validation.json": {"detail": "real communication failure"}}
        for i in range(3)
    ]})
    with pytest.raises(ValueError, match="plan failure budget exhausted"):
        module.start(tmp_path, task, {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0,
                                      "reason": "test same plan"})


def test_legacy_task_without_policy_keeps_global_three_failure_budget(tmp_path, monkeypatch):
    module = load("recover_framework")
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    plans = [
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0},
        {"tp": 4, "dp": 1, "graph": "eager", "mtp": 0},
        {"tp": 2, "dp": 1, "graph": "eager", "mtp": 3},
    ]
    (task / "ledger.json").write_text(json.dumps({"cases": [
        {"task": f"failed-{i}", "plan": plan} for i, plan in enumerate(plans)
    ]}))
    monkeypatch.setattr(module, "snapshot", lambda root, current_task: {"cases": [
        {"task": f"failed-{i}", "job-state.json": {"status": "failed"},
         "validation.json": {"detail": "real communication failure"}}
        for i in range(3)
    ]})
    with pytest.raises(ValueError, match="recovery budget exhausted"):
        module.start(tmp_path, task, {"tp": 1, "dp": 4, "graph": "eager", "mtp": 0,
                                      "reason": "legacy compatibility"})


@pytest.mark.parametrize(
    ("plan_quality_only", "policy_quality_only", "expected"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_recovery_start_emits_quality_only_argv_without_touching_device(tmp_path, monkeypatch,
                                                                          plan_quality_only,
                                                                          policy_quality_only,
                                                                          expected):
    module = load("recover_framework")
    root = tmp_path / "project"
    task = root / "task"
    (root / "scripts/ops").mkdir(parents=True)
    (root / "runs").mkdir()
    (task / "repair").mkdir(parents=True)
    (root / "scripts/ops/framework-recovery.local.json").write_text(json.dumps({
        "python": "/opt/python", "reference": "/ref.json", "weights": "/weights",
        "dataset": "/dataset", "tmux": "/tmux", "cann_source": "/cann.sh",
        "container": "container",
    }))
    if policy_quality_only:
        (task / "repair/EXECUTION_POLICY.json").write_text(json.dumps({
            "max_total_jobs": 20, "max_failures_per_plan": 3, "quality_only": True,
        }))
    monkeypatch.setattr(module, "snapshot", lambda root, current_task: {"cases": []})
    monkeypatch.setattr(module, "inventory", lambda root, current_task: {"available": [0, 1]})
    monkeypatch.setattr(module, "container", lambda *args, **kwargs: type(
        "Completed", (), {"returncode": 0, "stdout": "0 0 0 0 0\n1 1 1 1 1\n", "stderr": ""}
    )())
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: type(
        "Completed", (), {"returncode": 0, "stdout": "submitted", "stderr": ""}
    )())
    plan = {"tp": 2, "dp": 1, "graph": "eager", "mtp": 0, "reason": "test",
            "quality_only": plan_quality_only}
    module.start(root, task, plan)
    request = json.loads(next((root / "runs").glob("*/requested-job.json")).read_text())
    argv = request["argv"]
    command = " ".join(argv)
    assert ("--quality-only" in command) is expected


def test_write_scope_allows_orchestration_and_rejects_original_kernel(tmp_path):
    guard = Path(__file__).resolve().parents[1] / 'scripts/ops/task_write_guard.py'
    root = tmp_path / 'project'
    source = root / 'src/fullnet'
    source.mkdir(parents=True)
    script = root / 'scripts/ops/run_pangu92b.py'
    script.parent.mkdir(parents=True)
    script.write_text('# allowed framework runner')
    def decide(path):
        result = subprocess.run([sys.executable, '-B', str(guard), str(root), str(source), str(script)],
            input=json.dumps({'tool_input': {'file_path': str(path)}}), text=True, capture_output=True, check=True)
        return json.loads(result.stdout)['hookSpecificOutput']['permissionDecision']
    assert decide(source / 'execution.py') == 'allow'
    assert decide(script) == 'allow'
    assert decide(root / 'kernels/original.asc') == 'deny'
    assert decide(root / 'scripts/ops/task_write_guard.py') == 'deny'
    assert decide(tmp_path / 'shared/cann/file.py') == 'deny'
