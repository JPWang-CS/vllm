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
