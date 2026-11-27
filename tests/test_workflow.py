import json
import os
import socket
import sys
from pathlib import Path

import pytest

from fullnet.analysis import comparison, numeric_parity, report
from fullnet.config import resolve
from fullnet.evaluation import score, validate_samples
from fullnet.execution import plan, run_experiment
from fullnet.store import RunLock, inside, read_json, rows, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "models/test").mkdir(parents=True)
    (tmp_path / "models/test/model.yaml").write_text("name: test")
    monkeypatch.setenv("FULLNET_PYTHON", sys.executable)
    # Fixtures are read-only sources. All generated output stays below the test project.
    c = resolve(ROOT / "examples/offline/config.yaml", ROOT)
    c["model"] = "test"
    c["service"]["argv"][2] = str(ROOT / "examples/offline/server.py")
    c["evaluation"]["dataset"] = str(ROOT / "examples/offline/dataset.jsonl")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    c["service"]["argv"][-1] = str(port)
    c["service"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    return tmp_path, c


def test_real_http_lifecycle_and_report(project):
    root, c = project
    run = plan(root, c, "baseline")
    result = run_experiment(root, "test", "baseline")
    assert result["quality_status"] == "passed"
    assert result["numerical_parity"] == "not_run"
    assert len(rows(run / "raw/predictions.jsonl")) == 2
    assert read_json(run / "state.json")["phases"]["cleanup"]["service_exit"] is not None
    assert "100.000" in report(run).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already started"):
        run_experiment(root, "test", "baseline")
    with pytest.raises(FileExistsError):
        plan(root, c, "baseline")


def test_short_ipc_is_project_private(project):
    from fullnet.execution import private_environment
    root,c=project
    run=plan(root,c,'ipc-check')
    env=private_environment(run,c,root)
    ipc=Path(env['VLLM_RPC_BASE_PATH'])
    assert ipc.is_relative_to(root/'.i')
    assert env['TMPDIR']==str(ipc)
    assert ipc.is_dir()


def test_measurement_callback_follows_real_evaluation(project):
    root,c=project
    run=plan(root,c,'measurement')
    observed=[]
    def collect(path, config, result):
        assert read_json(path/'results.json')['protocol_pass']
        observed.append(result['quality_status'])
    run_experiment(root,'test','measurement',after_evaluation=collect)
    assert observed==['passed']


@pytest.mark.parametrize("mode", ["probe_failed", "probe_passed", "default", "smoke_failed"])
def test_before_evaluation_gate_order_failure_and_owned_cleanup(project, monkeypatch, mode):
    from unittest.mock import Mock
    from fullnet import execution

    root, c = project
    c.setdefault("measurements", {})["deferred_stages"] = ["profile", "performance"]
    c["service"]["smoke"] = {"enabled": True, "max_tokens": 8}
    run = plan(root, c, "early-gate")
    events = []

    class Process:
        pid = 10000001
        returncode = None
        def poll(self):
            return self.returncode

    process = Process()

    class Ready:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"data": [{"id": c["service"]["served_model"]}]}).encode()

    def smoke(*args):
        events.append("smoke")
        if mode == "smoke_failed":
            raise RuntimeError("smoke failed")

    def probe(path, config):
        assert path == run
        assert read_json(run / "state.json")["phases"]["smoke"]["status"] == "passed"
        assert "evaluation" not in read_json(run / "state.json")["phases"]
        events.append("probe")
        if mode == "probe_failed":
            raise RuntimeError("pd split probe failed: LLM_LINK_FAILED")

    result = {"protocol_pass": True, "quality_status": "not_specified",
              "attempted": 48, "succeeded": 48, "failed": 0,
              "skipped": 0, "timeout": 0, "samples": 48}
    evaluate = Mock(side_effect=lambda *args: events.append("evaluate") or result)

    def stop(owned, timeout):
        assert owned is process
        events.append("cleanup")
        process.returncode = -15

    monkeypatch.setattr(execution, "check_port", lambda c: None)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(execution.urllib.request, "urlopen", lambda *args, **kwargs: Ready())
    monkeypatch.setattr(execution, "smoke_request", smoke)
    monkeypatch.setattr(execution, "evaluate", evaluate)
    monkeypatch.setattr(execution, "stop_owned", stop)
    callback = None if mode == "default" else probe
    if mode in {"probe_failed", "smoke_failed"}:
        with pytest.raises(RuntimeError, match="probe failed" if mode == "probe_failed" else "smoke failed"):
            run_experiment(root, "test", "early-gate", before_evaluation=callback)
        evaluate.assert_not_called()
    else:
        assert run_experiment(root, "test", "early-gate", before_evaluation=callback) == result
        evaluate.assert_called_once()
    phases = read_json(run / "state.json")["phases"]
    assert phases["cleanup"]["service_exit"] == -15
    assert not (run / ".lock").exists()
    if mode == "probe_failed":
        assert events == ["smoke", "probe", "cleanup"]
        assert phases["before_evaluation"]["status"] == "failed"
        assert phases["evaluation"]["status"] == "not_run"
        assert phases["execution"]["status"] == "failed"
    elif mode == "smoke_failed":
        assert events == ["smoke", "cleanup"]
        assert "before_evaluation" not in phases
    else:
        assert events == (["smoke", "evaluate", "cleanup"] if mode == "default"
                          else ["smoke", "probe", "evaluate", "cleanup"])


def test_failed_missing_predictions_count_in_denominator():
    expected = [{"id": "a", "outputs": ["yes"]}, {"id": "b", "outputs": ["yes"]}]
    prediction = [{"id": "a", "sample": expected[0], "response": "yes", "success": True}]
    result = score(expected, prediction, {"scope": "test", "context_limit": 32, "metric": "exact_match", "minimum_score": 40})
    assert result["score"] == 50
    assert result["quality_status"] == "failed"
    assert result["groups"][0]["missing"] == 1


def test_official_metric_receives_failed_samples_as_empty(tmp_path):
    metrics = tmp_path / "metrics.py"
    metrics.write_text("def string_match_all(texts, refs):\n assert texts == ['yes', '']\n return 50.0\n")
    samples = [{"id": "a", "outputs": ["yes"]}, {"id": "b", "outputs": ["yes"]}]
    result = score(samples, [{"id": "a", "sample": samples[0], "success": True, "response": "yes"}],
                   {"metric": "ruler_official", "scope": "subset", "context_limit": 32, "official_metrics": str(metrics)})
    assert result["score"] == 50 and result["quality_status"] == "not_specified"


def test_budget_rejected_without_modifying_samples():
    sample = {"id": "a", "prompt": "keep this", "outputs": ["x"], "actual_chat_tokens": 30, "output_tokens": 3}
    with pytest.raises(ValueError, match="context budget"):
        validate_samples([sample], {"context_limit": 32})
    assert sample["prompt"] == "keep this"


def test_per_sample_context_limit_is_preserved():
    sample = {"id": "4k", "prompt": "keep", "outputs": ["x"],
              "context_limit": 4096, "actual_chat_tokens": 4090, "output_tokens": 30}
    with pytest.raises(ValueError, match="context budget"):
        validate_samples([sample], {"context_limit": 16384})


def test_path_escape_and_symlink(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        inside(tmp_path, tmp_path / "../elsewhere")
    with pytest.raises(ValueError):
        from fullnet.store import experiment
        experiment(tmp_path, "../bad", "run")


def test_existing_port_never_stopped(project):
    root, c = project
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        c["service"]["base_url"] = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        run = plan(root, c, "occupied")
        with pytest.raises(ValueError, match="already in use"):
            run_experiment(root, "test", "occupied")
        assert listener.fileno() >= 0
        assert read_json(run / "state.json")["phases"]["execution"]["status"] == "failed"


def test_changed_sample_rejected():
    with pytest.raises(ValueError, match="differs"):
        score([{"id": "x", "outputs": ["a"]}], [{"id": "x", "sample": {"id": "x", "outputs": ["b"]}}], {})


def test_lock_preserves_other_writer(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(FileExistsError):
            with RunLock(tmp_path):
                pass


def test_cross_hardware_comparison_rejected(project):
    root, c = project
    left = plan(root, c, "a")
    c["platform"] = "a3"
    right = plan(root, c, "b")
    with pytest.raises(ValueError, match="platform"):
        comparison(left, right)


def test_numerical_parity_shape_and_nonfinite(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_json(a, {"logits": {"shape": [2], "values": [1., 2.]}})
    write_json(b, {"logits": {"shape": [2], "values": [1., 2.001]}})
    assert numeric_parity(a, b, .01, 0)["passed"]
    assert not numeric_parity(a, b, 0, 0)["passed"]
    write_json(b, {"logits": {"shape": [3], "values": [1., 2.]}})
    with pytest.raises(ValueError, match="shape"):
        numeric_parity(a, b, 0, 0)
