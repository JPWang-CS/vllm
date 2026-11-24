import ast
import json
import os
import socket
import sys
from pathlib import Path

from fullnet.execution import plan, profiler_deferred, run_experiment
from fullnet.store import read_json


ROOT = Path(os.environ.get("FULLNET_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def test_quality_only_defers_profiler_setup_only_when_requested():
    assert profiler_deferred({"measurements": {"deferred_stages": ["profile", "performance"]}})
    assert not profiler_deferred({"measurements": {"deferred_stages": []}})
    assert not profiler_deferred({})


def test_quality_only_runner_contracts_keep_pd_and_defer_measurements():
    p92 = (ROOT / "scripts/ops/run_pangu92b.py").read_text(encoding="utf-8")
    p7 = (ROOT / "scripts/ops/run_pangu7b_matrix.py").read_text(encoding="utf-8")
    ast.parse(p92)
    ast.parse(p7)
    assert "--quality-only" in p92 and "--quality-only" in p7
    assert "deferred_stages" in p92 and "deferred_stages" in p7
    assert "if a.pd_mode=='mixed':" in p92
    assert "if a.pd_mode=='split':" in p92
    assert "status['profile']='deferred_by_user'" in p92
    assert 'status["profile"] = "deferred_by_user"' in p7
    assert "if not a.quality_only:" in p92
    assert "if a.quality_only:" in p7


def test_quality_runner_enables_real_smoke_contract():
    source = (ROOT / "scripts/ops/run_pangu7b_matrix.py").read_text(encoding="utf-8")
    assert '"smoke": {"enabled": True' in source
    assert '"max_tokens": 8' in source
    assert '"timeout": 60' in source


def _fixture_config(tmp_path, deferred, run_id):
    root = tmp_path / "project"
    root.mkdir()
    server = ROOT / "examples/offline/server.py"
    dataset = root / "dataset.jsonl"
    rows = [
        {"id": "fixture/0", "prompt": "hello", "outputs": ["hello"], "actual_chat_tokens": 1, "output_tokens": 8},
        {"id": "fixture/1", "prompt": "world", "outputs": ["world"], "actual_chat_tokens": 1, "output_tokens": 8},
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = {
        "schema_version": 1,
        "model": "workflow-fixture",
        "platform": "cpu-test",
        "identity": {"purpose": "quality-only fixture", "inference": False},
        "changes": [],
        "resolved_changes": [],
        "service": {"argv": [sys.executable, "-B", str(server), "--port", str(port)],
                     "base_url": f"http://127.0.0.1:{port}/v1", "served_model": "fixture",
                     "ready_timeout": 10, "request_timeout": 5, "shutdown_timeout": 2,
                     "smoke": {"enabled": True, "prompt": "smoke", "max_tokens": 8, "timeout": 5}},
        "evaluation": {"dataset": str(dataset), "scope": "fixed fixture", "metric": "exact_match",
                        "context_limit": 32, "minimum_score": None, "seed": 42,
                        "temperature": 0, "top_p": 1, "chat_template_kwargs": {}},
        "measurements": {"requests": 1, "warmups": 0, "deferred_stages": ["profile", "performance"] if deferred else []},
    }
    return root, config, run_id


def test_quality_only_fixture_skips_profiler_but_runs_fixed_counter(tmp_path, monkeypatch):
    root, config, run_id = _fixture_config(tmp_path, True, "deferred")
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "inherited-profiler-path")

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare_profiler_tools must not run in quality-only")

    monkeypatch.setattr("fullnet.measurements.prepare_profiler_tools", forbidden)
    run = plan(root, config, run_id)
    result = run_experiment(root, "workflow-fixture", run_id)
    state = read_json(run / "state.json")
    assert result["samples"] == result["attempted"] == result["succeeded"] == 2
    assert result["failed"] == result["skipped"] == result["timeout"] == 0
    assert result["quality_status"] == "not_specified"
    assert state["phases"]["profile"]["status"] == "deferred_by_user"
    assert state["phases"]["performance"]["status"] == "deferred_by_user"
    assert state["phases"]["cleanup"]["status"] == "process_group_signalled"
    assert read_json(run / "performance/profiler-tools.json")["acceptance"] == "deferred_by_user"


def test_default_fixture_still_prepares_profiler(tmp_path, monkeypatch):
    root, config, run_id = _fixture_config(tmp_path, False, "default")
    calls = []
    monkeypatch.setattr("fullnet.measurements.prepare_profiler_tools",
                        lambda run, c: calls.append(str(run)) or {"env_updates": {}})
    plan(root, config, run_id)
    result = run_experiment(root, "workflow-fixture", run_id)
    assert result["attempted"] == 2
    assert calls


def test_quality_only_finalization_failure_is_not_success():
    namespace = {}
    source = (ROOT / "scripts/ops/run_pangu7b_matrix.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "quality_phase_complete")
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, "run_pangu7b_matrix.py", "exec"), namespace)
    complete = namespace["quality_phase_complete"]
    base = {"protocol_pass": True, "device_empty_after_exit": True,
            "quality_status": "not_specified", "error": None,
            "finalization_error": None}
    assert complete(base, True)
    failed_export = {**base, "error": "RuntimeError"}
    failed_pd = {**base, "finalization_error": "PD export failed"}
    assert not complete(failed_export, True)
    assert not complete(failed_pd, True)
