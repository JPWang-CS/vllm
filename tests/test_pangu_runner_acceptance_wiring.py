"""CPU-only contract wiring tests for the three formal Pangu runners."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/ops/runner_acceptance.py"
SPEC = importlib.util.spec_from_file_location("runner_acceptance_under_test", HELPER_PATH)
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ACCEPTANCE)


PREFLIGHT = {
    "assets": {"adapter": "assets-check"},
    "device_gate": {"adapter": "fleet-evidence", "evidence_path": "${RUN}/device.json"},
    "model_import": {"adapter": "model-import", "python": "python"},
    "cleanup": {"adapter": "fleet-evidence", "evidence_path": "${RUN}/cleanup.json"},
}


def full_config(mode="full"):
    value = {
        "schema_version": 1, "model": "pangu-7b", "platform": "a5", "devices": [0],
        "identity": {"architecture": "PanguEmbeddedForCausalLM"},
        "service": {"smoke": {"enabled": True}},
        "evaluation": {"minimum_score": None},
        "acceptance": {"mode": mode, "expected_samples": 48, "preflight": copy.deepcopy(PREFLIGHT)},
    }
    if mode == "full":
        value["measurements"] = {
            "requests": 32, "profile_workload": {"input": 4096, "output": 128, "concurrency": 1},
            "workloads": [{"input": 128, "output": 128, "concurrency": 1}],
        }
    return value


def test_missing_acceptance_config_fails_before_any_runner_preparation(tmp_path):
    task = tmp_path / "runs" / "case"
    task.mkdir(parents=True)
    with pytest.raises(ValueError, match="missing --acceptance-config"):
        ACCEPTANCE.load_acceptance_config(None, task)


def test_missing_preflight_field_is_not_ignored(tmp_path):
    task = tmp_path / "runs" / "case"
    task.mkdir(parents=True)
    value = full_config()
    del value["acceptance"]["preflight"]["model_import"]
    path = task / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="model_import"):
        ACCEPTANCE.load_acceptance_config(path, task)


def test_acceptance_loader_uses_repository_yaml_loader(tmp_path):
    task = tmp_path / "runs" / "case"
    task.mkdir(parents=True)
    path = task / "config.yaml"
    path.write_text(
        "schema_version: 1\nmodel: pangu-7b\nplatform: a5\ndevices: [0]\n"
        "identity: {architecture: PanguEmbeddedForCausalLM}\n"
        "service: {smoke: {enabled: true}}\n"
        "evaluation: {minimum_score: null}\n"
        "acceptance:\n  mode: quality-only\n  expected_samples: 48\n  preflight:\n"
        "    assets: {adapter: assets-check}\n"
        "    device_gate: {adapter: fleet-evidence, evidence_path: '${RUN}/device.json'}\n"
        "    model_import: {adapter: model-import, python: python}\n"
        "    cleanup: {adapter: fleet-evidence, evidence_path: '${RUN}/cleanup.json'}\n",
        encoding="utf-8")
    acceptance, _ = ACCEPTANCE.load_acceptance_config(path, task, quality_only=True)
    assert acceptance["mode"] == "quality-only"


def test_plan_seam_receives_acceptance_without_device_calls(tmp_path):
    task = tmp_path / "runs" / "case"
    task.mkdir(parents=True)
    path = task / "config.json"
    path.write_text(json.dumps(full_config()), encoding="utf-8")
    acceptance, _ = ACCEPTANCE.load_acceptance_config(path, task)
    calls = []

    def fake_plan(root, config, run_id, *, quality_only=None):
        calls.append((root, config, run_id, quality_only))
        return "cpu-plan-only"

    result = ACCEPTANCE.plan_with_acceptance(
        fake_plan, tmp_path, {"model": "pangu-7b", "platform": "a5", "devices": [0],
                              "identity": {"architecture": "PanguEmbeddedForCausalLM"},
                              "service": {"smoke": {"enabled": True}},
                              "evaluation": {"minimum_score": None},
                              "measurements": full_config()["measurements"]},
        "case", acceptance, quality_only=False)
    assert result == "cpu-plan-only"
    assert calls[0][1]["acceptance"] == acceptance
    assert calls[0][3] is None


def test_result_status_does_not_require_after_evaluation_callback():
    status = {"profile": "not_run", "performance_phase": "not_run"}
    result = {"protocol_pass": True, "quality_status": "not_specified", "score": 12.0,
              "attempted": 48, "succeeded": 48, "failed": 0, "skipped": 0,
              "timeout": 0, "samples": 48}
    ACCEPTANCE.update_result_status(status, result, quality_only=False)
    assert status["protocol_pass"] is True
    assert status["attempted"] == 48
    assert status["quality_status"] == "not_specified"
    assert status["business_acceptance"] == "INCOMPLETE"
    assert status["execution_completed"] is None

    result["execution_completed"] = True
    result["business_acceptance"] = "ACCEPTED"
    result["acceptance_reasons"] = []
    ACCEPTANCE.update_result_status(status, result, quality_only=False)
    assert status["execution_completed"] is True
    assert status["business_acceptance"] == "ACCEPTED"


def test_all_formal_runners_use_acceptance_input_and_plan_seam():
    runners = ("validate_pangu7b.py", "run_pangu7b_matrix.py", "run_pangu92b.py")
    for name in runners:
        source = (ROOT / "scripts/ops" / name).read_text(encoding="utf-8")
        assert "--acceptance-config" in source
        assert "load_acceptance_config" in source
        assert "plan_with_acceptance" in source
        assert source.index("load_acceptance_config(") < source.index("def guard")
    matrix = (ROOT / "scripts/ops/run_pangu7b_matrix.py").read_text(encoding="utf-8")
    pangu92b = (ROOT / "scripts/ops/run_pangu92b.py").read_text(encoding="utf-8")
    assert "profile_workload" in matrix
    assert "profile_workload" in pangu92b
