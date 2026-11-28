"""CPU-only regression tests for the frozen fullnet acceptance contract."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fullnet.acceptance import (
    AcceptanceBlocked,
    _evidence_ok,
    _fresh_stage_status,
    _model_import_probe,
    _read_acceptance,
    freeze_contract,
)
from fullnet.assets import check_local_assets
from fullnet.cli import main as cli_main
from fullnet.execution import plan
from fullnet.store import write_json


def _config(tmp_path, platform="cpu-test", **extra):
    value = {
        "model": "fixture", "platform": platform, "devices": [0],
        "identity": {"weights": str(tmp_path / "weights"), "purpose": "test"},
        "service": {"argv": [sys.executable], "smoke": {"enabled": True}},
        "evaluation": {"minimum_score": None},
    }
    value.update(extra)
    return value


def test_deferred_stages_only_defer_explicit_stage_and_quality_only_defers_both(tmp_path):
    config = _config(tmp_path, measurements={"deferred_stages": ["profile"]})
    contract = freeze_contract(config, 2)
    assert contract["mode"] == "full"
    assert contract["deferred_stages"] == ["profile"]
    assert contract["required_stages"]["profile"] is False
    assert contract["required_stages"]["performance"] is True

    config["acceptance"] = {"mode": "quality-only"}
    contract = freeze_contract(config, 2)
    assert contract["deferred_stages"] == ["performance", "profile"]
    assert contract["required_stages"]["profile"] is False
    assert contract["required_stages"]["performance"] is False


def test_hardware_plan_rejects_missing_smoke_preflight_and_measurements_before_popen(tmp_path):
    config = _config(tmp_path, platform="a5")
    config["service"]["smoke"]["enabled"] = False
    with pytest.raises(ValueError, match="smoke.enabled"):
        freeze_contract(config, 48)


def test_plan_contract_failure_does_not_leave_empty_run_directory(tmp_path):
    root = tmp_path / "project"
    (root / "models" / "fixture").mkdir(parents=True)
    (root / "models" / "fixture" / "model.yaml").write_text("platforms: [a5]\n", encoding="utf-8")
    dataset = root / "dataset.jsonl"
    dataset.write_text(json.dumps({"id": "a", "prompt": "hello", "outputs": ["yes"],
                                  "actual_chat_tokens": 1, "output_tokens": 1}) + "\n", encoding="utf-8")
    config = _config(root, platform="a5", service={"argv": [sys.executable], "smoke": {"enabled": False}},
                     resolved_changes=[], evaluation={"dataset": str(dataset), "metric": "exact_match",
                     "context_limit": 32, "minimum_score": None})
    run = root / "models" / "fixture" / "experiments" / "bad-plan"
    with pytest.raises(ValueError, match="smoke.enabled"):
        plan(root, config, "bad-plan")
    assert not run.exists()
    config["service"]["smoke"] = {"enabled": True}
    with pytest.raises(ValueError, match="preflight"):
        freeze_contract(config, 48)


def test_hardware_plan_requires_full_measurement_contract(tmp_path):
    config = _config(tmp_path, platform="a5", service={"argv": [sys.executable], "smoke": {"enabled": True}},
                     acceptance={"preflight": {
                         "assets": {"adapter": "assets-check"},
                         "device_gate": {"adapter": "fleet-evidence", "evidence_path": "device.json"},
                         "model_import": {"adapter": "model-import", "python": sys.executable}}})
    with pytest.raises(ValueError, match="measurements"):
        freeze_contract(config, 48)


def test_assets_check_reads_frozen_external_weights_without_download_target(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    weights = tmp_path / "shared" / "weights"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({"architectures": ["Fixture"]}), encoding="utf-8")
    (weights / "model.bin").write_bytes(b"fixture-weight")
    (weights / "tokenizer.json").write_text("{}", encoding="utf-8")
    result = check_local_assets(weights)
    assert result["status"] == "files_present"
    assert result["target"] == str(weights.resolve())
    assert not (project / "assets" / "weights").exists()


def test_hardware_partial_defer_only_requires_the_non_deferred_measurement_contract(tmp_path):
    acceptance = {"preflight": {
        "assets": {"adapter": "assets-check"},
        "device_gate": {"adapter": "fleet-evidence", "evidence_path": "device.json"},
        "model_import": {"adapter": "model-import", "python": sys.executable}}}
    profile_deferred = _config(
        tmp_path / "profile-deferred", platform="a5",
        service={"argv": [sys.executable], "smoke": {"enabled": True}},
        acceptance=acceptance,
        measurements={"deferred_stages": ["profile"], "requests": 1,
                      "workloads": [{"input": 1, "output": 1, "concurrency": 1}]})
    contract = freeze_contract(profile_deferred, 48)
    assert contract["required_stages"]["profile"] is False
    assert contract["required_stages"]["performance"] is True

    performance_deferred = _config(
        tmp_path / "performance-deferred", platform="a5",
        service={"argv": [sys.executable], "smoke": {"enabled": True}},
        acceptance=acceptance,
        measurements={"deferred_stages": ["performance"],
                      "profile_workload": {"input": 1, "output": 1, "concurrency": 1}})
    contract = freeze_contract(performance_deferred, 48)
    assert contract["required_stages"]["profile"] is True
    assert contract["required_stages"]["performance"] is False

    missing_remaining = _config(
        tmp_path / "missing-remaining", platform="a5",
        service={"argv": [sys.executable], "smoke": {"enabled": True}},
        acceptance=acceptance,
        measurements={"deferred_stages": ["profile"]})
    with pytest.raises(ValueError, match="measurements.workloads"):
        freeze_contract(missing_remaining, 48)


def test_evidence_binding_rejects_missing_identity_wrong_devices_and_incomplete_mapping(tmp_path):
    contract = {"run": str(tmp_path / "run"), "devices": ["0"],
                "identity": {"weights": "weights"}}
    base = {"status": "passed", "run": contract["run"], "nonce": "n",
            "identity": {"weights": "weights"}, "source": "test",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "devices": ["0"], "healthy": True, "idle": True, "external_processes": [],
            "mapping": [{"physical_id": 1, "runtime_device": 0}]}
    assert _evidence_ok("device_gate", base, contract, nonce="n")[0]
    for bad in (
        {**base, "run": None},
        {**base, "devices": ["1"]},
        {**base, "mapping": []},
        {**base, "external_processes": [123]},
        {**base, "identity": {}},
    ):
        assert _evidence_ok("device_gate", bad, contract, nonce="n")[0] is False


def test_fleet_evidence_rejects_stale_future_and_wrong_source_job(tmp_path):
    contract = {"run": str(tmp_path / "run"), "devices": ["0"],
                "identity": {"weights": "weights"}, "freshness_ttl_seconds": 30}
    now = datetime.now(timezone.utc)
    base = {"status": "passed", "run": contract["run"], "identity": contract["identity"],
            "source": "registered-fleet-device-gate", "devices": ["0"], "healthy": True,
            "idle": True, "external_processes": [], "mapping": [{"physical_id": 1, "runtime_device": 0}],
            "source_job": {"job": "job-1", "exit": 0, "run": contract["run"], "devices": ["0"]}}
    assert _evidence_ok("device_gate", {**base, "observed_at": now.isoformat()}, contract, adapter="fleet-evidence")[0]
    assert not _evidence_ok("device_gate", {**base, "observed_at": (now - timedelta(seconds=31)).isoformat()}, contract, adapter="fleet-evidence")[0]
    assert not _evidence_ok("device_gate", {**base, "observed_at": (now + timedelta(seconds=1)).isoformat()}, contract, adapter="fleet-evidence")[0]
    stale_job = {**base, "observed_at": now.isoformat(), "source_job": {"job": "old", "exit": 0, "run": str(tmp_path / "other")}}
    assert not _evidence_ok("device_gate", stale_job, contract, adapter="fleet-evidence")[0]


def test_preflight_freshness_is_checked_at_gate_not_again_after_long_run(tmp_path):
    from fullnet.store import write_json

    run = tmp_path / "run"
    (run / "validation").mkdir(parents=True)
    now = datetime.now(timezone.utc)
    checked = (now - timedelta(minutes=30)).isoformat()
    contract = {"run": str(run), "devices": ["0"],
                "identity": {"weights": "weights"}, "freshness_ttl_seconds": 30}
    evidence = {"status": "passed", "run": str(run), "nonce": "gate-nonce",
                "identity": {"weights": "weights"}, "source": "preflight-command",
                "observed_at": checked, "devices": ["0"], "healthy": True, "idle": True,
                "external_processes": [], "mapping": [{"physical_id": 4, "runtime_device": 0}]}
    item = {"status": "passed", "adapter": "external-structured", "returncode": 0,
            "started_at": (now - timedelta(minutes=30, seconds=1)).isoformat(),
            "checked_at": checked, "freshness_checked": True,
            "freshness_ttl_seconds": 30, "evidence": evidence}
    write_json(run / "validation" / "preflight-device_gate.json", {
        "stage": "device_gate", "checked_at": checked,
        "freshness_ttl_seconds": 30, "evidence": evidence})
    value = {"contract": contract, "stages": {"device_gate": item}}
    config = _config(tmp_path)
    phases = {}
    # The evidence is 30 minutes old now, but was accepted at the gate and
    # its saved receipt/TTL remain internally consistent.
    assert _fresh_stage_status(run, config, contract, "device_gate", value, phases) == "passed"

    # The same old receipt without a recorded gate decision is rejected.
    item["freshness_checked"] = False
    assert _fresh_stage_status(run, config, contract, "device_gate", value, phases) == "failed"


def _verifiable_run(tmp_path):
    root = tmp_path / "project"
    run = root / "models" / "fixture" / "experiments" / "case"
    (run / "raw").mkdir(parents=True)
    (run / "validation").mkdir()
    now = datetime.now(timezone.utc)
    started = (now - timedelta(seconds=120)).isoformat()
    stopped = (now - timedelta(seconds=60)).isoformat()
    contract = {
        "schema": "fullnet_acceptance_contract", "version": 2,
        "run": str(run.resolve()), "platform": "a5", "hardware_validation": True,
        "devices": ["0"], "identity": {"weights": "weights"}, "python": sys.executable,
        "freshness_ttl_seconds": 300, "expected_samples": 1, "mode": "full",
        "required_stages": {"evaluation": True, "quality": True, "cleanup": True,
                             "assets": False, "device_gate": False, "model_import": False,
                             "startup": False, "smoke": False, "pd_kv": False,
                             "profile": False, "performance": False},
        "preflight": {"cleanup": {"adapter": "fleet-evidence",
                                    "evidence_path": str((run / "validation" / "cleanup.json").resolve())}},
    }
    config = {"model": "fixture", "platform": "a5", "devices": [0],
              "identity": {"weights": "weights"},
              "evaluation": {"metric": "exact_match", "scope": "fixture", "context_limit": 32,
                              "minimum_score": 100}, "acceptance_contract": contract}
    write_json(run / "config.json", config)
    (run / "dataset.jsonl").write_text(json.dumps({"id": "a", "outputs": ["yes"]}) + "\n", encoding="utf-8")
    (run / "raw" / "predictions.jsonl").write_text(json.dumps({
        "id": "a", "sample": {"id": "a", "outputs": ["yes"]},
        "response": "yes", "success": True}) + "\n", encoding="utf-8")
    write_json(run / "state.json", {"phase": "cleanup", "phases": {
        "plan": {"status": "passed"}, "evaluation": {"status": "passed"},
        "cleanup": {"status": "process_group_signalled", "cleanup_started_at": started,
                     "cleanup_stopped_at": stopped, "resource_verification": "unknown"}}})
    write_json(run / "acceptance.json", {"contract": contract, "stages": {
        "evaluation": {"status": "passed"}, "quality": {"status": "passed"}}})
    return root, run, contract, now


def test_verify_cli_unknown_then_valid_cleanup_receipt_and_stale_before_stop(tmp_path, capsys):
    root, run, contract, now = _verifiable_run(tmp_path)
    assert cli_main(["--root", str(root), "verify", "--model", "fixture", "--run", "case"]) == 1
    capsys.readouterr()

    receipt = {"status": "passed", "run": str(run.resolve()),
               "observed_at": (now - timedelta(seconds=30)).isoformat(),
               "source": "registered-fleet-cleanup", "identity": {"weights": "weights"},
               "devices": ["0"], "service_released": True, "device_processes": [], "ports": [],
               "source_job": {"job": "cleanup-1", "exit": 0,
                              "run": str(run.resolve()), "devices": ["0"]}}
    write_json(run / "validation" / "cleanup.json", receipt)
    assert cli_main(["--root", str(root), "verify", "--model", "fixture", "--run", "case"]) == 0
    capsys.readouterr()

    stale = {**receipt, "observed_at": (now - timedelta(seconds=90)).isoformat()}
    write_json(run / "validation" / "cleanup.json", stale)
    assert cli_main(["--root", str(root), "verify", "--model", "fixture", "--run", "case"]) == 1
    assert any("before or at the owned stop time" in reason
               for reason in json.loads((run / "acceptance.json").read_text())["reasons"])


def test_npu_smi_adapter_uses_full_info_name_before_health_proc(monkeypatch, tmp_path):
    import fullnet.acceptance as acceptance

    table = (
        "NPU ID                         Slot ID                        Chip ID                        Chip Phy-ID                    Chip Name\n"
        "0                              7                              0                              2                              Ascend950DT\n"
        "2                              3                              0                              4                              Ascend950DT\n"
    )
    acl = {"hosts": {"host-10": {"host": "host-10",
            "api": "aclrtGetUserDevIdByPhyDevId", "visible_unset": True,
            "source_job": "acl-job-1", "rows": [
                {"physical_id": 2, "rc": 0, "user_device": 1},
                {"physical_id": 4, "rc": 0, "user_device": 0}]}}}
    full_info = "| 2 | Ascend950DT | OK | 0 |\n"
    calls = []

    class Completed:
        def __init__(self, args, code=0, stdout="", stderr=""):
            self.args, self.returncode = args, code
            self.stdout, self.stderr = stdout, stderr

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv == ["npu-smi", "info", "-m"]:
            return Completed(argv, stdout=table)
        if argv == ["acl-probe"]:
            return Completed(argv, stdout=json.dumps(acl))
        if argv == ["npu-smi", "info"]:
            return Completed(argv, stdout=full_info)
        if argv == ["npu-smi", "info", "-t", "health", "-i", "2"]:
            # Deliberately omit name; full ``info`` must supply it.
            return Completed(argv, stdout="Health Status: OK\n")
        if argv == ["npu-smi", "info", "-t", "proc-mem", "-i", "2"]:
            return Completed(argv, stdout="No running processes found in NPU 2\n")
        raise AssertionError(argv)

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)
    contract = {"run": str(tmp_path / "run"), "platform": "a5", "devices": ["0"],
                "identity": {"weights": "weights"}}
    argv, completed = acceptance._npu_smi_probe(
        tmp_path / "run", {}, contract,
        {"tool": "npu-smi", "host": "host-10", "mapping_argv": ["acl-probe"]},
        {}, tmp_path, "nonce")
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence["mapping"][0]["board"] == 2
    assert evidence["mapping"][0]["physical_id"] == 4
    assert evidence["health_records"][0]["board"] == 2
    assert acceptance._evidence_ok("device_gate", evidence, contract, nonce="nonce")[0]
    assert calls.index(["npu-smi", "info", "-m"]) < calls.index(["npu-smi", "info"])
    assert ["npu-smi", "info", "-t", "health", "-i", "2"] in calls


def test_model_import_requires_actual_python_classfile_and_plugin_result(tmp_path):
    contract = {"run": str(tmp_path / "run"), "devices": ["0"],
                "identity": {"weights": str(tmp_path / "weights")}, "python": sys.executable}
    run = Path(contract["run"])
    run.mkdir()
    weights = Path(contract["identity"]["weights"])
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"architectures": ["FixtureArch"]}), encoding="utf-8")
    package = tmp_path / "fakepkgs"
    (package / "vllm" / "plugins").mkdir(parents=True)
    (package / "vllm" / "model_executor" / "models").mkdir(parents=True)
    (package / "vllm" / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "vllm" / "plugins" / "__init__.py").write_text(
        "import os, sys\n"
        "def load_general_plugins():\n"
        "    i = sys.argv.index('--model')\n"
        "    assert sys.argv[i + 1] == os.environ['EXPECTED_MODEL']\n"
        "    return ['fixture-plugin']\n", encoding="utf-8")
    registry = package / "vllm" / "model_executor" / "models" / "registry.py"
    registry.write_text(
        "class Fixture:\n    pass\n"
        "class ModelRegistry:\n"
        "    @staticmethod\n"
        "    def resolve_model_cls(architectures):\n"
        "        return (Fixture, architectures[0])\n", encoding="utf-8")
    (package / "vllm" / "model_executor" / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "vllm" / "model_executor" / "models" / "__init__.py").write_text("\n", encoding="utf-8")
    contract["service_argv"] = [sys.executable, "-m", "fixture.server", "--model",
                                 contract["identity"]["weights"], "--node-manager-port", "1234"]
    env = {"PYTHONPATH": str(package), "EXPECTED_MODEL": contract["identity"]["weights"]}
    argv, completed = _model_import_probe(run, {}, contract, {"python": sys.executable}, env, tmp_path, "tuple-nonce")
    evidence = json.loads(completed.stdout)
    assert completed.returncode == 0, (argv, completed.stdout, completed.stderr)
    assert evidence["model_class"] == "Fixture"
    assert evidence["resolved_architecture"] == "FixtureArch"
    assert evidence["model_context_argv"][evidence["model_context_argv"].index("--model") + 1] == contract["identity"]["weights"]
    assert _evidence_ok("model_import", evidence, contract, nonce="tuple-nonce")[0]

    registry.write_text(
        "class Fixture:\n    pass\n"
        "class LazyRegisteredModel:\n"
        "    def load_model_cls(self):\n        return Fixture\n"
        "class ModelRegistry:\n"
        "    @staticmethod\n"
        "    def resolve_model_cls(architectures):\n        return LazyRegisteredModel()\n", encoding="utf-8")
    _, completed = _model_import_probe(run, {}, contract, {"python": sys.executable}, env, tmp_path, "lazy-nonce")
    evidence = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert evidence["model_class"] == "Fixture"


def test_contract_tampering_is_rejected_and_old_runs_stay_read_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    contract = freeze_contract(_config(tmp_path), 2)
    contract["run"] = str(run)
    write_json(run / "config.json", {"acceptance_contract": contract})
    write_json(run / "acceptance.json", {"contract": {**contract, "required_stages": {}}, "stages": {}})
    with pytest.raises(AcceptanceBlocked, match="differs from frozen plan"):
        _read_acceptance(run)


def test_fresh_stage_status_does_not_trust_passed_labels_or_old_phase_fallback(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = _config(tmp_path)
    contract = freeze_contract(config, 2)
    contract["run"] = str(run)
    value = {"contract": contract, "stages": {
        "device_gate": {"status": "passed", "returncode": 0,
                         "evidence": {"status": "passed"}, "adapter": "external-structured"}},
    }
    phases = {"smoke": {"status": "passed"}, "profile": {"status": "passed"},
              "performance": {"status": "passed"}}
    assert _fresh_stage_status(run, config, contract, "device_gate", value, phases) == "failed"
    assert _fresh_stage_status(run, config, contract, "smoke", value, phases) == "missing"
    assert _fresh_stage_status(run, config, contract, "profile", value, phases) == "missing"
    assert _fresh_stage_status(run, config, contract, "performance", value, phases) == "missing"


def test_cleanup_signal_failure_wins_over_a_passed_verification_label(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = _config(tmp_path, platform="a5")
    contract = {"run": str(run), "devices": ["0"], "identity": config["identity"]}
    evidence = {"status": "passed", "run": str(run), "identity": config["identity"],
                "devices": ["0"], "service_released": True, "device_processes": [], "ports": []}
    value = {"contract": contract, "stages": {"cleanup": {"status": "passed", "evidence": evidence}}}
    phases = {"cleanup": {"status": "failed", "resource_verification": "passed"}}
    assert _fresh_stage_status(run, config, contract, "cleanup", value, phases) == "failed"
