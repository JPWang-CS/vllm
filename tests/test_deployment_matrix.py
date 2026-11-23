import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.ops import recover_framework


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ops/deployment_matrix.py"
SPEC = importlib.util.spec_from_file_location("deployment_matrix_under_test", SCRIPT)
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_matrix_enumerates_platforms_and_blocks_unsupported_modes(tmp_path):
    data = MATRIX.build_matrix("deployment-matrix-cpu")
    assert {case["platform"] for case in data["cases"]} == {"a3", "a5"}
    assert sum(case["scenario"] == "single_host_tpdp" and case["platform"] == "a5" and case["status"] == "planned"
               for case in data["cases"]) == 16
    assert all(case["status"] == "blocked" for case in data["cases"] if case["platform"] == "a3")
    assert {"pangu7b_single_card", "multi_host_tp", "multi_host_dp", "pd_mixed_single_instance",
            "same_host_pd_split", "pd_disaggregated"}.issubset(
        {case["scenario"] for case in data["cases"]})
    assert all(case["status"] == "blocked" for case in data["cases"]
               if case["scenario"] != "single_host_tpdp" and
               not (case["scenario"] == "pangu7b_single_card" and case["platform"] == "a5"))
    assert next(case for case in data["cases"]
                if case["scenario"] == "pangu7b_single_card" and case["platform"] == "a5")["status"] == "planned"
    assert all(case["launch_contract"]["fixed_evaluation_samples"] == 48 for case in data["cases"])
    assert all(case["launch_contract"]["requests"] == 32 and case["launch_contract"]["warmups"] == 2
               for case in data["cases"])
    assert not any(case["status"] == "passed" for case in data["cases"])


def test_plan_is_not_acceptance_and_existing_plan_is_never_overwritten(tmp_path):
    root = tmp_path.resolve()
    data = MATRIX.create_matrix(root, "deployment-matrix-plan")
    assert data["status"] == "incomplete"
    assert not any(case["status"] == "passed" for case in data["cases"])
    with pytest.raises(FileExistsError, match="already exists"):
        MATRIX.create_matrix(root, "deployment-matrix-plan")
    with pytest.raises(ValueError, match="blocked cases remain blocked"):
        MATRIX.submit_case(root, "deployment-matrix-plan", "a3-single-tp2dp1-eager-mtp0")


def _valid_evidence():
    case = {"model": "pangu-92b", "platform": "a5", "parallel": {"tp": 2, "dp": 1, "pp": 1}, "graph": "eager", "mtp": 0}
    job = {"status": "passed", "exit": 0}
    validation = {"model_requests": "completed", "attempted": 48, "succeeded": 48, "failed": 0,
                  "skipped": 0, "timeout": 0, "protocol_pass": True, "device_empty_after_exit": True}
    exp_state = {"phases": {"evaluation": {"status": "passed"}, "performance": {"status": "passed"},
                             "profile": {"status": "passed"}}}
    config = {"model": "pangu-92b", "platform": "a5", "identity": {"tp": 2, "dp": 1, "pp": 1, "graph": "eager", "mtp": 0},
              "measurements": {"requests": 32, "warmups": 2, "workloads": MATRIX.PERFORMANCE_WORKLOADS}}
    performance = [{"completed": 32} for _ in range(6)]
    profile = {"acceptance": "accepted", "has_device_kernel_evidence": True}
    return case, job, validation, exp_state, config, performance, profile


def test_acceptance_requires_real_48_32x2_and_device_profile_evidence():
    evidence = _valid_evidence()
    assert MATRIX.acceptance_reasons(*evidence) == []
    case, job, validation, exp_state, config, performance, profile = evidence
    validation["attempted"] = 47
    profile["has_device_kernel_evidence"] = False
    reasons = MATRIX.acceptance_reasons(case, job, validation, exp_state, config, performance, profile)
    assert any("48-sample" in reason for reason in reasons)
    assert any("device kernel" in reason for reason in reasons)


def test_fleet_inventory_and_asset_requests_match_broker_schema(tmp_path):
    root = tmp_path.resolve()
    task = root / "runs" / "matrix-task"
    task.mkdir(parents=True)
    request = recover_framework.fleet_inventory_request(root, task, "inventory-00", "10.1.2.3", "pd-case")
    request_path = Path(request["request"])
    value = json.loads(request_path.read_text())
    assert set(value) == {"request_id", "action", "host", "job", "script", "devices", "args"}
    assert value["action"] == "inventory" and value["devices"] == []

    asset = recover_framework.fleet_submit(root, task, "asset-00", "copy_asset", "10.1.2.3", "asset-job",
                                           devices=[], args=[], asset="pangu-92b")
    asset_request = json.loads(Path(asset["request"]).read_text())
    assert asset_request["action"] == "copy_asset" and asset_request["asset"] == "pangu-92b"
    assert set(asset_request) == {"request_id", "action", "host", "job", "script", "devices", "args", "asset"}


def test_fleet_run_script_is_confined_to_task_repair_and_status_reads_broker_reply(tmp_path):
    root = tmp_path.resolve()
    task = root / "runs" / "matrix-task"
    repair = task / "repair"
    repair.mkdir(parents=True)
    (repair / "worker.py").write_text("print('ok')\n")
    submitted = recover_framework.fleet_submit(root, task, "run-00", "run", "10.1.2.3", "pd-job",
                                               "repair/worker.py", [2, 3], ["--rank", "0"])
    request = json.loads(Path(submitted["request"]).read_text())
    assert request["script"] == "repair/worker.py" and request["devices"] == [2, 3]
    with pytest.raises(ValueError, match="relative"):
        recover_framework.fleet_submit(root, task, "run-01", "run", "10.1.2.3", "pd-job2",
                                       "repair/../outside.py", [2], [])
    response_path = task / "fleet" / "responses" / "run-00.json"
    response_path.parent.mkdir(parents=True)
    response_path.write_text(json.dumps({"submitted": True, "acceptance": "submitted_not_validated"}))
    result = recover_framework.fleet_status(root, task, "run-00")
    assert result["status"] == "received"
    assert result["response"]["acceptance"] == "submitted_not_validated"


def test_matrix_asset_copy_uses_registered_asset_allowlist(tmp_path):
    root = tmp_path.resolve()
    matrix = MATRIX.create_matrix(root, "deployment-matrix-assets")
    pd_case = next(c for c in matrix["cases"] if c["case_id"] == "a5-pd-disaggregated")
    request = MATRIX.submit_fleet_asset(root, matrix["task"], pd_case["case_id"], "10.1.2.3", "pangu-92b")
    payload = json.loads(Path(request["request"]).read_text())
    assert payload["action"] == "copy_asset" and payload["asset"] == "pangu-92b"
    with pytest.raises(ValueError, match="registered model assets"):
        recover_framework.fleet_submit(root, root / "runs" / matrix["task"], "asset-bad", "copy_asset",
                                       "10.1.2.3", "job", asset="other-model")


def test_missing_runtime_or_weight_evidence_is_blocked_not_passed():
    reasons = MATRIX.dependency_reasons({"python": True, "weights": False, "cann_source": False})
    assert reasons == ["weights unavailable", "cann_source unavailable"]
