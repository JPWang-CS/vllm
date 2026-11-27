import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ops/recover_framework.py"


def _module():
    spec = importlib.util.spec_from_file_location("recover_framework_guard", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(tmp_path, active=None):
    task = tmp_path / "task"
    repair = task / "repair"
    repair.mkdir(parents=True)
    if active is not None:
        (repair / "ACTIVE_BUSINESS.json").write_text(json.dumps(active), encoding="utf-8")
    (repair / "probe.py").write_text("print('ok')\n", encoding="utf-8")
    return task


EXCLUSIVE_HOST = "198.51.100.42"
OTHER_HOST = "192.0.2.10"


def _submit(module, root, task, *, action="run", host=EXCLUSIVE_HOST):
    return module.fleet_submit(
        root, task, "guard-request", action, host, "guard-job",
        script="repair/probe.py" if action == "run" else None,
        devices=[] if action == "run" else None,
        args=[],
    )


def test_active_exclusive_owner_blocks_only_exclusive_host_run(tmp_path):
    module = _module()
    task = _task(tmp_path, {
        "owner": "native-owner",
        "exclusive_active": True,
        "exclusive_host": EXCLUSIVE_HOST,
        "case_id": "native-a3-quality-09",
    })
    with pytest.raises(RuntimeError, match="exclusive host.*owner=native-owner.*native-a3-quality-09.*ACTIVE_BUSINESS.json"):
        _submit(module, tmp_path, task)
    assert not (task / "fleet/requests/guard-request.json").exists()


def test_active_exclusive_owner_allows_status_and_other_host(tmp_path, published_source_fixture):
    module = _module()
    task = _task(tmp_path, {
        "owner": "native-owner",
        "exclusive_active": True,
        "exclusive_host": EXCLUSIVE_HOST,
        "job": "native-a3-quality-09",
    })
    status = _submit(module, tmp_path, task, action="status")
    assert status["status"] == "submitted_to_broker_queue"
    task2 = _task(tmp_path / "other", {
        "owner": "native-owner",
        "exclusive_active": True,
        "exclusive_host": EXCLUSIVE_HOST,
        "job": "native-a3-quality-09",
    })
    published_source_fixture(tmp_path / "other", task2)
    other = _submit(module, tmp_path / "other", task2, host=OTHER_HOST)
    assert other["host"] == OTHER_HOST


def test_unconfigured_or_different_owner_preserves_existing_run_behavior(tmp_path, published_source_fixture):
    module = _module()
    task = _task(tmp_path, {"owner": "native-owner", "exclusive_active": False,
                            "exclusive_host": EXCLUSIVE_HOST})
    published_source_fixture(tmp_path, task)
    result = _submit(module, tmp_path, task)
    assert result["status"] == "submitted_to_broker_queue"
