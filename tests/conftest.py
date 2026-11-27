"""Fixtures for isolated public framework tests."""
import json
import pytest


@pytest.fixture
def published_source_fixture():
    def configure(root, task):
        receipt = root / ".runtime/framework-source.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"schema": 1, "branch": "main",
            "snapshot": str(root / "runs/release/published"),
            "files": ["src/fullnet/execution.py"]}), encoding="utf-8")
        heartbeat = task / "fleet/broker-heartbeat.json"
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.write_text(json.dumps({"framework_source_receipt_schema": 1,
            "broker_source_file": "scripts/ops/registered_fleet_bridge.py",
            "loaded_source_matches_disk": True}), encoding="utf-8")
    return configure
