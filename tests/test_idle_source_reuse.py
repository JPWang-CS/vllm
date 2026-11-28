"""Task-private semantic tests for the idle source reuse candidate."""

import importlib.util
import io
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "ops" / "registered_fleet_bridge.py"
SPEC = importlib.util.spec_from_file_location(
    "idle_reuse_bridge", SOURCE
)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


class MemorySFTP:
    def __init__(self, files):
        self.files = dict(files)
        self.writes = 0

    def lstat(self, path):
        if path in self.files:
            return SimpleNamespace(st_mode=stat.S_IFREG)
        if path in {"/source", "/source/scripts", "/target", "/target/scripts"}:
            return SimpleNamespace(st_mode=stat.S_IFDIR)
        raise FileNotFoundError(path)

    def open(self, path, mode):
        assert mode == "rb"
        return io.BytesIO(self.files[path])


def test_identical_idle_target_is_reused_readonly():
    source = MemorySFTP({"/source/scripts/probe.py": b"same"})
    target = MemorySFTP({"/target/scripts/probe.py": b"same"})

    comparison = BRIDGE.compare_active_target_source(
        source, "/source", target, "/target", ["scripts/probe.py"]
    )
    reused, mode = BRIDGE.target_source_reuse_decision(comparison, target_busy=False)

    assert reused["matches"] is True
    assert mode == "reuse_identical_idle"
    assert target.writes == 0
    assert all(record["write_performed"] is False for record in reused["files"])


def test_identical_active_target_keeps_original_active_reuse_mode():
    comparison = {"matches": True, "files": [], "mismatches": [], "files_checked": 0}

    reused, mode = BRIDGE.target_source_reuse_decision(comparison, target_busy=True)

    assert reused is comparison
    assert mode == "reuse_identical_active"


def test_changed_idle_target_falls_through_to_original_sync():
    comparison = {"matches": False, "files": [],
                  "mismatches": [{"path": "scripts/probe.py", "reason": "different content"}],
                  "files_checked": 1}

    reused, mode = BRIDGE.target_source_reuse_decision(comparison, target_busy=False)

    assert reused is None
    assert mode is None


def test_changed_active_target_still_waits_before_job_creation():
    comparison = {"matches": False, "files": [],
                  "mismatches": [{"path": "scripts/probe.py", "reason": "missing"}],
                  "files_checked": 1}

    with pytest.raises(BRIDGE.TargetProjectBusy, match="published source update is pending"):
        BRIDGE.target_source_reuse_decision(comparison, target_busy=True, busy_detail="target busy")


def test_public_bridge_has_distinct_idle_and_active_framework_sync_modes():
    source = SOURCE.read_text(encoding="utf-8")

    assert "reuse_identical_idle" in source
    assert "reuse_identical_active" in source
    assert "if value[\"host\"] != self.control_host and reused_source is None:" in source
