"""Task-private tests for the batched source capture candidate."""

import base64
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "ops" / "registered_fleet_bridge.py"
SPEC = importlib.util.spec_from_file_location("batch_source_bridge", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


def record(mode, data=b""):
    value = {"status": "present", "mode": mode, "size": len(data)}
    if stat.S_ISREG(mode):
        value["data"] = base64.b64encode(data).decode("ascii")
    return value


def payload(records):
    return {"schema": "fullnet.source-capture.v1", "records": records}


def test_capture_view_serves_captured_bytes_readonly():
    view = BRIDGE.CapturedRemote(payload({
        "/project": record(stat.S_IFDIR),
        "/project/scripts": record(stat.S_IFDIR),
        "/project/scripts/a.py": record(stat.S_IFREG | 0o644, b"abc"),
    }))

    assert view.lstat("/project/scripts/a.py").st_size == 3
    with view.open("/project/scripts/a.py", "rb") as stream:
        assert stream.read() == b"abc"
    with pytest.raises(ValueError, match="read-only"):
        view.open("/project/scripts/a.py", "wb")


def test_capture_limits_and_missing_or_symlink_records_fail_closed():
    oversized = b"x" * (BRIDGE.CAPTURE_MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="file limit"):
        BRIDGE.CapturedRemote(payload({"/project/a.py": record(stat.S_IFREG | 0o644, oversized)}))

    missing = BRIDGE.CapturedRemote(payload({"/project/a.py": {"status": "missing"}}))
    with pytest.raises(FileNotFoundError):
        missing.lstat("/project/a.py")

    linked = BRIDGE.CapturedRemote(payload({"/project": record(stat.S_IFDIR),
                                            "/project/a.py": record(stat.S_IFLNK)}))
    with pytest.raises(ValueError, match="source root|source contains"):
        BRIDGE.remote_regular_source(linked, "/project", "a.py")

    with pytest.raises(ValueError, match="invalid source capture bytes"):
        BRIDGE.CapturedRemote(payload({"/project/a.py": {
            "status": "present", "mode": stat.S_IFREG | 0o644,
            "size": 1, "data": "!not-base64!"}}))


def test_capture_script_real_subprocess_returns_bytes_and_lexical_parent_attrs(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    file_path = root / "scripts" / "a.py"
    file_path.write_bytes(b"captured")
    manifest = json.dumps([{"root": str(root), "files": ["scripts/a.py"]}])
    result = subprocess.run(
        [sys.executable, "-B", "-c", BRIDGE.CAPTURE_SCRIPT, manifest],
        capture_output=True, text=True, check=True,
    )
    view = BRIDGE.CapturedRemote(json.loads(result.stdout))
    assert view.lstat(str(root)).st_mode & stat.S_IFMT(stat.S_IFDIR)
    with view.open(str(file_path), "rb") as stream:
        assert stream.read() == b"captured"


def test_real_control_capture_includes_snapshot_ancestors_and_compare_passes(tmp_path):
    project = tmp_path / "project"
    snapshot = project / "runs" / "published"
    (project / "scripts").mkdir(parents=True)
    (snapshot / "scripts").mkdir(parents=True)
    (project / "scripts" / "a.py").write_bytes(b"same")
    (snapshot / "scripts" / "a.py").write_bytes(b"same")
    project_text = str(project).replace("\\", "/")
    snapshot_text = str(snapshot).replace("\\", "/")
    posix_manifest = BRIDGE.capture_manifest([
        {"root": "/project", "files": ["scripts/a.py"]},
        {"root": "/project/runs/published", "files": ["scripts/a.py"]},
    ])
    assert posix_manifest[0]["parents"][-2:] == ["runs", "runs/published"]
    manifest = [
        {"root": project_text, "files": ["scripts/a.py"], "parents": ["runs", "runs/published"]},
        {"root": snapshot_text, "files": ["scripts/a.py"], "parents": []},
    ]
    result = subprocess.run(
        [sys.executable, "-B", "-c", BRIDGE.CAPTURE_SCRIPT, json.dumps(manifest)],
        capture_output=True, text=True, check=True,
    )
    captured = json.loads(result.stdout)
    remapped = {}
    for path, value in captured["records"].items():
        remapped[path.replace("\\", "/").replace(project_text, "/project")] = value
    view = BRIDGE.CapturedRemote({"schema": captured["schema"], "records": remapped})
    comparison = BRIDGE.compare_control_to_snapshot(
        view, "/project", {"snapshot": "/project/runs/published", "files": ["scripts/a.py"]}
    )
    assert comparison["matches"] is True
    assert comparison["files_checked"] == 1


def test_capture_manifest_rejects_unsafe_or_unallowlisted_paths():
    with pytest.raises(ValueError):
        BRIDGE.capture_manifest([{"root": "/project", "files": ["../secret.py"]}])
    with pytest.raises(ValueError):
        BRIDGE.capture_manifest([{"root": "/project", "files": ["scripts/unknown.bin"]}])


def test_control_and_target_capture_each_use_one_rpc_and_no_sftp(tmp_path):
    class Client:
        def exec_command(self, _command):
            raise AssertionError("helper.call is the capture RPC surface")

    class Helper:
        def __init__(self):
            self.calls = []

        def call(self, _client, command, body=None, timeout=0):
            self.calls.append((command, body, timeout))
            return 0, json.dumps(payload({"/project": record(stat.S_IFDIR)})), ""

    broker = object.__new__(BRIDGE.Broker)
    broker.helper = Helper()
    control = Client()
    first = broker._capture_remote_view(control, [{"root": "/project", "files": ["scripts/a.py"]},
                                          {"root": "/project/runs/published", "files": ["scripts/a.py"]}])
    second = broker._capture_remote_view(control, [{"root": "/target", "files": ["scripts/a.py"]}])

    assert isinstance(first, BRIDGE.CapturedRemote)
    assert isinstance(second, BRIDGE.CapturedRemote)
    assert len(broker.helper.calls) == 2
    assert all("source-capture" not in call[0] or "python3" in call[0] for call in broker.helper.calls)
    assert all(call[2] == 120 for call in broker.helper.calls)


def test_plain_sftp_adapter_keeps_existing_path():
    broker = object.__new__(BRIDGE.Broker)

    class PlainClient:
        pass

    assert broker._capture_remote_view(PlainClient(), [{"root": "/project", "files": []}]) is None


def test_public_bridge_integrates_capture_before_existing_compare_paths():
    source = SOURCE.read_text(encoding="utf-8")
    compile(BRIDGE.CAPTURE_SCRIPT, "<capture-script>", "exec")
    assert "verified_source_view = self._capture_remote_view" in source
    assert "target_view = self._capture_remote_view" in source
    assert "compare_control_to_snapshot(verified_source_view or source" in source
    assert "compare_active_target_source(" in source
