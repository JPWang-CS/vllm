import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ops/registered_fleet_bridge.py"


def _module():
    spec = importlib.util.spec_from_file_location("registered_fleet_bridge_scope", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_project_excludes_runtime_and_large_model_trees():
    bridge = _module()
    for name in ("runtime", "assets", "build", "tmp", "weights", "experiments", "envs"):
        assert not bridge.copy_project_directory_allowed(name)
    assert not bridge.copy_project_directory_allowed("cache")
    assert bridge.copy_project_directory_allowed("cache", python_package=True)
    for name in ("scripts", "src", "models"):
        assert bridge.copy_project_directory_allowed(name)


def test_container_to_host_read_path_uses_longest_bind_destination():
    bridge = _module()
    mounts = [
        {"Type": "bind", "Source": "/host/root", "Destination": "/data2"},
        {"Type": "bind", "Source": "/host/assets", "Destination": "/data2/project/assets"},
    ]
    assert bridge.container_to_host_read_path(
        "/data2/project/assets/pangu/file.safetensors", mounts
    ) == "/host/assets/pangu/file.safetensors"


def test_container_to_host_read_path_maps_mount_root_and_rejects_unregistered():
    bridge = _module()
    mounts = [{"Type": "bind", "Source": "/host/project", "Destination": "/data2/project"}]
    assert bridge.container_to_host_read_path("/data2/project", mounts) == "/host/project"
    with pytest.raises(ValueError, match="no registered bind mount"):
        bridge.container_to_host_read_path("/other/assets", mounts)


def test_business_checkpoint_summary_is_independent_from_broker_heartbeat(tmp_path):
    bridge = _module()
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    (task / "repair/BUSINESS_CHECKPOINT.json").write_text(
        '{"cases":{"mixed":{"status":"completed"},"multi_host_tp":{"status":"pending"}}}',
        encoding="utf-8",
    )
    result = bridge.business_checkpoint_summary(task, ("mixed", "same_host_pd_split", "multi_host_tp"))
    assert result["controller_independent"] is True
    assert result["cases"]["mixed"]["state"] == "completed"
    assert result["cases"]["same_host_pd_split"]["state"] == "pending"
    assert result["cases"]["multi_host_tp"]["state"] == "pending"


def test_health_gate_accepts_a3_and_legacy_status_formats():
    bridge = _module()
    assert bridge.health_status_ok("Health: OK")
    assert bridge.health_status_ok("Health Status: OK")
    assert bridge.health_status_ok("| Ascend910 | OK |\n| Ascend910 | OK |")
    assert bridge.process_table_empty("No running processes found in NPU 1")
    assert bridge.process_table_empty("No process in device")


def test_health_gate_rejects_missing_or_bad_chip_status():
    bridge = _module()
    assert not bridge.health_status_ok("Health: Error")
    assert not bridge.health_status_ok("Health: OK\nHealth: Fault")
    assert not bridge.health_status_ok("Health")
    assert not bridge.process_table_empty("Process id 123 python3")


def test_generated_runner_has_isolated_health_gate_behavior():
    bridge = _module()
    namespace = {}
    prefix = bridge.RUNNER.split("j=pathlib.Path", 1)[0]
    exec(prefix, namespace)
    assert namespace["health_status_ok"]("Health: OK")
    assert namespace["health_status_ok"]("Health Status: OK")
    assert namespace["process_table_empty"]("No process in device.")
    assert not namespace["health_status_ok"]("Health: OK\nHealth: Fault")
    assert not namespace["process_table_empty"]("Process id 123 python3")


def test_atomic_sftp_write_completes_before_posix_rename_and_rejects_symlink():
    bridge = _module()

    class Output:
        def __init__(self, owner, path):
            self.owner, self.path, self.data = owner, path, bytearray()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.owner.files[self.path] = bytes(self.data)

        def write(self, data):
            self.data.extend(data)

    class SFTP:
        def __init__(self):
            self.files = {}
            self.modes = {}
            self.renames = []

        def lstat(self, path):
            if path not in self.files and path not in self.modes:
                raise FileNotFoundError(path)
            return SimpleNamespace(st_mode=self.modes.get(path, stat.S_IFREG))

        def open(self, path, mode):
            assert mode == "wx"
            assert path not in self.files
            return Output(self, path)

        def chmod(self, path, mode):
            self.modes[path] = mode

        def posix_rename(self, source, destination):
            assert source in self.files
            assert len(Path(source).name) < 255
            self.files[destination] = self.files.pop(source)
            self.renames.append((source, destination))

        def remove(self, path):
            self.files.pop(path, None)

    sftp = SFTP()
    bridge.atomic_sftp_write(sftp, "/project/src/module.py", b"new", "job-01")
    assert sftp.files["/project/src/module.py"] == b"new"
    assert len(sftp.renames) == 1
    assert not any("fleet-tmp" in path for path in sftp.files)
    sftp.files["/project/src/link.py"] = b"outside"
    sftp.modes["/project/src/link.py"] = stat.S_IFLNK
    with pytest.raises(ValueError, match="symlink"):
        bridge.atomic_sftp_write(sftp, "/project/src/link.py", b"new", "job-02")


def test_runner_device_leases_conflict_by_device_release_and_skip_empty(tmp_path, monkeypatch):
    bridge = _module()
    namespace = {}
    prefix = bridge.RUNNER.split("j=pathlib.Path", 1)[0]
    exec(prefix, namespace)
    held = set()

    class FakeFcntl(types.ModuleType):
        LOCK_EX = 1
        LOCK_NB = 2
        LOCK_UN = 4

        @staticmethod
        def flock(fd, flags):
            key = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            if flags == FakeFcntl.LOCK_UN:
                held.discard(key)
            elif key in held:
                raise BlockingIOError("busy")
            else:
                held.add(key)

    monkeypatch.setitem(sys.modules, "fcntl", FakeFcntl("fcntl"))
    project = tmp_path / "project"
    project.mkdir()
    first = namespace["acquire_device_leases"](project, [3, 1])
    assert sorted(path.name for path in (project / ".runtime" / "device-leases").iterdir()) == ["npu-1.lock", "npu-3.lock"]
    with pytest.raises(RuntimeError, match="device lease busy: 1"):
        namespace["acquire_device_leases"](project, [1])
    second = namespace["acquire_device_leases"](project, [2])
    namespace["release_device_leases"](first)
    reacquired = namespace["acquire_device_leases"](project, [1])
    namespace["release_device_leases"](second + reacquired)
    empty = namespace["acquire_device_leases"](project / "empty", [])
    assert empty == []
    assert not (project / "empty" / ".runtime" / "device-leases").exists()


def test_runner_device_leases_reject_runtime_and_lock_symlinks(tmp_path, monkeypatch):
    bridge = _module()
    namespace = {}
    exec(bridge.RUNNER.split("j=pathlib.Path", 1)[0], namespace)
    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(
        LOCK_EX=1, LOCK_NB=2, LOCK_UN=4, flock=lambda *_args: None))
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = root / ".runtime"
    try:
        runtime.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(RuntimeError, match="symlink"):
        namespace["acquire_device_leases"](root, [1])
    runtime.unlink()
    lease_dir = runtime / "device-leases"
    lease_dir.mkdir(parents=True)
    target = outside / "external.lock"
    target.write_bytes(b"")
    (lease_dir / "npu-1.lock").symlink_to(target)
    with pytest.raises(RuntimeError, match="escapes project"):
        namespace["acquire_device_leases"](root, [1])


def test_runner_skips_post_device_query_when_workload_did_not_start(tmp_path):
    bridge = _module()
    namespace = {}
    exec(bridge.RUNNER.split("j=pathlib.Path", 1)[0], namespace)
    job = tmp_path / "job"
    job.mkdir()
    namespace["write_device_post"](job, [1, 3], None, {}, False)
    assert "not_run" in (job / "device-1-post.txt").read_text()
    assert "not_run" in (job / "device-3-post.txt").read_text()


@pytest.mark.skipif(os.name == "nt", reason="real fcntl flock branch requires Linux")
def test_runner_device_leases_real_linux_branch(tmp_path):
    bridge = _module()
    namespace = {}
    exec(bridge.RUNNER.split("j=pathlib.Path", 1)[0], namespace)
    first = namespace["acquire_device_leases"](tmp_path / "project", [1])
    with pytest.raises(RuntimeError, match="device lease busy: 1"):
        namespace["acquire_device_leases"](tmp_path / "project", [1])
    namespace["release_device_leases"](first)
    second = namespace["acquire_device_leases"](tmp_path / "project", [1])
    namespace["release_device_leases"](second)


def _fake_proc(proc_root, pid, command, uid=None):
    proc = proc_root / str(pid)
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
    return proc


def test_controller_alive_accepts_current_controller_layout_and_legacy_glm_task(tmp_path):
    bridge = _module()
    proc_root = tmp_path / "proc"
    task = "/srv/project/runs/task"
    _fake_proc(proc_root, 101, ["python3", task + "/controllers/host/glm02/controller.py", task])
    _fake_proc(proc_root, 102, ["python3", task + "/glm_task.py", task])
    assert bridge.controller_process_alive(task, 101, proc_root=proc_root, uid=None)
    assert bridge.controller_process_alive(task, 102, proc_root=proc_root, uid=None)


def test_controller_alive_rejects_other_project_controller_and_uid_mismatch(tmp_path):
    bridge = _module()
    proc_root = tmp_path / "proc"
    task = "/srv/project/runs/task"
    _fake_proc(proc_root, 103, ["python3", "/other/runs/task/controllers/host/glm02/controller.py", task])
    _fake_proc(proc_root, 104, ["python3", task + "/controllers/host/glm02/controller.py", task])
    _fake_proc(proc_root, 105, ["python3", task + "2/glm_task.py", task + "2"])
    assert not bridge.controller_process_alive(task, 103, proc_root=proc_root, uid=None)
    assert not bridge.controller_process_alive(task, 104, proc_root=proc_root, uid=999999)
    assert not bridge.controller_process_alive(task, 105, proc_root=proc_root, uid=None)


def test_latest_controller_status_uses_real_controller_directory_and_unknown_terminal(tmp_path):
    bridge = _module()
    task = tmp_path / "runs" / "task"
    controller = task / "controllers" / "host" / "glm02"
    controller.mkdir(parents=True)
    (task / "STATE.json").write_text(json.dumps({"exit": 0, "child_alive": False}), encoding="utf-8")
    (controller / "STATE.json").write_text(json.dumps({"exit": 1, "child_alive": False}), encoding="utf-8")
    (controller / "RUN_IDENTITY.json").write_text(json.dumps({"controller_run": str(controller)}), encoding="utf-8")
    result = bridge.latest_controller_status(task)
    assert result["controller_dir"].replace("\\", "/").endswith("/controllers/host/glm02")
    assert result["status"] == "unknown"
    assert result["controller_independent"] is True


def test_latest_controller_status_scans_exact_controller_without_active(tmp_path):
    bridge = _module()
    task = tmp_path / "runs" / "task"
    controller = task / "controllers" / "host" / "glm02"
    controller.mkdir(parents=True)
    (controller / "STATE.json").write_text(json.dumps({"phase": "running"}), encoding="utf-8")
    (controller / "RUN_IDENTITY.json").write_text(json.dumps({"controller_run": str(controller)}), encoding="utf-8")
    proc_root = tmp_path / "proc"
    controller_main = (controller / "controller.py").as_posix()
    _fake_proc(proc_root, 201, ["python3", controller_main])
    result = bridge.latest_controller_status(task, proc_root=proc_root)
    assert result["status"] == "running"
    assert result["controller_alive"] is True
