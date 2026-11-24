import importlib.util
from pathlib import Path

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
