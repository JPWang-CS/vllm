"""Focused regression tests for fleet inventory's device execution surface."""

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ops/registered_fleet_bridge.py"
SPEC = importlib.util.spec_from_file_location("registered_fleet_bridge_inventory_surface", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def query(stdout, *, exit_code=0):
    return {"exit": exit_code, "stdout": stdout, "stderr": ""}


def inventory_probe(host_mapping, container_mapping, *, container_exit=0):
    container_payload = {
        "mapping": query(container_mapping, exit_code=container_exit),
        "devices": query("container-devices", exit_code=container_exit),
        "tool": {
            "path": "/project/.runtime/device-query/inventory-01/npu-smi",
            "realpath": "/project/.runtime/device-query/inventory-01/npu-smi",
            "execution_surface": "container",
            "ld_library_path": "/usr/local/Ascend/driver/lib64/driver",
        },
    }
    return {
        "host_raw": {
            "mapping": query(host_mapping),
            "devices": query("host-devices"),
        },
        # This mirrors the generated container probe: its JSON has no exit
        # field; the outer docker-exec result adds it after parsing.
        "container_query": dict(container_payload, exit=container_exit),
        "container_tool": {
            "path": "/project/.runtime/device-query/inventory-01/npu-smi",
            "realpath": "/project/.runtime/device-query/inventory-01/npu-smi",
            "execution_surface": "container",
            "source": "/usr/local/sbin/npu-smi",
        },
    }


def test_different_host_and_container_mappings_use_container_as_canonical():
    raw = inventory_probe("host-board-5-phy-3", "container-board-5-phy-6")

    result = BRIDGE.canonical_inventory_result(raw)

    assert result["host_raw"]["mapping"]["stdout"] == "host-board-5-phy-3"
    assert result["mapping"]["stdout"] == "container-board-5-phy-6"
    assert result["devices"]["stdout"] == "container-devices"
    assert result["execution_surface"] == "container"
    assert result["tool"]["path"].startswith("/project/.runtime/device-query/")


def test_container_tool_identity_is_preserved_for_queue_diagnostics():
    result = BRIDGE.canonical_inventory_result(
        inventory_probe("host-mapping", "container-mapping")
    )

    assert result["tool"]["execution_surface"] == "container"
    assert result["tool"]["realpath"] == result["tool"]["path"]
    assert result["tool"]["source"] == "/usr/local/sbin/npu-smi"


def test_generated_container_probe_payload_is_success_only_after_outer_exit_is_added():
    raw = inventory_probe("host-mapping", "container-mapping")

    assert "exit" not in {
        "mapping": raw["container_query"]["mapping"],
        "devices": raw["container_query"]["devices"],
        "tool": raw["container_query"]["tool"],
    }
    assert BRIDGE.canonical_inventory_result(raw)["execution_surface"] == "container"


def test_failed_container_query_does_not_fallback_to_host():
    raw = inventory_probe("host-board-5-phy-3", "container-unavailable", container_exit=1)

    with pytest.raises(RuntimeError, match="host evidence is diagnostic only"):
        BRIDGE.canonical_inventory_result(raw)


def test_inventory_branch_stages_and_executes_the_tool_inside_container():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    remote_sources = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "code" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and "source_tool=None" in node.value.value
    ]
    assert remote_sources
    source = remote_sources[-1]
    compile(source, "<inventory-remote>", "exec")
    remote_tree = ast.parse(source)
    container_sources = [
        node.value.value
        for node in ast.walk(remote_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "container_probe" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert container_sources
    compile(container_sources[0], "<container-probe>", "exec")

    assert "device-query" in source
    assert "docker','exec','--user','0','-i','-w'" in source
    assert "execution_surface':'container'" in source
    assert "o['host_raw']" in source
    assert "(pathlib.Path('/usr/local/sbin/npu-smi'),pathlib.Path('/usr/local/bin/npu-smi'))" in source
    assert "source_tool is None:raise RuntimeError('runner host-matched npu-smi ELF is unavailable')" in source
    assert "management_tool),'execution_surface':'host','query':'not_run'" in source
    assert "o['container_query']['exit']=0" in source
    assert "return canonical_inventory_result(raw)" in SOURCE.read_text(encoding="utf-8")
