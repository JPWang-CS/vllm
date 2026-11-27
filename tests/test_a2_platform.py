import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fullnet.cli import main as cli_main
from fullnet.config import resolve
from fullnet.execution import private_environment
from fullnet.pangu7b_device_guard import (configure_pangu7b_platform_env,
                                          parse_npu_smi_records, verify_npu_smi_output)


def _model_project(tmp_path, model="pangu-7b", platforms=("a2",)):
    model_dir = tmp_path / "models" / model
    model_dir.mkdir(parents=True)
    (model_dir / "model.yaml").write_text(
        yaml.safe_dump({"name": model, "family": "pangu", "platforms": list(platforms)}),
        encoding="utf-8",
    )
    return tmp_path


def _config(root, model="pangu-7b", platform="a2", devices=None):
    return {
        "schema_version": 1, "model": model, "platform": platform,
        "devices": [0] if devices is None else devices,
        "identity": {"device_model": "unverified", "weights": "weights", "dtype": "bfloat16"},
        "env": {}, "changes": [],
        "service": {"argv": [sys.executable, "-c", "pass"], "ready_timeout": 1,
                     "request_timeout": 1, "shutdown_timeout": 1},
        "evaluation": {"dataset": str(root / "data.jsonl"), "metric": "exact_match",
                        "context_limit": 16, "chat_template_kwargs": {}},
    }


def test_a2_config_and_cli_route_is_real_platform_not_a3(tmp_path, capsys):
    root = _model_project(tmp_path)
    (root / "data.jsonl").write_text("{}\n", encoding="utf-8")
    path = root / "a2.yaml"
    path.write_text(yaml.safe_dump(_config(root)), encoding="utf-8")
    assert resolve(path, root)["platform"] == "a2"

    result = cli_main(["--root", str(root), "configure", "--model", "pangu-7b",
                       "--platform", "a2", "--python", sys.executable,
                       "--weights", "weights", "--dataset", "data.jsonl",
                       "--official-metrics", "metrics.py", "--devices", "0"])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["config"].endswith("a2.local.yaml")
    generated = yaml.safe_load(Path(output["config"]).read_text(encoding="utf-8"))
    assert generated["platform"] == "a2"
    assert generated["service"]["base_url"].endswith(":18320/v1")


def test_a2_config_still_requires_device_identity_and_92b_manifest_stays_a5_only(tmp_path):
    root = _model_project(tmp_path, platforms=("a2",))
    no_device = root / "no-device.yaml"
    no_device.write_text(yaml.safe_dump(_config(root, devices=[])), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit devices"):
        resolve(no_device, root)

    restricted = _model_project(tmp_path / "restricted", model="pangu-92b", platforms=("a5",))
    restricted_path = restricted / "a2.yaml"
    restricted_path.write_text(yaml.safe_dump(_config(restricted, model="pangu-92b")), encoding="utf-8")
    with pytest.raises(ValueError, match="not declared"):
        resolve(restricted_path, restricted)


def test_a2_execution_sets_npu_visibility_but_does_not_bypass_gate():
    # The real IPC length gate is intentionally exercised with the short
    # project-local runtime root, not defeated with a monkeypatch or temp root.
    root = Path(__file__).parents[1].resolve()
    runtime_root = (root / ".runtime").resolve()
    assert runtime_root.is_relative_to(root)
    run = (runtime_root / f"a2-visibility-{os.getpid()}").resolve()
    assert run.is_relative_to(root)
    run.mkdir(parents=True)
    try:
        env = private_environment(run, {"platform": "a2", "devices": [0], "env": {}}, root)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0"
        assert env["ASCEND_VISIBLE_DEVICES"] == "0"
        ipc = Path(env["VLLM_RPC_BASE_PATH"])
        assert ipc.is_relative_to(root)
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_platform_hint_isolated_in_private_runner_env():
    inherited = {"ASCEND_COMPUTE_UNIT": "ascend950", "KEEP_RUNTIME_HINT": "source-value"}
    a2 = configure_pangu7b_platform_env(inherited, "a2")
    a3 = configure_pangu7b_platform_env(inherited, "a3")
    a5 = configure_pangu7b_platform_env(inherited, "a5")
    assert "ASCEND_COMPUTE_UNIT" not in a2
    assert "ASCEND_COMPUTE_UNIT" not in a3
    assert a5["ASCEND_COMPUTE_UNIT"] == "ascend950"
    assert a2["KEEP_RUNTIME_HINT"] == a3["KEEP_RUNTIME_HINT"] == "source-value"
    assert inherited["ASCEND_COMPUTE_UNIT"] == "ascend950"
    custom = configure_pangu7b_platform_env({"ASCEND_COMPUTE_UNIT": "custom910"}, "a2")
    assert custom["ASCEND_COMPUTE_UNIT"] == "custom910"


def test_a2_device_guard_accepts_910b2_and_rejects_unknown_or_wrong_platform():
    table = "| 1 | 910B2 | OK |\nNo running processes found in NPU 1 \n"
    assert verify_npu_smi_output(table, "a2", 1)["name"] == "910B2"
    kv = "Device ID=1 Name=910B2 Status=OK\nNo running processes found in NPU 1\n"
    assert verify_npu_smi_output(kv, "a2", 1)["name"] == "910B2"
    with pytest.raises(RuntimeError, match="not allowlisted"):
        verify_npu_smi_output(table.replace("910B2", "910X"), "a2", 1)
    with pytest.raises(RuntimeError, match="not allowlisted"):
        verify_npu_smi_output(table, "a3", 1)


def test_guard_uses_deidentified_a2_a3_a5_receipt_shapes():
    # These are the smallest in-repo forms extracted from the private raw
    # receipts. No host, run, or remote path is part of the public test.
    a2_b1 = "| 0     910B1 | OK | 0 |\n| No running processes found in NPU 0 |"
    a2_b2 = "| 0     910B2 | OK | 0 |\n| No running processes found in NPU 0 |"
    a2_b3 = "| 0     910B3 | OK | 0 |\n| No running processes found in NPU 0 |"
    a3 = ("| 2     Ascend910 | OK | 0 |\n"
          "| 2     Ascend910 | OK | - |\n"
          "| No running processes found in NPU 2 |")
    a5 = "| 0 | Ascend950DT | OK | 0 |\n| No running processes found in NPU 0 |"

    assert verify_npu_smi_output(a2_b1, "a2", 0)["name"].strip() == "910B1"
    assert verify_npu_smi_output(a2_b2, "a2", 0)["name"].strip() == "910B2"
    assert verify_npu_smi_output(a2_b3, "a2", 0)["name"].strip() == "910B3"
    assert len([v for v in parse_npu_smi_records(a3) if v["device_id"] == 2]) == 2
    assert verify_npu_smi_output(a3, "a3", 2)["name"].strip() == "Ascend910"
    assert verify_npu_smi_output(a5, "a5", 0)["name"].strip() == "Ascend950DT"

    a3_second_critical = []
    seen = 0
    for line in a3.splitlines():
        if re.search(r"\|\s*2\s+Ascend910\s+\|\s*OK\s*\|", line):
            seen += 1
            if seen == 2:
                line = line.replace("| OK", "| Critical", 1)
        a3_second_critical.append(line)
    with pytest.raises(RuntimeError, match="health is not OK"):
        verify_npu_smi_output("\n".join(a3_second_critical), "a3", 2)


@pytest.mark.parametrize("script", [
    "scripts/ops/run_pangu7b_matrix.py",
    "scripts/ops/validate_pangu7b.py",
    "scripts/ops/pangu_pipeline.py",
])
def test_7b_runner_cli_exposes_a2_without_running_device(script):
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([sys.executable, script, "--help"], cwd=root,
                            env=env, capture_output=True, text=True, check=True)
    assert "a2" in result.stdout
