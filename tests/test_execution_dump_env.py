"""CPU regression for NPU dump paths; no runtime/NPU imports or execution."""
import copy
import os
from pathlib import Path

from fullnet.execution import private_environment


def test_npu_dump_overrides_inherited_and_config_defaults_without_mutating_config(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    project = root
    monkeypatch.setenv("OMNI_DUMP_DIR", "/var/log/omni-npu/dump")
    for platform in ("a2", "a3", "a5", "cpu-test"):
        for configured in ({}, {"OMNI_DUMP_DIR": "/var/log/omni-npu/dump"}):
            run = tmp_path / f"{platform}-{bool(configured)}"
            run.mkdir()
            config = {"platform": platform, "devices": [1, 7], "env": configured}
            original = copy.deepcopy(config)
            env = private_environment(run, config, project)
            try:
                assert config == original
                if platform == "cpu-test":
                    assert env["OMNI_DUMP_DIR"] == "/var/log/omni-npu/dump"
                else:
                    dump = Path(env["OMNI_DUMP_DIR"])
                    assert dump == (run / "logs/omni-dump").resolve()
                    assert dump.is_dir() and dump.is_relative_to(run.resolve())
            finally:
                Path(env["VLLM_RPC_BASE_PATH"]).rmdir()
