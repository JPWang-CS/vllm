"""CPU coverage for the explicit job-private role environment seam."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
from unittest.mock import Mock

import pytest

from test_pd_readiness import _cpu_stub_modules

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/pangu-92b/adapters/serve_pd_split.py"


@pytest.fixture
def adapter(monkeypatch):
    for name, module in _cpu_stub_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(ADAPTER.parent))
    spec = importlib.util.spec_from_file_location("pd_role_environment_test_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sidecar(tmp_path):
    job = tmp_path / "job"
    hook = job / "private/hooks"
    hook.mkdir(parents=True)
    values = {}
    for role in ("producer", "consumer"):
        metrics = job / "metrics" / role
        metrics.mkdir(parents=True)
        values[role] = {"PYTHONPATH": str(hook), "VLLM_PLUGINS": "base,private_umbrella",
                        "FULLNET_KV_OBSERVER_ROLE": role, "FULLNET_KV_OBSERVER_METRICS": "1",
                        "PROMETHEUS_MULTIPROC_DIR": str(metrics), "FULLNET_TEST_RESERVE_BYTES": "4096"}
    path = job / "private/roles.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    base = {"FULLNET_FLEET_JOB_ROOT": str(job), "PYTHONPATH": "baseline-source",
            "ASCEND_HOME_PATH": "chosen-cann", "ASCEND_RT_VISIBLE_DEVICES": "inherited"}
    return path, values, base


def test_explicit_merge_preserves_baseline_and_final_device_choice(adapter, tmp_path):
    path, values, base = sidecar(tmp_path)
    original = base.copy()
    envs = adapter.role_environments(base, str(path))
    assert base == original
    for role, device in [("producer", "6"), ("consumer", "7")]:
        env = adapter.instance_env(envs[role], device)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == env["ASCEND_VISIBLE_DEVICES"] == device
        assert env["ASCEND_HOME_PATH"] == "chosen-cann"
        assert env["PYTHONPATH"].split(os.pathsep) == [values[role]["PYTHONPATH"], "baseline-source"]
        assert env["FULLNET_KV_OBSERVER_ROLE"] == role
    assert envs["producer"]["PROMETHEUS_MULTIPROC_DIR"] != envs["consumer"]["PROMETHEUS_MULTIPROC_DIR"]


def test_default_has_no_new_environment_behavior(adapter):
    base = {"PYTHONPATH": "existing", "ASCEND_RT_VISIBLE_DEVICES": "9"}
    result = adapter.role_environments(base)
    assert result == {"producer": base, "consumer": base}
    assert result["producer"] is not base and result["consumer"] is not base


@pytest.mark.parametrize("key", ["ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
                                 "ASCEND_HOME_PATH", "API_KEY", "TOKEN", "HOME", "FULLNET_REMOTE_PROJECT",
                                 "FULLNET_SECRET_RESERVE_BYTES", "HCCL_INTRA_ROCE_ENABLE"])
def test_sidecar_cannot_override_device_sdk_credentials_or_identity(adapter, tmp_path, key):
    path, values, base = sidecar(tmp_path)
    values["consumer"][key] = "forbidden"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="not allowed"):
        adapter.role_environments(base, str(path))


def test_outside_missing_or_symlink_sidecar_is_rejected(adapter, tmp_path):
    path, values, base = sidecar(tmp_path)
    outside = tmp_path / "roles.json"
    outside.write_text(path.read_text())
    with pytest.raises(ValueError, match="escapes"):
        adapter.role_environments(base, str(outside))
    with pytest.raises(FileNotFoundError): adapter.role_environments(base, str(path.parent / "missing"))
    link = path.parent / "symlink.json"
    try: link.symlink_to(path)
    except OSError: return  # Windows may deny creating a symlink; other path checks still ran.
    with pytest.raises(ValueError, match="symlink"):
        adapter.role_environments(base, str(link))


@pytest.mark.parametrize("change", ["wrong_role", "shared_metrics", "new_public_pythonpath", "non_string", "zero_reserve"])
def test_invalid_role_contract_fails_before_launch(adapter, tmp_path, change):
    path, values, base = sidecar(tmp_path)
    if change == "wrong_role": values["consumer"]["FULLNET_KV_OBSERVER_ROLE"] = "producer"
    if change == "shared_metrics": values["consumer"]["PROMETHEUS_MULTIPROC_DIR"] = values["producer"]["PROMETHEUS_MULTIPROC_DIR"]
    if change == "new_public_pythonpath": values["consumer"]["PYTHONPATH"] = str(tmp_path)
    if change == "non_string": values["consumer"]["VLLM_PLUGINS"] = []
    if change == "zero_reserve": values["consumer"]["FULLNET_TEST_RESERVE_BYTES"] = "0"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError): adapter.role_environments(base, str(path))


def test_main_passes_each_environment_before_popen_and_keeps_proxy_base(adapter, tmp_path, monkeypatch):
    path, values, base = sidecar(tmp_path)
    for key, value in base.items(): monkeypatch.setenv(key, value)
    argv = ["adapter", "--serve-adapter", "serve.py", "--model", "weights", "--proxy-script", "relay.py",
            "--producer-devices", "6", "--consumer-devices", "7", "--producer-port", "21001",
            "--consumer-port", "21002", "--proxy-port", "21003", "--front-port", "21000",
            "--logs", str(path.parent / "logs"), "--role-env-file", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    popen = Mock(side_effect=lambda *args, **kw: Mock(poll=lambda: None))
    monkeypatch.setattr(adapter.subprocess, "Popen", popen)
    monkeypatch.setattr(adapter, "fetch", lambda *args: (200, b'{"data":[{"id":"pangu-92b"}]}'))
    monkeypatch.setattr(adapter.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "shutdown", lambda code: None)
    adapter.main()
    assert popen.call_count == 3
    for i, role in enumerate(("producer", "consumer")):
        env = popen.call_args_list[i].kwargs["env"]
        assert env["FULLNET_KV_OBSERVER_ROLE"] == role
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == ("6" if role == "producer" else "7")
        assert env["PROMETHEUS_MULTIPROC_DIR"] == values[role]["PROMETHEUS_MULTIPROC_DIR"]
    assert popen.call_args_list[2].kwargs["env"].get("FULLNET_KV_OBSERVER_ROLE") is None
    values["consumer"]["ASCEND_HOME_PATH"] = "bad"
    path.write_text(json.dumps(values), encoding="utf-8")
    popen.reset_mock()
    with pytest.raises(ValueError): adapter.main()
    popen.assert_not_called()
