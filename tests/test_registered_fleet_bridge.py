import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("fleet_bridge", Path(__file__).parents[1] / "scripts/ops/registered_fleet_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def request(**changes):
    value = {"request_id": "probe-01", "action": "run", "host": "registered",
             "job": "first", "script": "repair/probe.py", "devices": [1, 3], "args": []}
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", [
    {"host": "other"}, {"script": "../secret.py"}, {"script": "/tmp/probe.py"},
    {"script": "repair/../../probe.py"}, {"script": "repair/probe.json"},
    {"request_id": "../../escape"}, {"devices": [1, 1]}, {"devices": [True]},
    {"devices": [-1]}, {"args": "shell command"}, {"action": "shell"},
    {"command": "arbitrary"},
])
def test_rejects_out_of_scope_request(changes):
    with pytest.raises((ValueError, TypeError)):
        bridge.validate_request(request(**changes), {"registered": {}})


def test_run_request_preserves_args_without_shell_interpolation():
    r = request(args=["literal $(command)", "space in argument"])
    assert bridge.validate_request(r, {"registered": {}})["args"] == r["args"]


def test_runtime_runner_compiles_and_records_job_exit():
    compile(bridge.RUNNER, "fleet_runner", "exec")
    assert "(j/'exit.txt').write_text" in bridge.RUNNER
    assert "No process in device" in bridge.RUNNER
    assert "os.killpg(child.pid,signal.SIGTERM)" in bridge.RUNNER
