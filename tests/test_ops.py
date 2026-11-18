import importlib.util
import os
import json
import sys
from pathlib import Path


def test_download_shutdown_excludes_docker_wrapper_and_other_tasks(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/ops/stop_asset_job.py'
    spec = importlib.util.spec_from_file_location('stop_asset_job', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'project'
    proc = tmp_path / 'proc'
    expected = [str(root/'scripts/ops/prepare_pangu_assets.py'), '--project', str(root), '--task', 'ours']
    commands = {
        10: ['docker', 'exec', 'container', 'python', '-B', *expected],
        11: ['python3', '-B', *expected],
        12: ['python3', '-B', *expected[:-1], 'other-task'],
        13: ['python3', '-B', str(tmp_path/'other-project/prepare_pangu_assets.py'), *expected[1:]],
    }
    for pid, argv in commands.items():
        path = proc/str(pid)
        path.mkdir(parents=True)
        (path/'cmdline').write_bytes(b'\0'.join(map(os.fsencode, argv))+b'\0')
    assert module.download_supervisors(root, 'ours', proc) == [11]


def test_deployment_suite_stops_on_busy_device_and_returns_failure(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/ops/run_deployment_suite.py'
    spec = importlib.util.spec_from_file_location('deployment_suite', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path/'project'
    task = root/'runs'/'suite'
    task.mkdir(parents=True)
    calls = []

    class BusyDevice:
        pid = 12345

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            case = root/'runs'/argv[argv.index('--task')+1]
            (case/'validation.json').write_text(json.dumps({
                'stage': 'device-pre', 'detail': 'selected device became busy'}))

        def wait(self):
            return 1

    monkeypatch.setattr(module.subprocess, 'Popen', BusyDevice)
    monkeypatch.setattr(sys, 'argv', ['suite', '--project', str(root),
        '--reference', str(root/'reference.json'), '--weights', str(root/'weights'),
        '--task', 'suite', '--devices', '3', '4', '5', '6'])
    assert module.main() == 1
    result = json.loads((task/'suite.json').read_text())
    assert len(calls) == 1
    assert result['status'] == 'incomplete'
    assert result['cases'][0]['status'] == 'failed'
    assert all(c['status'] == 'blocked_resource' for c in result['cases'][1:])
