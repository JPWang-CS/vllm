"""Real source-file comparisons and process fixtures; no servers or devices."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/ops' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_old_broker_or_changed_loaded_source_rejects_new_run(tmp_path):
    bridge = module('registered_fleet_bridge')
    assert not bridge.broker_heartbeat_compatible({'status': 'connected'})
    good = {'framework_source_receipt_schema': 1,
            'broker_source_file': 'scripts/ops/registered_fleet_bridge.py',
            'loaded_source_matches_disk': True}
    assert bridge.broker_heartbeat_compatible(good)
    assert module('recover_framework').broker_heartbeat_compatible(good)
    path = tmp_path / 'broker.py'; path.write_bytes(b'old')
    captured, content = bridge.bridge_source_snapshot(path)
    path.write_bytes(b'new')
    assert not bridge.loaded_bridge_matches_disk(captured, content)


def test_branch_name_alone_does_not_prove_current_source(tmp_path):
    bridge = module('registered_fleet_bridge')
    root = tmp_path / 'project'; snapshot = root / 'runs/release/published'
    rel = 'src/fullnet/execution.py'
    for base in (root, snapshot):
        target = base / rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'published')
    receipt = {'schema': 1, 'branch': 'main', 'snapshot': str(snapshot).replace('\\', '/'), 'files': [rel]}
    class Remote:
        def open(self, path, mode): return Path(path).open(mode)
    # POSIX remote paths are represented by local fixture paths via a mapping.
    class Mapped:
        def local(self, path):
            return root if path == '/project' else root / Path(path.removeprefix('/project/'))
        def open(self, path, mode):
            return self.local(path).open(mode)
        def lstat(self, path):
            return self.local(path).lstat()
    remote_receipt = dict(receipt, snapshot='/project/runs/release/published')
    assert bridge.compare_control_to_snapshot(Mapped(), '/project', remote_receipt)['matches']
    (root / rel).write_bytes(b'changed!!')
    assert not bridge.compare_control_to_snapshot(Mapped(), '/project', remote_receipt)['matches']
    with pytest.raises(ValueError):
        bridge.compare_control_to_snapshot(Mapped(), '/project', dict(remote_receipt, files=['../secret']))


def test_actual_processes_guard_hot_updates_and_ignores_stale_state(tmp_path):
    bridge = module('registered_fleet_bridge'); proc = tmp_path / 'proc'; proc.mkdir()
    for pid, argv, env in [(910001, b'python\0/project/models/pangu/adapters/serve.py\0', b''),
                           (910002, b'python\0spawn_main\0', b'FULLNET_RUN=/project/models/pangu/experiments/case\0'),
                           (910003, b'python\0/other/models/pangu/serve.py\0', b'')]:
        item=proc/str(pid);item.mkdir();(item/'cmdline').write_bytes(argv);(item/'environ').write_bytes(env)
    result=bridge.project_workload_processes('/project', proc)
    assert {x['pid'] for x in result['active']} == {910001, 910002}


def test_run_requires_receipt_but_status_and_inventory_remain_available(tmp_path):
    recover=module('recover_framework');task=tmp_path/'runs/fixture';(task/'repair').mkdir(parents=True)
    (task/'repair/probe.py').write_text('pass\n')
    with pytest.raises(RuntimeError, match='broker'):
        recover.fleet_submit(tmp_path,task,'new-run','run','192.0.2.1','job',script='repair/probe.py',devices=[])
    assert not (task/'fleet/requests/new-run.json').exists()
    value=recover.fleet_submit(tmp_path,task,'inventory','inventory','192.0.2.1','job',devices=[])
    assert value
