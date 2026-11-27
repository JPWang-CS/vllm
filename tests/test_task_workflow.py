import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fleet_submit_rejects_omitted_devices_before_queue_write(tmp_path):
    module = load("scripts/ops/recover_framework.py", "workflow_fix_recover")
    task = tmp_path / "task"
    (task / "repair").mkdir(parents=True)
    (task / "repair/probe.py").write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit host NPU ID list"):
        module.fleet_submit(tmp_path, task, "missing-devices", "run", "192.0.2.10", "job",
                            script="repair/probe.py", devices=None, args=[])
    assert not (task / "fleet/requests/missing-devices.json").exists()


def test_fleet_submit_cli_rejects_omitted_devices_without_creating_request(tmp_path):
    task_name = "fixture"
    project = tmp_path / "project"
    script = project / "scripts/ops/recover_framework.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/ops/recover_framework.py", script)
    (project / "runs" / task_name / "repair").mkdir(parents=True)
    request = project / "runs" / task_name / "fleet" / "requests" / "workflow-fix-cli-missing-devices.json"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--project", str(project),
         "--task", task_name, "--action", "fleet-submit", "--request-id",
         "workflow-fix-cli-missing-devices", "--fleet-action", "run", "--host",
         "192.0.2.10", "--job", "workflow-fix-cli"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "explicit host NPU ID list" in result.stderr
    assert not request.exists()


def test_cli_rejects_absolute_task_path_before_queue_or_directory_write(tmp_path):
    task_name = "fixture"
    project = tmp_path / "project"
    script = project / "scripts/ops/recover_framework.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/ops/recover_framework.py", script)
    (project / "runs" / task_name / "repair").mkdir(parents=True)
    task_path = project / "runs" / task_name
    request_id = "workflow-fix-cli-invalid-task-path"
    request = task_path / "fleet" / "requests" / (request_id + ".json")
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--project", str(project),
         "--task", str(task_path), "--action", "fleet-job-status", "--host",
         "192.0.2.10", "--job", "workflow-fix-cli", "--wait-seconds", "0"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "expected a task identifier, not a path" in result.stderr
    assert "--task fixture" in result.stderr
    assert not request.exists()


def test_cli_invalid_task_path_without_identifier_suffix_uses_placeholder(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", "scripts/ops/recover_framework.py", "--project", str(ROOT),
         "--task", str(tmp_path / "bad task"), "--action", "fleet-job-status", "--host",
         "192.0.2.10", "--job", "workflow-fix-cli", "--wait-seconds", "0"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "use --task <task-id>" in result.stderr
    assert "deployment-remaining-20261124-02" not in result.stderr


def test_cli_result_orders_status_summary_before_raw_response():
    module = load("scripts/ops/recover_framework.py", "cli_result_order")
    raw_response = {"file_metadata": {"state.json": {"json_valid": True}},
                    "state.json": "{" + ("x" * 100) + "}"}
    ordered = module.order_cli_result({
        "response": raw_response,
        "job_outcome": {"execution": "running"},
        "operator_update": {"id": "note-1"},
        "status": "received",
        "observation": "fresh_job_status",
        "response_path": "D:/task/response.json",
        "other": "preserved",
    })
    keys = list(ordered)
    assert keys[:5] == ["job_outcome", "operator_update", "status", "observation", "response_path"]
    assert keys[-1] == "response"
    assert ordered["response"] is raw_response
    encoded = json.dumps(ordered, ensure_ascii=False)
    assert encoded.find('"job_outcome"') < encoded.find('"response"')


def test_cli_result_normalizes_list_before_standards_assignment():
    module = load("scripts/ops/recover_framework.py", "cli_result_list_normalization")
    result = module.order_cli_result([{"request_id": "inventory-1"}])
    result["readable_server_standards"] = {"status": "readable"}
    result = module.order_cli_result(result)
    assert result["result"] == [{"request_id": "inventory-1"}]
    assert result["readable_server_standards"] == {"status": "readable"}


@pytest.mark.parametrize("module_name", ["workflow_fix_recover_checkpoint", "workflow_fix_bridge_checkpoint"])
def test_four_types_checkpoint_is_read_when_cases_is_empty(tmp_path, module_name):
    relative = "scripts/ops/recover_framework.py" if module_name.endswith("recover_checkpoint") else "scripts/ops/registered_fleet_bridge.py"
    module = load(relative, module_name)
    task = tmp_path / "task"
    task.mkdir()
    (task / "CHECKPOINT.json").write_text(json.dumps({
        "cases": {},
        "four_types": {
            "mixed": {"status": "running", "job": "job-mixed", "request": "req-mixed",
                       "host": "192.0.2.10", "next_action": "read status"},
        },
    }), encoding="utf-8")
    result = (module.business_checkpoint_status if hasattr(module, "business_checkpoint_status")
              else module.business_checkpoint_summary)(task, ("mixed",))
    assert result["checkpoint_field"] == "four_types"
    assert result["cases"]["mixed"]["state"] == "running"
    assert result["cases"]["mixed"]["job_id"] == "job-mixed"
    assert result["cases"]["mixed"]["next_action"] == "read status"


def test_invalid_checkpoint_is_reported_instead_of_being_treated_as_empty(tmp_path):
    module = load("scripts/ops/recover_framework.py", "workflow_fix_invalid_checkpoint")
    task = tmp_path / "task"
    task.mkdir()
    (task / "CHECKPOINT.json").write_text('{"cases": "not-a-map"}', encoding="utf-8")
    with pytest.raises(ValueError, match="business checkpoint cases"):
        module.business_checkpoint_status(task)


def test_public_fleet_metadata_is_whitelisted_and_has_no_receipts_or_credentials():
    module = load("scripts/ops/registered_fleet_bridge.py", "workflow_fix_metadata")
    targets = {
        "192.0.2.10": {
            "management": "/home/user/management", "project": "/home/user/management/projects/vllm",
            "container": "vllm", "protected_devices": [3], "model": "secret-model",
            "private_key": "/home/user/.ssh/id_rsa", "local_receipts": "C:/private/receipts",
        },
    }
    value = module.public_fleet_metadata("control", "/home/user/management/projects/vllm",
                                         "task-01", targets)
    assert set(value) == {"schema", "control_host", "control_project", "task", "targets"}
    assert set(value["targets"]["192.0.2.10"]) == {
        "host", "container", "project", "management", "protected_devices"
    }
    assert "private_key" not in json.dumps(value)
    assert "receipts" not in json.dumps(value)


def test_public_fleet_metadata_accepts_registered_sibling_project_and_rejects_symlink_resolution():
    module = load("scripts/ops/registered_fleet_bridge.py", "workflow_fix_metadata_scope")
    target = {"management": "/home/user/management", "project": "/home/user/projects/vllm",
              "container": "vllm", "protected_devices": []}
    value = module.public_fleet_metadata("control", "/home/user/management/projects/vllm", "task-01",
                                         {"192.0.2.10": target})
    assert value["targets"]["192.0.2.10"]["project"] == "/home/user/projects/vllm"
    with pytest.raises(ValueError, match="resolves outside"):
        module.validate_remote_path("/home/user/management/projects/vllm/runs/task-01",
                                   "/home/user/shared/task-01")


@pytest.mark.parametrize("value", [[None], [{"status": "running"}],
                                     [{"case": "same"}, {"name": "same"}],
                                     [{"case": ""}]])
def test_checkpoint_case_lists_reject_incomplete_or_duplicate_entries(tmp_path, value):
    module = load("scripts/ops/recover_framework.py", "workflow_fix_bad_checkpoint_list")
    task = tmp_path / "task"
    task.mkdir()
    (task / "CHECKPOINT.json").write_text(json.dumps({"cases": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="business checkpoint cases"):
        module.business_checkpoint_status(task)


def test_failure_context_keeps_earlier_device_error_and_separate_tail(tmp_path):
    module = load(".agents/skills/fullnet-debug/scripts/extract_failure_context.py", "failure_context")
    log = tmp_path / "run.log"
    log.write_text(
        "INFO startup\npagefault in AICPU userDevId3\nTraceback (most recent call last):\n"
        "RuntimeError: 507018 get kernel failed\nfinal ERROR after cleanup\n",
        encoding="utf-8",
    )
    result = module.extract([log], tmp_path, context_lines=1, max_events=8, tail_events=2,
                            max_file_bytes=1024 * 1024)
    assert result["first_traceback"]["line"] == 3
    assert result["first_error"]["line"] == 2
    assert result["first_error"]["kind"] == "device_error"
    assert result["first_error"]["path"] == str(log.resolve())
    assert result["tail_summary"][-1]["text"] == "final ERROR after cleanup"
    assert "root-cause" in result["note"]


def test_failure_context_skips_sensitive_path_and_redacts_sensitive_value(tmp_path):
    module = load(".agents/skills/fullnet-debug/scripts/extract_failure_context.py", "failure_context_sensitive")
    secret_dir = tmp_path / "credentials"
    secret_dir.mkdir()
    (secret_dir / "log.txt").write_text("ERROR token=do-not-read\n", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text('{"secret": "must-not-be-read", "message": "ERROR settings"}\n', encoding="utf-8")
    safe = tmp_path / "safe.log"
    safe.write_text("ERROR token=visible-secret\n", encoding="utf-8")
    result = module.extract([tmp_path], tmp_path, max_file_bytes=1024 * 1024)
    assert any(item["reason"] == "sensitive_path_not_read" for item in result["skipped"])
    assert str(settings.resolve()) not in {item["path"] for item in result["files"]}
    assert "must-not-be-read" not in json.dumps(result)
    assert all("visible-secret" not in json.dumps(item) for item in result["events"])


def test_failure_context_warning_burst_cannot_hide_later_real_failures(tmp_path):
    module = load(".agents/skills/fullnet-debug/scripts/extract_failure_context.py", "failure_context_warning_burst")
    log = tmp_path / "warning-burst.log"
    log.write_text(
        "".join(f"WARNING failed optional plugin {index}\n" for index in range(50))
        + "Traceback (most recent call last):\n"
        + "RuntimeError: service failed\n"
        + "pagefault in AICPU userDevId3\n",
        encoding="utf-8",
    )
    result = module.extract([log], tmp_path, context_lines=1, max_events=2, tail_events=2,
                            max_file_bytes=1024 * 1024)
    assert len(result["events"]) == 2
    assert result["events"][0]["kind"] == "traceback"
    assert result["events"][1]["kind"] == "actual_error"
    assert result["tail_summary"][-1]["kind"] == "device_error"
    assert all("WARNING" not in event["text"] for event in result["events"])


def test_failure_context_cli_fails_when_no_safe_log_was_read(tmp_path):
    helper = ROOT / ".agents/skills/fullnet-debug/scripts/extract_failure_context.py"
    result = subprocess.run(
        [sys.executable, str(helper), "missing.log", "--project-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    assert result.returncode != 0
    assert payload["status"] == "no_safe_logs"


def test_failure_context_cli_returns_zero_and_marks_no_candidates_for_readable_log(tmp_path):
    helper = ROOT / ".agents/skills/fullnet-debug/scripts/extract_failure_context.py"
    log = tmp_path / "clean.log"
    log.write_text("INFO service ready\nWARNING optional plugin skipped\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(helper), str(log), "--project-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["no_error_candidates"] is True


def test_inventory_refresh_uses_new_request_ids(monkeypatch, tmp_path):
    module = load("scripts/ops/recover_framework.py", "fresh_inventory_batch")
    timestamps = iter((100, 101))
    monkeypatch.setattr(module.time, "time_ns", lambda: next(timestamps))
    calls = []
    monkeypatch.setattr(module, "fleet_inventory_request",
                        lambda root, task, request, host, job: calls.append(request) or {"request_id": request})
    task = tmp_path / "batch"
    first = module.fleet_inventory_batch(tmp_path, task, ["192.0.2.10", "192.0.2.11"])
    second = module.fleet_inventory_batch(tmp_path, task, ["192.0.2.10", "192.0.2.11"])
    assert set(x["request_id"] for x in first["requests"]).isdisjoint(
        x["request_id"] for x in second["requests"])
    assert len(set(calls)) == 4


def test_inventory_invalid_prefix_fails_before_any_submission(monkeypatch, tmp_path):
    module = load("scripts/ops/recover_framework.py", "inventory_bad_prefix")
    calls = []
    monkeypatch.setattr(module, "fleet_inventory_request", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="invalid"):
        module.fleet_inventory_batch(tmp_path, tmp_path / "batch", ["192.0.2.10"],
                                     request_prefix="x" * 64)
    assert calls == []


def test_asset_source_does_not_require_control_host_in_device_targets(tmp_path):
    module = load("scripts/ops/registered_fleet_bridge.py", "control_asset_source")
    broker = object.__new__(module.Broker)
    broker.control_host = "cpu-control"
    broker.control_container = "personal-control"
    broker.control = object()
    broker.project = "/personal/project"
    broker.task = "batch"
    broker.local = tmp_path
    broker.targets = {"192.0.2.10": {
        "project": "/personal/target", "management": "/personal/tools",
        "container": "personal-target", "protected_devices": [],
    }}
    client = SimpleNamespace(close=lambda: None)
    broker.helper = SimpleNamespace(connect=lambda host: client)
    commands = []

    def call(connection, command, body=None, timeout=45):
        assert connection is broker.control
        commands.append(command)
        return json.dumps({"root": "/personal/project/assets/weights/pangu-7b", "files": []})

    broker.call = call
    with pytest.raises(ValueError, match="no real model weight shards"):
        broker.execute({"request_id": "copy-seven", "action": "copy_asset", "host": "192.0.2.10",
                        "job": "copy-seven", "asset": "pangu-7b"}, None)
    assert "personal-control" in commands[0]
    assert "personal-target" not in commands[0]


def test_asset_control_container_rejects_invalid_name():
    module = load("scripts/ops/registered_fleet_bridge.py", "invalid_control_container")
    broker = object.__new__(module.Broker)
    broker.control_container = "other; command"
    with pytest.raises(ValueError, match="invalid registered control container"):
        broker.asset_source_container()


def test_fleet_connects_with_registered_alias_before_remote_action(tmp_path):
    module = load("scripts/ops/registered_fleet_bridge.py", "registered_target_alias")
    broker = object.__new__(module.Broker)
    broker.targets = {"192.0.2.10": {
        "project": "/personal/target", "management": "/personal/tools",
        "container": "personal-target", "protected_devices": [],
        "ssh_alias": "registered-target",
    }}
    seen = []
    def connect(alias):
        seen.append(alias)
        raise RuntimeError("stop before remote operation")
    broker.helper = SimpleNamespace(connect=connect)
    with pytest.raises(RuntimeError, match="stop before"):
        broker.execute({"request_id": "fresh-inventory", "action": "inventory",
                        "host": "192.0.2.10"}, None)
    assert seen == ["registered-target"]


@pytest.mark.parametrize(('action', 'response', 'execution'), [
    ('run', {'submitted': True}, 'submitted_not_validated'),
    ('status', {'state.json': '{"status":"running"}'}, 'running'),
    ('status', {'state.json': '{"status":"exited","exit":0}'}, 'succeeded'),
    ('status', {'state.json': '{"status":"exited","exit":1}'}, 'failed'),
    ('status', {'exit.txt': '1\n'}, 'failed'),
    ('status', {'state.json': '{"status":"exited","exit":0}',
                'failure.json': '{"error":"RuntimeError"}'}, 'failed'),
    ('status', {'state.json': {'truncated': True}}, 'unknown'),
    ('status', {'state.json': '{broken'}, 'unknown'),
    ('status', {'error': 'SSHException'}, 'broker_failed'),
])
def test_job_outcome_separates_queue_execution_and_acceptance(action, response, execution):
    module = load("scripts/ops/recover_framework.py", "workflow_job_outcome")
    outcome = module.fleet_job_outcome(action, response)
    assert outcome['execution'] == execution
    assert outcome['business_acceptance'] == 'NOT_VALIDATED'


def test_operator_update_is_optional_bounded_task_local_json(tmp_path):
    module = load("scripts/ops/recover_framework.py", "task_operator_update")
    assert module.operator_update(tmp_path) is None
    repair = tmp_path / 'repair'
    repair.mkdir()
    note = repair / 'OPERATOR_UPDATE.json'
    note.write_text(json.dumps({'id': 'update-1', 'instructions': 'Read fresh inventory',
                                'paths': ['repair/MAIN_GUIDANCE.md']}))
    assert module.operator_update(tmp_path)['id'] == 'update-1'
    note.write_text('x' * 8193)
    with pytest.raises(ValueError, match='8192'):
        module.operator_update(tmp_path)
    note.write_text('{"id":"update-1","instructions":"note","unexpected":1}')
    with pytest.raises(ValueError, match='invalid operator'):
        module.operator_update(tmp_path)


def test_job_status_requests_actual_status_not_run_receipt(tmp_path, monkeypatch):
    module = load('scripts/ops/recover_framework.py', 'actual_job_status')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    (task / 'repair').mkdir()
    (task / 'repair/OPERATOR_UPDATE.json').write_text(json.dumps({
        'id': 'follow-case', 'instructions': 'Continue the real model smoke',
    }))
    submit = module.fleet_submit
    def respond(*args, **kwargs):
        result = submit(*args, **kwargs)
        response = task / 'fleet/responses' / (result['request_id'] + '.json')
        response.parent.mkdir(parents=True, exist_ok=True)
        response.write_text(json.dumps({'state.json': '{"status":"exited","exit":0}',
                                        'exit.txt': '0\n'}))
        return result
    monkeypatch.setattr(module, 'fleet_submit', respond)
    result = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert result['action'] == 'status'
    assert result['job_outcome']['execution'] == 'succeeded'
    assert result['job_outcome']['business_acceptance'] == 'NOT_VALIDATED'
    assert 'never a run receipt' in result['observation']
    assert result['job_outcome']['operator_update']['id'] == 'follow-case'


def test_job_status_reuses_pending_observation_request(tmp_path):
    module = load('scripts/ops/recover_framework.py', 'pending_job_status')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    module.fleet_submit(tmp_path, task, 'pending-status', 'status', '192.0.2.10', 'job-01')
    first = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    second = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert first['request_id'] == second['request_id'] == 'pending-status'
    assert first['status'] == 'pending'
    assert len(list((task / 'fleet/requests').glob('*.json'))) == 1


def test_pending_status_contract_has_dict_response_and_absolute_response_path(tmp_path):
    module = load('scripts/ops/recover_framework.py', 'pending_status_contract')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    module.fleet_submit(tmp_path, task, 'pending-contract', 'status', '192.0.2.10', 'job-01')
    result = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert result['status'] == 'pending'
    assert result['response'] == {}
    assert Path(result['response_path']).is_absolute()
    assert result['job_outcome'] == {'execution': 'unknown', 'business_acceptance': 'NOT_VALIDATED'}
    assert 'not a job-queued result' in result['observation']


def _terminal_status_response(exit_code=0, checked_at=1.0):
    return {
        'state.json': json.dumps({'status': 'exited', 'exit': exit_code}),
        'exit.txt': f'{exit_code}\n',
        'file_metadata': {'state.json': {'json_valid': True, 'truncated': False}},
        'broker_checked_at': checked_at,
    }


def test_terminal_status_is_cached_and_latest_response_is_reused(tmp_path, monkeypatch):
    module = load('scripts/ops/recover_framework.py', 'terminal_status_cache')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    repair = task / 'repair'
    repair.mkdir()
    note = repair / 'OPERATOR_UPDATE.json'
    note.write_text(json.dumps({'id': 'update-before', 'instructions': 'first'}))
    module.fleet_submit(tmp_path, task, 'terminal-old', 'status', '192.0.2.10', 'job-01')
    module.fleet_submit(tmp_path, task, 'terminal-new', 'status', '192.0.2.10', 'job-01')
    responses = task / 'fleet/responses'
    responses.mkdir(parents=True, exist_ok=True)
    old_response = responses / 'terminal-old.json'
    new_response = responses / 'terminal-new.json'
    old_response.write_text(json.dumps(_terminal_status_response(0, 10.0)))
    new_response.write_text(json.dumps(_terminal_status_response(0, 20.0)))
    os.utime(old_response, (100, 100))
    os.utime(new_response, (200, 200))

    submitted = []
    original_submit = module.fleet_submit
    def unexpected_query(*args, **kwargs):
        submitted.append(args[2])
        return original_submit(*args, **kwargs)
    monkeypatch.setattr(module, 'fleet_submit', unexpected_query)

    first = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    note.write_text(json.dumps({'id': 'update-after', 'instructions': 'refresh'}))
    second = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert submitted == []
    assert first['request_id'] == second['request_id'] == 'terminal-new'
    assert first['observation'] == 'cached_terminal_job_result; not a fresh device-idle proof'
    assert first['snapshot_at'] == second['snapshot_at'] == 20.0
    assert first['job_outcome']['execution'] == 'succeeded'
    assert first['operator_update']['id'] == 'update-before'
    assert second['operator_update']['id'] == 'update-after'
    assert second['job_outcome']['operator_update']['id'] == 'update-after'


@pytest.mark.parametrize('response', [
    {'state.json': json.dumps({'status': 'running'}),
     'file_metadata': {'state.json': {'json_valid': True, 'truncated': False}}},
    {'state.json': json.dumps({'status': 'exited', 'exit': 0})},
    {'error': 'SSHException'},
])
def test_nonterminal_or_old_status_response_is_refreshed(tmp_path, monkeypatch, response):
    module = load('scripts/ops/recover_framework.py', 'refresh_status_cache')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    module.fleet_submit(tmp_path, task, 'stale-status', 'status', '192.0.2.10', 'job-01')
    response_path = task / 'fleet/responses/stale-status.json'
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response))

    submitted = []
    original_submit = module.fleet_submit
    def refresh_query(*args, **kwargs):
        submitted.append(args[2])
        return original_submit(*args, **kwargs)
    monkeypatch.setattr(module, 'fleet_submit', refresh_query)
    result = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert len(submitted) == 1
    assert result['status'] == 'pending'
    assert result['response'] == {}
    assert result['request_id'] == submitted[0]


def test_oversized_terminal_response_is_not_read_or_cached(tmp_path, monkeypatch):
    module = load('scripts/ops/recover_framework.py', 'oversized_terminal_response')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    module.fleet_submit(tmp_path, task, 'oversized-status', 'status', '192.0.2.10', 'job-01')
    response_path = task / 'fleet/responses/oversized-status.json'
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text('x' * (1_000_001))

    original_read = module.read
    read_response = []
    def guarded_read(path):
        if Path(path) == response_path:
            read_response.append(str(path))
            raise AssertionError('oversized response was read')
        return original_read(path)
    monkeypatch.setattr(module, 'read', guarded_read)
    submitted = []
    original_submit = module.fleet_submit
    def refresh_query(*args, **kwargs):
        submitted.append(args[2])
        return original_submit(*args, **kwargs)
    monkeypatch.setattr(module, 'fleet_submit', refresh_query)

    result = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert read_response == []
    assert len(submitted) == 1
    assert result['status'] == 'pending'
    assert result['response'] == {}


def test_symlink_terminal_response_is_not_read_or_cached(tmp_path, monkeypatch):
    module = load('scripts/ops/recover_framework.py', 'symlink_terminal_response')
    task = tmp_path / 'runs/batch'
    task.mkdir(parents=True)
    module.fleet_submit(tmp_path, task, 'symlink-status', 'status', '192.0.2.10', 'job-01')
    responses = task / 'fleet/responses'
    responses.mkdir(parents=True, exist_ok=True)
    target = responses / 'terminal-target.json'
    target.write_text(json.dumps(_terminal_status_response()))
    response_path = responses / 'symlink-status.json'
    try:
        response_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation is unavailable on this platform')

    original_read = module.read
    read_response = []
    def guarded_read(path):
        if Path(path) == response_path:
            read_response.append(str(path))
            raise AssertionError('symlink response was read')
        return original_read(path)
    monkeypatch.setattr(module, 'read', guarded_read)
    submitted = []
    original_submit = module.fleet_submit
    def refresh_query(*args, **kwargs):
        submitted.append(args[2])
        return original_submit(*args, **kwargs)
    monkeypatch.setattr(module, 'fleet_submit', refresh_query)

    result = module.fleet_job_status(tmp_path, task, '192.0.2.10', 'job-01', 0)
    assert read_response == []
    assert len(submitted) == 1
    assert result['status'] == 'pending'
    assert result['response'] == {}


def test_cli_request_status_follows_run_receipt_to_real_job(tmp_path, monkeypatch, published_source_fixture):
    module = load('scripts/ops/recover_framework.py', 'follow_run_status')
    task = tmp_path / 'runs/batch'
    repair = task / 'repair'
    repair.mkdir(parents=True)
    (repair / 'cpu.py').write_text('print(1)')
    published_source_fixture(tmp_path, task)
    module.fleet_submit(tmp_path, task, 'run-01', 'run', '192.0.2.10', 'cpu-01',
                        'repair/cpu.py', devices=[])
    response = task / 'fleet/responses/run-01.json'
    response.parent.mkdir(parents=True)
    response.write_text('{"submitted":true,"acceptance":"submitted_not_validated"}')
    called = []
    def actual_status(root, current, host, job, wait):
        called.append((host, job))
        return {'action': 'status', 'job_outcome': {'execution': 'succeeded',
                                                  'business_acceptance': 'NOT_VALIDATED'}}
    monkeypatch.setattr(module, 'fleet_job_status', actual_status)
    result = module.fleet_request_status(tmp_path, task, 'run-01', 0)
    assert called == [('192.0.2.10', 'cpu-01')]
    assert result['job_outcome']['execution'] == 'succeeded'
    assert result['run_request_id'] == 'run-01'
    assert result['submission_response_path'] == str(response)


def test_cli_request_status_does_not_query_nonexistent_job_after_submit_failure(tmp_path, monkeypatch):
    module = load('scripts/ops/recover_framework.py', 'failed_submission_status')
    task = tmp_path / 'runs/batch'
    req = task / 'fleet/requests/run-01.json'
    req.parent.mkdir(parents=True)
    req.write_text(json.dumps({'action': 'run', 'host': '192.0.2.10', 'job': 'cpu-01'}))
    response = task / 'fleet/responses/run-01.json'
    response.parent.mkdir(parents=True)
    response.write_text('{"error":"SSHException"}')
    monkeypatch.setattr(module, 'fleet_job_status', lambda *_: pytest.fail('job was never submitted'))
    result = module.fleet_request_status(tmp_path, task, 'run-01', 0)
    assert result['job_outcome']['execution'] == 'broker_failed'


def test_runner_rejects_fragmented_json_instead_of_accepting_exit_zero(tmp_path):
    bridge = load('scripts/ops/registered_fleet_bridge.py', 'structured_output_gate')
    namespace = {}
    exec(bridge.RUNNER.split('j=pathlib.Path', 1)[0], namespace)
    (tmp_path / 'result.json').write_text('{"sample":')
    with pytest.raises(RuntimeError, match='one complete JSON'):
        namespace['validate_structured_outputs'](tmp_path)
    (tmp_path / 'result.json').write_text('{"sample":1}')
    assert namespace['validate_structured_outputs'](tmp_path) == ['result.json']


def test_historical_exit_zero_with_invalid_artifact_is_reported_separately():
    module = load('scripts/ops/recover_framework.py', 'historical_invalid_artifact')
    outcome = module.fleet_job_outcome('status', {
        'state.json': '{"status":"exited","exit":0}',
        'file_metadata': {'result.json': {'json_valid': False, 'json_error': 'unterminated string'}},
    })
    assert outcome['execution'] == 'succeeded'
    assert outcome['artifact_validation'] == 'failed'
    assert outcome['business_acceptance'] == 'NOT_VALIDATED'


def test_resource_gate_failure_is_not_reported_as_model_execution_failure():
    module = load('scripts/ops/recover_framework.py', 'resource_gate_outcome')
    result = module.fleet_job_outcome('status', {
        'state.json': '{"status":"exited","exit":1}',
        'failure.json': json.dumps({'error': 'RuntimeError',
                                    'detail': 'device 1 is not confirmed healthy and empty'}),
    })
    assert result['failure_layer'] == 'device_resource_gate'
    assert result['workload_execution'] == 'not_started'
    assert 'another registered host' in result['next_action']


def test_device_lease_busy_without_process_is_resource_gate_not_model_failure():
    module = load('scripts/ops/recover_framework.py', 'device_lease_busy_outcome')
    result = module.fleet_job_outcome('status', {
        'state.json': '{"status":"exited","exit":1}',
        'failure.json': json.dumps({'error': 'RuntimeError',
                                    'detail': 'device lease busy: runtime device 3 is held by another task'}),
    })
    assert result['failure_layer'] == 'device_resource_gate'
    assert result['workload_execution'] == 'not_started'
    assert result['reason'] == 'device_lease_busy'
    assert 'lease holder' in result['next_action']
    assert 'do not delete lease/lock files' in result['next_action']


def test_framework_staging_keeps_yaml_manifests_without_machine_local_configuration():
    module = load('scripts/ops/registered_fleet_bridge.py', 'yaml_model_staging')
    assert module.framework_source_file_allowed('model.yaml', 1024)
    assert module.framework_source_file_allowed('a2.example.yaml', 1024)
    assert module.framework_source_file_allowed('config.yml', 1024)
    assert not module.framework_source_file_allowed('a2.local.yaml', 1024)
    assert not module.framework_source_file_allowed('credentials.env', 1024)
    assert not module.framework_source_file_allowed('weights.safetensors', 1024)
    assert not module.framework_source_file_allowed('oversized.yaml', 2_000_001)
