"""CPU-only CLI fixtures: no SSH, devices or active project runs."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ops/recover_framework.py'
HOST = '192.0.2.10'
STOP = {'id': 'stop-now', 'instructions': 'STOP. Do not submit or resume. ' + '保留停止指令。' * 300,
        'paths': ['repair/STOP.json']}
HARNESS = r'''
import codecs,importlib.util,json,os,sys
from pathlib import Path
assert codecs.lookup(sys.stdout.encoding).name == codecs.lookup(os.environ['PYTHONIOENCODING']).name
script=Path(sys.argv[1]);root=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('compact_cli_fixture',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
response=json.loads((root/'fixture.json').read_text(encoding='utf-8'))
original=module.fleet_submit
def submit(*args,**kwargs):
    receipt=original(*args,**kwargs)
    if response is not None:
        path=Path(receipt['response_path']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(response if isinstance(response,str) else json.dumps(response),encoding='utf-8')
    return receipt
module.fleet_submit=submit
sys.argv=[str(script),'--project',str(root),'--task','fixture',*sys.argv[3:]]
module.main()
'''


def module():
    spec = importlib.util.spec_from_file_location('compact_status_test', SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def response(status='exited', exit_code=0, **extra):
    return {'state.json': json.dumps({'status': status, 'exit': exit_code}),
            'file_metadata': {'state.json': {'json_valid': True, 'truncated': False}},
            'workload.log': 'LARGE_RAW_LOG_MARKER' + 'x' * 300000, **extra}


@pytest.fixture
def cli(tmp_path):
    root = tmp_path / 'project'
    script = root / 'scripts/ops/recover_framework.py'
    script.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE, script)
    repair = root / 'runs/fixture/repair'
    repair.mkdir(parents=True)
    (repair / 'OPERATOR_UPDATE.json').write_text(json.dumps(STOP, ensure_ascii=False), encoding='utf-8')

    def invoke(raw, *args, child_encoding='utf-8', return_bytes=False):
        (root / 'fixture.json').write_text(json.dumps(raw), encoding='utf-8')
        child_env = dict(os.environ, PYTHONIOENCODING=child_encoding, PYTHONUTF8='0')
        result = subprocess.run([sys.executable, '-B', '-c', HARNESS, str(script), str(root), *args],
                                cwd=root, capture_output=True, env=child_env)
        if not return_bytes:
            result.stdout = result.stdout.decode('utf-8')
            result.stderr = result.stderr.decode('utf-8')
        return result, root
    return invoke


def status_args(compact=True):
    args = ['--action', 'fleet-job-status', '--host', HOST, '--job', 'job-01', '--wait-seconds', '0']
    return args + (['--compact'] if compact else [])


@pytest.mark.parametrize(('raw', 'execution', 'layer'), [
    (response('running', None), 'running', None),
    (response(exit_code=1, **{'failure.json': json.dumps({'error': 'RuntimeError', 'detail': 'actual failure'})}), 'failed', None),
    (response(exit_code=1, **{'failure.json': json.dumps({'error': 'RuntimeError', 'detail': 'device 1 is not confirmed healthy and empty'})}), 'failed', 'device_resource_gate'),
    (response(exit_code=1, **{'failure.json': json.dumps({'error': 'RuntimeError', 'detail': 'device lease busy: device 1 held'})}), 'failed', 'device_resource_gate'),
    (response(), 'succeeded', None),
    ({'exit.txt': '0\n'}, 'unknown', None),
    ({'submitted': True}, 'unknown', None),
    (None, 'unknown', None),
    ({'error': 'SSHException'}, 'broker_failed', None),
    ({'state.json': {'truncated': True, 'original_bytes': 400000}, 'exit.txt': '0\n'}, 'unknown', None),
    ({'state.json': '{broken', 'file_metadata': {'state.json': {'json_valid': False, 'json_error': 'invalid JSON'}}}, 'unknown', None),
])
def test_compact_cli_preserves_execution_and_stop_without_full_response(cli, raw, execution, layer):
    completed, root = cli(raw, *status_args())
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output['job_outcome']['execution'] == execution
    assert output['job_outcome'].get('failure_layer') == layer
    assert output['job_outcome']['business_acceptance'] == 'NOT_VALIDATED'
    assert output['operator_update'] == STOP
    assert output['job_outcome']['operator_update'] == STOP
    assert output['host'] == HOST and output['job'] == 'job-01'
    assert output['request_id']
    assert Path(output['response_path']).is_absolute()
    assert 'response' not in output and 'LARGE_RAW_LOG_MARKER' not in completed.stdout
    assert len(completed.stdout) < 20000
    if raw is not None:
        assert json.loads(Path(output['response_path']).read_text(encoding='utf-8')) == raw
    else:
        assert output['status'] == 'pending'
        assert not Path(output['response_path']).exists()


@pytest.mark.parametrize('kind', ['invalid_result', 'truncated_result', 'truncated_state_metadata', 'missing_state_metadata', 'advertised_missing_file'])
def test_exit_zero_never_masks_incomplete_compact_evidence(cli, kind):
    raw = response()
    if kind == 'invalid_result':
        raw['result.json'] = '{bad'
        raw['file_metadata']['result.json'] = {'json_valid': False, 'json_error': 'unfinished result'}
    elif kind == 'truncated_result':
        raw['result.json'] = {'truncated': True, 'path': '/job/result.json'}
    elif kind == 'truncated_state_metadata':
        raw['file_metadata']['state.json']['truncated'] = True
    elif kind == 'missing_state_metadata':
        raw['file_metadata'] = {}
    else:
        raw['file_metadata']['process.json'] = {'json_valid': True}
    completed, _ = cli(raw, *status_args())
    assert completed.returncode == 0, completed.stderr
    outcome = json.loads(completed.stdout)['job_outcome']
    assert outcome['execution'] == 'unknown'
    assert outcome['evidence_status'].startswith('incomplete_or_invalid')
    if kind == 'invalid_result':
        assert outcome['artifact_validation'] == 'failed'
        assert outcome['invalid_json_files']['result.json'] == 'unfinished result'


@pytest.mark.parametrize('error_key', ['first_error', 'raw_error'])
def test_recorded_model_error_is_preserved_with_source_and_raw_file_pointer(cli, error_key):
    raw = response(exit_code=1, **{
        'failure.json': json.dumps({'error': 'RuntimeError', 'detail': 'wrapper exit 1'}),
        'result.json': json.dumps({'failure_layer': 'model_import', error_key: {'type': 'ImportError', 'detail': 'missing actual model symbol',
                                               'traceback': 'large traceback' * 15000}}),
    })
    completed, _ = cli(raw, *status_args())
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    first = output['job_outcome']['first_error']
    assert output['job_outcome']['failure_layer'] == 'model_import'
    assert first['source_file'] == 'result.json'
    assert first['type'] == 'ImportError' and first['detail'] == 'missing actual model symbol'
    assert first['read_path'] == output['response_path']
    assert 'large traceback' not in completed.stdout


def test_normal_result_detail_and_expected_log_tail_do_not_invent_failure(cli):
    raw = response(**{'result.json': json.dumps({'status': 'completed', 'detail': 'normal result note'})})
    raw['file_metadata']['workload.log'] = {'truncated': True, 'snippet_start_byte': 500000}
    completed, _ = cli(raw, *status_args())
    assert completed.returncode == 0, completed.stderr
    outcome = json.loads(completed.stdout)['job_outcome']
    assert outcome['execution'] == 'succeeded'
    assert 'first_error' not in outcome


def test_default_cli_keeps_complete_original_response_and_stop(cli):
    raw = response(**{'result.json': '{bad', 'file_metadata': {
        'state.json': {'json_valid': True}, 'result.json': {'json_valid': False, 'json_error': 'old artifact invalid'}}})
    completed, _ = cli(raw, *status_args(compact=False))
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output['response'] == raw
    assert output['operator_update'] == STOP
    assert output['job_outcome']['execution'] == 'succeeded'  # Legacy behavior is unchanged.
    assert output['job_outcome']['artifact_validation'] == 'failed'
    assert 'LARGE_RAW_LOG_MARKER' in completed.stdout
    assert list(output)[-1] == 'response'


@pytest.mark.parametrize('compact', [False, True])
def test_cli_forces_utf8_json_from_non_utf8_child_stdio(cli, compact):
    raw = response()
    completed, _ = cli(raw, *status_args(compact=compact),
                       child_encoding='cp936', return_bytes=True)
    assert completed.returncode == 0, completed.stderr
    assert '保留停止指令。'.encode('utf-8') in completed.stdout
    output = json.loads(completed.stdout.decode('utf-8'))
    assert output['operator_update'] == STOP
    assert output['job_outcome']['execution'] == 'succeeded'
    if compact:
        assert 'response' not in output
    else:
        assert output['response'] == raw


def test_compact_renderer_does_not_modify_original_and_marks_long_error():
    tool = module()
    raw = response(exit_code=1, **{'failure.json': json.dumps({'error': 'RuntimeError', 'detail': 'e' * 5000})})
    full = {'response': raw, 'job_outcome': tool.fleet_job_outcome('status', raw),
            'response_path': '/project/response.json', 'operator_update': STOP, 'status': 'received'}
    original = copy.deepcopy(full)
    compact = tool.compact_job_status(full, HOST)
    assert full == original
    assert compact['job_outcome']['first_error']['truncated_fields'] == ['detail']
    assert len(compact['job_outcome']['first_error']['detail']) == 2048


@pytest.mark.parametrize(('args', 'message', 'code'), [
    (['--action', 'fleet-status', '--request-id', 'status-01', '--compact'], '--compact is only supported', 2),
    (['--action', 'fleet-job-status', '--job', 'job-01', '--compact'], 'requires --host and --job', 1),
    (['--action', 'fleet-job-status', '--host', HOST, '--job', '../escape', '--compact'], 'invalid request or job identifier', 1),
])
def test_cli_errors_remain_nonzero_and_do_not_queue(cli, args, message, code):
    completed, root = cli(None, *args)
    assert completed.returncode == code
    assert message in completed.stderr
    assert not list((root/'runs/fixture').glob('fleet/requests/*.json'))


def test_invalid_outer_response_json_is_cli_failure_not_compact_success(cli):
    completed, _ = cli('{broken outer response', *status_args())
    assert completed.returncode != 0
    assert 'JSONDecodeError' in completed.stderr
    assert not completed.stdout.strip()
