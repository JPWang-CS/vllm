"""The hourly entry must use live remote evidence and the public resume gate."""
import importlib.util
import json
from pathlib import Path
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ops/recover_framework.py'


def module():
    spec = importlib.util.spec_from_file_location('hourly_recovery', SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def setup(tmp_path):
    root = tmp_path / 'vllm'
    task = root / 'runs/job'
    (task / 'repair').mkdir(parents=True)
    prompt = task / 'original.txt'
    prompt.write_text('Original four-mode task. Do not repeat accepted runs.')
    launch = {'key': 'job-r01', 'remote_run': '/personal/projects/vllm/runs/job/controllers/host/glm01'}
    (task / 'GLM_LAUNCH.json').write_text(json.dumps(launch))
    plan = task / 'glm-attempts/job-r01/PLAN.json'
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({'host': 'registered-alias', 'remote_project': '/personal/projects/vllm',
                                'business_run': '/personal/projects/vllm/runs/job',
                                'prompt_file': str(prompt), 'argv': []}))
    return root, task


def public_state(state='MODEL_FAILED'):
    return {'state': state, 'failure': {'class': 'PROXY_AUTH_OR_ROUTE'}, 'remote_snapshot': {
        'terminal': {'exit': 1, 'model': 'glm-5.3-flash', 'child_alive': False,
                     'fifo_removed': True, 'credentials_persisted': False, 'cleanup_complete': True},
        'cleanup': {'model_waited': True, 'remaining_direct_children': []},
    }}


def fake_calls(task, status):
    seen = []

    def call(argv, **kwargs):
        seen.append(argv)
        assert 'docker' not in argv
        if '--status' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(status), '')
        assert '--resume' in argv
        assert argv[argv.index('--remote-project') + 1] == '/personal/projects/vllm'
        (task / 'GLM_LAUNCH.json').write_text(json.dumps({'key': 'job-r02'}))
        return subprocess.CompletedProcess(argv, 0, '{"state":"DRIVER_STARTED_NOT_TOOL_VERIFIED"}', '')

    return seen, call


def test_running_controller_is_observed_without_starting_another(tmp_path):
    root, task = setup(tmp_path)
    status = public_state('TOOLS_RUNNING')
    status['remote_snapshot'] = {'terminal': None, 'cleanup': None}
    seen, call = fake_calls(task, status)
    out = module().supervise_once(root, task, tmp_path, allow_resume=True, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': True}])
    assert out['action'] == 'CONTROLLER_RUNNING'
    assert len(seen) == 1


def test_remote_business_keeps_running_without_duplicate_submission(tmp_path):
    root, task = setup(tmp_path)
    seen, call = fake_calls(task, public_state())
    out = module().supervise_once(root, task, tmp_path, allow_resume=True, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': True}])
    assert out['action'] == 'BUSINESS_RUNNING'
    assert len(seen) == 1


def test_actual_cleanup_and_remote_idle_allow_resume_despite_old_407(tmp_path):
    root, task = setup(tmp_path)
    (task / 'PUBLIC_NETWORK_BLOCKER.json').write_text('{"latest_error":"old 407"}')
    (task / 'CHECKPOINT.json').write_text('{"cases":{"mixed":{"status":"running"}}}')
    seen, call = fake_calls(task, public_state())
    published = []
    out = module().supervise_once(root, task, tmp_path, allow_resume=True, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': False}],
                                 publish=lambda *args: published.append(args[-1]))
    assert out['action'] == 'RESUME_SUBMITTED'
    assert len(seen) == 2 and len(published) == 1
    assert json.loads((task / 'repair/PROCESS_CLEANUP.json').read_text())['model_waited'] is True
    assert '不要照抄旧说明' in (task / 'repair/SUPERVISOR_RESUME_PROMPT.txt').read_text(encoding='utf-8')
    assert json.loads((task / 'HOURLY_SUPERVISOR.json').read_text())['action'] == 'RESUME_SUBMITTED'


def test_missing_real_cleanup_is_not_replaced_by_empty_defaults(tmp_path):
    root, task = setup(tmp_path)
    status = public_state()
    status['remote_snapshot']['cleanup'] = {}
    seen, call = fake_calls(task, status)
    out = module().supervise_once(root, task, tmp_path, allow_resume=True, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': False}])
    assert out['action'] == 'BLOCKED_CONTROLLER_NOT_STRICTLY_TERMINAL'
    assert len(seen) == 1


def test_unobserved_remote_blocks_a_resume_instead_of_calling_local_docker(tmp_path):
    root, task = setup(tmp_path)
    seen, call = fake_calls(task, public_state())
    out = module().supervise_once(root, task, tmp_path, allow_resume=True, call=call,
                                 observe=lambda *args: [{'observed': False, 'uncertain': True}])
    assert out['action'] == 'BLOCKED_REMOTE_OBSERVATION'
    assert len(seen) == 1


def test_observation_only_never_starts_a_model(tmp_path):
    root, task = setup(tmp_path)
    seen, call = fake_calls(task, public_state())
    out = module().supervise_once(root, task, tmp_path, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': False}])
    assert out['action'] == 'RESUME_AVAILABLE_OBSERVE_ONLY'
    assert len(seen) == 1


def test_normal_model_completion_is_not_recorded_as_api_failure(tmp_path):
    root, task = setup(tmp_path)
    status = public_state('MODEL_COMPLETED')
    status['failure'] = None
    status['remote_snapshot']['terminal']['exit'] = 0
    seen, call = fake_calls(task, status)
    out = module().supervise_once(root, task, tmp_path, call=call,
                                 observe=lambda *args: [{'observed': True, 'active': False}])
    assert out['action'] == 'RESUME_AVAILABLE_OBSERVE_ONLY'
    assert not (task / 'repair/CONTROLLER_FAILURE.json').exists()
    completion = json.loads((task / 'repair/CONTROLLER_COMPLETION.json').read_text())
    assert completion['exit'] == 0
    assert completion['business_acceptance'] == 'NOT_INFERRED_FROM_CONTROLLER'
    assert len(seen) == 1
