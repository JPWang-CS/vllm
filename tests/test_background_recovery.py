import json
from pathlib import Path
import pytest
from scripts.ops import deployment_matrix as matrix, recover_framework as recover
from scripts.ops import registered_fleet_bridge as bridge


def test_broker_reconnect_preserves_controller_and_never_spawns_remote_ssh(tmp_path):
    task=tmp_path/'runs'/'case';task.mkdir(parents=True)
    state={'exit':1,'model':'glm-5.3-flash','child_alive':False,'fifo_removed':True,'credentials_persisted':False}
    (task/'STATE.json').write_text(json.dumps(state))
    value=recover.fleet_broker(tmp_path,task,'restart')
    assert value['controller_state_unchanged'] is True
    assert json.loads((task/'STATE.json').read_text())==state
    assert json.loads((task/'fleet/broker-control.json').read_text())['action']=='reconnect'


def test_matrix_revision_rejects_concurrent_stale_refresh(tmp_path):
    matrix.create_matrix(tmp_path,'case')
    first,path=matrix.load_matrix(tmp_path,'case')
    stale,_=matrix.load_matrix(tmp_path,'case')
    first['cases'][0]['status']='submitted'
    matrix.save_matrix(path,first)
    with pytest.raises(RuntimeError,match='changed during refresh'):
        matrix.save_matrix(path,stale)
    assert json.loads(path.read_text())['cases'][0]['status']=='submitted'
    # Released OS lock is usable even while the diagnostic lock file remains.
    current,_=matrix.load_matrix(tmp_path,'case')
    matrix.save_matrix(path,current)


def test_broker_ignores_nonterminal_control_message():
    assert bridge.terminal_controller_state({'reason':'restart','at':1}) is False
    assert bridge.terminal_controller_state({'exit':True}) is False
    assert bridge.terminal_controller_state({'exit':1,'child_alive':True}) is False
    assert bridge.terminal_controller_state({'exit':1,'model':'glm-5.3-flash','child_alive':False,'fifo_removed':True,'credentials_persisted':False}) is True


def test_matrix_refresh_records_failed_business_even_when_controller_exit_zero(tmp_path,monkeypatch):
    data=matrix.create_matrix(tmp_path,'case');entry=data['cases'][1]
    entry.update(latest_task='case-01',status='submitted')
    matrix.save_matrix(tmp_path/'runs/case/deployment-matrix.json',data)
    monkeypatch.setattr(recover,'snapshot',lambda *args:{'cases':[{'task':'case-01','job-state.json':{'status':'failed','exit':1}}]})
    refreshed=matrix.refresh_matrix(tmp_path,'case')
    assert refreshed['cases'][1]['status']=='failed'
    assert refreshed['counts']['failed']==1
