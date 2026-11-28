"""A busy source-update gate waits; identical active sources stay read-only."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('target_wait_bridge',ROOT/'scripts/ops/registered_fleet_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class FS:
    def __init__(self):
        self.writes = 0
    def stat(self,path):
        return Path(path).stat()
    def lstat(self,path):
        return Path(path).lstat()
    def listdir(self,path):
        return [p.name for p in Path(path).iterdir()]
    def open(self,path,mode='r'):
        if mode!='rb':
            self.writes += 1
        return Path(path).open('wb' if mode=='w' else mode)
    def posix_rename(self,source,target):
        Path(source).replace(target)
    def close(self):
        pass


def broker_fixture(tmp_path,execute):
    sf = FS()
    obj = object.__new__(bridge.Broker)
    obj.run = tmp_path/'run'
    obj.local = tmp_path/'receipts'
    obj.local.mkdir()
    obj.control = type('Control',(),{'open_sftp':lambda self:sf})()
    obj.helper = type('Helper',(),{'mkdirs':lambda self,fs,path,mode:Path(path).mkdir(parents=True,exist_ok=True)})()
    obj.execute = execute
    requests = obj.run/'fleet/requests'
    requests.mkdir(parents=True)
    return obj,requests


def request(directory,name,host='10.0.0.1'):
    (directory/(name+'.json')).write_text(json.dumps({'request_id':name,'action':'run','host':host,'job':name}))


def test_busy_request_keeps_same_request_without_final_failure_then_resumes(tmp_path):
    calls = []
    def execute(value,sf):
        calls.append(value['request_id'])
        if len(calls)==1:
            raise bridge.TargetProjectBusy('target project needs source update while active')
        return {'submitted':True,'job':value['job']}
    broker,requests = broker_fixture(tmp_path,execute)
    request(requests,'job1')
    response = broker.run/'fleet/responses/job1.json'
    waiting = broker.run/'fleet/waiting-job1.json'
    with patch.object(bridge.time,'time',return_value=2000):
        assert broker.poll()==0
    assert not response.exists()
    assert json.loads(waiting.read_text())['workload_started'] is False
    with patch.object(bridge.time,'time',return_value=2010):
        assert broker.poll()==0
    assert calls==['job1']
    with patch.object(bridge.time,'time',return_value=2031):
        assert broker.poll()==1
    assert calls==['job1','job1']
    assert json.loads(response.read_text())['submitted'] is True
    assert json.loads(waiting.read_text())['status']=='RELEASED'


def test_busy_host_does_not_block_other_queued_host(tmp_path):
    def execute(value,sf):
        if value['request_id']=='a':
            raise bridge.TargetProjectBusy('source update deferred')
        return {'submitted':True,'job':value['job']}
    broker,requests = broker_fixture(tmp_path,execute)
    request(requests,'a')
    request(requests,'b','10.0.0.2')
    assert broker.poll()==1
    assert not (broker.run/'fleet/responses/a.json').exists()
    assert json.loads((broker.run/'fleet/responses/b.json').read_text())['submitted'] is True


def test_unknown_runtime_error_remains_failure_and_is_not_retried(tmp_path):
    calls = []
    def execute(value,sf):
        calls.append(value['request_id'])
        raise RuntimeError('bad launch configuration')
    broker,requests = broker_fixture(tmp_path,execute)
    request(requests,'bad')
    assert broker.poll()==1
    result=json.loads((broker.run/'fleet/responses/bad.json').read_text())
    assert result['error']=='RuntimeError'
    assert not (broker.run/'fleet/waiting-bad.json').exists()
    assert broker.poll()==0
    assert calls==['bad']


def source_fixture(tmp_path,actual):
    source,target=tmp_path/'source',tmp_path/'target'
    for root in [source,target]:
        (root/'src').mkdir(parents=True)
    (source/'src/sample.py').write_bytes(b'published code\n')
    (target/'src/sample.py').write_bytes(actual)
    return source,target


def test_identical_active_target_is_verified_without_writes(tmp_path):
    source,target=source_fixture(tmp_path,b'published code\n')
    source_sf,target_sf=FS(),FS()
    result=bridge.compare_active_target_source(source_sf,str(source),target_sf,str(target),['src/sample.py'])
    assert result['matches'] is True and result['files_checked']==1
    assert result['files'][0]['write_performed'] is False
    assert source_sf.writes==target_sf.writes==0


def test_changed_active_target_is_preserved_and_not_accepted(tmp_path):
    source,target=source_fixture(tmp_path,b'active old code\n')
    sf=FS()
    result=bridge.compare_active_target_source(sf,str(source),sf,str(target),['src/sample.py'])
    assert result['matches'] is False
    assert result['mismatches']==[{'path':'src/sample.py','reason':'different content'}]
    assert (target/'src/sample.py').read_bytes()==b'active old code\n'
    assert sf.writes==0


def test_missing_active_source_remains_missing(tmp_path):
    source,target=source_fixture(tmp_path,b'old')
    (target/'src/sample.py').unlink()
    sf=FS()
    result=bridge.compare_active_target_source(sf,str(source),sf,str(target),['src/sample.py'])
    assert result['matches'] is False
    assert result['mismatches'][0]['reason']=='missing'
    assert not (target/'src/sample.py').exists() and sf.writes==0
