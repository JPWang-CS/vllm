"""CPU regressions for real-window KV completion, not HTTP/recompute success."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fullnet import measurements as m


@pytest.mark.parametrize('name',[
    'vllm:request_prefill_kv_computed_tokens_sum',
    'vllm:request_prefill_kv_computed_tokens_bucket',
    'omni_npu:worker_mem_reserved_bytes',
    'omni_npu:kv_transfer_failures_total{kind="pull"}',
    'omni_npu:kv_pull_submitted_total',
])
def test_noncompletion_counters_are_not_transfer(name):
    assert not m._has_positive_kv_metric({name:2})


@pytest.mark.parametrize('line',[
    'INFO vllm.kv_transfer.kv_connector="LLMDataDistConnector"',
    'INFO omni connector prom metrics registered: kv_transfer_failures + worker_mem_bytes',
    'INFO factory Creating v1 connector LLMDataDistConnector',
    'INFO pull_done req_id=r1',  # actual runtime emits this even on failure
    'INFO pull_block submitted req_id=r1',
])
def test_config_registration_or_submission_is_not_completion(line):
    assert not m._has_kv_transfer_log([line])


@pytest.mark.parametrize('mode',[
    'success','historical_failure','failure_counter','new_failure_counter',
    'fail_pull','link_failed','fallback','registration_only','counter_reset',
])
def test_probe_uses_window_and_failure_veto_even_with_positive_metrics(tmp_path,monkeypatch,mode):
    run=tmp_path/'models/pangu-7b/experiments/probe';logs=run/'logs'
    logs.mkdir(parents=True)
    for role in ('producer','consumer'):
        (logs/f'pd-split-{role}.log').write_text(
            'ERROR link failed: LLMStatusCode.LLM_LINK_FAILED\n'
            'WARNING Recovered from KV load failure: rescheduled\n',encoding='utf-8')
    monkeypatch.setattr(m,'_bench_environment',lambda *a:{})
    phases=[];monkeypatch.setattr(m,'state',lambda *a,**k:phases.append((a[1:],k)))
    def prepare(argv,**kwargs):
        Path(argv[argv.index('--output')+1]).write_text(json.dumps({'prompt':'test'})+'\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',prepare)
    failure='omni_npu:kv_transfer_failures_total{kind="pull"}'
    before={role:f'omni_npu:kv_transfer_completed_total 5\n{failure} 3\n' for role in ('producer','consumer')}
    after={role:f'omni_npu:kv_transfer_completed_total 7\n{failure} 3\n' for role in ('producer','consumer')}
    if mode=='failure_counter':after['consumer']=after['consumer'].replace(f'{failure} 3',f'{failure} 5')
    if mode=='new_failure_counter':
        before['consumer']='omni_npu:kv_transfer_completed_total 5\n'
        after['consumer']='omni_npu:kv_transfer_completed_total 7\n'+failure+' 2\n'
    if mode=='counter_reset':after['consumer']=after['consumer'].replace('completed_total 7','completed_total 1')
    if mode=='registration_only':
        before={r:'vllm:request_prefill_kv_computed_tokens_sum 0\n' for r in before}
        after={r:'vllm:request_prefill_kv_computed_tokens_sum 4224\n' for r in after}
    metric_calls=0
    class Response:
        status=200
        def __init__(self,value):self.value=value
        def read(self):return json.dumps(self.value).encode()
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def urlopen(request,**kwargs):
        nonlocal metric_calls
        if isinstance(request,str):
            metric_calls+=1
            return Response(before if metric_calls==1 else after)
        body=json.loads(request.data)
        for role in ('producer','consumer'):
            with (logs/f'pd-split-{role}.log').open('a',encoding='utf-8') as f:
                f.write('INFO vllm.kv_transfer.kv_connector="LLMDataDistConnector"\n')
                if role=='consumer':
                    f.write({'fail_pull':'INFO KV Transfer metrics: fail_pull=1\n',
                             'link_failed':'ERROR link failed: LLMStatusCode.LLM_LINK_FAILED\n',
                             'fallback':'WARNING Recovered from KV load failure: request rescheduled\n'}.get(mode,''))
        return Response({'usage':{'prompt_tokens':4096,'completion_tokens':body['max_tokens']}})
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    config={'service':{'argv':['selected-python'],'base_url':'http://localhost:1/v1','served_model':'pangu'},
            'evaluation':{},'identity':{'weights':'readonly-weights'}}
    accepted=mode in ('success','historical_failure')
    if accepted:
        assert m.pd_split_probe(run,config)['acceptance']=='passed'
    else:
        with pytest.raises(RuntimeError,match='pd split probe failed'):
            m.pd_split_probe(run,config)
    result=json.loads((run/'performance/pd-split-probe.json').read_text())
    assert result['acceptance']==('passed' if accepted else 'failed')
    assert all(w['offset_start']>0 for w in result['connector_log_windows'].values())
    assert phases[-1][0]==('pd_split_probe','passed' if accepted else 'failed')
    if mode=='registration_only':assert any('no kv-transfer activity' in s for s in result['reasons'])
    if mode in ('failure_counter','new_failure_counter'):assert any('failure counter increased' in s for s in result['reasons'])


def test_actual08_minimal_counter_and_log_structure_is_rejected():
    deltas={'producer':{'vllm:request_prefill_kv_computed_tokens_sum':4224},
            'consumer':{'vllm:request_prefill_kv_computed_tokens_sum':4224,
                        'omni_npu:kv_transfer_failures_total{kind="pull"}':2}}
    logs=['INFO vllm.kv_transfer.kv_connector="LLMDataDistConnector"',
          'INFO omni connector prom metrics registered: kv_transfer_failures + worker_mem_bytes']
    windows={r:{'activity':logs,'failure_count':0,'failure_lines':[]} for r in deltas}
    assert not any(m._has_positive_kv_metric(v) for v in deltas.values())
    assert not m._has_kv_transfer_log(logs)
    assert any('consumer KV failure counter increased' in s for s in m._pd_failure_reasons(deltas,windows))


def test_log_rotation_or_truncation_cannot_hide_failures(tmp_path):
    path=tmp_path/'consumer.log';path.write_bytes(b'history\n')
    info=path.stat();start={'identity':[info.st_dev,info.st_ino],'offset':info.st_size}
    path.write_bytes(b'')
    assert 'invalid_window' in m._pd_log_window(path,start)


def test_only_explicit_successful_completion_log_is_positive():
    assert m._has_kv_transfer_log(['INFO kv_pull_completed success=true blocks=32 req_id=r1'])
    assert not m._has_kv_transfer_log(['INFO kv_pull_completed success=false blocks=32 req_id=r1'])


def test_gauges_prefill_histograms_and_memory_drops_are_not_counter_resets():
    before='omni_npu:kv_worker_memory_bytes 100\nvllm:kv_cache_usage_perc 0.8\nvllm:request_prefill_kv_computed_tokens_sum 10\nomni_npu:kv_transfer_completed_total 4\nomni_npu:kv_transfer_failures_total 2\n'
    after='omni_npu:kv_worker_memory_bytes 20\nvllm:kv_cache_usage_perc 0.1\nvllm:request_prefill_kv_computed_tokens_sum 0\nomni_npu:kv_transfer_completed_total 5\nomni_npu:kv_transfer_failures_total 2\n'
    deltas=m._kv_metric_deltas({r:before for r in ('producer','consumer')},
                               {r:after for r in ('producer','consumer')})
    assert deltas=={r:{'omni_npu:kv_transfer_completed_total':1} for r in ('producer','consumer')}
    windows={r:{'failure_count':0} for r in deltas}
    assert m._pd_failure_reasons(deltas,windows)==[]
