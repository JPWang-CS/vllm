"""CPU request/counter joins; these fixtures are not NPU transfer evidence."""
import asyncio
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fullnet import measurements as m
from test_pd_readiness import RELAY


def counter(name,labels,value,*,kind='counter'):
    # json quoting matches Prometheus escaping for the label values used here.
    fields=','.join(k+'='+json.dumps(v,ensure_ascii=False) for k,v in reversed(list(labels.items())))
    return f'# TYPE {name} {kind}\n{name}{{{fields}}} {value}\n'


def snapshots(request='request-1',*,parts='1',ack='producer.example:9',value=1):
    labels={'prefill_req_id':request,'parts':parts,'ack_addr':ack}
    return {r:counter(n,labels,value) for r,n in m._PD_OBSERVER_COUNTERS.items()}


def join(before=None,after=None,request='request-1'):
    return m.pd_observer_request_evidence(before or {r:'' for r in m._PD_OBSERVER_COUNTERS},
        after or snapshots(request),prefill_req_id=request,ack_endpoint='tcp://producer.example:9')


def test_official_parser_handles_escaped_labels_and_label_order():
    request='request-"quoted"\\part'
    result=join(request=request)
    assert result['proof_status']=='PROVEN'
    for role,rows in result['counter_deltas'].items():
        assert rows[0]['labels']['prefill_req_id']==request
        assert rows[0]['delta']==1
    assert join(request='request-failure-word')['proof_status']=='PROVEN'


def test_official_parser_accepts_unicode_labels_and_sample_timestamps():
    after=snapshots('request-caf\u00e9')
    after={r:text.rsplit(' 1\n',1)[0]+' 1 1234567\n' for r,text in after.items()}
    assert join(after=after,request='request-caf\u00e9')['proof_status']=='PROVEN'


@pytest.mark.parametrize('case',[
    'wrong_request','parts','ack','missing_header','only_ack','zero','two',
    'gauge','untyped','duplicate','malformed','nan','inf','negative','reset',
    'stale_id','missing_series','extra_label','positive_and_failure','unrelated_completion',
])
def test_counter_pair_fails_closed(case):
    before={r:'' for r in m._PD_OBSERVER_COUNTERS};after=snapshots();request='request-1'
    if case=='wrong_request':after=snapshots('another-request')
    if case=='parts':after=snapshots(parts='2')
    if case=='ack':after=snapshots(ack='other.example:9')
    if case=='missing_header':request=None
    if case=='only_ack':after['consumer']=''
    if case in ('zero','two','nan','inf','negative'):
        after=snapshots(value={'zero':0,'two':2,'nan':'NaN','inf':'+Inf','negative':-1}[case])
    if case=='gauge':after={r:t.replace(' counter',' gauge') for r,t in after.items()}
    if case=='untyped':after={r:t.split('\n',1)[1] for r,t in after.items()}
    if case=='duplicate':after['consumer']+=after['consumer']
    if case=='malformed':after['consumer']='kv_pull_completed_requests_total{bad="oops} 1\n'
    if case=='reset':before=snapshots(value=2)
    if case=='stale_id':before=snapshots();after=snapshots(value=2)
    if case=='missing_series':before=snapshots('old-request')
    if case=='extra_label':after={r:t.replace('{','{extra="unexpected",') for r,t in after.items()}
    if case=='positive_and_failure':after['consumer']+=counter('omni_npu:kv_transfer_failures_total',{'kind':'pull'},2)
    if case=='unrelated_completion':after['consumer']+=snapshots('foreign')['consumer'].split('\n',1)[1]
    result=join(before,after,request)
    assert result['proof_status']=='NOT_PROVEN'
    assert result['reasons']


def test_gauges_are_not_failure_counter_resets_and_failure_is_checked_by_name():
    before={r:'vllm:kv_cache_usage_perc 0.8\n' for r in m._PD_OBSERVER_COUNTERS}
    after={r:t+'vllm:kv_cache_usage_perc 0.1\n' for r,t in snapshots().items()}
    assert join(before,after)['proof_status']=='PROVEN'


def test_optional_dependency_error_is_actionable(monkeypatch):
    real_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='prometheus_client.parser':raise ImportError('not installed')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match=r'fullnet\[metrics\]'):
        join()


@pytest.mark.parametrize('endpoint',[
    None,'','producer.example:9','tcp://host:0','tcp://host:65536','tcp://host:9/path',
    'tcp://user@host:9','tcp://host:9?query','tcp://host:9\r\nX: injected',
    'tcp://host:9 ','tcp://主机:9','tcp://host:9,tcp://host:10','tcp://[:::]:9',
    'tcp://999.999.999.999:9','tcp://bad..host:9',
])
def test_dynamic_endpoint_invalid_even_with_matching_completion_counts(endpoint):
    before={r:'' for r in m._PD_OBSERVER_COUNTERS}
    result=m.pd_observer_request_evidence(before,snapshots(),prefill_req_id='request-1',ack_endpoint=endpoint)
    assert result['proof_status']=='NOT_PROVEN'


def test_configured_expected_ack_is_additional_constraint_not_discovery():
    before={r:'' for r in m._PD_OBSERVER_COUNTERS};after=snapshots()
    def evidence(**kwargs):
        return m.pd_observer_request_evidence(before,after,prefill_req_id='request-1',**kwargs)
    assert evidence(ack_endpoint='tcp://producer.example:9')['proof_status']=='PROVEN'
    assert evidence(ack_endpoint='tcp://producer.example:9',ack_addr='producer.example:9')['proof_status']=='PROVEN'
    assert evidence(ack_endpoint='tcp://producer.example:9',ack_addr='producer.example:10')['proof_status']=='NOT_PROVEN'
    assert evidence(ack_endpoint=None,ack_addr='producer.example:9')['proof_status']=='NOT_PROVEN'


@pytest.mark.parametrize('request_id',['original-prefill','request-"quoted"\\opaque'])
def test_relay_attaches_original_runtime_request_header_without_changing_body(monkeypatch,request_id):
    params={'prefill_req_id':request_id,'opaque_runtime_field':{'a':[1,2]}}
    payload=b'{ "id": "different-response-id", "choices": [] }'
    calls=[]
    def post(url,body):
        calls.append(body)
        return (200,json.dumps({'kv_transfer_params':params}).encode(),'application/json') if len(calls)==1 else (200,payload,'application/json')
    async def body():return {'model':'pangu','messages':[],'stream':False}
    monkeypatch.setitem(RELAY.STATE,'urls',{'prefill':'http://producer','decode':'http://consumer'})
    monkeypatch.setattr(RELAY,'_post_json',post)
    response=asyncio.run(RELAY.chat(SimpleNamespace(json=body)))
    assert response.status_code==200 and response.body==payload
    assert response.headers=={m.PREFILL_REQUEST_ID_HEADER:request_id}
    assert RELAY.PREFILL_REQUEST_ID_HEADER==m.PREFILL_REQUEST_ID_HEADER
    assert calls[1]['kv_transfer_params']==params


@pytest.mark.parametrize('request_id',[True,42,'',' padded','a\rb','a\nb','a\x00b','非header字符'])
def test_relay_rejects_invalid_or_injected_request_id_before_decode(monkeypatch,request_id):
    calls=[]
    def post(*args):
        calls.append(args)
        return 200,json.dumps({'kv_transfer_params':{'prefill_req_id':request_id}}).encode(),'application/json'
    async def body():return {'messages':[]}
    monkeypatch.setattr(RELAY,'_post_json',post)
    monkeypatch.setitem(RELAY.STATE,'urls',{'prefill':'http://producer','decode':'http://consumer'})
    assert asyncio.run(RELAY.chat(SimpleNamespace(json=body))).status_code==502
    assert len(calls)==1


def test_relay_does_not_attach_success_header_on_consumer_error(monkeypatch):
    replies=iter([(200,b'{"kv_transfer_params":{"prefill_req_id":"r1"}}','application/json'),
                  (503,b'original-error','text/plain')])
    monkeypatch.setattr(RELAY,'_post_json',lambda *a:next(replies))
    monkeypatch.setitem(RELAY.STATE,'urls',{'prefill':'http://producer','decode':'http://consumer'})
    async def body():return {'messages':[]}
    response=asyncio.run(RELAY.chat(SimpleNamespace(json=body)))
    assert response.status_code==503 and response.body==b'original-error' and response.headers=={}


def setup_probe(tmp_path,monkeypatch,*,mode='success'):
    run=tmp_path/'models/pangu-7b/experiments/observer';logs=run/'logs';logs.mkdir(parents=True)
    for role in m._PD_OBSERVER_COUNTERS:
        (logs/f'pd-split-{role}.log').write_text('ERROR historical link failed\n',encoding='utf-8')
    config={'measurements':{'pd_split_observer':{'enabled':True,'ack_source':'runtime_handoff',
                                               'failure_evidence':'local_logs'}},
            'service':{'argv':['selected-python'],'base_url':'http://front/v1','served_model':'pangu'},
            'evaluation':{},'identity':{'weights':'readonly','tp':1,'dp':1,'pp':1}}
    monkeypatch.setattr(m,'_bench_environment',lambda *a:{})
    def prepare(argv,**kwargs):
        Path(argv[argv.index('--output')+1]).write_text(json.dumps({'prompt':'test'})+'\n')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',prepare)
    seen=[];completed=[]
    class Response:
        status=200
        def __init__(self,value,headers=None):self.data=json.dumps(value).encode();self.headers=headers or {}
        def read(self,size=-1):return self.data if size<0 else self.data[:size]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def urlopen(request,**kwargs):
        if isinstance(request,str):
            seen.append('snapshot')
            data={r:''.join(snapshots(req)[r] for req in completed) for r in m._PD_OBSERVER_COUNTERS}
            if mode=='failure_counter' and completed:
                data['consumer']+=counter('omni_npu:kv_transfer_failures_total',{'kind':'pull'},1)
            return Response(data)
        seen.append('chat');req='original-'+str(len(completed)+1);completed.append(req)
        if mode=='fallback':
            with (logs/'pd-split-consumer.log').open('a') as f:f.write('WARNING Recovered from KV load failure: rescheduled\n')
        if mode=='transient_link':
            with (logs/'pd-split-consumer.log').open('a') as f:f.write('ERROR link failed: retry then success\n')
        body=json.loads(request.data)
        return Response({'usage':{'completion_tokens':body['max_tokens']}},
                        {} if mode=='missing_header' else {m.PREFILL_REQUEST_ID_HEADER:req,
                            m.PREFILL_ACK_ENDPOINT_HEADER:'tcp://producer.example:9'})
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    return run,config,seen


def test_explicit_mode_records_each_header_and_two_role_before_after(tmp_path,monkeypatch):
    run,config,seen=setup_probe(tmp_path,monkeypatch)
    frozen=json.dumps(config,sort_keys=True)
    result=m.pd_split_probe(run,config)
    assert result['acceptance']=='passed' and result['proof_status']=='PROVEN'
    assert [r['prefill_req_id'] for r in result['requests']]==['original-1','original-2']
    assert all(r['ack_endpoint']=='tcp://producer.example:9' for r in result['requests'])
    assert 'ack_addr' not in config['measurements']['pd_split_observer']
    assert json.dumps(config,sort_keys=True)==frozen
    assert seen==['snapshot','snapshot','chat','snapshot','snapshot','chat','snapshot','snapshot']
    for spec in result['requests']:
        folder=run/'performance'/('pd-split-'+spec['request_class'])
        assert set(json.loads((folder/'metrics-before.json').read_text()))=={'producer','consumer'}
        assert set(json.loads((folder/'metrics-after.json').read_text()))=={'producer','consumer'}
    assert all(w['offset_start']>0 for w in result['connector_log_windows'].values())


@pytest.mark.parametrize('case',['missing_ack','duplicate_ack','duplicate_id','invalid_ack','configured_mismatch'])
def test_real_http_header_multiplicity_and_runtime_ack_are_enforced(tmp_path,monkeypatch,case):
    from email.message import Message
    run,config,seen=setup_probe(tmp_path,monkeypatch);original=m.urllib.request.urlopen
    if case=='configured_mismatch':config['measurements']['pd_split_observer']['ack_addr']='producer.example:10'
    def urlopen(request,**kwargs):
        response=original(request,**kwargs)
        if not isinstance(request,str):
            headers=Message()
            for name,value in response.headers.items():
                if case=='missing_ack' and name==m.PREFILL_ACK_ENDPOINT_HEADER:continue
                headers[name]='tcp://host:9/path' if case=='invalid_ack' and name==m.PREFILL_ACK_ENDPOINT_HEADER else value
            if case=='duplicate_ack':headers[m.PREFILL_ACK_ENDPOINT_HEADER]=response.headers[m.PREFILL_ACK_ENDPOINT_HEADER]
            if case=='duplicate_id':headers[m.PREFILL_REQUEST_ID_HEADER]=response.headers[m.PREFILL_REQUEST_ID_HEADER]
            response.headers=headers
        return response
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    with pytest.raises(RuntimeError,match='observer probe failed'):m.pd_split_probe(run,config)
    result=json.loads((run/'performance/pd-split-probe.json').read_text(encoding='utf-8'))
    assert result['proof_status']=='NOT_PROVEN' and seen.count('chat')==1
    assert 'handoff_headers' in result['requests'][0]


def test_explicit_relay_endpoint_keeps_front_metrics_and_real_handoff_separate(tmp_path,monkeypatch):
    run,config,_=setup_probe(tmp_path,monkeypatch)
    config['measurements']['pd_split_observer']['relay_base_url']='http://real-relay:7/v1/'
    original=m.urllib.request.urlopen;urls=[]
    def urlopen(request,**kwargs):
        urls.append(request if isinstance(request,str) else request.full_url)
        return original(request,**kwargs)
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    assert m.pd_split_probe(run,config)['proof_status']=='PROVEN'
    assert set(urls)=={'http://real-relay:7/v1/chat/completions','http://front/metrics_snapshot'}


def test_role_metrics_urls_read_real_prometheus_text(monkeypatch):
    texts=snapshots();urls={'producer':'http://producer.example:1/metrics',
                           'consumer':'http://consumer.example:2/metrics'};seen=[]
    class Response:
        status=200
        def __init__(self,data):self.data=data
        def read(self,size):return self.data[:size]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def urlopen(url,**kwargs):
        seen.append(url)
        return Response(texts[next(r for r,u in urls.items() if u==url)].encode())
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    after=m._pd_observer_snapshot('http://front',{'metrics_urls':urls})
    assert set(seen)==set(urls.values())
    assert join(after=after)['proof_status']=='PROVEN'


@pytest.mark.parametrize('mode',['missing_header','failure_counter','fallback','transient_link'])
def test_request_failure_stops_next_probe_even_with_positive_pair(tmp_path,monkeypatch,mode):
    run,config,seen=setup_probe(tmp_path,monkeypatch,mode=mode)
    with pytest.raises(RuntimeError,match='observer probe failed'):m.pd_split_probe(run,config)
    assert seen.count('chat')==1
    result=json.loads((run/'performance/pd-split-probe.json').read_text(encoding='utf-8'))
    assert result['acceptance']=='failed' and result['proof_status']=='NOT_PROVEN'
    if mode in ('fallback','transient_link'):
        assert result['requests'][0]['transfer_evidence']['proof_status']=='PROVEN'
        assert result['connector_log_windows']['consumer']['failure_count']==1


@pytest.mark.parametrize('mode',['counter','log'])
def test_failures_at_final_snapshot_are_not_outside_the_window(tmp_path,monkeypatch,mode):
    run,config,seen=setup_probe(tmp_path,monkeypatch)
    original=m.urllib.request.urlopen
    def urlopen(request,**kwargs):
        response=original(request,**kwargs)
        if seen.count('snapshot')==6:
            if mode=='log':
                with (run/'logs/pd-split-consumer.log').open('a') as stream:
                    stream.write('ERROR link failed: final-snapshot retry\n')
            else:
                data=json.loads(response.data)
                data['consumer']+=counter('omni_npu:kv_transfer_failures_total',{'kind':'pull'},1)
                response.data=json.dumps(data).encode()
        return response
    monkeypatch.setattr(m.urllib.request,'urlopen',urlopen)
    with pytest.raises(RuntimeError,match='observer probe failed'):m.pd_split_probe(run,config)
    result=json.loads((run/'performance/pd-split-probe.json').read_text(encoding='utf-8'))
    assert seen.count('chat')==2 and result['proof_status']=='NOT_PROVEN'
    assert all(r['transfer_evidence']['proof_status']=='PROVEN' for r in result['requests'])


@pytest.mark.parametrize('case',['missing_logs','remote_claim','parts2','no_dependency'])
def test_incomplete_coverage_fails_before_issuing_requests(tmp_path,monkeypatch,case):
    run,config,seen=setup_probe(tmp_path,monkeypatch)
    if case=='missing_logs':(run/'logs/pd-split-consumer.log').unlink()
    if case=='remote_claim':config['measurements']['pd_split_observer']['failure_evidence']='private_metrics'
    if case=='parts2':config['identity']['tp']=2
    if case=='no_dependency':
        def missing():raise RuntimeError('install fullnet[metrics]')
        monkeypatch.setattr(m,'_pd_observer_parser',missing)
    with pytest.raises(RuntimeError,match='observer probe failed'):m.pd_split_probe(run,config)
    assert seen==[]
    assert json.loads((run/'performance/pd-split-probe.json').read_text(encoding='utf-8'))['proof_status']=='NOT_PROVEN'


def test_default_path_never_imports_optional_parser(tmp_path,monkeypatch):
    run,config,_=setup_probe(tmp_path,monkeypatch)
    config['measurements']={}
    def forbidden():raise AssertionError('default path imported optional parser')
    monkeypatch.setattr(m,'_pd_observer_parser',forbidden)
    # Fixtures only supply observer counters; legacy strict remains failed.
    with pytest.raises(RuntimeError,match='pd split probe failed'):m.pd_split_probe(run,config)


@pytest.mark.parametrize('line',['ERROR err pull blocks: exception',
                               'ERROR err send pull done: timeout','ERROR err in task_pull_kv: failed'])
def test_actual_runtime_error_messages_are_failure_evidence(line):
    assert m._pd_log_failure(line)
