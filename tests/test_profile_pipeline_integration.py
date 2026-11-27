"""Real CPU parser subprocesses and actual rank/job trees; no model or NPU."""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import pytest
from fullnet import measurements as m

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("pipeline_recovery",ROOT/".agents/skills/fullnet-profile/scripts/profile_recovery.py")
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)
CLI="""import json,os,sys
from pathlib import Path
if '--help' in sys.argv: print('--collection-dir');raise SystemExit(0)
p=Path(sys.argv[sys.argv.index('--collection-dir')+1]);d=p.name.rsplit('_',1)[-1]
if os.environ.get('FAKE_CHILD_ERROR'):
 print('Traceback (most recent call last):\\nImportError: child sqlite missing');raise SystemExit(0)
if sys.argv[1:3]==['export','summary']:
 (p/'op_summary.csv').write_text('Task Type,Device_id,Task Duration(us)\\nAI_CORE,'+d+',2\\n')
if sys.argv[1:3]==['export','timeline']:
 (p/'msprof_fixture.json').write_text(json.dumps([{'ph':'X','cat':'aicore','dur':2,'args':{'Device_id':int(d)}}]))
"""

def fixture(tmp_path,monkeypatch):
 run=tmp_path/'project/models/fixture/experiments/case';run.mkdir(parents=True)
 cann=tmp_path/'cann';entry=cann/'tools/profiler/profiler_tool/analysis/msprof/msprof.py';entry.parent.mkdir(parents=True);entry.write_text(CLI)
 binary=cann/'tools/profiler/bin/msprof';binary.parent.mkdir(parents=True);binary.write_text('never execute ELF')
 (cann/'bin').mkdir();(cann/'bin/msprof').write_text('never execute ELF')
 monkeypatch.setenv('ASCEND_HOME_PATH',str(cann))
 config={'platform':'cpu-test','devices':[6,7],'env':{'VLLM_TORCH_PROFILER_DIR':str(run/'profiles')},'service':{'argv':[sys.executable]},'measurements':{'profile_expected_profs':2,'profile_expected_devices':[6,7]}}
 (run/'config.json').write_text(json.dumps(config))
 for role,dev in [('producer',6),('consumer',7)]:
  raw=run/'profiles'/role/'rank_ascend_pt'/f'PROF_{dev}'/f'device_{dev}'/'data';raw.mkdir(parents=True);(raw/'stars_soc').write_bytes(b'raw')
  (raw.parent/f'info.json.{dev}').write_text(json.dumps({'DeviceInfo':[{'id':dev}]}))
 return run,cann,config

def test_log_parent_is_created_and_launch_filename_is_saved(tmp_path):
 prefix=tmp_path/'new/logs/cli-help'
 got=r.run_command([str(tmp_path/'missing-python')],os.environ,tmp_path,prefix,5)
 assert not got['passed'];assert got['exit'] is None
 assert got['filename']==str(tmp_path/'missing-python')
 assert got['traceback']
 assert Path(str(prefix)+'.stderr').exists()

def test_legal_sdk_binary_symlink_copied_to_private_real_file(tmp_path):
 cann=tmp_path/'sdk';(cann/'bin').mkdir(parents=True);(cann/'tools/profiler/bin').mkdir(parents=True)
 source=cann/'tools/profiler/bin/msprof';source.write_text('ELF fixture')
 try:(cann/'bin/msprof').symlink_to(source)
 except OSError:pytest.skip('symlink unavailable')
 run=tmp_path/'run';run.mkdir();record=m.msprof_shadow_toolkit(cann,run)
 assert record.get('shadow'),record
 copied=Path(record['shadow'])/'tools/profiler/bin/msprof'
 assert not copied.is_symlink();assert copied.read_bytes()==source.read_bytes()
 assert (cann/'bin/msprof').is_symlink()

def test_prelaunch_actual_sqlite_and_official_entry_without_shadow(tmp_path,monkeypatch):
 run,cann,c=fixture(tmp_path,monkeypatch)
 tools=m.prepare_profiler_tools(run,c,launch_env=dict(os.environ,**c['env']))
 assert tools['parser_ready'];assert tools['parser_preflight']['sqlite']['passed']
 assert tools['parser_preflight']['cli']['passed']
 assert tools['parser_entry']

def test_dependency_failure_blocks_before_cli_and_copy(tmp_path,monkeypatch):
 run,cann,c=fixture(tmp_path,monkeypatch);bad=tmp_path/'bad';bad.mkdir();(bad/'sqlite3.py').write_text("raise ImportError('dependency sentinel')")
 env=dict(os.environ,**c['env'],PYTHONPATH=str(bad))
 with pytest.raises(RuntimeError,match='parser'):
  m.prepare_profiler_tools(run,c,launch_env=env)
 record=json.loads((run/'performance/profiler-tools.json').read_text())
 assert not record['parser_ready'];assert 'dependency sentinel' in record['parser_preflight']['sqlite']['stderr']
 assert not (run/'performance/parser-prelaunch/cli-help.stdout').exists()
 assert not list((run/'performance').glob('parse-offline*'))

def test_nested_roles_complete_cli_export_then_explicit_verify(tmp_path,monkeypatch):
 run,cann,c=fixture(tmp_path,monkeypatch)
 m.prepare_profiler_tools(run,c,launch_env=dict(os.environ,**c['env']))
 result=m._offline_parse_profiles(run)
 assert result['acceptance']=='passed',result
 assert result['verify']['observed_profs']==2
 assert result['verify']['observed_devices']==['6','7']
 assert all(x['passed'] for x in result['verify']['ranks'])
 before=list((run/'performance').glob('parse-offline*'))
 again=m._offline_parse_profiles(run)
 assert again['reused_existing'];assert list((run/'performance').glob('parse-offline*'))==before

def test_parent_exit_zero_child_failure_rejected(tmp_path,monkeypatch):
 run,cann,c=fixture(tmp_path,monkeypatch)
 m.prepare_profiler_tools(run,c,launch_env=dict(os.environ,**c['env']))
 monkeypatch.setenv('FAKE_CHILD_ERROR','1')
 result=m._offline_parse_profiles(run)
 assert result['acceptance']=='not_accepted'
 assert result['first_error']


def test_prepared_flag_alone_cannot_start_model(tmp_path,monkeypatch):
 spec=importlib.util.spec_from_file_location("quality_fixture_for_profile",ROOT/"tests/test_quality_only.py")
 q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 root,c,run_id=q._fixture_config(tmp_path,False,"prelaunch-rejected")
 c['env']={'VLLM_TORCH_PROFILER_DIR':str(root/'profiles')}
 from fullnet.execution import plan,run_experiment
 plan(root,c,run_id)
 observed=[]
 monkeypatch.setattr(m,'prepare_profiler_tools',lambda *a,**kw:{'prepared_before_launch':True,'env_updates':{}})
 monkeypatch.setattr('fullnet.execution.subprocess.Popen',lambda *a,**kw:observed.append(a))
 with pytest.raises(RuntimeError,match='parser is not ready'):
  run_experiment(root,'workflow-fixture',run_id)
 assert observed==[]
 assert not (root/'models/workflow-fixture/experiments'/run_id/'process.json').exists()


def test_missing_rank_and_wrong_raw_metadata_cannot_pass(tmp_path,monkeypatch):
 run,cann,c=fixture(tmp_path,monkeypatch)
 c['measurements']['profile_expected_profs']=3
 (run/'config.json').write_text(json.dumps(c))
 m.prepare_profiler_tools(run,c,launch_env=dict(os.environ,**c['env']))
 result=m._offline_parse_profiles(run)
 assert result['acceptance']=='not_accepted'
 assert result['verify']['observed_profs']==2
 assert result['verify']['expected_profs']==3
 # Same completed exports, corrected denominator, corrupted metadata: still fail.
 c['measurements']['profile_expected_profs']=2;(run/'config.json').write_text(json.dumps(c))
 copied=Path(result['workdir'])/'rank-0'
 record=json.loads((copied/'source.json').read_text());prof=Path(record['copy'])
 info=next(prof.glob('device_*/info.json.*'));info.write_text('{"DeviceInfo":[{"id":999}]}')
 failed=m._offline_parse_profiles(run)
 assert failed['reused_existing'] and failed['acceptance']=='not_accepted'
 assert failed['verify']['ranks'][0]['metadata_errors']


def test_one_rank_csv_index_cannot_bypass_complete_capture_gate(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from contextlib import nullcontext
 run,cann,c=fixture(tmp_path,monkeypatch)
 # One role has real device export evidence; the second role has raw only.
 prof=next((run/'profiles/producer').rglob('PROF_*'))
 (prof/'op_summary.csv').write_text('Task Type,Device_id,Task Duration(us)\nAI_CORE,6,2\n')
 (prof/'msprof_fixture.json').write_text(json.dumps([{'ph':'X','cat':'aicore','dur':2,'args':{'Device_id':6}}]))
 assert m.profile_artifact_index(run)['has_device_kernel_evidence']
 direct=m._verify_captured_profile_contract(run,c)
 assert direct['status']=='FAIL' and direct['observed_profs']==2
 c.update(evaluation={},identity={'weights':str(tmp_path/'weights')})
 c['service'].update(base_url='http://127.0.0.1:1/v1',served_model='fixture')
 monkeypatch.setattr(m,'_bench_environment',lambda *a:{})
 monkeypatch.setattr(m,'verify_profiler_tools',lambda *a:{})
 monkeypatch.setattr(m,'_capture_parse_evidence',lambda *a:{})
 monkeypatch.setattr(m.urllib.request,'urlopen',lambda *a,**kw:nullcontext(SimpleNamespace(status=200,read=lambda n:b'{}')))
 def benchmark(argv,**kw):
  if '--result-dir' in argv:
   folder=Path(argv[argv.index('--result-dir')+1]);(folder/'official.json').write_text('{"completed":1,"failed":0}')
  return SimpleNamespace(returncode=0)
 monkeypatch.setattr(m.subprocess,'run',benchmark)
 monkeypatch.setattr(m,'_offline_parse_profiles',lambda *a:{'acceptance':'not_accepted','first_error':'second rank failed'})
 with pytest.raises(RuntimeError,match='second rank failed'):
  m.collect_profile(run,c)
 assert json.loads((run/'state.json').read_text())['phases']['profile']['status']=='failed'
