"""Exercise dependency staging against real local source/job directory trees."""
import importlib.util,json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location("profile_stage_bridge",Path(__file__).parents[1]/"scripts/ops/registered_fleet_bridge.py")
b=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(b)
class SFTP:
 def lstat(self,path):return Path(path).lstat()
 def open(self,path,mode):return Path(path).open('xb' if mode=='wx' else mode)

def task(tmp_path,files):
 source=tmp_path/'project/runs/task';repair=source/'repair';repair.mkdir(parents=True)
 for rel,value in files.items():
  p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
 job=tmp_path/'target/project/runs/task/fleet-jobs/probe';job.mkdir(parents=True)
 return source,job

def stage(source,job):
 return b.stage_declared_repair_dependencies(SFTP(),SFTP(),str(source),str(job),'repair/probe.py',lambda p:Path(p).mkdir(parents=True,exist_ok=True))

def test_declared_json_md_reach_actual_job_layout(tmp_path):
 source,job=task(tmp_path,{'repair/probe.dependencies.json':json.dumps({'files':['repair/source/config.json','repair/README.md']}),'repair/source/config.json':'{"expected_devices":[6,7]}','repair/README.md':'offline only'})
 receipt=stage(source,job)
 assert len(receipt)==3 and all(x['verified'] for x in receipt)
 assert json.loads((job/'repair/source/config.json').read_text())['expected_devices']==[6,7]
 assert (job/'repair/README.md').read_text()=='offline only'

def test_missing_declared_file_blocks_submission(tmp_path):
 source,job=task(tmp_path,{'repair/staging-manifest.json':'{"files":["repair/source/missing.json"]}'})
 with pytest.raises(FileNotFoundError):stage(source,job)
 assert not (job/'runner.py').exists()

def test_escape_manifest_cannot_read_other_project(tmp_path):
 source,job=task(tmp_path,{'repair/staging-manifest.json':'{"files":["repair/../../secret.json"]}'})
 with pytest.raises(ValueError,match='scope'):stage(source,job)

def test_existing_wrong_dependency_is_rejected_not_overwritten(tmp_path):
 source,job=task(tmp_path,{'repair/staging-manifest.json':'{"files":["repair/source/config.json"]}','repair/source/config.json':'{"version":1}'})
 target=job/'repair/source/config.json';target.parent.mkdir(parents=True);target.write_text('{"version":2}')
 with pytest.raises(RuntimeError,match='differs'):stage(source,job)
 assert target.read_text()=='{"version":2}'

def test_no_manifest_keeps_default_empty_contract(tmp_path):
 source,job=task(tmp_path,{})
 assert stage(source,job)==[]
