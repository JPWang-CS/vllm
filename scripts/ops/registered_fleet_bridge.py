"""Workstation-owned SSH broker for task-scoped multi-host GLM jobs.

The server agent writes structured requests, never receives SSH credentials,
and uses the same private project on each registered target. Device jobs are
gated immediately before their script executes inside the personal container.
"""
from __future__ import annotations

import argparse
import base64
import os
import importlib.util
import io
import json
import posixpath
import re
import shlex
import stat
import time
import uuid
from pathlib import Path, PurePosixPath


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("invalid task, request or job identifier")
    return value


PROJECT_COPY_SKIP_DIRS = frozenset({
    "experiments", "__pycache__", "weights", "cache", "envs", "runtime",
    "assets", "build", "tmp", ".git", "runs", ".runtime", "secrets",
})
FRAMEWORK_SOURCE_SUFFIXES = frozenset({'.py', '.sh', '.md', '.toml', '.json', '.txt', '.yaml', '.yml'})
FRAMEWORK_SOURCE_ROOTS = ("scripts", "src", "models", ".agents", "requirements")
FRAMEWORK_SOURCE_RECEIPT = ".runtime/framework-source.json"
FRAMEWORK_SOURCE_RECEIPT_SCHEMA = 1
CAPTURE_MAX_FILE_BYTES = 2_000_000
CAPTURE_MAX_TOTAL_BYTES = 32 * 1024 * 1024
CAPTURE_MAX_RECORDS = 4096


class TargetProjectBusy(RuntimeError):
    """A run was deferred before its job directory or workload was created."""


def published_source_file_allowed(relative):
    path = PurePosixPath(relative)
    roots = set(FRAMEWORK_SOURCE_ROOTS) | {'docs', 'tests', 'examples'}
    root_files = {'AGENTS.md', 'CLAUDE.md', 'README.md', 'STATUS.md', 'LICENSE',
                  'pyproject.toml', '.gitignore', '.gitattributes'}
    return (str(path) in root_files or
            (path.parts and path.parts[0] in roots and
             (path.suffix in FRAMEWORK_SOURCE_SUFFIXES | {'.png', '.svg', '.jpg', '.jpeg', '.jsonl'})))


def bridge_source_snapshot(path=None):
    """Capture the loaded bridge bytes for freshness checks without hashing."""
    path = Path(path or __file__).resolve(strict=True)
    return path, path.read_bytes()


def loaded_bridge_matches_disk(path, loaded):
    """Detect a replaced bridge file without killing the running broker."""
    try:
        return Path(path).resolve(strict=True).read_bytes() == loaded
    except (OSError, ValueError):
        return False


_LOADED_BRIDGE_PATH, _LOADED_BRIDGE_CONTENT = bridge_source_snapshot()


def project_workload_processes(project, proc_root='/proc'):
    """Find actual project workloads, without treating old state files as live."""
    project = str(PurePosixPath(project)).rstrip('/')
    active = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            args = [v.decode('utf-8', 'replace') for v in argv if v]
            direct = any(a.startswith(project + '/models/') or
                         (a.startswith(project + '/runs/') and '/fleet-jobs/' in a)
                         for a in args)
            env = (entry / 'environ').read_bytes().split(b'\0')
            inherited = any(v.startswith(b'FULLNET_RUN=' + project.encode() + b'/models/') for v in env)
            if direct or inherited:
                active.append({'pid': int(entry.name), 'executable': Path(args[0]).name if args else 'unknown'})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # The container's reviewed update entry executes as its existing
            # privileged identity; an unreadable process makes the guard unknown.
            raise RuntimeError('cannot inspect process identity before source update')
    return {'active': active, 'busy': bool(active)}


def broker_heartbeat_compatible(value):
    """Return true only for a broker that loaded the freshness-aware bridge."""
    return (isinstance(value, dict)
            and value.get("framework_source_receipt_schema") == FRAMEWORK_SOURCE_RECEIPT_SCHEMA
            and value.get("broker_source_file") == "scripts/ops/registered_fleet_bridge.py"
            and value.get("loaded_source_matches_disk") is True)


def framework_source_manifest(root):
    """List the published framework files and sizes; no hashes or Git metadata."""
    root = Path(root).resolve(strict=True)
    files = []
    for top in FRAMEWORK_SOURCE_ROOTS:
        directory = root / top
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if any(part in PROJECT_COPY_SKIP_DIRS for part in Path(relative).parts[:-1]):
                continue
            size = path.stat().st_size
            if framework_source_file_allowed(relative, size):
                files.append({"path": relative, "bytes": size})
    return files


def validate_framework_receipt(receipt, manifest):
    """Require a published tracked-snapshot receipt, not a branch name alone."""
    if not isinstance(receipt, dict) or receipt.get("schema") != FRAMEWORK_SOURCE_RECEIPT_SCHEMA:
        raise ValueError("published framework snapshot receipt is missing or has an unsupported schema")
    if receipt.get("branch") != "main":
        raise ValueError("published framework snapshot must declare branch main")
    files = receipt.get("files")
    expected = sorted(item["path"] if isinstance(item, dict) else item for item in manifest)
    actual = sorted(files) if isinstance(files, list) and all(isinstance(item, str) for item in files) else None
    if not actual or len(actual) != len(set(actual)) or actual != expected or not isinstance(receipt.get("snapshot"), str):
        raise ValueError("published framework snapshot receipt does not match the source file list")
    return {"schema": receipt["schema"], "branch": "main",
            "snapshot": receipt["snapshot"], "files": expected}


def remote_regular_source(remote, root, relative):
    """Check every lexical component before reading a published source file."""
    import stat
    path = PurePosixPath(relative)
    current = PurePosixPath(root)
    attrs = remote.lstat(str(current))
    if not stat.S_ISDIR(attrs.st_mode):
        raise ValueError("source root is not a regular directory")
    for index, part in enumerate(path.parts):
        current = current / part
        attrs = remote.lstat(str(current))
        expected = stat.S_ISREG if index == len(path.parts) - 1 else stat.S_ISDIR
        if not expected(attrs.st_mode):
            raise ValueError("source contains a symlink or nonregular component: " + relative)
    return attrs


class CapturedRemoteAttrs:
    def __init__(self, mode, size):
        self.st_mode, self.st_size = mode, size


class CapturedRemote:
    """Read-only SFTP-shaped view backed by one remote capture response."""
    def __init__(self, payload):
        if not isinstance(payload, dict) or payload.get("schema") != "fullnet.source-capture.v1":
            raise ValueError("invalid source capture schema")
        records = payload.get("records")
        if not isinstance(records, dict) or len(records) > CAPTURE_MAX_RECORDS:
            raise ValueError("source capture record limit exceeded")
        self.records = records
        self.total_bytes = 0
        for path, record in records.items():
            if not isinstance(path, str) or not isinstance(record, dict):
                raise ValueError("invalid source capture record")
            if record.get("status") == "present" and record.get("data") is not None:
                try:
                    data = base64.b64decode(record["data"], validate=True)
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid source capture bytes") from exc
                if len(data) != record.get("size") or len(data) > CAPTURE_MAX_FILE_BYTES:
                    raise ValueError("source capture file limit exceeded")
                self.total_bytes += len(data)
                if self.total_bytes > CAPTURE_MAX_TOTAL_BYTES:
                    raise ValueError("source capture total limit exceeded")

    def lstat(self, path):
        record = self.records.get(str(path))
        if record is None or record.get("status") == "missing":
            raise FileNotFoundError(path)
        if record.get("status") != "present":
            raise OSError("source capture record is unusable: " + str(path))
        return CapturedRemoteAttrs(int(record["mode"]), int(record["size"]))

    def open(self, path, mode="rb"):
        if mode != "rb":
            raise ValueError("captured source view is read-only")
        record = self.records.get(str(path))
        self.lstat(path)
        if not stat.S_ISREG(int(record["mode"])):
            raise ValueError("captured source is not a regular file: " + str(path))
        return io.BytesIO(base64.b64decode(record["data"], validate=True))


CAPTURE_SCRIPT = r'''
import base64,json,os,stat,sys
MAX_FILE=2000000
MAX_TOTAL=32*1024*1024
manifest=json.loads(sys.argv[1])
if not isinstance(manifest,list) or len(manifest)>4096: raise ValueError('invalid capture manifest')
records={}; total=0
def add(path):
 global total
 path=os.path.abspath(path)
 if path in records:return
 try: attrs=os.lstat(path)
 except FileNotFoundError:
  records[path]={'status':'missing'}; return
 record={'status':'present','mode':attrs.st_mode,'size':attrs.st_size}
 if stat.S_ISREG(attrs.st_mode):
  if attrs.st_size>MAX_FILE: raise ValueError('captured file exceeds limit: '+path)
  flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
  fd=os.open(path,flags)
  try:
   actual=os.fstat(fd)
   if not stat.S_ISREG(actual.st_mode) or actual.st_size!=attrs.st_size: raise OSError('source changed during capture: '+path)
   data=os.read(fd,MAX_FILE+1)
  finally: os.close(fd)
  if len(data)>MAX_FILE: raise ValueError('captured file exceeds limit: '+path)
  total+=len(data)
  if total>MAX_TOTAL: raise ValueError('captured source total exceeds limit')
  record['size']=len(data); record['data']=base64.b64encode(data).decode('ascii')
 records[path]=record
 if len(records)>4096: raise ValueError('captured source record limit exceeded')
for item in manifest:
 if not isinstance(item,dict) or not isinstance(item.get('root'),str) or not isinstance(item.get('files'),list) or not isinstance(item.get('parents',[]),list): raise ValueError('invalid capture root')
 root=os.path.abspath(item['root'])
 add(root)
 for relative in [*item.get('parents',[]),*item['files']]:
  if not isinstance(relative,str) or os.path.isabs(relative) or '..' in relative.replace('\\','/').split('/'): raise ValueError('unsafe capture relative path')
  current=root
  for part in relative.replace('\\','/').split('/'):
   current=os.path.join(current,part); add(current)
   if records[current].get('status')=='present' and not stat.S_ISDIR(records[current]['mode']) and current!=os.path.join(root,*relative.replace('\\','/').split('/')): break
print(json.dumps({'schema':'fullnet.source-capture.v1','records':records},separators=(',',':')))
'''


def capture_manifest(roots):
    """Validate the allowlisted capture request before remote execution."""
    if not isinstance(roots, list) or not roots or len(roots) > 4:
        raise ValueError("invalid source capture roots")
    result = []
    for item in roots:
        if not isinstance(item, dict):
            raise ValueError("invalid source capture root")
        root = _absolute_posix_path(item.get("root"), "capture root")
        files = item.get("files")
        if not isinstance(files, list) or len(files) > CAPTURE_MAX_RECORDS:
            raise ValueError("invalid source capture files")
        safe = []
        for relative in files:
            path = PurePosixPath(relative)
            if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                    or not published_source_file_allowed(relative)):
                raise ValueError("invalid published source path")
            safe.append(relative)
        result.append({"root": str(root), "files": safe, "parents": []})
    base = PurePosixPath(result[0]["root"])
    for item in result[1:]:
        root = PurePosixPath(item["root"])
        if not root.is_relative_to(base):
            raise ValueError("capture roots do not share a registered project root")
        relative = root.relative_to(base).as_posix()
        if relative:
            item_base = result[0]["parents"]
            parts = relative.split("/")
            item_base.extend("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return result


def compare_active_target_source(source, source_root, target, target_root, files, progress=None):
    """Verify identical published files without changing an active target."""
    records, mismatches = [], []
    for relative in files:
        if progress:
            progress()
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or not published_source_file_allowed(relative):
            raise ValueError('invalid published source path')
        remote_regular_source(source, source_root, relative)
        with source.open(source_root+'/'+relative,'rb') as stream:
            expected = stream.read(2_000_001)
        if len(expected)>2_000_000:
            raise ValueError('published source exceeds transfer limit')
        try:
            remote_regular_source(target,target_root,relative)
            with target.open(target_root+'/'+relative,'rb') as stream:
                actual = stream.read(len(expected)+1)
        except FileNotFoundError:
            mismatches.append({'path':relative,'reason':'missing'})
            continue
        if actual != expected:
            mismatches.append({'path':relative,'reason':'different content'})
        else:
            records.append({'path':relative,'source_path':source_root+'/'+relative,
                'target_path':target_root+'/'+relative,'bytes':len(expected),
                'changed':False,'backup_path':None,'write_performed':False})
    return {'matches':not mismatches,'files_checked':len(files),
            'files':records,'mismatches':mismatches}


def target_source_reuse_decision(comparison, target_busy, busy_detail=''):
    """Return the read-only reuse mode, or preserve idle sync / active wait."""
    if comparison['matches']:
        return comparison, ('reuse_identical_active' if target_busy else 'reuse_identical_idle')
    if target_busy:
        raise TargetProjectBusy(busy_detail + '; published source update is pending: '
                                + json.dumps(comparison['mismatches'][:8]))
    return None, None


def compare_control_to_snapshot(remote, project_root, receipt, progress=None):
    """Compare control-project core files with the published snapshot contents."""
    snapshot = _absolute_posix_path(receipt["snapshot"], "published snapshot")
    project = _absolute_posix_path(project_root, "control project")
    runs = project / "runs"
    if not snapshot.is_relative_to(runs) or snapshot.name != "published":
        raise ValueError("published snapshot must be project/runs/<publish-task>/published")
    mismatches = []
    for relative in receipt["files"]:
        if progress:
            progress()
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or any(part in PROJECT_COPY_SKIP_DIRS - {"assets", "cache"} for part in path.parts[:-1])
                or '.local.' in relative
                or not published_source_file_allowed(relative)):
            raise ValueError("published snapshot contains an unsupported source path: " + relative)
        source_path = str(project / path)
        snapshot_path = str(snapshot / path)
        try:
            source_attrs = remote_regular_source(remote, project, relative)
            snapshot_attrs = remote_regular_source(remote, project, str(PurePosixPath(snapshot).relative_to(project) / path))
            if source_attrs.st_size > 2_000_000 or snapshot_attrs.st_size > 2_000_000:
                raise ValueError("published source exceeds the transfer limit: " + relative)
            with remote.open(source_path, "rb") as source_stream:
                source_data = source_stream.read(2_000_001)
            with remote.open(snapshot_path, "rb") as snapshot_stream:
                snapshot_data = snapshot_stream.read(2_000_001)
        except FileNotFoundError:
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        if source_data != snapshot_data:
            mismatches.append({"path": relative, "reason": "content"})
    return {"files_checked": len(receipt["files"]), "mismatches": mismatches,
            "matches": not mismatches, "source_root": str(project),
            "snapshot_root": str(snapshot)}


def compare_snapshot_to_remote(snapshot_root, remote, remote_root):
    """Compare the published snapshot to the control project by file contents."""
    snapshot_root = Path(snapshot_root).resolve(strict=True)
    manifest = framework_source_manifest(snapshot_root)
    mismatches = []
    for entry in manifest:
        relative = PurePosixPath(entry["path"])
        target = str(PurePosixPath(remote_root) / relative)
        try:
            attrs = remote.lstat(target)
        except FileNotFoundError:
            mismatches.append({"path": entry["path"], "reason": "missing"})
            continue
        import stat
        if not stat.S_ISREG(attrs.st_mode):
            mismatches.append({"path": entry["path"], "reason": "not_regular"})
            continue
        local_data = (snapshot_root / Path(*relative.parts)).read_bytes()
        if attrs.st_size != len(local_data):
            mismatches.append({"path": entry["path"], "reason": "size"})
            continue
        with remote.open(target, "rb") as stream:
            if stream.read(len(local_data) + 1) != local_data:
                mismatches.append({"path": entry["path"], "reason": "content"})
    return {"files_checked": len(manifest), "mismatches": mismatches,
            "matches": not mismatches, "source_root": str(snapshot_root),
            "remote_root": str(remote_root)}


def stage_declared_repair_dependencies(source, target, source_root, job, script, mkdirs):
    """Stage explicit non-code dependencies and verify the received files.

    Optional repair/staging-manifest.json and <script>.dependencies.json use
    {"files": ["repair/source/contract.json", ...]}. All paths stay task-local;
    absent declarations preserve the existing script-only staging contract.
    """
    import stat
    canonical=str(PurePosixPath(script).with_suffix('.dependencies.json'))
    mistyped=script+'.dependencies.json'
    try:source.lstat(source_root+'/'+canonical)
    except FileNotFoundError:
        try:source.lstat(source_root+'/'+mistyped)
        except FileNotFoundError:pass
        else:raise ValueError('misnamed dependency manifest: '+mistyped+'; use '+canonical)
    manifests=['repair/staging-manifest.json',canonical]
    declared=set()
    for rel in manifests:
        try:attrs=source.lstat(source_root+'/'+rel)
        except FileNotFoundError:continue
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size>131072:
            raise ValueError('invalid repair dependency manifest: '+rel)
        with source.open(source_root+'/'+rel,'rb') as stream:value=json.loads(stream.read())
        files=value.get('files') if isinstance(value,dict) else None
        if not isinstance(files,list) or any(not isinstance(p,str) for p in files):
            raise ValueError('repair dependency manifest requires a files list')
        declared.update([rel,*files])
    receipt=[]
    for rel in sorted(declared):
        path=PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='repair' or not framework_source_file_allowed(rel,0):
            raise ValueError('repair dependency escapes scope or has unsupported type: '+rel)
        # Reject linked parents as well as linked leaf files.
        for index in range(1,len(path.parts)+1):
            attrs=source.lstat(source_root+'/'+PurePosixPath(*path.parts[:index]).as_posix())
            if stat.S_ISLNK(attrs.st_mode):raise ValueError('repair dependency is a symlink: '+rel)
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size>2_000_000:
            raise ValueError('repair dependency is missing, nonregular or too large: '+rel)
        with source.open(source_root+'/'+rel,'rb') as stream:data=stream.read(2_000_001)
        if path.suffix in {'.py','.sh'}:data=data.replace(b'\r\n',b'\n')
        destination=job+'/'+rel
        mkdirs(str(PurePosixPath(destination).parent))
        try:
            target.lstat(destination)
        except FileNotFoundError:
            with target.open(destination,'wx') as stream:stream.write(data)
        attrs=target.lstat(destination)
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size!=len(data):
            raise RuntimeError('repair dependency transfer is incomplete: '+rel)
        with target.open(destination,'rb') as stream:received=stream.read(len(data)+1)
        if received!=data:raise RuntimeError('repair dependency differs after transfer: '+rel)
        receipt.append({'path':rel,'bytes':len(data),'verified':True})
    return receipt


def framework_source_file_allowed(name, size):
    """Include model manifests/templates while excluding machine-local config."""
    return (PurePosixPath(name).suffix in FRAMEWORK_SOURCE_SUFFIXES | {'.jsonl'}
            and '.local.' not in name and size <= 2_000_000)


def atomic_sftp_write(sftp, destination, data, unique_suffix):
    """Write a complete regular-file replacement, then atomically rename it."""
    import stat
    destination = str(PurePosixPath(destination))
    try:
        attrs = sftp.lstat(destination)
    except FileNotFoundError:
        attrs = None
    if attrs is not None:
        if stat.S_ISLNK(attrs.st_mode):
            raise ValueError("target source is a symlink")
        if not stat.S_ISREG(attrs.st_mode):
            raise ValueError("target source is not a regular file")
    suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", str(unique_suffix))[:32]
    temporary = str(PurePosixPath(destination).parent / (".fleet-tmp-" + suffix + "-" + uuid.uuid4().hex[:8]))
    try:
        with sftp.open(temporary, "wx") as output:
            output.write(data)
        sftp.chmod(temporary, stat.S_IMODE(attrs.st_mode) if attrs is not None else 0o644)
        sftp.posix_rename(temporary, destination)
    except BaseException:
        try:
            sftp.remove(temporary)
        except (FileNotFoundError, OSError):
            pass
        raise


def copy_project_directory_allowed(name, python_package=False):
    """Keep model runtime/assets/build trees read-only and out of staging."""
    return name not in PROJECT_COPY_SKIP_DIRS or (name == "cache" and python_package)


def remote_directory_contains_python(sftp, path):
    """Identify a Python package/cache without importing or executing it."""
    import stat
    try:
        return any(
            stat.S_ISREG(entry.st_mode)
            and (entry.filename == "__init__.py" or entry.filename.endswith(".py"))
            for entry in sftp.listdir_attr(path)
        )
    except (FileNotFoundError, OSError):
        return False


def business_checkpoint_summary(task_path, expected_cases=()):
    """Report per-case business state independently from broker/controller liveness."""
    task = Path(task_path)
    source = next((p for p in (task / "BUSINESS_CHECKPOINT.json", task / "CHECKPOINT.json",
                               task / "repair/BUSINESS_CHECKPOINT.json") if p.is_file()), None)
    raw = json.loads(source.read_text(encoding="utf-8")) if source else None
    raw_cases, checkpoint_field = checkpoint_cases(raw) if source else ({}, None)
    aliases = {"92b-mixed": "mixed", "same-host-pd-split": "same_host_pd_split",
               "multi-host-tp": "multi_host_tp", "cross-host-pd-split": "cross_host_pd_split"}
    raw_cases = {aliases.get(str(name), str(name)): value for name, value in raw_cases.items()}
    names = sorted(set(expected_cases) | set(raw_cases))
    completed = {"completed", "passed", "accepted", "done"}
    running = {"running", "starting", "active"}
    pending = {"pending", "queued", "waiting", "deferred", "submitted"}
    failed = {"failed", "error", "aborted", "cancelled"}
    blocked = {"blocked", "resource_blocked", "waiting_resource"}
    cases = {}
    for name in names:
        entry = raw_cases.get(name) or {}
        if not isinstance(entry, dict):
            entry = {"status": entry}
        status = str(entry.get("status", entry.get("phase", "pending"))).lower()
        submitted = bool(entry.get("request_id") or entry.get("fleet_request_id")
                         or entry.get("job_id") or entry.get("job") or status in running)
        if status in completed:
            state = "completed"
        elif status in running:
            state = "running"
        elif status in failed:
            state = "failed"
        elif status in blocked:
            state = "blocked"
        elif status in pending:
            state = "pending"
        else:
            state = "unknown"
        cases[name] = {"state": state, "raw_status": status, "submitted": submitted,
                       "request_id": entry.get("request_id") or entry.get("fleet_request_id"),
                       "job_id": entry.get("job_id") or entry.get("job"),
                       "experiment": entry.get("experiment") or entry.get("experiment_id"),
                       "next_action": entry.get("next_action"),
                       "last_error": entry.get("last_error"),
                       "resubmit_allowed": state == "pending" and not submitted}
    requests = []
    request_dir = task / "fleet/requests"
    if request_dir.is_dir():
        for path in request_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if value.get("action") == "run":
                requests.append(value.get("request_id") or path.stem)
    return {"checkpoint": str(source) if source else None, "cases": cases,
            "submitted_request_ids": sorted(str(v) for v in requests),
            "checkpoint_field": checkpoint_field, "controller_independent": True}


def checkpoint_cases(raw):
    """Read stable ``cases`` and the narrow legacy ``four_types`` alias."""
    if not isinstance(raw, dict):
        raise ValueError("business checkpoint must be a JSON object")

    def normalize(value, field):
        if isinstance(value, list):
            normalized = {}
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"business checkpoint {field} list item must be an object")
                name = item.get("case") or item.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"business checkpoint {field} list item needs case or name")
                if name in normalized:
                    raise ValueError(f"business checkpoint {field} has duplicate case name: {name}")
                normalized[name] = item
            value = normalized
        if not isinstance(value, dict):
            raise ValueError(f"business checkpoint {field} must be an object or list")
        if any(not isinstance(name, str) or not name.strip() or not isinstance(item, dict)
               for name, item in value.items()):
            raise ValueError(f"business checkpoint {field} entries must be named objects")
        return value

    cases = normalize(raw["cases"], "cases") if "cases" in raw else None
    legacy = normalize(raw["four_types"], "four_types") if "four_types" in raw else None
    if cases:
        return cases, "cases"
    if legacy:
        return legacy, "four_types"
    if cases is not None:
        return cases, "cases"
    if legacy is not None:
        return legacy, "four_types"
    raise ValueError("business checkpoint must contain cases (or legacy four_types)")


def container_to_host_read_path(path, mounts):
    """Map a container path through an explicit read-only/bind mount list."""
    container_path = PurePosixPath(path)
    if not container_path.is_absolute() or ".." in container_path.parts:
        raise ValueError("container source path must be absolute and traversal-free")
    candidates = []
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get("Type") != "bind":
            continue
        source = PurePosixPath(mount.get("Source", ""))
        destination = PurePosixPath(mount.get("Destination", ""))
        if (not source.is_absolute() or not destination.is_absolute()
                or ".." in source.parts or ".." in destination.parts):
            continue
        if container_path.is_relative_to(destination):
            candidates.append((len(destination.parts), source, destination))
    if not candidates:
        raise ValueError("container source path has no registered bind mount: " + str(path))
    _, source, destination = max(candidates, key=lambda item: item[0])
    relative = container_path.relative_to(destination)
    host_path = source / relative
    if not host_path.is_absolute() or ".." in host_path.parts:
        raise ValueError("mapped host source path is not canonical: " + str(host_path))
    return str(host_path)


def health_status_ok(text):
    """Accept both npu-smi Health: OK and Health Status: OK formats."""
    values = re.findall(r"Health(?:\s+Status)?\s*:\s*([A-Za-z]+)\b", text, re.IGNORECASE)
    if not values:
        # A3 npu-smi info tables put the status in the column after the device
        # name: | Ascend910 | OK |. Only treat explicit status cells as health.
        values = re.findall(r"\|\s*(OK|Healthy|Error|Warning|Fault|Unknown)\s*\|", text, re.IGNORECASE)
    return bool(values) and all(value.upper() == "OK" for value in values)


def process_table_empty(text):
    return bool(re.search(r"No process in device|No running processes found in NPU", text, re.IGNORECASE))


def canonical_inventory_result(raw):
    """Use only the target-container query as the inventory source of truth."""
    if not isinstance(raw, dict):
        raise RuntimeError("inventory probe did not return an object")
    container = raw.get("container_query")
    if not isinstance(container, dict) or container.get("exit") != 0:
        raise RuntimeError("container inventory query failed; host evidence is diagnostic only")
    for field in ("mapping", "devices"):
        value = container.get(field)
        if not isinstance(value, dict) or value.get("exit") != 0:
            raise RuntimeError(
                f"container inventory {field} query failed; host evidence is diagnostic only"
            )
    result = dict(raw)
    result["mapping"] = container["mapping"]
    result["devices"] = container["devices"]
    result["execution_surface"] = "container"
    result["tool"] = raw.get("container_tool")
    return result


def validate_request(value, targets):
    if set(value) - {"request_id", "action", "host", "job", "script", "devices", "args", "asset", "framework_source"}:
        raise ValueError("unexpected fleet request fields")
    identifier(value.get("request_id"))
    identifier(value.get("job", value["request_id"]))
    if value.get("host") not in targets:
        raise ValueError("target is not registered for this task")
    if value.get("action") not in {"inventory", "run", "status", "cancel", "copy_asset"}:
        raise ValueError("unsupported action")
    if value.get("action") == "copy_asset" and value.get("asset") not in {"pangu-7b", "pangu-92b"}:
        raise ValueError("only the two registered model assets may be transferred")
    devices = value.get("devices", [])
    if not isinstance(devices, list) or len(set(devices)) != len(devices) or any(type(x) is not int or not 0 <= x < 32 for x in devices):
        raise ValueError("invalid device selection")
    args = value.get("args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) or len(x) > 4096 for x in args):
        raise ValueError("arguments must be bounded strings")
    if value["action"] == "run":
        contract = value.get("framework_source")
        if (not isinstance(contract, dict)
                or contract.get("schema") != FRAMEWORK_SOURCE_RECEIPT_SCHEMA
                or contract.get("branch") != "main"
                or contract.get("receipt_path") != FRAMEWORK_SOURCE_RECEIPT):
            raise ValueError("run requires the published framework snapshot receipt contract")
        script = PurePosixPath(value.get("script", ""))
        if script.is_absolute() or ".." in script.parts or script.suffix not in {".py", ".sh"} or script.parts[0] != "repair":
            raise ValueError("script must be a task repair Python or Bash file")
    return value


def _absolute_posix_path(value, label):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute traversal-free path")
    return path


def validate_remote_path(expected, resolved, label="remote task path"):
    """Reject a path whose realpath escaped the registered project."""
    expected_path = _absolute_posix_path(expected, label)
    resolved_path = _absolute_posix_path(resolved, f"resolved {label}")
    if resolved_path != expected_path:
        raise ValueError(f"{label} resolves outside the registered project")
    return str(expected_path)


def public_fleet_metadata(control_host, control_project, task, targets):
    """Build the non-sensitive target manifest exposed to the remote task."""
    if not isinstance(control_host, str) or not control_host.strip():
        raise ValueError("control host is required")
    control_project = _absolute_posix_path(control_project, "control project")
    task = identifier(task)
    if not isinstance(targets, dict) or not targets:
        raise ValueError("fleet target map is required")
    safe_targets = {}
    for host, value in sorted(targets.items()):
        if not isinstance(host, str) or not host.strip() or not isinstance(value, dict):
            raise ValueError("fleet target metadata is malformed")
        management = _absolute_posix_path(value.get("management"), f"management for {host}")
        project = _absolute_posix_path(value.get("project"), f"project for {host}")
        container = value.get("container")
        if not isinstance(container, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container):
            raise ValueError(f"container for {host} is invalid")
        protected = value.get("protected_devices", [])
        if (not isinstance(protected, list) or len(set(protected)) != len(protected)
                or any(type(device) is not int or not 0 <= device < 32 for device in protected)):
            raise ValueError(f"protected devices for {host} are invalid")
        safe_targets[host] = {
            "host": host,
            "container": container,
            "project": str(project),
            "management": str(management),
            "protected_devices": sorted(protected),
        }
    return {
        "schema": "fullnet.fleet-config.v1",
        "control_host": control_host,
        "control_project": str(control_project),
        "task": task,
        "targets": safe_targets,
    }


# Runs inside the selected personal container, under the target job directory.
RUNNER = r'''
import json,os,pathlib,signal,subprocess,sys,time
import re
def health_status_ok(text):
 values=re.findall(r"Health(?:\s+Status)?\s*:\s*([A-Za-z]+)\b",text,re.IGNORECASE)
 if not values: values=re.findall(r"\|\s*(OK|Healthy|Error|Warning|Fault|Unknown)\s*\|",text,re.IGNORECASE)
 return bool(values) and all(value.upper()=='OK' for value in values)
def process_table_empty(text):
 return bool(re.search(r"No process in device|No running processes found in NPU",text,re.IGNORECASE))
def acquire_device_leases(root, devices):
 if not devices:return []
 root=pathlib.Path(root).resolve(strict=True)
 if not root.is_dir():raise RuntimeError('project root is not a directory')
 runtime_dir=root/'.runtime';lease_dir=runtime_dir/'device-leases'
 for directory in (runtime_dir,lease_dir):
  if directory.is_symlink():raise RuntimeError('device lease directory is a symlink')
  if directory.exists() and not directory.is_dir():raise RuntimeError('device lease path is not a directory')
  if not directory.resolve(strict=False).is_relative_to(root):raise RuntimeError('device lease directory escapes project')
  directory.mkdir(parents=True,exist_ok=True)
  if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(root):raise RuntimeError('device lease directory escapes project')
 leases=[]
 try:
  import fcntl
  for device in sorted(set(devices)):
   path=lease_dir/('npu-'+str(device)+'.lock')
   if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):raise RuntimeError('device lease file escapes project')
   flags=os.O_CREAT|os.O_RDWR
   if hasattr(os,'O_NOFOLLOW'):flags|=os.O_NOFOLLOW
   if hasattr(os,'O_CLOEXEC'):flags|=os.O_CLOEXEC
   try:fd=os.open(str(path),flags,0o600)
   except OSError as exc:raise RuntimeError('device lease file is unavailable: '+str(device)) from exc
   stream=os.fdopen(fd,'a+b',closefd=True)
   try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   except (BlockingIOError,OSError) as exc:
    stream.close();raise RuntimeError('device lease busy: '+str(device)) from exc
   leases.append(stream)
  return leases
 except BaseException:
  release_device_leases(leases);raise
def release_device_leases(leases):
 try:import fcntl
 except ImportError:fcntl=None
 for stream in reversed(leases):
  try:
   if fcntl is not None:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
  except OSError:pass
  try:stream.close()
  except OSError:pass
def write_device_post(job,devices,tool,env,workload_started):
 for d in devices:
  target=job/f'device-{d}-post.txt'
  if not workload_started:
   target.write_text('not_run: workload did not start; post device query skipped\n');continue
  try:
   p=subprocess.run([str(tool or pathlib.Path(c['management'])/'tools/bin/npu-smi'),'info','-t','proc-mem','-i',str(d)],capture_output=True,text=True,timeout=30,env=env)
   target.write_text(p.stdout+p.stderr)
  except Exception as e:target.write_text(str(e))
def validate_structured_outputs(directory):
 checked=[]
 for name in ('result.json','validation.json'):
  p=directory/name
  if not p.exists():continue
  if p.is_symlink() or not p.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
   raise RuntimeError(name+' resolves outside the job')
  if p.stat().st_size>16*1024*1024:
   raise RuntimeError(name+' exceeds compact result limit; keep bulk data separately')
  try:json.loads(p.read_text(encoding='utf-8'))
  except (ValueError,UnicodeError) as exc:
   raise RuntimeError(name+' must contain one complete JSON document: '+str(exc)) from exc
  checked.append(name)
 return checked
j=pathlib.Path(sys.argv[1]).resolve(strict=True); c=json.loads((j/'config.json').read_text())
root=pathlib.Path(c['project']).resolve(strict=True)
if not j.is_relative_to(root):raise ValueError('job outside project')
os.chdir(root)
for k,s in {'HOME':'home','TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_EXTENSIONS_DIR':'cache/torch','TRITON_CACHE_DIR':'cache/triton','ASCEND_PROCESS_LOG_PATH':'logs/ascend','ASCEND_WORK_PATH':'tmp/ascend'}.items():
 p=j/s;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
overlay=j/'repair'/'source'/'src'
python_roots=[str(overlay),str(root/'src')] if overlay.is_dir() else [str(root/'src')]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',FULLNET_REMOTE_PROJECT=str(root),FULLNET_FLEET_JOB_ROOT=str(j),FULLNET_SERVER_ENV_ROOT=c['management'],FULLNET_FLEET_DEVICES=json.dumps(c['devices']),PYTHONPATH=os.pathsep.join(python_roots))
os.environ.pop('ASCEND_RT_VISIBLE_DEVICES',None);os.environ.pop('ASCEND_VISIBLE_DEVICES',None)
# Host-matched npu-smi: the broker stages the host ELF into <job>/tools/npu-smi;
# containers without it fall back to the shared wrapper (which needs a real
# in-container npu-smi). Driver libs are prepended for either candidate.
dlenv=dict(os.environ);dlenv['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common'+((':'+dlenv['LD_LIBRARY_PATH']) if dlenv.get('LD_LIBRARY_PATH') else '')
tool=None
for cand in (j/'tools/npu-smi',pathlib.Path(c['management'])/'tools/bin/npu-smi'):
 try:
  if cand.is_file() and cand.stat().st_size>4096:tool=cand;break
 except OSError:pass
def write(name,value):
 p=j/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
child=None;code=1;leases=[];workload_started=False
try:
 if c['devices']:
  leases=acquire_device_leases(root,c['devices'])
  if tool is None:raise RuntimeError('no host-matched npu-smi tool staged or shared for this container')
  mapping=subprocess.run([str(tool),'info','-m'],capture_output=True,text=True,timeout=30,env=dlenv)
  (j/'device-mapping.txt').write_text(mapping.stdout+mapping.stderr)
  if mapping.returncode:raise RuntimeError('device mapping query failed')
  for d in c['devices']:
   h=subprocess.run([str(tool),'info','-t','health','-i',str(d)],capture_output=True,text=True,timeout=30,env=dlenv)
   p=subprocess.run([str(tool),'info','-t','proc-mem','-i',str(d)],capture_output=True,text=True,timeout=30,env=dlenv)
   (j/f'device-{d}-pre.txt').write_text(h.stdout+h.stderr+p.stdout+p.stderr)
   import re
   if h.returncode or p.returncode or not health_status_ok(h.stdout) or not process_table_empty(p.stdout):raise RuntimeError(f'device {d} is not confirmed healthy and empty')
  # Scripts derive logical/physical runtime IDs from the saved official mapping;
  # host NPU IDs are deliberately not reused as runtime device IDs here.
 write('state.json',{'status':'running','runner_pid':os.getpid(),'started_at':time.time(),'devices':c['devices']})
 argv=['bash',str(j/c['script'])] if c['script'].endswith('.sh') else ['python3','-B',str(j/c['script'])]
 with (j/'workload.log').open('x') as log:
  child=subprocess.Popen(argv+c['args'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  workload_started=True
  write('process.json',{'pid':child.pid,'pgid':child.pid,'argv':argv+c['args'],'job':c['job'],'project':str(root)})
  while child.poll() is None:
   if (j/'CANCEL').exists():
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(2)
  code=child.wait()
  if code==0:
   checked=validate_structured_outputs(j)
   write('output-check.json',{'status':'passed','checked_files':checked,'business_acceptance':'NOT_VALIDATED'})
except BaseException as e:
 if code==0:code=1
 write('failure.json',{'error':type(e).__name__,'detail':str(e)[:2000]})
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=45)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
 (j/'exit.txt').write_text(str(code)+'\n')
 write('state.json',{'status':'exited','exit':code,'finished_at':time.time(),'runner_pid':os.getpid()})
 # Device post-state is retained independently from the script exit status.
 try:write_device_post(j,c['devices'],tool,dlenv,workload_started)
 finally:release_device_leases(leases)
sys.exit(code)
'''


STATUS_READER = r'''
import json,pathlib,sys
JSON_LIMIT=256*1024
LOG_LIMIT=14000
JSON_FILES={'state.json','process.json','failure.json','result.json','validation.json'}
j=pathlib.Path(sys.argv[1]).resolve(strict=True);r=pathlib.Path(sys.argv[2]).resolve(strict=True)
if not j.is_relative_to(r/'runs'/sys.argv[3]/'fleet-jobs'):raise ValueError('job outside task')
if sys.argv[4]=='cancel':(j/'CANCEL').touch(exist_ok=True)
o={'job':str(j),'file_metadata':{}}
def safe_file(name):
 p=j/name
 if p.is_symlink():
  raise ValueError('status file symlink or path escape: '+str(p))
 if not p.is_file():return None
 if not p.resolve(strict=True).is_relative_to(j):
  raise ValueError('status file symlink or path escape: '+str(p))
 return p
def json_value(p):
 size=p.stat().st_size
 with p.open('rb') as stream:
  raw=stream.read(JSON_LIMIT+1)
 if size<=JSON_LIMIT and len(raw)<=JSON_LIMIT:
  value=raw.decode('utf-8',errors='replace')
  metadata={'path':str(p),'original_bytes':size,'truncated':False,
            'snippet_start_byte':0,'limit_bytes':JSON_LIMIT}
  try:json.loads(value);metadata['json_valid']=True
  except ValueError as exc:metadata.update(json_valid=False,json_error=str(exc))
  return value,metadata
 return {'status':'truncated','truncated':True,'path':str(p),'original_bytes':size,
         'limit_bytes':JSON_LIMIT,'snippet_start_byte':0},{'path':str(p),
         'original_bytes':size,'truncated':True,'snippet_start_byte':0,'limit_bytes':JSON_LIMIT}
def log_value(p):
 size=p.stat().st_size
 start=max(0,size-LOG_LIMIT)
 with p.open('rb') as stream:
  stream.seek(start)
  raw=stream.read(LOG_LIMIT)
 return raw.decode('utf-8',errors='replace'),{'path':str(p),'original_bytes':size,
         'truncated':start>0,'snippet_start_byte':start,'limit_bytes':LOG_LIMIT}
for n in ['state.json','process.json','failure.json','exit.txt','workload.log','runner.log','result.json','validation.json','report.md']:
 p=safe_file(n)
 if p is not None:
  value,metadata=json_value(p) if n in JSON_FILES else log_value(p)
  o[n]=value
  o['file_metadata'][n]=metadata
print(json.dumps(o,ensure_ascii=False))
'''


class Broker:
    def __init__(self, helper, control, project, task, targets, local):
        self.helper, self.control, self.project = helper, control, project
        self.task, self.targets, self.local = identifier(task), targets, Path(local)
        self.run = PurePosixPath(project) / "runs" / task
        self.bridge_path, self.loaded_bridge = _LOADED_BRIDGE_PATH, _LOADED_BRIDGE_CONTENT
        self.local.mkdir(parents=True, exist_ok=True)

    def freshness(self):
        return {
            "framework_source_receipt_schema": FRAMEWORK_SOURCE_RECEIPT_SCHEMA,
            "broker_source_file": "scripts/ops/registered_fleet_bridge.py",
            "loaded_source_matches_disk": loaded_bridge_matches_disk(
                self.bridge_path, self.loaded_bridge),
        }

    def _read_source_receipt(self, source):
        path = str(PurePosixPath(self.project) / FRAMEWORK_SOURCE_RECEIPT)
        with source.open(path, "rb") as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("published framework snapshot receipt is too large")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise ValueError("published framework snapshot receipt is invalid JSON") from exc

    def _check_published_source(self, source, receipt):
        if not loaded_bridge_matches_disk(self.bridge_path, self.loaded_bridge):
            raise RuntimeError("broker source changed on disk; restart broker before a new run")
        files = receipt.get("files") if isinstance(receipt, dict) else None
        validated = validate_framework_receipt(receipt, files or [])
        verified_source_view = self._capture_remote_view(
            self.control,
            [{"root": self.project, "files": validated["files"]},
             {"root": validated["snapshot"], "files": validated["files"]}],
            heartbeat_source=source,
        )
        self.verified_source_view = verified_source_view
        comparison = compare_control_to_snapshot(verified_source_view or source, self.project, validated,
                                                 lambda:self.heartbeat(source))
        if not comparison["matches"]:
            raise RuntimeError(
                "control project does not match the published tracked snapshot: "
                + json.dumps(comparison["mismatches"][:8], ensure_ascii=False))
        return validated, comparison

    def _target_has_active_workload(self, client, root, container):
        import inspect
        code = ('import json,sys,os\nfrom pathlib import Path,PurePosixPath\n'
                + inspect.getsource(project_workload_processes)
                + '\nprint(json.dumps(project_workload_processes(sys.argv[1])))')
        result = json.loads(self.call(
            client, shlex.join(["docker", "exec", "-i", container, "python3", "-B", "-c", code, root]),
            timeout=45))
        if result.get("busy"):
            raise TargetProjectBusy(
                "target project has an active run; refusing framework hot update: "
                + json.dumps(result.get("active", [])[:8]))
        return result

    def call(self, client, cmd, body=None, timeout=45):
        rc, out, err = self.helper.call(client, cmd, body, timeout=timeout)
        if rc:
            raise RuntimeError(f"remote command exit {rc}: {err[-1500:]} {out[-1000:]}")
        return out

    def _capture_remote_view(self, client, roots, heartbeat_source=None):
        """Capture allowlisted read-only source bytes in one remote exec."""
        if not callable(getattr(client, "exec_command", None)):
            return None
        manifest = capture_manifest(roots)
        command = shlex.join([
            "python3", "-B", "-c", CAPTURE_SCRIPT,
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        ])
        if heartbeat_source is not None:
            self.heartbeat(heartbeat_source)
        raw_output = self.call(client, command, timeout=120)
        if heartbeat_source is not None:
            self.heartbeat(heartbeat_source)
        raw = json.loads(raw_output)
        return CapturedRemote(raw)

    def atomic(self, sftp, name, value):
        temp = name + ".tmp"
        with sftp.open(temp, "w") as out:
            out.write(json.dumps(value, ensure_ascii=False).encode())
        sftp.posix_rename(temp, name)

    def heartbeat(self, source, force=False):
        """Publish liveness while bounded source checks and transfers progress."""
        now = time.time()
        if not force and now-getattr(self,'last_heartbeat',0)<15:
            return
        self.helper.mkdirs(source,str(self.run/'fleet'),0o700)
        value = {'status':'connected','at':now,'pid':os.getpid(),'task':self.task,
                 **self.freshness()}
        self.atomic(source,str(self.run/'fleet/broker-heartbeat.json'),value)
        (self.local/'broker-heartbeat.json').write_text(json.dumps(value),encoding='utf8')
        self.last_heartbeat = now

    def publish_target_metadata(self):
        """Publish a task-local, credential-free manifest on a fresh connection."""
        if not getattr(self, "control_host", None):
            raise ValueError("control host must be set before publishing fleet metadata")
        metadata = public_fleet_metadata(self.control_host, self.project, self.task, self.targets)
        run = str(self.run)
        validate_remote_path(
            str(_absolute_posix_path(self.project, "control project")),
            self.call(self.control, shlex.join(["realpath", "-e", self.project])).strip(),
            "control project",
        )
        validate_remote_path(run, self.call(self.control, shlex.join(["realpath", "-e", run])).strip())
        sf = self.control.open_sftp()
        target = str(self.run / "fleet-config.local.json")
        temporary = target + ".tmp"
        try:
            import stat
            for path in (target, temporary):
                try:
                    attrs = sf.lstat(path)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(attrs.st_mode):
                    raise ValueError("fleet metadata path must not be a symlink")
            self.atomic(sf, target, metadata)
        finally:
            sf.close()
        return {"path": target, "schema": metadata["schema"], "targets": sorted(metadata["targets"])}

    def asset_source_container(self):
        """The CPU control host need not be a device target."""
        container = getattr(self, 'control_container', 'w00939120')
        if not isinstance(container, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', container):
            raise ValueError('invalid registered control container')
        return container

    def execute(self, request, source):
        value = validate_request(request, self.targets)
        self.verified_source_view = None
        target = self.targets[value["host"]]
        root, management, con = target["project"], target["management"], target["container"]
        if set(value.get("devices", [])) & set(target.get("protected_devices", [])):
            raise ValueError("selected device is protected by an existing service")
        source_receipt = None
        source_comparison = None
        if value["action"] == "run":
            source_receipt = self._read_source_receipt(source)
            source_receipt, source_comparison = self._check_published_source(source, source_receipt)
        verified_source_view = self.verified_source_view
        client = self.helper.connect(target.get("ssh_alias", value["host"]))
        try:
            job = str(PurePosixPath(root) / "runs" / self.task / "fleet-jobs" / value.get("job", value["request_id"]))
            if value["action"] == "copy_asset":
                if value["host"] == self.control_host:
                    raise ValueError("asset already belongs to the control project")
                asset = value["asset"]
                origin = self.project + "/assets/weights/" + asset
                source_container = self.asset_source_container()
                code = "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]).resolve(strict=True); print(json.dumps({'root':str(p),'files':[{'path':str(f.relative_to(p)),'size':f.stat().st_size} for f in p.rglob('*') if f.is_file() and f.suffix in {'.safetensors','.json','.txt','.model','.tiktoken','.py','.jinja'} and f.resolve().is_relative_to(p)]}))"
                listing = json.loads(self.call(self.control, shlex.join(["docker", "exec", "-i", "--user", "0", source_container, "python3", "-B", "-c", code, origin]), timeout=60))
                if not any(x["path"].endswith(".safetensors") for x in listing["files"]):
                    raise ValueError("source has no real model weight shards")
                mounts_raw = self.call(self.control, shlex.join([
                    "docker", "inspect", "--format", "{{json .Mounts}}",
                    source_container,
                ]), timeout=30).strip()
                try:
                    mounts = json.loads(mounts_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("control container mount inspection was not valid JSON") from exc
                if not isinstance(mounts, list):
                    raise ValueError("control container mount inspection did not return a list")
                source_host_root = container_to_host_read_path(listing["root"], mounts)
                source_sf = self.control.open_sftp(); target_sf = client.open_sftp()
                transferred = 0
                progress_path = self.local / (value["request_id"] + "-copy-progress.json")
                remote_progress = job + "/copy-progress.json"
                progress = {
                    "request_id": value["request_id"], "job": value.get("job", value["request_id"]),
                    "asset": asset, "status": "copying", "source_container_root": listing["root"],
                    "source_host_root": source_host_root, "files_total": len(listing["files"]),
                    "files_completed": 0, "files_skipped_existing": 0, "transferred_bytes": 0,
                }

                def save_progress():
                    self.atomic(target_sf, remote_progress, progress)
                    temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
                    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(progress_path)

                try:
                    self.helper.mkdirs(target_sf, job, 0o700)
                    save_progress()
                    for entry in listing["files"]:
                        rel = PurePosixPath(entry["path"])
                        if rel.is_absolute() or ".." in rel.parts:raise ValueError("asset path outside model")
                        progress.update(current_file=rel.as_posix(), current_file_status="starting")
                        save_progress()
                        dest = root + "/assets/weights/" + asset + "/" + rel.as_posix()
                        self.helper.mkdirs(target_sf, str(PurePosixPath(dest).parent), 0o700)
                        actual = self.call(client, shlex.join(["realpath", "-e", str(PurePosixPath(dest).parent)])).strip()
                        if not PurePosixPath(actual).is_relative_to(PurePosixPath(root)):
                            raise ValueError("asset destination resolves outside target project")
                        try:
                            attrs = target_sf.lstat(dest)
                            import stat
                            if not stat.S_ISREG(attrs.st_mode):raise ValueError("existing target asset is not regular")
                            if attrs.st_size == entry["size"]:
                                progress["files_completed"] += 1
                                progress["files_skipped_existing"] += 1
                                progress.update(current_file_status="skipped_existing", transferred_bytes=transferred)
                                save_progress()
                                continue
                            raise ValueError("existing asset size differs; preserve it and inspect")
                        except FileNotFoundError:pass
                        partial = dest + "." + self.task + ".partial"
                        with source_sf.open(source_host_root + "/" + rel.as_posix(), "rb") as inp, target_sf.open(partial, "w") as out:
                            if hasattr(out, "set_pipelined"):
                                out.set_pipelined(True)
                            while True:
                                data = inp.read(4 * 1024 * 1024)
                                if not data:break
                                out.write(data);transferred += len(data)
                        if target_sf.stat(partial).st_size != entry["size"]:raise RuntimeError("incomplete asset transfer")
                        target_sf.rename(partial, dest)
                        progress["files_completed"] += 1
                        progress.update(current_file_status="completed", transferred_bytes=transferred)
                        save_progress()
                    progress.update(status="completed", current_file_status=None)
                    save_progress()
                except BaseException as exc:
                    progress.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:1000])
                    save_progress()
                    raise
                finally:
                    source_sf.close();target_sf.close()
                return {"asset":asset,"transferred_bytes":transferred,"files":len(listing["files"]),"target":root+"/assets/weights/"+asset,"acceptance":"files_transferred_model_not_validated"}
            if value["action"] == "inventory":
                code = r'''
import json,os,pathlib,shutil,stat,subprocess,sys
m=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2]);con=sys.argv[3];request_id=sys.argv[4]
o={'management':str(m),'project':str(r),'project_exists':r.exists(),'project_realpath':str(r.resolve()),'container':con}
def call(args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=60);return {'exit':p.returncode,'stdout':p.stdout[-16000:],'stderr':p.stderr[-1500:]}
 except Exception as e:return {'error':type(e).__name__,'detail':str(e)}
o['container_info']=call(['docker','inspect','--format','{{json .Mounts}}',con])
management_tool=m/'tools/bin/npu-smi'
o['host_raw']={'tool':{'path':str(management_tool),'execution_surface':'host','query':'not_run'}}
for label,args in [('addresses',['ip','-br','address']),('routes',['ip','route'])]:o[label]=call(args)

project_arg=r
root=project_arg.resolve(strict=True)
if not project_arg.is_absolute() or str(root)!=str(project_arg) or not root.is_dir():raise RuntimeError('project root is not a canonical directory')
runtime=root/'.runtime';device_query=runtime/'device-query'/request_id
for directory in (runtime,runtime/'device-query',device_query):
 if directory.is_symlink() or (directory.exists() and not directory.is_dir()):raise RuntimeError('device-query path is not a real directory')
 directory.mkdir(parents=True,exist_ok=True)
 if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(root):raise RuntimeError('device-query path escapes project')
source_tool=None
for candidate in (pathlib.Path('/usr/local/sbin/npu-smi'),pathlib.Path('/usr/local/bin/npu-smi')):
 try:
  mode=candidate.stat().st_mode
  if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size>4096 and mode&0o111:
   source_tool=candidate;break
 except OSError:pass
if source_tool is None:raise RuntimeError('runner host-matched npu-smi ELF is unavailable')
with source_tool.open('rb') as inp:
 if inp.read(4)!=b'\x7fELF':raise RuntimeError('runner host-matched npu-smi is not an ELF')
staged=device_query/'npu-smi'
if staged.exists() or staged.is_symlink():raise RuntimeError('inventory staged npu-smi already exists')
flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
if hasattr(os,'O_NOFOLLOW'):flags|=os.O_NOFOLLOW
fd=os.open(str(staged),flags,0o500)
try:
 with os.fdopen(fd,'wb') as out, source_tool.open('rb') as inp:shutil.copyfileobj(inp,out)
finally:
 try:os.chmod(staged,0o555)
 except OSError:pass
if staged.is_symlink() or not staged.resolve(strict=True).is_relative_to(root):raise RuntimeError('staged npu-smi escapes project')
container_probe=r"""import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);tool=pathlib.Path(sys.argv[2])
if not root.is_dir() or str(root.resolve(strict=True))!=str(root):raise RuntimeError('mounted project path is not canonical')
if tool.is_symlink() or not tool.is_file() or not tool.resolve(strict=True).is_relative_to(root):raise RuntimeError('container tool is outside mounted project')
env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common'+((':'+env['LD_LIBRARY_PATH']) if env.get('LD_LIBRARY_PATH') else '')
env.pop('ASCEND_RT_VISIBLE_DEVICES',None);env.pop('ASCEND_VISIBLE_DEVICES',None)
def query(args):
 try:
  p=subprocess.run([str(tool),*args],capture_output=True,text=True,timeout=20,env=env)
  return {'exit':p.returncode,'stdout':p.stdout[-16000:],'stderr':p.stderr[-1500:]}
 except Exception as e:return {'error':type(e).__name__,'detail':str(e)}
print(json.dumps({'mapping':query(['info','-m']),'devices':query(['info']),'tool':{'path':str(tool),'realpath':str(tool.resolve(strict=True)),'execution_surface':'container','ld_library_path':env.get('LD_LIBRARY_PATH')}}))"""
probe=call(['docker','exec','--user','0','-i','-w',str(root),con,'python3','-B','-c',container_probe,str(root),str(staged)])
if probe.get('exit')==0:
 try:
  o['container_query']=json.loads(probe.get('stdout',''))
  o['container_query']['exit']=0
 except (TypeError,ValueError):o['container_query']={'exit':1,'stdout':probe.get('stdout',''),'stderr':probe.get('stderr',''),'error':'invalid container probe JSON'}
else:o['container_query']=probe
o['container_tool']={'path':str(staged),'realpath':str(staged.resolve(strict=True)),'execution_surface':'container','source':str(source_tool)}
for label,base in [('model_runtime',m.parents[1]/'pangu-deploy'),('project_runtime',r/'envs'),('weights',r/'assets/weights')]:
 o[label]={'path':str(base),'exists':base.exists(),'entries':sorted(x.name for x in base.iterdir())[:40] if base.is_dir() else []}
print(json.dumps(o,ensure_ascii=False))
'''
                raw = json.loads(self.call(client, shlex.join(["python3", "-B", "-", management, root, con, value["request_id"]]), code, 180))
                return canonical_inventory_result(raw)
            if value["action"] in {"status", "cancel"}:
                code = STATUS_READER
                return json.loads(self.call(client, shlex.join(["docker", "exec", "-i", "--user", "0", con, "python3", "-B", "-", job, root, self.task, value["action"]]), code))
            # Realpaths and container mount visibility are checked before writing.
            check = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); assert str(p.resolve())==str(p), 'project symlink outside registered root'"
            self.call(client, shlex.join(["python3", "-B", "-c", check, root]))
            self.call(client, shlex.join(["docker", "exec", con, "test", "-d", root]))
            active_source = None
            reused_source = None
            reused_source_mode = None
            if value["action"] == "run" and value["host"] != self.control_host:
                try:
                    target_workload = self._target_has_active_workload(client, root, con)
                except TargetProjectBusy as exc:
                    target_view = self._capture_remote_view(
                        client, [{"root": root, "files": source_receipt["files"]}],
                        heartbeat_source=source)
                    if target_view is not None:
                        active_source = compare_active_target_source(
                            verified_source_view or source, self.project, target_view, root,
                            source_receipt['files'], lambda:self.heartbeat(source))
                    else:
                        check_sf = client.open_sftp()
                        try:
                            active_source = compare_active_target_source(
                                source,self.project,check_sf,root,source_receipt['files'],
                                lambda:self.heartbeat(source))
                        finally:
                            check_sf.close()
                    try:
                        reused_source, reused_source_mode = target_source_reuse_decision(
                            active_source, True, str(exc))
                    except TargetProjectBusy as wait:
                        raise wait from exc
                else:
                    # An idle target is safe to inspect before creating the job.
                    # Reuse is read-only; mismatches deliberately fall through to
                    # the existing synchronization path below.
                    target_view = self._capture_remote_view(
                        client, [{"root": root, "files": source_receipt["files"]}],
                        heartbeat_source=source)
                    if target_view is not None:
                        idle_source = compare_active_target_source(
                            verified_source_view or source, self.project, target_view, root,
                            source_receipt['files'], lambda:self.heartbeat(source))
                    else:
                        check_sf = client.open_sftp()
                        try:
                            idle_source = compare_active_target_source(
                                source,self.project,check_sf,root,source_receipt['files'],
                                lambda:self.heartbeat(source))
                        finally:
                            check_sf.close()
                    reused_source, reused_source_mode = target_source_reuse_decision(
                        idle_source, False)
            sf = client.open_sftp()
            try:
                try:
                    sf.stat(job)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("job already exists; inspect instead of duplicating")
                self.helper.mkdirs(sf, job, 0o700)
                source_root = str(self.run)
                # Copy only task-authored repair scripts; never credentials or environments.
                def copy_scripts(relative):
                    folder = source_root + "/" + relative
                    for ent in source.listdir_attr(folder):
                        self.heartbeat(source)
                        import stat
                        rel = relative + "/" + ent.filename
                        if stat.S_ISDIR(ent.st_mode):
                            if ent.filename in {"runtime", "assets", "build", "tmp", "cache", "envs", "experiments", "__pycache__", ".git"} and not (
                                ent.filename == "cache" and remote_directory_contains_python(
                                    source, source_root + "/" + rel
                                )
                            ):
                                continue
                            copy_scripts(rel)
                        elif stat.S_ISREG(ent.st_mode) and PurePosixPath(rel).suffix in {".py", ".sh"}:
                            if ent.st_size > 2_000_000:
                                raise ValueError("repair script exceeds transfer limit")
                            with source.open(source_root + "/" + rel, "rb") as inp:
                                data = inp.read()
                            self.helper.mkdirs(sf, str(PurePosixPath(job + "/" + rel).parent), 0o700)
                            with sf.open(job + "/" + rel, "wx") as out:
                                out.write(data.replace(b"\r\n", b"\n"))
                copy_scripts("repair")
                dependency_receipt=stage_declared_repair_dependencies(
                    source,sf,source_root,job,value['script'],
                    lambda parent:self.helper.mkdirs(sf,parent,0o700))
                self.atomic(sf,job+'/staging-dependencies.json',{'files':dependency_receipt})
                # Stage the current framework, retaining overwritten target sources.
                # This is a project transfer, not an environment/model installation.
                framework_sync = {
                    "schema": FRAMEWORK_SOURCE_RECEIPT_SCHEMA,
                    "source_policy": "published-main-tracked-snapshot",
                    "source_root": self.project,
                    "target_root": root,
                    "job": job,
                    "receipt_path": str(PurePosixPath(self.project) / FRAMEWORK_SOURCE_RECEIPT),
                    "source_files_checked": source_comparison["files_checked"] if source_comparison else None,
                    "source_matches_published_snapshot": source_comparison["matches"] if source_comparison else None,
                    "excluded_directories": sorted(PROJECT_COPY_SKIP_DIRS),
                    "files": [],
                }
                if value["host"] != self.control_host and reused_source is None:
                    import stat
                    for rel in source_receipt["files"]:
                        self.heartbeat(source)
                        # Source content has just been checked against the published
                        # snapshot. Never stage untracked scripts or local config.
                        remote_regular_source(source, self.project, rel)
                        with source.open(self.project + "/" + rel, "rb") as inp:
                            data = inp.read(2_000_001)
                        dest = root + "/" + rel
                        parent = str(PurePosixPath(dest).parent)
                        # Inspect lexical parents before mkdir can follow a link.
                        current = PurePosixPath(root)
                        for part in PurePosixPath(rel).parts[:-1]:
                            current = current / part
                            try:
                                attrs = sf.lstat(str(current))
                            except FileNotFoundError:
                                continue
                            if not stat.S_ISDIR(attrs.st_mode):
                                raise ValueError("target source parent is not a regular directory")
                        self.helper.mkdirs(sf, parent, 0o755)
                        changed = True
                        backup = None
                        try:
                            attrs = sf.lstat(dest)
                            if not stat.S_ISREG(attrs.st_mode):
                                raise ValueError("target source is not a regular file")
                            with sf.open(dest, "rb") as old:
                                old_data = old.read()
                            if old_data == data:
                                changed = False
                            else:
                                backup = job + "/source-before/" + rel
                                self.helper.mkdirs(sf, str(PurePosixPath(backup).parent), 0o700)
                                with sf.open(backup, "wx") as out:
                                    out.write(old_data)
                        except FileNotFoundError:
                            pass
                        framework_sync["files"].append({
                            "path": rel, "source_path": self.project + "/" + rel,
                            "target_path": dest, "bytes": len(data),
                            "changed": changed, "backup_path": backup})
                        if changed:
                            atomic_sftp_write(sf, dest, data, value["request_id"])
                        with sf.open(dest, "rb") as actual:
                            if actual.read(len(data) + 1) != data:
                                raise RuntimeError("target source failed content verification: " + rel)
                    framework_sync["status"] = "verified"
                elif reused_source is not None:
                    framework_sync.update(status='verified',mode=reused_source_mode,
                        files=reused_source['files'],source_writes_performed=False,
                        target_files_checked=reused_source['files_checked'])
                else:
                    framework_sync["status"] = "control_project_already_current"
                self.atomic(sf, job + "/framework-sync.json", framework_sync)
                # Confirm the selected source is a regular file after canonicalization.
                actual = self.call(self.control, shlex.join(["realpath", "-e", str(self.run / value["script"])] )).strip()
                if not PurePosixPath(actual).is_relative_to(self.run / "repair"):
                    raise ValueError("source script resolves outside task repair")
                # Stage the host-matched npu-smi ELF into the job directory so the
                # RUNNER device gate works in containers that ship no real npu-smi
                # (e.g. only a zero-byte /usr/local/bin/npu-smi stub).
                staged_tool = ""
                if value.get("devices"):
                    stage = ("mkdir -p " + shlex.quote(job + "/tools") +
                             " && for t in /usr/local/sbin/npu-smi /usr/local/bin/npu-smi; do "
                             "if [ -f \"$t\" ] && [ -x \"$t\" ] && [ \"$(stat -c %s \"$t\")\" -gt 4096 ]; then "
                             "cp \"$t\" " + shlex.quote(job + "/tools/npu-smi") +
                             " && chmod 755 " + shlex.quote(job + "/tools/npu-smi") +
                             " && echo \"$t\"; break; fi; done")
                    staged_tool = self.call(client, shlex.join(["bash", "-c", stage]), timeout=30).strip()
                config = {"project": root, "management": management, "container": con,
                          "job": value.get("job", value["request_id"]), "devices": value.get("devices", []),
                          "script": value["script"], "args": value.get("args", [])}
                for name, content in [("runner.py", RUNNER), ("config.json", json.dumps(config, ensure_ascii=False))]:
                    with sf.open(job + "/" + name, "wx") as out:
                        out.write(content.encode())
            finally:
                sf.close()
            sock = str(PurePosixPath(root) / ".runtime/fleet.sock")
            if len(sock.encode()) >= 104:
                raise ValueError("project tmux socket path is too long")
            self.call(client, shlex.join(["mkdir", "-p", str(PurePosixPath(sock).parent)]))
            argv = ["docker", "exec", "--user", "0", "-w", root, con, "python3", "-B", job + "/runner.py", job]
            command = shlex.join(argv) + " > " + shlex.quote(job + "/runner.log") + " 2>&1"
            session = "fleet-" + self.task[-24:] + "-" + config["job"][-24:]
            self.call(client, shlex.join([management + "/tools/bin/tmux", "-S", sock, "new-session", "-d", "-s", session, "-c", root, command]))
            return {"submitted": True, "host": value["host"], "project": root, "container": con,
                    "job": config["job"], "directory": job, "socket": sock, "session": session,
                    "staged_npu_smi": staged_tool or None,
                    "staged_dependencies": dependency_receipt,
                    "acceptance": "submitted_not_validated"}
        finally:
            client.close()

    def poll(self):
        sf = self.control.open_sftp()
        count = 0
        try:
            requests = str(self.run / "fleet/requests")
            responses = str(self.run / "fleet/responses")
            self.helper.mkdirs(sf, requests, 0o700)
            self.helper.mkdirs(sf, responses, 0o700)
            self.heartbeat(sf,force=True)
            for name in sorted(sf.listdir(requests)):
                self.heartbeat(sf)
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.json", name):
                    continue
                try:
                    sf.stat(responses + "/" + name)
                    continue
                except FileNotFoundError:
                    pass
                waiting_path = str(self.run/'fleet'/('waiting-'+name))
                waiting = None
                try:
                    with sf.open(waiting_path,'rb') as stream:
                        waiting = json.loads(stream.read())
                except (FileNotFoundError,ValueError):
                    pass
                if (isinstance(waiting,dict) and waiting.get('status')=='WAITING_TARGET_PROJECT'
                        and waiting.get('retry_at',0)>time.time()):
                    continue
                with sf.open(requests + "/" + name, "rb") as f:
                    raw = f.read(131073)
                if len(raw) > 131072:
                    result = {"error": "request too large"}
                else:
                    try:
                        request = json.loads(raw)
                        if request.get("request_id") != name[:-5]:
                            raise ValueError("request filename differs from request_id")
                        result = self.execute(request, sf)
                    except TargetProjectBusy as exc:
                        now = time.time()
                        record = {'status':'WAITING_TARGET_PROJECT','request_id':name[:-5],
                            'host':request.get('host'),'job':request.get('job'),
                            'workload_started':False,'job_created':False,
                            'first_seen':waiting.get('first_seen',now) if isinstance(waiting,dict) else now,
                            'last_seen':now,'retry_at':now+30,
                            'wait_count':waiting.get('wait_count',0)+1 if isinstance(waiting,dict) else 1,
                            'detail':str(exc)[:2500]}
                        self.atomic(sf,waiting_path,record)
                        (self.local/('waiting-'+name)).write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')
                        continue
                    except Exception as exc:
                        result = {"error": type(exc).__name__, "detail": str(exc)[:2500]}
                result["broker_checked_at"] = time.time()
                self.atomic(sf, responses + "/" + name, result)
                if isinstance(waiting,dict):
                    waiting.update(status='RELEASED',released_at=time.time())
                    self.atomic(sf,waiting_path,waiting)
                (self.local / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1
        finally:
            sf.close()
        return count


def terminal_controller_state(value):
    """Only the controller's completed schema ends an attempt."""
    return (isinstance(value, dict) and type(value.get('exit')) is int
            and value.get('model') == 'glm-5.3-flash'
            and value.get('child_alive') is False
            and value.get('fifo_removed') is True
            and value.get('credentials_persisted') is False)


def _normalized_controller_path(value):
    return PurePosixPath(posixpath.normpath(os.fsdecode(value).replace('\\', '/')))


def _controller_command_matches(args, task_dir, controller_path=None):
    """Match legacy glm_task or the current task-local controller.py layout."""
    task = _normalized_controller_path(task_dir)
    raw_task = str(task).encode()
    target = _normalized_controller_path(controller_path) if controller_path is not None else None
    task_controller_root = task / 'controllers'
    legacy_task_arg = raw_task in args
    for arg in args:
        try:
            path = _normalized_controller_path(arg)
        except (TypeError, ValueError):
            continue
        if path.name == 'glm_task.py':
            if legacy_task_arg:
                return True
        if path.name == 'controller.py':
            try:
                is_task_controller = path.is_relative_to(task_controller_root) or path == task / 'controller.py'
                if is_task_controller and (target is None or path == target):
                    return True
            except ValueError:
                continue
    return False


def controller_process_alive(task_dir, pid=None, proc_root=Path('/proc'), uid=None, controller_path=None):
    """Match this task's controller by UID and exact legacy/current path layout."""
    task_dir = str(task_dir)
    if uid is None:
        uid = os.getuid() if hasattr(os, 'getuid') else None
    candidate = proc_root / str(pid) if type(pid) is int else None
    if candidate is not None and candidate.exists():
        # An existing PID with a different identity must be rejected. Only a
        # missing/stale PID permits discovery of this exact task's controller.
        candidates = [candidate]
    else:
        try:
            candidates = [p for p in proc_root.iterdir() if p.name.isdecimal()]
        except OSError:
            candidates = []
    for process in candidates:
        try:
            if uid is not None and process.stat().st_uid != uid:
                continue
            args = process.joinpath('cmdline').read_bytes().split(b'\0')
            if _controller_command_matches(args, task_dir, controller_path):
                return True
        except OSError:
            continue
    return False


def latest_controller_status(task_dir, proc_root=Path('/proc')):
    """Read the newest real controllers/<host>/glmNN state, never task-root STATE."""
    task = Path(task_dir)
    candidates = []
    controllers = task / 'controllers'
    if controllers.is_dir():
        for controller_dir in controllers.glob('*/glm*'):
            state_path = controller_dir / 'STATE.json'
            identity_path = controller_dir / 'RUN_IDENTITY.json'
            active_path = controller_dir / 'ACTIVE.json'
            if not any(path.is_file() for path in (state_path, identity_path, active_path)):
                continue
            mtimes = [path.stat().st_mtime for path in (state_path, identity_path, active_path)
                      if path.is_file()]
            try:
                state = json.loads(state_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                state = None
            try:
                identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.is_file() else None
            except (OSError, ValueError):
                identity = None
            try:
                active = json.loads(active_path.read_text(encoding='utf-8')) if active_path.is_file() else {}
            except (OSError, ValueError):
                active = {}
            pid = active.get('controller_pid') or active.get('model_process', {}).get('pid')
            alive = controller_process_alive(
                task, pid, proc_root=proc_root,
                controller_path=controller_dir / 'controller.py')
            candidates.append((max(mtimes), controller_dir, state, identity, active, alive))
    if not candidates:
        return {'status': 'unknown', 'reason': 'no_controller_state_found', 'controller_independent': True}
    _, directory, state, identity, active, alive = max(candidates, key=lambda item: item[0])
    if alive:
        status = 'running'
    elif terminal_controller_state(state):
        status = 'exited'
    else:
        status = 'unknown'
    return {'status': status, 'controller_dir': str(directory), 'state': state,
            'run_identity': identity, 'active': active, 'controller_alive': alive,
            'controller_independent': True}


def reconcile_status(broker):
    """Refresh business cases separately from GLM/controller liveness."""
    code = r'''import json,pathlib,sys,time
r=pathlib.Path(sys.argv[1]);task=sys.argv[2];sys.path.insert(0,str(r))
from scripts.ops import deployment_matrix as matrix
from scripts.ops.registered_fleet_bridge import latest_controller_status, business_checkpoint_summary
p=r/'runs'/task;out={'at':time.time(),'task':task}
try:
 out['business']=business_checkpoint_summary(p,('mixed','same_host_pd_split','multi_host_tp','cross_host_pd_split'))
except Exception as e:
 out['business']={'controller_independent':True,'error':type(e).__name__,'detail':str(e)[:1000]}
if (p/'deployment-matrix.json').is_file():
 # A stale job-state file is not evidence of a live job.
 ledger=p/'ledger.json'
 if ledger.is_file():
  for item in json.loads(ledger.read_text()).get('cases',[]):
   ident=item.get('task','');j=r/'runs'/ident;f=j/'job-state.json'
   if not ident or not j.resolve().is_relative_to(r/'runs') or not f.is_file():continue
   value=json.loads(f.read_text());pid=value.get('pid')
   if value.get('status')!='running' or type(pid) is not int:continue
   try:
    raw=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes()
    alive=str(r).encode() in raw and ident.encode() in raw
   except OSError:alive=False
   if not alive:
    value.update(status='interrupted',reconciled_at=time.time(),reason='recorded business process is no longer present')
    tmp=f.with_suffix('.reconcile.tmp');tmp.write_text(json.dumps(value));tmp.replace(f)
 out['matrix']=matrix.refresh_matrix(r,task)
out['controller']=latest_controller_status(p)
out['controller_status']=out['controller']['status']
f=p/'BACKGROUND_STATUS.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(out,ensure_ascii=False));t.replace(f)
print(json.dumps({'task':task,'controller_status':out['controller_status'],'at':out['at']}))
'''
    return json.loads(broker.call(broker.control, shlex.join(['python3','-B','-',broker.project,broker.task]),code,90))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--management", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("registered_ssh", a.management / "runs/remote-model-agents-20261027/install_binary.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    registered = json.loads((a.management / "tools/ssh/server-inventory.json").read_text(encoding="utf-8"))["servers"]
    permitted = {x["ip"]: x for x in registered if not x["maintenance_forbidden"] and x["server_ai_required"]}
    for host, target in c["targets"].items():
        if host not in permitted or target["management"] != permitted[host]["project_root"]:
            raise ValueError("target differs from registered inventory")
        if target["project"] != target["management"] + "/projects/vllm":
            raise ValueError("fleet writes must stay in the registered vllm project")
        registered_alias = permitted[host]["alias"]
        if target.get("ssh_alias", registered_alias) != registered_alias:
            raise ValueError("SSH alias differs from registered inventory")
        target["ssh_alias"] = registered_alias
    receipt_root = Path(c['local_receipts'])
    receipt_root.mkdir(parents=True, exist_ok=True)
    heartbeat = receipt_root / 'broker-heartbeat.json'
    lock = receipt_root / 'broker.lock'
    # OS lock is released after crashes; the file is diagnostic, not a stale PID gate.
    lock_stream = lock.open('a+b')
    if os.name == 'nt':
        import msvcrt
        if lock.stat().st_size == 0:lock_stream.write(b'0');lock_stream.flush()
        lock_stream.seek(0)
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_refresh = 0
    while True:
        client = None
        try:
            client = helper.connect(c["control_host"])
            client.get_transport().set_keepalive(30)
            broker = Broker(helper, client, c["control_project"], c["task"], c["targets"], c["local_receipts"])
            broker.control_host = c["control_host"]
            broker.control_container = c.get("control_container", "w00939120")
            broker.asset_source_container()
            broker.publish_target_metadata()
            while True:
                broker.poll()
                if a.once:
                    return 0
                sf = client.open_sftp()
                reconnect = False
                try:
                    now = time.time()
                    value = {'status':'connected','at':now,'pid':os.getpid(),'task':c['task'], **broker.freshness()}
                    broker.atomic(sf, str(broker.run/'fleet/broker-heartbeat.json'), value)
                    heartbeat.write_text(json.dumps(value),encoding='utf-8')
                    try:
                        with sf.open(str(broker.run/'fleet/broker-control.json')) as stream:control=json.loads(stream.read())
                        if control.get('action') == 'reconnect':
                            sf.remove(str(broker.run/'fleet/broker-control.json'));reconnect=True
                    except (FileNotFoundError, ValueError):pass
                    try:sf.stat(str(broker.run/'fleet/STOP'));stopping=True
                    except FileNotFoundError:stopping=False
                finally:sf.close()
                if now-last_refresh >= 30:
                    try:
                        state=reconcile_status(broker)
                        (receipt_root/'background-status.json').write_text(json.dumps(state),encoding='utf-8')
                    except Exception as exc:
                        (receipt_root/'refresh-error.json').write_text(json.dumps({'error':type(exc).__name__,'detail':str(exc)[:1500],'at':now}),encoding='utf-8')
                    last_refresh=now
                if stopping:return 0
                if reconnect:break
                time.sleep(5)
        except (OSError, EOFError, helper.paramiko.SSHException) as exc:
            (Path(c["local_receipts"]) / "connection-error.json").write_text(json.dumps({"error":type(exc).__name__,"at":time.time()}), encoding="utf-8")
            heartbeat.write_text(json.dumps({'status':'reconnecting','at':time.time(),'pid':os.getpid(),'task':c['task'],'error':type(exc).__name__}),encoding='utf-8')
            if a.once:raise
            time.sleep(15)
        finally:
            if client:client.close()


if __name__ == "__main__":
    raise SystemExit(main())
