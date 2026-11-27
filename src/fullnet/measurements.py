"""Official service benchmark and server-side NPU profiling after quality requests."""
import csv,io,json,math,os,re,shutil,sqlite3,stat,subprocess,threading,time,urllib.request
from pathlib import Path
from .execution import private_environment
from .store import state,read_json,write_json


def ensure_msprof_tool_owner():
    """Inspect toolkit ownership; keep the source SDK strictly read-only."""
    home=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    if not home:
        return {'changed':0,'targets':[],'uid_after_unique':[], 'reason':'no-toolkit-home'}
    targets=[Path(home)/'bin/msprof',Path(home)/'tools/profiler']
    owners=[str(p.stat().st_uid) for p in targets if p.exists()]
    return {'changed':0,'chown_errors':0,'targets':[str(p) for p in targets],
            'uid_before_unique':sorted(set(owners)), 'uid_after_unique':sorted(set(owners)),
            'reason':'source-toolkit-read-only; use task-local tools before service launch'}


def msprof_shadow_toolkit(home,run,owner_uid=None):
    """Task-local copy of the msprof toolchain for an ownership mismatch.

    When the derived CANN tree is read-only or owned by another runtime user,
    torch_npu's owner check can reject the real tool, so CANNAnalyzeParser and
    the view parsers die while export only warns. This copies bin/msprof plus
    tools/profiler into this experiment's run subtree, assigns the copy to the
    service identity when the supervisor can do so, and PATH-prepends it so
    shutil.which('msprof') resolves the copy inside the analyzer subprocesses.
    Layout keeps a 'tools' path segment because torch_npu locates msprof.py by
    splitting the binary's realpath on 'tools' (msprof.py must exist under
    <prefix>/tools/profiler/profiler_tool/analysis/msprof/). The squashed
    original is only ever read. Device capture is unaffected; only the
    host-side analysis chain moves to the shadow.
    """
    run=Path(run).resolve(strict=True)
    src=Path(home).resolve(strict=True);shadow=run/'performance'/'msprof-shadow'
    bin_src=src/'bin/msprof';bin_dst=shadow/'tools/profiler/bin/msprof'
    try:
        if not bin_src.is_file():return {'shadow':None,'reason':'msprof binary missing'}
        for entry in (bin_src, src/'tools/profiler'):
            if not entry.resolve(strict=True).is_relative_to(src):
                return {'shadow':None,'reason':'toolkit entry resolves outside selected SDK'}
        for entry in (src/'tools/profiler').rglob('*'):
            if entry.is_symlink() and not entry.resolve(strict=True).is_relative_to(src):
                return {'shadow':None,'reason':'profiler dependency resolves outside selected SDK'}
        if shadow.exists():
            if shadow.is_symlink() or not shadow.resolve(strict=True).is_relative_to(run):
                return {'shadow':None,'reason':'shadow path escapes the experiment run'}
            shutil.rmtree(shadow)
        shadow.mkdir(parents=True,exist_ok=True)
        shutil.copytree(src/'tools/profiler',shadow/'tools/profiler',symlinks=False)
        bin_dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(bin_src,bin_dst)
        owner=None
        if os.name=='posix':
            import pwd
            uid=owner_uid if owner_uid is not None else os.geteuid()
            record=pwd.getpwuid(int(uid));owner=(record.pw_uid,record.pw_gid)
        for path in [shadow,*shadow.rglob('*')]:
            if path.is_symlink() or not path.resolve(strict=False).is_relative_to(run):
                continue
            try:
                if owner is not None:os.chown(path,owner[0],owner[1],follow_symlinks=False)
            except (OSError,AttributeError):pass
        st=bin_dst.stat()
        return {'shadow':str(shadow),'binary_uid':st.st_uid,
                'msprof_py':str(shadow/'tools/profiler/profiler_tool/analysis/msprof/msprof.py'),
                'msprof_py_exists':(shadow/'tools/profiler/profiler_tool/analysis/msprof/msprof.py').is_file()}
    except OSError as exc:
        return {'shadow':None,'error':type(exc).__name__+': '+str(exc)[:200]}


def profile_requested(c):
    argv=c.get('service',{}).get('argv',[])
    return bool(c.get('env',{}).get('VLLM_TORCH_PROFILER_DIR') or any(
        value=='--profiler-config' or str(value).startswith('--profiler-config=') for value in argv))


def _profile_recovery():
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'.agents/skills/fullnet-profile/scripts/profile_recovery.py'
    spec=importlib.util.spec_from_file_location('fullnet_profile_recovery',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _parser_prelaunch(run,c,record,launch_env):
    """Actual Python/SQLite/CLI checks, in the environment handed to the service."""
    helper=_profile_recovery();out=run/'performance'/'parser-prelaunch';out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ if launch_env is None else launch_env)
    if launch_env is None:env.update(c.get('env',{}))
    env.update(record.get('env_updates') or {})
    # A direct preparation caller also keeps HOME/cache/temp inside its run.
    if launch_env is None:
        for key,folder in [('HOME','home'),('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('XDG_CACHE_HOME','cache')]:
            target=out/folder;target.mkdir(exist_ok=True);env[key]=str(target)
    python=c['service']['argv'][0]
    home=env.get('ASCEND_HOME_PATH') or env.get('ASCEND_TOOLKIT_HOME') or record.get('home')
    entry=(record.get('shadow_toolkit') or {}).get('msprof_py')
    if not entry and home:entry=str(Path(home)/'tools/profiler/profiler_tool/analysis/msprof/msprof.py')
    record.update(parser_ready=False,parser_entry=entry,parser_python=python)
    result={'environment':{k:env[k] for k in ('PATH','LD_LIBRARY_PATH','PYTHONPATH','ASCEND_HOME_PATH','HOME','TMPDIR') if k in env},
            'tool_source':'private_shadow' if (record.get('shadow_toolkit') or {}).get('shadow') else 'selected_official_entry'}
    try:
        if not entry or not Path(entry).is_file():raise RuntimeError('official Python parser entry missing')
        settings=c.get('measurements',{})
        devices=settings.get('profile_expected_devices',c.get('devices',[]))
        count=settings.get('profile_expected_profs',len(devices))
        if (not isinstance(devices,list) or not devices or any(type(d) is not int or d<0 for d in devices)
                or len(set(devices))!=len(devices) or type(count) is not int or count<1):
            raise ValueError('expected profile rank/device contract is missing or invalid')
        record['expected_profile_contract']={'profs':count,'devices':[str(d) for d in devices]}
        # Prove the output is usable before any help/import/model process.
        with (out/'write-probe').open('x') as stream:stream.write('parser-output-writable')
        (out/'write-probe').unlink()
        result['sqlite']=helper.parser_dependency_preflight(python,env,run,out/'sqlite',30)
        if not result['sqlite']['passed']:raise RuntimeError(result['sqlite'].get('first_error','SQLite probe failed'))
        for name,args in [('cli',[]),('import',['import']),('summary',['export','summary']),('timeline',['export','timeline'])]:
            result[name]=helper.run_command([python,'-B',entry,*args,'--help'],env,run,out/(name+'-help'),60)
            text=(out/(name+'-help.stdout')).read_text(errors='replace')
            if not result[name]['passed'] or name!='cli' and '--collection-dir' not in text:
                raise RuntimeError('official parser CLI family probe failed: '+name)
        record['parser_ready']=True
    except (OSError,RuntimeError,ValueError) as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    record['parser_preflight']=result
    write_json(run/'performance/profiler-tools.json',record)
    if not record['parser_ready']:raise RuntimeError('profiler parser prelaunch failed: '+result['error'])


def prepare_profiler_tools(run,c,*,launch_env=None):
    """Prepare the msprof analysis toolchain BEFORE the service launches.

    The service spawns its profiler children only when start_profile is
    called, so ownership repair and PATH/LD_LIBRARY_PATH injection must be
    part of the launch environment; setting PATH in collect_profile after
    launch can never reach the running service. Also stages a real
    libsqlite3 copy next to the shadow (msprof_analysis.so links it and the
    squashed CANN tree copy fails for the analyzer). Returns the launch
    record; run_experiment merges record['env_updates'] into the Popen env.
    """
    run=Path(run).resolve(strict=True);out=run/'performance';out.mkdir(exist_ok=True)
    selected_env=launch_env if launch_env is not None else {**os.environ,**c.get('env',{})}
    home=selected_env.get('ASCEND_HOME_PATH') or selected_env.get('ASCEND_TOOLKIT_HOME')
    if os.name != 'posix':
        record={'home':home,'service_euid':None,'real_msprof_uid':None,
                'inspection':ensure_msprof_tool_owner(),'prepared_before_launch':True,
                'shadow_toolkit':{'shadow':None,'reason':'POSIX profiler ownership is not applicable'},
                'env_updates':{}}
        if profile_requested(c):_parser_prelaunch(run,c,record,launch_env)
        write_json(out/'profiler-tools.json',record)
        return record
    getuid=getattr(os,'getuid',lambda:None);geteuid=getattr(os,'geteuid',lambda:None)
    service_euid=int(c['service']['uid']) if c['service'].get('uid') and getuid()==0 else geteuid()
    bin_path=Path(home)/'bin/msprof' if home else None
    real_uid=bin_path.stat().st_uid if bin_path and bin_path.is_file() else None
    record={'home':home,'service_euid':service_euid,'real_msprof_uid':real_uid,
            'inspection':ensure_msprof_tool_owner(),'prepared_before_launch':True}
    if real_uid is None or real_uid not in (0,service_euid):
        if home:record['shadow_toolkit']=msprof_shadow_toolkit(home,run,service_euid)
        else:record['shadow_toolkit']={'shadow':None,'reason':'no-toolkit-home'}
    else:
        record['shadow_toolkit']={'shadow':None,'reason':'real-tool-owner-acceptable'}
    shadow=(record.get('shadow_toolkit') or {}).get('shadow')
    updates={}
    if shadow:
        shadow=Path(shadow);lib=shadow/'compat'/'lib';lib.mkdir(parents=True,exist_ok=True)
        try:
            for source in ('/usr/lib64/libsqlite3.so.0.8.6','/usr/lib64/libsqlite3.so.0','/usr/lib64/libsqlite3.so'):
                if Path(source).is_file():
                    shutil.copy2(source,lib/'libsqlite3.so');record['compat_sqlite_source']=source;break
            for path in [lib,*lib.iterdir()]:
                try:os.chown(path,0,0,follow_symlinks=False)
                except (OSError,AttributeError):pass
        except OSError as exc:record['compat_error']=type(exc).__name__+': '+str(exc)[:120]
        updates['PATH']=str(shadow/'tools/profiler/bin')+os.pathsep+os.environ.get('PATH','')
        updates['LD_LIBRARY_PATH']=str(lib)+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    record['env_updates']=updates
    if profile_requested(c):_parser_prelaunch(run,c,record,launch_env)
    write_json(out/'profiler-tools.json',record)
    return record


def verify_profiler_tools(run,c):
    """Re-check the launch-time preparation against the live service process.

    Reads the service's real /proc environment, resolves msprof exactly as
    the running service would, and records uids plus the parse command so a
    failed analysis stage is attributable to the tool the service sees. A run
    without launch evidence falls back to late preparation
    and says so instead of pretending PATH reached the service.
    """
    run=Path(run);out=run/'performance'
    try:launch=read_json(out/'profiler-tools.json')
    except (OSError,ValueError):
        launch=prepare_profiler_tools(run,c);launch['legacy_late_preparation']=True
    record={'launch':launch}
    try:pid=read_json(run/'process.json')['pid']
    except (OSError,KeyError,ValueError):pid=None
    record['service_pid']=pid
    environ={}
    if pid:
        try:
            raw=Path(f'/proc/{pid}/environ').read_bytes().decode(errors='replace')
            environ={part.split('=',1)[0]:part.split('=',1)[1] for part in raw.split('\0') if '=' in part}
        except OSError:pass
    service_path=environ.get('PATH','')
    record['service_path_head']=service_path.split(os.pathsep)[:4]
    record['service_ld_library_path_head']=environ.get('LD_LIBRARY_PATH','').split(os.pathsep)[:3]
    try:status=Path(f'/proc/{pid}/status').read_text(errors='replace') if pid else ''
    except OSError:status=''
    match=re.search(r'^Uid:\s+(\d+)\s+(\d+)',status,re.M)
    if match:record['service_uid_real_effective']=match.groups()
    which=None
    for entry in service_path.split(os.pathsep):
        if not entry:continue
        candidate=Path(entry)/'msprof'
        if candidate.is_file() and os.access(candidate,os.X_OK):which=str(candidate);break
    record['service_msprof_which']=which
    if which:
        real=str(Path(which).resolve());record['service_msprof_realpath']=real
        try:record['service_msprof_uid']=os.stat(real).st_uid
        except OSError:record['service_msprof_uid']=None
        msprof_py=real.split('tools')[0]+'tools/profiler/profiler_tool/analysis/msprof/msprof.py'
        record['service_msprof_py']=msprof_py
        try:record['service_msprof_py_uid']=os.stat(msprof_py).st_uid
        except OSError:record['service_msprof_py_uid']=None
    record['parse_command_template']=('torch_npu ProfilingParser (sync mode, in-service) -> '
                                      'msprof --export=on [--type=db] --output=<PROF_dir>; '
                                      'msprof --analyze=on [--type=db] --output=<PROF_dir>')
    return record


def _trace_device_kernel_events(path):
    import math
    from .profile_trace import iter_trace_events

    def norm(value):
        return re.sub(r'[^A-Z0-9]', '', str(value or '').upper())

    compute_types={'AICORE','AIVECTORCORE','MIXAIC','MIXAIV','AIC','AIV','AIVECTOR',
                   'KERNELAIC','KERNELAIVEC','KERNELMIXAIC','KERNELMIXAI','KERNELMIXAIV'}
    excluded=('CPU','AICPU','HCCL','COMMUNICATION','MEMCPY','SDMA','HOST')

    def has_device_metadata(args, metadata):
        for key,value in args.items():
            if norm(key) in {'DEVICEID','DEVICE','NPUID','NPU'} and str(value).strip().isdigit():
                return True
        return bool(re.search(r'(?i)\b(?:device(?:\s*id)?|npu)\D{0,8}\d+\b', metadata))

    process_names={};lanes={};process_metadata={}
    try:
        for event in iter_trace_events(path):
            if not isinstance(event,dict) or event.get('ph')!='M':continue
            args=event.get('args') or {}
            if not isinstance(args,dict):continue
            metadata=' '.join(f'{k}={v}' for k,v in args.items())
            if event.get('name') in ('process_name','process_labels'):
                pid=event.get('pid');process_names[pid]=str(args.get('name',''))
                record=process_metadata.setdefault(pid,{})
                direct=next((str(v).strip() for k,v in args.items()
                             if norm(k) in {'DEVICEID','DEVICE','NPUID','NPU'} and str(v).strip().isdigit()),None)
                match=re.search(r'(?i)\b(?:device(?:\s*id)?|npu)\D{0,8}\d+\b',metadata)
                if direct is not None:record['device']=direct
                elif match and 'device' not in record:record['device']=re.search(r'\d+',match.group()).group()
                record['metadata']=metadata
            elif event.get('name')=='thread_name':
                lanes[(event.get('pid'),event.get('tid'))]=str(args.get('name',''))
        count=0
        for event in iter_trace_events(path):
            if not isinstance(event,dict) or event.get('ph')!='X' or not event.get('name'):continue
            cat=str(event.get('cat',''));name=str(event.get('name',''))
            args=event.get('args') or {}
            if not isinstance(args,dict):args={}
            pid=event.get('pid');process=str(process_names.get(pid,''))
            metadata=str((process_metadata.get(pid) or {}).get('metadata',''))
            lane=str(lanes.get((pid,event.get('tid')),''))
            try:duration=float(event.get('dur',event.get('duration',0)))
            except (TypeError,ValueError):continue
            if not math.isfinite(duration) or duration<=0:continue
            task_values=[str(v) for k,v in args.items() if 'TASKTYPE' in norm(k) or norm(k) in {'KERNELTYPE','TASK'}]
            task_compute=any(norm(value) in compute_types for value in task_values)
            cat_compute=norm(cat) in {'AICORE','AIVECTOR','AIC','AIV'}
            lane_compute=norm(lane) in compute_types
            text=' '.join(norm(value) for value in (cat,name,process,metadata,lane,*task_values))
            if any(tag in text for tag in excluded):continue
            device_bound=has_device_metadata(args,metadata)
            if device_bound and (task_compute or cat_compute or lane_compute):
                count+=1
        return count
    except (OSError,ValueError,UnicodeError):
        return 0


def _csv_device_kernel_rows(path):
    import math

    def norm(value):
        return re.sub(r'[^A-Z0-9]', '', str(value or '').upper())

    compute_types={'AICORE','AIVECTORCORE','MIXAIC','MIXAIV','AIC','AIV','AIVECTOR',
                   'KERNELAIC','KERNELAIVEC','KERNELMIXAIC','KERNELMIXAI','KERNELMIXAIV'}
    excluded=('CPU','AICPU','HCCL','COMMUNICATION','MEMCPY','SDMA','HOST')
    count=0
    try:
        with path.open('r',encoding='utf-8-sig',newline='',errors='replace') as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames:return 0
            fields={norm(k):k for k in reader.fieldnames}
            kernel_key=next((v for k,v in fields.items() if 'KERNEL' in k or k in ('OPNAME','TASKNAME','NAME')),None)
            duration_key=next((v for k,v in fields.items() if 'DURATION' in k or 'EXECUTIONTIME' in k or 'TASKTIME' in k),None)
            task_key=next((v for k,v in fields.items() if k in ('TASKTYPE','KERNELTYPE','ACCELERATORCORE',
                           'ACCELERATORTYPE','CORETYPE','TYPE')),None)
            device_key=next((v for k,v in fields.items() if k in ('DEVICEID','DEVICE')),None)
            if not duration_key or not task_key or not device_key:return 0
            for row in reader:
                kernel=str(row.get(kernel_key,'device-row')).strip() if kernel_key else 'device-row'
                if not kernel or 'cpu' in kernel.lower():continue
                task=str(row.get(task_key,'')).strip()
                task_norm=norm(task)
                device=str(row.get(device_key,'')).strip() if device_key else ''
                if not device.isdigit() or not task_norm or task_norm in excluded or task_norm not in compute_types:continue
                try:duration=float(str(row.get(duration_key,'')).replace(',',''))
                except (TypeError,ValueError):continue
                if math.isfinite(duration) and duration>0:count+=1
    except OSError:
        return 0
    return count


def _relax_tree(base,limit=20000,run=None,exclude=()):
    """Make selected task artifacts readable without changing executability.

    msprof sets umask 027 and the root-run service inherits a restrictive
    umask, so parse outputs and state JSONs land as root-only (0600/0750)
    and CPU diagnostics running as the project owner cannot read them
    Metadata-only, bounded, and confined to the supplied task tree.
    """
    base=Path(base)
    changed=0
    try:
        boundary=Path(run).resolve(strict=True) if run is not None else base.resolve(strict=True)
        base=base.resolve(strict=True)
        if not base.is_relative_to(boundary):return changed
    except (OSError,RuntimeError):
        return changed
    excluded=set()
    for item in exclude:
        try:excluded.add(Path(item).resolve(strict=True))
        except (OSError,RuntimeError):continue
    stack=[base]
    while stack and changed<limit:
        current=stack.pop()
        try:
            for child in current.iterdir():
                resolved=child.resolve(strict=True)
                if child.is_symlink() or not resolved.is_relative_to(boundary) or resolved in excluded:continue
                current=child.stat().st_mode&0o777
                mode=(current|0o755) if child.is_dir() else (current|0o644)
                if current!=mode:
                    os.chmod(child,mode);changed+=1
                if child.is_dir():stack.append(child)
        except OSError:
            pass
    return changed


def _capture_parse_evidence(run,out,limit_bytes=65536,max_logs=40,max_dbs=8):
    """Preserve readable profiler parse evidence before the run is torn down.

    torch_npu view-parse tasks report only 'run failed' on stdout; their
    per-task ProfilerLogger files and the msprof sqlite DBs stay inside the
    msprof output tree. Copy log tails and probe DB tables so a failed parse
    can be diagnosed (and real device kernel rows can be counted) offline.
    """
    profiles=Path(run)/'profiles';evidence={'profiles_dir':str(profiles),'relaxed':0,'logs':[],'dbs':[]}
    try:evidence['relaxed']=_relax_tree(profiles,run=run)+_relax_tree(out,run=run,exclude=(out/'msprof-shadow',))
    except OSError as exc:evidence['relax_error']=str(exc)
    log_dir=out/'profile-parse-logs';log_dir.mkdir(parents=True,exist_ok=True)
    logs=sorted(profiles.rglob('*.log')) if profiles.exists() else []
    for source in logs[:max_logs]:
        try:
            with source.open('rb') as stream:
                stream.seek(max(0,source.stat().st_size-limit_bytes))
                data=stream.read()
            if source.is_symlink() or not source.resolve(strict=True).is_relative_to(Path(run).resolve(strict=True)):
                continue
            target=log_dir/source.name
            with target.open('wb') as output:
                shutil.copyfileobj(io.BytesIO(data),output)
            os.chmod(target,0o644)
            evidence['logs'].append({'source':str(source),'bytes':len(data),
                                     'tail':data.decode(errors='replace')})
        except OSError:
            continue
    for db in sorted(profiles.rglob('*.db'))[:max_dbs]:
        tables={}
        try:
            connection=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
            try:
                names=[r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                for table in names:
                    try:tables[table]=connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                    except sqlite3.Error as exc:tables[table]='error:'+type(exc).__name__
            finally:connection.close()
        except (sqlite3.Error,OSError) as exc:
            tables={'error':type(exc).__name__}
        evidence['dbs'].append({'path':str(db),'bytes':db.stat().st_size,'tables':tables})
    return evidence


def _offline_parse_profiles(run, output=None):
    """Recover saved raw data once, then explicitly verify the rank/device contract.

    Re-entry reads the previous receipt; it never replays the model or silently
    re-exports a completed/failed attempt. A repaired parser can be requested
    explicitly with a new output path.
    """
    from types import SimpleNamespace
    run=Path(run).resolve(strict=True);out=run/'performance';out.mkdir(exist_ok=True)
    helper=_profile_recovery()
    try:
        record=read_json(out/'profiler-tools.json');c=read_json(run/'config.json')
        settings=c.get('measurements',{})
        devices=[str(d) for d in settings.get('profile_expected_devices',c.get('devices',[]))]
        count=settings.get('profile_expected_profs',len(devices))
        if type(count) is not int or count<1 or not devices or len(set(devices))!=len(devices):
            raise ValueError('profile requires an explicit nonempty expected rank/device contract')
        existing=sorted(p.parent for p in out.glob('parse-offline-*/RECOVERY.json'))
        reused=output is None and bool(existing)
        work=existing[-1] if reused else Path(output) if output is not None else out/'parse-offline-01'
        work=work.resolve()
        if not work.is_relative_to(out.resolve()):raise ValueError('recovery output escapes performance directory')
        if reused:
            exported=read_json(work/'RECOVERY.json')
        else:
            entry=record.get('parser_entry') or (record.get('shadow_toolkit') or {}).get('msprof_py')
            home=record.get('home')
            if not entry and home:entry=str(Path(home)/'tools/profiler/profiler_tool/analysis/msprof/msprof.py')
            if not entry:raise ValueError('no official parser entry')
            env=dict(os.environ);env.update(c.get('env',{}));env.update(record.get('env_updates') or {})
            exported=helper.export(SimpleNamespace(source=run/'profiles',output=work,
                cann_root=Path(home),entry=Path(entry),python=c['service']['argv'][0],
                library_dir=[],timeout=settings.get('profile_parser_timeout',900),env=env))
        accepted=helper.verify_exports(work,count,devices) if exported.get('status')=='EXPORTED' else {'status':'FAIL'}
        write_json(work/'FINAL_ACCEPTANCE.json',accepted)
        errors=[]
        for path in sorted(work.glob('rank-*/*.stderr')):
            text=path.read_text(errors='replace')
            if text.strip():errors.append(_first_error(text))
        first=exported.get('first_error') or next((e for e in errors if e),None)
        passed=accepted.get('status')=='PASS'
        return {'workdir':str(work),'ranks':exported.get('ranks',[]),'verify':accepted,
                'acceptance':'passed' if passed else 'not_accepted',
                'reused_existing':reused,'first_error':None if passed else first or 'offline export/verify failed'}
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        return {'acceptance':'not_accepted','first_error':f'{type(exc).__name__}: {exc}',
                'workdir':str(work) if 'work' in locals() else None,'ranks':[]}


def _first_error(text):
    for line in str(text).splitlines():
        stripped=line.strip()
        if stripped and re.search(r'\b(error|exception|traceback|failed)\b',stripped,re.I):
            return stripped[:500]
    return next((line.strip()[:500] for line in str(text).splitlines() if line.strip()),None)


def _verify_captured_profile_contract(run,c):
    """A substantive index alone cannot prove that every captured rank exported."""
    helper=_profile_recovery();settings=c.get('measurements',{})
    expected={str(v) for v in settings.get('profile_expected_devices',c.get('devices',[]))}
    count=settings.get('profile_expected_profs',len(expected));ranks=[];observed=set()
    for raw in helper.raw_inventory(Path(run)/'profiles'):
        prof=Path(raw['path']);devices={d['name'].removeprefix('device_') for d in raw['devices'] if d['nonempty_files']}
        observed.update(devices);metadata=set();errors=[]
        for device in sorted(devices):
            try:
                info=read_json(prof/f'device_{device}'/f'info.json.{device}')
                metadata.update(str(v['id']) for v in info['DeviceInfo'])
            except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
        # torch_npu also places exports beside its single PROF, not inside it.
        base=prof.parent if len(list(prof.parent.glob('PROF_*')))==1 else prof
        evidence=helper.inspect(base,next(iter(devices)) if len(devices)==1 else None)
        csv_devices={d for item in evidence['csv'] for d in item['devices']}
        trace_devices={d for item in evidence['timelines'] for d in item['devices']}
        passed=(bool(devices) and devices==metadata==csv_devices==trace_devices and not errors
                and not evidence['parse_errors'] and evidence['csv_compute_rows']>0
                and evidence['csv_compute_rows']==evidence['timeline_compute_events'])
        ranks.append({'prof':str(prof),'devices':sorted(devices),'passed':passed,'metadata_errors':errors,'evidence':evidence})
    passed=bool(expected) and len(ranks)==count and observed==expected and all(r['passed'] for r in ranks)
    return {'status':'PASS' if passed else 'FAIL','expected_profs':count,'expected_devices':sorted(expected),
            'observed_profs':len(ranks),'observed_devices':sorted(observed),'ranks':ranks}


def profile_artifact_index(run,extra_dirs=()):
    root=Path(run);profiles=root/'profiles';items=[];candidates=[];seen=set()
    def add_candidate(path,kind):
        try:resolved=path.resolve(strict=True)
        except (OSError,RuntimeError):return
        key=str(resolved)
        if key not in seen:
            seen.add(key);candidates.append((resolved,kind))
    def scan(base,all_csv=False):
        for path in base.rglob('*'):
            if not path.is_file():continue
            name=path.name.lower()
            if path.suffix.lower()=='.csv' and (all_csv or name in {'device.csv','device_details.csv','kernel_details.csv'}
                                                or name.startswith(('op_summary','ascend_op_summary'))):
                add_candidate(path,'kernel_csv')
            elif name=='trace_view.json' or (name.startswith('msprof_') and name.endswith('.json')) \
                    or name.endswith('.pt.trace.json.gz'):
                add_candidate(path,'trace')
    if profiles.exists():scan(profiles)
    for extra in extra_dirs:
        extra=Path(extra)
        if extra.exists():scan(extra,all_csv=True)
    candidates.sort(key=lambda item:(item[1]!='kernel_csv',str(item[0])))
    for path,kind in candidates:
        exists=path.is_file();size=path.stat().st_size if exists else 0
        events=kernels=0;notes=[]
        if exists and kind=='kernel_csv':
            kernels=_csv_device_kernel_rows(path)
        elif exists:
            if size>256*1024*1024 and any(v['kind']=='kernel_csv' and v['substantive'] for v in items):
                notes.append('trace_parse_skipped:kernel_csv_evidence_present')
            else:
                events=_trace_device_kernel_events(path)
        items.append({'kind':kind,'path':str(path.resolve()),'exists':exists,'bytes':size,'notes':notes,
                      'device_kernel_events':events,'device_kernel_rows':kernels,
                      'substantive':bool(size and (events or kernels))})
    return {'artifacts':items,'has_device_kernel_evidence':any(v['substantive'] for v in items),
            'acceptance':'accepted' if any(v['substantive'] for v in items) else 'not_accepted',
            'source_directory':str(profiles.resolve()) if profiles.exists() else str(profiles)}


def _bench_environment(run,c,root):
    env=private_environment(run,c,root)
    env.update(VLLM_PLUGINS='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('VLLM_TORCH_PROFILER_DIR',None)
    return env


def collect_performance(run,c):
    """Six-group official benchmark; frozen request counts, lengths, concurrency."""
    root=Path(run).parents[3];run=Path(run);settings=c['measurements']
    out=run/'performance';out.mkdir(exist_ok=True);env=_bench_environment(run,c,root)
    if not (out/'profile-artifacts.json').is_file():
        write_json(out/'profile-artifacts.json',{'acceptance':'not_run','artifacts':[],
                    'has_device_kernel_evidence':False,'source_directory':str((run/'profiles').resolve())})
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    cli=[python,'-B','-m','vllm.entrypoints.cli.main','bench','serve']
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    body={'chat_template_kwargs':kwargs}
    if 'add_special_tokens' in c['evaluation']:body['add_special_tokens']=c['evaluation']['add_special_tokens']
    commands=[]
    def call(label,argv,folder,timeout=1800):
        record={'label':label,'argv':argv,'status':'running'};commands.append(record);write_json(out/'commands.json',commands)
        try:
            with (folder/(label+'.log')).open('w') as log:
                r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            record.update(exit=r.returncode,status='passed' if r.returncode==0 else 'failed')
            write_json(out/'commands.json',commands)
            if r.returncode:raise RuntimeError('measurement failed: '+label)
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__)
            if isinstance(exc,subprocess.TimeoutExpired):record['timeout']=True
            write_json(out/'commands.json',commands)
            raise
    call('official-help',cli+['--help'],out,120)
    def one(spec,folder,requests,warmups):
        folder.mkdir(parents=True,exist_ok=True);dataset=folder/'input.jsonl'
        call('prepare',[python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',c['identity']['weights'],
             '--tokens',str(spec['input']),'--output',str(dataset),'--kwargs',json.dumps(kwargs)],folder,180)
        args=cli+['--backend','openai-chat','--base-url',base,'--endpoint','/v1/chat/completions',
             '--model',c['service']['served_model'],'--tokenizer',c['identity']['weights'],'--trust-remote-code',
             '--dataset-name','custom','--dataset-path',str(dataset),'--skip-chat-template','--disable-shuffle',
             '--custom-output-len',str(spec['output']),'--num-prompts',str(requests),'--num-warmups',str(warmups),
             '--max-concurrency',str(spec['concurrency']),'--request-rate','inf','--ignore-eos','--temperature','0',
             '--extra-body',json.dumps(body),'--save-result','--save-detailed','--result-dir',str(folder),'--result-filename','official.json']
        call('benchmark',args,folder,settings.get('timeout_per_group',3600))
        raw=json.loads((folder/'official.json').read_text())
        if raw.get('completed')!=requests or raw.get('failed',0)!=0:raise RuntimeError('benchmark request count or failure check failed')
        return raw
    state(run,'performance','running')
    summary=[]
    for i,spec in enumerate(settings['workloads']):
        raw=one(spec,out/f'case-{i:02d}',settings.get('requests',32),settings.get('warmups',2))
        summary.append({'workload':spec,'completed':raw['completed'],**{k:raw.get(k) for k in
            ('output_throughput','request_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms','mean_e2el_ms')}})
        write_json(out/'summary.json',summary)
    state(run,'performance','passed',groups=len(summary))
    return summary


def collect_profile(run,c):
    """Server-side NPU profile capture with real device kernel evidence gate.

    Kept independent from the benchmark so a device trace exists before any
    long decode group can fail; the six-group contract stays untouched.
    """
    root=Path(run).parents[3];run=Path(run);settings=c['measurements']
    out=run/'performance';out.mkdir(exist_ok=True);env=_bench_environment(run,c,root)
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    cli=[python,'-B','-m','vllm.entrypoints.cli.main','bench','serve']
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    body={'chat_template_kwargs':kwargs}
    if 'add_special_tokens' in c['evaluation']:body['add_special_tokens']=c['evaluation']['add_special_tokens']
    commands=[]
    def call(label,argv,folder,timeout=1800):
        record={'label':label,'argv':argv,'status':'running'};commands.append(record);write_json(out/'profile-commands.json',commands)
        try:
            with (folder/(label+'.log')).open('w') as log:
                r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            record.update(exit=r.returncode,status='passed' if r.returncode==0 else 'failed')
            write_json(out/'profile-commands.json',commands)
            if r.returncode:raise RuntimeError('measurement failed: '+label)
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__)
            if isinstance(exc,subprocess.TimeoutExpired):record['timeout']=True
            write_json(out/'profile-commands.json',commands)
            raise
    def one(spec,folder,requests,warmups):
        folder.mkdir(parents=True,exist_ok=True);dataset=folder/'input.jsonl'
        call('prepare',[python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',c['identity']['weights'],
             '--tokens',str(spec['input']),'--output',str(dataset),'--kwargs',json.dumps(kwargs)],folder,180)
        args=cli+['--backend','openai-chat','--base-url',base,'--endpoint','/v1/chat/completions',
             '--model',c['service']['served_model'],'--tokenizer',c['identity']['weights'],'--trust-remote-code',
             '--dataset-name','custom','--dataset-path',str(dataset),'--skip-chat-template','--disable-shuffle',
             '--custom-output-len',str(spec['output']),'--num-prompts',str(requests),'--num-warmups',str(warmups),
             '--max-concurrency',str(spec['concurrency']),'--request-rate','inf','--ignore-eos','--temperature','0',
             '--extra-body',json.dumps(body),'--save-result','--save-detailed','--result-dir',str(folder),'--result-filename','official.json']
        call('benchmark',args,folder,settings.get('timeout_per_group',3600))
        raw=json.loads((folder/'official.json').read_text())
        if raw.get('completed')!=requests or raw.get('failed',0)!=0:raise RuntimeError('benchmark request count or failure check failed')
        return raw
    def control(action):
        req=urllib.request.Request(base+'/'+action+'_profile',method='POST')
        with urllib.request.urlopen(req,timeout=120) as response:
            write_json(out/(action+'-profile.json'),{'http_status':response.status,'body':response.read(8192).decode(errors='replace')})
    state(run,'profile','running')
    # Tool ownership and PATH were fixed up before the service launched
    # (prepare_profiler_tools); here we only verify what the live service
    # actually sees. A late PATH change cannot reach the running service's
    # profiler children.
    owner_repair=verify_profiler_tools(run,c)
    write_json(out/'profile-owner-repair.json',owner_repair)
    profile_error=None
    try:
        control('start')
        one(settings.get('profile_workload',{'input':4096,'output':128,'concurrency':1}),out/'profile-request',1,0)
    except BaseException as exc:
        profile_error=exc
    finally:
        try:control('stop')
        except BaseException as exc:
            if profile_error is None:profile_error=exc
        write_json(out/'profile-parse-evidence.json',_capture_parse_evidence(run,out))
        index=profile_artifact_index(run)
        direct=_verify_captured_profile_contract(run,c)
        write_json(out/'profile-direct-verify.json',direct)
        if direct['status']!='PASS':
            # Daemon-mode DP workers cannot run the in-service torch_npu parser
            # parse the captured *_ascend_pt trees offline instead of
            # failing with zero artifacts while valid device data is on disk.
            offline=_offline_parse_profiles(run)
            write_json(out/'profile-offline-parse.json',offline)
            extras=[Path(offline['workdir'])] if offline.get('workdir') else []
            index=profile_artifact_index(run,extra_dirs=extras)
            if offline.get('acceptance')!='passed' and profile_error is None:
                profile_error=RuntimeError('offline profile export/verify failed: '+str(offline.get('first_error')))
        write_json(out/'profile-artifacts.json',index)
    if profile_error is not None:
        state(run,'profile','failed',error=type(profile_error).__name__,detail=str(profile_error)[:1000],
              artifact_count=len(index['artifacts']),acceptance=index['acceptance'])
        raise profile_error
    if not index['has_device_kernel_evidence']:
        state(run,'profile','failed',reason='no substantive NPU kernel events or kernel CSV rows',
              artifact_count=len(index['artifacts']),
              acceptance=index['acceptance'])
        raise RuntimeError('profile contains no substantive device kernel events; CPU-only trace is insufficient')
    state(run,'profile','passed',artifact_count=len(index['artifacts']),
          device_kernel_events=sum(v['device_kernel_events'] for v in index['artifacts']),
          device_kernel_rows=sum(v['device_kernel_rows'] for v in index['artifacts']))
    return index


def _mixed_engine_lines(text,start,end):
    """Engine throughput lines inside the request window showing BOTH phases."""
    lines=[]
    for line in text.splitlines():
        if 'Avg prompt throughput' not in line or 'Avg generation throughput' not in line:
            continue
        match=re.search(r'(\d{4}-\d{2}-\d{2}|\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})',line)
        if match:
            date_part,time_part=match.groups()
            try:
                if len(date_part)==4:
                    moment=time.mktime(time.strptime(date_part+' '+time_part,'%Y-%m-%d %H:%M:%S'))
                else:
                    moment=time.mktime(time.strptime(str(time.localtime().tm_year)+'-'+date_part+' '+time_part,'%Y-%m-%d %H:%M:%S'))
            except ValueError:
                moment=None
            if moment is not None and not (start-5<=moment<=end+5):continue
        prompt=re.search(r'Avg prompt throughput:\s*([0-9.]+)',line)
        decode=re.search(r'Avg generation throughput:\s*([0-9.]+)',line)
        if not prompt or not decode:continue
        lines.append({'line':line.strip(),'prompt_tps':float(prompt.group(1)),
                      'generation_tps':float(decode.group(1)),
                      'mixed':float(prompt.group(1))>0 and float(decode.group(1))>0})
    return lines


def pd_mixed_probe(run,c):
    """Same-instance mixed prefill+decode evidence for PD-mixed acceptance.

    The already-ready single service instance receives two request classes at
    one shared wall-clock barrier: prefill-heavy (long input, short output)
    and decode-heavy (short input, long output). Evidence recorded: per-class
    request status/usage/timing, an overlapping execution window, and engine
    throughput lines from the same window with nonzero prompt AND generation
    rates on the instance. The frozen six-group benchmark, profile and
    quality chain run unchanged around this probe.
    """
    run=Path(run);out=run/'performance';out.mkdir(exist_ok=True)
    root=run.parents[3];env=_bench_environment(run,c,root)
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    timeout=c['service'].get('request_timeout',300)
    classes=[{'name':'prefill_heavy','input':4096,'output':64,'count':2},
             {'name':'decode_heavy','input':128,'output':1024,'count':2}]
    state(run,'pd_mixed_probe','running')
    prompts=[];records=[]
    for spec in classes:
        folder=out/('pd-mixed-'+spec['name']);folder.mkdir(parents=True,exist_ok=True)
        dataset=folder/'input.jsonl'
        prepared=subprocess.run([python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',
            c['identity']['weights'],'--tokens',str(spec['input']),'--output',str(dataset),
            '--kwargs',json.dumps(kwargs)],cwd=root,env=env,capture_output=True,text=True,timeout=300)
        if prepared.returncode:raise RuntimeError('mixed probe prompt preparation failed for '+spec['name'])
        prompts.append(json.loads(dataset.read_text().splitlines()[0])['prompt'])
    barrier=threading.Barrier(sum(spec['count'] for spec in classes))
    def launch(spec,prompt,index):
        record={'request_class':spec['name'],'index':index,'input_tokens_target':spec['input'],
                'output_tokens_target':spec['output']}
        body={'model':c['service']['served_model'],'messages':[{'role':'user','content':prompt}],
              'max_tokens':spec['output'],'temperature':0,'seed':0,'ignore_eos':True,'chat_template_kwargs':kwargs}
        barrier.wait()
        record['started']=time.time()
        try:
            request=urllib.request.Request(base+'/v1/chat/completions',method='POST',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=timeout) as response:
                payload=json.loads(response.read())
            record.update(http_status=response.status,prompt_tokens=payload.get('usage',{}).get('prompt_tokens'),
                          completion_tokens=payload.get('usage',{}).get('completion_tokens'))
        except BaseException as exc:
            record.update(http_status=None,error=type(exc).__name__,detail=str(exc)[:300])
        record['finished']=time.time();records.append(record)
    threads=[threading.Thread(target=launch,args=(spec,prompt,i))
             for spec,prompt in zip(classes,prompts) for i in range(spec['count'])]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=timeout+120)
    service_log=run/'logs/service.log'
    text=service_log.read_text(errors='replace') if service_log.is_file() else ''
    window_start=min((r['started'] for r in records if 'started' in r),default=0)
    window_end=max((r['finished'] for r in records if 'finished' in r),default=0)
    engine=_mixed_engine_lines(text,window_start,window_end)
    reasons=[]
    if len(records)!=sum(spec['count'] for spec in classes):reasons.append('not every mixed request returned')
    if any(r.get('http_status')!=200 for r in records):reasons.append('a mixed request did not return HTTP 200')
    if any(r.get('completion_tokens')!=r.get('output_tokens_target') for r in records):
        reasons.append('completion token count diverged from the ignore_eos target')
    overlap=records and all('started' in r and 'finished' in r for r in records) and \
        max(r['started'] for r in records)<min(r['finished'] for r in records)
    if not overlap:reasons.append('prefill and decode classes did not overlap in one window')
    if not any(v['mixed'] for v in engine):reasons.append('no engine line in the window shows prompt and generation throughput together')
    result={'probe':'pd_mixed_single_instance','requests':records,'overlap':overlap,
            'window':{'start':window_start,'end':window_end},'engine_lines':engine,
            'service_log':str(service_log),'acceptance':'passed' if not reasons else 'failed',
            'reasons':reasons}
    write_json(out/'pd-mixed-probe.json',result)
    if reasons:
        state(run,'pd_mixed_probe','failed',reasons=reasons)
        raise RuntimeError('pd mixed probe failed: '+'; '.join(reasons))
    state(run,'pd_mixed_probe','passed',requests=len(records),mixed_engine_lines=sum(v['mixed'] for v in engine))
    return result


def _pd_metric_kind(name):
    label=str(name).split('{',1)[0].lower()
    if re.search(r'(?:^|:)kv_(?:(?:pull|push|transfer)_(?:success|completed)(?:_(?:bytes|blocks|requests))?|transfer_(?:bytes|blocks))_total$',label):
        return 'completion'
    if label.endswith('_total') and re.search(r'(?:^|:)kv_(?:transfer|pull|push|link|load|recompute|fallback)_',label) \
            and any(tag in label for tag in ('fail','error','timeout','reject','recomput','fallback')):
        return 'failure'
    return None


def _kv_metric_deltas(before,after):
    """Prometheus-style kv counter deltas from /metrics_snapshot bodies."""
    def counters(text):
        values={}
        for line in text.splitlines():
            fields=line.split()
            if len(fields)==2 and _pd_metric_kind(fields[0]) is not None:
                try:values[fields[0]]=float(fields[1])
                except ValueError:pass
        return values
    deltas={}
    for role in ('producer','consumer'):
        b,a=counters(before.get(role,'')),counters(after.get(role,''))
        deltas[role]={name:a[name]-b.get(name,0) for name in a if a[name]!=b.get(name,0)}
    return deltas


def _connector_log_evidence(log_path):
    """Connector activity lines (registration/pull/push) from an engine log."""
    try:
        text=Path(log_path).read_text(errors='replace')
    except OSError:
        return []
    hits=[]
    for line in text.splitlines():
        if any(tag in line for tag in ('Llmdatadist','LLMDataDist','llmdatadist','pull_block','pull_done','worker[','register')):
            if 'INFO' in line or 'WARNING' in line or 'Error' in line:
                hits.append(line.strip()[:220])
    return hits[-12:]


def _has_positive_kv_metric(values):
    for name,value in values.items():
        # Transfer-specific counters only: memory bytes, prefill/recompute
        # histograms and failure counters do not prove a completed transfer.
        if math.isfinite(value) and value>0 and _pd_metric_kind(name)=='completion':
            return True
    return False


def _has_kv_transfer_log(lines):
    # Bare pull_done is also emitted on failure by this runtime. Require an
    # explicit successful completion; config/factory/register/submit is not one.
    return any(re.search(r'\b(?:pull_done|push_done|transfer_done|kv_pull_completed|kv_transfer_completed)\b',line.lower())
               and re.search(r'\b(?:success=true|status=success)\b',line.lower())
               and not any(tag in line.lower() for tag in ('fail','error','timeout','reject','register','factory','submit'))
               for line in lines)


def _pd_log_failure(line):
    lower=line.lower()
    if any(tag in lower for tag in ('llm_link_failed','link failed','link fail:',
                                     'kv load failure','kv load failed','err pull blocks',
                                     'err pull_blocks','err send pull done','err in task_pull_kv')):
        return True
    if ('kv' in lower or 'cache' in lower) and any(tag in lower for tag in ('recomput','fallback','rescheduled')):
        return True
    match=re.search(r'\bfail_pull\s*[:=]\s*(\d+(?:\.\d*)?(?:e[+\-]?\d+)?)',lower)
    return bool(match and float(match.group(1))>0)


def _pd_log_window(path,start):
    """Scan only appended bytes; retain bounded evidence but count every failure."""
    activity=[];failures=[];count=0
    try:
        with Path(path).open('rb') as stream:
            info=os.fstat(stream.fileno())
            if (start is not None and (info.st_dev,info.st_ino)!=tuple(start['identity'])) or info.st_size<(start or {}).get('offset',0):
                return {'activity':[], 'failure_count':0,'failure_lines':[], 'invalid_window':'log rotated or truncated'}
            offset=(start or {}).get('offset',0);stream.seek(offset)
            for raw in stream:
                line=raw.decode('utf-8',errors='replace').strip()
                if _pd_log_failure(line):
                    count+=1
                    if len(failures)<12:failures.append(line[:600])
                if any(tag in line.lower() for tag in ('llmdatadist','pull_block','pull_done','push_block','push_done','transfer_done','kv_transfer')):
                    activity.append(line[:600]);activity=activity[-12:]
            return {'activity':activity,'failure_count':count,'failure_lines':failures,
                    'offset_start':offset,'offset_end':stream.tell(),'identity':[info.st_dev,info.st_ino]}
    except OSError as exc:
        return {'activity':[],'failure_count':0,'failure_lines':[],'invalid_window':str(exc)}


def _pd_failure_reasons(deltas,windows):
    reasons=[]
    for role in ('producer','consumer'):
        for name,value in deltas[role].items():
            if not math.isfinite(value) or value<0:
                reasons.append(f'{role} KV counter nonfinite or reset: {name}')
            elif value>0 and any(tag in name.lower() for tag in ('fail','error','timeout','reject','recomput','fallback')):
                reasons.append(f'{role} KV failure counter increased: {name} (+{value})')
        window=windows[role]
        if window.get('invalid_window'):reasons.append(f'{role} log window invalid: '+window['invalid_window'])
        if window['failure_count']:reasons.append(f'{role} new KV link/pull/recompute failure log lines: {window["failure_count"]}')
    return reasons


PREFILL_REQUEST_ID_HEADER = 'X-Fullnet-Prefill-Request-Id'
PREFILL_ACK_ENDPOINT_HEADER = 'X-Fullnet-Prefill-Ack-Endpoint'
_PD_OBSERVER_COUNTERS = {'consumer':'kv_pull_completed_requests_total',
                         'producer':'kv_producer_done_ack_total'}


def _pd_observer_parser():
    # Default/legacy probes never import this optional dependency.
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError as exc:
        raise RuntimeError("observer mode requires fullnet[metrics]; install this checkout with "
                           "python -m pip install -e '.[metrics]'") from exc
    return text_string_to_metric_families


def _pd_observer_counters(text,parser):
    if not isinstance(text,str):raise ValueError('metrics body must be text')
    values={}
    for family in parser(text):
        for sample in family.samples:
            if sample.name not in _PD_OBSERVER_COUNTERS.values() and _pd_metric_kind(sample.name)!='failure':
                continue
            if family.type!='counter':raise ValueError('expected Counter: '+sample.name)
            key=(sample.name,tuple(sorted(sample.labels.items())))
            if key in values:raise ValueError('duplicate metric series: '+sample.name)
            value=float(sample.value)
            if not math.isfinite(value) or value<0:raise ValueError('nonfinite or negative Counter: '+sample.name)
            values[key]=value
    return values


def _pd_observer_deltas(before,after):
    parser=_pd_observer_parser();deltas={};reasons=[]
    for role in ('producer','consumer'):
        deltas[role]=[]
        try:
            b=_pd_observer_counters(before[role],parser);a=_pd_observer_counters(after[role],parser)
            if b.keys()-a.keys():raise ValueError('Counter series disappeared')
            for (name,labels),value in a.items():
                previous=b.get((name,labels),0);delta=value-previous
                if delta<0:raise ValueError('Counter reset: '+name)
                deltas[role].append({'name':name,'labels':dict(labels),'before':previous,'delta':delta})
                if _pd_metric_kind(name)=='failure' and delta>0:
                    reasons.append(f'{role} KV failure counter increased: {name} (+{delta})')
        except (KeyError,TypeError,ValueError) as exc:
            reasons.append(f'{role} invalid observer metrics: {exc}')
    return deltas,reasons


def _pd_ack_endpoint(value,*,runtime=False):
    """Validate an ASCII TCP authority; never infer a port or resolve a host."""
    if not isinstance(value,str) or len(value)>512 or not value.isascii():
        raise ValueError('ack endpoint must be a bounded ASCII TCP endpoint')
    if runtime and not value.startswith('tcp://'):
        raise ValueError('runtime ack endpoint must use tcp://')
    authority=value.removeprefix('tcp://')
    match=re.fullmatch(r'(?:\[[0-9a-fA-F:.]+\]|[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?):([0-9]{1,5})',authority)
    if not match or not 1<=int(match.group(1))<=65535:
        raise ValueError('invalid ack host:port (no whitespace, URL path or userinfo allowed)')
    host=authority[:authority.rfind(':')]
    if host.startswith('[') or re.fullmatch(r'[0-9.]+',host):
        import ipaddress
        ipaddress.ip_address(host.strip('[]'))
    elif any(not re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?',part) for part in host.split('.')):
        raise ValueError('invalid ack hostname')
    return authority


def _pd_handoff_headers(headers):
    """HTTPMessage.get_all preserves duplicate headers, unlike .get()."""
    result={}
    for name in (PREFILL_REQUEST_ID_HEADER,PREFILL_ACK_ENDPOINT_HEADER):
        values=headers.get_all(name,[]) if hasattr(headers,'get_all') else ([headers[name]] if name in headers else [])
        result[name]=values
    return result


def pd_observer_request_evidence(before,after,*,prefill_req_id,ack_endpoint,ack_addr=None):
    """Join one relay request to real consumer completion and producer ack.

    parts=1 only. An ack is not a push/copy proof. This pure join alone does
    not cover transient link failures or recompute; the probe also needs full
    bounded log windows. Labels are decoded by the official Prometheus parser.
    """
    deltas,reasons=_pd_observer_deltas(before,after)
    if (not isinstance(prefill_req_id,str) or not prefill_req_id or prefill_req_id!=prefill_req_id.strip()
            or any(ord(char)<32 or ord(char)==127 for char in prefill_req_id)):
        reasons.append('missing or invalid relay prefill request ID header')
    discovered=None
    try:
        discovered=_pd_ack_endpoint(ack_endpoint,runtime=True)
        if ack_addr is not None and _pd_ack_endpoint(ack_addr)!=discovered:
            reasons.append('runtime handoff ack endpoint differs from configured expected ack_addr')
    except ValueError as exc:reasons.append(str(exc))
    labels={'prefill_req_id':prefill_req_id,'parts':'1',
            'ack_addr':discovered}
    for role,name in _PD_OBSERVER_COUNTERS.items():
        rows=[row for row in deltas[role] if row['name']==name]
        matched=[row for row in rows if row['labels']==labels]
        if len(matched)!=1 or matched[0]['delta']!=1 or matched[0]['before']!=0:
            reasons.append(f'{role} requires one fresh request-bound {name} increment with parts=1 and matching ack_addr')
        if any(row['delta']>0 and row['labels']!=labels for row in rows):
            reasons.append(f'{role} unrelated or mismatched completion in request window')
    return {'proof_status':'PROVEN' if not reasons else 'NOT_PROVEN','reasons':reasons,
            'prefill_req_id':prefill_req_id,'handoff_ack_endpoint':ack_endpoint,
            'configured_ack_addr':ack_addr,'expected_labels':labels,'counter_deltas':deltas}


def _pd_observer_snapshot(base,options):
    urls=options.get('metrics_urls')
    if urls is None:
        with urllib.request.urlopen(base+'/metrics_snapshot',timeout=120) as response:
            if response.status!=200:raise ValueError('metrics_snapshot did not return HTTP 200')
            raw=response.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:raise ValueError('metrics snapshot exceeds 2 MiB')
        data=json.loads(raw)
    else:
        if not isinstance(urls,dict) or set(urls)!=set(_PD_OBSERVER_COUNTERS):
            raise ValueError('metrics_urls requires producer and consumer URLs')
        data={}
        for role,url in urls.items():
            if not isinstance(url,str) or not url.startswith(('http://','https://')):
                raise ValueError('invalid role metrics URL')
            with urllib.request.urlopen(url,timeout=120) as response:
                if response.status!=200:raise ValueError(role+' metrics did not return HTTP 200')
                raw=response.read(2*1024*1024+1)
            if len(raw)>2*1024*1024:raise ValueError('role metrics exceeds 2 MiB')
            data[role]=raw.decode('utf-8')
    if not isinstance(data,dict) or any(not isinstance(data.get(r),str) for r in _PD_OBSERVER_COUNTERS):
        raise ValueError('metrics snapshot must contain both role text bodies')
    return data


def _pd_split_observer_probe(run,c,options):
    """Explicit observer mode; remote-only failure metrics are NOT_PROVEN.

    measurements.pd_split_observer = {enabled: true, ack_source: 'runtime_handoff',
      failure_evidence: 'local_logs', relay_base_url: 'http://relay:port',
      metrics_urls: {producer: ..., consumer: ...}}
    metrics_urls is optional for same-host /metrics_snapshot. relay_base_url
    bypasses a front that strips response headers; it still calls the real
    relay, not either model role directly. Complete local
    producer/consumer logs remain mandatory until remote failure coverage is
    implemented and independently verified. ack_addr is an optional extra
    restriction, not a source for discovering the endpoint. Never falls back
    to legacy proof.
    """
    run=Path(run);out=run/'performance';out.mkdir(exist_ok=True)
    result={'probe':'same_host_pd_split','mode':'observer','proof_status':'NOT_PROVEN',
            'acceptance':'failed','requests':[],'reasons':[]}
    reasons=result['reasons'];paths={r:run/'logs'/f'pd-split-{r}.log' for r in _PD_OBSERVER_COUNTERS}
    starts={};whole_before=None
    state(run,'pd_split_probe','running')
    try:
        _pd_observer_parser()
        if options.get('failure_evidence')!='local_logs':
            raise ValueError('NOT_PROVEN: full remote link/pull/recompute failure coverage is not implemented; local_logs required')
        if options.get('ack_source')!='runtime_handoff':
            raise ValueError('observer mode requires frozen ack_source=runtime_handoff')
        if 'ack_addr' in options:_pd_ack_endpoint(options['ack_addr'])
        if any(type(c.get('identity',{}).get(k)) is not int or c['identity'][k]!=1 for k in ('tp','dp','pp')):
            raise ValueError('observer mode currently requires explicit TP=DP=PP=1 (parts=1)')
        for role,path in paths.items():
            info=path.stat();starts[role]={'identity':[info.st_dev,info.st_ino],'offset':info.st_size}
        root=run.parents[3];env=_bench_environment(run,c,root)
        python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
        relay=options.get('relay_base_url',base)
        if not isinstance(relay,str) or not relay.startswith(('http://','https://')):
            raise ValueError('relay_base_url must be the real HTTP relay endpoint')
        relay=relay.rstrip('/').removesuffix('/v1')
        kwargs=c['evaluation'].get('chat_template_kwargs',{})
        whole_before=_pd_observer_snapshot(base,options)
        for name,input_tokens,output_tokens in [('prefill_heavy',4096,64),('decode_heavy',128,1024)]:
            folder=out/('pd-split-'+name);folder.mkdir(exist_ok=True)
            record={'request_class':name,'input_tokens_target':input_tokens,'output_tokens_target':output_tokens}
            result['requests'].append(record);dataset=folder/'input.jsonl'
            prepared=subprocess.run([python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),
                '--model',c['identity']['weights'],'--tokens',str(input_tokens),'--output',str(dataset),
                '--kwargs',json.dumps(kwargs)],cwd=root,env=env,capture_output=True,text=True,timeout=300)
            (folder/'prepare.stdout').write_text(prepared.stdout,encoding='utf-8')
            (folder/'prepare.stderr').write_text(prepared.stderr,encoding='utf-8')
            if prepared.returncode:raise RuntimeError('split probe prompt preparation failed: '+name)
            prompt=json.loads(dataset.read_text().splitlines()[0])['prompt']
            before=_pd_observer_snapshot(base,options)
            write_json(folder/'metrics-before.json',before)
            body={'model':c['service']['served_model'],'messages':[{'role':'user','content':prompt}],
                  'max_tokens':output_tokens,'temperature':0,'seed':0,'ignore_eos':True,'chat_template_kwargs':kwargs}
            request=urllib.request.Request(relay+'/v1/chat/completions',method='POST',data=json.dumps(body).encode(),
                                           headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=c['service'].get('request_timeout',300)) as response:
                raw=response.read(8*1024*1024+1)
                if len(raw)>8*1024*1024:raise ValueError('probe response exceeds 8 MiB')
                (folder/'response.json').write_bytes(raw);payload=json.loads(raw)
                header_values=_pd_handoff_headers(response.headers);record['handoff_headers']=header_values
                if any(len(values)!=1 for values in header_values.values()):
                    raise ValueError('missing or duplicate runtime handoff response header')
                record.update(http_status=response.status,prefill_req_id=header_values[PREFILL_REQUEST_ID_HEADER][0],
                              ack_endpoint=header_values[PREFILL_ACK_ENDPOINT_HEADER][0],
                              completion_tokens=payload.get('usage',{}).get('completion_tokens'))
            after=_pd_observer_snapshot(base,options);write_json(folder/'metrics-after.json',after)
            proof=pd_observer_request_evidence(before,after,prefill_req_id=record['prefill_req_id'],
                ack_endpoint=record['ack_endpoint'],ack_addr=options.get('ack_addr'))
            record['transfer_evidence']=proof;reasons.extend(proof['reasons'])
            if record['http_status']!=200 or record['completion_tokens']!=output_tokens:
                reasons.append('request HTTP status or exact completion token contract failed: '+name)
            windows={r:_pd_log_window(p,starts[r]) for r,p in paths.items()}
            reasons.extend(_pd_failure_reasons({r:{} for r in paths},windows))
            if reasons:break
    except Exception as exc:
        reasons.append(f'{type(exc).__name__}: {exc}')
    finally:
        if whole_before is not None:
            try:
                whole_after=_pd_observer_snapshot(base,options)
                write_json(out/'pd-observer-window-metrics.json',{'before':whole_before,'after':whole_after})
                result['window_counter_deltas'],failures=_pd_observer_deltas(whole_before,whole_after)
                reasons.extend(failures)
            except Exception as exc:reasons.append('whole-window metrics failed: '+str(exc))
        if len(starts)==2:
            result['connector_log_windows']={r:_pd_log_window(p,starts[r]) for r,p in paths.items()}
            reasons.extend(_pd_failure_reasons({r:{} for r in paths},result['connector_log_windows']))
    if len(result['requests'])!=2:reasons.append('not every split request proved transfer')
    result['reasons']=list(dict.fromkeys(reasons))
    if not reasons:result.update(acceptance='passed',proof_status='PROVEN')
    write_json(out/'pd-split-probe.json',result)
    state(run,'pd_split_probe',result['acceptance'],requests=len(result['requests']),reasons=result['reasons'])
    if reasons:raise RuntimeError('pd split observer probe failed: '+'; '.join(result['reasons']))
    return result


def pd_split_probe(run,c):
    """Real KV-transfer evidence for the same-host PD split case.

    Sends one prefill-heavy and one decode-heavy request through the split
    front so the producer prefills, LLMDataDistConnector transfers KV and the
    consumer decodes. Acceptance: both requests complete with exact token
    counts AND kv-transfer activity on each instance (positive kv metric
    deltas on /metrics of both sides, or connector activity lines in both
    engine logs). Any new link/pull failure or recompute fallback vetoes
    acceptance, including when positive transfer counters also increase.
    Evidence recorded to performance/pd-split-probe.json.
    """
    options=c.get('measurements',{}).get('pd_split_observer')
    if options is not None:
        if not isinstance(options,dict) or type(options.get('enabled')) is not bool:
            raise ValueError('pd_split_observer requires an explicit boolean enabled')
        if options['enabled']:return _pd_split_observer_probe(run,c,options)
    run=Path(run);out=run/'performance';out.mkdir(exist_ok=True)
    root=run.parents[3];env=_bench_environment(run,c,root)
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    timeout=c['service'].get('request_timeout',300)
    classes=[{'name':'prefill_heavy','input':4096,'output':64},
             {'name':'decode_heavy','input':128,'output':1024}]
    state(run,'pd_split_probe','running')
    log_paths={role:run/'logs'/('pd-split-%s.log'%role) for role in ('producer','consumer')}
    log_starts={}
    for role,path in log_paths.items():
        try:
            info=path.stat();log_starts[role]={'identity':[info.st_dev,info.st_ino],'offset':info.st_size}
        except FileNotFoundError:log_starts[role]=None
    metrics_before=json.loads(urllib.request.urlopen(base+'/metrics_snapshot',timeout=120).read())
    requests=[]
    for spec in classes:
        folder=out/('pd-split-'+spec['name']);folder.mkdir(parents=True,exist_ok=True)
        dataset=folder/'input.jsonl'
        prepared=subprocess.run([python,'-B',str(root/'scripts/evaluation/prepare_bench_data.py'),'--model',
            c['identity']['weights'],'--tokens',str(spec['input']),'--output',str(dataset),
            '--kwargs',json.dumps(kwargs)],cwd=root,env=env,capture_output=True,text=True,timeout=300)
        if prepared.returncode:raise RuntimeError('split probe prompt preparation failed for '+spec['name'])
        prompt=json.loads(dataset.read_text().splitlines()[0])['prompt']
        body={'model':c['service']['served_model'],'messages':[{'role':'user','content':prompt}],
              'max_tokens':spec['output'],'temperature':0,'seed':0,'ignore_eos':True,'chat_template_kwargs':kwargs}
        record={'request_class':spec['name'],'input_tokens_target':spec['input'],'output_tokens_target':spec['output']}
        started=time.time()
        try:
            request=urllib.request.Request(base+'/v1/chat/completions',method='POST',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=timeout) as response:
                payload=json.loads(response.read())
            record.update(http_status=response.status,
                prompt_tokens=payload.get('usage',{}).get('prompt_tokens'),
                completion_tokens=payload.get('usage',{}).get('completion_tokens'))
        except BaseException as exc:
            record.update(http_status=None,error=type(exc).__name__,detail=str(exc)[:300])
        record['finished']=time.time();record['elapsed_s']=round(record['finished']-started,3)
        requests.append(record)
    metrics_after=json.loads(urllib.request.urlopen(base+'/metrics_snapshot',timeout=120).read())
    deltas=_kv_metric_deltas(metrics_before,metrics_after)
    windows={role:_pd_log_window(path,log_starts[role]) for role,path in log_paths.items()}
    logs={role:window['activity'] for role,window in windows.items()}
    reasons=_pd_failure_reasons(deltas,windows)
    if len(requests)!=len(classes):reasons.append('not every split request returned')
    if any(r.get('http_status')!=200 for r in requests):reasons.append('a split request did not return HTTP 200')
    if any(r.get('completion_tokens')!=r.get('output_tokens_target') for r in requests):
        reasons.append('completion token count diverged from the ignore_eos target')
    kv_metrics=all(_has_positive_kv_metric(deltas[role]) for role in ('producer','consumer'))
    kv_logs=all(_has_kv_transfer_log(logs[role]) for role in ('producer','consumer'))
    if not (kv_metrics or kv_logs):
        reasons.append('no kv-transfer activity observed (no kv metric delta on both sides and no connector log lines)')
    result={'probe':'same_host_pd_split','requests':requests,'kv_metric_deltas':deltas,
            'connector_log_lines':logs,'connector_log_windows':windows,
            'acceptance':'passed' if not reasons else 'failed','reasons':reasons}
    write_json(out/'pd-split-probe.json',result)
    if reasons:
        state(run,'pd_split_probe','failed',reasons=reasons)
        raise RuntimeError('pd split probe failed: '+'; '.join(reasons))
    state(run,'pd_split_probe','passed',requests=len(requests))
    return result


def collect(run,c,result):
    """Legacy order (performance then profile); new callers use the split stages."""
    collect_performance(run,c)
    collect_profile(run,c)
