"""Official service benchmark and server-side NPU profiling after quality requests."""
import csv,io,json,os,re,shutil,sqlite3,stat,subprocess,threading,time,urllib.request
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
        if bin_src.is_symlink() or (src/'tools/profiler').is_symlink():
            return {'shadow':None,'reason':'toolkit entry is a symlink; source remains read-only'}
        if shadow.exists():
            if shadow.is_symlink() or not shadow.resolve(strict=True).is_relative_to(run):
                return {'shadow':None,'reason':'shadow path escapes the experiment run'}
            shutil.rmtree(shadow)
        shadow.mkdir(parents=True,exist_ok=True)
        shutil.copytree(src/'tools/profiler',shadow/'tools/profiler',symlinks=True)
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


def prepare_profiler_tools(run,c):
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
    home=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    if os.name != 'posix':
        record={'home':home,'service_euid':None,'real_msprof_uid':None,
                'inspection':ensure_msprof_tool_owner(),'prepared_before_launch':True,
                'shadow_toolkit':{'shadow':None,'reason':'POSIX profiler ownership is not applicable'},
                'env_updates':{}}
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
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError,UnicodeError):
        return 0
    events=data.get('traceEvents',data.get('events',[])) if isinstance(data,dict) else data
    if not isinstance(events,list):return 0
    process_names={}
    for event in events:
        if isinstance(event,dict) and event.get('name') in ('process_name','process_labels'):
            args=event.get('args') or {};process_names[event.get('pid')]=str(args.get('name',''))
    count=0
    for event in events:
        if not isinstance(event,dict) or not event.get('name'):continue
        cat=str(event.get('cat','')).lower();name=str(event.get('name','')).lower()
        args=event.get('args') or {};process=str(process_names.get(event.get('pid'),'' )).lower()
        if 'cpu' in cat:continue
        try:duration=float(event.get('dur',event.get('duration',0)))
        except (TypeError,ValueError):continue
        if duration<=0:continue
        task_values=' '.join(str(v).upper() for k,v in args.items() if 'task' in str(k).lower() or 'type' in str(k).lower())
        task_device=any(tag in task_values for tag in ('AI_CORE','AICORE','AIVEC','AI_VEC','AI_CUBE','AICUBE'))
        keys={str(k).strip().lower().replace(' ','_') for k in args}
        device_id=any(k in keys for k in ('device_id','deviceid','device'))
        process_device=any(tag in process for tag in ('ascend hardware','ascend device','device'))
        structural_device=task_device or device_id or process_device
        if not structural_device:continue
        kernel='kernel' in cat or 'kernel' in name or any(tag in cat for tag in ('device','npu','ascend hardware','aicore','aiv'))
        if kernel:count+=1
    return count


def _csv_device_kernel_rows(path):
    count=0
    try:
        with path.open('r',encoding='utf-8-sig',newline='',errors='replace') as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames:return 0
            fields={str(k).strip().lower().replace(' ','_'):k for k in reader.fieldnames}
            kernel_key=next((v for k,v in fields.items() if 'kernel' in k or k in ('op_name','task_name','name')),None)
            duration_key=next((v for k,v in fields.items() if 'duration' in k or 'execution_time' in k),None)
            task_key=next((v for k,v in fields.items() if 'task_type' in k or 'tasktype' in k
                           or k in ('accelerator_core','accelerator_type','core_type','type')),None)
            device_key=next((v for k,v in fields.items() if k in ('device_id','deviceid','device')),None)
            if not duration_key or not (task_key or device_key):return 0
            for row in reader:
                kernel=str(row.get(kernel_key,'device-row')).strip() if kernel_key else 'device-row'
                if not kernel or 'cpu' in kernel.lower():continue
                task=str(row.get(task_key,'')).upper() if task_key else ''
                device=str(row.get(device_key,'')).strip() if device_key else ''
                if 'AICPU' in task or task in {'CPU','HOST'}:continue
                if not (any(tag in task for tag in ('AI_CORE','AICORE','AIVEC','AI_VEC','AI_CUBE','AICUBE',
                                                    'VECTOR_CORE','CUBE')) or device):continue
                try:duration=float(str(row.get(duration_key,'')).replace(',',''))
                except (TypeError,ValueError):continue
                if duration>0:count+=1
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


def _offline_parse_profiles(run):
    """Offline msprof import of the captured *_ascend_pt rank trees.

    Daemonized workers may not run the in-service torch_npu parser because the
    profiling data cannot be parsed from that process context while the
    raw device data on disk is valid. msprof import needs a writable
    collection dir and the captured trees may be owned by the service identity,
    so each rank tree is copied under performance/parse-offline/ first.
    CPU-only: no device access, no service interaction; bounded per rank.
    """
    run=Path(run).resolve(strict=True);out=run/'performance'
    try:record=read_json(out/'profiler-tools.json')
    except (OSError,ValueError):return {'skipped':'no profiler-tools.json'}
    msprof_py=(record.get('shadow_toolkit') or {}).get('msprof_py')
    if not msprof_py:return {'skipped':'no shadow msprof.py'}
    try:
        msprof_path=Path(msprof_py).resolve(strict=True)
        if not msprof_path.is_relative_to(run) or not msprof_path.is_file():
            return {'skipped':'shadow msprof.py is outside the experiment run'}
    except (OSError,RuntimeError):return {'skipped':'no shadow msprof.py'}
    try:python=read_json(run/'config.json')['service']['argv'][0]
    except (OSError,ValueError,KeyError):return {'skipped':'no service python'}
    env={'HOME':str(run/'home'),'ASCEND_HOME_PATH':record.get('home')}
    env.update(record.get('env_updates') or {})
    work=out/'parse-offline';work.mkdir(parents=True,exist_ok=True)
    parsed=[];first_error=None
    for rank_dir in sorted((run/'profiles').glob('*_ascend_pt')):
        result=rank_dir.name
        try:
            copy=work/rank_dir.name
            if rank_dir.is_symlink() or not rank_dir.resolve(strict=True).is_relative_to(run):
                raise ValueError('profile rank tree escapes the experiment run')
            if copy.exists():
                if copy.is_symlink() or not copy.resolve(strict=True).is_relative_to(run):
                    raise ValueError('offline parse copy escapes the experiment run')
            else:shutil.copytree(rank_dir,copy,symlinks=True)
            prof=next(iter(sorted(copy.glob('PROF_*'))))
            completed=subprocess.run([python,'-B',str(msprof_path),'import','-dir',str(prof)],
                                     cwd=work,capture_output=True,text=True,timeout=900)
            if completed.returncode:
                error_text=(completed.stderr or '')+'\n'+(completed.stdout or '')
                first_error=first_error or _first_error(error_text) or f'command exited {completed.returncode}'
                result+=f':exit-{completed.returncode}'
            else:
                result+=':passed'
        except (OSError,subprocess.TimeoutExpired,StopIteration,ValueError) as exc:
            first_error=first_error or f'{type(exc).__name__}: {exc}'
            result+=':'+type(exc).__name__
        parsed.append(result)
    return {'workdir':str(work),'ranks':parsed,'first_error':first_error,
            'acceptance':'parsed' if parsed and not first_error else 'not_accepted'}


def _first_error(text):
    for line in str(text).splitlines():
        stripped=line.strip()
        if stripped and re.search(r'\b(error|exception|traceback|failed)\b',stripped,re.I):
            return stripped[:500]
    return next((line.strip()[:500] for line in str(text).splitlines() if line.strip()),None)


def profile_artifact_index(run,extra_dirs=()):
    root=Path(run);profiles=root/'profiles';items=[]
    candidates=[]
    if profiles.exists():
        candidates.extend((p,'kernel_csv') for p in profiles.rglob('*')
                          if p.is_file() and p.suffix.lower()=='.csv' and
                          (p.name.lower() in {'device.csv','device_details.csv','kernel_details.csv'} or
                           p.name.lower().startswith(('op_summary','ascend_op_summary'))))
        candidates.extend((p,'trace') for p in profiles.rglob('trace_view.json'))
    for extra in extra_dirs:
        extra=Path(extra)
        if extra.exists():
            candidates.extend((p,'kernel_csv') for p in extra.rglob('*.csv'))
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
        if not index['has_device_kernel_evidence']:
            # Daemon-mode DP workers cannot run the in-service torch_npu parser
            # parse the captured *_ascend_pt trees offline instead of
            # failing with zero artifacts while valid device data is on disk.
            offline=_offline_parse_profiles(run)
            write_json(out/'profile-offline-parse.json',offline)
            extras=[Path(offline['workdir'])] if offline.get('workdir') else []
            index=profile_artifact_index(run,extra_dirs=extras)
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


def _kv_metric_deltas(before,after):
    """Prometheus-style kv counter deltas from /metrics_snapshot bodies."""
    def counters(text):
        values={}
        for line in text.splitlines():
            fields=line.split()
            if len(fields)==2 and 'kv' in fields[0].lower():
                try:values[fields[0]]=float(fields[1])
                except ValueError:pass
        return values
    deltas={}
    for role in ('producer','consumer'):
        b,a=counters(before.get(role,'')),counters(after.get(role,''))
        deltas[role]={name:a[name]-b[name] for name in a if name in b and a[name]!=b[name]}
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
        label=str(name).lower()
        if value>0 and any(tag in label for tag in ('byte','block','success','transfer','pull','push')) \
                and not any(tag in label for tag in ('fail','error','timeout','reject')):
            return True
    return False


def _has_kv_transfer_log(lines):
    return any(any(tag in line.lower() for tag in ('pull_block','pull_done','push_block',
                                                    'push_done','kv_transfer','transfer_done'))
               and not any(tag in line.lower() for tag in ('fail','error','timeout','reject'))
               for line in lines)


def pd_split_probe(run,c):
    """Real KV-transfer evidence for the same-host PD split case.

    Sends one prefill-heavy and one decode-heavy request through the split
    front so the producer prefills, LLMDataDistConnector transfers KV and the
    consumer decodes. Acceptance: both requests complete with exact token
    counts AND kv-transfer activity on each instance (positive kv metric
    deltas on /metrics of both sides, or connector activity lines in both
    engine logs). Evidence recorded to performance/pd-split-probe.json.
    """
    run=Path(run);out=run/'performance';out.mkdir(exist_ok=True)
    root=run.parents[3];env=_bench_environment(run,c,root)
    python=c['service']['argv'][0];base=c['service']['base_url'].removesuffix('/v1')
    kwargs=c['evaluation'].get('chat_template_kwargs',{})
    timeout=c['service'].get('request_timeout',300)
    classes=[{'name':'prefill_heavy','input':4096,'output':64},
             {'name':'decode_heavy','input':128,'output':1024}]
    state(run,'pd_split_probe','running')
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
    logs={role:_connector_log_evidence(run/'logs'/('pd-split-%s.log'%role)) for role in ('producer','consumer')}
    reasons=[]
    if len(requests)!=len(classes):reasons.append('not every split request returned')
    if any(r.get('http_status')!=200 for r in requests):reasons.append('a split request did not return HTTP 200')
    if any(r.get('completion_tokens')!=r.get('output_tokens_target') for r in requests):
        reasons.append('completion token count diverged from the ignore_eos target')
    kv_metrics=all(_has_positive_kv_metric(deltas[role]) for role in ('producer','consumer'))
    kv_logs=all(_has_kv_transfer_log(logs[role]) for role in ('producer','consumer'))
    if not (kv_metrics or kv_logs):
        reasons.append('no kv-transfer activity observed (no kv metric delta on both sides and no connector log lines)')
    result={'probe':'same_host_pd_split','requests':requests,'kv_metric_deltas':deltas,
            'connector_log_lines':logs,'acceptance':'passed' if not reasons else 'failed','reasons':reasons}
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
