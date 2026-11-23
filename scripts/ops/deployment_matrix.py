"""Plan, submit, refresh and audit the Pangu deployment validation matrix.

Only A5 single-host TP/DP cases currently submit through the reviewed native
runner. Other modes remain explicitly blocked until a real platform adapter is
implemented and the tool-enabled agent supplies real execution evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops import recover_framework

MATRIX_FILE = "deployment-matrix.json"
PARALLEL = ((2, 1), (4, 1), (2, 2), (1, 4))
GRAPHS = ("eager", "full_decode_only")
MTP = (0, 3)
PERFORMANCE = {"requests": 32, "warmups": 2, "workload_groups": 6}
PERFORMANCE_WORKLOADS = [
    {"input": 128, "output": 128, "concurrency": 1},
    {"input": 4096, "output": 128, "concurrency": 1},
    {"input": 15999, "output": 128, "concurrency": 1},
    {"input": 128, "output": 1024, "concurrency": 1},
    {"input": 4096, "output": 1024, "concurrency": 4},
    {"input": 128, "output": 4096, "concurrency": 4},
]


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", value):
        raise ValueError("invalid deployment matrix task identifier")
    return value


def _case_id(*parts):
    value = "-".join(str(part).lower().replace("_", "-") for part in parts)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value):
        raise ValueError("generated case identifier is invalid")
    return value


def _single_case(platform, tp, dp, graph, mtp):
    supported = platform == "a5"
    return {"case_id": _case_id(platform, "single", f"tp{tp}dp{dp}", graph, f"mtp{mtp}"),
            "model": "pangu-92b", "platform": platform, "scenario": "single_host_tpdp", "placement": "single_host",
            "hosts_required": 1, "required_npus": tp * dp,
            "parallel": {"tp": tp, "dp": dp, "pp": 1}, "graph": graph, "mtp": mtp,
            "launch_contract": {"mode": "native_pangu92b", "fixed_evaluation_samples": 48,
                                "max_batched_tokens": 10240, "max_seqs": 56,
                                "disable_aligned_decode": True,
                                **PERFORMANCE, "workloads": PERFORMANCE_WORKLOADS,
                                "profile": "real_device_kernel_required"},
            "status": "planned" if supported else "blocked",
            "reason": None if supported else "A3 runner/platform adapter is not implemented; no A5 result may stand in for A3",
            "attempts": []}


def build_matrix(task_id):
    task_id = _id(task_id)
    cases = []
    for platform in ("a3", "a5"):
        # The A5 single-card 7B runner (scripts/ops/run_pangu7b_matrix.py) joins
        # the full acceptance chain; A3 still has no vLLM runtime root.
        cases.append({"case_id": _case_id(platform, "pangu7b-single-card"), "model": "pangu-7b",
                      "platform": platform, "scenario": "pangu7b_single_card", "placement": "single_host",
                      "hosts_required": 1, "required_npus": 1, "parallel": {"tp": 1, "dp": 1, "pp": 1},
                      "graph": "eager", "mtp": 0,
                      "launch_contract": {"mode": "pangu7b-single-card", "fixed_evaluation_samples": 48,
                                          **PERFORMANCE, "workloads": PERFORMANCE_WORKLOADS,
                                          "profile": "real_device_kernel_required"},
                      "status": "planned" if platform == "a5" else "blocked",
                      "reason": None if platform == "a5" else "A3 provides no pangu-deploy vLLM runtime root; validator quality support alone does not satisfy matrix acceptance",
                      "attempts": []})
        for tp, dp in PARALLEL:
            for graph in GRAPHS:
                for mtp in MTP:
                    cases.append(_single_case(platform, tp, dp, graph, mtp))
        for scenario, parallel, split in (
            ("multi_host_tp", {"tp": 4, "dp": 1, "pp": 1}, [2, 2]),
            ("multi_host_dp", {"tp": 1, "dp": 4, "pp": 1}, [2, 2]),
        ):
            for graph in GRAPHS:
                for mtp in MTP:
                    cases.append({"case_id": _case_id(platform, scenario, graph, f"mtp{mtp}"),
                                  "model": "pangu-92b", "platform": platform, "scenario": scenario, "placement": "multi_host",
                                  "hosts_required": 2, "devices_per_host": split,
                                  "parallel": parallel, "graph": graph, "mtp": mtp,
                                  "launch_contract": {"mode": "distributed_adapter_required", "fixed_evaluation_samples": 48,
                                                      **PERFORMANCE, "workloads": PERFORMANCE_WORKLOADS,
                                                      "profile": "real_device_kernel_required"},
                                  "status": "blocked", "reason": "current runner has no multi-host rendezvous/host-rank adapter",
                                  "attempts": []})
        for scenario, host_count, roles in (
            ("pd_mixed_single_instance", 1, {"instance": "one service instance handles both prefill and decode"}),
            ("same_host_pd_split", 1, {"prefill": "independent service process; device group requires inventory", "decode": "independent service process; device group requires inventory"}),
            ("pd_disaggregated", 2, {"prefill": "host 0, dedicated device group", "decode": "host 1, dedicated device group"}),
        ):
            cases.append({"case_id": _case_id(platform, scenario), "model": "pangu-92b", "platform": platform,
                          "scenario": scenario,
                          "placement": {"pd_mixed_single_instance": "single_instance_mixed",
                                        "same_host_pd_split": "same_host_split",
                                        "pd_disaggregated": "disaggregated"}[scenario],
                          "hosts_required": host_count, "roles": roles,
                          "launch_contract": {"mode": "prefill_decode_adapter_required", "fixed_evaluation_samples": 48,
                                              **PERFORMANCE, "workloads": PERFORMANCE_WORKLOADS,
                                              "profile": "real_device_kernel_required"},
                          "status": "blocked", "reason": "P/D endpoints, role-specific launch and aggregate acceptance are not implemented",
                          "attempts": []})
    return {"schema_version": 1, "task": task_id, "status": "incomplete", "platforms": ["a3", "a5"],
            "acceptance_contract": {"quality_samples": 48, "performance_requests_per_group": 32,
                                    "performance_warmups_per_group": 2, "performance_groups": 6,
                                    "profile_requires_device_kernel_evidence": True,
                                    "fixture_counts_as_device_acceptance": False},
            "cases": cases}


def matrix_path(root, task_id, create=False):
    root = Path(root).resolve(strict=True)
    task_id = _id(task_id)
    task = root / "runs" / task_id
    if create:
        task.mkdir(mode=0o700, parents=True, exist_ok=True)
    task = task.resolve(strict=True)
    if not task.is_relative_to(root):
        raise ValueError("matrix task resolves outside project")
    return task, task / MATRIX_FILE


def save_matrix(path, value, *, create=False):
    path = Path(path)
    lock = path.with_name(path.name + '.lock')
    stream = lock.open('a+b')
    acquired = False
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        if os.name == 'nt':
            import msvcrt
            if lock.stat().st_size == 0:stream.write(b'0');stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        acquired = True
        if create and path.exists():
            raise FileExistsError('matrix already exists; refusing to overwrite')
        if not create and not path.exists():
            raise FileNotFoundError(path)
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if not create and previous.get('_revision',0)!=value.get('_revision',0):
            raise RuntimeError('matrix changed during refresh; reload before retrying')
        value['_revision']=previous.get('_revision',0)+1
        with temporary.open('x',encoding='utf-8') as output:
            json.dump(value,output,ensure_ascii=False,indent=2)
            output.write('\n');output.flush();os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        if acquired:
            if os.name == 'nt':
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def create_matrix(root, task_id):
    task, path = matrix_path(root, task_id, create=True)
    value = build_matrix(task_id)
    value["task_directory"] = str(task)
    save_matrix(path, value, create=True)
    return value


def load_matrix(root, task_id):
    _, path = matrix_path(root, task_id)
    return json.loads(path.read_text(encoding="utf-8")), path


def _case(matrix, case_id):
    for item in matrix["cases"]:
        if item["case_id"] == case_id:
            return item
    raise KeyError("unknown matrix case: " + str(case_id))


def acceptance_reasons(case, job_state, validation, experiment_state, config, performance, profile):
    reasons = []
    if job_state.get("status") != "passed" or job_state.get("exit") != 0:
        reasons.append("business job exit is not zero")
    expected_identity = case.get("parallel") or {}
    identity = config.get("identity", {})
    if config.get("platform") != case["platform"]:
        reasons.append("frozen experiment platform differs")
    if config.get("model") != case.get("model"):
        reasons.append("frozen model differs")
    if any(identity.get(key) != expected_identity.get(key) for key in ("tp", "dp", "pp")):
        reasons.append("frozen parallel configuration differs")
    if identity.get("graph") != case.get("graph") or identity.get("mtp") != case.get("mtp"):
        reasons.append("frozen graph/MTP configuration differs")
    if case.get("scenario") == "pd_mixed_single_instance":
        if identity.get("pd_mode") != "mixed":
            reasons.append("frozen pd mode differs from mixed single-instance")
        if validation.get("pd_mixed_probe") != "passed":
            reasons.append("same-instance mixed prefill+decode probe is not passed")
    if case.get("scenario") == "same_host_pd_split":
        if identity.get("pd_mode") != "split":
            reasons.append("frozen pd mode differs from same-host split")
        if validation.get("pd_split_probe") != "passed":
            reasons.append("same-host split KV-transfer probe is not passed")
    measure = config.get("measurements", {})
    if (measure.get("requests") != PERFORMANCE["requests"] or measure.get("warmups") != PERFORMANCE["warmups"] or
            measure.get("workloads") != PERFORMANCE_WORKLOADS):
        reasons.append("frozen performance contract differs from 32+2, six groups")
    if (validation.get("model_requests") != "completed" or validation.get("attempted") != 48 or
            validation.get("succeeded") != 48 or validation.get("failed") != 0 or
            validation.get("skipped") != 0 or validation.get("timeout") != 0 or
            validation.get("protocol_pass") is not True):
        reasons.append("fixed 48-sample evaluation did not fully pass protocol")
    phases = experiment_state.get("phases", {})
    if phases.get("evaluation", {}).get("status") != "passed":
        reasons.append("evaluation phase is not passed")
    if phases.get("performance", {}).get("status") != "passed":
        reasons.append("performance phase is not passed")
    if not isinstance(performance, list) or len(performance) != PERFORMANCE["workload_groups"]:
        reasons.append("six real performance summaries are missing")
    elif any(type(row.get("completed")) is not int or row["completed"] != PERFORMANCE["requests"] for row in performance):
        reasons.append("performance group did not complete all 32 requests")
    if phases.get("profile", {}).get("status") != "passed":
        reasons.append("profile phase is not passed")
    if profile.get("acceptance") != "accepted" or profile.get("has_device_kernel_evidence") is not True:
        reasons.append("profile lacks accepted device kernel evidence")
    if validation.get("device_empty_after_exit") is not True:
        reasons.append("post-run device-empty gate is not passed")
    return reasons


def dependency_preflight(root):
    """Check existing model/runtime files in the personal container without sourcing CANN."""
    config = recover_framework.read(Path(root) / "scripts/ops/framework-recovery.local.json")
    reference = recover_framework.read(config["reference"])
    bootstrap = reference["bootstrap"]
    probe_input = {"python": bootstrap.get("python", config.get("python")),
                   "reference": config["reference"], "weights": config["weights"],
                   "dataset": config.get("dataset"),
                   "model_extra_source": bootstrap.get("model_extra_source"),
                   "cann_source": config["cann_source"],
                   "source_roots": bootstrap.get("source_roots", []),
                   "vendor_roots": bootstrap.get("vendor_roots", [])}
    probe = r'''import json,pathlib,sys
c=json.loads(sys.argv[1]);checks={}
for k in ('python','reference','dataset','model_extra_source','cann_source'):
 p=c.get(k)
 checks[k]=bool(p) and pathlib.Path(p).is_file()
w=c.get('weights')
checks['weights']=bool(w) and pathlib.Path(w).is_dir()
for i,p in enumerate(c['source_roots']):checks['source_root_'+str(i)]=pathlib.Path(p).is_dir()
for i,p in enumerate(c['vendor_roots']):checks['vendor_root_'+str(i)]=pathlib.Path(p).is_dir()
w=pathlib.Path(c['weights']);idx=w/'model.safetensors.index.json'
try:
 data=json.loads(idx.read_text());shards=set(data.get('weight_map',{}).values())
 checks['weight_shards']=bool(shards) and all((w/s).is_file() for s in shards)
except Exception:checks['weight_shards']=False
try:
 rows=[json.loads(s) for s in pathlib.Path(c['dataset']).read_text().splitlines() if s.strip()]
 checks['fixed_48']=len(rows)==48 and len({r.get('id') for r in rows})==48
except Exception:checks['fixed_48']=False
print(json.dumps(checks))
'''
    result = recover_framework.container(Path(root),
        [probe_input["python"], "-B", "-c", probe, json.dumps(probe_input, ensure_ascii=False)], timeout=90)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-800:]
        return ["read-only dependency probe failed" + (": " + detail if detail else "")]
    try:
        checks = json.loads(result.stdout)
    except ValueError:
        return ["read-only dependency probe returned invalid JSON"]
    return dependency_reasons(checks)


def dependency_reasons(checks):
    return [key + " unavailable" for key, exists in checks.items() if exists is not True]


def _matrix_summary(matrix):
    counts = {}
    for case in matrix["cases"]:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    if all(case["status"] == "passed" for case in matrix["cases"]):
        status = "passed"
    elif any(case["status"] in ("submitted", "running") for case in matrix["cases"]):
        status = "running"
    elif any(case["status"] in ("failed", "failed_to_submit") for case in matrix["cases"]):
        status = "failed"
    elif all(case["status"] == "blocked" for case in matrix["cases"]):
        status = "blocked"
    else:
        status = "incomplete"
    matrix["status"] = status
    matrix["counts"] = counts
    return matrix


def _submit_pangu7b(root, task, path, matrix, case):
    """Submit the single-card 7B matrix runner on one confirmed empty device.

    Runtime root and model python are derived from the registered reference
    bootstrap (same pangu-deploy stack the 92B runner uses). A3 stays blocked
    until that host actually provides a vLLM runtime root.
    """
    import shlex, subprocess, time
    case_id = case["case_id"]
    if case["platform"] != "a5":
        reason = "A3 provides no pangu-deploy vLLM runtime root; a3-pangu7b stays blocked"
        case.update(status="blocked", reason=reason)
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case_id, "status": "blocked", "reason": reason}
    reference = recover_framework.read(recover_framework.read(root / "scripts/ops/framework-recovery.local.json")["reference"])
    bootstrap = reference["bootstrap"]
    source_roots = [Path(v) for v in bootstrap.get("source_roots", [])]
    runtime_root = None
    for candidate in source_roots:
        if candidate.name == "vllm" and candidate.parent.name == "opt":
            runtime_root = candidate.parents[3]  # <rt>/source/image/opt/vllm -> <rt>
            break
    model_python = Path(bootstrap.get("python")) if bootstrap.get("python") else None
    weights = root / "assets/weights/pangu-7b"
    problems = []
    if runtime_root is None or not (runtime_root / "source/image/opt/vllm").is_dir():
        problems.append("vLLM runtime root missing under registered bootstrap")
    if model_python is None or not model_python.is_file():
        problems.append("pangu7b-image model python missing")
    if not (weights / "config.json").is_file() or not list(weights.glob("*.safetensors")):
        problems.append("pangu-7b weight shards missing")
    if problems:
        case.update(status="blocked_dependency", reason="; ".join(problems))
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case_id, "status": "blocked_dependency", "reason": "; ".join(problems)}
    cards = recover_framework.inventory(root, task)["available"]
    if not cards:
        reason = "insufficient healthy empty devices: need 1, available []"
        case.update(status="blocked_resource", reason=reason)
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case_id, "status": "blocked_resource", "reason": reason}
    card = cards[0]
    mapping = recover_framework.container(root, [str(root / 'tools/npu-smi'), 'info', '-m'], source=True)
    (task / 'device-mapping.txt').write_text(mapping.stdout + mapping.stderr, encoding='utf-8')
    physical = None
    for line in mapping.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5 and all(v.isdigit() for v in fields[:4]) and int(fields[0]) == card:
            physical = int(fields[3])
    if mapping.returncode or physical is None:
        case.update(status="failed_to_submit", reason="device mapping unavailable")
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case_id, "status": "failed_to_submit", "reason": "device mapping unavailable"}
    index = len(case.get("attempts", []))
    ident = f"{task.name}-7b{index + 1:02d}"
    if len(ident) > 63:
        raise ValueError("run identifier too long")
    run = root / "runs" / ident
    run.mkdir(mode=0o755)
    args = [recover_framework.read(root / "scripts/ops/framework-recovery.local.json")["python"], "-B",
            str(root / "scripts/ops/run_pangu7b_matrix.py"), "--project", str(root),
            "--runtime-root", str(runtime_root), "--model-python", str(model_python),
            "--weights", str(weights), "--platform", "a5",
            "--device-index", str(physical), "--npu-id", str(card), "--task", ident,
            # 18321 is held by the protected 7B service on NPU 7; matrix runs
            # must not collide with it (plan() refuses to bind an used port).
            "--port", "18323"]
    # Reuse the frozen 48-sample subset generated by an earlier attempt of this
    # case (identical recipe: lengths 4096/8192/16384 x8, seed 42). Data
    # generation takes minutes of CPU-only work between the device-pre and
    # before-launch guards; on a host with rotating foreign jobs that window
    # lost attempt -7b01. Reusing the frozen file keeps the samples identical
    # and shrinks the unguarded window to seconds.
    for attempt in reversed(case.get("attempts", [])):
        prior = root / "runs" / str(attempt.get("task", "")) / "data" / "all.jsonl"
        if prior.is_file():
            try:
                dataset_rows = [json.loads(s) for s in prior.read_text().splitlines() if s.strip()]
                if len(dataset_rows) == 48 and len({r.get("id") for r in dataset_rows}) == 48:
                    recover_framework.write(run / "reused-dataset.json", {"dataset": str(prior), "rows": 48})
                    args.extend(["--dataset", str(prior)])
                    break
            except Exception:
                pass
    # A first submission has no attempts yet; the task may declare the shared
    # frozen 48-sample subset (identical recipe: lengths 4096/8192/16384 x8,
    # seed 42) in repair/FROZEN_DATASET.json. Same validation as above, so a
    # wrong or missing declaration just falls through to fresh generation.
    if "--dataset" not in args:
        frozen_path = task / "repair" / "FROZEN_DATASET.json"
        if frozen_path.is_file():
            try:
                declared = recover_framework.read(frozen_path)
                candidate = Path(declared["dataset"]).resolve(strict=True)
                if candidate.is_relative_to(root):
                    dataset_rows = [json.loads(s) for s in candidate.read_text().splitlines() if s.strip()]
                    if len(dataset_rows) == 48 and len({r.get("id") for r in dataset_rows}) == 48:
                        recover_framework.write(run / "reused-dataset.json", {"dataset": str(candidate),
                                                                             "rows": 48, "declared_by": str(frozen_path)})
                        args.extend(["--dataset", str(candidate)])
            except (OSError, ValueError, KeyError):
                pass
    config = recover_framework.read(root / "scripts/ops/framework-recovery.local.json")
    body = "set -e; source " + shlex.quote(config["cann_source"]) + "; exec " + shlex.join(args)
    command = ["docker", "exec", "--user", "0", "-w", str(root), "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONPATH=" + str(root / "src"), "-e", "FULLNET_CA_BUNDLE=" + str(root / "cache/trust/workstation-ca.pem"),
               config["container"], "bash", "--noprofile", "--norc", "-c", body]
    write_json = recover_framework.write
    write_json(run / "requested-job.json", {"argv": command})
    write_json(run / "recovery-plan.json", {"case": case_id, "devices": [card], "runtime_physical_ids": [physical]})
    receipt = subprocess.run([sys.executable, "-B", str(root / "scripts/ops/submit_job.py"), "--project", str(root),
                              "--task", ident, "--config", str(run / "requested-job.json"), "--tmux", config["tmux"]],
                             cwd=root, capture_output=True, text=True, timeout=30)
    if receipt.returncode:
        case.update(status="failed_to_submit", reason=receipt.stderr[-500:])
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case_id, "status": "failed_to_submit", "reason": receipt.stderr[-500:]}
    # Register every 7B attempt in the recovery ledger so
    # snapshot()/refresh_matrix() could not observe them and the case sat at
    # "submitted" regardless of the run outcome. Register the attempt so the
    # shared ledger drives start()'s single-job check and 7B-aware refresh.
    ledger_path = task / "ledger.json"
    ledger = recover_framework.read(ledger_path) if ledger_path.exists() else {"cases": []}
    ledger["cases"].append({"task": ident, "devices": [card], "submitted_at": time.time(),
                            "submit_exit": receipt.returncode})
    recover_framework.write(ledger_path, ledger)
    case.setdefault("attempts", []).append({"task": ident, "devices": [card], "submitted": True})
    case.update(status="submitted", latest_task=ident, reason=None)
    save_matrix(path, _matrix_summary(matrix))
    return {"case": case_id, "task": ident, "devices": [card], "submitted": True}


def _submit_multinode(root, task, path, matrix, case):
    """Distributed-adapter submit path for the 2-host x 2-NPU A5 cases.

    Implements the launch_contract mode "distributed_adapter_required"
    (RECOVERY.md: implement the missing distributed adapter, do not leave
    the skeleton blocked as if done). Rank/bundle mapping and the cluster
    contract come from models/pangu-92b/adapters/multinode_adapter.py on the
    fork's RayDistributedExecutor surface (VLLM_RAY_BUNDLE_INDICES /
    VLLM_RAY_PER_WORKER_GPUS). Every gate records REAL evidence into the
    task fleet dir; an unreachable verdict maps to blocked_external
    (resource/infra per spec), never to a runtime failure.
    """
    root = Path(root).resolve(strict=True)
    if case["platform"] != "a5":
        raise ValueError("A3 multi-host cases remain blocked until an A3 runtime exists")
    targets = json.loads((root / "runs" / task.name / "repair" / "FLEET_TARGETS.json").read_text(encoding="utf-8"))
    excluded_host = os.environ.get("FULLNET_LOCAL_HOST")
    hosts = sorted(h for h, t in targets.items()
                   if t.get("model") == "950DT" and h != excluded_host)[:2]
    sys.path.insert(0, str(root / "models/pangu-92b/adapters"))
    from multinode_adapter import plan as adapter_plan, preflight as adapter_preflight
    evidence = {"case": case["case_id"], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if len(hosts) != 2:
        evidence["verdict"] = "unreachable"
        evidence["detail"] = f"only {len(hosts)} registered 950DT fleet peers; contract needs 2"
        plan_value = None
    else:
        plan_value = adapter_plan(hosts, case["devices_per_host"], case["parallel"])
        evidence["plan"] = plan_value
        # Pre-formation gate: probe the peers' SSH port (always listening on
        # the hosts); the ray port only opens after cluster formation, and the
        # peer CONTAINER network is probed by the fleet net_probe.sh jobs.
        evidence["local_preflight"] = adapter_preflight([(h, 22) for h in hosts])
    heartbeat_file = task / "fleet" / "broker-heartbeat.json"
    heartbeat = json.loads(heartbeat_file.read_text(encoding="utf-8")) if heartbeat_file.is_file() else {}
    age = time.time() - heartbeat.get("at", 0)
    broker_alive = heartbeat.get("status") == "connected" and age < 120
    evidence["broker"] = {"status": heartbeat.get("status"), "age_seconds": round(age, 1),
                          "alive": broker_alive, "at_epoch": heartbeat.get("at")}
    evidence_path = task / "fleet" / f"multinode-preflight-{case['case_id']}.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(evidence_path, 0o644)
    if plan_value is None:
        case.update(status="blocked_external", reason=evidence["detail"] + f" (see {evidence_path.name})")
    elif not broker_alive:
        case.update(status="blocked_external",
                    reason=(f"fleet broker dead since epoch {heartbeat.get('at')} "
                            f"(reconnect requested via fleet-broker restart); see {evidence_path.name}"))
    elif evidence["local_preflight"]["verdict"] != "reachable":
        case.update(status="blocked_external",
                    reason=(f"peer TCP preflight unreachable on ssh/22; "
                            f"see {evidence_path.name}"))
    else:
        # Broker alive and peers TCP-reachable from the submit host. The
        # remaining unknown is the peer CONTAINER network (the recorded
        # loopback-only family). Container probes run as repair/net_probe.sh
        # fleet jobs; their responses carry the verdict for the cluster step.
        index = matrix["cases"].index(case)
        round_index = len(case.get("multinode_probe_rounds", []))
        requests = []
        for host_index, host in enumerate(hosts):
            request_id = f"mnet-{index:02d}-r{round_index:02d}-h{host_index:02d}"
            # Device-free probe (devices=[]): each peer's container reports its
            # own addresses and TCP reachability to the OTHER fleet host only,
            # so a single-host link failure cannot look like full reachability.
            peers = [other for other in hosts if other != host]
            requests.append(recover_framework.fleet_submit(
                root, task, request_id, "run", host, case["case_id"],
                "repair/net_probe.sh", [], peers))
        case.setdefault("multinode_probe_requests", []).extend(
            item["request_id"] for item in requests)
        case.setdefault("multinode_probe_rounds", []).append(
            {"hosts": hosts, "request_ids": [item["request_id"] for item in requests]})
        case.update(status="preflight_submitted", reason=None)
        save_matrix(path, _matrix_summary(matrix))
        return {"case": case["case_id"], "requests": [item["request_id"] for item in requests]}
    save_matrix(path, _matrix_summary(matrix))
    return case


def _submit_pd_disaggregated(root, task, path, matrix, case):
    """Distributed-adapter submit path for the cross-host PD contract.

    The same-host split adapter (serve_pd_split.py + pd_relay_proxy.py)
    proved real KV transfer between two instances on ONE host; the
    disaggregated contract moves the consumer to a second host, so the new
    unknown is the cross-host KV endpoint between two containers. This path
    records the PD rank map (multinode_adapter.pd_plan), the SSH preflight,
    and broker liveness, then submits the same device-free container-network
    probes as the TP case. A real device phase additionally needs weights +
    runtime on BOTH peers; the fleet inventory responses already refute that
    (weights.exists=false on both 950DT peers), and finalization records the
    honest verdict instead of starting a device phase that cannot pass.
    """
    root = Path(root).resolve(strict=True)
    if case["platform"] != "a5":
        raise ValueError("A3 PD cases remain blocked until an A3 runtime exists")
    targets = json.loads((root / "runs" / task.name / "repair" / "FLEET_TARGETS.json").read_text(encoding="utf-8"))
    excluded_host = os.environ.get("FULLNET_LOCAL_HOST")
    hosts = sorted(h for h, t in targets.items()
                   if t.get("model") == "950DT" and h != excluded_host)[:2]
    sys.path.insert(0, str(root / "models/pangu-92b/adapters"))
    from multinode_adapter import pd_plan as adapter_pd_plan, preflight as adapter_preflight
    evidence = {"case": case["case_id"], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if len(hosts) != 2:
        evidence["verdict"] = "unreachable"
        evidence["detail"] = f"only {len(hosts)} registered 950DT fleet peers; contract needs 2"
        plan_value = None
    else:
        # Per-role TP2 mirrors the same-host split contract (two TP2
        # instances), one role per host over the frozen 2x2 NPU placement.
        plan_value = adapter_pd_plan(hosts, case.get("devices_per_host", [2, 2]), 2)
        evidence["plan"] = plan_value
        evidence["local_preflight"] = adapter_preflight([(h, 22) for h in hosts])
    heartbeat_file = task / "fleet" / "broker-heartbeat.json"
    heartbeat = json.loads(heartbeat_file.read_text(encoding="utf-8")) if heartbeat_file.is_file() else {}
    age = time.time() - heartbeat.get("at", 0)
    broker_alive = heartbeat.get("status") == "connected" and age < 120
    evidence["broker"] = {"status": heartbeat.get("status"), "age_seconds": round(age, 1),
                          "alive": broker_alive, "at_epoch": heartbeat.get("at")}
    evidence_path = task / "fleet" / f"multinode-preflight-{case['case_id']}.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(evidence_path, 0o644)
    if plan_value is None:
        case.update(status="blocked_external", reason=evidence["detail"] + f" (see {evidence_path.name})")
    elif not broker_alive:
        case.update(status="blocked_external",
                    reason=(f"fleet broker dead since epoch {heartbeat.get('at')} "
                            f"(reconnect requested via fleet-broker restart); see {evidence_path.name}"))
    elif evidence["local_preflight"]["verdict"] != "reachable":
        case.update(status="blocked_external",
                    reason=(f"peer TCP preflight unreachable on ssh/22; "
                            f"see {evidence_path.name}"))
    else:
        index = matrix["cases"].index(case)
        round_index = len(case.get("multinode_probe_rounds", []))
        requests = []
        for host_index, host in enumerate(hosts):
            request_id = f"pdnet-{index:02d}-r{round_index:02d}-h{host_index:02d}"
            peers = [other for other in hosts if other != host]
            requests.append(recover_framework.fleet_submit(
                root, task, request_id, "run", host, case["case_id"],
                "repair/net_probe.sh", [], peers))
        case.setdefault("multinode_probe_requests", []).extend(
            item["request_id"] for item in requests)
        case.setdefault("multinode_probe_rounds", []).append(
            {"hosts": hosts, "request_ids": [item["request_id"] for item in requests]})
        case.update(status="preflight_submitted", reason=None)
    save_matrix(path, _matrix_summary(matrix))
    return case if plan_value is None or not broker_alive or \
        evidence["local_preflight"]["verdict"] != "reachable" else \
        {"case": case["case_id"], "requests": case["multinode_probe_requests"][-2:]}


def submit_case(root, task_id, case_id, retry=False):
    root = Path(root).resolve(strict=True)
    task, path = matrix_path(root, task_id)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    case = _case(matrix, case_id)
    # The single-card 7B runner joins the matrix through its own branch: an A5
    # submit transitions the case out of "blocked" explicitly; A3 stays blocked.
    if case["model"] == "pangu-7b" and case["scenario"] == "pangu7b_single_card":
        return _submit_pangu7b(root, task, path, matrix, case)
    if case["scenario"] == "pd_disaggregated":
        return _submit_pd_disaggregated(root, task, path, matrix, case)
    if case["scenario"] in ("multi_host_tp", "multi_host_dp"):
        return _submit_multinode(root, task, path, matrix, case)
    pd_mixed_supported = case["model"] == "pangu-92b" and \
        case["scenario"] in ("pd_mixed_single_instance", "same_host_pd_split")
    if pd_mixed_supported:
        # The case's "blocked" reason was the missing P/D adapter, which now
        # exists (pd_mixed_probe / pd_split_probe with real KV-transfer
        # evidence); an A5 submit unblocks it explicitly and A3 stays blocked
        # with its own reason.
        if case["platform"] != "a5":
            raise ValueError("A3 PD cases remain blocked until an A3 runtime exists")
    retryable = {"blocked_resource", "failed", "failed_to_submit", "incomplete", "interrupted"}
    if case["status"] != "planned" and not pd_mixed_supported and not (retry and case["status"] in retryable):
        raise ValueError("blocked cases remain blocked; only planned or explicitly retried supported cases may be submitted")
    if case["model"] != "pangu-92b" or case["platform"] != "a5" or \
            case["scenario"] not in ("single_host_tpdp", "pd_mixed_single_instance", "same_host_pd_split"):
        raise ValueError("case has no supported native single-host A5 launcher")
    dependencies = dependency_preflight(root)
    if dependencies:
        case.update(status="blocked_dependency", reason="; ".join(dependencies))
        save_matrix(path, _matrix_summary(matrix))
        return case
    contract = case["launch_contract"]
    pd_mode = None
    if case["scenario"] == "pd_mixed_single_instance":
        # Single-instance mixed P/D: the case carries no parallel block, so the
        # launch contract resolves it (TP2 DP1 eager, no MTP) and the same
        # frozen quality/performance/profile chain proves both phases.
        pd_mode = "mixed"
        case["parallel"] = {"tp": 2, "dp": 1, "pp": 1}
        case["graph"], case["mtp"] = "eager", 0
    if case["scenario"] == "same_host_pd_split":
        # Same-host split: two independent TP2 instances on disjoint device
        # pairs (4 NPUs total) plus the relay proxy; eager so no graph capture
        # runs twice on the same host. The real-KV-transfer probe runs inside
        # the frozen chain before profile/performance.
        pd_mode = "split"
        case["parallel"] = {"tp": 2, "dp": 1, "pp": 1}
        case["graph"], case["mtp"] = "eager", 0
    plan = {"tp": case["parallel"]["tp"], "dp": case["parallel"]["dp"], "graph": case["graph"],
            "mtp": case["mtp"],
            "max_batched_tokens": contract.get("max_batched_tokens", 4096),
            "max_seqs": contract.get("max_seqs", 8),
            "disable_aligned_decode": contract.get("disable_aligned_decode", True),
            "pd_mode": pd_mode,
            "reason": "deployment matrix native single-host A5 case: " + case_id}
    try:
        receipt = recover_framework.start(root, task, plan)
    except RuntimeError as exc:
        reason = str(exc)
        if "insufficient healthy empty devices" in reason:
            case.update(status="blocked_resource", reason=reason)
        else:
            case.update(status="failed_to_submit", reason=reason)
        save_matrix(path, _matrix_summary(matrix))
        return case
    attempt = {"task": receipt["submitted"], "devices": receipt["devices"], "submitted": True}
    case.setdefault("attempts", []).append(attempt)
    case.update(status="submitted", latest_task=attempt["task"], reason=None)
    save_matrix(path, _matrix_summary(matrix))
    return {"case": case_id, **attempt}


def refresh_matrix(root, task_id):
    root = Path(root).resolve(strict=True)
    task, path = matrix_path(root, task_id)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    snapshot = recover_framework.snapshot(root, task)
    observed = {item.get("task"): item for item in snapshot.get("cases", [])}
    for case in matrix["cases"]:
        if case.get("multinode_probe_requests"):
            # Keep the distributed-adapter gate truthful: pull the container
            # network probe statuses without waiting; the verdict that admits
            # the cluster step lands in the case once responses exist.
            case["multinode_probe_status"] = [
                recover_framework.fleet_status(root, task, request_id, 0)
                for request_id in case["multinode_probe_requests"]]
        run_id = case.get("latest_task")
        if not run_id:
            continue
        record = observed.get(run_id, {})
        job = record.get("job-state.json", {})
        if job.get("status") == "running":
            case["status"] = "running"
            continue
        if job.get('status') == 'interrupted':
            case.update(status='interrupted',reason=job.get('reason','business process interrupted'))
            continue
        if job.get("status") == "failed" or job.get("exit") not in (None, 0):
            # Spec mapping: a run rejected because the target device was taken
            # by another user between inventory and launch is a resource
            # conflict (blocked), not a runtime failure (failed).
            detail = str(record.get("validation.json", {}).get("detail", ""))
            if any(tag in detail for tag in ("target device is occupied", "port already in use",
                                             "insufficient healthy empty devices")):
                case.update(status="blocked_resource", reason=detail, exit=job.get("exit"))
            else:
                case.update(status="failed", reason="business job exited nonzero", exit=job.get("exit"))
            continue
        if job.get("status") != "passed":
            case["status"] = "submitted"
            continue
        if case.get("scenario") == "pangu7b_single_card":
            # The 7B runner keeps its experiment under models/pangu-7b and its
            # own validation.json is the outcome truth (exit 0 only on a full
            # pass), so acceptance is derived from those fields directly.
            validation = record.get("validation.json", {})
            reasons = []
            if job.get("exit") != 0:
                reasons.append("business job exit is not zero")
            if (validation.get("attempted") != 48 or validation.get("succeeded") != 48 or
                    validation.get("failed") != 0 or validation.get("skipped") != 0 or
                    validation.get("timeout") != 0 or validation.get("protocol_pass") is not True or
                    validation.get("evaluation_phase") != "passed"):
                reasons.append("fixed 48-sample evaluation did not fully pass protocol")
            if validation.get("profile_phase") != "passed":
                reasons.append("profile phase is not passed")
            elif (record.get("profile-artifacts.json") or {}).get("has_device_kernel_evidence") is not True:
                reasons.append("profile lacks accepted device kernel evidence")
            if validation.get("performance_phase") != "passed":
                reasons.append("performance phase is not passed")
            if validation.get("device_empty_after_exit") is not True:
                reasons.append("post-run device-empty gate is not passed")
            case["status"] = "passed" if not reasons else "incomplete"
            case["acceptance_reasons"] = reasons
            case["exit"] = job.get("exit")
            continue
        reasons = acceptance_reasons(case, job, record.get("validation.json", {}),
                                     record.get("state.json", {}), record.get("config.json", {}),
                                     record.get("summary.json"), record.get("profile-artifacts.json", {}))
        case["status"] = "passed" if not reasons else "incomplete"
        case["acceptance_reasons"] = reasons
        case["exit"] = job.get("exit")
    save_matrix(path, _matrix_summary(matrix))
    return matrix


def submit_fleet_inventory(root, task_id, case_id, hosts):
    root = Path(root).resolve(strict=True)
    task, path = matrix_path(root, task_id)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    case = _case(matrix, case_id)
    if case["scenario"] in {"single_host_tpdp", "pangu7b_single_card"} and case["status"] != "blocked":
        raise ValueError("use the local NPU preflight for supported A5 single-host cases")
    if len(hosts) != case["hosts_required"]:
        raise ValueError(f"case requires exactly {case['hosts_required']} hosts")
    if len(set(hosts)) != len(hosts):
        raise ValueError("fleet host list contains duplicates")
    index = matrix["cases"].index(case)
    round_index = len(case.get("fleet_inventory_rounds", []))
    requests = []
    for host_index, host in enumerate(hosts):
        request_id = f"inv-{index:02d}-r{round_index:02d}-h{host_index:02d}"
        requests.append(recover_framework.fleet_inventory_request(root, task, request_id, host, case_id))
    case["fleet_hosts"] = hosts
    case.setdefault("fleet_inventory_requests", []).extend(item["request_id"] for item in requests)
    case.setdefault("fleet_inventory_rounds", []).append({"hosts": hosts,
        "request_ids": [item["request_id"] for item in requests]})
    save_matrix(path, _matrix_summary(matrix))
    return requests


def submit_fleet_asset(root, task_id, case_id, host, asset):
    root = Path(root).resolve(strict=True)
    task, path = matrix_path(root, task_id)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    case = _case(matrix, case_id)
    if asset not in {"pangu-7b", "pangu-92b"}:
        raise ValueError("only registered model assets may be copied")
    index = matrix["cases"].index(case)
    round_index = len(case.get("asset_requests", []))
    request_id = f"asset-{index:02d}-r{round_index:02d}-{'7b' if asset == 'pangu-7b' else '92b'}"
    request = recover_framework.fleet_submit(root, task, request_id, "copy_asset", host, case_id,
                                             devices=[], args=[], asset=asset)
    case.setdefault("asset_requests", []).append({"host": host, **request})
    save_matrix(path, _matrix_summary(matrix))
    return request


def read_fleet_inventory(root, task_id, case_id, wait_seconds=0):
    root = Path(root).resolve(strict=True)
    task, path = matrix_path(root, task_id)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    case = _case(matrix, case_id)
    responses = [recover_framework.fleet_status(root, task, request_id, wait_seconds)
                 for request_id in case.get("fleet_inventory_requests", [])]
    case["fleet_inventory_status"] = responses
    for asset_request in case.get("asset_requests", []):
        asset_request["result"] = recover_framework.fleet_status(root, task, asset_request["request_id"], wait_seconds)
    save_matrix(path, _matrix_summary(matrix))
    return responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--action", choices=("plan", "status", "submit", "refresh", "fleet-inventory", "fleet-asset", "fleet-status"), required=True)
    parser.add_argument("--case")
    parser.add_argument("--retry", action="store_true", help="Submit a fresh experiment for a retryable supported case")
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--host")
    parser.add_argument("--asset", choices=("pangu-7b", "pangu-92b"))
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    root = args.project.resolve(strict=True)
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("project root must be this checkout")
    if args.action == "plan":
        result = create_matrix(root, args.task)
    elif args.action == "submit":
        if not args.case: raise ValueError("--case is required")
        result = submit_case(root, args.task, args.case, retry=args.retry)
    elif args.action == "refresh":
        result = refresh_matrix(root, args.task)
    elif args.action == "fleet-inventory":
        if not args.case or not args.hosts: raise ValueError("--case and --hosts are required")
        result = submit_fleet_inventory(root, args.task, args.case, args.hosts)
    elif args.action == "fleet-asset":
        if not args.case or not args.host or not args.asset: raise ValueError("--case, --host and --asset are required")
        result = submit_fleet_asset(root, args.task, args.case, args.host, args.asset)
    elif args.action == "fleet-status":
        if not args.case: raise ValueError("--case is required")
        result = read_fleet_inventory(root, args.task, args.case, args.wait_seconds)
    else:
        matrix, _ = load_matrix(root, args.task)
        result = _matrix_summary(matrix)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # A plan/refresh/query that successfully records an incomplete or blocked
    # matrix is a successful orchestration command; acceptance lives in JSON.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
