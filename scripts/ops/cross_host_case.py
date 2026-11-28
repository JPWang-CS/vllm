"""Public, fail-closed cross-host case coordinator.

This module is the small orchestration seam between a case-local manifest and
the existing ``recover_framework``/``registered_fleet_bridge`` fleet API.  It
does not open SSH, select a host by a hard-coded address, or create a second
backend.  It persists request/job/attempt identity in the current task and
only allows a resource requeue when the current evidence proves that no model
work started.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "fullnet.cross-host-case.v1"
STATE_SCHEMA = "fullnet.cross-host-state.v1"
RESOURCE_GATE_REQUEUE = "RESOURCE_GATE_REQUEUE"
MAX_JSON_BYTES = 1_000_000


class CaseContractError(ValueError):
    """The manifest or current evidence cannot prove the requested action."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.ops import recover_framework as RECOVER  # noqa: E402

BRIDGE = _load_module("cross_host_registered_fleet_bridge",
                     PROJECT_ROOT / "scripts/ops/registered_fleet_bridge.py")
ADAPTER = _load_module("cross_host_multinode_adapter",
                       PROJECT_ROOT / "models/pangu-92b/adapters/multinode_adapter.py")


def _read_json(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise CaseContractError(f"invalid or oversized JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CaseContractError(f"JSON object required: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    if path.is_symlink():
        raise CaseContractError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise CaseContractError(f"invalid {label}")
    return value


def _canonical_host(value: str) -> str:
    # Fleet requests are intentionally restricted to the existing registered
    # IPv4 contract.  Hosts remain data in the manifest, never source code.
    import ipaddress
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise CaseContractError(f"host must be a registered IPv4 address: {value!r}") from exc
    if address.version != 4 or str(address) != value:
        raise CaseContractError(f"host must be canonical IPv4: {value!r}")
    return value


def _role_ranks(value, world, label):
    if not isinstance(value, list) or not value or any(type(rank) is not int for rank in value):
        raise CaseContractError(f"{label} must be a non-empty integer list")
    ranks = sorted(set(value))
    if len(ranks) != len(value) or any(rank < 0 or rank >= world for rank in ranks):
        raise CaseContractError(f"{label} contains invalid rank")
    return ranks


def load_case_spec(path: Path) -> dict:
    """Validate and normalize the case-local role/candidate contract."""
    value = _read_json(Path(path).resolve(strict=True))
    if value.get("schema") != SCHEMA:
        raise CaseContractError(f"manifest schema must be {SCHEMA}")
    case_id = _identifier(value.get("case_id"), "case_id")
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise CaseContractError("case model is required")
    scenario = value.get("scenario", "multi_host_tp")
    if scenario not in {"multi_host_tp", "pd_split", "cross_host_pd_split"}:
        raise CaseContractError("unsupported cross-host scenario")
    world = value.get("world_size")
    if scenario == "multi_host_tp":
        if type(world) is not int or world < 2:
            raise CaseContractError("multi_host_tp world_size must be an integer >= 2")
        expected_ranks = _role_ranks(value.get("expected_ranks"), world, "expected_ranks")
        if expected_ranks != list(range(world)):
            raise CaseContractError("TP expected_ranks must cover every global rank exactly once")
    else:
        # P/D has two independent TP1 role worlds.  There is no global TP2
        # rank map to synthesize from producer and consumer.
        expected_ranks = list(value.get("expected_ranks", []))
        if any(type(rank) is not int or rank < 0 for rank in expected_ranks):
            raise CaseContractError("P/D expected_ranks must be non-negative integers when present")
    roles = value.get("roles")
    if not isinstance(roles, dict) or len(roles) < 2:
        raise CaseContractError("at least two current-case roles are required")
    if scenario != "multi_host_tp" and set(roles) != {"producer", "consumer"}:
        raise CaseContractError("P/D scenarios require producer and consumer roles")
    normalized_roles = {}
    for role, raw in roles.items():
        role = _identifier(role, "role")
        if not isinstance(raw, dict):
            raise CaseContractError(f"role {role} must be an object")
        host = _canonical_host(raw.get("host"))
        task_id = _identifier(raw.get("task_id"), f"{role}.task_id")
        role_world = raw.get("expected_world") if scenario != "multi_host_tp" else raw.get("expected_world", world)
        if type(role_world) is not int or role_world < 1:
            raise CaseContractError(f"{role}.expected_world must be a positive integer")
        if scenario == "multi_host_tp" and role_world != world:
            raise CaseContractError(f"{role}.expected_world differs from case world_size")
        ranks = _role_ranks(raw.get("expected_ranks"), role_world, f"{role}.expected_ranks")
        if raw.get("model", model) != model:
            raise CaseContractError(f"{role}.model differs from case model")
        normalized_roles[role] = {
            **raw, "host": host, "task_id": task_id,
            "expected_world": role_world, "expected_ranks": ranks, "model": model,
        }
    hosts = [raw["host"] for raw in normalized_roles.values()]
    if len(set(hosts)) < 2:
        raise CaseContractError("current role hosts must span at least two hosts")
    if scenario == "multi_host_tp":
        role_ranks = sorted(rank for role in normalized_roles.values() for rank in role["expected_ranks"])
        if role_ranks != expected_ranks:
            raise CaseContractError("TP role expected_ranks must cover the global rank map exactly once")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CaseContractError("candidate host list is required")
    normalized_candidates = []
    seen = set()
    for raw in candidates:
        if not isinstance(raw, dict):
            raise CaseContractError("candidate must be an object")
        role = raw.get("role")
        if role not in normalized_roles:
            raise CaseContractError(f"candidate role is not in current roles: {role!r}")
        host = _canonical_host(raw.get("host"))
        key = (role, host, tuple(raw.get("device_ids", [])))
        if key in seen:
            raise CaseContractError(f"duplicate candidate: {key}")
        seen.add(key)
        device_ids = raw.get("device_ids")
        if (not isinstance(device_ids, list) or not device_ids or
                any(type(device) is not int or not 0 <= device < 32 for device in device_ids)):
            raise CaseContractError(f"candidate {role}@{host} needs device_ids")
        gates = {name: raw.get(name) for name in (
            "cpu_prepared", "source_ready", "environment_ready", "network_ready")}
        if any(value is not True for value in gates.values()):
            raise CaseContractError(f"candidate {role}@{host} has an incomplete CPU gate")
        normalized_candidates.append({**raw, "role": role, "host": host,
                                      "device_ids": sorted(set(device_ids)), **gates})
    return {**value, "case_id": case_id, "scenario": scenario, "model": model, "world_size": world,
            "expected_ranks": expected_ranks, "roles": normalized_roles,
            "candidates": normalized_candidates}


def _path_under(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CaseContractError(f"{label} escapes task root: {path}") from exc
    return path


def _task_input(task: Path, path: Path | None, default_name: str) -> Path:
    candidate = task / "repair" / default_name if path is None else Path(path)
    candidate = _path_under(candidate, task, "task evidence path")
    if candidate.is_symlink() or not candidate.is_file():
        raise CaseContractError(f"task evidence file is missing: {candidate}")
    return candidate


def _response_payload(raw: dict) -> dict:
    """Return the standard broker response while preserving its raw fields."""
    response = raw.get("response") if isinstance(raw.get("response"), dict) else raw
    if not isinstance(response, dict):
        raise CaseContractError("broker response is not an object")
    return response


def _complete_file(response: dict, name: str, kind: str) -> bool:
    metadata = response.get("file_metadata", {}).get(name, {})
    if not isinstance(metadata, dict) or metadata.get("truncated") is True:
        return False
    if kind == "json":
        return metadata.get("json_valid") is True and isinstance(response.get(name), dict)
    return isinstance(response.get(name), str)


def _no_model_workload(response: dict, role: str) -> tuple[bool, dict]:
    reasons = []
    presence = response.get("file_presence")
    if not isinstance(presence, dict):
        reasons.append(f"{role}: file_presence is missing")
    else:
        for name in ("process.json", "workload.log"):
            if presence.get(name) is not False:
                reasons.append(f"{role}: {name} absence is not explicit")
    validation = response.get("validation.json")
    if validation is not None:
        if not _complete_file(response, "validation.json", "json"):
            reasons.append(f"{role}: validation.json is incomplete")
        elif validation.get("model_started") is not False:
            reasons.append(f"{role}: model_started is not explicitly false")
    result = response.get("result.json")
    if result is not None:
        if not _complete_file(response, "result.json", "json"):
            reasons.append(f"{role}: result.json is incomplete")
        elif result.get("model_started") is not False:
            reasons.append(f"{role}: result model_started is not explicitly false")
    return not reasons, {"role": role, "reasons": reasons,
                         "file_presence": presence, "validation": validation,
                         "result": result}


def _resource_failure(response: dict, role: str) -> tuple[bool, dict]:
    reasons = []
    if not _complete_file(response, "state.json", "json"):
        reasons.append(f"{role}: state.json missing/truncated")
    state = response.get("state.json") or {}
    if state.get("status") != "exited" or state.get("exit") != 1:
        reasons.append(f"{role}: state is not terminal exit 1")
    if not _complete_file(response, "failure.json", "json"):
        reasons.append(f"{role}: failure.json missing/truncated")
    failure = response.get("failure.json") or {}
    detail = str(failure.get("detail", ""))
    if not ("not confirmed healthy and empty" in detail or "device lease busy:" in detail):
        reasons.append(f"{role}: failure is not an anchored device resource gate")
    if not _complete_file(response, "exit.txt", "text") or response.get("exit.txt", "").strip() != "1":
        reasons.append(f"{role}: exit.txt is missing or not exactly 1")
    return not reasons, {"role": role, "reasons": reasons,
                         "state": state, "failure": failure}


def safe_requeue_decision(*, intents, request_paths, observations, cleanup,
                          expected_roles, unsubmitted_roles=None,
                          model_started=False):
    """Classify a resource failure without turning a model failure into replay.

    ``intents`` are persisted role intents, while ``request_paths`` are the
    actual task-local request files.  An intent without its request file is a
    never-queued race and may be discarded; an intent with a request is queued
    and can only be retried after current binding, no-workload, resource-gate
    and cleanup evidence.  Roles not yet submitted need an explicit
    task-local no-request/no-business record.
    """
    intents = list(intents or [])
    expected_roles = set(expected_roles or ())
    queued = [entry for entry in intents if entry.get("request_id") in request_paths]
    queued_roles = {entry.get("role") for entry in queued}
    unsubmitted_roles = dict(unsubmitted_roles or {})
    active = []
    for role, response in (observations or {}).items():
        state = (response or {}).get("state.json") or {}
        if state.get("status") == "running":
            active.append(role)
    if active:
        return {"allowed": False, "action": "adopt_active_job",
                "reason": "active job exists; do not submit another request",
                "active_roles": sorted(active)}
    if model_started:
        return {"allowed": False, "action": "stop",
                "reason": "model execution has started; never replay automatically"}
    if not queued and intents:
        missing = sorted(expected_roles - set(unsubmitted_roles))
        if missing:
            return {"allowed": False, "action": "stop",
                    "reason": "unsubmitted role evidence is missing", "roles": missing}
        if any(value.get("request_present") is not False or
               value.get("business_started") is not False
               for value in unsubmitted_roles.values()):
            return {"allowed": False, "action": "stop",
                    "reason": "unsubmitted role has a request or business evidence"}
        return {"allowed": True, "action": "discard_never_queued_intent",
                "category": RESOURCE_GATE_REQUEUE,
                "cleanup": {"verified": True, "jobs": [],
                             "note": "no queued intent; nothing to clean"},
                "evidence": {"intents": intents, "request_paths": sorted(request_paths)}}
    if not queued or not queued_roles.issubset(expected_roles):
        return {"allowed": False, "action": "stop",
                "reason": "queued roles do not match the current case contract",
                "queued_roles": sorted(queued_roles)}
    missing = sorted(expected_roles - queued_roles - set(unsubmitted_roles))
    if missing:
        return {"allowed": False, "action": "stop",
                "reason": "unsubmitted role evidence is missing", "roles": missing}
    if any(value.get("request_present") is not False or
           value.get("business_started") is not False
           for value in unsubmitted_roles.values()):
        return {"allowed": False, "action": "stop",
                "reason": "unsubmitted role has a request or business evidence"}
    if not isinstance(cleanup, dict) or cleanup.get("verified") is not True:
        return {"allowed": False, "action": "stop",
                "reason": "cleanup is not verified for the current case"}
    cleanup_jobs = cleanup.get("jobs")
    cleanup_by_role = {item.get("role"): item for item in cleanup_jobs or []
                       if isinstance(item, dict)}
    if (not isinstance(cleanup_jobs, list) or set(cleanup_by_role) != queued_roles):
        return {"allowed": False, "action": "stop",
                "reason": "cleanup does not cover every submitted role"}
    for entry in queued:
        item = cleanup_by_role.get(entry.get("role"))
        response = (observations or {}).get(entry.get("role")) or {}
        binding = response.get("_request_binding")
        if not (isinstance(binding, dict) and
                all(binding.get(key) == entry.get(key) for key in ("request_id", "host", "job"))):
            return {"allowed": False, "action": "stop",
                    "reason": "current response is not bound to submitted request/job",
                    "role": entry.get("role")}
        if (item.get("verified") is not True or
                any(item.get(key) != entry.get(key) for key in ("role", "host", "job", "request_id"))):
            return {"allowed": False, "action": "stop",
                    "reason": "cleanup binding does not match submitted request/job",
                    "role": entry.get("role")}
    evidence = {}
    resource_roles = []
    for role in sorted(queued_roles):
        response = observations.get(role) or {}
        resource_ok, resource = _resource_failure(response, role)
        workload_ok, workload = _no_model_workload(response, role)
        evidence[role] = {"resource": resource, "no_workload": workload}
        if resource_ok:
            resource_roles.append(role)
        if not workload_ok:
            return {"allowed": False, "action": "stop",
                    "reason": "current complete no-workload proof is incomplete",
                    "evidence": evidence}
    if not resource_roles:
        return {"allowed": False, "action": "stop",
                "reason": "no submitted role has a complete resource-gate failure",
                "evidence": evidence}
    return {"allowed": True, "action": "archive_resource_requeue",
            "category": RESOURCE_GATE_REQUEUE, "cleanup": cleanup,
            "evidence": evidence}


def archive_resource_requeue(task: Path, state: dict, evidence: dict) -> dict:
    """Archive the failed attempt and reset only this case's role queue."""
    task = Path(task).resolve(strict=True)
    repair = _path_under(task / "repair", task, "repair directory")
    repair.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    archive = repair / f"CROSS_HOST_STATE.resource-gate-{stamp}.json"
    _write_json_atomic(archive, state)
    current = dict(state)
    current.update({"schema": STATE_SCHEMA, "status": RESOURCE_GATE_REQUEUE,
                    "entries": [], "requeue_archive": str(archive),
                    "requeue_evidence": evidence})
    current_path = repair / "CROSS_HOST_STATE.json"
    _write_json_atomic(current_path, current)
    return {"status": "WAITING_RESOURCES", "archive": str(archive),
            "state": str(current_path), "entries": []}


def _fresh_stamp(raw: dict, now: float, ttl: float) -> bool:
    try:
        stamp = float(raw.get("broker_checked_at", raw.get("checked_at")))
    except (TypeError, ValueError):
        return False
    return 0 <= now - stamp < ttl


def _device_proof(raw: dict, device_ids: list[int]) -> tuple[bool, dict]:
    """Use the current broker inventory output; fail closed if a device is not proved."""
    container = raw.get("container_query") if isinstance(raw.get("container_query"), dict) else raw
    devices = container.get("devices") if isinstance(container, dict) else None
    if not isinstance(devices, dict) or devices.get("exit") != 0:
        return False, {"reason": "container device inventory query failed"}
    text = str(devices.get("stdout", ""))
    rows = container.get("device_rows") if isinstance(container, dict) else None
    if isinstance(rows, list):
        by_id = {row.get("device_id"): row for row in rows if isinstance(row, dict)}
        missing = [device for device in device_ids
                   if not (by_id.get(device, {}).get("health") == "OK"
                           and by_id.get(device, {}).get("idle") is True
                           and by_id.get(device, {}).get("process_list_clear") is True)]
        return not missing, {"source": "structured_container_inventory", "missing": missing}
    missing = []
    for device in device_ids:
        health = bool(re.search(rf"\|\s*{device}\s*\|[^\n]*\bOK\b", text, re.I))
        idle = bool(re.search(rf"No running processes found in NPU\s+{device}\b", text, re.I))
        if not (health and idle):
            missing.append(device)
    return not missing, {"source": "container_npu_smi_text", "missing": missing}


def _cpu_prepare_proof(raw: dict, candidate: dict) -> tuple[bool, dict]:
    """Consume a complete current CPU-prepare fleet receipt.

    The four manifest booleans describe intent only.  They do not authorize a
    device job.  The receipt must bind host/request/job and contain terminal
    CPU preparation, source-list and network-probe evidence.
    """
    prepared = raw.get("cpu_prepare") if isinstance(raw, dict) else None
    reasons = []
    if not isinstance(prepared, dict) or not prepared.get("response_path"):
        return False, {"reasons": ["complete CPU prepare fleet receipt is missing"]}
    response = _response_payload(prepared)
    binding = prepared.get("_request_binding") or response.get("_request_binding")
    if not isinstance(binding, dict):
        reasons.append("CPU prepare request/job binding is missing")
    else:
        for key, expected in (("host", candidate["host"]),
                              ("request_id", candidate.get("prepare_request_id")),
                              ("job", candidate.get("prepare_job"))):
            if expected is not None and binding.get(key) != expected:
                reasons.append(f"CPU prepare {key} binding differs")
    if not _complete_file(response, "state.json", "json"):
        reasons.append("CPU prepare state.json is missing or incomplete")
    state = response.get("state.json") or {}
    if state.get("status") != "exited" or state.get("exit") != 0:
        reasons.append("CPU prepare job did not exit 0")
    if not _complete_file(response, "result.json", "json"):
        reasons.append("CPU prepare result.json is missing or incomplete")
    result = response.get("result.json") or {}
    if result.get("status") != "CPU_PREPARED" or result.get("no_model_started") is not True:
        reasons.append("CPU prepare result is not a no-model CPU_PREPARED result")
    source = result.get("source_manifest")
    network = result.get("network_probe")
    if (not isinstance(source, dict) or source.get("complete") is not True or
            not isinstance(source.get("files"), list) or not source["files"]):
        reasons.append("complete source manifest is missing")
    if (not isinstance(network, dict) or network.get("complete") is not True or
            network.get("reachable") is not True):
        reasons.append("complete network probe is missing")
    return not reasons, {"reasons": reasons, "response_path": prepared.get("response_path"),
                         "state": state, "result": result}


def eligible_candidates(spec: dict, inventory_by_host: dict, *, now=None, ttl=180.0) -> dict:
    """Return only candidates passing CPU gates and a fresh container inventory."""
    observed = time.time() if now is None else float(now)
    result = {}
    reasons = {}
    for candidate in spec["candidates"]:
        role, host = candidate["role"], candidate["host"]
        raw = inventory_by_host.get(host)
        if not isinstance(raw, dict) or not _fresh_stamp(raw, observed, ttl):
            reasons.setdefault(role, []).append({"host": host, "reason": "inventory is missing or stale"})
            continue
        prepare_ok, prepare_proof = _cpu_prepare_proof(raw, candidate)
        if not prepare_ok:
            reasons.setdefault(role, []).append({"host": host, "reason": "CPU prepare/source/network gate not proved",
                                                 "proof": prepare_proof})
            continue
        device_ok, proof = _device_proof(raw, candidate["device_ids"])
        if not device_ok:
            reasons.setdefault(role, []).append({"host": host, "reason": "device gate not proved", "proof": proof})
            continue
        result.setdefault(role, []).append({**candidate, "inventory_at": raw.get("broker_checked_at", raw.get("checked_at")),
                                             "inventory_proof": proof, "cpu_prepare_proof": prepare_proof})
    return {"candidates": result, "rejected": reasons, "observed_at": observed}


def select_two_phase(first: dict, second: dict, roles) -> dict:
    """Intersect two fresh scans by full host/device identity, never by role only."""
    selected = {}
    for role in roles:
        left = first.get(role, [])
        right = second.get(role, [])
        right_keys = {(_candidate_key(value)) for value in right}
        matches = [value for value in left if _candidate_key(value) in right_keys]
        if len(matches) != 1:
            raise CaseContractError(f"two-phase fresh gate has {len(matches)} matches for role {role}")
        selected[role] = matches[0]
    return selected


def _candidate_key(value: dict) -> tuple:
    return (value.get("role"), value.get("host"), tuple(value.get("device_ids", [])),
            value.get("board_id"), value.get("physical_id"), value.get("runtime_device"))


def _remote_rank_records(spec: dict, state: dict, observations: dict) -> list[dict]:
    """Build actor records from complete fleet role responses.

    The control host never globs ``.r`` for all roles.  Each selected role's
    own remote job must publish complete validation fields (rank lines and
    source paths) through the broker status response.  The request/job/host
    binding comes from the local case state and is checked against the remote
    response before the fields are merged.
    """
    records = []
    entries = {entry.get("role"): entry for entry in state.get("entries", [])}
    for role, role_spec in spec["roles"].items():
        entry = entries.get(role)
        observed = observations.get(role)
        if not isinstance(entry, dict) or not isinstance(observed, dict):
            raise CaseContractError(f"fleet evidence missing for current role: {role}")
        status = observed.get("status") if isinstance(observed.get("status"), dict) else observed
        response = _response_payload(status)
        if not status.get("response_path") or not isinstance(response, dict):
            raise CaseContractError(f"complete fleet response missing for role: {role}")
        job_name = Path(str(response.get("job", status.get("job", "")))).name
        if job_name != entry.get("job"):
            raise CaseContractError(f"fleet response job does not bind to role {role}")
        validation = response.get("validation.json")
        result = response.get("result.json")
        if not _complete_file(response, "validation.json", "json"):
            # A finished role may carry the same contract in result.json; it
            # is still a complete broker file, not a control-side log read.
            if not (_complete_file(response, "result.json", "json") and
                    isinstance(result.get("validation"), dict)):
                raise CaseContractError(f"remote validation evidence missing for role: {role}")
            validation = result["validation"]
        envelope = validation.get("actor_evidence") or validation.get("actor_evidence_envelope")
        if not isinstance(envelope, dict) or envelope.get("schema") != "fullnet.actor-evidence.v1":
            raise CaseContractError(f"complete actor evidence envelope missing for role: {role}")
        for key, expected in (("case_id", spec["case_id"]), ("model", role_spec["model"]),
                              ("role", role), ("host", role_spec["host"]),
                              ("job", entry["job"]), ("request_id", entry["request_id"])):
            if envelope.get(key) != expected:
                raise CaseContractError(f"remote actor envelope {key} is not bound for role: {role}")
        paths = envelope.get("log_paths")
        rank_raw = envelope.get("rank_facts")
        completion_raw = envelope.get("completion_facts")
        if not isinstance(paths, list) or not paths:
            raise CaseContractError(f"remote actor log paths missing for role: {role}")
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise CaseContractError(f"remote actor path is empty for role: {role}")
        path_pattern = re.compile(
            rf"[/\\]{re.escape(role_spec['task_id'])}-{re.escape(role)}"
            rf"[/\\]session_[^/\\]+[/\\]logs[/\\]worker-[^/\\]+\.out$")
        if any(not path_pattern.search(path) for path in paths):
            raise CaseContractError(f"remote actor path is not bound to current role task: {role}")
        if isinstance(rank_raw, dict):
            rank_items = list(rank_raw.values())
        elif isinstance(rank_raw, list):
            rank_items = rank_raw
        else:
            rank_items = []
        if not isinstance(completion_raw, list) or not rank_items or not completion_raw:
            raise CaseContractError(f"remote actor rank/completion facts are incomplete for role: {role}")
        rank_facts = {}
        for item in rank_items:
            if not isinstance(item, dict) or not isinstance(item.get("line"), str):
                continue
            match = ADAPTER._RANK_WORLD_RE.search(item["line"])
            if not match:
                continue
            rank, world = ADAPTER._rank_world_values(match)
            if item.get("rank") != rank or item.get("world_size") != world:
                raise CaseContractError(f"remote rank fact disagrees with its source line for role: {role}")
            path = item.get("path")
            if path not in paths:
                raise CaseContractError(f"remote rank fact path is not in envelope for role: {role}")
            rank_facts[str(rank)] = {
                "rank": rank, "world_size": world, "host": role_spec["host"],
                "role": role, "model": role_spec["model"],
                "weights_loaded": False, "path": path,
                "case_id": spec["case_id"], "job": entry["job"],
                "request_id": entry["request_id"],
            }
        completion_facts = []
        for item in completion_raw:
            if not isinstance(item, dict) or type(item.get("rank")) is not int:
                continue
            item_path = item.get("path")
            lines = item.get("lines")
            if not isinstance(lines, list) or item_path not in paths:
                continue
            completed = [line for line in lines
                         if isinstance(line, str) and ADAPTER._WEIGHTS_RE.search(line)]
            if not completed:
                continue
            fact = rank_facts.get(str(item["rank"]))
            if fact is None or fact["path"] != item_path:
                raise CaseContractError(f"completion fact is not bound to rank fact for role: {role}")
            fact["weights_loaded"] = True
            completion_facts.append({"rank": item["rank"], "path": item_path,
                                     "lines": completed, "weights_loaded": True})
        expected_role_ranks = set(role_spec["expected_ranks"])
        if set(rank_facts) != {str(rank) for rank in expected_role_ranks}:
            raise CaseContractError(f"remote actor ranks are incomplete for role: {role}")
        if any(not rank_facts[str(rank)]["weights_loaded"] for rank in expected_role_ranks):
            raise CaseContractError(f"remote actor completion evidence is incomplete for role: {role}")
        records.append({
            "role": role, "host": role_spec["host"], "task_id": role_spec["task_id"],
            "model": role_spec["model"], "case_id": spec["case_id"],
            "job": entry["job"], "request_id": entry["request_id"],
            "log_paths": paths, "completion_facts": completion_facts,
            "records": [{"rank_facts": rank_facts,
                                               "ranks": sorted(int(rank) for rank in rank_facts),
                                               "world_sizes": sorted({fact["world_size"] for fact in rank_facts.values()}),
                                               "weights_loaded": all(fact["weights_loaded"] for fact in rank_facts.values()),
                                               "path": paths[0],
                                               "completion_facts": completion_facts}],
            "ranks": sorted(int(rank) for rank in rank_facts),
            "world_sizes": sorted({fact["world_size"] for fact in rank_facts.values()}),
            "weights_loaded": all(fact["weights_loaded"] for fact in rank_facts.values()),
            "rank_facts": rank_facts,
        })
    return records


class CrossHostCase:
    """Case-local coordinator using the existing public fleet functions."""

    def __init__(self, project: Path, task: Path, manifest: Path):
        self.project = Path(project).resolve(strict=True)
        self.task = Path(task).resolve(strict=True)
        if not self.task.is_relative_to(self.project / "runs"):
            raise CaseContractError("task must be inside project/runs")
        self.spec_path = Path(manifest).resolve(strict=True)
        self.spec = load_case_spec(self.spec_path)
        self.state_path = self.task / "repair" / "CROSS_HOST_STATE.json"
        self.state = (_read_json(self.state_path) if self.state_path.is_file() else {
            "schema": STATE_SCHEMA, "case_id": self.spec["case_id"],
            "status": "CPU_PREPARED", "entries": [], "attempts": [],
        })
        if self.state.get("case_id") != self.spec["case_id"]:
            raise CaseContractError("state case_id differs from current manifest")

    def save(self) -> None:
        _write_json_atomic(self.state_path, self.state)

    def inventory_requests(self) -> dict:
        hosts = sorted({candidate["host"] for candidate in self.spec["candidates"]})
        prefix = f"{self.spec['case_id'][:32]}-inv-{time.time_ns()}"
        result = RECOVER.fleet_inventory_batch(self.project, self.task, hosts,
                                               request_prefix=prefix)
        self.state.setdefault("attempts", []).append({
            "kind": "inventory", "at": time.time(), "hosts": hosts,
            "requests": result.get("requests", []),
        })
        self.save()
        return result

    def record_selection(self, first_scan: dict, second_scan: dict) -> dict:
        """Persist the two fresh scans that authorize a later device submit."""
        first = first_scan.get("candidates", first_scan)
        second = second_scan.get("candidates", second_scan)
        selected = select_two_phase(first, second, self.spec["roles"])
        for role, candidate in selected.items():
            if (candidate.get("inventory_at") is None or
                    not isinstance(candidate.get("inventory_proof"), dict)):
                raise CaseContractError(f"selected candidate lacks fresh inventory proof: {role}")
        self.state["selection"] = {
            "verified": True, "first": first_scan, "second": second_scan,
            "selected": selected, "at": time.time(),
        }
        self.save()
        return selected

    def submit_role(self, role: str, *, host: str, job: str,
                    request_id: str, script: str, devices: list[int], args: list[str]) -> dict:
        role = _identifier(role, "role")
        _identifier(job, "job")
        _identifier(request_id, "request_id")
        role_spec = self.spec["roles"].get(role)
        if role_spec is None or role_spec["host"] != host:
            raise CaseContractError("submit role/host is not bound to the current case")
        if not isinstance(devices, list) or not devices or any(type(value) is not int for value in devices):
            raise CaseContractError("a role submit requires an explicit device list")
        entries = self.state.setdefault("entries", [])
        existing = next((entry for entry in entries if entry.get("role") == role), None)
        expected = {"role": role, "host": host, "job": job, "request_id": request_id,
                    "script": script, "devices": devices, "args": args}
        if existing:
            if any(existing.get(key) != value for key, value in expected.items()):
                raise CaseContractError("existing role identity differs; do not replay")
            return {"status": "adopted_existing_intent", "entry": existing}
        request_path = self.task / "fleet" / "requests" / f"{request_id}.json"
        if request_path.is_file():
            request = _read_json(request_path)
            if any(request.get(key) != value for key, value in {
                    "request_id": request_id, "host": host, "job": job,
                    "script": script, "devices": devices, "args": args}.items()):
                raise CaseContractError("existing fleet request differs; do not overwrite")
            entry = expected | {"state": "queued", "adopted": True}
            entries.append(entry); self.save()
            return {"status": "adopted_existing_request", "entry": entry}
        selection = self.state.get("selection")
        selected = selection.get("selected", {}).get(role) if isinstance(selection, dict) else None
        if not isinstance(selection, dict) or selection.get("verified") is not True:
            raise CaseContractError("two-phase fresh candidate selection is required before submit")
        if (not isinstance(selected, dict) or selected.get("host") != host or
                selected.get("device_ids") != sorted(set(devices))):
            raise CaseContractError("submit does not match the current two-phase selected candidate")
        receipt = RECOVER.fleet_submit(self.project, self.task, request_id, "run",
                                       host, job, script, devices, args)
        entry = expected | {"state": "submitted", "receipt": receipt,
                            "attempt": len(self.state.setdefault("attempts", [])) + 1}
        entries.append(entry)
        self.state["attempts"].append({"kind": "submit", "at": time.time(),
                                        "role": role, "request_id": request_id,
                                        "job": job, "receipt": receipt})
        self.save()
        return receipt

    def status(self) -> dict:
        observations = {}
        for entry in self.state.get("entries", []):
            observations[entry["role"]] = {
                "entry": entry,
                "status": RECOVER.fleet_job_status(
                self.project, self.task, entry["host"], entry["job"], wait_seconds=0)
            }
        self.state["observations"] = observations
        self.save()
        return observations

    def collect_rank_evidence(self) -> dict:
        if self.spec.get("scenario", "multi_host_tp") != "multi_host_tp":
            raise CaseContractError(
                "rank-evidence is only valid for multi_host_tp; use the frozen P/D KV-transfer gate"
            )
        observations = self.state.get("observations")
        if set(entry.get("role") for entry in self.state.get("entries", [])) != set(self.spec["roles"]):
            raise CaseContractError("current role request/job entries are incomplete")
        if not isinstance(observations, dict) or set(observations) != set(self.spec["roles"]):
            observations = self.status()
        records = _remote_rank_records(self.spec, self.state, observations)
        rank_hosts = {}
        for role, role_spec in self.spec["roles"].items():
            for rank in role_spec["expected_ranks"]:
                rank_hosts[rank] = role_spec["host"]
        evidence = ADAPTER.rank_evidence(
            actor_records=records,
            expected_ranks=self.spec["expected_ranks"],
            expected_world=self.spec["world_size"],
            expected_hosts={role["host"] for role in self.spec["roles"].values()},
            expected_model=self.spec["model"], expected_rank_hosts=rank_hosts,
            case_id=self.spec["case_id"],
        )
        evidence["case_id"] = self.spec["case_id"]
        evidence["model"] = self.spec["model"]
        evidence["roles"] = records
        self.state["rank_evidence"] = evidence
        self.save()
        return evidence

    def decide_requeue(self, evidence: dict) -> dict:
        request_paths = {entry["request_id"] for entry in self.state.get("entries", [])
                         if (self.task / "fleet/requests" / f"{entry['request_id']}.json").is_file()}
        raw_observations = evidence.get("observations", self.state.get("observations", {}))
        observations = {}
        for role, value in (raw_observations or {}).items():
            if isinstance(value, dict) and isinstance(value.get("status"), dict):
                payload = dict(_response_payload(value["status"]))
                entry = value.get("entry")
                if isinstance(entry, dict):
                    payload["_request_binding"] = {key: entry.get(key)
                                                   for key in ("request_id", "host", "job")}
                observations[role] = payload
            else:
                observations[role] = value
        decision = safe_requeue_decision(
            intents=self.state.get("entries", []), request_paths=request_paths,
            observations=observations,
            cleanup=evidence.get("cleanup", self.state.get("cleanup")),
            expected_roles=set(self.spec["roles"]),
            unsubmitted_roles=evidence.get("unsubmitted_roles", {}),
            model_started=bool(evidence.get("model_started", False)),
        )
        if decision.get("allowed"):
            return archive_resource_requeue(self.task, self.state, decision)
        return decision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action", choices=("validate", "inventory", "select", "submit", "status", "rank-evidence", "requeue"), required=True)
    parser.add_argument("--role")
    parser.add_argument("--host")
    parser.add_argument("--job")
    parser.add_argument("--request-id")
    parser.add_argument("--script")
    parser.add_argument("--devices", type=int, nargs="*")
    parser.add_argument("--args-json", default="[]")
    parser.add_argument("--first-json", type=Path)
    parser.add_argument("--second-json", type=Path)
    parser.add_argument("--evidence-json", type=Path)
    args = parser.parse_args(argv)
    project = args.project.resolve(strict=True)
    task = (project / "runs" / _identifier(args.task, "task")).resolve(strict=True)
    case = CrossHostCase(project, task, args.manifest)
    if args.action == "validate":
        result = {"status": "CPU_PREPARED", "case_id": case.spec["case_id"],
                  "model": case.spec["model"], "roles": sorted(case.spec["roles"]),
                  "candidates": len(case.spec["candidates"]), "no_model_started": True}
    elif args.action == "inventory":
        result = case.inventory_requests()
    elif args.action == "select":
        first_path = _task_input(task, args.first_json, "CROSS_HOST_FIRST_SCAN.json")
        second_path = _task_input(task, args.second_json, "CROSS_HOST_SECOND_SCAN.json")
        result = case.record_selection(_read_json(first_path), _read_json(second_path))
    elif args.action == "submit":
        if not all((args.role, args.host, args.job, args.request_id, args.script)):
            parser.error("submit requires role, host, job, request-id and script")
        try:
            submit_args = json.loads(args.args_json)
        except ValueError as exc:
            raise CaseContractError("--args-json must be a JSON list") from exc
        if not isinstance(submit_args, list):
            raise CaseContractError("--args-json must be a JSON list")
        result = case.submit_role(args.role, host=args.host, job=args.job,
                                  request_id=args.request_id, script=args.script,
                                  devices=args.devices or [], args=submit_args)
    elif args.action == "status":
        result = case.status()
    elif args.action == "requeue":
        evidence_path = _task_input(task, args.evidence_json, "CROSS_HOST_REQUEUE_EVIDENCE.json")
        result = case.decide_requeue(_read_json(evidence_path))
    else:
        result = case.collect_rank_evidence()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "rank-evidence" and result.get("has_cross_host_ranks") is not True:
        return 1
    if args.action == "requeue":
        decision_action = result.get("action")
        if result.get("allowed") is False and decision_action != "adopt_active_job":
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaseContractError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": type(exc).__name__,
                          "detail": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
