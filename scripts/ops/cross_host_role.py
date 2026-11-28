"""Collect and publish one current cross-host role's actor evidence.

This is intended to run inside the selected host/container job.  It calls the
public Pangu adapter against that role's own Ray session directory, binds the
result to the current case/request/job identity, and atomically merges it into
the job's ``validation.json`` while preserving its existing stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import cross_host_case as CASE  # noqa: E402


def _job_path(project: Path, raw: str | Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project / path
    path = CASE._path_under(path, project, label)
    if path.is_symlink() or not path.is_dir():
        raise CASE.CaseContractError(f"{label} must be a real directory: {path}")
    if path.name != path.name.strip() or path.name in {"", ".", ".."}:
        raise CASE.CaseContractError(f"invalid {label}: {path}")
    return path


def _identity_file(job: Path, name: str, expected: dict) -> None:
    path = job / name
    if not path.is_file() or path.is_symlink():
        return
    value = CASE._read_json(path)
    for key in ("case_id", "case", "model", "role", "host", "job", "request_id"):
        if key in value and value[key] != expected.get(key):
            raise CASE.CaseContractError(
                f"current job identity mismatch in {name}: {key}={value[key]!r}"
            )


def _require_fleet_config(project: Path, job_root: Path, expected: dict) -> dict:
    """Consume the real registered_fleet_bridge runner config, not a fixture."""
    path = job_root / "config.json"
    if path.is_symlink() or not path.is_file():
        raise CASE.CaseContractError(f"current fleet job config.json is missing: {path}")
    value = CASE._read_json(path)
    if value.get("project") != str(project) or value.get("job") != expected["job"]:
        raise CASE.CaseContractError("fleet config project/job identity differs")
    for key in ("management", "container", "script"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise CASE.CaseContractError(f"fleet config lacks {key}")
    if (not isinstance(value.get("args"), list) or
            any(not isinstance(arg, str) for arg in value["args"])):
        raise CASE.CaseContractError("fleet config args are invalid")
    devices = value.get("devices")
    if (not isinstance(devices, list) or any(type(device) is not int for device in devices)):
        raise CASE.CaseContractError("fleet config devices are invalid")
    return value


def _require_role_metadata(job: Path, expected: dict) -> None:
    """Require current role/plan metadata to carry case/request/rank facts."""
    candidates = []
    for name in ("role-spec.json", "plan.json"):
        path = job / name
        if path.is_file() and not path.is_symlink():
            value = CASE._read_json(path)
            candidates.append((name, value))
            for key in ("case_id", "case", "request_id", "role", "host"):
                if key in value and key in expected and value[key] != expected[key]:
                    raise CASE.CaseContractError(f"current {name} identity mismatch: {key}")
    if not candidates:
        raise CASE.CaseContractError("current role-spec.json or plan.json is missing")
    has_case_request = any(
        (value.get("case_id") or value.get("case")) == expected["case_id"] and
        value.get("request_id") == expected["request_id"]
        for _, value in candidates
    )
    has_ranks = any(
        isinstance(value.get("expected_ranks"), list) and
        isinstance(value.get("expected_world", value.get("world_size", value.get("world"))), int)
        for _, value in candidates
    )
    if not has_case_request or not has_ranks:
        raise CASE.CaseContractError("current role/plan metadata lacks case/request/rank facts")


def collect_publish(*, project: Path, job_root: Path, role: str, host: str,
                    task_id: str, case_id: str, model: str, request_id: str,
                    job: str, stage: str | None = None,
                    validation_path: Path | None = None) -> dict:
    """Collect one role locally and publish a complete actor envelope."""
    project = Path(project).resolve(strict=True)
    job_root = _job_path(project, job_root, "job root")
    role = CASE._identifier(role, "role")
    host = CASE._canonical_host(host)
    task_id = CASE._identifier(task_id, "task_id")
    case_id = CASE._identifier(case_id, "case_id")
    request_id = CASE._identifier(request_id, "request_id")
    job = CASE._identifier(job, "job")
    if job_root.name != job:
        raise CASE.CaseContractError("job root basename does not match current job")
    expected = {"case_id": case_id, "model": model, "role": role,
                "host": host, "job": job, "request_id": request_id}
    _require_fleet_config(project, job_root, expected)
    _require_role_metadata(job_root, expected)
    _identity_file(job_root, "role-spec.json", expected)
    _identity_file(job_root, "plan.json", expected)
    _identity_file(job_root, "job-config.json", expected)

    validation_path = (job_root / "validation.json" if validation_path is None
                       else CASE._path_under(validation_path, job_root, "validation path"))
    existing = CASE._read_json(validation_path) if validation_path.is_file() else {}
    existing_stage = existing.get("stage")
    if existing_stage is not None and not isinstance(existing_stage, str):
        raise CASE.CaseContractError("existing validation stage is invalid")
    publish_stage = existing_stage or stage
    if not isinstance(publish_stage, str) or not publish_stage.strip():
        raise CASE.CaseContractError("stage is required when validation has no stage")

    role_spec = {role: {"host": host, "task_id": task_id,
                        "expected_ranks": _expected_ranks_from_job(job_root, role),
                        "expected_world": _expected_world_from_job(job_root),
                        "model": model, "job": job, "request_id": request_id}}
    records = CASE.ADAPTER.collect_actor_rank_evidence(
        project, role_spec, case_id=case_id, model=model)
    if len(records) != 1:
        raise CASE.CaseContractError("current role actor evidence is missing")
    envelope = dict(records[0])
    envelope["schema"] = "fullnet.actor-evidence.v1"
    for key, value in expected.items():
        if envelope.get(key) != value:
            raise CASE.CaseContractError(f"collected actor evidence identity mismatch: {key}")
    if not envelope.get("completion_facts"):
        raise CASE.CaseContractError("actor evidence has no loading completion fact")

    merged = dict(existing)
    merged.update({"schema": existing.get("schema", "tp.runtime-progress.v1"),
                   "stage": publish_stage, "observed_at": time.time(),
                   "case_id": case_id, "model": model, "role": role,
                   "host": host, "job": job, "request_id": request_id,
                   "actor_evidence": envelope})
    CASE._write_json_atomic(validation_path, merged)
    return {"status": "ACTOR_EVIDENCE_PUBLISHED", "stage": publish_stage,
            "validation": str(validation_path), "actor_evidence": envelope}


def _expected_ranks_from_job(job: Path, role: str) -> list[int]:
    for name in ("role-spec.json", "plan.json", "job-config.json"):
        path = job / name
        if not path.is_file() or path.is_symlink():
            continue
        value = CASE._read_json(path)
        for container in (value, value.get("roles", {}), value.get("role_specs", {})):
            if isinstance(container, dict):
                item = container.get(role) if container is not value else container
                if isinstance(item, dict) and isinstance(item.get("expected_ranks"), list):
                    return [int(rank) for rank in item["expected_ranks"]]
        if isinstance(value.get("expected_ranks"), list):
            return [int(rank) for rank in value["expected_ranks"]]
    raise CASE.CaseContractError("current job config lacks expected_ranks")


def _expected_world_from_job(job: Path) -> int:
    for name in ("role-spec.json", "plan.json", "job-config.json"):
        path = job / name
        if not path.is_file() or path.is_symlink():
            continue
        value = CASE._read_json(path)
        for key in ("expected_world", "world_size", "world"):
            if type(value.get(key)) is int and value[key] > 0:
                return value[key]
    raise CASE.CaseContractError("current job config lacks expected world size")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(os.environ.get("FULLNET_REMOTE_PROJECT", PROJECT_ROOT)))
    parser.add_argument("--job-root", type=Path, default=Path(os.environ.get("FULLNET_FLEET_JOB_ROOT", "")))
    parser.add_argument("--role", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--validation-path", type=Path)
    args = parser.parse_args(argv)
    if not str(args.job_root):
        parser.error("--job-root or FULLNET_FLEET_JOB_ROOT is required")
    result = collect_publish(
        project=args.project, job_root=args.job_root, role=args.role,
        host=args.host, task_id=args.task_id, case_id=args.case_id,
        model=args.model, request_id=args.request_id, job=args.job,
        stage=args.stage, validation_path=args.validation_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CASE.CaseContractError, ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": type(exc).__name__,
                          "detail": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
