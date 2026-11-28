"""Shared acceptance-contract wiring for the three formal Pangu runners."""
from __future__ import annotations

import copy
from pathlib import Path

from fullnet.acceptance import freeze_contract
from fullnet.config import load


REQUIRED_PREFLIGHT = ("assets", "device_gate", "model_import", "cleanup")
SUPPORTED_ADAPTERS = {
    "assets-check", "npu-smi", "model-import", "fleet-evidence", "external-structured",
}


def _read(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"acceptance config is missing or symlinked: {path}")
    return load(path)


def load_acceptance_config(path: Path | None, task: Path, *, expected_samples=48,
                           quality_only=False) -> tuple[dict, dict]:
    """Read a task-local full config and its acceptance segment, fail closed."""
    if path is None:
        raise ValueError(
            "missing --acceptance-config; provide the current task-local full config "
            "containing acceptance.preflight and the frozen evaluation contract"
        )
    task = Path(task).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(task):
        raise ValueError("--acceptance-config must be inside the current task directory")
    raw = _read(path)
    acceptance = raw.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("acceptance config must contain an acceptance mapping")
    mode = acceptance.get("mode")
    if mode not in {"full", "quality-only"}:
        raise ValueError("acceptance.mode must be full or quality-only")
    if acceptance.get("expected_samples") != expected_samples:
        raise ValueError(f"acceptance.expected_samples must equal {expected_samples}")
    preflight = acceptance.get("preflight")
    if not isinstance(preflight, dict):
        raise ValueError("acceptance.preflight is required and must be a mapping")
    for stage in REQUIRED_PREFLIGHT:
        spec = preflight.get(stage)
        if not isinstance(spec, dict) or spec.get("adapter") not in SUPPORTED_ADAPTERS:
            raise ValueError(f"acceptance.preflight.{stage} needs a supported adapter")
        if spec.get("adapter") == "fleet-evidence" and not isinstance(spec.get("evidence_path"), str):
            raise ValueError(f"acceptance.preflight.{stage}.evidence_path is required")
    # Validate the supplied complete config before device/model/download work.
    # quality_only only defers profile/performance; it does not waive preflight.
    probe = copy.deepcopy(raw)
    if quality_only:
        probe.setdefault("acceptance", {})["mode"] = "quality-only"
    freeze_contract(probe, expected_samples, quality_only=quality_only)
    return copy.deepcopy(acceptance), raw


def plan_with_acceptance(plan_fn, root, config: dict, run_id: str,
                         acceptance: dict, *, quality_only=False):
    """Pass the frozen source acceptance segment through the real plan seam."""
    value = copy.deepcopy(config)
    value["acceptance"] = copy.deepcopy(acceptance)
    # This repeats the core validation immediately before plan and protects
    # against a runner accidentally dropping fields while constructing c.
    freeze_contract(value, 48, quality_only=quality_only)
    return plan_fn(root, value, run_id, quality_only=quality_only or None)


def update_result_status(status: dict, result: dict, *, quality_only=False) -> dict:
    """Record protocol/quality/counts after run_experiment, independent of callbacks."""
    status.update(
        score=result.get("score"),
        quality_status=result.get("quality_status"),
        protocol_pass=result.get("protocol_pass"),
        attempted=result.get("attempted"),
        succeeded=result.get("succeeded"),
        failed=result.get("failed"),
        skipped=result.get("skipped"),
        timeout=result.get("timeout"),
        samples=result.get("samples"),
    )
    status["execution_completed"] = result.get("execution_completed")
    status["business_acceptance"] = result.get("business_acceptance") \
        if isinstance(result.get("business_acceptance"), str) else "INCOMPLETE"
    status["acceptance_reasons"] = result.get("acceptance_reasons", [])
    if result.get("protocol_pass") is not True:
        status["protocol_error"] = "protocol_pass is not true"
    if result.get("quality_status") == "failed":
        status["quality_error"] = "quality_status is failed"
    return status
