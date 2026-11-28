"""Frozen fullnet stage contracts and evidence-based acceptance.

This module deliberately keeps orchestration evidence separate from the old
``state.json`` phase names.  A process exit code, a successful HTTP fixture,
or a termination signal is not, by itself, business acceptance evidence.
"""
from __future__ import annotations

import json
import copy
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .assets import check_local_assets
from .device_mapping import bind_runtime_mapping, parse_npu_smi_mapping
from .pangu7b_device_guard import verify_npu_smi_output
from .store import read_json, state, utc, write_json


CONTRACT_VERSION = 2
HARDWARE_PLATFORMS = {"a2", "a3", "a5"}
STAGES = (
    "assets",
    "device_gate",
    "model_import",
    "startup",
    "smoke",
    "pd_kv",
    "evaluation",
    "quality",
    "profile",
    "performance",
    "cleanup",
)


class AcceptanceBlocked(RuntimeError):
    """A frozen gate is missing or has invalid evidence."""


def _settings(config):
    value = config.get("acceptance", {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("acceptance must be a mapping")
    return value


def _deferred(config, settings, quality_only=None):
    measurements = config.get("measurements", {})
    values = measurements.get("deferred_stages", []) if isinstance(measurements, dict) else []
    if not isinstance(values, list) or any(v not in {"profile", "performance"} for v in values):
        raise ValueError("measurements.deferred_stages must contain only profile/performance")
    mode = settings.get("mode", "full")
    if quality_only is True:
        mode = "quality-only"
    if mode not in {"full", "quality-only"}:
        raise ValueError("acceptance.mode must be full or quality-only")
    deferred = set(values)
    if mode == "quality-only":
        deferred.update({"profile", "performance"})
    return mode, sorted(deferred)


def freeze_contract(config, sample_count, *, quality_only=None):
    """Build the immutable contract written by :func:`execution.plan`."""
    settings = _settings(config)
    mode, deferred = _deferred(config, settings, quality_only)
    hardware = config.get("platform") in HARDWARE_PLATFORMS
    measurements = config.get("measurements", {})
    measurements = measurements if isinstance(measurements, dict) else {}
    pd_mode = settings.get("pd_mode") or measurements.get("pd_mode")
    pd_required = bool(settings.get("pd_kv_required", pd_mode in {"split", "mixed"}))
    preflight = settings.get("preflight", {})
    if preflight is None:
        preflight = {}
    if not isinstance(preflight, dict):
        raise ValueError("acceptance.preflight must be a mapping")
    expected = settings.get("expected_samples", sample_count)
    if type(expected) is not int or expected <= 0:
        raise ValueError("acceptance.expected_samples must be a positive integer")
    if expected != sample_count:
        raise ValueError(f"frozen sample contract {expected} differs from dataset {sample_count}")
    freshness_ttl = settings.get("freshness_ttl_seconds", 300)
    if type(freshness_ttl) is not int or not 1 <= freshness_ttl <= 3600:
        raise ValueError("acceptance.freshness_ttl_seconds must be an integer in [1, 3600]")
    _validate_new_plan_requirements(config, hardware, mode, deferred, pd_required, preflight)
    required = {stage: True for stage in STAGES}
    if not hardware:
        for stage in ("assets", "device_gate", "model_import", "pd_kv", "profile", "performance", "cleanup"):
            if stage in {"profile", "performance"} and measurements:
                required[stage] = stage not in deferred
            else:
                required[stage] = bool(settings.get(f"{stage}_required", False))
    if not pd_required:
        required["pd_kv"] = False
    for stage in deferred:
        required[stage] = False
    required["quality"] = True
    return {
        "schema": "fullnet_acceptance_contract",
        "version": CONTRACT_VERSION,
        "mode": mode,
        "platform": config.get("platform"),
        "hardware_validation": hardware,
        "devices": [str(v) for v in config.get("devices", [])],
        "identity": copy.deepcopy(config.get("identity", {})),
        "python": (config.get("service", {}).get("argv") or [None])[0],
        "service_argv": copy.deepcopy(config.get("service", {}).get("argv", [])),
        "expected_samples": expected,
        "freshness_ttl_seconds": freshness_ttl,
        "quality": {
            "minimum_score": config.get("evaluation", {}).get("minimum_score"),
            "status_when_missing_threshold": "NOT_SPECIFIED",
        },
        "pd_kv_required": pd_required,
        "pd_mode": pd_mode,
        "deferred_stages": deferred,
        "required_stages": required,
        "preflight": preflight,
        "created_at": utc(),
    }


def initialize(run, contract):
    write_json(Path(run) / "acceptance.json", {
        "contract": contract,
        "stages": {},
        "execution_completed": False,
        "business_acceptance": "INCOMPLETE",
        "reasons": ["execution has not completed"],
        "updated_at": utc(),
    })


def _read_acceptance(run):
    path = Path(run) / "acceptance.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
        raise ValueError("acceptance.json is not a valid frozen acceptance record")
    config_path = Path(run) / "config.json"
    if config_path.is_file():
        frozen = read_json(config_path).get("acceptance_contract")
        if not isinstance(frozen, dict) or value["contract"] != frozen:
            raise AcceptanceBlocked("acceptance.json contract differs from frozen plan; create a new plan")
    return value


def record(run, stage, status, **evidence):
    if stage not in STAGES and stage not in {"before_evaluation", "pd_split_probe", "pd_mixed_probe"}:
        raise ValueError(f"unknown acceptance stage: {stage}")
    value = _read_acceptance(run)
    if value is None:
        return
    value.setdefault("stages", {})[stage] = {"status": status, "at": utc(), **evidence}
    value["updated_at"] = utc()
    write_json(Path(run) / "acceptance.json", value)
    state(run, stage, status, **evidence)


def _validate_new_plan_requirements(config, hardware, mode, deferred, pd_required, preflight):
    """Reject incomplete new hardware plans before a model process can start."""
    if not hardware:
        return
    smoke = config.get("service", {}).get("smoke", {})
    if not isinstance(smoke, dict) or smoke.get("enabled") is not True:
        raise ValueError("hardware plan requires service.smoke.enabled=true")
    for stage in ("assets", "device_gate", "model_import"):
        spec = preflight.get(stage)
        if not isinstance(spec, dict) or spec.get("adapter") not in {
                "assets-check", "npu-smi", "model-import", "fleet-evidence", "external-structured"}:
            raise ValueError(f"acceptance.preflight.{stage} needs a supported adapter")
    cleanup_spec = preflight.get("cleanup")
    if isinstance(cleanup_spec, dict) and cleanup_spec.get("adapter") == "fleet-evidence" \
            and not isinstance(cleanup_spec.get("evidence_path"), str):
        raise ValueError("acceptance.preflight.cleanup.evidence_path is required for fleet-evidence")
    if mode == "full":
        measurements = config.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError("full hardware plan requires a frozen measurements mapping")
        workloads = measurements.get("workloads")
        profile_workload = measurements.get("profile_workload")
        if "performance" not in deferred and (not isinstance(workloads, list) or not workloads):
            raise ValueError("full hardware plan requires measurements.workloads")
        if "profile" not in deferred and not isinstance(profile_workload, dict):
            raise ValueError("full hardware plan requires measurements.profile_workload")
        if ("performance" not in deferred
                and (type(measurements.get("requests", 0)) is not int or measurements["requests"] <= 0)):
            raise ValueError("full hardware plan requires positive measurements.requests")
    if pd_required and not (preflight.get("pd_kv") or config.get("measurements", {}).get("pd_mode")):
        # The actual split/mixed probe is an existing measurements adapter.  A
        # frozen pd_mode is enough to select it; no generic JSON command is
        # accepted as a substitute for that implementation.
        raise ValueError("P/D plan requires frozen acceptance.pd_mode or measurements.pd_mode")


def _command_spec(contract, stage):
    spec = contract.get("preflight", {}).get(stage)
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"acceptance.preflight.{stage} must be an adapter mapping")
    if spec.get("adapter") == "external-structured" and not isinstance(spec.get("argv"), list):
        raise ValueError(f"acceptance.preflight.{stage}.argv is required for external-structured")
    if spec.get("adapter") == "fleet-evidence" and not isinstance(spec.get("evidence_path"), str):
        raise ValueError(f"acceptance.preflight.{stage}.evidence_path is required for fleet-evidence")
    if spec.get("adapter") not in {"assets-check", "npu-smi", "model-import", "fleet-evidence", "external-structured"}:
        raise ValueError(f"unsupported preflight adapter for {stage}")
    if "argv" in spec and (not isinstance(spec["argv"], list) or any(not isinstance(v, str) or not v for v in spec["argv"])):
        raise ValueError(f"acceptance.preflight.{stage}.argv contains an invalid argument")
    return spec


def _json_from_stdout(stdout):
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Permit a command to log before its final JSON object without treating
        # arbitrary text as acceptance evidence.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        try:
            value = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            return None
    return value if isinstance(value, dict) else None


def _identity_matches(observed, expected):
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    return all(observed.get(key) == value for key, value in expected.items())


def _parse_utc(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _fresh_observation(evidence, contract):
    value = evidence.get("observed_at")
    parsed = _parse_utc(value)
    if parsed is None:
        return False, "observed_at must be ISO-8601 with a UTC offset"
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    if age < 0:
        return False, "evidence observed_at is in the future"
    if age > contract.get("freshness_ttl_seconds", 300):
        return False, "evidence is older than the frozen freshness budget"
    return True, None


def _evidence_ok(stage, evidence, contract, *, nonce=None, adapter=None, check_freshness=True):
    if not isinstance(evidence, dict) or evidence.get("status") not in {"passed", "PASS", "ok"}:
        return False, "command did not return a passed JSON evidence object"
    if evidence.get("run") != str(contract.get("run")):
        return False, "evidence run does not exactly identify this experiment"
    if not _identity_matches(evidence.get("identity"), contract.get("identity", {})):
        return False, "evidence identity does not match the frozen plan"
    if check_freshness:
        fresh, reason = _fresh_observation(evidence, contract)
        if not fresh:
            return False, reason
    elif _parse_utc(evidence.get("observed_at")) is None:
        return False, "observed_at must be ISO-8601 with a UTC offset"
    if not evidence.get("source"):
        return False, "evidence lacks source binding"
    if nonce is not None and evidence.get("nonce") != nonce:
        return False, "evidence nonce does not come from this command invocation"
    if adapter == "fleet-evidence":
        job = evidence.get("source_job")
        if not isinstance(job, dict) or job.get("exit") != 0 or not job.get("job"):
            return False, "fleet evidence lacks a successful source job binding"
        if (job.get("run") or job.get("business_run")) != str(contract.get("run")):
            return False, "fleet source_job is not bound to this experiment run"
        if "devices" in job and [str(v) for v in job["devices"]] != [str(v) for v in contract.get("devices", [])]:
            return False, "fleet source_job devices do not match the frozen contract"
    expected = set(contract.get("devices", []))
    observed = evidence.get("devices") or evidence.get("runtime_devices")
    if stage == "device_gate":
        if evidence.get("healthy") is not True:
            return False, "device evidence does not prove healthy=true"
        if evidence.get("idle") is not True or evidence.get("external_processes") != []:
            return False, "device evidence does not prove idle with no external processes"
        if not isinstance(observed, list) or {str(v) for v in observed} != expected:
            return False, "device evidence does not match the frozen device contract"
        mapping = evidence.get("mapping") or evidence.get("device_mapping")
        if not isinstance(mapping, list) or len(mapping) != len(expected):
            return False, "device evidence has no complete physical/runtime mapping"
        runtime_ids = set()
        for row in mapping:
            if not isinstance(row, dict) or row.get("runtime_device") is None or row.get("physical_id") is None:
                return False, "device mapping row lacks physical_id/runtime_device"
            runtime_ids.add(str(row["runtime_device"]))
        if runtime_ids != expected:
            return False, "device mapping runtime devices do not match the frozen contract"
    if stage == "model_import":
        required = ("python", "python_path", "model_class", "classfile", "plugins_loaded", "plugin_loader_status")
        if any(not evidence.get(key) for key in required):
            return False, "model import evidence lacks python/classfile/plugin details"
        if evidence.get("plugins_loaded") is not True or evidence.get("plugin_loader_status") not in {"passed", "PASS"}:
            return False, "model plugin loading was not proven"
        if contract.get("python") and evidence.get("python") != contract.get("python"):
            return False, "model import used a different Python executable"
    return True, None


def _stored_freshness(run, stage, item, contract, phases):
    """Return (recorded, valid) for a preflight-time freshness decision."""
    if item.get("freshness_checked") is not True:
        return False, False
    checked = _parse_utc(item.get("checked_at"))
    started = _parse_utc(item.get("started_at"))
    if started is None and stage == "cleanup":
        started = _parse_utc(phases.get("cleanup", {}).get("cleanup_started_at"))
    if checked is None or started is None or checked < started:
        return True, False
    if item.get("freshness_ttl_seconds") != contract.get("freshness_ttl_seconds", 300):
        return True, False
    if checked > datetime.now(timezone.utc):
        return True, False
    startup = _parse_utc(phases.get("startup", {}).get("at"))
    if stage != "cleanup" and startup is not None and checked > startup:
        return True, False
    receipt_name = "cleanup-verified.json" if stage == "cleanup" else f"preflight-{stage}.json"
    receipt = Path(run) / "validation" / receipt_name
    try:
        saved = read_json(receipt)
    except (OSError, ValueError):
        return True, False
    if (saved.get("stage") != stage or saved.get("checked_at") != item.get("checked_at")
            or saved.get("freshness_ttl_seconds") != item.get("freshness_ttl_seconds")
            or saved.get("evidence") != item.get("evidence")):
        return True, False
    return True, True


def _adapter_environment(env, run, nonce, devices=None):
    child = dict(env)
    child["FULLNET_RUN"] = str(Path(run).resolve())
    child["FULLNET_PREFLIGHT_NONCE"] = nonce
    if devices is None:
        config_path = Path(run) / "config.json"
        devices = read_json(config_path).get("devices", []) if config_path.is_file() else []
    child["FULLNET_PREFLIGHT_DEVICES"] = json.dumps(devices)
    return child


def _model_import_probe(run, config, contract, spec, env, root, nonce):
    model = contract.get("identity", {}).get("weights")
    python = spec.get("python") or contract.get("python")
    if not isinstance(python, str) or not python:
        raise AcceptanceBlocked("model-import adapter needs the selected model Python")
    probe = Path(run) / "validation" / "model-import-probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    source = """import importlib, inspect, json, os, sys
result = {"status": "failed", "run": os.environ.get("FULLNET_RUN"),
          "nonce": os.environ.get("FULLNET_PREFLIGHT_NONCE"),
          "python": sys.executable, "python_path": list(sys.path),
          "model_context_argv": list(sys.argv[1:]),
          "identity": %r, "source": "fullnet.model-import.adapter",
          "observed_at": %r, "plugins_loaded": False,
          "plugin_loader_status": "not_run"}
try:
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    result["plugins_loaded"] = True
    result["plugin_loader_status"] = "passed"
except Exception as exc:
    result["plugin_loader_status"] = type(exc).__name__
    result["error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False)); raise SystemExit(2)
try:
    model_path = %r
    architectures = []
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
            architectures = json.load(stream).get("architectures", [])
    except Exception:
        pass
    registry = importlib.import_module("vllm.model_executor.models.registry")
    holder = getattr(registry, "ModelRegistry", registry)
    resolver = getattr(holder, "resolve_model_cls", None) or getattr(holder, "load_model_cls", None)
    if resolver is None:
        raise RuntimeError("vllm model registry has no resolve_model_cls/load_model_cls")
    resolved = resolver(architectures)
    architecture = None
    if isinstance(resolved, tuple):
        cls, architecture = resolved[0], (resolved[1] if len(resolved) > 1 else None)
    else:
        cls = resolved
    # Recent registries return LazyRegisteredModel; older registries return the
    # class directly, and some return (class, architecture). Resolve the lazy
    # entry only through its public load_model_cls contract.
    if hasattr(cls, "load_model_cls") and not inspect.isclass(cls):
        cls = cls.load_model_cls()
    result.update(status="passed", model_class=cls.__name__, classfile=inspect.getfile(cls),
                  resolved_architecture=architecture,
                  registry_resolution="lazy.load_model_cls" if architecture is None and hasattr(resolved, "load_model_cls") else "resolve_model_cls")
except Exception as exc:
    result.update(error=type(exc).__name__, detail=str(exc))
    print(json.dumps(result, ensure_ascii=False)); raise SystemExit(3)
print(json.dumps(result, ensure_ascii=False))
""" % (contract.get("identity", {}), utc(), model)
    probe.write_text(source, encoding="utf-8")
    frozen_service_argv = contract.get("service_argv") or []
    context_argv = list(frozen_service_argv[1:]) if len(frozen_service_argv) > 1 else []
    if "--model" not in context_argv:
        context_argv.extend(["--model", str(model)])
    probe_argv = [python, "-B", str(probe), *context_argv]
    completed = subprocess.run(probe_argv, cwd=root,
                               env=_adapter_environment(env, run, nonce, contract.get("devices", [])), capture_output=True,
                               text=True, timeout=float(spec.get("timeout", 300)), check=False)
    return probe_argv, completed


def _npu_smi_probe(run, config, contract, spec, env, root, nonce):
    tool = spec.get("tool")
    if not isinstance(tool, str) or not tool:
        raise AcceptanceBlocked("npu-smi adapter needs a host-matched tool path")
    child = _adapter_environment(env, run, nonce, contract.get("devices", []))
    outputs = []
    mapping_query = subprocess.run([tool, "info", "-m"], cwd=root, env=child,
                                   capture_output=True, text=True, timeout=35, check=False)
    mapping_text = mapping_query.stdout + mapping_query.stderr
    if mapping_query.returncode:
        return [tool, "info", "-m"], mapping_query
    mapping_run = spec.get("mapping_argv")
    if not isinstance(mapping_run, list) or not mapping_run:
        return [tool, "info", "-m"], subprocess.CompletedProcess([tool, "info", "-m"], 1, "",
            "npu-smi adapter requires a fresh ACL physical-to-runtime mapping_argv")
    mapping = subprocess.run(mapping_run, cwd=root, env=child, capture_output=True, text=True,
                             timeout=float(spec.get("mapping_timeout", 120)), check=False)
    if mapping.returncode:
        return mapping_run, mapping
    try:
        acl = _json_from_stdout(mapping.stdout)
        bound = bind_runtime_mapping(mapping_text, acl, spec["host"],
                                     required_board_ids=spec.get("required_board_ids"))
    except Exception as exc:
        return mapping_run, subprocess.CompletedProcess(mapping_run, 1, "", str(exc))
    expected_runtime = set(contract.get("devices", []))
    selected = [row for row in bound["rows"] if str(row.get("runtime_device")) in expected_runtime]
    if {str(row.get("runtime_device")) for row in selected} != expected_runtime:
        return mapping_run, subprocess.CompletedProcess(mapping_run, 1, "",
            "ACL mapping has no complete selected runtime-device set")
    for row in selected:
        board = row.get("board")
        # ``info -t health`` on real A3/A5 containers often omits the device
        # name.  The existing parser is intentionally fail-closed on that
        # shape, so obtain the complete official ``info`` table first and
        # combine it with the selected board's health/proc receipts.
        full_info = subprocess.run([tool, "info"], cwd=root, env=child,
                                   capture_output=True, text=True, timeout=35, check=False)
        health = subprocess.run([tool, "info", "-t", "health", "-i", str(board)], cwd=root,
                                env=child, capture_output=True, text=True, timeout=35, check=False)
        proc = subprocess.run([tool, "info", "-t", "proc-mem", "-i", str(board)], cwd=root,
                              env=child, capture_output=True, text=True, timeout=35, check=False)
        if full_info.returncode:
            return [tool, "info"], full_info
        if health.returncode or proc.returncode:
            return [tool, "info", "-t", "health/proc-mem", "-i", str(board)], subprocess.CompletedProcess(
                [tool, "info", "-t", "health/proc-mem", "-i", str(board)],
                health.returncode or proc.returncode, health.stdout + proc.stdout,
                health.stderr + proc.stderr)
        text = (full_info.stdout + full_info.stderr + health.stdout + health.stderr
                + proc.stdout + proc.stderr)
        try:
            verify_npu_smi_output(text, contract["platform"], int(board))
        except Exception as exc:
            return [tool, "info", "-t", "health/proc-mem", "-i", str(board)], subprocess.CompletedProcess(
                [tool, "info", "-t", "health/proc-mem", "-i", str(board)], 1, "", str(exc))
        outputs.append({"runtime_device": str(row["runtime_device"]), "board": board,
                        "full_info_returncode": full_info.returncode,
                        "health_returncode": health.returncode,
                        "proc_returncode": proc.returncode, "stdout": text})
    normalized_mapping = [{**row, "physical_id": row["chip_phy"],
                           "source_chip_phy": row["chip_phy"]} for row in selected]
    evidence = {"status": "passed", "run": str(Path(run).resolve()), "nonce": nonce,
                "identity": contract.get("identity", {}), "source": "fullnet.npu-smi.adapter",
                "observed_at": utc(), "devices": contract.get("devices", []), "healthy": True,
                "idle": True, "external_processes": [], "mapping": normalized_mapping,
                "health_records": outputs}
    return mapping_run, subprocess.CompletedProcess(mapping_run, 0, json.dumps(evidence), "")


def run_preflight(run, config, env, root):
    """Run fresh controlled preflight commands before the model service."""
    value = _read_acceptance(run)
    if value is None:
        raise AcceptanceBlocked("experiment has no frozen acceptance contract; create a new plan")
    contract = value["contract"]
    hardware = contract.get("hardware_validation", False)
    for stage in ("assets", "device_gate", "model_import"):
        if not hardware and not contract["required_stages"].get(stage):
            record(run, stage, "not_applicable", evidence_kind="fixture", hardware_validation=False)
            continue
        spec = _command_spec(contract, stage)
        if spec is None:
            record(run, stage, "blocked", reason=f"missing fresh {stage} command evidence")
            raise AcceptanceBlocked(f"missing fresh {stage} preflight evidence")
        started = utc()
        nonce = uuid.uuid4().hex
        argv = spec.get("argv") or [spec["adapter"]]
        try:
            adapter = spec["adapter"]
            if adapter == "assets-check":
                weights = contract.get("identity", {}).get("weights")
                checked = check_local_assets(weights)
                evidence = {"status": "passed" if checked.get("status") == "files_present" else "failed",
                            "run": str(Path(run).resolve()), "nonce": nonce,
                            "identity": contract.get("identity", {}), "source": "fullnet.assets.check_local",
                            "observed_at": utc(), "assets": checked}
                completed = subprocess.CompletedProcess(["fullnet", "assets", "check"], 0 if evidence["status"] == "passed" else 1,
                                                         json.dumps(evidence, ensure_ascii=False), "")
            elif adapter == "model-import":
                argv, completed = _model_import_probe(run, config, contract, spec, env, root, nonce)
            elif adapter == "npu-smi":
                argv, completed = _npu_smi_probe(run, config, contract, spec, env, root, nonce)
            elif adapter == "fleet-evidence":
                path = Path(spec.get("evidence_path", ""))
                if not path.is_absolute():
                    path = Path(root) / path
                evidence = read_json(path)
                completed = subprocess.CompletedProcess(["fleet-evidence", str(path)], 0,
                                                         json.dumps(evidence, ensure_ascii=False), "")
                argv = completed.args
            else:
                argv = spec["argv"]
                child_env = _adapter_environment(env, run, nonce)
                completed = subprocess.run(argv, cwd=root, env=child_env, capture_output=True,
                                           text=True, timeout=float(spec.get("timeout", 300)), check=False)
            if adapter == "model-import" and completed.returncode == 0:
                evidence = _json_from_stdout(completed.stdout)
            elif adapter == "npu-smi" and completed.returncode == 0:
                evidence = _json_from_stdout(completed.stdout)
            elif adapter in {"fleet-evidence", "assets-check"}:
                evidence = _json_from_stdout(completed.stdout)
            else:
                evidence = _json_from_stdout(completed.stdout)
            if adapter == "fleet-evidence":
                # A fleet job is the producer of this evidence; it cannot echo
                # the local nonce because it runs before this supervisor.
                ok, reason = _evidence_ok(stage, evidence, contract, adapter=adapter)
            else:
                ok, reason = _evidence_ok(stage, evidence, contract, nonce=nonce, adapter=adapter)
            freshness_checked = completed.returncode == 0 and ok
            checked_at = utc() if freshness_checked else None
            if freshness_checked:
                receipt = Path(run) / "validation" / f"preflight-{stage}.json"
                write_json(receipt, {"stage": stage, "checked_at": checked_at,
                                     "freshness_ttl_seconds": contract.get("freshness_ttl_seconds", 300),
                                     "evidence": evidence})
            record(run, stage, "passed" if completed.returncode == 0 and ok else "failed",
                   adapter=adapter, argv=completed.args, cwd=str(root), started_at=started,
                   returncode=completed.returncode, stdout=completed.stdout,
                   stderr=completed.stderr, evidence=evidence,
                   freshness_checked=freshness_checked, checked_at=checked_at,
                   freshness_ttl_seconds=contract.get("freshness_ttl_seconds", 300),
                   reason=None if completed.returncode == 0 and ok else (reason or "preflight command failed"))
            if completed.returncode != 0 or not ok:
                raise AcceptanceBlocked(f"{stage} preflight evidence failed")
        except subprocess.TimeoutExpired as exc:
            record(run, stage, "failed", adapter=spec["adapter"], argv=argv, cwd=str(root), started_at=started,
                   returncode=None, stdout=exc.stdout or "", stderr=exc.stderr or "", reason="timeout")
            raise AcceptanceBlocked(f"{stage} preflight timed out") from exc
    return True


def verify_cleanup(run, config, env, root, *, keep_ready=False):
    """Record resource release separately from signalling the owned process."""
    if keep_ready:
        record(run, "cleanup", "deferred_by_user", resource_verification="deferred_by_user",
               reason="keep_ready was explicitly requested")
        return "deferred_by_user"
    if config.get("platform") not in HARDWARE_PLATFORMS:
        record(run, "cleanup", "not_applicable", resource_verification="not_applicable",
               hardware_validation=False)
        return "not_applicable"
    value = _read_acceptance(run)
    spec = (value or {}).get("contract", {}).get("preflight", {}).get("cleanup")
    if isinstance(spec, dict) and spec.get("adapter") == "fleet-evidence":
        try:
            ok, reason, evidence = _cleanup_receipt(run, config)
            contract = (value or {}).get("contract", {})
            checked_at = utc() if ok else None
            if ok:
                write_json(Path(run) / "validation" / "cleanup-verified.json", {
                    "stage": "cleanup", "checked_at": checked_at,
                    "freshness_ttl_seconds": contract.get("freshness_ttl_seconds", 300),
                    "evidence": evidence})
            record(run, "cleanup", "passed" if ok else "failed", resource_verification="passed" if ok else "failed",
                   adapter="fleet-evidence", evidence=evidence,
                   freshness_checked=ok, checked_at=checked_at,
                   freshness_ttl_seconds=contract.get("freshness_ttl_seconds", 300),
                   reason=None if ok else (reason or "fresh cleanup evidence failed"))
            return "passed" if ok else "failed"
        except (OSError, ValueError, KeyError) as exc:
            record(run, "cleanup", "failed", resource_verification="failed",
                   adapter="fleet-evidence", reason=f"cleanup evidence unreadable: {type(exc).__name__}")
            return "failed"
    if isinstance(spec, dict) and isinstance(spec.get("argv"), list):
        nonce = uuid.uuid4().hex
        try:
            completed = subprocess.run(spec["argv"], cwd=root, env=_adapter_environment(env, run, nonce), capture_output=True,
                                       text=True, timeout=float(spec.get("timeout", 120)), check=False)
            evidence = _json_from_stdout(completed.stdout)
            expected = (value or {}).get("contract", {})
            ok = (completed.returncode == 0 and isinstance(evidence, dict)
                  and evidence.get("status") in {"passed", "PASS", "ok"}
                  and evidence.get("run") == str(Path(run).resolve())
                  and evidence.get("nonce") == nonce
                  and _identity_matches(evidence.get("identity"), expected.get("identity", {}))
                  and evidence.get("devices") == expected.get("devices")
                  and evidence.get("service_released") is True
                  and evidence.get("device_processes") == []
                  and evidence.get("ports") == [])
            record(run, "cleanup", "passed" if ok else "failed", resource_verification="passed" if ok else "failed",
                   argv=spec["argv"], returncode=completed.returncode, stdout=completed.stdout,
                   stderr=completed.stderr, evidence=evidence, reason=None if ok else "fresh cleanup evidence failed")
            return "passed" if ok else "failed"
        except subprocess.TimeoutExpired:
            record(run, "cleanup", "failed", resource_verification="failed", reason="cleanup verification timed out")
            return "failed"
    record(run, "cleanup", "unknown", resource_verification="unknown",
           reason="process signal is not proof of device/port release")
    return "unknown"


def _cleanup_receipt(run, config):
    """Read and validate a fleet cleanup receipt after the owned stop window."""
    value = _read_acceptance(run)
    if value is None:
        return False, "legacy experiment has no frozen acceptance contract", None
    contract = value["contract"]
    phases = read_json(Path(run) / "state.json").get("phases", {})
    cleanup = phases.get("cleanup", {})
    stopped = _parse_utc(cleanup.get("cleanup_stopped_at"))
    started = _parse_utc(cleanup.get("cleanup_started_at"))
    if started is None or stopped is None or stopped < started:
        return False, "cleanup stop window is missing or invalid", None
    spec = contract.get("preflight", {}).get("cleanup", {})
    if not isinstance(spec, dict) or spec.get("adapter") != "fleet-evidence":
        return False, "cleanup verification requires the fleet-evidence adapter", None
    path = Path(str(spec.get("evidence_path", "")).replace("${RUN}", str(Path(run).resolve())))
    if not path.is_absolute():
        path = Path(run) / path
    try:
        evidence = read_json(path)
    except (OSError, ValueError) as exc:
        return False, f"cleanup receipt is missing or unreadable: {type(exc).__name__}", None
    ok, reason = _evidence_ok("cleanup", evidence, contract, adapter="fleet-evidence")
    observed = _parse_utc(evidence.get("observed_at")) if isinstance(evidence, dict) else None
    if not ok:
        return False, reason or "cleanup receipt failed evidence validation", evidence
    if observed is None or observed <= stopped:
        return False, "cleanup receipt was observed before or at the owned stop time", evidence
    if evidence.get("service_released") is not True or evidence.get("device_processes") != [] \
            or evidence.get("ports") != []:
        return False, "cleanup receipt does not prove service/device/port release", evidence
    return True, None, evidence


def verify_run(run, config):
    """Verify a completed run without starting a service or rerunning workload."""
    value = _read_acceptance(run)
    if value is None:
        return {"execution_completed": None, "business_acceptance": "LEGACY_READ_ONLY",
                "verification_status": "not_run",
                "acceptance_reasons": ["legacy experiment is read-only"]}
    contract = value["contract"]
    reasons = []
    if contract.get("hardware_validation"):
        ok, reason, evidence = _cleanup_receipt(run, config)
        checked_at = utc()
        if ok:
            receipt = Path(run) / "validation" / "cleanup-verified.json"
            write_json(receipt, {"stage": "cleanup", "checked_at": checked_at,
                                 "freshness_ttl_seconds": contract.get("freshness_ttl_seconds", 300),
                                 "evidence": evidence})
            cleanup_before = read_json(Path(run) / "state.json").get("phases", {}).get("cleanup", {})
            record(run, "cleanup", "passed", adapter="fleet-evidence", evidence=evidence,
                   freshness_checked=True, checked_at=checked_at,
                   freshness_ttl_seconds=contract.get("freshness_ttl_seconds", 300),
                   resource_verification="passed")
            state(run, "cleanup", "process_group_signalled",
                  **{key: cleanup_before[key] for key in ("service_exit", "cleanup_started_at", "cleanup_stopped_at") if key in cleanup_before},
                  device_release="passed", resource_verification="passed", cleanup_verified_at=checked_at)
        else:
            reasons.append(reason or "cleanup receipt rejected")
            cleanup_before = read_json(Path(run) / "state.json").get("phases", {}).get("cleanup", {})
            record(run, "cleanup", "failed" if evidence else "unknown",
                   adapter="fleet-evidence", evidence=evidence,
                   resource_verification="failed" if evidence else "unknown", reason=reason)
            state(run, "cleanup", "process_group_signalled" if cleanup_before.get("status") != "failed" else "failed",
                  **{key: cleanup_before[key] for key in ("service_exit", "cleanup_started_at", "cleanup_stopped_at") if key in cleanup_before},
                  device_release="failed" if evidence else "unknown",
                  resource_verification="failed" if evidence else "unknown")
    from .analysis import report
    summary = summarize(run, config)
    if reasons:
        summary["acceptance_reasons"] = list(dict.fromkeys(reasons + summary.get("acceptance_reasons", [])))
        summary["business_acceptance"] = "INCOMPLETE"
        acceptance_path = Path(run) / "acceptance.json"
        acceptance = read_json(acceptance_path)
        acceptance["reasons"] = summary["acceptance_reasons"]
        acceptance["business_acceptance"] = "INCOMPLETE"
        write_json(acceptance_path, acceptance)
    report(run)
    return {**summary, "verification_status": "passed" if summary.get("business_acceptance") == "ACCEPTED" else "incomplete"}


def _stage_status(value, stage):
    return value.get("stages", {}).get(stage, {}).get("status")


def _fresh_evaluation(run, config):
    """Re-score the frozen dataset and raw predictions; never trust results.json."""
    prediction_path = Path(run) / "raw" / "predictions.jsonl"
    dataset_path = Path(run) / "dataset.jsonl"
    if not prediction_path.is_file() or not dataset_path.is_file():
        return None, "raw evaluation evidence is missing"
    samples = []
    predictions = []
    try:
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                predictions.append(json.loads(line))
        from .evaluation import score
        fresh = score(samples, predictions, config["evaluation"])
        expected = (read_json(Path(run) / "acceptance.json")["contract"]["expected_samples"]
                    if (Path(run) / "acceptance.json").is_file() else len(samples))
        if len(samples) != expected:
            return None, f"frozen sample count is {len(samples)}, expected {expected}"
        return fresh, None
    except Exception as exc:
        return None, f"raw evaluation evidence is invalid: {type(exc).__name__}: {exc}"


def _fresh_stage_status(run, config, contract, stage, value, phases):
    """Derive acceptance stage status from raw artifacts, not editable labels."""
    if stage in {"assets", "device_gate", "model_import"}:
        item = value.get("stages", {}).get(stage, {})
        evidence = item.get("evidence")
        adapter = item.get("adapter")
        if not isinstance(evidence, dict):
            return "missing"
        recorded, stored_ok = _stored_freshness(run, stage, item, contract, phases)
        if recorded and not stored_ok:
            return "failed"
        if adapter == "fleet-evidence":
            ok, _ = _evidence_ok(stage, evidence, contract, adapter=adapter,
                                 check_freshness=not recorded)
        else:
            ok, _ = _evidence_ok(stage, evidence, contract,
                                 nonce=(evidence or {}).get("nonce"), adapter=adapter,
                                 check_freshness=not recorded)
        return "passed" if ok and item.get("returncode") == 0 else "failed"
    if stage == "smoke":
        if not (Path(run) / "smoke.json").is_file():
            return "missing"
        try:
            return "passed" if read_json(Path(run) / "smoke.json").get("status") == "passed" else "failed"
        except (OSError, ValueError):
            return "failed"
    if stage == "pd_kv":
        filename = "pd-split-probe.json" if contract.get("pd_mode") == "split" else "pd-mixed-probe.json"
        path = Path(run) / "performance" / filename
        try:
            return "passed" if read_json(path).get("acceptance") == "passed" else "failed"
        except Exception:
            return "failed"
    if stage == "profile":
        try:
            from .measurements import _verify_captured_profile_contract, profile_artifact_index
            artifact = read_json(Path(run) / "performance" / "profile-artifacts.json")
            direct = _verify_captured_profile_contract(run, config)
            index = profile_artifact_index(run)
            return "passed" if (phases.get(stage, {}).get("status") == "passed"
                                  and artifact.get("acceptance") in {"accepted", "passed"}
                                  and direct.get("status") == "PASS"
                                  and index.get("has_device_kernel_evidence") is True) else "failed"
        except (OSError, ValueError):
            return "missing"
    if stage == "performance":
        try:
            summary_path = Path(run) / "performance" / "summary.json"
            if not summary_path.is_file():
                return "missing"
            summary = read_json(summary_path)
            measurements = config.get("measurements", {})
            workloads = measurements.get("workloads", [])
            requests = measurements.get("requests")
            complete = (phases.get(stage, {}).get("status") == "passed"
                        and isinstance(summary, list) and len(summary) == len(workloads)
                        and all(row.get("completed") == requests and row.get("workload") == workload
                                for row, workload in zip(summary, workloads)))
            return "passed" if complete else "failed"
        except Exception:
            return "failed"
    if stage == "cleanup":
        cleanup = phases.get("cleanup", {})
        if cleanup.get("status") == "failed" or cleanup.get("resource_verification") in {"failed", "unknown", "deferred_by_user"}:
            return cleanup.get("status", "failed")
        item = value.get("stages", {}).get(stage, {})
        evidence = item.get("evidence")
        if config.get("platform") in HARDWARE_PLATFORMS:
            contract_devices = (contract.get("devices") or [])
            if item.get("adapter") == "fleet-evidence":
                recorded, stored_ok = _stored_freshness(run, "cleanup", item, contract, phases)
                if recorded:
                    fresh = stored_ok
                else:
                    fresh, _ = _evidence_ok("cleanup", evidence, contract, adapter="fleet-evidence") if isinstance(evidence, dict) else (False, "missing")
            else:
                fresh = True
            if (not isinstance(evidence, dict) or item.get("status") != "passed"
                    or not fresh
                    or evidence.get("run") != str(Path(run).resolve())
                    or not _identity_matches(evidence.get("identity"), contract.get("identity", {}))
                    or evidence.get("devices") != contract_devices
                    or evidence.get("service_released") is not True
                    or evidence.get("device_processes") != []
                    or evidence.get("ports") != []):
                return "failed" if evidence else "missing"
            return "passed" if cleanup.get("resource_verification") == "passed" else "failed"
        return "not_applicable"
    return phases.get(stage, {}).get("status", _stage_status(value, stage) or "missing")


def summarize(run, config, result=None):
    """Compute independent execution and business outcomes from fresh evidence."""
    value = _read_acceptance(run)
    if value is None:
        # Old runs remain readable but are intentionally not upgraded in place.
        return {"execution_completed": None, "business_acceptance": "LEGACY_READ_ONLY",
                "acceptance_reasons": ["legacy experiment has no frozen acceptance contract"]}
    contract = value["contract"]
    state_value = read_json(Path(run) / "state.json") if (Path(run) / "state.json").is_file() else {"phases": {}}
    phases = state_value.get("phases", {})
    evaluation, evaluation_error = _fresh_evaluation(run, config)
    if evaluation is None:
        evaluation = {}
        reasons = [evaluation_error or "raw evaluation evidence is unavailable"]
    else:
        reasons = []
    protocol = bool(evaluation.get("protocol_pass")) and phases.get("evaluation", {}).get("status") == "passed"
    if not protocol:
        reasons.append("evaluation protocol is not complete")
    if evaluation.get("quality_status") != "passed":
        reasons.append(f"quality_status={evaluation.get('quality_status', 'not_run')}")
    for stage, required in contract.get("required_stages", {}).items():
        if not required:
            continue
        status = _fresh_stage_status(run, config, contract, stage, value, phases)
        if status != "passed":
            reasons.append(f"{stage} evidence is {status or 'missing'}")
    if contract.get("hardware_validation") and contract.get("mode") != "quality-only":
        cleanup = phases.get("cleanup", {})
        if cleanup.get("status") == "failed" or cleanup.get("resource_verification") != "passed":
            reasons.append("resource release is not independently verified")
    execution_completed = protocol and phases.get("cleanup", {}).get("status") in {"process_group_signalled", "not_applicable", "unknown", "passed"}
    if not execution_completed:
        reasons.insert(0, "execution did not complete its frozen protocol")
    if not contract.get("hardware_validation"):
        reasons.append("fixture/offline evidence cannot validate hardware")
    business = "ACCEPTED" if not reasons else "INCOMPLETE"
    value.update(execution_completed=bool(execution_completed), business_acceptance=business,
                 reasons=list(dict.fromkeys(reasons)), updated_at=utc())
    write_json(Path(run) / "acceptance.json", value)
    return {"execution_completed": bool(execution_completed), "business_acceptance": business,
            "acceptance_reasons": list(dict.fromkeys(reasons)),
            "acceptance_contract_version": contract.get("version")}
