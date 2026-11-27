from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import comparison, numeric_parity, report
from .assets import assets, doctor
from .setup import configure
from .export import export_run
from .config import resolve
from .execution import plan, run_experiment
from .measurements import collect_performance, collect_profile, pd_mixed_probe, pd_split_probe
from .store import experiment, read_json, write_json


def _positive_int(value, name, *, allow_zero=False):
    valid = type(value) is int and (value >= 0 if allow_zero else value > 0)
    if not valid:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"measurements.{name} must be a {bound} integer")


def _workload_shape(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"measurements.{name} must be a mapping")
    for field in ("input", "output", "concurrency"):
        _positive_int(value.get(field), f"{name}.{field}")


def _has_profiler_config(argv):
    if not isinstance(argv, list):
        return False
    for index, value in enumerate(argv):
        if value == "--profiler-config" and index + 1 < len(argv) and argv[index + 1]:
            return True
        if isinstance(value, str) and value.startswith("--profiler-config=") and value[len("--profiler-config="):]:
            return True
    return False


def _positive_context_limit(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _measurement_context_limit(config):
    """Return the explicit service/evaluation context contract, if any."""
    service = config.get("service", {})
    argv = service.get("argv") if isinstance(service, dict) else None
    service_limit = None
    service_source = None
    if isinstance(argv, (list, tuple)):
        for index, value in enumerate(argv):
            candidate = None
            source = None
            if value == "--max-model-len":
                source = "service.argv --max-model-len"
                if index + 1 < len(argv):
                    candidate = argv[index + 1]
            elif isinstance(value, str) and value.startswith("--max-model-len="):
                source = "service.argv --max-model-len="
                candidate = value[len("--max-model-len="):]
            if source is not None:
                service_limit = _positive_context_limit(candidate)
                service_source = source if service_limit is not None else None
    if service_limit is not None:
        return service_limit, service_source

    evaluation = config.get("evaluation", {})
    evaluation_limit = evaluation.get("context_limit") if isinstance(evaluation, dict) else None
    evaluation_limit = _positive_context_limit(evaluation_limit)
    if evaluation_limit is not None:
        return evaluation_limit, "evaluation.context_limit"
    return None, None


def _validate_workload_context(workload, name, limit, source):
    total = workload["input"] + workload["output"]
    if total > limit:
        raise ValueError(
            f"{name} exceeds {source} limit {limit}: "
            f"input + output = {workload['input']} + {workload['output']} = {total}"
        )


def _validate_measurement_contract(config, args):
    if not (args.profile or args.performance):
        return
    measurements = config.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("requested measurement stages require a measurements mapping; create a new plan")
    deferred = measurements.get("deferred_stages", [])
    if not isinstance(deferred, list):
        raise ValueError("measurements.deferred_stages must be a list in the frozen config")
    conflicts = [stage for stage, enabled in (("profile", args.profile), ("performance", args.performance))
                 if enabled and stage in deferred]
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"requested measurement stage is deferred in frozen config: {joined}; create a new plan without deferred_stages")
    if args.performance:
        workloads = measurements.get("workloads")
        if not isinstance(workloads, list) or not workloads:
            raise ValueError("measurements.workloads must be a non-empty list for --performance")
        for index, workload in enumerate(workloads):
            _workload_shape(workload, f"workloads[{index}]")
        _positive_int(measurements.get("requests", 32), "requests")
        _positive_int(measurements.get("warmups", 2), "warmups", allow_zero=True)
        _positive_int(measurements.get("timeout_per_group", 3600), "timeout_per_group")
    if args.profile:
        profile_workload = measurements.get("profile_workload", {"input": 4096, "output": 128, "concurrency": 1})
        _workload_shape(profile_workload, "profile_workload")
    context_limit, context_source = _measurement_context_limit(config)
    if context_limit is not None:
        if args.profile:
            _validate_workload_context(profile_workload, "measurements.profile_workload", context_limit, context_source)
        if args.performance:
            for index, workload in enumerate(measurements["workloads"]):
                _validate_workload_context(workload, f"measurements.workloads[{index}]", context_limit, context_source)
    if args.profile:
        env = config.get("env", {})
        service = config.get("service", {})
        env_enabled = isinstance(env, dict) and bool(env.get("VLLM_TORCH_PROFILER_DIR"))
        argv = service.get("argv") if isinstance(service, dict) else None
        if not env_enabled and not _has_profiler_config(argv):
            raise ValueError("--profile requires VLLM_TORCH_PROFILER_DIR or service.argv --profiler-config; create a new plan with server-side profile enabled")


def _measurement_callbacks(args, root):
    requested = args.profile or args.performance or args.pd_probe is not None
    if not requested:
        return {}
    config = read_json(experiment(root, args.model, args.run) / "config.json")
    _validate_measurement_contract(config, args)

    def before_evaluation(run, config):
        pd_split_probe(run, config)

    def after_evaluation(run, config, result):
        if args.pd_probe == "mixed":
            pd_mixed_probe(run, config)
        if args.profile:
            collect_profile(run, config)
        if args.performance:
            collect_performance(run, config)

    callbacks = {}
    if args.pd_probe == "split":
        callbacks["before_evaluation"] = before_evaluation
    if args.profile or args.performance or args.pd_probe == "mixed":
        callbacks["after_evaluation"] = after_evaluation
    return callbacks


def main(argv=None):
    p = argparse.ArgumentParser(prog="fullnet", description="Model-scoped inference experiments")
    p.add_argument("--root", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    cmd = sub.add_parser("export")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--run", required=True)
    cmd.add_argument("--destination", type=Path, required=True)
    cmd = sub.add_parser("configure")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--platform", choices=("a2", "a3", "a5"), required=True)
    cmd.add_argument("--python", type=Path, required=True)
    cmd.add_argument("--weights", type=Path, required=True)
    cmd.add_argument("--dataset", type=Path, required=True)
    cmd.add_argument("--official-metrics", type=Path, required=True)
    cmd.add_argument("--devices", nargs="+", type=int, required=True)
    cmd.add_argument("--context-limit", type=int, default=16384)
    cmd = sub.add_parser("assets")
    cmd.add_argument("action", choices=("plan", "verify-source", "fetch", "check"))
    cmd.add_argument("--model", required=True)
    for command in ("check", "plan"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", type=Path, required=True)
        if command == "plan":
            cmd.add_argument("--run", required=True)
    for command in ("run", "status", "report"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--run", required=True)
        if command == 'run':
            cmd.add_argument('--keep-ready', action='store_true', help='Retain the validated service in its task until stopped')
            cmd.add_argument('--profile', action='store_true', help='Collect the server-side device profile after evaluation')
            cmd.add_argument('--performance', action='store_true', help='Collect the frozen performance workloads after evaluation')
            cmd.add_argument('--pd-probe', choices=('mixed', 'split'),
                             help='Run split KV gate before evaluation, or mixed probe after evaluation')
    cmd = sub.add_parser("compare")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--baseline", required=True)
    cmd.add_argument("--candidate", required=True)
    cmd = sub.add_parser("parity")
    cmd.add_argument("--reference", type=Path, required=True)
    cmd.add_argument("--candidate", type=Path, required=True)
    cmd.add_argument("--atol", type=float, required=True)
    cmd.add_argument("--rtol", type=float, required=True)
    args = p.parse_args(argv)
    try:
        root = args.root.resolve()
        if args.command == "doctor":
            result = doctor()
        elif args.command == "export":
            result = export_run(root, experiment(root, args.model, args.run), args.destination)
        elif args.command == "configure":
            result = configure(root, args.model, args.platform, args.python, args.weights, args.dataset,
                               args.official_metrics, args.devices, args.context_limit)
        elif args.command == "assets":
            result = assets(root, args.model, args.action)
        elif args.command in ("check", "plan"):
            c = resolve(args.config, root)
            result = {"model": c["model"], "platform": c["platform"], "config_valid": True}
            if args.command == "plan":
                result["experiment"] = str(plan(root, c, args.run))
        elif args.command == "run":
            callbacks = _measurement_callbacks(args, root)
            result = run_experiment(root, args.model, args.run, keep_ready=args.keep_ready,
                                    **callbacks)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["protocol_pass"] and result["quality_status"] != "failed" else 1
        elif args.command == "status":
            result = read_json(experiment(root, args.model, args.run) / "state.json")
        elif args.command == "report":
            result = {"report": str(report(experiment(root, args.model, args.run)))}
        elif args.command == "compare":
            candidate = experiment(root, args.model, args.candidate)
            result = comparison(experiment(root, args.model, args.baseline), candidate)
            write_json(candidate / "comparison.json", result)
        else:
            result = numeric_parity(args.reference, args.candidate, args.atol, args.rtol)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["passed"] else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        p.exit(2, f"fullnet: {exc}\n")
