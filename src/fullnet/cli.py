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
from .store import experiment, read_json, write_json


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
    cmd.add_argument("--platform", choices=("a3", "a5"), required=True)
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
            result = run_experiment(root, args.model, args.run, keep_ready=args.keep_ready)
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
