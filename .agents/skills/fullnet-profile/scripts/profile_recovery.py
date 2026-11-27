"""Inspect and recover saved Ascend PROF data without running a model or NPU.

The exporter writes only a new output directory. The original capture is read-only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

COMPUTE_TYPES = {"kernelaic", "kernelaivec", "kernelmixaic", "kernelmixai", "aicore", "aivector", "aivectorcore", "aic", "aiv", "mixaic", "mixaiv"}

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def csv_evidence(path):
    accepted, other, task_counts = [], 0, {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fields = {normalized(k): v for k, v in row.items() if k}
            task = normalized(fields.get("tasktype", ""))
            task_counts[fields.get("tasktype", "unknown")] = task_counts.get(fields.get("tasktype", "unknown"), 0) + 1
            duration = next((v for k, v in fields.items() if "duration" in k and positive(v)), None)
            device = next((v for k, v in fields.items() if k in ("deviceid", "device") and str(v).isdigit()), None)
            # Explicit compute task types; device_id alone also labels communication/host rows.
            compute = task in COMPUTE_TYPES
            if compute and device is not None and duration is not None:
                accepted.append({"device": str(device), "task_type": fields.get("tasktype"),
                                 "duration": duration, "name": fields.get("opname", fields.get("name", ""))})
            else:
                other += 1
    return {"path": str(path), "compute_rows": len(accepted), "other_rows": other,
            "devices": sorted({r["device"] for r in accepted}), "task_type_counts": task_counts, "examples": accepted[:3]}


def timeline_evidence(path, source_device=None):
    # Reuse the strict repository reader, including JSON closure/trailing checks.
    reader = Path(__file__).resolve().parents[4] / "src/fullnet/profile_trace.py"
    spec = importlib.util.spec_from_file_location("fullnet_recovery_trace", reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata, threads, process_devices = {}, {}, {}
    for event in module.iter_trace_events(path):
        if event.get("ph") == "M":
            metadata.setdefault(str(event.get("pid")), []).append(str(event.get("args", {})))
            if event.get("name") == "thread_name":
                threads[(str(event.get("pid")), str(event.get("tid")))] = str(event.get("args", {}).get("name", ""))
            if event.get("name") in {"process_labels", "process_name"}:
                match = re.search(r"\b(?:NPU|Device)\s*(?:Id)?\s*[:=]?\s*(\d+)\b", str(event.get("args", {})), re.IGNORECASE)
                if match:
                    process_devices[str(event.get("pid"))] = match.group(1)
    process_descriptions = {pid: normalized(" ".join(values)) for pid, values in metadata.items()}
    accepted = []
    for event in module.iter_trace_events(path):
        if event.get("ph") != "X" or not positive(event.get("dur")):
            continue
        category = normalized(event.get("cat", ""))
        if category in {"cpuop", "queue", "pythonfunction", "userannotation", "runtime"}:
            continue
        args = event.get("args") or {}
        # Large official timelines also contain millions of non-kernel CCU/runtime events.
        if isinstance(args, dict) and "Task Type" not in args and category not in {"kernel", "aicore", "aivector", "aic", "aiv"}:
            if not any(normalized(key) == "tasktype" for key in args):
                continue
        fields = {normalized(k): v for k, v in args.items()} if isinstance(args, dict) else {}
        desc = process_descriptions.get(str(event.get("pid")), "")
        task = normalized(fields.get("tasktype", ""))
        lane = normalized(threads.get((str(event.get("pid")), str(event.get("tid"))), ""))
        if any(word in task + lane + category for word in ("aicpu", "hccl", "communication", "memcpy")):
            continue
        explicit = task in COMPUTE_TYPES
        # Ascend official exports use 'aicore' category or a Hardware process lane.
        device_lane = category in {"aicore", "aivector", "aic", "aiv"} or (category == "kernel" and "hardware" in desc and lane in COMPUTE_TYPES)
        if not (explicit or device_lane):
            continue
        accepted.append({"name": event.get("name"), "duration_us": event["dur"],
                         "category": event.get("cat"), "pid": event.get("pid"),
                         "device": fields.get("deviceid", fields.get("device", process_devices.get(str(event.get("pid")), source_device)))})
    return {"path": str(path), "compute_events": len(accepted), "examples": accepted[:3],
            "devices": sorted({str(x["device"]) for x in accepted if x["device"] is not None}),
            "unbound_events": sum(x["device"] is None for x in accepted), "source_device_scope": source_device,
            "complete": True}


def raw_inventory(root):
    root = Path(root).resolve()
    raws = [root] if root.name.startswith("PROF_") else sorted(p for p in root.rglob("PROF_*") if p.is_dir())
    output = []
    for raw in raws:
        devices = []
        for device in sorted(raw.glob("device_*")):
            data = device / "data"
            files = [p for p in data.rglob("*") if p.is_file()] if data.is_dir() else []
            nonempty = [p for p in files if p.stat().st_size > 0 and not p.name.endswith(".done")]
            devices.append({"name": device.name, "files": len(files), "nonempty_files": len(nonempty),
                            "bytes": sum(p.stat().st_size for p in files),
                            "examples": [p.name for p in nonempty[:8]]})
        output.append({"path": str(raw), "devices": devices})
    return output


def inspect(root, source_device=None):
    root = Path(root).resolve()
    csvs, timelines, errors = [], [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".csv" and (path.name.startswith("op_summary") or "kernel" in path.name.lower()):
                csvs.append(csv_evidence(path))
            elif path.name == "trace_view.json" or path.name.startswith("msprof_") and path.suffix == ".json" or path.name.endswith(".pt.trace.json.gz"):
                timelines.append(timeline_evidence(path, source_device))
        except (ValueError, OSError, csv.Error) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    raws = raw_inventory(root)
    csv_count = sum(x["compute_rows"] for x in csvs)
    timeline_count = sum(x["compute_events"] for x in timelines)
    raw_present = any(d["nonempty_files"] for r in raws for d in r["devices"])
    state = "device_export_present" if csv_count or timeline_count else "raw_present_export_missing" if raw_present else "device_capture_unconfirmed"
    return {"root": str(root), "classification": state, "acceptance": "device_evidence_present" if csv_count or timeline_count else "incomplete", "raw": raws, "csv": csvs,
            "timelines": timelines, "csv_compute_rows": csv_count,
            "timeline_compute_events": timeline_count, "parse_errors": errors,
            "device_evidence": bool(csv_count or timeline_count)}


def run_command(argv, env, cwd, prefix, timeout):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    launch_error = None
    launch_filename = launch_traceback = None
    try:
        completed = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout, stderr, code = exc.stdout or "", exc.stderr or "", None
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    except OSError as exc:
        launch_filename = exc.filename or str(argv[0])
        launch_traceback = traceback.format_exc()
        launch_error = f"{type(exc).__name__}: {exc}; filename={launch_filename!r}"
        stdout, stderr, code = "", launch_error, None
    Path(str(prefix) + ".stdout").write_text(stdout, encoding="utf-8")
    Path(str(prefix) + ".stderr").write_text(stderr, encoding="utf-8")
    # Official parsers can print a child traceback and still exit zero.
    fatal = bool(launch_error or re.search(r"Traceback \(most recent call last\)|ImportError:|ModuleNotFoundError:|ERR04012|ERR04005|error while loading shared libraries|Failed to export CANN[^\n]*Profiling data|export\s*\|\s*SUB_PROCESS\s*\|\s*Failed", stdout + stderr))
    result = {"argv": argv, "exit": code, "child_failure": fatal, "timed_out": timed_out,
              "passed": code == 0 and not fatal and not timed_out}
    if launch_error:
        result["launch_error"] = launch_error
        result.update(filename=launch_filename, traceback=launch_traceback)
    save(str(prefix) + ".json", result)
    return result


def manifest(root):
    return {str(p.relative_to(root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in root.rglob("*") if p.is_file()}


def first_error(stdout, stderr):
    """Return the most useful first diagnostic, preferring exception/error lines."""
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    exception = re.compile(r"^(?:ImportError|ModuleNotFoundError|OSError|RuntimeError|Exception)(?::| )", re.IGNORECASE)
    specific = re.compile(r"(?:error|failed|failure|exception)", re.IGNORECASE)
    return next((line for line in lines if exception.search(line)),
                next((line for line in lines if specific.search(line)), lines[0] if lines else "no diagnostic output"))


def parser_dependency_preflight(python, env, cwd, prefix, timeout):
    """Prove that the parser Python can import sqlite3 in the exact CLI environment."""
    code = (
        "import sqlite3, sys; "
        "print('python_executable=' + sys.executable); "
        "print('python_version=' + sys.version.replace('\\n', ' ')); "
        "print('sqlite_module_version=' + str(getattr(sqlite3, 'version', 'not_exposed'))); "
        "print('sqlite_runtime_version=' + str(sqlite3.sqlite_version))"
    )
    result = run_command([python, "-B", "-X", "utf8", "-c", code], env, cwd, prefix, timeout)
    stdout_path = Path(str(prefix) + ".stdout")
    stderr_path = Path(str(prefix) + ".stderr")
    stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    result.update({
        "stage": "parser_dependency_preflight",
        "stdout": stdout,
        "stderr": stderr,
        "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
    })
    if not result["passed"]:
        result["first_error"] = first_error(stdout, stderr)
    save(str(prefix) + ".json", result)
    return result


def export(args):
    source, output, cann = args.source.resolve(strict=True), args.output.resolve(), args.cann_root.resolve(strict=True)
    if output == source or source in output.parents or output in source.parents or output.exists():
        raise ValueError("use a new output directory outside the original capture")
    entry = args.entry.resolve(strict=True) if args.entry else cann / "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    if not entry.is_file():
        raise ValueError("official Python entry not found; select --entry after checking this SDK")
    raws = raw_inventory(source)
    if not raws or not any(d["nonempty_files"] for r in raws for d in r["devices"]):
        raise ValueError("no nonempty device raw data; diagnose capture before attempting export")
    output.mkdir(parents=True)
    # SDKs differ in owner checks. Record identity and preserve an actual CLI
    # rejection; do not reject a usable read-only official entry by uid alone.
    env = dict(getattr(args, "env", None) or os.environ)
    for name in ("home", "tmp", "cache"):
        (output / name).mkdir()
    env.update(HOME=str(output / "home"), TMPDIR=str(output / "tmp"), TMP=str(output / "tmp"), TEMP=str(output / "tmp"),
               XDG_CACHE_HOME=str(output / "cache"), ASCEND_WORK_PATH=str(output / "work"),
               ASCEND_HOME_PATH=str(cann), PYTHONDONTWRITEBYTECODE="1")
    if args.library_dir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(str(p.resolve()) for p in args.library_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    preflight = parser_dependency_preflight(args.python, env, output, output / "parser-dependency-preflight", args.timeout)
    context = {"source": str(source), "tool": str(entry), "python": args.python,
               "euid": os.geteuid() if hasattr(os, "geteuid") else None, "tool_owner": entry.stat().st_uid,
               "library_dirs": [str(p) for p in args.library_dir],
               "parser_dependency_preflight": preflight, "new_device_compute": False}
    save(output / "context.json", context)
    if not preflight["passed"]:
        receipt = {"status": "FAIL", "failure_stage": "parser_dependency_preflight",
                   "first_error": preflight.get("first_error", "parser dependency preflight failed"),
                   "parser_dependency_preflight": preflight, "official_cli": "not_run",
                   "original_metadata_unchanged": None, "new_device_compute": False,
                   "scope": "parser_dependency_preflight_only"}
        save(output / "RECOVERY.json", receipt)
        return receipt
    for name, command in [("cli-help", ["--help"]), ("import-help", ["import", "--help"]),
                          ("summary-help", ["export", "summary", "--help"]), ("timeline-help", ["export", "timeline", "--help"])]:
        help_result = run_command([args.python, "-B", str(entry), *command], env, output, output / name, args.timeout)
        help_text = Path(str(output / name) + ".stdout").read_text(encoding="utf-8")
        if not help_result["passed"] or name != "cli-help" and "--collection-dir" not in help_text:
            raise RuntimeError("BLOCKED_CLI_FAMILY: check help receipts and select this SDK's supported offline command family")
    results = []
    for index, item in enumerate(raws):
        original = Path(item["path"])
        if original.is_symlink() or any(p.is_symlink() for p in original.rglob("*")):
            raise ValueError("raw capture contains links; review and make a bounded regular-file snapshot first")
        work = output / f"rank-{index}"
        work.mkdir()
        copied = work / original.name
        before_source = manifest(original)
        save(work / "original-before.json", before_source)
        shutil.copytree(original, copied, copy_function=shutil.copy2)
        before_copy = manifest(copied)
        save(work / "source.json", {"original": str(original), "copy": str(copied)})
        commands = [("import", ["import", "--collection-dir", str(copied)]),
                    ("summary", ["export", "summary", "--collection-dir", str(copied), "--format", "csv"]),
                    ("timeline", ["export", "timeline", "--collection-dir", str(copied)])]
        steps = []
        for name, argv in commands:
            receipt = run_command([args.python, "-B", str(entry), *argv], env, output, work / name, args.timeout)
            steps.append(receipt)
            if not receipt["passed"]:
                break
        expected = {d["name"].removeprefix("device_") for d in item["devices"] if d["nonempty_files"]}
        evidence = inspect(copied, next(iter(expected)) if len(expected) == 1 else None)
        after_copy = manifest(copied)
        generated = {str(copied / path) for path, value in after_copy.items() if before_copy.get(path) != value}
        # Old exports copied from the capture cannot satisfy a no-op exporter.
        evidence["csv"] = [c for c in evidence["csv"] if c["path"] in generated]
        evidence["timelines"] = [t for t in evidence["timelines"] if t["path"] in generated]
        evidence["csv_compute_rows"] = sum(c["compute_rows"] for c in evidence["csv"])
        evidence["timeline_compute_events"] = sum(t["compute_events"] for t in evidence["timelines"])
        evidence["device_evidence"] = bool(evidence["csv_compute_rows"] or evidence["timeline_compute_events"])
        after_source = manifest(original)
        save(work / "original-after.json", after_source)
        save(work / "generated-files.json", sorted(generated))
        observed = {d for c in evidence["csv"] for d in c["devices"]}
        trace_devices = {d for t in evidence["timelines"] for d in t["devices"]}
        passed = len(steps) == 3 and all(s["passed"] for s in steps) and evidence["csv_compute_rows"] > 0 and evidence["timeline_compute_events"] > 0 and expected == observed == trace_devices and before_source == after_source
        result = {"original": str(original), "copy": str(copied), "steps": steps,
                  "expected_devices": sorted(expected), "observed_devices": sorted(observed), "timeline_devices": sorted(trace_devices), "original_metadata_unchanged": before_source == after_source, "passed": passed, "evidence": evidence}
        save(work / "verification.json", result)
        results.append(result)
    receipt = {"status": "EXPORTED" if results and all(r["passed"] for r in results) else "FAIL",
               "contract_acceptance": "pending_explicit_verify",
               "original_metadata_unchanged": all(r["original_metadata_unchanged"] for r in results),
               "parser_dependency_preflight": preflight, "new_device_compute": False,
               "scope": "discovered_profiles_only", "ranks": results}
    save(output / "RECOVERY.json", receipt)
    return receipt


def verify_exports(root, expected_profs, expected_devices):
    """Check a completed export against the capture contract, without re-exporting."""
    root = Path(root).resolve()
    records = []
    for work in sorted(root.glob("rank-*")):
        source_file = work / "source.json"
        if not source_file.is_file():
            continue
        source = json.loads(source_file.read_text(encoding="utf-8"))
        copied = Path(source["copy"]).resolve(strict=True)
        if not copied.is_relative_to(work.resolve()):
            raise ValueError("PROF copy resolves outside its rank work directory")
        raw = raw_inventory(copied)
        devices = {d["name"].removeprefix("device_") for r in raw for d in r["devices"] if d["nonempty_files"]}
        raw_metadata, metadata_errors = {}, []
        for device in sorted(devices):
            info = copied / f"device_{device}" / f"info.json.{device}"
            try:
                value = json.loads(info.read_text(encoding="utf-8"))
                declared = {str(v["id"]) for v in value["DeviceInfo"]}
                raw_metadata[device] = {"path": str(info), "devices": sorted(declared)}
                if declared != {device}:
                    metadata_errors.append(f"raw metadata device mismatch: {device}: {sorted(declared)}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                metadata_errors.append(f"raw device metadata unreadable: {info}: {exc}")
        scoped = next(iter(devices)) if len(devices) == 1 else None
        evidence = inspect(copied, scoped)
        csv_devices = {d for c in evidence["csv"] for d in c["devices"]}
        trace_devices = {d for t in evidence["timelines"] for d in t["devices"]}
        steps = [json.loads((work / (n + ".json")).read_text(encoding="utf-8")) if (work / (n + ".json")).exists() else {} for n in ("import", "summary", "timeline")]
        generated_file = work / "generated-files.json"
        generated = ({str(Path(p).resolve()) for p in json.loads(generated_file.read_text(encoding="utf-8"))}
                     if generated_file.exists() else set())
        new_csv = sum(c["compute_rows"] for c in evidence["csv"] if c["path"] in generated)
        new_trace = sum(t["compute_events"] for t in evidence["timelines"] if t["path"] in generated)
        before_path, after_path = work / "original-before.json", work / "original-after.json"
        metadata_same = before_path.exists() and after_path.exists() and json.loads(before_path.read_text(encoding="utf-8")) == json.loads(after_path.read_text(encoding="utf-8"))
        passed = (all(s.get("passed") for s in steps) and new_csv > 0 and new_csv == new_trace
                  and csv_devices == devices == trace_devices and metadata_same
                  and not metadata_errors and not evidence["parse_errors"])
        records.append({"rank_dir": str(work), "expected_devices_from_raw": sorted(devices),
                        "csv_devices": sorted(csv_devices), "timeline_devices": sorted(trace_devices),
                        "new_csv_compute_rows": new_csv, "new_timeline_compute_events": new_trace,
                        "original_metadata_unchanged": metadata_same, "passed": passed, "evidence": evidence})
        records[-1].update(raw_metadata_binding=raw_metadata, metadata_errors=metadata_errors)
    observed = {d for r in records for d in r["expected_devices_from_raw"]}
    contract = len(records) == expected_profs and observed == set(expected_devices)
    result = {"status": "PASS" if contract and all(r["passed"] for r in records) else "FAIL",
              "expected_profs": expected_profs, "expected_devices": sorted(expected_devices),
              "observed_profs": len(records), "observed_devices": sorted(observed),
              "passed_profs": sum(bool(r["passed"]) for r in records),
              "contract_matches": contract, "ranks": records, "new_device_compute": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("inspect")
    check.add_argument("root", type=Path)
    check.add_argument("--output", type=Path)
    recover = sub.add_parser("export")
    recover.add_argument("--source", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--cann-root", type=Path, required=True)
    recover.add_argument("--entry", type=Path)
    recover.add_argument("--python", default=sys.executable)
    recover.add_argument("--library-dir", type=Path, action="append", default=[])
    recover.add_argument("--timeout", type=int, default=900)
    verify = sub.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("--expected-profs", type=int, required=True)
    verify.add_argument("--expected-devices", nargs="+", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect(args.root)
        if args.output:
            save(args.output, result)
    elif args.command == "verify":
        result = verify_exports(args.root, args.expected_profs, args.expected_devices)
        save(args.output, result)
    else:
        try:
            result = export(args)
        except Exception as exc:
            result = {"status": "FAIL", "error": repr(exc), "new_device_compute": False,
                      "original_modification_check": "see per-rank manifests when available"}
            if args.output.is_dir():
                save(args.output / "FAILED.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status", "PASS") in {"PASS", "EXPORTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
