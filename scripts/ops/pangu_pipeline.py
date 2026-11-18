"""Download real Pangu weights, then execute the fixed model validation workflow."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--model-python", type=Path, required=True)
    p.add_argument("--platform", choices=("a3", "a5"), required=True)
    p.add_argument("--device-index", type=int, required=True)
    p.add_argument("--npu-id", type=int, required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    task = (root / "runs" / a.task).resolve(strict=True)
    if not task.is_relative_to(root):
        raise ValueError("task outside project")
    # A running fetch from this project must be inspected before any resubmission.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"fullnet" in args and b"fetch" in args and (proc / "cwd").resolve() == root:
                raise RuntimeError("another model fetch is active in this project")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    steps = []
    def call(label, argv, timeout):
        record = {"stage": label, "argv": argv, "started": time.time()}
        steps.append(record)
        (task / "pipeline.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2))
        with (task / (label + "-pipeline.log")).open("x") as output:
            child = subprocess.Popen(argv, cwd=root, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                record.update(exit=child.returncode, interrupted=True, ended=time.time())
                (task / "pipeline.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2))
                raise
        record.update(exit=code, ended=time.time())
        (task / "pipeline.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if code:
            raise RuntimeError(label + " failed; inspect its log before a new attempt")
    call("assets", [sys.executable, "-B", str(root / "scripts/ops/prepare_pangu_assets.py"),
         "--project", str(root), "--python", str(a.model_python), "--task", a.task,
         "--download-timeout", "14400"], timeout=15000)
    call("model", [sys.executable, "-B", str(root / "scripts/ops/validate_pangu7b.py"),
         "--project", str(root), "--runtime-root", str(a.runtime_root),
         "--model-python", str(a.model_python), "--weights", str(root / "assets/weights/pangu-7b"),
         "--platform", a.platform, "--device-index", str(a.device_index),
         "--npu-id", str(a.npu_id), "--task", a.task], timeout=14400)


if __name__ == "__main__":
    main()
