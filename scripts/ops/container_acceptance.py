"""Remote skill/CLI acceptance with a local HTTP fixture and an asset inventory.

No model inference or NPU initialization. Invoked by the approved remote agent.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--personal-root", type=Path, required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    run = (root / "runs" / a.task).resolve(strict=True)
    if not run.is_relative_to(root) or root.name != "vllm":
        raise ValueError("invalid project task")
    env = os.environ.copy()
    for key, suffix in {"HOME":"home", "XDG_CACHE_HOME":"cache", "PIP_CACHE_DIR":"cache/pip",
                        "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp", "HF_HOME":"cache/huggingface"}.items():
        target = run / suffix
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    env["FULLNET_PYTHON"] = sys.executable
    results = []
    def call(label, argv):
        with (run / (label + ".log")).open("w") as output:
            result = subprocess.run(argv, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
        record = {"step": label, "argv": argv, "exit": result.returncode}
        results.append(record)
        (run / "steps.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if result.returncode:
            raise RuntimeError("acceptance step failed: " + label)
    try:
        import yaml
    except ImportError:
        deps = root / "envs/fullnet-deps"
        call("private-dependencies", [sys.executable, "-B", "-m", "pip", "install", "--target", str(deps), "PyYAML>=6,<7"])
        env["PYTHONPATH"] += os.pathsep + str(deps)
    call("inventory", [sys.executable, "-B", str(root / "scripts/ops/container_inventory.py"),
                       "--project", str(root), "--personal-root", str(a.personal_root), "--output", str(run / "inventory.json")])
    prefix = [sys.executable, "-B", "-m", "fullnet"]
    call("doctor", prefix + ["doctor"])
    call("download-plan", prefix + ["assets", "plan", "--model", "qwen3-0.6b"])
    call("check", prefix + ["check", "--config", "examples/offline/config.yaml"])
    call("plan", prefix + ["plan", "--config", "examples/offline/config.yaml", "--run", a.task])
    call("run", prefix + ["run", "--model", "workflow-fixture", "--run", a.task])
    call("report", prefix + ["report", "--model", "workflow-fixture", "--run", a.task])
    with (run / "skill-acceptance.json").open("w") as output:
        json.dump({"workflow": "fullnet-validate", "fixture_only": True, "steps": results,
                   "model_inference": "not_run", "quality_on_real_model": "not_run"}, output, indent=2)


if __name__ == "__main__":
    main()
