"""Read-only container inventory; run by the remote agent, no device initialization."""
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        return {"argv": argv, "exit": p.returncode, "stdout": p.stdout[-30000:], "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "error": "timeout"}
    except OSError as exc:
        return {"argv": argv, "error": type(exc).__name__, "detail": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--personal-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    target = a.output.resolve()
    if not target.is_relative_to(root):
        raise ValueError("output outside project")
    # Inspect metadata only; importing torch_npu can initialize a device on some versions.
    packages = {}
    for pkg in ("torch", "torch-npu", "vllm", "vllm-ascend", "omni-npu", "transformers", "PyYAML"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    models, sources = [], []
    base = a.personal_root.resolve(strict=True)
    skip = {".git", "cache", ".cache", "node_modules", "envs", ".venv", "site-packages", "proc", "sys"}
    for current, dirs, files in os.walk(base):
        here = Path(current)
        if len(here.relative_to(base).parts) >= 7:
            dirs[:] = []
        dirs[:] = [v for v in dirs if v not in skip and not v.startswith(".")]
        if "config.json" in files:
            try:
                config = json.loads((here / "config.json").read_text())
                if "model_type" in config or "architectures" in config:
                    models.append({"path": str(here), "model_type": config.get("model_type"),
                                   "architectures": config.get("architectures"), "dtype": config.get("torch_dtype"),
                                   "weights_present": bool(list(here.glob("*.safetensors")) or list(here.glob("*.bin")))})
            except (OSError, ValueError):
                pass
        if here.name in ("vllm", "omni-npu", "vllm-ascend", "pangu-deploy"):
            sources.append(str(here))
    result = {"uid": os.getuid(), "python": shutil.which("python3"), "packages": packages,
              "tools": {v: shutil.which(v) for v in ("msprof", "npu-smi", "tmux", "gcc", "cmake")},
              "devices": command(["npu-smi", "info"]), "models": models, "sources": sources,
              "project": str(root), "device_computation": "not_started"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
