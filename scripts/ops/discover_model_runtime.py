"""Locate read-only model/runtime assets beneath an authorized personal root."""
import argparse
import glob
import importlib.metadata
import json
import os
import stat
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--personal-root", type=Path, required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    project = a.project.resolve(strict=True)
    base = a.personal_root.resolve(strict=True)
    run = (project / "runs" / a.task).resolve(strict=True)
    if not run.is_relative_to(project):
        raise ValueError("invalid task")
    models, packages, sources, wheels, interpreters, configurations = [], [], [], [], [], []
    count = 0
    skip = {".git", "__pycache__", "node_modules", "profiles", "profiler", "runs", "logs", "assets", "hf-xet", "xet", "opp", "op_impl", "include", "man", "locale"}
    for current, dirs, files in os.walk(base):
        here = Path(current)
        count += 1
        if count > 50000:
            break
        depth = len(here.relative_to(base).parts)
        dirs[:] = [v for v in dirs if v not in skip and (depth < 18)]
        if "config.json" in files:
            try:
                c = json.loads((here / "config.json").read_text())
                if c.get("architectures"):
                    models.append({"path": str(here), "architectures": c.get("architectures"),
                                   "model_type": c.get("model_type"), "hidden_size": c.get("hidden_size"),
                                   "layers": c.get("num_hidden_layers"), "max_position_embeddings": c.get("max_position_embeddings"),
                                   "weights_bytes": sum(v.stat().st_size for v in here.glob("*.safetensors"))})
            except (OSError, ValueError):
                pass
        if here.name in ("site-packages", "dist-packages"):
            selected = {}
            for dist in importlib.metadata.distributions(path=[str(here)]):
                name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
                if name in ("torch", "torch-npu", "vllm", "omni-npu", "vllm-ascend", "transformers", "huggingface-hub"):
                    selected[name] = dist.version
            if selected:
                packages.append({"path": str(here), "packages": selected})
            dirs[:] = [v for v in dirs if v in ("vllm", "omni_npu", "omni_models", "omni_cache", "omniinfer")]
        if here.name in ("vllm", "omni_npu", "omni_models", "omni_cache", "omniinfer") and "__init__.py" in files:
            sources.append(str(here))
        if here.name in ("bin", "python", "python3.11", "python3.10"):
            for filename in files:
                if filename in ("python", "python3", "python3.11", "python3.10"):
                    interpreters.append(str(here / filename))
        for filename in files:
            if filename.endswith(".whl") and filename.startswith(("vllm", "torch", "omni", "transformers", "huggingface")):
                wheels.append(str(here / filename))
            if filename.endswith(".json") and (filename.lower().startswith("7b") or "-7b." in filename.lower() or "runtime" in filename.lower()):
                configurations.append(str(here / filename))
    devices = []
    for filename in glob.glob("/dev/davinci[0-9]*"):
        data = Path(filename).stat()
        devices.append({"path": filename, "major": os.major(data.st_rdev), "minor": os.minor(data.st_rdev)})
    tools = []
    for filename in ("/usr/local/bin/npu-smi", "/usr/local/Ascend/driver/tools/npu-smi", "/usr/bin/npu-smi"):
        path = Path(filename)
        if path.exists():
            info = path.stat()
            tools.append({"path": filename, "realpath": str(path.resolve()), "mode": oct(stat.S_IMODE(info.st_mode)), "executable": os.access(path, os.X_OK)})
    result = {"personal_root": str(base), "directories_scanned": count, "scan_limit_reached": count > 50000,
              "models": models, "packages": packages, "sources": sources, "wheels": wheels,
              "interpreters": interpreters, "configurations": configurations, "mapped_devices": devices,
              "npu_smi_paths": tools, "visibility": {k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_HOME_PATH")},
              "device_initialization": "not_started"}
    (run / "model-runtime.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
