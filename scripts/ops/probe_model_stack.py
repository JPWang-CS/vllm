"""Probe a read-only deployment stack before creating a model experiment."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    project = a.project.resolve(strict=True)
    run = (project / "runs" / a.task).resolve(strict=True)
    if not run.is_relative_to(project):
        raise ValueError("invalid run")
    runtime = a.runtime_root.resolve(strict=True)
    env = os.environ.copy()
    for key, suffix in {"HOME":"home", "XDG_CACHE_HOME":"cache", "HF_HOME":"cache/hf", "HF_HUB_CACHE":"cache/hf/hub",
                        "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp", "ASCEND_PROCESS_LOG_PATH":"logs/ascend",
                        "ASCEND_CACHE_PATH":"cache/ascend"}.items():
        path = run / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    source_roots = [runtime / "source/image/opt/vllm",
                    runtime / "source/image/workspace/omniinfer/components/omni-npu/src",
                    runtime / "source/image/workspace/omniinfer/components/omni-models",
                    runtime / "source/image/workspace/omniinfer/components/omni-cache",
                    runtime / "source/image/workspace/omniinfer/components/omni-eplb",
                    runtime / "source/image/workspace/omniinfer", project / "src"]
    env["PYTHONPATH"] = os.pathsep.join(str(v) for v in source_roots if v.is_dir())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    result = {"disk_free": shutil.disk_usage(project).free, "python": str(a.python), "source_roots": env["PYTHONPATH"],
              "device_computation": "not_started", "stages": []}
    def call(name, argv, timeout=60):
        try:
            completed = subprocess.run(argv, env=env, cwd=project, capture_output=True, text=True, timeout=timeout)
            record = {"stage": name, "exit": completed.returncode, "stdout": completed.stdout[-15000:], "stderr": completed.stderr[-10000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            record = {"stage": name, "error": type(exc).__name__, "detail": str(exc)}
        result["stages"].append(record)
        (run / "stack-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record
    # Preserve a non-executable container tool. Use only a new private copy.
    original = Path("/usr/local/bin/npu-smi")
    if original.is_file() and original.read_bytes()[:4] == b"\x7fELF":
        private = run / "bin/npu-smi"
        private.parent.mkdir(parents=True)
        shutil.copyfile(original, private)
        private.chmod(0o700)
        call("device_status", [str(private), "info"], timeout=30)
    else:
        result["stages"].append({"stage": "device_status", "error": "no_registered_npu_smi_elf"})
    # Use a file to preserve literal newlines and avoid shell/code quoting changes.
    probe = run / "import_probe.py"
    probe.write_text('import json,importlib\nout={}\nfor name in ["torch","torch_npu","vllm","omni_npu","transformers","huggingface_hub"]:\n try:\n  module=importlib.import_module(name); out[name]={"file":str(getattr(module,"__file__",None))}\n except Exception as exc:\n  out[name]={"error":type(exc).__name__,"detail":str(exc)}\nprint(json.dumps(out))\n')
    call("imports", [str(a.python), "-B", str(probe)], timeout=120)
    (run / "stack-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
