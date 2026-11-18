"""Use framework commands to obtain Pangu assets inside the current project."""
import argparse
import json
import os
import shutil
import subprocess
import ssl
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--download-timeout", type=int, default=14400)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    run = (root / "runs" / a.task).resolve(strict=True)
    if not run.is_relative_to(root) or shutil.disk_usage(root).free < 40 * 1024**3:
        raise ValueError("invalid run or less than 40 GiB free project storage")
    env = os.environ.copy()
    for key, suffix in {"HOME":"home", "XDG_CACHE_HOME":"cache", "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp",
                        "HF_HOME":"cache/hf", "HF_HUB_CACHE":"cache/hf/hub", "HF_XET_CACHE":"cache/hf/xet"}.items():
        path = run / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1", NO_PROXY="localhost,127.0.0.1,::1")
    steps = []
    # Check the CDN with a small range before starting multi-GB downloads.
    probe_url = "https://huggingface.co/openpangu/openPangu-Embedded-7B-V1.1/resolve/main/model-00004-of-00004.safetensors"
    bundle = env.get("FULLNET_CA_BUNDLE")
    context = ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()
    probe = urllib.request.Request(probe_url, headers={"Range": "bytes=0-15"})
    with urllib.request.urlopen(probe, context=context, timeout=60) as reply:
        sample = reply.read(16)
        (run / "cdn-probe.json").write_text(json.dumps({"status": reply.status, "bytes": len(sample), "certificate_verification": True}))
    for action in ("verify-source", "fetch", "check"):
        argv = [str(a.python), "-B", "-m", "fullnet", "assets", action, "--model", "pangu-7b"]
        with (run / (action + ".log")).open("w") as output:
            result = subprocess.run(argv, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=a.download_timeout if action == "fetch" else 1800)
        steps.append({"action": action, "argv": argv, "exit": result.returncode})
        (run / "asset-steps.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2))
        print(json.dumps(steps[-1], ensure_ascii=False), flush=True)
        if result.returncode:
            raise RuntimeError("asset preparation failed: " + action)
    (run / "assets-ready.json").write_text(json.dumps({"model":"pangu-7b", "source":"openpangu/openPangu-Embedded-7B-V1.1",
        "path": str(root / "assets/weights/pangu-7b"), "steps":steps, "model_inference":"not_run"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
