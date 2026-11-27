"""Generate an editable launch config; device stacks remain independently managed."""
import os
import json
import subprocess
from pathlib import Path

import yaml

from .config import load
from .store import inside, name


def model_packages(python):
    """Read versions from the selected model interpreter without importing torch."""
    script = '''import importlib.metadata as m,json
out={}
for name in ["torch","torch-npu","vllm","vllm-ascend","omni-npu","transformers"]:
 try:out[name]=m.version(name)
 except m.PackageNotFoundError:out[name]=None
print(json.dumps(out))
'''
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([str(python), "-B", "-c", script], capture_output=True,
                            text=True, env=env, timeout=30, check=True)
    return json.loads(result.stdout.strip())


def configure(root, model, platform, python, weights, dataset, metrics, devices, context_limit):
    directory = inside(root, Path(root) / "models" / name(model))
    manifest = load(directory / "model.yaml")
    if platform not in manifest.get("platforms", []):
        raise ValueError("platform is not declared for this model")
    if not devices or len(set(devices)) != len(devices) or any(v < 0 for v in devices):
        raise ValueError("select distinct nonnegative devices")
    if context_limit <= 0:
        raise ValueError("context_limit must be positive")
    target = inside(root, directory / "configs" / (platform + ".local.yaml"))
    target.parent.mkdir(parents=True, exist_ok=True)
    info = model_packages(python)
    model_path = str(Path(weights).resolve())
    port = {"a2": 18320, "a3": 18321, "a5": 18322}[platform]
    kwargs = {"enable_thinking": False} if manifest.get("family") == "qwen3" else {"thinking": False}
    config = {"schema_version": 1, "model": model, "platform": platform, "devices": devices,
              "identity": {"device_model": "unverified", "model_python": str(python), "packages": info,
                           "weights": model_path, "dtype": "bfloat16", "quantization": "none",
                           "vllm_version": info.get("vllm"), "platform_plugin_version": info.get("vllm-ascend"),
                           "cann_path": os.environ.get("ASCEND_HOME_PATH", "not_set")},
              "env": {}, "changes": [],
              "service": {"argv": [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
                                   "--served-model-name", model, "--host", "127.0.0.1", "--port", str(port),
                                   "--dtype", "bfloat16", "--tensor-parallel-size", str(len(devices)),
                                   "--max-model-len", str(context_limit), "--max-num-seqs", "4", "--enforce-eager", "--seed", "0"],
                          "base_url": f"http://127.0.0.1:{port}/v1", "served_model": model,
                          "ready_timeout": 1800, "request_timeout": 180, "shutdown_timeout": 20},
              "evaluation": {"dataset": str(Path(dataset).resolve()), "scope": "RULER niah_single_1 + vt, fixed two-task subset",
                             "metric": "ruler_official", "official_metrics": str(Path(metrics).resolve()),
                             "context_limit": context_limit, "seed": 42, "temperature": 0, "top_p": 1,
                             "chat_template_kwargs": kwargs, "minimum_score": None}}
    if manifest.get("family") == "pangu":
        config["service"]["argv"].append("--trust-remote-code")
        config["evaluation"]["add_special_tokens"] = False
    with target.open("x", encoding="utf-8") as output:
        yaml.safe_dump(config, output, allow_unicode=True, sort_keys=False)
    return {"config": str(target), "status": "generated", "device_execution": "not_run"}
