"""Downloadable model sources with project-private cache and explicit credentials."""
from __future__ import annotations

import importlib.metadata
import os
import json
import urllib.parse
import urllib.request
import ssl
import struct
from pathlib import Path

from .config import load
from .store import inside, name, utc, write_json


def check_safetensors(path):
    """Validate structure and stored byte ranges without loading tensor payloads."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("incomplete safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 64 * 1024**2 or length + 8 > size:
            raise ValueError("invalid safetensors header; possible truncated file or LFS pointer")
        header = json.loads(stream.read(length))
    spans = []
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        start, end = tensor["data_offsets"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= size - 8 - length:
            raise ValueError("safetensors tensor range exceeds file")
        spans.append((start, end))
    if not spans:
        raise ValueError("safetensors contains no tensors")
    spans.sort()
    if spans[0][0] != 0 or spans[-1][1] != size - 8 - length or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("safetensors byte ranges are incomplete or overlap")
    return len(spans)


def check_local_assets(path):
    """Read-only validation of an already selected model directory.

    This is intentionally independent of the download manifest and never
    creates a cache, writes a receipt beside the weights, or fetches anything.
    """
    target = Path(path).resolve(strict=True)
    if not target.is_dir():
        raise ValueError("frozen identity.weights must resolve to a directory")
    config = target / "config.json"
    if not config.is_file():
        raise ValueError("model assets are missing config.json")
    from .store import read_json
    metadata = read_json(config)
    files = sorted(target.glob("*.safetensors")) + sorted(target.glob("*.bin"))
    if not files:
        raise ValueError("model assets contain no supported weight files")
    for item in files:
        if item.suffix == ".safetensors":
            check_safetensors(item)
        elif item.stat().st_size == 0:
            raise ValueError(f"empty weight file: {item.name}")
    for index in target.glob("*.index.json"):
        shards = set(read_json(index).get("weight_map", {}).values())
        if not shards or any(not inside(target, target / shard).is_file() for shard in shards):
            raise ValueError(f"missing weight shard referenced by {index.name}")
    tokenizer = next((target / name for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
                      if (target / name).is_file()), None)
    if tokenizer is None:
        raise ValueError("model assets are missing tokenizer files")
    return {"status": "files_present", "target": str(target),
            "config_architectures": metadata.get("architectures", []),
            "weight_files": len(files),
            "weight_bytes": sum(item.stat().st_size for item in files),
            "tokenizer": str(tokenizer), "device_execution": "not_run"}


def model_assets(root, model):
    manifest = load(inside(root, Path(root) / "models" / name(model) / "model.yaml"))
    source = manifest.get("download", {})
    if source.get("provider") != "huggingface" or not source.get("repo_id"):
        raise ValueError("model needs a declared download provider/repo_id; supply the authorized Pangu source in model.yaml")
    target = inside(root, Path(root) / "assets/weights" / model)
    return source, target


def assets(root, model, action):
    source, target = model_assets(root, model)
    cache = inside(root, Path(root) / "cache/huggingface")
    result = {"model": model, "provider": source["provider"], "repo_id": source["repo_id"],
              "revision": source.get("revision", "main"), "target": str(target), "cache": str(cache)}
    if action == "plan":
        return result
    if action == "verify-source":
        url = "https://huggingface.co/api/models/" + urllib.parse.quote(source["repo_id"], safe="/") + "/revision/" + urllib.parse.quote(result["revision"], safe="")
        token = os.environ.get(source.get("token_env", "HF_TOKEN"))
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token} if token else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            metadata = json.load(response)
        names = [v["rfilename"] for v in metadata.get("siblings", [])]
        weight_files = [v for v in names if v.endswith((".safetensors", ".bin"))]
        if "config.json" not in names or not weight_files:
            raise ValueError("source lacks config or supported weight files")
        result.update(status="source_accessible", weight_files=weight_files,
                      tokenizer_present=any(v in names for v in ("tokenizer.json", "tokenizer.model", "vocab.json")),
                      weights_downloaded=False)
        return result
    if action == "fetch":
        # requests uses certifi while urllib uses the system trust store. Respect
        # explicit policy and reuse a trusted system bundle for managed networks.
        bundle = os.environ.get("FULLNET_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or ssl.get_default_verify_paths().cafile
        if bundle and not os.environ.get("REQUESTS_CA_BUNDLE"):
            if not Path(bundle).is_file():
                raise ValueError("configured CA bundle is missing")
            os.environ["REQUESTS_CA_BUNDLE"] = bundle
            os.environ["SSL_CERT_FILE"] = bundle
        for key, relative in {"HF_HOME":"", "HF_HUB_CACHE":"hub", "HF_XET_CACHE":"xet"}.items():
            path = cache / relative
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import snapshot_download
        token_name = source.get("token_env", "HF_TOKEN")
        token = os.environ.get(token_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        # Hub cache location is read at module import; explicitly pass cache_dir/local_dir.
        snapshot_download(repo_id=source["repo_id"], revision=result["revision"],
                          local_dir=target, cache_dir=cache, token=token or False,
                          allow_patterns=source.get("allow_patterns"), ignore_patterns=source.get("ignore_patterns"))
    config = target / "config.json"
    files = list(target.glob("*.safetensors")) + list(target.glob("*.bin"))
    if not config.is_file() or not files:
        raise ValueError("model download is incomplete: config.json and weight files are required")
    from .store import read_json
    for index in target.glob("*.index.json"):
        shards = set(read_json(index).get("weight_map", {}).values())
        if not shards or any(not inside(target, target / shard).is_file() for shard in shards):
            raise ValueError("missing weight shard")
    for path in files:
        if path.suffix == ".safetensors":
            check_safetensors(path)
        elif path.stat().st_size == 0:
            raise ValueError("empty weight file")
    if not any((target / filename).is_file() for filename in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("tokenizer files missing")
    result.update(status="files_present", weight_files=len(files), at=utc(), device_execution="not_run")
    write_json(inside(root, Path(root) / "assets" / (model + "-download.json")), result)
    return result


def doctor():
    packages = {}
    for pkg in ("fullnet-lab", "PyYAML", "torch", "torch-npu", "vllm", "vllm-ascend", "transformers", "huggingface-hub"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    return {"packages": packages, "device_initialization": "not_performed",
            "environment": {key: os.environ.get(key) for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH")}}
