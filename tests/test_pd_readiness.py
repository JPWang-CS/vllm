import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _cpu_stub_modules():
    """Load adapter route code without requiring the server-only FastAPI stack."""
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def get(self, *args, **kwargs):
            return lambda function: function

        def post(self, *args, **kwargs):
            return lambda function: function

    class Request:
        pass

    fastapi.FastAPI = FastAPI
    fastapi.Request = Request
    concurrency = types.ModuleType("fastapi.concurrency")

    async def run_in_threadpool(function, *args, **kwargs):
        return function(*args, **kwargs)

    concurrency.run_in_threadpool = run_in_threadpool
    responses = types.ModuleType("fastapi.responses")

    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.status_code = status_code
            self.body = json.dumps(content).encode()

    class Response:
        def __init__(self, content=b"", status_code=200, media_type=None):
            self.status_code = status_code
            self.body = content
            self.media_type = media_type

    responses.JSONResponse = JSONResponse
    responses.Response = Response
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    return {
        "fastapi": fastapi,
        "fastapi.concurrency": concurrency,
        "fastapi.responses": responses,
        "uvicorn": uvicorn,
    }


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, _cpu_stub_modules(), clear=False):
    SERVE = _load("serve_pd_split_readiness", "models/pangu-92b/adapters/serve_pd_split.py")
    RELAY = _load("pd_relay_readiness", "models/pangu-92b/adapters/pd_relay_proxy.py")


def _models(model):
    return json.dumps({"data": [{"id": model, "object": "model"}]}).encode()


def _model_list(models):
    return json.dumps({"data": [{"id": model, "object": "model"} for model in models]}).encode()


def _response(value):
    if isinstance(value, dict):
        return 200, value
    return value.status_code, json.loads(value.body.decode())


@pytest.mark.parametrize("module,roles", [(SERVE, ("producer", "consumer")),
                                           (RELAY, ("prefill", "decode"))])
def test_pd_readiness_returns_openai_models_only_when_both_match(monkeypatch, module, roles):
    if module is SERVE:
        module.STATE.clear()
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        def fetch(url, timeout=30):
            return 200, _models("pangu-92b")
        monkeypatch.setattr(module, "fetch", fetch)
    else:
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        module.STATE["model"] = "pangu-92b"
        def fetch(url, timeout=30):
            return 200, _models("pangu-92b")
        monkeypatch.setattr(module, "_get", fetch)
    status, body = _response(asyncio.run(module.readiness()))
    assert status == 200
    assert body["data"] == [{"id": "pangu-92b", "object": "model", "owned_by": "vllm"}]


@pytest.mark.parametrize("module,roles", [(SERVE, ("producer", "consumer")),
                                           (RELAY, ("prefill", "decode"))])
def test_pd_readiness_accepts_same_ids_in_reverse_order(monkeypatch, module, roles):
    if module is SERVE:
        module.STATE.clear()
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        def fetch(url, timeout=30):
            role = next(role for role, base in module.STATE["urls"].items() if url.startswith(base))
            ids = ["pangu-92b", "alias"] if role == roles[0] else ["alias", "pangu-92b"]
            return 200, _model_list(ids)
        monkeypatch.setattr(module, "fetch", fetch)
    else:
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        module.STATE["model"] = "pangu-92b"
        def fetch(url, timeout=30):
            role = next(role for role, base in module.STATE["urls"].items() if url.startswith(base))
            ids = ["pangu-92b", "alias"] if role == roles[0] else ["alias", "pangu-92b"]
            return 200, _model_list(ids)
        monkeypatch.setattr(module, "_get", fetch)
    status, body = _response(asyncio.run(module.readiness()))
    assert status == 200
    assert {item["id"] for item in body["data"]} == {"pangu-92b", "alias"}


@pytest.mark.parametrize("module,roles", [(SERVE, ("producer", "consumer")),
                                           (RELAY, ("prefill", "decode"))])
@pytest.mark.parametrize("case", ["not_ready", "missing_data", "mismatched_model"])
def test_pd_readiness_rejects_unready_malformed_or_mismatched_backends(
    monkeypatch, module, roles, case
):
    if module is SERVE:
        module.STATE.clear()
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        def fetch(url, timeout=30):
            role = next(role for role, base in module.STATE["urls"].items() if url.startswith(base))
            if case == "not_ready" and role == roles[1]:
                return 503, b"starting"
            if case == "missing_data" and role == roles[1]:
                return 200, json.dumps({"status": "ready"}).encode()
            model = "other-model" if case == "mismatched_model" and role == roles[1] else "pangu-92b"
            return 200, _models(model)
        monkeypatch.setattr(module, "fetch", fetch)
    else:
        module.STATE["urls"] = {role: "http://" + role for role in roles}
        module.STATE["model"] = "pangu-92b"
        def fetch(url, timeout=30):
            role = next(role for role, base in module.STATE["urls"].items() if url.startswith(base))
            if case == "not_ready" and role == roles[1]:
                return 503, b"starting"
            if case == "missing_data" and role == roles[1]:
                return 200, json.dumps({"status": "ready"}).encode()
            model = "other-model" if case == "mismatched_model" and role == roles[1] else "pangu-92b"
            return 200, _models(model)
        monkeypatch.setattr(module, "_get", fetch)
    status, body = _response(asyncio.run(module.readiness()))
    assert status == 503
    assert body["status"] == "waiting"
