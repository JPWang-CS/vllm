import asyncio
import importlib.util
import json
import os
import shutil
import signal
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "models/pangu-92b/adapters"
sys.path.insert(0, str(ADAPTERS))


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
        def __init__(self, content=b"", status_code=200, media_type=None, headers=None):
            self.status_code = status_code
            self.body = content
            self.media_type = media_type
            self.headers = dict(headers or {})

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
MULTI = _load("multinode_adapter_readiness", "models/pangu-92b/adapters/multinode_adapter.py")


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


def test_serve_readiness_route_uses_custom_served_model_for_both_roles(monkeypatch):
    SERVE.STATE.clear()
    SERVE.STATE["served_model_name"] = "pangu"
    SERVE.STATE["urls"] = {"producer": "http://producer", "consumer": "http://consumer"}

    def fetch(url, timeout=30):
        assert url.endswith("/v1/models")
        return 200, _models("pangu")

    monkeypatch.setattr(SERVE, "fetch", fetch)
    status, body = _response(asyncio.run(SERVE.readiness()))
    assert status == 200
    assert body["data"] == [{"id": "pangu", "object": "model", "owned_by": "vllm"}]


def test_child_exit_codes_ignore_state_metadata():
    class Child:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    state = {"producer": Child(17), "consumer": Child(None), "proxy": Child(None),
             "served_model_name": "pangu", "urls": {"producer": "http://p"}}
    assert SERVE.child_exit_codes(state) == {"producer": 17, "consumer": None, "proxy": None}

    state["producer"] = Child(None)
    assert SERVE.all_children_alive(state) is True
    state["consumer"] = Child(0)
    assert SERVE.all_children_alive(state) is False


def _profile_request(path):
    async def body():
        return b'{"profile":true}'
    return SimpleNamespace(url=SimpleNamespace(path=path), body=body,
                           headers={"Content-Type": "application/json"})


def test_profile_control_posts_to_both_roles(monkeypatch):
    SERVE.STATE["urls"] = {"producer": "http://producer", "consumer": "http://consumer"}
    calls = []

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def read(self):
            return b'{"ok":true}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data, timeout))
        return Response()

    monkeypatch.setattr(SERVE.urllib.request, "urlopen", urlopen)
    result = _response(asyncio.run(SERVE.profile_control(_profile_request("/start_profile"))))
    assert result[0] == 200
    assert {call[0] for call in calls} == {"http://producer/start_profile", "http://consumer/start_profile"}
    assert {call[1] for call in calls} == {"POST"}
    assert {call[2] for call in calls} == {b'{"profile":true}'}


def test_profile_control_one_role_failure_is_not_200(monkeypatch):
    SERVE.STATE["urls"] = {"producer": "http://producer", "consumer": "http://consumer"}
    calls = []

    class Response:
        def __init__(self, status):
            self.status = status
            self.headers = {"Content-Type": "text/plain"}

        def read(self):
            return b"failed" if self.status != 200 else b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        calls.append((request.full_url, request.get_method()))
        return Response(503 if request.full_url.startswith("http://consumer/") else 200)

    monkeypatch.setattr(SERVE.urllib.request, "urlopen", urlopen)
    result = _response(asyncio.run(SERVE.profile_control(_profile_request("/stop_profile"))))
    assert result[0] == 500
    assert {call[0] for call in calls} == {"http://producer/stop_profile", "http://consumer/stop_profile"}
    assert {call[1] for call in calls} == {"POST"}
    assert result[1]["consumer"]["http_status"] == 503


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


def test_role_argv_preserves_connector_keys_and_parameterizes_model_bind_and_tp():
    argv = SERVE.role_argv(
        "python3", "serve.py", "/models/7b", "prefill", 18333, 0.7,
        "3,6", 7100, served_model_name="pangu-7b", bind_address="0.0.0.0",
        tensor_parallel_size=2, max_model_len=16384,
        kv_cache_dtype="auto", block_size=128, expert_parallel=False)
    assert argv[argv.index("--served-model-name") + 1] == "pangu-7b"
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "auto"
    assert "--enable-expert-parallel" not in argv
    config = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    assert config == {"kv_connector": "LLMDataDistConnector",
                      "kv_role": "kv_producer", "kv_port": 7100}


def test_supervisor_urls_follow_advertised_role_endpoints():
    urls = SERVE.service_urls(
        "10.0.0.3", "10.0.0.4", "0.0.0.0",
        {"producer": 18333, "consumer": 18334, "proxy": 18335},
        proxy_advertise="10.0.0.9")
    assert urls["producer"] == "http://10.0.0.3:18333"
    assert urls["consumer"] == "http://10.0.0.4:18334"
    assert urls["proxy"] == "http://10.0.0.9:18335"


def test_cross_host_pd_plan_and_role_launches_are_explicit_and_not_verified():
    plan = MULTI.pd_plan(
        ["prefill-host", "decode-host"], [2, 2], tp=2,
        served_model="pangu-7b", device_ids=[[3, 6], [1, 4]],
        bind_addresses=["0.0.0.0", "0.0.0.0"],
        advertise_addresses=["10.0.0.3", "10.0.0.4"],
    )
    assert plan["endpoint_verified"] is False
    assert plan["endpoints"]["prefill_http"] == "http://10.0.0.3:18333"
    assert plan["producer"]["device_ids"] == [3, 6]
    assert plan["consumer"]["device_ids"] == [1, 4]

    sys.modules["serve_pd_split"] = SERVE
    launches = MULTI.pd_role_launches(
        plan, "python3", "serve.py", "pd_relay_proxy.py", "/models/7b")
    assert launches["roles"]["prefill"]["host"] == "prefill-host"
    assert launches["roles"]["decode"]["host"] == "decode-host"
    assert launches["roles"]["prefill"]["argv"][
        launches["roles"]["prefill"]["argv"].index("--host") + 1] == "0.0.0.0"
    assert launches["roles"]["prefill"]["kv_config"]["kv_role"] == "kv_producer"
    assert launches["relay"]["argv"][-1] == "18335"
    assert launches["endpoint_verified"] is False


def test_ray_runtime_path_uses_project_scope_and_byte_limit(tmp_path):
    selected = MULTI.short_runtime_path(ROOT)
    assert selected["path"].startswith(str(ROOT.resolve()))
    assert selected["bytes"] <= 107
    assert selected["max_path_bytes"] <= 107
    assert selected["worker_temp_dir"] == "inherited_from_head_internal_kv"
    with pytest.raises(ValueError, match="107"):
        MULTI.validate_runtime_path(tmp_path / ("x" * 200), tmp_path)


def test_pipeline_parallelism_is_explicitly_not_claimed():
    with pytest.raises(ValueError, match="pipeline parallelism"):
        MULTI.plan(["a", "b"], [2, 2], {"tp": 2, "dp": 1, "pp": 2})


def test_ray_node_entry_uses_project_local_short_sockets_and_scoped_cleanup():
    paths = MULTI.ray_project_paths(ROOT, "r7")
    assert Path(paths["paths"]["plasma_store_socket"]).parts[-3:] == (".r", "r7", "p")
    assert Path(paths["paths"]["raylet_socket"]).parts[-3:] == (".r", "r7", "r")
    script = str((ROOT / "models/pangu-92b/adapters/multinode_adapter.py").resolve())
    head = MULTI.ray_node_argv(
        "python3", script, ROOT, "r7", "head", "10.0.0.3", ray_port=29600)
    worker = MULTI.ray_node_argv(
        "python3", script, ROOT, "r7", "worker", "10.0.0.4",
        ray_port=29600, head_address="10.0.0.3:29600")
    assert head[head.index("ray-node") + 1:head.index("ray-node") + 3] == [
        "--role", "head"]
    assert worker[worker.index("--head-address") + 1] == "10.0.0.3:29600"
    source = (ROOT / "models/pangu-92b/adapters/multinode_adapter.py").read_text()
    assert "plasma_store_socket_name=paths" in source
    assert "raylet_socket_name=paths" in source
    assert "RAY_NODE_MAJOR_MINOR = (2, 48)" in source
    assert "kill_all_processes(check_alive=False, allow_graceful=True)" in source


def test_cluster_scripts_can_emit_python_node_commands(tmp_path):
    plan = MULTI.plan(["head", "worker"], [2, 2], {"tp": 4, "dp": 1},
                       advertise_address="10.0.0.3", bind_address="10.0.0.4")
    script = str((ROOT / "models/pangu-92b/adapters/multinode_adapter.py").resolve())
    commands = MULTI.cluster_scripts(
        plan, project_root=ROOT, task_id="r7", adapter_script=script)
    assert "ray-node" in commands["head"]
    assert "--role head" in commands["head"]
    assert "--role worker" in commands["worker"]
    assert "ray start" not in commands["head"]


def _install_fake_ray(monkeypatch, node_cls):
    ray = types.ModuleType("ray")
    ray.__version__ = "2.48.0"
    private = types.ModuleType("ray._private")
    node_module = types.ModuleType("ray._private.node")
    parameter_module = types.ModuleType("ray._private.parameter")
    node_module.Node = node_cls

    class RayParams:
        last = None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            type(self).last = self

    parameter_module.RayParams = RayParams
    private.node = node_module
    private.parameter = parameter_module
    ray._private = private
    for name, module in {
        "ray": ray,
        "ray._private": private,
        "ray._private.node": node_module,
        "ray._private.parameter": parameter_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return RayParams


@pytest.fixture
def project_runtime_dir():
    """Use a short, project-owned directory for real 107-byte path checks."""
    root = ROOT.resolve(strict=True)
    candidate = (root / ".runtime" /
                 f"ray-test-{os.getpid()}-{uuid.uuid4().hex[:8]}" ).resolve()
    candidate.relative_to(root)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir()
    try:
        yield candidate
    finally:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
        if resolved.is_symlink():
            raise AssertionError("test runtime path unexpectedly became a symlink")
        if resolved.exists():
            shutil.rmtree(resolved)


def test_ray_node_main_fake_lifecycle_uses_own_node_and_records_term(project_runtime_dir, monkeypatch):
    callbacks = {}

    class FakeEvent:
        def __init__(self):
            self.called = False

        def wait(self, _timeout):
            if not self.called:
                self.called = True
                callbacks[signal.SIGTERM](signal.SIGTERM, None)
            return True

        def set(self):
            self.called = True

    class FakeNode:
        instance = None

        def __init__(self, params, head, shutdown_at_exit, spawn_reaper):
            self.params = params
            self.head = head
            self.address = "10.0.0.3:29600"
            self.all_processes = {"fake_child": []}
            self.cleaned = False
            type(self).instance = self

        def kill_all_processes(self, check_alive, allow_graceful):
            self.cleaned = check_alive is False and allow_graceful is True

    ray_params = _install_fake_ray(monkeypatch, FakeNode)
    monkeypatch.setattr(MULTI.threading, "Event", FakeEvent)
    monkeypatch.setattr(MULTI.signal, "signal",
                        lambda sig, handler: callbacks.setdefault(sig, handler))
    script = str((ROOT / "models/pangu-92b/adapters/multinode_adapter.py").resolve())
    result = MULTI.ray_node_main([
        "--role", "worker", "--project-root", str(project_runtime_dir), "--task-id", "t1",
        "--node-ip", "10.0.0.4", "--ray-port", "29600",
        "--head-address", "10.0.0.3:29600", "--num-gpus", "0",
        "--ray-resources-json", "{}",
    ])
    assert result == 0
    assert FakeNode.instance.cleaned is True
    assert ray_params.last.gcs_address == "10.0.0.3:29600"
    assert Path(ray_params.last.plasma_store_socket_name).parts[-3:] == (".r", "t1", "p")
    assert Path(ray_params.last.raylet_socket_name).parts[-3:] == (".r", "t1", "r")
    status_path = project_runtime_dir / ".r" / "t1" / "node-status.json"
    exit_path = project_runtime_dir / ".r" / "t1" / "node-exit.json"
    assert json.loads(status_path.read_text())["status"] == "stopped"
    assert json.loads(exit_path.read_text())["status"] == "stopped"


def test_ray_node_main_constructor_failure_is_failed_not_stopped(project_runtime_dir, monkeypatch):
    class FailingNode:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("constructor boom")

    _install_fake_ray(monkeypatch, FailingNode)
    monkeypatch.setattr(MULTI.signal, "signal", lambda *_args: None)
    with pytest.raises(RuntimeError, match="constructor boom"):
        MULTI.ray_node_main([
            "--role", "head", "--project-root", str(project_runtime_dir), "--task-id", "t2",
            "--node-ip", "10.0.0.3", "--ray-port", "29600", "--num-gpus", "0",
            "--ray-resources-json", "{}",
        ])
    status = json.loads((project_runtime_dir / ".r" / "t2" / "node-status.json").read_text())
    exit_record = json.loads((project_runtime_dir / ".r" / "t2" / "node-exit.json").read_text())
    assert status["status"] == "failed"
    assert status["failure"]["message"] == "constructor boom"
    assert exit_record["status"] == "failed"


def test_relay_accepts_only_runtime_returned_kv_params():
    params = {"prefill_req_id": "req-1", "remote_cluster_id": 7}
    payload = json.dumps({"kv_transfer_params": params}).encode()
    assert RELAY.extract_runtime_kv_transfer_params(payload) == params
    assert RELAY.extract_runtime_kv_transfer_params(
        json.dumps({"kv_transfer_params": {}}).encode()) is None
    assert RELAY.extract_runtime_kv_transfer_params(
        json.dumps({"text": "independent generation"}).encode()) is None
