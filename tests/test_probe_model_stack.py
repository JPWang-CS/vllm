import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_model_stack_inventory", ROOT / "scripts/ops/probe_model_stack.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
static_python_package_inventory = MODULE.static_python_package_inventory
routing_import_probe_source = MODULE.routing_import_probe_source
generic_import_probe_source = MODULE.generic_import_probe_source
probe_main = MODULE.main


def test_static_serve_package_inventory_finds_cache_python_package_without_import(tmp_path):
    root = tmp_path / "runtime" / "vllm" / "entrypoints" / "serve"
    (root / "cache").mkdir(parents=True)
    (root / "__init__.py").write_text("from .router import router\n", encoding="utf-8")
    (root / "router.py").write_text("from .cache import registry\n", encoding="utf-8")
    (root / "cache" / "__init__.py").write_text("registry = {}\n", encoding="utf-8")
    (root / "jinja").mkdir()
    (root / "jinja" / "index.jinja").write_text("{{ value }}\n", encoding="utf-8")
    (root / "static").mkdir()
    (root / "static" / "app.js").write_text("const app = {};\n", encoding="utf-8")
    (root / "pkgdata").mkdir()
    (root / "pkgdata" / "routes.json").write_text("{}\n", encoding="utf-8")
    result = static_python_package_inventory([tmp_path / "runtime"])
    assert result["status"] == "PASS"
    assert result["source_complete"] is True
    assert any(path.replace("\\", "/").endswith("serve/cache/__init__.py") for path in result["files"])
    assert any(path.replace("\\", "/").endswith("serve/jinja/index.jinja") for path in result["package_data"])
    assert any(path.replace("\\", "/").endswith("serve/static/app.js") for path in result["package_data"])
    assert any(path.replace("\\", "/").endswith("serve/pkgdata/routes.json") for path in result["package_data"])
    assert any(".cache" in item["imports"] for item in result["imports"])
    assert result["routing_modules"] == ["vllm.entrypoints.serve.router"]


def test_static_inventory_reports_missing_internal_module_without_importing(tmp_path):
    root = tmp_path / "runtime" / "vllm" / "entrypoints" / "serve"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("from .missing_router import router\n", encoding="utf-8")
    result = static_python_package_inventory([tmp_path / "runtime"])
    assert result["status"] == "PASS"
    assert result["source_complete"] is False
    assert result["missing_modules"] == ["vllm.entrypoints.serve.missing_router"]


def test_router_probe_uses_real_imports_and_never_attaches_partial_fastapi_app():
    source = routing_import_probe_source(["vllm.entrypoints.serve.router"])
    assert "importlib.import_module" in source
    assert "FastAPI" not in source
    assert "register_vllm_serve_api_routers" not in source
    assert '"fastapi_attach": "not_called"' in source


def test_generic_import_probe_child_exits_nonzero_on_any_failed_import(tmp_path):
    probe = tmp_path / "import_probe.py"
    probe.write_text(generic_import_probe_source(("json", "module_that_does_not_exist")), encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(probe)], capture_output=True, text=True)
    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["json"]["status"] == "PASS"
    assert output["module_that_does_not_exist"]["status"] == "FAIL"


def _main_args(monkeypatch, project, runtime, task):
    monkeypatch.setattr("sys.argv", [
        "probe_model_stack.py", "--project", str(project), "--runtime-root", str(runtime),
        "--python", "python", "--task", task,
    ])


def _serve_root(runtime, init_source):
    root = runtime / "source/image/opt/vllm/vllm/entrypoints/serve"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(init_source, encoding="utf-8")
    return root


def test_main_stops_before_real_import_when_source_is_missing(tmp_path, monkeypatch):
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    task = "probe-missing"
    (project / "runs" / task).mkdir(parents=True)
    runtime.mkdir()
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: calls.append(args) or None)
    _main_args(monkeypatch, project, runtime, task)
    assert probe_main() == 2
    result = json.loads((project / "runs" / task / "stack-probe.json").read_text(encoding="utf-8"))
    assert result["source_status"] == "failed"
    assert result["exit"] == 2
    assert result["import_status"] == "not_run"
    assert calls == []


def test_main_stops_before_real_import_when_source_is_incomplete(tmp_path, monkeypatch):
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    task = "probe-incomplete"
    (project / "runs" / task).mkdir(parents=True)
    _serve_root(runtime, "from .missing_router import router\n")
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: calls.append(args) or None)
    _main_args(monkeypatch, project, runtime, task)
    assert probe_main() == 2
    result = json.loads((project / "runs" / task / "stack-probe.json").read_text(encoding="utf-8"))
    assert result["source_status"] == "failed"
    assert result["stages"][0]["result"]["source_complete"] is False
    assert result["import_status"] == "not_run"
    assert calls == []


def test_main_propagates_router_child_failure_before_generic_import(tmp_path, monkeypatch):
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    task = "probe-router-fail"
    (project / "runs" / task).mkdir(parents=True)
    _serve_root(runtime, "from .router import router\n")
    (runtime / "source/image/opt/vllm/vllm/entrypoints/serve" / "router.py").write_text(
        "router = object()\n", encoding="utf-8"
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return type("Completed", (), {"returncode": 1, "stdout": "router-fail", "stderr": ""})()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    _main_args(monkeypatch, project, runtime, task)
    assert probe_main() == 3
    result = json.loads((project / "runs" / task / "stack-probe.json").read_text(encoding="utf-8"))
    assert result["routing_status"] == "failed"
    assert result["import_status"] == "not_run"
    assert len(calls) == 1
    assert calls[0][-1].endswith("routing_import_probe.py")


def test_main_propagates_generic_import_failure_and_keeps_device_unknown_separate(tmp_path, monkeypatch):
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    task = "probe-import-fail"
    (project / "runs" / task).mkdir(parents=True)
    _serve_root(runtime, "# no explicit routers\n")

    def run(argv, **kwargs):
        assert argv[-1].endswith("import_probe.py")
        return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "torch_npu failed"})()

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    _main_args(monkeypatch, project, runtime, task)
    assert probe_main() == 4
    result = json.loads((project / "runs" / task / "stack-probe.json").read_text(encoding="utf-8"))
    assert result["routing_status"] == "not_available"
    assert result["import_status"] == "failed"
    assert result["device_status"] == "unknown"
    assert result["exit"] == 4
