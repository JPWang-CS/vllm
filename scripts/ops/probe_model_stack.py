"""Probe a read-only deployment stack before creating a model experiment."""
import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path


PACKAGE_DATA_SUFFIXES = frozenset({
    ".py", ".jinja", ".jinja2", ".json", ".yaml", ".yml", ".txt",
    ".html", ".css", ".js", ".svg",
})


def _module_name_for_file(path, package_root, package):
    relative = path.relative_to(package_root)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join([package, *parts]) if parts else package


def _relative_import_name(path, package_root, package, node):
    """Resolve a relative ImportFrom using only its source path."""
    current = _module_name_for_file(path, package_root, package).split(".")
    if path.name != "__init__.py":
        current.pop()
    base = current[:len(current) - max(0, node.level - 1)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _module_name_set(package_root, package, files):
    return {
        _module_name_for_file(path, package_root, package)
        for path in files
        if path.suffix.lower() == ".py"
    }


def static_python_package_inventory(source_roots, package="vllm.entrypoints.serve"):
    """Inventory a Python package without importing it or touching a device."""
    relative = Path(*package.split("."))
    package_root = None
    package_kind = None
    for source in source_roots:
        directory = source / relative
        module = source / relative.with_suffix(".py")
        if directory.is_dir():
            package_root, package_kind = directory, "package"
            break
        if module.is_file():
            package_root, package_kind = module, "module"
            break
    if package_root is None:
        return {"status": "MISSING", "package": package, "files": [], "python_files": [],
                "package_data": [], "imports": [], "routing_modules": [], "missing_modules": []}
    files = [package_root] if package_kind == "module" else sorted(
        path for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix.lower() in PACKAGE_DATA_SUFFIXES
    )
    python_files = [path for path in files if path.suffix.lower() == ".py"]
    available_modules = _module_name_set(package_root, package, python_files)
    imports = []
    routing_modules = set()
    internal_imports = set()
    init_path = package_root / "__init__.py" if package_kind == "package" else None
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
                internal_imports.update(alias.name for alias in node.names
                                        if alias.name.startswith(package + "."))
                if path == init_path:
                    routing_modules.update(alias.name for alias in node.names
                                           if alias.name.startswith(package + "."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                resolved = (_relative_import_name(path, package_root, package, node)
                            if node.level else node.module)
                names.add("." * node.level + node.module)
                if resolved.startswith(package + "."):
                    internal_imports.add(resolved)
                if path == init_path and resolved.startswith(package + "."):
                    routing_modules.add(resolved)
            elif isinstance(node, ast.ImportFrom) and node.level and node.module is None:
                base = _relative_import_name(path, package_root, package, node)
                for alias in node.names:
                    resolved = base + "." + alias.name
                    if resolved.startswith(package + "."):
                        internal_imports.add(resolved)
        imports.append({"file": str(path), "imports": sorted(names)})
    package_data = [str(path) for path in files if path.suffix.lower() != ".py"]
    missing_modules = sorted(name for name in internal_imports if name not in available_modules)
    return {"status": "PASS", "package": package, "root": str(package_root),
            "kind": package_kind, "files": [str(path) for path in files],
            "python_files": [str(path) for path in python_files],
            "package_data": package_data, "imports": imports,
            "routing_modules": sorted(routing_modules), "missing_modules": missing_modules,
            "source_complete": not missing_modules}


def routing_import_probe_source(modules):
    """Return a real-module-only probe; it never fabricates FastAPI state."""
    encoded = json.dumps(sorted(set(modules)), ensure_ascii=False)
    return f'''import importlib, json, sys
modules = json.loads({encoded!r})
out = []
failed = False
for name in modules:
    try:
        module = importlib.import_module(name)
        out.append({{"module": name, "status": "PASS", "file": str(getattr(module, "__file__", None))}})
    except Exception as exc:
        failed = True
        out.append({{"module": name, "status": "FAIL", "error": type(exc).__name__, "detail": str(exc)}})
print(json.dumps({{"modules": out, "model_load": "not_started", "fastapi_attach": "not_called"}}, ensure_ascii=False))
sys.exit(1 if failed else 0)
'''


def generic_import_probe_source(modules=("torch", "torch_npu", "vllm", "omni_npu", "transformers", "huggingface_hub")):
    encoded = json.dumps(list(modules), ensure_ascii=False)
    return f'''import importlib, json, sys
modules = json.loads({encoded!r})
out = {{}}
failed = False
for name in modules:
    try:
        module = importlib.import_module(name)
        out[name] = {{"status": "PASS", "file": str(getattr(module, "__file__", None))}}
    except Exception as exc:
        failed = True
        out[name] = {{"status": "FAIL", "error": type(exc).__name__, "detail": str(exc)}}
print(json.dumps(out))
sys.exit(1 if failed else 0)
'''


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
    result = {"disk_free": shutil.disk_usage(project).free, "python": str(a.python), "source_roots": env["PYTHONPATH"],
              "device_computation": "not_started", "source_status": "not_run",
              "routing_status": "not_run", "import_status": "not_run", "device_status": "unknown",
              "stages": []}
    inventory = static_python_package_inventory(source_roots)
    result["stages"].append({"stage": "python_package_inventory", "result": inventory})
    probe_path = run / "stack-probe.json"

    def save():
        probe_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def finish(code):
        result["exit"] = code
        save()
        return code

    result["source_status"] = ("passed" if inventory.get("status") == "PASS"
                                and inventory.get("source_complete") else "failed")
    save()
    if result["source_status"] != "passed":
        # Do not run any real Python import after a missing/incomplete source
        # package.  The inventory and failure are already durable above.
        return finish(2)

    def call(name, argv, timeout=60):
        try:
            completed = subprocess.run(argv, env=env, cwd=project, capture_output=True, text=True, timeout=timeout)
            record = {"stage": name, "exit": completed.returncode, "stdout": completed.stdout[-15000:], "stderr": completed.stderr[-10000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            record = {"stage": name, "error": type(exc).__name__, "detail": str(exc)}
        result["stages"].append(record)
        save()
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    if inventory.get("routing_modules"):
        routing_probe = run / "routing_import_probe.py"
        routing_probe.write_text(routing_import_probe_source(inventory["routing_modules"]), encoding="utf-8")
        routing_record = call("routing_module_imports", [str(a.python), "-B", str(routing_probe)], timeout=120)
        if routing_record.get("exit") != 0:
            result["routing_status"] = "failed"
            return finish(3)
        result["routing_status"] = "passed"
    else:
        result["routing_status"] = "not_available"
    save()

    # Preserve a non-executable container tool. Use only a new private copy.
    original = Path("/usr/local/bin/npu-smi")
    if original.is_file() and original.read_bytes()[:4] == b"\x7fELF":
        private = run / "bin/npu-smi"
        private.parent.mkdir(parents=True)
        shutil.copyfile(original, private)
        private.chmod(0o700)
        device_record = call("device_status", [str(private), "info"], timeout=30)
        result["device_status"] = "observed" if device_record.get("exit") == 0 else "unknown"
    else:
        result["stages"].append({"stage": "device_status", "error": "no_registered_npu_smi_elf"})

    # Use a file to preserve literal newlines and avoid shell/code quoting
    # changes.  The child returns nonzero when any import failed.
    probe = run / "import_probe.py"
    probe.write_text(generic_import_probe_source(), encoding="utf-8")
    import_record = call("imports", [str(a.python), "-B", str(probe)], timeout=120)
    result["import_status"] = "passed" if import_record.get("exit") == 0 else "failed"
    if import_record.get("exit") != 0:
        return finish(4)
    return finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
