"""Readable, model-scoped configuration and change selection."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

from .store import inside, name

SUPPORTED_NPU_PLATFORMS = ("a2", "a3", "a5")
SUPPORTED_PLATFORMS = SUPPORTED_NPU_PLATFORMS + ("cpu-test",)


def load(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    return data


def resolve(path, root):
    root = Path(root).resolve()
    c = copy.deepcopy(load(path))
    if c.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    model = name(c.get("model"))
    manifest_path = inside(root, root / "models" / model / "model.yaml")
    if not manifest_path.is_file():
        raise ValueError("model.yaml is required")
    if c.get("platform") not in SUPPORTED_PLATFORMS:
        raise ValueError("platform must be a2, a3 or a5 (cpu-test is for offline workflow tests)")
    manifest = load(manifest_path)
    if c["platform"] not in manifest.get("platforms", []):
        raise ValueError(f"platform {c['platform']} is not declared for model {model}")
    service = c.get("service", {})
    argv = service.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
        raise ValueError("service.argv must be a nonempty argument list")
    if c["platform"] != "cpu-test" and not c.get("devices"):
        raise ValueError("explicit devices are required")
    if c["platform"] != "cpu-test" and not c.get("identity"):
        raise ValueError("device, weights and runtime identity are required")
    for key in ("ready_timeout", "request_timeout", "shutdown_timeout"):
        if float(service.get(key, 30)) <= 0:
            raise ValueError(f"{key} must be positive")
    evaluation = c.get("evaluation", {})
    if evaluation.get("metric") not in ("exact_match", "ruler_official"):
        raise ValueError("evaluation.metric must be exact_match or ruler_official")
    if int(evaluation.get("context_limit", 0)) <= 0:
        raise ValueError("evaluation.context_limit must be positive")
    if not isinstance(evaluation.get("chat_template_kwargs", {}), dict):
        raise ValueError("chat_template_kwargs must be a mapping")
    # Secrets are injected by name at runtime, never persisted in resolved configuration.
    for key, value in c.get("env", {}).items():
        if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", key, re.I):
            raise ValueError(f"use a runtime credential environment reference, not env.{key}")
        if not isinstance(value, str):
            raise ValueError(f"environment value must be a string: {key}")
    changes, enabled = [], c.get("changes", [])
    if not isinstance(enabled, list) or len(set(enabled)) != len(enabled):
        raise ValueError("changes must be a list of distinct identifiers")
    for change in enabled:
        directory = inside(root, root / "models" / model / "changes" / name(change))
        spec = load(directory / "change.yaml")
        if c["platform"] not in spec.get("platforms", []):
            raise ValueError(f"change {change} does not support {c['platform']}")
        if set(spec.get("conflicts", [])) & set(enabled):
            raise ValueError(f"conflicting changes: {change}")
        if not set(spec.get("requires", [])).issubset(enabled):
            raise ValueError(f"unsatisfied change dependency: {change}")
        if not spec.get("verify_argv"):
            raise ValueError(f"change {change} needs verify_argv to prove activation")
        changes.append({"name": change, "spec": spec})
    c["resolved_changes"] = changes
    return c


def expand(value, root, run):
    value = str(value).replace("${PROJECT}", str(root)).replace("${RUN}", str(run))
    value = os.path.expandvars(value)
    if re.search(r"\$\{[^}]+\}", value) or "<fill-" in value:
        raise ValueError(f"unresolved configuration value: {value}")
    return value


def runtime(c, root, run):
    def visit(value):
        if isinstance(value, str):
            return expand(value, root, run)
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        return value
    c = visit(copy.deepcopy(c))
    for change in c["resolved_changes"]:
        spec = change["spec"]
        for key, value in spec.get("env", {}).items():
            if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", key, re.I):
                raise ValueError("change env cannot contain credentials")
            c["env"][key] = expand(value, root, run)
        spec["verify_argv"] = [expand(v, root, run) for v in spec["verify_argv"]]
    evaluation = c["evaluation"]
    evaluation["dataset"] = expand(evaluation["dataset"], root, run)
    if evaluation.get("official_metrics"):
        evaluation["official_metrics"] = expand(evaluation["official_metrics"], root, run)
    return c
