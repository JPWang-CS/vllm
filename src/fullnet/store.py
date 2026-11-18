"""Project boundaries and atomic experiment records."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project: {path}")
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        if os.name == 'posix' and os.geteuid() == 0:
            owner = path.parent.stat()
            os.fchown(fd, owner.st_uid, owner.st_gid)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    result = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    identifiers = [row.get("id") for row in result]
    if not result or any(not isinstance(v, str) or not v for v in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("dataset must contain nonempty unique string ids")
    return result


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def experiment(root, model, run):
    return inside(root, Path(root) / "models" / name(model) / "experiments" / name(run))


def state(run, phase, status, **extra):
    path = Path(run) / "state.json"
    data = read_json(path) if path.exists() else {"phases": {}}
    data.update(updated_at=utc(), phase=phase)
    data["phases"][phase] = {"status": status, "at": utc(), **extra}
    write_json(path, data)
    return data


class RunLock:
    """Single writer; a crashed writer leaves evidence for explicit recovery."""
    def __init__(self, run):
        self.path = Path(run) / ".lock"

    def __enter__(self):
        with self.path.open("x", encoding="utf-8") as output:
            output.write(json.dumps({"pid": os.getpid(), "at": utc()}))
        return self

    def __exit__(self, *unused):
        self.path.unlink()
