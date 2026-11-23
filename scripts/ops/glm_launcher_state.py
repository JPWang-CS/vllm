"""Strict parsing helpers for the registered GLM controller state file."""
from __future__ import annotations

import json
import shlex
from typing import Any


_TERMINAL_FIELDS = {
    "exit": int,
    "model": str,
    "runtime": str,
    "fifo_removed": bool,
    "child_alive": bool,
    "credentials_persisted": bool,
}


def parse_terminal_state(value: Any) -> dict[str, Any] | None:
    """Return a valid completed controller record; ignore nonterminal state.

    The controller publishes STATE.json only after its child exits. Other writers
    may use the same filename for progress or restart notes, so file existence
    alone is never a completion signal.
    """
    if not isinstance(value, dict):
        return None
    for name, expected in _TERMINAL_FIELDS.items():
        item = value.get(name)
        if expected is int:
            if type(item) is not int:
                return None
        elif not isinstance(item, expected):
            return None
    if not value["model"] or not value["runtime"]:
        return None
    if (value["fifo_removed"] is not True or value["child_alive"] is not False or
            value["credentials_persisted"] is not False):
        return None
    return value


def decode_terminal_state(payload: bytes | str) -> dict[str, Any] | None:
    """Decode JSON bytes/text then apply the shared terminal schema."""
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return parse_terminal_state(value)


def should_stop_proxy(*, terminal_state: dict[str, Any] | None, controller_submitted: bool) -> bool:
    """Keep the tunnel for any submitted controller until its real terminal record."""
    return terminal_state is not None or not controller_submitted


def canonical_repair_bash_rule(project: str, supplied: str) -> str:
    """Return the one reviewed Bash prefix accepted for framework repair tasks."""
    entry = f"python3 -B {project}/scripts/ops/recover_framework.py"
    if supplied not in (entry, entry + ":*"):
        raise ValueError("repair tasks must use the recover_framework.py project entrypoint")
    return entry + ":*"


def bash_matches_argv(command: str, expected: list[str]) -> bool:
    """Match equivalent shell quoting while rejecting extra/chained commands."""
    try:
        return shlex.split(command, posix=True) == expected
    except ValueError:
        return False
