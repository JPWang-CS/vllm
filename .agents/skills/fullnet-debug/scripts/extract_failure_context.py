#!/usr/bin/env python3
"""Stream bounded failure context from task and service/PD logs.

The helper is deliberately evidence-oriented: it records the first observed
traceback/error and nearby lines, while retaining a separate tail summary.
The first observed error is not declared to be the root cause.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
import sys


SENSITIVE_PARTS = {
    ".netrc", ".claude", ".codex", "credential", "credentials", "password",
    "passwd", "secret", "secrets", "token", "tokens", "private", "id_rsa",
}
REDACTIONS = (
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(password\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(token\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"), r"\1<redacted>"),
)
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
DEVICE_ERROR_RE = re.compile(
    r"(?i)\b(?:pagefault|aicpu|507018|11003|get kernel failed|userdevid|"
    r"device\s+(?:error|fault|exception)|kernel\s+(?:launch|lookup)\s+failed)\b"
)
ACTUAL_ERROR_RE = re.compile(
    r"(?i)(?:\b(?:error|fatal|failed|failure)\b|"
    r"^[\w.]+(?:Error|Exception|Fault)(?::|\s))"
)
WARNING_RE = re.compile(r"(?i)\b(?:warn|warning)\b")


def _sensitive(path: Path) -> bool:
    for part in (value.lower() for value in path.parts):
        if (part in SENSITIVE_PARTS or part.startswith(".env")
                or any(part.endswith(suffix) for suffix in (".pem", ".key"))):
            return True
    name = path.name.lower()
    if name == "settings.json" or ("settings" in name and name.endswith(".json")):
        return True
    return False


def _directory_log_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".log" or (
        path.suffix.lower() == ".txt" and (name.startswith("device") or name.startswith("exit"))
    )


def _redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text[:2000]


def _kind(text: str) -> str | None:
    if TRACEBACK_RE.search(text):
        return "traceback"
    if DEVICE_ERROR_RE.search(text):
        return "device_error"
    # Startup warnings such as "WARNING failed plugin probe" are not actual
    # failures.  Do not let a warning burst exhaust the bounded event budget.
    if WARNING_RE.search(text):
        return None
    if ACTUAL_ERROR_RE.search(text):
        return "actual_error"
    return None


def _files(inputs, project_root: Path):
    found = []
    skipped = []
    for raw in inputs:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(project_root):
                raise ValueError("path is outside project root")
        except (OSError, ValueError) as exc:
            skipped.append({"path": str(candidate), "reason": type(exc).__name__ + ": " + str(exc)[:200]})
            continue
        if _sensitive(resolved):
            skipped.append({"path": str(resolved), "reason": "sensitive_path_not_read"})
            continue
        if resolved.is_file() and not resolved.is_symlink():
            found.append(resolved)
        elif resolved.is_dir():
            for item in sorted(resolved.rglob("*")):
                if item.is_symlink() or not item.is_file():
                    continue
                try:
                    item_resolved = item.resolve(strict=True)
                    if not item_resolved.is_relative_to(project_root):
                        skipped.append({"path": str(item), "reason": "symlink_or_path_escape"})
                        continue
                except OSError as exc:
                    skipped.append({"path": str(item), "reason": type(exc).__name__})
                    continue
                if _sensitive(item_resolved):
                    skipped.append({"path": str(item_resolved), "reason": "sensitive_path_not_read"})
                    continue
                if not _directory_log_file(item_resolved):
                    continue
                found.append(item_resolved)
        else:
            skipped.append({"path": str(resolved), "reason": "not_a_regular_file_or_directory"})
    return sorted(set(found), key=lambda value: str(value)), skipped[:128]


def extract(inputs, project_root, *, context_lines=3, max_events=32, tail_events=12,
            max_file_bytes=64 * 1024 * 1024, max_line_bytes=16 * 1024):
    if context_lines < 0 or context_lines > 20:
        raise ValueError("context-lines must be between 0 and 20")
    if max_events < 1 or max_events > 128:
        raise ValueError("max-events must be between 1 and 128")
    if tail_events < 1 or tail_events > 64:
        raise ValueError("tail-events must be between 1 and 64")
    if max_file_bytes < 1024 or max_file_bytes > 512 * 1024 * 1024:
        raise ValueError("max-file-bytes is outside the bounded range")
    if max_line_bytes < 256 or max_line_bytes > 1024 * 1024:
        raise ValueError("max-line-bytes is outside the bounded range")

    project_root = project_root.resolve(strict=True)
    paths, skipped = _files(inputs, project_root)
    events = []
    tails = deque(maxlen=tail_events)
    active_contexts = []
    before = deque(maxlen=context_lines)
    file_stats = []
    first_traceback = None
    first_error = None

    for path in paths:
        line_number = 0
        bytes_seen = 0
        truncated = False
        try:
            with path.open("rb") as stream:
                while bytes_seen < max_file_bytes:
                    raw = stream.readline(max_line_bytes + 1)
                    if not raw:
                        break
                    bytes_seen += len(raw)
                    line_number += 1
                    if len(raw) > max_line_bytes and not raw.endswith(b"\n"):
                        truncated = True
                        while stream.readline(max_line_bytes):
                            pass
                    text = _redact(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                    for capture in active_contexts[:]:
                        capture["context"].append({"line": line_number, "text": text})
                        capture["remaining"] -= 1
                        if capture["remaining"] <= 0:
                            active_contexts.remove(capture)
                    kind = _kind(text)
                    if kind:
                        event = {"path": str(path), "line": line_number, "kind": kind, "text": text}
                        tails.append({"path": str(path), "line": line_number, "kind": kind, "text": text})
                        if len(events) < max_events:
                            event["context"] = list(before) + [{"line": line_number, "text": text}]
                            events.append(event)
                            active_contexts.append({"event": event, "context": event["context"],
                                                    "remaining": context_lines})
                        if kind == "traceback" and first_traceback is None:
                            first_traceback = event
                        if kind != "traceback" and first_error is None:
                            first_error = event
                    before.append({"line": line_number, "text": text})
                    if bytes_seen >= max_file_bytes:
                        truncated = True
                        break
        except (OSError, UnicodeError) as exc:
            skipped.append({"path": str(path), "reason": type(exc).__name__ + ": " + str(exc)[:200]})
            continue
        file_stats.append({"path": str(path), "lines": line_number, "bytes": bytes_seen,
                           "truncated": truncated})

    return {
        "schema": "fullnet.failure-context.v1",
        "status": "ok" if file_stats else "no_safe_logs",
        "error_candidates_found": bool(events),
        "no_error_candidates": not events,
        "first_traceback": first_traceback,
        "first_error": first_error,
        "events": events,
        "tail_summary": list(tails),
        "files": file_stats,
        "skipped": skipped,
        "note": "first observed traceback/error is evidence, not a root-cause conclusion",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="task, service, or PD log files/directories")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--context-lines", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--tail-events", type=int, default=12)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-line-bytes", type=int, default=16 * 1024)
    args = parser.parse_args(argv)
    try:
        result = extract(args.paths, args.project_root, context_lines=args.context_lines,
                         max_events=args.max_events, tail_events=args.tail_events,
                         max_file_bytes=args.max_file_bytes, max_line_bytes=args.max_line_bytes)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": "fullnet.failure-context.v1", "status": "invalid_input",
                          "error": type(exc).__name__, "detail": str(exc)[:500]}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
