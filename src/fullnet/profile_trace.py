"""Streaming readers for profiler trace JSON without materializing event arrays."""
from __future__ import annotations

import codecs
import gzip
import json


MAX_ROOT_BUFFER = 8 << 20


def iter_trace_events(path):
    """Yield trace events from a top-level array or an events/traceEvents array."""
    opener = gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")
    decoder = json.JSONDecoder()
    incremental = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf, pos, in_items = "", 0, False
    root_kind, header, tail, root_closed = None, "", "", False
    expect_value, after_comma = True, False
    with opener as handle:
        while True:
            chunk = handle.read(1 << 20)
            decoded = incremental.decode(chunk) if chunk else ""
            if root_closed:
                tail += decoded
                if len(tail) > MAX_ROOT_BUFFER:
                    raise ValueError("root JSON tail exceeds bounded validation buffer")
                if chunk:
                    continue
                candidate = "[]" + tail if root_kind == "array" else header + "[]" + tail
                try:
                    json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid root closure or trailing JSON: {exc}") from exc
                return
            if decoded:
                buf += decoded
            if not in_items:
                stripped = buf.lstrip()
                if stripped.startswith("["):
                    root_kind, buf, pos, in_items = "array", stripped[1:], 0, True
                    continue
                if len(buf) > MAX_ROOT_BUFFER:
                    raise ValueError("trace JSON header exceeds bounded validation buffer")
                if not chunk and not buf:
                    return
                positions = [idx for idx in (buf.find('"traceEvents"'), buf.find('"events"'))
                             if idx >= 0]
                if not positions:
                    if not chunk:
                        raise ValueError("traceEvents/events key not found")
                    continue
                bracket = buf.find("[", min(positions))
                if bracket < 0:
                    if not chunk:
                        raise ValueError("trace event array not found")
                    continue
                header, root_kind = buf[:bracket], "object"
                try:
                    json.loads(header + "[]}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid trace JSON header: {exc}") from exc
                buf, pos, in_items = buf[bracket + 1:], 0, True
                continue
            while True:
                n = len(buf)
                while pos < n and buf[pos] in " \t\r\n":
                    pos += 1
                if pos >= n:
                    break
                if expect_value:
                    if buf[pos] == "]":
                        if after_comma:
                            raise ValueError("trailing comma in trace event array")
                        tail, root_closed = buf[pos + 1:], True
                        if len(tail) > MAX_ROOT_BUFFER:
                            raise ValueError("root JSON tail exceeds bounded validation buffer")
                        buf, pos = "", 0
                        break
                    if buf[pos] == ",":
                        raise ValueError("unexpected comma in trace event array")
                else:
                    if buf[pos] == ",":
                        pos += 1
                        expect_value, after_comma = True, True
                        continue
                    if buf[pos] == "]":
                        tail, root_closed = buf[pos + 1:], True
                        if len(tail) > MAX_ROOT_BUFFER:
                            raise ValueError("root JSON tail exceeds bounded validation buffer")
                        buf, pos = "", 0
                        break
                    raise ValueError("missing comma between trace events")
                try:
                    event, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if not chunk:
                        raise ValueError(f"truncated/malformed trace JSON near byte {pos}")
                    break
                pos = end
                expect_value, after_comma = False, False
                yield event
            if root_closed:
                continue
            if pos:
                buf, pos = buf[pos:], 0
            if not chunk:
                raise ValueError("trailing or truncated trace JSON")
