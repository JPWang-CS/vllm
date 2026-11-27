"""Fail-closed parsing for the Pangu 7B real-device preflight.

The remote ``npu-smi`` wrapper has emitted both table rows and key/value
records. This parser accepts only those structured forms and an explicit
platform/model allowlist; a generic ``Ascend*`` match is not enough to pass
a platform-specific gate.
"""
from __future__ import annotations

import re


_PLATFORM_DEVICE_NAMES = {
    "a2": frozenset(("910b1", "ascend910b1", "910b2", "ascend910b2",
                      "910b3", "ascend910b3")),
    "a3": frozenset(("910", "ascend910", "910c", "ascend910c")),
    "a5": frozenset(("950dt", "ascend950dt")),
}


def configure_pangu7b_platform_env(base_env: dict[str, str], platform: str) -> dict[str, str]:
    """Return a private runner env with only the platform hint normalized.

    A2/A3 must not inherit an A5 compute-unit hint from the launching shell,
    but other runtime values remain untouched. The input mapping and
    ``os.environ`` are never modified.
    """
    if platform not in _PLATFORM_DEVICE_NAMES:
        raise ValueError(f"unsupported device platform: {platform}")
    env = dict(base_env)
    hint = env.get("ASCEND_COMPUTE_UNIT")
    is_950_hint = isinstance(hint, str) and re.match(r"^(?:ascend)?950", _normalise(hint))
    if platform == "a5":
        env["ASCEND_COMPUTE_UNIT"] = "ascend950"
    elif is_950_hint:
        env.pop("ASCEND_COMPUTE_UNIT", None)
    return env


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _append(records, seen, current):
    if not current or current.get("device_id") is None or not current.get("name"):
        return
    item = {"device_id": current["device_id"], "name": current["name"],
            "status": current.get("status", "")}
    # Do not deduplicate: A3 reports two chip-health rows under one NPU id,
    # and every row must be healthy before that board can pass.
    records.append(item)


def parse_npu_smi_records(text: str) -> list[dict[str, object]]:
    """Parse only rows/records with an explicit device id and name."""
    records, seen = [], set()
    current = None
    for line in text.splitlines():
        cells = [part.strip() for part in line.split("|")]
        if len(cells) >= 3:
            # npu-smi 25.x emits both ``| 0 | Ascend950DT | OK |`` and
            # ``| 0     910B2 | OK |``. The latter is also the A3 form
            # with two chip rows sharing the same NPU id.
            merged = re.fullmatch(r"(\d+)\s+([A-Za-z0-9][A-Za-z0-9 _-]*)", cells[1])
            if (merged and re.search(r"[A-Za-z]", merged.group(2)) and
                    re.fullmatch(r"[A-Za-z][A-Za-z_-]*", cells[2])):
                _append(records, seen, {"device_id": int(merged.group(1)),
                                        "name": merged.group(2).strip(),
                                        "status": cells[2]})
                continue
            if cells[1].isdigit() and len(cells) >= 4:
                name, status = cells[2], cells[3]
                # Do not mistake the follow-up Chip/Bus row for a health row.
                if re.search(r"[A-Za-z]", name) and re.fullmatch(r"[A-Za-z][A-Za-z_-]*", status):
                    _append(records, seen, {"device_id": int(cells[1]),
                                            "name": name, "status": status})
                    continue

        device = re.search(r"\b(?:NPU|Device)(?:\s+ID)?\s*[:=]\s*(\d+)\b", line, re.I)
        if device:
            _append(records, seen, current)
            current = {"device_id": int(device.group(1))}
        if current is None:
            continue
        name = re.search(
            r"\bName\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9 _-]*?)(?=\s+\b(?:Status|Health|State)\s*[:=]|\s*$)",
            line, re.I)
        if name:
            current["name"] = name.group(1).strip()
        status = re.search(r"\b(?:Status|Health|State)\s*[:=]\s*([A-Za-z0-9_-]+)", line, re.I)
        if status:
            current["status"] = status.group(1)
    _append(records, seen, current)
    return records


def verify_npu_smi_output(text: str, platform: str, npu_id: int) -> dict[str, object]:
    """Validate one idle device, raising on every unknown/ambiguous case."""
    allowed = _PLATFORM_DEVICE_NAMES.get(platform)
    if allowed is None:
        raise RuntimeError(f"unsupported device platform: {platform}")
    records = parse_npu_smi_records(text)
    targets = [item for item in records if item["device_id"] == npu_id]
    if not targets:
        raise RuntimeError(f"target NPU {npu_id} has no structured name/status record")
    for target in targets:
        normalized = _normalise(str(target["name"]))
        if normalized not in allowed:
            raise RuntimeError(f"target NPU {npu_id} device name is not allowlisted for {platform}: {target['name']}")
        if _normalise(str(target.get("status", ""))) != "ok":
            raise RuntimeError(f"target NPU {npu_id} health is not OK: {target.get('status', '')}")
    if not re.search(r"No running processes found in NPU\s+" + re.escape(str(npu_id)) + r"\b", text, re.I):
        raise RuntimeError(f"target NPU {npu_id} is occupied or process visibility is unconfirmed")
    return targets[0]
