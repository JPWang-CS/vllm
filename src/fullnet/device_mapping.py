"""Pure-CPU parsing and binding of npu-smi physical devices to ACL runtimes.

The npu-smi ``info -m`` tables use NPU ID as the board identifier.  A3
exposes Chip Logic ID while A5 exposes Slot ID; neither field is a runtime
device selector.  Runtime identity comes only from host-bound ACL evidence
for ``aclrtGetUserDevIdByPhyDevId``.
"""
from __future__ import annotations

from typing import Any


ACL_API = "aclrtGetUserDevIdByPhyDevId"
_HEADERS = ("NPU ID", "Slot ID", "Chip ID", "Chip Logic ID", "Chip Phy-ID", "Chip Name")


def _int_field(fields: dict[str, str], name: str, *, optional: bool = False) -> int | None:
    value = fields.get(name, "").strip()
    if optional and value in {"", "-"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mapping field {name!r} is not an integer: {value!r}") from exc


def _columns(header: str) -> list[tuple[int, str]]:
    found = sorted((header.index(label), label) for label in _HEADERS if label in header)
    names = {label for _, label in found}
    required = {"NPU ID", "Chip ID", "Chip Phy-ID", "Chip Name"}
    if not required.issubset(names):
        raise ValueError("mapping header must contain NPU ID, Chip ID, Chip Phy-ID and Chip Name")
    if "Slot ID" not in names and "Chip Logic ID" not in names:
        raise ValueError("mapping header must contain Slot ID or Chip Logic ID")
    return found


def parse_npu_smi_mapping(text: str) -> dict[str, Any]:
    """Parse an A3 or A5 fixed-width ``npu-smi info -m`` table.

    Returned row keys deliberately distinguish ``slot``, ``chip``,
    ``chip_phy`` and optional ``chip_logic``.  MCU rows and rows without a
    numeric Chip Phy-ID are not device mappings and are excluded.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("npu-smi mapping text is required")
    lines = [line.expandtabs() for line in text.splitlines()]
    header_index = next(
        (index for index, line in enumerate(lines)
         if "NPU ID" in line and "Chip Phy-ID" in line and "Chip Name" in line
         and ("Slot ID" in line or "Chip Logic ID" in line)),
        None,
    )
    if header_index is None:
        raise ValueError("npu-smi mapping header not found")
    header = lines[header_index]
    columns = _columns(header)
    starts = [start for start, _ in columns]
    rows: list[dict[str, Any]] = []
    for raw in lines[header_index + 1:]:
        if not raw.strip() or set(raw.strip()) <= {"+", "=", "-"}:
            continue
        values = [raw[start:end].strip() for start, end in
                  zip(starts, starts[1:] + [None])]
        fields = {label: value for (_, label), value in zip(columns, values)}
        name = fields.get("Chip Name", "").strip()
        if not name or name.casefold() == "mcu":
            continue
        try:
            board = _int_field(fields, "NPU ID")
            chip = _int_field(fields, "Chip ID")
            chip_phy = _int_field(fields, "Chip Phy-ID")
            slot = _int_field(fields, "Slot ID", optional=True)
            chip_logic = _int_field(fields, "Chip Logic ID", optional=True)
        except ValueError:
            continue
        if board is None or chip is None or chip_phy is None:
            continue
        rows.append({
            "board": board,
            "slot": slot,
            "chip": chip,
            "chip_phy": chip_phy,
            "chip_logic": chip_logic,
            "chip_name": name,
        })
    if not rows:
        raise ValueError("npu-smi mapping contains no physical device rows")
    return {"header": header.strip(), "rows": rows}


def _acl_envelope(evidence: dict[str, Any], expected_host: str) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ValueError("ACL mapping evidence must be an object")
    value: Any = evidence
    hosts = evidence.get("hosts")
    if isinstance(hosts, dict):
        if expected_host not in hosts or not isinstance(hosts[expected_host], dict):
            raise ValueError(f"ACL mapping aggregate has no host envelope for {expected_host!r}")
        value = hosts[expected_host]
    if not isinstance(value, dict) or value.get("host") != expected_host:
        raise ValueError(f"ACL mapping host does not match expected host {expected_host!r}")
    if value.get("api") != ACL_API:
        raise ValueError(f"ACL mapping must use {ACL_API}")
    if value.get("visible_unset") is not True:
        raise ValueError("ACL mapping requires visible_unset=true")
    source_job = value.get("source_job")
    if isinstance(source_job, str):
        if not source_job.strip():
            raise ValueError("ACL mapping source_job is required")
    elif isinstance(source_job, dict):
        if source_job.get("exit") != 0 or not source_job.get("job"):
            raise ValueError("ACL mapping source_job must record a successful job")
    else:
        raise ValueError("ACL mapping source_job is required")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ACL mapping rows are required")
    return value


def _acl_rows(value: dict[str, Any]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen_phy: set[int] = set()
    seen_runtime: set[int] = set()
    for row in value["rows"]:
        if not isinstance(row, dict) or type(row.get("physical_id")) is not int \
                or type(row.get("user_device")) is not int or row.get("rc") != 0:
            raise ValueError("ACL rows require integer physical_id/user_device and rc=0")
        physical = row["physical_id"]
        runtime = row["user_device"]
        if physical < 0 or runtime < 0:
            raise ValueError("ACL physical_id and user_device must be non-negative")
        if physical in seen_phy or runtime in seen_runtime:
            raise ValueError("ACL mapping must be unique by physical_id and user_device")
        seen_phy.add(physical)
        seen_runtime.add(runtime)
        result.append({"physical_id": physical, "user_device": runtime, "rc": 0})
    return result


def _validate_optional_mappings(value: dict[str, Any], acl_rows: list[dict[str, int]]) -> None:
    mappings = value.get("mappings")
    if mappings is None and isinstance(value.get("_aggregate_mappings"), list):
        mappings = value["_aggregate_mappings"]
    if mappings is None:
        return
    if not isinstance(mappings, list) or len(mappings) != len(acl_rows):
        raise ValueError("ACL mappings must be a complete list when present")
    expected = {(row["physical_id"], row["user_device"]) for row in acl_rows}
    actual: set[tuple[int, int]] = set()
    for row in mappings:
        if not isinstance(row, dict) or row.get("api") != ACL_API \
                or type(row.get("board_id")) is not int \
                or type(row.get("physical_id")) is not int \
                or type(row.get("runtime_device")) is not int:
            raise ValueError("ACL mappings require board_id, physical_id, runtime_device and matching api")
        key = (row["physical_id"], row["runtime_device"])
        if key in actual:
            raise ValueError("ACL mappings contain duplicate physical/runtime pairs")
        actual.add(key)
    if actual != expected:
        raise ValueError("ACL mappings do not cover the ACL result rows exactly")


def bind_runtime_mapping(npu_text: str, acl_evidence: dict[str, Any], expected_host: str,
                         *, required_board_ids: list[int] | None = None) -> dict[str, Any]:
    """Bind host-scoped ACL user devices to npu-smi boards by Chip Phy-ID.

    Every ACL row must match exactly one parsed physical row and every parsed
    physical row must have one ACL result.  No Slot ID, Chip ID or positional
    fallback is used.  The caller must attach the actual host to a raw ACL
    probe result before calling this function.
    """
    if not isinstance(expected_host, str) or not expected_host.strip():
        raise ValueError("expected_host is required")
    parsed = parse_npu_smi_mapping(npu_text)
    if required_board_ids is not None:
        if (not isinstance(required_board_ids, list) or not required_board_ids
                or any(type(board) is not int or board < 0 for board in required_board_ids)
                or len(set(required_board_ids)) != len(required_board_ids)):
            raise ValueError("required_board_ids must be a non-empty unique integer list")
        selected = set(required_board_ids)
        parsed_rows = [row for row in parsed["rows"] if row["board"] in selected]
        if not parsed_rows or {row["board"] for row in parsed_rows} != selected:
            raise ValueError("required_board_ids contains a board absent from npu-smi mapping")
    else:
        parsed_rows = parsed["rows"]
    value = _acl_envelope(acl_evidence, expected_host)
    if isinstance(acl_evidence.get("mappings"), list) and "mappings" not in value:
        value = dict(value)
        value["_aggregate_mappings"] = acl_evidence["mappings"]
    acl_rows = _acl_rows(value)
    _validate_optional_mappings(value, acl_rows)
    by_phy: dict[int, dict[str, Any]] = {}
    for row in parsed["rows"]:
        if row["chip_phy"] in by_phy:
            raise ValueError(f"npu-smi mapping duplicates Chip Phy-ID {row['chip_phy']}")
        by_phy[row["chip_phy"]] = row
    selected_by_phy = {row["chip_phy"]: row for row in parsed_rows}
    acl_physical = {row["physical_id"] for row in acl_rows}
    if acl_physical != set(selected_by_phy):
        raise ValueError("ACL rows and required npu-smi physical rows are not a complete set")
    bound: list[dict[str, Any]] = []
    for acl in acl_rows:
        source = selected_by_phy[acl["physical_id"]]
        bound.append({**source, "runtime_device": acl["user_device"], "acl_rc": 0})
    bound.sort(key=lambda row: (row["board"], row["chip_phy"]))
    return {
        "schema": "device_runtime_mapping.v1",
        "host": expected_host,
        "api": ACL_API,
        "visible_unset": True,
        "source_job": value["source_job"],
        "required_board_ids": sorted({row["board"] for row in parsed_rows}),
        "rows": bound,
        "complete": True,
        "unique": True,
    }
