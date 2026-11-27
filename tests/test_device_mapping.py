"""CPU tests for strict npu-smi/ACL runtime mapping."""
from __future__ import annotations

import pytest

from fullnet.device_mapping import bind_runtime_mapping, parse_npu_smi_mapping


def a3_table() -> str:
    return (
        "NPU ID                         Chip ID                        Chip Logic ID                  Chip Phy-ID                    Chip Name\n"
        "0                              0                              7                              2                              Ascend910\n"
        "2                              0                              9                              4                              Ascend910\n"
        "2                              1                              10                             5                              Ascend910\n"
        "2                              2                              -                              -                              Mcu\n"
    )


def a5_table() -> str:
    return (
        "NPU ID                         Slot ID                        Chip ID                        Chip Phy-ID                    Chip Name\n"
        "0                              7                              0                              2                              Ascend950DT\n"
        "2                              3                              0                              4                              Ascend950DT\n"
    )


def a5_full_table() -> str:
    header = f"{'NPU ID':<15}{'Slot ID':<15}{'Chip ID':<15}{'Chip Phy-ID':<15}{'Chip Name':<15}"
    rows = [f"{board:<15}{board + 1:<15}{0:<15}{phy:<15}{'Ascend950DT':<15}"
            for board, phy in enumerate((2, 1, 4, 6))]
    return header + "\n" + "\n".join(rows)


def acl(host: str = "host-10", rows=None, *, visible=True, api="aclrtGetUserDevIdByPhyDevId"):
    rows = rows or [
        {"physical_id": 2, "rc": 0, "user_device": 1},
        {"physical_id": 4, "rc": 0, "user_device": 3},
    ]
    return {
        "schema": "acl_mapping_all.v1",
        "hosts": {host: {"host": host, "api": api, "visible_unset": visible,
                         "source_job": "acl-probe-01", "rows": rows}},
    }


def test_parse_a3_keeps_chip_logic_and_omits_mcu():
    parsed = parse_npu_smi_mapping(a3_table())
    assert [(row["board"], row["chip"], row["chip_phy"], row["chip_logic"], row["slot"])
            for row in parsed["rows"]] == [(0, 0, 2, 7, None), (2, 0, 4, 9, None), (2, 1, 5, 10, None)]


def test_parse_a5_does_not_turn_slot_into_runtime_or_physical():
    rows = parse_npu_smi_mapping(a5_table())["rows"]
    assert rows[0]["slot"] == 7
    assert rows[0]["chip_phy"] == 2
    assert rows[0]["slot"] != rows[0]["chip_phy"]


def test_bind_nonidentity_runtime_by_phy_and_board():
    result = bind_runtime_mapping(a5_table(), acl(), "host-10")
    assert [(row["board"], row["chip_phy"], row["runtime_device"])
            for row in result["rows"]] == [(0, 2, 1), (2, 4, 3)]
    assert result["complete"] is True
    assert result["unique"] is True


def test_bind_selected_board_subset_does_not_require_other_driver_rows():
    result = bind_runtime_mapping(a5_full_table(), acl(), "host-10", required_board_ids=[0, 2])
    assert [(row["board"], row["chip_phy"], row["runtime_device"])
            for row in result["rows"]] == [(0, 2, 1), (2, 4, 3)]
    assert result["required_board_ids"] == [0, 2]


def test_bind_selected_board_missing_acl_row_fails():
    evidence = acl(rows=[{"physical_id": 2, "rc": 0, "user_device": 1}])
    with pytest.raises(ValueError, match="complete set"):
        bind_runtime_mapping(a5_full_table(), evidence, "host-10", required_board_ids=[0, 2])


@pytest.mark.parametrize("bad", [
    acl(rows=[{"physical_id": 2, "rc": 0, "user_device": 1}]),
    acl(rows=[{"physical_id": 7, "rc": 0, "user_device": 1},
             {"physical_id": 4, "rc": 0, "user_device": 3}]),
])
def test_missing_or_unknown_physical_fails_closed(bad):
    with pytest.raises(ValueError):
        bind_runtime_mapping(a5_table(), bad, "host-10")


def test_host_mismatch_fails_before_binding():
    with pytest.raises(ValueError, match="host"):
        bind_runtime_mapping(a5_table(), acl("host-10"), "host-115")


@pytest.mark.parametrize("rows", [
    [{"physical_id": 2, "rc": 1, "user_device": 1},
     {"physical_id": 4, "rc": 0, "user_device": 3}],
    [{"physical_id": 2, "rc": 0, "user_device": 1},
     {"physical_id": 2, "rc": 0, "user_device": 3}],
    [{"physical_id": 2, "rc": 0, "user_device": 1},
     {"physical_id": 4, "rc": 0, "user_device": 1}],
])
def test_negative_rc_and_duplicate_rows_fail(rows):
    with pytest.raises(ValueError):
        bind_runtime_mapping(a5_table(), acl(rows=rows), "host-10")


def test_slot_fallback_is_rejected():
    evidence = acl(rows=[
        {"physical_id": 7, "rc": 0, "user_device": 1},
        {"physical_id": 3, "rc": 0, "user_device": 3},
    ])
    with pytest.raises(ValueError):
        bind_runtime_mapping(a5_table(), evidence, "host-10")


def test_raw_acl_probe_without_outer_host_is_rejected():
    raw = {"api": "aclrtGetUserDevIdByPhyDevId", "visible_unset": True,
           "source_job": "acl-probe-01", "rows": [
               {"physical_id": 2, "rc": 0, "user_device": 1},
               {"physical_id": 4, "rc": 0, "user_device": 3},
           ]}
    with pytest.raises(ValueError, match="host"):
        bind_runtime_mapping(a5_table(), raw, "host-10")
