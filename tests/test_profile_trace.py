import csv
import gzip
import json

import pytest

from fullnet.measurements import _csv_device_kernel_rows, _trace_device_kernel_events
from fullnet.profile_trace import iter_trace_events


def _x(task="AI_CORE", device=None, duration=2, cat="kernel", pid=1):
    args = {"Task Type": task}
    if device is not None:
        args["Device_id"] = device
    return {"name": "kernel", "ph": "X", "cat": cat, "pid": pid,
            "args": args, "dur": duration, "ts": "2.5"}


def test_streaming_yields_first_array_item_before_large_tail(tmp_path):
    path = tmp_path / "large-prefix.json"
    first = json.dumps(_x(device=1)).encode()
    path.write_bytes(b"[" + first + b"," + (b" " * (2 << 20)) + b"BROKEN]")
    first_event = next(iter_trace_events(path))
    assert first_event["args"]["Task Type"] == "AI_CORE"


def test_array_object_events_object_and_gzip(tmp_path):
    events = [_x(device=1), _x("MIX_AIV", device=1)]
    array = tmp_path / "array.json"
    array.write_text(json.dumps(events), encoding="utf-8")
    obj = tmp_path / "events.json"
    obj.write_text(json.dumps({"events": events}), encoding="utf-8")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    zipped = tmp_path / "trace.json.gz"
    with gzip.open(zipped, "wb") as handle:
        handle.write(trace.read_bytes())
    assert len(list(iter_trace_events(array))) == 2
    assert len(list(iter_trace_events(obj))) == 2
    assert len(list(iter_trace_events(zipped))) == 2


def test_root_closure_header_and_trailing_validation(tmp_path):
    valid_array = tmp_path / "valid-array.json"
    valid_array.write_text(json.dumps([_x(device=1)]) + "  \n", encoding="utf-8")
    assert len(list(iter_trace_events(valid_array))) == 1

    trailing = tmp_path / "trailing.json"
    trailing.write_text(json.dumps([_x(device=1)]) + "junk", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_trace_events(trailing))

    missing_object_close = tmp_path / "missing-object-close.json"
    missing_object_close.write_text(json.dumps({"traceEvents": [_x(device=1)]})[:-1],
                                    encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_trace_events(missing_object_close))

    bad_header = tmp_path / "bad-header.json"
    bad_header.write_text('{"traceEvents" broken:[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_trace_events(bad_header))

    valid_object_tail = tmp_path / "valid-object-tail.json"
    valid_object_tail.write_text(json.dumps({"traceEvents": [_x(device=1)], "meta": 1}),
                                 encoding="utf-8")
    assert len(list(iter_trace_events(valid_object_tail))) == 1


@pytest.mark.parametrize("text", [
    '[{"a":1}{"b":2}]',
    '[{"a":1},,{"b":2}]',
    '[{"a":1},]',
])
def test_array_requires_single_comma_between_items(tmp_path, text):
    path = tmp_path / "bad-separator.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_trace_events(path))


def test_truncated_and_late_or_missing_metadata(tmp_path):
    truncated = tmp_path / "truncated.json"
    truncated.write_text('[{"ph":"X","name":"broken"}', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_trace_events(truncated))

    late = tmp_path / "late.json"
    late.write_text(json.dumps({"traceEvents": [
        _x(device=None),
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "NPU Device 1"}},
    ]}), encoding="utf-8")
    assert _trace_device_kernel_events(late) == 1

    missing = tmp_path / "missing-m.json"
    missing.write_text(json.dumps({"traceEvents": [_x(device=None)]}), encoding="utf-8")
    assert _trace_device_kernel_events(missing) == 0


def test_compute_gate_excludes_comm_and_invalid_duration(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"traceEvents": [
        _x("HCCL", device=1, cat="kernel"),
        _x("COMMUNICATION", device=1),
        _x("AI_CORE", device=1, duration=0),
        _x("AI_CORE", device=1, duration="nan"),
        _x("AI_CORE", device=1, duration="inf"),
        _x("KERNELMIXAIC", device=1),
    ]}), encoding="utf-8")
    assert _trace_device_kernel_events(path) == 1


def test_actual_csv_headers_and_tasktime_duration(tmp_path):
    path = tmp_path / "task_time.csv"
    path.write_text("Device_id,Kernel Name,Task Type,TaskTime(us)\n"
                    "1,legacy,KERNELMIXAIC,3\n"
                    "1,cpu,CPU,5\n"
                    "1,bad,AI_CORE,nan\n", encoding="utf-8")
    assert _csv_device_kernel_rows(path) == 1
