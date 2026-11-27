"""Offline behavioral checks; no SDK, model, network or NPU required."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import profile_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def trace(self, events, name="trace_view.json"):
        p = self.root / name
        p.write_text(json.dumps(events), encoding="utf-8")
        return p

    def test_cpu_queue_and_kernel_name_do_not_pass(self):
        p = self.trace([{"ph": "X", "cat": "cpu_op", "name": "my_kernel", "dur": 5},
                        {"ph": "X", "cat": "queue", "name": "kernel_launch", "dur": 10}])
        self.assertEqual(recovery.timeline_evidence(p)["compute_events"], 0)
        self.assertEqual(recovery.inspect(self.root)["acceptance"], "incomplete")

    def test_raw_data_distinguishes_export_failure_from_missing_capture(self):
        self.assertEqual(recovery.inspect(self.root)["classification"], "device_capture_unconfirmed")
        p = self.root / "PROF_example/device_4/data/stars_soc.data.4.slice_0"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"raw-capture")
        self.assertEqual(recovery.inspect(self.root)["classification"], "raw_present_export_missing")

    def test_only_complete_positive_compute_events_pass(self):
        events = [{"ph": "M", "pid": 4, "args": {"name": "Ascend Hardware"}}]
        for phase, duration, task in [("X", 8, "KERNEL_AIVEC"), ("B", 8, "KERNEL_AIVEC"),
                                      ("X", 0, "KERNEL_AIVEC"), ("X", 5, "KERNEL_AICPU"),
                                      ("X", 5, "HCCL"), ("X", -1, "KERNEL_AIVEC")]:
            events.append({"ph": phase, "pid": 4, "cat": "kernel", "dur": duration,
                           "args": {"Task Type": task}, "name": task})
        self.assertEqual(recovery.timeline_evidence(self.trace(events))["compute_events"], 1)

    def test_csv_requires_compute_task_device_and_duration(self):
        p = self.root / "op_summary_1.csv"
        p.write_text("Op Name,Task Type,Device_id,Task Duration(us)\n"
                     "vector,KERNEL_AIVEC,4,8\ncommunication,HCCL,4,8\n"
                     "aicpu,KERNEL_AICPU,4,8\nzero,KERNEL_AIVEC,4,0\n"
                     "host,KERNEL_AIVEC,,3\n", encoding="utf-8")
        self.assertEqual(recovery.csv_evidence(p)["compute_rows"], 1)

    def test_hardware_process_does_not_make_hccl_thread_compute(self):
        events = [{"ph": "M", "name": "process_name", "pid": 4, "args": {"name": "Ascend Hardware"}},
                  {"ph": "M", "name": "thread_name", "pid": 4, "tid": 7, "args": {"name": "HCCL"}},
                  {"ph": "X", "pid": 4, "tid": 7, "cat": "kernel", "dur": 10,
                   "name": "HcomAllToAll", "args": {"Device_id": 4}}]
        self.assertEqual(recovery.timeline_evidence(self.trace(events))["compute_events"], 0)

    def verify_fixture(self, timeline_device):
        work = self.root / "rank-0"
        raw = work / "PROF_test"
        (raw / "device_3/data").mkdir(parents=True)
        (raw / "device_3/data/stars_soc.data").write_bytes(b"raw")
        recovery.save(raw / "device_3/info.json.3", {"DeviceInfo": [{"id": 3}]})
        csv = raw / "op_summary_1.csv"
        csv.write_text("Task Type,Device_id,Task Duration(us)\nAI_VECTOR_CORE,3,9\n")
        timeline = raw / "msprof_1.json"
        timeline.write_text(json.dumps([{"ph": "X", "cat": "aicore", "dur": 9, "args": {"Device_id": timeline_device}}]))
        recovery.save(work / "source.json", {"copy": str(raw)})
        recovery.save(work / "generated-files.json", [str(csv), str(timeline)])
        for step in ("import", "summary", "timeline"):
            recovery.save(work / (step + ".json"), {"exit": 0, "passed": True})
        for phase in ("before", "after"):
            recovery.save(work / ("original-" + phase + ".json"), {"raw": {"size": 3, "mtime_ns": 1}})

    def test_csv_timeline_device_mismatch_fails(self):
        self.verify_fixture(4)
        receipt = recovery.verify_exports(self.root, 1, ["3"])
        self.assertEqual(receipt["status"], "FAIL")
        self.assertFalse(receipt["ranks"][0]["passed"])

    def test_missing_whole_rank_fails_explicit_contract(self):
        self.verify_fixture(3)
        self.assertEqual(recovery.verify_exports(self.root, 1, ["3"])["status"], "PASS")
        self.assertEqual(recovery.verify_exports(self.root, 2, ["3", "4"])["status"], "FAIL")

    def test_parent_zero_with_child_import_error_fails(self):
        command = [sys.executable, "-c", "import sys; print('ImportError: libsqlite3.so', file=sys.stderr)"]
        result = recovery.run_command(command, dict(os.environ), self.root, self.root / "child", 10)
        self.assertEqual(result["exit"], 0)
        self.assertTrue(result["child_failure"])
        self.assertFalse(result["passed"])
        self.assertIn("ImportError", (self.root / "child.stderr").read_text())

    def test_historical_export_failure_rejected_but_warning_allowed(self):
        failed = recovery.run_command([sys.executable, "-c", "print('export | SUB_PROCESS | Failed; ERR04005')"],
                                      dict(os.environ), self.root, self.root / "export-fail", 10)
        warning = recovery.run_command([sys.executable, "-c", "print('WARNING: unknown communication operator')"],
                                       dict(os.environ), self.root, self.root / "warning", 10)
        self.assertFalse(failed["passed"])
        self.assertTrue(warning["passed"])

    def test_timeout_keeps_partial_logs_and_failed_receipt(self):
        command = [sys.executable, "-u", "-c", "import time; print('started',flush=True); time.sleep(5)"]
        result = recovery.run_command(command, dict(os.environ), self.root, self.root / "timeout", .2)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["passed"])
        self.assertIn("started", (self.root / "timeout.stdout").read_text())

    def test_noop_export_cannot_reuse_preexisting_outputs(self):
        raw = self.root / "capture/PROF_example"
        (raw / "device_4/data").mkdir(parents=True)
        (raw / "device_4/data/stars_soc.data").write_bytes(b"raw")
        out = raw / "mindstudio_profiler_output"
        out.mkdir()
        (out / "op_summary_old.csv").write_text("Task Type,Device_id,Task Duration(us)\nKERNEL_AIVEC,4,9\n")
        (out / "msprof_old.json").write_text(json.dumps([{"ph": "X", "cat": "aicore", "dur": 9}]))
        entry = self.root / "msprof.py"
        entry.write_text("import sys\nif '--help' in sys.argv: print('--collection-dir')\n")
        args = argparse.Namespace(source=raw.parent, output=self.root / "recovered", cann_root=self.root,
                                  entry=entry, python=sys.executable, timeout=10, library_dir=[])
        receipt = recovery.export(args)
        self.assertEqual(receipt["status"], "FAIL")
        self.assertEqual(receipt["ranks"][0]["evidence"]["csv_compute_rows"], 0)
        self.assertTrue(receipt["ranks"][0]["original_metadata_unchanged"])


if __name__ == "__main__":
    unittest.main()
