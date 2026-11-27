"""Parser dependency preflight checks; no SDK, model, network or NPU required."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "fullnet-profile" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import profile_recovery as recovery  # noqa: E402


FAKE_CLI = r'''
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
receipt = os.environ.get("CLI_ENV_RECEIPT")
if receipt:
    with Path(receipt).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "argv": argv,
            "preflight_sentinel": os.environ.get("PREFLIGHT_SENTINEL"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        }) + "\n")
if "--help" in argv:
    print("--collection-dir")
    raise SystemExit(0)

collection = Path(argv[argv.index("--collection-dir") + 1])
if argv[:2] == ["export", "summary"]:
    (collection / "op_summary_generated.csv").write_text(
        "Task Type,Device_id,Task Duration(us)\nAI_VECTOR_CORE,4,9\n", encoding="utf-8")
elif argv[:2] == ["export", "timeline"]:
    (collection / "msprof_generated.json").write_text(json.dumps([
        {"ph": "X", "cat": "aicore", "dur": 9, "args": {"Device_id": 4}}
    ]), encoding="utf-8")
'''


class ParserPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "capture"
        raw = self.source / "PROF_rank0" / "device_4" / "data"
        raw.mkdir(parents=True)
        (raw / "stars_soc.data").write_bytes(b"raw-capture")
        self.cann = self.root / "cann"
        self.cann.mkdir()
        self.entry = self.root / "msprof.py"
        self.entry.write_text(FAKE_CLI, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, output):
        return argparse.Namespace(
            source=self.source,
            output=output,
            cann_root=self.cann,
            entry=self.entry,
            python=sys.executable,
            timeout=20,
            library_dir=[],
        )

    def test_failed_sqlite_gate_blocks_official_cli_and_copy(self):
        shadow = self.root / "shadow"
        shadow.mkdir()
        (shadow / "sqlite3.py").write_text(
            "raise ImportError('libsqlite3.so: cannot open shared object file')\n",
            encoding="utf-8")
        output = self.root / "failed-preflight"
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(shadow)}, clear=False):
            receipt = recovery.export(self.args(output))

        self.assertEqual(receipt["status"], "FAIL")
        self.assertEqual(receipt["failure_stage"], "parser_dependency_preflight")
        self.assertEqual(receipt["first_error"], "ImportError: libsqlite3.so: cannot open shared object file")
        self.assertEqual(receipt["official_cli"], "not_run")
        self.assertFalse((output / "rank-0").exists())
        self.assertTrue((output / "RECOVERY.json").is_file())
        preflight = receipt["parser_dependency_preflight"]
        self.assertFalse(preflight["passed"])
        self.assertIn("-X", preflight["argv"])
        self.assertIn("utf8", preflight["argv"])
        self.assertIn("ImportError: libsqlite3.so", preflight["stderr"])

    def test_same_environment_reaches_preflight_and_official_cli(self):
        output = self.root / "env-forwarded"
        cli_receipt = self.root / "cli-env.jsonl"
        private_ld = str(self.root / "parser-libs")
        with mock.patch.dict(os.environ, {
            "PREFLIGHT_SENTINEL": "same-task-environment",
            "LD_LIBRARY_PATH": private_ld,
            "CLI_ENV_RECEIPT": str(cli_receipt),
        }, clear=False):
            receipt = recovery.export(self.args(output))

        self.assertEqual(receipt["status"], "EXPORTED")
        self.assertEqual(receipt["parser_dependency_preflight"]["ld_library_path"], private_ld)
        cli_records = [json.loads(line) for line in cli_receipt.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(cli_records), 7)  # four help calls plus import/summary/timeline
        self.assertTrue(all(item["preflight_sentinel"] == "same-task-environment" for item in cli_records))
        self.assertTrue(all(item["ld_library_path"] == private_ld for item in cli_records))
        self.assertEqual(receipt["ranks"][0]["expected_devices"], ["4"])

    def test_successful_preflight_preserves_original_export_contract(self):
        output = self.root / "successful-export"
        receipt = recovery.export(self.args(output))

        self.assertEqual(receipt["status"], "EXPORTED")
        self.assertTrue(receipt["parser_dependency_preflight"]["passed"])
        self.assertIn("python_executable=", receipt["parser_dependency_preflight"]["stdout"])
        self.assertIn("sqlite_runtime_version=", receipt["parser_dependency_preflight"]["stdout"])
        self.assertEqual(receipt["ranks"][0]["steps"][-1]["passed"], True)
        self.assertTrue(receipt["ranks"][0]["original_metadata_unchanged"])


if __name__ == "__main__":
    unittest.main()
