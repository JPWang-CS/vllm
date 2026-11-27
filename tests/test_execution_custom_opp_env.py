"""CPU-only regression tests for the task-private runtime environment."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fullnet.execution import private_environment


class CustomOppEnvironmentTest(unittest.TestCase):
    def test_empty_custom_opp_is_removed_and_nonempty_values_are_preserved(self):
        cases = (
            ("", {}, None),
            ("/inherited", {"ASCEND_CUSTOM_OPP_PATH": ""}, None),
            ("/inherited", {}, "/inherited"),
            ("/inherited", {"ASCEND_CUSTOM_OPP_PATH": "/configured"}, "/configured"),
            ("/inherited", {"ASCEND_CUSTOM_OPP_PATH": "  "}, "  "),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "task"
            run.mkdir()
            for inherited, config_env, expected in cases:
                with self.subTest(inherited=inherited, config_env=config_env):
                    with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": inherited}, clear=False):
                        env = private_environment(run, {"platform": "cpu-test", "env": config_env}, root)
                    if expected is None:
                        self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", env)
                    else:
                        self.assertEqual(env["ASCEND_CUSTOM_OPP_PATH"], expected)
                    self.assertTrue(Path(env["VLLM_RPC_BASE_PATH"]).resolve().is_relative_to((root / ".i").resolve()))

    def test_missing_custom_opp_stays_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "task"
            run.mkdir()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
                env = private_environment(run, {"platform": "cpu-test", "env": {}}, root)
            self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", env)


if __name__ == "__main__":
    unittest.main()
