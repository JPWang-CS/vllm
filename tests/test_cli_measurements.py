"""Focused tests for the public measurement flags on ``fullnet run``."""
from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from fullnet import cli


TMP_BASE = Path(__file__).parents[1] / "tmp" / "cli-measurements-focused"


VALID_MEASUREMENTS = {
    "deferred_stages": [], "workloads": [{"input": 128, "output": 64, "concurrency": 1}],
    "requests": 2, "warmups": 0, "timeout_per_group": 60,
    "profile_workload": {"input": 128, "output": 64, "concurrency": 1},
}


def make_run(root: Path, deferred=(), measurements=None, env=None, argv=None, evaluation=None,
             required=(), pd_mode=None):
    run = root / "models" / "test" / "experiments" / "r1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "model": "test",
        "measurements": dict(VALID_MEASUREMENTS if measurements is None else measurements,
                              deferred_stages=list(deferred)),
        "env": {"VLLM_TORCH_PROFILER_DIR": "/run/profiles"} if env is None else env,
        "service": {"argv": ["python", "serve.py"] if argv is None else argv},
        "evaluation": {} if evaluation is None else evaluation,
        "acceptance_contract": {
            "version": 2, "mode": "full", "pd_mode": pd_mode,
            "required_stages": {stage: stage in required for stage in ("profile", "performance")},
        },
    }), encoding="utf-8")
    return run


def result():
    return {"protocol_pass": True, "quality_status": "passed"}


class CliMeasurementsTest(unittest.TestCase):
    def project(self):
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=TMP_BASE)

    def test_default_run_uses_frozen_optional_stages_and_does_not_collect(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root)
            with patch.object(cli, "run_experiment", return_value=result()) as run, \
                    patch.object(cli, "collect_profile") as profile, \
                    patch.object(cli, "collect_performance") as performance, \
                    patch.object(cli, "pd_mixed_probe") as mixed, \
                    patch.object(cli, "pd_split_probe") as split:
                self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1"]), 0)
            run.assert_called_once_with(root.resolve(), "test", "r1", keep_ready=False)
            profile.assert_not_called()
            performance.assert_not_called()
            mixed.assert_not_called()
            split.assert_not_called()

    def test_callback_order_and_keep_ready(self):
        for probe_flag, probe_name in (("mixed", "pd_mixed_probe"), ("split", "pd_split_probe")):
            with self.subTest(probe=probe_flag), self.project() as raw:
                root = Path(raw)
                make_run(root, pd_mode=probe_flag)
                order = []
                fake_probe = Mock(side_effect=lambda run, config: order.append(probe_name))
                fake_profile = Mock(side_effect=lambda run, config: order.append("profile"))
                fake_performance = Mock(side_effect=lambda run, config: order.append("performance"))

                def fake_run(root_arg, model, run_id, *, before_evaluation=None,
                             after_evaluation=None, keep_ready=False):
                    self.assertTrue(keep_ready)
                    self.assertTrue(callable(before_evaluation))
                    if before_evaluation is not None:
                        before_evaluation(Path("run"), {"measurements": {}})
                    order.append("evaluation")
                    after_evaluation(Path("run"), {"measurements": {}}, result())
                    return result()

                with patch.object(cli, "run_experiment", side_effect=fake_run) as run, \
                        patch.object(cli, "pd_mixed_probe", fake_probe), \
                        patch.object(cli, "pd_split_probe", fake_probe), \
                        patch.object(cli, "collect_profile", fake_profile), \
                        patch.object(cli, "collect_performance", fake_performance):
                    self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1",
                                               "--pd-probe", probe_flag, "--profile", "--performance",
                                               "--keep-ready"]), 0)
                expected = [probe_name, "evaluation"]
                self.assertEqual(order, expected + ["profile", "performance"])
                run.assert_called_once()

    def test_split_only_passes_before_callback_without_after_callback(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root, measurements={}, pd_mode="split")
            with patch.object(cli, "run_experiment", return_value=result()) as run:
                self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test",
                                           "--run", "r1", "--pd-probe", "split"]), 0)
            self.assertTrue(callable(run.call_args.kwargs["before_evaluation"]))
            self.assertNotIn("after_evaluation", run.call_args.kwargs)

    def test_split_probe_failure_propagates_before_evaluation_and_collectors(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root, pd_mode="split")
            evaluate = Mock()
            def fake_run(root_arg, model, run_id, *, before_evaluation=None,
                         after_evaluation=None, keep_ready=False):
                before_evaluation(Path("run"), {})
                evaluate()
                after_evaluation(Path("run"), {}, result())
                return result()
            with patch.object(cli, "run_experiment", side_effect=fake_run), \
                    patch.object(cli, "pd_split_probe", side_effect=RuntimeError("KV gate failed")) as split, \
                    patch.object(cli, "collect_profile") as profile, \
                    patch.object(cli, "collect_performance") as performance:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1",
                              "--pd-probe", "split", "--profile", "--performance"])
            self.assertEqual(raised.exception.code, 2)
            split.assert_called_once()
            evaluate.assert_not_called()
            profile.assert_not_called()
            performance.assert_not_called()

    def test_collector_failure_is_nonzero(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root)

            def fake_run(root_arg, model, run_id, *, after_evaluation=None, keep_ready=False):
                after_evaluation(Path("run"), {"measurements": {}}, result())
                return result()

            with patch.object(cli, "run_experiment", side_effect=fake_run), \
                    patch.object(cli, "collect_profile", side_effect=RuntimeError("profile failed")):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--profile"])
            self.assertEqual(raised.exception.code, 2)

    def test_deferred_stage_rejected_before_run(self):
        for flag, stage in (("--profile", "profile"), ("--performance", "performance")):
            with self.subTest(stage=stage), self.project() as raw:
                root = Path(raw)
                make_run(root, deferred=(stage,))
                with patch.object(cli, "run_experiment") as run:
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", flag])
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_missing_measurements_rejected_before_run(self):
        with self.project() as raw:
            root = Path(raw)
            run_path = make_run(root)
            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            config.pop("measurements")
            (run_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch.object(cli, "run_experiment") as run:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--profile"])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_performance_shape_and_counts_rejected_before_run(self):
        cases = [
            {**VALID_MEASUREMENTS, "workloads": []},
            {**VALID_MEASUREMENTS, "workloads": [{"input": 0, "output": 64, "concurrency": 1}]},
            {**VALID_MEASUREMENTS, "requests": 0},
            {**VALID_MEASUREMENTS, "warmups": -1},
        ]
        for measurements in cases:
            with self.subTest(measurements=measurements), self.project() as raw:
                root = Path(raw)
                make_run(root, measurements=measurements)
                with patch.object(cli, "run_experiment") as run:
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--performance"])
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_profile_requires_server_enablement(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root, env={}, argv=["python", "serve.py"])
            with patch.object(cli, "run_experiment") as run:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--profile"])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_profile_config_flag_forms_are_accepted(self):
        for argv in (("python", "serve.py", "--profiler-config", "profile.json"),
                     ("python", "serve.py", "--profiler-config=profile.json")):
            with self.subTest(argv=argv), self.project() as raw:
                root = Path(raw)
                make_run(root, env={}, argv=list(argv))
                with patch.object(cli, "run_experiment", return_value=result()) as run:
                    self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--profile"]), 0)
                run.assert_called_once()

    def test_workload_equal_to_context_limit_is_accepted(self):
        measurements = {**VALID_MEASUREMENTS,
                        "profile_workload": {"input": 96, "output": 32, "concurrency": 1},
                        "workloads": [{"input": 80, "output": 48, "concurrency": 1}]}
        with self.project() as raw:
            root = Path(raw)
            make_run(root, measurements=measurements,
                     argv=["python", "serve.py", "--max-model-len", "128"],
                     evaluation={"context_limit": 64})
            with patch.object(cli, "run_experiment", return_value=result()) as run:
                self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1",
                                           "--profile", "--performance"]), 0)
            run.assert_called_once()

    def test_overlong_profile_and_performance_workloads_are_rejected_by_group(self):
        cases = [
            ("--profile", {"profile_workload": {"input": 97, "output": 32, "concurrency": 1}},
             "measurements.profile_workload"),
            ("--performance", {"workloads": [{"input": 96, "output": 33, "concurrency": 1}]},
             "measurements.workloads[0]"),
        ]
        for flag, changes, group in cases:
            with self.subTest(flag=flag), self.project() as raw:
                root = Path(raw)
                measurements = {**VALID_MEASUREMENTS, **changes}
                make_run(root, measurements=measurements,
                         argv=["python", "serve.py", "--max-model-len=128"],
                         evaluation={"context_limit": 256})
                with patch.object(cli, "run_experiment") as run, redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", flag])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(group, stderr.getvalue())
                self.assertIn("service.argv", stderr.getvalue())
                run.assert_not_called()

    def test_both_max_model_len_argv_forms_are_enforced(self):
        for argv in (["python", "serve.py", "--max-model-len", "128"],
                     ["python", "serve.py", "--max-model-len=128"]):
            with self.subTest(argv=argv), self.project() as raw:
                root = Path(raw)
                make_run(root,
                         measurements={**VALID_MEASUREMENTS,
                                       "workloads": [{"input": 128, "output": 1, "concurrency": 1}]},
                         argv=argv)
                with patch.object(cli, "run_experiment") as run:
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--performance"])
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_evaluation_context_limit_is_used_without_service_limit(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root,
                     measurements={**VALID_MEASUREMENTS,
                                   "workloads": [{"input": 128, "output": 1, "concurrency": 1}]},
                     evaluation={"context_limit": 128})
            with patch.object(cli, "run_experiment") as run, redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1", "--performance"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("evaluation.context_limit", stderr.getvalue())
            run.assert_not_called()

    def test_missing_context_limit_keeps_existing_measurement_semantics(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root,
                     measurements={**VALID_MEASUREMENTS,
                                   "profile_workload": {"input": 10000, "output": 10000, "concurrency": 1},
                                   "workloads": [{"input": 10000, "output": 10000, "concurrency": 1}]},
                     evaluation={})
            with patch.object(cli, "run_experiment", return_value=result()) as run:
                self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1",
                                           "--profile", "--performance"]), 0)
            run.assert_called_once()

    def test_frozen_required_collectors_run_without_measurement_flags(self):
        with self.project() as raw:
            root = Path(raw)
            make_run(root, required=("profile", "performance"))
            order = []

            def fake_run(root_arg, model, run_id, *, after_evaluation=None, keep_ready=False):
                order.append("evaluation")
                self.assertTrue(callable(after_evaluation))
                after_evaluation(Path("run"), {}, result())
                return result()

            with patch.object(cli, "run_experiment", side_effect=fake_run), \
                    patch.object(cli, "collect_profile", side_effect=lambda *args: order.append("profile")), \
                    patch.object(cli, "collect_performance", side_effect=lambda *args: order.append("performance")):
                self.assertEqual(cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1"]), 0)
            self.assertEqual(order, ["evaluation", "profile", "performance"])

    def test_legacy_config_is_rejected_before_run(self):
        with self.project() as raw:
            root = Path(raw)
            directory = make_run(root)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            del config["acceptance_contract"]
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch.object(cli, "run_experiment") as run, redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--root", str(root), "run", "--model", "test", "--run", "r1"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("legacy", stderr.getvalue())
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
