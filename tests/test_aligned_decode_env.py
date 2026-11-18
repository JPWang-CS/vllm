import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ops/run_pangu92b.py"
SPEC = importlib.util.spec_from_file_location("run_pangu92b_aligned_decode", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_explicit_disable_wins_over_reference_and_inherited_values():
    env = {"OMNI_HYBRID_ALIGNED_DECODE": "1"}
    RUNNER.apply_reference_env_overrides(env, {"OMNI_HYBRID_ALIGNED_DECODE": "1"})
    effective = RUNNER.apply_aligned_decode_plan(env, disable=True)
    assert effective == "0"
    assert env["OMNI_HYBRID_ALIGNED_DECODE"] == "0"
    RUNNER.assert_aligned_decode_plan(env, {"OMNI_HYBRID_ALIGNED_DECODE": "0"}, effective)


def test_default_removes_reference_benchmark_override_and_records_runtime_default():
    env = {"OMNI_HYBRID_ALIGNED_DECODE": "1"}
    RUNNER.apply_reference_env_overrides(env, {"OMNI_HYBRID_ALIGNED_DECODE": "1"})
    effective = RUNNER.apply_aligned_decode_plan(env)
    assert effective is None
    assert "OMNI_HYBRID_ALIGNED_DECODE" not in env
    RUNNER.assert_aligned_decode_plan(env, {}, effective)


def test_explicit_enable_and_frozen_environment_assertion():
    env = {"OMNI_HYBRID_ALIGNED_DECODE": "0"}
    RUNNER.apply_reference_env_overrides(env, {"OMNI_HYBRID_ALIGNED_DECODE": "0"})
    effective = RUNNER.apply_aligned_decode_plan(env, enable=True)
    assert effective == "1"
    RUNNER.assert_aligned_decode_plan(env, {"OMNI_HYBRID_ALIGNED_DECODE": "1"}, effective)
    with pytest.raises(AssertionError, match="differs"):
        RUNNER.assert_aligned_decode_plan(env, {"OMNI_HYBRID_ALIGNED_DECODE": "0"}, effective)


def test_enable_and_disable_are_mutually_exclusive():
    with pytest.raises(ValueError, match="both enabled and disabled"):
        RUNNER.apply_aligned_decode_plan({}, disable=True, enable=True)
