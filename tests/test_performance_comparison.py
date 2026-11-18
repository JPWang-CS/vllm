import json

import pytest

from fullnet.analysis import comparison


WORKLOAD = {"input": 128, "output": 32, "concurrency": 1}


def make_run(path, *, metrics=None, completed=4, workload=WORKLOAD, profile_workload=None):
    path.mkdir(parents=True)
    samples = [{"id": "sample-1", "prompt": "x", "outputs": ["ok"]}]
    config = {
        "model": "fixture", "platform": "a5", "devices": [0], "identity": {"dtype": "bf16"},
        "evaluation": {"scope": "fixed", "metric": "exact_match", "dataset": str(path / "dataset.jsonl")},
        "service": {"served_model": "fixture", "argv": ["python", "serve.py", "--port", "10001"]},
        "measurements": {"requests": 4, "warmups": 1, "workloads": [WORKLOAD],
                          "profile_workload": profile_workload or {"input": 4096}},
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    (path / "raw").mkdir()
    pred = {"id": "sample-1", "sample": samples[0], "success": True, "response": "ok"}
    (path / "raw/predictions.jsonl").write_text(json.dumps(pred) + "\n")
    (path / "state.json").write_text(json.dumps({"phases": {"performance": {"status": "passed"}}}))
    if metrics is not None:
        summary_workload = WORKLOAD if workload is None else workload
        summary = [{"workload": summary_workload, "completed": completed, **metrics}]
        (path / "performance").mkdir()
        (path / "performance/summary.json").write_text(json.dumps(summary, allow_nan=True))
    return path


def test_summary_comparison_uses_explicit_ratio_directions_and_keeps_numeric_unverified(tmp_path):
    reference = make_run(tmp_path / "reference", metrics={
        "output_throughput": 100.0, "mean_ttft_ms": 20.0, "mean_tpot_ms": 4.0,
    }, profile_workload={"input": 8192})
    candidate = make_run(tmp_path / "candidate", metrics={
        "output_throughput": 250.0, "mean_ttft_ms": 10.0, "mean_tpot_ms": 2.0,
    }, profile_workload={"input": 16384})

    result = comparison(reference, candidate)
    assert result["performance"] == "compared"
    assert result["numerical_parity"] == "not_run"
    workload = result["performance_comparison"]["workloads"][0]
    assert workload["completed_samples"] == {"reference": 4, "candidate": 4}
    assert workload["metrics"]["output_throughput"]["ratio_direction"] == "candidate/reference"
    assert workload["metrics"]["output_throughput"]["ratio"] == 2.5
    assert workload["metrics"]["mean_ttft_ms"]["ratio_direction"] == "reference/candidate"
    assert workload["metrics"]["mean_ttft_ms"]["ratio"] == 2.0
    assert workload["metrics"]["mean_tpot_ms"]["reference_value"] == 4.0


def test_absent_summaries_remain_not_measured(tmp_path):
    reference = make_run(tmp_path / "reference")
    candidate = make_run(tmp_path / "candidate")
    result = comparison(reference, candidate)
    assert result["performance"] == "not_measured"
    assert result["performance_comparison"]["status"] == "not_measured"
    assert result["performance_comparison"]["workloads"] == []
    assert result["numerical_parity"] == "not_run"


@pytest.mark.parametrize("completed,workload,reason", [
    (3, WORKLOAD, "completed sample count"),
    (4, {"input": 256, "output": 32, "concurrency": 1}, "does not match the contract"),
])
def test_incomplete_or_different_workloads_are_not_comparable(tmp_path, completed, workload, reason):
    reference = make_run(tmp_path / "reference", metrics={"output_throughput": 100.0})
    candidate = make_run(tmp_path / "candidate", metrics={"output_throughput": 120.0},
                         completed=completed, workload=workload)
    result = comparison(reference, candidate)
    assert result["performance"] == "not_comparable"
    assert reason in result["performance_comparison"]["reason"]


def test_missing_zero_and_nonfinite_metrics_have_no_ratio(tmp_path):
    reference = make_run(tmp_path / "reference", metrics={
        "output_throughput": 100.0, "request_throughput": 1e-308,
        "mean_ttft_ms": 0.0, "mean_tpot_ms": float("nan"),
    })
    candidate = make_run(tmp_path / "candidate", metrics={
        "output_throughput": 0.0, "request_throughput": 1e308,
        "mean_ttft_ms": 10.0, "mean_tpot_ms": 5.0,
    })
    result = comparison(reference, candidate)
    metrics = result["performance_comparison"]["workloads"][0]["metrics"]
    assert result["performance"] == "compared"
    assert metrics["output_throughput"]["ratio"] is None
    assert metrics["output_throughput"]["reason"] == "zero_or_non_finite_or_non_positive_metric"
    assert metrics["mean_ttft_ms"]["ratio"] is None
    assert metrics["mean_tpot_ms"]["ratio"] is None
    assert metrics["mean_tpot_ms"]["reason"] == "zero_or_non_finite_or_non_positive_metric"
    assert metrics["request_throughput"]["ratio"] is None
    assert metrics["request_throughput"]["reason"] == "non_finite_ratio"
    assert metrics["p95_ttft_ms"]["ratio"] is None
    assert metrics["p95_ttft_ms"]["reason"] == "missing_metric"
