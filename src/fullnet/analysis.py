"""Compare equivalent experiments and generate compact evidence reports."""
from pathlib import Path
import math
import numbers

from .store import read_json, rows, write_json


_PERFORMANCE_METRICS = {
    "output_throughput": ("throughput", "tokens/s"),
    "request_throughput": ("throughput", "requests/s"),
    "mean_ttft_ms": ("latency", "ms"),
    "p95_ttft_ms": ("latency", "ms"),
    "mean_tpot_ms": ("latency", "ms"),
    "p95_tpot_ms": ("latency", "ms"),
    "mean_e2el_ms": ("latency", "ms"),
}


def _performance_comparison(reference_dir, candidate_dir, reference_config, candidate_config):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    ref_path = reference_dir / "performance" / "summary.json"
    cand_path = candidate_dir / "performance" / "summary.json"
    result = {"reference_summary": str(ref_path), "candidate_summary": str(cand_path), "workloads": []}
    ref_exists, cand_exists = ref_path.is_file(), cand_path.is_file()
    if not ref_exists and not cand_exists:
        return "not_measured", {**result, "status": "not_measured", "reason": "both performance summaries are absent"}
    if not ref_exists or not cand_exists:
        return "not_comparable", {**result, "status": "not_comparable", "reason": "performance summary exists for only one run"}

    ref_measure = reference_config.get("measurements")
    cand_measure = candidate_config.get("measurements")
    if not isinstance(ref_measure, dict) or not isinstance(cand_measure, dict):
        return "not_comparable", {**result, "status": "not_comparable", "reason": "measurement configuration is missing"}
    # Profile-only settings do not change the official benchmark workload.
    profile_keys = {"profile", "profile_workload"}
    ref_bench = {k: v for k, v in ref_measure.items() if k not in profile_keys | {"workloads"}}
    cand_bench = {k: v for k, v in cand_measure.items() if k not in profile_keys | {"workloads"}}
    if ref_bench != cand_bench:
        return "not_comparable", {**result, "status": "not_comparable", "reason": "benchmark measurement parameters differ"}
    if ref_measure.get("workloads") != cand_measure.get("workloads"):
        return "not_comparable", {**result, "status": "not_comparable", "reason": "workload contracts differ"}
    requests = ref_measure.get("requests")
    workloads = ref_measure.get("workloads")
    if type(requests) is not int or requests <= 0 or not isinstance(workloads, list) or not workloads:
        return "not_comparable", {**result, "status": "not_comparable", "reason": "requests or workload contract is invalid"}

    try:
        ref_state = read_json(reference_dir / "state.json").get("phases", {}).get("performance", {}).get("status")
        cand_state = read_json(candidate_dir / "state.json").get("phases", {}).get("performance", {}).get("status")
        ref_summary, cand_summary = read_json(ref_path), read_json(cand_path)
    except (OSError, ValueError, AttributeError) as exc:
        return "not_comparable", {**result, "status": "not_comparable", "reason": f"performance evidence cannot be read: {type(exc).__name__}"}
    if ref_state != "passed" or cand_state != "passed":
        return "not_comparable", {**result, "status": "not_comparable", "reason": "both performance phases must be passed"}
    if not isinstance(ref_summary, list) or not isinstance(cand_summary, list):
        return "not_comparable", {**result, "status": "not_comparable", "reason": "summary must be a workload list"}
    if len(ref_summary) != len(workloads) or len(cand_summary) != len(workloads):
        return "not_comparable", {**result, "status": "not_comparable", "reason": "summary workload count differs from the contract"}

    comparisons = []
    for index, (workload, ref_row, cand_row) in enumerate(zip(workloads, ref_summary, cand_summary)):
        if not isinstance(ref_row, dict) or not isinstance(cand_row, dict):
            return "not_comparable", {**result, "status": "not_comparable", "reason": f"summary row {index} is invalid"}
        if ref_row.get("workload") != workload or cand_row.get("workload") != workload:
            return "not_comparable", {**result, "status": "not_comparable", "reason": f"workload {index} does not match the contract"}
        ref_completed, cand_completed = ref_row.get("completed"), cand_row.get("completed")
        if (type(ref_completed) is not int or type(cand_completed) is not int or
                ref_completed != requests or cand_completed != requests or ref_completed != cand_completed):
            return "not_comparable", {**result, "status": "not_comparable", "reason": f"completed sample count differs at workload {index}"}
        metrics = {}
        for name, (kind, unit) in _PERFORMANCE_METRICS.items():
            ref_value, cand_value = ref_row.get(name), cand_row.get(name)
            values = {"throughput": (ref_value, cand_value, "candidate/reference"),
                      "latency": (ref_value, cand_value, "reference/candidate")}[kind]
            left, right, direction = values
            metric = {"unit": unit, "reference_value": left, "candidate_value": right,
                      "ratio_direction": direction, "ratio": None}
            valid = lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x)) and x > 0
            if left is None or right is None:
                metric.update(ratio_status="not_calculable", reason="missing_metric")
            elif not valid(left) or not valid(right):
                metric.update(ratio_status="not_calculable", reason="zero_or_non_finite_or_non_positive_metric")
                if isinstance(left, float) and not math.isfinite(left): metric["reference_value"] = repr(left)
                if isinstance(right, float) and not math.isfinite(right): metric["candidate_value"] = repr(right)
            else:
                ratio = float(right) / float(left) if kind == "throughput" else float(left) / float(right)
                if math.isfinite(ratio):
                    metric.update(ratio_status="calculated", ratio=ratio)
                else:
                    metric.update(ratio_status="not_calculable", reason="non_finite_ratio")
            metrics[name] = metric
        comparisons.append({"workload_index": index, "workload": workload,
                            "completed_samples": {"reference": ref_completed, "candidate": cand_completed},
                            "metrics": metrics})
    return "compared", {**result, "status": "compared", "reason": None, "workloads": comparisons}


def comparison(left, right):
    a, b = (read_json(Path(p) / "config.json") for p in (left, right))
    # Change names/activation env differ by design; numerical workload and hardware do not.
    for key in ("model", "platform", "devices", "identity", "evaluation"):
        av, bv = a.get(key), b.get(key)
        if key == "evaluation":
            av, bv = dict(av), dict(bv)
            av.pop("dataset", None)
            bv.pop("dataset", None)
        if av != bv:
            raise ValueError(f"comparison condition differs: {key}")
    for key in ("served_model",):
        if a["service"][key] != b["service"][key]:
            raise ValueError(f"service condition differs: {key}")
    def workload_argv(config):
        args = list(config["service"]["argv"])
        # A service port is operational; all other launch differences need a separate analysis.
        if "--port" in args:
            args[args.index("--port") + 1] = "<port>"
        return args
    if workload_argv(a) != workload_argv(b):
        raise ValueError("launch arguments differ; describe the system configuration comparison separately")
    if rows(Path(left) / "dataset.jsonl") != rows(Path(right) / "dataset.jsonl"):
        raise ValueError("frozen inputs differ")
    pa, pb = (rows(Path(p) / "raw/predictions.jsonl") for p in (left, right))
    da, db = ({v["id"]: v for v in predictions} for predictions in (pa, pb))
    ids = [v["id"] for v in rows(Path(left) / "dataset.jsonl")]
    complete = set(da) == set(db) == set(ids)
    mismatches = [i for i in ids if not da.get(i, {}).get("success") or not db.get(i, {}).get("success")
                  or da[i]["response"] != db[i]["response"]]
    performance, performance_details = _performance_comparison(left, right, a, b)
    return {"left": str(left), "right": str(right), "samples": len(ids), "complete": complete,
            "text_mismatches": mismatches, "text_match": complete and not mismatches,
            "numerical_parity": "not_run", "performance": performance,
            "performance_comparison": performance_details}


def numeric_parity(reference, candidate, atol, rtol):
    """Compare exported stages; no PyTorch dependency in the orchestration layer."""
    import math
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be nonnegative")
    a, b = read_json(reference), read_json(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("numerical stages differ")
    results = []
    for stage in a:
        x, y = a[stage], b[stage]
        if x.get("shape") != y.get("shape") or len(x["values"]) != len(y["values"]):
            raise ValueError(f"shape differs: {stage}")
        if math.prod(x["shape"]) != len(x["values"]) or not x["values"]:
            raise ValueError(f"invalid exported tensor: {stage}")
        checks = [math.isfinite(u) and math.isfinite(v) and abs(v-u) <= atol + rtol*abs(u)
                  for u, v in zip(x["values"], y["values"])]
        results.append({"stage": stage, "passed": all(checks), "elements": len(checks),
                        "mismatches": sum(not check for check in checks)})
    return {"passed": all(v["passed"] for v in results), "atol": atol, "rtol": rtol, "stages": results}


def report(run):
    run = Path(run)
    c, state = read_json(run / "config.json"), read_json(run / "state.json")
    result = read_json(run / "results.json") if (run / "results.json").exists() else {}
    acceptance = read_json(run / "acceptance.json") if (run / "acceptance.json").exists() else {}
    lines = [f"# {c['model']} / {run.name}", "", f"平台：{c['platform']}。设备：{c.get('devices', [])}。",
             "", "| 检查 | 结果 |", "|---|---|"]
    for phase, value in state["phases"].items():
        lines.append(f"| {phase} | {value['status']} |")
    lines.extend([f"| 任务精度 | {result.get('quality_status', 'not_run')} |",
                  f"| 数值一致性 | {result.get('numerical_parity', 'not_run')} |",
                  f"| execution_completed | {acceptance.get('execution_completed', 'legacy_read_only')} |",
                  f"| business_acceptance | {acceptance.get('business_acceptance', 'LEGACY_READ_ONLY')} |",
                  ""])
    if acceptance.get("reasons"):
        lines += ["验收阻断：" + "；".join(str(v) for v in acceptance["reasons"]), ""]
    if result:
        score = result.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        lines += [f"评测范围：{result['scope']}。按固定样本数加权得分：{score_text}。", "",
                  "| 任务 | 上下文长度 | 样本数 | 得分 | 请求失败 | 截断 |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {v['task']} | {v['context_limit']} | {v['samples']} | {v['score']:.3f} | {v['protocol_failures']} | {v['truncated']} |"
                  for v in result['groups']]
    lines += ["", "配置、原始请求与响应保存在本实验目录。性能需使用固定负载另行采集。", ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return run / "report.md"


def status(run):
    """Read-only status view; legacy runs are never upgraded or rewritten."""
    run = Path(run)
    value = read_json(run / "state.json")
    acceptance_path = run / "acceptance.json"
    if acceptance_path.is_file():
        acceptance = read_json(acceptance_path)
        value = {**value, "execution_completed": acceptance.get("execution_completed"),
                 "business_acceptance": acceptance.get("business_acceptance"),
                 "acceptance_reasons": acceptance.get("reasons", []),
                 "acceptance_contract_version": acceptance.get("contract", {}).get("version")}
    else:
        value = {**value, "execution_completed": None,
                 "business_acceptance": "LEGACY_READ_ONLY",
                 "acceptance_reasons": ["legacy experiment has no frozen acceptance contract"]}
    return value
