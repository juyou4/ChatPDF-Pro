"""检查 RAGAS 结果是否满足发布/回归门槛。

该脚本只读取评测 JSON，不发起网络请求。联网评测由手动或 nightly 工作流
负责；脚本严格区分“未评估”“NaN”“真实低分”，避免把固定测试通过误报成
概率校准。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[1] / "benchmarks" / "ragas_gate.v1.json"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ratio(value: Any, denominator: Any) -> float:
    number = _number(value)
    total = _number(denominator)
    if number is None or total is None or total <= 0:
        return 0.0
    return number / total


def evaluate_gate(result: dict[str, Any], thresholds: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回结构化检查结果；不直接退出，便于单测和 CI 展示。"""
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    scores = result.get("ragas_scores") if isinstance(result.get("ragas_scores"), dict) else {}
    stats = summary.get("ragas_metric_stats") if isinstance(summary.get("ragas_metric_stats"), dict) else {}
    sample_policy = thresholds.get("sample_policy") if isinstance(thresholds.get("sample_policy"), dict) else {}

    total_samples = int(_number(summary.get("total_samples")) or 0)
    error_samples = int(_number(summary.get("error_samples")) or 0)
    ragas_samples = summary.get("ragas_eval_samples") if isinstance(summary.get("ragas_eval_samples"), dict) else {}
    evaluated_samples = int(_number(ragas_samples.get("ragas_evaluated_samples")) or 0)

    min_total = int(_number(sample_policy.get("min_total_samples")) or 0)
    min_evaluated = int(_number(sample_policy.get("min_ragas_evaluated_samples")) or 0)
    max_error_ratio = _number(sample_policy.get("max_error_ratio"))
    if total_samples < min_total:
        failures.append({"kind": "sample_count", "field": "total_samples", "actual": total_samples, "minimum": min_total})
    if evaluated_samples < min_evaluated:
        failures.append({"kind": "sample_count", "field": "ragas_evaluated_samples", "actual": evaluated_samples, "minimum": min_evaluated})
    if max_error_ratio is not None and _ratio(error_samples, total_samples) > max_error_ratio:
        failures.append({
            "kind": "error_ratio",
            "actual": _ratio(error_samples, total_samples),
            "maximum": max_error_ratio,
        })

    required_metrics = thresholds.get("required_metrics") if isinstance(thresholds.get("required_metrics"), dict) else {}
    baseline_scores = baseline.get("ragas_scores") if isinstance(baseline, dict) and isinstance(baseline.get("ragas_scores"), dict) else {}
    allowed_regression = thresholds.get("allowed_regression") if isinstance(thresholds.get("allowed_regression"), dict) else {}

    metric_results: dict[str, dict[str, Any]] = {}
    for metric, policy in required_metrics.items():
        policy = policy if isinstance(policy, dict) else {}
        score = _number(scores.get(metric))
        metric_stat = stats.get(metric) if isinstance(stats.get(metric), dict) else {}
        valid_count = int(_number(metric_stat.get("valid_count")) or 0)
        total_count = int(_number(metric_stat.get("total_count")) or 0)
        nan_count = int(_number(metric_stat.get("nan_count")) or 0)
        item = {
            "score": score,
            "valid_count": valid_count,
            "total_count": total_count,
            "nan_count": nan_count,
            "status": "pass",
        }
        if score is None:
            item["status"] = "missing_or_non_finite"
            failures.append({"kind": "metric_missing", "metric": metric})
        min_valid = int(_number(policy.get("min_valid_count")) or 0)
        if valid_count < min_valid:
            item["status"] = "insufficient_valid_samples"
            failures.append({"kind": "metric_valid_count", "metric": metric, "actual": valid_count, "minimum": min_valid})
        max_nan_ratio = _number(policy.get("max_nan_ratio"))
        if max_nan_ratio is None:
            max_nan_ratio = _number(sample_policy.get("max_nan_ratio"))
        if max_nan_ratio is not None and _ratio(nan_count, total_count) > max_nan_ratio:
            item["status"] = "nan_ratio"
            failures.append({"kind": "metric_nan_ratio", "metric": metric, "actual": _ratio(nan_count, total_count), "maximum": max_nan_ratio})
        minimum = _number(policy.get("min"))
        if score is not None and minimum is not None and score < minimum:
            item["status"] = "below_absolute_minimum"
            failures.append({"kind": "metric_minimum", "metric": metric, "actual": score, "minimum": minimum})
        baseline_score = _number(baseline_scores.get(metric))
        delta = _number(allowed_regression.get(metric))
        if score is not None and baseline_score is not None and delta is not None and score < baseline_score - delta:
            item["status"] = "baseline_regression"
            failures.append({
                "kind": "baseline_regression",
                "metric": metric,
                "actual": score,
                "baseline": baseline_score,
                "allowed_regression": delta,
            })
        metric_results[metric] = item

    optional_metrics = thresholds.get("optional_metrics") if isinstance(thresholds.get("optional_metrics"), list) else []
    for metric in optional_metrics:
        if metric not in scores:
            warnings.append({"kind": "optional_metric_missing", "metric": str(metric)})

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "samples": {
            "total": total_samples,
            "errors": error_samples,
            "ragas_evaluated": evaluated_samples,
        },
        "metrics": metric_results,
        "gate_version": str(thresholds.get("version") or "ragas-gate-v1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 ChatPDF RAGAS 结果门槛")
    parser.add_argument("--results", required=True, help="eval_ragas 输出的 JSON")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="门槛 JSON")
    parser.add_argument("--baseline", default="", help="可选历史结果 JSON，用于检查允许回归幅度")
    args = parser.parse_args()

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    report = evaluate_gate(result, thresholds, baseline)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
