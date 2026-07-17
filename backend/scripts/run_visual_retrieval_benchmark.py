"""Offline benchmark for the default visual-retriever contract.

The benchmark deliberately executes the production ID-only retriever contract
without invoking a visual model. Returned IDs are hydrated exclusively from the
current canonical modal-asset index. Gold annotations use stable document
coordinates (page/kind/figure_id/block_id/bbox) instead of the
parse-identity-derived ``asset_id``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.visual_retriever import (  # noqa: E402
    DEFAULT_VISUAL_RETRIEVER,
    VisualRetrieverRequest,
    deterministic_ranked_assets,
    execute_visual_retriever,
)


SCHEMA_VERSION = 1
CUTOFFS = (1, 3, 5)
_KIND_ALIASES = {
    "fig": "figure",
    "figure": "figure",
    "image": "figure",
    "picture": "figure",
    "chart": "figure",
    "table": "table",
    "formula": "formula",
    "equation": "formula",
    "eq": "formula",
    "visual": "visual_enrichment",
    "visual_enrichment": "visual_enrichment",
}
_IDENTITY_KEYS = (
    "version",
    "route",
    "generation",
    "source_hash",
    "revision",
    "index_id",
)


class BenchmarkError(ValueError):
    """Raised when a visual retrieval benchmark manifest is malformed."""


@dataclass(frozen=True)
class Threshold:
    metric: str
    operator: str
    value: float
    source: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"找不到 JSON 文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"JSON 无法解析：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"JSON 根节点必须为对象：{path}")
    return value


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _unwrap_index(value: dict[str, Any]) -> dict[str, Any]:
    """Accept a raw modal index or a small wrapper used by exported fixtures."""
    nested = value.get("modal_asset_index") or value.get("index")
    if isinstance(nested, dict) and isinstance(nested.get("assets"), list):
        return nested
    return value


def _load_index(
    holder: dict[str, Any],
    *,
    base_dir: Path,
    label: str,
) -> dict[str, Any]:
    inline = holder.get("index")
    index_path = holder.get("index_path")
    if isinstance(inline, dict) and isinstance(index_path, str) and index_path.strip():
        raise BenchmarkError(f"{label} 同时设置了 index 和 index_path")
    if isinstance(inline, dict):
        index = _unwrap_index(inline)
    elif isinstance(index_path, str) and index_path.strip():
        index = _unwrap_index(_read_json(_resolve_path(index_path, base_dir=base_dir)))
    else:
        raise BenchmarkError(f"{label} 缺少 index 或 index_path")
    assets = index.get("assets")
    if not isinstance(assets, list):
        raise BenchmarkError(f"{label} 的 index.assets 必须为数组")
    return index


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _kind(value: Any) -> str:
    raw = _clean_text(value).lower()
    return _KIND_ALIASES.get(raw, raw)


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def bbox_iou(left: Any, right: Any) -> float:
    """Return standard IoU for two [x0, y0, x1, y1] boxes, or zero when invalid."""
    first = _bbox(left)
    second = _bbox(right)
    if first is None or second is None:
        return 0.0
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    if intersection <= 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _validate_gold(target: Any, *, case_id: str, position: int) -> dict[str, Any]:
    if not isinstance(target, dict):
        raise BenchmarkError(f"{case_id}.gold[{position}] 必须为对象")
    page = _positive_int(target.get("page"))
    kind = _kind(target.get("kind"))
    if not page or not kind:
        raise BenchmarkError(f"{case_id}.gold[{position}] 必须包含正数 page 和 kind")
    target_bbox = _bbox(target.get("bbox")) if "bbox" in target else None
    if "bbox" in target and target_bbox is None:
        raise BenchmarkError(f"{case_id}.gold[{position}].bbox 必须是有效区域")
    figure_id = _clean_text(target.get("figure_id"))
    block_id = _clean_text(target.get("block_id"))
    if not (figure_id or block_id or target_bbox):
        raise BenchmarkError(
            f"{case_id}.gold[{position}] 至少应包含 figure_id、block_id 或 bbox 之一"
        )
    try:
        relevance = float(target.get("relevance", 1.0))
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{case_id}.gold[{position}].relevance 必须为数字") from exc
    if not math.isfinite(relevance) or relevance <= 0:
        raise BenchmarkError(f"{case_id}.gold[{position}].relevance 必须大于零")
    try:
        threshold = float(target.get("bbox_iou_threshold", 0.5))
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{case_id}.gold[{position}].bbox_iou_threshold 必须为数字") from exc
    if not 0.0 <= threshold <= 1.0:
        raise BenchmarkError(f"{case_id}.gold[{position}].bbox_iou_threshold 必须在 0 到 1 之间")
    return {
        "page": page,
        "kind": kind,
        "figure_id": figure_id,
        "block_id": block_id,
        "bbox": target_bbox,
        "bbox_iou_threshold": threshold,
        "relevance": relevance,
    }


def _identity_matches(target: dict[str, Any], asset: dict[str, Any]) -> bool:
    if _positive_int(asset.get("page")) != target["page"]:
        return False
    if _kind(asset.get("kind")) != target["kind"]:
        return False
    if target["figure_id"] and _clean_text(asset.get("figure_id")) != target["figure_id"]:
        return False
    if target["block_id"] and _clean_text(asset.get("block_id")) != target["block_id"]:
        return False
    return True


def _match(target: dict[str, Any], asset: dict[str, Any]) -> tuple[bool, float]:
    if not _identity_matches(target, asset):
        return False, 0.0
    iou = bbox_iou(target["bbox"], asset.get("bbox")) if target["bbox"] else 0.0
    if target["bbox"] and iou < target["bbox_iou_threshold"]:
        return False, iou
    return True, iou


def _page_hit(target: dict[str, Any], results: Iterable[dict[str, Any]]) -> bool:
    return any(_positive_int(asset.get("page")) == target["page"] for asset in results)


def _bbox_hit(target: dict[str, Any], results: Iterable[dict[str, Any]]) -> tuple[bool, float]:
    if not target["bbox"]:
        return False, 0.0
    best_iou = 0.0
    for asset in results:
        if not _identity_matches(target, asset):
            continue
        best_iou = max(best_iou, bbox_iou(target["bbox"], asset.get("bbox")))
    return best_iou >= target["bbox_iou_threshold"], best_iou


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def _scope_identity(index: dict[str, Any]) -> dict[str, str]:
    identity = {key: _clean_text(index.get(key)) for key in _IDENTITY_KEYS}
    identity["identity_hash"] = hashlib.sha256(_canonical_json(identity)).hexdigest()[:16]
    return identity


def _safe_result(asset: dict[str, Any], *, rank: int, gold: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [_match(target, asset) for target in gold]
    matching_targets = [position for position, (matched, _iou) in enumerate(matches) if matched]
    ious = [iou for _matched, iou in matches if iou > 0]
    return {
        "rank": rank,
        "page": _positive_int(asset.get("page")),
        "kind": _kind(asset.get("kind")),
        "figure_id": _clean_text(asset.get("figure_id")),
        "block_id": _clean_text(asset.get("block_id")),
        "bbox": _bbox(asset.get("bbox")) or [],
        "score": asset.get("score"),
        "matching_gold_positions": matching_targets,
        "best_gold_bbox_iou": round(max(ious, default=0.0), 6),
    }


def _dcg(relevances: list[float]) -> float:
    return sum(value / math.log2(position + 2) for position, value in enumerate(relevances))


def _case_index(case: dict[str, Any], *, default_index: dict[str, Any], base_dir: Path, case_id: str) -> dict[str, Any]:
    if "index" not in case and "index_path" not in case:
        return default_index
    return _load_index(case, base_dir=base_dir, label=f"case {case_id}")


def _canonical_assets_by_id(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the first canonical asset for each non-empty asset ID."""
    assets_by_id: dict[str, dict[str, Any]] = {}
    for raw_asset in index.get("assets", []):
        if not isinstance(raw_asset, dict):
            continue
        asset_id = _clean_text(raw_asset.get("asset_id"))
        if asset_id and asset_id not in assets_by_id:
            assets_by_id[asset_id] = raw_asset
    return assets_by_id


def _retrieve_assets(
    index: dict[str, Any],
    *,
    query: str,
    reference: str,
    page: int,
    kinds: list[Any] | None,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the default ID-only contract and hydrate from the current index."""
    request = VisualRetrieverRequest(
        query=query,
        reference=reference,
        page=page,
        kinds=tuple(_clean_text(kind) for kind in (kinds or []) if _clean_text(kind)),
        limit=limit,
    )
    execution = execute_visual_retriever(
        DEFAULT_VISUAL_RETRIEVER,
        request=request,
        modal_asset_index=index,
    )
    assets_by_id = _canonical_assets_by_id(index)
    trusted_scores = {
        _clean_text(asset.get("asset_id")): asset.get("score")
        for asset in deterministic_ranked_assets(
            request=request,
            modal_asset_index=index,
        )
        if isinstance(asset, dict) and _clean_text(asset.get("asset_id"))
    }
    hydrated: list[dict[str, Any]] = []
    for asset_id in execution.asset_ids:
        canonical = assets_by_id.get(asset_id)
        if canonical is None:
            continue
        asset = dict(canonical)
        asset["score"] = trusted_scores.get(asset_id, 0.0)
        hydrated.append(asset)
        if len(hydrated) >= limit:
            break
    return hydrated, execution.diagnostics()


def run_benchmark(manifest: dict[str, Any], *, manifest_path: Path | None = None) -> dict[str, Any]:
    """Execute a manifest and return a JSON-serialisable report."""
    if int(manifest.get("schema_version", 0)) != SCHEMA_VERSION:
        raise BenchmarkError(f"仅支持 schema_version={SCHEMA_VERSION}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("manifest.cases 必须是非空数组")
    base_dir = manifest_path.parent if manifest_path else Path.cwd()
    default_index = _load_index(manifest, base_dir=base_dir, label="manifest")

    positive_target_count = 0
    bbox_target_count = 0
    negative_case_count = 0
    negative_false_positives = 0
    latency_ms: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    target_hits = {cutoff: 0 for cutoff in CUTOFFS}
    page_hits = {cutoff: 0 for cutoff in CUTOFFS}
    bbox_hits = {cutoff: 0 for cutoff in CUTOFFS}
    case_reports: list[dict[str, Any]] = []
    scope_identities: dict[str, dict[str, Any]] = {}

    for position, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkError(f"cases[{position}] 必须为对象")
        case_id = _clean_text(case.get("id")) or f"case-{position + 1}"
        negative = bool(case.get("negative", False))
        raw_gold = case.get("gold", [])
        if not isinstance(raw_gold, list):
            raise BenchmarkError(f"{case_id}.gold 必须为数组")
        gold = [_validate_gold(item, case_id=case_id, position=i) for i, item in enumerate(raw_gold)]
        if negative and gold:
            raise BenchmarkError(f"负例 {case_id} 不应包含 gold")
        if not negative and not gold:
            raise BenchmarkError(f"正例 {case_id} 至少需要一个 gold；空结果请标记 negative=true")

        index = _case_index(case, default_index=default_index, base_dir=base_dir, case_id=case_id)
        identity = _scope_identity(index)
        scope_identities[identity["identity_hash"]] = identity
        raw_kinds = case.get("kinds")
        if raw_kinds is not None and not isinstance(raw_kinds, list):
            raise BenchmarkError(f"{case_id}.kinds 必须为数组")
        try:
            requested_page = max(0, int(case.get("page", 0) or 0))
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(f"{case_id}.page 必须为整数") from exc
        started = time.perf_counter()
        results, retriever_diagnostics = _retrieve_assets(
            index,
            query=_clean_text(case.get("query")),
            reference=_clean_text(case.get("reference")),
            page=requested_page,
            kinds=raw_kinds,
            limit=max(CUTOFFS),
        )
        elapsed = (time.perf_counter() - started) * 1000
        latency_ms.append(elapsed)
        if not isinstance(results, list):
            raise BenchmarkError(f"{case_id} 的检索器返回了非数组结果")

        if negative:
            negative_case_count += 1
            if results:
                negative_false_positives += 1
            case_reports.append({
                "id": case_id,
                "type": "negative",
                "scope_identity": identity,
                "retriever": retriever_diagnostics,
                "latency_ms": round(elapsed, 6),
                "result_count": len(results),
                "false_positive": bool(results),
                "results": [_safe_result(item, rank=i + 1, gold=[]) for i, item in enumerate(results)],
            })
            continue

        positive_target_count += len(gold)
        bbox_target_count += sum(1 for target in gold if target["bbox"])
        ranks_by_target: list[int | None] = []
        for target in gold:
            rank = next((rank for rank, asset in enumerate(results, start=1) if _match(target, asset)[0]), None)
            ranks_by_target.append(rank)
        first_relevant_rank = min((rank for rank in ranks_by_target if rank is not None), default=None)
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        for cutoff in CUTOFFS:
            top_results = results[:cutoff]
            target_hits[cutoff] += sum(
                any(_match(target, asset)[0] for asset in top_results) for target in gold
            )
            page_hits[cutoff] += sum(_page_hit(target, top_results) for target in gold)
            bbox_hits[cutoff] += sum(_bbox_hit(target, top_results)[0] for target in gold if target["bbox"])

        graded_relevances: list[float] = []
        for asset in results[:max(CUTOFFS)]:
            grades = [target["relevance"] for target in gold if _match(target, asset)[0]]
            graded_relevances.append(max(grades, default=0.0))
        ideal = sorted((target["relevance"] for target in gold), reverse=True)[:max(CUTOFFS)]
        ideal_dcg = _dcg(ideal)
        ndcgs.append(_dcg(graded_relevances) / ideal_dcg if ideal_dcg else 0.0)
        case_reports.append({
            "id": case_id,
            "type": "positive",
            "scope_identity": identity,
            "retriever": retriever_diagnostics,
            "latency_ms": round(elapsed, 6),
            "result_count": len(results),
            "first_relevant_rank": first_relevant_rank,
            "gold_target_count": len(gold),
            "matched_gold_positions": [i for i, rank in enumerate(ranks_by_target) if rank is not None],
            "results": [_safe_result(item, rank=i + 1, gold=gold) for i, item in enumerate(results)],
        })

    metrics: dict[str, Any] = {
        **{f"recall_at_{cutoff}": round(target_hits[cutoff] / positive_target_count, 6) if positive_target_count else 0.0 for cutoff in CUTOFFS},
        **{f"page_recall_at_{cutoff}": round(page_hits[cutoff] / positive_target_count, 6) if positive_target_count else 0.0 for cutoff in CUTOFFS},
        **{f"bbox_iou_recall_at_{cutoff}": round(bbox_hits[cutoff] / bbox_target_count, 6) if bbox_target_count else 0.0 for cutoff in CUTOFFS},
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6) if reciprocal_ranks else 0.0,
        "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 6) if ndcgs else 0.0,
        "negative_fpr": round(negative_false_positives / negative_case_count, 6) if negative_case_count else 0.0,
        "latency_ms": {
            "p50": _percentile(latency_ms, 0.5),
            "p95": _percentile(latency_ms, 0.95),
            "mean": round(sum(latency_ms) / len(latency_ms), 6) if latency_ms else 0.0,
        },
    }
    return {
        "benchmark": "chatpdf.visual_retrieval",
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": _manifest_hash(manifest),
        "manifest_path": str(manifest_path.resolve()) if manifest_path else "",
        "scope": manifest.get("scope") if isinstance(manifest.get("scope"), dict) else {},
        "scope_identities": list(scope_identities.values()),
        "counts": {
            "case_count": len(cases),
            "positive_target_count": positive_target_count,
            "bbox_target_count": bbox_target_count,
            "negative_case_count": negative_case_count,
            "negative_false_positive_count": negative_false_positives,
        },
        "metrics": metrics,
        "cases": case_reports,
    }


def _thresholds_from_value(value: Any, *, source: str) -> list[Threshold]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise BenchmarkError(f"{source} 必须为对象")
    parsed: list[Threshold] = []
    for raw_name, raw_rule in value.items():
        name = _clean_text(raw_name)
        if not name:
            raise BenchmarkError(f"{source} 含有空指标名")
        if isinstance(raw_rule, dict):
            for key, operator in (("min", ">="), ("max", "<=")):
                if key in raw_rule:
                    try:
                        parsed.append(Threshold(name, operator, float(raw_rule[key]), source))
                    except (TypeError, ValueError) as exc:
                        raise BenchmarkError(f"{source}.{name}.{key} 必须为数字") from exc
            continue
        try:
            numeric = float(raw_rule)
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(f"{source}.{name} 必须为数字或 {{min,max}}") from exc
        if name.startswith("min_"):
            parsed.append(Threshold(name[4:], ">=", numeric, source))
        elif name.startswith("max_"):
            parsed.append(Threshold(name[4:], "<=", numeric, source))
        else:
            parsed.append(Threshold(name, ">=", numeric, source))
    return parsed


def _parse_cli_threshold(value: str) -> Threshold:
    for operator in (">=", "<="):
        if operator in value:
            metric, raw_limit = value.split(operator, 1)
            try:
                return Threshold(metric.strip(), operator, float(raw_limit.strip()), "command_line")
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"阈值必须是 metric>=value 或 metric<=value：{value}") from exc
    raise argparse.ArgumentTypeError(f"阈值必须是 metric>=value 或 metric<=value：{value}")


def _metric_value(report: dict[str, Any], metric: str) -> float:
    value: Any = report.get("metrics", {}).get(metric)
    if value is None and metric.startswith("latency_ms."):
        value = report.get("metrics", {}).get("latency_ms", {}).get(metric.split(".", 1)[1])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BenchmarkError(f"阈值引用了不存在或非数值的指标：{metric}")
    return float(value)


def evaluate_thresholds(report: dict[str, Any], thresholds: list[Threshold]) -> list[dict[str, Any]]:
    evaluations = []
    for threshold in thresholds:
        actual = _metric_value(report, threshold.metric)
        passed = actual >= threshold.value if threshold.operator == ">=" else actual <= threshold.value
        evaluations.append({
            "metric": threshold.metric,
            "operator": threshold.operator,
            "expected": threshold.value,
            "actual": actual,
            "source": threshold.source,
            "passed": passed,
        })
    return evaluations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线评测 ChatPDF 模态资产检索")
    parser.add_argument(
        "--manifest",
        default=str(BACKEND_DIR / "benchmarks" / "visual_retrieval_manifest.v1.json"),
        help="评测清单 JSON 路径",
    )
    parser.add_argument("--output", help="可选的 JSON 报告输出路径")
    parser.add_argument(
        "--threshold",
        action="append",
        type=_parse_cli_threshold,
        default=[],
        help="追加检查，例如 recall_at_3>=0.9 或 negative_fpr<=0.05；可重复使用",
    )
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    try:
        manifest = _read_json(manifest_path)
        report = run_benchmark(manifest, manifest_path=manifest_path)
        thresholds = _thresholds_from_value(manifest.get("thresholds"), source="manifest.thresholds")
        thresholds.extend(args.threshold)
        evaluations = evaluate_thresholds(report, thresholds)
        report["threshold_checks"] = evaluations
        report["passed"] = all(item["passed"] for item in evaluations)
    except BenchmarkError as exc:
        print(f"BENCHMARK_INVALID: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
