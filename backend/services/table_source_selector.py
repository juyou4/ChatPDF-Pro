"""Offline table-source selector for RAG evaluation.

The selector compares table bundles from multiple parsers for the same paper
and builds a mixed ``structured_table_bundles`` list.  It is intentionally
LLM-free and keeps the decision trace in metadata so evaluation runs can be
audited table by table.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from statistics import median
from typing import Any


MIXED_TABLE_RAG_INDEX_SOURCE = "mixed_table"
MIXED_TABLE_SELECTOR_VERSION = "mixed-table-selector-v1"

_SOURCE_TIE_PRIORITY = {
    "pdf_native": 0,
    "mineru_pipeline": 1,
    "mineru_vlm": 2,
}


@dataclass
class TableCandidate:
    source: str
    bundle: dict[str, Any]
    table_ref: str
    align_key: str
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def build_mixed_table_rag_data(
    source_documents: dict[str, dict[str, Any]],
    *,
    base_source: str = "mineru_vlm",
) -> dict[str, Any]:
    """Build RAG data with per-table selected bundles.

    Args:
        source_documents: Mapping from source label to a document JSON or its
            ``data`` object. Expected labels are ``pdf_native``,
            ``mineru_pipeline`` and ``mineru_vlm``; extra labels are accepted.
        base_source: Source used for pages/full_text. Tables are selected
            independently and may come from any source.
    """
    source_data = {
        label: _doc_data(doc)
        for label, doc in (source_documents or {}).items()
        if isinstance(doc, dict)
    }
    if not source_data:
        raise ValueError("source_documents is empty")

    if base_source not in source_data:
        base_source = next(iter(source_data.keys()))
    base = source_data[base_source]

    candidates = _collect_candidates(source_data)
    grouped = _group_candidates(candidates)
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for align_key, group in sorted(grouped.items(), key=lambda item: _group_sort_key(item[1])):
        _score_group(group)
        winner = sorted(
            group,
            key=lambda cand: (
                -cand.score,
                _SOURCE_TIE_PRIORITY.get(cand.source, 99),
                cand.source,
            ),
        )[0]
        selected_bundle = _annotate_selected_bundle(winner, group)
        selected.append(selected_bundle)
        decisions.append(_decision_record(align_key, winner, group))

    full_text = str(base.get("full_text") or "")
    pages = copy.deepcopy(base.get("pages") or [])
    quality_report = _quality_report(source_data, decisions, base_source)
    source_hash = _mixed_source_hash(source_data, decisions, base_source)

    return {
        "full_text": full_text,
        "pages": pages,
        "structured_table_bundles": selected,
        "quality_report": quality_report,
        "source_hash": source_hash,
        "index_source": MIXED_TABLE_RAG_INDEX_SOURCE,
        "normalizer_version": MIXED_TABLE_SELECTOR_VERSION,
        "base_source": base_source,
        "table_selection_decisions": decisions,
    }


def _doc_data(doc: dict[str, Any]) -> dict[str, Any]:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else doc
    return data if isinstance(data, dict) else {}


def _collect_candidates(source_data: dict[str, dict[str, Any]]) -> list[TableCandidate]:
    candidates: list[TableCandidate] = []
    for source, data in source_data.items():
        for bundle in data.get("structured_table_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            caption = str(bundle.get("table_caption") or "")
            table_id = str(bundle.get("table_id") or "")
            table_ref = _extract_table_ref(table_id) or _extract_table_ref(caption)
            if not table_ref:
                # This offline selector is scoped to paper-style numbered
                # tables.  Generated pdf-native IDs without a Table N caption
                # cannot be safely aligned across parsers and would add extra
                # tables rather than choosing among equivalent candidates.
                continue
            caption_key = _caption_fingerprint(caption or table_id)
            align_key = f"{table_ref}:{caption_key[:48]}"
            candidates.append(TableCandidate(
                source=source,
                bundle=bundle,
                table_ref=table_ref,
                align_key=align_key,
            ))
    return candidates


def _group_candidates(candidates: list[TableCandidate]) -> dict[str, list[TableCandidate]]:
    by_ref: dict[str, list[TableCandidate]] = defaultdict(list)
    no_ref: dict[str, list[TableCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.table_ref:
            by_ref[candidate.table_ref].append(candidate)
        else:
            no_ref[candidate.align_key].append(candidate)

    grouped: dict[str, list[TableCandidate]] = {}
    for ref, items in by_ref.items():
        # Table N is necessary but not sufficient: same-number tables can exist
        # in appendices or malformed parser output.  Cluster by caption
        # fingerprint with fuzzy matching so OCR spelling variants still align.
        for cluster_key, cluster_items in _cluster_by_caption(items):
            grouped[cluster_key] = cluster_items
    grouped.update(no_ref)
    return grouped


def _cluster_by_caption(items: list[TableCandidate]) -> list[tuple[str, list[TableCandidate]]]:
    clusters: list[tuple[str, list[TableCandidate]]] = []
    for candidate in items:
        best_index = -1
        best_score = 0.0
        for idx, (cluster_key, cluster_items) in enumerate(clusters):
            score = _caption_similarity(_align_caption_key(candidate.align_key), _align_caption_key(cluster_key))
            if score > best_score:
                best_index = idx
                best_score = score
        if best_index >= 0 and best_score >= 0.45:
            clusters[best_index][1].append(candidate)
        else:
            clusters.append((candidate.align_key, [candidate]))
    return clusters


def _align_caption_key(align_key: str) -> str:
    return str(align_key or "").split(":", 1)[-1]


def _caption_similarity(left: str, right: str) -> float:
    left_tokens = [token for token in re.split(r"[-\s]+", str(left or "")) if token]
    right_tokens = [token for token in re.split(r"[-\s]+", str(right or "")) if token]
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    jaccard = len(left_set & right_set) / max(1, len(left_set | right_set))
    prefix_hits = 0
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token != right_token:
            break
        prefix_hits += 1
    prefix = prefix_hits / max(1, min(len(left_tokens), len(right_tokens), 10))
    sequence = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    return max(jaccard, prefix, sequence)


def _score_group(group: list[TableCandidate]) -> None:
    row_key_union: set[str] = set()
    numeric_counter: Counter[str] = Counter()
    for candidate in group:
        candidate.metrics = _bundle_metrics(candidate.bundle)
        row_key_union.update(candidate.metrics["row_keys"])
        numeric_counter.update(set(candidate.metrics["numeric_tokens"]))

    consensus_numbers = {token for token, count in numeric_counter.items() if count >= 2}
    for candidate in group:
        metrics = candidate.metrics
        row_keys = metrics["row_keys"]
        numeric_tokens = set(metrics["numeric_tokens"])
        row_key_coverage = len(row_keys & row_key_union) / max(1, len(row_key_union))
        numeric_consensus = len(numeric_tokens & consensus_numbers) / max(1, len(numeric_tokens))
        metrics["row_key_coverage"] = row_key_coverage
        metrics["numeric_consensus"] = numeric_consensus
        metrics["cross_source_missing_row_count"] = max(0, len(row_key_union - row_keys))

        score = 0.0
        score += min(metrics["data_row_count"], 40) * 0.65
        score += min(metrics["max_col_count"], 24) * 0.16
        score += min(metrics["numeric_token_count"], 160) * 0.035
        score += row_key_coverage * 8.0
        score += numeric_consensus * 10.0
        score += metrics["row_key_ratio"] * 4.0
        score += metrics["header_bound_row_ratio"] * 3.0
        score += 1.5 if candidate.table_ref else 0.0
        score += min(len(str(candidate.bundle.get("table_caption") or "")), 180) / 180.0
        score -= metrics["empty_cell_ratio"] * 8.0
        score -= metrics["duplicate_header_ratio"] * 6.0
        score -= metrics["blank_header_ratio"] * 10.0
        score -= min(metrics["cross_source_missing_row_count"], 12) * 0.35
        score -= 3.0 if metrics["table8_pollution"] else 0.0
        score -= 2.0 if metrics["suspicious_concatenated_number_count"] else 0.0
        score -= min(metrics["overpacked_metric_cell_count"], 24) * 0.45

        candidate.score = round(score, 4)
        candidate.reasons = _candidate_reasons(candidate)


def _bundle_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in (bundle.get("evidence_units") or [])
        if isinstance(row, dict) and str(row.get("evidence_unit_type") or "").lower() == "table_row"
    ]
    data_rows = [row for row in rows if not row.get("is_header_row")]
    cell_counts = [int(row.get("cell_count") or len(row.get("cell_evidence_units") or [])) for row in rows]
    non_empty_cells = 0
    empty_cells = 0
    header_bound_rows = 0
    row_keys: set[str] = set()
    all_text_parts: list[str] = [
        str(bundle.get("table_caption") or ""),
        str(bundle.get("table_header") or ""),
        str(bundle.get("table_body_markdown") or ""),
    ]

    for row in data_rows:
        row_text = str(row.get("row_text") or row.get("content") or row.get("raw_row_text") or "")
        all_text_parts.append(row_text)
        if _looks_header_bound(row_text):
            header_bound_rows += 1
        row_key = _row_key(row)
        if row_key:
            row_keys.add(row_key)
        cells = row.get("cell_evidence_units") or []
        if isinstance(cells, list) and cells:
            for cell in cells:
                value = str((cell or {}).get("content") or (cell or {}).get("cell_text") or "").strip()
                if value:
                    non_empty_cells += 1
                else:
                    empty_cells += 1

    text = "\n".join(all_text_parts)
    numeric_tokens = _numeric_tokens(text)
    max_cols = max(cell_counts or [0])
    median_cols = int(median(cell_counts)) if cell_counts else 0
    total_cells = non_empty_cells + empty_cells
    header = str(bundle.get("table_header") or "")

    return {
        "row_count": len(rows),
        "data_row_count": len(data_rows),
        "max_col_count": max_cols,
        "median_col_count": median_cols,
        "empty_cell_ratio": empty_cells / total_cells if total_cells else 0.0,
        "numeric_token_count": len(numeric_tokens),
        "numeric_tokens": numeric_tokens,
        "row_keys": row_keys,
        "row_key_count": len(row_keys),
        "row_key_ratio": len(row_keys) / max(1, len(data_rows)),
        "header_bound_row_ratio": header_bound_rows / max(1, len(data_rows)),
        "blank_header_ratio": _blank_header_ratio(header),
        "duplicate_header_ratio": _duplicate_header_ratio(header),
        "table8_pollution": bool(re.search(r"\bTable\s*8\b", text, re.IGNORECASE) and re.search(r"\bTable\s*7\b", text, re.IGNORECASE)),
        "suspicious_concatenated_number_count": len(re.findall(r"\d+\.\d{3,}\b", text)),
        "overpacked_metric_cell_count": _overpacked_metric_cell_count(data_rows),
    }


def _annotate_selected_bundle(winner: TableCandidate, group: list[TableCandidate]) -> dict[str, Any]:
    bundle = copy.deepcopy(winner.bundle)
    bundle["selected_source"] = winner.source
    bundle["selection_reason"] = "; ".join(winner.reasons)
    bundle["table_selector_score"] = winner.score
    bundle["table_selector_version"] = MIXED_TABLE_SELECTOR_VERSION
    bundle["table_selector_candidates"] = [
        {
            "source": candidate.source,
            "score": candidate.score,
            "bundle_id": candidate.bundle.get("bundle_id", ""),
            "table_id": candidate.bundle.get("table_id", ""),
            "data_row_count": candidate.metrics.get("data_row_count", 0),
            "max_col_count": candidate.metrics.get("max_col_count", 0),
            "numeric_consensus": round(float(candidate.metrics.get("numeric_consensus", 0.0)), 4),
            "row_key_coverage": round(float(candidate.metrics.get("row_key_coverage", 0.0)), 4),
            "empty_cell_ratio": round(float(candidate.metrics.get("empty_cell_ratio", 0.0)), 4),
            "blank_header_ratio": round(float(candidate.metrics.get("blank_header_ratio", 0.0)), 4),
            "overpacked_metric_cell_count": int(candidate.metrics.get("overpacked_metric_cell_count", 0) or 0),
        }
        for candidate in sorted(group, key=lambda item: item.source)
    ]
    bundle["source"] = winner.source
    for row in bundle.get("evidence_units") or []:
        if isinstance(row, dict):
            row["selected_source"] = winner.source
            row["selection_reason"] = bundle["selection_reason"]
            row["source"] = winner.source
            for cell in row.get("cell_evidence_units") or []:
                if isinstance(cell, dict):
                    cell["selected_source"] = winner.source
                    cell["source"] = winner.source
    return bundle


def _decision_record(align_key: str, winner: TableCandidate, group: list[TableCandidate]) -> dict[str, Any]:
    return {
        "align_key": align_key,
        "table_ref": winner.table_ref,
        "table_id": winner.bundle.get("table_id", ""),
        "table_caption": winner.bundle.get("table_caption", ""),
        "selected_source": winner.source,
        "selection_reason": "; ".join(winner.reasons),
        "score": winner.score,
        "candidates": [
            {
                "source": candidate.source,
                "score": candidate.score,
                "bundle_id": candidate.bundle.get("bundle_id", ""),
                "table_id": candidate.bundle.get("table_id", ""),
                "metrics": _jsonable_metrics(candidate.metrics),
                "reasons": candidate.reasons,
            }
            for candidate in sorted(group, key=lambda item: item.source)
        ],
    }


def _candidate_reasons(candidate: TableCandidate) -> list[str]:
    metrics = candidate.metrics
    reasons = [
        f"rows={metrics.get('data_row_count', 0)}",
        f"cols={metrics.get('max_col_count', 0)}",
        f"numeric={metrics.get('numeric_token_count', 0)}",
        f"row_key_coverage={metrics.get('row_key_coverage', 0.0):.2f}",
        f"numeric_consensus={metrics.get('numeric_consensus', 0.0):.2f}",
    ]
    if metrics.get("table8_pollution"):
        reasons.append("penalty=adjacent_table_pollution")
    if metrics.get("suspicious_concatenated_number_count"):
        reasons.append("penalty=concatenated_numbers")
    if metrics.get("blank_header_ratio"):
        reasons.append(f"penalty=blank_headers:{metrics.get('blank_header_ratio', 0.0):.2f}")
    if metrics.get("overpacked_metric_cell_count"):
        reasons.append(f"penalty=overpacked_metric_cells:{metrics.get('overpacked_metric_cell_count', 0)}")
    return reasons


def _quality_report(source_data: dict[str, dict[str, Any]], decisions: list[dict[str, Any]], base_source: str) -> dict[str, Any]:
    selected_counts = Counter(decision["selected_source"] for decision in decisions)
    return {
        "selector_version": MIXED_TABLE_SELECTOR_VERSION,
        "base_source": base_source,
        "source_table_counts": {
            source: len((data.get("structured_table_bundles") or []))
            for source, data in source_data.items()
        },
        "selected_source_counts": dict(sorted(selected_counts.items())),
        "table_count": len(decisions),
        "decisions": decisions,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def _mixed_source_hash(source_data: dict[str, dict[str, Any]], decisions: list[dict[str, Any]], base_source: str) -> str:
    payload = {
        "version": MIXED_TABLE_SELECTOR_VERSION,
        "base_source": base_source,
        "sources": {
            source: {
                "full_text_len": len(str(data.get("full_text") or "")),
                "table_count": len(data.get("structured_table_bundles") or []),
                "bundle_ids": [
                    bundle.get("bundle_id", "")
                    for bundle in data.get("structured_table_bundles") or []
                    if isinstance(bundle, dict)
                ],
            }
            for source, data in source_data.items()
        },
        "selected": [
            {
                "align_key": item.get("align_key", ""),
                "selected_source": item.get("selected_source", ""),
                "bundle_id": next(
                    (
                        candidate.get("bundle_id", "")
                        for candidate in item.get("candidates", [])
                        if candidate.get("source") == item.get("selected_source")
                    ),
                    "",
                ),
            }
            for item in decisions
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, set):
            result[key] = sorted(value)
        elif key == "numeric_tokens":
            result[key] = sorted(set(value))[:80]
        else:
            result[key] = value
    return result


def _group_sort_key(group: list[TableCandidate]) -> tuple[int, str]:
    pages = []
    refs = []
    for candidate in group:
        bundle = candidate.bundle
        page = int(bundle.get("page_start") or (bundle.get("pages") or [9999])[0] or 9999)
        pages.append(page)
        refs.append(candidate.table_ref or str(bundle.get("table_id") or bundle.get("table_caption") or ""))
    return (min(pages or [9999]), min(refs or [""]))


def _extract_table_ref(value: str) -> str:
    match = re.search(r"\b(?:Table|TABLE|表)\s*\.?\s*([A-Za-z0-9IVXLC]+(?:\.\d+)?)", value or "", re.IGNORECASE)
    if not match:
        return ""
    token = match.group(1).upper()
    return f"table-{token}"


def _caption_fingerprint(value: str) -> str:
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").casefold())
    tokens = [token for token in text.split() if token and token not in {"table", "the", "and", "of", "on", "for"}]
    return "-".join(tokens[:14]) or "uncaptioned"


def _numeric_tokens(text: str) -> list[str]:
    tokens = []
    for match in re.finditer(r"(?<![A-Za-z])[-+−]?\d+(?:[.,]\d+)?%?", text or ""):
        token = match.group(0).replace("−", "-").replace(",", "")
        # Ignore standalone table labels and very long concatenation artifacts.
        if re.fullmatch(r"\d+", token) and int(token) <= 30:
            continue
        if len(token) > 12:
            continue
        tokens.append(token)
    return tokens


def _row_key(row: dict[str, Any]) -> str:
    for value in (
        row.get("row_id"),
        row.get("row_label"),
        row.get("method"),
        row.get("model"),
    ):
        key = _normalize_row_key(value)
        if key:
            return key
    cells = row.get("cell_evidence_units") or []
    if isinstance(cells, list):
        for cell in cells[:2]:
            if not isinstance(cell, dict):
                continue
            key = _normalize_row_key(cell.get("content") or cell.get("cell_text"))
            if key:
                return key
    text = str(row.get("row_text") or row.get("content") or "")
    if ":" in text:
        first = text.split(";", 1)[0]
        key = _normalize_row_key(first.split(":", 1)[-1])
        if key:
            return key
    return ""


def _normalize_row_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" :：;|,")
    if not text:
        return ""
    if len(text) > 80:
        return ""
    lowered = text.casefold()
    if lowered in {"method", "model", "metrics", "datasets", "id", "row", "column"}:
        return ""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lowered)


def _looks_header_bound(text: str) -> bool:
    return len([part for part in re.split(r";|\n", text or "") if ":" in part]) >= 2


def _duplicate_header_ratio(header: str) -> float:
    cells = [re.sub(r"\s+", " ", part).strip().casefold() for part in str(header or "").split("|")]
    cells = [cell for cell in cells if cell]
    if not cells:
        return 0.0
    counts = Counter(cells)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(cells)


def _blank_header_ratio(header: str) -> float:
    parts = [re.sub(r"\s+", " ", part).strip() for part in str(header or "").split("|")]
    parts = [part for part in parts if part or len(parts) > 1]
    if not parts:
        return 0.0
    return sum(1 for part in parts if not part) / len(parts)


def _overpacked_metric_cell_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    metric_key_pattern = re.compile(
        r"\b(?:ap|ap50|ap75|map|nds|fid|acc|accuracy|precision|recall|score|metric|mate|mase|maoe|mave|maae)\b",
        re.IGNORECASE,
    )
    for row in rows:
        row_text = str(row.get("row_text") or row.get("content") or row.get("raw_row_text") or "")
        for part in re.split(r"\s*;\s*", row_text):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            if not metric_key_pattern.search(key):
                continue
            if "/" in value or "," in value:
                continue
            decimal_values = re.findall(r"(?<![A-Za-z])[-+−]?\d+\.\d+", value)
            if len(decimal_values) >= 2:
                count += 1
    return count
