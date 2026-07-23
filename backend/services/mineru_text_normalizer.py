"""Normalize MinerU parse results into ChatPDF RAG-native inputs.

This module is intentionally separate from ``mineru_block_index_service``:
the block index is for immersive reading anchors, while this normalizer builds
the ``pages/full_text/structured_table_bundles`` shape consumed by the RAG
indexing pipeline.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from services.document_geometry import (
    normalize_bbox as normalize_document_bbox,
    normalize_page_size,
    visual_geometry,
)

MINERU_RAG_INDEX_SOURCE = "mineru"
MINERU_RAG_NORMALIZER_VERSION = "mineru-rag-v2"
# MinerU is the selected primary parser for a full document. Publishing an
# otherwise healthy-looking index with one missing page makes every downstream
# capability silently incomplete, so this is intentionally a full-coverage
# contract rather than a best-effort OCR threshold.
MINERU_MIN_PAGE_COVERAGE = 1.0

_INCLUDE_TEXT_TYPES = {
    "text",
    "plain_text",
    "paragraph",
    "para",
    "body_text",
    "list",
    "list_item",
    "code",
    "code_body",
    "ref_text",
    "reference",
    "references",
    "title",
    "heading",
    "caption",
    "image_caption",
    "table_caption",
    "chart_caption",
}
_CAPTION_ONLY_TYPES = {"image", "chart"}
_TABLE_TYPES = {"table", "table_body"}
_FORMULA_TYPES = {"equation", "interline_equation", "inline_equation", "formula"}
_EXCLUDE_TYPES = {
    "header",
    "page_number",
    "page_footnote",
    "aside_text",
    "discarded",
    "discarded_block",
    "abandon",
    "footer",
    "header_footer",
}
_HTML_TAG_RE = re.compile(r"</?(?:table|thead|tbody|tfoot|tr|td|th)\b", re.IGNORECASE)


class _TableHTMLParser(HTMLParser):
    """Small table parser that preserves row/col spans without extra deps."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._cell_parts: list[str] = []
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"} and self._current_row is not None:
            attr_map = {key.lower(): value for key, value in attrs}
            self._current_cell = {
                "tag": tag,
                "rowspan": _positive_int(attr_map.get("rowspan"), 1),
                "colspan": _positive_int(attr_map.get("colspan"), 1),
            }
            self._cell_parts = []
            return
        if tag in {"br", "p", "div"} and self._current_cell is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None:
            cell = dict(self._current_cell)
            cell["text"] = _clean_cell_text("".join(self._cell_parts))
            if self._current_row is not None:
                self._current_row.append(cell)
            self._current_cell = None
            self._cell_parts = []
            return
        if tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None
            return
        if tag == "table":
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None and data:
            self._cell_parts.append(data)


def build_mineru_page_ledger(
    mineru_payload: dict[str, Any],
    *,
    expected_page_count: int = 0,
    block_pages: set[int] | list[int] | tuple[int, ...] | None = None,
    text_pages: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build a page-level parse ledger without inventing successful pages.

    ``expected_page_count`` must come from the source PDF. When it is unknown,
    the ledger reports the pages present in MinerU output but deliberately does
    not infer missing leading pages from a sparse page number.
    """
    payload = mineru_payload if isinstance(mineru_payload, dict) else {}
    try:
        expected_count = max(0, int(expected_page_count or 0))
    except (TypeError, ValueError):
        expected_count = 0

    observed_pages = _payload_page_numbers(payload)
    reported_failed_pages = _reported_failed_page_numbers(payload)
    block_page_numbers = _positive_page_numbers(block_pages)
    text_page_numbers = _positive_page_numbers(text_pages)
    consumable_pages = block_page_numbers | text_page_numbers

    if expected_count > 0:
        expected_pages = set(range(1, expected_count + 1))
        missing_pages = expected_pages - observed_pages - consumable_pages
        empty_pages = (observed_pages & expected_pages) - consumable_pages - reported_failed_pages
        failed_pages = missing_pages | empty_pages | (reported_failed_pages & expected_pages)
        successful_pages = (expected_pages & consumable_pages) - reported_failed_pages
        coverage = len(successful_pages) / expected_count
        ledger_pages = sorted(expected_pages | observed_pages | reported_failed_pages | consumable_pages)
    else:
        expected_pages = observed_pages | reported_failed_pages | consumable_pages
        empty_pages = observed_pages - consumable_pages - reported_failed_pages
        failed_pages = set(reported_failed_pages) | empty_pages
        successful_pages = consumable_pages - reported_failed_pages
        coverage = (len(successful_pages) / len(expected_pages)) if expected_pages else 0.0
        ledger_pages = sorted(expected_pages)

    page_ledger: list[dict[str, Any]] = []
    for page_num in ledger_pages:
        provider_failed = page_num in reported_failed_pages
        missing = (
            expected_count > 0
            and page_num in expected_pages
            and page_num not in observed_pages
            and page_num not in consumable_pages
        )
        empty_result = page_num in observed_pages and page_num not in consumable_pages
        if provider_failed:
            status = "failed"
            reason = "provider_reported_failure"
        elif missing:
            status = "failed"
            reason = "missing_from_result"
        elif empty_result:
            status = "failed"
            reason = "empty_page_result"
        elif page_num in consumable_pages:
            status = "success"
            reason = ""
        else:
            status = "unknown"
            reason = "not_observed"
        page_ledger.append({
            "page": page_num,
            "status": status,
            "reason": reason,
            "raw_present": page_num in observed_pages,
            "has_blocks": page_num in block_page_numbers,
            "has_text": page_num in text_page_numbers,
        })

    unexpected_pages = sorted(
        page for page in observed_pages
        if expected_count > 0 and page > expected_count
    )
    if not consumable_pages or coverage < MINERU_MIN_PAGE_COVERAGE or failed_pages:
        quality_status = "failed"
    else:
        quality_status = "success"

    return {
        "quality_status": quality_status,
        "expected_page_count": expected_count,
        "observed_page_count": len(observed_pages & expected_pages) if expected_count else len(observed_pages),
        "covered_page_count": len(successful_pages),
        "coverage": round(float(coverage), 4),
        "failed_pages": sorted(failed_pages),
        "unexpected_pages": unexpected_pages,
        "page_ledger": page_ledger,
    }


def normalize_mineru_for_rag(
    mineru_payload: dict[str, Any],
    *,
    source_hash: str | None = None,
    normalizer_version: str | None = None,
    page_sizes: dict[int, list[float] | tuple[float, float]] | None = None,
    expected_page_count: int = 0,
) -> dict[str, Any]:
    """Convert MinerU payload into RAG-native pages/full_text/table bundles."""
    payload = mineru_payload if isinstance(mineru_payload, dict) else {}
    version = normalizer_version or MINERU_RAG_NORMALIZER_VERSION
    source_hash = source_hash or _payload_hash(payload)

    items = _extract_content_items(payload)
    page_quality = build_mineru_page_ledger(
        payload,
        expected_page_count=expected_page_count,
    )
    middle_geometry_warnings: list[str] = []
    if items:
        items, middle_geometry_warnings = _enrich_content_table_geometry(
            items,
            payload.get("middle_json"),
        )
    if not items:
        quality = _quality_report(
            kept_counts={},
            filtered_counts={},
            table_count=0,
            evidence_unit_count=0,
            malformed_table_count=0,
            warnings=["missing_mineru_structural_items"],
            failure_reasons=["MinerU 结果中没有可消费的结构化块"],
            page_quality=page_quality,
        )
        return {
            "full_text": "",
            "pages": [],
            "structured_table_bundles": [],
            "quality_report": quality,
            "source_hash": source_hash,
            "index_source": MINERU_RAG_INDEX_SOURCE,
            "normalizer_version": version,
            **page_quality,
        }

    pages_parts: dict[int, list[str]] = {}
    structured_table_bundles: list[dict[str, Any]] = []
    raw_type_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}
    warnings: list[str] = list(middle_geometry_warnings)
    unknown_types: set[str] = set()
    malformed_table_count = 0
    evidence_unit_count = 0

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_type = _normalize_type(item.get("type") or item.get("block_type") or item.get("category"))
        normalized_raw_type = raw_type or "unknown"
        _inc(raw_type_counts, normalized_raw_type)
        page_num = _page_num(item)

        if raw_type in _EXCLUDE_TYPES:
            _inc(filtered_counts, raw_type)
            continue

        text = ""
        if raw_type in _INCLUDE_TEXT_TYPES:
            text = _clean_text(_extract_text(item, ("text", "content", "blocks", "lines", "spans")))
        elif raw_type in _CAPTION_ONLY_TYPES:
            text = _clean_text(_extract_text(item, (
                "image_caption",
                "img_caption",
                "chart_caption",
                "caption",
                "image_footnote",
                "chart_footnote",
                "footnote",
            )))
        elif raw_type in _FORMULA_TYPES:
            text = _format_equation(_extract_text(item, ("latex", "text", "content")))
        elif raw_type in _TABLE_TYPES:
            bundle, table_text, table_warnings = _build_table_bundle(
                item,
                page_num,
                item_index,
                page_size=_resolve_page_size(item, page_num, page_sizes),
            )
            warnings.extend(table_warnings)
            if bundle:
                structured_table_bundles.append(bundle)
                evidence_unit_count += len(bundle.get("evidence_units") or [])
            else:
                malformed_table_count += 1
            text = table_text
        else:
            # MinerU evolves its paragraph taxonomy frequently. A text-bearing
            # unknown block is still evidence, so retain it as generic text and
            # make the schema drift visible in the quality report.
            text = _clean_text(_extract_text(item, ("text", "content", "html", "latex", "blocks", "lines", "spans")))
            if text:
                unknown_types.add(normalized_raw_type)
                warnings.append(f"unknown_text_type_downgraded:{normalized_raw_type}:p{page_num}:i{item_index}")
            else:
                unknown_types.add(normalized_raw_type)
                _inc(filtered_counts, f"{normalized_raw_type}:unsupported")
                warnings.append(f"unknown_type_dropped:{normalized_raw_type}:p{page_num}:i{item_index}")
                continue

        if not text:
            _inc(filtered_counts, f"{raw_type}:empty")
            continue

        if _HTML_TAG_RE.search(text):
            warnings.append(f"html_stripped:{raw_type}:p{page_num}:i{item_index}")
            text = _strip_html_tags(text)
        text = _clean_text(text)
        if not text:
            _inc(filtered_counts, f"{raw_type}:empty_after_clean")
            continue

        _inc(kept_counts, raw_type)
        pages_parts.setdefault(page_num, []).append(text)

    pages = [
        {"page": page_num, "content": "\n\n".join(parts).strip()}
        for page_num, parts in sorted(pages_parts.items())
        if "\n\n".join(parts).strip()
    ]
    full_text = "\n\n".join(page["content"] for page in pages).strip()
    page_quality = build_mineru_page_ledger(
        payload,
        expected_page_count=expected_page_count,
        text_pages={int(page["page"]) for page in pages},
    )
    quality = _quality_report(
        kept_counts=kept_counts,
        raw_type_counts=raw_type_counts,
        filtered_counts=filtered_counts,
        table_count=len(structured_table_bundles),
        evidence_unit_count=evidence_unit_count,
        malformed_table_count=malformed_table_count,
        warnings=warnings,
        failure_reasons=[],
        page_quality=page_quality,
        unknown_types=unknown_types,
    )

    return {
        "full_text": full_text,
        "pages": pages,
        "structured_table_bundles": structured_table_bundles,
        "quality_report": quality,
        "source_hash": source_hash,
        "index_source": MINERU_RAG_INDEX_SOURCE,
        "normalizer_version": version,
        **page_quality,
    }


def validate_mineru_rag_data(
    normalized: dict[str, Any],
    *,
    original_full_text: str = "",
    min_text_ratio: float = 0.6,
    min_page_coverage: float = MINERU_MIN_PAGE_COVERAGE,
) -> tuple[bool, list[str]]:
    """Quality gate before replacing an existing RAG index."""
    failures: list[str] = []
    pages = normalized.get("pages") if isinstance(normalized, dict) else None
    full_text = str((normalized or {}).get("full_text") or "")
    bundles = normalized.get("structured_table_bundles") or []
    try:
        expected_page_count = max(0, int((normalized or {}).get("expected_page_count") or 0))
    except (TypeError, ValueError):
        expected_page_count = 0
    try:
        coverage = float((normalized or {}).get("coverage") or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0
    quality_status = str((normalized or {}).get("quality_status") or "failed").strip().lower()
    failed_pages = [
        page
        for page in ((normalized or {}).get("failed_pages") or [])
        if isinstance(page, (int, float, str)) and str(page).strip()
    ]

    if not isinstance(pages, list) or not pages:
        failures.append("pages_empty")
    if not full_text.strip():
        failures.append("full_text_empty")
    if expected_page_count > 0 and coverage < float(min_page_coverage):
        failures.append("page_coverage_incomplete")
    if failed_pages:
        failures.append("page_parse_failed")
    if quality_status != "success":
        failures.append("parse_quality_not_complete")
    original_len = len(str(original_full_text or "").strip())
    if original_len >= 200 and len(full_text.strip()) < int(original_len * min_text_ratio):
        failures.append("full_text_too_short")
    if _HTML_TAG_RE.search(full_text):
        failures.append("html_tag_in_full_text")
    for page in pages or []:
        content = str((page or {}).get("content") or "")
        if _HTML_TAG_RE.search(content):
            failures.append("html_tag_in_page_text")
            break
    for bundle in bundles if isinstance(bundles, list) else []:
        if not isinstance(bundle, dict):
            failures.append("invalid_table_bundle")
            continue
        body = str(bundle.get("table_body_markdown") or "")
        if _HTML_TAG_RE.search(body):
            failures.append("html_tag_in_table_markdown")
        page_numbers = bundle.get("pages") or []
        if not all(isinstance(page, int) and page > 0 for page in page_numbers):
            failures.append("table_page_not_1_based")
        if not _table_markdown_has_consistent_columns(body):
            failures.append("table_markdown_inconsistent_columns")

    return not failures, failures


def _extract_content_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload.get("content_list_json")
    content_items = [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []
    middle = payload.get("middle_json")
    if isinstance(middle, list):
        middle_items = [item for item in middle if isinstance(item, dict)]
    elif isinstance(middle, dict):
        middle_items = _middle_json_content_items(middle)
    else:
        middle_items = []
    return _merge_content_items(content_items, middle_items)


def _middle_json_content_items(middle_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose middle JSON blocks to the RAG normalizer as a lossless fallback."""
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        return []
    items: list[dict[str, Any]] = []
    for page_info in pdf_info:
        if not isinstance(page_info, dict):
            continue
        page_num = _page_num(page_info)
        for block in _middle_json_page_blocks(page_info):
            item = dict(block)
            item.setdefault("page_idx", page_num - 1)
            if page_info.get("page_size") and not item.get("page_size"):
                item["page_size"] = page_info["page_size"]
            items.append(item)
    return items


def _merge_content_items(
    primary_items: list[dict[str, Any]],
    fallback_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep content-list ordering but append only middle-JSON blocks it lacks."""
    merged = [dict(item) for item in primary_items]
    for item in fallback_items:
        if not any(_content_items_overlap(existing, item) for existing in merged):
            merged.append(dict(item))
    return merged


def _content_items_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _page_num(left) != _page_num(right):
        return False
    left_type = _normalize_type(left.get("type") or left.get("block_type") or left.get("category"))
    right_type = _normalize_type(right.get("type") or right.get("block_type") or right.get("category"))
    left_text = _content_item_text_fingerprint(left)
    right_text = _content_item_text_fingerprint(right)
    if left_text and left_text == right_text and len(left_text) >= 12:
        return True
    if left_type != right_type:
        return False
    left_bbox = _normalize_bbox(left.get("bbox") or left.get("layout_bbox") or left.get("block_bbox"))
    right_bbox = _normalize_bbox(right.get("bbox") or right.get("layout_bbox") or right.get("block_bbox"))
    return bool(left_bbox and left_bbox == right_bbox and (left_text == right_text or not left_text or not right_text))


def _content_item_text_fingerprint(item: dict[str, Any]) -> str:
    value = _extract_text(item, ("text", "content", "html", "latex", "table_body", "blocks", "lines", "spans"))
    return re.sub(r"\s+", " ", _clean_text(value)).strip().lower()


def _enrich_content_table_geometry(
    items: list[dict[str, Any]],
    middle_json: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fill missing table overview geometry only when the source table is provably the same.

    MinerU's standard middle JSON carries a page-level table body bbox, but not
    row or cell bboxes.  HTML identity lets us safely join that bbox to the
    content-list table without turning page order into a geometry guess.
    """
    candidates = _middle_json_table_candidates(middle_json)
    if not candidates:
        return items, []

    by_page: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_page.setdefault(candidate["page"], []).append(candidate)

    enriched = [dict(item) for item in items]
    warnings: list[str] = []
    used_candidates: set[int] = set()
    for item_index, item in enumerate(enriched):
        raw_type = _normalize_type(item.get("type") or item.get("block_type") or item.get("category"))
        if raw_type != "table":
            continue
        page_num = _page_num(item)
        table_html = _table_html_from_item(item)
        fingerprint = _table_html_fingerprint(table_html)
        if not fingerprint:
            continue

        matching = [
            candidate
            for candidate in by_page.get(page_num, [])
            if candidate["html_fingerprint"] == fingerprint and candidate["candidate_index"] not in used_candidates
        ]
        if not matching:
            if not _item_table_bbox(item):
                warnings.append(f"middle_json_table_geometry_unmatched:p{page_num}:i{item_index}")
            continue

        # Matching HTML is the primary identity.  When identical tables repeat
        # on one page, consuming them in page order is the only remaining stable
        # disambiguator; no bbox is assigned when the HTML does not match.
        candidate = matching[0]
        used_candidates.add(candidate["candidate_index"])
        if not _item_table_bbox(item):
            item["bbox"] = list(candidate["bbox"])
            item["_table_geometry_source"] = "middle_json_table"
        if len(candidate["pages"]) > 1:
            item["_table_pages"] = list(candidate["pages"])

    return enriched, warnings


def _middle_json_table_candidates(middle_json: Any) -> list[dict[str, Any]]:
    if not isinstance(middle_json, dict):
        return []
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        return []

    candidates: list[dict[str, Any]] = []
    previous_page_candidates: list[dict[str, Any]] = []
    for page_info in pdf_info:
        if not isinstance(page_info, dict):
            continue
        page_num = _page_num(page_info)
        source_page_size = normalize_page_size(page_info.get("page_size"))
        page_candidates: list[dict[str, Any]] = []
        has_deleted_table_body = False
        for block in _middle_json_page_blocks(page_info):
            if _normalize_type(block.get("type") or block.get("block_type") or block.get("category")) != "table":
                continue
            has_deleted_table_body = has_deleted_table_body or _middle_table_lines_deleted(block)
            table_html = _middle_table_html(block)
            bbox = _middle_bbox_to_normalized(_middle_table_bbox(block), source_page_size)
            fingerprint = _table_html_fingerprint(table_html)
            if not bbox or not fingerprint:
                continue
            page_candidates.append({
                "page": page_num,
                "bbox": bbox,
                "pages": [page_num],
                "html_fingerprint": fingerprint,
            })

        # MinerU merges a continuation page's HTML into the previous page's
        # final table, then marks the continuation body's lines as deleted.  A
        # multi-page bundle makes the visual verifier abstain instead of treating
        # the first-page overview as evidence for rows from later pages.
        if has_deleted_table_body and previous_page_candidates:
            previous_page_candidates[-1]["pages"].append(page_num)

        candidates.extend(page_candidates)
        previous_page_candidates = page_candidates

    for index, candidate in enumerate(candidates):
        candidate["candidate_index"] = index
    return candidates


def _middle_json_page_blocks(page_info: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("para_blocks", "preproc_blocks", "layout_blocks", "blocks"):
        blocks = page_info.get(key)
        if isinstance(blocks, list) and blocks:
            return [block for block in blocks if isinstance(block, dict)]
    return []


def _middle_table_children(block: dict[str, Any]) -> list[dict[str, Any]]:
    children = block.get("blocks")
    return [child for child in children if isinstance(child, dict)] if isinstance(children, list) else []


def _middle_table_body(block: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            child
            for child in _middle_table_children(block)
            if _normalize_type(child.get("type") or child.get("block_type")) == "table_body"
        ),
        {},
    )


def _middle_table_html(block: dict[str, Any]) -> str:
    body = _middle_table_body(block)
    for value in (
        body.get("html"),
        body.get("table_body"),
        block.get("html"),
        block.get("table_body"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    lines = body.get("lines") if isinstance(body.get("lines"), list) else []
    for line in lines:
        spans = line.get("spans") if isinstance(line, dict) and isinstance(line.get("spans"), list) else []
        for span in spans:
            if not isinstance(span, dict):
                continue
            value = span.get("html") or span.get("table_body")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _middle_table_bbox(block: dict[str, Any]) -> list[float]:
    for record in (block, _middle_table_body(block)):
        for key in ("bbox", "table_body_bbox", "layout_bbox", "block_bbox"):
            bbox = normalize_document_bbox(record.get(key)) if isinstance(record, dict) else []
            if bbox:
                return bbox
    body = _middle_table_body(block)
    lines = body.get("lines") if isinstance(body.get("lines"), list) else []
    for line in lines:
        spans = line.get("spans") if isinstance(line, dict) and isinstance(line.get("spans"), list) else []
        for span in spans:
            if not isinstance(span, dict):
                continue
            bbox = normalize_document_bbox(span.get("bbox") or span.get("table_body_bbox"))
            if bbox:
                return bbox
    return []


def _middle_table_lines_deleted(block: dict[str, Any]) -> bool:
    if block.get("lines_deleted") is True:
        return True
    body = _middle_table_body(block)
    return body.get("lines_deleted") is True


def _middle_bbox_to_normalized(bbox: Any, page_size: list[float]) -> list[float]:
    raw_bbox = normalize_document_bbox(bbox)
    if not raw_bbox or not page_size:
        return []
    width, height = page_size
    if (
        raw_bbox[0] < 0
        or raw_bbox[1] < 0
        or raw_bbox[2] > width
        or raw_bbox[3] > height
    ):
        return []
    return [
        round(raw_bbox[0] / width * 1000.0, 3),
        round(raw_bbox[1] / height * 1000.0, 3),
        round(raw_bbox[2] / width * 1000.0, 3),
        round(raw_bbox[3] / height * 1000.0, 3),
    ]


def _table_html_from_item(item: dict[str, Any]) -> str:
    for key in ("table_body", "html", "table_html"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _table_html_fingerprint(value: str) -> str:
    text = str(value or "").strip()
    if not text or not _HTML_TAG_RE.search(text):
        return ""
    canonical = re.sub(r"\s+", "", html.unescape(text)).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _item_table_bbox(item: dict[str, Any]) -> list[float]:
    for key in ("bbox", "table_body_bbox"):
        bbox = normalize_document_bbox(item.get(key))
        if bbox:
            return bbox
    return []


def _build_table_bundle(
    item: dict[str, Any],
    page_num: int,
    item_index: int,
    *,
    page_size: list[float],
) -> tuple[dict[str, Any] | None, str, list[str]]:
    warnings: list[str] = []
    html_table = str(item.get("table_body") or item.get("html") or item.get("table_html") or "").strip()
    caption = _clean_text(_extract_text(item, ("table_caption", "caption")))
    footnote = _clean_text(_extract_text(item, ("table_footnote",)))
    bbox = _normalize_bbox(item.get("bbox") or item.get("table_body_bbox"))
    geometry = visual_geometry(
        bbox,
        coordinate_space="normalized_0_1000",
        page_size=page_size,
    )
    table_id = _extract_table_reference(caption) or f"MinerU Table p{page_num}-{item_index + 1}"
    bundle_id = f"mineru:p{page_num}:table:{item_index + 1}"

    matrix: list[list[str]] = []
    if html_table:
        try:
            matrix = _html_table_to_matrix(html_table)
        except Exception as exc:
            warnings.append(f"table_html_parse_failed:p{page_num}:i{item_index}:{exc}")

    if not matrix:
        fallback_text = _strip_html_tags(html_table) if html_table else _extract_text(item, ("text", "content"))
        fallback_text = _clean_text(fallback_text)
        if fallback_text:
            text = "\n\n".join(part for part in (f"[TABLE] {caption}" if caption else "[TABLE]", fallback_text) if part)
            return None, text, warnings or [f"table_fallback_plain_text:p{page_num}:i{item_index}"]
        return None, "", warnings or [f"table_empty:p{page_num}:i{item_index}"]

    body_markdown = _matrix_to_markdown(matrix)
    page_markdown = _matrix_to_markdown(matrix, caption=caption)
    header = " | ".join(matrix[0]).strip() if matrix else ""
    cell_bboxes, row_bboxes = _extract_trustworthy_table_geometry(item, matrix, bbox)
    evidence_units = _build_table_evidence_units(
        bundle_id=bundle_id,
        table_id=table_id,
        caption=caption,
        header=header,
        matrix=matrix,
        page_num=page_num,
        bbox=bbox,
        page_size=page_size,
        cell_bboxes=cell_bboxes,
        row_bboxes=row_bboxes,
    )
    table_pages = _table_pages_for_item(item, page_num)
    bundle = {
        "bundle_id": bundle_id,
        "evidence_unit_id": f"{bundle_id}::table_bundle",
        "table_id": table_id,
        "table_caption": caption,
        "table_header": header,
        "table_body_markdown": body_markdown,
        "table_markdown": body_markdown,
        "html_table": html_table,
        "table_footnote": footnote,
        "page_start": table_pages[0],
        "page_end": table_pages[-1],
        "pages": table_pages,
        "page_index": page_num - 1,
        "bounding_box": bbox,
        "bounding_boxes": [bbox] if bbox else [],
        **geometry,
        "table_geometry_source": str(item.get("_table_geometry_source") or ("content_list" if bbox else "none")),
        "source_ids": [],
        "source": "mineru",
        "row_cell_geometry_available": bool(cell_bboxes or row_bboxes),
        "evidence_units": evidence_units,
    }
    bundle_lines = ["[Structured Table Bundle]"]
    if caption:
        bundle_lines.extend(["", caption])
    if table_id:
        bundle_lines.extend(["", "[Table ID]", table_id])
    if header:
        bundle_lines.extend(["", "[Header]", header])
    if body_markdown:
        bundle_lines.extend(["", "[Body]", body_markdown])
    bundle["bundle_text"] = "\n".join(bundle_lines).strip()

    page_text_parts = []
    page_text_parts.append(page_markdown)
    if footnote:
        page_text_parts.extend(["[Table Footnote]", footnote])
    return bundle, "\n".join(page_text_parts).strip(), warnings


def _table_pages_for_item(item: dict[str, Any], page_num: int) -> list[int]:
    pages = [page_num]
    raw_pages = item.get("_table_pages")
    if isinstance(raw_pages, list):
        for value in raw_pages:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0 and page not in pages:
                pages.append(page)
    return sorted(pages)


def _html_table_to_matrix(table_html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(table_html or "")
    parser.close()
    return _expand_spans(parser.rows)


def _expand_spans(rows: list[list[dict[str, Any]]]) -> list[list[str]]:
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    max_cols = 0

    for row_idx, row in enumerate(rows):
        grid_row: list[str] = []
        col_idx = 0

        def fill_pending() -> None:
            nonlocal col_idx
            while (row_idx, col_idx) in pending:
                grid_row.append(pending.pop((row_idx, col_idx)))
                col_idx += 1

        fill_pending()
        for cell in row:
            fill_pending()
            text = _clean_cell_text(cell.get("text", ""))
            rowspan = max(1, int(cell.get("rowspan") or 1))
            colspan = max(1, int(cell.get("colspan") or 1))
            for dx in range(colspan):
                grid_row.append(text)
                if rowspan > 1:
                    for dy in range(1, rowspan):
                        pending[(row_idx + dy, col_idx + dx)] = text
            col_idx += colspan
        fill_pending()
        max_cols = max(max_cols, len(grid_row))
        grid.append(grid_row)

    # Flush rows created only by rowspan.
    row_idx = len(grid)
    while any(key[0] == row_idx for key in pending):
        grid_row = []
        col_idx = 0
        while any(key[0] == row_idx and key[1] >= col_idx for key in pending):
            if (row_idx, col_idx) in pending:
                grid_row.append(pending.pop((row_idx, col_idx)))
            else:
                grid_row.append("")
            col_idx += 1
        max_cols = max(max_cols, len(grid_row))
        grid.append(grid_row)
        row_idx += 1

    if max_cols <= 0:
        return []
    for row in grid:
        while len(row) < max_cols:
            row.append("")
    return grid


def _matrix_to_markdown(matrix: list[list[str]], *, caption: str = "") -> str:
    if not matrix:
        return ""
    max_cols = max(len(row) for row in matrix)
    normalized_rows: list[list[str]] = []
    for row in matrix:
        cells = [_clean_cell_text(cell).replace("|", "\\|") for cell in row]
        while len(cells) < max_cols:
            cells.append("")
        normalized_rows.append(cells)

    lines: list[str] = []
    if caption:
        lines.append(f"[TABLE] {caption}")
        lines.append("")
    lines.append("| " + " | ".join(normalized_rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in normalized_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_header_bound_row_text(header_cells: list[str], row: list[str]) -> str:
    """Build self-contained row text as ``header: value`` pairs."""
    parts: list[str] = []
    fallback_parts: list[str] = []
    for idx, raw_value in enumerate(row):
        value = _clean_cell_text(raw_value)
        if not value:
            continue
        fallback_parts.append(value)
        header = _clean_cell_text(header_cells[idx] if idx < len(header_cells) else "")
        if not header:
            header = f"Column {idx + 1}"
        if header == value:
            parts.append(value)
        else:
            parts.append(f"{header}: {value}")
    return "; ".join(parts).strip() or " | ".join(fallback_parts).strip()


def _mineru_column_header_paths(matrix: list[list[str]]) -> list[str]:
    if not matrix:
        return []
    header_depth = 1
    if len(matrix) >= 2:
        first = [_clean_cell_text(cell) for cell in matrix[0]]
        second = [_clean_cell_text(cell) for cell in matrix[1]]
        first_has_merged_hint = len(set(cell for cell in first if cell)) < len([cell for cell in first if cell])
        second_numeric_ratio = 0.0
        second_values = [cell for cell in second if cell]
        if second_values:
            numeric_count = sum(1 for cell in second_values if re.fullmatch(r"[-+−]?\d+(?:[.,]\d+)?%?", cell))
            second_numeric_ratio = numeric_count / len(second_values)
        if first_has_merged_hint and second_values and second_numeric_ratio < 0.45:
            header_depth = 2

    header_rows = matrix[:header_depth]
    max_cols = max(len(row) for row in matrix)
    paths: list[str] = []
    for col_idx in range(max_cols):
        parts: list[str] = []
        for row in header_rows:
            part = _clean_cell_text(row[col_idx] if col_idx < len(row) else "")
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        paths.append(" ".join(parts).strip() or f"Column {col_idx + 1}")
    return paths


def _build_table_evidence_units(
    *,
    bundle_id: str,
    table_id: str,
    caption: str,
    header: str,
    matrix: list[list[str]],
    page_num: int,
    bbox: list[float],
    page_size: list[float],
    cell_bboxes: dict[tuple[int, int], list[float]] | None = None,
    row_bboxes: dict[int, list[float]] | None = None,
) -> list[dict[str, Any]]:
    evidence_units: list[dict[str, Any]] = []
    geometry = visual_geometry(
        bbox,
        coordinate_space="normalized_0_1000",
        page_size=page_size,
    )
    header_cells = matrix[0] if matrix else []
    header_paths = _mineru_column_header_paths(matrix)
    cell_bboxes = cell_bboxes or {}
    row_bboxes = row_bboxes or {}
    for row_idx, row in enumerate(matrix, start=1):
        is_header = row_idx == 1
        cell_units: list[dict[str, Any]] = []
        for col_idx, cell_text in enumerate(row, start=1):
            header_path = header_paths[col_idx - 1] if col_idx - 1 < len(header_paths) else f"Column {col_idx}"
            cell_bbox = cell_bboxes.get((row_idx, col_idx))
            cell_geometry = visual_geometry(
                cell_bbox or bbox,
                coordinate_space="normalized_0_1000",
                page_size=page_size,
            )
            cell_units.append({
                "evidence_unit_id": f"{bundle_id}::table_cell::p{page_num}::r{row_idx}::c{col_idx}",
                "evidence_unit_type": "table_cell",
                "table_bundle_id": bundle_id,
                "table_id": table_id,
                "table_caption": caption,
                "table_header": header,
                "page": page_num,
                "row_idx": row_idx,
                "row_number": row_idx,
                "col_idx": col_idx,
                "column_number": col_idx,
                "col_id": header_path,
                "column_header": header_path,
                "header_path": header_path,
                "bbox": cell_bbox or bbox,
                "bounding_box": cell_bbox or bbox,
                **cell_geometry,
                "geometry_source": "mineru_cell" if cell_bbox else "table_fallback",
                "visual_crop_eligible": bool(cell_bbox),
                "cell_text": cell_text,
                "content": cell_text,
                "is_header_row": is_header,
                "source": "mineru",
            })
        raw_row_text = " | ".join(cell for cell in row if cell).strip()
        row_text = raw_row_text if is_header else _build_header_bound_row_text(header_cells, row)
        row_bbox = row_bboxes.get(row_idx)
        if not row_bbox and row and all(cell_bboxes.get((row_idx, col_idx)) for col_idx in range(1, len(row) + 1)):
            row_bbox = _merge_normalized_bboxes([cell_bboxes[(row_idx, col_idx)] for col_idx in range(1, len(row) + 1)])
        row_geometry = visual_geometry(
            row_bbox or bbox,
            coordinate_space="normalized_0_1000",
            page_size=page_size,
        )
        evidence_units.append({
            "evidence_unit_id": f"{bundle_id}::table_row::p{page_num}::r{row_idx}",
            "evidence_unit_type": "table_row",
            "table_bundle_id": bundle_id,
            "table_id": table_id,
            "table_caption": caption,
            "table_header": header,
            "page": page_num,
            "row_idx": row_idx,
            "row_number": row_idx,
            "bbox": row_bbox or bbox,
            "bounding_box": row_bbox or bbox,
            **row_geometry,
            "geometry_source": "mineru_row" if row_bbox else "table_fallback",
            "visual_crop_eligible": bool(row_bbox),
            "row_id": next((cell for cell in row if cell), ""),
            "row_text": row_text,
            "raw_row_text": raw_row_text,
            "row_numbers": " ".join(cell for cell in row[1:] if cell),
            "content": row_text,
            "is_header_row": is_header,
            "cell_count": len(cell_units),
            "cell_evidence_unit_ids": [cell["evidence_unit_id"] for cell in cell_units],
            "cell_evidence_units": cell_units,
            "source": "mineru",
        })
    return evidence_units


def _extract_trustworthy_table_geometry(
    item: dict[str, Any],
    matrix: list[list[str]],
    table_bbox: list[float],
) -> tuple[dict[tuple[int, int], list[float]], dict[int, list[float]]]:
    """Read only explicit MinerU table geometry; never infer it from HTML.

    MinerU variants expose this either as ``table_cells``/``cells`` or
    ``table_rows``/``rows``.  Coordinates must be normalized 0--1000 and lie
    inside the declared table bbox.  Otherwise row/cell retrieval stays tied
    to the whole-table overview and is marked non-croppable.
    """
    cell_bboxes: dict[tuple[int, int], list[float]] = {}
    row_bboxes: dict[int, list[float]] = {}
    rows = len(matrix)
    cols = max((len(row) for row in matrix), default=0)
    if not table_bbox or not rows or not cols:
        return cell_bboxes, row_bboxes

    def valid_bbox(value: Any) -> list[float]:
        bbox = _normalize_bbox(value)
        if not bbox or any(value < 0 or value > 1000 for value in bbox):
            return []
        if bbox[0] < table_bbox[0] or bbox[1] < table_bbox[1] or bbox[2] > table_bbox[2] or bbox[3] > table_bbox[3]:
            return []
        return bbox

    def normalized_index(value: Any, limit: int, *, zero_based: bool) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        if zero_based:
            index += 1
        return index if 1 <= index <= limit else 0

    raw_cells = item.get("table_cells") or item.get("cells") or item.get("cell_bboxes") or []
    if isinstance(raw_cells, list):
        zero_based = any(isinstance(cell, dict) and (cell.get("row_idx") == 0 or cell.get("col_idx") == 0) for cell in raw_cells)
        for cell in raw_cells:
            if not isinstance(cell, dict):
                continue
            row = normalized_index(cell.get("row_idx", cell.get("row_index", cell.get("row"))), rows, zero_based=zero_based)
            col = normalized_index(cell.get("col_idx", cell.get("col_index", cell.get("column", cell.get("col")))), cols, zero_based=zero_based)
            bbox = valid_bbox(cell.get("bbox") or cell.get("cell_bbox"))
            if row and col and bbox:
                cell_bboxes[(row, col)] = bbox

    raw_rows = item.get("table_rows") or item.get("rows") or item.get("row_bboxes") or []
    if isinstance(raw_rows, list):
        zero_based = any(isinstance(row, dict) and row.get("row_idx") == 0 for row in raw_rows)
        for row_data in raw_rows:
            if not isinstance(row_data, dict):
                continue
            row = normalized_index(row_data.get("row_idx", row_data.get("row_index", row_data.get("row"))), rows, zero_based=zero_based)
            bbox = valid_bbox(row_data.get("bbox") or row_data.get("row_bbox"))
            if row and bbox:
                row_bboxes[row] = bbox
    return cell_bboxes, row_bboxes


def _merge_normalized_bboxes(bboxes: list[list[float]]) -> list[float]:
    return [
        min(bbox[0] for bbox in bboxes), min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes), max(bbox[3] for bbox in bboxes),
    ]


def _quality_report(
    *,
    kept_counts: dict[str, int],
    raw_type_counts: dict[str, int] | None = None,
    filtered_counts: dict[str, int],
    table_count: int,
    evidence_unit_count: int,
    malformed_table_count: int,
    warnings: list[str],
    failure_reasons: list[str],
    page_quality: dict[str, Any] | None = None,
    unknown_types: set[str] | None = None,
) -> dict[str, Any]:
    raw_counts = dict(sorted((raw_type_counts or {}).items()))
    emitted_counts = dict(sorted(kept_counts.items()))
    report = {
        "kept_type_counts": emitted_counts,
        "raw_type_counts": raw_counts,
        "emitted_type_counts": emitted_counts,
        "filtered_type_counts": dict(sorted(filtered_counts.items())),
        "unknown_types": sorted(str(value) for value in (unknown_types or set()) if value),
        "table_count": table_count,
        "evidence_unit_count": evidence_unit_count,
        "malformed_table_count": malformed_table_count,
        "warnings": warnings,
        "failure_reasons": failure_reasons,
    }
    if isinstance(page_quality, dict):
        report.update({
            "quality_status": page_quality.get("quality_status", "failed"),
            "expected_page_count": page_quality.get("expected_page_count", 0),
            "observed_page_count": page_quality.get("observed_page_count", 0),
            "covered_page_count": page_quality.get("covered_page_count", 0),
            "coverage": page_quality.get("coverage", 0.0),
            "failed_pages": list(page_quality.get("failed_pages") or []),
            "unexpected_pages": list(page_quality.get("unexpected_pages") or []),
            "page_ledger": list(page_quality.get("page_ledger") or []),
        })
    return report


def _payload_page_numbers(payload: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    content = payload.get("content_list_json")
    if isinstance(content, list):
        pages.update(_page_num(item) for item in content if isinstance(item, dict))

    middle = payload.get("middle_json")
    if isinstance(middle, list):
        pages.update(_page_num(item) for item in middle if isinstance(item, dict))
    elif isinstance(middle, dict):
        pdf_info = middle.get("pdf_info")
        if isinstance(pdf_info, list):
            pages.update(_page_num(item) for item in pdf_info if isinstance(item, dict))
    return {page for page in pages if page > 0}


def _reported_failed_page_numbers(payload: dict[str, Any]) -> set[int]:
    values: list[Any] = []
    for key in ("failed_pages", "fail_pages", "failed_page_numbers"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
    return _positive_page_numbers(values)


def _positive_page_numbers(values: Any) -> set[int]:
    result: set[int] = set()
    if values is None:
        return result
    try:
        iterator = iter(values)
    except TypeError:
        iterator = iter((values,))
    for value in iterator:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            result.add(page)
    return result


def _inc(counter: dict[str, int], key: str) -> None:
    key = key or "unknown"
    counter[key] = counter.get(key, 0) + 1


def _extract_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, str) and child.strip():
                    parts.append(child.strip())
                elif isinstance(child, dict):
                    nested = _extract_text(
                        child,
                        ("text", "content", "html", "latex", "table_body", "blocks", "lines", "spans"),
                    )
                    if nested:
                        parts.append(nested)
    return "\n".join(_dedupe(parts)).strip()


def _format_equation(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    if value.startswith("$") or value.startswith("\\[") or value.startswith("\\("):
        return value
    return f"$$\n{value}\n$$"


def _table_markdown_has_consistent_columns(markdown: str) -> bool:
    rows = []
    for line in (markdown or "").splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        if set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        rows.append([cell.strip() for cell in _split_markdown_table_row(line)])
    if not rows:
        return bool(not markdown.strip())
    col_count = len(rows[0])
    return all(len(row) == col_count for row in rows)


def _split_markdown_table_row(line: str) -> list[str]:
    value = str(line or "").strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current))
            current = []
            continue
        current.append(char)
    cells.append("".join(current))
    return cells


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _extract_table_reference(caption: str) -> str:
    match = re.search(r"(?:Table|TABLE|表)\s*\.?\s*([A-Za-z0-9IVXLC]+)", caption or "", re.IGNORECASE)
    return f"Table {match.group(1)}" if match else ""


def _normalize_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _page_num(item: dict[str, Any]) -> int:
    if "page_idx" in item:
        try:
            return max(1, int(item["page_idx"]) + 1)
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(item.get("page") or item.get("page_id") or 1))
    except (TypeError, ValueError):
        return 1


def _resolve_page_size(
    item: dict[str, Any],
    page_num: int,
    page_sizes: dict[int, list[float] | tuple[float, float]] | None,
) -> list[float]:
    """Prefer the original PDF page size; keep item metadata as a fallback."""
    if isinstance(page_sizes, dict):
        known = normalize_page_size(page_sizes.get(page_num))
        if known:
            return known
    for key in ("page_size", "pageSize", "page_dimensions"):
        known = normalize_page_size(item.get(key))
        if known:
            return known
    return []


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    bbox = []
    for item in value:
        if isinstance(item, (int, float)):
            bbox.append(float(item))
    return bbox[:4] if len(bbox) >= 4 else []


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _clean_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = html.unescape(value)
    value = re.sub(r"<sup\b[^>]*>(.*?)</sup>", r"^\1", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<sub\b[^>]*>(.*?)</sub>", r"_\1", value, flags=re.IGNORECASE | re.DOTALL)
    value = _strip_html_tags(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _clean_cell_text(text: str) -> str:
    value = _clean_text(text)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(text or ""))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = re.sub(r"\s+", " ", value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
