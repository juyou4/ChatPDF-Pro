"""Deterministic citation-quality measurements for regression and runtime audits."""

from __future__ import annotations

import re
from typing import Any

from services.citation_alignment_service import (
    claim_support_score,
    citation_support_text,
    extract_atomic_claims,
    strip_inline_citations,
)


def _inline_refs(text: str) -> list[int]:
    refs: list[int] = []
    for match in re.finditer(r"(?<![A-Za-z_])(?:\[(\d{1,3})\]|【(\d{1,3})】)", str(text or "")):
        value = int(match.group(1) or match.group(2))
        if value not in refs:
            refs.append(value)
    return refs


def _ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    if denominator <= 0:
        return empty
    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def _span_is_in_source(citation: dict, span: str) -> bool:
    if not span:
        return False
    source = citation_support_text(citation)
    if not source:
        return False
    compact = lambda value: re.sub(r"\s+", "", str(value or "")).lower()
    return compact(span) in compact(source)


def _is_factual_claim(claim: dict) -> bool:
    text = strip_inline_citations(str(claim.get("claim_text") or "")).strip()
    if not text:
        return False
    # Questions and UI-style requests are not answer claims and should not
    # lower completeness when a caller measures an intermediate draft.
    return not bool(re.match(r"^(?:请(?:问|说明|解释)?|说明|解释|什么|哪些|如何|是否|能否)", text))


def compute_citation_quality_metrics(
    answer: str,
    citations: list[dict],
    *,
    verifier: dict | None = None,
    min_support_score: float = 0.24,
) -> dict[str, Any]:
    """Compute correctness/completeness without relying on a model or a gold set.

    These are operational metrics, not a replacement for a human-labelled
    benchmark.  An optional verifier result adds the contradiction escape rate;
    all other measures are derived from the final answer and authorized
    citation records.
    """

    citation_map: dict[int, dict] = {}
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref"))
        except (TypeError, ValueError):
            continue
        if ref > 0:
            citation_map[ref] = citation

    claims = [claim for claim in extract_atomic_claims(answer) if _is_factual_claim(claim)]
    cited_claim_count = 0
    supported_claim_count = 0
    cited_ref_count = 0
    unsupported_ref_count = 0
    span_total = 0
    span_exact = 0
    for claim in claims:
        refs = [ref for ref in claim.get("existing_refs") or [] if ref in citation_map]
        if not refs:
            continue
        cited_claim_count += 1
        cited_ref_count += len(refs)
        supported_refs = [
            ref
            for ref in refs
            if claim_support_score(claim.get("claim_text") or "", citation_map[ref])
            >= max(0.0, float(min_support_score))
        ]
        if supported_refs:
            supported_claim_count += 1
        unsupported_ref_count += max(0, len(refs) - len(supported_refs))
        for ref in supported_refs:
            citation = citation_map[ref]
            spans = citation.get("support_spans") if isinstance(citation.get("support_spans"), list) else []
            claim_spans = [
                item for item in spans
                if isinstance(item, dict) and str(item.get("claim_id") or "") == str(claim.get("claim_id") or "")
            ]
            span = str(
                (claim_spans[0] if claim_spans else {}).get("text")
                or citation.get("support_span")
                or ""
            ).strip()
            span_total += 1
            if _span_is_in_source(citation, span):
                span_exact += 1

    contradiction_total = 0
    contradiction_escaped = 0
    if isinstance(verifier, dict):
        claim_by_id = {str(claim.get("claim_id")): claim for claim in claims}
        for verdict in verifier.get("verdicts") or []:
            if not isinstance(verdict, dict):
                continue
            if str(verdict.get("status") or "").lower() != "contradicted":
                continue
            contradiction_total += 1
            claim = claim_by_id.get(str(verdict.get("claim_id") or ""))
            if claim and _inline_refs(claim.get("raw_text") or ""):
                contradiction_escaped += 1

    factual_claim_count = len(claims)
    return {
        "citation_correctness": _ratio(supported_claim_count, cited_claim_count),
        "citation_completeness": _ratio(
            supported_claim_count,
            factual_claim_count,
            empty=1.0,
        ),
        "overcitation": _ratio(unsupported_ref_count, cited_ref_count, empty=0.0),
        "span_precision": _ratio(span_exact, span_total),
        "contradiction_escape_rate": _ratio(
            contradiction_escaped,
            contradiction_total,
            empty=0.0,
        ),
        "counts": {
            "claim_count": factual_claim_count,
            "cited_claim_count": cited_claim_count,
            "supported_claim_count": supported_claim_count,
            "cited_ref_count": cited_ref_count,
            "unsupported_ref_count": unsupported_ref_count,
            "span_total": span_total,
            "span_exact": span_exact,
            "contradiction_total": contradiction_total,
            "contradiction_escaped": contradiction_escaped,
        },
        "threshold": round(max(0.0, float(min_support_score)), 4),
    }
