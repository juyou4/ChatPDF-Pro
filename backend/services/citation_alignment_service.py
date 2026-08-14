"""Deterministic claim-to-evidence alignment helpers.

The retrieval query and an answer sentence answer different questions.  This
module keeps those scores separate and provides a conservative, dependency-free
alignment pass that can run after generation and before citations are published.
It is intentionally a gate, not a replacement for the optional LLM verifier.
"""

from __future__ import annotations

import re
from typing import Any


_INLINE_CITATION_RE = re.compile(r"(?<![A-Za-z_])(?:\[(\d{1,3})\]|【(\d{1,3})】)")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:\s*%|e[-+]?\d+)?")
_CLAUSE_RE = re.compile(
    r".+?(?:[。！？!?；;]|(?:(?:，|,)(?=\s*(?:但|然而|同时|因此|而|且|可是|不过|but\b|however\b|while\b|whereas\b|therefore\b)))|(?<!\d)\.(?=\s|$)|$)",
    re.S,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|unable|fail|failed|lack|lacks)\b"
    r"|(?:没有|未|无法|不能|不可|不支持|不具备|缺乏|无)"
)
_POSITIVE_DIRECTION_RE = re.compile(
    r"\b(?:better|higher|greater|more|improv(?:e|ed|es|ement)|increase(?:d|s)?|above|outperform(?:ed|s)?)\b"
    r"|(?:优于|高于|提升|提高|增加|超过|领先|更好|较高)"
)
_NEGATIVE_DIRECTION_RE = re.compile(
    r"\b(?:worse|lower|less|decreas(?:e|ed|es|ing)|below|underperform(?:ed|s)?)\b"
    r"|(?:低于|下降|减少|少于|不如|落后|更低|较差)"
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|thus|hence|caus(?:e|al|ed)|leads? to|results? in|due to)\b"
    r"|(?:因此|从而|导致|造成|使得|由于|归因于|因而|说明)"
)
_LIMITATION_RE = re.compile(
    r"\b(?:limitation|limitations|constraint|constrain(?:ed|s)?|caveat|cannot generalize|future work)\b"
    r"|(?:局限|限制|不足|约束|无法泛化|未来工作|缺点|问题在于)"
)
_SUPERLATIVE_RE = re.compile(
    r"\b(?:best|worst|highest|lowest|largest|smallest|state[- ]of[- ]the[- ]art|significant(?:ly)?)\b"
    r"|(?:最高|最低|最大|最小|最佳|最差|显著|领先|唯一|首次)"
)
_COMPARISON_OP_RE = re.compile(
    r"\b(?:better\s+than|worse\s+than|higher\s+than|lower\s+than|"
    r"greater\s+than|less\s+than|outperform(?:s|ed)?|underperform(?:s|ed)?)\b"
    r"|(?:优于|高于|超过|领先|不如|低于|少于|落后|更高|更低|较高|较低)",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"(?:\b(?:on|in|for|under|within|across|dataset|benchmark)\b|在|于|针对|对于|数据集|基准)"
    r"\s*[:：]?\s*([A-Za-z][A-Za-z0-9_.+/-]*|[\u4e00-\u9fff]{2,16})",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with", "what", "which", "why", "where", "when", "do", "does",
    "请", "什么", "哪些", "如何", "是否", "这个", "那个", "论文", "方法", "结果",
    "主要", "内容", "说明", "解释", "可以", "进行", "使用", "采用",
}
_GENERIC_ENTITY_TERMS = {
    "method", "model", "baseline", "approach", "system", "paper", "authors",
    "result", "results", "accuracy", "precision", "recall", "score", "metric",
    "方法", "模型", "基线", "结果", "准确率", "性能", "指标", "论文", "作者",
}
_CONSERVATIVE_CLAIM_PREFIX_RE = re.compile(
    r"^(?:根据当前已授权证据，无法确认[:：]|"
    r"根据当前检索证据，无法确认[:：]|"
    r"当前已授权证据与该陈述不一致[，,：:]?)"
)


def strip_inline_citations(text: str = "") -> str:
    """Remove only citation markers, preserving mathematical ``x[1]`` text."""

    return re.sub(r"(?<![A-Za-z_])(?:\[\d{1,3}\]|【\d{1,3}】)", "", str(text or "")).strip()


_SOURCE_REFERENCE_GROUP_RE = re.compile(
    r"(?<![A-Za-z_])[\[【]\d{1,3}(?:\s*[,，、\-–—]\s*\d{1,3})+[\]】]"
)


def strip_source_reference_markers(text: str = "") -> str:
    """Remove the source paper's own numeric reference markers from evidence text.

    Academic PDFs are full of ``[12]`` / ``[3,4]`` markers that use the same
    syntax as our citation numbers. Inside a prompt they are indistinguishable
    from the evidence list's ``[n]`` prefixes, so a model can copy one and
    produce a citation that passes every format and range check while pointing
    at unrelated evidence.

    Author-year forms such as ``(Zhang et al., 2019)`` are deliberately left
    alone: they cannot be mistaken for our ``[n]`` syntax, and any regex wide
    enough to catch them also eats real data like ``(n=120, 2020 cohort)``.

    Callers that compare model-produced spans against stored evidence must run
    the stored text through this same function, otherwise the span will carry
    no marker while the source still does and the match will fail.
    """
    return strip_inline_citations(_SOURCE_REFERENCE_GROUP_RE.sub("", str(text or "")))


def tokenize(text: str = "", *, informative_only: bool = False) -> list[str]:
    lowered = str(text or "").lower()
    atoms = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    if informative_only:
        atoms = [atom for atom in atoms if atom not in _STOPWORDS]
    bigrams = []
    for index in range(len(atoms) - 1):
        left, right = atoms[index], atoms[index + 1]
        if len(left) == 1 and len(right) == 1 and re.match(r"[\u4e00-\u9fff]", left) and re.match(r"[\u4e00-\u9fff]", right):
            bigrams.append(left + right)
    return [*atoms, *bigrams]


def _claim_numbers(text: str) -> set[str]:
    values = set()
    for raw in _NUMBER_RE.findall(str(text or "")):
        value = re.sub(r"\s+", "", raw).lower()
        if value:
            values.add(value)
    return values


def _direction(text: str) -> int:
    value = str(text or "").lower()
    positive = bool(_POSITIVE_DIRECTION_RE.search(value))
    negative = bool(_NEGATIVE_DIRECTION_RE.search(value))
    if positive == negative:
        return 0
    return 1 if positive else -1


def _entity_terms(text: str) -> set[str]:
    """Extract stable method/dataset labels for wrong-row detection.

    General Chinese prose is intentionally ignored here.  The guard focuses
    on names that identify a row or comparison subject (``Ours``, ``RIDE``,
    ``ImageNet-LT`` and explicit ``方法: X`` labels), so a neighboring table
    row cannot inherit a citation merely because it shares the metric name.
    """
    value = str(text or "").lower()
    terms = {
        token.strip("._+-")
        for token in re.findall(r"[a-z][a-z0-9_.+/-]*", value)
        if len(token.strip("._+-")) >= 2
    }
    terms -= _GENERIC_ENTITY_TERMS
    for match in re.finditer(
        r"(?:方法|模型|基线|method|model|baseline)\s*(?:为|是|[:：])?\s*"
        r"([a-z][a-z0-9_.+/-]*|[\u4e00-\u9fff]{2,16})",
        value,
        re.IGNORECASE,
    ):
        candidate = match.group(1).strip("._+-").lower()
        if candidate and candidate not in _GENERIC_ENTITY_TERMS and len(candidate) >= 2:
            terms.add(candidate)
    return terms


def _operand_key(text: str) -> frozenset[str]:
    terms = _entity_terms(text)
    if terms:
        return frozenset(terms)
    tokens = [token for token in tokenize(text, informative_only=True) if token not in _STOPWORDS]
    return frozenset(tokens[-6:])


def _comparison_relations(text: str) -> list[tuple[frozenset[str], frozenset[str], int]]:
    """Return ``(subject, object, direction)`` tuples for explicit comparisons."""
    value = str(text or "")
    relations: list[tuple[frozenset[str], frozenset[str], int]] = []
    direction_map = {
        "better": 1, "higher": 1, "greater": 1, "outperform": 1,
        "worse": -1, "lower": -1, "less": -1, "underperform": -1,
        "优于": 1, "高于": 1, "超过": 1, "领先": 1, "更高": 1, "较高": 1,
        "不如": -1, "低于": -1, "少于": -1, "落后": -1, "更低": -1, "较低": -1,
    }
    for match in _COMPARISON_OP_RE.finditer(value):
        raw_operator = match.group(0).lower()
        operator = next((key for key in direction_map if key in raw_operator), "")
        direction = direction_map.get(operator, 0)
        if not direction:
            continue
        left = re.split(r"[。！？!?；;，,]|\b(?:and|while|whereas|but)\b|但|然而", value[:match.start()], flags=re.IGNORECASE)[-1]
        right = re.split(r"[。！？!?；;，,]|\b(?:and|while|whereas|but)\b|但|然而", value[match.end():], flags=re.IGNORECASE)[0]
        subject = _operand_key(left)
        object_ = _operand_key(right)
        if subject and object_ and subject != object_:
            relations.append((subject, object_, direction))
    return relations


def _comparison_mismatch(claim: str, support: str) -> bool:
    claim_relations = _comparison_relations(claim)
    support_relations = _comparison_relations(support)
    if not claim_relations or not support_relations:
        return False
    for claim_subject, claim_object, claim_direction in claim_relations:
        for support_subject, support_object, support_direction in support_relations:
            same_order = bool(claim_subject & support_subject) and bool(claim_object & support_object)
            reversed_order = bool(claim_subject & support_object) and bool(claim_object & support_subject)
            if reversed_order and not same_order:
                return True
            if same_order and claim_direction != support_direction:
                return True
    return False


def _condition_mismatch(claim: str, support: str) -> bool:
    claim_conditions = {item.lower() for item in _CONDITION_RE.findall(str(claim or ""))}
    support_conditions = {item.lower() for item in _CONDITION_RE.findall(str(support or ""))}
    return bool(claim_conditions and support_conditions and claim_conditions.isdisjoint(support_conditions))


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(str(text or "").lower()))


def citation_support_text(citation: dict | None) -> str:
    """Collect proof-bearing fields without treating opaque IDs as evidence."""

    if not isinstance(citation, dict):
        return ""
    parts = [
        citation.get("support_span"),
        citation.get("highlight_text"),
        citation.get("source_text"),
        citation.get("display_text"),
        citation.get("_full_text"),
        citation.get("numeric_table_exact_context_row_text"),
        citation.get("numeric_table_exact_context_caption"),
        citation.get("numeric_table_exact_context_header"),
        citation.get("table_caption"),
        citation.get("table_header"),
        citation.get("row_text"),
    ]
    for collection_key in ("evidence_units", "cell_evidence_units"):
        for unit in citation.get(collection_key) or []:
            if isinstance(unit, dict):
                parts.extend((unit.get("content"), unit.get("text"), unit.get("value")))
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


def query_relevance_score(query: str, evidence: str | dict) -> float:
    """Score whether evidence helps answer the query (not whether it supports a claim)."""

    query_tokens = set(tokenize(query, informative_only=True))
    evidence_text = citation_support_text(evidence) if isinstance(evidence, dict) else str(evidence or "")
    evidence_tokens = set(tokenize(evidence_text, informative_only=True))
    if not query_tokens or not evidence_tokens:
        return 0.0
    overlap = len(query_tokens & evidence_tokens) / max(1, len(query_tokens))
    query_compact = re.sub(r"\s+", "", str(query or "").lower())
    evidence_compact = re.sub(r"\s+", "", evidence_text.lower())
    phrase_bonus = 0.18 if len(query_compact) >= 8 and query_compact in evidence_compact else 0.0
    return max(0.0, min(1.0, overlap * 0.82 + phrase_bonus))


def claim_support_score(claim: str, citation: dict | None) -> float:
    """Estimate direct support for one claim using lexical and structural guards."""

    claim_text = strip_inline_citations(claim)
    support_text = citation_support_text(citation)
    claim_tokens = set(tokenize(claim_text, informative_only=True))
    support_tokens = set(tokenize(support_text, informative_only=True))
    if not claim_tokens or not support_tokens:
        return 0.0

    overlap = len(claim_tokens & support_tokens) / max(1, len(claim_tokens))
    score = overlap * 0.68

    claim_numbers = _claim_numbers(claim_text)
    if claim_numbers:
        support_numbers = _claim_numbers(support_text)
        number_ratio = len(claim_numbers & support_numbers) / max(1, len(claim_numbers))
        score += number_ratio * 0.28
        if number_ratio < 1.0:
            score -= 0.22 * (1.0 - number_ratio)

    compact_claim = re.sub(r"\s+", "", claim_text.lower())
    compact_support = re.sub(r"\s+", "", support_text.lower())
    if len(compact_claim) >= 10 and compact_claim in compact_support:
        score += 0.24

    # Figure/table identity is direct evidence even when the answer and
    # caption use different languages.  It prevents a correct “图 2” caption
    # from being discarded solely because Chinese and English tokens differ.
    claim_object = re.search(r"(?:figure|fig\.?|table|image|chart|图|表|图片)\s*[-_:：#]?\s*(\d+)", claim_text, re.IGNORECASE)
    citation_identity = " ".join(
        str(citation.get(key) or "")
        for key in ("group_id", "context_id", "table_id", "figure_id", "asset_id")
    ) if isinstance(citation, dict) else ""
    if claim_object and re.search(
        rf"(?:figure|fig\.?|table|image|chart|图|表|图片)\s*[-_:：#]?\s*{re.escape(claim_object.group(1))}",
        citation_identity,
        re.IGNORECASE,
    ):
        score += 0.3

    if _has_negation(claim_text) != _has_negation(support_text):
        score -= 0.2
    claim_direction = _direction(claim_text)
    support_direction = _direction(support_text)
    if claim_direction and support_direction and claim_direction != support_direction:
        score -= 0.22
    if _comparison_mismatch(claim_text, support_text):
        # ``A > B`` and ``B > A`` can share every metric token; subject/object
        # order is therefore a stronger signal than lexical overlap.
        score -= 0.34
    claim_entities = _entity_terms(claim_text)
    support_entities = _entity_terms(support_text)
    if claim_entities and support_entities and claim_entities.isdisjoint(support_entities):
        # Prevent a same-table neighboring row (same metric and number) from
        # being accepted solely because its method name was omitted from the
        # generated sentence.
        score -= 0.28
    if _condition_mismatch(claim_text, support_text):
        score -= 0.18

    return max(0.0, min(1.0, score))


def _evidence_is_complementary(claim: str, first: dict, second: dict) -> bool:
    """Require genuinely different proof before permitting a second citation."""
    first_text = citation_support_text(first)
    second_text = citation_support_text(second)
    if not first_text or not second_text:
        return False
    first_tokens = set(tokenize(first_text, informative_only=True))
    second_tokens = set(tokenize(second_text, informative_only=True))
    if not first_tokens or not second_tokens:
        return False
    overlap = len(first_tokens & second_tokens) / max(1, min(len(first_tokens), len(second_tokens)))
    if overlap >= 0.9:
        return False
    claim_entities = _entity_terms(claim)
    first_entities = _entity_terms(first_text)
    second_entities = _entity_terms(second_text)
    if claim_entities and first_entities and second_entities:
        # Both records must contribute at least one distinct claim-relevant
        # entity/condition; duplicate rows with different IDs are not enough.
        first_hits = claim_entities & first_entities
        second_hits = claim_entities & second_entities
        if first_hits and second_hits and first_hits == second_hits:
            return False
    return True


def _is_conservative_claim(claim: str) -> bool:
    """Conservative verifier fallbacks must stay citation-free.

    The fallback intentionally retains the rejected claim for transparency.
    Re-running lexical alignment on that retained text would otherwise attach
    the same evidence again and undo the fail-closed decision.
    """
    return bool(_CONSERVATIVE_CLAIM_PREFIX_RE.search(strip_inline_citations(claim).strip()))


def rank_claim_evidence(
    claim: str,
    citations: list[dict],
) -> list[tuple[float, int, dict]]:
    """Coarse-rank authorized records for one claim.

    Hybrid BM25/vector retrieval and any upstream reranker already run before
    this service.  If a record carries an explicit claim/reranker score, it is
    used only as a tie-break after the deterministic claim-support score; the
    query-relevance score is deliberately never treated as claim support.
    """
    scored: list[tuple[float, float, int, dict]] = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref"))
        except (TypeError, ValueError):
            continue
        if ref <= 0:
            continue
        support = claim_support_score(claim, citation)
        rerank = 0.0
        for key in ("claim_rerank_score", "alignment_rerank_score", "rerank_score"):
            raw = citation.get(key)
            if isinstance(raw, (int, float)):
                rerank = max(0.0, min(1.0, float(raw)))
                break
        scored.append((support, rerank, ref, citation))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [(support, ref, citation) for support, _rerank, ref, citation in scored]


def claim_risk_reasons(claim: str, *, top_score: float = 0.0, score_gap: float | None = None) -> list[str]:
    """Return deterministic reasons for sending a claim to the semantic verifier.

    The verifier is intentionally selective.  Numeric, directional, causal,
    negative, limitation and superlative claims are cheap to identify locally
    and expensive to get wrong.  A small score gap is also risky because the
    top two chunks may belong to adjacent rows/sections.
    """

    text = strip_inline_citations(claim)
    reasons: list[str] = []
    if _claim_numbers(text):
        reasons.append("number")
    if _direction(text):
        reasons.append("comparison_direction")
    if _has_negation(text):
        reasons.append("negation")
    if _CAUSAL_RE.search(text):
        reasons.append("causal")
    if _LIMITATION_RE.search(text):
        reasons.append("limitation")
    if _SUPERLATIVE_RE.search(text):
        reasons.append("superlative")
    if score_gap is not None and score_gap < 0.12:
        reasons.append("ambiguous_top_two")
    if top_score < 0.24:
        reasons.append("weak_deterministic_support")
    return reasons


def _verifier_evidence_id(citation: dict, ref: int) -> str:
    """Build a stable, ref-addressable ID without trusting opaque model IDs."""

    return f"ref:{int(ref)}"


def build_claim_verifier_candidates(
    answer: str,
    citations: list[dict],
    *,
    min_score: float = 0.24,
    max_candidates: int = 8,
    max_evidence_per_claim: int = 2,
) -> dict[str, Any]:
    """Build a bounded verifier payload from already-authorized citations.

    This function never searches or invents evidence.  It only ranks the
    citation records already passed by provenance authorization and includes a
    claim when deterministic signals say that a semantic check is worthwhile.
    ``claim_ref_map`` is kept outside the model payload so verdict application
    can map ``ref:<n>`` back to the exact displayed citation.
    """

    normalized: list[dict[str, Any]] = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref"))
        except (TypeError, ValueError):
            continue
        if ref <= 0 or not citation_support_text(citation):
            continue
        item = dict(citation)
        item["ref"] = ref
        normalized.append(item)

    claims = extract_atomic_claims(answer)
    candidates: list[dict[str, Any]] = []
    claim_ref_map: dict[str, dict[str, Any]] = {}
    for claim in claims:
        scored = rank_claim_evidence(claim["claim_text"], normalized)
        if not scored:
            continue
        top_score = float(scored[0][0])
        second_score = float(scored[1][0]) if len(scored) > 1 else None
        score_gap = (top_score - second_score) if second_score is not None else None
        risk_reasons = claim_risk_reasons(
            claim["claim_text"],
            top_score=top_score,
            score_gap=score_gap,
        )
        if not risk_reasons:
            continue

        qualified = [row for row in scored if row[0] >= max(0.0, float(min_score))]
        if not qualified:
            qualified = scored[:1]
        # A second excerpt is useful only when it is genuinely competitive;
        # otherwise it tends to be a nearby row or adjacent section that the
        # verifier may incorrectly treat as corroboration.
        selected = qualified[:1]
        if (
            len(qualified) > 1
            and max_evidence_per_claim > 1
            and score_gap is not None
            and score_gap < 0.12
            and _evidence_is_complementary(
                claim["claim_text"],
                qualified[0][2],
                qualified[1][2],
            )
        ):
            first_identity = str(
                qualified[0][2].get("evidence_id")
                or qualified[0][2].get("group_id")
                or qualified[0][2].get("block_id")
                or qualified[0][1]
            )
            second_identity = str(
                qualified[1][2].get("evidence_id")
                or qualified[1][2].get("group_id")
                or qualified[1][2].get("block_id")
                or qualified[1][1]
            )
            if first_identity != second_identity:
                selected.append(qualified[1])
        evidence: list[dict[str, str]] = []
        refs: list[int] = []
        scores: list[dict[str, Any]] = []
        for score, ref, citation in selected:
            evidence_text = citation_support_text(citation)
            if not evidence_text:
                continue
            evidence.append({
                "evidence_id": _verifier_evidence_id(citation, ref),
                "text": evidence_text[:900],
            })
            refs.append(int(ref))
            scores.append({"ref": int(ref), "score": round(float(score), 4)})
        if not evidence:
            continue
        claim_id = str(claim["claim_id"])
        candidates.append({
            "claim_id": claim_id,
            "claim_kind": "high_risk_answer_claim",
            "claim_text": claim["claim_text"],
            "evidence": evidence,
        })
        claim_ref_map[claim_id] = {
            "refs": refs,
            "scores": scores,
            "risk_reasons": risk_reasons,
            "start": int(claim["start"]),
            "end": int(claim["end"]),
            "raw_text": claim["raw_text"],
            "claim_text": claim["claim_text"],
        }
        if len(candidates) >= max(1, int(max_candidates)):
            break

    return {
        "claims": candidates,
        "claim_ref_map": claim_ref_map,
        "diagnostics": {
            "claim_count": len(claims),
            "candidate_count": len(candidates),
            "max_candidates": max(1, int(max_candidates)),
            "min_score": round(max(0.0, float(min_score)), 4),
        },
    }


def apply_claim_verifier_decisions(
    answer: str,
    verifier: dict | None,
    *,
    conservative_prefix: str = "根据当前已授权证据，无法确认：",
) -> tuple[str, dict[str, Any]]:
    """Apply a fail-closed fallback when semantic verification found a risk.

    The normal path tries one evidence-locked LLM repair first.  This helper is
    the deterministic fallback for timeouts, missing credentials, or rejected
    repair output.  It never adds a citation or new factual content.
    """

    payload = verifier if isinstance(verifier, dict) else {}
    verdicts = payload.get("verdicts") if isinstance(payload.get("verdicts"), list) else []
    claim_ref_map = payload.get("claim_ref_map") if isinstance(payload.get("claim_ref_map"), dict) else {}
    if not verdicts or not claim_ref_map or not answer:
        return str(answer or ""), {"applied": False, "reason": "no_verdicts"}

    replacements: list[tuple[int, int, str]] = []
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0, "uncertain": 0}
    for raw in verdicts:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "uncertain").strip().lower()
        if status not in counts:
            status = "uncertain"
        counts[status] += 1
        if status == "supported":
            continue
        info = claim_ref_map.get(str(raw.get("claim_id") or ""))
        if not isinstance(info, dict):
            continue
        start = info.get("start")
        end = info.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        core = strip_inline_citations(str(info.get("raw_text") or info.get("claim_text") or "")).strip()
        if not core:
            continue
        if status == "contradicted":
            replacement = "当前已授权证据与该陈述不一致，无法确认其成立。"
        elif status == "uncertain":
            replacement = f"{conservative_prefix} {core}"
        else:
            replacement = f"{conservative_prefix} {core}"
        replacements.append((start, end, replacement))

    rewritten = str(answer)
    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten, {
        "applied": bool(replacements),
        "counts": counts,
        "replaced_claim_count": len(replacements),
        "mode": "deterministic_fail_closed",
    }


def _attach_refs(text: str, refs: list[int]) -> str:
    clean = strip_inline_citations(text).rstrip()
    if not clean or not refs:
        return clean
    marker = "".join(f"[{ref}]" for ref in refs)
    if clean[-1:] in "。！？!?；;":
        return f"{clean[:-1]}{marker}{clean[-1]}"
    return f"{clean}{marker}"


def extract_atomic_claims(answer: str) -> list[dict[str, Any]]:
    """Split answer prose into replaceable sentence spans."""

    source = str(answer or "")
    claims: list[dict[str, Any]] = []
    offset = 0
    in_code_fence = False
    for line in source.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        if re.match(r"^\s*```", line_body):
            in_code_fence = not in_code_fence
            offset += len(line)
            continue
        if in_code_fence or not line_body.strip() or re.match(r"^\s{0,3}#{1,6}\s", line_body):
            offset += len(line)
            continue
        line_start = offset
        for match in _CLAUSE_RE.finditer(line_body):
            raw = match.group(0)
            raw_match_offset = match.start()
            # A citation marker normally follows the sentence terminator.  The
            # clause regex intentionally stops at that terminator, so attach a
            # leading marker from the next match back to the preceding claim
            # before parsing the next sentence.  Otherwise ``[1]`` would be
            # attributed to the following claim and completeness metrics would
            # report a false missing citation.
            if claims:
                marker_prefix = re.match(
                    r"^\s*(?:(?:\[\d{1,3}\]|【\d{1,3}】)\s*)+",
                    raw,
                )
                if marker_prefix:
                    prefix = marker_prefix.group(0)
                    previous = claims[-1]
                    previous["raw_text"] = str(previous.get("raw_text") or "") + prefix
                    previous["end"] = line_start + raw_match_offset + len(prefix)
                    previous["existing_refs"] = list(previous.get("existing_refs") or [])
                    previous["existing_refs"].extend(
                        int(item.group(1) or item.group(2))
                        for item in _INLINE_CITATION_RE.finditer(prefix)
                    )
                    raw = raw[len(prefix):]
                    raw_match_offset += len(prefix)
                    if not raw.strip():
                        continue
            left_trim = len(raw) - len(raw.lstrip())
            right_trimmed = raw.rstrip()
            if not right_trimmed:
                continue
            start = line_start + raw_match_offset + left_trim
            end = line_start + raw_match_offset + len(right_trimmed)
            core = strip_inline_citations(right_trimmed).strip(" -*•\t")
            if len(tokenize(core, informative_only=True)) < 2:
                continue
            claims.append({
                "claim_id": f"claim-{len(claims) + 1}",
                "claim_text": core[:320],
                "raw_text": right_trimmed,
                "start": start,
                "end": end,
                "existing_refs": [int(item.group(1) or item.group(2)) for item in _INLINE_CITATION_RE.finditer(right_trimmed)],
            })
        offset += len(line)
    return claims


def _find_support_span(claim: str, citation: dict, score: float) -> dict:
    source = str(
        citation.get("source_text")
        or citation.get("display_text")
        or citation.get("_full_text")
        or citation.get("highlight_text")
        or ""
    ).strip()
    if not source:
        return {"text": "", "start": None, "end": None, "score": round(score, 4)}

    candidates: list[tuple[float, int, int, str]] = []
    for match in re.finditer(r"[^\n。！？!?；;，,]+(?:[。！？!?；;，,]|$)", source):
        text = match.group(0).strip()
        if text:
            local_score = claim_support_score(claim, {"source_text": text})
            candidates.append((local_score, match.start(), match.end(), text))
    highlight = str(citation.get("highlight_text") or "").strip()
    if highlight:
        highlight_score = claim_support_score(claim, {"source_text": highlight})
        index = source.find(highlight)
        candidates.append((highlight_score, index, index + len(highlight), highlight))
    if not candidates:
        candidates = [(score, 0, min(len(source), 320), source[:320])]
    best_score, start, end, text = max(candidates, key=lambda item: (item[0], -item[1]))
    if len(text) > 360:
        text = text[:360].rstrip() + "..."
        end = start + len(text)
    return {
        "text": text,
        "start": start if start >= 0 else None,
        "end": end if start >= 0 else None,
        "score": round(max(score, best_score), 4),
    }


def align_answer_citations(
    answer: str,
    citations: list[dict],
    *,
    min_score: float = 0.24,
    max_refs_per_claim: int = 2,
) -> dict[str, Any]:
    """Bind each atomic claim to its strongest authorized citation(s).

    The caller is responsible for authorization filtering.  Returned citations
    retain all candidates for diagnostics, while ``selected_refs`` identifies
    the records that may be published for this answer.
    """

    normalized = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        try:
            ref = int(citation.get("ref"))
        except (TypeError, ValueError):
            continue
        if ref <= 0:
            continue
        item = dict(citation)
        item["ref"] = ref
        normalized.append(item)
    claims = extract_atomic_claims(answer)
    if not normalized or not claims:
        return {"answer": answer, "citations": normalized, "selected_refs": [], "bindings": [], "diagnostics": {"claim_count": len(claims)}}

    updated = {int(item["ref"]): item for item in normalized}
    selected_refs: set[int] = set()
    replacements: list[tuple[int, int, str]] = []
    bindings: list[dict[str, Any]] = []
    unsupported_count = 0
    # 同一段证据被反复拿来支撑不同结论会制造虚假的高覆盖率。独占键刻意带上
    # 该结论涉及的数字集合：一张表可以合法支撑多个**不同数值**的结论，按整块
    # 一刀切会误杀正当的一表多用。这里只标记不删除——误判独占不应让正确证据
    # 消失，但下游度量需要能看见这个信号。
    occupied_spans: set[tuple[int, str, frozenset]] = set()
    reused_span_count = 0
    for claim in claims:
        if _is_conservative_claim(claim["claim_text"]):
            unsupported_count += 1
            replacements.append((
                int(claim["start"]),
                int(claim["end"]),
                _attach_refs(claim["raw_text"], []),
            ))
            bindings.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "status": "uncertain",
                "refs": [],
                "claim_support": [],
                "support_spans": [],
            })
            continue
        scored = rank_claim_evidence(claim["claim_text"], normalized)
        qualified = [row for row in scored if row[0] >= max(0.0, float(min_score))]
        chosen = qualified[:1]
        if len(qualified) > 1 and max_refs_per_claim > 1:
            first_score, _first_ref, first_item = qualified[0]
            second_score, _second_ref, second_item = qualified[1]
            first_identity = str(first_item.get("evidence_id") or first_item.get("group_id") or first_item.get("block_id") or "")
            second_identity = str(second_item.get("evidence_id") or second_item.get("group_id") or second_item.get("block_id") or "")
            if (
                second_score >= max(0.32, first_score * 0.82)
                and second_identity != first_identity
                and _evidence_is_complementary(claim["claim_text"], first_item, second_item)
            ):
                chosen.append(qualified[1])

        refs = [int(row[1]) for row in chosen]
        if not refs:
            unsupported_count += 1
        selected_refs.update(refs)
        span_records = []
        claim_number_key = frozenset(_claim_numbers(claim["claim_text"]))
        for score, ref, _item in chosen:
            record = updated[ref]
            span = _find_support_span(claim["claim_text"], record, score)
            occupancy_key = (
                int(ref),
                re.sub(r"\s+", "", str(span["text"] or "")).lower(),
                claim_number_key,
            )
            reused = occupancy_key in occupied_spans
            if reused:
                reused_span_count += 1
            else:
                occupied_spans.add(occupancy_key)
            span_record = {
                "ref": ref,
                "score": round(score, 4),
                "text": span["text"],
                "start": span["start"],
                "end": span["end"],
                "reused": reused,
            }
            span_records.append(span_record)
            existing_spans = list(record.get("support_spans") or []) if isinstance(record.get("support_spans"), list) else []
            existing_spans.append({"claim_id": claim["claim_id"], **span_record})
            record["support_spans"] = existing_spans[-8:]
            if not record.get("support_span") or score >= float(record.get("claim_support_score") or 0.0):
                record["support_span"] = span["text"]
                record["support_span_start"] = span["start"]
                record["support_span_end"] = span["end"]
                record["claim_support_score"] = round(score, 4)
                if span["text"]:
                    record["highlight_text"] = span["text"]
                    record["alignment_status"] = "claim_span_matched"

        replacements.append((int(claim["start"]), int(claim["end"]), _attach_refs(claim["raw_text"], refs)))
        bindings.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "status": "supported" if refs else "unsupported",
            "refs": refs,
            "claim_support": [
                {"ref": int(row[1]), "score": round(float(row[0]), 4)}
                for row in chosen
            ],
            "support_spans": span_records,
            "span_reused": any(item.get("reused") for item in span_records),
        })

    rewritten = str(answer or "")
    for start, end, replacement in reversed(replacements):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return {
        "answer": rewritten,
        "citations": list(updated.values()),
        "selected_refs": sorted(selected_refs),
        "bindings": bindings,
        "diagnostics": {
            "claim_count": len(claims),
            "supported_claim_count": len(claims) - unsupported_count,
            "unsupported_claim_count": unsupported_count,
            "min_claim_support_score": round(float(min_score), 4),
            "max_refs_per_claim": max(1, int(max_refs_per_claim)),
            "reused_span_count": reused_span_count,
        },
    }
