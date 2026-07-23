"""Shared document block-role classification for outline and reading services."""
from __future__ import annotations

import re
from typing import Any


BLOCK_ROLE_VERSION = 1

ROLE_BODY = "body"
ROLE_HEADING = "heading"
ROLE_CAPTION = "caption"
ROLE_TABLE = "table"
ROLE_FIGURE = "figure"
ROLE_VISUAL_ENRICHMENT = "visual_enrichment"
ROLE_ARTIFACT = "artifact"
ROLE_AUTHOR = "author"
ROLE_AFFILIATION = "affiliation"
ROLE_CONTACT = "contact"
ROLE_PUBLICATION_HEADER = "publication_header"
ROLE_REFERENCE = "reference"
ROLE_KEYWORDS = "keywords"

FRONT_MATTER_ROLES = frozenset({ROLE_AUTHOR, ROLE_AFFILIATION, ROLE_CONTACT})
OUTLINE_EXCLUDED_ROLES = frozenset({
    ROLE_ARTIFACT,
    ROLE_AUTHOR,
    ROLE_AFFILIATION,
    ROLE_CONTACT,
    ROLE_PUBLICATION_HEADER,
    ROLE_REFERENCE,
    ROLE_KEYWORDS,
})
KEYWORD_EXCLUDED_ROLES = frozenset({
    ROLE_ARTIFACT,
    ROLE_AUTHOR,
    ROLE_AFFILIATION,
    ROLE_CONTACT,
    ROLE_PUBLICATION_HEADER,
    ROLE_REFERENCE,
})

_KNOWN_ROLES = frozenset({
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_CAPTION,
    ROLE_TABLE,
    ROLE_FIGURE,
    ROLE_VISUAL_ENRICHMENT,
    *OUTLINE_EXCLUDED_ROLES,
})
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.IGNORECASE)
_ORCID_RE = re.compile(r"\borcid(?:\.org)?\b|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_KEYWORDS_RE = re.compile(r"^\s*(?:keywords?|index\s+terms?|关键词|关键字)\s*[:：]", re.IGNORECASE)
_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:references|bibliography|参考文献)\s*$",
    re.IGNORECASE,
)
_ACKNOWLEDGMENT_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)(?:acknowledg(?:e)?ments?|致谢|致謝)\s*$",
    re.IGNORECASE,
)
_POST_REFERENCE_TEMPLATE_RE = re.compile(
    r"^\s*(?:[a-z0-9-]+\s+)?(?:paper\s+)?checklist\b",
    re.IGNORECASE,
)
_PUBLICATION_STRONG_RE = re.compile(
    r"\b(?:copyright|all\s+rights\s+reserved|doi|issn|isbn|licensed|authorized|downloaded)\b|"
    r"(?:版权所有|保留所有权利)",
    re.IGNORECASE,
)
_PUBLICATION_CONTEXT_RE = re.compile(
    r"\b(?:proceedings|transactions?|journal|conference|technical\s+report|preprint|"
    r"volume|vol\.?|issue|no\.?|pp\.?)\b|(?:会议论文集|期刊|预印本|出版)",
    re.IGNORECASE,
)
_AFFILIATION_RE = re.compile(
    r"\b(?:dept\.?|department|institute|university|college|school|academy|faculty|campus|"
    r"laborator(?:y|ies)|lab|research\s+center|research\s+centre|corporation|inc\.?|ltd\.?)\b|"
    r"(?:大学(?!生)|学院|研究院|研究所|实验室|重点实验室|研究中心|工程中心|技术中心|"
    r"(?:计算机|信息|电子|自动化|软件|人工智能|医学|物理|数学|化学|生物|材料)系|"
    r"作者单位|通讯作者|通信作者)",
    re.IGNORECASE,
)
_AUTHOR_SEPARATOR_RE = re.compile(r"[,，、;；]")
_AUTHOR_NAME_PART_RE = re.compile(
    r"^(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+|[A-Z]\.)"
    r"(?:\s+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+|[A-Z]\.)){1,3}"
    r"\s*(?:\d{1,2}|\*|†|‡|§)*$"
)
_AUTHOR_MARKED_NAME_RE = re.compile(
    r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+"
    r"(?:\s+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+|[A-Z]\.)){1,3}"
    r"\s*(?:\*|†|‡|§|\d{1,2})(?=\s|$)"
)
_COMPACT_MARKED_AUTHOR_RE = re.compile(
    r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+\s+"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+"
    r"(?:\d{1,2}|\*|⋆|†|‡|§)"
)
_CJK_AUTHOR_LIST_RE = re.compile(
    r"^\s*[\u4e00-\u9fff]{2,4}(?:\s*[\d*†‡]+)?"
    r"(?:\s*[,，、;；]\s*[\u4e00-\u9fff]{2,4}(?:\s*[\d*†‡]+)?){1,}\s*$"
)
_REFERENCE_ENTRY_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+[.)])\s+"
    r"(?:[A-Z][A-Za-z'\-]+\s*,\s*(?:[A-Z]\.|[A-Z][A-Za-z'\-]+)|"
    r"[\u4e00-\u9fff]{2,4}\s*[,，])",
    re.IGNORECASE,
)
_BODY_SENTENCE_RE = re.compile(
    r"(?:[。！？!?]\s*$|\.\s*$)|\b(?:we|this\s+(?:paper|study|work)|results?|method|dataset|"
    r"evaluate|investigate|show|demonstrate|propose|introduce)\b|"
    r"\b(?:participants?|students?|patients?|users?)\s+"
    r"(?:are|were|was|is|show|showed|reported|completed|used|received|performed|participated)\b",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def is_reference_heading(value: Any) -> bool:
    """Whether a heading marks the terminal bibliography section."""
    return bool(_REFERENCE_HEADING_RE.match(_clean_text(value)))


def is_acknowledgment_heading(value: Any) -> bool:
    """Recognize English and Chinese acknowledgement headings consistently."""
    return bool(_ACKNOWLEDGMENT_HEADING_RE.match(_clean_text(value)))


def is_post_reference_template_artifact(value: Any) -> bool:
    """Recognize terminal submission-template material such as NeurIPS checklists."""
    return bool(_POST_REFERENCE_TEMPLATE_RE.match(_clean_text(value)))


def _decision(role: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "role": role,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "reason": reason,
        "version": BLOCK_ROLE_VERSION,
    }


def _looks_like_english_author_byline(value: str) -> bool:
    parts = [
        part.strip()
        for part in re.split(r"[,，、;；]|\s+(?:and|&)\s+", value)
        if part.strip()
    ]
    return len(parts) >= 2 and all(_AUTHOR_NAME_PART_RE.fullmatch(part) for part in parts)


def classify_front_matter_text(text: Any) -> dict[str, Any] | None:
    """Classify metadata-like text using multiple signals, not institution words alone."""
    value = _clean_text(text)
    if not value:
        return None
    if _EMAIL_RE.search(value):
        return _decision(ROLE_CONTACT, 0.99, "email")
    if _ORCID_RE.search(value):
        return _decision(ROLE_CONTACT, 0.98, "orcid")
    if _KEYWORDS_RE.match(value):
        return _decision(ROLE_KEYWORDS, 0.99, "keyword_line")
    if is_reference_heading(value) or _REFERENCE_ENTRY_RE.match(value):
        return _decision(ROLE_REFERENCE, 0.96, "reference")
    if _PUBLICATION_STRONG_RE.search(value) and len(value) <= 320:
        return _decision(ROLE_PUBLICATION_HEADER, 0.97, "strong_publication_cue")
    publication_hits = len(_PUBLICATION_CONTEXT_RE.findall(value))
    if (
        publication_hits
        and len(value) <= 240
        and not _BODY_SENTENCE_RE.search(value)
        and (publication_hits >= 2 or re.search(r"\b(?:19|20)\d{2}\b", value))
    ):
        return _decision(ROLE_PUBLICATION_HEADER, 0.86, "publication_metadata")

    affiliation_hits = len(_AFFILIATION_RE.findall(value))
    if affiliation_hits:
        confidence = 0.55
        reasons = ["affiliation_cue"]
        if affiliation_hits >= 2:
            confidence += 0.12
            reasons.append("multiple_cues")
        if len(value) <= 180:
            confidence += 0.15
            reasons.append("short_line")
        if not _BODY_SENTENCE_RE.search(value):
            confidence += 0.16
            reasons.append("non_sentence")
        if re.match(r"^\s*(?:\d+|[*†‡§])\s*", value):
            confidence += 0.08
            reasons.append("affiliation_marker")
        if _URL_RE.search(value):
            confidence += 0.05
            reasons.append("url")
        if confidence >= 0.78:
            return _decision(ROLE_AFFILIATION, confidence, "+".join(reasons))

    # Run the looser name-shape detector after institution cues. Capitalized
    # organization names (for example, "Department ... University") otherwise
    # look like a comma-separated author byline.
    separators = len(_AUTHOR_SEPARATOR_RE.findall(value))
    if (
        len(value) <= 320
        and separators >= 1
        and _looks_like_english_author_byline(value)
        and not _BODY_SENTENCE_RE.search(value)
    ):
        confidence = 0.9 + (0.03 if separators >= 2 else 0.0)
        return _decision(ROLE_AUTHOR, confidence, "author_byline")
    if len(value) <= 180 and _CJK_AUTHOR_LIST_RE.fullmatch(value):
        return _decision(ROLE_AUTHOR, 0.92, "cjk_author_byline")
    if (
        len(value) <= 220
        and len(_AUTHOR_MARKED_NAME_RE.findall(value)) >= 2
    ):
        return _decision(ROLE_AUTHOR, 0.86, "author_markers")
    if (
        len(value) <= 420
        and len(_COMPACT_MARKED_AUTHOR_RE.findall(value)) >= 3
    ):
        return _decision(ROLE_AUTHOR, 0.92, "compact_author_markers")
    return None


def classify_block_role(block: dict[str, Any], *, section_title: str = "") -> dict[str, Any]:
    existing_role = str(block.get("content_role") or "").strip().lower()
    try:
        existing_version = int(block.get("block_role_version") or 0)
    except (TypeError, ValueError):
        existing_version = 0
    if existing_role in _KNOWN_ROLES and existing_version == BLOCK_ROLE_VERSION:
        return _decision(
            existing_role,
            float(block.get("role_confidence") or 1.0),
            str(block.get("role_reason") or "persisted"),
        )

    block_type = str(block.get("type") or "paragraph").strip().lower()
    text = _clean_text(block.get("text"))
    try:
        page = max(1, int(block.get("page") or 1))
    except (TypeError, ValueError):
        page = 1
    section = _clean_text(section_title or block.get("section_title")).lower()

    if block_type == ROLE_ARTIFACT:
        return _decision(ROLE_ARTIFACT, 1.0, "artifact_type")

    if block_type == "heading":
        heading_metadata = classify_front_matter_text(text)
        if heading_metadata and heading_metadata["role"] in {
            ROLE_CONTACT,
            ROLE_PUBLICATION_HEADER,
            ROLE_REFERENCE,
            ROLE_KEYWORDS,
        }:
            return heading_metadata
        return _decision(ROLE_HEADING, 1.0, "heading_type")

    # Explicit visual/table types take precedence over metadata wording inside
    # their content. A first-page table mentioning a university is still a table.
    structural_roles = {
        "heading": ROLE_HEADING,
        "caption": ROLE_CAPTION,
        "table": ROLE_TABLE,
        "figure": ROLE_FIGURE,
        "image": ROLE_FIGURE,
        "visual_enrichment": ROLE_VISUAL_ENRICHMENT,
    }
    if block_type in structural_roles:
        return _decision(structural_roles[block_type], 1.0, f"{block_type}_type")

    if is_reference_heading(section):
        return _decision(ROLE_REFERENCE, 0.99, "reference_section")

    front_matter = classify_front_matter_text(text)
    if front_matter:
        role = str(front_matter["role"])
        confidence = float(front_matter["confidence"])
        globally_decisive = role in {ROLE_CONTACT, ROLE_REFERENCE, ROLE_KEYWORDS}
        likely_front_matter = page <= 2 or section in {"title", "abstract", "摘要"}
        if globally_decisive or likely_front_matter or confidence >= 0.92:
            return front_matter

    if block_type in structural_roles:
        return _decision(structural_roles[block_type], 1.0, f"{block_type}_type")
    return _decision(ROLE_BODY, 0.8, "content_default")


def annotate_block_role(block: dict[str, Any], *, section_title: str = "") -> dict[str, Any]:
    item = dict(block)
    decision = classify_block_role(item, section_title=section_title)
    item["content_role"] = decision["role"]
    item["role_confidence"] = decision["confidence"]
    item["role_reason"] = decision["reason"]
    item["block_role_version"] = decision["version"]
    return item


def looks_like_affiliation_or_author_line(text: Any) -> bool:
    decision = classify_front_matter_text(text)
    return bool(decision and decision["role"] in FRONT_MATTER_ROLES)


def block_is_outline_excluded(block: dict[str, Any], *, section_title: str = "") -> bool:
    return classify_block_role(block, section_title=section_title)["role"] in OUTLINE_EXCLUDED_ROLES


def block_is_keyword_excluded(block: dict[str, Any], *, section_title: str = "") -> bool:
    return classify_block_role(block, section_title=section_title)["role"] in KEYWORD_EXCLUDED_ROLES
