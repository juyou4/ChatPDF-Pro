"""学术论文 DocDetails 抽取（本地启发式，默认不联网）。

从首页/文首抽取 title / authors / year / doi / arxiv_id / venue / abstract_preview，
写入文档的 ``paper_metadata`` 字段，供速览、对话身份行与引用导出使用。

设计约束：
- 桌面产品默认离线：不强制 Crossref / Semantic Scholar
- 失败时降级到 filename stem，不阻塞上传
- 解析代际变化后应重算（source_hash / parse_generation 绑定）
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

PAPER_METADATA_VERSION = "v1"

_DOI_RE = re.compile(
    r"\b(?:doi[:\s]*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b",
    re.IGNORECASE,
)
_ARXIV_RE = re.compile(
    r"\b(?:arXiv[:\s]*)(\d{4}\.\d{4,5})(?:v\d+)?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(
    r"(?:^|[^\d])((?:19|20)\d{2})(?:[^\d]|$)",
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
_INTRO_RE = re.compile(r"^\s*(?:1[\.\s]+)?introduction\b", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"^\s*(?:related\s+work|method|methodology|experiments?|conclusion|"
    r"references|acknowledg|background|preliminar)\b",
    re.IGNORECASE,
)
_AFFILIATION_RE = re.compile(
    r"(?:university|institute|laboratory|lab\b|college|department|school|"
    r"huawei|google|microsoft|facebook|meta|alibaba|tencent|bytedance|"
    r"academy|research\s+center|中心|大学|研究所|实验室)",
    re.IGNORECASE,
)
_STOP_TITLE_LINES = re.compile(
    r"^\s*(?:arxiv|doi|http|www\.|preprint|copyright|©|all\s+rights|"
    r"proceedings|conference|workshop|volume|pages?\s*\d)\b",
    re.IGNORECASE,
)


@dataclass
class PaperMetadata:
    """Normalized academic identity for one document."""

    version: str = PAPER_METADATA_VERSION
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    abstract_preview: str = ""
    source: str = "heuristic"  # heuristic | filename | cached
    confidence: float = 0.0
    extracted_at: str = ""
    parse_generation: str = ""
    document_source_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["authors"] = list(self.authors or [])
        return payload

    def short_citation(self) -> str:
        """Author et al., Year style label for prompts and UI."""
        authors = [a for a in (self.authors or []) if a]
        year = str(self.year) if self.year else ""
        if authors:
            first = authors[0]
            # Keep last token as surname when "First Last"
            parts = first.replace(",", " ").split()
            surname = parts[-1] if parts else first
            if len(authors) == 1:
                who = surname
            elif len(authors) == 2:
                other = authors[1].replace(",", " ").split()
                who = f"{surname} & {(other[-1] if other else authors[1])}"
            else:
                who = f"{surname} et al."
        else:
            who = ""
        title = (self.title or "").strip()
        bits = [b for b in (who, year) if b]
        head = ", ".join(bits) if bits else ""
        if title and head:
            return f"{head}. {title}"
        return head or title or ""

    def identity_line(self) -> str:
        """One-line identity for chat system prompt."""
        cite = self.short_citation()
        extras: list[str] = []
        if self.doi:
            extras.append(f"DOI {self.doi}")
        if self.arxiv_id:
            extras.append(f"arXiv:{self.arxiv_id}")
        if self.venue:
            extras.append(self.venue[:80])
        extra = f" ({'; '.join(extras)})" if extras else ""
        if cite:
            return f"论文身份：{cite}{extra}"
        return ""


def _clean_line(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    text = text.replace("\\*", "").replace("*", "")
    return text


def _is_authorish_line(line: str) -> bool:
    text = _clean_line(line)
    if not text or len(text) < 3 or len(text) > 220:
        return False
    if _ABSTRACT_RE.match(text) or _INTRO_RE.match(text) or _SECTION_RE.match(text):
        return False
    if _AFFILIATION_RE.search(text) and not re.search(r"[A-Z][a-z]+\s+[A-Z]", text):
        return False
    if _EMAIL_RE.search(text):
        return True
    # Multiple capitalized name tokens, optionally comma/and separated.
    names = re.findall(r"\b[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3}\b", text)
    if len(names) >= 2 and not text.endswith("."):
        return True
    if len(names) == 1 and "," in text:
        return True
    # Chinese author lines: 2-4 char names separated by space/顿号
    zh_names = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    if len(zh_names) >= 2 and len(text) <= 40:
        return True
    return False


def _split_space_separated_western_names(text: str) -> list[str]:
    """Split 'First Last First Last ...' lines common on paper title pages."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    if len(tokens) < 4:
        return []
    # Heuristic: most academic names are 2 tokens (First Last); allow 3 for
    # middle initials / particles when the middle token is short.
    names: list[str] = []
    i = 0
    while i < len(tokens):
        remaining = len(tokens) - i
        middle = tokens[i + 1] if remaining >= 3 else ""
        # Only treat single-letter / initial tokens as middle names (John A Smith).
        # Do NOT treat short surnames like "Wu" as initials.
        is_initial = bool(re.fullmatch(r"[A-Za-z]\.?", middle)) and len(middle.rstrip(".")) == 1
        if remaining >= 3 and is_initial:
            names.append(" ".join(tokens[i : i + 3]))
            i += 3
            continue
        if remaining >= 2:
            names.append(" ".join(tokens[i : i + 2]))
            i += 2
            continue
        break
    return names if len(names) >= 2 else []


def _split_authors(line: str) -> list[str]:
    text = _EMAIL_RE.sub(" ", _clean_line(line))
    text = re.sub(r"\s+", " ", text).strip(" ,;|")
    if not text:
        return []
    # Prefer explicit separators.
    if re.search(r"[,;|&、，]|\band\b", text, flags=re.IGNORECASE):
        chunks = re.split(r"\s*(?:,|;|\||\band\b|&|、|，)\s*", text, flags=re.IGNORECASE)
    else:
        space_names = _split_space_separated_western_names(text)
        if space_names:
            chunks = space_names
        else:
            chunks = [text]
    authors: list[str] = []
    for chunk in chunks:
        name = re.sub(r"\s+", " ", chunk).strip(" .")
        if not name or len(name) < 2:
            continue
        if _AFFILIATION_RE.search(name) and len(name) > 40:
            continue
        # Drop pure emails / urls
        if "@" in name or name.lower().startswith("http"):
            continue
        authors.append(name[:80])
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for author in authors:
        key = author.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(author)
    return ordered[:12]


def _sample_head_text(
    full_text: str = "",
    pages: Optional[list] = None,
    *,
    max_chars: int = 3500,
) -> str:
    page_bits: list[str] = []
    for page in (pages or [])[:2]:
        if not isinstance(page, dict):
            continue
        bit = str(page.get("text") or page.get("content") or "").strip()
        if bit:
            page_bits.append(bit)
    if page_bits:
        joined = "\n\n".join(page_bits)
        return joined[:max_chars]
    return str(full_text or "")[:max_chars]


def _filename_stem_title(filename: str) -> str:
    stem = str(filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"\.(pdf|docx?|pptx?|txt|md)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:180]


def extract_paper_metadata(
    *,
    full_text: str = "",
    pages: Optional[list] = None,
    filename: str = "",
    parse_generation: str = "",
    document_source_hash: str = "",
) -> PaperMetadata:
    """Extract DocDetails from local text only."""
    head = _sample_head_text(full_text, pages)
    lines = [_clean_line(line) for line in head.splitlines()]
    lines = [line for line in lines if line]

    doi = ""
    arxiv_id = ""
    year: Optional[int] = None
    for line in lines[:40]:
        if not doi:
            match = _DOI_RE.search(line)
            if match:
                doi = match.group(1).rstrip(").,;")
        if not arxiv_id:
            match = _ARXIV_RE.search(line)
            if match:
                arxiv_id = match.group(1)
        if year is None:
            # Prefer years near venue / copyright lines later; collect candidates.
            pass

    year_candidates: list[int] = []
    for line in lines[:50]:
        for match in _YEAR_RE.finditer(line):
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 1990 <= value <= datetime.now(timezone.utc).year + 1:
                year_candidates.append(value)
    if year_candidates:
        # Prefer the first plausible year in the head (often venue line / title year).
        year = year_candidates[0]

    # Locate abstract boundary
    abstract_idx = None
    for idx, line in enumerate(lines[:60]):
        if _ABSTRACT_RE.match(line):
            abstract_idx = idx
            break

    title_lines: list[str] = []
    author_lines: list[str] = []
    venue = ""
    body_start = abstract_idx if abstract_idx is not None else min(len(lines), 25)

    # Title: first non-boilerplate lines before authors/abstract
    i = 0
    while i < body_start and i < 12:
        line = lines[i]
        i += 1
        if _STOP_TITLE_LINES.match(line) or _EMAIL_RE.search(line):
            continue
        if _is_authorish_line(line) and title_lines:
            author_lines.append(line)
            break
        if _AFFILIATION_RE.search(line) and title_lines:
            if not venue:
                venue = line[:120]
            break
        if len(line) < 5:
            continue
        # Titles are usually longer than 8 chars and not pure digits
        if re.fullmatch(r"[\d\W]+", line):
            continue
        title_lines.append(line)
        # Multi-line title: allow one continuation if next line is title-like
        if i < body_start:
            nxt = lines[i]
            if (
                not _is_authorish_line(nxt)
                and not _AFFILIATION_RE.search(nxt)
                and not _ABSTRACT_RE.match(nxt)
                and 5 <= len(nxt) <= 160
                and not nxt.endswith(".")
            ):
                # Only take continuation if it looks like title case / mixed
                if re.search(r"[A-Za-z\u4e00-\u9fff]", nxt):
                    title_lines.append(nxt)
                    i += 1
        break

    # Authors after title until affiliation/abstract
    while i < body_start and i < 20:
        line = lines[i]
        i += 1
        if _ABSTRACT_RE.match(line) or _INTRO_RE.match(line):
            break
        if _AFFILIATION_RE.search(line) and not _is_authorish_line(line):
            if not venue:
                venue = line[:120]
            # skip subsequent affiliation-only lines
            continue
        if _is_authorish_line(line):
            author_lines.append(line)
            continue
        if author_lines:
            break

    authors: list[str] = []
    for line in author_lines:
        authors.extend(_split_authors(line))
    # de-dupe again
    deduped: list[str] = []
    seen: set[str] = set()
    for author in authors:
        key = author.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(author)
    authors = deduped[:12]

    title = " ".join(title_lines).strip()
    title = re.sub(r"\s+", " ", title)
    if len(title) > 240:
        title = title[:240].rstrip()

    abstract_preview = ""
    if abstract_idx is not None:
        abs_lines = []
        for line in lines[abstract_idx + 1 : abstract_idx + 12]:
            if _INTRO_RE.match(line) or _SECTION_RE.match(line):
                break
            abs_lines.append(line)
        abstract_preview = re.sub(r"\s+", " ", " ".join(abs_lines)).strip()[:400]

    confidence = 0.25
    source = "heuristic"
    if title and authors:
        confidence = 0.82
    elif title:
        confidence = 0.6
    elif authors:
        confidence = 0.45
    if doi or arxiv_id:
        confidence = min(0.95, confidence + 0.1)

    if not title:
        title = _filename_stem_title(filename)
        source = "filename"
        confidence = min(confidence, 0.35)

    return PaperMetadata(
        version=PAPER_METADATA_VERSION,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        venue=venue,
        abstract_preview=abstract_preview,
        source=source,
        confidence=round(confidence, 3),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        parse_generation=str(parse_generation or "").strip(),
        document_source_hash=str(document_source_hash or "").strip(),
    )


def paper_metadata_from_dict(raw: Any) -> Optional[PaperMetadata]:
    if not isinstance(raw, dict):
        return None
    authors_raw = raw.get("authors") or []
    authors = [str(item).strip() for item in authors_raw if str(item).strip()] if isinstance(authors_raw, list) else []
    year = raw.get("year")
    try:
        year_i = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year_i = None
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return PaperMetadata(
        version=str(raw.get("version") or PAPER_METADATA_VERSION),
        title=str(raw.get("title") or "").strip(),
        authors=authors,
        year=year_i,
        doi=str(raw.get("doi") or "").strip(),
        arxiv_id=str(raw.get("arxiv_id") or "").strip(),
        venue=str(raw.get("venue") or "").strip(),
        abstract_preview=str(raw.get("abstract_preview") or "").strip(),
        source=str(raw.get("source") or "cached"),
        confidence=confidence,
        extracted_at=str(raw.get("extracted_at") or ""),
        parse_generation=str(raw.get("parse_generation") or "").strip(),
        document_source_hash=str(raw.get("document_source_hash") or "").strip(),
    )


def _parse_identity_from_doc(doc: dict) -> tuple[str, str]:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    # Prefer explicit fields; fall back to parse_manifest-like nests.
    generation = str(
        doc.get("parse_generation")
        or data.get("parse_generation")
        or (doc.get("parse_manifest") or {}).get("generation")
        or ""
    ).strip()
    source_hash = str(
        doc.get("document_source_hash")
        or data.get("document_source_hash")
        or data.get("source_hash")
        or (doc.get("parse_manifest") or {}).get("source_hash")
        or ""
    ).strip()
    return generation, source_hash


def ensure_paper_metadata(
    doc: dict,
    *,
    force: bool = False,
    persist_callback=None,
) -> dict[str, Any]:
    """Ensure ``doc['paper_metadata']`` is present and identity-fresh.

    Optionally call ``persist_callback(doc)`` when metadata is newly written.
    """
    if not isinstance(doc, dict):
        return {}
    generation, source_hash = _parse_identity_from_doc(doc)
    existing = paper_metadata_from_dict(doc.get("paper_metadata"))
    if (
        existing
        and not force
        and existing.title
        and (
            not generation
            or not existing.parse_generation
            or existing.parse_generation == generation
        )
        and (
            not source_hash
            or not existing.document_source_hash
            or existing.document_source_hash == source_hash
        )
    ):
        # Refresh identity fields if they were empty on an older cache.
        if generation and not existing.parse_generation:
            existing.parse_generation = generation
        if source_hash and not existing.document_source_hash:
            existing.document_source_hash = source_hash
        payload = existing.to_dict()
        doc["paper_metadata"] = payload
        return payload

    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    meta = extract_paper_metadata(
        full_text=str(data.get("full_text") or data.get("text") or ""),
        pages=data.get("pages") if isinstance(data.get("pages"), list) else None,
        filename=str(doc.get("filename") or ""),
        parse_generation=generation,
        document_source_hash=source_hash,
    )
    payload = meta.to_dict()
    doc["paper_metadata"] = payload
    if callable(persist_callback):
        try:
            persist_callback(doc)
        except Exception:
            pass
    return payload


def format_paper_identity_prompt(paper_metadata: Any) -> str:
    """System-prompt snippet; empty when identity is unknown."""
    meta = (
        paper_metadata
        if isinstance(paper_metadata, PaperMetadata)
        else paper_metadata_from_dict(paper_metadata)
    )
    if not meta:
        return ""
    line = meta.identity_line()
    if not line:
        return ""
    extra = ""
    if meta.abstract_preview:
        extra = f"\n摘要线索：{meta.abstract_preview[:220]}"
    return f"【文献身份】\n{line}{extra}"
