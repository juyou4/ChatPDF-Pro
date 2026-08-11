"""AI 辅助的文档章节大纲。

这个服务刻意与 reading_outline_service 分离：
- section outline：原文题名、章节、子章节，用于导航
- reading outline：中文 AI 阅读笔记和结构化总结
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.chat_service import call_ai_api
from services.completion_outcome import (
    IncompleteCompletionError,
    require_publishable_completion,
)
from services.document_block_roles import (
    FRONT_MATTER_ROLES,
    ROLE_PUBLICATION_HEADER,
    annotate_block_role,
    classify_block_role,
    is_post_reference_template_artifact,
    is_reference_heading,
    looks_like_affiliation_or_author_line,
)
from services.document_parse_state import read_parse_manifest
from services.structured_json import (
    StructuredJSONError,
    parse_json_object,
    structured_json_request_params,
)

logger = logging.getLogger(__name__)

SECTION_OUTLINE_VERSION = 12
SECTION_OUTLINE_PROMPT_VERSION = "section-outline-v12"
FAILED_GENERATION_COOLDOWN_SECONDS = 60.0
# The published block outline no longer truncates long documents. Keep the
# recovery budget aligned so it can repair every heading in a thesis/textbook.
MAX_CANDIDATES = 600
MAX_CANDIDATE_TEXT = 260

_RE_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.?\s+)(.+)$")
_RE_ROMAN_HEADING = re.compile(r"^\s*([IVXLCM]+)\.\s+(.+)$")
_RE_ALPHA_HEADING = re.compile(r"^\s*([A-Z])\.\s+(.+)$")
_RE_CANONICAL_HEADING = re.compile(
    r"^\s*(abstract|introduction|background|related\s+work|method|methods|methodology|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion|limitations?|"
    r"preliminar(?:y|ies)|implementation|future\s+work|references|acknowledg(?:e)?ments?|"
    r"appendix|supplementary\s+material)\s*$",
    re.IGNORECASE,
)
_RE_CAPTION_OR_LABEL = re.compile(
    r"^\s*(fig(?:ure)?\.?|table|图|表)\s*([0-9]+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_RE_DECIMAL_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+\.\d+(?![A-Za-z])")
_RE_PUBLICATION_HEADER_CUE = re.compile(
    r"\b(vol\.?|no\.?|pp\.?|transactions?|journal|proceedings|conference|copyright|"
    r"authorized|licensed|downloaded|doi|issn|isbn|technical\s+report)\b",
    re.IGNORECASE,
)
_GENERIC_BOOKMARK_TITLES = {
    "全文",
    "全文书签",
    "full text",
    "fulltext",
    "document",
    "article",
    "paper",
    "contents",
    "content",
}
_RE_ALGORITHM_LINE = re.compile(
    r"^\s*(input|output|require|ensure|initialize|initialise|update|repeat|return|for\s+each|for\s+"
    r"|while\s+|if\s+|else\b|end\b|stage\s*\d+)\b",
    re.IGNORECASE,
)
_RE_REFERENCE_ENTRY = re.compile(
    r"^\s*(?:\[\d+\]|\d+[\.)])\s+"
    r"(?:[A-Z][A-Za-z'\-]+,\s+(?:[A-Z]\.|[A-Z][A-Za-z'\-]+)|"
    r"(?:[A-Z]\.\s*)?[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.?))",
    re.IGNORECASE,
)
_RE_KEYWORDS_LINE = re.compile(r"^\s*keywords?\s*[:：]\s+.+", re.IGNORECASE)
_RE_AUTHOR_INITIAL_REFERENCE = re.compile(
    r"^\s*\d+[\.)]\s+[A-Z][A-Za-z'\-]+,\s+(?:[A-Z]\.\s*)+",
    re.IGNORECASE,
)


def get_section_outline_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "section_outlines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_section_outline_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_section_outline_dir(data_dir) / f"{doc_id}.json"


def load_section_outline(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_section_outline_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != SECTION_OUTLINE_VERSION:
            return None
        return data
    except Exception as exc:
        logger.warning("[SectionOutline] Failed to load %s: %s", path, exc)
        return None


def save_section_outline(data_dir: Path | str, doc_id: str, outline: dict[str, Any]) -> None:
    if not _has_parse_identity(outline):
        logger.warning("[SectionOutline] Skip cache without parse identity for %s", doc_id)
        return
    path = get_section_outline_path(data_dir, doc_id)
    outline["version"] = SECTION_OUTLINE_VERSION
    outline["doc_id"] = doc_id
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(outline, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    except Exception as exc:
        logger.warning("[SectionOutline] Failed to save %s: %s", path, exc)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


async def get_or_create_section_outline(
    *,
    data_dir: Path | str,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    force: bool = False,
    cache_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """按 PDF 书签 -> AI -> 启发式的顺序返回文档章节树。"""
    source_hash = _source_hash(block_index)
    parse_generation, document_source_hash = _parse_identity(doc_id, doc)
    fallback = build_fallback_section_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )
    if _mineru_structure_is_navigation_ready(block_index, fallback):
        mineru_outline = build_fallback_section_outline(
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            source_hash=source_hash,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            source="mineru",
        )
        mineru_meta = block_index.get("mineru_meta")
        if isinstance(mineru_meta, dict):
            mineru_outline.setdefault("meta", {})["structure_version"] = mineru_meta.get("structure_version")
        logger.info("[SectionOutline] Using verified MinerU structure for %s", doc_id)
        return mineru_outline
    bookmark_outline = _build_bookmark_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )
    if bookmark_outline and _bookmark_outline_quality_ok(bookmark_outline, fallback, block_index):
        return bookmark_outline
    if bookmark_outline:
        logger.info("[SectionOutline] Ignore low-quality PDF bookmark outline for %s", doc_id)

    provider_lower = (provider or "").lower()
    can_call_model = bool(api_key) or provider_lower in {"local", "ollama"}
    cached = load_section_outline(data_dir, doc_id)
    healthy_cached: dict[str, Any] | None = None
    if (
        cached
        and cached.get("version") == SECTION_OUTLINE_VERSION
        and _matches_parse_identity(cached, parse_generation, document_source_hash)
        and cached.get("source_hash") == source_hash
    ):
        if cached.get("source") == "ai":
            if _section_outline_quality_ok(cached, fallback, block_index):
                healthy_cached = cached
                model_matches = _matches_ai_generation(cached, provider=provider, model=model)
                if not can_call_model or (not force and model_matches):
                    logger.info(
                        "[AI-Audit] purpose=section_outline doc=%s provider=%s model=%s status=cache_hit",
                        doc_id,
                        cached.get("provider") or "",
                        cached.get("model") or "",
                    )
                    return cached
                logger.info(
                    "[SectionOutline] Ignore cached AI outline doc=%s model_match=%s",
                    doc_id,
                    model_matches,
                )
            else:
                logger.info("[SectionOutline] Ignore low-quality cached AI outline for %s", doc_id)
        elif not force and _matches_failed_generation(cached, provider=provider, model=model):
            return cached
        elif not can_call_model:
            return cached
    if cached:
        logger.info(
            "[SectionOutline] Ignore stale cache doc=%s parse_match=%s source_match=%s",
            doc_id,
            _matches_parse_identity(cached, parse_generation, document_source_hash),
            cached.get("source_hash") == source_hash,
        )

    if not can_call_model:
        return fallback

    try:
        logger.info(
            "[AI-Audit] purpose=section_outline doc=%s provider=%s model=%s status=start",
            doc_id,
            provider,
            model,
        )
        generated = await _generate_ai_section_outline(
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )
        outline = _normalize_section_outline(
            raw=generated,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            source_hash=source_hash,
            source="ai",
            model=model,
            provider=provider,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
        if not outline.get("items"):
            raise ValueError("empty section outline")
        if not _section_outline_quality_ok(outline, fallback, block_index):
            raise ValueError("low-quality section outline")
    except Exception as exc:
        logger.warning(
            "[AI-Audit] purpose=section_outline doc=%s provider=%s model=%s status=failed error=%s",
            doc_id,
            provider,
            model,
            exc,
        )
        if healthy_cached is not None:
            logger.warning(
                "[SectionOutline] Refresh failed for %s; preserving verified AI cache: %s",
                doc_id,
                exc,
            )
            return healthy_cached
        fallback.setdefault("meta", {})["generation_error"] = str(exc)
        fallback.setdefault("meta", {})["generation_error_at"] = time.time()
        fallback.setdefault("meta", {})["provider"] = provider
        fallback.setdefault("meta", {})["model"] = model
        writer = cache_writer or (lambda value: save_section_outline(data_dir, doc_id, value))
        writer(fallback)
        return fallback

    # A route can inject a generation fence here.  Keep it outside the model
    # failure handler so a stale result cannot be reported as a valid fallback.
    writer = cache_writer or (lambda value: save_section_outline(data_dir, doc_id, value))
    writer(outline)
    logger.info(
        "[AI-Audit] purpose=section_outline doc=%s provider=%s model=%s status=success",
        doc_id,
        provider,
        model,
    )
    return outline


def _matches_failed_generation(
    cached: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> bool:
    """相同模型的自动生成失败缓存可复用，手动 force 会在调用前绕过。"""
    meta = cached.get("meta") if isinstance(cached.get("meta"), dict) else {}
    if not str(meta.get("generation_error") or "").strip():
        return False
    if (
        str(meta.get("provider") or "").strip().lower() != str(provider or "").strip().lower()
        or str(meta.get("model") or "").strip() != str(model or "").strip()
    ):
        return False
    try:
        age = time.time() - float(meta.get("generation_error_at"))
    except (TypeError, ValueError):
        return False
    return 0 <= age < FAILED_GENERATION_COOLDOWN_SECONDS


def _matches_ai_generation(
    cached: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> bool:
    return (
        str(cached.get("provider") or "").strip().lower()
        == str(provider or "").strip().lower()
        and str(cached.get("model") or "").strip() == str(model or "").strip()
    )


def build_fallback_section_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str | None = None,
    parse_generation: str = "",
    document_source_hash: str = "",
    source: str = "heuristic",
) -> dict[str, Any]:
    raw_items = []
    for idx, item in enumerate(get_structural_outline_items(block_index)):
        if not isinstance(item, dict):
            continue
        raw_items.append({
            "id": item.get("section_id") or f"section_{idx + 1}",
            "section_id": item.get("section_id") or f"s{idx + 1}",
            "title": item.get("title"),
            "level": item.get("level") or 1,
            "page": item.get("page") or 1,
            "first_block": item.get("first_block"),
            "source": item.get("source"),
        })

    if not _outline_items_quality_ok(raw_items):
        raw_items = _heading_outline_items_from_blocks(block_index)

    raw = {"items": raw_items}
    return _normalize_section_outline(
        raw=raw,
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash or _source_hash(block_index),
        source=source,
        model="",
        provider="",
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )


def _mineru_structure_is_navigation_ready(
    block_index: dict[str, Any],
    outline: dict[str, Any],
) -> bool:
    """Prefer MinerU's published heading tree over a second LLM reconstruction."""
    if str(block_index.get("source") or "").strip().lower() != "mineru_vlm":
        return False
    mineru_meta = block_index.get("mineru_meta")
    if not isinstance(mineru_meta, dict) or mineru_meta.get("structure_degraded") is not False:
        return False
    items = outline.get("flat_items") or _flatten_outline_items(outline.get("items") or [])
    return _outline_items_quality_ok(items)


def get_structural_outline_items(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    """Return navigation anchors, completing only unverifiable MinerU outlines.

    Modern MinerU indexes publish a structure diagnostic.  When it confirms
    complete heading coverage, the published outline remains the sole source
    of truth.  Older indexes may have the right body blocks but an incomplete
    heading list because their raw MinerU artifact is no longer available for
    an upgrade.  In that narrow case, merge deterministic heading recovery
    into the existing anchors so downstream consumers do not silently lose
    sections or fall back to a flat document excerpt.
    """
    published = [
        dict(item)
        for item in (block_index.get("outline") or [])
        if isinstance(item, dict)
    ]
    source = str(block_index.get("source") or "").strip().lower()
    mineru_meta = block_index.get("mineru_meta")
    structure_verified = (
        source.startswith("mineru")
        and isinstance(mineru_meta, dict)
        and mineru_meta.get("structure_degraded") is False
    )
    if not source.startswith("mineru") or structure_verified:
        return published

    recovered = recover_section_outline_items(block_index)
    if not recovered:
        return published
    if published and all(_is_generic_outline_title(str(item.get("title") or "")) for item in published):
        # A lone "Full text" bookmark is a parser placeholder, not a real
        # section.  Keeping it beside recovered headings makes it a false root
        # in the reading skeleton and causes its evidence range to swallow the
        # document again.
        published = []
    return _merge_recovered_outline_items(published, recovered, block_index)


def _merge_recovered_outline_items(
    published: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
    block_index: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add missing recovered headings without disturbing published anchors."""
    merged = [dict(item) for item in published]
    by_anchor: dict[str, int] = {}
    by_title_page: dict[tuple[str, int], int] = {}

    def normalized_title(item: dict[str, Any]) -> str:
        return " ".join(str(item.get("title") or "").split()).casefold()

    def page_number(item: dict[str, Any]) -> int:
        try:
            return max(1, int(item.get("page") or 1))
        except (TypeError, ValueError):
            return 1

    for index, item in enumerate(merged):
        anchor = str(item.get("first_block") or "").strip()
        if anchor:
            by_anchor.setdefault(anchor, index)
        title = normalized_title(item)
        if title:
            by_title_page.setdefault((title, page_number(item)), index)

    for recovered_item in recovered:
        item = dict(recovered_item)
        anchor = str(item.get("first_block") or "").strip()
        if anchor and anchor in by_anchor:
            continue
        title = normalized_title(item)
        title_page_key = (title, page_number(item)) if title else None
        # A legacy outline can name the correct heading but point at a nearby
        # paragraph.  Keep its section id while repairing that anchor from the
        # deterministic recovery result.
        existing_index = by_title_page.get(title_page_key) if title_page_key else None
        if existing_index is not None:
            existing = merged[existing_index]
            if anchor:
                existing["first_block"] = anchor
                by_anchor[anchor] = existing_index
            existing.setdefault("source", item.get("source") or "mineru_heading_recovery")
            continue
        merged.append(item)
        new_index = len(merged) - 1
        if anchor:
            by_anchor[anchor] = new_index
        if title_page_key:
            by_title_page[title_page_key] = new_index

    positions: dict[str, tuple[int, int]] = {}
    for page_index, page in enumerate(block_index.get("pages") or []):
        if not isinstance(page, dict):
            continue
        try:
            page_num = max(1, int(page.get("page") or page_index + 1))
        except (TypeError, ValueError):
            page_num = page_index + 1
        for block_order, block in enumerate(page.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if block_id:
                positions.setdefault(block_id, (page_num, block_order))

    def reading_order(item_with_index: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, item = item_with_index
        anchor = str(item.get("first_block") or "").strip()
        position = positions.get(anchor)
        if position is not None:
            return position[0], position[1], index
        return page_number(item), 1_000_000, index

    return [item for _, item in sorted(enumerate(merged), key=reading_order)]

def recover_section_outline_items(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    """从块序列恢复稳定的章节锚点，兼容 MinerU 将标题标成 text 的结果。"""
    items: list[dict[str, Any]] = []
    is_mineru = str(block_index.get("source") or "").strip().lower().startswith("mineru")
    body_started = False
    references_seen = False
    appendix_started = False
    bare_appendix_started = False
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block_order, block in enumerate(page.get("blocks", []) or []):
            title = _clean_title(block.get("text"))
            if not title or _is_generic_outline_title(title) or _is_noise_text(title):
                continue
            signals = _candidate_signals(block)
            recovered_mineru_heading = bool(
                is_mineru
                and block.get("type") == "paragraph"
                and _looks_like_short_heading(title)
            )
            if block.get("type") != "heading" and not signals and not recovered_mineru_heading:
                continue
            if _looks_like_non_section_title(title):
                continue

            is_appendix = bool(re.match(
                r"^\s*(?:appendix|appendices|supplementary\s+material)\b",
                title,
                re.IGNORECASE,
            ))
            bare_appendix_match = re.match(
                r"^\s*([A-Z])(?:\.(\d+(?:\.\d+)*))?[.)]?\s+(.+)$",
                title,
            )
            is_bare_appendix = bool(
                bare_appendix_match
                and _looks_like_short_heading(title)
                and not _looks_like_post_references_noise(title)
            )
            if references_seen and not appendix_started:
                if not (is_appendix or is_bare_appendix):
                    continue
                appendix_started = True
                bare_appendix_started = is_bare_appendix and not is_appendix
            elif references_seen:
                if _looks_like_post_references_noise(title):
                    continue
                if bare_appendix_started and not is_bare_appendix:
                    continue

            structured_heading = bool(set(signals) & {
                "heuristic_heading",
                "canonical_heading",
                "numbered_heading",
                "roman_heading",
                "alpha_subsection",
            })
            # MinerU content_list 常把论文题名也标成 text。正文尚未开始时，
            # 跳过首页最前面的非结构化短行，避免把论文题名当第一章。
            if (
                page_num == 1
                and not body_started
                and block_order < 5
                and not structured_heading
            ):
                continue

            if is_reference_heading(title):
                references_seen = True
            if structured_heading or recovered_mineru_heading:
                body_started = True

            level = _safe_level(block.get("level"), 1)
            numbered = _RE_NUMBERED_HEADING.match(title)
            if numbered:
                level = min(4, len(numbered.group(1).split(".")))
            elif references_seen and is_bare_appendix and bare_appendix_match:
                appendix_path = bare_appendix_match.group(2)
                level = 1 + len(appendix_path.split(".")) if appendix_path else 1
            elif _RE_ALPHA_HEADING.match(title):
                level = 2
            elif _RE_ROMAN_HEADING.match(title):
                level = 1

            block_id = str(block.get("block_id") or f"p{page_num}_heading_{len(items) + 1}")
            items.append({
                "id": f"section_{len(items) + 1}",
                "section_id": f"recovered_{block_id}",
                "title": title,
                "level": level,
                "page": page_num,
                "first_block": block_id,
                "source": "mineru_heading_recovery" if recovered_mineru_heading else "heading",
            })
            if len(items) >= MAX_CANDIDATES:
                return items
    return items


def _heading_outline_items_from_blocks(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    return recover_section_outline_items(block_index)


def _outline_items_quality_ok(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    titles = [
        _clean_title(item.get("title"))
        for item in items
        if isinstance(item, dict) and _clean_title(item.get("title"))
    ]
    if not titles:
        return False
    if len(titles) < 2:
        return False
    sources = {
        str(item.get("source") or "").lower()
        for item in items
        if isinstance(item, dict)
    }
    if sources == {"toc"} and len(titles) < 3:
        return False
    useful_titles = [title for title in titles if not _is_generic_outline_title(title)]
    if not useful_titles:
        return False
    suspicious_count = sum(1 for title in useful_titles if _looks_like_non_section_title(title))
    return suspicious_count < max(1, len(useful_titles) // 2)


def _build_bookmark_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> dict[str, Any] | None:
    outline = block_index.get("outline", []) or []
    if not outline or not all((item.get("source") == "toc") for item in outline if isinstance(item, dict)):
        return None
    raw = {"items": outline}
    return _normalize_section_outline(
        raw=raw,
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
        source="toc",
        model="",
        provider="",
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )


async def _generate_ai_section_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, Any]:
    candidates = _select_section_candidates(block_index)
    if not candidates:
        return {}

    system = (
        "你是学术 PDF 章节结构解析器。你的任务是从候选文本块中还原原文的标题、章节和子章节目录树，"
        "用于 PDF 导航。只输出 JSON，不要输出 Markdown 或解释。"
    )
    user = (
        "请生成原文档的章节目录树，不要生成“研究背景/核心创新/实验结果/结论价值”这类总结框架，"
        "除非这些词本来就是原文章节标题。\n\n"
        "输出 JSON 格式：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\"id\":\"sec_1\",\"title\":\"Abstract\",\"level\":1,\"page\":1,\"first_block\":\"p1_b2\",\"children\":[]},\n"
        "    {\"id\":\"sec_2\",\"title\":\"I. Introduction\",\"level\":1,\"page\":1,\"first_block\":\"p1_b5\",\"children\":[\n"
        "      {\"id\":\"sec_2_1\",\"title\":\"A. Problem Setup\",\"level\":2,\"page\":2,\"first_block\":\"p2_b3\",\"children\":[]}\n"
        "    ]}\n"
        "  ]\n"
        "}\n\n"
        "规则：\n"
        "1. title 尽量保留原文标题和编号；不要输出论文题名、作者、单位等首页元信息，通常从 Abstract/Introduction 开始。\n"
        "2. 必须使用候选里的 block_id 作为 first_block，不允许编造 ID。\n"
        "3. 过滤页眉页脚、页码、LPIPS/PSNR 等纯指标数值行、Fig/Table 标签和表格标题。\n"
        "4. 正确识别罗马数字章节（II.）为 level 1，字母子节（A.）通常为 level 2。\n"
        "5. 算法伪代码步骤（Input/Output/Update/Initialize/Return 等）和图内标注不是章节。\n"
        "6. 不要输出作者姓名、作者单位、邮箱、通讯作者脚注、会议/期刊页眉、参考文献条目；References 章节标题本身可以保留，但 References 下面的编号文献不是子章节。\n"
        "7. 若候选 signals 包含 layout_title，说明它来自 DocLayout-YOLO Title 区域；优先保留其文本，只判断层级和父子关系。\n"
        "8. 层级必须连续，不要从 level 1 直接跳到 level 3；最多 4 级。\n"
        "9. 若不确定，宁可少输出，不要把正文句子当章节标题。\n\n"
        f"文档名：{doc.get('filename', doc_id)}\n"
        f"候选块：{json.dumps(candidates, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    custom_params = structured_json_request_params(provider, model)
    response = await call_ai_api(
        messages=messages,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=3600,
        temperature=0.1,
        custom_params=custom_params,
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))
    content = _extract_content(response)
    try:
        require_publishable_completion(response, operation="section outline")
        return parse_json_object(content, allow_partial=False)
    except (StructuredJSONError, IncompleteCompletionError):
        logger.info("[SectionOutline] Invalid JSON from %s/%s; retrying once", provider, model)

    retry_response = await call_ai_api(
        messages=[
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上面的输出不是合法 JSON。请重新输出完整、严格合法的 JSON 对象；"
                    "保留既定字段、章节层级和 first_block，只输出 JSON，不要解释或 Markdown。"
                ),
            },
        ],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=3600,
        temperature=0,
        custom_params=custom_params,
    )
    if isinstance(retry_response, dict) and retry_response.get("error"):
        raise RuntimeError(retry_response.get("error"))
    try:
        require_publishable_completion(retry_response, operation="section outline retry")
        return parse_json_object(_extract_content(retry_response), allow_partial=False)
    except (StructuredJSONError, IncompleteCompletionError) as exc:
        raise StructuredJSONError("模型两次返回的章节结构格式均不完整") from exc


def _normalize_section_outline(
    *,
    raw: dict[str, Any],
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str,
    source: str,
    model: str,
    provider: str,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> dict[str, Any]:
    block_map = _flatten_blocks(block_index)
    used_ids: set[str] = set()
    raw_items = raw.get("items") if isinstance(raw, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []
    raw_has_children = any(isinstance(item, dict) and item.get("children") for item in raw_items)

    def normalize_node(node: dict[str, Any], idx_path: str, inherited_level: int = 1) -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        title = _clean_title(node.get("title") or node.get("label") or node.get("name"))
        if not title or _is_noise_text(title):
            return None

        raw_id = str(node.get("id") or node.get("section_id") or f"sec_{idx_path}").strip()
        node_id = _unique_id(_slug_id(raw_id) or f"sec_{idx_path}", used_ids)
        level = _safe_level(node.get("level"), inherited_level)
        raw_first_block = _valid_first_block(node.get("first_block") or node.get("block_id"), block_map)
        evidence_ids = _valid_block_ids(node.get("evidence_block_ids") or node.get("block_ids"), block_map)
        requested_page = _safe_page(node.get("page"), block_map.get(raw_first_block) if raw_first_block else None)
        if requested_page == 1 and _looks_like_front_matter_title(title):
            return None
        first_block = _resolve_section_anchor(
            title=title,
            requested_page=requested_page,
            raw_block_id=raw_first_block,
            evidence_ids=evidence_ids,
            block_map=block_map,
        )
        evidence_ids = _normalize_section_evidence(first_block, evidence_ids, block_map)

        page = _safe_page(node.get("page"), block_map.get(first_block) if first_block else None)
        children = []
        for child_idx, child in enumerate(node.get("children") or []):
            child_node = normalize_node(child, f"{idx_path}_{child_idx + 1}", level + 1)
            if child_node:
                children.append(child_node)

        pages = sorted({int(block_map[bid]["page"]) for bid in evidence_ids if bid in block_map})
        if not pages and page:
            pages = [page]

        return {
            "id": node_id,
            "section_id": node.get("section_id") or node_id,
            "title": title,
            "level": level,
            "page": page,
            "first_block": first_block,
            "evidence_block_ids": evidence_ids[:4],
            "evidence": {
                "block_ids": evidence_ids[:4],
                "pages": pages,
                "primary_page": page,
            },
            "source": source,
            "children": children,
        }

    items = []
    for idx, item in enumerate(raw_items):
        normalized = normalize_node(item, str(idx + 1))
        if normalized:
            items.append(normalized)

    if items and not raw_has_children:
        items = _nest_flat_items(items)

    if not items:
        first_page = (block_index.get("pages") or [{}])[0].get("page", 1)
        first_block = _first_block_id(block_index, first_page)
        items = [{
            "id": "full_text",
            "section_id": "full_text",
            "title": "全文",
            "level": 1,
            "page": int(first_page or 1),
            "first_block": first_block,
            "evidence_block_ids": [first_block] if first_block else [],
            "evidence": {
                "block_ids": [first_block] if first_block else [],
                "pages": [int(first_page or 1)],
                "primary_page": int(first_page or 1),
            },
            "source": source,
            "children": [],
        }]

    return {
        "version": SECTION_OUTLINE_VERSION,
        "doc_id": doc_id,
        "source": source,
        "source_hash": source_hash,
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
        "title": _filename_title(doc),
        "items": items,
        "flat_items": _flatten_outline_items(items),
        "created_at": time.time(),
        "model": model,
        "provider": provider,
        "meta": {
            "candidate_count": len(_select_section_candidates(block_index)),
            "block_count": len(block_map),
            "page_count": len(block_index.get("pages", []) or []),
        },
    }


def _section_outline_quality_ok(
    outline: dict[str, Any],
    fallback: dict[str, Any],
    block_index: dict[str, Any],
) -> bool:
    ai_count = len(outline.get("flat_items") or _flatten_outline_items(outline.get("items") or []))
    fallback_count = len(fallback.get("flat_items") or _flatten_outline_items(fallback.get("items") or []))
    if ai_count <= 0:
        return False
    if fallback_count >= 6 and ai_count < max(3, int(fallback_count * 0.5)):
        return False

    titles = [
        str(item.get("title") or "")
        for item in (outline.get("flat_items") or _flatten_outline_items(outline.get("items") or []))
    ]
    suspicious_count = sum(1 for title in titles if _looks_like_non_section_title(title))
    if suspicious_count and suspicious_count >= max(1, ai_count // 3):
        return False
    if not _outline_anchors_are_monotonic(outline, block_index):
        return False
    return True


def _bookmark_outline_quality_ok(
    outline: dict[str, Any],
    fallback: dict[str, Any],
    block_index: dict[str, Any],
) -> bool:
    items = outline.get("flat_items") or _flatten_outline_items(outline.get("items") or [])
    fallback_items = fallback.get("flat_items") or _flatten_outline_items(fallback.get("items") or [])
    item_count = len(items)
    fallback_count = len(fallback_items)
    if item_count <= 0:
        return False
    titles = [str(item.get("title") or "").strip() for item in items]
    normalized_titles = {
        re.sub(r"\s+", " ", title.lower()).strip()
        for title in titles
        if title
    }
    if item_count == 1:
        return False
    if fallback_count >= 6 and item_count < max(3, int(fallback_count * 0.5)):
        return False
    suspicious_count = sum(1 for title in titles if _looks_like_non_section_title(title))
    if suspicious_count and suspicious_count >= max(1, item_count // 3):
        return False
    return _outline_anchors_are_monotonic(outline, block_index)


def _outline_anchors_are_monotonic(
    outline: dict[str, Any],
    block_index: dict[str, Any],
) -> bool:
    """Reject outlines that move backwards through the published reading order.

    Heading numbering is only a signal: unnumbered sections, appendices and
    front matter make it unsafe as a sorting key.  ``first_block`` anchors are
    stronger evidence because every downstream feature already consumes the
    block-index reading order.  Unknown anchors remain allowed so an otherwise
    valid outline can still be normalized or repaired by the fallback path.
    """
    positions: dict[str, tuple[int, int]] = {}
    for page_index, page in enumerate(block_index.get("pages") or []):
        page_num = int(page.get("page") or page_index + 1)
        for block_index_on_page, block in enumerate(page.get("blocks") or []):
            block_id = str(block.get("block_id") or "").strip()
            if block_id:
                positions[block_id] = (page_num, block_index_on_page)

    previous: tuple[int, int] | None = None
    items = outline.get("flat_items") or _flatten_outline_items(outline.get("items") or [])
    for item in items:
        if not isinstance(item, dict):
            continue
        position = positions.get(str(item.get("first_block") or "").strip())
        if position is None:
            continue
        if previous is not None and position < previous:
            return False
        previous = position
    return True


def _select_section_candidates(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = list(_flatten_blocks(block_index).values())
    selected: dict[str, dict[str, Any]] = {}
    is_mineru = str(block_index.get("source") or "").strip().lower().startswith("mineru")
    yolo_primary = any(
        block.get("layout_primary_classifier") == "doclayout_yolo"
        for block in ordered
    )

    def add(block: dict[str, Any], signal: str) -> None:
        block_id = str(block.get("block_id") or "")
        text = str(block.get("text") or "").strip()
        if not block_id or not text or _is_noise_text(text):
            return
        if yolo_primary and block.get("layout_region") != "Title":
            return
        if yolo_primary and int(block.get("page") or 1) == 1 and _looks_like_front_matter_title(text):
            return
        item = selected.get(block_id)
        if not item:
            item = {
                "block_id": block_id,
                "page": int(block.get("page") or 1),
                "type": block.get("type") or "paragraph",
                "text": _limit(text, MAX_CANDIDATE_TEXT),
                "font_size": block.get("font_size"),
                "is_bold": bool(block.get("is_bold")),
                "level_hint": block.get("level"),
                "layout_region": block.get("layout_region") or "",
                "layout_score": block.get("layout_score"),
                "signals": [],
            }
            selected[block_id] = item
        if signal not in item["signals"]:
            item["signals"].append(signal)

    for block in ordered:
        signals = _candidate_signals(block, yolo_primary=yolo_primary)
        if (
            is_mineru
            and not yolo_primary
            and block.get("type") == "paragraph"
            and _looks_like_short_heading(str(block.get("text") or ""))
        ):
            signals.append("mineru_short_heading")
        for signal in signals:
            add(block, signal)

    if not yolo_primary:
        seen_pages: set[int] = set()
        for block in ordered:
            page = int(block.get("page") or 1)
            if page in seen_pages or block.get("type") == "artifact":
                continue
            text = str(block.get("text") or "").strip()
            if text and not _is_noise_text(text):
                add(block, "page_first_block")
                seen_pages.add(page)

    return list(selected.values())[:MAX_CANDIDATES]


def _candidate_signals(block: dict[str, Any], *, yolo_primary: bool = False) -> list[str]:
    text = " ".join(str(block.get("text") or "").split())
    if not text or _is_noise_text(text):
        return []
    if yolo_primary:
        if block.get("type") == "heading" and block.get("layout_region") == "Title":
            return ["layout_title"]
        return []
    signals = []
    if block.get("type") == "heading":
        signals.append("heuristic_heading")
    if _RE_CANONICAL_HEADING.match(text):
        signals.append("canonical_heading")
    if _RE_NUMBERED_HEADING.match(text):
        signals.append("numbered_heading")
    if _RE_ROMAN_HEADING.match(text):
        signals.append("roman_heading")
    if _RE_ALPHA_HEADING.match(text):
        signals.append("alpha_subsection")
    if bool(block.get("is_bold")) and _looks_like_short_heading(text):
        signals.append("bold_short_line")
    return signals


def _looks_like_short_heading(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if len(stripped) > 180 or stripped.endswith((".", ",", ";", ":")):
        return False
    if _looks_like_non_section_title(stripped):
        return False
    words = stripped.split()
    if not 1 <= len(words) <= 16:
        return False
    if _RE_CAPTION_OR_LABEL.match(stripped):
        return False
    if _RE_CANONICAL_HEADING.match(stripped):
        return True
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return True
    title_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", stripped)
    if not title_words:
        return False
    small_words = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}
    content_words = [word for word in title_words if word.lower() not in small_words]
    if not content_words:
        return False
    capitalized = sum(1 for word in content_words if word[:1].isupper())
    return capitalized / max(len(content_words), 1) >= 0.55


def _flatten_blocks(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    section_titles: dict[str, str] = {}
    for page in block_index.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            section_id = str(block.get("section_id") or "").strip()
            text = str(block.get("text") or "").strip()
            if section_id and block.get("type") == "heading" and text:
                section_titles.setdefault(section_id, text)

    result: dict[str, dict[str, Any]] = {}
    current_heading = ""
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            text = str(block.get("text") or "").strip()
            if block.get("type") == "heading" and text:
                current_heading = text
            section_id = str(block.get("section_id") or "").strip()
            section_title = section_titles.get(section_id) or current_heading
            item = dict(block)
            item["page"] = page_num
            item["text"] = text
            item["section_title"] = section_title
            result[block_id] = annotate_block_role(item, section_title=section_title)
    return result


def _nest_flat_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for item in items:
        node = dict(item)
        node["children"] = list(node.get("children") or [])
        level = int(node.get("level") or 1)
        while stack and int(stack[-1].get("level") or 1) >= level:
            stack.pop()
        if stack:
            stack[-1].setdefault("children", []).append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _flatten_outline_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], level: int) -> None:
        for node in nodes:
            item = {k: v for k, v in node.items() if k != "children"}
            item["level"] = level
            flat.append(item)
            walk(node.get("children") or [], level + 1)

    walk(items, 1)
    return flat


def _valid_first_block(value: Any, block_map: dict[str, dict[str, Any]]) -> str | None:
    block_id = str(value or "").strip()
    return block_id if block_id in block_map else None


def _valid_block_ids(value: Any, block_map: dict[str, dict[str, Any]]) -> list[str]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    result: list[str] = []
    for item in raw:
        block_id = str(item or "").strip()
        if block_id and block_id in block_map and block_id not in result:
            result.append(block_id)
    return result


def _resolve_section_anchor(
    *,
    title: str,
    requested_page: int | None,
    raw_block_id: str | None,
    evidence_ids: list[str],
    block_map: dict[str, dict[str, Any]],
) -> str | None:
    raw_block = block_map.get(raw_block_id or "")
    if (
        raw_block
        and _title_match_score(title, str(raw_block.get("text") or "")) == 4
        and _block_is_valid_section_anchor(raw_block, title, allow_loose=False)
    ):
        return raw_block_id

    for page in (requested_page, None):
        matched = _match_block_by_title(title, block_map, page=page, heading_only=True)
        if matched:
            return matched

    for block_id in evidence_ids:
        block = block_map.get(block_id)
        if block and _block_is_valid_section_anchor(block, title, allow_loose=False):
            return block_id

    for page in (requested_page, None):
        matched = _match_block_by_title(title, block_map, page=page, heading_only=False)
        if matched:
            return matched

    if requested_page:
        return _first_section_block_id(block_map, requested_page)
    return None


def _normalize_section_evidence(
    first_block: str | None,
    evidence_ids: list[str],
    block_map: dict[str, dict[str, Any]],
) -> list[str]:
    result: list[str] = []
    if first_block:
        result.append(first_block)
    for block_id in evidence_ids:
        if block_id in result:
            continue
        block = block_map.get(block_id)
        if not block or _block_is_publication_header(block):
            continue
        if block.get("type") == "artifact":
            continue
        result.append(block_id)
    return result


def _match_block_by_title(
    title: str,
    block_map: dict[str, dict[str, Any]],
    *,
    page: int | None = None,
    heading_only: bool = False,
) -> str | None:
    for allow_loose in (False, True):
        best: tuple[int, str] | None = None
        for block in block_map.values():
            if page and int(block.get("page") or 1) != int(page):
                continue
            if heading_only and block.get("type") != "heading":
                continue
            if _block_is_valid_section_anchor(block, title, allow_loose=allow_loose):
                block_id = str(block.get("block_id") or "")
                score = _title_match_score(title, str(block.get("text") or ""))
                if block_id and (best is None or score > best[0]):
                    best = (score, block_id)
        if best:
            return best[1]
    return None


def _block_is_valid_section_anchor(block: dict[str, Any], title: str, *, allow_loose: bool) -> bool:
    block_type = block.get("type") or "paragraph"
    if block_type in {"artifact", "caption", "figure", "table"}:
        return False
    text = str(block.get("text") or "").strip()
    if not text or _block_is_publication_header(block):
        return False
    score = _title_match_score(title, text)
    if block_type == "heading" and score >= 3:
        return True
    if _is_noise_text(text):
        return False
    if score >= 3:
        return True
    if block_type == "heading" and score >= 2:
        return True
    return allow_loose and score >= 2 and _block_text_starts_with_title(title, text)


def _first_section_block_id(block_map: dict[str, dict[str, Any]], page_num: int) -> str | None:
    page_blocks = [
        block for block in block_map.values()
        if int(block.get("page") or 1) == int(page_num) and not _block_is_publication_header(block)
    ]
    for block in page_blocks:
        if block.get("type") == "heading" and block.get("block_id"):
            return str(block["block_id"])
    for block in page_blocks:
        if block.get("block_id") and block.get("type") not in {"artifact", "caption", "figure", "table"}:
            return str(block["block_id"])
    return None


def _block_is_publication_header(block: dict[str, Any]) -> bool:
    text = " ".join(str(block.get("text") or "").split())
    if not text:
        return True
    if block.get("type") == "artifact":
        return True
    role = classify_block_role(
        block,
        section_title=str(block.get("section_title") or ""),
    )["role"]
    if role in FRONT_MATTER_ROLES or role == ROLE_PUBLICATION_HEADER:
        return True
    return bool(_RE_PUBLICATION_HEADER_CUE.search(text)) and len(text.split()) <= 18


def _title_match_score(title: str, text: str) -> int:
    title_variants = _norm_title_variants(title)
    text_variants = _norm_title_variants(text)
    title_variants = {item for item in title_variants if item}
    text_variants = {item for item in text_variants if item}
    if not title_variants or not text_variants:
        return 0
    if title_variants & text_variants:
        return 4
    for title_norm in title_variants:
        if len(title_norm) < 4:
            continue
        for text_norm in text_variants:
            if text_norm.startswith(title_norm) or title_norm.startswith(text_norm):
                return 3
            if title_norm in text_norm:
                return 2
    return 0


def _block_text_starts_with_title(title: str, text: str) -> bool:
    title_variants = _norm_title_variants(title)
    text_variants = _norm_title_variants(text)
    return any(
        text_norm.startswith(title_norm)
        for title_norm in title_variants
        for text_norm in text_variants
        if len(title_norm) >= 4
    )


def _norm_title_variants(value: str) -> set[str]:
    norm = _norm_text(value)
    if not norm:
        return set()
    variants = {norm}
    stripped = _strip_heading_prefix_norm(norm)
    if stripped:
        variants.add(stripped)
    return variants


def _strip_heading_prefix_norm(norm: str) -> str:
    text = " ".join(str(norm or "").split())
    if not text:
        return ""
    text = re.sub(r"^(?:\d+\s+){1,4}", "", text).strip()
    text = re.sub(
        r"^(?:[ivxlcm]+|[a-z])(?:\s+\d+){0,4}\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    parts = text.split()
    if len(parts) >= 2 and re.fullmatch(r"[ivxlcm]+|[a-z]", parts[0], re.IGNORECASE):
        text = " ".join(parts[1:]).strip()
    return text


def _first_block_id(block_index: dict[str, Any], page_num: int) -> str | None:
    for page in block_index.get("pages", []) or []:
        if int(page.get("page") or 1) != int(page_num or 1):
            continue
        for block in page.get("blocks", []) or []:
            if block.get("block_id") and block.get("type") != "artifact":
                return str(block["block_id"])
    return None


def _safe_level(value: Any, fallback: int) -> int:
    try:
        level = int(value)
    except (TypeError, ValueError):
        level = fallback
    return max(1, min(level, 6))


def _safe_page(value: Any, block: dict[str, Any] | None) -> int:
    if block and block.get("page"):
        return int(block.get("page") or 1)
    try:
        page = int(value)
    except (TypeError, ValueError):
        page = 1
    return max(1, page)


def _is_noise_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if len(stripped) < 2:
        return True
    lower = stripped.lower()
    if "arxiv:" in lower:
        return True
    if _RE_CAPTION_OR_LABEL.match(stripped):
        return True
    if _looks_like_non_section_title(stripped):
        return True
    if re.match(r"^\s*\d{3,}\s+[A-Z]", stripped) and _RE_PUBLICATION_HEADER_CUE.search(stripped):
        return True
    if re.fullmatch(r"[\d\s.,:/()%+\-=×xX<>~]+", stripped):
        return True
    if _looks_like_numeric_measurement_line(stripped):
        return True
    compact = re.sub(r"\s+", "", stripped)
    if compact and sum(ch.isdigit() for ch in compact) / max(len(compact), 1) > 0.50:
        return True
    return False


def _is_generic_outline_title(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return normalized in _GENERIC_BOOKMARK_TITLES


def _looks_like_non_section_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    if _RE_KEYWORDS_LINE.match(stripped):
        return True
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        try:
            first_number = int(numbered.group(1).split(".")[0])
        except ValueError:
            first_number = 0
        if first_number > 50 or len(numbered.group(2).strip()) > 80:
            return True
    if _RE_ALGORITHM_LINE.match(stripped):
        return True
    if re.search(r"\b(stage|step)\s*\d+\b", stripped, re.IGNORECASE) and re.search(r"\b(update|initialize|input|output)\b", stripped, re.IGNORECASE):
        return True
    if _looks_like_reference_entry(stripped):
        return True
    if _looks_like_affiliation_or_author_line(stripped):
        return True
    if _looks_like_run_in_paragraph_heading(stripped):
        return True
    return False


def _looks_like_front_matter_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    return True


def _looks_like_run_in_paragraph_heading(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_CANONICAL_HEADING.match(stripped) or _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    words = stripped.split()
    if len(words) > 12:
        return True
    if stripped.endswith(".") and len(words) >= 4:
        return True
    if re.search(r"\.\s+[A-Z]", stripped):
        return True
    if len(words) >= 9 and re.search(r"\[[0-9,\s]+\].*\b(is|are|was|were|has|have)\b", stripped, re.IGNORECASE):
        return True
    if len(words) >= 9 and re.search(r"\b(is|are|was|were)\s+(a|an|the|more)\b", stripped, re.IGNORECASE):
        return True
    return False


def _looks_like_reference_entry(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if _RE_REFERENCE_ENTRY.match(stripped):
        return True
    if _RE_AUTHOR_INITIAL_REFERENCE.match(stripped):
        return True
    numbered = _RE_NUMBERED_HEADING.match(stripped)
    if numbered:
        try:
            first_number = int(numbered.group(1).split(".")[0])
        except ValueError:
            first_number = 0
        if first_number > 50:
            return True
        if len(numbered.group(2)) > 80 and re.search(r"\bet\s+al\.?\b|,\s+[A-Z]\.", stripped, re.IGNORECASE):
            return True
    return bool(re.match(r"^\s*\d+[\.)]?\s+", stripped) and re.search(r"\bet\s+al\.?\b", stripped, re.IGNORECASE))


def _looks_like_post_references_noise(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if is_post_reference_template_artifact(stripped):
        return True
    if _RE_CANONICAL_HEADING.match(stripped):
        return False
    return bool(_looks_like_reference_entry(stripped) or re.match(r"^\s*(?:\[\d+\]|\d+[\.)])\s+", stripped))


def _looks_like_affiliation_or_author_line(text: str) -> bool:
    return looks_like_affiliation_or_author_line(text)


def _looks_like_numeric_measurement_line(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False
    if _RE_NUMBERED_HEADING.match(stripped) or _RE_ROMAN_HEADING.match(stripped) or _RE_ALPHA_HEADING.match(stripped):
        return False
    decimal_tokens = _RE_DECIMAL_TOKEN.findall(stripped)
    if not decimal_tokens:
        return False
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", stripped)
    number_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)
    return len(alpha_tokens) <= 2 and (len(number_tokens) >= 2 or len(stripped.split()) <= 8)


def _source_hash(block_index: dict[str, Any]) -> str:
    """Fingerprint the evidence structure, not only the visible text."""
    pages = []
    for page_order, page in enumerate(block_index.get("pages", []) or []):
        if not isinstance(page, dict):
            continue
        blocks = []
        for block_order, block in enumerate(page.get("blocks", []) or []):
            if not isinstance(block, dict):
                continue
            blocks.append({
                "order": block_order,
                "block_id": str(block.get("block_id") or ""),
                "type": str(block.get("type") or ""),
                "text": " ".join(str(block.get("text") or "").split()),
                "level": block.get("level"),
                "section_id": block.get("section_id"),
            })
        pages.append({
            "order": page_order,
            "page": page.get("page"),
            "blocks": blocks,
        })
    payload = {
        "block_index_version": block_index.get("version"),
        "block_index_source": block_index.get("source"),
        "pages": pages,
        "outline": block_index.get("outline") or [],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _parse_identity(doc_id: str, doc: dict[str, Any]) -> tuple[str, str]:
    """Return the active primary parse identity for a cached outline."""
    manifest = read_parse_manifest(doc, doc_id=doc_id)
    return (
        str(manifest.get("generation") or "").strip(),
        str(manifest.get("source_hash") or "").strip(),
    )


def _has_parse_identity(outline: dict[str, Any]) -> bool:
    return bool(
        str(outline.get("parse_generation") or "").strip()
        and str(outline.get("document_source_hash") or "").strip()
    )


def _matches_parse_identity(
    outline: dict[str, Any],
    parse_generation: str,
    document_source_hash: str,
) -> bool:
    return (
        _has_parse_identity(outline)
        and str(outline.get("parse_generation") or "").strip() == parse_generation
        and str(outline.get("document_source_hash") or "").strip() == document_source_hash
    )


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(message.get("content"), str):
                return message["content"]
    return str(response or "")


def _parse_json_object(content: str) -> dict[str, Any]:
    return parse_json_object(content, allow_partial=False)


def _clean_title(value: Any) -> str:
    title = " ".join(str(value or "").split())
    title = re.sub(r"^(title|section)\s*[:：]\s*", "", title, flags=re.IGNORECASE)
    return _limit(title, 180)


def _filename_title(doc: dict[str, Any]) -> str:
    name = str(doc.get("filename") or "当前文档")
    return re.sub(r"\.[A-Za-z0-9]+$", "", name).strip() or "当前文档"


def _slug_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip()).strip("_").lower()
    return text[:72]


def _unique_id(base: str, used_ids: set[str]) -> str:
    node_id = base
    suffix = 2
    while node_id in used_ids:
        node_id = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(node_id)
    return node_id


def _norm_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower())
    return " ".join(text.split())


def _limit(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."
