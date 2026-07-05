"""AI 辅助的文档章节大纲。

这个服务刻意与 reading_outline_service 分离：
- section outline：原文题名、章节、子章节，用于导航
- reading outline：中文 AI 阅读笔记和结构化总结
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from services.chat_service import call_ai_api

logger = logging.getLogger(__name__)

SECTION_OUTLINE_VERSION = 4
MAX_CANDIDATES = 180
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
_RE_ALGORITHM_LINE = re.compile(
    r"^\s*(input|output|require|ensure|initialize|initialise|update|repeat|return|for\s+each|for\s+"
    r"|while\s+|if\s+|else\b|end\b|stage\s*\d+)\b",
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
    path = get_section_outline_path(data_dir, doc_id)
    outline["version"] = SECTION_OUTLINE_VERSION
    outline["doc_id"] = doc_id
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[SectionOutline] Failed to save %s: %s", path, exc)


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
) -> dict[str, Any]:
    """按 PDF 书签 -> AI -> 启发式的顺序返回文档章节树。"""
    source_hash = _source_hash(block_index)
    bookmark_outline = _build_bookmark_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
    )
    if bookmark_outline:
        return bookmark_outline

    provider_lower = (provider or "").lower()
    can_call_model = bool(api_key) or provider_lower in {"local", "ollama"}
    fallback = build_fallback_section_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
    )
    cached = None if force else load_section_outline(data_dir, doc_id)
    if cached and cached.get("source_hash") == source_hash:
        if cached.get("source") == "ai":
            if _section_outline_quality_ok(cached, fallback):
                return cached
            logger.info("[SectionOutline] Ignore low-quality cached AI outline for %s", doc_id)
        elif not can_call_model:
            return cached

    if not can_call_model:
        return fallback

    try:
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
        )
        if not outline.get("items"):
            raise ValueError("empty section outline")
        if not _section_outline_quality_ok(outline, fallback):
            raise ValueError("low-quality section outline")
        save_section_outline(data_dir, doc_id, outline)
        return outline
    except Exception as exc:
        logger.warning("[SectionOutline] AI generation failed for %s: %s", doc_id, exc)
        fallback.setdefault("meta", {})["generation_error"] = str(exc)
        return fallback


def build_fallback_section_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str | None = None,
) -> dict[str, Any]:
    raw_items = []
    for idx, item in enumerate(block_index.get("outline", []) or []):
        if not isinstance(item, dict):
            continue
        raw_items.append({
            "id": item.get("section_id") or f"section_{idx + 1}",
            "section_id": item.get("section_id") or f"s{idx + 1}",
            "title": item.get("title"),
            "level": item.get("level") or 1,
            "page": item.get("page") or 1,
            "first_block": item.get("first_block"),
        })

    raw = {"items": raw_items}
    return _normalize_section_outline(
        raw=raw,
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash or _source_hash(block_index),
        source="heuristic",
        model="",
        provider="",
    )


def _build_bookmark_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str,
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
        "1. title 尽量保留原文标题和编号；可包含论文题名作为第一个 level 1 节点。\n"
        "2. 必须使用候选里的 block_id 作为 first_block，不允许编造 ID。\n"
        "3. 过滤页眉页脚、页码、LPIPS/PSNR 等纯指标数值行、Fig/Table 标签和表格标题。\n"
        "4. 正确识别罗马数字章节（II.）为 level 1，字母子节（A.）通常为 level 2。\n"
        "5. 算法伪代码步骤（Input/Output/Update/Initialize/Return 等）和图内标注不是章节。\n"
        "6. 若不确定，宁可少输出，不要把正文句子当章节标题。\n\n"
        f"文档名：{doc.get('filename', doc_id)}\n"
        f"候选块：{json.dumps(candidates, ensure_ascii=False)}"
    )
    response = await call_ai_api(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=3600,
        temperature=0.1,
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))
    return _parse_json_object(_extract_content(response))


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


def _section_outline_quality_ok(outline: dict[str, Any], fallback: dict[str, Any]) -> bool:
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
    return True


def _select_section_candidates(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = list(_flatten_blocks(block_index).values())
    selected: dict[str, dict[str, Any]] = {}

    def add(block: dict[str, Any], signal: str) -> None:
        block_id = str(block.get("block_id") or "")
        text = str(block.get("text") or "").strip()
        if not block_id or not text or _is_noise_text(text):
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
                "signals": [],
            }
            selected[block_id] = item
        if signal not in item["signals"]:
            item["signals"].append(signal)

    for block in ordered:
        for signal in _candidate_signals(block):
            add(block, signal)

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


def _candidate_signals(block: dict[str, Any]) -> list[str]:
    text = " ".join(str(block.get("text") or "").split())
    if not text or _is_noise_text(text):
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
    result: dict[str, dict[str, Any]] = {}
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            item = dict(block)
            item["page"] = page_num
            item["text"] = str(block.get("text") or "").strip()
            result[block_id] = item
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
    for page in (requested_page, None):
        matched = _match_block_by_title(title, block_map, page=page, heading_only=True)
        if matched:
            return matched

    raw_block = block_map.get(raw_block_id or "")
    if raw_block and _block_is_valid_section_anchor(raw_block, title, allow_loose=False):
        return raw_block_id

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
        for block in block_map.values():
            if page and int(block.get("page") or 1) != int(page):
                continue
            if heading_only and block.get("type") != "heading":
                continue
            if _block_is_valid_section_anchor(block, title, allow_loose=allow_loose):
                return str(block.get("block_id") or "")
    return None


def _block_is_valid_section_anchor(block: dict[str, Any], title: str, *, allow_loose: bool) -> bool:
    block_type = block.get("type") or "paragraph"
    if block_type in {"artifact", "caption", "figure", "table"}:
        return False
    text = str(block.get("text") or "").strip()
    if not text or _is_noise_text(text) or _block_is_publication_header(block):
        return False
    score = _title_match_score(title, text)
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


def _looks_like_non_section_title(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if _RE_ALGORITHM_LINE.match(stripped):
        return True
    if re.search(r"\b(stage|step)\s*\d+\b", stripped, re.IGNORECASE) and re.search(r"\b(update|initialize|input|output)\b", stripped, re.IGNORECASE):
        return True
    return False


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
    parts = [f"version:{block_index.get('version')}"]
    for block in _flatten_blocks(block_index).values():
        parts.append(
            f"{block.get('block_id')}:{block.get('type')}:{block.get('level')}:{block.get('is_bold')}:{block.get('text')}"
        )
    payload = "\n".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


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
