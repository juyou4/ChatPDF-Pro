"""Deterministic, parse-identity-bound block inventory helpers.

Semantic retrieval answers relevance questions.  It must not pretend to be a
complete enumeration when a user asks for every formula, table, figure, or
reference in a document.  This module reads the published block index in page
order and exposes explicit pagination/coverage metadata instead.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


_EXPLICIT_FULL_SCOPE_RE = re.compile(
    r"(?:所有|全部|全量|逐[一個个]|每(?:一)?(?:个|條|条|項|项)|完整(?:列出|清单|列表)?|总数|一共|"
    r"\b(?:all|every|full|complete|entire|enumerate)\b)",
    re.IGNORECASE,
)
_LIST_SCOPE_RE = re.compile(
    r"(?:清单|列表|汇总|catalog(?:ue)?|inventory|\blist\b)",
    re.IGNORECASE,
)
_DOCUMENT_SCOPE_RE = re.compile(
    r"(?:文中|文内|本文|本篇|论文(?:中|内)?|文章(?:中|内)?|文档(?:中|内)?|"
    r"\b(?:this|the)\s+(?:document|paper|article)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 中文单字中心词的词边界闸门
# ---------------------------------------------------------------------------
# 「图」「表」既可以是被枚举的中心词（"所有图" / "文中所有的表"），也可以只是
# 复合词的一半（"试图" / "意图" / "图书" / "图灵" / "表现" / "发表"）。中文没有
# 空格，`\b` 在这里完全失效，而工作区里没有可复用的轻量中文分词方案（TrustRAG
# 那份纯 Python 分词器要 datrie + 71MB 词典，为一条路由规则引入不成比例）。
#
# 这里做的是 maximum munch（最长匹配优先）的微缩实现，用**两道肯定上下文**
# 而不是一张永远列不全的复合词黑名单：
#
#   LEFT  —— 中心词左边必须是限定词/量词/连词/方位词，或任何非汉字（数字、
#            标点、拉丁字母、空白、句首）。这道闸负责挡掉 X图/X表 型复合词：
#            试图、意图、企图、力图、妄图、蓝图、地图、版图、拼图、构图、
#            代表、发表、手表、外表、仪表、报表、年表、量表 —— 它们的首字
#            一个都不在限定词集合里，无需逐个列举。
#   RIGHT —— 中心词右边必须是句读/助词/连词/系动词/方位词，或任何非汉字
#            （含句尾）。这道闸负责挡掉 图X/表X 型复合词：图书、图灵、图例、
#            图标、图层、图腾、图论、图谱、图案、表现、表明、表示、表达、
#            表面、表征、表述、表决。
#
# 两道闸是 AND 关系，方向互补：LEFT 管左复合、RIGHT 管右复合。任何一侧放开
# 都会漏（实测：只留 RIGHT 时「本文所有发表的论文」会被误判成表格清单）。
#
# 以「图」「表」收尾的**真**图名（示意图 / 流程图 / 柱状图 …）走显式复合词
# 分支，并且排在裸字分支前面：re 从左往右扫描，"示意图" 在「示」的位置就先
# 命中，不会退化成「意图」——这正是最长匹配的效果，靠扫描顺序拿到，不靠回溯。
# 写成转义而不是字面汉字：这两个是 CJK 基本区的首尾码位（U+4E00 / U+9FFF），
# 字面形式在编辑器/编码往返里最容易被悄悄改坏，而它一旦坏了两道闸会同时失效。
_CJK = "%s-%s" % (chr(0x4E00), chr(0x9FFF))
_HEAD_LEFT = (
    rf"(?:^|(?<=[^{_CJK}])|(?<=[的有全部些张幅个条种和与及或中里页出含]))"
)
_HEAD_RIGHT = (
    rf"(?:$|(?=[^{_CJK}])|(?=[的和与及或等都有是在中里吗呢]))"
)
# 以「图」收尾且确实是文档插图的复合词。收录标准：整词在学术语境下只可能指
# 版面里的一张图，不存在非图义项。
_FIGURE_COMPOUND_HEADS = (
    "示意|流程|结构|架构|框架|拓扑|曲线|折线|柱状|条形|散点|直方|饼|雷达|热力|热"
)

_KIND_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("formula", re.compile(r"(?:公式|方程|等式|\b(?:formula|equation|equations|math)\b)", re.IGNORECASE)),
    ("table", re.compile(
        r"(?:表格|表\s*\d+|"
        rf"数据表{_HEAD_RIGHT}|统计表{_HEAD_RIGHT}|附表{_HEAD_RIGHT}|"
        rf"{_HEAD_LEFT}表{_HEAD_RIGHT}|"
        r"\b(?:table|tables)\b)",
        re.IGNORECASE,
    )),
    ("figure", re.compile(
        r"(?:图片|图像|图形|插图|附图|截图|图\s*\d+|"
        rf"图表{_HEAD_RIGHT}|(?:{_FIGURE_COMPOUND_HEADS})图{_HEAD_RIGHT}|"
        rf"{_HEAD_LEFT}图{_HEAD_RIGHT}|"
        r"\b(?:figure|figures|image|images|chart|charts)\b)",
        re.IGNORECASE,
    )),
    ("reference", re.compile(r"(?:参考文献|文献列表|引用列表|\b(?:reference|references|bibliography|citations?)\b)", re.IGNORECASE)),
    ("metadata", re.compile(r"(?:作者|机构|单位|通讯作者|邮箱|doi|\b(?:author|authors|affiliation|contact|metadata)\b)", re.IGNORECASE)),
)

_KIND_BLOCK_TYPES = {
    "formula": {"formula", "equation", "math", "display_formula", "inline_formula"},
    "table": {"table", "table_caption", "table_text"},
    "figure": {"figure", "image", "chart", "picture", "figure_caption"},
    "reference": {"reference", "references", "bibliography", "citation"},
    "metadata": {"author", "authors", "affiliation", "contact", "publication_header", "metadata"},
}
_KIND_CONTENT_ROLES = {
    "reference": {"reference", "references", "bibliography", "citation"},
    "metadata": {"author", "authors", "affiliation", "contact", "publication_header", "metadata"},
}


def detect_inventory_kinds(question: str) -> tuple[str, ...]:
    """Return every requested complete block kind in stable kind order.

    类目边界（判定放宽前先读这段）：

    * **是 inventory** —— 用户要的是一份完整清单，漏一条就算错答案。
      下游会绕开语义 Top-K，直接按页序枚举已发布的块索引。
      例："列出本文所有的公式" / "第 4 到 6 页里的所有图" / "list all the figures"。
    * **不是 inventory** —— 用户要的是若干条相关证据，Top-K 采样即可满足。
      例："第 3 页讲了什么" / "列出公式的参数" / "表 3 里的 F1 是多少"。

    判定是 ``kind ∧ scope`` 的合取：光有中心词（图/表/公式）不够，还必须有
    显式的完整性诉求（所有/全部/每一个/all/every…），或者"清单类说法 +
    指向当前文档"。两道闸缺一不可——kind 侧刻意只做词形判断，不承担
    "用户到底要不要全量"的语义责任。
    """
    normalized = str(question or "").strip()
    if not normalized:
        return ()
    matched_kinds = tuple(
        kind for kind, pattern in _KIND_PATTERNS if pattern.search(normalized)
    )
    if not matched_kinds:
        return ()
    if _EXPLICIT_FULL_SCOPE_RE.search(normalized):
        return matched_kinds
    # A bare request such as "列出公式的参数" is ordinary QA, not an
    # exhaustive document scan.  List-like phrasing becomes inventory mode
    # only when it is explicitly scoped to the current document.
    if _LIST_SCOPE_RE.search(normalized) and _DOCUMENT_SCOPE_RE.search(normalized):
        return matched_kinds
    return ()


def detect_inventory_kind(question: str) -> str | None:
    """Compatibility wrapper for callers that only support one kind."""
    kinds = detect_inventory_kinds(question)
    return kinds[0] if kinds else None


def _page_number(page_record: dict, fallback: int) -> int:
    for key in ("page", "page_number", "number"):
        try:
            value = int(page_record.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return fallback


def _block_text(block: dict) -> str:
    for key in ("text", "content", "markdown", "caption", "ocr_text"):
        text = str(block.get(key) or "").strip()
        if text:
            return text
    return ""


def _block_type(block: dict) -> str:
    return str(
        block.get("block_type") or block.get("type") or block.get("kind") or ""
    ).strip().casefold()


def _content_role(block: dict) -> str:
    return str(block.get("content_role") or block.get("role") or "").strip().casefold()


def _coerce_bbox(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return []


def _normalize_page_ranges(
    page_ranges: Iterable[Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    """Normalize untrusted page locators without expanding large ranges."""
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_range in page_ranges or ():
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
            continue
        try:
            start = int(raw_range[0])
            end = int(raw_range[1])
        except (TypeError, ValueError):
            continue
        if start <= 0 or end <= 0:
            continue
        value = (min(start, end), max(start, end))
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _page_matches_ranges(page: int, page_ranges: tuple[tuple[int, int], ...]) -> bool:
    return not page_ranges or any(start <= page <= end for start, end in page_ranges)


def _iter_kind_blocks(
    block_index: dict,
    kind: str,
    *,
    page_ranges: tuple[tuple[int, int], ...] = (),
) -> Iterable[dict]:
    allowed_types = _KIND_BLOCK_TYPES.get(kind, set())
    allowed_roles = _KIND_CONTENT_ROLES.get(kind, set())
    pages = block_index.get("pages") if isinstance(block_index, dict) else []
    if not isinstance(pages, list):
        return
    ordinal = 0
    for fallback_page, page_record in enumerate(pages, start=1):
        if not isinstance(page_record, dict):
            continue
        page = _page_number(page_record, fallback_page)
        if not _page_matches_ranges(page, page_ranges):
            continue
        blocks = page_record.get("blocks")
        if not isinstance(blocks, list):
            continue
        for page_order, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = _block_type(block)
            content_role = _content_role(block)
            if block_type not in allowed_types and content_role not in allowed_roles:
                continue
            text = _block_text(block)
            block_id = str(block.get("block_id") or block.get("id") or "").strip()
            # A location identity is mandatory.  Empty synthetic blocks cannot
            # be part of an exhaustive user-visible inventory.
            if not text or not block_id:
                continue
            ordinal += 1
            yield {
                "ordinal": ordinal,
                "page": page,
                "page_order": page_order,
                "block_id": block_id,
                "block_type": block_type,
                "content_role": content_role,
                "text": text,
                "bbox": _coerce_bbox(block.get("bbox")),
                "section_id": block.get("section_id"),
                "section_path": block.get("section_path"),
                "figure_id": block.get("figure_id") or block.get("asset_id"),
                "table_id": block.get("table_id"),
            }


def enumerate_block_inventory(
    block_index: dict,
    kind: str,
    *,
    cursor: int = 0,
    limit: int = 100,
    page_ranges: Iterable[Sequence[int]] | None = None,
) -> dict:
    """Enumerate published blocks in deterministic page/block order.

    ``cursor`` is a zero-based stable offset within the active block-index
    revision.  Callers receive the parse identity alongside the next cursor so
    a later request cannot accidentally continue a different parse generation.
    """
    normalized_kind = str(kind or "").strip().casefold()
    if normalized_kind not in _KIND_BLOCK_TYPES:
        raise ValueError(f"unsupported inventory kind: {kind}")
    try:
        offset = max(0, int(cursor or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        page_size = max(1, min(500, int(limit or 100)))
    except (TypeError, ValueError):
        page_size = 100

    normalized_ranges = _normalize_page_ranges(page_ranges)
    all_items = list(
        _iter_kind_blocks(
            block_index,
            normalized_kind,
            page_ranges=normalized_ranges,
        )
    )
    page_items = all_items[offset:offset + page_size]
    next_offset = offset + len(page_items)
    total = len(all_items)
    return {
        "kind": normalized_kind,
        "items": page_items,
        "total": total,
        "cursor": offset,
        "next_cursor": next_offset if next_offset < total else None,
        "has_more": next_offset < total,
        "coverage_complete": next_offset >= total,
        "page_ranges": [list(item) for item in normalized_ranges],
        "parse_generation": str(block_index.get("parse_generation") or ""),
        "document_source_hash": str(block_index.get("document_source_hash") or ""),
        "block_index_hash": str(
            block_index.get("block_index_hash") or block_index.get("block_index_revision") or ""
        ),
    }


def inventory_citations(inventory: dict, *, start_ref: int = 1) -> list[dict]:
    """Build citations that retain the original block/page/bbox identity."""
    citations: list[dict] = []
    for index, item in enumerate(inventory.get("items") or [], start=start_ref):
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("block_id") or "").strip()
        if not block_id:
            continue
        page = int(item.get("page") or 0)
        text = str(item.get("text") or "").strip()
        citations.append({
            "ref": index,
            "source_ref": index,
            "group_id": f"inventory:{inventory.get('kind', 'block')}:{block_id}",
            "context_id": block_id,
            "chunk_id": block_id,
            "block_id": block_id,
            "evidence_id": f"block:{block_id}",
            "page_range": [page, page] if page > 0 else [],
            "bbox": item.get("bbox") or [],
            "highlight_text": text[:1200],
            "_full_text": text,
            "block_type": item.get("block_type"),
            "content_role": item.get("content_role"),
            "inventory_kind": inventory.get("kind"),
            "parse_generation": inventory.get("parse_generation"),
            "document_source_hash": inventory.get("document_source_hash"),
        })
    return citations


def build_inventory_context(
    inventory: dict,
    *,
    max_chars: int = 72_000,
) -> tuple[str, dict]:
    """Format a page of deterministic inventory evidence for the answer model.

    The returned diagnostics make any context-window cut explicit.  The caller
    must never label a partial page as a complete document inventory.
    """
    lines = [
        "The following is a deterministic block inventory in document order.",
        f"kind={inventory.get('kind')} total={inventory.get('total', 0)} "
        f"cursor={inventory.get('cursor', 0)} has_more={bool(inventory.get('has_more'))}",
    ]
    used = sum(len(line) + 1 for line in lines)
    included = 0
    omitted = 0
    for index, item in enumerate(inventory.get("items") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        record = (
            f"[{index}] page={item.get('page')} block={item.get('block_id')} "
            f"type={item.get('block_type')}\n{text}\n"
        )
        if used + len(record) > max_chars:
            omitted = len(inventory.get("items") or []) - included
            break
        lines.append(record)
        used += len(record)
        included += 1
    diagnostics = {
        "returned_count": len(inventory.get("items") or []),
        "context_included_count": included,
        "context_omitted_count": omitted,
        "has_more": bool(inventory.get("has_more")),
        "coverage_complete": bool(inventory.get("coverage_complete")) and omitted == 0,
    }
    if omitted or inventory.get("has_more"):
        lines.append(
            "Coverage note: this response must state the returned range and must not claim "
            "details for items omitted from the supplied evidence page."
        )
    return "\n".join(lines), diagnostics
