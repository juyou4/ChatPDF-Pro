"""
检索工具执行层

统一封装所有检索工具的调用，供 RetrievalAgent 使用。
支持的工具：
- vector_search: 向量语义搜索
- keyword_search: BM25 关键词搜索
- grep: 精确文本搜索
- regex_search: 正则表达式搜索
- boolean_search: 布尔逻辑搜索
- fetch_group: 获取指定意群的详细内容
- map: 获取文档结构概览（意群地图）
"""

import logging
import re
from typing import Any, Dict, List, Optional

from services.grep_service import grep_search
from services.bm25_service import bm25_search
from services.advanced_search import AdvancedSearchService
from services.formula_text import formula_term_matches, looks_formula_like
from services.query_analyzer import analyze_evidence_need, expand_academic_bilingual_terms

logger = logging.getLogger(__name__)

_advanced_search = AdvancedSearchService()


class DocContext:
    """文档上下文，封装工具执行所需的文档数据"""

    def __init__(
        self,
        doc_id: str,
        full_text: str,
        chunks: List[str],
        pages: List[dict],
        semantic_groups: Optional[List] = None,
        vector_store_dir: str = "",
        api_key: str = "",
        use_rerank: bool = False,
        reranker_model: str = "",
        rerank_provider: str = "",
        rerank_api_key: str = "",
        rerank_endpoint: str = "",
    ):
        self.doc_id = doc_id
        self.full_text = full_text
        self.chunks = chunks
        self.pages = pages
        self.semantic_groups = semantic_groups or []
        self.vector_store_dir = vector_store_dir
        self.api_key = api_key
        self.use_rerank = bool(use_rerank)
        self.reranker_model = reranker_model or ""
        self.rerank_provider = rerank_provider or ""
        self.rerank_api_key = rerank_api_key or ""
        self.rerank_endpoint = rerank_endpoint or ""


def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    doc_ctx: DocContext,
) -> Dict[str, Any]:
    """统一工具调度

    Args:
        tool_name: 工具名称
        args: 工具参数
        doc_ctx: 文档上下文

    Returns:
        工具执行结果，包含 results 列表和 summary 字符串
    """
    try:
        if tool_name == "vector_search":
            return _exec_vector_search(args, doc_ctx)
        elif tool_name == "keyword_search":
            return _exec_keyword_search(args, doc_ctx)
        elif tool_name == "grep":
            return _exec_grep(args, doc_ctx)
        elif tool_name == "regex_search":
            return _exec_regex_search(args, doc_ctx)
        elif tool_name == "boolean_search":
            return _exec_boolean_search(args, doc_ctx)
        elif tool_name == "fetch":
            return _exec_fetch_group(args, doc_ctx)
        elif tool_name == "map":
            return _exec_map(args, doc_ctx)
        else:
            return {"error": f"未知工具: {tool_name}", "results": []}
    except Exception as e:
        logger.error(f"[RetrievalTools] 工具 {tool_name} 执行失败: {e}")
        return {"error": str(e), "results": []}


def _group_value(group: Any, key: str, default: Any = None) -> Any:
    if isinstance(group, dict):
        return group.get(key, default)
    return getattr(group, key, default)


def _as_page_range(value: Any) -> list:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [value[0], value[1]]
    return [0, 0]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _is_distinctive_query_anchor(term: str) -> bool:
    """识别问题中的技术锚点，给专有术语/编号/公式符号更稳定的排序权重。"""
    raw = str(term or "").strip()
    normalized = raw.strip(".,;:()[]{}，。；：、")
    if len(normalized) < 3:
        return False
    if re.search(r"[\d_/%\-]", normalized):
        return True
    if re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", normalized):
        return True
    if normalized.isupper() and len(normalized) >= 3:
        return True
    return len(normalized) >= 8 and not re.fullmatch(r"[a-z]+", normalized.lower())


def _find_page_for_text(text: str, pages: List[dict]) -> int:
    snippet = re.sub(r"\s+", " ", str(text or "")[:120]).strip().lower()
    if not snippet:
        return 0
    for idx, page in enumerate(pages or []):
        page_text = re.sub(r"\s+", " ", str(page.get("text", "") or page.get("content", ""))).lower()
        if snippet[:60] and snippet[:60] in page_text:
            return idx + 1
        if snippet[:36] and snippet[:36] in page_text:
            return idx + 1
    return 0


def _find_page_for_offset(offset: Any, full_text: str, pages: List[dict]) -> int:
    try:
        target = int(offset)
    except (TypeError, ValueError):
        return 0
    if target < 0:
        return 0
    cursor = 0
    source_text = str(full_text or "")
    for idx, page in enumerate(pages or []):
        page_text = str(page.get("text", "") or page.get("content", "") or "")
        if not page_text:
            continue
        found_at = source_text.find(page_text, cursor)
        if found_at < 0:
            found_at = source_text.find(page_text)
        if found_at < 0:
            continue
        end_at = found_at + len(page_text)
        if found_at <= target <= end_at:
            return idx + 1
        cursor = max(cursor, end_at)
    return 0


def _normalize_page_number(value: Any, text: str, pages: List[dict]) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        page = 0
    if 1 <= page <= len(pages or []):
        return page
    return _find_page_for_text(text, pages)


def _tool_result_score(query: str, text: str, base_score: float = 0.0) -> float:
    query_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", query or "")
    bridge_terms = expand_academic_bilingual_terms(query)
    haystack = str(text or "").lower()
    score = float(base_score or 0.0)
    lexical_boost = 0.0
    anchor_boost = 0.0
    for term in _dedupe_preserve_order(query_terms):
        normalized = term.lower()
        if len(normalized) < 2:
            continue
        if normalized in haystack:
            lexical_boost += 0.35 if " " in normalized else 0.18
            if _is_distinctive_query_anchor(term):
                anchor_boost += 0.08
    for term in _dedupe_preserve_order(bridge_terms):
        normalized = term.lower()
        if len(normalized) < 2:
            continue
        if normalized in haystack:
            lexical_boost += 0.18 if " " in normalized else 0.08
    if looks_formula_like(query) or looks_formula_like(text):
        formula_hits = 0
        for term in _dedupe_preserve_order(query_terms):
            if len(term) >= 2 and formula_term_matches(term, text):
                formula_hits += 1
        if formula_hits:
            lexical_boost += min(0.35, formula_hits * 0.12)
    score += min(lexical_boost, 0.9)
    score += min(anchor_boost, 0.24)
    if re.search(r"\d", text or ""):
        score += 0.05
    return score


def compute_document_aware_evidence_score(
    query: str,
    chunk_text: str,
    doc_key_phrases: list[str] | None = None,
    base_score: float = 0.0,
) -> float:
    """计算文档感知的证据评分，融合查询词法匹配和文档关键短语命中。

    与 _tool_result_score 的区别：
    - 额外考虑文档级关键短语（从文档全文中提取的高频术语）
    - 对文档关键短语命中给予额外加分（表示该 chunk 包含文档核心内容）

    Args:
        query: 用户查询
        chunk_text: 候选证据文本
        doc_key_phrases: 文档级关键短语列表（从 extract_document_bilingual_terms 获取）
        base_score: 基础分数（如向量相似度）

    Returns:
        [0, 1] 的综合评分
    """
    # 基础词法评分
    score = _tool_result_score(query, chunk_text, base_score)

    # 文档关键短语加分
    if doc_key_phrases:
        chunk_lower = str(chunk_text or "").lower()
        phrase_hits = 0
        for phrase in doc_key_phrases:
            if phrase and phrase.lower() in chunk_lower:
                phrase_hits += 1
        # 每命中一个关键短语加 0.05，最多加 0.3
        phrase_bonus = min(0.3, phrase_hits * 0.05)
        score = min(1.0, score + phrase_bonus)

    return score


def _result_key(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("chunk_id", "parent_id", "doc_id"):
        value = item.get(key)
        if value:
            return f"{key}:{value}"
    text = item.get("chunk") or item.get("child_chunk") or item.get("raw_chunk_text") or ""
    return str(text)[:120]


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _result_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("chunk") or item.get("child_chunk") or item.get("raw_chunk_text") or "")


def _group_page_range(group: Any) -> list:
    return _as_page_range(_group_value(group, "page_range", [0, 0]))


def _find_group_for_page(page: int, semantic_groups: list) -> str:
    if not page:
        return ""
    for group in semantic_groups or []:
        page_range = _group_page_range(group)
        try:
            start = int(page_range[0])
            end = int(page_range[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start and end and start <= page <= end:
            return str(_group_value(group, "group_id", "") or "")
    return ""


def _find_group_for_text(text: str, semantic_groups: list) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return ""
    probes = [normalized[:160], normalized[:96], normalized[:48]]
    for group in semantic_groups or []:
        group_text = " ".join(
            str(_group_value(group, key, "") or "")
            for key in ("full_text", "digest", "summary")
        )
        group_norm = re.sub(r"\s+", " ", group_text).strip().lower()
        if not group_norm:
            continue
        if any(probe and probe in group_norm for probe in probes):
            return str(_group_value(group, "group_id", "") or "")
    return ""


def _search_result_to_tool_item(
    result: dict,
    *,
    ctx: DocContext,
    source: str,
    query: str,
) -> dict:
    snippet = str(result.get("context_snippet") or result.get("chunk") or "")
    page = _find_page_for_offset(result.get("match_offset"), ctx.full_text, ctx.pages) or _find_page_for_text(snippet, ctx.pages)
    group_id = _find_group_for_page(page, ctx.semantic_groups) or _find_group_for_text(snippet, ctx.semantic_groups)
    offset = result.get("match_offset")
    try:
        offset_text = str(int(offset))
    except (TypeError, ValueError):
        offset_text = ""
    evidence_id = f"text-offset:{offset_text}" if offset_text else ""
    item = {
        "chunk": snippet,
        "raw_chunk_text": snippet,
        "source": source,
        "retrieval_type": f"agent_{source}",
        "page": page,
        "group_id": group_id,
        "context_id": group_id or (f"page:{page}" if page else ""),
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,
        "score": result.get("score", 1.0),
        "match_text": result.get("match_text") or result.get("keyword") or "",
        "match_offset": result.get("match_offset"),
    }
    if query:
        item["query"] = query
    return item


def _extract_table_id_from_text(text: str) -> str:
    match = re.search(r"\bTable\s+\d+[A-Za-z]?\b|表\s*\d+[A-Za-z]?", text or "", re.IGNORECASE)
    return match.group(0).strip() if match else ""


def _looks_like_table_query(query: str) -> bool:
    query_text = str(query or "")
    query_lower = query_text.lower()
    try:
        if "numeric_table" in analyze_evidence_need(query_text):
            return True
    except Exception:
        pass
    return any(
        token in query_lower
        for token in (
            "table", "dataset", "metric", "accuracy", "acc", "score",
            "many", "med.", "medium", "few", "表", "表格", "数据集", "指标",
            "数值", "数字", "分别", "多少",
        )
    )


def _has_table_evidence(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    chunk_type = str(item.get("chunk_type") or item.get("block_type") or "").strip().lower()
    if chunk_type in {"table", "table_row", "table_cell", "caption"}:
        return True
    if any(
        item.get(key)
        for key in (
            "structured_table_bundle",
            "table_bundle_id",
            "table_id",
            "table_row_evidence",
            "numeric_table_exact_context_row_text",
            "evidence_units",
            "cell_evidence_units",
        )
    ):
        return True
    return "[structured table bundle]" in _result_text(item).lower()


def _ensure_table_result_selected(query: str, selected: list[dict], candidates: list[dict], limit: int) -> list[dict]:
    if not _looks_like_table_query(query) or not candidates:
        return selected[:limit]
    if any(_has_table_evidence(item) for item in selected):
        return selected[:limit]

    scored_tables: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(candidates):
        if not _has_table_evidence(item):
            continue
        text = _result_text(item)
        if not text:
            continue
        score = _tool_result_score(query, text, item.get("similarity", item.get("score", 0.0)))
        if item.get("structured_table_bundle") or "[structured table bundle]" in text.lower():
            score += 0.45
        if item.get("evidence_units") or item.get("cell_evidence_units"):
            score += 0.2
        caption = f"{item.get('table_id') or ''} {item.get('table_caption') or ''}".lower()
        query_lower = str(query or "").lower()
        if caption and any(part and part in query_lower for part in re.split(r"\s+", caption)[:6]):
            score += 0.25
        scored_tables.append((float(score), idx, item))

    if not scored_tables:
        return selected[:limit]

    scored_tables.sort(key=lambda row: (-row[0], row[1]))
    best = scored_tables[0][2]
    best_key = _result_key(best)
    if best_key and any(_result_key(item) == best_key for item in selected):
        return selected[:limit]

    trimmed = selected[: max(0, limit)]
    if limit <= 0:
        return []
    if len(trimmed) < limit:
        return [*trimmed, best]
    if not trimmed:
        return [best]
    return [*trimmed[:-1], best]


def _interleave_ranked_results(primary: list[dict], secondary: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        for source in (primary, secondary):
            if idx >= len(source):
                continue
            item = source[idx]
            key = _result_key(item)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                return selected
    return selected


def _format_tool_chunk(
    text: str,
    *,
    page: int = 0,
    group_id: str = "",
    chunk_idx: Any = None,
    source: str = "",
    context_id: Any = None,
    evidence_id: Any = None,
    child_chunk_id: Any = None,
    parent_id: Any = None,
    chunk_type: Any = None,
    table_id: Any = None,
    table_bundle_id: Any = None,
    evidence_unit_id: Any = None,
) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    tags = []
    if source:
        tags.append(f"source:{source}")
    if page:
        tags.append(f"页码:{page}")
    if group_id:
        tags.append(f"group_id:{group_id}")
    if context_id:
        tags.append(f"context_id:{context_id}")
    if evidence_id:
        tags.append(f"evidence_id:{evidence_id}")
    if chunk_idx is not None:
        tags.append(f"chunk_id:{chunk_idx}")
    if child_chunk_id:
        tags.append(f"child_chunk_id:{child_chunk_id}")
    if parent_id:
        tags.append(f"parent_id:{parent_id}")
    if chunk_type:
        tags.append(f"chunk_type:{chunk_type}")
    if table_id:
        tags.append(f"table_id:{table_id}")
    if table_bundle_id:
        tags.append(f"table_bundle_id:{table_bundle_id}")
    if evidence_unit_id:
        tags.append(f"evidence_unit_id:{evidence_unit_id}")
    return f"【检索证据 | {' | '.join(tags)}】\n{body[:1500]}" if tags else body[:1500]


def _build_tool_candidate_meta(
    item: dict,
    *,
    ctx: DocContext,
    page: int = 0,
    group_id: str = "",
    chunk_idx: Any = None,
) -> dict:
    meta = {
        "context_id": item.get("context_id"),
        "evidence_id": item.get("evidence_id"),
        "chunk_id": item.get("chunk_id"),
        "child_chunk_id": item.get("child_chunk_id"),
        "chunk_idx": chunk_idx,
        "group_id": group_id,
        "page": page,
        "parent_id": item.get("parent_id"),
        "doc_id": item.get("doc_id") or ctx.doc_id,
        "score": item.get("score", 0.0),
        "similarity": item.get("similarity"),
    }
    for key in (
        "chunk_type",
        "block_type",
        "page_range",
        "table_pages",
        "structured_table_bundle",
        "table_bundle_id",
        "evidence_unit_id",
        "table_id",
        "table_caption",
        "table_header",
        "numeric_table_exact_context_row_text",
        "numeric_table_exact_context_caption",
        "numeric_table_exact_context_header",
        "table_footnote",
        "table_bbox",
        "table_bboxes",
        "table_source_ids",
        "evidence_units",
        "cell_evidence_units",
        "table_row_evidence",
        "table_row_slice_kind",
        "table_row_raw_text",
        "table_row_bbox",
        "cell_evidence_ids",
        "source",
    ):
        value = item.get(key)
        if _has_value(value):
            meta[key] = value

    text = _result_text(item)
    if "[structured table bundle]" in text.lower():
        meta.setdefault("structured_table_bundle", True)
        meta.setdefault("chunk_type", item.get("chunk_type") or "table")
        table_id = item.get("table_id") or _extract_table_id_from_text(text)
        if table_id:
            meta.setdefault("table_id", table_id)
    return meta


def _format_structure_lines(structure: Any, chunk_indices: Any = None) -> list[str]:
    if not isinstance(structure, dict):
        structure = {}
    lines: list[str] = []
    ordered = structure.get("orderedElements") or structure.get("ordered_elements") or []
    if isinstance(ordered, list):
        for elem in ordered[:8]:
            if not isinstance(elem, dict):
                continue
            content = elem.get("content") or elem.get("text") or elem.get("title") or ""
            elem_type = elem.get("type") or "item"
            if content:
                lines.append(f"{elem_type}: {content}")
    for label, keys in [
        ("章节", ("sections", "section")),
        ("要点", ("keyPoints", "key_points")),
        ("图表", ("figures", "tables")),
        ("公式", ("formulas", "equations")),
    ]:
        values = []
        for key in keys:
            raw = structure.get(key)
            if isinstance(raw, list):
                values.extend(str(x) for x in raw if x)
            elif raw:
                values.append(str(raw))
        if values:
            lines.append(f"{label}: {'; '.join(values[:6])}")
    if chunk_indices:
        values = list(chunk_indices)[:8] if isinstance(chunk_indices, (list, tuple)) else [chunk_indices]
        lines.append(f"chunks: {', '.join(str(x) for x in values)}")
    return lines[:10]


def _exec_vector_search(args: dict, ctx: DocContext) -> dict:
    """向量语义搜索"""
    from services.embedding_service import search_document_chunks

    query = args.get("query", "")
    # 适度放宽 agent 工具召回上限，给后续 rerank/上下文预算选择保留更多候选。
    limit = max(1, min(int(args.get("limit", 16) or 16), 24))
    retrieval_limit = max(limit * 2, 32)

    if not query:
        return {"results": [], "chunk_meta": [], "summary": "查询为空"}

    try:
        use_rerank = bool(ctx.use_rerank)
        rerank_provider = (ctx.rerank_provider or "").strip().lower().replace("siliconflow", "silicon")
        reranker_model = (ctx.reranker_model or "").strip()
        rerank_api_key = (ctx.rerank_api_key or "").strip()
        rerank_endpoint = (ctx.rerank_endpoint or "").strip()
        search_output = search_document_chunks(
            ctx.doc_id,
            query,
            vector_store_dir=ctx.vector_store_dir,
            pages=ctx.pages,
            api_key=ctx.api_key,
            top_k=retrieval_limit,
            candidate_k=max(retrieval_limit * 4, 80),
            use_rerank=use_rerank,
            reranker_model=reranker_model or None,
            rerank_provider=rerank_provider or None,
            rerank_api_key=rerank_api_key or None,
            rerank_endpoint=rerank_endpoint or None,
            enable_query_expansion_override=False,
        )
        results = search_output[0] if isinstance(search_output, tuple) else search_output
        if not isinstance(results, list):
            results = []
        # 提取 chunk 文本和元数据
        chunks_found = []
        chunk_meta = []
        candidate_meta = []
        ranked_results = sorted(
            results,
            key=lambda item: _tool_result_score(query, item.get("chunk") or item.get("child_chunk") or item.get("raw_chunk_text") or "", item.get("similarity", item.get("score", 0.0))),
            reverse=True,
        )
        for r in results:
            if not isinstance(r, dict):
                continue
            chunk_text = r.get("chunk") or r.get("child_chunk") or r.get("raw_chunk_text") or ""
            if not chunk_text:
                continue
            page = _normalize_page_number(r.get("page"), chunk_text, ctx.pages)
            group_id = r.get("group_id") or ""
            candidate_meta.append(_build_tool_candidate_meta(
                r,
                ctx=ctx,
                page=page or 0,
                group_id=group_id,
                chunk_idx=r.get("chunk_id"),
            ))
        selected_results = _interleave_ranked_results(results, ranked_results, limit)
        selected_results = _ensure_table_result_selected(query, selected_results, ranked_results, limit)
        for r in selected_results:
            if not isinstance(r, dict):
                continue
            chunk_text = r.get("chunk") or r.get("child_chunk") or r.get("raw_chunk_text") or ""
            if chunk_text:
                page = _normalize_page_number(r.get("page"), chunk_text, ctx.pages)
                group_id = r.get("group_id") or ""
                chunk_idx = r.get("chunk_id")
                chunks_found.append(_format_tool_chunk(
                    chunk_text,
                    page=page or 0,
                    group_id=group_id,
                    chunk_idx=chunk_idx,
                    source="vector",
                    context_id=r.get("context_id"),
                    evidence_id=r.get("evidence_id"),
                    child_chunk_id=r.get("child_chunk_id"),
                    parent_id=r.get("parent_id"),
                    chunk_type=r.get("chunk_type") or r.get("block_type"),
                    table_id=r.get("table_id"),
                    table_bundle_id=r.get("table_bundle_id"),
                    evidence_unit_id=r.get("evidence_unit_id"),
                ))
                chunk_meta.append(_build_tool_candidate_meta(
                    r,
                    ctx=ctx,
                    page=page or 0,
                    group_id=group_id,
                    chunk_idx=chunk_idx,
                ))

        return {
            "results": chunks_found,
            "chunk_meta": chunk_meta,
            "candidate_meta": candidate_meta,
            "result_count": len(chunks_found),
            "summary": f"向量搜索 \"{query}\" 返回 {len(chunks_found)} 个结果",
            "candidate_k": max(retrieval_limit * 4, 80),
        }
    except Exception as e:
        logger.warning(f"[RetrievalTools] vector_search 失败: {e}")
        return {"results": [], "chunk_meta": [], "result_count": 0, "summary": f"向量搜索失败: {e}"}


def _exec_keyword_search(args: dict, ctx: DocContext) -> dict:
    """BM25 关键词搜索"""
    from services.embedding_service import _build_chunk_idx_to_group_map, _load_group_data

    keywords = args.get("keywords", [])
    # P3: keyword_search default 8→12、cap 20→24，对齐 vector_search 的 limit_gap 修复
    limit = max(1, min(int(args.get("limit", 12) or 12), 24))

    if not keywords:
        return {"results": [], "chunk_meta": [], "summary": "关键词为空"}

    # 将关键词列表组合为查询字符串
    raw_terms = keywords if isinstance(keywords, list) else [str(keywords)]
    expanded_terms = []
    for term in raw_terms:
        expanded_terms.extend(expand_academic_bilingual_terms(str(term)))
    query_terms = _dedupe_preserve_order([str(item) for item in raw_terms] + expanded_terms)
    query = " ".join(query_terms)

    results = bm25_search(ctx.doc_id, query, ctx.chunks, top_k=max(limit * 2, 24))

    # 构建 chunk_idx -> group_id 映射
    group_chunk_map = _load_group_data(ctx.doc_id)
    chunk_idx_to_group = _build_chunk_idx_to_group_map(group_chunk_map)

    chunks_found = []
    chunk_meta = []
    candidate_meta = []
    ranked_results = sorted(
        results,
        key=lambda item: _tool_result_score(query, item.get("chunk", ""), item.get("score", 0.0)),
        reverse=True,
    )
    for r in ranked_results:
        if not isinstance(r, dict):
            continue
        chunk_text = r.get("chunk", "")
        if not chunk_text:
            continue
        chunk_idx = r.get("index")
        page = _find_page_for_text(chunk_text, ctx.pages)
        group_id = chunk_idx_to_group.get(chunk_idx, "") if isinstance(chunk_idx, int) else ""
        candidate_meta.append(_build_tool_candidate_meta(
            r,
            ctx=ctx,
            page=page,
            group_id=group_id,
            chunk_idx=chunk_idx,
        ))
    for r in ranked_results[:limit]:
        chunk_text = r.get("chunk", "")
        if chunk_text:
            chunk_idx = r.get("index")
            page = _find_page_for_text(chunk_text, ctx.pages)
            group_id = chunk_idx_to_group.get(chunk_idx, "") if isinstance(chunk_idx, int) else ""
            chunks_found.append(_format_tool_chunk(
                chunk_text,
                page=page,
                group_id=group_id,
                chunk_idx=chunk_idx,
                source="bm25",
                context_id=r.get("context_id"),
                evidence_id=r.get("evidence_id"),
                child_chunk_id=r.get("child_chunk_id"),
                parent_id=r.get("parent_id"),
                chunk_type=r.get("chunk_type") or r.get("block_type"),
                table_id=r.get("table_id"),
                table_bundle_id=r.get("table_bundle_id"),
                evidence_unit_id=r.get("evidence_unit_id"),
            ))
            chunk_meta.append(_build_tool_candidate_meta(
                r,
                ctx=ctx,
                page=page,
                group_id=group_id,
                chunk_idx=chunk_idx,
            ))

    return {
        "results": chunks_found,
        "chunk_meta": chunk_meta,
        "candidate_meta": candidate_meta,
        "result_count": len(chunks_found),
        "summary": f"BM25搜索 {keywords} 返回 {len(chunks_found)} 个结果",
    }


def _exec_grep(args: dict, ctx: DocContext) -> dict:
    """精确文本搜索"""
    query = args.get("query", "")
    limit = max(1, min(int(args.get("limit", 20) or 20), 30))
    context = args.get("context", 2000)
    case_insensitive = args.get("caseInsensitive", True)

    if not query:
        return {"results": [], "summary": "查询为空"}

    terms = _dedupe_preserve_order([*(str(query or "").split("|")), *expand_academic_bilingual_terms(str(query or ""))])
    expanded_query = "|".join(terms[:24])

    results = grep_search(
        query=expanded_query,
        text=ctx.full_text,
        limit=limit,
        context_chars=context,
        case_insensitive=case_insensitive,
    )

    chunks_found = []
    chunk_meta = []
    candidate_meta = []
    for r in results:
        item = _search_result_to_tool_item(r, ctx=ctx, source="grep", query=query)
        snippet = item.get("chunk")
        if not snippet:
            continue
        chunks_found.append(_format_tool_chunk(
            snippet,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
            source="grep",
            context_id=item.get("context_id"),
            evidence_id=item.get("evidence_id"),
        ))
        meta = _build_tool_candidate_meta(
            item,
            ctx=ctx,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
        )
        chunk_meta.append(meta)
        candidate_meta.append(meta)

    return {
        "results": chunks_found,
        "chunk_meta": chunk_meta,
        "candidate_meta": candidate_meta,
        "result_count": len(chunks_found),
        "summary": f"GREP搜索 \"{query}\" 返回 {len(chunks_found)} 个结果",
    }


def _exec_regex_search(args: dict, ctx: DocContext) -> dict:
    """正则表达式搜索"""
    pattern = args.get("pattern", "")
    limit = args.get("limit", 10)
    context = args.get("context", 1500)

    if not pattern:
        return {"results": [], "summary": "正则模式为空"}

    try:
        results = _advanced_search.regex_search(
            pattern=pattern,
            text=ctx.full_text,
            limit=limit,
            context_chars=context,
        )
    except ValueError as e:
        return {"results": [], "summary": f"正则语法错误: {e}"}

    chunks_found = []
    chunk_meta = []
    candidate_meta = []
    for r in results:
        item = _search_result_to_tool_item(r, ctx=ctx, source="regex", query=pattern)
        snippet = item.get("chunk")
        if not snippet:
            continue
        chunks_found.append(_format_tool_chunk(
            snippet,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
            source="regex",
            context_id=item.get("context_id"),
            evidence_id=item.get("evidence_id"),
        ))
        meta = _build_tool_candidate_meta(
            item,
            ctx=ctx,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
        )
        chunk_meta.append(meta)
        candidate_meta.append(meta)

    return {
        "results": chunks_found,
        "chunk_meta": chunk_meta,
        "candidate_meta": candidate_meta,
        "result_count": len(chunks_found),
        "summary": f"正则搜索 \"{pattern}\" 返回 {len(chunks_found)} 个结果",
    }


def _exec_boolean_search(args: dict, ctx: DocContext) -> dict:
    """布尔逻辑搜索"""
    query = args.get("query", "")
    limit = args.get("limit", 10)
    context = args.get("context", 1500)

    if not query:
        return {"results": [], "summary": "查询为空"}

    results = _advanced_search.boolean_search(
        query=query,
        text=ctx.full_text,
        limit=limit,
        context_chars=context,
    )

    chunks_found = []
    chunk_meta = []
    candidate_meta = []
    for r in results:
        item = _search_result_to_tool_item(r, ctx=ctx, source="boolean", query=query)
        snippet = item.get("chunk")
        if not snippet:
            continue
        chunks_found.append(_format_tool_chunk(
            snippet,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
            source="boolean",
            context_id=item.get("context_id"),
            evidence_id=item.get("evidence_id"),
        ))
        meta = _build_tool_candidate_meta(
            item,
            ctx=ctx,
            page=item.get("page") or 0,
            group_id=item.get("group_id") or "",
            chunk_idx=item.get("chunk_id"),
        )
        chunk_meta.append(meta)
        candidate_meta.append(meta)

    return {
        "results": chunks_found,
        "chunk_meta": chunk_meta,
        "candidate_meta": candidate_meta,
        "result_count": len(chunks_found),
        "summary": f"布尔搜索 \"{query}\" 返回 {len(chunks_found)} 个结果",
    }


def _exec_fetch_group(args: dict, ctx: DocContext) -> dict:
    """获取指定意群的详细内容"""
    group_id = args.get("groupId", "")
    granularity = args.get("granularity", "full")

    if not group_id:
        return {"results": [], "summary": "意群 ID 为空"}

    # 在 semantic_groups 中查找
    group = None
    for g in ctx.semantic_groups:
        gid = g.group_id if hasattr(g, "group_id") else g.get("group_id", "")
        if gid == group_id:
            group = g
            break

    if group is None:
        return {"results": [], "summary": f"未找到意群 {group_id}"}

    # 按粒度获取文本
    if granularity == "full":
        text = getattr(group, "full_text", "") or group.get("full_text", "") if isinstance(group, dict) else getattr(group, "full_text", "")
    elif granularity == "digest":
        text = getattr(group, "digest", "") or group.get("digest", "") if isinstance(group, dict) else getattr(group, "digest", "")
    else:
        text = getattr(group, "summary", "") or group.get("summary", "") if isinstance(group, dict) else getattr(group, "summary", "")

    if not text:
        # 降级：尝试获取更高粒度
        for attr in ["full_text", "digest", "summary"]:
            text = getattr(group, attr, "") if hasattr(group, attr) else group.get(attr, "") if isinstance(group, dict) else ""
            if text:
                break

    # 截取合理长度
    text = text[:8000] if text else ""

    keywords = getattr(group, "keywords", []) if hasattr(group, "keywords") else group.get("keywords", []) if isinstance(group, dict) else []
    page_range = _as_page_range(
        getattr(group, "page_range", [0, 0])
        if hasattr(group, "page_range")
        else group.get("page_range", [0, 0])
        if isinstance(group, dict)
        else [0, 0]
    )

    context_id = str(group_id)
    evidence_id = f"{context_id}:{granularity}"
    chunk = _format_tool_chunk(
        text,
        page=page_range[0] if page_range and page_range[0] == page_range[-1] else 0,
        group_id=group_id,
        chunk_idx=evidence_id,
        source="fetch",
        context_id=context_id,
        evidence_id=evidence_id,
    ) if text else ""
    meta_item = {
        "chunk": text,
        "raw_chunk_text": text,
        "source": "fetch",
        "retrieval_type": "agent_fetch_group",
        "group_id": group_id,
        "context_id": context_id,
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,
        "page_range": page_range,
        "score": 1.0,
    }
    meta = _build_tool_candidate_meta(
        meta_item,
        ctx=ctx,
        page=page_range[0] if page_range and page_range[0] == page_range[-1] else 0,
        group_id=group_id,
        chunk_idx=evidence_id,
    ) if text else None

    return {
        "results": [chunk] if chunk else [],
        "result_count": 1 if text else 0,
        "group_id": group_id,
        "context_id": context_id,
        "evidence_id": evidence_id,
        "granularity": granularity,
        "keywords": keywords,
        "page_range": page_range,
        "chunk_meta": [meta] if meta else [],
        "candidate_meta": [meta] if meta else [],
        "summary": f"获取意群 {group_id} ({granularity})，{len(text)} 字符",
    }


def _exec_map(args: dict, ctx: DocContext) -> dict:
    """获取文档结构概览（意群地图）"""
    limit = args.get("limit", 50)
    include_structure = args.get("includeStructure", args.get("include_structure", True))

    if not ctx.semantic_groups:
        return {"results": [], "summary": "无意群数据"}

    map_entries = []
    for g in ctx.semantic_groups[:limit]:
        group_id = _group_value(g, "group_id", "")
        if not group_id:
            continue
        structure = _group_value(g, "structure", {}) or {}
        chunk_indices = _group_value(g, "chunk_indices", []) or []
        entry = {
            "group_id": group_id,
            "char_count": _group_value(g, "char_count", 0) or 0,
            "keywords": _group_value(g, "keywords", []) or [],
            "summary": (_group_value(g, "summary", "") or "")[:200],
            "page_range": _as_page_range(_group_value(g, "page_range", [0, 0])),
        }
        if include_structure:
            structure_lines = _format_structure_lines(structure, chunk_indices)
            if structure_lines:
                entry["structure"] = structure_lines
        map_entries.append(entry)

    # 构建地图文本
    map_lines = []
    for e in map_entries:
        kw = "、".join(e["keywords"]) if e["keywords"] else "无"
        lines = [
            f"【{e['group_id']}】{e['char_count']}字 | 页码:{e['page_range'][0]}-{e['page_range'][1]} | 关键词:{kw}",
        ]
        if e["summary"]:
            lines.append(f"  摘要:{e['summary']}")
        for structure_line in e.get("structure", []):
            lines.append(f"  {structure_line}")
        map_lines.append("\n".join(lines))

    map_text = "\n".join(map_lines)

    return {
        "results": [map_text] if map_text else [],
        "result_count": len(map_entries),
        "map_entries": map_entries,
        "summary": f"文档地图：{len(map_entries)} 个意群",
    }
