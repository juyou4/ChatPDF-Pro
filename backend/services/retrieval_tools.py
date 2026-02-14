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
from typing import Any, Dict, List, Optional

from services.grep_service import grep_search
from services.bm25_service import bm25_search
from services.advanced_search import AdvancedSearchService

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
    ):
        self.doc_id = doc_id
        self.full_text = full_text
        self.chunks = chunks
        self.pages = pages
        self.semantic_groups = semantic_groups or []
        self.vector_store_dir = vector_store_dir
        self.api_key = api_key


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


def _exec_vector_search(args: dict, ctx: DocContext) -> dict:
    """向量语义搜索"""
    from services.embedding_service import search_document_chunks

    query = args.get("query", "")
    limit = args.get("limit", 10)

    if not query:
        return {"results": [], "summary": "查询为空"}

    try:
        results = search_document_chunks(
            ctx.doc_id,
            query,
            vector_store_dir=ctx.vector_store_dir,
            pages=ctx.pages,
            api_key=ctx.api_key,
            top_k=limit,
            candidate_k=max(limit * 2, 20),
            use_rerank=False,
        )
        # 提取 chunk 文本
        chunks_found = []
        for r in results[:limit]:
            chunk_text = r.get("chunk", "")
            if chunk_text:
                # 截取前 1500 字符避免上下文过长
                chunks_found.append(chunk_text[:1500])

        return {
            "results": chunks_found,
            "result_count": len(chunks_found),
            "summary": f"向量搜索 \"{query}\" 返回 {len(chunks_found)} 个结果",
        }
    except Exception as e:
        logger.warning(f"[RetrievalTools] vector_search 失败: {e}")
        return {"results": [], "result_count": 0, "summary": f"向量搜索失败: {e}"}


def _exec_keyword_search(args: dict, ctx: DocContext) -> dict:
    """BM25 关键词搜索"""
    keywords = args.get("keywords", [])
    limit = args.get("limit", 8)

    if not keywords:
        return {"results": [], "summary": "关键词为空"}

    # 将关键词列表组合为查询字符串
    query = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

    results = bm25_search(ctx.doc_id, query, ctx.chunks, top_k=limit)

    chunks_found = []
    for r in results:
        chunk_text = r.get("chunk", "")
        if chunk_text:
            chunks_found.append(chunk_text[:1500])

    return {
        "results": chunks_found,
        "result_count": len(chunks_found),
        "summary": f"BM25搜索 {keywords} 返回 {len(chunks_found)} 个结果",
    }


def _exec_grep(args: dict, ctx: DocContext) -> dict:
    """精确文本搜索"""
    query = args.get("query", "")
    limit = args.get("limit", 20)
    context = args.get("context", 2000)
    case_insensitive = args.get("caseInsensitive", True)

    if not query:
        return {"results": [], "summary": "查询为空"}

    results = grep_search(
        query=query,
        text=ctx.full_text,
        limit=limit,
        context_chars=context,
        case_insensitive=case_insensitive,
    )

    chunks_found = [r["context_snippet"][:1500] for r in results if r.get("context_snippet")]

    return {
        "results": chunks_found,
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

    chunks_found = [r["context_snippet"][:1500] for r in results if r.get("context_snippet")]

    return {
        "results": chunks_found,
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

    chunks_found = [r["context_snippet"][:1500] for r in results if r.get("context_snippet")]

    return {
        "results": chunks_found,
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

    return {
        "results": [text] if text else [],
        "result_count": 1 if text else 0,
        "group_id": group_id,
        "granularity": granularity,
        "keywords": keywords,
        "summary": f"获取意群 {group_id} ({granularity})，{len(text)} 字符",
    }


def _exec_map(args: dict, ctx: DocContext) -> dict:
    """获取文档结构概览（意群地图）"""
    limit = args.get("limit", 50)

    if not ctx.semantic_groups:
        return {"results": [], "summary": "无意群数据"}

    map_entries = []
    for g in ctx.semantic_groups[:limit]:
        if hasattr(g, "group_id"):
            # SemanticGroup 对象
            entry = {
                "group_id": g.group_id,
                "char_count": g.char_count,
                "keywords": g.keywords,
                "summary": g.summary[:200] if g.summary else "",
                "page_range": list(g.page_range),
            }
        elif isinstance(g, dict):
            entry = {
                "group_id": g.get("group_id", ""),
                "char_count": g.get("char_count", 0),
                "keywords": g.get("keywords", []),
                "summary": (g.get("summary", "") or "")[:200],
                "page_range": g.get("page_range", [0, 0]),
            }
        else:
            continue
        map_entries.append(entry)

    # 构建地图文本
    map_lines = []
    for e in map_entries:
        kw = "、".join(e["keywords"]) if e["keywords"] else "无"
        map_lines.append(
            f"【{e['group_id']}】{e['char_count']}字 | 页码:{e['page_range'][0]}-{e['page_range'][1]} | 关键词:{kw}\n  摘要:{e['summary']}"
        )

    map_text = "\n".join(map_lines)

    return {
        "results": [map_text] if map_text else [],
        "result_count": len(map_entries),
        "map_entries": map_entries,
        "summary": f"文档地图：{len(map_entries)} 个意群",
    }
