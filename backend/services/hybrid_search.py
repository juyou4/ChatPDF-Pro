"""
混合检索服务 - 融合向量检索和BM25检索结果

设计哲学（参考paper-burner-x）：
- 向量检索：语义理解，处理同义词和语义相似
- BM25检索：关键词匹配，精确定位特定术语
- RRF融合：互补优势，提高检索鲁棒性

使用RRF (Reciprocal Rank Fusion) 算法融合两路结果：
- 不依赖分数归一化（向量距离和BM25分数量纲不同）
- 只依赖排名，简单稳定

P3.3c 扩展（借鉴 RAGPaper.query_rewriter）：
- intersection: 仅保留所有 query 都命中的 chunks（高 precision 场景）
- weighted_avg: 多 query 平均 score（不依赖 rank）
- union: 去重并集（保留所有结果）
"""
from typing import Dict, List, Optional, Tuple


_QUERY_TYPE_PARAMS = {
    # (alpha=向量权重, min_score=最小保留分数阈值, candidate_multiplier=候选池扩展倍率)
    "extraction":  (0.35, 0.05, 1.5),  # 提取题：BM25 关键词首要
    "overview":    (0.65, 0.04, 2.5),  # 概览题：语义完备性首要，需要较大候选池
    "analytical":  (0.55, 0.05, 2.0),  # 分析题：小居向量
    "specific":    (0.50, 0.06, 1.5),  # 具体题：均衡
}
_DEFAULT_QUERY_TYPE_PARAMS = (0.50, 0.05, 1.5)


def _merge_retrieval_sources(target: dict, source_item: dict, fallback: str = "") -> None:
    sources = []
    existing = target.get("retrieval_sources")
    if isinstance(existing, list):
        sources.extend(str(v).strip() for v in existing if str(v).strip())
    elif isinstance(existing, str) and existing.strip():
        sources.extend(v.strip() for v in existing.replace(",", "+").split("+") if v.strip())
    incoming = source_item.get("retrieval_sources")
    if isinstance(incoming, list):
        candidates = incoming
    elif isinstance(incoming, str):
        candidates = incoming.replace(",", "+").split("+")
    else:
        candidates = [fallback] if fallback else []
    for value in candidates:
        normalized = str(value or "").strip()
        if normalized and normalized not in sources:
            sources.append(normalized)
    if sources:
        target["retrieval_sources"] = sources
        target["retrieval_source"] = "+".join(sources)


def get_query_type_params(query_type: Optional[str]) -> Tuple[float, float, float]:
    """根据查询类型返回 (alpha, min_score, candidate_multiplier)"""
    if not query_type:
        return _DEFAULT_QUERY_TYPE_PARAMS
    return _QUERY_TYPE_PARAMS.get(query_type, _DEFAULT_QUERY_TYPE_PARAMS)


def reciprocal_rank_fusion(
    *result_lists: List[dict],
    k: int = 60,
    top_k: int = 10,
    chunk_key: str = 'chunk'
) -> List[dict]:
    """
    RRF融合多路检索结果
    
    Args:
        *result_lists: 多个检索结果列表，每项需包含chunk_key字段
        k: RRF参数，控制排名衰减速度（默认60）
        top_k: 返回结果数量
        chunk_key: 用于标识chunk的字段名
        
    Returns:
        融合后的结果列表，按RRF分数降序排列
    """
    # chunk文本 -> (rrf_score, 原始结果dict)
    scores: Dict[str, float] = {}
    chunk_data: Dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            chunk_text = item.get(chunk_key, '')
            if not chunk_text:
                continue

            rrf_score = 1.0 / (k + rank + 1)
            scores[chunk_text] = scores.get(chunk_text, 0.0) + rrf_score

            # 保留第一次出现的完整数据（通常是向量检索的，包含更多元数据）
            if chunk_text not in chunk_data:
                chunk_data[chunk_text] = item.copy()
            _merge_retrieval_sources(chunk_data[chunk_text], item)

    # 按RRF分数排序
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_text, rrf_score in sorted_chunks[:top_k]:
        item = chunk_data[chunk_text]
        item['rrf_score'] = rrf_score
        item['hybrid'] = True
        results.append(item)

    return results


def hybrid_search_merge(
    vector_results: List[dict],
    bm25_results: List[dict],
    top_k: int = 10,
    alpha: float = 0.5,
    query_type: Optional[str] = None,
) -> List[dict]:
    """
    混合检索：融合向量检索和BM25检索结果
    
    Args:
        vector_results: 向量检索结果
        bm25_results: BM25检索结果
        top_k: 返回结果数量
        alpha: 向量检索权重（0-1），默认0.5表示平等权重
        query_type: 查询类型（可选），用于动态调整 alpha/min_score
        
    Returns:
        融合后的结果列表
    """
    # 根据查询类型动态调整 alpha 和 min_score
    alpha_adj, min_score, _ = get_query_type_params(query_type)
    # 只有在未显式指定 alpha 时才用动态分配
    if alpha == 0.5:
        alpha = alpha_adj
    if not bm25_results:
        return vector_results[:top_k]
    if not vector_results:
        return bm25_results[:top_k]

    # 使用加权RRF
    k = 60
    scores: Dict[str, float] = {}
    chunk_data: Dict[str, dict] = {}

    # 向量检索结果（权重alpha）
    for rank, item in enumerate(vector_results):
        chunk_text = item.get('chunk', '')
        if not chunk_text:
            continue
        rrf_score = alpha / (k + rank + 1)
        scores[chunk_text] = scores.get(chunk_text, 0.0) + rrf_score
        if chunk_text not in chunk_data:
            chunk_data[chunk_text] = item.copy()
        _merge_retrieval_sources(chunk_data[chunk_text], item, "vector")

    # BM25检索结果（权重1-alpha）
    for rank, item in enumerate(bm25_results):
        chunk_text = item.get('chunk', '')
        if not chunk_text:
            continue
        rrf_score = (1 - alpha) / (k + rank + 1)
        scores[chunk_text] = scores.get(chunk_text, 0.0) + rrf_score
        if chunk_text not in chunk_data:
            chunk_data[chunk_text] = item.copy()
        _merge_retrieval_sources(chunk_data[chunk_text], item, "bm25")

    # 排序
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_text, rrf_score in sorted_chunks[:top_k]:
        item = chunk_data[chunk_text]
        item['rrf_score'] = rrf_score
        item['hybrid'] = True
        results.append(item)

    return results


# ═══════════════════════════════════════════════════
# P3.3c 多查询合并策略（借鉴 RAGPaper.query_rewriter._merge_results）
# ═══════════════════════════════════════════════════

def merge_multi_query_results(
    result_lists: List[List[dict]],
    *,
    mode: str = "rrf",
    top_k: int = 10,
    chunk_key: str = "chunk",
    rrf_k: int = 60,
) -> List[dict]:
    """合并多个 query 的检索结果（用于 multi-query expansion）

    Args:
        result_lists: 每个 query 的检索结果列表
        mode: "rrf" / "intersection" / "weighted_avg" / "union"
        top_k: 返回结果数量
        chunk_key: chunk 文本字段名
        rrf_k: RRF 衰减参数（仅 mode=rrf 时使用）

    Returns:
        合并后的结果列表

    模式说明：
    - rrf: 排名融合，鲁棒性强（默认推荐）
    - intersection: 只保留所有 query 都命中的 chunks，高 precision 场景
    - weighted_avg: 平均原始分数（每个 chunk 的 similarity 字段平均）
    - union: 去重并集，最大召回
    """
    if not result_lists:
        return []
    valid_lists = [r for r in result_lists if r]
    if len(valid_lists) <= 1:
        return (valid_lists[0] if valid_lists else [])[:top_k]

    if mode == "intersection":
        chunk_count: Dict[str, int] = {}
        chunk_data: Dict[str, dict] = {}
        chunk_score_sum: Dict[str, float] = {}
        for results in valid_lists:
            seen_in_this_list = set()
            for item in results:
                ct = item.get(chunk_key, "")
                if not ct or ct in seen_in_this_list:
                    continue
                seen_in_this_list.add(ct)
                chunk_count[ct] = chunk_count.get(ct, 0) + 1
                chunk_score_sum[ct] = chunk_score_sum.get(ct, 0.0) + float(
                    item.get("similarity") or item.get("score") or 0.0
                )
                if ct not in chunk_data:
                    chunk_data[ct] = item.copy()
        n_queries = len(valid_lists)
        intersected = [(ct, chunk_score_sum[ct]) for ct, cnt in chunk_count.items() if cnt == n_queries]
        intersected.sort(key=lambda x: x[1], reverse=True)
        out = []
        for ct, score in intersected[:top_k]:
            item = chunk_data[ct]
            item["multi_query_intersect_score"] = score
            item["multi_query_merge_mode"] = "intersection"
            out.append(item)
        if not out:
            # 交集为空兜底到 rrf
            return merge_multi_query_results(valid_lists, mode="rrf", top_k=top_k, chunk_key=chunk_key, rrf_k=rrf_k)
        return out

    if mode == "weighted_avg":
        chunk_score_sum: Dict[str, float] = {}
        chunk_count: Dict[str, int] = {}
        chunk_data: Dict[str, dict] = {}
        for results in valid_lists:
            for item in results:
                ct = item.get(chunk_key, "")
                if not ct:
                    continue
                score = float(item.get("similarity") or item.get("score") or 0.0)
                chunk_score_sum[ct] = chunk_score_sum.get(ct, 0.0) + score
                chunk_count[ct] = chunk_count.get(ct, 0) + 1
                if ct not in chunk_data:
                    chunk_data[ct] = item.copy()
        averaged = sorted(
            ((ct, chunk_score_sum[ct] / max(1, chunk_count[ct])) for ct in chunk_score_sum),
            key=lambda x: x[1],
            reverse=True,
        )
        out = []
        for ct, avg_score in averaged[:top_k]:
            item = chunk_data[ct]
            item["multi_query_avg_score"] = avg_score
            item["multi_query_merge_mode"] = "weighted_avg"
            out.append(item)
        return out

    if mode == "union":
        seen = set()
        out: List[dict] = []
        for results in valid_lists:
            for item in results:
                ct = item.get(chunk_key, "")
                if not ct or ct in seen:
                    continue
                seen.add(ct)
                copy = item.copy()
                copy["multi_query_merge_mode"] = "union"
                out.append(copy)
                if len(out) >= top_k:
                    return out
        return out

    # 默认 rrf
    return reciprocal_rank_fusion(*valid_lists, k=rrf_k, top_k=top_k, chunk_key=chunk_key)
