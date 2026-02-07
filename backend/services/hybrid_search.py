"""
混合检索服务 - 融合向量检索和BM25检索结果

设计哲学（参考paper-burner-x）：
- 向量检索：语义理解，处理同义词和语义相似
- BM25检索：关键词匹配，精确定位特定术语
- RRF融合：互补优势，提高检索鲁棒性

使用RRF (Reciprocal Rank Fusion) 算法融合两路结果：
- 不依赖分数归一化（向量距离和BM25分数量纲不同）
- 只依赖排名，简单稳定
"""
from typing import Dict, List, Optional, Tuple


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
    alpha: float = 0.5
) -> List[dict]:
    """
    混合检索：融合向量检索和BM25检索结果
    
    Args:
        vector_results: 向量检索结果
        bm25_results: BM25检索结果
        top_k: 返回结果数量
        alpha: 向量检索权重（0-1），默认0.5表示平等权重
        
    Returns:
        融合后的结果列表
    """
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

    # BM25检索结果（权重1-alpha）
    for rank, item in enumerate(bm25_results):
        chunk_text = item.get('chunk', '')
        if not chunk_text:
            continue
        rrf_score = (1 - alpha) / (k + rank + 1)
        scores[chunk_text] = scores.get(chunk_text, 0.0) + rrf_score
        if chunk_text not in chunk_data:
            chunk_data[chunk_text] = item.copy()

    # 排序
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_text, rrf_score in sorted_chunks[:top_k]:
        item = chunk_data[chunk_text]
        item['rrf_score'] = rrf_score
        item['hybrid'] = True
        results.append(item)

    return results
