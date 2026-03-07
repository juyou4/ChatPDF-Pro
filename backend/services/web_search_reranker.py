"""
联网搜索结果向量重排服务

用余弦相似度替代纯词法 token-overlap 评分，对 web 搜索结果进行语义重排。
当 embedding 模型不可用时静默降级，返回原始顺序结果。
"""

import asyncio
import logging
import os
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SNIPPET_JOIN_SEP = " "


def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """批量余弦相似度（已归一化向量直接点积）"""
    q = query_vec.flatten().astype("float32")
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(len(doc_vecs), dtype="float32")
    q = q / q_norm

    d = doc_vecs.astype("float32")
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    d = d / norms
    return (d @ q).astype("float32")


def _get_doc_embedding_model(doc_id: str, vector_store_dir: str) -> Optional[str]:
    """从文档向量索引元数据中读取 embedding 模型 ID，不命中返回 None。"""
    chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")
    if not os.path.exists(chunks_path):
        return None
    try:
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data.get("embedding_model")
    except Exception as e:
        logger.debug(f"读取文档 embedding 模型失败 doc_id={doc_id}: {e}")
    return None


async def rerank_web_results(
    query: str,
    results: list[dict],
    *,
    doc_id: str = "",
    vector_store_dir: str = "",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.2,
) -> list[dict]:
    """用向量相似度对 web 搜索结果进行语义重排。

    先尝试获取文档的 embedding 模型，失败时降级返回原始结果（保留词法重排结果）。

    Args:
        query: 搜索查询
        results: 已经过词法重排的搜索结果列表 [{title, url, snippet}]
        doc_id: 文档 ID，用于查找 embedding 模型
        vector_store_dir: 向量存储目录
        api_key: Embedding API key
        api_host: Embedding API base URL（可选）
        top_k: 最终返回条数
        threshold: 余弦相似度过滤阈值（低于此值的结果被丢弃）

    Returns:
        语义重排后的结果列表（降级时返回原始 results[:top_k]）
    """
    if not results:
        return results

    if len(results) == 1:
        return results[:top_k]

    embedding_model_id = None
    if doc_id and vector_store_dir:
        embedding_model_id = _get_doc_embedding_model(doc_id, vector_store_dir)

    if not embedding_model_id:
        logger.debug("联网搜索向量重排：未找到 embedding 模型，跳过重排")
        return results[:top_k]

    try:
        from services.embedding_service import get_embedding_function

        embed_fn = get_embedding_function(embedding_model_id, api_key, api_host)

        texts = [
            f"{r.get('title', '')} {r.get('snippet', '')}".strip() or r.get("url", "")
            for r in results
        ]

        def _sync_embed():
            q_vec = embed_fn([query])
            d_vecs = embed_fn(texts)
            return q_vec, d_vecs

        q_vec_raw, d_vecs_raw = await asyncio.to_thread(_sync_embed)
        q_vec = np.array(q_vec_raw).reshape(1, -1)
        d_vecs = np.array(d_vecs_raw)

        scores = _cosine_similarity(q_vec, d_vecs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        max_score = ranked[0][0] if ranked else 0.0
        if max_score < 0.05:
            logger.debug(f"联网搜索向量重排：max_score={max_score:.3f} 过低，返回原始顺序")
            return results[:top_k]

        effective_threshold = max(threshold, max_score * 0.5)
        filtered = [r for s, r in ranked if s >= effective_threshold]
        if not filtered:
            filtered = [ranked[0][1]]

        logger.info(
            f"联网搜索向量重排：{len(results)} → {len(filtered)} 条 "
            f"(model={embedding_model_id}, max_score={max_score:.3f})"
        )
        return filtered[:top_k]

    except Exception as e:
        logger.warning(f"联网搜索向量重排失败，降级返回词法结果: {e}")
        return results[:top_k]
