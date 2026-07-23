"""
联网搜索结果向量重排服务

用余弦相似度替代纯词法 token-overlap 评分，对 web 搜索结果进行语义重排。
当 embedding 模型不可用时静默降级，返回原始顺序结果。
"""

import asyncio
import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException

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


def _load_doc_index_payload(doc_id: str, vector_store_dir: str) -> Optional[dict]:
    """读取文档向量索引元数据，供 embedding 身份校验复用。"""
    chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")
    if not os.path.exists(chunks_path):
        return None
    try:
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.debug(f"读取文档 embedding 身份失败 doc_id={doc_id}: {e}")
    return None


def _build_skip_diagnostic(reason: str, message: str, **extra: Any) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "applied": False,
        "reason": reason,
        "message": message,
    }
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            diagnostic[key] = value
    return diagnostic


def _build_identity_skip_diagnostic(exc: HTTPException) -> dict[str, Any]:
    detail = str(getattr(exc, "detail", "") or "").strip()
    if "必须显式提供 embedding_model/provider/api_host" in detail:
        return _build_skip_diagnostic(
            "request_embedding_identity_missing",
            "联网语义重排已跳过：未提供与文档索引绑定的 embedding_model/provider/api_host",
        )
    if "Embedding 模型不能为空" in detail or "Embedding API 地址" in detail:
        return _build_skip_diagnostic(
            "index_embedding_identity_invalid",
            "联网语义重排已跳过：文档索引缺少完整有效的 embedding 身份",
        )
    return _build_skip_diagnostic(
        "embedding_identity_mismatch",
        "联网语义重排已跳过：请求 embedding 身份与文档索引不一致",
    )


def _finalize_rerank_result(
    results: list[dict],
    *,
    top_k: int,
    return_diagnostic: bool,
    diagnostic: Optional[dict[str, Any]] = None,
):
    trimmed = results[:top_k]
    if return_diagnostic:
        return trimmed, diagnostic
    return trimmed


async def rerank_web_results(
    query: str,
    results: list[dict],
    *,
    doc_id: str = "",
    vector_store_dir: str = "",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.2,
    return_diagnostic: bool = False,
) -> list[dict] | tuple[list[dict], Optional[dict[str, Any]]]:
    """用向量相似度对 web 搜索结果进行语义重排。

    先校验请求 embedding 身份与文档索引绑定身份一致，失败时安全跳过并返回原始结果。

    Args:
        query: 搜索查询
        results: 已经过词法重排的搜索结果列表 [{title, url, snippet}]
        doc_id: 文档 ID，用于查找 embedding 模型
        vector_store_dir: 向量存储目录
        api_key: Embedding API key（仅允许使用专用 embedding key）
        api_host: 保留兼容参数，不参与身份校验和请求
        embedding_model: 请求显式提供的 embedding 模型
        embedding_provider: 请求显式提供的 embedding provider
        embedding_api_host: 请求显式提供的 embedding API base URL
        top_k: 最终返回条数
        threshold: 余弦相似度过滤阈值（低于此值的结果被丢弃）
        return_diagnostic: 为 True 时返回 (results, diagnostic)

    Returns:
        语义重排后的结果列表；return_diagnostic=True 时返回 (results, diagnostic)
    """
    if not results:
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
        )

    if len(results) == 1:
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
        )

    index_payload = None
    if doc_id and vector_store_dir:
        index_payload = _load_doc_index_payload(doc_id, vector_store_dir)

    if not index_payload:
        diagnostic = _build_skip_diagnostic(
            "index_embedding_identity_unavailable",
            "联网语义重排已跳过：未找到文档索引的 embedding 身份",
        )
        logger.info("联网搜索向量重排跳过：%s", diagnostic["reason"])
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
            diagnostic=diagnostic,
        )

    try:
        from services.embedding_service import (
            KEYLESS_EMBEDDING_PROVIDERS,
            _resolve_verified_query_embedding_identity,
            get_embedding_function,
        )

        verified_embedding = _resolve_verified_query_embedding_identity(
            index_payload,
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_api_host=embedding_api_host,
        )
    except HTTPException as exc:
        diagnostic = _build_identity_skip_diagnostic(exc)
        logger.info("联网搜索向量重排跳过：%s", diagnostic["reason"])
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
            diagnostic=diagnostic,
        )
    except Exception as exc:
        diagnostic = _build_skip_diagnostic(
            "embedding_identity_verification_failed",
            "联网语义重排已跳过：无法校验 embedding 身份",
            error_type=type(exc).__name__,
        )
        logger.warning("联网搜索向量重排身份校验失败，已跳过: %s", type(exc).__name__)
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
            diagnostic=diagnostic,
        )

    if (
        verified_embedding["provider"] not in KEYLESS_EMBEDDING_PROVIDERS
        and not verified_embedding.get("api_key")
    ):
        diagnostic = _build_skip_diagnostic(
            "embedding_api_key_missing",
            "联网语义重排已跳过：缺少文档绑定的 embedding_api_key",
        )
        logger.info("联网搜索向量重排跳过：%s", diagnostic["reason"])
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
            diagnostic=diagnostic,
        )

    try:
        embed_fn = get_embedding_function(
            verified_embedding["model"],
            api_key=verified_embedding.get("api_key"),
            base_url=verified_embedding["api_host"],
            allow_model_fallback=False,
        )
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
            return _finalize_rerank_result(
                results,
                top_k=top_k,
                return_diagnostic=return_diagnostic,
            )

        effective_threshold = max(threshold, max_score * 0.5)
        filtered = [r for s, r in ranked if s >= effective_threshold]
        if not filtered:
            filtered = [ranked[0][1]]

        logger.info(
            f"联网搜索向量重排：{len(results)} → {len(filtered)} 条 "
            f"(model={verified_embedding['model']}, max_score={max_score:.3f})"
        )
        return _finalize_rerank_result(
            filtered,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
        )

    except Exception as e:
        diagnostic = _build_skip_diagnostic(
            "rerank_unavailable",
            "联网语义重排暂不可用，已保留原始联网结果",
            error_type=type(e).__name__,
        )
        logger.warning("联网搜索向量重排失败，降级返回词法结果: %s", type(e).__name__)
        return _finalize_rerank_result(
            results,
            top_k=top_k,
            return_diagnostic=return_diagnostic,
            diagnostic=diagnostic,
        )
