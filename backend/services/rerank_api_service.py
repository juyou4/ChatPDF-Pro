import httpx
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def cohere_rerank(query: str, documents: List[str], model: str, api_key: str, endpoint: Optional[str] = None, timeout: float = 30.0):
    """调用 Cohere rerank API，返回 (index, score) 列表"""
    url = endpoint or "https://api.cohere.com/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "rerank-multilingual-v3.0",
        "query": query,
        "documents": documents,
    }
    logger.info(f"[rerank_api] Cohere rerank: url={url}, model={payload['model']}, docs={len(documents)}")
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    scores = []
    for item in results:
        idx = item.get("index")
        score = float(item.get("relevance_score", 0.0))
        scores.append((idx, score))
    return scores


def jina_rerank(query: str, documents: List[str], model: str, api_key: str, endpoint: Optional[str] = None, timeout: float = 30.0):
    """调用 Jina rerank API，返回 (index, score) 列表"""
    url = endpoint or "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": [{"text": d} for d in documents],
    }
    logger.info(f"[rerank_api] Jina rerank: url={url}, model={payload['model']}, docs={len(documents)}")
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    scores = []
    for item in results:
        idx = item.get("index")
        score = float(item.get("score", 0.0))
        scores.append((idx, score))
    return scores


# Provider ID -> 默认 rerank endpoint 的映射
# 这些 provider 都使用与 Cohere 兼容的 rerank API 格式（返回 relevance_score）
OPENAI_LIKE_RERANK_ENDPOINTS = {
    "silicon": "https://api.siliconflow.cn/v1/rerank",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank",
}


def openai_like_rerank(query: str, documents: List[str], model: str, api_key: str, endpoint: Optional[str] = None, provider: str = "silicon", timeout: float = 30.0):
    """调用 OpenAI 兼容的 rerank API（硅基流动、阿里云等）

    这些服务的 rerank API 格式与 Cohere 兼容：
    - 请求体: {"model": ..., "query": ..., "documents": [...]}
    - 响应体: {"results": [{"index": N, "relevance_score": X}, ...]}

    Args:
        query: 查询文本
        documents: 待重排的文档列表
        model: 模型名称
        api_key: API 密钥
        endpoint: 自定义 endpoint（可选，默认根据 provider 自动选择）
        provider: 提供商 ID（用于选择默认 endpoint）
        timeout: 超时时间（秒）

    Returns:
        (index, score) 元组列表
    """
    url = endpoint or OPENAI_LIKE_RERANK_ENDPOINTS.get(provider, f"https://api.siliconflow.cn/v1/rerank")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    logger.info(f"[rerank_api] OpenAI-like rerank: provider={provider}, url={url}, model={model}, docs={len(documents)}")
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    scores = []
    for item in results:
        idx = item.get("index")
        # 兼容 relevance_score（Cohere/SiliconFlow 格式）和 score（Jina 格式）
        score = float(item.get("relevance_score", item.get("score", 0.0)))
        scores.append((idx, score))
    return scores
