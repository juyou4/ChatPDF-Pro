import httpx
from typing import List, Optional


def cohere_rerank(query: str, documents: List[str], model: str, api_key: str, endpoint: Optional[str] = None, timeout: float = 30.0):
    """Call Cohere rerank API and return list of scores with indices."""
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
    """Call Jina rerank API and return list of scores with indices."""
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
