from typing import List, Optional

import httpx
from sentence_transformers import CrossEncoder
from services import rerank_api_service


class RerankService:
    """重排服务：支持本地 CrossEncoder + 云端 Cohere/Jina"""

    def __init__(self):
        self._cache = {}

    def _get_model(self, model_name: str) -> CrossEncoder:
        if model_name not in self._cache:
            self._cache[model_name] = CrossEncoder(model_name)
        return self._cache[model_name]

    def _rerank_local(self, query: str, candidates: List[dict], model_name: str) -> List[dict]:
        model = self._get_model(model_name)
        pairs = [(query, item["chunk"]) for item in candidates]
        scores = model.predict(pairs)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
            item["reranked"] = True
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def _rerank_cohere(self, query: str, candidates: List[dict], model_name: str, api_key: str, endpoint: Optional[str], timeout: float) -> List[dict]:
        scores = rerank_api_service.cohere_rerank(
            query=query,
            documents=[c["chunk"] for c in candidates],
            model=model_name,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
        )
        for idx, score in scores:
            if idx is None or idx >= len(candidates):
                continue
            candidates[idx]["rerank_score"] = float(score)
            candidates[idx]["reranked"] = True
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def _rerank_jina(self, query: str, candidates: List[dict], model_name: str, api_key: str, endpoint: Optional[str], timeout: float) -> List[dict]:
        scores = rerank_api_service.jina_rerank(
            query=query,
            documents=[c["chunk"] for c in candidates],
            model=model_name,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
        )
        for idx, score in scores:
            if idx is None or idx >= len(candidates):
                continue
            candidates[idx]["rerank_score"] = float(score)
            candidates[idx]["reranked"] = True
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        model_name: Optional[str] = None,
        provider: str = "local",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0
    ) -> List[dict]:
        if not candidates:
            return []

        model_name = model_name or "BAAI/bge-reranker-base"
        provider = (provider or "local").lower()

        try:
            if provider == "cohere":
                if not api_key:
                    raise ValueError("Cohere rerank 需要提供 api_key")
                return self._rerank_cohere(query, candidates, model_name, api_key, endpoint, timeout)
            if provider == "jina":
                if not api_key:
                    raise ValueError("Jina rerank 需要提供 api_key")
                return self._rerank_jina(query, candidates, model_name, api_key, endpoint, timeout)

            # 默认走本地 CrossEncoder
            return self._rerank_local(query, candidates, model_name)
        except Exception:
            # 回退到原有排序
            return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


rerank_service = RerankService()
