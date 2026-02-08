import logging
from typing import List, Optional

import httpx
from sentence_transformers import CrossEncoder
from models.api_key_selector import select_api_key
from services import rerank_api_service

logger = logging.getLogger(__name__)


class RerankService:
    """重排服务：支持本地 CrossEncoder + 云端 Cohere/Jina"""

    def __init__(self):
        self._cache = {}

    def _get_model(self, model_name: str) -> CrossEncoder:
        if model_name not in self._cache:
            logger.info(f"[RerankService] 加载本地模型: {model_name}（首次加载可能需要下载）")
            try:
                self._cache[model_name] = CrossEncoder(model_name)
                logger.info(f"[RerankService] 模型 {model_name} 加载完成")
            except Exception as e:
                logger.error(f"[RerankService] 模型 {model_name} 加载失败: {e}")
                raise
        return self._cache[model_name]

    @staticmethod
    def _normalize_rerank_scores(candidates: List[dict]) -> None:
        """将 rerank_score 归一化为 0-100 的百分比，写入 similarity / similarity_percent

        这样前端 formatSimilarity 直接读取 similarity_percent 就能显示
        与排序一致的匹配度，而不是原始向量距离。
        """
        if not candidates:
            return
        scores = [c.get("rerank_score", 0) for c in candidates]
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        score_range = max_score - min_score if max_score != min_score else 1

        for item in candidates:
            raw = item.get("rerank_score", 0)
            # 线性映射到 40-99 区间（避免出现 0% 或 100%）
            normalized = 40 + (raw - min_score) / score_range * 59
            item["similarity"] = round(normalized / 100, 4)
            item["similarity_percent"] = round(normalized, 2)

    def _rerank_local(self, query: str, candidates: List[dict], model_name: str) -> List[dict]:
        logger.info(f"[RerankService] 本地重排序: model={model_name}, 候选数={len(candidates)}")
        model = self._get_model(model_name)
        pairs = [(query, item["chunk"]) for item in candidates]
        scores = model.predict(pairs)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
            item["reranked"] = True
        sorted_results = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        self._normalize_rerank_scores(sorted_results)
        return sorted_results

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
        sorted_results = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        self._normalize_rerank_scores(sorted_results)
        return sorted_results

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
        sorted_results = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        self._normalize_rerank_scores(sorted_results)
        return sorted_results

    def _rerank_openai_like(self, query: str, candidates: List[dict], model_name: str, api_key: str, endpoint: Optional[str], provider: str, timeout: float) -> List[dict]:
        """通用 OpenAI 兼容 rerank（硅基流动、阿里云等）"""
        scores = rerank_api_service.openai_like_rerank(
            query=query,
            documents=[c["chunk"] for c in candidates],
            model=model_name,
            api_key=api_key,
            endpoint=endpoint,
            provider=provider,
            timeout=timeout,
        )
        for idx, score in scores:
            if idx is None or idx >= len(candidates):
                continue
            candidates[idx]["rerank_score"] = float(score)
            candidates[idx]["reranked"] = True
        sorted_results = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        self._normalize_rerank_scores(sorted_results)
        return sorted_results

    # 支持云端 rerank API 的 provider 列表
    # 这些 provider 都使用 OpenAI 兼容的 rerank API 格式
    CLOUD_RERANK_PROVIDERS = {"cohere", "jina", "silicon", "aliyun", "openai", "moonshot", "deepseek", "zhipu", "minimax"}

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
            # 云端 provider 需要 API Key，从 Key 池中随机选择一个有效 Key
            if provider in self.CLOUD_RERANK_PROVIDERS:
                actual_key = select_api_key(api_key) if api_key else None
                if not actual_key:
                    raise ValueError(f"{provider} rerank 需要提供 api_key")

                if provider == "cohere":
                    return self._rerank_cohere(query, candidates, model_name, actual_key, endpoint, timeout)
                if provider == "jina":
                    return self._rerank_jina(query, candidates, model_name, actual_key, endpoint, timeout)

                # OpenAI 兼容的云端 rerank provider（硅基流动、阿里云等）
                return self._rerank_openai_like(query, candidates, model_name, actual_key, endpoint, provider, timeout)

            # 默认走本地 CrossEncoder
            return self._rerank_local(query, candidates, model_name)
        except Exception as e:
            # 记录错误日志后回退到原有排序
            logger.warning(f"[RerankService] 重排序失败 (provider={provider}): {e}", exc_info=True)
            return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


rerank_service = RerankService()
