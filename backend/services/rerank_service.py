import json
import logging
from typing import List, Optional

import httpx
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except (ImportError, OSError):
    _HAS_CROSS_ENCODER = False
from models.api_key_selector import select_api_key
from services import rerank_api_service

logger = logging.getLogger(__name__)


class RerankService:
    """重排服务：支持本地 CrossEncoder + 云端 Cohere/Jina"""

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _candidate_text(item: dict) -> str:
        text = (item.get("rerank_text") or item.get("chunk") or "").strip()
        return text or (item.get("chunk") or "")

    def _get_model(self, model_name: str):
        if not _HAS_CROSS_ENCODER:
            raise ValueError(
                "本地 rerank 模型不可用（sentence-transformers 未安装）。"
                "请使用远程 rerank API（Cohere/Jina/硅基流动等），"
                "或安装完整依赖: pip install -r requirements.txt"
            )
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
        pairs = [(query, self._candidate_text(item)) for item in candidates]
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
            documents=[self._candidate_text(c) for c in candidates],
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
            documents=[self._candidate_text(c) for c in candidates],
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
            documents=[self._candidate_text(c) for c in candidates],
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

    def _rerank_llm(self, query: str, candidates: List[dict], model_name: str, api_key: str, endpoint: Optional[str], provider: str, timeout: float) -> List[dict]:
        """使用通用 LLM 进行相关性评分重排序

        参考 kotaemon LLMTrulensScoring：让 LLM 对每个文档打 0-10 分。
        使用批量评分 prompt 一次评多个文档，减少 API 调用次数。

        适用场景：没有专用 rerank API 但有 LLM API 时的回退方案。
        """
        import asyncio
        import concurrent.futures
        from services.chat_service import call_ai_api

        def _run_async_local(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            return asyncio.run(coro)

        BATCH_SIZE = 5  # 每批评分的文档数
        scored_candidates = []

        for batch_start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[batch_start:batch_start + BATCH_SIZE]

            # 构建批量评分 prompt
            docs_text = ""
            for i, item in enumerate(batch):
                chunk = self._candidate_text(item)[:800]  # 截断避免超长
                docs_text += f"\n[Document {i + 1}]\n{chunk}\n"

            system_prompt = (
                "You are a relevance scoring assistant. "
                "Score each document's relevance to the query on a scale of 0-10. "
                "Output ONLY a JSON array of scores, e.g. [8, 3, 7, 5, 2]. "
                "No explanation needed."
            )
            user_prompt = f"Query: {query}\n\nDocuments:{docs_text}\n\nScores (JSON array):"

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = _run_async_local(call_ai_api(
                    messages=messages,
                    api_key=api_key,
                    model=model_name,
                    provider=provider,
                    endpoint=endpoint or "",
                    max_tokens=100,
                    temperature=0.0,
                ))

                # 解析分数
                scores_text = response.strip()
                # 提取 JSON 数组
                match = __import__('re').search(r'\[([\d,\s.]+)\]', scores_text)
                if match:
                    scores = json.loads(f"[{match.group(1)}]")
                else:
                    scores = json.loads(scores_text)

                for i, item in enumerate(batch):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    item["rerank_score"] = score / 10.0  # 归一化到 0-1
                    item["reranked"] = True
                    item["rerank_method"] = "llm"
                    scored_candidates.append(item)

            except Exception as e:
                logger.warning(f"[RerankService] LLM 批量评分失败: {e}")
                # 失败时保留原始分数
                for item in batch:
                    item["rerank_score"] = item.get("similarity", 0)
                    scored_candidates.append(item)

        sorted_results = sorted(scored_candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        self._normalize_rerank_scores(sorted_results)
        logger.info(f"[RerankService] LLM 重排序完成: {len(sorted_results)} 个候选")
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
            # LLM rerank 模式：使用通用聊天模型评分
            if provider == "llm":
                actual_key = select_api_key(api_key) if api_key else None
                if not actual_key:
                    raise ValueError("LLM rerank 需要提供 api_key")
                # model_name 格式: "provider:model"，如 "openai:gpt-4o-mini"
                llm_provider = "openai"
                llm_model = model_name
                if ":" in model_name:
                    llm_provider, llm_model = model_name.split(":", 1)
                return self._rerank_llm(query, candidates, llm_model, actual_key, endpoint, llm_provider, timeout)

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
