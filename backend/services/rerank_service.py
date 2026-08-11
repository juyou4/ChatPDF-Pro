import json
import importlib.util
import logging
import os
import threading
import time
from typing import List, Optional

import httpx
CrossEncoder = None
_HAS_CROSS_ENCODER = importlib.util.find_spec("sentence_transformers") is not None
_CROSS_ENCODER_IMPORT_LOCK = threading.Lock()
from models.api_key_selector import select_api_key
from services import rerank_api_service

logger = logging.getLogger(__name__)


def _get_cross_encoder_class():
    """只在首次使用本地重排时加载 sentence-transformers/PyTorch。"""
    global CrossEncoder, _HAS_CROSS_ENCODER
    if not _HAS_CROSS_ENCODER:
        raise ImportError("sentence-transformers 未安装")
    if CrossEncoder is None:
        with _CROSS_ENCODER_IMPORT_LOCK:
            if CrossEncoder is None:
                try:
                    from sentence_transformers import CrossEncoder as cross_encoder_class
                except (ImportError, OSError):
                    _HAS_CROSS_ENCODER = False
                    raise
                CrossEncoder = cross_encoder_class
    return CrossEncoder


class LocalRerankModelUnavailable(RuntimeError):
    """本地 rerank 模型当前不可用，可直接降级到原始排序。"""


class RerankService:
    """重排服务：支持本地 CrossEncoder + 云端 Cohere/Jina"""

    LOCAL_LOAD_FAILURE_TTL_SECONDS = 300.0
    LOCAL_ALLOW_DOWNLOAD_ENV = "CHATPDF_LOCAL_RERANK_ALLOW_DOWNLOAD"
    RERANK_CHUNK_CHAR_LIMIT = 1800
    RERANK_CHUNK_OVERLAP = 180
    RERANK_MAX_CHUNKS_PER_CANDIDATE = 6

    def __init__(self):
        self._cache = {}
        self._load_failures = {}
        self._load_lock = threading.RLock()

    @staticmethod
    def _candidate_text(item: dict) -> str:
        text = (item.get("rerank_text") or item.get("chunk") or "").strip()
        return text or (item.get("chunk") or "")

    @classmethod
    def _split_long_rerank_text(cls, text: str) -> List[str]:
        value = " ".join(str(text or "").split())
        limit = max(400, int(cls.RERANK_CHUNK_CHAR_LIMIT))
        if len(value) <= limit:
            return [value] if value else []
        overlap = max(0, min(int(cls.RERANK_CHUNK_OVERLAP), limit // 3))
        chunks: List[str] = []
        start = 0
        while start < len(value) and len(chunks) < cls.RERANK_MAX_CHUNKS_PER_CANDIDATE:
            end = min(len(value), start + limit)
            if end < len(value):
                boundary = max(value.rfind("。", start, end), value.rfind(".", start, end), value.rfind("\n", start, end))
                if boundary > start + limit * 0.55:
                    end = boundary + 1
            chunk = value[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(value):
                break
            start = max(end - overlap, start + 1)
        return chunks or [value[:limit]]

    @classmethod
    def _expand_candidates_for_rerank(cls, candidates: List[dict]) -> tuple[List[dict], bool]:
        expanded: List[dict] = []
        used_chunking = False
        for parent_idx, item in enumerate(candidates):
            text = cls._candidate_text(item)
            pieces = cls._split_long_rerank_text(text)
            if len(pieces) > 1:
                used_chunking = True
            for chunk_idx, piece in enumerate(pieces or [""]):
                child = dict(item)
                child["rerank_text"] = piece
                child["_rerank_parent_index"] = parent_idx
                child["_rerank_chunk_index"] = chunk_idx
                child["_rerank_chunk_count"] = len(pieces)
                child["_rerank_original_chars"] = len(text)
                expanded.append(child)
        return expanded, used_chunking

    @classmethod
    def _aggregate_chunked_rerank_results(cls, reranked: List[dict], originals: List[dict]) -> List[dict]:
        best_by_parent: dict[int, dict] = {}
        for item in reranked:
            try:
                parent_idx = int(item.get("_rerank_parent_index"))
            except (TypeError, ValueError):
                continue
            current = best_by_parent.get(parent_idx)
            score = float(item.get("rerank_score", item.get("similarity", 0.0)) or 0.0)
            current_score = float(current.get("rerank_score", current.get("similarity", 0.0)) or 0.0) if current else float("-inf")
            if current is None or score > current_score:
                best_by_parent[parent_idx] = item

        aggregated: List[dict] = []
        for parent_idx, original in enumerate(originals):
            item = dict(original)
            best = best_by_parent.get(parent_idx)
            if best:
                for key in ("rerank_score", "similarity", "similarity_percent", "reranked", "rerank_method", "combined_score"):
                    if key in best:
                        item[key] = best[key]
                chunk_count = int(best.get("_rerank_chunk_count") or 1)
                if chunk_count > 1:
                    item["rerank_chunked"] = True
                    item["rerank_chunk_count"] = chunk_count
                    item["rerank_best_chunk_index"] = int(best.get("_rerank_chunk_index") or 0)
                    item["rerank_original_chars"] = int(best.get("_rerank_original_chars") or 0)
            else:
                item.setdefault("rerank_score", item.get("similarity", 0.0))
            aggregated.append(item)
        aggregated.sort(key=lambda x: x.get("rerank_score", x.get("similarity", 0.0)), reverse=True)
        cls._normalize_rerank_scores(aggregated)
        return aggregated

    @classmethod
    def _allow_local_model_download(cls) -> bool:
        raw = os.environ.get(cls.LOCAL_ALLOW_DOWNLOAD_ENV, "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _compact_error(exc: Exception) -> str:
        text = " ".join(str(exc).split())
        if len(text) > 300:
            return text[:297] + "..."
        return text or exc.__class__.__name__

    def _get_model(self, model_name: str):
        if not _HAS_CROSS_ENCODER:
            raise ValueError(
                "本地 rerank 模型不可用（sentence-transformers 未安装）。"
                "请使用远程 rerank API（Cohere/Jina/硅基流动等），"
                "或安装完整依赖: pip install -r requirements.txt"
            )
        with self._load_lock:
            failure = self._load_failures.get(model_name)
            now = time.monotonic()
            if failure:
                expires_at = float(failure.get("expires_at") or 0.0)
                if now < expires_at:
                    remaining = max(1, int(round(expires_at - now)))
                    reason = failure.get("reason") or "unknown error"
                    raise LocalRerankModelUnavailable(
                        f"本地 rerank 模型 {model_name} 最近加载失败，"
                        f"{remaining}s 内跳过重复加载: {reason}"
                    )
                self._load_failures.pop(model_name, None)
            if model_name not in self._cache:
                allow_download = self._allow_local_model_download()
                local_files_only = not allow_download
                suffix = (
                    "允许自动下载"
                    if allow_download
                    else f"仅使用本地缓存；设置 {self.LOCAL_ALLOW_DOWNLOAD_ENV}=1 可允许自动下载"
                )
                logger.info(f"[RerankService] 加载本地模型: {model_name}（{suffix}）")
                try:
                    cross_encoder_class = _get_cross_encoder_class()
                    self._cache[model_name] = cross_encoder_class(
                        model_name,
                        local_files_only=local_files_only,
                    )
                    logger.info(f"[RerankService] 模型 {model_name} 加载完成")
                except Exception as e:
                    reason = self._compact_error(e)
                    self._load_failures[model_name] = {
                        "expires_at": now + self.LOCAL_LOAD_FAILURE_TTL_SECONDS,
                        "reason": reason,
                    }
                    if allow_download:
                        logger.error(f"[RerankService] 模型 {model_name} 加载失败: {reason}")
                    else:
                        logger.info(f"[RerankService] 本地模型 {model_name} 未命中缓存，已跳过联网加载: {reason}")
                    raise LocalRerankModelUnavailable(reason) from e
            self._load_failures.pop(model_name, None)
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
        from services.completion_outcome import require_publishable_completion

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
                require_publishable_completion(response, operation="LLM rerank")

                # 解析分数
                if isinstance(response, dict):
                    choices = response.get("choices") or []
                    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
                    scores_text = str(message.get("content") or "") if isinstance(message, dict) else ""
                else:
                    scores_text = str(response or "")
                scores_text = scores_text.strip()
                # 提取 JSON 数组
                match = __import__('re').search(r'\[([\d,\s.]+)\]', scores_text)
                if match:
                    scores = json.loads(f"[{match.group(1)}]")
                else:
                    scores = json.loads(scores_text)

                if not isinstance(scores, list) or len(scores) != len(batch):
                    raise ValueError(
                        f"LLM rerank score count mismatch: expected={len(batch)} actual={len(scores) if isinstance(scores, list) else 0}"
                    )
                if any(not isinstance(score, (int, float)) or isinstance(score, bool) for score in scores):
                    raise ValueError("LLM rerank returned a non-numeric score")

                for i, item in enumerate(batch):
                    score = max(0.0, min(10.0, float(scores[i])))
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
        original_candidates = list(candidates)
        rerank_candidates, used_chunking = self._expand_candidates_for_rerank(original_candidates)
        if used_chunking:
            logger.info(
                "[RerankService] 长证据切片: 原候选=%s, rerank片段=%s, limit=%s",
                len(original_candidates),
                len(rerank_candidates),
                self.RERANK_CHUNK_CHAR_LIMIT,
            )
        candidates_for_provider = rerank_candidates if used_chunking else candidates

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
                result = self._rerank_llm(query, candidates_for_provider, llm_model, actual_key, endpoint, llm_provider, timeout)
                return self._aggregate_chunked_rerank_results(result, original_candidates) if used_chunking else result

            # 云端 provider 需要 API Key，从 Key 池中随机选择一个有效 Key
            # 动态 Provider 不应被固定白名单挡在本地回退之外。只要调用方
            # 明确提供了 rerank endpoint 和 API Key，就按 OpenAI/Cohere
            # 兼容的 ``query + documents`` 协议执行；local/llm 仍保持原路径。
            remote_with_explicit_endpoint = (
                provider not in {"local", "llm"}
                and bool(str(endpoint or "").strip())
            )
            if provider in self.CLOUD_RERANK_PROVIDERS or remote_with_explicit_endpoint:
                actual_key = select_api_key(api_key) if api_key else None
                if not actual_key:
                    raise ValueError(f"{provider} rerank 需要提供 api_key")

                if provider == "cohere":
                    result = self._rerank_cohere(query, candidates_for_provider, model_name, actual_key, endpoint, timeout)
                    return self._aggregate_chunked_rerank_results(result, original_candidates) if used_chunking else result
                if provider == "jina":
                    result = self._rerank_jina(query, candidates_for_provider, model_name, actual_key, endpoint, timeout)
                    return self._aggregate_chunked_rerank_results(result, original_candidates) if used_chunking else result

                # OpenAI 兼容的云端 rerank provider（硅基流动、阿里云等）
                result = self._rerank_openai_like(query, candidates_for_provider, model_name, actual_key, endpoint, provider, timeout)
                return self._aggregate_chunked_rerank_results(result, original_candidates) if used_chunking else result

            # 默认走本地 CrossEncoder
            result = self._rerank_local(query, candidates_for_provider, model_name)
            return self._aggregate_chunked_rerank_results(result, original_candidates) if used_chunking else result
        except Exception as e:
            # 记录错误日志后回退到原有排序
            if isinstance(e, LocalRerankModelUnavailable):
                logger.info(f"[RerankService] 重排序降级 (provider={provider}): {e}")
            else:
                logger.warning(f"[RerankService] 重排序失败 (provider={provider}): {e}", exc_info=True)
            return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


rerank_service = RerankService()
