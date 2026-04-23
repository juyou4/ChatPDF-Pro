"""
RAGAS 评估脚本 - 评估 Chatpdf RAG 系统质量

使用 RAGAS 框架对 Chatpdf 的 RAG 管线进行端到端质量评估。

评估指标：
  - Faithfulness（忠实性）：答案是否有上下文支撑，防止幻觉
  - AnswerRelevancy（答案相关性）：答案是否回答了问题
  - LLMContextPrecisionWithoutReference（上下文精度）：检索内容是否有用
  - LLMContextRecall（上下文召回）：上下文是否覆盖了所需信息（需 ground_truth）

使用方式：
  # 使用配置文件（推荐）
  python -m tests.eval_ragas --config tests/ragas_eval_config.json

  # 命令行直接传参
  python -m tests.eval_ragas \\
    --backend-url http://localhost:8000 \\
    --doc-id <doc_id> \\
    --api-key <key> \\
    --model gpt-4o-mini \\
    --provider openai \\
    --questions tests/ragas_sample_questions.json

支持的 RAGAS 版本：0.1.x 和 0.2.x（自动检测）
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────── 数据结构 ───────────────────────────────


@dataclass
class QuestionItem:
    """单个评估问题"""
    question: str
    ground_truth: Optional[str] = None  # 有则计算 context_recall，无则跳过
    question_type: str = ""
    note: str = ""
    facet: str = ""
    difficulty: str = ""
    payload_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectedSample:
    """从 Chatpdf 后端收集的单个样本"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    latency_ms: float
    error: Optional[str] = None
    citations_count: int = 0
    question_type: str = ""
    note: str = ""
    facet: str = ""
    difficulty: str = ""
    request_overrides: Dict[str, Any] = field(default_factory=dict)
    retrieval_diagnostics: Dict[str, Any] = field(default_factory=dict)


def _metric_value_to_float(value: Any) -> Optional[float]:
    """将 RAGAS 返回值安全转换为 float，并过滤 NaN。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return score


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


_ANSWER_CORRECTNESS_EMBED_MAX_CHARS = 224
_ANSWER_CORRECTNESS_EMBED_TAIL_CHARS = 48


def _truncate_for_answer_correctness_embedding(
    text: str,
    max_chars: int = _ANSWER_CORRECTNESS_EMBED_MAX_CHARS,
    tail_chars: int = _ANSWER_CORRECTNESS_EMBED_TAIL_CHARS,
) -> str:
    """限制 AnswerCorrectness 的 embedding 输入长度，规避 512-token 提供商上限。"""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return " "
    if len(normalized) <= max_chars:
        return normalized

    tail_chars = max(0, min(tail_chars, max_chars // 2))
    head_chars = max_chars - tail_chars - 5
    if head_chars <= 0:
        return normalized[:max_chars]
    return f"{normalized[:head_chars]} ... {normalized[-tail_chars:]}"


class _AnswerCorrectnessEmbeddingProxy:
    """仅给 AnswerCorrectness 用的 embeddings 代理，避免长文本触发 413。"""

    def __init__(self, inner_embeddings: Any):
        self.inner_embeddings = inner_embeddings

    def _truncate(self, text: str) -> str:
        return _truncate_for_answer_correctness_embedding(text)

    def embed_query(self, text: str) -> List[float]:
        return self.inner_embeddings.embed_query(self._truncate(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.inner_embeddings.embed_documents([self._truncate(text) for text in texts])

    async def aembed_query(self, text: str) -> List[float]:
        return await self.inner_embeddings.aembed_query(self._truncate(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.inner_embeddings.aembed_documents([self._truncate(text) for text in texts])

    async def embed_text(self, text: str, is_async: bool = True) -> List[float]:
        truncated = self._truncate(text)
        if hasattr(self.inner_embeddings, "embed_text"):
            return await self.inner_embeddings.embed_text(truncated, is_async=is_async)
        if is_async and hasattr(self.inner_embeddings, "aembed_query"):
            return await self.inner_embeddings.aembed_query(truncated)
        return self.inner_embeddings.embed_query(truncated)

    async def embed_texts(self, texts: List[str], is_async: bool = True) -> List[List[float]]:
        normalized = [self._truncate(text) for text in texts]
        if hasattr(self.inner_embeddings, "embed_texts"):
            return await self.inner_embeddings.embed_texts(normalized, is_async=is_async)
        if is_async and hasattr(self.inner_embeddings, "aembed_documents"):
            return await self.inner_embeddings.aembed_documents(normalized)
        return self.inner_embeddings.embed_documents(normalized)

    def set_run_config(self, run_config: Any) -> None:
        if hasattr(self.inner_embeddings, "set_run_config"):
            self.inner_embeddings.set_run_config(run_config)


def _extract_retrieval_diagnostics(sample: CollectedSample) -> Dict[str, Any]:
    diagnostics = sample.retrieval_diagnostics or {}
    if isinstance(diagnostics.get("retrieval"), dict):
        return diagnostics.get("retrieval") or {}
    return diagnostics if isinstance(diagnostics, dict) else {}


def _average_diagnostic(samples: List[CollectedSample], key: str) -> Optional[float]:
    values: List[float] = []
    for sample in samples:
        diag = _extract_retrieval_diagnostics(sample)
        value = diag.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _build_eval_warnings(question_items: List[QuestionItem], embed_model: str) -> List[str]:
    warnings: List[str] = []
    model_lower = (embed_model or "").lower()

    if any(_contains_cjk(item.question) for item in question_items):
        if "-en" in model_lower or "english" in model_lower:
            warnings.append(
                f"当前 RAGAS embedding 模型 `{embed_model}` 偏英文，中文问题的 AnswerRelevancy 可能失真。"
            )

    synthesis_without_gt = sum(
        1
        for item in question_items
        if item.question_type in {"analytical", "overview"} and not (item.ground_truth or "").strip()
    )
    if synthesis_without_gt:
        warnings.append(
            f"有 {synthesis_without_gt} 个 analytical/overview 问题未提供 ground_truth，Faithfulness 更容易受总结/外推型答案影响。"
        )

    overrides_count = sum(1 for item in question_items if item.payload_overrides)
    if overrides_count:
        warnings.append(
            f"评测集中有 {overrides_count} 个问题带有单题检索覆盖参数，跨实验对比时请确保这些覆盖参数保持一致。"
        )

    return warnings


def _normalize_eval_text(text: str) -> str:
    """压缩空白，避免评测文本因换行/空格膨胀。"""
    return re.sub(r"\s+", " ", text or "").strip()


def _trim_answer_correctness_text(
    text: str,
    *,
    max_tokens: int = 380,
    max_cjk_chars: int = 380,
    max_text_chars: int = 1200,
) -> str:
    """为 AnswerCorrectness 的 embedding 路径做保守裁剪，绕开 512-token 限制。"""
    normalized = _normalize_eval_text(text)
    if not normalized:
        return ""

    # 裁掉常见引用尾巴，优先保留直接答案内容。
    normalized = re.split(r"(?:参考文献|引用来源|CITATION LIST)\s*[:：]?", normalized, maxsplit=1)[0].strip()
    normalized = re.sub(r"\[(?:\d+|[^\]]{1,24})\]", "", normalized)

    char_limit = max_cjk_chars if _contains_cjk(normalized) else max_text_chars
    normalized = normalized[:char_limit].strip()

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(normalized)
        if len(token_ids) > max_tokens:
            normalized = encoding.decode(token_ids[:max_tokens]).strip()
    except Exception:
        pass

    return normalized


def _build_answer_correctness_sample(sample_cls: Any, sample: Any) -> Any:
    """构造仅供 AnswerCorrectness 使用的裁剪样本。"""
    row = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
    return sample_cls(
        user_input=_normalize_eval_text(row.get("user_input") or ""),
        response=_trim_answer_correctness_text(row.get("response") or ""),
        reference=_trim_answer_correctness_text(row.get("reference") or ""),
        retrieved_contexts=[],
    )


class _AnswerCorrectnessEmbeddingProxy:
    """仅在 AnswerSimilarity 的 embedding 调用前做更激进裁剪。"""

    def __init__(self, base_embeddings: Any):
        self._base_embeddings = base_embeddings

    def _prepare(self, text: str) -> str:
        return _trim_answer_correctness_text(
            text,
            max_tokens=220,
            max_cjk_chars=220,
            max_text_chars=700,
        )

    async def aembed_text(self, text: str) -> Any:
        prepared = self._prepare(text)
        if hasattr(self._base_embeddings, "aembed_text"):
            return await self._base_embeddings.aembed_text(prepared)
        return await self._base_embeddings.embed_text(prepared)

    async def embed_text(self, text: str) -> Any:
        prepared = self._prepare(text)
        if hasattr(self._base_embeddings, "embed_text"):
            return await self._base_embeddings.embed_text(prepared)
        return await self._base_embeddings.aembed_text(prepared)

    def set_run_config(self, run_config: Any) -> None:
        if hasattr(self._base_embeddings, "set_run_config"):
            self._base_embeddings.set_run_config(run_config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_embeddings, name)


# ─────────────────────────────── Chatpdf 客户端 ───────────────────────────────


class ChatpdfClient:
    """调用 Chatpdf 后端 /chat 接口，提取 answer + contexts"""

    def __init__(
        self,
        backend_url: str,
        doc_id: str,
        api_key: str,
        model: str,
        provider: str,
        api_host: str = "",
        top_k: int = 10,
        candidate_k: int = 20,
        use_rerank: bool = False,
        reranker_model: str = "",
        rerank_provider: str = "",
        rerank_api_key: str = "",
        rerank_endpoint: str = "",
        embedding_api_key: str = "",
        enable_jieba_bm25: bool = True,
        num_expand_context_chunk: int = 1,
        enable_memory: bool = True,
        enable_glossary: bool = True,
        enable_graphrag: bool = False,
        enable_agent_retrieval: bool = False,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.doc_id = doc_id
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.api_host = api_host
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.use_rerank = use_rerank
        self.reranker_model = reranker_model
        self.rerank_provider = rerank_provider
        self.rerank_api_key = rerank_api_key
        self.rerank_endpoint = rerank_endpoint
        self.embedding_api_key = embedding_api_key
        self.enable_jieba_bm25 = enable_jieba_bm25
        self.num_expand_context_chunk = num_expand_context_chunk
        self.enable_memory = enable_memory
        self.enable_glossary = enable_glossary
        self.enable_graphrag = enable_graphrag
        self.enable_agent_retrieval = enable_agent_retrieval

    def _build_payload(self, question: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
        payload = {
            "doc_id": self.doc_id,
            "question": question,
            "api_key": self.api_key,
            "model": self.model,
            "api_provider": self.provider,
            "enable_vector_search": True,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "use_rerank": self.use_rerank,
            "enable_jieba_bm25": self.enable_jieba_bm25,
            "num_expand_context_chunk": self.num_expand_context_chunk,
            "stream_output": False,
            "enable_memory": self.enable_memory,
            "enable_glossary": self.enable_glossary,
            "enable_graphrag": self.enable_graphrag,
            "enable_agent_retrieval": self.enable_agent_retrieval,
            "enable_web_search": False,
            "chat_history": [],
        }
        if self.api_host:
            payload["api_host"] = self.api_host
        if self.embedding_api_key:
            payload["embedding_api_key"] = self.embedding_api_key
        if self.use_rerank and self.reranker_model:
            payload["reranker_model"] = self.reranker_model
        if self.use_rerank and self.rerank_provider:
            payload["rerank_provider"] = self.rerank_provider
        if self.use_rerank and self.rerank_api_key:
            payload["rerank_api_key"] = self.rerank_api_key
        if self.use_rerank and self.rerank_endpoint:
            payload["rerank_endpoint"] = self.rerank_endpoint
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    payload[key] = value
        return payload

    def _extract_contexts(self, retrieval_meta: dict) -> List[str]:
        """从 retrieval_meta 中提取上下文片段文本，用于 RAGAS 评估

        优先使用 context_segments（完整意群文本），仅在无 segments 时
        回退到 highlight_text。避免两者同时出现导致内容重复、
        ContextPrecision 被短片段拉低。
        """
        seen: set = set()
        contexts: List[str] = []

        # 优先：context_segments（完整上下文文本）
        for seg in retrieval_meta.get("context_segments") or []:
            if isinstance(seg, dict):
                text = (seg.get("text") or "").strip()
            else:
                text = str(seg).strip()
            if text and text not in seen:
                seen.add(text)
                contexts.append(text)

        # 回退：仅当 context_segments 为空时使用 highlight_text
        if not contexts:
            for cit in retrieval_meta.get("citations") or []:
                text = (cit.get("highlight_text") or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    contexts.append(text)

        return contexts

    async def query(
        self,
        question: str,
        timeout: float = 120.0,
        payload_overrides: Optional[Dict[str, Any]] = None,
    ) -> CollectedSample:
        """调用 /chat 接口获取答案和上下文"""
        t0 = time.perf_counter()
        payload = self._build_payload(question, payload_overrides)

        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(
                    f"{self.backend_url}/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            latency = (time.perf_counter() - t0) * 1000
            detail = ""
            try:
                detail = e.response.text[:300]
            except Exception:
                pass
            return CollectedSample(
                question=question, answer="", contexts=[],
                ground_truth=None, latency_ms=round(latency, 1),
                error=f"HTTP {e.response.status_code}: {detail}",
                request_overrides=payload_overrides or {},
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return CollectedSample(
                question=question, answer="", contexts=[],
                ground_truth=None, latency_ms=round(latency, 1),
                error=str(e),
                request_overrides=payload_overrides or {},
            )

        latency = (time.perf_counter() - t0) * 1000
        answer = data.get("answer", "") or ""
        retrieval_meta = data.get("retrieval_meta") or {}
        contexts = self._extract_contexts(retrieval_meta)
        citations_count = len(retrieval_meta.get("citations") or [])
        retrieval_diagnostics = retrieval_meta.get("diagnostics") or {}

        return CollectedSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=None,
            latency_ms=round(latency, 1),
            citations_count=citations_count,
            request_overrides=payload_overrides or {},
            retrieval_diagnostics=retrieval_diagnostics,
        )


# ─────────────────────────────── RAGAS 版本检测 ───────────────────────────────


def _detect_ragas_version() -> str:
    """检测已安装的 RAGAS 版本，返回 'v2'（0.2+）或 'v1'（0.1.x）"""
    try:
        import ragas
        ver = getattr(ragas, "__version__", "0.0.0")
        major, minor = int(ver.split(".")[0]), int(ver.split(".")[1])
        if major >= 1 or (major == 0 and minor >= 2):
            return "v2"
        return "v1"
    except ImportError:
        raise ImportError(
            "未安装 ragas，请运行：pip install ragas>=0.2\n"
            "或安装评估专用依赖：pip install -r requirements-eval.txt"
        )


# ─────────────────────────────── RAGAS 评估（v2 API）───────────────────────────────


def _import_ragas_metrics():
    """兼容 ragas 0.2.x / 0.4.x 的指标导入"""
    try:
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )

    try:
        from ragas.metrics.collections import LLMContextRecall
    except ImportError:
        try:
            from ragas.metrics import LLMContextRecall
        except ImportError:
            LLMContextRecall = None

    try:
        # 对 evaluate() 旧评测链路优先使用 ragas.metrics 下的兼容类；
        # collections 版本虽然更新，但不兼容当前脚本的旧入口。
        from ragas.metrics import AnswerCorrectness
    except ImportError:
        try:
            from ragas.metrics.collections import AnswerCorrectness
        except ImportError:
            AnswerCorrectness = None

    return (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        AnswerCorrectness,
    )


def _init_answer_correctness(metric_cls: Any, ragas_llm: Any, ragas_embeddings: Any) -> Any:
    """兼容不同 RAGAS 版本下 AnswerCorrectness 的构造签名。"""
    if metric_cls is None:
        return None

    init_attempts = (
        {"llm": ragas_llm, "embeddings": ragas_embeddings},
        {"llm": ragas_llm},
        {},
    )
    last_error: Optional[Exception] = None
    for kwargs in init_attempts:
        try:
            return metric_cls(**kwargs)
        except TypeError as e:
            last_error = e
        except Exception as e:
            last_error = e
            break

    if last_error is not None:
        logger.warning(f"AnswerCorrectness 初始化失败，跳过：{last_error}")
    return None


def _build_answer_correctness_metric(
    metric_cls: Any,
    judge_model: str,
    judge_api_key: str,
    judge_api_base: str,
    fallback_embeddings: Any,
) -> Any:
    """为 ragas 0.4.x 单独构造 AnswerCorrectness 所需的 modern LLM。"""
    if metric_cls is None:
        return None

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

        llm_client_kwargs: dict = {"api_key": judge_api_key}
        if judge_api_base:
            llm_client_kwargs["base_url"] = judge_api_base
        modern_llm = llm_factory(
            judge_model,
            provider="openai",
            client=OpenAI(**llm_client_kwargs),
            temperature=0,
            max_tokens=2048,
        )
        compatibility_embeddings = (
            _AnswerCorrectnessEmbeddingProxy(fallback_embeddings)
            if fallback_embeddings is not None
            else None
        )
        return _init_answer_correctness(metric_cls, modern_llm, compatibility_embeddings)
    except Exception as e:
        logger.warning(f"AnswerCorrectness modern 初始化失败，跳过：{e}")
        return None


def _series_metric_summary(values: List[Any]) -> tuple[Optional[float], Dict[str, int], List[Optional[float]]]:
    """从 DataFrame 单列结果中提取平均分、统计信息和逐条分数。"""
    per_values = [_metric_value_to_float(value) for value in values]
    valid_values = [value for value in per_values if value is not None]
    avg_score = float(sum(valid_values) / len(valid_values)) if valid_values else None
    stats = {
        "valid_count": len(valid_values),
        "nan_count": int(len(values) - len(valid_values)),
        "total_count": int(len(values)),
    }
    return avg_score, stats, per_values


def _evaluate_metric_samplewise(
    metric_label: str,
    metric_builder,
    dataset_samples: List[Any],
    sample_indices: List[int],
    run_cfg: Any,
    non_data_cols: set[str],
) -> tuple[dict, dict]:
    """逐条评估慢指标，避免批量并发时的超时与大输出失控。"""
    from ragas import evaluate, EvaluationDataset
    try:
        from ragas.run_config import RunConfig
        sample_run_cfg = RunConfig(timeout=240, max_retries=1, max_wait=120)
    except Exception:
        sample_run_cfg = run_cfg

    per_metric_values: dict[str, List[Optional[float]]] = {}

    for local_idx, original_idx in enumerate(sample_indices):
        try:
            metric = metric_builder()
        except Exception as e:
            logger.warning(f"{metric_label} 单题初始化失败，跳过 idx={original_idx + 1}: {e}")
            continue

        if metric is None:
            continue

        try:
            single_dataset = EvaluationDataset(samples=[dataset_samples[original_idx]])
            single_result = (
                evaluate(dataset=single_dataset, metrics=[metric], run_config=sample_run_cfg)
                if sample_run_cfg
                else evaluate(dataset=single_dataset, metrics=[metric])
            )
            single_df = single_result.to_pandas()
        except Exception as e:
            logger.warning(f"{metric_label} 单题评估失败，跳过 idx={original_idx + 1}: {e}")
            continue

        metric_cols = [col for col in single_df.columns if col not in non_data_cols]
        if not metric_cols:
            continue

        for col in metric_cols:
            if col not in per_metric_values:
                per_metric_values[col] = [None] * len(sample_indices)
            per_metric_values[col][local_idx] = _metric_value_to_float(single_df.iloc[0].get(col))

    scores: dict = {}
    metric_stats: dict = {}
    for col, values in per_metric_values.items():
        avg_score, stats, normalized_values = _series_metric_summary(values)
        metric_stats[col] = stats
        if avg_score is not None:
            scores[col] = avg_score
        per_metric_values[col] = normalized_values

    return scores, per_metric_values


def _run_ragas_v2(
    samples: List[CollectedSample],
    judge_model: str,
    judge_api_key: str,
    judge_api_base: str,
    embed_model: str,
    embed_api_key: str,
    embed_api_base: str,
) -> dict:
    """RAGAS 0.2+ / 0.4.x 评估"""
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        AnswerCorrectness,
    ) = _import_ragas_metrics()

    # 构建 LLM judge
    llm_kwargs: dict = {"model": judge_model, "api_key": judge_api_key, "temperature": 0, "request_timeout": 120}
    if judge_api_base:
        llm_kwargs["base_url"] = judge_api_base
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))

    # 构建 Embeddings（支持 local:model-name 格式，使用 HuggingFace 本地模型）
    if embed_model.startswith("local:"):
        local_model_name = embed_model[len("local:"):]
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=local_model_name)
            )
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=local_model_name)
            )
    else:
        emb_kwargs: dict = {"model": embed_model, "api_key": embed_api_key or judge_api_key}
        if embed_api_base or judge_api_base:
            emb_kwargs["base_url"] = embed_api_base or judge_api_base
        ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(**emb_kwargs))

    has_gt = any(s.ground_truth for s in samples)

    # 指标列表
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        LLMContextPrecisionWithoutReference(llm=ragas_llm),
    ]
    # 构建 dataset
    dataset_samples = []
    for s in samples:
        kwargs: dict = {
            "user_input": s.question,
            "response": s.answer,
            "retrieved_contexts": s.contexts if s.contexts else ["（无检索结果）"],
        }
        if s.ground_truth:
            kwargs["reference"] = s.ground_truth
        dataset_samples.append(SingleTurnSample(**kwargs))

    dataset = EvaluationDataset(samples=dataset_samples)
    try:
        from ragas.run_config import RunConfig
        run_cfg = RunConfig(timeout=120, max_retries=3, max_wait=60)
    except Exception:
        run_cfg = None
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_cfg) if run_cfg else evaluate(dataset=dataset, metrics=metrics)
    # RAGAS 0.4.x: 用 to_pandas() 提取各指标平均分
    df = result.to_pandas()
    non_data_cols = {"user_input", "response", "retrieved_contexts", "reference"}
    scores: dict = {}
    metric_cols = [col for col in df.columns if col not in non_data_cols]
    metric_stats: dict = {}
    for col in metric_cols:
        avg_score, stats, _ = _series_metric_summary(df[col].tolist())
        metric_stats[col] = stats
        if avg_score is not None:
            scores[col] = avg_score

    # 保存逐条分数用于诊断
    per_sample_scores = []
    for i, row in df.iterrows():
        item = {"index": i + 1, "question": row.get("user_input", "")[:80]}
        for col in metric_cols:
            item[col] = _metric_value_to_float(row.get(col))
        per_sample_scores.append(item)

    gt_indices = [idx for idx, sample in enumerate(samples) if (sample.ground_truth or "").strip()]
    if has_gt and gt_indices:
        gt_dataset = EvaluationDataset(samples=[dataset_samples[idx] for idx in gt_indices])
        answer_correctness_samples = [
            _build_answer_correctness_sample(SingleTurnSample, dataset_samples[idx])
            for idx in gt_indices
        ]
        gt_metric_builders = []

        if LLMContextRecall is not None:
            gt_metric_builders.append((
                "LLMContextRecall",
                lambda: LLMContextRecall(llm=ragas_llm),
            ))
        if AnswerCorrectness is not None:
            gt_metric_builders.append((
                "AnswerCorrectness",
                lambda: _build_answer_correctness_metric(
                    AnswerCorrectness,
                    judge_model=judge_model,
                    judge_api_key=judge_api_key,
                    judge_api_base=judge_api_base,
                    fallback_embeddings=ragas_embeddings,
                ),
            ))

        for metric_label, metric_builder in gt_metric_builders:
            if metric_label == "AnswerCorrectness":
                ac_scores, ac_per_values = _evaluate_metric_samplewise(
                    metric_label=metric_label,
                    metric_builder=metric_builder,
                    dataset_samples=answer_correctness_samples,
                    sample_indices=list(range(len(answer_correctness_samples))),
                    run_cfg=run_cfg,
                    non_data_cols=non_data_cols,
                )
                for col, stats in (
                    (col, {"valid_count": len([v for v in values if v is not None]),
                           "nan_count": len([v for v in values if v is None]),
                           "total_count": len(values)})
                    for col, values in ac_per_values.items()
                ):
                    metric_stats[col] = stats
                for col, avg_score in ac_scores.items():
                    scores[col] = avg_score
                for item in per_sample_scores:
                    item.setdefault("answer_correctness", None)
                for col, values in ac_per_values.items():
                    for local_idx, original_idx in enumerate(gt_indices):
                        per_sample_scores[original_idx][col] = values[local_idx]
                continue

            try:
                metric = metric_builder()
            except Exception as e:
                logger.warning(f"{metric_label} 初始化失败，跳过：{e}")
                continue

            if metric is None:
                continue

            try:
                gt_result = (
                    evaluate(dataset=gt_dataset, metrics=[metric], run_config=run_cfg)
                    if run_cfg
                    else evaluate(dataset=gt_dataset, metrics=[metric])
                )
                gt_df = gt_result.to_pandas()
            except Exception as e:
                logger.warning(f"{metric_label} 评估失败，跳过：{e}")
                continue

            gt_metric_cols = [col for col in gt_df.columns if col not in non_data_cols]
            if not gt_metric_cols:
                continue

            for col in gt_metric_cols:
                avg_score, stats, per_values = _series_metric_summary(gt_df[col].tolist())
                metric_stats[col] = stats
                if avg_score is not None:
                    scores[col] = avg_score
                for item in per_sample_scores:
                    item.setdefault(col, None)
                for local_idx, original_idx in enumerate(gt_indices):
                    per_sample_scores[original_idx][col] = per_values[local_idx]

    scores["_per_sample"] = per_sample_scores
    scores["_meta"] = {
        "ragas_version": "v2",
        "sample_count": len(samples),
        "ground_truth_sample_count": len(gt_indices),
        "metric_stats": metric_stats,
    }

    return scores


# ─────────────────────────────── RAGAS 评估（v1 API）───────────────────────────────


def _run_ragas_v1(
    samples: List[CollectedSample],
    judge_model: str,
    judge_api_key: str,
    judge_api_base: str,
    embed_model: str,
    embed_api_key: str,
    embed_api_base: str,
) -> dict:
    """RAGAS 0.1.x 旧版 API 评估"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset as HFDataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm_kwargs: dict = {"model": judge_model, "openai_api_key": judge_api_key, "temperature": 0}
    if judge_api_base:
        llm_kwargs["openai_api_base"] = judge_api_base
    llm = ChatOpenAI(**llm_kwargs)

    emb_kwargs: dict = {"model": embed_model, "openai_api_key": embed_api_key or judge_api_key}
    if embed_api_base or judge_api_base:
        emb_kwargs["openai_api_base"] = embed_api_base or judge_api_base
    embeddings = OpenAIEmbeddings(**emb_kwargs)

    metrics = [faithfulness, answer_relevancy, context_precision]
    has_gt = any(s.ground_truth for s in samples)

    data_dict: dict = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts if s.contexts else ["（无检索结果）"] for s in samples],
    }
    if has_gt:
        try:
            from ragas.metrics import context_recall
            metrics.append(context_recall)
        except ImportError:
            pass
        try:
            from ragas.metrics import answer_correctness
            metrics.append(answer_correctness)
        except ImportError:
            pass
        data_dict["ground_truths"] = [s.ground_truth or "" for s in samples]

    hf_dataset = HFDataset.from_dict(data_dict)
    result = evaluate(hf_dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    scores = {}
    for key, value in dict(result).items():
        metric_value = _metric_value_to_float(value)
        scores[key] = metric_value if metric_value is not None else value
    scores["_meta"] = {
        "ragas_version": "v1",
        "sample_count": len(samples),
    }
    return scores


# ─────────────────────────────── 主评估入口 ───────────────────────────────


def run_ragas_evaluation(
    samples: List[CollectedSample],
    judge_model: str,
    judge_api_key: str,
    judge_api_base: str = "",
    embed_model: str = "text-embedding-3-small",
    embed_api_key: str = "",
    embed_api_base: str = "",
) -> dict:
    """运行 RAGAS 评估，自动适配 v1/v2 API"""
    valid = [s for s in samples if not s.error and s.answer and s.contexts]
    if not valid:
        logger.warning("没有有效样本（answer 非空 + contexts 非空），跳过 RAGAS 评估")
        return {}

    logger.info(f"RAGAS 评估样本数: {len(valid)}/{len(samples)}")
    version = _detect_ragas_version()
    logger.info(f"检测到 RAGAS API 版本: {version}")

    if version == "v2":
        return _run_ragas_v2(
            valid, judge_model, judge_api_key, judge_api_base,
            embed_model, embed_api_key, embed_api_base,
        )
    else:
        return _run_ragas_v1(
            valid, judge_model, judge_api_key, judge_api_base,
            embed_model, embed_api_key, embed_api_base,
        )


# ─────────────────────────────── 报告输出 ───────────────────────────────

_METRIC_LABELS: Dict[str, str] = {
    "faithfulness": "忠实性 (Faithfulness)",
    "answer_relevancy": "答案相关性 (AnswerRelevancy)",
    "llm_context_precision_without_reference": "上下文精度 (ContextPrecision)",
    "context_precision": "上下文精度 (ContextPrecision)",
    "llm_context_recall": "上下文召回 (ContextRecall)",
    "context_recall": "上下文召回 (ContextRecall)",
    "context_entity_recall": "实体召回 (EntityRecall)",
    "answer_correctness": "答案正确性 (AnswerCorrectness)",
}


def _score_bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def print_summary_table(samples: List[CollectedSample], ragas_scores: dict) -> None:
    """格式化打印评估报告"""
    print("\n" + "=" * 72)
    print("   ChatPDF RAG 系统 · RAGAS 评估报告")
    print("=" * 72)

    total = len(samples)
    errors = sum(1 for s in samples if s.error)
    valid = total - errors
    valid_samples = [s for s in samples if not s.error]
    avg_ctx = sum(len(s.contexts) for s in valid_samples) / max(valid, 1)
    avg_lat = sum(s.latency_ms for s in valid_samples) / max(valid, 1)
    avg_dup = _average_diagnostic(valid_samples, "duplicate_chunk_ratio")
    avg_group_cov = _average_diagnostic(valid_samples, "unique_group_coverage")
    avg_ref_pollution = _average_diagnostic(valid_samples, "reference_pollution_ratio")
    avg_focus_ratio = _average_diagnostic(valid_samples, "focus_mode_avg_compression_ratio")
    avg_path_diversity = _average_diagnostic(valid_samples, "path_diversity_ratio")
    avg_singleton_paths = _average_diagnostic(valid_samples, "singleton_path_count")

    print(f"\n  样本统计")
    print(f"  {'总样本数':<22} {total}")
    print(f"  {'有效样本数':<22} {valid}")
    print(f"  {'错误样本数':<22} {errors}")
    print(f"  {'平均检索片段数':<22} {avg_ctx:.1f}")
    print(f"  {'平均响应时间':<22} {avg_lat:.0f} ms")
    if avg_dup is not None:
        print(f"  {'平均重复片段率':<22} {avg_dup:.3f}")
    if avg_group_cov is not None:
        print(f"  {'平均意群覆盖率':<22} {avg_group_cov:.3f}")
    if avg_ref_pollution is not None:
        print(f"  {'平均参考污染率':<22} {avg_ref_pollution:.3f}")
    if avg_focus_ratio is not None:
        print(f"  {'Focus Mode 平均压缩率':<22} {avg_focus_ratio:.3f}")
    if avg_path_diversity is not None:
        print(f"  {'平均路径多样性':<22} {avg_path_diversity:.3f}")
    if avg_singleton_paths is not None:
        print(f"  {'平均孤立路径数':<22} {avg_singleton_paths:.1f}")

    if ragas_scores:
        print(f"\n  RAGAS 指标（满分 1.0）")
        print(f"  {'指标名称':<42} {'分数':>6}  {'进度'}")
        print("  " + "─" * 65)
        metric_stats = (ragas_scores.get("_meta") or {}).get("metric_stats", {})
        for key, val in ragas_scores.items():
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            label = _METRIC_LABELS.get(key, key)
            stats = metric_stats.get(key) or {}
            total_count = stats.get("total_count", 0)
            valid_count = stats.get("valid_count", 0)
            if total_count and valid_count != total_count:
                label = f"{label} [{valid_count}/{total_count}]"
            bar = _score_bar(score)
            print(f"  {label:<42} {score:.4f}  {bar}")
    else:
        print("\n  ⚠️  RAGAS 评估未运行（未配置 judge_api_key 或评估失败）")

    print(f"\n  各样本详情")
    print(f"  {'#':<4} {'问题':<34} {'回答字数':>7} {'片段数':>6} {'延迟ms':>8}  状态")
    print("  " + "─" * 66)
    for i, s in enumerate(samples):
        q = (s.question[:32] + "..") if len(s.question) > 34 else s.question
        status = "✗ " + (s.error[:25] if s.error else "") if s.error else "✓"
        print(
            f"  {i+1:<4} {q:<34} {len(s.answer):>7} {len(s.contexts):>6} "
            f"{s.latency_ms:>8.0f}  {status}"
        )
    print("=" * 72)


def save_results(
    samples: List[CollectedSample],
    ragas_scores: dict,
    output_path: str,
    run_config: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """保存评估结果到 JSON 文件"""
    valid_samples = [s for s in samples if not s.error]
    avg_dup = _average_diagnostic(valid_samples, "duplicate_chunk_ratio")
    avg_group_cov = _average_diagnostic(valid_samples, "unique_group_coverage")
    avg_ref_pollution = _average_diagnostic(valid_samples, "reference_pollution_ratio")
    avg_numeric_hit_quality = _average_diagnostic(valid_samples, "numeric_table_hit_quality")
    out = {
        "ragas_scores": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in ragas_scores.items()},
        "summary": {
            "total_samples": len(samples),
            "valid_samples": sum(1 for s in samples if not s.error),
            "error_samples": sum(1 for s in samples if s.error),
            "avg_contexts": (
                sum(len(s.contexts) for s in samples if not s.error)
                / max(sum(1 for s in samples if not s.error), 1)
            ),
            "avg_latency_ms": (
                sum(s.latency_ms for s in samples if not s.error)
                / max(sum(1 for s in samples if not s.error), 1)
            ),
            "avg_duplicate_chunk_ratio": avg_dup,
            "avg_unique_group_coverage": avg_group_cov,
            "avg_reference_pollution_ratio": avg_ref_pollution,
            "avg_numeric_table_hit_quality": avg_numeric_hit_quality,
            "avg_focus_mode_compression_ratio": _average_diagnostic(valid_samples, "focus_mode_avg_compression_ratio"),
            "avg_path_diversity_ratio": _average_diagnostic(valid_samples, "path_diversity_ratio"),
            "avg_singleton_path_count": _average_diagnostic(valid_samples, "singleton_path_count"),
            "avg_unique_path_count": _average_diagnostic(valid_samples, "unique_path_count"),
        },
        "samples": [
            {
                "index": i + 1,
                "question": s.question,
                "answer_preview": s.answer[:500] + "..." if len(s.answer) > 500 else s.answer,
                "answer_length": len(s.answer),
                "contexts_count": len(s.contexts),
                "contexts_preview": [c[:300] for c in s.contexts[:3]],
                "ground_truth": s.ground_truth,
                "question_type": s.question_type,
                "note": s.note,
                "request_overrides": s.request_overrides,
                "latency_ms": s.latency_ms,
                "error": s.error,
                "retrieval_diagnostics": s.retrieval_diagnostics,
            }
            for i, s in enumerate(samples)
        ],
    }
    if run_config:
        out["run_config"] = run_config
    if warnings:
        out["warnings"] = warnings
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存 → {output_path}")

    # 同时保存 CSV（方便 Excel 查看）
    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "问题", "题型", "难度", "答案字数", "片段数", "延迟ms", "重复率", "意群覆盖", "参考污染率", "Focus压缩率", "路径多样性", "孤立路径数", "错误", "ground_truth"])
        for i, s in enumerate(samples):
            diag = _extract_retrieval_diagnostics(s)
            writer.writerow([
                i + 1, s.question,
                getattr(s, 'question_type', '') or "",
                getattr(s, 'difficulty', '') or "",
                len(s.answer),
                len(s.contexts), s.latency_ms,
                diag.get("duplicate_chunk_ratio", ""),
                diag.get("unique_group_coverage", ""),
                diag.get("reference_pollution_ratio", ""),
                diag.get("focus_mode_avg_compression_ratio", ""),
                diag.get("path_diversity_ratio", ""),
                diag.get("singleton_path_count", ""),
                s.error or "", s.ground_truth or "",
            ])
        if ragas_scores:
            writer.writerow([])
            writer.writerow(["RAGAS指标", "分数"])
            for k, v in ragas_scores.items():
                if isinstance(v, (int, float)):
                    writer.writerow([_METRIC_LABELS.get(k, k), f"{float(v):.4f}"])
    logger.info(f"CSV 结果已保存  → {csv_path}")


# ─────────────────────────────── 主流程 ───────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    # ── 加载配置文件 ──
    config: dict = {}
    config_path = args.config or "tests/ragas_eval_config.json"
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"已加载配置文件: {config_path}")

    def _get(attr: str, default: Any = None) -> Any:
        """优先级：命令行参数 > 配置文件 > 环境变量 > 默认值"""
        cli_val = getattr(args, attr.replace("-", "_"), None)
        if cli_val is not None:
            return cli_val
        if attr in config:
            return config[attr]
        env_key = "RAGAS_" + attr.upper().replace("-", "_")
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        return default

    backend_url: str = _get("backend_url", "http://localhost:8000")
    doc_id: str = _get("doc_id", "")
    api_key: str = _get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    model: str = _get("model", "gpt-4o-mini")
    provider: str = _get("provider", "openai")
    api_host: str = _get("api_host", "")
    embedding_api_key: str = _get("embedding_api_key", "") or api_key
    top_k: int = int(_get("top_k", 10))
    candidate_k: int = int(_get("candidate_k", 20))
    use_rerank: bool = bool(_get("use_rerank", False))
    reranker_model: str = _get("reranker_model", "")
    rerank_provider: str = _get("rerank_provider", "")
    rerank_api_key: str = _get("rerank_api_key", "") or api_key
    rerank_endpoint: str = _get("rerank_endpoint", "")
    enable_jieba_bm25: bool = str(_get("enable_jieba_bm25", True)).lower() not in ("false", "0", "no")
    num_expand_context_chunk: int = int(_get("num_expand_context_chunk", 1))
    enable_memory: bool = str(_get("enable_memory", True)).lower() not in ("false", "0", "no")
    enable_glossary: bool = str(_get("enable_glossary", True)).lower() not in ("false", "0", "no")
    enable_graphrag: bool = str(_get("enable_graphrag", False)).lower() in ("true", "1", "yes")
    enable_agent_retrieval: bool = str(_get("enable_agent_retrieval", False)).lower() in ("true", "1", "yes")
    judge_model: str = _get("judge_model", "") or model
    judge_api_key: str = _get("judge_api_key", "") or api_key
    judge_api_base: str = _get("judge_api_base", "") or api_host
    embed_model: str = _get("embed_model", "text-embedding-3-small")
    embed_api_key: str = _get("embed_api_key", "") or judge_api_key
    embed_api_base: str = _get("embed_api_base", "") or judge_api_base
    output_path: str = _get("output", "tests/ragas_results.json")
    skip_ragas: bool = bool(_get("skip_ragas", False))

    # ── 加载问题 ──
    questions_data: List[Any] = config.get("questions") or []
    questions_file: str = _get("questions", "") or config.get("questions_file", "")
    if questions_file and Path(questions_file).exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        logger.info(f"已加载问题文件: {questions_file}（{len(questions_data)} 条）")

    if not questions_data:
        logger.error(
            "未找到测试问题！请通过以下方式之一提供：\n"
            "  1. --questions tests/ragas_sample_questions.json\n"
            "  2. 在配置文件 questions_file 字段中指定路径\n"
            "  3. 在配置文件 questions 数组中直接内嵌问题"
        )
        sys.exit(1)

    if not doc_id:
        logger.error(
            "未指定 doc_id！请通过 --doc-id 参数或配置文件 doc_id 字段指定。\n"
            "doc_id 是上传 PDF 后返回的文档 ID，可在前端设置中查看。"
        )
        sys.exit(1)

    # 解析问题列表
    question_items: List[QuestionItem] = []
    for item in questions_data:
        if isinstance(item, str):
            question_items.append(QuestionItem(question=item))
        elif isinstance(item, dict):
            _meta_keys = {"question", "ground_truth", "_type", "_note", "_facet", "_difficulty"}
            payload_overrides = {
                key: value
                for key, value in item.items()
                if key not in _meta_keys
            }
            question_items.append(QuestionItem(
                question=item["question"],
                ground_truth=item.get("ground_truth"),
                question_type=item.get("_type", ""),
                note=item.get("_note", ""),
                facet=item.get("_facet", ""),
                difficulty=item.get("_difficulty", ""),
                payload_overrides=payload_overrides,
            ))

    eval_warnings = _build_eval_warnings(question_items, embed_model)
    for warning in eval_warnings:
        logger.warning(f"[评估告警] {warning}")

    run_config = {
        "backend_url": backend_url,
        "doc_id": doc_id,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "use_rerank": use_rerank,
        "reranker_model": reranker_model,
        "rerank_provider": rerank_provider,
        "enable_jieba_bm25": enable_jieba_bm25,
        "num_expand_context_chunk": num_expand_context_chunk,
        "enable_memory": enable_memory,
        "enable_glossary": enable_glossary,
        "enable_graphrag": enable_graphrag,
        "enable_agent_retrieval": enable_agent_retrieval,
        "judge_model": judge_model,
        "embed_model": embed_model,
        "questions_count": len(question_items),
    }

    # ── 收集样本 ──
    client = ChatpdfClient(
        backend_url=backend_url,
        doc_id=doc_id,
        api_key=api_key,
        model=model,
        provider=provider,
        api_host=api_host,
        embedding_api_key=embedding_api_key,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        reranker_model=reranker_model,
        rerank_provider=rerank_provider,
        rerank_api_key=rerank_api_key,
        rerank_endpoint=rerank_endpoint,
        enable_jieba_bm25=enable_jieba_bm25,
        num_expand_context_chunk=num_expand_context_chunk,
        enable_memory=enable_memory,
        enable_glossary=enable_glossary,
        enable_graphrag=enable_graphrag,
        enable_agent_retrieval=enable_agent_retrieval,
    )

    rerank_info = f" | rerank={reranker_model or 'cross-encoder'}" if use_rerank else ""
    extra_info = f" | memory={enable_memory} | glossary={enable_glossary}"
    if enable_graphrag:
        extra_info += " | graphrag"
    if enable_agent_retrieval:
        extra_info += " | agent"
    logger.info(
        f"开始收集数据：{len(question_items)} 个问题 | "
        f"doc_id={doc_id} | model={provider}/{model} | "
        f"top_k={top_k} | bm25={enable_jieba_bm25} | expand={num_expand_context_chunk}{rerank_info}{extra_info}"
    )
    samples: List[CollectedSample] = []
    for i, qi in enumerate(question_items):
        logger.info(f"  [{i+1}/{len(question_items)}] {qi.question[:60]}")
        sample = await client.query(qi.question, payload_overrides=qi.payload_overrides)
        sample.ground_truth = qi.ground_truth
        sample.question_type = qi.question_type
        sample.note = qi.note
        sample.facet = qi.facet
        sample.difficulty = qi.difficulty
        sample.request_overrides = qi.payload_overrides
        samples.append(sample)
        if sample.error:
            logger.warning(f"    ✗ 错误: {sample.error}")
        else:
            logger.info(
                f"    ✓ 答案: {len(sample.answer)} 字 | "
                f"上下文片段: {len(sample.contexts)} 个 | "
                f"耗时: {sample.latency_ms:.0f}ms"
            )

    # ── RAGAS 评估 ──
    ragas_scores: dict = {}
    if skip_ragas:
        logger.info("--skip-ragas 已设置，跳过 RAGAS LLM 评估（仅收集数据）")
    elif not judge_api_key:
        logger.warning("未配置 judge_api_key，跳过 RAGAS LLM 评估（仅统计数据）")
    else:
        logger.info(
            f"开始 RAGAS 评估 | judge={judge_model} | "
            f"embed={embed_model}"
        )
        try:
            ragas_scores = run_ragas_evaluation(
                samples=samples,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_api_base=judge_api_base,
                embed_model=embed_model,
                embed_api_key=embed_api_key,
                embed_api_base=embed_api_base,
            )
            meta = ragas_scores.setdefault("_meta", {})
            if isinstance(meta, dict) and eval_warnings:
                meta["warnings"] = eval_warnings
        except ImportError as e:
            logger.error(f"依赖缺失: {e}")
        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}", exc_info=True)

    # ── 输出报告 ──
    print_summary_table(samples, ragas_scores)
    save_results(samples, ragas_scores, output_path, run_config=run_config, warnings=eval_warnings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 RAGAS 对 Chatpdf RAG 系统进行端到端质量评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用配置文件
  python -m tests.eval_ragas --config tests/ragas_eval_config.json

  # 命令行参数（仅收集数据，不运行 RAGAS LLM 评分）
  python -m tests.eval_ragas --doc-id abc123 --api-key sk-xxx \\
    --model gpt-4o-mini --provider openai --skip-ragas
        """,
    )
    parser.add_argument("--config", help="配置文件路径（JSON）")
    parser.add_argument("--backend-url", dest="backend_url", help="Chatpdf 后端地址")
    parser.add_argument("--doc-id", dest="doc_id", help="文档 ID")
    parser.add_argument("--api-key", dest="api_key", help="LLM API Key")
    parser.add_argument("--model", help="LLM 模型名称（如 gpt-4o-mini）")
    parser.add_argument("--provider", help="LLM 提供商（如 openai）")
    parser.add_argument("--api-host", dest="api_host", help="自定义 API Host / Base URL")
    parser.add_argument("--embedding-api-key", dest="embedding_api_key", help="Embedding API Key")
    parser.add_argument("--top-k", dest="top_k", type=int, help="检索 top-k（默认 10）")
    parser.add_argument("--use-rerank", dest="use_rerank", action="store_true", help="启用重排序")
    parser.add_argument("--questions", help="问题文件路径（JSON 数组）")
    parser.add_argument("--judge-model", dest="judge_model", help="RAGAS LLM judge 模型")
    parser.add_argument("--judge-api-key", dest="judge_api_key", help="RAGAS judge API Key")
    parser.add_argument("--judge-api-base", dest="judge_api_base", help="RAGAS judge API Base URL")
    parser.add_argument("--embed-model", dest="embed_model", help="RAGAS embedding 模型")
    parser.add_argument("--embed-api-key", dest="embed_api_key", help="RAGAS embedding API Key")
    parser.add_argument("--embed-api-base", dest="embed_api_base", help="RAGAS embedding API Base URL")
    parser.add_argument("--output", help="结果输出路径（JSON，默认 tests/ragas_results.json）")
    parser.add_argument("--skip-ragas", dest="skip_ragas", action="store_true",
                        help="仅收集数据，跳过 RAGAS LLM 评分（节省 API 费用）")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
