"""
P3.7 树式递归查询分解检索（Tree Structured Query Decomposition Retrieval）
借鉴 ragflow rag/advanced_rag/tree_structured_query_decomposition_retrieval.py

核心流程：
1. 用初始 query 检索一次
2. 用 LLM 做 sufficiency_check：当前内容是否足够回答问题？
3. 不足 → 调用 multi_queries_gen 生成 K 个子查询
4. 并发递归子查询，深度上限 max_depth
5. 合并所有 chunks，去重返回

设计：
- max_depth=2（保守，避免深度爆炸）
- 每层 timeout 30s
- sufficiency_check JSON 解析失败 → 默认 sufficient=true（不递归）
- gate 严格：仅在 query_type ∈ {analytical, overview, comparison} 且首轮 chunks < 5 时触发
"""
import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _chunk_merge_key(ch: dict) -> str:
    """Passage identity for recursive merge. chunk_id=0 is valid."""
    chunk_id = ch.get("chunk_id")
    if chunk_id is None or isinstance(chunk_id, bool) or chunk_id == "":
        return str(ch.get("chunk") or ch.get("text") or "")[:100]
    return f"chunk:{chunk_id}"


# ragflow sufficiency_check.md 中文版
_SUFFICIENCY_CHECK_PROMPT = """你是一个严谨的研究助手。请判断给定的"已检索内容"是否足以全面回答用户的问题。

## 用户问题
{question}

## 已检索内容
{context}

## 输出要求
仅输出 JSON，不要包含任何额外文字：
```json
{{
  "is_sufficient": true | false,
  "missing_information": ["缺失信息1", "缺失信息2", ...],
  "confidence": 0.0-1.0
}}
```

## 评估标准
- is_sufficient=true：内容覆盖问题的核心方面，足以给出可信回答
- is_sufficient=false：缺少关键事实/数据/对比/解释，需要进一步检索
- missing_information：列出 1-3 个具体缺口，例如「方法 X 的实验数据」「与 baseline 的具体对比」
- confidence：你对此判断的置信度

只输出 JSON，禁止解释。"""

# multi-queries gen 提示词（基于缺失信息生成子查询）
_MULTI_QUERIES_GEN_PROMPT = """根据用户问题以及已知的"缺失信息"，生成 {k} 个子查询，分别针对每个缺失方面。

## 用户问题
{question}

## 缺失信息
{missing}

## 输出
每行输出一个子查询，不超过 30 字，不加编号或前缀。"""


_DEFAULT_MAX_DEPTH = 2
_DEFAULT_K_PER_LEVEL = 3
_DEFAULT_TIMEOUT_PER_LEVEL = 30.0


# 触发条件 gate
_TREE_DECOMPOSE_QUERY_TYPES = {"analytical", "overview", "comparison"}
_TREE_DECOMPOSE_MIN_INITIAL_CHUNKS = 5


def should_use_tree_decomposition(
    *,
    query_type: str,
    initial_chunk_count: int,
    enabled: bool,
) -> Tuple[bool, str]:
    """判断是否应启用树式递归分解。

    Returns:
        (should_use, gate_reason)
    """
    if not enabled:
        return False, "disabled"
    if query_type not in _TREE_DECOMPOSE_QUERY_TYPES:
        return False, f"query_type_blocked:{query_type}"
    if initial_chunk_count >= _TREE_DECOMPOSE_MIN_INITIAL_CHUNKS:
        return False, f"sufficient_initial:{initial_chunk_count}"
    return True, f"matched:{query_type}_chunks={initial_chunk_count}"


async def sufficiency_check(
    *,
    question: str,
    context: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """LLM 判断当前 context 是否足以回答问题。

    Returns:
        {"is_sufficient": bool, "missing_information": [str], "confidence": float, "error": str?}
    """
    default = {
        # 最大深度已经提供硬停止条件。判定失败时不能反向宣称证据充足，
        # 否则输出上限恰好会让证据最少的请求提前停止。
        "is_sufficient": False,
        "missing_information": [question] if question else [],
        "confidence": 0.0,
        "error": "",
    }
    if not question or not context:
        default["error"] = "empty_input"
        return default

    try:
        from services.chat_service import call_ai_api
        from services.completion_outcome import require_publishable_completion

        # 截断 context 避免 prompt 过长
        ctx_for_check = context[:8000] if len(context) > 8000 else context

        prompt = _SUFFICIENCY_CHECK_PROMPT.format(
            question=question, context=ctx_for_check
        )
        response = await asyncio.wait_for(
            call_ai_api(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=400,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        require_publishable_completion(response, operation="tree sufficiency check")

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                default["error"] = str(response.get("error"))
                return default
            content = response.get("content") or ""
            if not content and "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content", "")
        else:
            content = str(response or "")

        # 解析 JSON
        parsed = _parse_json_response(content)
        if parsed is None:
            default["error"] = "json_parse_failed"
            return default

        return {
            "is_sufficient": bool(parsed.get("is_sufficient", True)),
            "missing_information": list(parsed.get("missing_information", []))[:5],
            "confidence": float(parsed.get("confidence", 0.5) or 0.5),
            "error": "",
        }

    except asyncio.TimeoutError:
        default["error"] = "timeout"
        return default
    except Exception as e:
        default["error"] = f"exception:{str(e)[:120]}"
        return default


async def gen_sub_queries(
    *,
    question: str,
    missing_information: List[str],
    k: int,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    timeout: float = 15.0,
) -> List[str]:
    """基于缺失信息生成 k 个子查询。"""
    if not missing_information:
        return []

    try:
        from services.chat_service import call_ai_api
        from services.completion_outcome import require_publishable_completion
        from services.intent_constraints import IntentConstraintSet

        prompt = _MULTI_QUERIES_GEN_PROMPT.format(
            question=question,
            missing="；".join(missing_information[:3]),
            k=k,
        )
        response = await asyncio.wait_for(
            call_ai_api(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=200,
                temperature=0.5,
            ),
            timeout=timeout,
        )
        require_publishable_completion(response, operation="tree sub-query generation")

        content = ""
        if isinstance(response, dict):
            content = response.get("content") or ""
            if not content and "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content", "")
        else:
            content = str(response or "")

        if not content:
            return []

        sub_queries = []
        for line in content.strip().splitlines():
            line = line.strip()
            line = re.sub(r"^[\d\.\-\)、]+\s*", "", line)
            line = line.strip("\"'`")
            if line and len(line) > 3 and line != question:
                sub_queries.append(line)
        sub_queries = sub_queries[:k]
        validation = IntentConstraintSet.from_text(question).validate_subquestions(sub_queries)
        if not validation.allowed:
            logger.warning(
                "[TreeDecompose] 子查询违反原始意图约束，已丢弃: %s",
                validation.violations,
            )
            return []
        return sub_queries

    except Exception as e:
        logger.warning(f"[TreeDecompose] sub-query 生成失败: {e}")
        return []


async def research(
    *,
    question: str,
    retrieve_fn: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    max_depth: int = _DEFAULT_MAX_DEPTH,
    k_per_level: int = _DEFAULT_K_PER_LEVEL,
    timeout_per_level: float = _DEFAULT_TIMEOUT_PER_LEVEL,
    _depth: int = 0,
    _trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """递归检索：每层做 sufficiency_check，不足时分解为子查询并行递归。

    Args:
        question: 当前问题（顶层为用户原始问题）
        retrieve_fn: 异步检索函数，输入 query 字符串，返回 chunk 列表
        api_key/model/provider/endpoint: LLM 配置
        max_depth: 最大递归深度
        k_per_level: 每层生成的子查询数
        timeout_per_level: 每层 sufficiency_check 总超时

    Returns:
        (merged_chunks, diagnostics)
    """
    if _trace is None:
        _trace = []

    diagnostics: Dict[str, Any] = {
        "trace": _trace,
        "max_depth": max_depth,
        "total_queries": 0,
        "total_chunks": 0,
        "stopped_reason": "",
    }

    if _depth >= max_depth:
        diagnostics["stopped_reason"] = "max_depth_reached"
        return [], diagnostics

    started = time.perf_counter()
    try:
        chunks = await asyncio.wait_for(retrieve_fn(question), timeout=timeout_per_level)
    except asyncio.TimeoutError:
        diagnostics["stopped_reason"] = "retrieve_timeout"
        return [], diagnostics
    except Exception as e:
        diagnostics["stopped_reason"] = f"retrieve_error:{str(e)[:60]}"
        return [], diagnostics

    chunks = chunks or []
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    _trace.append({
        "depth": _depth,
        "question": question[:80],
        "chunk_count": len(chunks),
        "elapsed_ms": elapsed_ms,
    })

    # 顶层若已检索丰富，不再递归
    if _depth + 1 >= max_depth:
        diagnostics["total_chunks"] = len(chunks)
        diagnostics["total_queries"] = 1
        diagnostics["stopped_reason"] = "max_depth_reached"
        return chunks, diagnostics

    # 构造 context 字符串做充分性判断
    context_text = "\n\n".join(
        (ch.get("chunk") or ch.get("text") or "")[:1500]
        for ch in chunks[:5]
        if isinstance(ch, dict)
    ).strip()

    if not context_text:
        # 当前层无内容，可继续递归
        suff = {"is_sufficient": False, "missing_information": [question]}
    else:
        suff = await sufficiency_check(
            question=question,
            context=context_text,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            timeout=timeout_per_level / 2,
        )

    if suff.get("is_sufficient", True):
        diagnostics["total_chunks"] = len(chunks)
        diagnostics["total_queries"] = 1
        diagnostics["stopped_reason"] = "sufficient"
        return chunks, diagnostics

    missing = suff.get("missing_information", [])
    sub_queries = await gen_sub_queries(
        question=question,
        missing_information=missing,
        k=k_per_level,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        timeout=timeout_per_level / 2,
    )

    if not sub_queries:
        diagnostics["total_chunks"] = len(chunks)
        diagnostics["total_queries"] = 1
        diagnostics["stopped_reason"] = "no_sub_queries"
        return chunks, diagnostics

    # 并行递归子查询
    child_results = await asyncio.gather(
        *[
            research(
                question=sq,
                retrieve_fn=retrieve_fn,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_depth=max_depth,
                k_per_level=k_per_level,
                timeout_per_level=timeout_per_level,
                _depth=_depth + 1,
                _trace=_trace,
            )
            for sq in sub_queries
        ],
        return_exceptions=True,
    )

    # 合并 chunks（按 chunk_id 或 text 前缀去重）
    merged = list(chunks)
    seen_keys: set = set()
    for ch in chunks:
        if isinstance(ch, dict):
            key = _chunk_merge_key(ch)
            if key:
                seen_keys.add(key)

    for cr in child_results:
        if isinstance(cr, Exception):
            continue
        if not isinstance(cr, tuple) or len(cr) != 2:
            continue
        child_chunks, _child_diag = cr
        for ch in (child_chunks or []):
            if not isinstance(ch, dict):
                continue
            key = _chunk_merge_key(ch)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(ch)

    diagnostics["total_chunks"] = len(merged)
    diagnostics["total_queries"] = 1 + len(sub_queries)
    diagnostics["stopped_reason"] = "merged_with_children"
    diagnostics["sub_queries"] = sub_queries
    diagnostics["missing_information"] = missing
    return merged, diagnostics


def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中解析 JSON（含 markdown code block 兼容）"""
    if not content:
        return None
    content = content.strip()

    # 直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # markdown code block
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 第一对 { }
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(content[first:last + 1])
        except json.JSONDecodeError:
            pass

    return None
