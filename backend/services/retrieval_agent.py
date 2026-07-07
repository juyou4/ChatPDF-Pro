"""
多轮 Agent 式检索服务

参考 paper-burner-x 的 streaming-multi-hop 架构，实现：
- LLM 作为"检索规划助手"，不回答问题，只规划检索策略
- 多轮迭代：每轮执行搜索→评估结果→决定是否需要更多信息
- 丰富的工具集：vector_search, grep, keyword_search, regex_search, boolean_search, fetch, map
- 搜索历史去重，避免重复查询
- 任务追踪 (taskStatus)
- 流式进度反馈
"""

import asyncio
import json
import logging
import math
import hashlib
import re
import time
from collections import Counter
from typing import Any, AsyncGenerator, Dict, List, Optional

from config import settings
from services.chat_service import call_ai_api
from services.formula_text import build_formula_alias_text, looks_formula_like, normalize_formula_text, technical_anchor_matches
from services.query_analyzer import (
    analyze_evidence_need,
    analyze_query_type,
    extract_document_bilingual_terms,
    extract_hl_ll_terms,
)
from services.retrieval_tool_schemas import TOOL_SCHEMAS
from services.retrieval_tools import DocContext, execute_tool
from utils.middleware import RetryMiddleware

logger = logging.getLogger(__name__)


_TOOL_RESULT_EVIDENCE_FIELDS = (
    "text",
    "content",
    "context_segment_text",
    "numeric_table_exact_context_row_text",
    "table_row_boundary_text",
    "table_row_raw_text",
    "source_text",
    "display_text",
    "highlight_text",
    "chunk",
    "child_chunk",
    "raw_chunk_text",
    "summary",
)

_TOOL_RESULT_PROVENANCE_FIELDS = (
    ("source", "source"),
    ("page", "页码"),
    ("page_number", "页码"),
    ("group_id", "group_id"),
    ("context_id", "context_id"),
    ("evidence_id", "evidence_id"),
    ("chunk_id", "chunk_id"),
    ("child_chunk_id", "child_chunk_id"),
    ("parent_id", "parent_id"),
    ("chunk_type", "chunk_type"),
    ("table_id", "table_id"),
    ("table_bundle_id", "table_bundle_id"),
    ("evidence_unit_id", "evidence_unit_id"),
)


def _stringify_tool_result_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts: list[str] = []
        seen: set[str] = set()
        for key in _TOOL_RESULT_EVIDENCE_FIELDS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = re.sub(r"\s+", " ", value).strip()
                dedupe_key = text.lower()
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        try:
            return json.dumps(item, ensure_ascii=False)
        except Exception:
            return str(item)
    return str(item or "")


def _format_page_range_value(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        try:
            start = int(value[0])
            end = int(value[-1] if len(value) > 1 else value[0])
        except (TypeError, ValueError):
            return ""
        if start <= 0 and end <= 0:
            return ""
        if start <= 0:
            start = end
        if end <= 0:
            end = start
        return str(start) if start == end else f"{start}-{end}"
    try:
        page = int(value)
    except (TypeError, ValueError):
        return ""
    return str(page) if page > 0 else ""


def _format_tool_result_with_provenance(item: Any, text: str) -> str:
    if not isinstance(item, dict) or not text:
        return text
    fields: list[str] = []
    seen_labels: set[str] = set()
    page_range_text = _format_page_range_value(item.get("page_range"))
    if page_range_text:
        fields.append(f"页码:{page_range_text}")
        seen_labels.add("页码")
    for key, label in _TOOL_RESULT_PROVENANCE_FIELDS:
        value = item.get(key)
        if value in (None, "") or label in seen_labels:
            continue
        if isinstance(value, (list, dict)):
            continue
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        if not normalized:
            continue
        fields.append(f"{label}:{normalized}")
        seen_labels.add(label)
    if not fields:
        return text
    return f"【检索证据 | {' | '.join(fields)}】\n{text}"


def _dedupe_terms(items: list) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items or []:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _extract_formula_search_terms(question: str, max_terms: int = 10) -> list[str]:
    """Extract generic math aliases from a question for exact-search tools."""
    if not looks_formula_like(question):
        return []
    normalized = normalize_formula_text(question)
    alias_text = build_formula_alias_text(question)
    candidates: list[str] = []
    for source in (normalized, alias_text):
        candidates.extend(
            re.findall(
                r"\b[a-z]+(?:_bar)?(?:_[a-z0-9]+)+\b|\bsqrt\b|\b[a-z]\^[0-9]+\b|"
                r"\b(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)(?:_bar)?\b",
                source,
                re.IGNORECASE,
            )
        )
    expanded: list[str] = []
    for term in _dedupe_terms(candidates):
        expanded.append(term)
        if "_" in term:
            expanded.append(term.replace("_", " "))
            expanded.append(term.replace("_", ""))
    return _dedupe_terms(expanded)[:max_terms]


_QUESTION_ANCHOR_STOPWORDS = {
    "about", "above", "answer", "are", "based", "does", "from", "how", "main",
    "method", "paper", "problem", "proposed", "result", "results", "that", "the",
    "their", "this", "uses", "what", "when", "where", "which", "why", "什么", "哪些",
    "如何", "论文", "方法", "主要", "问题", "区别", "请", "解释", "说明",
}


def _extract_question_anchor_terms(question: str, max_terms: int = 24) -> list[str]:
    """Extract question-side technical anchors for generic sufficiency checks."""
    text = str(question or "")
    candidates = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}",
        text,
    )
    candidates.extend(_extract_formula_search_terms(text, max_terms=12))
    anchors: list[str] = []
    for term in _dedupe_terms(candidates):
        clean = term.strip(" .,;:()[]{}，。；：、")
        lower = clean.casefold()
        if not clean or lower in _QUESTION_ANCHOR_STOPWORDS:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?%?", clean):
            anchors.append(clean)
            continue
        if any(ch.isdigit() for ch in clean) or "_" in clean or "-" in clean or any(ch.isupper() for ch in clean[1:]):
            anchors.append(clean)
            continue
        if len(clean) >= 4 and not re.fullmatch(r"[\u4e00-\u9fff]+", clean):
            anchors.append(clean)
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", clean):
            anchors.append(clean)
    return _dedupe_terms(anchors)[:max_terms]


def _question_anchor_coverage(question: str, evidence_text: str) -> dict:
    anchors = _extract_question_anchor_terms(question)
    if not anchors:
        return {"anchors": [], "matched": [], "missing": [], "coverage": 1.0, "required": False}
    evidence_lower = str(evidence_text or "").casefold()
    matched: list[str] = []
    for anchor in anchors:
        if technical_anchor_matches(anchor, evidence_text):
            matched.append(anchor)
    missing = [anchor for anchor in anchors if anchor not in set(matched)]
    coverage = len(matched) / max(len(anchors), 1)
    return {
        "anchors": anchors,
        "matched": matched,
        "missing": missing,
        "coverage": round(coverage, 4),
        "required": len(anchors) >= 2,
    }


def _sub_question_evidence_coverage(sub_questions: List[str] | None, evidence_text: str) -> dict:
    """Check whether retrieved evidence covers decomposed sub-question anchors."""
    items: list[dict[str, Any]] = []
    required_count = 0
    covered_count = 0
    for idx, sub_question in enumerate(sub_questions or []):
        text = re.sub(r"\s+", " ", str(sub_question or "")).strip()
        if not text:
            continue
        anchors = _extract_question_anchor_terms(text, max_terms=8)
        if not anchors:
            items.append({
                "index": idx,
                "sub_question": text,
                "anchors": [],
                "matched": [],
                "missing": [],
                "required_matches": 0,
                "covered": True,
                "required": False,
            })
            continue
        matched = [
            anchor
            for anchor in anchors
            if technical_anchor_matches(anchor, evidence_text)
        ]
        required_matches = 1 if len(anchors) <= 2 else min(2, max(1, math.ceil(len(anchors) * 0.5)))
        covered = len(matched) >= required_matches
        required_count += 1
        if covered:
            covered_count += 1
        matched_set = set(matched)
        items.append({
            "index": idx,
            "sub_question": text,
            "anchors": anchors,
            "matched": matched,
            "missing": [anchor for anchor in anchors if anchor not in matched_set],
            "required_matches": required_matches,
            "covered": covered,
            "required": True,
        })
    coverage = covered_count / max(required_count, 1)
    return {
        "required": required_count > 0,
        "required_count": required_count,
        "covered_count": covered_count,
        "coverage": round(coverage, 4),
        "uncovered": [
            item["sub_question"]
            for item in items
            if item.get("required") and not item.get("covered")
        ],
        "items": items,
    }


def _evidence_independence_key(meta: Dict[str, Any], text: str, *, fallback_scope: str = "") -> str:
    """Return a generic source-level key for sufficiency independence checks.

    Strong evidence ids represent a concrete unit and can stand alone. Coarser
    provenance such as context/group/page/table is intentionally treated as one
    source, so repeated windows from the same source do not trigger early stop.
    """
    if not isinstance(meta, dict):
        meta = {}
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    for field in ("evidence_id", "chunk_id", "child_chunk_id"):
        value = re.sub(r"\s+", " ", str(meta.get(field) or "")).strip().casefold()
        if value:
            return f"{field}:{value}"
    scoped_parts: list[str] = []
    for field in ("context_id", "parent_id", "group_id", "table_bundle_id", "table_id"):
        value = re.sub(r"\s+", " ", str(meta.get(field) or "")).strip().casefold()
        if value:
            scoped_parts.append(f"{field}:{value}")
    if scoped_parts:
        return "|".join(scoped_parts)
    if normalized:
        text_digest = hashlib.sha1(normalized[:1200].encode("utf-8", errors="ignore")).hexdigest()
        return f"text-exact:{text_digest}"
    page_range = meta.get("page_range")
    if isinstance(page_range, list) and page_range:
        return f"page:{page_range[0]}-{page_range[-1]}"
    if meta.get("page"):
        return f"page:{meta.get('page')}"
    digest = hashlib.sha1(normalized[:600].encode("utf-8", errors="ignore")).hexdigest()
    return f"text:{fallback_scope}:{digest}"


def _context_part_dedupe_key(meta: Dict[str, Any], text: str) -> str:
    """Return a provenance-aware key for final-context duplicate removal."""
    if not isinstance(meta, dict):
        meta = {}
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    prefix = normalized[:160]
    suffix = normalized[-160:] if len(normalized) > 160 else ""
    digest = hashlib.blake2b(normalized.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    text_fp = f"{len(normalized)}:{prefix}:{suffix}:{digest}"
    for field in ("evidence_id", "chunk_id", "child_chunk_id"):
        value = re.sub(r"\s+", " ", str(meta.get(field) or "")).strip().casefold()
        if value:
            return f"id:{field}:{value}"
    scoped_parts: list[str] = []
    for field in ("context_id", "parent_id", "group_id", "table_bundle_id", "table_id"):
        value = re.sub(r"\s+", " ", str(meta.get(field) or "")).strip().casefold()
        if value:
            scoped_parts.append(f"{field}:{value}")
    if scoped_parts:
        return f"scoped:{'|'.join(scoped_parts)}:{text_fp}"
    return f"text:{text_fp}"


def _tool_result_dedupe_key(item: Any, text: str) -> str:
    """Return a generic key for merging repeated tool hits before final budgeting."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    prefix = normalized[:160]
    suffix = normalized[-160:] if len(normalized) > 160 else ""
    digest = hashlib.blake2b(normalized.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    text_fp = f"{len(normalized)}:{prefix}:{suffix}:{digest}"
    if not isinstance(item, dict):
        return f"text:{text_fp}"

    for field in ("evidence_id", "chunk_id", "child_chunk_id"):
        value = re.sub(r"\s+", " ", str(item.get(field) or "")).strip().casefold()
        if value:
            return f"id:{field}:{value}"
    scoped_parts: list[str] = []
    for field in ("context_id", "parent_id", "group_id", "table_bundle_id", "table_id", "evidence_unit_id"):
        value = re.sub(r"\s+", " ", str(item.get(field) or "")).strip().casefold()
        if value:
            scoped_parts.append(f"{field}:{value}")
    if scoped_parts:
        return f"scoped:{'|'.join(scoped_parts)}:{text_fp}"
    return f"text:{text_fp}"


def _merge_uncovered_sub_questions(
    sub_questions: List[str] | None,
    query_uncovered: List[str] | None,
    evidence_uncovered: List[str] | None,
) -> list[str]:
    """Merge query-planning and evidence-coverage gaps in sub-question order."""
    if not sub_questions:
        return []
    gap_keys = {
        re.sub(r"\s+", " ", str(item or "")).strip().casefold()
        for item in [*(query_uncovered or []), *(evidence_uncovered or [])]
        if str(item or "").strip()
    }
    merged: list[str] = []
    for sub_question in sub_questions:
        text = re.sub(r"\s+", " ", str(sub_question or "")).strip()
        if text and text.casefold() in gap_keys:
            merged.append(text)
    return merged

# Agent 系统提示词模板（v2 精简版 - 参考 paper-burner-x ReAct v2.0）
# 设计原则：信任 LLM 判断、删除冗余规则、保留工具定义和 JSON 格式
_AGENT_SYSTEM_PROMPT = """你是文档检索规划助手。任务：分析用户问题 → 选工具 → 输出 JSON 检索计划。**不回答用户问题**，仅规划检索。

## 可用工具

- `vector_search(query, limit=10)` 语义搜索（同义词/相关概念）
- `grep(query, limit=20, context=2000, caseInsensitive=true)` 字面搜索；query 用 `|` 分隔表 OR
- `keyword_search(keywords=[...], limit=8)` BM25 多关键词加权
- `regex_search(pattern, limit=10, context=1500)` 正则匹配
- `boolean_search(query, limit=10, context=1500)` 布尔逻辑：AND/OR/NOT
{group_tools}
## 策略
- vector_search 擅长语义、grep 擅长精确，**复杂问题并发组合**
- 检查【搜索历史】避免重复搜索
- 内容足够即 `final: true`；每轮最多 5 个操作

## 输出（严格 JSON）
```json
{{
  "operations": [{{"tool":"...","args":{{...}}}}],
  "final": true | false,
  "taskStatus": {{"completed": [...], "current": "...", "pending": [...]}}
}}
```"""

_GROUP_TOOLS_TEMPLATE = """- `fetch(groupId)` 获取意群完整内容（论述/公式/数据）
- `map(limit=50, includeStructure=true)` 文档结构概览（意群 ID/字数/关键词/摘要/结构）
"""

# ---------------------------------------------------------------------------
# Planner_Hint 动态状态提示（参考 paper-burner-x react/engine 的"动态警告块"）
# 在每轮 user message 中按需注入，引导 Planner_LLM 避免：
#   - 首轮直接 final=true（缺乏检索）
#   - 重复搜索（与上一轮相同的工具+参数）
#   - 空结果回路（上轮检索 0 命中却仍在原方向继续）
#   - 过度检索（信息已充足时仍未终止）
#   - 最终轮未设置 final=true
# 所有提示均使用中文，便于和 prompt 中的中文上下文一致。
# ---------------------------------------------------------------------------
_HINT_FIRST_ROUND = "🚨 首轮必须至少调用一个检索工具，禁止直接 final=true"
_HINT_DUPLICATE = "⚠️ 检测到与上轮相同的搜索（工具+参数），请更换工具或换关键词"
_HINT_EMPTY = "💡 上轮检索无结果，请尝试不同的关键词或更宽泛的查询"
_HINT_UNCOVERED_SUBQUESTIONS_PREFIX = "🧭 仍有子问题未覆盖，请优先补检索："
_HINT_SUFFICIENT = "✓ 信息可能已充足，若无明显空缺可考虑 final=true"
_HINT_FINAL_ROUND = "🚨 已是最终轮，必须设置 final=true"


def _format_uncovered_subquestion_hint(uncovered_sub_questions: List[str] | None = None) -> str:
    items = [re.sub(r"\s+", " ", str(item or "")).strip() for item in (uncovered_sub_questions or [])]
    items = [item for item in items if item]
    if not items:
        return ""
    preview = []
    for item in items[:3]:
        preview.append(item[:90] + ("..." if len(item) > 90 else ""))
    more = len(items) - len(preview)
    suffix = f"；另有 {more} 条" if more > 0 else ""
    return f"{_HINT_UNCOVERED_SUBQUESTIONS_PREFIX}{'；'.join(preview)}{suffix}"


def _compute_planner_hints(
    *,
    round_idx: int,
    max_rounds: int,
    last_round_calls: List[Dict[str, Any]],
    last_round_total_hits: int,
    duplicate_detected: bool,
    sufficiency_level: str,
    uncovered_sub_questions: List[str] | None = None,
) -> List[str]:
    """根据当前轮状态生成 Planner_Hint 列表

    优先级（列表首位为最高优先级）：
        final > duplicate > empty > sufficient > first_round

    触发条件：
    - final:       round_idx == max_rounds - 1（最终轮，必须收尾）
    - duplicate:   duplicate_detected=True（与上轮工具+参数相同）
    - empty:       round_idx > 0 且 last_round_total_hits == 0 且 last_round_calls 非空
                   （仅当上轮真的执行了工具但 0 命中时才提示）
    - sufficient:  sufficiency_level == "sufficient"（充足度评估命中）
    - first_round: round_idx == 0 且非最终轮（避免与 final 同时出现）

    全部条件都不满足时返回空列表，调用方应跳过【动态提示】区块的注入。

    Args:
        round_idx: 当前轮次索引（0-based）
        max_rounds: 总轮数上限
        last_round_calls: 上一轮的工具调用记录列表
        last_round_total_hits: 上一轮所有工具命中总数
        duplicate_detected: 是否检测到与上轮重复的搜索
        sufficiency_level: _assess_sufficiency 返回的充足度级别字符串

    Returns:
        按优先级排序的 hint 字符串列表（首位优先级最高）
    """
    hints: List[str] = []
    is_final = round_idx == max_rounds - 1

    # 优先级 1：最终轮提示（最高）
    if is_final:
        hints.append(_HINT_FINAL_ROUND)

    # 优先级 2：重复搜索提示
    if duplicate_detected:
        hints.append(_HINT_DUPLICATE)

    # 优先级 3：空结果提示（仅当上轮确实跑过工具但没命中时才适用）
    if round_idx > 0 and last_round_total_hits == 0 and last_round_calls:
        hints.append(_HINT_EMPTY)

    # 优先级 4：子问题覆盖缺口提示
    uncovered_hint = _format_uncovered_subquestion_hint(uncovered_sub_questions)
    if round_idx > 0 and uncovered_hint:
        hints.append(uncovered_hint)

    # 优先级 5：信息充足提示
    if sufficiency_level == "sufficient":
        hints.append(_HINT_SUFFICIENT)

    # 优先级 6：首轮提示（仅在非最终轮触发，避免与 final 冲突）
    if round_idx == 0 and not is_final:
        hints.append(_HINT_FIRST_ROUND)

    return hints


class RetrievalAgent:
    """多轮检索规划 Agent"""

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str,
        endpoint: str = "",
        max_rounds: int = 3,  # P1.4 优化：5→3，实测 2-3 轮足够，减少 planner 串行延迟
        temperature: float = 0.3,
        planner_retries: int = 1,
        max_context_tokens: int = 12000,
        max_iterations: int = 3,
        max_tool_calls: int = 12,
        context_compress_threshold: int = 16000,
        max_tool_concurrency: int = 5,
        sub_questions: Optional[List[str]] = None,
        use_rerank: bool = False,
        reranker_model: str = "",
        rerank_provider: str = "",
        rerank_api_key: str = "",
        rerank_endpoint: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.endpoint = endpoint
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.planner_retries = max(0, int(planner_retries or 0))
        self.max_context_tokens = max(500, int(max_context_tokens or 12000))
        self.max_iterations = max(1, min(int(max_iterations or max_rounds or 3), 10))
        self.max_tool_calls = max(1, int(max_tool_calls or 12))
        self.context_compress_threshold = max(0, int(context_compress_threshold or 0))
        self.max_tool_concurrency = max(1, min(int(max_tool_concurrency or 5), 5))
        self.sufficiency_threshold_chars: int = 2000
        self.sufficiency_min_sources: int = 2
        self.sub_questions: Optional[List[str]] = sub_questions  # 由 decompose 拆分的子问题列表
        self.backfilled_groups: set = set()  # 跨轮去重：已回填的 group_id 集合
        self.use_rerank = bool(use_rerank)
        self.reranker_model = reranker_model or ""
        self.rerank_provider = (rerank_provider or "").strip().lower().replace("siliconflow", "silicon")
        self.rerank_api_key = rerank_api_key or ""
        self.rerank_endpoint = rerank_endpoint or ""
        self.diagnostics: Dict[str, Any] = {}
        # P1: partial state 引用，便于 agent_total_timeout 时由外层调 snapshot_partial_diagnostics()
        # 直接拿到已积累的 candidate_pool/search_history，避免 candidate_gap 误归 no_candidate_pool_trace
        self._partial_state: Dict[str, Any] = {
            "search_history": [],
            "search_results": [],
            "fetched_content": {},
            "detail": [],
            "context_text": "",
            "context_budget": {},
            "external_rerank_cache": {},
        }
        # P4: 用于 child→parent chunk 扩展，命中 chunk 所属 group 的所有兄弟 chunk_idx
        # 也加入 candidate_pool（扩大 ID 维度召回，对应 candidate_pool_exact_id_gap）
        self._doc_ctx: Optional[DocContext] = None
        self._group_chunk_map: Optional[Dict[str, List[int]]] = None

    def _external_rerank_skip_reason(self) -> str:
        if not self.use_rerank:
            return "disabled"
        if not bool(getattr(settings, "agent_external_rerank_enabled", False)):
            return "disabled_by_flag"
        provider = self.rerank_provider or "local"
        cloud_providers = {"cohere", "jina", "silicon", "aliyun", "openai", "moonshot", "deepseek", "zhipu", "minimax", "llm"}
        if provider in cloud_providers and not self.rerank_api_key:
            return "provider_disabled_no_key"
        return ""

    def _should_apply_external_rerank(self) -> bool:
        return self._external_rerank_skip_reason() == ""

    @staticmethod
    def _extract_tool_chunk_meta(chunk: str) -> Dict[str, Any]:
        if not isinstance(chunk, str):
            return {"text": "", "rerank_text": ""}
        lines = chunk.splitlines()
        header = lines[0] if lines else ""
        body = "\n".join(lines[1:]).strip() if header.startswith("【检索证据") else chunk.strip()
        meta: Dict[str, Any] = {"text": body, "rerank_text": body}
        if not header.startswith("【检索证据"):
            return meta
        for match in re.finditer(r"([A-Za-z_\u4e00-\u9fff]+)[:：]([^|】]+)", header):
            key = str(match.group(1) or "").strip().lower()
            value = str(match.group(2) or "").strip()
            if not key or not value:
                continue
            if key == "页码":
                page_match = re.match(r"^\s*(\d+)(?:\s*[-~－—]\s*(\d+))?\s*$", value)
                if page_match:
                    try:
                        start = int(page_match.group(1))
                        end = int(page_match.group(2) or start)
                        if start > 0 and end > 0:
                            meta["page"] = start
                            meta["page_range"] = [start, end]
                    except (TypeError, ValueError):
                        pass
                continue
            normalized_key = {
                "group_id": "group_id",
                "context_id": "context_id",
                "evidence_id": "evidence_id",
                "chunk_id": "chunk_id",
                "child_chunk_id": "child_chunk_id",
                "parent_id": "parent_id",
                "source": "source",
            }.get(key, key)
            meta[normalized_key] = value
        return meta

    def _external_rerank_cache_key(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        basis = [question.strip()]
        for item in candidates:
            basis.append(
                "|".join([
                    str(item.get("chunk_id") or ""),
                    str(item.get("child_chunk_id") or ""),
                    str(item.get("parent_id") or ""),
                    str(item.get("group_id") or ""),
                    str(item.get("page") or ""),
                    str(item.get("text") or "")[:200],
                ])
            )
        joined = "\u241e".join(basis)
        return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()

    def _apply_external_rerank(
        self,
        question: str,
        scored_chunks: List[tuple],
    ) -> tuple[List[tuple], Dict[str, Any]]:
        diag = {
            "applied": False,
            "provider": self.rerank_provider or ("local" if self.use_rerank else ""),
            "model": self.reranker_model or ("BAAI/bge-reranker-base" if self.use_rerank else ""),
            "input_count": 0,
            "output_count": 0,
            "top_score": 0.0,
            "median_score": 0.0,
            "min_score": 0.0,
            "elapsed_ms": 0.0,
            "error": None,
            "cache_hit": False,
        }
        skip_reason = self._external_rerank_skip_reason()
        if skip_reason:
            if skip_reason != "disabled":
                diag["error"] = skip_reason
            return scored_chunks, diag
        top_candidates = list(scored_chunks[: min(30, len(scored_chunks))])
        if len(top_candidates) < 2:
            diag["error"] = "insufficient_candidates"
            return scored_chunks, diag
        candidate_payload: List[Dict[str, Any]] = []
        for score, idx, chunk in top_candidates:
            meta = self._extract_tool_chunk_meta(chunk)
            candidate_payload.append({
                "chunk": meta.get("text") or chunk,
                "rerank_text": meta.get("rerank_text") or meta.get("text") or chunk,
                "context_id": meta.get("context_id"),
                "evidence_id": meta.get("evidence_id"),
                "chunk_id": meta.get("chunk_id"),
                "child_chunk_id": meta.get("child_chunk_id"),
                "parent_id": meta.get("parent_id"),
                "group_id": meta.get("group_id"),
                "page": meta.get("page"),
                "similarity": float(score or 0.0),
                "_orig_score": float(score or 0.0),
                "_orig_idx": idx,
                "_orig_chunk": chunk,
            })
        diag["input_count"] = len(candidate_payload)
        cache = self._partial_state.setdefault("external_rerank_cache", {})
        cache_key = self._external_rerank_cache_key(question, candidate_payload)
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and cached:
            for item in candidate_payload:
                item["rerank_score"] = float(cached.get(int(item.get("_orig_idx", 0)), item.get("similarity", 0.0)) or 0.0)
            reranked = sorted(
                candidate_payload,
                key=lambda item: (-float(item.get("rerank_score", 0.0)), int(item.get("_orig_idx", 0))),
            )
            diag["cache_hit"] = True
            diag["applied"] = True
        else:
            try:
                from services.rerank_service import rerank_service
                started = time.perf_counter()
                reranked = rerank_service.rerank(
                    query=question,
                    candidates=[dict(item) for item in candidate_payload],
                    model_name=self.reranker_model or None,
                    provider=self.rerank_provider or "local",
                    api_key=self.rerank_api_key or None,
                    endpoint=self.rerank_endpoint or None,
                    timeout=10.0,
                )
                diag["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if not isinstance(reranked, list) or not reranked:
                    diag["error"] = "empty_rerank_result"
                    return scored_chunks, diag
                cache[cache_key] = {
                    int(item.get("_orig_idx", 0)): float(item.get("rerank_score", item.get("similarity", 0.0)) or 0.0)
                    for item in reranked
                }
                diag["applied"] = True
            except Exception as exc:
                diag["error"] = str(exc)
                return scored_chunks, diag
        rerank_scores = [float(item.get("rerank_score", item.get("similarity", 0.0)) or 0.0) for item in reranked]
        if rerank_scores:
            sorted_scores = sorted(rerank_scores)
            diag["top_score"] = round(max(rerank_scores), 4)
            diag["median_score"] = round(sorted_scores[len(sorted_scores) // 2], 4)
            diag["min_score"] = round(min(rerank_scores), 4)
        diag["output_count"] = len(reranked)
        reordered: List[tuple] = []
        for item in reranked:
            reordered.append((float(item.get("rerank_score", item.get("similarity", 0.0)) or 0.0), int(item.get("_orig_idx", 0)), item.get("_orig_chunk", "")))
        appended_indices = {int(item.get("_orig_idx", -1)) for item in reranked}
        for score, idx, chunk in scored_chunks:
            if idx in appended_indices:
                continue
            reordered.append((score, idx, chunk))
        return reordered, diag

    async def run(
        self,
        question: str,
        doc_ctx: DocContext,
        doc_name: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行多轮检索，流式 yield 进度和最终上下文

        Yields:
            进度事件: {"type": "retrieval_progress", "phase": str, "message": str}
            最终结果: {"type": "retrieval_complete", "context": str, "detail": list}
        """
        has_groups = bool(doc_ctx.semantic_groups)
        group_tools = _GROUP_TOOLS_TEMPLATE if has_groups else ""

        system_prompt = _AGENT_SYSTEM_PROMPT.format(group_tools=group_tools)

        # P4: 缓存 doc_ctx 与 group_chunk_map，供 _candidate_summary_from_result 做
        # child→parent chunk_idx 扩展（命中 chunk 所属 group 的兄弟 chunk_idx 进入候选池）
        self._doc_ctx = doc_ctx
        self._group_chunk_map = None
        try:
            from services.embedding_service import _load_group_data
            self._group_chunk_map = _load_group_data(doc_ctx.doc_id)
        except Exception as exc:
            logger.debug(f"[RetrievalAgent] _load_group_data 失败，跳过 child→parent 扩展: {exc}")
            self._group_chunk_map = None

        # 状态
        fetched_content: Dict[str, dict] = {}  # group_id -> {granularity, text}
        search_results: List[str] = []  # 累积的搜索结果片段
        search_history: List[dict] = []  # 搜索历史
        task_status = {"completed": [], "current": "", "pending": []}
        # P1: 把状态对象引用绑定到 _partial_state，使外层 timeout 时通过
        # snapshot_partial_diagnostics() 仍能读到当前已累积的 partial 数据
        self._partial_state["search_history"] = search_history
        self._partial_state["search_results"] = search_results
        self._partial_state["fetched_content"] = fetched_content
        self.diagnostics = {
            "planner_rounds": [],
            "tool_timings": [],
            "context_budget": {},
            "errors": [],
            "fallback_reason": "",
            "iteration_count": 0,
            "tool_call_count": 0,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "context_compress_threshold": self.context_compress_threshold,
            "compressed_context_chars": 0,
            "compression_count": 0,
            "forced_initial_search": False,
            "initial_search_blueprint_tools": [],
            "forced_initial_search_injected_tools": [],
            "forced_initial_search_terms": {},
            "forced_initial_search_grep_query": "",
            "default_initial_search_tools": [],
            "default_initial_search_terms": {},
            "default_initial_search_grep_query": "",
            "final_transition": {},
            "final_transition_reason": "",
        }

        yield {
            "type": "retrieval_progress",
            "phase": "agent_start",
            "message": "正在分析问题，规划检索策略...",
        }

        loop_limit = min(self.max_rounds, self.max_iterations)
        tool_call_count = 0

        # ----------------------------------------------------------------
        # 上一轮状态（供下一轮 ``_compute_planner_hints`` 使用）
        #   - last_round_executed_calls: 上一轮成功执行的工具调用列表
        #     ([{"tool": str, "query": str}, ...]，第 0 轮为空)
        #   - last_round_total_hits:     上一轮所有工具命中总数（第 0 轮为 0）
        #   - last_round_duplicate_detected: 上一轮 planner 是否输出过被
        #     ``_is_duplicate_search`` 判定为重复并被跳过的工具调用
        # ----------------------------------------------------------------
        last_round_executed_calls: List[Dict[str, Any]] = []
        last_round_total_hits: int = 0
        last_round_duplicate_detected: bool = False

        for round_idx in range(loop_limit):
            self.diagnostics["iteration_count"] = round_idx + 1
            yield {
                "type": "retrieval_progress",
                "phase": "round_start",
                "round": round_idx + 1,
                "message": f"第 {round_idx + 1} 轮取材...",
            }

            # ----------------------------------------------------------------
            # 计算并注入本轮的 Planner_Hint（动态状态提示）
            #   - 必须在 ``_call_planner`` 之前执行，使提示进入 user message
            #   - 优先级合并由 ``_compute_planner_hints`` 内部完成：
            #     final > duplicate > empty > sufficient > first_round
            #   - sufficiency_level 取自上一轮 ``_assess_sufficiency`` 的 level
            #     字段；尚未评估时（如第 0 轮）为空字符串
            # 同时按 Property 6 把 hints 列表追加到诊断中，确保
            # ``planner_hints_per_round`` 长度严格等于已执行轮数
            # ----------------------------------------------------------------
            sufficiency_level = (self.diagnostics.get("sufficiency") or {}).get("level", "")
            sub_question_coverage = task_status.get("sub_question_coverage") or []
            query_uncovered_sub_questions = [
                sq
                for idx, sq in enumerate(self.sub_questions or [])
                if idx >= len(sub_question_coverage) or not sub_question_coverage[idx]
            ]
            evidence_uncovered_sub_questions = (
                (self.diagnostics.get("sufficiency") or {})
                .get("sub_question_evidence_coverage", {})
                .get("uncovered", [])
            )
            uncovered_sub_questions = _merge_uncovered_sub_questions(
                self.sub_questions,
                query_uncovered_sub_questions,
                evidence_uncovered_sub_questions,
            )
            hints = _compute_planner_hints(
                round_idx=round_idx,
                max_rounds=loop_limit,
                last_round_calls=last_round_executed_calls,
                last_round_total_hits=last_round_total_hits,
                duplicate_detected=last_round_duplicate_detected,
                sufficiency_level=sufficiency_level,
                uncovered_sub_questions=uncovered_sub_questions,
            )
            self.diagnostics.setdefault("planner_hints_per_round", []).append(list(hints))
            self.diagnostics.setdefault("uncovered_sub_questions_per_round", []).append(list(uncovered_sub_questions))
            self.diagnostics.setdefault("uncovered_sub_questions_by_reason_per_round", []).append({
                "query": list(query_uncovered_sub_questions),
                "evidence": list(evidence_uncovered_sub_questions or []),
                "merged": list(uncovered_sub_questions),
            })

            # 本轮统计累加器（在工具执行循环中累计，循环末尾覆盖 last_round_*）
            current_round_executed_calls: List[Dict[str, Any]] = []
            current_round_total_hits: int = 0
            # 本轮是否在去重检查处发现 planner 输出了重复搜索
            duplicate_detected_this_round: bool = False

            # 构建用户消息（注入本轮的动态提示）
            user_content = self._build_user_message(
                question, doc_name, search_results, search_history,
                fetched_content, task_status, round_idx,
                hints=hints,
            )

            # 调用 LLM 规划
            yield {
                "type": "retrieval_progress",
                "phase": "planning",
                "round": round_idx + 1,
                "message": "LLM 规划中...",
            }

            plan = await self._call_planner(system_prompt, user_content, round_idx + 1)
            if plan is None:
                planner_error = self.diagnostics.get("last_error") or "planner_failed"
                logger.warning(f"[RetrievalAgent] 第 {round_idx + 1} 轮规划失败: {planner_error}")
                if not search_results and not fetched_content:
                    operations = self._build_default_operations(question)
                    if operations:
                        is_final = True
                        self.diagnostics["fallback_reason"] = "planner_error_default_search"
                        self.diagnostics["planner_fallback_error"] = planner_error
                        yield {
                            "type": "retrieval_progress",
                            "phase": "planner_error",
                            "round": round_idx + 1,
                            "message": "检索规划失败，执行默认组合检索...",
                            "error": planner_error,
                        }
                    else:
                        yield {
                            "type": "retrieval_progress",
                            "phase": "planner_error",
                            "round": round_idx + 1,
                            "message": "检索规划失败，准备使用已获取内容或降级上下文",
                            "error": planner_error,
                        }
                        self._record_final_transition(
                            "planner_error_no_plan",
                            round_no=round_idx + 1,
                            phase="planner_error",
                            detail={"planner_error": planner_error},
                        )
                        break
                else:
                    yield {
                        "type": "retrieval_progress",
                        "phase": "planner_error",
                        "round": round_idx + 1,
                        "message": "检索规划失败，准备使用已获取内容或降级上下文",
                        "error": planner_error,
                    }
                    self._record_final_transition(
                        "planner_error_with_partial_evidence",
                        round_no=round_idx + 1,
                        phase="planner_error",
                        detail={"planner_error": planner_error},
                    )
                    break
            else:
                operations = plan.get("operations", [])
                if not isinstance(operations, list):
                    operations = []
                is_final = plan.get("final", False)

                # 更新任务追踪
                new_status = plan.get("taskStatus")
                if new_status and isinstance(new_status, dict):
                    task_status = {
                        "completed": new_status.get("completed", task_status["completed"]),
                        "current": new_status.get("current", ""),
                        "pending": new_status.get("pending", []),
                    }

                if round_idx == 0:
                    operations = self._ensure_initial_search(operations, question)

            if not operations and not is_final and not search_results and not fetched_content:
                operations = self._build_default_operations(question)
                if operations:
                    is_final = True
                    self.diagnostics["fallback_reason"] = "empty_plan_default_search"
                    yield {
                        "type": "retrieval_progress",
                        "phase": "planning",
                        "round": round_idx + 1,
                        "message": "规划结果为空，执行默认组合检索...",
                    }

            if not operations and is_final and uncovered_sub_questions and round_idx < loop_limit - 1:
                fallback_question = str(uncovered_sub_questions[0] or "").strip()
                operations = self._build_default_operations(fallback_question)
                if operations:
                    is_final = False
                    self.diagnostics["fallback_reason"] = self.diagnostics.get("fallback_reason") or "planner_final_uncovered_subquestions"
                    self.diagnostics.setdefault("uncovered_subquestion_fallback_queries", []).append(fallback_question)
                    yield {
                        "type": "retrieval_progress",
                        "phase": "subquestion_gap",
                        "round": round_idx + 1,
                        "message": "仍有子问题未覆盖，执行补充检索...",
                    }

            if not operations and is_final:
                self._record_final_transition(
                    "planner_final",
                    round_no=round_idx + 1,
                    phase="planning",
                )
                break

            # 如果没有操作但不是最终，也结束（防止死循环）
            if not operations:
                self._record_final_transition(
                    "empty_plan_no_operations",
                    round_no=round_idx + 1,
                    phase="planning",
                )
                break

            if tool_call_count >= self.max_tool_calls:
                self.diagnostics["fallback_reason"] = self.diagnostics.get("fallback_reason") or "max_tool_calls_reached"
                yield {
                    "type": "retrieval_progress",
                    "phase": "loop_guard",
                    "round": round_idx + 1,
                    "message": "工具调用达到上限，停止继续检索。",
                }
                self._record_final_transition(
                    "max_tool_calls_reached",
                    round_no=round_idx + 1,
                    phase="loop_guard",
                )
                break

            prepared_ops = []
            seen_ops = set()
            remaining_tool_calls = self.max_tool_calls - tool_call_count
            for op in operations[:5]:
                if len(prepared_ops) >= remaining_tool_calls:
                    self.diagnostics["fallback_reason"] = self.diagnostics.get("fallback_reason") or "max_tool_calls_reached"
                    break
                normalized = self._normalize_operation(op)
                if not normalized:
                    continue
                tool_name, tool_args, query_key = normalized
                op_key = (tool_name, query_key)
                if query_key and (self._is_duplicate_search(search_history, tool_name, query_key) or op_key in seen_ops):
                    # 标记本轮检测到重复（供下一轮 Planner_Hint 触发 _HINT_DUPLICATE）
                    duplicate_detected_this_round = True
                    logger.info(f"[RetrievalAgent] 跳过重复搜索: {tool_name} {query_key}")
                    yield {
                        "type": "retrieval_progress",
                        "phase": "tool_result",
                        "round": round_idx + 1,
                        "message": f"跳过重复检索: {tool_name}",
                        "tool": tool_name,
                        "result_count": 0,
                    }
                    continue
                seen_ops.add(op_key)
                prepared_ops.append((tool_name, tool_args, query_key))
                yield {
                    "type": "retrieval_progress",
                    "phase": "executing",
                    "round": round_idx + 1,
                    "message": f"执行 {tool_name}...",
                    "tool": tool_name,
                }

            # 本轮收集的 chunk_meta（仅 vector_search/keyword_search 工具）
            # 用于 Group_Backfill 阶段提取 group_id（Requirements 4.2, 4.7）
            round_chunk_meta: List[dict] = []

            for batch_start in range(0, len(prepared_ops), self.max_tool_concurrency):
                batch = prepared_ops[batch_start:batch_start + self.max_tool_concurrency]
                batch_results = await asyncio.gather(*[
                    self._execute_tool_async(tool_name, tool_args, doc_ctx)
                    for tool_name, tool_args, _query_key in batch
                ])
                for (tool_name, tool_args, query_key), executed in zip(batch, batch_results):
                    tool_call_count += 1
                    self.diagnostics["tool_call_count"] = tool_call_count
                    result = executed["result"]
                    result_count = result.get("result_count", len(result.get("results", [])))
                    self._record_candidate_pool_trace(round_idx + 1, tool_name, query_key, result, result_count)
                    search_history.append({
                        "tool": tool_name,
                        "query": query_key,
                        "resultCount": result_count,
                    })
                    self.diagnostics["tool_timings"].append({
                        "round": round_idx + 1,
                        "tool": tool_name,
                        "query": query_key,
                        "result_count": result_count,
                        "elapsed_ms": executed["elapsed_ms"],
                        "error": result.get("error", ""),
                    })
                    self._merge_tool_result(tool_name, tool_args, result, search_results, fetched_content)

                    # 追踪子问题覆盖情况（Requirements 5.6）
                    self._track_sub_question_coverage(tool_args, task_status)

                    # 收集 vector_search/keyword_search 的 chunk_meta（供 Group_Backfill 使用）
                    if tool_name in ("vector_search", "keyword_search"):
                        chunk_meta = result.get("chunk_meta") or []
                        round_chunk_meta.extend(chunk_meta)

                    # 累计本轮工具调用统计（供下一轮 Planner_Hint 使用）
                    current_round_executed_calls.append({"tool": tool_name, "query": query_key})
                    current_round_total_hits += result_count

                    yield {
                        "type": "retrieval_progress",
                        "phase": "tool_result",
                        "round": round_idx + 1,
                        "message": result.get("summary", f"{tool_name} 完成"),
                        "tool": tool_name,
                        "result_count": result_count,
                        "elapsed_ms": executed["elapsed_ms"],
                    }

            # ----------------------------------------------------------------
            # Group_Backfill 阶段：将命中 chunk 所属语义组的 digest 回填到上下文
            # 当 enable_parent_backfill=True 时执行回填；否则跳过但仍记录 0
            # （Requirements 4.7, 4.8）
            # ----------------------------------------------------------------
            if settings.enable_parent_backfill and round_chunk_meta:
                group_ids = self._collect_group_ids(round_chunk_meta)
                backfill_count = self._apply_group_backfill(
                    group_ids, fetched_content, doc_ctx, self.backfilled_groups
                )
            else:
                backfill_count = 0
            self.diagnostics.setdefault("group_backfill_count_per_round", []).append(backfill_count)

            if tool_call_count >= self.max_tool_calls and not is_final:
                self.diagnostics["fallback_reason"] = self.diagnostics.get("fallback_reason") or "max_tool_calls_reached"
                yield {
                    "type": "retrieval_progress",
                    "phase": "loop_guard",
                    "round": round_idx + 1,
                    "message": "工具调用达到上限，使用已获取内容生成上下文。",
                }
                self._record_final_transition(
                    "max_tool_calls_reached",
                    round_no=round_idx + 1,
                    phase="loop_guard",
                )
                break

            # Phase 2.1：信息充足性 gate - LLM 认为还不够时也可能自动停止
            if not is_final and round_idx > 0:
                suf = self._assess_sufficiency(question, search_results, fetched_content, search_history)
                self.diagnostics["sufficiency"] = suf
                sub_question_coverage = task_status.get("sub_question_coverage") or []
                query_uncovered_sub_questions = [
                    sq
                    for idx, sq in enumerate(self.sub_questions or [])
                    if idx >= len(sub_question_coverage) or not sub_question_coverage[idx]
                ]
                evidence_uncovered_sub_questions = (
                    suf.get("sub_question_evidence_coverage", {}).get("uncovered", [])
                )
                uncovered_sub_questions = _merge_uncovered_sub_questions(
                    self.sub_questions,
                    query_uncovered_sub_questions,
                    evidence_uncovered_sub_questions,
                )
                self.diagnostics["latest_uncovered_sub_questions"] = list(uncovered_sub_questions)
                if suf["level"] == "sufficient":
                    is_final = True
                    self.diagnostics["sufficiency_early_stop"] = True
                    yield {
                        "type": "retrieval_progress",
                        "phase": "sufficiency_gate",
                        "round": round_idx + 1,
                        "message": f"信息已充足（{suf['total_chars']}字/{suf['unique_sources']}源），提前停止检索。",
                    }

            # 如果标记为最终，结束
            if is_final:
                # 更新跟踪变量（虽然即将 break，保持一致性）
                last_round_executed_calls = current_round_executed_calls
                last_round_total_hits = current_round_total_hits
                last_round_duplicate_detected = duplicate_detected_this_round
                if self.diagnostics.get("sufficiency_early_stop"):
                    reason = "sufficiency_early_stop"
                    phase = "sufficiency_gate"
                else:
                    reason = self.diagnostics.get("fallback_reason") or "planner_final"
                    phase = "planning" if reason == "planner_final" else "fallback"
                self._record_final_transition(
                    reason,
                    round_no=round_idx + 1,
                    phase=phase,
                )
                break

            # ----------------------------------------------------------------
            # 轮末更新：将本轮统计覆盖到 last_round_* 变量，供下一轮
            # ``_compute_planner_hints`` 读取（Requirements 3.2, 3.3, 3.4）
            # ----------------------------------------------------------------
            last_round_executed_calls = current_round_executed_calls
            last_round_total_hits = current_round_total_hits
            last_round_duplicate_detected = duplicate_detected_this_round

        self._record_final_transition(
            "loop_exhausted",
            round_no=self.diagnostics.get("iteration_count", 0),
            phase="loop",
        )

        # 构建最终上下文
        final_context, detail, context_budget = self._build_final_context(
            question, search_results, fetched_content
        )
        self.diagnostics["context_budget"] = context_budget

        # 写入子问题与覆盖情况到诊断，便于前端取用（Requirements 5.7）
        self.diagnostics["sub_questions"] = self.sub_questions or []
        if task_status.get("sub_question_coverage"):
            self.diagnostics["sub_question_coverage"] = task_status["sub_question_coverage"]

        retrieval_diagnostics = self._build_agent_retrieval_diagnostics(
            search_history=search_history,
            search_results=search_results,
            fetched_content=fetched_content,
            detail=detail,
            context_text=final_context,
            context_budget=context_budget,
        )

        yield {
            "type": "retrieval_progress",
            "phase": "complete",
            "message": f"检索完成，共获取 {len(search_results)} 个片段，{len(fetched_content)} 个意群",
        }

        yield {
            "type": "retrieval_complete",
            "context": final_context,
            "detail": detail,
            "search_history": search_history,
            "task_status": task_status,
            "diagnostics": self.diagnostics,
            "retrieval_diagnostics": retrieval_diagnostics,
        }

    def _build_user_message(
        self,
        question: str,
        doc_name: str,
        search_results: List[str],
        search_history: List[dict],
        fetched_content: Dict[str, dict],
        task_status: dict,
        round_idx: int,
        hints: List[str] = (),
    ) -> str:
        """构建每轮发送给 planner LLM 的用户消息

        参数:
            hints: 由 ``_compute_planner_hints`` 计算得到的动态提示文本列表；
                非空时会在消息最前面注入 ``【动态提示】`` 区块，便于 Planner_LLM
                感知首轮/重复搜索/空结果/充足/最终轮等状态。默认 ``()``（空元组）
                以避免使用可变默认值。
        """
        parts = []

        # 动态提示块：必须在所有既有内容之前注入，确保 Planner_LLM 在
        # 阅读问题与历史前先看到本轮的状态指示（参考设计文档需求 3.1-3.5）。
        if hints:
            hint_lines = [f"- {h}" for h in hints if h]
            if hint_lines:
                parts.append("【动态提示】\n" + "\n".join(hint_lines))

        # 子问题列表：第 0 轮且 sub_questions 非空时注入，引导 Planner_LLM
        # 为每个子问题分派检索（Requirements 5.4, 5.5）
        if round_idx == 0 and self.sub_questions:
            sq_lines = [f"{i+1}. {sq}" for i, sq in enumerate(self.sub_questions)]
            parts.append(
                "【子问题列表】（请尽量为每个子问题分派至少一次检索）：\n"
                + "\n".join(sq_lines)
            )

        parts.append(f"文档名称: {doc_name}")
        parts.append(f"\n用户问题:\n{question}")

        # 搜索历史
        if search_history:
            recent = search_history[-8:]
            history_lines = []
            for s in recent:
                tool_label = {
                    "vector_search": "向量",
                    "keyword_search": "BM25",
                    "grep": "GREP",
                    "regex_search": "正则",
                    "boolean_search": "布尔",
                    "fetch": "获取意群",
                    "map": "文档地图",
                }.get(s["tool"], s["tool"])
                status = f"✓ {s['resultCount']}个结果" if s["resultCount"] > 0 else "✗ 无结果"
                history_lines.append(f"- {tool_label} \"{s['query']}\" → {status}")
            parts.append(f"\n【搜索历史】(避免重复搜索):\n" + "\n".join(history_lines))

        # 任务追踪
        if round_idx > 0 and (task_status["completed"] or task_status["current"]):
            status_parts = []
            if task_status["completed"]:
                status_parts.append(f"已完成: {'; '.join(task_status['completed'])}")
            if task_status["current"]:
                status_parts.append(f"上轮任务: {task_status['current']}")
            if task_status["pending"]:
                status_parts.append(f"待完成: {'; '.join(task_status['pending'])}")
            parts.append(f"\n【任务追踪】\n" + "\n".join(status_parts))

        # 已获取内容摘要
        fetched_summary = "无"
        all_content = []

        # search_results 中的片段
        for i, chunk in enumerate(search_results[:15]):
            preview = chunk[:500] + "..." if len(chunk) > 500 else chunk
            all_content.append(f"[片段{i+1}]\n{preview}")

        # fetched_content 中的意群
        for gid, data in fetched_content.items():
            preview = data["text"][:500] + "..." if len(data["text"]) > 500 else data["text"]
            all_content.append(f"【{gid}】({data['granularity']})\n{preview}")

        if all_content:
            fetched_summary = "\n\n".join(all_content)
            fetched_summary = self._compress_context_summary(fetched_summary)

        parts.append(f"\n【已获取内容】:\n{fetched_summary}")

        return "\n".join(parts)

    async def _call_planner(self, system_prompt: str, user_content: str, round_no: int) -> Optional[dict]:
        # 判断是否使用原生函数调用模式
        use_native = settings.use_native_tools and self._provider_supports_tools(self.provider)
        tools = TOOL_SCHEMAS if use_native else None

        last_error = ""
        for attempt in range(self.planner_retries + 1):
            attempt_content = user_content
            if attempt > 0:
                attempt_content = (
                    f"{user_content}\n\n【上次规划输出无法解析】\n"
                    "请重新输出严格 JSON，不要包含解释性文字，格式必须包含 operations、final、taskStatus。"
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attempt_content},
            ]
            started = time.perf_counter()
            record = {"round": round_no, "attempt": attempt + 1, "ok": False, "elapsed_ms": 0, "error": ""}
            try:
                # 单次 LLM 调用 timeout 防护：避免 deepseek 等模型卡住时整个 Agent 被无限阻塞
                # 参考 agentic-rag-for-dummies 的 GRAPH_RECURSION_LIMIT 思路，把超时下沉到每次调用
                _planner_timeout = max(15.0, float(getattr(settings, "agent_planner_timeout", 30.0) or 30.0))
                response = await asyncio.wait_for(
                    call_ai_api(
                        messages=messages,
                        api_key=self.api_key,
                        model=self.model,
                        provider=self.provider,
                        endpoint=self.endpoint,
                        middlewares=[RetryMiddleware(retries=1, delay=0.2)],
                        max_tokens=2000,
                        temperature=self.temperature,
                        tools=tools,
                        purpose="agent",
                    ),
                    timeout=_planner_timeout,
                )
                record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if isinstance(response, dict) and response.get("error"):
                    last_error = str(response.get("error"))
                    record["error"] = last_error
                else:
                    choices = response.get("choices") if isinstance(response, dict) else None
                    message = (choices or [{}])[0].get("message", {}) if choices else {}
                    tool_calls = message.get("tool_calls") or []
                    content = (message.get("content") or "").strip()
                    record["content_preview"] = content[:300]

                    # 优先使用原生 tool_calls 解析
                    if use_native and tool_calls:
                        plan = self._tool_calls_to_plan(tool_calls, content)
                        record["ok"] = True
                        self.diagnostics["planner_rounds"].append(record)
                        self.diagnostics.setdefault("planner_invocation_mode", []).append("native_tools")
                        return plan

                    # 兜底：JSON 文本解析
                    plan = self._parse_plan_json(content)
                    if plan is not None:
                        record["ok"] = True
                        self.diagnostics["planner_rounds"].append(record)
                        self.diagnostics.setdefault("planner_invocation_mode", []).append("json_fallback")
                        return plan
                    last_error = "planner_json_parse_failed"
                    record["error"] = last_error
            except asyncio.TimeoutError:
                # 单次 LLM 调用超时（>=15s 阈值），视为 attempt 失败而非整轮卡死
                # 这条分支防止 deepseek-chat 等模型在网络抖动 / 推理慢时卡住整个 Agent
                record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                last_error = f"planner_call_timeout(>{_planner_timeout:.0f}s)"
                record["error"] = last_error
                logger.warning(f"[RetrievalAgent] planner LLM 调用超时（{_planner_timeout:.0f}s）")
            except Exception as e:
                record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                last_error = str(e)
                record["error"] = last_error
                logger.error(f"[RetrievalAgent] LLM 调用异常: {e}")
            self.diagnostics["planner_rounds"].append(record)
        # 所有重试均失败，记录 json_fallback 模式
        self.diagnostics.setdefault("planner_invocation_mode", []).append("json_fallback")
        self.diagnostics["last_error"] = last_error
        self.diagnostics["errors"].append({"type": "planner", "message": last_error})
        return None

    def _parse_plan_json(self, content: str) -> Optional[dict]:
        """从 LLM 输出中解析 JSON 检索计划"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试找到第一个 { 和最后一个 }
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"[RetrievalAgent] 无法解析 JSON: {content[:200]}")
        return None

    @staticmethod
    def _provider_supports_tools(provider: str) -> bool:
        """判断 provider 是否支持原生函数调用（Native Tool Calls）"""
        return provider in {
            "openai", "deepseek", "grok", "anthropic", "gemini",
            "silicon", "aliyun", "moonshot", "zhipu", "minimax", "qwen", "doubao",
        }

    def _tool_calls_to_plan(self, tool_calls: list, text_content: str) -> dict:
        """将 LLM 原生 tool_calls 响应解析为 Agent 检索计划

        Args:
            tool_calls: LLM 响应中的 tool_calls 列表（OpenAI 标准格式）
            text_content: LLM 响应中的文本内容（用于检测 FINAL_ANSWER 指示）

        Returns:
            与 _parse_plan_json 返回格式一致的 plan 字典
        """
        operations = []
        for tc in tool_calls[:self.max_tool_calls]:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                logger.warning(f"[RetrievalAgent] tool_call arguments 解析失败: {fn.get('arguments')}")
            operations.append({"tool": name, "args": args, "rationale": ""})
        # 模型可在文本中给出 final 指示，简单启发式判断
        final = bool(operations) is False or "FINAL_ANSWER" in (text_content or "").upper()
        return {"operations": operations, "final": final, "taskStatus": {}}

    def _normalize_operation(self, op: dict) -> Optional[tuple[str, dict, str]]:
        if not isinstance(op, dict):
            return None
        tool_name = str(op.get("tool", "") or "").strip()
        if not tool_name:
            return None
        tool_args = op.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}
        if "limit" in tool_args:
            try:
                tool_args["limit"] = max(1, min(int(tool_args.get("limit") or 10), 20))
            except Exception:
                tool_args["limit"] = 10
        query_key = self._operation_query_key(tool_name, tool_args)
        return tool_name, tool_args, query_key

    def _operation_query_key(self, tool_name: str, tool_args: dict) -> str:
        value = (
            tool_args.get("query")
            or tool_args.get("pattern")
            or tool_args.get("groupId")
            or tool_args.get("group_id")
            or tool_args.get("keywords")
            or ""
        )
        if isinstance(value, list):
            value = "|".join(str(x) for x in value)
        return str(value).strip()

    def _build_initial_search_bundle(self, question: str) -> dict:
        q = (question or "").strip()
        if not q:
            return {
                "operations": [],
                "terms": {"high_level": [], "low_level": []},
                "grep_query": "",
                "tool_names": [],
            }
        # 提取文档级术语桥接和关键短语（从文档全文中动态提取，替代全局硬编码）
        doc_bridges = None
        doc_key_phrases: list[str] = []
        if self._doc_ctx and self._doc_ctx.full_text:
            try:
                doc_bridges, doc_key_phrases = extract_document_bilingual_terms(self._doc_ctx.full_text)
            except Exception:
                pass
        terms = extract_hl_ll_terms(q, doc_bridges=doc_bridges)
        # 存储关键短语供证据评分使用
        self._doc_key_phrases = doc_key_phrases
        bilingual_terms = terms.get("bilingual") or []
        high_level_query = " ".join(_dedupe_terms([*(terms.get("high_level") or []), *bilingual_terms[:4]])) or q
        formula_terms = _extract_formula_search_terms(q)
        low_level_terms = _dedupe_terms([
            *formula_terms,
            *(terms.get("low_level") or []),
            *bilingual_terms,
            *(terms.get("high_level") or []),
        ])[:16] or [q]
        grep_query = self._build_or_grep_query(q, {**terms, "formula": formula_terms})
        operations = [
            {"tool": "vector_search", "args": {"query": high_level_query, "limit": 14}},
            {"tool": "keyword_search", "args": {"keywords": low_level_terms, "limit": 14}},
            {"tool": "map", "args": {"limit": 20, "includeStructure": True}},
            {"tool": "grep", "args": {"query": grep_query, "limit": 12, "context": 2000, "caseInsensitive": True}},
        ]
        return {
            "operations": operations,
            "terms": terms,
            "grep_query": grep_query,
            "tool_names": [op.get("tool", "") for op in operations],
        }

    def _build_or_grep_query(self, question: str, terms: dict) -> str:
        candidates = []
        for term in [
            *(terms.get("formula") or []),
            *(terms.get("low_level") or []),
            *(terms.get("bilingual") or []),
            *(terms.get("high_level") or []),
        ]:
            text = str(term or "").strip()
            if len(text) < 2:
                continue
            candidates.append(text[:40])
        if not candidates:
            candidates = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?", question or "")
        ordered = []
        seen = set()
        for item in candidates:
            key = item.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return "|".join(ordered[:12])[:260] or (question or "")[:80]

    def _build_default_operations(self, question: str) -> list[dict]:
        bundle = self._build_initial_search_bundle(question)
        if bundle["operations"]:
            self.diagnostics["default_initial_search_tools"] = list(bundle.get("tool_names") or [])
            self.diagnostics["default_initial_search_terms"] = dict(bundle.get("terms") or {})
            self.diagnostics["default_initial_search_grep_query"] = bundle.get("grep_query") or ""
        return list(bundle.get("operations") or [])

    def _ensure_initial_search(self, operations: list, question: str) -> list:
        bundle = self._build_initial_search_bundle(question)
        if not bundle["operations"]:
            return operations
        existing = set()
        for op in operations:
            if isinstance(op, dict):
                existing.add(str(op.get("tool", "")).strip())
        injected = []
        self.diagnostics["initial_search_blueprint_tools"] = list(bundle.get("tool_names") or [])
        for op in bundle["operations"]:
            tool_name = str(op.get("tool", "") or "").strip()
            if tool_name and tool_name not in existing:
                injected.append(op)
        if injected:
            self.diagnostics["forced_initial_search"] = True
            self.diagnostics["forced_initial_search_terms"] = dict(bundle.get("terms") or {})
            self.diagnostics["forced_initial_search_grep_query"] = bundle.get("grep_query") or ""
            self.diagnostics["forced_initial_search_injected_tools"] = [
                str(op.get("tool", "") or "") for op in injected if str(op.get("tool", "") or "")
            ]
        return [*injected, *operations]

    def _compress_context_summary(self, text: str) -> str:
        threshold = self.context_compress_threshold
        if not threshold or not text or len(text) <= threshold:
            return text
        head_chars = max(1000, int(threshold * 0.6))
        tail_chars = max(500, threshold - head_chars)
        compressed = (
            text[:head_chars].rstrip()
            + f"\n\n【context_summary】中间 {max(0, len(text) - head_chars - tail_chars)} 字符已压缩，仅保留首尾高置信片段。\n\n"
            + text[-tail_chars:].lstrip()
        )
        saved = max(0, len(text) - len(compressed))
        self.diagnostics["compressed_context_chars"] = self.diagnostics.get("compressed_context_chars", 0) + saved
        self.diagnostics["compression_count"] = self.diagnostics.get("compression_count", 0) + 1
        return compressed

    async def _execute_tool_async(self, tool_name: str, tool_args: dict, doc_ctx: DocContext) -> dict:
        started = time.perf_counter()
        # P1: 单工具 timeout 上限（参考 ragflow COMPONENT_EXEC_TIMEOUT=12）
        # 单工具卡住只丢弃该工具结果，不拖垮整轮 Agent，partial candidate_pool 仍能积累
        tool_timeout = max(2.0, float(getattr(settings, "agent_tool_timeout", 12.0) or 12.0))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(execute_tool, tool_name, tool_args, doc_ctx),
                timeout=tool_timeout,
            )
        except asyncio.TimeoutError:
            self.diagnostics.setdefault("errors", []).append({
                "type": "tool_timeout",
                "tool": tool_name,
                "timeout_s": tool_timeout,
            })
            result = {
                "results": [],
                "summary": f"{tool_name} 超时（>{tool_timeout:.0f}s）",
                "error": f"tool_timeout(>{tool_timeout:.0f}s)",
                "result_count": 0,
                "chunk_meta": [],
                "candidate_meta": [],
            }
        return {
            "result": result if isinstance(result, dict) else {"results": [], "summary": f"{tool_name} 返回异常格式"},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _merge_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        result: dict,
        search_results: List[str],
        fetched_content: Dict[str, dict],
    ) -> None:
        tool_results = result.get("results", [])
        if tool_name == "fetch":
            group_id = tool_args.get("groupId", "") or tool_args.get("group_id", "")
            gran = result.get("granularity", "full")
            if tool_results:
                fetched_content[group_id] = {
                    "granularity": gran,
                    "text": tool_results[0],
                    "page_range": result.get("page_range") or [],
                    "keywords": result.get("keywords") or [],
                    "context_id": result.get("context_id") or group_id,
                    "evidence_id": result.get("evidence_id") or f"{group_id}:{gran}",
                }
        elif tool_name == "map":
            if tool_results:
                map_text = _stringify_tool_result_item(tool_results[0])
                if map_text:
                    search_results.append(f"【文档地图】\n{map_text[:3000]}")
        else:
            seen_result_keys: set[str] = set()
            for existing in search_results:
                existing_meta = self._extract_tool_chunk_meta(existing)
                existing_text = str(existing_meta.get("text") or existing or "")
                seen_result_keys.add(_context_part_dedupe_key(existing_meta, existing_text))
            for chunk in tool_results:
                chunk_text = _stringify_tool_result_item(chunk)
                result_key = _tool_result_dedupe_key(chunk, chunk_text)
                if result_key in seen_result_keys:
                    self.diagnostics.setdefault("tool_result_dedup_removed", 0)
                    self.diagnostics["tool_result_dedup_removed"] += 1
                    continue
                chunk_text = _format_tool_result_with_provenance(chunk, chunk_text)
                if chunk_text and chunk_text not in search_results:
                    search_results.append(chunk_text)
                    seen_result_keys.add(result_key)

    def _track_sub_question_coverage(self, tool_args: dict, task_status: dict):
        """追踪子问题覆盖情况

        使用子问题中的通用技术锚点匹配每次工具的 query/keywords 文本。
        命中即将 task_status["sub_question_coverage"][i] 置 true（且不会回退）。
        初始化时长度等于 len(sub_questions)。

        Requirements: 5.6
        """
        if not self.sub_questions:
            return
        coverage = task_status.setdefault("sub_question_coverage", [False] * len(self.sub_questions))
        query_text = (tool_args.get("query") or " ".join(tool_args.get("keywords") or []) or "").strip()
        if not query_text:
            return
        for i, sq in enumerate(self.sub_questions):
            sub_question_text = str(sq or "").strip()
            if sub_question_text and sub_question_text in query_text:
                coverage[i] = True
                continue
            anchors = _extract_question_anchor_terms(sq, max_terms=8)
            if anchors:
                matched = [anchor for anchor in anchors if technical_anchor_matches(anchor, query_text)]
                required = 1 if len(anchors) <= 2 else min(2, max(1, math.ceil(len(anchors) * 0.5)))
                if len(matched) >= required:
                    coverage[i] = True
                continue
            prefix = sub_question_text[:8]
            if len(prefix) >= 8 and prefix in query_text:
                coverage[i] = True


    def _collect_group_ids(self, results: list) -> list:
        """从 vector_search/keyword_search 结果的 chunk_meta 中按命中顺序提取去重 group_id

        参数:
            results: chunk_meta 列表，每条为 dict，包含 chunk_idx、group_id、score

        返回:
            按命中顺序去重的 group_id 列表
        """
        seen, ordered = set(), []
        for r in results:
            gid = r.get("group_id") if isinstance(r, dict) else None
            if gid and gid not in seen:
                seen.add(gid)
                ordered.append(gid)
        return ordered

    def _apply_group_backfill(
        self,
        new_group_ids: list,
        fetched_content: dict,
        doc_ctx: "DocContext",
        backfilled_groups: set,
        max_per_round: int = 5,
    ) -> int:
        """按 group_id 去重并最多回填 max_per_round 次，返回实际回填条数

        对每个尚未回填的 group_id，调用 fetch 工具获取 digest 粒度文本并
        注入 fetched_content。单条 fetch 抛异常时跳过；累计失败 ≥ 3 时
        本轮停止回填并写入 diagnostics["errors"]。

        参数:
            new_group_ids: 本轮命中的去重 group_id 列表
            fetched_content: 已获取内容字典（group_id -> {granularity, text}）
            doc_ctx: 文档上下文
            backfilled_groups: 跨轮去重集合，记录已回填的 group_id
            max_per_round: 单轮最大回填数，默认 5

        返回:
            本轮实际成功回填的条数
        """
        count = 0
        fail_count = 0
        for gid in new_group_ids:
            if count >= max_per_round:
                break
            if fail_count >= 3:
                # 累计失败达到阈值，停止本轮回填并记录错误
                self.diagnostics.setdefault("errors", []).append({
                    "type": "group_backfill_abort",
                    "reason": "cumulative_failures>=3",
                })
                break
            if gid in backfilled_groups:
                continue
            if gid in fetched_content:
                # 已通过其他途径获取过该 group 的内容，跳过
                backfilled_groups.add(gid)
                continue
            try:
                result = execute_tool("fetch", {"groupId": gid, "granularity": "digest"}, doc_ctx)
                text = (result.get("results") or [""])[0]
                if text:
                    fetched_content[gid] = {
                        "granularity": "digest",
                        "text": text,
                        "page_range": result.get("page_range") or [],
                        "keywords": result.get("keywords") or [],
                    }
                    backfilled_groups.add(gid)
                    count += 1
            except Exception as e:
                fail_count += 1
                logger.warning(f"[RetrievalAgent] group_backfill fetch 失败 gid={gid}: {e}")
        return count

    def _assess_sufficiency(
        self,
        question: str,
        search_results: List[str],
        fetched_content: Dict[str, dict],
        search_history: List[dict],
    ) -> Dict[str, Any]:
        """Phase 2.1：评估当前检索信息是否充足（借鉴 paper-burner-x 启发式）"""
        total_chars = sum(len(s) for s in search_results)
        total_chars += sum(len(d["text"]) for d in fetched_content.values())
        unique_tools = set(h["tool"] for h in search_history if h.get("resultCount", 0) > 0)
        successful_calls = sum(1 for h in search_history if h.get("resultCount", 0) > 0)
        unique_sources = len(unique_tools)
        evidence_keys: set[str] = set()
        for idx, text in enumerate(search_results):
            meta = self._extract_tool_chunk_meta(text)
            evidence_text = str(meta.get("text") or text or "")
            normalized = re.sub(r"\s+", " ", evidence_text).strip()
            if normalized:
                evidence_keys.add(_evidence_independence_key(meta, normalized, fallback_scope=f"search:{idx}"))
        for group_id, data in fetched_content.items():
            text = str((data or {}).get("text") or "").strip()
            normalized_text = re.sub(r"\s+", " ", text).strip()
            if normalized_text or group_id:
                fetch_meta = {
                    "group_id": group_id,
                    "context_id": (data or {}).get("context_id") or group_id,
                    "evidence_id": (data or {}).get("evidence_id") or "",
                    "page_range": (data or {}).get("page_range") or [],
                }
                evidence_keys.add(
                    _evidence_independence_key(fetch_meta, normalized_text, fallback_scope=f"fetch:{group_id}")
                )
        independent_evidence_count = len(evidence_keys)
        required_independent_evidence = max(1, min(self.sufficiency_min_sources, 2))
        evidence_text = "\n".join([*search_results, *(str(d.get("text") or "") for d in fetched_content.values())])
        anchor_report = _question_anchor_coverage(question, evidence_text)
        anchor_coverage = float(anchor_report.get("coverage", 1.0) or 0.0)
        anchor_required = bool(anchor_report.get("required"))
        sub_question_report = _sub_question_evidence_coverage(self.sub_questions, evidence_text)

        if (
            successful_calls >= self.sufficiency_min_sources
            and independent_evidence_count >= required_independent_evidence
            and total_chars >= self.sufficiency_threshold_chars
        ):
            level = "sufficient"
        elif successful_calls >= 1 and total_chars >= self.sufficiency_threshold_chars * 0.5:
            level = "maybe_sufficient"
        else:
            level = "insufficient"
        if anchor_required and anchor_coverage < 0.5 and level == "sufficient":
            level = "maybe_sufficient"
        if anchor_required and anchor_coverage < 0.25:
            level = "insufficient"
        if (
            sub_question_report.get("required")
            and sub_question_report.get("uncovered")
            and level == "sufficient"
        ):
            level = "maybe_sufficient"

        return {
            "level": level,
            "total_chars": total_chars,
            "successful_calls": successful_calls,
            "unique_sources": unique_sources,
            "independent_evidence_count": independent_evidence_count,
            "required_independent_evidence": required_independent_evidence,
            "threshold_chars": self.sufficiency_threshold_chars,
            "min_sources": self.sufficiency_min_sources,
            "question_anchor_coverage": anchor_report,
            "sub_question_evidence_coverage": sub_question_report,
        }

    def _record_final_transition(
        self,
        reason: str,
        *,
        round_no: Optional[int] = None,
        phase: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录 Agent 检索循环的最终退出原因，供评估诊断按路径聚合。"""
        if not reason or self.diagnostics.get("final_transition_reason"):
            return
        sufficiency = self.diagnostics.get("sufficiency") or {}
        transition: Dict[str, Any] = {
            "reason": reason,
            "round": max(0, int(round_no or 0)),
            "phase": phase or "",
            "tool_call_count": max(0, int(self.diagnostics.get("tool_call_count") or 0)),
            "fallback_reason": self.diagnostics.get("fallback_reason") or "",
            "sufficiency_level": sufficiency.get("level", ""),
        }
        if detail:
            transition["detail"] = dict(detail)
        self.diagnostics["final_transition"] = transition
        self.diagnostics["final_transition_reason"] = reason

    def _is_duplicate_search(
        self, history: List[dict], tool_name: str, query_key: str
    ) -> bool:
        """检查是否为重复搜索"""
        if not query_key:
            return False
        for h in history:
            if h["tool"] == tool_name and h["query"] == query_key:
                return True
        return False

    def _tool_source_family(self, tool_name: str) -> str:
        mapping = {
            "vector_search": "vector",
            "keyword_search": "bm25",
            "grep": "lexical",
            "regex_search": "lexical",
            "boolean_search": "lexical",
            "fetch": "semantic_group",
            "map": "semantic_map",
        }
        return mapping.get(str(tool_name or "").strip(), str(tool_name or "unknown").strip() or "unknown")

    def _append_ordered(self, values: list, value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in values:
            values.append(text)

    def _append_page(self, pages: list[int], value: Any) -> None:
        try:
            page = int(value)
        except (TypeError, ValueError):
            return
        if page > 0 and page not in pages:
            pages.append(page)

    def _candidate_summary_from_result(
        self,
        tool_name: str,
        result: dict,
        *,
        use_candidate_meta: bool = True,
    ) -> Dict[str, Any]:
        pages: list[int] = []
        ids: list[str] = []
        group_ids: list[str] = []
        chunk_ids: list[str] = []
        table_ids: list[str] = []
        table_bundle_ids: list[str] = []
        evidence_unit_ids: list[str] = []
        if use_candidate_meta:
            meta_items = result.get("candidate_meta") or result.get("chunk_meta") or []
        else:
            meta_items = result.get("chunk_meta") or []
        for meta in meta_items:
            if not isinstance(meta, dict):
                continue
            self._append_page(pages, meta.get("page"))
            for key in ("context_id", "evidence_id", "chunk_id", "child_chunk_id", "chunk_idx", "parent_id", "doc_id"):
                value = meta.get(key)
                self._append_ordered(ids, value)
                if key in {"chunk_id", "child_chunk_id", "chunk_idx"}:
                    self._append_ordered(chunk_ids, value)
            group_id = meta.get("group_id")
            self._append_ordered(group_ids, group_id)
            self._append_ordered(ids, group_id)
            table_id = meta.get("table_id")
            self._append_ordered(table_ids, table_id)
            self._append_ordered(ids, table_id)
            table_bundle_id = meta.get("table_bundle_id")
            self._append_ordered(table_bundle_ids, table_bundle_id)
            self._append_ordered(ids, table_bundle_id)
            evidence_unit_id = meta.get("evidence_unit_id")
            self._append_ordered(evidence_unit_ids, evidence_unit_id)
            self._append_ordered(ids, evidence_unit_id)
            for unit in (meta.get("evidence_units") or meta.get("cell_evidence_units") or []):
                if not isinstance(unit, dict):
                    continue
                self._append_ordered(evidence_unit_ids, unit.get("evidence_unit_id"))
                self._append_ordered(ids, unit.get("evidence_unit_id"))
                self._append_ordered(table_ids, unit.get("table_id"))
                self._append_ordered(table_bundle_ids, unit.get("table_bundle_id"))
            if meta.get("chunk_idx") is not None:
                self._append_ordered(ids, f"chunk:{meta.get('chunk_idx')}")
            if meta.get("page"):
                self._append_ordered(ids, f"page:{meta.get('page')}")
        for entry in result.get("map_entries") or []:
            if not isinstance(entry, dict):
                continue
            group_id = entry.get("group_id")
            self._append_ordered(group_ids, group_id)
            self._append_ordered(ids, group_id)
            page_range = entry.get("page_range") or []
            if isinstance(page_range, list) and page_range:
                try:
                    start = int(page_range[0])
                    end = int(page_range[-1])
                except (TypeError, ValueError):
                    start = end = 0
                for page in range(start, end + 1):
                    self._append_page(pages, page)
                    self._append_ordered(ids, f"page:{page}")
        if result.get("group_id"):
            self._append_ordered(group_ids, result.get("group_id"))
            self._append_ordered(ids, result.get("group_id"))
        for text in result.get("results") or []:
            if not isinstance(text, str):
                continue
            for match in re.finditer(r"页码[:：]\s*(\d+)", text):
                self._append_page(pages, match.group(1))
                self._append_ordered(ids, f"page:{match.group(1)}")
            for match in re.finditer(r"group_id[:：]\s*([A-Za-z0-9_.:\-]+)", text):
                self._append_ordered(group_ids, match.group(1))
                self._append_ordered(ids, match.group(1))
            for match in re.finditer(r"chunk_id[:：]\s*([A-Za-z0-9_.:\-]+)", text):
                self._append_ordered(chunk_ids, match.group(1))
                self._append_ordered(ids, match.group(1))
        # P4: child→parent 扩展 - 仅对宽候选（candidate_meta）扩展，selected 不变。
        # 把每个命中 group 的同组兄弟 chunk_idx 也加入 candidate_pool.ids/chunk_ids，
        # 让候选池诊断能识别同一语义组内的相关证据，不必盲目扩 query。
        if use_candidate_meta and self._group_chunk_map and group_ids:
            sibling_chunk_count = 0
            for gid in list(group_ids)[:10]:  # 限制扩展 group 数，避免诊断爆炸
                indices = self._group_chunk_map.get(gid)
                if not isinstance(indices, list):
                    continue
                for idx in indices[:40]:  # 单 group 最多 40 个兄弟
                    if not isinstance(idx, int) or idx < 0:
                        continue
                    sibling_id = f"chunk:{idx}"
                    self._append_ordered(ids, sibling_id)
                    self._append_ordered(chunk_ids, sibling_id)
                    sibling_chunk_count += 1
                    if sibling_chunk_count >= 200:
                        break
                if sibling_chunk_count >= 200:
                    break
        return {
            "source": self._tool_source_family(tool_name),
            "pages": pages[:40],
            "ids": ids[:200] if use_candidate_meta else ids[:80],
            "group_ids": group_ids[:40],
            "chunk_ids": chunk_ids[:200] if use_candidate_meta else chunk_ids[:40],
            "table_ids": table_ids[:40],
            "table_bundle_ids": table_bundle_ids[:40],
            "evidence_unit_ids": evidence_unit_ids[:80],
        }

    def _record_candidate_pool_trace(
        self,
        round_no: int,
        tool_name: str,
        query_key: str,
        result: dict,
        result_count: int,
    ) -> None:
        result_dict = result if isinstance(result, dict) else {}
        summary = self._candidate_summary_from_result(tool_name, result_dict, use_candidate_meta=True)
        selected_summary = self._candidate_summary_from_result(tool_name, result_dict, use_candidate_meta=False)
        trace = {
            "round": round_no,
            "tool": tool_name,
            "query": query_key,
            "result_count": result_count,
            "selected_count": len(result_dict.get("chunk_meta") or []),
            "candidate_count": len(result_dict.get("candidate_meta") or result_dict.get("chunk_meta") or []),
            "selected_pages": selected_summary.get("pages") or [],
            "selected_ids": selected_summary.get("ids") or [],
            "selected_group_ids": selected_summary.get("group_ids") or [],
            "selected_chunk_ids": selected_summary.get("chunk_ids") or [],
            "selected_table_ids": selected_summary.get("table_ids") or [],
            "selected_table_bundle_ids": selected_summary.get("table_bundle_ids") or [],
            "selected_evidence_unit_ids": selected_summary.get("evidence_unit_ids") or [],
            **summary,
        }
        self.diagnostics.setdefault("candidate_pool_trace", []).append(trace)

    def _build_candidate_pool_summary(self) -> Dict[str, Any]:
        pages: list[int] = []
        ids: list[str] = []
        group_ids: list[str] = []
        chunk_ids: list[str] = []
        table_ids: list[str] = []
        table_bundle_ids: list[str] = []
        evidence_unit_ids: list[str] = []
        selected_pages: list[int] = []
        selected_ids: list[str] = []
        selected_group_ids: list[str] = []
        selected_chunk_ids: list[str] = []
        selected_table_ids: list[str] = []
        selected_table_bundle_ids: list[str] = []
        selected_evidence_unit_ids: list[str] = []
        by_tool: list[dict] = []
        selected_count = 0
        candidate_count = 0
        for item in self.diagnostics.get("candidate_pool_trace") or []:
            if not isinstance(item, dict):
                continue
            by_tool.append(item)
            selected_count += max(0, int(item.get("selected_count") or 0))
            candidate_count += max(0, int(item.get("candidate_count") or 0))
            for page in item.get("pages") or []:
                self._append_page(pages, page)
            for item_id in item.get("ids") or []:
                self._append_ordered(ids, item_id)
            for group_id in item.get("group_ids") or []:
                self._append_ordered(group_ids, group_id)
            for chunk_id in item.get("chunk_ids") or []:
                self._append_ordered(chunk_ids, chunk_id)
            for table_id in item.get("table_ids") or []:
                self._append_ordered(table_ids, table_id)
            for table_bundle_id in item.get("table_bundle_ids") or []:
                self._append_ordered(table_bundle_ids, table_bundle_id)
            for evidence_unit_id in item.get("evidence_unit_ids") or []:
                self._append_ordered(evidence_unit_ids, evidence_unit_id)
            for page in item.get("selected_pages") or []:
                self._append_page(selected_pages, page)
            for item_id in item.get("selected_ids") or []:
                self._append_ordered(selected_ids, item_id)
            for group_id in item.get("selected_group_ids") or []:
                self._append_ordered(selected_group_ids, group_id)
            for chunk_id in item.get("selected_chunk_ids") or []:
                self._append_ordered(selected_chunk_ids, chunk_id)
            for table_id in item.get("selected_table_ids") or []:
                self._append_ordered(selected_table_ids, table_id)
            for table_bundle_id in item.get("selected_table_bundle_ids") or []:
                self._append_ordered(selected_table_bundle_ids, table_bundle_id)
            for evidence_unit_id in item.get("selected_evidence_unit_ids") or []:
                self._append_ordered(selected_evidence_unit_ids, evidence_unit_id)
        return {
            "pages": pages,
            "ids": ids,
            "group_ids": group_ids,
            "chunk_ids": chunk_ids,
            "table_ids": table_ids,
            "table_bundle_ids": table_bundle_ids,
            "evidence_unit_ids": evidence_unit_ids,
            "selected_pages": selected_pages,
            "selected_ids": selected_ids,
            "selected_group_ids": selected_group_ids,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_table_ids": selected_table_ids,
            "selected_table_bundle_ids": selected_table_bundle_ids,
            "selected_evidence_unit_ids": selected_evidence_unit_ids,
            "by_tool": by_tool,
            "selected_count": selected_count,
            "candidate_count": candidate_count,
        }

    def snapshot_partial_diagnostics(self, fallback_reason: str = "") -> Dict[str, Any]:
        """P1: 在 agent_total_timeout 等中断时，由外层调用拿到当前已累积的 partial diagnostics。

        作用：把 _partial_state 里的 search_history/search_results/fetched_content
        以及当前 self.diagnostics（含 candidate_pool_trace）组合成与
        ``_build_agent_retrieval_diagnostics`` 同结构的字典，使外层能合并到
        ``retrieval_meta.diagnostics.retrieval.candidate_pool``，避免诊断盲点。

        参数:
            fallback_reason: 写入返回 dict 的 ``fallback_reason``，用于 miss
                report 区分 ``agent_timeout_no_candidate_pool``。
        返回:
            ``{"retrieval": {...}, "context_assembly": {...}}``，缺失字段以默认 0/[]/"" 填充
        """
        try:
            partial = self._build_agent_retrieval_diagnostics(
                search_history=list(self._partial_state.get("search_history") or []),
                search_results=list(self._partial_state.get("search_results") or []),
                fetched_content=dict(self._partial_state.get("fetched_content") or {}),
                detail=list(self._partial_state.get("detail") or []),
                context_text=str(self._partial_state.get("context_text") or ""),
                context_budget=dict(self._partial_state.get("context_budget") or {}),
            )
        except Exception as exc:
            logger.warning(f"[RetrievalAgent] snapshot_partial_diagnostics 构建失败: {exc}")
            partial = {
                "retrieval": {"candidate_pool": self._build_candidate_pool_summary()},
                "context_assembly": {},
            }
        if fallback_reason:
            for key in ("retrieval", "context_assembly"):
                section = partial.get(key)
                if isinstance(section, dict):
                    section["fallback_reason"] = fallback_reason
        return partial

    def _build_agent_retrieval_diagnostics(
        self,
        *,
        search_history: List[dict],
        search_results: List[str],
        fetched_content: Dict[str, dict],
        detail: List[dict],
        context_text: str,
        context_budget: Dict[str, Any],
    ) -> Dict[str, Any]:
        success_history = [h for h in search_history if h.get("resultCount", 0) > 0]
        source_counts = Counter(self._tool_source_family(h.get("tool", "")) for h in success_history)
        total_sources = sum(source_counts.values())
        source_entropy = 0.0
        if total_sources > 0:
            for count in source_counts.values():
                p = count / total_sources
                if p > 0:
                    source_entropy -= p * math.log2(p)

        token_limit = max(0, int((context_budget or {}).get("limit_tokens") or 0))
        token_used = max(0, int((context_budget or {}).get("after_tokens") or 0))
        dedup_removed = max(0, int((context_budget or {}).get("dedup_removed") or 0))
        tool_result_dedup_removed = max(0, int(self.diagnostics.get("tool_result_dedup_removed") or 0))

        retrieval_diag = {
            "source_mix": dict(source_counts),
            "source_mix_entropy": round(source_entropy, 4),
            "successful_tool_calls": len(success_history),
            "zero_result_tool_calls": max(0, len(search_history) - len(success_history)),
            "search_result_count": len(search_results),
            "fetched_group_count": len(fetched_content),
            "detail_count": len(detail),
            "forced_initial_search": bool(self.diagnostics.get("forced_initial_search")),
            "initial_search_blueprint_tools": list(self.diagnostics.get("initial_search_blueprint_tools") or []),
            "forced_initial_search_injected_tools": list(self.diagnostics.get("forced_initial_search_injected_tools") or []),
            "forced_initial_search_terms": dict(self.diagnostics.get("forced_initial_search_terms") or {}),
            "forced_initial_search_grep_query": self.diagnostics.get("forced_initial_search_grep_query") or "",
            "default_initial_search_used": self.diagnostics.get("fallback_reason") == "empty_plan_default_search",
            "default_initial_search_tools": list(self.diagnostics.get("default_initial_search_tools") or []),
            "default_initial_search_terms": dict(self.diagnostics.get("default_initial_search_terms") or {}),
            "default_initial_search_grep_query": self.diagnostics.get("default_initial_search_grep_query") or "",
            "rerank_applied": False,
            "final_external_rerank": dict(self.diagnostics.get("final_external_rerank") or {}),
            "dedup_removed": dedup_removed,
            "tool_result_dedup_removed": tool_result_dedup_removed,
            "dedup_ratio": round(dedup_removed / max(len(search_results), 1), 4) if search_results else 0.0,
            "candidate_pool": self._build_candidate_pool_summary(),
        }
        context_diag = {
            "source_mix": dict(source_counts),
            "source_mix_entropy": round(source_entropy, 4),
            "dedup_removed": dedup_removed,
            "dedup_ratio": round(dedup_removed / max(len(search_results), 1), 4) if search_results else 0.0,
            "rerank_applied": False,
            "final_external_rerank": dict(self.diagnostics.get("final_external_rerank") or {}),
            "token_budget_used": token_used,
            "token_budget_limit": token_limit,
            "token_budget_ratio": round(token_used / token_limit, 4) if token_limit else 0.0,
            "tool_result_dedup_removed": tool_result_dedup_removed,
            "parts_before": max(0, int((context_budget or {}).get("parts_before") or 0)),
            "parts_after": max(0, int((context_budget or {}).get("parts_after") or 0)),
            "truncated": bool((context_budget or {}).get("truncated")),
            "context_chars": len(context_text or ""),
            "detail_count": len(detail),
        }
        return {
            "retrieval": retrieval_diag,
            "context_assembly": context_diag,
        }

    def _try_expand_chunk_to_group_fulltext(
        self, chunk: str, expanded_groups: set
    ) -> str:
        """P1: 如果 chunk 属于某个语义组且该组有 full_text，则用 full_text 替换截断片段。

        借鉴 ragflow 的章节感知分块思路：检索到的 chunk 往往是截断的片段，
        而语义组的 full_text 包含完整的段落/章节内容，更适合概述型问题。
        """
        if not self._doc_ctx or not self._doc_ctx.semantic_groups:
            return chunk

        meta = self._extract_tool_chunk_meta(chunk)
        chunk_type = str(meta.get("chunk_type") or "").strip().lower()
        if (
            chunk_type in {"table", "table_row", "table_cell", "caption"}
            or meta.get("table_id")
            or meta.get("table_bundle_id")
            or "[structured table" in str(chunk or "").lower()
        ):
            return chunk

        # 从格式化 chunk 中提取 group_id
        import re as _re
        gid_match = _re.search(r"group_id[:：]\s*(\S+)", chunk[:300])
        if not gid_match:
            return chunk
        gid = gid_match.group(1).strip()
        if not gid or gid in expanded_groups:
            return chunk

        # 查找语义组
        for g in self._doc_ctx.semantic_groups:
            g_id = g.group_id if hasattr(g, "group_id") else g.get("group_id", "")
            if g_id != gid:
                continue
            full_text = (
                getattr(g, "full_text", "") or ""
                if not isinstance(g, dict)
                else g.get("full_text", "") or ""
            )
            if not full_text or len(full_text) < len(chunk) * 1.5:
                # full_text 不存在或不比 chunk 长太多，不替换
                return chunk
            # 用 full_text 替换，保留原始元数据头
            header_end = chunk.find("】\n")
            if header_end > 0:
                header = chunk[: header_end + 2]
                expanded_groups.add(gid)
                return header + full_text
            break
        return chunk

    def _build_search_result_detail(self, chunk: str, index: int) -> Dict[str, Any]:
        meta = self._extract_tool_chunk_meta(chunk)
        text = re.sub(r"\s+", " ", str(meta.get("text") or chunk or "")).strip()
        group_id = str(meta.get("group_id") or "").strip()
        page = meta.get("page")
        parsed_page_range = meta.get("page_range")
        page_range = (
            parsed_page_range
            if isinstance(parsed_page_range, list) and parsed_page_range
            else [page, page] if isinstance(page, int) and page > 0 else []
        )
        context_id = str(meta.get("context_id") or group_id or f"agent-search-{index + 1}").strip()
        evidence_id = str(meta.get("evidence_id") or "").strip()
        if not evidence_id:
            chunk_id = meta.get("chunk_id") or meta.get("child_chunk_id") or meta.get("parent_id")
            evidence_id = f"{context_id}:{chunk_id}" if chunk_id not in (None, "") else f"{context_id}:{index + 1}"
        detail: Dict[str, Any] = {
            "group_id": group_id,
            "context_id": context_id,
            "evidence_id": evidence_id,
            "retrieval_type": "agent_search_result",
            "granularity": "chunk",
            "char_count": len(text),
            "text": text[:1400],
            "page_range": page_range,
        }
        for key in (
            "source",
            "chunk_id",
            "child_chunk_id",
            "parent_id",
            "chunk_type",
            "table_id",
            "table_bundle_id",
            "evidence_unit_id",
            "table_caption",
            "table_header",
            "numeric_table_exact_context_row_text",
            "numeric_table_exact_context_caption",
            "numeric_table_exact_context_header",
            "table_row_boundary_text",
            "table_row_raw_text",
            "table_row_evidence",
            "table_row_slice_kind",
        ):
            value = meta.get(key)
            if value not in (None, ""):
                detail[key] = value
        return detail

    def _extract_final_context_anchor_terms(self, question: str, max_terms: int = 24) -> list[str]:
        """Combine root-question and decomposed sub-question anchors for final context assembly."""
        anchors = _extract_question_anchor_terms(question, max_terms=max_terms)
        seen = {item.casefold() for item in anchors}
        remaining = max(0, max_terms - len(anchors))
        if remaining <= 0:
            return anchors[:max_terms]
        for sub_question in (self.sub_questions or [])[:3]:
            for anchor in _extract_question_anchor_terms(str(sub_question or ""), max_terms=8):
                key = anchor.casefold()
                if key not in seen:
                    seen.add(key)
                    anchors.append(anchor)
                    remaining -= 1
                    if remaining <= 0:
                        return anchors[:max_terms]
        return anchors[:max_terms]

    def _prioritize_anchor_coverage(self, question: str, chunks: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """把覆盖新增问题锚点的证据提前，提升多约束问题的最终上下文召回。

        该步骤只使用问题文本中抽取的通用术语/数字/公式锚点和候选 chunk 文本，
        不依赖论文标题、表号、答案或评估集内容；原始相关性排序仍作为稳定次序。
        """
        anchors = self._extract_final_context_anchor_terms(question, max_terms=16)
        if len(anchors) < 2 or len(chunks) < 2:
            return chunks, {"applied": False, "anchor_count": len(anchors)}

        scored: list[tuple[int, int, int, int, str, list[str]]] = []
        for idx, chunk in enumerate(chunks):
            text = str(chunk or "")
            matched = [
                anchor
                for anchor in anchors
                if technical_anchor_matches(anchor, text)
            ]
            scored.append((len(matched), self._extract_page_no_from_chunk(text), idx, idx, chunk, matched))

        selected_indices: list[int] = []
        covered: set[str] = set()
        remaining = set(range(len(scored)))
        seed_limit = min(len(chunks), max(2, min(len(anchors), 8)))
        while remaining and len(selected_indices) < seed_limit:
            best_idx = None
            best_key = None
            for item_idx in remaining:
                match_count, _page, original_idx, _stable_idx, _chunk, matched = scored[item_idx]
                new_matches = [anchor for anchor in matched if anchor not in covered]
                key = (len(new_matches), match_count, -original_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = item_idx
            if best_idx is None or not best_key or best_key[0] <= 0:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            covered.update(scored[best_idx][5])

        if not selected_indices:
            return chunks, {
                "applied": False,
                "anchor_count": len(anchors),
                "covered_count": 0,
                "missing": anchors,
            }

        ordered_indices = selected_indices + [idx for idx in range(len(chunks)) if idx not in set(selected_indices)]
        reordered = [scored[idx][4] for idx in ordered_indices]
        original_positions = [scored[idx][2] for idx in ordered_indices[: len(selected_indices)]]
        return reordered, {
            "applied": original_positions != sorted(original_positions) or original_positions != list(range(len(original_positions))),
            "anchor_count": len(anchors),
            "covered_count": len(covered),
            "coverage": round(len(covered) / max(len(anchors), 1), 4),
            "covered": [anchor for anchor in anchors if anchor in covered],
            "missing": [anchor for anchor in anchors if anchor not in covered],
            "seed_count": len(selected_indices),
            "seed_original_positions": original_positions,
        }

    def _build_final_context(
        self,
        question: str,
        search_results: List[str],
        fetched_content: Dict[str, dict],
    ) -> tuple:
        """构建最终上下文和详情

        Returns:
            (context_string, detail_list)
        """
        context_parts = []
        context_details: list[Dict[str, Any]] = []

        try:
            from services.embedding_service import filter_reference_trap_texts
        except Exception:
            filter_reference_trap_texts = None

        # 添加搜索结果片段（去重，限制总量）
        seen = set()
        dedup_removed = 0
        expanded_groups: set = set()  # P1: 跟踪已展开的语义组，避免重复
        filtered_search_results = (
            filter_reference_trap_texts(search_results, question)
            if filter_reference_trap_texts is not None
            else search_results
        )
        # P6: evidence-aware rerank - 按 question 对 search_results 做 lexical+semantic rerank。
        # 借鉴 ragflow `retrieval_by_toc` 与 kotaemon `LLMTrulensScoring` 思路，但用本地
        # _tool_result_score 避免额外 LLM 调用，减少相关候选被低质量片段挤出 token budget。
        # P7: rerank 后做 page round-robin，确保 page 多样性优先，避免单页高词频
        # chunk 占满 budget 后挤出其他页的互补证据。借鉴 LangChain `MMR` 与 ragflow
        # `retrieval_with_diversity` 的多样性思路。
        rerank_scores: List[float] = []
        try:
            from services.retrieval_tools import _tool_result_score as _final_rerank_score
            scored = [
                (_final_rerank_score(question, chunk, 0.0), idx, chunk)
                for idx, chunk in enumerate(filtered_search_results)
            ]
            # 稳定排序：按分数降序，分数相同保留原始顺序
            scored.sort(key=lambda x: (-x[0], x[1]))
            rerank_scores = [score for score, _idx, _chunk in scored]
            external_reranked, external_rerank_diag = self._apply_external_rerank(question, scored)
            self.diagnostics["final_external_rerank"] = external_rerank_diag
            if external_rerank_diag.get("applied"):
                scored = external_reranked
                rerank_scores = [score for score, _idx, _chunk in scored]
            # P7: 按 page round-robin 重排
            #   1) 从 chunk 字符串提取 "页码:N" 作 page 标签，没标签的归 page 0
            #   2) 按 page 分组，组内保留 rerank 顺序
            #   3) round-robin 取每 page 第 1/2/3... 个，直到全部输出
            page_pattern = re.compile(r"页码[:：]\s*(\d+)")
            page_groups: Dict[int, List[tuple]] = {}
            page_order: List[int] = []  # 记录 page 首次出现顺序
            for score, idx, chunk in scored:
                match = page_pattern.search(chunk[:300])
                page_no = int(match.group(1)) if match else 0
                if page_no not in page_groups:
                    page_groups[page_no] = []
                    page_order.append(page_no)
                page_groups[page_no].append((score, idx, chunk))
            interleaved: List[str] = []
            slot = 0
            while True:
                emitted_in_round = False
                for page_no in page_order:
                    bucket = page_groups[page_no]
                    if slot < len(bucket):
                        interleaved.append(bucket[slot][2])
                        emitted_in_round = True
                if not emitted_in_round:
                    break
                slot += 1
            filtered_search_results = interleaved
            self.diagnostics["final_rerank_applied"] = True
            self.diagnostics["final_rerank_page_diversity_pages"] = len(page_order)
            if rerank_scores:
                self.diagnostics["final_rerank_top_score"] = round(max(rerank_scores), 4)
                self.diagnostics["final_rerank_median_score"] = round(
                    sorted(rerank_scores)[len(rerank_scores) // 2], 4
                )
                self.diagnostics["final_rerank_count"] = len(rerank_scores)
        except Exception as exc:
            logger.debug(f"[RetrievalAgent] final rerank 失败，按原顺序构建: {exc}")
            self.diagnostics["final_rerank_applied"] = False
            self.diagnostics.setdefault("final_external_rerank", {
                "applied": False,
                "provider": self.rerank_provider or ("local" if self.use_rerank else ""),
                "model": self.reranker_model or ("BAAI/bge-reranker-base" if self.use_rerank else ""),
                "input_count": 0,
                "output_count": 0,
                "top_score": 0.0,
                "median_score": 0.0,
                "min_score": 0.0,
                "elapsed_ms": 0.0,
                "error": str(exc),
                "cache_hit": False,
            })
        filtered_search_results, anchor_seed_stats = self._prioritize_anchor_coverage(
            question,
            filtered_search_results,
        )
        self.diagnostics["final_anchor_coverage_seed"] = anchor_seed_stats
        for chunk in filtered_search_results:
            chunk_meta = self._extract_tool_chunk_meta(chunk)
            chunk_key = _context_part_dedupe_key(chunk_meta, str(chunk_meta.get("text") or chunk or ""))
            if chunk_key in seen:
                dedup_removed += 1
                continue
            seen.add(chunk_key)
            # P1: 尝试将截断 chunk 替换为其所属语义组的 full_text
            expanded = self._try_expand_chunk_to_group_fulltext(chunk, expanded_groups)
            context_parts.append(expanded)
            context_details.append(self._build_search_result_detail(expanded, len(context_details)))

        # 添加意群内容（按 gid 跨轮去重）
        seen_groups: set = set()
        for gid, data in fetched_content.items():
            # 跳过已合并到上下文的 group，避免同一语义组重复出现
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            group_text = data["text"]
            if filter_reference_trap_texts is not None:
                filtered_group = filter_reference_trap_texts([group_text], question)
                if not filtered_group:
                    continue
                group_text = filtered_group[0]
            context_parts.append(f"【{gid} - {data['granularity']}】\n{group_text}")
            context_id = str(data.get("context_id") or gid or f"agent-fetch-{len(context_details) + 1}").strip()
            evidence_id = str(
                data.get("evidence_id")
                or f"{context_id}:{data['granularity']}"
            ).strip()
            context_details.append({
                "group_id": gid,
                "context_id": context_id,
                "evidence_id": evidence_id,
                "retrieval_type": "agent_fetch_group",
                "granularity": data["granularity"],
                "char_count": len(group_text),
                "text": group_text[:1400],
                "page_range": data.get("page_range") or [],
                "keywords": data.get("keywords") or [],
            })

        context_parts, context_details, unified_anchor_stats = self._prioritize_context_parts_by_anchor_coverage(
            question,
            context_parts,
            context_details,
        )
        self.diagnostics["final_unified_anchor_coverage"] = unified_anchor_stats

        raw_before_tokens = self._estimate_tokens("\n\n".join(context_parts))
        context_parts, page_seed_stats = self._compact_page_seeds_for_budget(context_parts)
        if page_seed_stats.get("compacted_count"):
            self.diagnostics["final_context_page_seed_compaction"] = page_seed_stats
            compacted_indices = set(page_seed_stats.get("compacted_indices") or [])
            for idx in compacted_indices:
                if 0 <= idx < len(context_parts) and idx < len(context_details):
                    item = dict(context_details[idx])
                    item["text"] = re.sub(r"\s+", " ", context_parts[idx]).strip()[:1400]
                    item["char_count"] = len(context_parts[idx])
                    item["compacted"] = True
                    context_details[idx] = item
        before_tokens = self._estimate_tokens("\n\n".join(context_parts))
        total_tokens = 0
        trimmed_parts = []
        trimmed_detail: list[Dict[str, Any]] = []
        budget_anchor_terms = self._extract_final_context_anchor_terms(question, max_terms=16)
        budget_covered_anchors: set[str] = set()
        budget_skipped_parts = 0
        deferred_truncation: tuple[int, str, set[str], int] | None = None
        for idx, part in enumerate(context_parts):
            part_tokens = self._estimate_tokens(part)
            if total_tokens + part_tokens > self.max_context_tokens:
                remaining = self.max_context_tokens - total_tokens
                part_matches = {
                    anchor
                    for anchor in budget_anchor_terms
                    if technical_anchor_matches(anchor, part)
                }
                adds_new_anchor = bool(part_matches - budget_covered_anchors)
                if remaining > 100 and (adds_new_anchor or not trimmed_parts):
                    # 先暂存可截断候选，继续扫描后续短证据；短证据若能完整进入上下文，
                    # 往往比立即截断一个长段更有利于保留多锚点证据。
                    new_anchor_count = len(part_matches - budget_covered_anchors)
                    if (
                        deferred_truncation is None
                        or new_anchor_count > len(deferred_truncation[2] - budget_covered_anchors)
                    ):
                        deferred_truncation = (idx, part, set(part_matches), remaining)
                    budget_skipped_parts += 1
                    continue
                budget_skipped_parts += 1
                continue
            trimmed_parts.append(part)
            if idx < len(context_details):
                trimmed_detail.append(context_details[idx])
            for anchor in budget_anchor_terms:
                if technical_anchor_matches(anchor, part):
                    budget_covered_anchors.add(anchor)
            total_tokens += part_tokens

        if deferred_truncation is not None:
            idx, part, part_matches, _remaining_at_deferral = deferred_truncation
            remaining = self.max_context_tokens - total_tokens
            if remaining > 100 and (part_matches - budget_covered_anchors or not trimmed_parts):
                trimmed = self._trim_to_tokens(part, remaining) + "...(截断)"
                trimmed_parts.append(trimmed)
                item = dict(context_details[idx]) if idx < len(context_details) else {}
                if item:
                    item["text"] = re.sub(r"\s+", " ", trimmed).strip()[:1400]
                    item["char_count"] = len(trimmed)
                    item["truncated"] = True
                    trimmed_detail.append(item)
                budget_covered_anchors.update(part_matches)

        context_string = "\n\n".join(trimmed_parts)
        detail = trimmed_detail
        budget = {
            "limit_tokens": self.max_context_tokens,
            "raw_before_tokens": raw_before_tokens,
            "before_tokens": before_tokens,
            "after_tokens": self._estimate_tokens(context_string),
            "truncated": raw_before_tokens > self.max_context_tokens,
            "parts_before": len(context_parts),
            "parts_after": len(trimmed_parts),
            "dedup_removed": dedup_removed,
            "raw_search_results": len(search_results),
            "filtered_search_results": len(filtered_search_results),
            "page_seed_compacted": int(page_seed_stats.get("compacted_count") or 0),
            "detail_before_trim": len(context_details),
            "detail_after_trim": len(detail),
            "budget_skipped_parts": budget_skipped_parts,
            "budget_anchor_covered_count": len(budget_covered_anchors),
        }
        return context_string, detail, budget

    def _prioritize_context_parts_by_anchor_coverage(
        self,
        question: str,
        context_parts: List[str],
        context_details: List[Dict[str, Any]],
    ) -> tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """在最终预算裁剪前统一重排 search/fetch 证据。

        只依据问题中的通用术语、数字、公式锚点与证据文本的匹配情况，让能覆盖
        新锚点的证据先进入上下文。这样 fetch(group) 得到的完整意群也能与普通
        search chunk 公平竞争 token budget，而不是固定排在后面。
        """
        anchors = self._extract_final_context_anchor_terms(question, max_terms=16)
        if len(anchors) < 2 or len(context_parts) < 2:
            return context_parts, context_details, {"applied": False, "anchor_count": len(anchors)}
        try:
            numeric_table_query = "numeric_table" in (analyze_evidence_need(question) or [])
        except Exception:
            numeric_table_query = bool(re.search(r"\btable\s*\d+\b|表\s*\d+", question or "", re.I))

        scored: list[dict[str, Any]] = []
        for idx, part in enumerate(context_parts):
            text = str(part or "")
            matched = [
                anchor
                for anchor in anchors
                if technical_anchor_matches(anchor, text)
            ]
            detail = context_details[idx] if idx < len(context_details) else {}
            chunk_type = str((detail or {}).get("chunk_type") or "").strip().lower()
            table_evidence = (
                chunk_type in {"table", "table_row", "table_cell"}
                or bool((detail or {}).get("table_id") or (detail or {}).get("table_bundle_id"))
                or "[structured table" in text.lower()
            )
            scored.append({
                "idx": idx,
                "part": part,
                "detail": detail,
                "matched": matched,
                "match_count": len(matched),
                "page": self._extract_page_no_from_chunk(text),
                "retrieval_type": str((detail or {}).get("retrieval_type") or ""),
                "table_evidence": table_evidence,
            })

        selected: list[int] = []
        covered: set[str] = set()
        remaining = set(range(len(scored)))
        seed_limit = min(len(scored), max(2, min(len(anchors), 8)))
        while remaining and len(selected) < seed_limit:
            best_idx = None
            best_key = None
            for item_idx in remaining:
                item = scored[item_idx]
                new_matches = [anchor for anchor in item["matched"] if anchor not in covered]
                # fetch_group 往往是更完整的语义单元；但数值表格题更需要表格/表行
                # 证据进入最终上下文，避免 digest 把精确数值证据挤出 RAGAS/context_segments。
                table_bonus = 2 if numeric_table_query and item.get("table_evidence") else 0
                fetch_bonus = 1 if item["retrieval_type"] == "agent_fetch_group" and not numeric_table_query else 0
                key = (len(new_matches), item["match_count"], table_bonus, fetch_bonus, -int(item["idx"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = item_idx
            if best_idx is None or not best_key or best_key[0] <= 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            covered.update(scored[best_idx]["matched"])

        if not selected:
            return context_parts, context_details, {
                "applied": False,
                "anchor_count": len(anchors),
                "covered_count": 0,
                "missing": anchors,
            }

        selected_set = set(selected)
        ordered = selected + [idx for idx in range(len(scored)) if idx not in selected_set]
        reordered_parts = [scored[idx]["part"] for idx in ordered]
        reordered_details = [scored[idx]["detail"] for idx in ordered if scored[idx]["detail"]]
        original_positions = [int(scored[idx]["idx"]) for idx in ordered[: len(selected)]]
        return reordered_parts, reordered_details, {
            "applied": original_positions != sorted(original_positions) or original_positions != list(range(len(original_positions))),
            "anchor_count": len(anchors),
            "covered_count": len(covered),
            "coverage": round(len(covered) / max(len(anchors), 1), 4),
            "covered": [anchor for anchor in anchors if anchor in covered],
            "missing": [anchor for anchor in anchors if anchor not in covered],
            "seed_count": len(selected),
            "seed_original_positions": original_positions,
        }

    def _extract_page_no_from_chunk(self, text: str) -> int:
        """从检索证据头中提取页码；没有页码时返回 0。"""
        if not text:
            return 0
        match = re.search(r"页码[:：]\s*(\d+)", text[:300])
        if not match:
            return 0
        try:
            return max(0, int(match.group(1)))
        except (TypeError, ValueError):
            return 0

    def _trim_evidence_chunk_to_tokens(self, text: str, limit_tokens: int) -> str:
        """截断检索证据时保留首行元数据，避免引用 ID/page 信息丢失。"""
        if self._estimate_tokens(text) <= limit_tokens:
            return text
        lines = (text or "").splitlines()
        if lines and lines[0].startswith("【检索证据"):
            header = lines[0]
            body = "\n".join(lines[1:]).strip()
            remaining = limit_tokens - self._estimate_tokens(header)
            if remaining > 80 and body:
                return f"{header}\n{self._trim_to_tokens(body, remaining)}...(页种子截断)"
        return self._trim_to_tokens(text, limit_tokens) + "...(页种子截断)"

    def _compact_page_seeds_for_budget(self, context_parts: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """对不同页的首条证据做预算保护。

        当候选池已经覆盖多个相关页时，最终上下文或引用窗口仍可能被少数
        长 chunk 挤占。这里仅压缩每个 page 的第一条检索证据，不改变页间
        round-robin 顺序，保证更多页能进入 token budget，同时保留证据头里的
        page/chunk/context ID。
        """
        raw_tokens = self._estimate_tokens("\n\n".join(context_parts))
        if raw_tokens <= self.max_context_tokens or not context_parts:
            return context_parts, {"applied": False, "compacted_count": 0}

        pages = [self._extract_page_no_from_chunk(part) for part in context_parts]
        unique_pages = [p for p in dict.fromkeys(p for p in pages if p > 0)]
        if len(unique_pages) <= 1:
            return context_parts, {"applied": False, "compacted_count": 0}

        seed_cap = max(160, min(900, int(self.max_context_tokens * 0.18)))
        seen_pages: set[int] = set()
        compacted: List[str] = []
        compacted_pages: List[int] = []
        compacted_indices: List[int] = []
        for idx, (part, page_no) in enumerate(zip(context_parts, pages)):
            if page_no > 0 and page_no not in seen_pages:
                seen_pages.add(page_no)
                trimmed = self._trim_evidence_chunk_to_tokens(part, seed_cap)
                if trimmed != part:
                    compacted_pages.append(page_no)
                    compacted_indices.append(idx)
                compacted.append(trimmed)
            else:
                compacted.append(part)

        return compacted, {
            "applied": bool(compacted_pages),
            "compacted_count": len(compacted_pages),
            "compacted_pages": compacted_pages,
            "compacted_indices": compacted_indices,
            "unique_pages": len(unique_pages),
            "seed_token_cap": seed_cap,
            "raw_before_tokens": raw_tokens,
            "after_compaction_tokens": self._estimate_tokens("\n\n".join(compacted)),
        }

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = max(0, len(text) - chinese_chars)
        return int(chinese_chars * 1.5 + other_chars * 0.25) + 1

    def _trim_to_tokens(self, text: str, limit_tokens: int) -> str:
        if self._estimate_tokens(text) <= limit_tokens:
            return text
        approx_chars = max(200, int(limit_tokens * 2.5))
        trimmed = text[:approx_chars]
        while trimmed and self._estimate_tokens(trimmed) > limit_tokens:
            trimmed = trimmed[: max(1, int(len(trimmed) * 0.85))]
        return trimmed
