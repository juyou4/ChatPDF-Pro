"""统一聊天意图合同。

该模块只负责描述用户想做什么，不负责生成检索查询或执行检索。
检索模板、框选文本和术语扩展不得回流并改变这里的判定结果。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Literal, Sequence

from services.modal_asset_service import detect_query_modalities
from services.query_analyzer import get_retrieval_strategy


INTENT_DECISION_VERSION = "v1"

InteractionMode = Literal[
    "default",
    "selection",
    "image",
    "preset",
    "retry_failed_turn",
]
IntentTask = Literal[
    "qa",
    "summarize",
    "extract",
    "explain",
    "compare",
    "calculate",
    "translate",
    "continue",
]
IntentScope = Literal["document", "section", "page", "selection", "image"]
RoutePolicy = Literal["off", "auto", "force"]
GraphMode = Literal["local", "global", "hybrid"]


_SUMMARY_RE = re.compile(
    r"(?:总结|概括|概述|简述|大意|主要内容|讲了什么|讲什么|"
    r"\b(?:summary|summarize|overview|outline|main\s+idea)\b)",
    re.IGNORECASE,
)
_TRANSLATE_RE = re.compile(r"(?:翻译|译成|中译|英译|\btranslat(?:e|ion)\b)", re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"(?:比较|对比|相比|区别|差异|异同|优缺点|利弊|"
    r"\b(?:compare|comparison|difference|versus|vs\.?|pros?\s+and\s+cons?)\b)",
    re.IGNORECASE,
)
_CALCULATE_RE = re.compile(
    r"(?:计算(?!机)|求和|平均值|差多少|高多少|低多少|提升多少|下降多少|百分点|"
    r"\b(?:calculate|compute|sum|average|delta|difference\s+between|percentage\s+point)\b)",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"(?:解释|说明|讲解|介绍|为什么|为何|原因|原理|机制|如何|怎么|含义|意思|"
    r"\b(?:explain|why|how|reason|mechanism|principle|meaning)\b)",
    re.IGNORECASE,
)
_CONTINUE_RE = re.compile(
    r"^\s*(?:继续|接着说|继续上一个回答|展开一点|go\s+on|continue)\s*[。.!！?？]*\s*$",
    re.IGNORECASE,
)
_PAGE_SCOPE_RE = re.compile(
    r"(?:第\s*\d+\s*页|\bpage\s*\d+\b|\bp\.?\s*\d+\b)",
    re.IGNORECASE,
)
_SECTION_SCOPE_RE = re.compile(
    r"(?:第\s*[\d一二三四五六七八九十]+\s*(?:章|节)|章节|部分|附录|"
    r"方法部分|实验部分|相关工作|结论部分|"
    r"\b(?:section|chapter|appendix|methodology|related\s+work)\b)",
    re.IGNORECASE,
)
_AMBIGUOUS_REFERENCE_RE = re.compile(
    r"(?:这个|那个|它|这块|那块|这部分|那部分|这里|那里|前者|后者|第二个|"
    r"\b(?:it|this|that|the\s+former|the\s+latter|the\s+second\s+one)\b)",
    re.IGNORECASE,
)
_DOCUMENT_SELF_REFERENCE_RE = re.compile(
    r"(?:这篇(?:论文|文章)|本文|该(?:论文|文章|文档)|当前文档|"
    r"\bthis\s+(?:paper|article|document)\b|\bthe\s+(?:paper|article|document)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentDecision:
    """一次聊天请求唯一、不可变的语义判定。"""

    intent_id: str
    version: str
    original_question: str
    intent_question: str
    interaction_mode: InteractionMode
    task: IntentTask
    scope: IntentScope
    modalities: tuple[str, ...]
    query_type: str
    evidence_need: tuple[str, ...]
    top_k: int
    reasoning: str
    agent_policy: RoutePolicy
    web_policy: RoutePolicy
    graph_mode: GraphMode
    confidence: float
    is_ambiguous: bool
    clarification_question: str
    matched_rules: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["modalities"] = list(self.modalities)
        result["evidence_need"] = list(self.evidence_need)
        result["matched_rules"] = list(self.matched_rules)
        return result

    def to_retrieval_strategy(self) -> dict:
        """兼容现有 query_type/evidence_need/top_k 消费者。"""
        return {
            "query_type": self.query_type,
            "evidence_need": list(self.evidence_need),
            "top_k": self.top_k,
            "reasoning": self.reasoning,
            "intent_id": self.intent_id,
            "intent_version": self.version,
        }


@dataclass(frozen=True)
class ChatTurnContext:
    """贯穿单次聊天的查询身份，区分意图问题与检索问题。"""

    original_question: str
    effective_question: str
    intent_question: str
    retrieval_query: str
    intent: IntentDecision
    parse_generation: str = ""
    document_source_hash: str = ""

    def with_retrieval_query(self, retrieval_query: str) -> "ChatTurnContext":
        return replace(self, retrieval_query=str(retrieval_query or self.intent_question).strip())

    def to_meta(self) -> dict:
        return {
            "original_question": self.original_question,
            "effective_question": self.effective_question,
            "intent_question": self.intent_question,
            "retrieval_query": self.retrieval_query,
            "parse_generation": self.parse_generation,
            "document_source_hash": self.document_source_hash,
            "intent": self.intent.to_dict(),
        }


def normalize_route_policy(value: object, *, enabled: bool = False) -> RoutePolicy:
    normalized = str(value or "").strip().lower()
    if normalized in {"off", "auto", "force"}:
        return normalized  # type: ignore[return-value]
    return "auto" if enabled else "off"


def infer_interaction_mode(
    *,
    explicit_mode: object = "",
    selected_text: object = "",
    has_images: bool = False,
    retry_resolved: bool = False,
) -> InteractionMode:
    normalized = str(explicit_mode or "").strip().lower()
    # 请求中的实际附件/框选/重试状态比可选提示字段更可靠。
    if has_images:
        return "image"
    if str(selected_text or "").strip():
        return "selection"
    if retry_resolved:
        return "retry_failed_turn"
    if normalized in {"default", "selection", "image", "preset", "retry_failed_turn"}:
        return normalized  # type: ignore[return-value]
    return "default"


def prepare_chat_intent(
    *,
    original_question: str,
    intent_question: str | None = None,
    interaction_mode: object = "",
    selected_text: object = "",
    has_images: bool = False,
    retry_resolved: bool = False,
    enable_agent: bool = False,
    force_agent: bool = False,
    enable_web: bool = False,
    web_policy: object = "",
) -> IntentDecision:
    """基于已经完成上下文消歧的问题生成一次性意图判定。"""
    original = str(original_question or "").strip()
    question = str(intent_question or original).strip() or original
    mode = infer_interaction_mode(
        explicit_mode=interaction_mode,
        selected_text=selected_text,
        has_images=has_images,
        retry_resolved=retry_resolved,
    )
    strategy = get_retrieval_strategy(question)
    evidence_need = tuple(_dedupe_strings(strategy.get("evidence_need") or []))
    query_type = str(strategy.get("query_type") or "specific")
    modalities = tuple(detect_query_modalities(question))
    task, task_rule = _infer_task(question, query_type, retry_resolved=retry_resolved)
    scope, scope_rule = _infer_scope(question, mode)
    agent_policy = "force" if force_agent else ("auto" if enable_agent else "off")
    normalized_web_policy = normalize_route_policy(web_policy, enabled=enable_web)
    graph_mode = _infer_graph_mode(task, query_type, evidence_need)

    ambiguous_reference = bool(_AMBIGUOUS_REFERENCE_RE.search(question))
    is_ambiguous = bool(
        mode == "default"
        and ambiguous_reference
        and not _DOCUMENT_SELF_REFERENCE_RE.search(question)
        and len(question) <= 32
    )
    matched_rules = _dedupe_strings([
        f"interaction:{mode}",
        task_rule,
        scope_rule,
        *(f"modality:{item}" for item in modalities if item != "text"),
        *(f"evidence:{item}" for item in evidence_need),
    ])
    if is_ambiguous:
        confidence = 0.35
        clarification_question = "你说的“这个”具体指文档中的哪项内容？可以补充名称、章节或页码。"
        matched_rules.append("ambiguity:unresolved_reference")
    elif mode in {"selection", "image", "retry_failed_turn"}:
        confidence = 0.98
        clarification_question = ""
    elif task != "qa" or evidence_need:
        confidence = 0.9
        clarification_question = ""
    else:
        confidence = 0.65
        clarification_question = ""

    identity_payload = {
        "version": INTENT_DECISION_VERSION,
        "question": question,
        "interaction_mode": mode,
        "task": task,
        "scope": scope,
        "modalities": modalities,
        "query_type": query_type,
        "evidence_need": evidence_need,
        "agent_policy": agent_policy,
        "web_policy": normalized_web_policy,
    }
    intent_id = hashlib.sha256(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return IntentDecision(
        intent_id=intent_id,
        version=INTENT_DECISION_VERSION,
        original_question=original,
        intent_question=question,
        interaction_mode=mode,
        task=task,
        scope=scope,
        modalities=modalities,
        query_type=query_type,
        evidence_need=evidence_need,
        top_k=int(strategy.get("top_k") or 10),
        reasoning=str(strategy.get("reasoning") or ""),
        agent_policy=agent_policy,  # type: ignore[arg-type]
        web_policy=normalized_web_policy,
        graph_mode=graph_mode,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        clarification_question=clarification_question,
        matched_rules=tuple(matched_rules),
    )


def build_chat_turn_context(
    *,
    original_question: str,
    effective_question: str,
    intent_question: str,
    intent: IntentDecision,
    retrieval_query: str = "",
    parse_identity: dict | None = None,
) -> ChatTurnContext:
    identity = parse_identity if isinstance(parse_identity, dict) else {}
    return ChatTurnContext(
        original_question=str(original_question or "").strip(),
        effective_question=str(effective_question or original_question or "").strip(),
        intent_question=str(intent_question or effective_question or original_question or "").strip(),
        retrieval_query=str(retrieval_query or intent_question or effective_question or original_question or "").strip(),
        intent=intent,
        parse_generation=str(
            identity.get("parse_generation") or identity.get("generation") or ""
        ).strip(),
        document_source_hash=str(
            identity.get("document_source_hash") or identity.get("source_hash") or ""
        ).strip(),
    )


def _infer_task(question: str, query_type: str, *, retry_resolved: bool) -> tuple[IntentTask, str]:
    if _TRANSLATE_RE.search(question):
        return "translate", "task:translate"
    if _CONTINUE_RE.fullmatch(question) and not retry_resolved:
        return "continue", "task:continue"
    if _COMPARE_RE.search(question):
        return "compare", "task:compare"
    if _CALCULATE_RE.search(question):
        return "calculate", "task:calculate"
    if _SUMMARY_RE.search(question):
        return "summarize", "task:summarize"
    if _EXPLAIN_RE.search(question):
        return "explain", "task:explain"
    if query_type == "extraction":
        return "extract", "task:extract"
    return "qa", "task:qa"


def _infer_scope(question: str, mode: InteractionMode) -> tuple[IntentScope, str]:
    if mode == "image":
        return "image", "scope:image"
    if mode == "selection":
        return "selection", "scope:selection"
    if _PAGE_SCOPE_RE.search(question):
        return "page", "scope:page"
    if _SECTION_SCOPE_RE.search(question):
        return "section", "scope:section"
    return "document", "scope:document"


def _infer_graph_mode(
    task: IntentTask,
    query_type: str,
    evidence_need: Sequence[str],
) -> GraphMode:
    needs = set(evidence_need)
    if task == "summarize" and query_type == "overview":
        return "global"
    if (
        task in {"explain", "compare"}
        or query_type == "analytical"
        or needs & {"section_explanation", "analysis_explanation", "comparison_multi_aspect"}
    ):
        return "hybrid"
    return "local"


def _dedupe_strings(items: Sequence[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
