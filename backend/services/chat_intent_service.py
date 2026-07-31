"""统一聊天意图合同。

该模块只负责描述用户想做什么，不负责生成检索查询或执行检索。
检索模板、框选文本和术语扩展不得回流并改变这里的判定结果。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import inspect
import json
import re
from typing import Literal, Sequence

from services.block_inventory_service import detect_inventory_kinds
from services.modal_asset_service import detect_query_modalities
from services.query_analyzer import count_content_terms, get_retrieval_strategy


INTENT_DECISION_VERSION = "v3"
INTENT_TRACE_VERSION = "intent_trace_v2"
# rule_version 覆盖的规则模块：其中任何一处正则/常量改动都会让 hash 变化。
_RULE_SOURCE_MODULES = (
    "services.chat_intent_service",
    "services.query_analyzer",
    "services.modal_asset_service",
)
# 惰性缓存：import 期不读任何文件，第一次真正需要 rule_version 时才取源码。
_RULE_VERSION_CACHE: str | None = None

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
    "inventory",
]
IntentScope = Literal["document", "section", "page", "selection", "image"]
RoutePolicy = Literal["off", "auto", "force"]
GraphMode = Literal["local", "global", "hybrid"]


_SUMMARY_RE = re.compile(
    r"(?:总结|概括|概述|简述|大意|主要内容|讲了什么|讲什么|"
    r"\b(?:summary|summarize|overview|outline|main\s+idea)\b)",
    re.IGNORECASE,
)
# 「译为」原本只写在 _TRANSLATION_TARGET_RE 的触发词表里，这里却漏了，
# 于是「把这一段译为德语」根本建不起 translate 操作，目标语抽取也就永远不会被调用。
# 两张表必须覆盖同一组触发词。
_TRANSLATE_RE = re.compile(r"(?:翻译|译成|译为|中译|英译|\btranslat(?:e|ion)\b)", re.IGNORECASE)
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
# 页码定位器。同一个语义有多种写法时，按 RAGFlow QUESTION_PATTERN
# (rag/nlp/__init__.py:75-87) 的做法把写法**显式列成一张模式表**、最长最具体的
# 形态排最前，而不是堆一条带层层可选组的巨型正则——后者在「第3页到第5页」上
# 会先匹配失败再回溯出错误切分。
#
# 两条不变量决定了这两个正则的边界，放宽任何一条都会踩已满分的类目：
#   1) 中文侧一律以「页」收尾。这是与 _SECTION_SCOPE_RE 划界的唯一凭据：
#      「第3章到第5章」必须完全不命中，否则 _infer_scope 会把 section 抢成 page。
#   2) 英文侧必须先出现 pages/page/pp/p 关键词才认数字。裸的「3-5」「Top-1」
#      不是页码；这也是「第 5 页表 2」「Table 1 on page 4」两条 CI 硬闸
#      (numeric_table_guard) 不被表号/图号污染的原因。
# alternation 内 pages 必须排在 page 前、pp 必须排在 p 前：正则是最左优先，
# 反了就只吃到 page 再卡在剩下的 s 上（en_inventory_page_scope_004 的原始病因）。
_PAGE_SCOPE_RE = re.compile(
    r"(?:第\s*\d+\s*页|\b(?:pages|page|pp|p)\s*\.?\s*\d+\b)",
    re.IGNORECASE,
)
_PAGE_RANGE_RE = re.compile(
    # 中文四种写法，两端都带「页」的最长形态排最前：
    #   第3页到第5页 / 第3页-第5页 → 第3到5页 / 第3-5页 / 第3到第5页 → 3到5页
    r"第\s*(\d+)\s*页\s*(?:到|至|[-~])\s*第?\s*(\d+)\s*页"
    r"|第\s*(\d+)\s*(?:到|至|[-~])\s*第?\s*(\d+)\s*页"
    r"|(\d+)\s*(?:到|至)\s*(\d+)\s*页"
    # 英文：pages/page/pp./p. + to/through/连字符。分隔符写成 \bto\b 而不是裸 to，
    # 免得「page 4 into Korean」(en_translate_target_003) 多出一个故障源。
    r"|\b(?:pages|page|pp|p)\s*\.?\s*(\d+)\s*(?:\bto\b|\bthrough\b|[-~])\s*"
    r"(?:pages|page|pp|p)?\s*\.?\s*(\d+)\b",
    re.IGNORECASE,
)
# 解析前先把分隔符变体折成一种窄形态（paper-qa types.py:959-964 的 `--`→`-`
# 思路），比在字符类里无限堆 Unicode 变体可控得多。
# 只做 1:1 字符映射 + 折叠连续连字符，**不删空格**：paper-qa 敢 replace(" ","")
# 是因为它的输入是 bibtex 的 pages 字段值，放到自然语言问句上会把
# 「pages 3 to 5」压成「pages3to5」，英文词边界 \b 全毁。
_PAGE_CHAR_TRANSLATION = str.maketrans(
    {
        **{chr(0xFF10 + offset): str(offset) for offset in range(10)},  # 全角数字０-９
        "－": "-",  # U+FF0D 全角连字符
        "—": "-",  # U+2014 em dash
        "–": "-",  # U+2013 en dash
        "−": "-",  # U+2212 minus
        "‐": "-",  # U+2010
        "‑": "-",  # U+2011
        "―": "-",  # U+2015
        "～": "~",  # U+FF5E 全角波浪
        "〜": "~",  # U+301C wave dash
    }
)
_REPEATED_DASH_RE = re.compile(r"-{2,}")
_SECTION_SCOPE_RE = re.compile(
    r"(?:第\s*[\d一二三四五六七八九十]+\s*(?:章|节)|章节|部分|附录|"
    r"方法部分|实验部分|相关工作|结论部分|"
    r"\b(?:section|chapter|appendix|methodology|related\s+work)\b)",
    re.IGNORECASE,
)
# 指代线索。这只是「可能有回指」的信号，不等于歧义——最终是否需要澄清由
# prepare_chat_intent 里的可检索性闸门决定（见 count_content_terms）。
#
# 中文没有词边界，裸字命中会把词内夹字当成指代：「应该」的「该」、「尤其/其他/
# 其实」的「其」。这是第二道防线：这类字面上就不成立的匹配必须在正则层先排掉，
# 不能全指望闸门兜底。
#   - 「该」的后缀改必填并扩表，前置 (?<!应)(?<!活) 排除「应该/活该」；
#   - 裸「其」加 (?!中|他|它|她|余|次|实|间|所) 与 (?<!尤)(?<!与)(?<!极)，
#     真代词用法（「其推理开销」）仍然命中。
# 另补齐两处漏判：中文复数人称代词（英文侧的 they/them 走 \b 词边界，对中文无效）
# 与 one 类回指（原来只硬编码了 the second one 一种）。
_AMBIGUOUS_REFERENCE_RE = re.compile(
    r"(?:这个|那个|这些|那些|它们?|他们|她们|这块|那块|这部分|那部分|这里|那里|上述|"
    r"(?<!应)(?<!活)该(?:方法|模型|模块|结果|部分|文|论文|工作|章节|表|图|数据集|实验|方向)|"
    r"(?<!尤)(?<!与)(?<!极)其(?!中|他|它|她|余|次|实|间|所)|"
    r"前者|后者|第二个|另一个|另外一个|"
    r"\b(?:it|this|that|they|them|he|she|the\s+former|the\s+latter|"
    r"(?:the\s+(?:first|second|third|last|other|next|previous)|which)\s+ones?)\b)",
    re.IGNORECASE,
)
# 自指：指代对象就是当前文档本身，单文档语境下不存在歧义。
# 原来只认 this paper/article/document，漏掉了学术问句里最常见的
# this work / this study / this method / the authors / these results。
_DOCUMENT_SELF_REFERENCE_RE = re.compile(
    r"(?:这篇(?:论文|文章|工作)|这个工作|本文|本论文|该(?:论文|文章|文档|工作)|当前文档|"
    r"\bthis\s+(?:paper|article|document|work|study|approach|method|model|"
    r"framework|system|dataset|architecture|technique)\b|"
    r"\bthe\s+(?:paper|article|document|authors?)\b|"
    r"\bthese\s+(?:results|experiments|findings)\b)",
    re.IGNORECASE,
)
# 纯推进语：没有任何检索内容的追问（「再说说」「然后呢」「Tell me more」）。
# 这类问句连指代词都没有，_AMBIGUOUS_REFERENCE_RE 完全看不见，但它们同样
# 离开上一轮就无法检索。用 fullmatch 而不是 search：「继续讲解注意力机制」
# 是有明确目标的请求，不能落进来。
_BARE_FOLLOW_UP_RE = re.compile(
    r"\s*(?:请|麻烦)?\s*(?:"
    r"继续(?:说说|讲讲|说|讲)?|接着(?:说说|讲讲|说|讲)?|再(?:说说|讲讲|说|讲|来点|多说点)|"
    r"然后(?:呢|吧)?|还有(?:呢|吗)?|多说(?:一点|一些|点)?|展开(?:说说|讲讲|一点)?|"
    r"tell\s+me\s+more|more(?:\s+details?|\s+please)?|what\s+else|anything\s+else|"
    r"go\s+on|and\s+then|keep\s+going"
    r")\s*[。.!！?？,，]*\s*",
    re.IGNORECASE,
)
_EXPLICIT_DOCUMENT_EVIDENCE_RE = re.compile(
    r"(?:本文|文档|论文|文章|第\s*\d+\s*页|表\s*\d+|图\s*\d+|"
    r"\b(?:this|the)\s+(?:paper|article|document)\b|"
    r"\b(?:table|figure|fig\.?|page)\s*\d+\b)",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"[，,。；;、\n]|(?:而是|但是|但)|\b(?:instead|rather\s+than)\b",
    re.IGNORECASE,
)
# 否定词后面允许粘连任意多个口水副词（"不要只是总结"/"don't just summarize"），
# 不再穷举组合。但必须锚到子句前缀末尾（`\s*$`）：否定线索要紧邻被否定的动作词，
# 中间只允许粘连词。否则否定的是状语而不是动词——"不需要很详细地总结"是"要总结，
# 但别太细"，"不要用英文总结"是"用中文总结"，两者都不是禁止总结。
#
# 中英语序相反，必须分开处理：
#   中文修饰语在动词【前】——"不需要*很详细地*总结"，所以否定线索必须锚到前缀末尾
#   （`\s*$`，中间只允许粘连副词）；隔着状语就说明否定的是状语而不是动词。
#   英文修饰语在动词【后】——"don't summarize *too briefly*"，前缀里剩下的是
#   "don't give me an" / "I don't want a" 这类，不能锚定，否则否定会整个失效。
#
# 「跳过/skip the」是不带否定词的禁止说法，必须锚到前缀末尾（`\s*$`）——否则
# "Skip the summary and translate the abstract" 里的 translate 也会跟着被否定。
_NEGATION_CUE_RE = re.compile(
    r"(?:不要|不用|别|不必|无需|不需要|禁止|不能)"
    r"(?:\s*(?:帮我|给我|再|去|给|只是|仅仅|单纯|一味|简单地|光|只))*\s*$"
    r"|(?:跳过|略过|不看)\s*$"
    r"|(?:not\s+to|do\s+not|don't|doesn't|won't|no\s+need\s+to)\b"
    r"|\b(?:skip|omit)\s+(?:the|a|an|any|your)?\s*$",
    re.IGNORECASE,
)
# 否定的作用域落在【动词之后】的程度状语上时，被否定的是"太简略"而不是"总结"：
# "Don't summarize too briefly" 用户要的仍然是总结。这与 Phase 0 给中文定的
# 「否定线索必须紧邻动词，隔着状语就说明否定的是状语」是同一条规则的镜像——
# 中文修饰语在动词前（靠前缀锚点解决），英文在动词后（只能看后缀）。
# 只收 too/overly/excessively 这类无歧义的程度副词；so 兼作目的连词，不收。
_EN_DEGREE_SCOPE_RE = re.compile(
    r"\s*(?:way\s+)?(?:too|overly|excessively)\s+\w+",
    re.IGNORECASE,
)
# 「被否定动作之外的残余抽取请求」。只在 summarize 被明确禁止、且 query_type
# 恰好是 overview 时才参与判定——那个 overview 正是被用户否定掉的那句话产生的，
# 本身不可信。正常问句与 query_type 已经正确的问句都走不到这里。
_EXTRACT_REQUEST_RE = re.compile(
    r"(?:列出|列举|提取|抽取|摘出|具体数值|准确数值|原始数值|"
    r"\b(?:extract|verbatim|exact)\b)",
    re.IGNORECASE,
)
# 翻译目标语的规范表：每个 code 至少覆盖「X文」「X语」两种说法与英文名。
_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "zh": ("中文", "汉语", "中国话", "简体中文", "繁体中文", "chinese", "mandarin"),
    "en": ("英文", "英语", "english"),
    "ja": ("日文", "日语", "日本语", "日本語", "japanese"),
    "ko": ("韩文", "韩语", "朝鲜语", "korean"),
    "fr": ("法文", "法语", "french"),
    "de": ("德文", "德语", "german"),
    "ru": ("俄文", "俄语", "russian"),
    "es": ("西班牙文", "西班牙语", "西语", "spanish"),
}
_LANGUAGE_ALIAS_TO_CODE: dict[str, str] = {
    alias.lower(): code
    for code, aliases in _LANGUAGE_ALIASES.items()
    for alias in aliases
}
# 别名按长度倒序参与匹配，避免"日"类短别名先吃掉"日文"这样的长别名。
_TRANSLATION_TARGET_RE = re.compile(
    r"(?:译成|译为|翻成|翻译成|翻译为|转成|转为|\binto\b|\bto\b)\s*("
    + "|".join(
        re.escape(alias)
        for alias in sorted(_LANGUAGE_ALIAS_TO_CODE, key=len, reverse=True)
    )
    + r")",
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
    # Derived once with the frozen modalities/interaction mode. Downstream
    # gates must consume this instead of reclassifying raw question text.
    visual_intent: bool
    query_type: str
    evidence_need: tuple[str, ...]
    top_k: int
    reasoning: str
    agent_policy: RoutePolicy
    web_policy: RoutePolicy
    graph_mode: GraphMode
    # A deterministic rule-match strength, not a calibrated probability.
    decision_strength: float
    is_ambiguous: bool
    clarification_question: str
    matched_rules: tuple[str, ...]
    operations: tuple[dict[str, str], ...] = ()
    page_ranges: tuple[tuple[int, int], ...] = ()
    inventory_kinds: tuple[str, ...] = ()
    evidence_sources: tuple[str, ...] = ("document",)
    continuation_ref: dict[str, str] | None = None
    ambiguities: tuple[dict[str, str], ...] = ()

    @property
    def confidence(self) -> float:
        """Compatibility shim for internal callers; omitted from serialized output."""
        return self.decision_strength

    def to_dict(self) -> dict:
        result = asdict(self)
        result["modalities"] = list(self.modalities)
        result["evidence_need"] = list(self.evidence_need)
        result["matched_rules"] = list(self.matched_rules)
        result["operations"] = [dict(item) for item in self.operations]
        result["page_ranges"] = [list(item) for item in self.page_ranges]
        result["inventory_kinds"] = list(self.inventory_kinds)
        result["evidence_sources"] = list(self.evidence_sources)
        result["continuation_ref"] = dict(self.continuation_ref or {}) or None
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

    @property
    def resolved_question(self) -> str:
        """The one question downstream consumers must use for this turn."""
        return self.intent_question or self.effective_question or self.original_question

    def with_retrieval_query(self, retrieval_query: str) -> "ChatTurnContext":
        return replace(self, retrieval_query=str(retrieval_query or self.intent_question).strip())

    def to_meta(self) -> dict:
        return {
            "original_question": self.original_question,
            "effective_question": self.effective_question,
            "resolved_question": self.resolved_question,
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
    clarification_resolved: bool = False,
    unresolved_continuation: bool = False,
    continuation_ref: dict[str, str] | None = None,
) -> IntentDecision:
    """Build one frozen, compatibility-preserving decision for a chat turn."""
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
    visual_intent = bool(
        set(modalities) & {"figure", "table", "formula", "layout"}
    ) or mode == "image"
    operations = _infer_operations(question, query_type, retry_resolved=retry_resolved)
    task, task_rule = _infer_task(
        question,
        query_type,
        retry_resolved=retry_resolved,
        operations=operations,
    )
    page_ranges = extract_page_ranges(question)
    inventory_kinds = tuple(detect_inventory_kinds(question))
    scope, scope_rule = _infer_scope(question, mode)
    agent_policy = "force" if (force_agent and enable_agent) else ("auto" if enable_agent else "off")
    normalized_web_policy = normalize_route_policy(web_policy, enabled=enable_web)
    graph_mode = _infer_graph_mode(task, query_type, evidence_need)
    evidence_sources = _infer_evidence_sources(question, mode)

    ambiguous_reference = bool(_AMBIGUOUS_REFERENCE_RE.search(question))
    # 已绑定上一轮的续写请求由调用方消解，不算歧义。
    bare_follow_up = bool(_BARE_FOLLOW_UP_RE.fullmatch(question) and not continuation_ref)
    # 可检索性闸门。判据是离散的 `== 0`：问句里一个实义词都没有，才谈得上必须
    # 先澄清；只要有一个可检索目标（「其中哪个模块」→ 模块），就直接去检索，
    # 哪怕句子里带代词。不要折算成分数阈值——分数会让这里重蹈 confidence
    # 纯装饰的覆辙，而"判错就零检索"的代价远高于"带着代词多检索一次"。
    content_terms = count_content_terms(question)
    is_ambiguous = bool(
        mode == "default"
        and (
            unresolved_continuation
            or (
                (ambiguous_reference or bare_follow_up)
                and content_terms == 0
                and not clarification_resolved
                and not _DOCUMENT_SELF_REFERENCE_RE.search(question)
                and len(question) <= 96
            )
        )
    )
    ambiguities: list[dict[str, str]] = []
    matched_rules = _dedupe_strings([
        f"interaction:{mode}",
        task_rule,
        scope_rule,
        *(f"operation:{item['kind']}:{item['polarity']}" for item in operations),
        *(f"page_range:{start}-{end}" for start, end in page_ranges),
        *(f"inventory:{item}" for item in inventory_kinds),
        *(f"evidence_source:{item}" for item in evidence_sources),
        *(f"modality:{item}" for item in modalities if item != "text"),
        *(f"evidence:{item}" for item in evidence_need),
    ])
    if clarification_resolved:
        matched_rules.append("clarification:resolved")
    if continuation_ref:
        matched_rules.append("continuation:bound")
    if (
        mode == "default"
        and not is_ambiguous
        and (ambiguous_reference or bare_follow_up)
        and content_terms > 0
    ):
        # 闸门放行的观测点：指代正则确实命中了，但问句里有可检索目标，
        # 所以这一轮不澄清直接检索。没有这条 trace 就看不出闸门是否在工作。
        matched_rules.append("retrievability:content_terms_present")
    if is_ambiguous:
        decision_strength = 0.35
        if unresolved_continuation or (bare_follow_up and not ambiguous_reference):
            clarification_question = "你想继续展开上一轮中的哪一项内容？可以补充主题、章节或页码。"
            ambiguities.append({"kind": "continuation", "missing": "prior_turn"})
            matched_rules.append("ambiguity:unresolved_continuation")
        else:
            clarification_question = "你说的“这个”具体指文档中的哪项内容？可以补充名称、章节或页码。"
            ambiguities.append({"kind": "reference", "missing": "referent"})
            matched_rules.append("ambiguity:unresolved_reference")
    elif mode in {"selection", "image", "retry_failed_turn"}:
        decision_strength = 0.98
        clarification_question = ""
    elif task != "qa" or evidence_need:
        decision_strength = 0.9
        clarification_question = ""
    else:
        decision_strength = 0.65
        clarification_question = ""

    identity_payload = {
        "version": INTENT_DECISION_VERSION,
        "question": question,
        "interaction_mode": mode,
        "task": task,
        "scope": scope,
        "page_ranges": page_ranges,
        "operations": operations,
        "inventory_kinds": inventory_kinds,
        "evidence_sources": evidence_sources,
        "modalities": modalities,
        "visual_intent": visual_intent,
        "query_type": query_type,
        "evidence_need": evidence_need,
        "agent_policy": agent_policy,
        "web_policy": normalized_web_policy,
        "continuation_source": str((continuation_ref or {}).get("source_question_hash") or ""),
    }
    intent_id = hashlib.sha256(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    raw_top_k = strategy.get("top_k")
    try:
        top_k = max(0, int(raw_top_k)) if raw_top_k is not None else 10
    except (TypeError, ValueError):
        top_k = 10
    return IntentDecision(
        intent_id=intent_id,
        version=INTENT_DECISION_VERSION,
        original_question=original,
        intent_question=question,
        interaction_mode=mode,
        task=task,
        scope=scope,
        modalities=modalities,
        visual_intent=visual_intent,
        query_type=query_type,
        evidence_need=evidence_need,
        top_k=top_k,
        reasoning=str(strategy.get("reasoning") or ""),
        agent_policy=agent_policy,  # type: ignore[arg-type]
        web_policy=normalized_web_policy,
        graph_mode=graph_mode,
        decision_strength=decision_strength,
        is_ambiguous=is_ambiguous,
        clarification_question=clarification_question,
        matched_rules=tuple(matched_rules),
        operations=operations,
        page_ranges=page_ranges,
        inventory_kinds=inventory_kinds,
        evidence_sources=evidence_sources,
        continuation_ref=dict(continuation_ref or {}) or None,
        ambiguities=tuple(ambiguities),
    )


def apply_llm_clarification(
    intent: IntentDecision,
    *,
    is_clear: bool,
    clarification_question: str = "",
    source: str = "llm",
) -> IntentDecision:
    """Overlay a cheap-model clarity judgment onto a frozen intent decision.

    Only touches the ambiguity fields; never rewrites task/scope/evidence.
    The overlay is bidirectional: the model may raise an ambiguity the rules
    missed, and may also clear an ambiguity the rules raised by mistake.
    """
    if is_clear:
        if not intent.is_ambiguous:
            matched = _dedupe_strings([*intent.matched_rules, f"clarification_llm:clear:{source}"])
            return replace(intent, matched_rules=tuple(matched))
        matched = _dedupe_strings([
            *intent.matched_rules,
            f"clarification_llm:clear:{source}",
            "clarification_llm:override_rule",
        ])
        return replace(
            intent,
            is_ambiguous=False,
            clarification_question="",
            decision_strength=_unambiguous_decision_strength(intent),
            matched_rules=tuple(matched),
        )
    if intent.is_ambiguous:
        return intent

    question = str(clarification_question or "").strip() or (
        "你的问题还不够具体。可以补充指代对象、章节、页码或比较目标吗？"
    )
    ambiguities = list(intent.ambiguities or ())
    ambiguities.append({"kind": "llm_unclear", "missing": "specificity", "source": str(source or "llm")})
    matched = _dedupe_strings([
        *intent.matched_rules,
        f"clarification_llm:unclear:{source}",
        "ambiguity:llm_unclear",
    ])
    return replace(
        intent,
        is_ambiguous=True,
        clarification_question=question[:280],
        decision_strength=min(float(intent.decision_strength or 0.5), 0.4),
        matched_rules=tuple(matched),
        ambiguities=tuple(ambiguities),
    )


def _unambiguous_decision_strength(intent: IntentDecision) -> float:
    """Rule-match strength this intent carries when not flagged as ambiguous.

    Mirrors the non-ambiguous branches of ``prepare_chat_intent`` so an LLM
    override does not leave the turn stuck at the 0.35 ambiguity score.
    """
    if intent.interaction_mode in {"selection", "image", "retry_failed_turn"}:
        return 0.98
    if intent.task != "qa" or intent.evidence_need:
        return 0.9
    return 0.65


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


def build_intent_trace(
    intent: IntentDecision,
    question: str,
    resolved_question: str,
    retrieval_meta: dict | None = None,
) -> dict:
    """意图判定的唯一 trace 构造点（成功路径与失败路径都必须走这里）。

    纯观测：只把已经定下来的判据抄成结构化记录，不读取也不改变任何判定。
    ``question_hash`` 取用户原问题（澄清改写后仍可去重），``question_preview``
    与 ``lang`` 取规则实际匹配的 resolved_question；问题原文不落盘。
    运行期字段从 ``retrieval_meta`` 取，取不到写 None。
    """
    raw_question = str(question or "")
    matched_question = str(resolved_question or raw_question)
    meta = retrieval_meta if isinstance(retrieval_meta, dict) else {}
    operations = tuple(getattr(intent, "operations", ()) or ())

    translate_target_code = ""
    for operation in operations:
        if operation.get("kind") == "translate" and operation.get("polarity") == "requested":
            translate_target_code = str(operation.get("target_language_code") or "")
            break

    # agent_gate / fallback_reason 是本仓库既有的运行态字段，作为别名兜底。
    agent_used = meta.get("agent_used")
    if agent_used is None and isinstance(meta.get("agent_gate"), dict):
        agent_used = meta["agent_gate"].get("use_agent")
    degraded_to = meta.get("degraded_to")
    if degraded_to is None and meta.get("degraded"):
        degraded_to = meta.get("fallback_reason")
    skipped_reason = meta.get("retrieval_skipped_reason")

    return {
        "trace_version": INTENT_TRACE_VERSION,
        "rule_version": _rule_version(),
        "question_hash": hashlib.sha256(raw_question.encode("utf-8")).hexdigest()[:16],
        "question_preview": matched_question[:40],
        "lang": _detect_lang(matched_question),
        "is_ambiguous": bool(getattr(intent, "is_ambiguous", False)),
        "matched_rules": list(getattr(intent, "matched_rules", ()) or ()),
        "task": str(getattr(intent, "task", "") or ""),
        "query_type": str(getattr(intent, "query_type", "") or ""),
        "evidence_need": list(getattr(intent, "evidence_need", ()) or ()),
        "modalities": list(getattr(intent, "modalities", ()) or ()),
        "inventory_kinds": list(getattr(intent, "inventory_kinds", ()) or ()),
        "scope": str(getattr(intent, "scope", "") or ""),
        "page_ranges": [list(item) for item in (getattr(intent, "page_ranges", ()) or ())],
        "translate_target_code": translate_target_code or None,
        "operations": [dict(item) for item in operations],
        "top_k": int(getattr(intent, "top_k", 0) or 0),
        "decision_strength": float(getattr(intent, "decision_strength", 0.0) or 0.0),
        "retrieval_skipped_reason": str(skipped_reason) if skipped_reason else None,
        "agent_used": bool(agent_used) if agent_used is not None else None,
        "degraded_to": str(degraded_to) if degraded_to else None,
    }


def _rule_version() -> str:
    """意图规则模块源码的短 hash：规则一改就变，改回来就复原。"""
    global _RULE_VERSION_CACHE
    if _RULE_VERSION_CACHE is not None:
        return _RULE_VERSION_CACHE
    digest = hashlib.sha256()
    resolved = False
    for name in _RULE_SOURCE_MODULES:
        digest.update(name.encode("utf-8"))
        try:
            source = inspect.getsource(importlib.import_module(name))
        except Exception:
            # 冻结打包等场景拿不到源码：跳过该模块，绝不让 trace 构造失败。
            continue
        digest.update(source.encode("utf-8", "replace"))
        resolved = True
    _RULE_VERSION_CACHE = digest.hexdigest()[:12] if resolved else "unavailable"
    return _RULE_VERSION_CACHE


# 仅供 trace 归因使用的粗粒度语种标记，不参与任何判定。
_TRACE_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_TRACE_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _detect_lang(text: str) -> str:
    """返回 zh / en / mixed / unknown。

    按「CJK 字符数」对「拉丁词数」的占比判定，因此"解释一下 Transformer"
    这类夹带英文术语的中文问句仍记为 zh，纯英文问句记为 en。
    """
    sample = str(text or "")
    cjk = len(_TRACE_CJK_CHAR_RE.findall(sample))
    latin = len(_TRACE_LATIN_WORD_RE.findall(sample))
    if not cjk and not latin:
        return "unknown"
    if not latin:
        return "zh"
    if not cjk:
        return "en"
    ratio = cjk / float(cjk + latin)
    if ratio >= 0.6:
        return "zh"
    if ratio <= 0.25:
        return "en"
    return "mixed"


def _infer_task(
    question: str,
    query_type: str,
    *,
    retry_resolved: bool,
    operations: Sequence[dict[str, str]] | None = None,
) -> tuple[IntentTask, str]:
    del retry_resolved
    allowed = {
        "qa", "summarize", "extract", "explain", "compare",
        "calculate", "translate", "continue", "inventory",
    }
    prohibited_kinds = {
        str(item.get("kind") or "")
        for item in operations or ()
        if item.get("polarity") == "prohibited"
    }
    for operation in operations or ():
        if operation.get("polarity") != "requested":
            continue
        kind = str(operation.get("kind") or "")
        if kind == "qa":
            # _infer_operations 在「一个 requested 都没有」时补的兜底项，
            # 不是用户显式请求的动作。这里必须让位给下面按 query_type / 残余请求
            # 的推导，否则「别再只给概述了，列出表3的数值」会永远停在 qa——
            # 兜底 qa 抢在 query_type == "extraction" 之前返回。
            continue
        if kind in allowed:
            # A later request can intentionally narrow an earlier negation:
            # "不要总结全文，只总结第 3 页" and "不要翻译全文，只翻译摘要".
            # Operation records preserve both clauses for diagnostics, but the
            # positive, scoped action is still the turn's primary task.
            return kind, f"task:{kind}"  # type: ignore[return-value]
    if query_type == "extraction":
        return "extract", "task:extract"
    if (
        "summarize" in prohibited_kinds
        and query_type == "overview"
        and _EXTRACT_REQUEST_RE.search(str(question or ""))
    ):
        # 用户把 summary/overview 明确否掉了，而 query_type 的 overview 恰恰是
        # 那句被否定的话产生的，不可信；此时只能直接看问句里的残余抽取请求。
        return "extract", "task:extract_residual"
    return "qa", "task:qa"


def _normalize_page_text(text: str) -> str:
    """页码解析专用的局部归一化。

    只在 extract_page_ranges 内部对一份副本使用，**绝不能全局套用**：
    TrustRAG/RAGFlow 的 _strQ2B 会把「，。？（）」一并半角化，而 _CLAUSE_SPLIT_RE
    （否定作用域切子句）和 _AMBIGUOUS_REFERENCE_RE 都吃中文标点，全局归一化会
    连带改掉 negation_rewrite / zh_true_ambiguous 这两个已满分类目的判定。
    这里只动数字与连字符族，两类字符都不参与那些正则。

    _PAGE_SCOPE_RE / _PAGE_RANGE_RE 只出现在本函数体内，没有任何 match 偏移量
    外泄给调用方，所以即便折叠连字符改变了串长也不会污染别处。
    """
    return _REPEATED_DASH_RE.sub("-", str(text or "").translate(_PAGE_CHAR_TRANSLATION))


def _page_pair(match: re.Match) -> tuple[int, int] | None:
    """从任一 alternation 分支取出成对页码；取不出就返回 None。

    每个分支恰好两个捕获组，未命中的分支全是 None，所以「前两个非 None 组」
    就是这次命中的起止页。取不满两个（悬空分隔符、半截表达式）一律丢弃而不是
    抛异常——opendataloader-pdf Config.java:710-732 在受控 CLI 语法下抛
    IllegalArgumentException 是对的，但这里是意图识别链路上的软信号，抛异常
    会炸掉整条问答链。同理逆序保持 min/max 交换语义，不像它那样拒绝 5-3。
    """
    numbers: list[int] = []
    for group in match.groups():
        if group is None:
            continue
        try:
            numbers.append(int(group))
        except (TypeError, ValueError):
            return None
        if len(numbers) == 2:
            break
    if len(numbers) != 2:
        return None
    start, end = numbers
    if start <= 0 or end <= 0:
        return None
    return (min(start, end), max(start, end))


def extract_page_ranges(question: str) -> tuple[tuple[int, int], ...]:
    """Return explicit page locators in source order, including ranges."""
    raw = str(question or "")
    normalized = _normalize_page_text(raw)
    if not normalized:
        # 归一化是破坏性变换，剥空则回退原文（RAGFlow rmWWW 的 `if not txt: txt = otxt`）。
        # 注意只守这一步：最终返回空元组是「问句里本来就没页码」的合法高频结果，
        # 不能给返回值加回退。
        normalized = raw
    found: list[tuple[int, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for match in _PAGE_RANGE_RE.finditer(normalized):
        value = _page_pair(match)
        if value is None or value in seen:
            continue
        found.append((match.start(), value))
        seen.add(value)
    # 先扫范围、再把范围抹掉扫单点，这样「第3到5页」不会又被拆出 (3,3)/(5,5)。
    # 抹除用**等长空白**而不是单个空格：第二趟的 match.start() 因此仍与 normalized
    # 对齐，两趟产物才能按原文偏移量归并成 docstring 承诺的 source order
    # （否则「第2页和第4到6页」会得到 [[4,6],[2,2]]，而 eval_intent._norm_ranges
    # 是有序比较）。
    without_ranges = _PAGE_RANGE_RE.sub(lambda m: " " * (m.end() - m.start()), normalized)
    for match in _PAGE_SCOPE_RE.finditer(without_ranges):
        number = re.search(r"\d+", match.group(0))
        if not number:
            continue
        page = int(number.group(0))
        if page <= 0:
            # 与 _page_pair 共用同一条哨兵约定：0 页不存在，解析不出就统一丢弃。
            continue
        value = (page, page)
        if value in seen:
            continue
        found.append((match.start(), value))
        seen.add(value)
    found.sort(key=lambda item: item[0])
    return tuple(value for _, value in found)


def is_continuation_request(question: str) -> bool:
    return bool(_CONTINUE_RE.fullmatch(str(question or "").strip()))


def _clause_prefix(question: str, index: int) -> str:
    """返回 index 之前、且与 index 处于同一子句内的片段。

    否定作用域不跨子句：「不要总结，找出第2页结论」中的"不要"只管"总结"。
    """
    text = str(question or "")
    bound = 0
    for separator in _CLAUSE_SPLIT_RE.finditer(text):
        if separator.end() > index:
            break
        bound = separator.end()
    return text[bound:max(bound, index)]


def _clause_end(question: str, index: int) -> int:
    """返回 index 所在子句的结束位置（下一个分隔符之前，或文末）。"""
    text = str(question or "")
    for separator in _CLAUSE_SPLIT_RE.finditer(text):
        if separator.start() >= index:
            return separator.start()
    return len(text)


def _infer_operations(
    question: str,
    query_type: str,
    *,
    retry_resolved: bool,
) -> tuple[dict[str, str], ...]:
    if is_continuation_request(question) and not retry_resolved:
        return ({"kind": "continue", "polarity": "requested"},)

    patterns: tuple[tuple[IntentTask, object], ...] = (
        ("translate", _TRANSLATE_RE),
        ("compare", _COMPARE_RE),
        ("calculate", _CALCULATE_RE),
        ("summarize", _SUMMARY_RE),
        ("explain", _EXPLAIN_RE),
    )
    matches: list[tuple[int, int, dict[str, str]]] = []
    serial = 0
    for kind, raw_pattern in patterns:
        pattern = raw_pattern if isinstance(raw_pattern, re.Pattern) else None
        if pattern is None:
            continue
        for match in pattern.finditer(question):
            prefix = _clause_prefix(question, match.start())
            polarity = "prohibited" if _NEGATION_CUE_RE.search(prefix) else "requested"
            if polarity == "prohibited" and _EN_DEGREE_SCOPE_RE.match(
                question, match.end(), _clause_end(question, match.end())
            ):
                # 否定落在动词之后的程度状语上（"don't summarize too briefly"）：
                # 用户要的仍然是这个动作，只是别做过头。
                polarity = "requested"
            operation = {"kind": kind, "polarity": polarity}
            if kind == "translate":
                target_language, target_code = _extract_translation_target(question, match.start())
                if target_language:
                    operation["target_language"] = target_language
                if target_code:
                    operation["target_language_code"] = target_code
            matches.append((match.start(), serial, operation))
            serial += 1

    if query_type == "inventory":
        matches.append((0, serial, {"kind": "inventory", "polarity": "requested"}))
    elif query_type == "extraction" and not matches:
        matches.append((0, serial, {"kind": "extract", "polarity": "requested"}))

    matches.sort(key=lambda item: (item[0], item[1]))
    operations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, operation in matches:
        key = (operation["kind"], operation["polarity"])
        if key in seen:
            continue
        seen.add(key)
        operations.append(operation)
    if not any(item.get("polarity") == "requested" for item in operations):
        operations.append({"kind": "qa", "polarity": "requested"})
    return tuple(operations)


def _extract_translation_target(question: str, start: int) -> tuple[str, str]:
    """返回 (原文别名, canonical code)。

    只认「触发词在目标语之前」的写法，避免"把这段中文翻译一下"把源语当成目标语；
    命中多处时取离翻译动作最近的一处。
    """
    text = str(question or "")
    window_start = max(0, start - 16)
    # 向后的边界取「同一子句的末尾」而不是定长字符窗口：
    #   - 定长窗口（原来 48 字符）会截断带范围限定的长句，
    #     "translate the second paragraph on page 4 into Korean" 的 Korean 正好被切掉；
    #   - 子句边界同时挡住了跨子句误读，
    #     "Translate the abstract for me, the original is in German" 的 German 在逗号
    #     之后，是源语不是目标语，不该被取进来。
    window_end = max(start, _clause_end(text, start))
    window = text[window_start:window_end]
    anchor = start - window_start
    best: re.Match[str] | None = None
    best_distance: int | None = None
    for match in _TRANSLATION_TARGET_RE.finditer(window):
        distance = abs(match.start() - anchor)
        if best_distance is None or distance <= best_distance:
            best = match
            best_distance = distance
    if best is None:
        return "", ""
    alias = str(best.group(1) or "").strip().lower()
    return alias, _LANGUAGE_ALIAS_TO_CODE.get(alias, "")


def _infer_scope(question: str, mode: InteractionMode) -> tuple[IntentScope, str]:
    if mode == "image":
        return "image", "scope:image"
    if mode == "selection":
        return "selection", "scope:selection"
    if extract_page_ranges(question):
        return "page", "scope:page"
    if _SECTION_SCOPE_RE.search(question):
        return "section", "scope:section"
    return "document", "scope:document"


def _infer_evidence_sources(question: str, mode: InteractionMode) -> tuple[str, ...]:
    sources: list[str] = []
    if mode == "image":
        sources.append("attachment")
        if _EXPLICIT_DOCUMENT_EVIDENCE_RE.search(question):
            sources.append("document")
    elif mode == "selection":
        sources.extend(["selection", "document"])
    else:
        sources.append("document")
    return tuple(_dedupe_strings(sources))


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
