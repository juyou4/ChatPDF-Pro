"""Agent 路径的 cheap-model 澄清判定（阶段 3.1 起兼产问题分解信号）。

规则层（chat_intent_service）先判 is_ambiguous；只要 Agent 即将启用，
就再用一次 cheap model 判断问句是否足够清晰——这一步是双向的，
既能补判规则漏掉的模糊，也能翻转规则的误判。
失败时 fail-open（不阻断检索）。

阶段 3.1（降级版）在**同一次调用**上追加 ``sub_questions`` 字段：
判定与生成合一，用数组长度当"要不要拆"的信号（不新增 needs_decompose 布尔，
避免"布尔说 true 但数组为空"的自相矛盾态）。这样 agent 侧那次独立的
``decompose_question`` 调用可以省掉——净往返数不增加。

三条硬约束：

1. **numeric_table 硬短路**：表格/数值类问句发的是与改造前逐字节相同的精简
   system prompt，分解相关的 token 一个都不进 LLM，且无条件把 sub_questions
   当空处理，回落确定性规则路径。
2. **fail-open**：解析失败/超时/异常一律回落（is_clear=True、sub_questions=[]），
   绝不抛、绝不返回 None。
3. **语义等价锁**：LLM 返回的子问题必须只用原问句里已有的信息，出现原问句
   没有的实体/数字/汉字即整体丢弃（见 ``subquestions_preserve_semantics``）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Iterable, Optional, Sequence

from services.completion_outcome import require_publishable_completion
from services.intent_constraints import IntentConstraintSet
from services.structured_json import parse_json_object

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3

# ---------------------------------------------------------------------------
# System prompts
#
# 两份变体刻意共享同一段 is_clear 措辞：numeric_table 走精简版时，模型看到的
# 澄清指令必须与改造前**逐字节相同**，否则红线 3 的"短路"会顺带改变澄清行为。
# ---------------------------------------------------------------------------
_CLARIFY_CORE = (
    "You judge whether a user question about a single document is clear enough "
    "to retrieve evidence without asking the user for more detail.\n"
    "Return ONLY JSON:\n"
    '{\n  "is_clear": true|false,\n  "clarification_question": "..."\n}\n'
    "Set is_clear=false only when the referent, scope, comparison target, or task "
    "is genuinely underspecified and history does not resolve it. "
    "If is_clear=true, leave clarification_question empty. "
    "If is_clear=false, write one short clarification question in the user's language."
)

# 精简版 == 改造前的 CLARIFY_SYSTEM_PROMPT（numeric_table 短路专用）
CLARIFY_SYSTEM_PROMPT = _CLARIFY_CORE
CLARIFY_SYSTEM_PROMPT_NO_DECOMPOSE = _CLARIFY_CORE

# 完整版：判定与生成合一。
# 全文用英文书写（含"语言跟随"这一条本身）——用中文写这条指令会把英文提问的
# 输出也带偏（TrustRAG deepresearch/action.py 的实测教训）。
CLARIFY_SYSTEM_PROMPT_WITH_DECOMPOSE = (
    "You judge a user question about a single document on two axes at once:\n"
    "(a) whether it is clear enough to retrieve evidence without asking the user "
    "for more detail, and\n"
    "(b) whether it must be split into independently retrievable sub-questions.\n"
    "\n"
    "Return ONLY JSON, nothing else. No explanatory text, no markdown code fences:\n"
    "{\n"
    '  "is_clear": true|false,\n'
    '  "clarification_question": "...",\n'
    '  "sub_questions": []\n'
    "}\n"
    "\n"
    "## is_clear\n"
    "Set is_clear=false only when the referent, scope, comparison target, or task "
    "is genuinely underspecified and history does not resolve it.\n"
    "If is_clear=true, leave clarification_question empty.\n"
    "If is_clear=false, write one short clarification question in the user's language.\n"
    "\n"
    "## sub_questions\n"
    "Return [] unless the question asks for two or more things that must be looked up "
    "SEPARATELY in the document.\n"
    "\n"
    "Return [] - these are ONE question, not several:\n"
    "- One quantity measured under several conditions (\"the accuracy on different "
    "datasets\", \"results across several benchmarks\"). A word that merely MODIFIES a "
    "noun - \"different\", \"various\", \"several\" - is NOT a split signal.\n"
    "- Asking for the CAUSE, reason, or mechanism of a difference (\"why does this gap "
    "appear\"). The difference is the topic, not two targets.\n"
    "- Any single lookup of a number, name, setting, or yes/no fact.\n"
    "\n"
    "Split ONLY when one of these holds:\n"
    "- The question names two or more DISTINCT targets to be described or contrasted "
    "(\"compare X and Y\", \"differences between X and Y\", \"the pros and cons of X and Y\").\n"
    "- The question chains two or more DISTINCT tasks (\"summarize A and then list B\", "
    "\"compare A and explain why B\").\n"
    "\n"
    "## Hard constraints on sub_questions - violating ANY ONE means return [] instead\n"
    "1. At most 3 items. Fewer is better. If one item is enough, return [] - never pad "
    "the list.\n"
    "2. Every item must be answerable from the same single document.\n"
    "3. SEMANTIC EQUIVALENCE LOCK: the union of the items must mean exactly what the "
    "original asks - no more, no less.\n"
    "   - Every item must be explicitly derived from the original question.\n"
    "   - Do NOT expand, enrich, generalize, define, or reinterpret.\n"
    "   - Do NOT introduce any entity, model, metric, dataset, section, table, figure, "
    "year, or task that is not written in the original question.\n"
    "   - Splitting \"compare A and B\" yields one question about A and one about B. It "
    "must NOT yield \"what is A\", \"the history of A\", or \"how A relates to C\".\n"
    "4. Copy every proper noun, model name, metric name, table/figure number and numeral "
    "EXACTLY as written in the original. If there are acronyms or words you are not "
    "familiar with, do not try to rephrase or expand them.\n"
    "5. LANGUAGE FOLLOWING: write every item in the SAME LANGUAGE as the user's question "
    "- Chinese question, Chinese items; English question, English items. Never mix "
    "languages inside one item. Keep proper nouns in their original form regardless of "
    "the surrounding language.\n"
    "6. If you are not confident, return []. Returning [] is always safe.\n"
    "\n"
    "## Examples (sub_questions field only)\n"
    "\"这个方法在几个不同数据集上的表现如何？\" -> []\n"
    "\"这里出现的这种性能落差是什么原因造成的？\" -> []\n"
    "\"Which optimizer did the authors use?\" -> []\n"
    "\"Compare the throughput of method X and method Y, and explain why X is faster.\" ->\n"
    "  [\"What is the throughput of method X and of method Y?\", \"Why is method X faster?\"]\n"
    "\"先说明模型结构，再列出作者提到的不足。\" ->\n"
    "  [\"模型结构是什么？\", \"作者提到的不足有哪些？\"]"
)

CLARIFY_USER_PROMPT = (
    "Recent chat history (may be empty):\n{history}\n\n"
    "User question:\n{question}\n"
)

_VAGUE_HINT_RE = re.compile(
    r"(?:这个|那个|它|其|这块|那块|这种方法|该模型|好在哪|怎么样|如何评价|"
    r"\b(?:it|this|that|they|them|the\s+(?:method|model|approach|result))\b|"
    r"\bhow\s+(?:good|well)\b|\bwhat\s+about\b)",
    re.IGNORECASE,
)

# 触发门放宽用的召回向正则。它与 decompose_service.should_decompose 取并集：
# 规则侧的英文触发词由另一条工作线负责补齐，这里不能把 LLM 路径的存亡
# 押在那条线的落地时间上（今天的 _DECOMPOSE_TRIGGERS 是纯中文正则，
# 英文一条都不触发）。规则补齐后这条正则就退化成它的子集。
_DECOMPOSE_GATE_FALLBACK_RE = re.compile(
    r"比较|对比|优缺点|异同|分别|各自"
    r"|有(?:什么|哪些|何)\s*(?:不同|差异|区别|差别)"
    r"|先[^。？！]{0,24}?[，,]?\s*(?:再|然后|接着|之后)"
    r"|\bcompare\b|\bcomparison\b|\bcontrast\b|\bversus\b|\bvs\.?\b"
    r"|\bdifferences?\s+between\b|\bpros\s+and\s+cons\b"
    r"|\band\s+then\b"
    r"|\band\s+(?:also\s+)?(?:explain|list|describe|summari[sz]e|analy[sz]e|discuss|"
    r"evaluate|compare|outline|state)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# numeric_table 硬短路判定
# ---------------------------------------------------------------------------
def is_numeric_table_question(
    question: str,
    evidence_need: Sequence[str] | None = None,
) -> bool:
    """表格/数值类问句：分解相关的 token 一个都不许进 LLM 链。

    路由层能给出冻结好的 ``evidence_need``（权威值）；拿不到时再自己算一遍。
    任何异常都判 True（宁可短路，也不让红线 3 因为一个 ImportError 失守）。
    """
    for item in evidence_need or ():
        if str(item or "").strip().lower() == "numeric_table":
            return True
    text = str(question or "").strip()
    if not text:
        return False
    try:
        from services.query_analyzer import (
            _is_numeric_table_cost_query,
            analyze_evidence_need,
        )

        if _is_numeric_table_cost_query(text.lower()):
            return True
        return "numeric_table" in (analyze_evidence_need(text) or [])
    except Exception as exc:  # pragma: no cover - 只在 query_analyzer 不可用时
        logger.debug("[Clarification] numeric_table 判定失败，按短路处理: %s", exc)
        return True


def _looks_decomposable(text: str) -> bool:
    """触发门放宽用：这一句是否**可能**需要拆。

    取 ``should_decompose``（规则侧权威判定）与召回向正则的并集：
    命中规则的那批回合今天本来就要为 ``decompose_question`` 付一次往返，
    改造后那次调用被省掉，往返 1:1 互换；正则多出来的那部分是英文场景的
    保底（规则侧英文触发词补齐后即被吸收）。
    """
    try:
        from services.decompose_service import should_decompose

        if should_decompose(text):
            return True
    except Exception:  # pragma: no cover - decompose_service 不可用时不阻断
        logger.debug("[Clarification] should_decompose 不可用，仅用兜底正则")
    return bool(_DECOMPOSE_GATE_FALLBACK_RE.search(text))


def should_attempt_llm_clarification(
    *,
    question: str,
    use_agent: bool,
    already_ambiguous: bool,
    clarification_resolved: bool,
    agent_policy: str = "auto",
    evidence_need: Sequence[str] | None = None,
) -> bool:
    """Cheap prefilter: only spend an LLM call when the Agent path would run.

    ``already_ambiguous`` is kept in the signature for call-site readability but
    no longer blocks the call: the rule layer over-fires on ordinary questions,
    so the LLM must also be allowed to review (and overturn) a rule-level
    ambiguity verdict. Only a turn that already resumed from a clarification
    ticket is skipped outright.

    阶段 3.1 放宽了一条：需要分解的问句通常很长（"Compare the pros and cons of
    BM25 and dense retrieval..." 80 字符）且不含指代 hint，旧门会全部挡掉。
    放宽的边界严格对齐 ``should_decompose``——这批回合今天本来就要为
    ``decompose_question`` 付一次往返，改造后那次调用被省掉，净往返不变。
    numeric_table 问句不参与放宽（红线 3）。
    """
    del already_ambiguous
    if clarification_resolved:
        return False
    if not use_agent:
        return False
    if str(agent_policy or "").strip().lower() == "off":
        return False
    text = str(question or "").strip()
    if not text or len(text) > 240:
        return False
    # Always check short questions on agent path; longer ones only with vague hints.
    if len(text) <= 48:
        return True
    if _VAGUE_HINT_RE.search(text):
        return True
    # 放宽分支：表格/数值类问句一律不放宽，保持确定性路径。
    if is_numeric_table_question(text, evidence_need):
        return False
    return _looks_decomposable(text)


def _history_snippet(chat_history: Sequence[dict] | None, *, limit: int = 4) -> str:
    lines: list[str] = []
    for item in list(chat_history or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip() or "user"
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:240]}")
    return "\n".join(lines) if lines else "(empty)"


def _response_content(response: Any) -> str:
    if not isinstance(response, dict) or response.get("error"):
        return ""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


# ---------------------------------------------------------------------------
# 解析：剥 <think> → parse_json_object（json.loads strict=False + 大括号切片）
#       → 正则兜底。任何一步失败都只降级，不抛。
# ---------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)

_IS_CLEAR_RE = re.compile(r'"is_clear"\s*:\s*(true|false)', re.IGNORECASE)
_CLARIFY_Q_RE = re.compile(r'"clarification_question"\s*:\s*"((?:[^"\\]|\\.)*)"')
_SUBQ_ARRAY_RE = re.compile(r'"sub_?questions"\s*:\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL)
_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def strip_think_blocks(content: Any) -> str:
    """剥掉推理模型的 think 块。

    cheap model 常被配成推理模型，``<think>`` 里出现的大括号会让"首尾大括号
    切片"切到推理文本上（RAGFlow generator.py 同款处理）。
    """
    text = str(content or "").strip()
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub(" ", text)
    if re.search(r"</think\s*>", text, re.IGNORECASE):
        text = _THINK_TAIL_RE.sub(" ", text)
    if re.search(r"<think\b", text, re.IGNORECASE):
        text = _THINK_OPEN_RE.sub(" ", text)
    return text.strip()


def _unescape_json_string(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"'))
    except Exception:
        return raw.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")


def _regex_recover(text: str) -> dict[str, Any] | None:
    """JSON 彻底解析不出来时的兜底：逐字段正则捞。

    返回 None 表示连 is_clear 都没捞到——此时视为完全失败，走 fail-open。
    """
    if not text:
        return None
    is_clear_match = _IS_CLEAR_RE.search(text)
    subs_match = _SUBQ_ARRAY_RE.search(text)
    if not is_clear_match and not subs_match:
        return None
    recovered: dict[str, Any] = {}
    if is_clear_match:
        recovered["is_clear"] = is_clear_match.group(1).lower() == "true"
    clarify_match = _CLARIFY_Q_RE.search(text)
    if clarify_match:
        recovered["clarification_question"] = _unescape_json_string(clarify_match.group(1))
    if subs_match:
        recovered["sub_questions"] = [
            _unescape_json_string(item)
            for item in _JSON_STRING_RE.findall(subs_match.group(1))
        ]
    return recovered


def _parse_clarify_payload(content: Any) -> tuple[dict[str, Any], str]:
    """(payload, parse_source)。payload 为空 dict 表示彻底失败。"""
    text = strip_think_blocks(content)
    if not text:
        return {}, "empty"
    try:
        return parse_json_object(text, allow_partial=False), "llm"
    except Exception:
        recovered = _regex_recover(text)
        if recovered is None:
            return {}, "unparsed"
        return recovered, "llm_regex"


# ---------------------------------------------------------------------------
# 后置校验：语义等价锁（防扩写）
#
# 这是比 prompt 硬得多的一层，且完全离线可测。方向与 query_rewriter 的
# _llm_rewrite_preserves_semantics 相反：改写防的是"丢信息"（source ⊆ candidate），
# 分解防的是"加信息"（candidate ⊆ source），所以不能直接复用那个函数——
# 拆分本身就会丢掉 "compare" 这类谓词，套过去会把所有正确拆分都判死。
# ---------------------------------------------------------------------------
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#]*")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 拆分时允许新引入的英文脚手架词（疑问词 / 系动词 / 介词 / 通用任务动词）。
# 刻意不含任何实体性名词——专名、模型名、指标名、数据集名一律不许新增。
_LATIN_SCAFFOLD_TOKENS = frozenset({
    "a", "an", "the", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "did", "done", "has", "have", "had", "hav",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "of", "in", "on", "at", "by", "for", "to", "from", "with", "within",
    "without", "about", "as", "into", "onto", "over", "under", "between",
    "and", "or", "but", "if", "than", "then", "so",
    # 注意 _norm_latin_token("this") -> "thi"
    "that", "thi", "this", "these", "those", "it", "its", "they", "them", "their",
    "there", "here", "not", "no", "any", "some", "each", "both", "one", "once",
    "other", "another", "such", "same", "respectively",
    "paper", "document", "author", "accord", "according",
})

# These are task-language artifacts, not document entities. Requiring them to
# survive every split would reject a sound parallel decomposition solely because
# "Compare ... as described" becomes two evidence lookups that omit those
# boilerplate verbs.
_TASK_LANGUAGE_TOKENS = frozenset({
    "compare", "comparison", "contrast", "versus", "vs", "pro", "con",
    "advantage", "disadvantage", "explain", "describe", "describ", "analy",
    "discuss", "why", "reason", "cause", "mechanism", "summarize", "summary",
    "overview", "list", "enumerate", "catalogue", "translate", "translation",
    "limitation", "drawback", "weakness",
})

# 拆分时允许新引入的中文脚手架字（疑问 / 系词 / 助词 / 指示 / 量词）。
# 刻意**不含任何实体性字，也不含任务动词**：任务动词本来就写在原问句里，
# 把它们加进白名单只会给扩写留口子。
_CJK_SCAFFOLD_CHARS = frozenset(
    "是不否有无的地得了着过吗呢吧呀啊么什哪些多少几个种"
    "怎样如何为何请和与及跟同在中对于把被这那此该其它他她们"
)

_REQUIRED_TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("compare", re.compile(r"比较|对比|区别|差异|异同|\b(?:compare|contrast|versus|vs\.?)\b", re.IGNORECASE)),
    ("explain", re.compile(r"解释|说明|分析|\b(?:explain|describe|analy[sz]e|discuss)\b", re.IGNORECASE)),
    ("reason", re.compile(r"原因|为什么|为何|机制|\b(?:why|reason|cause|mechanism)\b", re.IGNORECASE)),
    ("summarize", re.compile(r"总结|概述|摘要|\b(?:summari[sz]e|overview)\b", re.IGNORECASE)),
    ("list", re.compile(r"列出|清单|列表|哪些|\b(?:list|enumerate|catalogue?)\b", re.IGNORECASE)),
    ("translate", re.compile(r"翻译|\btranslate\b", re.IGNORECASE)),
    ("limitations", re.compile(r"局限|不足|缺点|\b(?:limitation|drawback|weakness|con)\b", re.IGNORECASE)),
)

# A valid comparison decomposition often turns one relation into parallel
# entity-specific lookups ("pros and cons of A" + "pros and cons of B").
# It need not repeat the literal word "compare", but it must retain a shared
# comparison dimension. This deliberately excludes plain definition questions
# such as "What is A? / What is B?".
_COMPARISON_DIMENSION_RE = re.compile(
    r"(?:优点|缺点|优势|劣势|异同|差异|区别|表现|效果|结果|指标|准确率|性能|"
    r"优缺点|利弊|pros?|cons?|advantage|disadvantage|trade[ -]?off|"
    r"difference|similarit(?:y|ies)|performance|result|metric|accuracy)",
    re.IGNORECASE,
)


def _required_task_labels(text: str) -> set[str]:
    return {
        label for label, pattern in _REQUIRED_TASK_PATTERNS
        if pattern.search(str(text or ""))
    }


def _coerce_llm_bool(value: Any, *, default: bool) -> bool:
    """Parse LLM booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return default


def _norm_latin_token(token: str) -> str:
    """粗粒度词形归一：source / candidate 走同一个函数，一致即可。"""
    text = token.lower().strip("+#")
    if len(text) > 3 and text.endswith("ies"):
        text = text[:-3] + "y"
    elif len(text) > 3 and text.endswith("es") and not text.endswith("ses"):
        text = text[:-2]
    elif len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    if len(text) > 4 and text.endswith("ing"):
        text = text[:-3]
    elif len(text) > 4 and text.endswith("ed"):
        text = text[:-2]
    return text


def _latin_tokens(text: str) -> set[str]:
    return {_norm_latin_token(tok) for tok in _LATIN_TOKEN_RE.findall(text or "")}


def normalize_sub_questions(
    raw: Any,
    *,
    source_question: str = "",
    limit: int = MAX_SUB_QUESTIONS,
) -> list[str]:
    """strip / 丢空 / 去重 / 丢掉与原句相同的项 / 截断到 limit。"""
    if not isinstance(raw, (list, tuple)):
        return []
    source_key = re.sub(r"\s+", "", str(source_question or "")).strip().lower()
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = re.sub(r"\s+", " ", item).strip().strip("-•*").strip()
        if len(text) < 4 or len(text) > 200:
            continue
        key = re.sub(r"\s+", "", text).lower()
        if not key or key in seen:
            continue
        if source_key and key == source_key:
            # 整句照抄不是子问题
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def subquestions_preserve_semantics(source: str, subs: Sequence[str]) -> bool:
    """子问题是否只用了原问句里已有的信息。

    四道检查，任一不过 → 整组丢弃、回落规则：
      1. 数量 2..3（1 条等价于没拆）；
      2. 新增数字：candidate 的数字集合必须 ⊆ source；
      3. 新增英文 token：归一后必须 ⊆ source ∪ 脚手架词表；
      4. 新增汉字：必须 ⊆ source ∪ 脚手架字表；
      另加长度护栏：拼接串不得显著长于原句（扩写的典型表征）。
    """
    source_text = str(source or "").strip()
    items = [str(item or "").strip() for item in (subs or [])]
    items = [item for item in items if item]
    if not source_text or not (2 <= len(items) <= MAX_SUB_QUESTIONS):
        return False

    candidate = " ".join(items)
    if len(candidate) > int(len(source_text) * 2.5) + 40:
        return False

    source_numbers = set(_NUMBER_RE.findall(source_text))
    if set(_NUMBER_RE.findall(candidate)) - source_numbers:
        return False

    source_latin = _latin_tokens(source_text)
    candidate_latin = _latin_tokens(candidate)
    if candidate_latin - source_latin - _LATIN_SCAFFOLD_TOKENS:
        return False

    source_cjk = set(_CJK_CHAR_RE.findall(source_text))
    if set(_CJK_CHAR_RE.findall(candidate)) - source_cjk - _CJK_SCAFFOLD_CHARS:
        return False

    # The final answer still receives the root question and executes its task;
    # subquestions exist only to retrieve evidence. Do not reject a faithful
    # "summarize X and explain Y" split merely because each evidence lookup is
    # phrased as a direct question. Preserve only constraints that would be
    # impossible to reconstruct from independent definitions: comparison,
    # causality and explicitly requested limitations.
    source_tasks = _required_task_labels(source_text)
    candidate_tasks = _required_task_labels(candidate)
    if "compare" in source_tasks and (
        "compare" not in candidate_tasks
        and not _COMPARISON_DIMENSION_RE.search(candidate)
    ):
        return False
    if "reason" in source_tasks and "reason" not in candidate_tasks:
        return False
    if "limitations" in source_tasks and "limitations" not in candidate_tasks:
        return False

    # Proper nouns/acronyms are the most reliable cross-language entity signal.
    # They must survive a split; otherwise an LLM could replace a comparison of
    # BM25 and dense retrieval with two generic definition questions.
    source_entities = source_latin - _LATIN_SCAFFOLD_TOKENS - _TASK_LANGUAGE_TOKENS
    if source_entities - candidate_latin:
        return False

    # Final authority shared with the standalone decompose call and Planner.
    # Keep the mature lexical checks above, then require the canonical intent
    # contract as a second independent boundary.
    return IntentConstraintSet.from_text(source_text).validate_subquestions(items).allowed


def _blank_result(source: str) -> dict[str, Any]:
    return {
        "is_clear": True,
        "clarification_question": "",
        "sub_questions": [],
        "sub_questions_source": source,
        "decompose_decided": False,
        "source": source,
    }


async def assess_question_clarity(
    *,
    question: str,
    chat_history: Sequence[dict] | None = None,
    api_key: str = "",
    model: str = "",
    provider: str = "",
    endpoint: str = "",
    call_ai_api: Optional[Callable[..., Any]] = None,
    evidence_need: Sequence[str] | None = None,
    allow_decomposition: bool = True,
) -> dict[str, Any]:
    """Return {is_clear, clarification_question, sub_questions, source, ...}.

    ``sub_questions`` 最多 3 条；只有一组经结构与语义校验的子问题才会设置
    ``decompose_decided=True``。空列表、畸形输出和语义拒绝都必须回落规则分解，
    不能把一次不可靠的 LLM 输出误当成"权威地无需拆分"。

    Fail-open: on any error, is_clear=True and sub_questions=[] so retrieval continues.
    """
    text = str(question or "").strip()
    result = _blank_result("fail_open")
    if not text or not api_key or not model:
        return _blank_result("skipped_missing_input")

    if call_ai_api is None:
        from services.chat_service import call_ai_api as _call_ai_api
        call_ai_api = _call_ai_api

    # 红线 3：表格/数值类问句发精简版 prompt，分解相关一个 token 都不进 LLM。
    numeric_table = is_numeric_table_question(text, evidence_need)
    decompose_enabled = bool(allow_decomposition) and not numeric_table
    system_prompt = (
        CLARIFY_SYSTEM_PROMPT_WITH_DECOMPOSE
        if decompose_enabled
        else CLARIFY_SYSTEM_PROMPT_NO_DECOMPOSE
    )
    # 三条中文子问题约 90-120 token；精简版沿用原来的 120 不动。
    max_tokens = 320 if decompose_enabled else 120
    skip_reason = (
        "skipped_numeric_table" if numeric_table
        else ("skipped_disabled" if not allow_decomposition else "")
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": CLARIFY_USER_PROMPT.format(
                history=_history_snippet(chat_history),
                question=text,
            ),
        },
    ]
    try:
        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint or "",
            max_tokens=max_tokens,
            temperature=0.0,
            purpose="intent_clarification",
        )
        require_publishable_completion(response, operation="intent clarification")
        content = _response_content(response)
        parsed, parse_source = _parse_clarify_payload(content)
        if not parsed:
            # 彻底解析不出来：is_clear 也不可信，整体 fail-open。
            out = _blank_result(f"unparsed:{parse_source}")
            out["source"] = "fail_open"
            out["sub_questions_source"] = "unparsed"
            return out

        is_clear = _coerce_llm_bool(parsed.get("is_clear", True), default=True)
        clarify = str(parsed.get("clarification_question") or "").strip()
        if not is_clear and len(clarify) < 4:
            clarify = "你的问题还不够具体。可以补充指代对象、章节、页码或比较目标吗？"
        if is_clear:
            clarify = ""

        sub_questions: list[str] = []
        decompose_decided = False
        if not decompose_enabled:
            sub_source = skip_reason or "skipped_disabled"
        else:
            candidates = normalize_sub_questions(
                parsed.get("sub_questions"),
                source_question=text,
            )
            if len(candidates) < 2:
                # 0 或 1 条都等价于"不用拆"——绝不制造"布尔 true + 空数组"的矛盾态。
                sub_source = "llm_empty"
            elif not subquestions_preserve_semantics(text, candidates):
                logger.debug(
                    "[Clarification] 子问题被语义等价锁丢弃: %s -> %s", text, candidates
                )
                sub_source = "rejected_semantics"
            else:
                sub_questions = candidates
                # Only a structurally valid and semantically complete group
                # can suppress the independent decomposition fallback.
                decompose_decided = True
                sub_source = parse_source

        return {
            "is_clear": is_clear,
            "clarification_question": clarify[:280],
            "sub_questions": sub_questions,
            "sub_questions_source": sub_source,
            "decompose_decided": decompose_decided,
            "source": parse_source,
        }
    except Exception as exc:
        logger.debug("[Clarification] LLM clarity check failed: %s", exc)
        out = _blank_result(f"error:{type(exc).__name__}")
        out["sub_questions_source"] = f"error:{type(exc).__name__}"
        return out


def build_decomposition_signals(
    *,
    source_question: str,
    clarity: dict | None = None,
) -> dict[str, Any]:
    """打包传给消费端的分解信号。

    必须带 ``source_question``：路由层的 resolved_question 与 agent 消费点的
    decomposition_question 在有引用/检索改写的回合可能不是同一个字符串，
    消费端要先严格比对，不等就当作没有 hint——否则会把 A 问句的子问题
    套到 B 问句上。
    """
    payload = clarity if isinstance(clarity, dict) else {}
    subs = payload.get("sub_questions")
    source = str(payload.get("sub_questions_source") or "not_attempted")
    decided = bool(payload.get("decompose_decided"))
    if not (isinstance(subs, list) and len(subs) >= 2) and not is_numeric_table_question(source_question):
        from services.paper_section_router import rule_decompose_facets

        facet_subs = rule_decompose_facets(source_question, limit=4)
        if len(facet_subs) >= 2:
            subs = facet_subs
            decided = True
            source = "rule_paper_facets"
    return {
        "source_question": str(source_question or "").strip(),
        "sub_questions": list(subs) if isinstance(subs, list) else [],
        "decided": decided,
        "source": source,
    }


def read_decomposition_signals(
    signals: Any,
    question: str,
) -> tuple[list[str], bool, str]:
    """(sub_questions, decided, source)。身份不匹配或结构畸形一律当没有信号。"""
    try:
        if not isinstance(signals, dict):
            return [], False, "absent"
        expected = str(signals.get("source_question") or "").strip()
        actual = str(question or "").strip()
        if not expected or expected != actual:
            return [], False, "question_mismatch"
        raw = signals.get("sub_questions")
        source = str(signals.get("source") or "")
        limit = 4 if source == "rule_paper_facets" else MAX_SUB_QUESTIONS
        subs = normalize_sub_questions(raw, source_question=actual, limit=limit)
        if len(subs) < 2:
            subs = []
        return subs, bool(signals.get("decided")), str(signals.get("source") or "")
    except Exception:  # pragma: no cover - 结构畸形也绝不阻断检索
        return [], False, "signal_error"


__all__ = [
    "CLARIFY_SYSTEM_PROMPT",
    "CLARIFY_SYSTEM_PROMPT_NO_DECOMPOSE",
    "CLARIFY_SYSTEM_PROMPT_WITH_DECOMPOSE",
    "CLARIFY_USER_PROMPT",
    "MAX_SUB_QUESTIONS",
    "assess_question_clarity",
    "build_decomposition_signals",
    "is_numeric_table_question",
    "normalize_sub_questions",
    "read_decomposition_signals",
    "should_attempt_llm_clarification",
    "strip_think_blocks",
    "subquestions_preserve_semantics",
]
