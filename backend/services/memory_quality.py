"""自动记忆质量守卫，隔离传输/检索故障产生的伪 QA。"""

from __future__ import annotations

import re


_RETRY_CONTROL_QUESTIONS = {
    "继续",
    "重新回答",
    "重新生成",
    "重答",
    "再回答一次",
    "再答一次",
    "请重新回答",
    "请重答",
    "continue",
    "retry",
    "tryagain",
    "regenerate",
}

_AUTOMATIC_FAILURE_PREFIXES = (
    "当前请求未能返回可用检索证据",
    "根据当前已检索到的文档证据，可先给出以下保守回答",
    "根据提供的文档片段，无法回答",
    "根据当前检索到的文档片段，无法回答",
    "当前检索到的文档片段不足以回答",
    "当前可用的文档片段不足以回答",
    "检索到的内容仅包含",
)

_AUTOMATIC_SOURCE_TYPES = {"auto_qa", "llm_distilled", "compressed", "working_memory"}
_QA_SEGMENT_RE = re.compile(
    r"(?:^|\s)Q:\s*(?P<question>.*?)\s+A:\s*(?P<answer>.*?)(?=(?:\s+Q:\s*)|\Z)",
    re.DOTALL,
)
_DOCUMENT_ABSENCE_RE = re.compile(
    r"\u672a(?:\u7ed9\u51fa|\u8bf4\u660e|\u63d0\u4f9b|\u516c\u5f00)|\u6ca1\u6709(?:\u7ed9\u51fa|\u8bf4\u660e|\u63d0\u4f9b)|\u672a\u62ab\u9732|\u4e0d(?:\u6e05\u695a|\u660e\u786e)|\u65e0\u6cd5(?:\u786e\u8ba4|\u5f97\u77e5)|"
    r"\b(?:does not|doesn't|did not|has not|have not)\s+(?:give|provide|describe|specify|disclose)\b|"
    r"\bnot\s+(?:given|provided|described|specified|disclosed)\b|\b(?:unclear|unknown)\b",
    re.IGNORECASE,
)
_ARCHITECTURE_BROAD_RE = re.compile(
    r"\u67b6\u6784|\u7ed3\u6784|\u62d3\u6251|\u673a\u5236|\u4ea4\u4e92|\u6d41\u7a0b|\u7f51\u7edc|\u68c0\u6d4b\u5934|"
    r"architecture|structure|topology|mechanism|interaction|pipeline|network|detector",
    re.IGNORECASE,
)
_ARCHITECTURE_DETAIL_SCOPE_RE = re.compile(
    r"\u9010\u5c42|\u5c42\u6570|\u901a\u9053|\u7ef4\u5ea6|\u5f20\u91cf|\u6295\u5f71|\u5f52\u4e00\u5316|\u914d\u7f6e|\u8d85\u53c2|\u5177\u4f53(?:\u6a21\u5757|\u5b57\u6bb5|\u5b9e\u73b0)|\u5b9e\u73b0\u7ec6\u8282|"
    r"\b(?:layer(?:s)?|channel(?:s)?|dimension(?:s)?|tensor(?:s)?|projection|normalization|"
    r"configuration|hyperparameter(?:s)?|implementation(?:\s+details?)?)\b",
    re.IGNORECASE,
)
_CLAIM_BOUNDARY_RE = re.compile(r"(?:\r?\n+|(?<=[\u3002\uff01\uff1f!?])\s*)")
_DOCUMENT_ABSENCE_ASSERTION_RE = re.compile(
    r"\u672a(?:\u7ed9\u51fa|\u8bf4\u660e|\u63d0\u4f9b|\u516c\u5f00|\u62ab\u9732|\u8ba8\u8bba|\u6d89\u53ca|\u5305\u542b|\u62a5\u544a|\u5c55\u793a|\u8bb0\u8f7d|\u8fdb\u884c|\u5f00\u5c55|\u91c7\u7528|\u4f7f\u7528|\u8bc4\u4f30|\u6bd4\u8f83|\u6d4b\u8bd5)|\u6ca1\u6709(?:\u7ed9\u51fa|\u8bf4\u660e|\u63d0\u4f9b|\u516c\u5f00|\u62ab\u9732|\u8ba8\u8bba|\u6d89\u53ca|\u5305\u542b|\u62a5\u544a|\u5c55\u793a|\u8bb0\u8f7d|\u8fdb\u884c|\u5f00\u5c55|\u91c7\u7528|\u4f7f\u7528|\u8bc4\u4f30|\u6bd4\u8f83|\u6d4b\u8bd5)?|\u4e0d(?:\u5305\u542b|\u6d89\u53ca|\u8ba8\u8bba|\u63d0\u4f9b|\u62ab\u9732|\u8bf4\u660e)|\u65e0(?:\u5173|\u8bb0\u8f7d|\u63d0\u53ca)|"
    r"\b(?:does\s+not|doesn't|did\s+not|has\s+not|have\s+not)\s+(?:give|provide|describe|specify|disclose|mention|discuss|include|report|contain|cover|state|perform|conduct|run|use|evaluate|compare)\b|"
    r"\b(?:not\s+(?:given|provided|described|specified|disclosed|mentioned|discussed|included|reported|contained|covered|documented|shown|stated)|no\s+(?:mention|discussion|report|evidence|description|details?))\b|"
    r"\b(?:no|without)\s+(?:(?:[a-z][a-z0-9_-]*\s+){0,6}?)(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?(?:given|provided|described|specified|disclosed|mentioned|discussed|included|reported|contained|covered|documented|shown|stated|performed|conducted|run)\b",
    re.IGNORECASE,
)
_EVIDENCE_LIMITED_SCOPE_RE = re.compile(r"(?:\u5f53\u524d|\u73b0\u6709|\u5df2\u68c0\u7d22|\u5df2\u63d0\u4f9b|\u53ef\u89c1|\u7ed9\u5b9a|\u6240\u63d0\u4f9b).{0,32}(?:\u8bc1\u636e|\u4e0a\u4e0b\u6587|\u7247\u6bb5|\u68c0\u7d22|\u6458\u5f55|\u8282\u9009)|\b(?:current(?:ly)?|retrieved|provided|visible|available)\s+(?:evidence|context|excerpt|passage|snippet|material)\b", re.IGNORECASE)
_EXPLICIT_ABSENCE_EVIDENCE_RE = re.compile(r"(?:\u4f5c\u8005|\u8bba\u6587|\u6587\u4e2d|\u672c\u6587).{0,28}(?:\u660e\u786e|\u76f4\u63a5|\u6e05\u695a).{0,28}(?:\u8bf4\u660e|\u5199\u660e|\u6307\u51fa|\u58f0\u660e|\u62ab\u9732)|\b(?:the\s+)?(?:paper|document|article|study|authors?)\s+(?:explicitly|directly|clearly)\s+(?:state|states|stated|note|notes|noted|say|says|said|mention|mentions|mentioned|disclose|discloses|disclosed)\b", re.IGNORECASE)
_INLINE_CITATION_RE = re.compile(r"(?<!!)(?:\[\d{1,3}\](?!\()|\u3010\d{1,3}\u3011)")


def is_unscoped_architecture_absence_claim(value: str) -> bool:
    """Detect an unsafe broad architecture-absence claim in generated text.

    A paper may omit layer/channel/projection details while still documenting
    its architecture-level topology. For QA records, inspect only the answer
    so detail words in the user's question cannot accidentally make an unsafe
    answer look scoped.
    """
    text = str(value or "").strip()
    if not text:
        return False

    qa_segments = list(_QA_SEGMENT_RE.finditer(text))
    candidates = (
        [match.group("answer").strip() for match in qa_segments]
        if qa_segments
        else [piece.strip() for piece in _CLAIM_BOUNDARY_RE.split(text)]
    )
    return any(
        candidate
        and _DOCUMENT_ABSENCE_RE.search(candidate)
        and _ARCHITECTURE_BROAD_RE.search(candidate)
        and not _ARCHITECTURE_DETAIL_SCOPE_RE.search(candidate)
        for candidate in candidates
    )


def find_unscoped_document_absence_claims(value: str) -> list[str]:
    """Return unsupported source-absence claims that lack an evidence boundary.

    Callers only pass generated assistant text, so an omitted subject still has
    document scope: ``No ablation study is reported.`` must not become a
    persistent fact merely because it does not repeat the word ``paper``.
    """
    text = str(value or "").strip()
    if not text:
        return []

    qa_segments = list(_QA_SEGMENT_RE.finditer(text))
    candidates = (
        [match.group("answer").strip() for match in qa_segments]
        if qa_segments
        else [piece.strip() for piece in _CLAIM_BOUNDARY_RE.split(text)]
    )
    unsafe: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if not _DOCUMENT_ABSENCE_ASSERTION_RE.search(candidate):
            continue
        if _EVIDENCE_LIMITED_SCOPE_RE.search(candidate):
            continue
        if (
            _EXPLICIT_ABSENCE_EVIDENCE_RE.search(candidate)
            and _INLINE_CITATION_RE.search(candidate)
        ):
            continue
        unsafe.append(candidate)
    return unsafe


def is_unscoped_document_absence_claim(value: str) -> bool:
    """Whether generated text turns an incomplete context into a document fact."""
    return bool(find_unscoped_document_absence_claims(value))


def is_unsafe_automatic_document_answer(value: str) -> bool:
    """Reject operational failures and unsupported document-wide absence claims."""
    return bool(
        is_unusable_automatic_answer(value)
        or is_unscoped_document_absence_claim(value)
    )

def normalize_retry_control_question(question: str) -> str:
    """返回控制型重试话术的规范形式；普通问题返回空字符串。"""
    normalized = re.sub(r"[\s，。！？、,.!?;；:：]+", "", str(question or "")).casefold()
    return normalized if normalized in _RETRY_CONTROL_QUESTIONS else ""


def _is_operational_failure_text(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text.startswith(("❌", "⚠️ AI未返回")):
        return True
    if text.startswith(_AUTOMATIC_FAILURE_PREFIXES):
        return True

    has_retrieval_preamble = any(
        marker in text
        for marker in (
            "文档片段",
            "检索到的内容",
            "检索证据",
            "当前可用的内容",
        )
    )
    has_operational_failure = bool(re.search(
        r"(?:无法|不能|不足以)(?:基于文档|回答|确定|确认)|(?:未出现|未包含).*(?:所需|相关|核心)",
        text,
    ))
    return has_retrieval_preamble and has_operational_failure


def sanitize_automatic_memory_content(value: str, source_type: str = "auto_qa") -> str:
    """移除自动记忆中的故障片段，同时保留同一压缩条目里的有效事实。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if str(source_type or "").strip().lower() not in _AUTOMATIC_SOURCE_TYPES:
        return text

    qa_segments = list(_QA_SEGMENT_RE.finditer(text))
    if qa_segments:
        retained = []
        for match in qa_segments:
            question = match.group("question").strip()
            answer = match.group("answer").strip()
            if (
                not question
                or _is_operational_failure_text(answer)
                or is_unscoped_document_absence_claim(answer)
            ):
                continue
            retained.append(f"Q: {question}\nA: {answer}")
        return "\n\n".join(retained)

    if _is_operational_failure_text(text) or is_unscoped_document_absence_claim(text):
        return ""
    return text


def is_unusable_automatic_answer(value: str) -> bool:
    """判断答案是否只是上游空流、检索不足或重试失败的运行态文案。"""
    return not sanitize_automatic_memory_content(value, "auto_qa")
