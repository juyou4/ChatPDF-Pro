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
            if not question or _is_operational_failure_text(answer):
                continue
            retained.append(f"Q: {question}\nA: {answer}")
        return "\n\n".join(retained)

    if _is_operational_failure_text(text):
        return ""
    return text


def is_unusable_automatic_answer(value: str) -> bool:
    """判断答案是否只是上游空流、检索不足或重试失败的运行态文案。"""
    return not sanitize_automatic_memory_content(value, "auto_qa")
