"""联网检索策略的唯一权威实现。

`routes/chat_routes.py` 与 `services/agent_retrieval_service.py` 曾各自维护一份
逐字重复的 `_resolve_web_search_mode`，两份实现漂移时会让"路由层认为不联网、
Agent 层却发起出站查询"这类不一致悄悄发生。此模块把该判定收敛为单一来源，
两个调用方一律 import 转发。

本模块只做纯策略换算，不含任何 I/O 与网络调用，因此可被 route 与 service 双向
安全导入而不产生循环依赖。
"""

from __future__ import annotations

import re
from typing import Any, Literal

WebSearchMode = Literal["off", "auto", "force"]

_VALID_MODES: frozenset[str] = frozenset({"off", "auto", "force"})


# 这组规则只识别「请系统替我发起一次外部检索」的直接指令，不能把讨论功能
# 本身的问句误当成出站授权。正则先把空白折叠，因而同时覆盖“联网 搜索一下”。
_DIRECT_WEB_COMMAND_RE = re.compile(
    r"^(?:(?:请|请你|麻烦|麻烦你|帮我|帮忙|给我|你|你能|你可以|"
    r"能否|可否|可以|能不能|是否可以|麻烦您))*"
    r"(?:联网|上网|网络|网页)(?:搜索|检索|查询|查找|搜|查)",
    re.IGNORECASE,
)
_DIRECT_ENGLISH_WEB_COMMAND_RE = re.compile(
    r"^(?:(?:please|couldyou|canyou|helpm(e)?))*"
    r"(?:search|lookup|lookfor|google)(?:the)?web",
    re.IGNORECASE,
)
_WEB_ACTION_RE = re.compile(
    r"(?:联网|上网|网络|网页)(?:搜索|检索|查询|查找|搜|查)",
    re.IGNORECASE,
)
_COMMAND_PREFIX_RE = re.compile(
    r"^(?:(?:请|请你|麻烦|麻烦你|帮我|帮忙|给我|你|你能|你可以|"
    r"能否|可否|可以|能不能|是否可以|麻烦您))*",
    re.IGNORECASE,
)
_COMMAND_SUFFIX_RE = re.compile(r"(?:一下|下|看看|吧|好吗|可以吗|行吗|呢|呀|啊|！|。|，|、|？|\?)+$")
_NEGATED_WEB_ACTION_RE = re.compile(
    r"(?:不要|别|不用|不需要|无需|禁止|关闭|关掉|停止)(?:再)?"
    r"(?:联网|上网|网络|网页)?(?:搜索|检索|查询|查找|搜|查)",
    re.IGNORECASE,
)


def _normalized_question(question: Any) -> str:
    return re.sub(r"\s+", "", str(question or "")).strip().lower()


def _is_web_search_explanation_question(normalized_question: str) -> bool:
    """Return whether the user is asking *about* web search rather than using it."""
    if not normalized_question:
        return False
    explanations = (
        "什么是联网搜索",
        "联网搜索是什么",
        "什么是网络搜索",
        "网络搜索是什么",
        "what is web search",
        "web search是什么",
    )
    if any(item in normalized_question for item in explanations):
        return True
    return bool(
        _WEB_ACTION_RE.match(normalized_question)
        and normalized_question[len(_WEB_ACTION_RE.match(normalized_question).group(0)):]
        .startswith(("是什么", "有什么", "怎么", "为什么", "能否", "是否"))
    )


def is_web_search_opt_out(question: Any) -> bool:
    """Whether this turn explicitly withdraws the default web-search preference."""
    return bool(_NEGATED_WEB_ACTION_RE.search(_normalized_question(question)))


def is_explicit_web_search_request(question: Any) -> bool:
    """Whether ``question`` directly authorizes one outbound web search.

    A toolbar switch expresses a default preference. A phrase such as
    ``"你联网搜索一下"`` is a per-turn user instruction and must not be left to
    the LLM auto gate. Negated requests and questions describing the feature are
    deliberately excluded so this helper is safe to use at the network boundary.
    """
    normalized = _normalized_question(question)
    if not normalized or is_web_search_opt_out(normalized):
        return False
    if _is_web_search_explanation_question(normalized):
        return False
    return bool(
        _DIRECT_WEB_COMMAND_RE.match(normalized)
        or _DIRECT_ENGLISH_WEB_COMMAND_RE.match(normalized)
    )


def is_command_only_explicit_web_search_request(question: Any) -> bool:
    """Return true for a direct request with no topic of its own.

    It lets the chat route safely bind an instruction such as ``"联网搜索一下"``
    to the preceding *user* question, instead of exporting the command text as
    a low-quality query. A request that includes a topic remains self-contained.
    """
    normalized = _normalized_question(question)
    if not is_explicit_web_search_request(normalized):
        return False
    remaining = _COMMAND_PREFIX_RE.sub("", normalized, count=1)
    remaining = _WEB_ACTION_RE.sub("", remaining, count=1)
    remaining = _COMMAND_SUFFIX_RE.sub("", remaining)
    return not remaining


def resolve_web_search_mode(request: Any) -> WebSearchMode:
    """把请求上的联网开关折算成三态策略。

    显式的 `web_search_mode` 优先；缺省或取值非法时，退回布尔开关
    `enable_web_search`：开为 `auto`（是否真的联网由后续判定决定），
    关为 `off`。
    """
    explicit = str(getattr(request, "web_search_mode", "") or "").strip().lower()
    if explicit in _VALID_MODES:
        return explicit  # type: ignore[return-value]
    return "auto" if bool(getattr(request, "enable_web_search", False)) else "off"


def resolve_effective_web_search_mode(request: Any, question: Any) -> WebSearchMode:
    """Resolve the per-turn network policy after considering direct consent.

    An explicit user command has stronger authority than the saved toolbar
    preference for this one turn. Callers that have already determined a
    command-only request has no usable topic should freeze their route as
    ``off`` rather than invoking this helper again.
    """
    if is_web_search_opt_out(question):
        return "off"
    if is_explicit_web_search_request(question):
        return "force"
    return resolve_web_search_mode(request)
