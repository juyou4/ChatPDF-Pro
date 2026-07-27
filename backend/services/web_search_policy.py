"""联网检索策略的唯一权威实现。

`routes/chat_routes.py` 与 `services/agent_retrieval_service.py` 曾各自维护一份
逐字重复的 `_resolve_web_search_mode`，两份实现漂移时会让"路由层认为不联网、
Agent 层却发起出站查询"这类不一致悄悄发生。此模块把该判定收敛为单一来源，
两个调用方一律 import 转发。

本模块只做纯策略换算，不含任何 I/O 与网络调用，因此可被 route 与 service 双向
安全导入而不产生循环依赖。
"""

from __future__ import annotations

from typing import Any, Literal

WebSearchMode = Literal["off", "auto", "force"]

_VALID_MODES: frozenset[str] = frozenset({"off", "auto", "force"})


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
