"""证据上下文的统一 token 预算协调

在此之前记忆预算（默认 800）与文档检索预算（agent_context_max_tokens=12000）
是两本互不知晓的账：

- 文档上下文很长时，记忆照样按 800 硬塞，两边加起来可能顶爆模型窗口；
- 文档上下文很短时，记忆也拿不到多出来的余量。

这里让记忆知道"别人已经占了多少"，在总预算里取剩余额度，
并用下限保证记忆不会被完全挤没、用上限防止记忆反过来吃掉文档的空间。
"""
import logging

from services.context_injector import estimate_text_tokens

logger = logging.getLogger(__name__)

DEFAULT_TOTAL_EVIDENCE_TOKENS = 13000
DEFAULT_MEMORY_FLOOR_TOKENS = 200


def resolve_memory_token_budget(
    *,
    document_context: str = "",
    web_search_context: str = "",
    glossary_context: str = "",
    memory_ceiling: int,
    total_budget: int = DEFAULT_TOTAL_EVIDENCE_TOKENS,
    memory_floor: int = DEFAULT_MEMORY_FLOOR_TOKENS,
) -> dict[str, int]:
    """算出本轮记忆实际可用的 token 预算。

    Args:
        memory_ceiling: 记忆的配置上限（即原来的 memory_injection_token_budget）
        total_budget: 全部证据（文档 + 联网 + 记忆 + 术语）的总预算
        memory_floor: 记忆的保底额度——文档再长也要留一点，
            否则长文档会话里记忆等于被静默关掉

    Returns:
        含 resolved / ceiling / floor / others_used / total 的字典，
        既用于分配也用于回传给前端做透明化。
    """
    ceiling = max(0, int(memory_ceiling))
    floor = max(0, min(int(memory_floor), ceiling))
    total = max(0, int(total_budget))

    others_used = (
        estimate_text_tokens(document_context)
        + estimate_text_tokens(web_search_context)
        + estimate_text_tokens(glossary_context)
    )
    slack = total - others_used

    if slack >= ceiling:
        resolved = ceiling
    elif slack <= floor:
        # 文档已经吃满：记忆退到保底额度，而不是归零
        resolved = floor
    else:
        resolved = slack

    if resolved < ceiling:
        logger.debug(
            "[ContextBudget] 记忆预算收紧至 %d（上限 %d）：其它证据已占 %d/%d",
            resolved, ceiling, others_used, total,
        )

    return {
        "resolved": resolved,
        "ceiling": ceiling,
        "floor": floor,
        "others_used": others_used,
        "total": total,
    }
