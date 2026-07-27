"""
Selected Text 页码定位模块

在文档页面数据中定位用户框选文本所在的页码位置。
支持精确子串匹配、模糊匹配和跨页检测。
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def locate_selected_text(
    selected_text: str,
    pages: list[dict],
) -> dict:
    """定位 selected_text 在文档中的页码位置

    匹配策略：
    1. 精确子串匹配：在每页 content 中查找 selected_text
    2. 模糊匹配：取 selected_text 前 80 字符进行子串匹配
    3. 回退：返回默认页码 1，并把 ``match`` 标成 ``"none"``

    结果里的 ``match`` 字段是必读的。回退到第 1 页并不代表文本真的在第 1 页——
    MinerU 重建后的 pages 是规范化文本（去页眉页脚、双栏重排、表格转 markdown），
    和 PDF 文本层里选中的原始串经常对不上。调用方把这个回退当成真实页码用，
    就会在 prompt 里向模型断言一个错误页码，并把用户跳转到第 1 页。

    Args:
        selected_text: 用户框选的文本
        pages: 文档页面数据列表，每项包含 {"page": int, "content": str}

    Returns:
        {"page_start": int, "page_end": int, "match": "exact"|"prefix"|"none"}
    """
    # 空文本或空页面列表，直接回退
    if not selected_text or not selected_text.strip() or not pages:
        logger.warning("selected_text 为空或页面数据为空，返回默认页码 1")
        return {"page_start": 1, "page_end": 1, "match": "none"}

    # 策略 1：精确子串匹配
    matched_pages = _exact_match(selected_text, pages)
    if matched_pages:
        return _build_result(matched_pages, "exact")

    # 策略 2：模糊匹配（取前 80 字符）
    prefix = selected_text[:80]
    matched_pages = _exact_match(prefix, pages)
    if matched_pages:
        return _build_result(matched_pages, "prefix")

    # 策略 3：回退
    logger.warning(
        "selected_text 无法在任何页面中匹配到，返回默认页码 1。"
        f"文本前 40 字符: {selected_text[:40]!r}"
    )
    return {"page_start": 1, "page_end": 1, "match": "none"}


def selected_page_is_resolved(selected_page_info: Optional[dict]) -> bool:
    """Whether a locator result actually found the text.

    Legacy results predate the ``match`` field; treat them as resolved so an
    older cached payload does not suddenly lose its page label.
    """
    if not isinstance(selected_page_info, dict):
        return False
    return str(selected_page_info.get("match") or "exact").strip().lower() != "none"


def _exact_match(text: str, pages: list[dict]) -> list[int]:
    """在每页 content 中查找 text，返回匹配到的页码列表"""
    matched = []
    for page in pages:
        content = page.get("content", "")
        page_num = page.get("page", 1)
        if text in content:
            matched.append(page_num)
    return matched


def _build_result(matched_pages: list[int], match: str) -> dict:
    """根据匹配到的页码列表构建返回结果，支持跨页"""
    return {
        "page_start": min(matched_pages),
        "page_end": max(matched_pages),
        "match": match,
    }
