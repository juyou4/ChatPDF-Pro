"""
Grep 精确文本搜索服务

提供精确字面文本搜索（类似 grep），支持：
- 单个关键词精确匹配
- 多关键词 OR 逻辑（用 | 分隔）
- 大小写不敏感（可选）
- 上下文片段提取（可配置前后字符数）

参考 paper-burner-x 的 grep 工具设计：
- 适合搜索专有名词、特定数字、固定术语
- 支持 OR 逻辑：query 可用 | 分隔多个关键词
- 返回包含该短语的原文片段（前后 context_chars 字上下文）
"""

import re
from typing import List, Optional


def grep_search(
    query: str,
    text: str,
    limit: int = 20,
    context_chars: int = 2000,
    case_insensitive: bool = True,
) -> List[dict]:
    """精确文本搜索（grep 风格）

    Args:
        query: 搜索关键词，支持 | 分隔多关键词 OR 逻辑
               例如 "雷曼|Lehman" 匹配包含"雷曼"或"Lehman"的位置
        text: 要搜索的全文
        limit: 最大返回结果数，默认 20
        context_chars: 上下文片段的前后字符数，默认 2000
        case_insensitive: 是否大小写不敏感，默认 True

    Returns:
        匹配结果列表，每项包含:
        - match_text: 匹配的文本
        - match_offset: 匹配在全文中的偏移量
        - context_snippet: 上下文片段
        - score: 相关性分数（grep 固定为 1.0）
        - keyword: 匹配到的关键词
    """
    if not query or not text:
        return []

    # 解析 OR 逻辑：按 | 分隔关键词
    keywords = [kw.strip() for kw in query.split("|") if kw.strip()]
    if not keywords:
        return []

    # 构建正则表达式：将关键词用 | 连接，每个关键词转义特殊字符
    escaped_keywords = [re.escape(kw) for kw in keywords]
    pattern_str = "|".join(escaped_keywords)

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        pattern = re.compile(pattern_str, flags)
    except re.error:
        return []

    results = []
    seen_offsets = set()  # 去重：避免重叠区域产生重复结果

    for match in pattern.finditer(text):
        if len(results) >= limit:
            break

        match_start = match.start()
        match_end = match.end()

        # 跳过零宽度匹配
        if match_start == match_end:
            continue

        # 去重：如果与已有结果的上下文窗口重叠过多，跳过
        overlap = False
        for seen_start in seen_offsets:
            if abs(match_start - seen_start) < context_chars // 4:
                overlap = True
                break
        if overlap:
            continue

        seen_offsets.add(match_start)

        # 提取上下文片段
        context_start = max(0, match_start - context_chars)
        context_end = min(len(text), match_end + context_chars)

        # 尝试对齐到句子边界（向前找句号/换行，向后找句号/换行）
        context_start = _snap_to_boundary(text, context_start, direction="backward")
        context_end = _snap_to_boundary(text, context_end, direction="forward")

        context_snippet = text[context_start:context_end]

        results.append({
            "match_text": match.group(0),
            "match_offset": match_start,
            "context_snippet": context_snippet,
            "score": 1.0,
            "keyword": match.group(0),
        })

    return results


def _snap_to_boundary(text: str, pos: int, direction: str = "backward", max_search: int = 100) -> int:
    """将位置对齐到最近的句子/段落边界

    Args:
        text: 全文
        pos: 当前位置
        direction: "backward" 向前找边界，"forward" 向后找边界
        max_search: 最大搜索范围

    Returns:
        对齐后的位置
    """
    boundary_chars = "。\n.！？!?；;"

    if direction == "backward":
        search_start = max(0, pos - max_search)
        for i in range(pos, search_start, -1):
            if i < len(text) and text[i] in boundary_chars:
                return i + 1
        return pos
    else:  # forward
        search_end = min(len(text), pos + max_search)
        for i in range(pos, search_end):
            if text[i] in boundary_chars:
                return i + 1
        return pos
