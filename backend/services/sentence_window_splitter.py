"""Sentence Window 分块策略

参考 kotaemon 的 SentenceWindowSplitter：
- 按句子切分文本
- 每个句子节点保留前后各 window_size 句作为上下文窗口
- 检索时用单句嵌入匹配（精确），上下文组装时用窗口文本（完整）

无需 llama-index 依赖，纯 Python 实现。
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 中英文句子分割正则
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[。！？.!?])\s*'
    r'|(?<=\n)\s*'
)


def split_sentences(text: str) -> list[str]:
    """将文本按句子分割

    支持中英文句号、感叹号、问号和换行符。

    Args:
        text: 输入文本

    Returns:
        句子列表（去除空白句子）
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def build_sentence_windows(
    text: str,
    window_size: int = 3,
    page: int = 0,
) -> list[dict]:
    """将文本按句子切分并为每句构建上下文窗口

    Args:
        text: 输入文本
        window_size: 前后各保留的句子数
        page: 页码（用于 metadata）

    Returns:
        列表，每项包含:
        - sentence: 原始单句文本（用于嵌入匹配）
        - window: 包含上下文窗口的完整文本（用于 LLM 上下文）
        - page: 页码
        - sentence_idx: 句子在原文中的索引
    """
    sentences = split_sentences(text)

    if not sentences:
        return []

    results = []
    for i, sentence in enumerate(sentences):
        # 计算窗口范围
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        # 构建窗口文本
        window_sentences = sentences[start:end]
        window_text = " ".join(window_sentences)

        results.append({
            "sentence": sentence,
            "window": window_text,
            "page": page,
            "sentence_idx": i,
        })

    logger.debug(
        f"[SentenceWindow] {len(sentences)} 句, "
        f"window_size={window_size}, page={page}"
    )
    return results


def build_sentence_window_chunks(
    pages: list[dict],
    window_size: int = 3,
    min_sentence_len: int = 5,
) -> tuple[list[str], list[str], dict]:
    """从多页文本构建 sentence window 分块

    Args:
        pages: [{"page": int, "text": str}, ...]
        window_size: 前后各保留的句子数
        min_sentence_len: 最短句子长度（过滤噪声）

    Returns:
        (chunks, windows, metadata)
        - chunks: 单句列表（用于嵌入索引）
        - windows: 对应的窗口文本列表（用于 LLM 上下文）
        - metadata: {"chunk_to_window": {idx: window_text}, "chunk_to_page": {idx: page}}
    """
    chunks = []
    windows = []
    chunk_to_page = {}

    for page_data in pages:
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")

        if not text:
            continue

        sentence_windows = build_sentence_windows(
            text, window_size=window_size, page=page_num
        )

        for sw in sentence_windows:
            sentence = sw["sentence"]
            if len(sentence) < min_sentence_len:
                continue

            idx = len(chunks)
            chunks.append(sentence)
            windows.append(sw["window"])
            chunk_to_page[idx] = page_num

    logger.info(
        f"[SentenceWindow] 生成 {len(chunks)} 个句子分块, "
        f"window_size={window_size}"
    )

    metadata = {
        "chunk_to_window": {i: w for i, w in enumerate(windows)},
        "chunk_to_page": chunk_to_page,
    }

    return chunks, windows, metadata
