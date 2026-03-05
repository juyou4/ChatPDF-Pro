"""邻居 chunk 上下文扩展服务

命中的 chunk 前后各扩展 N 个邻居 chunk，将孤立匹配扩展为连续上下文段落。
作为 sentence window 和 semantic group 的补充策略。

参考 chatpdf1 的 num_expand_context_chunk 机制。
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def expand_context_chunks(
    results: List[dict],
    all_chunks: List[str],
    expand_n: int = 1,
    chunk_key: str = "chunk",
) -> List[dict]:
    """对检索结果进行邻居 chunk 扩展

    将每个命中 chunk 的前后各 expand_n 个 chunk 合并为一个扩展文本，
    写入 result["expanded_chunk"] 字段，原始 chunk 保持不变。

    Args:
        results: 检索结果列表，每项需包含 chunk_key 和可选的 index 字段
        all_chunks: 完整的 chunk 列表（按文档顺序排列）
        expand_n: 前后各扩展的 chunk 数，0 = 不扩展
        chunk_key: chunk 文本的字段名

    Returns:
        扩展后的结果列表（原地修改 + 返回）
    """
    if expand_n <= 0 or not all_chunks or not results:
        return results

    # 构建 chunk 文本 → 索引的反向映射（用于没有 index 字段的情况）
    chunk_to_idx = {}
    for i, c in enumerate(all_chunks):
        if c not in chunk_to_idx:
            chunk_to_idx[c] = i

    expanded_count = 0
    seen_ranges = set()

    for item in results:
        chunk_text = item.get(chunk_key, "")
        # 优先使用 index 字段，否则通过文本匹配查找
        idx = item.get("index")
        if idx is None:
            idx = chunk_to_idx.get(chunk_text)
        if idx is None:
            continue

        # 计算扩展范围
        start = max(0, idx - expand_n)
        end = min(len(all_chunks), idx + expand_n + 1)

        # 避免重复扩展相同范围
        range_key = (start, end)
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)

        # 合并邻居 chunk
        if start == idx and end == idx + 1:
            # 无扩展（首尾 chunk）
            continue

        expanded_text = "\n".join(all_chunks[start:end])
        item["expanded_chunk"] = expanded_text
        expanded_count += 1

    if expanded_count > 0:
        logger.debug(f"[ChunkExpander] 扩展 {expanded_count}/{len(results)} 个 chunk (±{expand_n})")

    return results
