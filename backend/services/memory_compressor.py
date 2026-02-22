"""记忆压缩整合模块

负责将多条相关记忆合并压缩为精简事实，减少 token 占用。
支持 LLM 压缩（优先）和文本截断合并（降级）两种策略。
"""
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from services.memory_store import MemoryEntry

logger = logging.getLogger(__name__)


class MemoryCompressor:
    """记忆压缩整合器

    当同一文档的记忆条目数量超过阈值时，触发压缩流程。
    优先使用 LLM 将多条记忆合并为精简事实，失败时降级为截断合并。
    压缩后的条目数量不超过 5 条。
    """

    def __init__(self, compression_threshold: int = 20):
        """初始化压缩器

        Args:
            compression_threshold: 触发压缩的条目数量阈值，默认 20
        """
        self.compression_threshold = compression_threshold

    def should_compress(self, doc_id: str, entries: list[MemoryEntry]) -> bool:
        """判断是否需要触发压缩

        当指定 doc_id 下的记忆条目数量超过压缩阈值时返回 True。

        Args:
            doc_id: 文档 ID
            entries: 所有记忆条目列表

        Returns:
            是否需要压缩
        """
        count = sum(1 for e in entries if e.doc_id == doc_id)
        return count > self.compression_threshold

    def compress(
        self,
        entries: list[MemoryEntry],
        api_key: str = None,
        model: str = None,
        api_provider: str = None,
    ) -> list[MemoryEntry]:
        """压缩多条记忆为精简事实

        优先使用 LLM 压缩，失败时降级为截断合并。
        返回不超过 5 条压缩后的记忆条目。

        Args:
            entries: 待压缩的记忆条目列表
            api_key: LLM API 密钥（可选）
            model: LLM 模型名称（可选）
            api_provider: LLM 提供商（可选）

        Returns:
            压缩后的记忆条目列表（最多 5 条）
        """
        if not entries:
            return []

        # 记录压缩前条目数
        original_count = len(entries)
        logger.info(f"[MemoryCompressor] 开始压缩，原始条目数: {original_count}")

        # 计算原始记忆中的最大 importance
        max_importance = max(e.importance for e in entries)

        # 尝试 LLM 压缩
        if api_key and model and api_provider:
            try:
                llm_results = self._llm_compress(entries, api_key, model, api_provider)
                if llm_results:
                    # 将 LLM 返回的文本列表转换为 MemoryEntry
                    compressed = []
                    for text in llm_results[:5]:
                        entry = MemoryEntry(
                            id=str(uuid.uuid4()),
                            content=text,
                            source_type="compressed",
                            created_at=datetime.now(timezone.utc).isoformat(),
                            doc_id=entries[0].doc_id if entries else None,
                            importance=max_importance,
                        )
                        compressed.append(entry)
                    logger.info(
                        f"[MemoryCompressor] LLM 压缩完成，"
                        f"压缩前: {original_count} 条，压缩后: {len(compressed)} 条"
                    )
                    return compressed
            except Exception as e:
                logger.warning(f"[MemoryCompressor] LLM 压缩失败，降级为截断合并: {e}")

        # 降级为截断合并
        result = self._fallback_compress(entries)
        logger.info(
            f"[MemoryCompressor] 截断合并完成，"
            f"压缩前: {original_count} 条，压缩后: {len(result)} 条"
        )
        return result

    def _llm_compress(
        self,
        entries: list[MemoryEntry],
        api_key: str,
        model: str,
        api_provider: str,
    ) -> Optional[list[str]]:
        """使用 LLM 将多条记忆合并为精简事实

        Args:
            entries: 待压缩的记忆条目列表
            api_key: LLM API 密钥
            model: LLM 模型名称
            api_provider: LLM 提供商

        Returns:
            压缩后的文本列表（最多 5 条），失败返回 None
        """
        # 拼接所有记忆内容
        memory_texts = "\n".join(
            f"- {e.content}" for e in entries if e.content.strip()
        )

        prompt = (
            "请将以下多条记忆信息合并压缩为不超过 5 条精简事实陈述。\n"
            "每条事实应独立成句，保留最重要的信息。\n"
            "请直接输出事实列表，每行一条，以 '- ' 开头。\n\n"
            f"原始记忆：\n{memory_texts}"
        )

        try:
            # 动态导入 provider，避免循环依赖
            from providers.chat_provider import get_chat_provider

            provider = get_chat_provider(api_provider)
            response = provider.chat(
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            # 解析 LLM 返回的事实列表
            if response and response.strip():
                facts = []
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        facts.append(line[2:].strip())
                    elif line:
                        facts.append(line)
                return facts[:5] if facts else None

        except Exception as e:
            logger.warning(f"[MemoryCompressor] LLM 压缩调用失败: {e}")

        return None

    def _fallback_compress(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """降级策略：按时间排序拼接截断

        将记忆按 created_at 排序，均匀分为 5 组，
        每组内的记忆内容拼接合并为一条压缩记忆。

        Args:
            entries: 待压缩的记忆条目列表

        Returns:
            压缩后的记忆条目列表（最多 5 条）
        """
        if not entries:
            return []

        # 如果条目数 <= 5，直接标记为 compressed 并返回
        if len(entries) <= 5:
            result = []
            max_importance = max(e.importance for e in entries)
            for e in entries:
                compressed_entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=e.content,
                    source_type="compressed",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    doc_id=e.doc_id,
                    importance=max_importance,
                    memory_tier=e.memory_tier,
                    tags=list(e.tags),
                )
                result.append(compressed_entry)
            return result

        # 按 created_at 排序
        sorted_entries = sorted(entries, key=lambda e: e.created_at)

        # 计算原始记忆中的最大 importance
        max_importance = max(e.importance for e in entries)

        # 均匀分为 5 组
        chunk_count = 5
        chunk_size = math.ceil(len(sorted_entries) / chunk_count)
        chunks = []
        for i in range(0, len(sorted_entries), chunk_size):
            chunks.append(sorted_entries[i : i + chunk_size])

        # 每组合并为一条记忆
        result = []
        doc_id = entries[0].doc_id if entries else None
        for chunk in chunks[:5]:
            # 拼接组内所有记忆内容
            merged_content = " ".join(e.content for e in chunk if e.content.strip())
            if not merged_content.strip():
                continue

            compressed_entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=merged_content,
                source_type="compressed",
                created_at=datetime.now(timezone.utc).isoformat(),
                doc_id=doc_id,
                importance=max_importance,
            )
            result.append(compressed_entry)

        return result
