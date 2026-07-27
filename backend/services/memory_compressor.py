"""记忆压缩整合模块

当同一文档的记忆条目数量超过阈值时，把多条记忆合并为精简事实。

压缩失败不是小概率事件（模型限流、超长输入、JSON 抽风），所以这里用
三级降级链而不是"失败即截断"：

  1. 整体压缩：把全部条目交给 LLM
  2. 剔除超长条目后重试：超长条目单独归档，并在结果里留下占位注记，
     让用户知道有东西没被摘要，而不是静默丢失
  3. 截断合并：不调 LLM，按时间分组拼接

压缩 prompt 采用"续接指令"式六段结构（借鉴 Chatbox），比自由式事实列表
更能保住用户约束与未决问题；并要求区分同名不同对象（借鉴 LightRAG），
避免多文档场景把同名概念压成一团。
"""
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from services.memory_llm import call_llm_sync, parse_bullet_list
from services.memory_store import MemoryEntry

logger = logging.getLogger(__name__)

# 默认值；可被 config 覆盖
DEFAULT_MAX_COMPRESSED = 5
DEFAULT_SUMMARY_LIMIT = 180
# 单条记忆超过这个字符数就算"超长"，第二级降级会把它剔出摘要输入
DEFAULT_OVERSIZED_CHARS = 2000

_COMPRESSION_SYSTEM_PROMPT = """你是论文阅读助手的记忆压缩器。把多条零散记忆合并成一份精简的"续接说明"，
让后续对话仅凭它就能接上下文。

按下面的顺序输出，每条一行、以 '- ' 开头，最多 {max_items} 条：
1. 用户约束：用户明确提过的要求与偏好，原话里的限定要逐字保留（如"回答用中文""必须给页码"）
2. 已定结论：已经确认的事实与结论，数值、单位、专有名词保持原样
3. 未决问题：尚未解决或用户追问过但没答完的点
4. 下一步指引：后续回答应当注意什么

冲突处理：
- 若两条记忆矛盾，先判断是不是同名不同对象（不同论文可以都有叫 baseline 的方法，
  不同表格可以都有 Acc 列）。是的话分别保留并各自标明所属对象。
- 确属同一对象的内部矛盾，则并列呈现并标注不确定，不要擅自二选一。

只输出列表本身，不要标题、编号或任何解释。"""


class MemoryCompressor:
    """记忆压缩整合器"""

    def __init__(
        self,
        compression_threshold: int = 20,
        max_compressed: int = DEFAULT_MAX_COMPRESSED,
        summary_limit: int = DEFAULT_SUMMARY_LIMIT,
        oversized_chars: int = DEFAULT_OVERSIZED_CHARS,
    ):
        """初始化压缩器

        Args:
            compression_threshold: 触发压缩的条目数量阈值
            max_compressed: 压缩后最多保留几条
            summary_limit: summary 字段截断长度
            oversized_chars: 单条记忆超过多少字符算"超长"
        """
        self.compression_threshold = compression_threshold
        self.max_compressed = max(1, int(max_compressed))
        self.summary_limit = max(20, int(summary_limit))
        self.oversized_chars = max(200, int(oversized_chars))

    # ==================== 触发判断 ====================

    def should_compress(self, doc_id: str, entries: list[MemoryEntry]) -> bool:
        """同一 doc_id 下的条目数超过阈值时触发。"""
        count = sum(1 for e in entries if e.doc_id == doc_id)
        return count > self.compression_threshold

    # ==================== 压缩主流程 ====================

    def compress(
        self,
        entries: list[MemoryEntry],
        api_key: str = None,
        model: str = None,
        api_provider: str = None,
    ) -> list[MemoryEntry]:
        """把多条记忆压缩为精简事实，走三级降级链。"""
        if not entries:
            return []

        original_count = len(entries)
        logger.info(f"[MemoryCompressor] 开始压缩，原始条目数: {original_count}")

        if api_key and model and api_provider:
            texts, skipped = self._llm_compress_with_degradation(
                entries, api_key, model, api_provider
            )
            if texts:
                compressed = self._build_compressed_entries(
                    texts, entries, source_label="llm", skipped=skipped
                )
                logger.info(
                    "[MemoryCompressor] LLM 压缩完成: %d -> %d 条（%d 条超长已归档未摘要）",
                    original_count,
                    len(compressed),
                    len(skipped),
                )
                return compressed

        result = self._fallback_compress(entries)
        logger.info(
            "[MemoryCompressor] 截断合并完成: %d -> %d 条", original_count, len(result)
        )
        return result

    def _llm_compress_with_degradation(
        self,
        entries: list[MemoryEntry],
        api_key: str,
        model: str,
        api_provider: str,
    ) -> tuple[Optional[list[str]], list[MemoryEntry]]:
        """两级 LLM 尝试，返回 (压缩文本列表, 被剔除的超长条目)。"""
        # 第一级：整体压缩
        texts = self._llm_compress(entries, api_key, model, api_provider)
        if texts:
            return texts, []

        # 第二级：剔除超长条目后重试，并带上首次失败的上下文
        normal, oversized = self._split_oversized(entries)
        if oversized and normal:
            logger.info(
                "[MemoryCompressor] 整体压缩失败，剔除 %d 条超长记忆后重试",
                len(oversized),
            )
            texts = self._llm_compress(
                normal,
                api_key,
                model,
                api_provider,
                prior_failure="上一次整体压缩失败，可能是输入过长；本次已剔除超长条目。",
            )
            if texts:
                return texts, oversized

        return None, []

    def _split_oversized(
        self, entries: list[MemoryEntry]
    ) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        normal: list[MemoryEntry] = []
        oversized: list[MemoryEntry] = []
        for entry in entries:
            if len(entry.content or "") > self.oversized_chars:
                oversized.append(entry)
            else:
                normal.append(entry)
        return normal, oversized

    def _llm_compress(
        self,
        entries: list[MemoryEntry],
        api_key: str,
        model: str,
        api_provider: str,
        prior_failure: str = "",
    ) -> Optional[list[str]]:
        """调用 LLM 合并记忆，失败返回 None。"""
        memory_texts = "\n".join(
            f"- {e.content}" for e in entries if (e.content or "").strip()
        )
        if not memory_texts.strip():
            return None

        user_content = f"原始记忆：\n{memory_texts}"
        if prior_failure:
            user_content = f"（{prior_failure}）\n\n{user_content}"

        messages = [
            {
                "role": "system",
                "content": _COMPRESSION_SYSTEM_PROMPT.format(max_items=self.max_compressed),
            },
            {"role": "user", "content": user_content},
        ]

        try:
            response = call_llm_sync(
                messages,
                api_key=api_key,
                model=model,
                provider=api_provider,
                max_tokens=800,
            )
        except Exception as exc:
            logger.warning(f"[MemoryCompressor] LLM 压缩调用失败: {exc}")
            return None

        facts = parse_bullet_list(response or "", limit=self.max_compressed)
        return facts or None

    # ==================== 结果构造 ====================

    def _summarize(self, text: str) -> str:
        if len(text) <= self.summary_limit:
            return text
        return text[: self.summary_limit] + "..."

    def _build_compressed_entries(
        self,
        texts: list[str],
        entries: list[MemoryEntry],
        *,
        source_label: str,
        skipped: list[MemoryEntry] = None,
    ) -> list[MemoryEntry]:
        skipped = skipped or []
        doc_id = entries[0].doc_id if entries else None
        max_importance = max((e.importance for e in entries), default=0.5)
        source_ids = [e.id for e in entries]

        compressed: list[MemoryEntry] = []
        for text in texts[: self.max_compressed]:
            compressed.append(
                MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=text,
                    source_type="compressed",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    doc_id=doc_id,
                    importance=max_importance,
                    memory_kind="consolidated",
                    memory_scope="document" if doc_id else "profile",
                    title="压缩记忆",
                    summary=self._summarize(text),
                    derived_from=source_ids,
                    trace={
                        "kind": "compression",
                        "source_type": source_label,
                        "source_entry_ids": source_ids,
                    },
                )
            )

        # 超长条目没进摘要，必须留下可发现的痕迹，否则等于静默丢信息。
        if skipped:
            note = (
                f"[{len(skipped)} 条超长记忆未被摘要，已归档保留原文，"
                f"可在记忆面板按来源链查看]"
            )
            compressed.append(
                MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=note,
                    source_type="compressed",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    doc_id=doc_id,
                    importance=max_importance,
                    memory_kind="consolidated",
                    memory_scope="document" if doc_id else "profile",
                    title="未摘要条目提示",
                    summary=note,
                    derived_from=[e.id for e in skipped],
                    trace={
                        "kind": "compression",
                        "source_type": "oversized_placeholder",
                        "source_entry_ids": [e.id for e in skipped],
                    },
                )
            )
        return compressed

    # ==================== 第三级降级 ====================

    def _fallback_compress(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """不调 LLM：按时间排序分组拼接。"""
        if not entries:
            return []

        max_importance = max(e.importance for e in entries)
        doc_id = entries[0].doc_id if entries else None

        if len(entries) <= self.max_compressed:
            result = []
            for e in entries:
                result.append(
                    MemoryEntry(
                        id=str(uuid.uuid4()),
                        content=e.content,
                        source_type="compressed",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        doc_id=e.doc_id,
                        importance=max_importance,
                        memory_tier=e.memory_tier,
                        tags=list(e.tags),
                        memory_kind="consolidated",
                        memory_scope="document" if e.doc_id else "profile",
                        title="压缩记忆",
                        summary=e.summary or self._summarize(e.content),
                        derived_from=[e.id],
                        trace={
                            "kind": "compression",
                            "source_type": "fallback",
                            "source_entry_ids": [e.id],
                        },
                    )
                )
            return result

        sorted_entries = sorted(entries, key=lambda e: e.created_at)
        chunk_size = math.ceil(len(sorted_entries) / self.max_compressed)
        chunks = [
            sorted_entries[i : i + chunk_size]
            for i in range(0, len(sorted_entries), chunk_size)
        ]

        result: list[MemoryEntry] = []
        for chunk in chunks[: self.max_compressed]:
            merged_content = " ".join(e.content for e in chunk if e.content.strip())
            if not merged_content.strip():
                continue
            result.append(
                MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=merged_content,
                    source_type="compressed",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    doc_id=doc_id,
                    importance=max_importance,
                    memory_kind="consolidated",
                    memory_scope="document" if doc_id else "profile",
                    title="压缩记忆",
                    summary=self._summarize(merged_content),
                    derived_from=[e.id for e in chunk],
                    trace={
                        "kind": "compression",
                        "source_type": "fallback",
                        "source_entry_ids": [e.id for e in chunk],
                    },
                )
            )
        return result
