"""缓冲器内容保持 round-trip 属性测试

Feature: chatpdf-performance-optimization, Property 2: 缓冲器内容保持 round-trip

使用 hypothesis 进行属性测试，验证 _buffered_stream 缓冲器在任意 chunk 序列下
不丢失、不重复、不修改任何内容。

**Validates: Requirements 2.1, 2.3, 2.4**
"""
import sys
import os
import asyncio
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 辅助函数
# ============================================================

async def _make_async_stream(chunks):
    """将 chunk 列表转换为异步生成器"""
    for chunk in chunks:
        yield chunk


def _run_buffered_stream(input_chunks, buffer_size):
    """同步运行 _buffered_stream 并收集输出

    Args:
        input_chunks: 输入 chunk 列表
        buffer_size: 缓冲字符数阈值

    Returns:
        输出 chunk 列表
    """
    async def _collect():
        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = buffer_size
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(input_chunks)):
                output.append(chunk)
            return output

    return asyncio.get_event_loop().run_until_complete(_collect())


# ============================================================
# Hypothesis 策略：生成随机 chunk 序列
# ============================================================

# 内容字符串策略：包含中英文、空字符串等
content_strategy = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z", "S")),
    min_size=0,
    max_size=50,
)

# 单个内容 chunk 策略
content_chunk_strategy = st.fixed_dictionaries({
    "content": content_strategy,
    "reasoning_content": content_strategy,
    "done": st.just(False),
})

# 缓冲大小策略：0（直通）到 100
buffer_size_strategy = st.integers(min_value=0, max_value=100)


def build_chunk_sequence(content_chunks, done_content, done_reasoning):
    """构建完整的 chunk 序列（内容 chunks + 终止 chunk）"""
    chunks = list(content_chunks)
    chunks.append({
        "content": done_content,
        "reasoning_content": done_reasoning,
        "done": True,
    })
    return chunks


# ============================================================
# Property 2：缓冲器内容保持 round-trip
# **Validates: Requirements 2.1, 2.3, 2.4**
# ============================================================

class TestP2BufferRoundTrip:
    """Property 2: 缓冲器内容保持 round-trip

    对于任意非空 chunk 序列（每个 chunk 包含 content 字符串，
    最后一个 chunk 的 done=True），经过缓冲器处理后，
    所有输出 chunk 的 content 拼接结果等于所有输入 chunk 的 content 拼接结果。
    即缓冲器不丢失、不重复、不修改任何内容。

    **Validates: Requirements 2.1, 2.3, 2.4**
    """

    @given(
        content_chunks=st.lists(content_chunk_strategy, min_size=0, max_size=20),
        done_content=content_strategy,
        done_reasoning=content_strategy,
        buffer_size=buffer_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_content_roundtrip(
        self, content_chunks, done_content, done_reasoning, buffer_size
    ):
        """属性：所有输出 content 拼接 == 所有输入 content 拼接"""
        input_chunks = build_chunk_sequence(
            content_chunks, done_content, done_reasoning
        )

        output_chunks = _run_buffered_stream(input_chunks, buffer_size)

        # 计算输入侧 content 拼接
        input_content = "".join(c.get("content", "") for c in input_chunks)
        # 计算输出侧 content 拼接
        output_content = "".join(c.get("content", "") for c in output_chunks)

        assert output_content == input_content, (
            f"content round-trip 失败！\n"
            f"buffer_size={buffer_size}\n"
            f"输入 content='{input_content}'\n"
            f"输出 content='{output_content}'"
        )

    @given(
        content_chunks=st.lists(content_chunk_strategy, min_size=0, max_size=20),
        done_content=content_strategy,
        done_reasoning=content_strategy,
        buffer_size=buffer_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_reasoning_content_roundtrip(
        self, content_chunks, done_content, done_reasoning, buffer_size
    ):
        """属性：所有输出 reasoning_content 拼接 == 所有输入 reasoning_content 拼接"""
        input_chunks = build_chunk_sequence(
            content_chunks, done_content, done_reasoning
        )

        output_chunks = _run_buffered_stream(input_chunks, buffer_size)

        # 计算输入侧 reasoning_content 拼接
        input_reasoning = "".join(
            c.get("reasoning_content", "") for c in input_chunks
        )
        # 计算输出侧 reasoning_content 拼接
        output_reasoning = "".join(
            c.get("reasoning_content", "") for c in output_chunks
        )

        assert output_reasoning == input_reasoning, (
            f"reasoning_content round-trip 失败！\n"
            f"buffer_size={buffer_size}\n"
            f"输入 reasoning='{input_reasoning}'\n"
            f"输出 reasoning='{output_reasoning}'"
        )

    @given(
        content_chunks=st.lists(content_chunk_strategy, min_size=0, max_size=20),
        done_content=content_strategy,
        done_reasoning=content_strategy,
        buffer_size=buffer_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_done_chunk_always_present(
        self, content_chunks, done_content, done_reasoning, buffer_size
    ):
        """属性：输出序列的最后一个 chunk 必须是 done=True"""
        input_chunks = build_chunk_sequence(
            content_chunks, done_content, done_reasoning
        )

        output_chunks = _run_buffered_stream(input_chunks, buffer_size)

        # 输出不应为空（至少有 done chunk）
        assert len(output_chunks) > 0, "输出序列不应为空"

        # 最后一个 chunk 必须是 done=True
        assert output_chunks[-1].get("done") is True, (
            f"输出序列最后一个 chunk 应为 done=True，"
            f"实际为 {output_chunks[-1]}"
        )
