"""缓冲器减少输出频率属性测试

Feature: chatpdf-performance-optimization, Property 3: 缓冲器减少输出频率

使用 hypothesis 进行属性测试，验证 _buffered_stream 缓冲器在处理多个小 chunk 时
输出的 chunk 数量严格小于输入的 chunk 数量。

**Validates: Requirements 2.1**
"""
import sys
import os
import asyncio
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck, assume

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
# Hypothesis 策略：生成小 chunk 序列
# ============================================================

@st.composite
def small_chunks_with_buffer(draw):
    """生成 buffer_size 和一组内容长度严格小于 buffer_size 的 chunk 序列

    约束：
    - buffer_size >= 5（确保有足够空间合并）
    - 至少 3 个内容 chunk，每个 content 长度在 [1, buffer_size-1] 之间
    - 所有小 chunk 的 content 总长度 >= buffer_size（确保至少能触发一次合并）
    - 最后追加一个 done=True 终止 chunk
    """
    # 生成 buffer_size，范围 5~100
    buffer_size = draw(st.integers(min_value=5, max_value=100))

    # 生成 3~20 个小 chunk
    num_chunks = draw(st.integers(min_value=3, max_value=20))

    content_chunks = []
    for _ in range(num_chunks):
        # 每个 chunk 的 content 长度在 [1, buffer_size-1]
        content_len = draw(st.integers(min_value=1, max_value=buffer_size - 1))
        content = draw(
            st.text(
                alphabet=st.characters(categories=("L", "N")),
                min_size=content_len,
                max_size=content_len,
            )
        )
        content_chunks.append({
            "content": content,
            "reasoning_content": "",
            "done": False,
        })

    # 确保所有小 chunk 的 content 总长度 >= buffer_size
    # 这样缓冲器至少会合并一次，从而减少输出数量
    total_content_len = sum(len(c["content"]) for c in content_chunks)
    assume(total_content_len >= buffer_size)

    # 追加终止 chunk
    all_chunks = content_chunks + [{"content": "", "reasoning_content": "", "done": True}]

    return buffer_size, content_chunks, all_chunks


# ============================================================
# Property 3：缓冲器减少输出频率
# **Validates: Requirements 2.1**
# ============================================================

class TestP3BufferReducesFrequency:
    """Property 3: 缓冲器减少输出频率

    对于任意包含 2 个以上小 chunk（每个 content 长度 < buffer_size）的序列，
    缓冲器输出的 chunk 数量严格小于输入的 chunk 数量。

    **Validates: Requirements 2.1**
    """

    @given(data=small_chunks_with_buffer())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_output_count_less_than_input(self, data):
        """属性：缓冲器输出 chunk 数量 < 输入 chunk 数量

        当所有输入 chunk 的 content 长度都小于 buffer_size 时，
        缓冲器会将多个小 chunk 合并，因此输出数量必然减少。
        """
        buffer_size, content_chunks, all_chunks = data

        # 输入 chunk 总数（含终止 chunk）
        input_count = len(all_chunks)

        # 运行缓冲器
        output_chunks = _run_buffered_stream(all_chunks, buffer_size)

        # 输出 chunk 数量
        output_count = len(output_chunks)

        assert output_count < input_count, (
            f"缓冲器未减少输出频率！\n"
            f"buffer_size={buffer_size}\n"
            f"输入 chunk 数={input_count}（含终止 chunk）\n"
            f"输出 chunk 数={output_count}\n"
            f"输入 content 长度列表={[len(c['content']) for c in content_chunks]}\n"
            f"输出 content 列表={[c.get('content', '') for c in output_chunks]}"
        )
