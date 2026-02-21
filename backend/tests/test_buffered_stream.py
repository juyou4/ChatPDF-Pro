"""_buffered_stream 异步生成器单元测试

测试 chat_routes.py 中 _buffered_stream 函数的各种场景：
- 直通模式（buffer_size=0）
- 正常缓冲合并
- error chunk 立即刷新
- done chunk 立即刷新
- 空流处理
- reasoning_content 累积

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""
import sys
import os
import asyncio
from unittest.mock import patch

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 辅助函数：构造异步生成器模拟原始流
# ============================================================

async def _make_async_stream(chunks):
    """将 chunk 列表转换为异步生成器"""
    for chunk in chunks:
        yield chunk


# ============================================================
# 单元测试
# ============================================================

class TestBufferedStreamPassthrough:
    """直通模式测试：stream_buffer_size=0 时不做任何缓冲"""

    @pytest.mark.asyncio
    async def test_passthrough_when_buffer_size_zero(self):
        """buffer_size=0 时所有 chunk 原样透传"""
        chunks = [
            {"content": "你", "reasoning_content": "", "done": False},
            {"content": "好", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 0
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 直通模式：输出数量等于输入数量
        assert len(output) == len(chunks)
        for i, chunk in enumerate(output):
            assert chunk == chunks[i]

    @pytest.mark.asyncio
    async def test_passthrough_with_error_chunk(self):
        """buffer_size=0 时 error chunk 也原样透传"""
        chunks = [
            {"content": "部分", "reasoning_content": "", "done": False},
            {"error": "连接超时"},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 0
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        assert len(output) == 2
        assert output[0] == chunks[0]
        assert output[1] == chunks[1]


class TestBufferedStreamMerge:
    """缓冲合并测试：多个小 chunk 合并为大 chunk"""

    @pytest.mark.asyncio
    async def test_merge_small_chunks(self):
        """多个小 chunk 合并后发送"""
        # buffer_size=10，每个 chunk 3 字符，需要 4 个才能达到阈值
        chunks = [
            {"content": "你好啊", "reasoning_content": "", "done": False},
            {"content": "世界真", "reasoning_content": "", "done": False},
            {"content": "美丽呀", "reasoning_content": "", "done": False},
            {"content": "是不是", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 10
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 拼接所有输出的 content 应等于输入的 content 拼接
        input_content = "".join(c["content"] for c in chunks)
        output_content = "".join(c["content"] for c in output)
        assert output_content == input_content

        # 输出 chunk 数量应少于输入（合并效果）
        # 4 个内容 chunk + 1 个 done chunk -> 合并后应少于 5 个
        assert len(output) < len(chunks)

    @pytest.mark.asyncio
    async def test_large_chunk_immediate_send(self):
        """单个 chunk 已超过阈值时立即发送"""
        chunks = [
            {"content": "这是一段很长的文本内容超过阈值", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 5
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 第一个 chunk 超过阈值，应立即发送
        assert output[0]["content"] == "这是一段很长的文本内容超过阈值"
        assert output[-1]["done"] is True


class TestBufferedStreamFlush:
    """刷新行为测试：error/done 时立即刷新缓冲区"""

    @pytest.mark.asyncio
    async def test_flush_on_done(self):
        """收到 done=True 时刷新缓冲区中的剩余内容"""
        chunks = [
            {"content": "你好", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 20  # 阈值大于内容，不会自动触发
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 应有 2 个输出：刷新的缓冲内容 + done chunk
        assert len(output) == 2
        assert output[0]["content"] == "你好"
        assert output[0]["done"] is False
        assert output[1]["done"] is True

    @pytest.mark.asyncio
    async def test_flush_on_error(self):
        """收到 error chunk 时刷新缓冲区中的剩余内容"""
        chunks = [
            {"content": "部分", "reasoning_content": "思考", "done": False},
            {"error": "API 错误"},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 20
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 应有 2 个输出：刷新的缓冲内容 + error chunk
        assert len(output) == 2
        assert output[0]["content"] == "部分"
        assert output[0]["reasoning_content"] == "思考"
        assert output[1].get("error") == "API 错误"

    @pytest.mark.asyncio
    async def test_no_flush_when_buffer_empty_on_done(self):
        """缓冲区为空时收到 done，不应产生额外的刷新 chunk"""
        # 第一个 chunk 刚好达到阈值被发送，然后 done 到来时缓冲区为空
        chunks = [
            {"content": "12345", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 5
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 第一个 chunk 达到阈值被发送，done chunk 直接转发，无额外刷新
        assert len(output) == 2
        assert output[0]["content"] == "12345"
        assert output[1]["done"] is True


class TestBufferedStreamReasoningContent:
    """reasoning_content 累积测试"""

    @pytest.mark.asyncio
    async def test_reasoning_content_accumulated(self):
        """reasoning_content 也应正确累积和发送"""
        chunks = [
            {"content": "答", "reasoning_content": "思考步骤1", "done": False},
            {"content": "案", "reasoning_content": "思考步骤2", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 20
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 所有 reasoning_content 拼接应完整
        input_reasoning = "".join(c["reasoning_content"] for c in chunks)
        output_reasoning = "".join(c.get("reasoning_content", "") for c in output)
        assert output_reasoning == input_reasoning


class TestBufferedStreamEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_stream_with_only_done(self):
        """只有 done chunk 的空流"""
        chunks = [
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 20
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        assert len(output) == 1
        assert output[0]["done"] is True

    @pytest.mark.asyncio
    async def test_single_content_chunk_then_done(self):
        """单个内容 chunk 后紧跟 done"""
        chunks = [
            {"content": "完整回答", "reasoning_content": "", "done": False},
            {"content": "", "reasoning_content": "", "done": True},
        ]

        with patch("routes.chat_routes.settings") as mock_settings:
            mock_settings.stream_buffer_size = 20
            from routes.chat_routes import _buffered_stream

            output = []
            async for chunk in _buffered_stream(_make_async_stream(chunks)):
                output.append(chunk)

        # 内容应完整保留
        output_content = "".join(c["content"] for c in output)
        assert output_content == "完整回答"
