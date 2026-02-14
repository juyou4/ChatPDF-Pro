"""OpenAI Provider 属性测试

使用 hypothesis 进行属性测试，验证请求体构建逻辑的正确性。
"""
import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from hypothesis import given, strategies as st, settings

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.openai_provider import OpenAICompatibleProvider


# ============================================================
# 辅助函数：调用 provider.chat 并捕获构建的请求体
# ============================================================

async def capture_request_body(**kwargs):
    """调用 OpenAICompatibleProvider.chat 并捕获发送的请求体

    通过 mock httpx.AsyncClient 拦截 post 请求，返回构建的 body。
    """
    provider = OpenAICompatibleProvider()
    captured_body = {}

    # 构造 mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    # mock httpx.AsyncClient 的 post 方法，捕获 json 参数
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            model="gpt-4",
            **kwargs,
        )

    # 从 mock 调用中提取 json 参数（即请求体）
    call_kwargs = mock_client.post.call_args
    captured_body = call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))
    return captured_body


# ============================================================
# P2: 可选参数透传属性测试
# **Validates: Requirements 6.1, 6.2, 6.5**
# ============================================================

# 可选参数的 hypothesis 策略：每个参数要么是 None，要么是有效值
optional_temperature = st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
optional_top_p = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
optional_max_tokens = st.one_of(st.none(), st.integers(min_value=1, max_value=128000))


class TestP2OptionalParamsPassthrough:
    """P2: 可选参数透传属性

    对于 temperature、top_p、max_tokens 三个可选参数的任意组合：
    - 值为 None 的参数不出现在请求体中
    - 值非 None 的参数以正确的值出现在请求体中

    **Validates: Requirements 6.1, 6.2, 6.5**
    """

    @given(
        temperature=optional_temperature,
        top_p=optional_top_p,
        max_tokens=optional_max_tokens,
    )
    @settings(max_examples=100)
    def test_optional_params_passthrough(self, temperature, top_p, max_tokens):
        """属性：None 参数不出现在请求体中，非 None 参数正确出现"""
        body = asyncio.get_event_loop().run_until_complete(
            capture_request_body(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )

        # 核心字段始终存在
        assert "model" in body
        assert "messages" in body
        assert "stream" in body

        # temperature: None 时不出现，非 None 时值正确
        if temperature is None:
            assert "temperature" not in body, "temperature 为 None 时不应出现在请求体中"
        else:
            assert body["temperature"] == temperature, f"temperature 应为 {temperature}，实际为 {body.get('temperature')}"

        # top_p: None 时不出现，非 None 时值正确
        if top_p is None:
            assert "top_p" not in body, "top_p 为 None 时不应出现在请求体中"
        else:
            assert body["top_p"] == top_p, f"top_p 应为 {top_p}，实际为 {body.get('top_p')}"

        # max_tokens: None 时不出现，非 None 时值正确
        if max_tokens is None:
            assert "max_tokens" not in body, "max_tokens 为 None 时不应出现在请求体中"
        else:
            assert body["max_tokens"] == max_tokens, f"max_tokens 应为 {max_tokens}，实际为 {body.get('max_tokens')}"


# ============================================================
# P3: 自定义参数合并属性测试
# **Validates: Requirements 4.5, 6.3**
# ============================================================

# 核心字段名称，自定义参数不应覆盖这些字段
CORE_FIELDS = {"model", "messages", "stream"}

# 自定义参数的 key 策略：生成非核心字段的字符串 key
custom_param_keys = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
).filter(lambda k: k not in CORE_FIELDS)

# 自定义参数的 value 策略：支持 string / number / boolean
custom_param_values = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    st.booleans(),
)

# 自定义参数字典策略
custom_params_strategy = st.dictionaries(
    keys=custom_param_keys,
    values=custom_param_values,
    min_size=1,
    max_size=10,
)


class TestP3CustomParamsMerge:
    """P3: 自定义参数合并属性

    对于任意自定义参数字典，合并到请求体后：
    - 自定义参数的 key-value 全部出现在请求体中
    - 不覆盖核心字段（model、messages、stream）

    **Validates: Requirements 4.5, 6.3**
    """

    @given(custom_params=custom_params_strategy)
    @settings(max_examples=100)
    def test_custom_params_all_present(self, custom_params):
        """属性：自定义参数的所有 key-value 都出现在请求体中"""
        body = asyncio.get_event_loop().run_until_complete(
            capture_request_body(custom_params=custom_params)
        )

        for key, value in custom_params.items():
            assert key in body, f"自定义参数 '{key}' 应出现在请求体中"
            assert body[key] == value, f"自定义参数 '{key}' 的值应为 {value!r}，实际为 {body[key]!r}"

    @given(custom_params=custom_params_strategy)
    @settings(max_examples=100)
    def test_custom_params_no_core_override(self, custom_params):
        """属性：自定义参数不覆盖核心字段（model、messages、stream）"""
        body = asyncio.get_event_loop().run_until_complete(
            capture_request_body(custom_params=custom_params)
        )

        # 核心字段应保持原始值
        assert body["model"] == "gpt-4", "model 字段不应被自定义参数覆盖"
        assert body["messages"] == [{"role": "user", "content": "hello"}], "messages 字段不应被自定义参数覆盖"
        assert body["stream"] is False, "stream 字段不应被自定义参数覆盖"

    @given(
        custom_params=st.dictionaries(
            keys=st.sampled_from(["model", "messages", "stream"]),
            values=custom_param_values,
            min_size=1,
            max_size=3,
        )
    )
    @settings(max_examples=50)
    def test_core_fields_overwritten_when_in_custom_params(self, custom_params):
        """边界验证：当 custom_params 包含核心字段名时，body.update 会覆盖

        注意：这是当前实现的行为（body.update(custom_params)），
        设计文档注释说"不覆盖已有核心字段由调用方保证"。
        此测试记录当前行为，确认核心字段确实会被覆盖。
        """
        body = asyncio.get_event_loop().run_until_complete(
            capture_request_body(custom_params=custom_params)
        )

        # 当 custom_params 包含核心字段时，body.update 会覆盖
        for key in custom_params:
            if key in CORE_FIELDS:
                assert body[key] == custom_params[key], (
                    f"body.update 应将 '{key}' 覆盖为 custom_params 的值"
                )
