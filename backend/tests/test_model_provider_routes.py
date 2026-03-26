"""模型管理路由回归测试"""

import asyncio
import os
import sys

# 将 backend 目录加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.model_provider_routes as model_provider_routes
from routes.model_provider_routes import ModelUpsertRequest


def test_get_models_includes_openai_compatible_dynamic_model(monkeypatch):
    """OpenAI 兼容 provider 的动态模型应正确归属到原 provider。"""
    monkeypatch.setattr(
        model_provider_routes,
        "load_dynamic_models",
        lambda: {
            "acme-embed-1": {
                "name": "Acme Embed 1",
                "provider": "silicon",
                "type": "embedding",
                "base_url": "https://api.siliconflow.cn/v1",
            }
        },
    )
    monkeypatch.setattr(model_provider_routes, "load_dynamic_providers", lambda: {})

    result = asyncio.run(model_provider_routes.get_models())

    assert "acme-embed-1" in result["silicon"]["models"]


def test_upsert_custom_model_normalizes_provider_and_metadata(monkeypatch):
    """保存自定义模型时应同时写入 provider_type 和后端使用的 snake_case 字段。"""
    monkeypatch.setattr(model_provider_routes, "load_dynamic_models", lambda: {})
    monkeypatch.setattr(model_provider_routes, "load_dynamic_providers", lambda: {})

    captured = {}

    def _capture(models):
        captured["models"] = models

    monkeypatch.setattr(model_provider_routes, "save_dynamic_models", _capture)

    req = ModelUpsertRequest(
        modelId="embedding-3-custom",
        name="Zhipu Embedding 3",
        providerId="zhipu",
        type="embedding",
        metadata={
            "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
            "maxTokens": 8192,
            "modelName": "embedding-3",
        },
    )

    asyncio.run(model_provider_routes.upsert_custom_model(req))

    saved = captured["models"]["embedding-3-custom"]
    assert saved["provider"] == "openai"
    assert saved["provider_id"] == "zhipu"
    assert saved["provider_type"] == "openai"
    assert saved["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
    assert saved["max_tokens"] == 8192
    assert saved["model_name"] == "embedding-3"


def test_get_models_exposes_official_doubao_chat_ids(monkeypatch):
    """豆包聊天模型列表应暴露官方可用的版本号模型 ID。"""
    monkeypatch.setattr(model_provider_routes, "load_dynamic_models", lambda: {})
    monkeypatch.setattr(model_provider_routes, "load_dynamic_providers", lambda: {})

    result = asyncio.run(model_provider_routes.get_models())
    doubao_models = result["doubao"]["models"]

    assert "doubao-seed-2-0-pro-260215" in doubao_models
    assert "doubao-seed-2-0-lite-260215" in doubao_models
    assert "doubao-seed-2-0-mini-260215" in doubao_models
    assert "doubao-seed-code-preview-latest" in doubao_models
    assert "doubao-seed-2-0-pro" not in doubao_models


def test_get_models_exposes_official_current_chat_ids(monkeypatch):
    """高风险 provider 的 chat 模型 id 应与当前官方/兼容 id 对齐。"""
    monkeypatch.setattr(model_provider_routes, "load_dynamic_models", lambda: {})
    monkeypatch.setattr(model_provider_routes, "load_dynamic_providers", lambda: {})

    result = asyncio.run(model_provider_routes.get_models())

    assert "gpt-5.2" in result["openai"]["models"]
    assert "gpt-5.1" in result["openai"]["models"]
    assert "gpt-5" in result["openai"]["models"]
    assert "gpt-5-mini" in result["openai"]["models"]
    assert "gpt-5-nano" in result["openai"]["models"]

    assert "claude-opus-4-1-20250805" in result["anthropic"]["models"]
    assert "claude-opus-4-20250514" in result["anthropic"]["models"]
    assert "claude-sonnet-4-20250514" in result["anthropic"]["models"]
    assert "claude-3-7-sonnet-20250219" in result["anthropic"]["models"]
    assert "claude-3-5-haiku-20241022" in result["anthropic"]["models"]

    assert "gemini-3-pro-preview" in result["gemini"]["models"]
    assert "gemini-3-flash-preview" in result["gemini"]["models"]

    assert "grok-4.20-beta-latest-non-reasoning" in result["grok"]["models"]
    assert "grok-4-1-fast-reasoning" in result["grok"]["models"]

    assert "qwen3.5-flash" in result["aliyun"]["models"]
    assert "kimi-latest" in result["moonshot"]["models"]
    assert "kimi-thinking-preview" in result["moonshot"]["models"]
    assert "kimi-k2-0905-preview" in result["moonshot"]["models"]
    assert "kimi-k2-turbo-preview" in result["moonshot"]["models"]
    assert "glm-4.6" in result["zhipu"]["models"]
    assert "glm-4-air-250414" in result["zhipu"]["models"]
    assert "MiniMax-M2.5" in result["minimax"]["models"]
    assert "MiniMax-M2.1" in result["minimax"]["models"]
    assert "MiniMax-M2" in result["minimax"]["models"]
    assert "deepseek-ai/DeepSeek-V3.2" in result["silicon"]["models"]
    assert "Qwen/Qwen3-32B" in result["silicon"]["models"]

    assert "kimi-k2.5" not in result["moonshot"]["models"]
    assert "kimi-k2" not in result["moonshot"]["models"]
    assert "MiniMax-Text-01" not in result["minimax"]["models"]
    assert "abab6.5s-chat" not in result["minimax"]["models"]
    assert "glm-4-air" not in result["zhipu"]["models"]
    assert "deepseek-ai/DeepSeek-V3" not in result["silicon"]["models"]
    assert "Qwen/Qwen3-235B-A22B" not in result["silicon"]["models"]
