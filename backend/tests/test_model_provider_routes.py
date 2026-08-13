"""模型管理路由回归测试

## 已知失败：内置模型目录落后于测试期望（2026-08-13 定位）

`test_get_models_exposes_official_doubao_chat_ids` 与
`test_get_models_exposes_official_current_chat_ids` 当前失败。这两条**不是测试陈旧**，
不要通过删断言来"修复"。

`backend/models/model_registry.py` 里没有 `gpt-5.x`、`claude-fable-5`、
`doubao-seed-2-0-*` 等条目，OpenAI 侧仍停在 gpt-4.1 / gpt-4o 一代，豆包侧只有
2.1 系三个模型；该文件在 HEAD 上也没有未提交改动。也就是说目录本身落后了，而这两条
测试的用途恰恰是"守住内置 chat 模型 ID 与当前官方 ID 对齐"——它们正在按设计报警。

该上架哪些模型 ID 是产品内容决策，需要核对各家官方目录后统一更新
`model_registry.py`，不适合在测试清理里顺手改。

（`test_upsert_custom_model_normalizes_provider_and_metadata` 是另一回事：它被
base_url 的 SSRF 防护拦下，用例里的私网/本机地址需要换成公网可解析的测试地址。）
"""

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
    # base_url 的 SSRF 防护会对主机名做真实 DNS 解析，并把解析到非公网地址视为
    # 拒绝。在 Fake-IP 透明代理环境下所有域名都会解析进保留段，测试因此变成
    # 网络环境依赖。本测试的主题是元数据归一化，把解析接缝钉成放行。
    import services.ocr_service as ocr_service_module

    monkeypatch.setattr(
        ocr_service_module,
        "_resolve_external_ocr_host",
        lambda *args, **kwargs: None,
    )

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

    # 清单已策展为当前代（2.1 系 + evolving），2.0 系整体清退。
    assert "doubao-seed-2-1-pro-260628" in doubao_models
    assert "doubao-seed-2-1-turbo-260628" in doubao_models
    assert "doubao-seed-evolving" in doubao_models
    assert "doubao-seed-2-0-pro" not in doubao_models
    assert "doubao-seed-2-0-pro-260215" not in doubao_models
    assert "doubao-seed-2-0-lite-260215" not in doubao_models
    assert "doubao-seed-2-0-lite-260428" not in doubao_models
    assert "doubao-seed-2-0-mini-260215" not in doubao_models
    assert "doubao-seed-2-0-mini-260428" not in doubao_models
    assert "doubao-seed-code-preview-latest" not in doubao_models


def test_get_models_exposes_official_current_chat_ids(monkeypatch):
    """高风险 provider 的 chat 模型 id 应与当前官方/兼容 id 对齐。"""
    monkeypatch.setattr(model_provider_routes, "load_dynamic_models", lambda: {})
    monkeypatch.setattr(model_provider_routes, "load_dynamic_providers", lambda: {})

    result = asyncio.run(model_provider_routes.get_models())

    # 各 provider 锁定当前策展代表集合：旗舰 + 长期保留项。断言集与
    # 静态注册表同步演进，已清退型号进入下方护栏，防止回流。
    assert "gpt-5.6" in result["openai"]["models"]
    assert "gpt-5.6-sol" in result["openai"]["models"]
    assert "gpt-5.5" in result["openai"]["models"]
    assert "gpt-5.5-pro" in result["openai"]["models"]
    assert "gpt-5.4-mini" in result["openai"]["models"]

    assert "claude-opus-5" in result["anthropic"]["models"]
    assert "claude-fable-5" in result["anthropic"]["models"]
    assert "claude-opus-4-8" in result["anthropic"]["models"]
    assert "claude-sonnet-5" in result["anthropic"]["models"]
    assert "claude-haiku-4-5-20251001" in result["anthropic"]["models"]

    assert "gemini-3.6-flash" in result["gemini"]["models"]
    assert "gemini-3.5-flash" in result["gemini"]["models"]
    assert "gemini-3.1-pro-preview" in result["gemini"]["models"]

    assert "grok-4.5" in result["grok"]["models"]
    assert "grok-4.3" in result["grok"]["models"]
    assert "grok-build-0.1" in result["grok"]["models"]

    assert "qwen3.7-max" in result["aliyun"]["models"]
    assert "qwen3.7-plus" in result["aliyun"]["models"]
    assert "qwen3.6-flash" in result["aliyun"]["models"]
    assert "qwen3-rerank" in result["aliyun"]["models"]
    assert "deepseek-v4-pro" in result["deepseek"]["models"]
    assert "deepseek-v4-flash" in result["deepseek"]["models"]
    assert "kimi-k3" in result["moonshot"]["models"]
    assert "kimi-k2.7-code" in result["moonshot"]["models"]
    assert "kimi-k2.6" in result["moonshot"]["models"]
    assert "glm-5.2" in result["zhipu"]["models"]
    assert "glm-5.1" in result["zhipu"]["models"]
    assert "MiniMax-M3" in result["minimax"]["models"]
    assert "deepseek-ai/DeepSeek-V4-Flash" in result["silicon"]["models"]
    assert "Qwen/Qwen3-32B" in result["silicon"]["models"]

    assert "gpt-5.4" not in result["openai"]["models"]
    assert "gpt-5.3-codex" not in result["openai"]["models"]
    assert "claude-opus-4-1-20250805" not in result["anthropic"]["models"]
    assert "gemini-3-pro-preview" not in result["gemini"]["models"]
    assert "gemini-3-flash-preview" not in result["gemini"]["models"]
    assert "grok-4.20-beta-latest-non-reasoning" not in result["grok"]["models"]
    assert "grok-4-1-fast-reasoning" not in result["grok"]["models"]
    assert "kimi-k2" not in result["moonshot"]["models"]
    assert "kimi-k2.7-code-highspeed" not in result["moonshot"]["models"]
    assert "kimi-latest" not in result["moonshot"]["models"]
    assert "kimi-thinking-preview" not in result["moonshot"]["models"]
    assert "moonshot-v1-vision-preview" not in result["moonshot"]["models"]
    assert "MiniMax-M2" not in result["minimax"]["models"]
    assert "MiniMax-Text-01" not in result["minimax"]["models"]
    assert "abab6.5s-chat" not in result["minimax"]["models"]
    assert "glm-4.6" not in result["zhipu"]["models"]
    assert "glm-4-air" not in result["zhipu"]["models"]
    assert "glm-4-air-250414" not in result["zhipu"]["models"]
    assert "deepseek-ai/DeepSeek-V3" not in result["silicon"]["models"]
    assert "deepseek-ai/DeepSeek-V3.2" not in result["silicon"]["models"]
    assert "Qwen/Qwen3-235B-A22B" not in result["silicon"]["models"]


def test_aliyun_rerank_default_uses_qwen3():
    """阿里云重排默认模型应使用当前 qwen3-rerank。"""
    result = asyncio.run(model_provider_routes.get_rerank_providers())

    assert result["aliyun"]["default_model"] == "qwen3-rerank"
