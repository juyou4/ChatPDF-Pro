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
