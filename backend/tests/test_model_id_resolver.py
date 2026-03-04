"""model_id_resolver 兼容性测试"""

import os
import sys

# 将 backend 目录加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.model_id_resolver import normalize_deprecated_model_id, resolve_model_id


def test_normalize_deprecated_model_id_plain_key():
    assert (
        normalize_deprecated_model_id("Qwen/Qwen-Embedding-8B")
        == "Qwen/Qwen3-Embedding-8B"
    )
    assert (
        normalize_deprecated_model_id("text-embedding-ada-002")
        == "text-embedding-3-small"
    )
    assert (
        normalize_deprecated_model_id("embo-01")
        == "minimax-embedding-v2"
    )


def test_normalize_deprecated_model_id_composite_key():
    assert (
        normalize_deprecated_model_id("silicon:Qwen/Qwen-Embedding-8B")
        == "silicon:Qwen/Qwen3-Embedding-8B"
    )
    assert (
        normalize_deprecated_model_id("openai:text-embedding-ada-002")
        == "openai:text-embedding-3-small"
    )
    assert (
        normalize_deprecated_model_id("minimax:embo-01")
        == "minimax:minimax-embedding-v2"
    )


def test_resolve_model_id_maps_deprecated_qwen_key():
    registry_key, config = resolve_model_id("silicon:Qwen/Qwen-Embedding-8B")
    assert registry_key == "Qwen/Qwen3-Embedding-8B"
    assert isinstance(config, dict)
    assert config.get("base_url") == "https://api.siliconflow.cn/v1"


def test_resolve_model_id_maps_deprecated_openai_key():
    registry_key, config = resolve_model_id("openai:text-embedding-ada-002")
    assert registry_key == "text-embedding-3-small"
    assert isinstance(config, dict)
    assert config.get("base_url") == "https://api.openai.com/v1"


def test_resolve_model_id_maps_deprecated_minimax_key():
    registry_key, config = resolve_model_id("minimax:embo-01")
    assert registry_key == "minimax-embedding-v2"
    assert isinstance(config, dict)
    assert "minimax" in (config.get("base_url") or "")
