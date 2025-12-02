import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from models import model_detector


def test_is_embedding_model_basic():
    assert model_detector.is_embedding_model("text-embedding-3-small")
    assert model_detector.is_embedding_model("BAAI/bge-m3")
    assert model_detector.is_embedding_model("gte-large")


def test_is_rerank_model_basic():
    assert model_detector.is_rerank_model("bge-reranker-base")
    assert not model_detector.is_rerank_model("text-embedding-3-small")


def test_get_model_provider_inference():
    assert model_detector.get_model_provider("kimi") == "moonshot"
    assert model_detector.get_model_provider("glm-4") == "zhipu"
    assert model_detector.get_model_provider("claude-3") == "anthropic"
    # default fallback
    assert model_detector.get_model_provider("some-unknown-model") == "openai"
