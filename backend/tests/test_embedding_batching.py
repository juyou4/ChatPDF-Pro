"""embedding 分批与上下文约束回归测试"""

import os
import sys
from types import SimpleNamespace

import numpy as np

# 将 backend 目录加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import get_embedding_function, get_chunk_params
import services.embedding_service as embedding_service_module


class _FakeOpenAIClient:
    """最小化 OpenAI client mock，记录 embeddings.create 调用"""

    def __init__(self):
        self.calls = []
        self._counter = 0
        self.embeddings = self

    def create(self, model, input):
        batch = list(input)
        self.calls.append({"model": model, "input": batch})
        data = []
        for _ in batch:
            # 返回稳定维度向量，便于断言 shape
            data.append(SimpleNamespace(embedding=[float(self._counter), 1.0]))
            self._counter += 1
        return SimpleNamespace(data=data)


class _FailThenSuccessClient:
    """第一次按指定模型失败，回退模型后成功。"""

    def __init__(self, fail_model: str):
        self.fail_model = fail_model
        self.calls = []
        self.embeddings = self

    def create(self, model, input):
        batch = list(input)
        self.calls.append({"model": model, "input": batch})
        if model == self.fail_model:
            raise RuntimeError("Error code: 400 - {'code': 20012, 'message': 'Model does not exist.'}")
        data = [SimpleNamespace(embedding=[1.0, 2.0]) for _ in batch]
        return SimpleNamespace(data=data)


class _StructuredModelNotFoundClient:
    """抛出带 response.json() 的结构化模型不存在错误。"""

    def __init__(self):
        self.calls = []
        self.embeddings = self

    def create(self, model, input):
        self.calls.append({"model": model, "input": list(input)})

        class _Resp:
            @staticmethod
            def json():
                return {"code": 20012, "message": "Model does not exist."}

        class _Exc(Exception):
            def __init__(self):
                super().__init__("Error code: 400")
                self.response = _Resp()

        raise _Exc()


def test_remote_embedding_batches_large_requests(monkeypatch):
    """大批量文本应自动拆分为多次 embeddings 请求，避免单次 token 超限。"""
    fake_client = _FakeOpenAIClient()

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "BAAI/bge-m3",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)

    embed_fn = get_embedding_function("silicon:BAAI/bge-m3", api_key="sk-user")
    texts = ["测" * 700 for _ in range(20)]
    vectors = embed_fn(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(texts)
    assert len(fake_client.calls) >= 2
    assert sum(len(call["input"]) for call in fake_client.calls) == len(texts)


def test_remote_embedding_truncates_single_oversized_input(monkeypatch):
    """单条超长文本应自动缩短后重试，而不是直接失败。"""
    fake_client = _FakeOpenAIClient()

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "BAAI/bge-m3",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)

    embed_fn = get_embedding_function("silicon:BAAI/bge-m3", api_key="sk-user")
    long_text = "测" * 12000
    vectors = embed_fn([long_text])

    assert vectors.shape[0] == 1
    assert len(fake_client.calls) == 1
    sent_text = fake_client.calls[0]["input"][0]
    assert len(sent_text) < len(long_text)


def test_get_chunk_params_respects_small_max_context():
    """小上下文模型（max_tokens=512）不应被固定下限放大到 1000。"""
    chunk_size, chunk_overlap = get_chunk_params(
        "BAAI/bge-large-zh-v1.5",
        base_chunk_size=1200,
        base_overlap=200
    )

    assert chunk_size <= 512
    assert chunk_overlap < chunk_size


def test_remote_embedding_uses_model_name_for_request(monkeypatch):
    """远程 embedding 请求应优先使用 config.model_name，而非 registry key。"""
    fake_client = _FakeOpenAIClient()

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "silicon:legacy-key",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "model_name": "BAAI/bge-m3",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)

    embed_fn = get_embedding_function("silicon:BAAI/bge-m3", api_key="sk-user")
    vectors = embed_fn(["hello"])

    assert vectors.shape[0] == 1
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["model"] == "BAAI/bge-m3"


def test_remote_embedding_fallback_when_model_not_found(monkeypatch):
    """模型不存在时应自动回退到可用 embedding 模型并继续执行。"""
    fake_client = _FailThenSuccessClient(fail_model="BAAI/bge-m3")

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "BAAI/bge-m3",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(
        embedding_service_module,
        "_fetch_available_model_ids",
        lambda _api_base, _api_key: ["Qwen/Qwen-Embedding-8B", "BAAI/bge-reranker-v2-m3"],
    )

    embed_fn = get_embedding_function("silicon:BAAI/bge-m3", api_key="sk-user")
    vectors = embed_fn(["test"])

    assert vectors.shape[0] == 1
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0]["model"] == "BAAI/bge-m3"
    assert fake_client.calls[1]["model"] == "Qwen/Qwen-Embedding-8B"


def test_remote_embedding_fallback_skips_failed_deprecated_model(monkeypatch):
    """首选模型失败后，回退不应再次选回同一旧模型。"""
    fake_client = _FailThenSuccessClient(fail_model="Qwen/Qwen-Embedding-8B")

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "Qwen/Qwen-Embedding-8B",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "model_name": "Qwen/Qwen-Embedding-8B",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(
        embedding_service_module,
        "_fetch_available_model_ids",
        lambda _api_base, _api_key: ["Qwen/Qwen-Embedding-8B", "Qwen/Qwen3-Embedding-8B"],
    )

    embed_fn = get_embedding_function("silicon:Qwen/Qwen-Embedding-8B", api_key="sk-user")
    vectors = embed_fn(["test"])

    assert vectors.shape[0] == 1
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0]["model"] == "Qwen/Qwen-Embedding-8B"
    assert fake_client.calls[1]["model"] == "Qwen/Qwen3-Embedding-8B"


def test_remote_embedding_fallback_alias_for_legacy_openai_model(monkeypatch):
    """旧 OpenAI embedding ID 失败时，应回退到 text-embedding-3-small。"""
    fake_client = _FailThenSuccessClient(fail_model="text-embedding-ada-002")

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "legacy-openai",
            {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "model_name": "text-embedding-ada-002",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(
        embedding_service_module,
        "_fetch_available_model_ids",
        lambda _api_base, _api_key: ["text-embedding-ada-002", "text-embedding-3-small"],
    )

    embed_fn = get_embedding_function("openai:text-embedding-ada-002", api_key="sk-user")
    vectors = embed_fn(["test"])

    assert vectors.shape[0] == 1
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0]["model"] == "text-embedding-ada-002"
    assert fake_client.calls[1]["model"] == "text-embedding-3-small"


def test_model_not_found_detection_with_structured_error(monkeypatch):
    """结构化错误体（code=20012）也应触发模型不存在提示。"""
    fake_client = _StructuredModelNotFoundClient()

    monkeypatch.setattr(
        embedding_service_module,
        "resolve_model_id",
        lambda _model_id: (
            "BAAI/bge-m3",
            {
                "provider": "openai",
                "base_url": "https://api.siliconflow.cn/v1",
                "max_tokens": 8192,
            },
        ),
    )
    monkeypatch.setattr(embedding_service_module, "select_api_key", lambda _k: "sk-test")
    monkeypatch.setattr(embedding_service_module, "_get_openai_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(embedding_service_module, "_fetch_available_model_ids", lambda *_args, **_kwargs: [])

    embed_fn = get_embedding_function("silicon:BAAI/bge-m3", api_key="sk-user")
    try:
        embed_fn(["test"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Embedding模型" in str(e)
