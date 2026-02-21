"""测试 MemoryRetriever 记忆混合检索器

验证需求：
- 4.2: 复用 hybrid_search_merge 进行混合检索（向量 + BM25）
- 4.3: 用户发起查询时先检索相关记忆
- 4.5: 记忆索引为空时跳过检索，返回空结果
- 4.6: 返回最多 top_k 条最相关记忆
"""
import sys
import os

import numpy as np
import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.memory_store import MemoryStore, MemoryEntry
from services.memory_index import MemoryIndex
from services.memory_retriever import MemoryRetriever


# ==================== 辅助工具 ====================


def _fake_embed_fn(texts, api_key=None, **kwargs):
    """模拟 embedding 函数，返回固定维度的确定性向量"""
    result = []
    for text in texts:
        np.random.seed(hash(text) % (2**31))
        result.append(np.random.randn(8).astype(np.float32))
    return np.array(result, dtype=np.float32)


def _make_entry(content: str, source_type: str = "manual", doc_id: str = None) -> MemoryEntry:
    """创建测试用记忆条目"""
    return MemoryEntry(
        content=content,
        source_type=source_type,
        doc_id=doc_id,
        importance=1.0 if source_type in ("manual", "liked") else 0.5,
    )


@pytest.fixture
def data_dir(tmp_path):
    """创建临时数据目录"""
    d = tmp_path / "memory"
    d.mkdir()
    return str(d)


@pytest.fixture
def memory_store(data_dir):
    """创建 MemoryStore 实例"""
    return MemoryStore(data_dir)


@pytest.fixture
def memory_index(data_dir, monkeypatch):
    """创建已 mock embedding 的 MemoryIndex 实例"""
    index_dir = os.path.join(data_dir, "memory_index")
    os.makedirs(index_dir, exist_ok=True)
    mi = MemoryIndex(index_dir, embedding_model_id="test-model")
    monkeypatch.setattr(mi, "_embed_texts", _fake_embed_fn)
    return mi


@pytest.fixture
def retriever(memory_store, memory_index):
    """创建 MemoryRetriever 实例"""
    return MemoryRetriever(memory_store, memory_index)


# ==================== 空索引测试 ====================


class TestEmptyIndex:
    """测试空索引时的行为（需求 4.5）"""

    def test_retrieve_empty_store(self, retriever):
        """无记忆条目时返回空列表"""
        results = retriever.retrieve("任何查询")
        assert results == []

    def test_retrieve_empty_query(self, retriever):
        """空查询返回空列表"""
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_whitespace_query(self, retriever):
        """纯空白查询返回空列表"""
        results = retriever.retrieve("   ")
        assert results == []

    def test_build_memory_context_empty(self, retriever):
        """空记忆列表返回空字符串"""
        assert retriever.build_memory_context([]) == ""


# ==================== 混合检索测试 ====================


class TestHybridRetrieval:
    """测试混合检索功能（需求 4.2, 4.3）"""

    def _add_entries(self, memory_store, memory_index, entries):
        """向 store 和 index 中添加记忆条目"""
        for entry in entries:
            memory_store.add_entry(entry)
            memory_index.add_entry(entry.id, entry.content)

    def test_retrieve_returns_results(self, retriever, memory_store, memory_index):
        """有记忆条目时能返回检索结果"""
        entries = [
            _make_entry("机器学习是人工智能的一个分支"),
            _make_entry("深度学习使用神经网络进行特征提取"),
            _make_entry("Transformer 架构改变了 NLP 领域"),
        ]
        self._add_entries(memory_store, memory_index, entries)

        results = retriever.retrieve("机器学习")
        assert len(results) > 0
        # 每个结果应包含必要字段
        for r in results:
            assert "entry_id" in r
            assert "text" in r
            assert "source_type" in r

    def test_retrieve_top_k_limit(self, retriever, memory_store, memory_index):
        """检索结果数量不超过 top_k（需求 4.6）"""
        entries = [
            _make_entry(f"记忆条目 {i}：关于机器学习的内容")
            for i in range(10)
        ]
        self._add_entries(memory_store, memory_index, entries)

        results = retriever.retrieve("机器学习", top_k=3)
        assert len(results) <= 3

    def test_retrieve_top_k_1(self, retriever, memory_store, memory_index):
        """top_k=1 时只返回一条结果"""
        entries = [
            _make_entry("机器学习基础"),
            _make_entry("深度学习进阶"),
        ]
        self._add_entries(memory_store, memory_index, entries)

        results = retriever.retrieve("学习", top_k=1)
        assert len(results) <= 1

    def test_retrieve_result_fields(self, retriever, memory_store, memory_index):
        """检索结果包含正确的字段"""
        entry = _make_entry("Transformer 注意力机制", source_type="liked", doc_id="doc1")
        memory_store.add_entry(entry)
        memory_index.add_entry(entry.id, entry.content)

        results = retriever.retrieve("注意力")
        assert len(results) > 0
        r = results[0]
        assert r["entry_id"] == entry.id
        assert r["text"] == entry.content
        assert r["source_type"] == "liked"
        assert r["doc_id"] == "doc1"

    def test_retrieve_fewer_than_top_k(self, retriever, memory_store, memory_index):
        """记忆条目少于 top_k 时返回所有条目"""
        entries = [_make_entry("唯一的记忆条目")]
        self._add_entries(memory_store, memory_index, entries)

        results = retriever.retrieve("记忆", top_k=5)
        assert len(results) <= 1


# ==================== BM25 检索测试 ====================


class TestBM25Search:
    """测试 BM25 检索路径"""

    def test_bm25_search_empty_entries(self, retriever):
        """空条目列表返回空结果"""
        results = retriever._bm25_search("查询", [], top_k=3)
        assert results == []

    def test_bm25_search_returns_results(self, retriever, memory_store, memory_index):
        """BM25 能检索到包含关键词的记忆"""
        entries = [
            _make_entry("Python 编程语言"),
            _make_entry("Java 编程语言"),
            _make_entry("机器学习算法"),
        ]
        for e in entries:
            memory_store.add_entry(e)
            memory_index.add_entry(e.id, e.content)

        results = retriever._bm25_search("编程", entries, top_k=3)
        assert len(results) > 0
        # 结果应包含 chunk 和 entry_id 字段
        for r in results:
            assert "chunk" in r
            assert "entry_id" in r


# ==================== 上下文格式化测试 ====================


class TestBuildMemoryContext:
    """测试记忆上下文格式化"""

    def test_format_single_memory(self, retriever):
        """单条记忆的格式化"""
        memories = [{"source_type": "manual", "text": "用户偏好中文回答"}]
        context = retriever.build_memory_context(memories)
        assert "用户历史记忆：" in context
        assert "- [manual] 用户偏好中文回答" in context

    def test_format_multiple_memories(self, retriever):
        """多条记忆的格式化"""
        memories = [
            {"source_type": "auto_qa", "text": "关于 Transformer 的讨论"},
            {"source_type": "liked", "text": "注意力机制的解释"},
            {"source_type": "manual", "text": "用户关注 NLP 领域"},
        ]
        context = retriever.build_memory_context(memories)
        lines = context.strip().split("\n")
        # 第一行是标题，后面是记忆条目
        assert lines[0] == "用户历史记忆："
        assert len(lines) == 4  # 标题 + 3 条记忆

    def test_format_preserves_source_type(self, retriever):
        """格式化保留来源类型标签"""
        memories = [
            {"source_type": "liked", "text": "重要内容"},
        ]
        context = retriever.build_memory_context(memories)
        assert "[liked]" in context

    def test_format_empty_returns_empty(self, retriever):
        """空列表返回空字符串"""
        assert retriever.build_memory_context([]) == ""
