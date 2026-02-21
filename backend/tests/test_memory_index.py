"""测试 MemoryIndex 记忆向量索引管理

验证需求 4.1：
- 使用 embedding_service 为记忆条目生成向量并添加到 FAISS 索引
- 向量检索 top_k 条记忆
- 索引持久化（save / load）
- 索引重建（rebuild）
- 条目移除（remove_entry）
"""
import sys
import os
import pickle

import numpy as np
import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.memory_index import MemoryIndex
from services.memory_store import MemoryEntry


# ==================== 辅助工具 ====================


def _fake_embed_fn(texts, api_key=None, **kwargs):
    """模拟 embedding 函数，返回固定维度的确定性向量"""
    # 基于文本哈希生成确定性向量，维度为 8
    result = []
    for text in texts:
        np.random.seed(hash(text) % (2**31))
        result.append(np.random.randn(8).astype(np.float32))
    return np.array(result, dtype=np.float32)


@pytest.fixture
def index_dir(tmp_path):
    """创建临时索引目录"""
    d = tmp_path / "memory_index"
    d.mkdir()
    return str(d)


@pytest.fixture
def memory_index(index_dir, monkeypatch):
    """创建已 mock embedding 函数的 MemoryIndex 实例"""
    mi = MemoryIndex(index_dir, embedding_model_id="test-model")
    # mock _embed_texts 避免加载真实模型
    monkeypatch.setattr(mi, "_embed_texts", _fake_embed_fn)
    return mi


# ==================== 基础功能测试 ====================


class TestMemoryIndexBasic:
    """测试 MemoryIndex 基础功能"""

    def test_init_empty(self, memory_index):
        """初始化后索引应为空"""
        assert memory_index.index is None
        assert memory_index.entry_ids == []
        assert memory_index.texts == []

    def test_add_entry(self, memory_index):
        """添加条目后索引应包含该条目"""
        memory_index.add_entry("id-1", "测试记忆内容")

        assert memory_index.index is not None
        assert memory_index.index.ntotal == 1
        assert "id-1" in memory_index.entry_ids
        assert "测试记忆内容" in memory_index.texts

    def test_add_multiple_entries(self, memory_index):
        """添加多条记忆后索引数量应正确"""
        memory_index.add_entry("id-1", "第一条记忆")
        memory_index.add_entry("id-2", "第二条记忆")
        memory_index.add_entry("id-3", "第三条记忆")

        assert memory_index.index.ntotal == 3
        assert len(memory_index.entry_ids) == 3
        assert len(memory_index.texts) == 3

    def test_search_empty_index(self, memory_index):
        """空索引搜索应返回空列表"""
        results = memory_index.search("查询文本")
        assert results == []

    def test_search_returns_results(self, memory_index):
        """搜索应返回结果列表"""
        memory_index.add_entry("id-1", "机器学习基础知识")
        memory_index.add_entry("id-2", "深度学习神经网络")

        results = memory_index.search("机器学习")
        assert len(results) > 0
        assert all("entry_id" in r for r in results)
        assert all("similarity" in r for r in results)
        assert all("text" in r for r in results)

    def test_search_top_k_limit(self, memory_index):
        """搜索结果数量不应超过 top_k"""
        for i in range(10):
            memory_index.add_entry(f"id-{i}", f"记忆内容 {i}")

        results = memory_index.search("记忆", top_k=3)
        assert len(results) <= 3

    def test_search_top_k_exceeds_total(self, memory_index):
        """top_k 超过总条目数时应返回所有条目"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.add_entry("id-2", "记忆二")

        results = memory_index.search("记忆", top_k=10)
        assert len(results) == 2

    def test_search_similarity_range(self, memory_index):
        """相似度应在 (0, 1] 范围内"""
        memory_index.add_entry("id-1", "测试内容")
        results = memory_index.search("测试")

        for r in results:
            assert 0 < r["similarity"] <= 1.0


# ==================== 移除条目测试 ====================


class TestMemoryIndexRemove:
    """测试条目移除功能"""

    def test_remove_entry(self, memory_index):
        """移除条目后索引应不包含该条目"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.add_entry("id-2", "记忆二")

        memory_index.remove_entry("id-1")

        assert memory_index.index.ntotal == 1
        assert "id-1" not in memory_index.entry_ids
        assert "id-2" in memory_index.entry_ids

    def test_remove_nonexistent_entry(self, memory_index):
        """移除不存在的条目不应报错"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.remove_entry("nonexistent-id")

        assert memory_index.index.ntotal == 1
        assert "id-1" in memory_index.entry_ids

    def test_remove_all_entries(self, memory_index):
        """移除所有条目后索引应为空"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.add_entry("id-2", "记忆二")

        memory_index.remove_entry("id-1")
        memory_index.remove_entry("id-2")

        assert memory_index.index is None
        assert memory_index.entry_ids == []
        assert memory_index.texts == []

    def test_remove_preserves_order(self, memory_index):
        """移除中间条目后剩余条目顺序应正确"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.add_entry("id-2", "记忆二")
        memory_index.add_entry("id-3", "记忆三")

        memory_index.remove_entry("id-2")

        assert memory_index.entry_ids == ["id-1", "id-3"]
        assert memory_index.texts == ["记忆一", "记忆三"]
        assert memory_index.index.ntotal == 2


# ==================== 持久化测试 ====================


class TestMemoryIndexPersistence:
    """测试索引持久化（save / load）"""

    def test_save_and_load(self, index_dir, monkeypatch):
        """保存后重新加载应恢复索引状态"""
        mi1 = MemoryIndex(index_dir, embedding_model_id="test-model")
        monkeypatch.setattr(mi1, "_embed_texts", _fake_embed_fn)

        mi1.add_entry("id-1", "记忆一")
        mi1.add_entry("id-2", "记忆二")

        # 创建新实例并加载
        mi2 = MemoryIndex(index_dir, embedding_model_id="test-model")
        success = mi2.load()

        assert success is True
        assert mi2.entry_ids == ["id-1", "id-2"]
        assert mi2.texts == ["记忆一", "记忆二"]
        assert mi2.index is not None
        assert mi2.index.ntotal == 2

    def test_load_nonexistent(self, index_dir):
        """加载不存在的索引应返回 False"""
        mi = MemoryIndex(index_dir, embedding_model_id="test-model")
        success = mi.load()

        assert success is False
        assert mi.index is None
        assert mi.entry_ids == []

    def test_load_model_mismatch(self, index_dir, monkeypatch):
        """embedding 模型不一致时加载应返回 False"""
        mi1 = MemoryIndex(index_dir, embedding_model_id="model-a")
        monkeypatch.setattr(mi1, "_embed_texts", _fake_embed_fn)
        mi1.add_entry("id-1", "记忆一")

        mi2 = MemoryIndex(index_dir, embedding_model_id="model-b")
        success = mi2.load()

        assert success is False
        assert mi2.index is None

    def test_save_empty_index(self, index_dir):
        """保存空索引应正常工作"""
        mi = MemoryIndex(index_dir, embedding_model_id="test-model")
        mi.save()

        meta_path = os.path.join(index_dir, "memory.pkl")
        assert os.path.exists(meta_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        assert meta["entry_ids"] == []
        assert meta["texts"] == []

    def test_load_corrupted_meta(self, index_dir):
        """元数据损坏时加载应返回 False"""
        meta_path = os.path.join(index_dir, "memory.pkl")
        with open(meta_path, "wb") as f:
            f.write(b"corrupted data")

        mi = MemoryIndex(index_dir, embedding_model_id="test-model")
        success = mi.load()

        assert success is False
        assert mi.index is None

    def test_load_missing_index_file(self, index_dir):
        """元数据存在但索引文件缺失时应返回 False"""
        meta_path = os.path.join(index_dir, "memory.pkl")
        meta = {
            "entry_ids": ["id-1"],
            "texts": ["记忆一"],
            "embedding_model": "test-model",
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        mi = MemoryIndex(index_dir, embedding_model_id="test-model")
        success = mi.load()

        assert success is False

    def test_load_inconsistent_count(self, index_dir, monkeypatch):
        """FAISS 索引条目数与元数据不一致时应返回 False"""
        import faiss

        # 创建一个有 1 条向量的索引
        index = faiss.IndexFlatL2(8)
        index.add(np.random.randn(1, 8).astype(np.float32))
        faiss.write_index(index, os.path.join(index_dir, "memory.index"))

        # 但元数据有 2 条记录
        meta = {
            "entry_ids": ["id-1", "id-2"],
            "texts": ["记忆一", "记忆二"],
            "embedding_model": "test-model",
        }
        with open(os.path.join(index_dir, "memory.pkl"), "wb") as f:
            pickle.dump(meta, f)

        mi = MemoryIndex(index_dir, embedding_model_id="test-model")
        success = mi.load()

        assert success is False


# ==================== 重建索引测试 ====================


class TestMemoryIndexRebuild:
    """测试索引重建功能"""

    def test_rebuild_with_entries(self, memory_index):
        """重建索引应包含所有传入的条目"""
        entries = [
            MemoryEntry(id="id-1", content="记忆一"),
            MemoryEntry(id="id-2", content="记忆二"),
            MemoryEntry(id="id-3", content="记忆三"),
        ]

        memory_index.rebuild(entries)

        assert memory_index.index is not None
        assert memory_index.index.ntotal == 3
        assert memory_index.entry_ids == ["id-1", "id-2", "id-3"]
        assert memory_index.texts == ["记忆一", "记忆二", "记忆三"]

    def test_rebuild_empty(self, memory_index):
        """用空列表重建应清空索引"""
        memory_index.add_entry("id-1", "记忆一")
        memory_index.rebuild([])

        assert memory_index.index is None
        assert memory_index.entry_ids == []
        assert memory_index.texts == []

    def test_rebuild_replaces_existing(self, memory_index):
        """重建应替换现有索引"""
        memory_index.add_entry("old-id", "旧记忆")

        entries = [
            MemoryEntry(id="new-1", content="新记忆一"),
            MemoryEntry(id="new-2", content="新记忆二"),
        ]
        memory_index.rebuild(entries)

        assert "old-id" not in memory_index.entry_ids
        assert memory_index.entry_ids == ["new-1", "new-2"]
        assert memory_index.index.ntotal == 2

    def test_rebuild_and_search(self, memory_index):
        """重建后应能正常搜索"""
        entries = [
            MemoryEntry(id="id-1", content="Python 编程语言"),
            MemoryEntry(id="id-2", content="JavaScript 前端开发"),
        ]
        memory_index.rebuild(entries)

        results = memory_index.search("编程")
        assert len(results) > 0
