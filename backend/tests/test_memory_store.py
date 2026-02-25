"""测试 MemoryStore 记忆持久化存储层

验证需求 1.1, 1.2, 1.3, 1.4, 1.5, 1.6：
- 用户画像存储为 user_profile.json
- 文档会话记忆存储为 sessions/{doc_id}_session.json
- 目录结构自动创建
- 文件不存在时返回空默认结构
- MemoryEntry 包含所有必需字段
- JSON 序列化/反序列化往返一致性
"""
import sys
import os
import json

import pytest

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.memory_store import MemoryEntry, MemoryStore


# ==================== MemoryEntry 测试 ====================


class TestMemoryEntry:
    """测试 MemoryEntry 数据类"""

    def test_default_values(self):
        """默认值应正确设置"""
        entry = MemoryEntry()
        assert entry.id  # UUID 不为空
        assert entry.content == ""
        assert entry.source_type == "manual"
        assert entry.created_at  # 时间戳不为空
        assert entry.doc_id is None
        assert entry.importance == 0.5
        assert entry.memory_tier == "short_term"
        assert entry.tags == []

    def test_custom_values(self):
        """自定义值应正确设置"""
        entry = MemoryEntry(
            id="test-id",
            content="测试内容",
            source_type="liked",
            created_at="2024-01-15T10:30:00Z",
            doc_id="doc-123",
            importance=1.0,
            memory_tier="long_term",
            tags=["concept", "fact"],
        )
        assert entry.id == "test-id"
        assert entry.content == "测试内容"
        assert entry.source_type == "liked"
        assert entry.doc_id == "doc-123"
        assert entry.importance == 1.0
        assert entry.memory_tier == "long_term"
        assert entry.tags == ["concept", "fact"]

    def test_to_dict(self):
        """to_dict 应返回包含所有字段的字典（含 memory_tier 和 tags）"""
        entry = MemoryEntry(
            id="test-id",
            content="测试内容",
            source_type="manual",
            created_at="2024-01-15T10:30:00Z",
            doc_id=None,
            importance=1.0,
            memory_tier="long_term",
            tags=["concept"],
        )
        d = entry.to_dict()
        assert d["id"] == "test-id"
        assert d["content"] == "测试内容"
        assert d["source_type"] == "manual"
        assert d["created_at"] == "2024-01-15T10:30:00Z"
        assert d["doc_id"] is None
        assert d["importance"] == 1.0
        assert d["memory_tier"] == "long_term"
        assert d["tags"] == ["concept"]

    def test_from_dict(self):
        """from_dict 应正确还原 MemoryEntry（含 memory_tier 和 tags）"""
        data = {
            "id": "test-id",
            "content": "测试内容",
            "source_type": "liked",
            "created_at": "2024-01-15T10:30:00Z",
            "doc_id": "doc-123",
            "importance": 0.8,
            "memory_tier": "long_term",
            "tags": ["fact", "method"],
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.id == "test-id"
        assert entry.content == "测试内容"
        assert entry.source_type == "liked"
        assert entry.doc_id == "doc-123"
        assert entry.importance == 0.8
        assert entry.memory_tier == "long_term"
        assert entry.tags == ["fact", "method"]

    def test_roundtrip_serialization(self):
        """序列化后反序列化应得到等价对象（需求 1.6, 9.4）"""
        entry = MemoryEntry(
            id="roundtrip-id",
            content="往返测试内容",
            source_type="auto_qa",
            created_at="2024-06-01T12:00:00Z",
            doc_id="doc-abc",
            importance=0.7,
            memory_tier="long_term",
            tags=["concept", "conclusion"],
        )
        d = entry.to_dict()
        # 模拟 JSON 序列化/反序列化
        json_str = json.dumps(d, ensure_ascii=False)
        restored_data = json.loads(json_str)
        restored = MemoryEntry.from_dict(restored_data)

        assert restored.id == entry.id
        assert restored.content == entry.content
        assert restored.source_type == entry.source_type
        assert restored.created_at == entry.created_at
        assert restored.doc_id == entry.doc_id
        assert restored.importance == entry.importance
        assert restored.memory_tier == entry.memory_tier
        assert restored.tags == entry.tags

    def test_from_dict_with_missing_fields(self):
        """from_dict 缺少字段时应使用默认值（含新增字段向后兼容）"""
        entry = MemoryEntry.from_dict({})
        assert entry.id  # 应生成 UUID
        assert entry.content == ""
        assert entry.source_type == "manual"
        assert entry.doc_id is None
        assert entry.importance == 0.5
        assert entry.memory_tier == "short_term"
        assert entry.tags == []

    def test_from_dict_old_data_without_new_fields(self):
        """旧版数据（不含 memory_tier 和 tags）应向后兼容，使用默认值"""
        old_data = {
            "id": "old-entry",
            "content": "旧版记忆内容",
            "source_type": "auto_qa",
            "created_at": "2024-01-01T00:00:00Z",
            "doc_id": "doc-old",
            "importance": 0.6,
            "hit_count": 3,
            "last_hit_at": "2024-01-10T00:00:00Z",
        }
        entry = MemoryEntry.from_dict(old_data)
        # 旧字段正常还原
        assert entry.id == "old-entry"
        assert entry.content == "旧版记忆内容"
        assert entry.hit_count == 3
        # 新字段使用默认值
        assert entry.memory_tier == "short_term"
        assert entry.tags == []


# ==================== MemoryStore 测试 ====================


class TestMemoryStoreInit:
    """测试 MemoryStore 初始化和目录创建（需求 1.3）"""

    def test_creates_directories(self, tmp_path):
        """初始化时应自动创建目录结构"""
        data_dir = str(tmp_path / "memory")
        store = MemoryStore(data_dir)
        assert os.path.isdir(data_dir)
        assert os.path.isdir(os.path.join(data_dir, "sessions"))
        assert os.path.isdir(os.path.join(data_dir, "memory_index"))


class TestMemoryStoreProfile:
    """测试用户画像读写（需求 1.1, 1.4）"""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(str(tmp_path / "memory"))

    def test_load_profile_default(self, store):
        """文件不存在时应返回默认结构"""
        profile = store.load_profile()
        assert profile["focus_areas"] == []
        assert profile["keyword_frequencies"] == {}
        assert profile["entries"] == []
        assert profile["updated_at"] == ""

    def test_save_and_load_profile(self, store):
        """保存后应能正确加载"""
        profile = {
            "focus_areas": ["机器学习"],
            "keyword_frequencies": {"机器学习": 5},
            "entries": [],
            "updated_at": "2024-01-15T10:30:00Z",
        }
        store.save_profile(profile)
        loaded = store.load_profile()
        assert loaded["focus_areas"] == ["机器学习"]
        assert loaded["keyword_frequencies"]["机器学习"] == 5

    def test_load_corrupted_profile(self, store):
        """损坏的 JSON 文件应返回默认结构"""
        with open(store.profile_path, "w") as f:
            f.write("这不是有效的 JSON")
        profile = store.load_profile()
        assert profile["focus_areas"] == []


class TestMemoryStoreSession:
    """测试文档会话记忆读写（需求 1.2, 1.4）"""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(str(tmp_path / "memory"))

    def test_load_session_default(self, store):
        """文件不存在时应返回默认结构"""
        session = store.load_session("nonexistent-doc")
        assert session["doc_id"] == "nonexistent-doc"
        assert session["qa_summaries"] == []
        assert session["important_memories"] == []
        assert session["last_accessed"] == ""

    def test_save_and_load_session(self, store):
        """保存后应能正确加载"""
        session = {
            "doc_id": "doc-123",
            "qa_summaries": [{"id": "q1", "question": "问题", "answer": "回答"}],
            "important_memories": [],
            "last_accessed": "2024-01-15T11:00:00Z",
        }
        store.save_session("doc-123", session)
        loaded = store.load_session("doc-123")
        assert loaded["doc_id"] == "doc-123"
        assert len(loaded["qa_summaries"]) == 1

    def test_session_file_path(self, store):
        """session 文件路径应正确"""
        path = store._session_path("doc-abc")
        assert path.endswith("doc-abc_session.json")


class TestMemoryStoreCRUD:
    """测试记忆条目 CRUD 操作"""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(str(tmp_path / "memory"))

    def test_add_entry_without_doc_id(self, store):
        """无 doc_id 的条目应存入 profile"""
        entry = MemoryEntry(
            id="entry-1",
            content="全局记忆",
            source_type="manual",
            importance=1.0,
        )
        store.add_entry(entry)
        profile = store.load_profile()
        assert len(profile["entries"]) == 1
        assert profile["entries"][0]["id"] == "entry-1"

    def test_add_entry_with_doc_id(self, store):
        """有 doc_id 的条目应存入 session"""
        entry = MemoryEntry(
            id="entry-2",
            content="文档记忆",
            source_type="liked",
            doc_id="doc-123",
            importance=1.0,
        )
        store.add_entry(entry)
        session = store.load_session("doc-123")
        assert len(session["important_memories"]) == 1
        assert session["important_memories"][0]["id"] == "entry-2"

    def test_get_all_entries(self, store):
        """应汇总 profile 和所有 session 中的条目"""
        # 添加 profile 条目
        entry1 = MemoryEntry(id="e1", content="全局", source_type="manual")
        store.add_entry(entry1)

        # 添加 session 条目
        entry2 = MemoryEntry(id="e2", content="文档", source_type="liked", doc_id="doc-1")
        store.add_entry(entry2)

        all_entries = store.get_all_entries()
        ids = [e.id for e in all_entries]
        assert "e1" in ids
        assert "e2" in ids

    def test_delete_entry_from_profile(self, store):
        """应能从 profile 中删除条目"""
        entry = MemoryEntry(id="del-1", content="待删除", source_type="manual")
        store.add_entry(entry)
        assert store.delete_entry("del-1") is True
        profile = store.load_profile()
        assert len(profile["entries"]) == 0

    def test_delete_entry_from_session(self, store):
        """应能从 session 中删除条目"""
        entry = MemoryEntry(
            id="del-2", content="待删除", source_type="liked", doc_id="doc-1"
        )
        store.add_entry(entry)
        assert store.delete_entry("del-2") is True
        session = store.load_session("doc-1")
        assert len(session["important_memories"]) == 0

    def test_delete_nonexistent_entry(self, store):
        """删除不存在的条目应返回 False"""
        assert store.delete_entry("nonexistent-id") is False

    def test_update_entry_in_profile(self, store):
        """应能更新 profile 中的条目内容"""
        entry = MemoryEntry(id="upd-1", content="原始内容", source_type="manual")
        store.add_entry(entry)
        assert store.update_entry("upd-1", "更新后的内容") is True
        profile = store.load_profile()
        assert profile["entries"][0]["content"] == "更新后的内容"

    def test_update_entry_in_session(self, store):
        """应能更新 session 中的条目内容"""
        entry = MemoryEntry(
            id="upd-2", content="原始内容", source_type="liked", doc_id="doc-1"
        )
        store.add_entry(entry)
        assert store.update_entry("upd-2", "更新后的内容") is True
        session = store.load_session("doc-1")
        assert session["important_memories"][0]["content"] == "更新后的内容"

    def test_update_nonexistent_entry(self, store):
        """更新不存在的条目应返回 False"""
        assert store.update_entry("nonexistent-id", "新内容") is False

    def test_clear_all(self, store):
        """清空后所有数据应为空"""
        # 添加一些数据
        store.add_entry(MemoryEntry(id="c1", content="全局", source_type="manual"))
        store.add_entry(
            MemoryEntry(id="c2", content="文档", source_type="liked", doc_id="doc-1")
        )
        store.clear_all()

        profile = store.load_profile()
        assert profile["entries"] == []
        assert store.get_all_entries() == []

    def test_add_multiple_entries_same_profile(self, store):
        """向 profile 添加多个条目应全部保留"""
        for i in range(5):
            entry = MemoryEntry(id=f"multi-{i}", content=f"内容{i}", source_type="manual")
            store.add_entry(entry)
        profile = store.load_profile()
        assert len(profile["entries"]) == 5


# ==================== 属性测试（Property-Based Testing） ====================

from hypothesis import given, settings
from hypothesis import strategies as st


class TestMemoryEntrySerializationProperty:
    """Feature: chatpdf-memory-system, Property 1: MemoryEntry 序列化往返一致性

    **Validates: Requirements 1.6**

    对任意有效的 MemoryEntry 对象，序列化为 JSON 后再反序列化，
    应得到与原始对象等价的 MemoryEntry。
    """

    # 构建 MemoryEntry 的 Hypothesis 策略（含新增字段 memory_tier 和 tags）
    memory_entry_strategy = st.builds(
        MemoryEntry,
        id=st.uuids().map(str),
        content=st.text(min_size=0, max_size=500),
        source_type=st.sampled_from(["auto_qa", "manual", "liked", "keyword", "compressed"]),
        created_at=st.datetimes().map(lambda dt: dt.isoformat()),
        doc_id=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        memory_tier=st.sampled_from(["working", "short_term", "long_term", "archived"]),
        tags=st.lists(st.sampled_from(["concept", "fact", "preference", "method", "conclusion", "correction"]), max_size=6, unique=True),
    )

    @given(entry=memory_entry_strategy)
    @settings(max_examples=100)
    def test_property_serialization_roundtrip(self, entry: MemoryEntry):
        """属性测试：MemoryEntry 序列化往返一致性

        对任意生成的 MemoryEntry，to_dict -> JSON 序列化 -> JSON 反序列化 -> from_dict
        应得到与原始对象完全等价的 MemoryEntry。
        """
        # 序列化为字典，再转 JSON 字符串，再解析回字典，再还原为 MemoryEntry
        d = entry.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        restored_data = json.loads(json_str)
        restored = MemoryEntry.from_dict(restored_data)

        # 验证所有字段一致
        assert restored.id == entry.id, f"id 不一致: {restored.id} != {entry.id}"
        assert restored.content == entry.content, f"content 不一致"
        assert restored.source_type == entry.source_type, f"source_type 不一致"
        assert restored.created_at == entry.created_at, f"created_at 不一致"
        assert restored.doc_id == entry.doc_id, f"doc_id 不一致"
        assert restored.importance == entry.importance, f"importance 不一致"
        assert restored.memory_tier == entry.memory_tier, f"memory_tier 不一致"
        assert restored.tags == entry.tags, f"tags 不一致"


class TestDefaultStructureProperty:
    """Feature: chatpdf-memory-system, Property 2: 不存在的文件返回默认结构

    **Validates: Requirements 1.4, 1.5, 2.4, 3.5**

    对任意随机生成的 doc_id（对应文件不存在），调用 load_session(doc_id) 应返回
    包含所有必需字段的默认结构，且不抛出异常。同理，load_profile() 在文件不存在时
    应返回包含所有必需字段的默认结构。
    """

    # 生成合法的 doc_id 字符串（避免文件系统非法字符）
    doc_id_strategy = st.text(
        alphabet=st.sampled_from(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        ),
        min_size=1,
        max_size=100,
    )

    @given(doc_id=doc_id_strategy)
    @settings(max_examples=100)
    def test_property_load_session_default_structure(self, doc_id: str):
        """属性测试：不存在的文件调用 load_session 返回包含所有必需字段的默认结构

        对任意随机 doc_id，在空目录中调用 load_session 应：
        1. 不抛出异常
        2. 返回包含 doc_id、qa_summaries、important_memories、last_accessed 的字典
        3. doc_id 字段值与传入参数一致
        4. 列表字段为空列表
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(os.path.join(tmpdir, "memory"))
            session = store.load_session(doc_id)

            # 验证所有必需字段存在
            assert "doc_id" in session, "缺少 doc_id 字段"
            assert "qa_summaries" in session, "缺少 qa_summaries 字段"
            assert "important_memories" in session, "缺少 important_memories 字段"
            assert "last_accessed" in session, "缺少 last_accessed 字段"

            # 验证默认值正确
            assert session["doc_id"] == doc_id, f"doc_id 不匹配: {session['doc_id']} != {doc_id}"
            assert session["qa_summaries"] == [], "qa_summaries 应为空列表"
            assert session["important_memories"] == [], "important_memories 应为空列表"
            assert session["last_accessed"] == "", "last_accessed 应为空字符串"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_property_load_profile_default_structure(self, data):
        """属性测试：不存在的文件调用 load_profile 返回包含所有必需字段的默认结构

        在随机生成的空目录中调用 load_profile 应：
        1. 不抛出异常
        2. 返回包含 focus_areas、keyword_frequencies、entries、updated_at 的字典
        3. 列表/字典字段为空
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 使用随机子目录名确保每次都是全新目录
            subdir = data.draw(st.text(
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
                min_size=1,
                max_size=20,
            ))
            store = MemoryStore(os.path.join(tmpdir, subdir, "memory"))
            profile = store.load_profile()

            # 验证所有必需字段存在
            assert "focus_areas" in profile, "缺少 focus_areas 字段"
            assert "keyword_frequencies" in profile, "缺少 keyword_frequencies 字段"
            assert "entries" in profile, "缺少 entries 字段"
            assert "updated_at" in profile, "缺少 updated_at 字段"

            # 验证默认值正确
            assert profile["focus_areas"] == [], "focus_areas 应为空列表"
            assert profile["keyword_frequencies"] == {}, "keyword_frequencies 应为空字典"
            assert profile["entries"] == [], "entries 应为空列表"
            assert profile["updated_at"] == "", "updated_at 应为空字符串"
