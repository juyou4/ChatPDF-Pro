"""
语义意群服务单元测试

测试 SemanticGroup 数据类和 SemanticGroupService 的
save_groups / load_groups 持久化方法。
"""

import json
import os
import tempfile

import pytest

from services.semantic_group_service import (
    SCHEMA_VERSION,
    SemanticGroup,
    SemanticGroupService,
)


# ---- 辅助函数 ----

def _make_group(
    group_id: str = "group-0",
    chunk_indices: list = None,
    char_count: int = 4800,
    summary: str = "这是一段摘要",
    digest: str = "这是一段精要内容",
    full_text: str = "这是完整的原始文本内容",
    keywords: list = None,
    page_range: tuple = (1, 3),
    summary_status: str = "ok",
    llm_meta: dict = None,
) -> SemanticGroup:
    """创建测试用的 SemanticGroup 实例"""
    return SemanticGroup(
        group_id=group_id,
        chunk_indices=chunk_indices or [0, 1, 2],
        char_count=char_count,
        summary=summary,
        digest=digest,
        full_text=full_text,
        keywords=keywords or ["关键词1", "关键词2", "关键词3"],
        page_range=page_range,
        summary_status=summary_status,
        llm_meta=llm_meta,
    )


# ---- SemanticGroup 数据类测试 ----

class TestSemanticGroup:
    """SemanticGroup 数据类测试"""

    def test_创建基本意群(self):
        """测试基本的意群创建"""
        group = _make_group()
        assert group.group_id == "group-0"
        assert group.chunk_indices == [0, 1, 2]
        assert group.char_count == 4800
        assert group.summary == "这是一段摘要"
        assert group.digest == "这是一段精要内容"
        assert group.full_text == "这是完整的原始文本内容"
        assert group.keywords == ["关键词1", "关键词2", "关键词3"]
        assert group.page_range == (1, 3)
        assert group.summary_status == "ok"
        assert group.llm_meta is None

    def test_创建带llm_meta的意群(self):
        """测试带 LLM 元数据的意群创建"""
        meta = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "prompt_version": "v1",
            "created_at": "2024-01-01T00:00:00Z",
        }
        group = _make_group(llm_meta=meta)
        assert group.llm_meta == meta
        assert group.llm_meta["model"] == "gpt-4o-mini"

    def test_to_dict转换(self):
        """测试 to_dict 方法将意群转为字典"""
        group = _make_group(page_range=(2, 5))
        d = group.to_dict()
        assert isinstance(d, dict)
        assert d["group_id"] == "group-0"
        assert d["page_range"] == [2, 5]  # tuple 转为 list
        assert d["summary_status"] == "ok"

    def test_from_dict还原(self):
        """测试 from_dict 方法从字典还原意群"""
        data = {
            "group_id": "group-1",
            "chunk_indices": [3, 4],
            "char_count": 3200,
            "summary": "摘要文本",
            "digest": "精要文本",
            "full_text": "完整文本",
            "keywords": ["kw1", "kw2"],
            "page_range": [4, 6],
            "summary_status": "failed",
            "llm_meta": {"model": "claude-3"},
        }
        group = SemanticGroup.from_dict(data)
        assert group.group_id == "group-1"
        assert group.chunk_indices == [3, 4]
        assert group.page_range == (4, 6)  # list 转回 tuple
        assert group.summary_status == "failed"
        assert group.llm_meta == {"model": "claude-3"}

    def test_from_dict默认值(self):
        """测试 from_dict 缺少可选字段时使用默认值"""
        data = {
            "group_id": "group-0",
            "chunk_indices": [0],
            "char_count": 100,
            "summary": "s",
            "digest": "d",
            "full_text": "f",
            "keywords": [],
        }
        group = SemanticGroup.from_dict(data)
        assert group.page_range == (0, 0)
        assert group.summary_status == "ok"
        assert group.llm_meta is None

    def test_to_dict_from_dict往返一致(self):
        """测试 to_dict 和 from_dict 的往返一致性"""
        original = _make_group(
            group_id="group-5",
            chunk_indices=[10, 11, 12, 13],
            char_count=5500,
            summary="往返测试摘要",
            digest="往返测试精要",
            full_text="往返测试全文",
            keywords=["a", "b", "c", "d"],
            page_range=(7, 9),
            summary_status="truncated",
            llm_meta={"model": "test", "temperature": 0.5},
        )
        restored = SemanticGroup.from_dict(original.to_dict())
        assert restored.group_id == original.group_id
        assert restored.chunk_indices == original.chunk_indices
        assert restored.char_count == original.char_count
        assert restored.summary == original.summary
        assert restored.digest == original.digest
        assert restored.full_text == original.full_text
        assert restored.keywords == original.keywords
        assert restored.page_range == original.page_range
        assert restored.summary_status == original.summary_status
        assert restored.llm_meta == original.llm_meta


# ---- SemanticGroupService 持久化测试 ----

class TestSemanticGroupServicePersistence:
    """SemanticGroupService 的 save_groups / load_groups 测试"""

    def setup_method(self):
        """每个测试方法前创建临时目录和服务实例"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = SemanticGroupService()

    def test_保存和加载空列表(self):
        """测试保存和加载空的意群列表"""
        self.service.save_groups("doc-empty", [], self.temp_dir)
        loaded = self.service.load_groups("doc-empty", self.temp_dir)
        assert loaded is not None
        assert loaded == []

    def test_保存和加载单个意群(self):
        """测试保存和加载单个意群"""
        group = _make_group()
        self.service.save_groups("doc-single", [group], self.temp_dir)
        loaded = self.service.load_groups("doc-single", self.temp_dir)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].group_id == "group-0"
        assert loaded[0].page_range == (1, 3)

    def test_保存和加载多个意群(self):
        """测试保存和加载多个意群"""
        groups = [
            _make_group(group_id="group-0", page_range=(1, 2)),
            _make_group(group_id="group-1", chunk_indices=[3, 4], page_range=(3, 5)),
            _make_group(group_id="group-2", chunk_indices=[5, 6, 7], page_range=(6, 8)),
        ]
        self.service.save_groups("doc-multi", groups, self.temp_dir)
        loaded = self.service.load_groups("doc-multi", self.temp_dir)
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0].group_id == "group-0"
        assert loaded[1].group_id == "group-1"
        assert loaded[2].group_id == "group-2"

    def test_JSON文件包含schema_version(self):
        """测试保存的 JSON 文件包含 schema_version 字段"""
        group = _make_group()
        self.service.save_groups("doc-schema", [group], self.temp_dir)

        file_path = os.path.join(self.temp_dir, "doc-schema.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_JSON文件包含完整元数据(self):
        """测试保存的 JSON 文件包含 doc_id、doc_hash、created_at、config"""
        group = _make_group()
        config = {"target_chars": 4000, "min_chars": 2000, "max_chars": 5000}
        self.service.save_groups(
            "doc-meta", [group], self.temp_dir,
            doc_hash="sha256:abc123", config=config,
        )

        file_path = os.path.join(self.temp_dir, "doc-meta.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["doc_id"] == "doc-meta"
        assert data["doc_hash"] == "sha256:abc123"
        assert "created_at" in data
        assert data["config"] == config

    def test_加载不存在的文件返回None(self):
        """测试加载不存在的文件返回 None"""
        loaded = self.service.load_groups("nonexistent", self.temp_dir)
        assert loaded is None

    def test_加载损坏的JSON文件返回None(self):
        """测试加载损坏的 JSON 文件返回 None"""
        file_path = os.path.join(self.temp_dir, "doc-corrupt.json")
        with open(file_path, "w") as f:
            f.write("{invalid json content!!!")

        loaded = self.service.load_groups("doc-corrupt", self.temp_dir)
        assert loaded is None

    def test_schema_version不匹配返回None(self):
        """测试 schema_version 不匹配时返回 None"""
        file_path = os.path.join(self.temp_dir, "doc-old.json")
        data = {
            "schema_version": 999,  # 不匹配的版本号
            "doc_id": "doc-old",
            "groups": [],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = self.service.load_groups("doc-old", self.temp_dir)
        assert loaded is None

    def test_groups字段缺失返回None(self):
        """测试 groups 字段缺失时返回 None"""
        file_path = os.path.join(self.temp_dir, "doc-nogroups.json")
        data = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": "doc-nogroups",
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = self.service.load_groups("doc-nogroups", self.temp_dir)
        assert loaded is None

    def test_groups字段类型错误返回None(self):
        """测试 groups 字段不是列表时返回 None"""
        file_path = os.path.join(self.temp_dir, "doc-badtype.json")
        data = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": "doc-badtype",
            "groups": "not a list",
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = self.service.load_groups("doc-badtype", self.temp_dir)
        assert loaded is None

    def test_意群数据字段缺失返回None(self):
        """测试意群数据中必需字段缺失时返回 None"""
        file_path = os.path.join(self.temp_dir, "doc-badgroup.json")
        data = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": "doc-badgroup",
            "groups": [{"group_id": "group-0"}],  # 缺少必需字段
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = self.service.load_groups("doc-badgroup", self.temp_dir)
        assert loaded is None

    def test_保存自动创建目录(self):
        """测试 save_groups 自动创建不存在的目录"""
        nested_dir = os.path.join(self.temp_dir, "a", "b", "c")
        group = _make_group()
        self.service.save_groups("doc-nested", [group], nested_dir)

        assert os.path.exists(os.path.join(nested_dir, "doc-nested.json"))
        loaded = self.service.load_groups("doc-nested", nested_dir)
        assert loaded is not None
        assert len(loaded) == 1

    def test_保存带llm_meta的意群(self):
        """测试保存和加载带 LLM 元数据的意群"""
        meta = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "prompt_version": "v1",
            "created_at": "2024-01-01T00:00:00Z",
        }
        group = _make_group(llm_meta=meta)
        self.service.save_groups("doc-llm", [group], self.temp_dir)
        loaded = self.service.load_groups("doc-llm", self.temp_dir)
        assert loaded is not None
        assert loaded[0].llm_meta == meta

    def test_保存不同summary_status(self):
        """测试保存和加载不同 summary_status 的意群"""
        groups = [
            _make_group(group_id="g-ok", summary_status="ok"),
            _make_group(group_id="g-failed", summary_status="failed"),
            _make_group(group_id="g-truncated", summary_status="truncated"),
        ]
        self.service.save_groups("doc-status", groups, self.temp_dir)
        loaded = self.service.load_groups("doc-status", self.temp_dir)
        assert loaded is not None
        assert loaded[0].summary_status == "ok"
        assert loaded[1].summary_status == "failed"
        assert loaded[2].summary_status == "truncated"

    def test_JSON文件使用UTF8编码(self):
        """测试 JSON 文件正确处理中文内容"""
        group = _make_group(
            summary="这是中文摘要测试",
            digest="这是中文精要内容，包含特殊字符：①②③",
            full_text="完整的中文文本，包含 English 混合内容",
            keywords=["人工智能", "自然语言处理", "深度学习"],
        )
        self.service.save_groups("doc-utf8", [group], self.temp_dir)
        loaded = self.service.load_groups("doc-utf8", self.temp_dir)
        assert loaded is not None
        assert loaded[0].summary == "这是中文摘要测试"
        assert "①②③" in loaded[0].digest
        assert loaded[0].keywords == ["人工智能", "自然语言处理", "深度学习"]

    def test_默认config值(self):
        """测试不传 config 参数时使用默认值"""
        group = _make_group()
        self.service.save_groups("doc-defconfig", [group], self.temp_dir)

        file_path = os.path.join(self.temp_dir, "doc-defconfig.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["config"] == {
            "target_chars": 5000,
            "min_chars": 2500,
            "max_chars": 6000,
        }


# ---- _detect_boundary 硬边界检测测试 ----

class TestDetectBoundary:
    """_detect_boundary 硬边界检测测试"""

    def setup_method(self):
        """每个测试方法前创建服务实例"""
        self.service = SemanticGroupService()

    # -- 页面边界 --

    def test_不同页码触发页面边界(self):
        """不同页码时应检测到硬边界"""
        assert self.service._detect_boundary("文本A", "文本B", page=1, next_page=2) is True

    def test_相同页码无页面边界(self):
        """相同页码且无其他边界时不应切分"""
        assert self.service._detect_boundary("普通文本", "普通后续文本", page=1, next_page=1) is False

    # -- 标题边界：编号模式 --

    def test_编号标题_单级编号(self):
        """下一个分块以 '1. 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "1. 引言", page=1, next_page=1) is True

    def test_编号标题_多级编号(self):
        """下一个分块以 '1.1 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "1.1 方法论", page=1, next_page=1) is True

    def test_编号标题_三级编号(self):
        """下一个分块以 '2.3.4 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "2.3.4 实验结果", page=1, next_page=1) is True

    # -- 标题边界：全大写行 --

    def test_全大写英文标题(self):
        """下一个分块以全大写英文行开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "INTRODUCTION", page=1, next_page=1) is True

    def test_全大写带空格(self):
        """全大写英文行带空格也应检测到"""
        assert self.service._detect_boundary("前文", "RELATED WORK", page=1, next_page=1) is True

    def test_非全大写不触发(self):
        """混合大小写不应触发全大写标题检测"""
        assert self.service._detect_boundary("前文", "Introduction to the topic", page=1, next_page=1) is False

    # -- 标题边界：Markdown 标题 --

    def test_markdown一级标题(self):
        """下一个分块以 '# 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "# 第一章", page=1, next_page=1) is True

    def test_markdown二级标题(self):
        """下一个分块以 '## 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "## 方法", page=1, next_page=1) is True

    def test_markdown三级标题(self):
        """下一个分块以 '### 标题' 开头时应检测到标题边界"""
        assert self.service._detect_boundary("前文", "### 子节", page=1, next_page=1) is True

    # -- 表格/代码块边界 --

    def test_表格行边界(self):
        """下一个分块以表格行开头时应检测到边界"""
        assert self.service._detect_boundary("前文", "| 列1 | 列2 | 列3 |", page=1, next_page=1) is True

    def test_代码块开始标记(self):
        """下一个分块以 ``` 开头时应检测到边界"""
        assert self.service._detect_boundary("前文", "```python\nprint('hello')", page=1, next_page=1) is True

    def test_代码块纯标记(self):
        """下一个分块以纯 ``` 开头时应检测到边界"""
        assert self.service._detect_boundary("前文", "```", page=1, next_page=1) is True

    # -- 无边界情况 --

    def test_普通中文文本无边界(self):
        """普通中文文本之间不应有边界"""
        assert self.service._detect_boundary(
            "这是一段普通的中文文本。",
            "这是后续的中文文本内容。",
            page=1, next_page=1
        ) is False

    def test_空字符串next_chunk无边界(self):
        """空的 next_chunk 不应触发边界"""
        assert self.service._detect_boundary("前文", "", page=1, next_page=1) is False

    def test_多行分块只检测第一行(self):
        """多行分块只检测第一行，第二行有标题不应触发"""
        chunk = "普通第一行\n# 这是第二行的标题"
        assert self.service._detect_boundary("前文", chunk, page=1, next_page=1) is False


# ---- _aggregate_chunks 分块聚合测试 ----

class TestAggregateChunks:
    """_aggregate_chunks 分块聚合测试"""

    def setup_method(self):
        """每个测试方法前创建服务实例"""
        self.service = SemanticGroupService()

    def test_空分块列表返回空列表(self):
        """空分块列表应返回空的候选意群列表"""
        result = self.service._aggregate_chunks([], [], target_chars=5000)
        assert result == []

    def test_单个分块生成单个意群(self):
        """单个分块应生成一个候选意群"""
        chunks = ["这是一段文本"]
        pages = [1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 1
        assert result[0]["chunk_indices"] == [0]
        assert result[0]["full_text"] == "这是一段文本"
        assert result[0]["page_range"] == (1, 1)

    def test_小分块聚合到一个意群(self):
        """多个小分块应聚合到同一个意群（未达到目标字符数）"""
        chunks = ["短文本A", "短文本B", "短文本C"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(
            chunks, pages, target_chars=5000, min_chars=2500, max_chars=6000
        )
        assert len(result) == 1
        assert result[0]["chunk_indices"] == [0, 1, 2]
        assert result[0]["page_range"] == (1, 1)

    def test_达到目标字符数时切分(self):
        """累计字符数达到目标时应切分"""
        # 每个分块 2000 字符，目标 5000
        chunk_text = "A" * 2000
        chunks = [chunk_text] * 5
        pages = [1] * 5
        result = self.service._aggregate_chunks(
            chunks, pages, target_chars=5000, min_chars=2500, max_chars=6000
        )
        # 前 3 个分块累计 6000 字符（含换行符），达到目标后切分
        assert len(result) >= 2

    def test_超过max_chars时强制切分(self):
        """加入分块后超过 max_chars 时应强制切分"""
        # 每个分块 3500 字符，max_chars=6000
        chunk_text = "B" * 3500
        chunks = [chunk_text] * 4
        pages = [1] * 4
        result = self.service._aggregate_chunks(
            chunks, pages, target_chars=5000, min_chars=2500, max_chars=6000
        )
        # 第 1+2 个分块 = 7000 > 6000，所以第 2 个分块前应切分
        # 每个意群最多包含 1 个分块（因为 3500 < 5000 但 3500+3500=7000 > 6000）
        for candidate in result:
            # 每个候选意群的原始分块字符总和不应超过 max_chars
            raw_chars = sum(len(chunks[i]) for i in candidate["chunk_indices"])
            # char_count 是拼接后的（含换行符），但原始字符总和也不应过大
            assert raw_chars <= 6000 or len(candidate["chunk_indices"]) == 1

    def test_页面边界强制切分(self):
        """不同页码的分块之间应强制切分"""
        chunks = ["页面1的文本", "页面2的文本", "页面2的后续"]
        pages = [1, 2, 2]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 2
        assert result[0]["chunk_indices"] == [0]
        assert result[0]["page_range"] == (1, 1)
        assert result[1]["chunk_indices"] == [1, 2]
        assert result[1]["page_range"] == (2, 2)

    def test_标题边界强制切分(self):
        """标题行开头的分块之前应强制切分"""
        chunks = ["前面的内容", "1.1 新章节标题", "章节内容"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 2
        assert result[0]["chunk_indices"] == [0]
        assert result[1]["chunk_indices"] == [1, 2]

    def test_markdown标题边界切分(self):
        """Markdown 标题开头的分块之前应强制切分"""
        chunks = ["前面的内容", "## 新章节", "章节内容"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 2
        assert result[0]["chunk_indices"] == [0]
        assert result[1]["chunk_indices"] == [1, 2]

    def test_表格边界切分(self):
        """表格行开头的分块之前应强制切分"""
        chunks = ["前面的内容", "| 列1 | 列2 |", "后续内容"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 2
        assert result[0]["chunk_indices"] == [0]

    def test_代码块边界切分(self):
        """代码块标记开头的分块之前应强制切分"""
        chunks = ["前面的内容", "```python\ncode here", "后续内容"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 2
        assert result[0]["chunk_indices"] == [0]

    def test_候选意群包含正确的字段(self):
        """每个候选意群应包含 chunk_indices、full_text、char_count、page_range"""
        chunks = ["文本内容A", "文本内容B"]
        pages = [1, 2]
        result = self.service._aggregate_chunks(chunks, pages)
        for candidate in result:
            assert "chunk_indices" in candidate
            assert "full_text" in candidate
            assert "char_count" in candidate
            assert "page_range" in candidate
            assert isinstance(candidate["chunk_indices"], list)
            assert isinstance(candidate["full_text"], str)
            assert isinstance(candidate["char_count"], int)
            assert isinstance(candidate["page_range"], tuple)

    def test_char_count等于full_text长度(self):
        """char_count 应等于 full_text 的实际长度"""
        chunks = ["文本A", "文本B", "文本C"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        for candidate in result:
            assert candidate["char_count"] == len(candidate["full_text"])

    def test_所有分块索引被覆盖(self):
        """所有分块索引应被恰好覆盖一次"""
        chunks = ["A" * 3000, "B" * 3000, "# 新章节", "C" * 2000, "D" * 2000]
        pages = [1, 1, 1, 1, 1]
        result = self.service._aggregate_chunks(
            chunks, pages, target_chars=5000, min_chars=2500, max_chars=6000
        )
        # 收集所有分块索引
        all_indices = []
        for candidate in result:
            all_indices.extend(candidate["chunk_indices"])
        # 排序后应等于 [0, 1, 2, 3, 4]
        assert sorted(all_indices) == list(range(len(chunks)))

    def test_page_range跨多页(self):
        """意群的 page_range 应正确反映跨页情况"""
        # 同一页的多个分块
        chunks = ["文本1", "文本2", "文本3"]
        pages = [3, 3, 5]
        result = self.service._aggregate_chunks(chunks, pages)
        # 页面 3→5 有页面边界，应切分
        assert result[0]["page_range"] == (3, 3)
        assert result[1]["page_range"] == (5, 5)

    def test_连续多个硬边界(self):
        """连续多个硬边界应产生多个小意群"""
        chunks = ["# 章节1", "# 章节2", "# 章节3"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        # 每个 Markdown 标题都是硬边界，应产生 3 个意群
        assert len(result) == 3
        assert result[0]["chunk_indices"] == [0]
        assert result[1]["chunk_indices"] == [1]
        assert result[2]["chunk_indices"] == [2]

    def test_full_text由换行符拼接(self):
        """多个分块的 full_text 应由换行符拼接"""
        chunks = ["第一段", "第二段", "第三段"]
        pages = [1, 1, 1]
        result = self.service._aggregate_chunks(chunks, pages)
        assert len(result) == 1
        assert result[0]["full_text"] == "第一段\n第二段\n第三段"


# ---- _generate_summary 和 _extract_keywords 测试 ----

from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


def _run_async(coro):
    """辅助函数：在同步测试中运行异步协程"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestGenerateSummary:
    """_generate_summary 方法测试"""

    def test_短文本直接返回_不调用LLM(self):
        """当原文本短于 max_length 时，直接返回原文，status 为 ok"""
        service = SemanticGroupService()
        short_text = "这是一段很短的文本"
        summary, status = _run_async(service._generate_summary(short_text, 80))
        assert summary == short_text
        assert status == "ok"

    def test_无api_key时降级为截断(self):
        """未配置 API key 时，降级为文本截断，status 为 failed"""
        service = SemanticGroupService(api_key="")
        long_text = "这是一段" * 100  # 400 字符
        summary, status = _run_async(service._generate_summary(long_text, 80))
        assert len(summary) <= 80
        assert summary == long_text[:80]
        assert status == "failed"

    def test_LLM调用成功返回摘要(self):
        """LLM 调用成功时，返回生成的摘要，status 为 ok"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 50

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "这是LLM生成的摘要"
            summary, status = _run_async(service._generate_summary(long_text, 80))
            assert summary == "这是LLM生成的摘要"
            assert status == "ok"
            mock_llm.assert_called_once()

    def test_LLM返回超长结果时截断(self):
        """LLM 返回的结果超过 max_length 时，截断到 max_length"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "原始文本" * 100

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # 返回超过 80 字的结果
            mock_llm.return_value = "A" * 200
            summary, status = _run_async(service._generate_summary(long_text, 80))
            assert len(summary) <= 80
            assert status == "ok"

    def test_LLM返回空内容时降级(self):
        """LLM 返回空内容时，降级为文本截断"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "原始文本内容" * 100

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "   "  # 空白内容
            summary, status = _run_async(service._generate_summary(long_text, 80))
            assert summary == long_text[:80]
            assert status == "failed"

    def test_LLM调用异常时降级为截断(self):
        """LLM API 调用抛出异常时，降级为文本截断，status 为 failed"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "原始文本内容" * 100

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API 连接超时")
            summary, status = _run_async(service._generate_summary(long_text, 80))
            assert len(summary) <= 80
            assert summary == long_text[:80]
            assert status == "failed"

    def test_digest粒度_max_length_1000(self):
        """使用 max_length=1000 生成 digest 粒度摘要"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "详细的文本内容，" * 500  # 很长的文本

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "这是一段精要内容，包含了文档的主要论点和关键发现。"
            summary, status = _run_async(service._generate_summary(long_text, 1000))
            assert len(summary) <= 1000
            assert status == "ok"

    def test_digest粒度_降级截断到1000字(self):
        """digest 粒度 LLM 失败时，截断到 1000 字"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "A" * 5000

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API 错误")
            summary, status = _run_async(service._generate_summary(long_text, 1000))
            assert len(summary) == 1000
            assert summary == long_text[:1000]
            assert status == "failed"

    def test_输入文本超过8000字时截断输入(self):
        """输入文本超过 8000 字符时，传给 LLM 的文本应被截断"""
        service = SemanticGroupService(api_key="test-key")
        very_long_text = "X" * 20000

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "摘要结果"
            _run_async(service._generate_summary(very_long_text, 80))
            # 验证传给 LLM 的 prompt 中文本被截断
            call_args = mock_llm.call_args[0][0]
            # prompt 中的文本部分不应包含完整的 20000 字符
            assert len(call_args) < 20000


class TestExtractKeywords:
    """_extract_keywords 方法测试"""

    def test_无api_key时返回空列表(self):
        """未配置 API key 时，返回空列表"""
        service = SemanticGroupService(api_key="")
        keywords = _run_async(service._extract_keywords("一些文本内容"))
        assert keywords == []

    def test_LLM调用成功_逗号分隔(self):
        """LLM 返回逗号分隔的关键词"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "人工智能,深度学习,自然语言处理,机器翻译"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert keywords == ["人工智能", "深度学习", "自然语言处理", "机器翻译"]

    def test_LLM调用成功_中文逗号分隔(self):
        """LLM 返回中文逗号分隔的关键词"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "人工智能，深度学习，自然语言处理，机器翻译"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert keywords == ["人工智能", "深度学习", "自然语言处理", "机器翻译"]

    def test_LLM调用成功_顿号分隔(self):
        """LLM 返回顿号分隔的关键词"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "人工智能、深度学习、自然语言处理、机器翻译"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert keywords == ["人工智能", "深度学习", "自然语言处理", "机器翻译"]

    def test_LLM返回带编号的关键词(self):
        """LLM 返回带编号前缀的关键词，应去除编号"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "1. 人工智能\n2. 深度学习\n3. 自然语言处理\n4. 机器翻译"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert "人工智能" in keywords
            assert "深度学习" in keywords
            assert len(keywords) == 4

    def test_LLM返回超过6个关键词时截断(self):
        """LLM 返回超过 6 个关键词时，只保留前 6 个"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "kw1,kw2,kw3,kw4,kw5,kw6,kw7,kw8"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert len(keywords) == 6

    def test_LLM返回空内容时返回空列表(self):
        """LLM 返回空内容时，返回空列表"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "  "
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert keywords == []

    def test_LLM调用异常时返回空列表(self):
        """LLM API 调用抛出异常时，返回空列表"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("网络错误")
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert keywords == []

    def test_LLM返回带破折号前缀的关键词(self):
        """LLM 返回带 - 前缀的关键词，应去除前缀"""
        service = SemanticGroupService(api_key="test-key")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "- 人工智能\n- 深度学习\n- NLP\n- 机器翻译"
            keywords = _run_async(service._extract_keywords("一些文本"))
            assert "人工智能" in keywords
            assert "深度学习" in keywords
            assert len(keywords) == 4


class TestBuildLlmMeta:
    """_build_llm_meta 方法测试"""

    def test_返回正确的元数据结构(self):
        """_build_llm_meta 应返回包含 model、temperature、prompt_version、created_at 的字典"""
        service = SemanticGroupService(
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.3,
            prompt_version="v1",
        )
        meta = service._build_llm_meta()
        assert meta["model"] == "gpt-4o-mini"
        assert meta["temperature"] == 0.3
        assert meta["prompt_version"] == "v1"
        assert "created_at" in meta

    def test_自定义模型参数(self):
        """使用自定义模型参数时，元数据应反映这些参数"""
        service = SemanticGroupService(
            api_key="key",
            model="claude-3-haiku",
            temperature=0.5,
            prompt_version="v2",
        )
        meta = service._build_llm_meta()
        assert meta["model"] == "claude-3-haiku"
        assert meta["temperature"] == 0.5
        assert meta["prompt_version"] == "v2"


class TestCallLlm:
    """_call_llm 方法测试"""

    def test_无api_key时_generate_summary降级(self):
        """确保在没有 API key 的情况下，_generate_summary 不会调用 _call_llm"""
        service = SemanticGroupService(api_key="")
        text = "A" * 200
        summary, status = _run_async(service._generate_summary(text, 80))
        # 没有 API key 时应直接降级，不调用 LLM
        assert status == "failed"
        assert len(summary) <= 80

    def test_构造函数默认参数(self):
        """验证构造函数的默认参数值"""
        service = SemanticGroupService()
        assert service.api_key == ""
        assert service.model == "gpt-4o-mini"
        assert service.provider == "openai"
        assert service.endpoint == ""
        assert service.temperature == 0.3
        assert service.prompt_version == "v1"

    def test_构造函数自定义参数(self):
        """验证构造函数接受自定义参数"""
        service = SemanticGroupService(
            api_key="sk-test",
            model="gpt-4",
            provider="deepseek",
            endpoint="https://api.deepseek.com/v1/chat/completions",
            temperature=0.5,
            prompt_version="v2",
        )
        assert service.api_key == "sk-test"
        assert service.model == "gpt-4"
        assert service.provider == "deepseek"
        assert service.endpoint == "https://api.deepseek.com/v1/chat/completions"
        assert service.temperature == 0.5
        assert service.prompt_version == "v2"


# ---- generate_groups 集成测试 ----


class TestGenerateGroups:
    """generate_groups 方法测试"""

    def test_空分块列表返回空列表(self):
        """空分块列表应返回空的意群列表"""
        service = SemanticGroupService()
        result = _run_async(service.generate_groups([], []))
        assert result == []

    def test_单个短分块_无需LLM(self):
        """单个短分块（短于 80 字）不需要调用 LLM，直接返回原文作为 summary 和 digest"""
        service = SemanticGroupService(api_key="")
        chunks = ["这是一段短文本"]
        pages = [1]
        result = _run_async(service.generate_groups(chunks, pages))
        assert len(result) == 1
        group = result[0]
        assert group.group_id == "group-0"
        assert group.chunk_indices == [0]
        assert group.full_text == "这是一段短文本"
        # 短文本直接返回原文，summary_status 为 ok
        assert group.summary == "这是一段短文本"
        assert group.digest == "这是一段短文本"
        assert group.summary_status == "ok"
        assert group.page_range == (1, 1)
        assert group.llm_meta is not None
        assert isinstance(group.keywords, list)

    def test_多个分块生成多个意群_页面边界切分(self):
        """不同页码的分块应被切分为不同意群"""
        service = SemanticGroupService(api_key="")
        chunks = ["页面1内容", "页面2内容"]
        pages = [1, 2]
        result = _run_async(service.generate_groups(chunks, pages))
        assert len(result) == 2
        assert result[0].group_id == "group-0"
        assert result[0].page_range == (1, 1)
        assert result[1].group_id == "group-1"
        assert result[1].page_range == (2, 2)

    def test_group_id格式正确(self):
        """每个意群的 group_id 应为 'group-{index}' 格式"""
        service = SemanticGroupService(api_key="")
        chunks = ["# 章节1", "# 章节2", "# 章节3"]
        pages = [1, 1, 1]
        result = _run_async(service.generate_groups(chunks, pages))
        for i, group in enumerate(result):
            assert group.group_id == f"group-{i}"

    def test_LLM成功时_summary和digest正确生成(self):
        """LLM 调用成功时，summary 和 digest 应由 LLM 生成"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 200  # 超过 1000 字

        call_count = 0

        async def mock_generate_summary(text, max_length):
            nonlocal call_count
            call_count += 1
            if max_length == 80:
                return "LLM生成的摘要", "ok"
            elif max_length == 1000:
                return "LLM生成的精要内容", "ok"
            return text[:max_length], "ok"

        async def mock_extract_keywords(text):
            return ["关键词1", "关键词2", "关键词3"]

        with patch.object(service, "_generate_summary", side_effect=mock_generate_summary):
            with patch.object(service, "_extract_keywords", side_effect=mock_extract_keywords):
                result = _run_async(service.generate_groups([long_text], [1]))

        assert len(result) == 1
        group = result[0]
        assert group.summary == "LLM生成的摘要"
        assert group.digest == "LLM生成的精要内容"
        assert group.keywords == ["关键词1", "关键词2", "关键词3"]
        assert group.summary_status == "ok"
        assert call_count == 2  # summary 和 digest 各调用一次

    def test_summary失败时_status为failed(self):
        """summary 生成失败时，summary_status 应为 failed"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 200

        async def mock_generate_summary(text, max_length):
            if max_length == 80:
                return text[:80], "failed"  # summary 失败
            return "精要内容", "ok"

        async def mock_extract_keywords(text):
            return []

        with patch.object(service, "_generate_summary", side_effect=mock_generate_summary):
            with patch.object(service, "_extract_keywords", side_effect=mock_extract_keywords):
                result = _run_async(service.generate_groups([long_text], [1]))

        assert result[0].summary_status == "failed"

    def test_digest失败时_status为failed(self):
        """digest 生成失败时，summary_status 应为 failed"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 200

        async def mock_generate_summary(text, max_length):
            if max_length == 80:
                return "摘要", "ok"
            return text[:1000], "failed"  # digest 失败

        async def mock_extract_keywords(text):
            return []

        with patch.object(service, "_generate_summary", side_effect=mock_generate_summary):
            with patch.object(service, "_extract_keywords", side_effect=mock_extract_keywords):
                result = _run_async(service.generate_groups([long_text], [1]))

        assert result[0].summary_status == "failed"

    def test_两者都失败时_status为failed(self):
        """summary 和 digest 都失败时，summary_status 应为 failed"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 200

        async def mock_generate_summary(text, max_length):
            return text[:max_length], "failed"

        async def mock_extract_keywords(text):
            return []

        with patch.object(service, "_generate_summary", side_effect=mock_generate_summary):
            with patch.object(service, "_extract_keywords", side_effect=mock_extract_keywords):
                result = _run_async(service.generate_groups([long_text], [1]))

        assert result[0].summary_status == "failed"

    def test_两者都成功时_status为ok(self):
        """summary 和 digest 都成功时，summary_status 应为 ok"""
        service = SemanticGroupService(api_key="test-key")
        long_text = "这是一段很长的文本内容，" * 200

        async def mock_generate_summary(text, max_length):
            return "生成的内容", "ok"

        async def mock_extract_keywords(text):
            return ["kw1"]

        with patch.object(service, "_generate_summary", side_effect=mock_generate_summary):
            with patch.object(service, "_extract_keywords", side_effect=mock_extract_keywords):
                result = _run_async(service.generate_groups([long_text], [1]))

        assert result[0].summary_status == "ok"

    def test_llm_meta包含正确字段(self):
        """每个意群的 llm_meta 应包含 model、temperature、prompt_version、created_at"""
        service = SemanticGroupService(
            api_key="",
            model="gpt-4o-mini",
            temperature=0.3,
            prompt_version="v1",
        )
        chunks = ["短文本"]
        pages = [1]
        result = _run_async(service.generate_groups(chunks, pages))
        meta = result[0].llm_meta
        assert meta is not None
        assert meta["model"] == "gpt-4o-mini"
        assert meta["temperature"] == 0.3
        assert meta["prompt_version"] == "v1"
        assert "created_at" in meta

    def test_char_count与full_text长度一致(self):
        """每个意群的 char_count 应等于 full_text 的长度"""
        service = SemanticGroupService(api_key="")
        chunks = ["文本A", "文本B", "文本C"]
        pages = [1, 1, 1]
        result = _run_async(service.generate_groups(chunks, pages))
        for group in result:
            assert group.char_count == len(group.full_text)

    def test_返回的是SemanticGroup实例(self):
        """generate_groups 返回的每个元素应为 SemanticGroup 实例"""
        service = SemanticGroupService(api_key="")
        chunks = ["文本内容"]
        pages = [1]
        result = _run_async(service.generate_groups(chunks, pages))
        assert len(result) == 1
        assert isinstance(result[0], SemanticGroup)

    def test_自定义聚合参数传递(self):
        """自定义的 target_chars、min_chars、max_chars 应正确传递给 _aggregate_chunks"""
        service = SemanticGroupService(api_key="")
        # 每个分块 100 字符，target=150 时应在 2 个分块后切分
        # 注意：使用小写字母避免触发全大写标题检测
        chunks = ["a" * 100, "b" * 100, "c" * 100]
        pages = [1, 1, 1]
        result = _run_async(service.generate_groups(
            chunks, pages, target_chars=150, min_chars=50, max_chars=300
        ))
        # 前 2 个分块累计 200 >= target 150，第 3 个分块前切分
        assert len(result) == 2

    def test_无api_key时降级行为(self):
        """无 API key 时，长文本的 summary 应为截断，status 为 failed"""
        service = SemanticGroupService(api_key="")
        long_text = "这是一段很长的文本内容用于测试降级行为，" * 100  # 超过 1000 字
        chunks = [long_text]
        pages = [1]
        result = _run_async(service.generate_groups(chunks, pages))
        assert len(result) == 1
        group = result[0]
        # 无 API key 时 summary 和 digest 都降级为截断
        assert len(group.summary) <= 80
        assert len(group.digest) <= 1000
        assert group.summary_status == "failed"
        assert group.keywords == []  # 无 API key 时关键词为空
