"""
上下文构建器单元测试

测试 ContextBuilder 类的 build_context 方法，验证：
- 格式化上下文字符串包含正确的引用编号、意群标识、粒度级别、页码范围和关键词
- 引文映射列表（citations）正确生成
- 边界情况处理（空列表、无关键词等）
"""

import pytest

from services.context_builder import ContextBuilder
from services.semantic_group_service import SemanticGroup


def _make_group(
    group_id: str = "group-0",
    chunk_indices: list = None,
    char_count: int = 100,
    summary: str = "这是摘要",
    digest: str = "这是精要内容",
    full_text: str = "这是完整的文本内容",
    keywords: list = None,
    page_range: tuple = (1, 3),
    summary_status: str = "ok",
    llm_meta: dict = None,
) -> SemanticGroup:
    """创建测试用的 SemanticGroup 对象"""
    return SemanticGroup(
        group_id=group_id,
        chunk_indices=chunk_indices or [0, 1, 2],
        char_count=char_count,
        summary=summary,
        digest=digest,
        full_text=full_text,
        keywords=keywords if keywords is not None else ["关键词1", "关键词2"],
        page_range=page_range,
        summary_status=summary_status,
        llm_meta=llm_meta,
    )


class TestBuildContext:
    """测试 build_context 方法"""

    def test_空列表返回空字符串和空引文(self):
        """空的 selections 列表应返回空字符串和空引文列表"""
        builder = ContextBuilder()
        context, citations = builder.build_context([])
        assert context == ""
        assert citations == []

    def test_单个意群_full粒度_格式正确(self):
        """单个意群使用 full 粒度时，上下文格式应包含所有必要元素"""
        builder = ContextBuilder()
        group = _make_group(
            group_id="group-0",
            full_text="这是完整的文本内容",
            keywords=["机器学习", "深度学习"],
            page_range=(1, 3),
        )
        selections = [{"group": group, "granularity": "full", "tokens": 100}]

        context, citations = builder.build_context(selections)

        # 验证引用编号
        assert "[1]" in context
        # 验证意群标识
        assert "group-0" in context
        # 验证粒度级别
        assert "full" in context
        # 验证页码范围
        assert "页码: 1-3" in context
        # 验证关键词
        assert "机器学习" in context
        assert "深度学习" in context
        # 验证内容
        assert "这是完整的文本内容" in context

    def test_单个意群_digest粒度_使用digest文本(self):
        """使用 digest 粒度时，应使用 digest 字段的文本"""
        builder = ContextBuilder()
        group = _make_group(
            group_id="group-1",
            digest="这是精要内容文本",
            full_text="这是完整文本不应出现",
            page_range=(5, 8),
        )
        selections = [{"group": group, "granularity": "digest", "tokens": 50}]

        context, citations = builder.build_context(selections)

        assert "这是精要内容文本" in context
        assert "digest" in context
        assert "页码: 5-8" in context

    def test_单个意群_summary粒度_使用summary文本(self):
        """使用 summary 粒度时，应使用 summary 字段的文本"""
        builder = ContextBuilder()
        group = _make_group(
            group_id="group-2",
            summary="这是摘要文本",
            page_range=(10, 10),
        )
        selections = [{"group": group, "granularity": "summary", "tokens": 20}]

        context, citations = builder.build_context(selections)

        assert "这是摘要文本" in context
        assert "summary" in context
        assert "页码: 10-10" in context

    def test_多个意群_引用编号递增(self):
        """多个意群时，引用编号应从 1 开始递增"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id="group-0", page_range=(1, 2)),
            _make_group(group_id="group-1", page_range=(3, 5)),
            _make_group(group_id="group-2", page_range=(6, 8)),
        ]
        selections = [
            {"group": groups[0], "granularity": "full", "tokens": 100},
            {"group": groups[1], "granularity": "digest", "tokens": 50},
            {"group": groups[2], "granularity": "summary", "tokens": 20},
        ]

        context, citations = builder.build_context(selections)

        # 验证引用编号
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        # 验证各意群标识
        assert "group-0" in context
        assert "group-1" in context
        assert "group-2" in context

    def test_多个意群_citations映射正确(self):
        """多个意群时，citations 映射应正确包含 ref、group_id 和 page_range"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id="group-0", page_range=(1, 2)),
            _make_group(group_id="group-1", page_range=(3, 5)),
        ]
        selections = [
            {"group": groups[0], "granularity": "full", "tokens": 100},
            {"group": groups[1], "granularity": "digest", "tokens": 50},
        ]

        _, citations = builder.build_context(selections)

        assert len(citations) == 2
        assert citations[0] == {"ref": 1, "group_id": "group-0", "page_range": [1, 2]}
        assert citations[1] == {"ref": 2, "group_id": "group-1", "page_range": [3, 5]}

    def test_无关键词时不显示关键词行(self):
        """当意群没有关键词时，上下文中不应包含关键词行"""
        builder = ContextBuilder()
        group = _make_group(group_id="group-0", keywords=[])
        selections = [{"group": group, "granularity": "full", "tokens": 100}]

        context, _ = builder.build_context(selections)

        assert "关键词:" not in context

    def test_上下文块之间用双换行分隔(self):
        """多个意群的上下文块之间应用双换行分隔"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id="group-0"),
            _make_group(group_id="group-1"),
        ]
        selections = [
            {"group": groups[0], "granularity": "full", "tokens": 100},
            {"group": groups[1], "granularity": "full", "tokens": 100},
        ]

        context, _ = builder.build_context(selections)

        # 两个上下文块之间应有双换行
        assert "\n\n" in context

    def test_头部格式完整(self):
        """验证头部格式：[ref]【group_id - granularity | 页码: start-end】"""
        builder = ContextBuilder()
        group = _make_group(group_id="group-5", page_range=(10, 20))
        selections = [{"group": group, "granularity": "digest", "tokens": 50}]

        context, _ = builder.build_context(selections)

        # 验证完整的头部格式
        assert "[1]【group-5 - digest | 页码: 10-20】" in context


class TestCitationRefNumbering:
    """测试引用编号分配（需求 9.1）

    验证 build_context 返回的 citations 列表中 ref 编号从 1 开始递增。
    """

    def test_单个意群_ref编号为1(self):
        """单个意群时，citations 中 ref 编号应为 1"""
        builder = ContextBuilder()
        group = _make_group(group_id="group-0", page_range=(1, 2))
        selections = [{"group": group, "granularity": "full", "tokens": 100}]

        _, citations = builder.build_context(selections)

        assert len(citations) == 1
        assert citations[0]["ref"] == 1

    def test_多个意群_ref编号从1递增(self):
        """多个意群时，citations 中 ref 编号应从 1 开始连续递增"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id=f"group-{i}", page_range=(i * 10, i * 10 + 5))
            for i in range(5)
        ]
        selections = [
            {"group": g, "granularity": "digest", "tokens": 50}
            for g in groups
        ]

        _, citations = builder.build_context(selections)

        assert len(citations) == 5
        for i, citation in enumerate(citations):
            assert citation["ref"] == i + 1, (
                f"第 {i} 个 citation 的 ref 应为 {i + 1}，实际为 {citation['ref']}"
            )

    def test_ref编号与上下文中的引用编号一致(self):
        """citations 中的 ref 编号应与上下文字符串中的 [n] 引用编号一致"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id="group-0", page_range=(1, 3)),
            _make_group(group_id="group-1", page_range=(4, 6)),
            _make_group(group_id="group-2", page_range=(7, 9)),
        ]
        selections = [
            {"group": g, "granularity": "full", "tokens": 100}
            for g in groups
        ]

        context, citations = builder.build_context(selections)

        # 验证上下文中包含与 citations ref 对应的引用编号
        for citation in citations:
            ref_marker = f"[{citation['ref']}]"
            assert ref_marker in context, (
                f"上下文中应包含引用编号 {ref_marker}"
            )


class TestCitationMapping:
    """测试 citations 映射正确性（需求 9.2）

    验证每个 citation 的 group_id 和 page_range 与输入的意群对象匹配。
    """

    def test_单个意群_映射字段正确(self):
        """单个意群时，citation 的 group_id 和 page_range 应与输入匹配"""
        builder = ContextBuilder()
        group = _make_group(group_id="group-42", page_range=(15, 20))
        selections = [{"group": group, "granularity": "summary", "tokens": 30}]

        _, citations = builder.build_context(selections)

        assert citations[0]["group_id"] == "group-42"
        assert citations[0]["page_range"] == [15, 20]

    def test_多个意群_每个映射与输入匹配(self):
        """多个意群时，每个 citation 的字段应与对应输入意群匹配"""
        builder = ContextBuilder()
        test_data = [
            ("group-a", (1, 5)),
            ("group-b", (6, 10)),
            ("group-c", (11, 15)),
            ("group-d", (16, 20)),
        ]
        groups = [
            _make_group(group_id=gid, page_range=pr)
            for gid, pr in test_data
        ]
        selections = [
            {"group": g, "granularity": "digest", "tokens": 50}
            for g in groups
        ]

        _, citations = builder.build_context(selections)

        assert len(citations) == len(test_data)
        for i, (expected_gid, expected_pr) in enumerate(test_data):
            assert citations[i]["group_id"] == expected_gid
            assert citations[i]["page_range"] == list(expected_pr)

    def test_page_range为元组时转换为列表(self):
        """page_range 在 SemanticGroup 中为元组，在 citations 中应转换为列表"""
        builder = ContextBuilder()
        group = _make_group(group_id="group-0", page_range=(3, 7))
        selections = [{"group": group, "granularity": "full", "tokens": 100}]

        _, citations = builder.build_context(selections)

        # 验证 page_range 是列表类型
        assert isinstance(citations[0]["page_range"], list)
        assert citations[0]["page_range"] == [3, 7]


class TestBuildCitationPrompt:
    """测试 build_citation_prompt 方法（需求 9.1）

    验证引文指示提示词的生成逻辑。
    """

    def test_空citations返回空字符串(self):
        """空的 citations 列表应返回空字符串"""
        builder = ContextBuilder()
        prompt = builder.build_citation_prompt([])
        assert prompt == ""

    def test_单个引用_提示词格式正确(self):
        """单个引用时，提示词应包含引用编号和来源信息"""
        builder = ContextBuilder()
        citations = [
            {"ref": 1, "group_id": "group-0", "page_range": [1, 3]},
        ]

        prompt = builder.build_citation_prompt(citations)

        # 验证提示词非空
        assert prompt != ""
        # 验证包含引用编号 [1]
        assert "[1]" in prompt
        # 验证包含来源信息
        assert "group-0" in prompt
        # 验证包含页码信息
        assert "1-3" in prompt
        # 验证包含引文使用指示
        assert "引用" in prompt

    def test_多个引用_提示词包含所有编号和来源(self):
        """多个引用时，提示词应包含所有引用编号和来源信息"""
        builder = ContextBuilder()
        citations = [
            {"ref": 1, "group_id": "group-0", "page_range": [1, 3]},
            {"ref": 2, "group_id": "group-1", "page_range": [4, 6]},
            {"ref": 3, "group_id": "group-2", "page_range": [7, 10]},
        ]

        prompt = builder.build_citation_prompt(citations)

        # 验证包含所有引用编号
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        # 验证包含所有来源标识
        assert "group-0" in prompt
        assert "group-1" in prompt
        assert "group-2" in prompt
        # 验证包含所有页码范围
        assert "1-3" in prompt
        assert "4-6" in prompt
        assert "7-10" in prompt

    def test_提示词包含引文使用说明(self):
        """提示词应包含指导 LLM 使用引用编号的说明"""
        builder = ContextBuilder()
        citations = [
            {"ref": 1, "group_id": "group-0", "page_range": [1, 5]},
        ]

        prompt = builder.build_citation_prompt(citations)

        # 验证包含引用格式说明（如 [1]、[2] 等格式）
        assert "[1]" in prompt
        # 验证包含引用使用指示关键词
        assert "引用" in prompt or "标注" in prompt


class TestMultiGroupCitationMapping:
    """测试多意群引文映射（需求 9.1, 9.2）

    验证多个意群时 citations 列表长度和内容正确。
    """

    def test_citations长度等于selections长度(self):
        """citations 列表长度应等于 selections 列表长度"""
        builder = ContextBuilder()
        n = 7
        groups = [
            _make_group(group_id=f"group-{i}", page_range=(i, i + 2))
            for i in range(n)
        ]
        selections = [
            {"group": g, "granularity": "summary", "tokens": 20}
            for g in groups
        ]

        _, citations = builder.build_context(selections)

        assert len(citations) == n

    def test_混合粒度不影响citations生成(self):
        """不同粒度的意群应都正确生成 citation 映射"""
        builder = ContextBuilder()
        groups = [
            _make_group(group_id="group-0", page_range=(1, 3)),
            _make_group(group_id="group-1", page_range=(4, 6)),
            _make_group(group_id="group-2", page_range=(7, 9)),
        ]
        granularities = ["full", "digest", "summary"]
        selections = [
            {"group": g, "granularity": gran, "tokens": 50}
            for g, gran in zip(groups, granularities)
        ]

        _, citations = builder.build_context(selections)

        # 验证所有意群都有对应的 citation
        assert len(citations) == 3
        # 验证 ref 编号递增
        assert [c["ref"] for c in citations] == [1, 2, 3]
        # 验证 group_id 映射正确
        assert [c["group_id"] for c in citations] == [
            "group-0", "group-1", "group-2"
        ]
        # 验证 page_range 映射正确
        assert [c["page_range"] for c in citations] == [
            [1, 3], [4, 6], [7, 9]
        ]

    def test_大量意群_citations完整性(self):
        """大量意群时，citations 列表应完整且编号正确"""
        builder = ContextBuilder()
        n = 20
        groups = [
            _make_group(
                group_id=f"group-{i}",
                page_range=(i * 5 + 1, (i + 1) * 5),
            )
            for i in range(n)
        ]
        selections = [
            {"group": g, "granularity": "summary", "tokens": 10}
            for g in groups
        ]

        _, citations = builder.build_context(selections)

        # 验证长度
        assert len(citations) == n
        # 验证编号从 1 到 n 连续递增
        refs = [c["ref"] for c in citations]
        assert refs == list(range(1, n + 1))
        # 验证每个 group_id 正确
        for i, citation in enumerate(citations):
            assert citation["group_id"] == f"group-{i}"
            assert citation["page_range"] == [i * 5 + 1, (i + 1) * 5]
