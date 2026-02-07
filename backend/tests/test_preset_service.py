"""
预设问题服务单元测试

测试 preset_service.py 中的预设问题列表、系统提示词常量和 get_generation_prompt 函数。
"""

import pytest

from services.preset_service import (
    PRESET_QUESTIONS,
    MERMAID_SYSTEM_PROMPT,
    MINDMAP_SYSTEM_PROMPT,
    get_generation_prompt,
)


class TestPresetQuestions:
    """测试预设问题列表"""

    def test_预设问题列表包含5个问题(self):
        """验证预设问题列表包含正确数量的问题"""
        assert len(PRESET_QUESTIONS) == 5

    def test_每个预设问题包含必要字段(self):
        """验证每个预设问题都包含 id、label 和 query 字段"""
        for question in PRESET_QUESTIONS:
            assert "id" in question, f"预设问题缺少 id 字段: {question}"
            assert "label" in question, f"预设问题缺少 label 字段: {question}"
            assert "query" in question, f"预设问题缺少 query 字段: {question}"

    def test_预设问题id唯一(self):
        """验证所有预设问题的 id 互不相同"""
        ids = [q["id"] for q in PRESET_QUESTIONS]
        assert len(ids) == len(set(ids)), f"预设问题 id 存在重复: {ids}"

    def test_预设问题包含指定项目(self):
        """验证预设问题列表包含需求中指定的所有项目"""
        ids = {q["id"] for q in PRESET_QUESTIONS}
        expected_ids = {"summarize", "formulas", "methods", "mindmap", "flowchart"}
        assert ids == expected_ids, f"预设问题 id 集合不匹配: {ids} != {expected_ids}"


class TestSystemPrompts:
    """测试系统提示词常量"""

    def test_mermaid提示词非空(self):
        """验证 Mermaid 流程图系统提示词非空"""
        assert len(MERMAID_SYSTEM_PROMPT) > 0

    def test_mindmap提示词非空(self):
        """验证思维导图系统提示词非空"""
        assert len(MINDMAP_SYSTEM_PROMPT) > 0

    def test_mermaid提示词包含mermaid关键内容(self):
        """验证 Mermaid 提示词包含 Mermaid 语法相关内容"""
        assert "mermaid" in MERMAID_SYSTEM_PROMPT.lower() or "Mermaid" in MERMAID_SYSTEM_PROMPT

    def test_mindmap提示词包含思维导图关键内容(self):
        """验证思维导图提示词包含思维导图相关内容"""
        assert "思维导图" in MINDMAP_SYSTEM_PROMPT


class TestGetGenerationPrompt:
    """测试 get_generation_prompt 函数"""

    def test_思维导图查询返回mindmap提示词(self):
        """包含"思维导图"关键词的查询应返回思维导图提示词"""
        result = get_generation_prompt("生成思维导图")
        assert result == MINDMAP_SYSTEM_PROMPT

    def test_流程图查询返回mermaid提示词(self):
        """包含"流程图"关键词的查询应返回 Mermaid 提示词"""
        result = get_generation_prompt("生成流程图")
        assert result == MERMAID_SYSTEM_PROMPT

    def test_普通查询返回None(self):
        """不包含生成类关键词的查询应返回 None"""
        result = get_generation_prompt("请总结本文的主要内容")
        assert result is None

    def test_空查询返回None(self):
        """空字符串查询应返回 None"""
        result = get_generation_prompt("")
        assert result is None

    def test_英文mindmap关键词匹配(self):
        """英文 mindmap 关键词也应匹配"""
        result = get_generation_prompt("generate a mindmap")
        assert result == MINDMAP_SYSTEM_PROMPT

    def test_英文flowchart关键词匹配(self):
        """英文 flowchart 关键词也应匹配"""
        result = get_generation_prompt("create a flowchart")
        assert result == MERMAID_SYSTEM_PROMPT

    def test_大小写不敏感匹配(self):
        """关键词匹配应不区分大小写"""
        result = get_generation_prompt("Generate a MINDMAP")
        assert result == MINDMAP_SYSTEM_PROMPT

    def test_脑图关键词匹配(self):
        """包含"脑图"关键词的查询应返回思维导图提示词"""
        result = get_generation_prompt("帮我画一个脑图")
        assert result == MINDMAP_SYSTEM_PROMPT

    def test_mermaid关键词匹配(self):
        """包含"mermaid"关键词的查询应返回 Mermaid 提示词"""
        result = get_generation_prompt("用mermaid画图")
        assert result == MERMAID_SYSTEM_PROMPT
