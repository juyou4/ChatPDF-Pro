"""
高级搜索服务单元测试

测试 AdvancedSearchService 的正则表达式搜索和布尔逻辑搜索功能。
"""

import pytest

from services.advanced_search import AdvancedSearchService


class TestRegexSearch:
    """正则表达式搜索测试"""

    def setup_method(self):
        """每个测试方法前初始化服务实例"""
        self.service = AdvancedSearchService()

    def test_基本正则匹配(self):
        """测试基本的正则表达式匹配"""
        text = "Python 3.10 和 Python 3.11 是常用版本"
        results = self.service.regex_search(r"Python \d+\.\d+", text)

        assert len(results) == 2
        assert results[0]["match_text"] == "Python 3.10"
        assert results[1]["match_text"] == "Python 3.11"

    def test_匹配结果包含正确的偏移量(self):
        """测试匹配结果的 match_offset 正确"""
        text = "hello world hello"
        results = self.service.regex_search(r"hello", text)

        assert len(results) == 2
        assert results[0]["match_offset"] == 0
        assert results[1]["match_offset"] == 12

    def test_匹配结果包含上下文片段(self):
        """测试匹配结果包含上下文片段"""
        text = "前缀文本" * 100 + "目标关键词" + "后缀文本" * 100
        results = self.service.regex_search(r"目标关键词", text, context_chars=50)

        assert len(results) == 1
        snippet = results[0]["context_snippet"]
        assert "目标关键词" in snippet
        # 上下文片段应包含前后文本
        assert len(snippet) > len("目标关键词")

    def test_匹配结果score固定为1(self):
        """测试正则搜索的 score 固定为 1.0"""
        text = "test pattern here"
        results = self.service.regex_search(r"pattern", text)

        assert len(results) == 1
        assert results[0]["score"] == 1.0

    def test_无效正则抛出ValueError(self):
        """测试无效正则表达式语法抛出 ValueError"""
        with pytest.raises(ValueError, match="正则表达式语法错误"):
            self.service.regex_search(r"[invalid", "some text")

    def test_无效正则_未闭合括号(self):
        """测试未闭合括号的正则表达式"""
        with pytest.raises(ValueError):
            self.service.regex_search(r"(unclosed", "some text")

    def test_空模式返回空列表(self):
        """测试空模式返回空列表"""
        results = self.service.regex_search("", "some text")
        assert results == []

    def test_空文本返回空列表(self):
        """测试空文本返回空列表"""
        results = self.service.regex_search(r"pattern", "")
        assert results == []

    def test_无匹配返回空列表(self):
        """测试无匹配时返回空列表"""
        results = self.service.regex_search(r"不存在的模式", "这是一段普通文本")
        assert results == []

    def test_limit参数限制结果数量(self):
        """测试 limit 参数限制返回结果数量"""
        text = "a " * 100
        results = self.service.regex_search(r"a", text, limit=5)
        assert len(results) == 5

    def test_大小写不敏感匹配(self):
        """测试正则搜索默认大小写不敏感"""
        text = "Hello HELLO hello"
        results = self.service.regex_search(r"hello", text)
        assert len(results) == 3

    def test_中文正则匹配(self):
        """测试中文正则表达式匹配"""
        text = "深度学习和机器学习是人工智能的分支"
        # 使用非贪婪匹配，确保分别匹配"深度学习"和"机器学习"
        results = self.service.regex_search(r"[\u4e00-\u9fff]{2}学习", text)

        assert len(results) == 2
        assert results[0]["match_text"] == "深度学习"
        assert results[1]["match_text"] == "机器学习"


class TestBooleanSearch:
    """布尔逻辑搜索测试"""

    def setup_method(self):
        """每个测试方法前初始化服务实例"""
        self.service = AdvancedSearchService()

    def test_单词项搜索(self):
        """测试单个词项的搜索"""
        text = "深度学习是一种机器学习方法"
        results = self.service.boolean_search("深度学习", text)

        assert len(results) >= 1
        assert results[0]["match_text"] == "深度学习"

    def test_AND搜索_两个词项都存在(self):
        """测试 AND 搜索：两个词项都在窗口范围内"""
        text = "深度学习和卷积神经网络是计算机视觉的核心技术"
        results = self.service.boolean_search("深度学习 AND 卷积", text)

        assert len(results) >= 1
        # 匹配文本应包含两个词项之间的内容
        assert "深度学习" in text[results[0]["match_offset"]:results[0]["match_offset"] + len(results[0]["match_text"])] or \
               "卷积" in text[results[0]["match_offset"]:results[0]["match_offset"] + len(results[0]["match_text"])]

    def test_AND搜索_词项不在窗口内返回空(self):
        """测试 AND 搜索：词项距离超出窗口范围时返回空"""
        # 构造两个词项距离超过 500 字符的文本
        text = "深度学习" + "x" * 600 + "卷积网络"
        results = self.service.boolean_search("深度学习 AND 卷积网络", text)
        assert len(results) == 0

    def test_OR搜索(self):
        """测试 OR 搜索：任一词项出现即可"""
        text = "深度学习是一种方法。另外，卷积网络也很重要。"
        results = self.service.boolean_search("深度学习 OR 卷积网络", text)

        assert len(results) >= 1

    def test_NOT搜索_排除词项(self):
        """测试 NOT 搜索：排除包含指定词项的结果"""
        # 构造两个"深度学习"出现位置，第一个靠近"图像"（在窗口内），
        # 第二个远离"图像"（超出 500 字符窗口）
        text = "深度学习和图像识别。" + "x" * 600 + "深度学习和文本分析。"
        results = self.service.boolean_search("深度学习 NOT 图像", text)

        # 第一个"深度学习"靠近"图像"应被排除，第二个应保留
        assert len(results) >= 1
        # 保留的结果应是远离"图像"的那个
        for result in results:
            assert result["match_offset"] > 100  # 应是第二个"深度学习"

    def test_结果按score降序排列(self):
        """测试布尔搜索结果按 score 降序排列"""
        text = "深度学习和卷积网络。深度学习方法。深度学习和卷积网络以及循环网络。"
        results = self.service.boolean_search(
            "深度学习 OR 卷积网络", text
        )

        # 验证结果按 score 降序排列
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"], (
                f"结果未按 score 降序排列: "
                f"results[{i}].score={results[i]['score']} < "
                f"results[{i+1}].score={results[i+1]['score']}"
            )

    def test_结果包含上下文片段(self):
        """测试布尔搜索结果包含上下文片段"""
        text = "前缀" * 50 + "深度学习是核心技术" + "后缀" * 50
        results = self.service.boolean_search("深度学习", text, context_chars=30)

        assert len(results) >= 1
        snippet = results[0]["context_snippet"]
        assert "深度学习" in snippet

    def test_空查询返回空列表(self):
        """测试空查询返回空列表"""
        results = self.service.boolean_search("", "some text")
        assert results == []

    def test_空文本返回空列表(self):
        """测试空文本返回空列表"""
        results = self.service.boolean_search("query", "")
        assert results == []

    def test_无匹配返回空列表(self):
        """测试无匹配时返回空列表"""
        results = self.service.boolean_search("不存在的词", "这是一段普通文本")
        assert results == []

    def test_limit参数限制结果数量(self):
        """测试 limit 参数限制返回结果数量"""
        text = "关键词 " * 100
        results = self.service.boolean_search("关键词", text, limit=3)
        assert len(results) <= 3

    def test_大小写不敏感(self):
        """测试布尔搜索大小写不敏感"""
        text = "Deep Learning and Machine Learning"
        results = self.service.boolean_search("deep AND learning", text)
        assert len(results) >= 1


class TestParseBooleanQuery:
    """布尔查询解析测试"""

    def setup_method(self):
        self.service = AdvancedSearchService()

    def test_解析AND查询(self):
        """测试解析 AND 查询"""
        must, should, not_terms = self.service._parse_boolean_query(
            "深度学习 AND 卷积网络"
        )
        assert "深度学习" in must
        assert "卷积网络" in must
        assert len(should) == 0
        assert len(not_terms) == 0

    def test_解析OR查询(self):
        """测试解析 OR 查询"""
        must, should, not_terms = self.service._parse_boolean_query(
            "深度学习 OR 卷积网络"
        )
        # 第一个词项作为 must，第二个作为 should
        assert "深度学习" in must
        assert "卷积网络" in should

    def test_解析NOT查询(self):
        """测试解析 NOT 查询"""
        must, should, not_terms = self.service._parse_boolean_query(
            "深度学习 NOT 图像"
        )
        assert "深度学习" in must
        assert "图像" in not_terms

    def test_解析复合查询(self):
        """测试解析复合布尔查询"""
        must, should, not_terms = self.service._parse_boolean_query(
            "CNN AND 对比 OR 分析 NOT 图像"
        )
        assert "CNN" in must
        assert "对比" in must
        assert "分析" in should
        assert "图像" in not_terms
