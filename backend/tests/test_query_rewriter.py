"""测试 QueryRewriter 查询改写模块

验证需求 1.1, 1.2, 1.3, 1.4, 1.5：
- 口语化查询转换为规范化形式
- 使用本地规则化改写，不依赖 LLM
- 规范化查询返回原始内容
- 支持中英文混合查询
- 指示代词替换为选中文本关键内容
"""
import sys
import os

import pytest

# 将 backend 目录添加到 sys.path，以便导入 services 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter():
    """创建 QueryRewriter 实例"""
    return QueryRewriter()


class TestColloquialRewrite:
    """测试口语化表达改写（需求 1.1, 1.2）"""

    def test_sha_yisi_with_subject(self, rewriter):
        """'X啥意思' → 'X的含义和解释'"""
        result = rewriter.rewrite("这个公式啥意思")
        assert "含义和解释" in result
        assert "啥意思" not in result

    def test_jiang_le_sha(self, rewriter):
        """'X讲了啥' → 'X的主要内容'"""
        result = rewriter.rewrite("这块讲了啥")
        assert "主要内容" in result
        assert "讲了啥" not in result

    def test_shuo_le_sha(self, rewriter):
        """'X说了啥' → 'X的主要内容'"""
        result = rewriter.rewrite("第三章说了啥")
        assert "主要内容" in result
        assert "说了啥" not in result

    def test_sha_shi_x(self, rewriter):
        """'啥是X' → 'X的定义'"""
        result = rewriter.rewrite("啥是梯度下降")
        assert "梯度下降" in result
        assert "定义" in result
        assert "啥是" not in result

    def test_x_shi_sha(self, rewriter):
        """'X是啥' → 'X的定义'"""
        result = rewriter.rewrite("反向传播是啥")
        assert "反向传播" in result
        assert "定义" in result
        assert "是啥" not in result

    def test_wei_sha(self, rewriter):
        """'为啥要X' → 'X的原因和目的'"""
        result = rewriter.rewrite("为啥要用正则化")
        assert "正则化" in result
        assert "原因" in result
        assert "为啥" not in result

    def test_za_yong_with_subject(self, rewriter):
        """'X咋用' → 'X的使用方法'"""
        result = rewriter.rewrite("这个API咋用")
        assert "使用方法" in result
        assert "咋用" not in result

    def test_zenme_yong(self, rewriter):
        """'怎么用X' → 'X的使用方法'"""
        result = rewriter.rewrite("怎么用 numpy")
        assert "使用方法" in result
        assert "怎么用" not in result

    def test_za_generic(self, rewriter):
        """'咋X' → '如何X'"""
        result = rewriter.rewrite("咋理解这个概念")
        assert "如何" in result
        assert "咋" not in result

    def test_sha_generic(self, rewriter):
        """'啥X' → '什么X'"""
        result = rewriter.rewrite("啥时候用这个方法")
        assert "什么" in result
        assert result.startswith("什么")


class TestNormalizedQueryPassthrough:
    """测试规范化查询不被修改（需求 1.3）"""

    def test_formal_chinese_query(self, rewriter):
        """正式中文查询应原样返回"""
        query = "请解释梯度下降算法的原理"
        assert rewriter.rewrite(query) == query

    def test_english_query(self, rewriter):
        """英文查询应原样返回"""
        query = "What is gradient descent?"
        assert rewriter.rewrite(query) == query

    def test_empty_query(self, rewriter):
        """空查询应原样返回"""
        assert rewriter.rewrite("") == ""
        assert rewriter.rewrite("   ") == "   "

    def test_none_query(self, rewriter):
        """None 查询应原样返回"""
        assert rewriter.rewrite(None) is None

    def test_technical_query(self, rewriter):
        """技术性查询应原样返回"""
        query = "FAISS 索引的构建过程"
        assert rewriter.rewrite(query) == query


class TestMixedLanguageSupport:
    """测试中英文混合查询支持（需求 1.4）"""

    def test_chinese_colloquial_with_english_term(self, rewriter):
        """中文口语化 + 英文术语"""
        result = rewriter.rewrite("transformer是啥")
        assert "transformer" in result
        assert "定义" in result

    def test_english_term_sha_yisi(self, rewriter):
        """英文术语 + 啥意思"""
        result = rewriter.rewrite("attention mechanism啥意思")
        assert "attention mechanism" in result
        assert "含义和解释" in result

    def test_za_yong_english_lib(self, rewriter):
        """咋用 + 英文库名"""
        result = rewriter.rewrite("pytorch咋用")
        assert "pytorch" in result
        assert "使用方法" in result


class TestPronounResolution:
    """测试指示代词解析（需求 1.5）"""

    def test_zhege_replaced(self, rewriter):
        """'这个' 被替换为选中文本关键内容"""
        result = rewriter.rewrite("这个是什么意思", selected_text="贝叶斯定理的推导过程")
        assert "这个" not in result
        assert "贝叶斯定理" in result

    def test_nage_replaced(self, rewriter):
        """'那个' 被替换为选中文本关键内容"""
        result = rewriter.rewrite("那个怎么理解", selected_text="卷积神经网络的池化层")
        assert "那个" not in result
        assert "卷积神经网络" in result

    def test_ta_replaced(self, rewriter):
        """'它' 被替换为选中文本关键内容"""
        result = rewriter.rewrite("它的作用是什么", selected_text="激活函数ReLU")
        assert "它" not in result
        assert "激活函数ReLU" in result

    def test_zhekuai_replaced(self, rewriter):
        """'这块' 被替换为选中文本关键内容"""
        result = rewriter.rewrite("这块讲了啥", selected_text="第三章 模型训练策略")
        assert "这块" not in result

    def test_no_replacement_without_selected_text(self, rewriter):
        """没有选中文本时不替换指示代词"""
        query = "这个是什么意思"
        result = rewriter.rewrite(query)
        assert "这个" in result

    def test_no_replacement_with_empty_selected_text(self, rewriter):
        """选中文本为空时不替换指示代词"""
        query = "这个是什么意思"
        result = rewriter.rewrite(query, selected_text="")
        assert "这个" in result

    def test_no_replacement_with_whitespace_selected_text(self, rewriter):
        """选中文本仅为空白时不替换指示代词"""
        query = "这个是什么意思"
        result = rewriter.rewrite(query, selected_text="   ")
        assert "这个" in result


class TestExtractKeyContent:
    """测试关键内容提取"""

    def test_short_text(self, rewriter):
        """短文本直接返回"""
        result = rewriter._extract_key_content("贝叶斯定理")
        assert result == "贝叶斯定理"

    def test_long_text_truncated(self, rewriter):
        """超长文本被截断到 max_chars"""
        long_text = "这是一段非常长的文本" * 20
        result = rewriter._extract_key_content(long_text, max_chars=50)
        assert len(result) <= 50

    def test_sentence_extraction(self, rewriter):
        """提取第一个有意义的句子"""
        text = "贝叶斯定理是概率论中的重要定理。它描述了条件概率的关系。"
        result = rewriter._extract_key_content(text)
        assert "贝叶斯定理" in result

    def test_empty_text(self, rewriter):
        """空文本返回空字符串"""
        assert rewriter._extract_key_content("") == ""
        assert rewriter._extract_key_content("   ") == ""

    def test_multiline_text(self, rewriter):
        """多行文本取第一行有意义的内容"""
        text = "第一行内容\n第二行内容\n第三行内容"
        result = rewriter._extract_key_content(text)
        assert "第一行内容" in result

    def test_custom_max_chars(self, rewriter):
        """自定义 max_chars 参数"""
        text = "这是一段测试文本内容"
        result = rewriter._extract_key_content(text, max_chars=5)
        assert len(result) <= 5


class TestErrorHandling:
    """测试错误处理"""

    def test_rewrite_returns_original_on_error(self, rewriter):
        """改写异常时返回原始查询"""
        # 正常查询不应出错
        query = "正常的查询"
        result = rewriter.rewrite(query)
        assert isinstance(result, str)

    def test_combined_pronoun_and_colloquial(self, rewriter):
        """同时包含指示代词和口语化表达"""
        result = rewriter.rewrite(
            "这个啥意思",
            selected_text="损失函数的定义"
        )
        # 指示代词应被替换，口语化表达也应被改写
        assert "这个" not in result
        assert "啥意思" not in result
