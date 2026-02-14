"""
KeywordExtractor 单元测试

测试关键词提取、频率更新和关注领域识别功能
"""
import pytest

from services.keyword_extractor import KeywordExtractor, STOP_WORDS


@pytest.fixture
def extractor():
    return KeywordExtractor()


class TestExtractKeywords:
    """测试 extract_keywords 方法"""

    def test_empty_query(self, extractor):
        """空查询返回空列表"""
        assert extractor.extract_keywords("") == []
        assert extractor.extract_keywords("   ") == []
        assert extractor.extract_keywords(None) == []

    def test_chinese_query(self, extractor):
        """中文查询能正确提取关键词"""
        keywords = extractor.extract_keywords("机器学习和深度学习的区别")
        assert len(keywords) > 0
        # 所有关键词长度 >= 2
        for kw in keywords:
            assert len(kw) >= 2

    def test_english_query(self, extractor):
        """英文查询能正确提取关键词"""
        keywords = extractor.extract_keywords("transformer attention mechanism")
        assert "transformer" in keywords
        assert "attention" in keywords
        assert "mechanism" in keywords

    def test_mixed_query(self, extractor):
        """中英文混合查询"""
        keywords = extractor.extract_keywords("transformer模型的attention机制")
        assert "transformer" in keywords
        assert "attention" in keywords

    def test_filters_stop_words(self, extractor):
        """过滤停用词"""
        keywords = extractor.extract_keywords("what is the transformer model")
        # "what", "is", "the" 是停用词，不应出现
        for kw in keywords:
            assert kw not in STOP_WORDS

    def test_filters_short_tokens(self, extractor):
        """过滤长度 < 2 的 token"""
        keywords = extractor.extract_keywords("I am a test")
        for kw in keywords:
            assert len(kw) >= 2

    def test_deduplication(self, extractor):
        """关键词去重"""
        keywords = extractor.extract_keywords("transformer transformer transformer")
        assert keywords.count("transformer") == 1

    def test_keywords_are_lowercase_substrings(self, extractor):
        """提取的关键词是原文的子串（小写化后）"""
        query = "Transformer Attention Mechanism"
        keywords = extractor.extract_keywords(query)
        query_lower = query.lower()
        for kw in keywords:
            assert kw.lower() in query_lower


class TestUpdateFrequency:
    """测试 update_frequency 方法"""

    def test_new_keywords(self, extractor):
        """新关键词初始频率为 1"""
        profile = {"keyword_frequencies": {}}
        result = extractor.update_frequency(profile, ["机器学习", "深度学习"])
        assert result["keyword_frequencies"]["机器学习"] == 1
        assert result["keyword_frequencies"]["深度学习"] == 1

    def test_increment_existing(self, extractor):
        """已有关键词频率递增"""
        profile = {"keyword_frequencies": {"机器学习": 2}}
        result = extractor.update_frequency(profile, ["机器学习"])
        assert result["keyword_frequencies"]["机器学习"] == 3

    def test_mixed_new_and_existing(self, extractor):
        """混合新旧关键词"""
        profile = {"keyword_frequencies": {"机器学习": 1}}
        result = extractor.update_frequency(profile, ["机器学习", "transformer"])
        assert result["keyword_frequencies"]["机器学习"] == 2
        assert result["keyword_frequencies"]["transformer"] == 1

    def test_empty_keywords(self, extractor):
        """空关键词列表不改变频率"""
        profile = {"keyword_frequencies": {"机器学习": 2}}
        result = extractor.update_frequency(profile, [])
        assert result["keyword_frequencies"]["机器学习"] == 2

    def test_missing_keyword_frequencies_field(self, extractor):
        """profile 缺少 keyword_frequencies 字段时自动创建"""
        profile = {}
        result = extractor.update_frequency(profile, ["test"])
        assert "keyword_frequencies" in result
        assert result["keyword_frequencies"]["test"] == 1

    def test_updates_timestamp(self, extractor):
        """更新后设置 updated_at 时间戳"""
        profile = {"keyword_frequencies": {}}
        result = extractor.update_frequency(profile, ["test"])
        assert "updated_at" in result


class TestGetFocusAreas:
    """测试 get_focus_areas 方法"""

    def test_default_threshold(self, extractor):
        """默认阈值 3：频率 >= 3 的关键词被识别为关注领域"""
        profile = {
            "keyword_frequencies": {
                "机器学习": 5,
                "transformer": 3,
                "cnn": 2,
                "rnn": 1,
            }
        }
        focus = extractor.get_focus_areas(profile)
        assert "机器学习" in focus
        assert "transformer" in focus
        assert "cnn" not in focus
        assert "rnn" not in focus

    def test_custom_threshold(self, extractor):
        """自定义阈值"""
        profile = {
            "keyword_frequencies": {
                "机器学习": 5,
                "transformer": 3,
                "cnn": 2,
            }
        }
        focus = extractor.get_focus_areas(profile, threshold=2)
        assert "机器学习" in focus
        assert "transformer" in focus
        assert "cnn" in focus

    def test_sorted_by_frequency(self, extractor):
        """结果按频率降序排列"""
        profile = {
            "keyword_frequencies": {
                "cnn": 3,
                "机器学习": 10,
                "transformer": 5,
            }
        }
        focus = extractor.get_focus_areas(profile, threshold=3)
        assert focus == ["机器学习", "transformer", "cnn"]

    def test_empty_frequencies(self, extractor):
        """空频率表返回空列表"""
        profile = {"keyword_frequencies": {}}
        assert extractor.get_focus_areas(profile) == []

    def test_missing_keyword_frequencies(self, extractor):
        """缺少 keyword_frequencies 字段返回空列表"""
        profile = {}
        assert extractor.get_focus_areas(profile) == []

    def test_threshold_boundary(self, extractor):
        """阈值边界：恰好等于阈值的关键词应被包含"""
        profile = {"keyword_frequencies": {"test": 3}}
        assert "test" in extractor.get_focus_areas(profile, threshold=3)
        assert "test" not in extractor.get_focus_areas(profile, threshold=4)


# ============================================================
# 属性测试（Property-Based Testing）
# ============================================================
from hypothesis import given, settings
from hypothesis import strategies as st


class TestPropertyKeywordFrequencyThreshold:
    """
    Feature: chatpdf-memory-system, Property 3: 关键词频率阈值触发关注领域

    **Validates: Requirements 2.1, 2.3**

    对于任意关键词和阈值 T，如果该关键词被更新 N 次（N >= T），
    则应出现在 get_focus_areas 的返回结果中；如果 N < T，则不应出现。
    """

    @given(
        keyword=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=2,
            max_size=20,
        ),
        n=st.integers(min_value=0, max_value=50),
        threshold=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_keyword_frequency_threshold_triggers_focus_area(self, keyword, n, threshold):
        """
        属性测试：关键词更新 N 次后，根据阈值 T 判断是否出现在关注领域中

        - N >= T 时，关键词应出现在 get_focus_areas 结果中
        - N < T 时，关键词不应出现在 get_focus_areas 结果中
        """
        extractor = KeywordExtractor()
        profile = {"keyword_frequencies": {}}

        # 将关键词更新 N 次（每次调用 update_frequency 频率 +1）
        for _ in range(n):
            profile = extractor.update_frequency(profile, [keyword])

        # 验证频率值正确
        if n > 0:
            assert profile["keyword_frequencies"].get(keyword) == n

        # 获取关注领域
        focus_areas = extractor.get_focus_areas(profile, threshold=threshold)

        # 核心属性：N >= T 时应出现，N < T 时不应出现
        if n >= threshold:
            assert keyword in focus_areas, (
                f"关键词 '{keyword}' 更新 {n} 次后应出现在关注领域中（阈值={threshold}）"
            )
        else:
            assert keyword not in focus_areas, (
                f"关键词 '{keyword}' 仅更新 {n} 次不应出现在关注领域中（阈值={threshold}）"
            )

class TestPropertyKeywordSubstring:
    """
    Feature: chatpdf-memory-system, Property 4: 关键词提取结果为原文子串

    **Validates: Requirements 2.2**

    对于任意非空查询文本，extract_keywords(query) 返回的每个关键词
    （转为小写后）应是原始查询文本（转为小写后）的子串。
    """

    @given(
        query=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_property_4_keywords_are_substrings_of_query(self, query):
        """
        属性测试：提取的每个关键词（小写）都是原始查询文本（小写）的子串

        使用 Hypothesis 生成包含字母、数字、标点和空格的随机文本，
        验证 _tokenize 产生的 token 始终来源于原文。
        """
        extractor = KeywordExtractor()
        keywords = extractor.extract_keywords(query)
        query_lower = query.lower()

        for kw in keywords:
            assert kw.lower() in query_lower, (
                f"关键词 '{kw}' 不是查询文本 '{query}' 的子串（小写比较）"
            )

