"""引用相关性优化 - 属性测试

Feature: chatpdf-citation-relevance

使用 Hypothesis 进行属性测试，验证引用相关性优化各模块的正确性。
"""
import sys
import os

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, strategies as st, settings, assume
from services.selected_text_locator import locate_selected_text


# ============================================================
# Hypothesis 策略：生成文档页面数据
# ============================================================

@st.composite
def pages_with_substring(draw):
    """生成文档页面列表和从某页 content 中提取的子串

    返回 (pages, selected_text, expected_page_num)：
    - pages: 至少 1 页的文档页面数据列表
    - selected_text: 从 target_page 的 content 中提取的非空子串
    - expected_page_num: 该子串所在页的页码
    """
    # 生成 1~5 页，每页有唯一前缀确保内容不重复
    num_pages = draw(st.integers(min_value=1, max_value=5))
    pages = []
    for i in range(num_pages):
        page_num = i + 1
        # 每页内容带唯一标识前缀，避免跨页误匹配
        unique_prefix = f"【第{page_num}页独有】"
        body = draw(st.text(
            alphabet=st.characters(categories=("L", "N", "P")),
            min_size=10,
            max_size=200,
        ))
        content = unique_prefix + body
        pages.append({"page": page_num, "content": content})

    # 随机选择一页作为目标页
    target_idx = draw(st.integers(min_value=0, max_value=num_pages - 1))
    target_page = pages[target_idx]
    target_content = target_page["content"]

    # 从目标页 content 中提取一个非空子串
    start = draw(st.integers(min_value=0, max_value=len(target_content) - 1))
    # 子串至少 1 个字符
    end = draw(st.integers(min_value=start + 1, max_value=len(target_content)))
    selected_text = target_content[start:end]

    # 确保子串非空且不是纯空白
    assume(selected_text.strip())

    # 确保子串只在目标页出现（排除跨页匹配干扰）
    for idx, page in enumerate(pages):
        if idx != target_idx:
            assume(selected_text not in page["content"])

    return pages, selected_text, target_page["page"]


# ============================================================
# Property 7: 页码定位正确性
# Feature: chatpdf-citation-relevance, Property 7: 页码定位正确性
# **Validates: Requirements 5.1**
# ============================================================

class TestProperty7PageLocationCorrectness:
    """Property 7: 页码定位正确性

    对于任意文档页面数据列表和从某页 content 中提取的子串作为 selected_text，
    调用 locate_selected_text(selected_text, pages) 返回的 page_start
    应等于该页的页码。

    **Validates: Requirements 5.1**
    """

    @given(data=pages_with_substring())
    @settings(max_examples=100, deadline=None)
    def test_property_7_page_start_matches_source_page(self, data):
        """属性：定位结果的 page_start 等于子串来源页的页码"""
        pages, selected_text, expected_page_num = data

        result = locate_selected_text(selected_text, pages)

        assert result["page_start"] == expected_page_num, (
            f"页码定位不正确！\n"
            f"selected_text={selected_text!r}\n"
            f"期望 page_start={expected_page_num}\n"
            f"实际结果={result}\n"
            f"页面内容={[(p['page'], p['content'][:50]) for p in pages]}"
        )


from services.query_rewriter import QueryRewriter


# ============================================================
# Hypothesis 策略：QueryRewriter 测试用生成器
# ============================================================

# QueryRewriter 实例（无状态，可复用）
_rewriter = QueryRewriter()

# 指示代词列表（与 QueryRewriter.PRONOUN_PATTERNS 保持一致）
_PRONOUNS = QueryRewriter.PRONOUN_PATTERNS

# 口语化关键字，生成 Property 3 查询时需要排除
_COLLOQUIAL_KEYWORDS = ['啥', '咋', '为啥', '怎么用']


@st.composite
def query_with_pronoun(draw):
    """生成包含至少一个指示代词的查询字符串

    策略：
    1. 生成一个不含代词的基础查询前缀
    2. 随机选择一个代词
    3. 生成一个不含代词的基础查询后缀
    4. 拼接为完整查询

    返回 (query, pronoun)
    """
    # 使用中文字符作为基础文本，避免意外包含代词
    prefix = draw(st.text(
        alphabet=st.sampled_from('文档内容分析研究方法数据结果说明'),
        min_size=2, max_size=10,
    ))
    suffix = draw(st.text(
        alphabet=st.sampled_from('是什么含义解释原因目的作用'),
        min_size=2, max_size=10,
    ))
    pronoun = draw(st.sampled_from(_PRONOUNS))

    query = f"{prefix}{pronoun}{suffix}"

    # 确保查询确实包含该代词
    assume(pronoun in query)

    return query, pronoun


@st.composite
def meaningful_selected_text(draw, min_size=2, max_size=200):
    """生成有意义的 selected_text

    确保：
    - 非空且非纯空白
    - 至少包含 min_size 个字符的实际文本内容
    - _extract_key_content 能从中提取出非空结果
    - 不包含任何指示代词（避免替换后 key_content 又引入代词）
    """
    # 使用不含代词字符的中文字符集
    text = draw(st.text(
        alphabet=st.sampled_from('机器学习是人工智能的一个分支通过数据训练模型来进行预测和决策'),
        min_size=min_size, max_size=max_size,
    ))
    assume(text.strip())
    assume(len(text.strip()) >= min_size)

    # 确保不包含任何指示代词
    key = _rewriter._extract_key_content(text)
    assume(key)
    for pronoun in _PRONOUNS:
        assume(pronoun not in key)

    return text


@st.composite
def query_without_pronouns_or_colloquial(draw):
    """生成不包含任何指示代词和口语化关键字的查询字符串

    确保查询经过 _replace_colloquial 后不会被改变，
    这样 rewrite 的第三步（语义增强）才会被触发。
    """
    # 使用安全的中文字符集，不包含代词和口语化关键字中的字符
    query = draw(st.text(
        alphabet=st.sampled_from('文档内容分析研究方法数据结果说明定义概念'),
        min_size=2, max_size=30,
    ))
    assume(query.strip())

    # 确保不包含任何指示代词
    for pronoun in _PRONOUNS:
        assume(pronoun not in query)

    # 确保不包含口语化关键字
    for kw in _COLLOQUIAL_KEYWORDS:
        assume(kw not in query)

    # 双重验证：口语化替换不应改变查询
    colloquial_result = _rewriter._replace_colloquial(query)
    assume(colloquial_result == query)

    return query


# ============================================================
# Property 2: 指示代词解析替换
# Feature: chatpdf-citation-relevance, Property 2: 指示代词解析替换
# **Validates: Requirements 2.2**
# ============================================================

class TestProperty2PronounResolution:
    """Property 2: 指示代词解析替换

    对于任意包含至少一个指示代词的查询字符串和任意非空 selected_text，
    调用 QueryRewriter.rewrite(query, selected_text) 后，
    返回的改写查询中不应再包含被替换的指示代词，
    且改写结果应包含 selected_text 的关键内容片段。

    **Validates: Requirements 2.2**
    """

    @given(
        query_data=query_with_pronoun(),
        selected_text=meaningful_selected_text(),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_2_pronoun_replaced_with_key_content(self, query_data, selected_text):
        """属性：代词被替换，且改写结果包含 selected_text 的关键内容"""
        query, pronoun = query_data

        rewritten = _rewriter.rewrite(query, selected_text)

        # 提取 selected_text 的关键内容（与 QueryRewriter 内部逻辑一致）
        key_content = _rewriter._extract_key_content(selected_text)

        # 验证 1：改写后不应再包含原始代词
        assert pronoun not in rewritten, (
            f"改写后仍包含指示代词！\n"
            f"原始查询: {query!r}\n"
            f"代词: {pronoun!r}\n"
            f"改写结果: {rewritten!r}\n"
            f"selected_text: {selected_text!r}"
        )

        # 验证 2：改写结果应包含 selected_text 的关键内容
        assert key_content in rewritten, (
            f"改写结果未包含 selected_text 的关键内容！\n"
            f"原始查询: {query!r}\n"
            f"关键内容: {key_content!r}\n"
            f"改写结果: {rewritten!r}\n"
            f"selected_text: {selected_text!r}"
        )


# ============================================================
# Property 3: 无代词查询的语义增强
# Feature: chatpdf-citation-relevance, Property 3: 无代词查询的语义增强
# **Validates: Requirements 2.3**
# ============================================================

class TestProperty3SemanticAugmentation:
    """Property 3: 无代词查询的语义增强

    对于任意不包含任何指示代词的查询字符串和任意非空 selected_text
    （长度 >= 2 个字符），调用 QueryRewriter.rewrite(query, selected_text) 后，
    返回的改写查询长度应大于原始查询长度。

    **Validates: Requirements 2.3**
    """

    @given(
        query=query_without_pronouns_or_colloquial(),
        selected_text=meaningful_selected_text(min_size=2),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_3_augmented_query_longer_than_original(self, query, selected_text):
        """属性：无代词查询经语义增强后长度大于原始查询"""
        rewritten = _rewriter.rewrite(query, selected_text)

        assert len(rewritten) > len(query), (
            f"语义增强后查询长度未增加！\n"
            f"原始查询: {query!r} (长度={len(query)})\n"
            f"改写结果: {rewritten!r} (长度={len(rewritten)})\n"
            f"selected_text: {selected_text!r}\n"
            f"key_content: {_rewriter._extract_key_content(selected_text)!r}"
        )


from services.context_builder import ContextBuilder
import re

# ContextBuilder 实例（无状态，可复用）
_context_builder = ContextBuilder()

# 关键词分割正则（与 _extract_relevant_snippet 内部一致）
_KEYWORD_SPLIT_RE = re.compile(r'[\s,;，。；、？！?!：:""\'\'""\u201c\u201d\u2018\u2019]+')

# 句子边界字符（与 _extract_relevant_snippet 内部一致）
_BOUNDARY_CHARS = '。\n.！？!?；;'


def _extract_keywords(query: str, selected_text: str = "") -> list[str]:
    """模拟 _extract_relevant_snippet 内部的关键词提取逻辑"""
    combined = query
    if selected_text:
        combined = f"{query} {selected_text[:100]}"
    terms = [
        t for t in _KEYWORD_SPLIT_RE.split(combined.lower())
        if t and len(t) >= 2
    ]
    return terms


# ============================================================
# 预定义安全填充句子（不含任何关键词，用于确定性构造长文本）
# ============================================================

# 这些句子不包含关键词池中的任何词，避免 assume 过滤
_SAFE_FILLER_SENTENCES = [
    "本章介绍了相关背景知识",
    "实验结果表明该方案可行",
    "第二节讨论了具体实现细节",
    "综合以上分析可以得出结论",
    "附录中列出了完整的参考文献",
    "图表展示了各项指标的变化趋势",
    "下面将从三个方面展开论述",
    "该方案已在多个场景中得到验证",
    "进一步的优化工作正在进行中",
    "以上就是本文的主要贡献和创新点",
]

# 关键词池（长度 >= 2，不含分隔符，不出现在安全填充句子中）
_KEYWORD_POOL = [
    '机器学习', '深度学习', '神经网络', '自然语言', '数据分析',
    '向量检索', '文本分类', '模型训练', '特征提取', '语义理解',
]


# ============================================================
# Hypothesis 策略：_extract_relevant_snippet 测试用生成器
# ============================================================

@st.composite
def text_with_keyword_hit(draw, max_len=200):
    """生成包含关键词命中的测试数据

    策略（确定性构造，无 assume 过滤）：
    1. 从预定义关键词池中选择一个关键词
    2. 用安全填充句子拼接出长文本（> max_len），在中间插入关键词
    3. 生成包含该关键词的查询字符串

    返回 (text, query, selected_text, keyword)
    """
    keyword = draw(st.sampled_from(_KEYWORD_POOL))

    # 随机选择若干安全填充句子拼接前半部分
    num_before = draw(st.integers(min_value=4, max_value=7))
    before_indices = draw(st.lists(
        st.integers(min_value=0, max_value=len(_SAFE_FILLER_SENTENCES) - 1),
        min_size=num_before, max_size=num_before,
    ))
    filler_before = "。".join(_SAFE_FILLER_SENTENCES[i] for i in before_indices)

    # 后半部分
    num_after = draw(st.integers(min_value=1, max_value=3))
    after_indices = draw(st.lists(
        st.integers(min_value=0, max_value=len(_SAFE_FILLER_SENTENCES) - 1),
        min_size=num_after, max_size=num_after,
    ))
    filler_after = "。".join(_SAFE_FILLER_SENTENCES[i] for i in after_indices)

    # 组装文本：填充 + 关键词 + 填充
    text = filler_before + "。" + keyword + "是重要的研究方向。" + filler_after

    # 决定关键词放在 query 还是 selected_text 中（或两者都有）
    # 注意：关键词分割使用空格/标点，所以查询中关键词必须用分隔符隔开
    source_choice = draw(st.sampled_from(['query', 'selected_text', 'both']))

    if source_choice == 'query':
        query = f"请解释 {keyword} 的原理"
        selected_text = ""
    elif source_choice == 'selected_text':
        # query 需要有长度 >= 2 的词才能触发非回退逻辑
        query = "请解释 概念"
        selected_text = f"关于 {keyword} 的研究"
    else:
        query = f"分析 {keyword}"
        selected_text = f"{keyword} 相关内容"

    return text, query, selected_text, keyword


@st.composite
def long_text_with_mid_keyword(draw, max_len=200):
    """生成关键词在文本中间/后部的长文本

    确定性构造，确保：
    - 前半部分 > max_len + 100（关键词不在第一个窗口内）
    - 关键词前有句子边界字符
    - 片段不从文本开头开始

    返回 (text, query, max_len_val)
    """
    boundary = draw(st.sampled_from(list(_BOUNDARY_CHARS)))
    keyword = draw(st.sampled_from(_KEYWORD_POOL))

    # 用安全填充句子构造前半部分（> max_len + 100）
    # 每个句子约 10-15 字符，8 个句子 + 边界字符 ≈ 100-150 字符
    # 需要更多句子确保长度足够
    num_sentences = draw(st.integers(min_value=15, max_value=20))
    sent_indices = draw(st.lists(
        st.integers(min_value=0, max_value=len(_SAFE_FILLER_SENTENCES) - 1),
        min_size=num_sentences, max_size=num_sentences,
    ))
    front_part = boundary.join(_SAFE_FILLER_SENTENCES[i] for i in sent_indices)

    # 后半部分填充
    num_after = draw(st.integers(min_value=2, max_value=4))
    after_indices = draw(st.lists(
        st.integers(min_value=0, max_value=len(_SAFE_FILLER_SENTENCES) - 1),
        min_size=num_after, max_size=num_after,
    ))
    back_filler = boundary.join(_SAFE_FILLER_SENTENCES[i] for i in after_indices)

    # 组装：前半部分 + 边界 + 关键词 + 后半部分
    text = front_part + boundary + keyword + "是核心技术" + boundary + back_filler

    query = f"请解释 {keyword} 的原理"

    return text, query, max_len


# ============================================================
# Property 4: 关键词命中时片段包含关键词
# Feature: chatpdf-citation-relevance, Property 4: 关键词命中时片段包含关键词
# **Validates: Requirements 3.1, 3.3**
# ============================================================

class TestProperty4KeywordInSnippet:
    """Property 4: 关键词命中时片段包含关键词

    对于任意文本字符串、查询字符串和 selected_text，如果查询或 selected_text
    中的关键词（长度 >= 2）在文本中存在，调用
    _extract_relevant_snippet(text, query, selected_text=selected_text)
    返回的片段应包含至少一个命中的关键词。

    **Validates: Requirements 3.1, 3.3**
    """

    @given(data=text_with_keyword_hit())
    @settings(max_examples=100, deadline=None)
    def test_property_4_snippet_contains_hit_keyword(self, data):
        """属性：当关键词在文本中命中时，返回的片段包含至少一个命中关键词"""
        text, query, selected_text, keyword = data

        snippet = _context_builder._extract_relevant_snippet(
            text, query, max_len=200, selected_text=selected_text
        )

        # 提取所有关键词
        all_keywords = _extract_keywords(query, selected_text)

        # 找出在文本中命中的关键词
        text_lower = text.lower()
        hit_keywords = [kw for kw in all_keywords if kw.lower() in text_lower]

        # 至少应有一个命中关键词（由生成器保证）
        assert hit_keywords, (
            f"生成器错误：没有命中的关键词\n"
            f"关键词: {all_keywords}\n"
            f"文本前100字符: {text[:100]!r}"
        )

        # 验证：片段中应包含至少一个命中的关键词
        snippet_lower = snippet.lower()
        found_in_snippet = [kw for kw in hit_keywords if kw.lower() in snippet_lower]

        assert found_in_snippet, (
            f"片段中未包含任何命中的关键词！\n"
            f"命中关键词: {hit_keywords}\n"
            f"片段: {snippet!r}\n"
            f"查询: {query!r}\n"
            f"selected_text: {selected_text!r}\n"
            f"文本长度: {len(text)}"
        )


# ============================================================
# Property 5: 高亮片段句子边界对齐
# Feature: chatpdf-citation-relevance, Property 5: 高亮片段句子边界对齐
# **Validates: Requirements 3.4**
# ============================================================

class TestProperty5SentenceBoundaryAlignment:
    """Property 5: 高亮片段句子边界对齐

    对于任意长度超过 max_len 的文本字符串和任意查询，如果
    _extract_relevant_snippet 返回的片段不是从文本的第一个字符开始，
    那么片段起始位置的前一个字符应该是句子分隔符（句号、换行、问号、
    感叹号等）或在分隔符后 30 字符范围内。

    **Validates: Requirements 3.4**
    """

    @given(data=long_text_with_mid_keyword())
    @settings(max_examples=100, deadline=None)
    def test_property_5_snippet_aligns_to_sentence_boundary(self, data):
        """属性：非开头片段的起始位置在句子边界附近"""
        text, query, max_len_val = data

        snippet = _context_builder._extract_relevant_snippet(
            text, query, max_len=max_len_val
        )

        # 如果片段为空或文本太短，跳过
        assume(snippet)
        assume(len(text) > max_len_val)

        # 找到片段在原文中的起始位置
        snippet_stripped = snippet
        snippet_start = text.find(snippet_stripped)

        # 如果无法精确定位（strip 可能导致），尝试在文本中搜索
        if snippet_start == -1:
            # 片段可能被 strip 过，尝试查找片段的前几个字符
            if len(snippet_stripped) > 10:
                snippet_start = text.find(snippet_stripped[:50])

        # 如果仍然找不到，跳过此测试用例
        assume(snippet_start != -1)

        # 如果片段从文本开头开始，无需验证边界对齐
        if snippet_start == 0:
            return

        # 验证：片段起始位置前的字符应该是句子边界字符，
        # 或者在句子边界字符后 30 字符范围内
        found_boundary = False

        # 检查前 30 个字符范围内是否有句子边界字符
        search_start = max(0, snippet_start - 30)
        for i in range(snippet_start - 1, search_start - 1, -1):
            if text[i] in _BOUNDARY_CHARS:
                found_boundary = True
                break

        assert found_boundary, (
            f"片段起始位置未对齐到句子边界！\n"
            f"片段起始位置: {snippet_start}\n"
            f"片段前30字符: {text[max(0, snippet_start-30):snippet_start]!r}\n"
            f"片段前5字符: {text[max(0, snippet_start-5):snippet_start]!r}\n"
            f"片段开头: {snippet[:50]!r}\n"
            f"边界字符集: {_BOUNDARY_CHARS!r}"
        )


from routes.chat_routes import _build_fused_context, _build_selected_text_citation


# ============================================================
# Hypothesis 策略：融合逻辑测试用生成器
# ============================================================

@st.composite
def chinese_text(draw, min_size=1, max_size=500):
    """生成非空中文文本"""
    text = draw(st.text(
        alphabet=st.sampled_from(
            '机器学习是人工智能的一个分支它通过数据训练模型来进行预测和决策'
            '深度学习神经网络自然语言处理文本分类向量检索语义理解知识图谱'
        ),
        min_size=min_size,
        max_size=max_size,
    ))
    assume(text.strip())
    return text


@st.composite
def valid_page_info(draw):
    """生成有效的 page_info 字典，包含 page_start 和 page_end"""
    page_start = draw(st.integers(min_value=1, max_value=100))
    page_end = draw(st.integers(min_value=page_start, max_value=page_start + 50))
    return {"page_start": page_start, "page_end": page_end}


# ============================================================
# Property 1: 融合上下文包含 selected_text 且位置在前
# Feature: chatpdf-citation-relevance, Property 1
# **Validates: Requirements 1.1, 1.2**
# ============================================================

class TestProperty1FusedContextSelectedTextFirst:
    """Property 1: 融合上下文包含 selected_text 且位置在前

    对于任意非空 selected_text 和非空 retrieval_context，
    调用 _build_fused_context(selected_text, retrieval_context, page_info)
    返回的融合上下文字符串必须同时包含 selected_text 原文，
    且 selected_text 在字符串中的起始位置必须小于 retrieval_context
    中任意片段的起始位置。

    **Validates: Requirements 1.1, 1.2**
    """

    @given(
        selected_text=chinese_text(min_size=1, max_size=500),
        retrieval_context=chinese_text(min_size=1, max_size=2000),
        page_info=valid_page_info(),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_fused_context_contains_selected_text_first(
        self, selected_text, retrieval_context, page_info
    ):
        """属性：融合上下文包含 selected_text 且位置在 retrieval_context 之前"""
        # 确保 selected_text 和 retrieval_context 不同，
        # 避免 index() 找到同一位置导致误判
        assume(selected_text != retrieval_context)
        # 确保 retrieval_context 不是 selected_text 的子串（反之亦然），
        # 避免嵌套匹配干扰位置判断
        assume(retrieval_context not in selected_text)
        assume(selected_text not in retrieval_context)

        fused = _build_fused_context(selected_text, retrieval_context, page_info)

        # 验证 1：融合上下文包含 selected_text 原文
        assert selected_text in fused, (
            f"融合上下文未包含 selected_text！\n"
            f"selected_text: {selected_text!r}\n"
            f"融合上下文: {fused!r}"
        )

        # 验证 2：融合上下文包含 retrieval_context
        assert retrieval_context in fused, (
            f"融合上下文未包含 retrieval_context！\n"
            f"retrieval_context: {retrieval_context!r}\n"
            f"融合上下文: {fused!r}"
        )

        # 验证 3：selected_text 的起始位置在 retrieval_context 之前
        selected_pos = fused.index(selected_text)
        retrieval_pos = fused.index(retrieval_context)
        assert selected_pos < retrieval_pos, (
            f"selected_text 未在 retrieval_context 之前！\n"
            f"selected_text 位置: {selected_pos}\n"
            f"retrieval_context 位置: {retrieval_pos}\n"
            f"融合上下文: {fused!r}"
        )


# ============================================================
# Property 6: 基础 citation 结构完整性
# Feature: chatpdf-citation-relevance, Property 6
# **Validates: Requirements 4.2**
# ============================================================

class TestProperty6CitationStructureIntegrity:
    """Property 6: 基础 citation 结构完整性

    对于任意非空 selected_text 和有效的 page_info（包含 page_start 和 page_end），
    调用 _build_selected_text_citation(selected_text, page_info) 返回的 citation
    字典必须包含 "ref"、"group_id"、"page_range"、"highlight_text" 四个键，
    且 page_range 与 page_info 一致，highlight_text 长度不超过 200。

    **Validates: Requirements 4.2**
    """

    @given(
        selected_text=chinese_text(min_size=1, max_size=500),
        page_info=valid_page_info(),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_6_citation_has_required_keys_and_valid_values(
        self, selected_text, page_info
    ):
        """属性：citation 包含所有必需键且值正确"""
        citation = _build_selected_text_citation(selected_text, page_info)

        # 验证 1：包含所有必需键
        required_keys = {"ref", "group_id", "page_range", "highlight_text"}
        assert required_keys.issubset(citation.keys()), (
            f"citation 缺少必需键！\n"
            f"期望键: {required_keys}\n"
            f"实际键: {set(citation.keys())}\n"
            f"citation: {citation}"
        )

        # 验证 2：page_range 与 page_info 一致
        expected_range = [page_info["page_start"], page_info["page_end"]]
        assert citation["page_range"] == expected_range, (
            f"page_range 与 page_info 不一致！\n"
            f"期望: {expected_range}\n"
            f"实际: {citation['page_range']}\n"
            f"page_info: {page_info}"
        )

        # 验证 3：highlight_text 长度不超过 200
        assert len(citation["highlight_text"]) <= 200, (
            f"highlight_text 长度超过 200！\n"
            f"长度: {len(citation['highlight_text'])}\n"
            f"highlight_text: {citation['highlight_text']!r}"
        )

        # 验证 4：ref 固定为 1
        assert citation["ref"] == 1, (
            f"ref 不等于 1！实际值: {citation['ref']}"
        )

        # 验证 5：group_id 固定为 "selected-text"
        assert citation["group_id"] == "selected-text", (
            f"group_id 不等于 'selected-text'！实际值: {citation['group_id']!r}"
        )
