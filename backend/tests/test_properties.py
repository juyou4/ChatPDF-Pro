"""
属性测试集合

使用 hypothesis 库对 ChatPDF RAG 系统的核心组件进行属性测试，
验证系统在所有有效输入下都能保持正确性属性。

测试框架：pytest + hypothesis
"""

import json
import math
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.semantic_group_service import SemanticGroup, SemanticGroupService
from services.token_budget import TokenBudgetManager


# ---- 自定义 Hypothesis 策略 ----

# 有效的 summary_status 值
summary_status_strategy = st.sampled_from(["ok", "failed", "truncated"])

# 生成有效的 llm_meta 字典或 None
llm_meta_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "model": st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("L", "N", "P"),
            whitelist_characters="-_."
        )),
        "temperature": st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        "prompt_version": st.text(min_size=1, max_size=10, alphabet=st.characters(
            whitelist_categories=("L", "N"),
            whitelist_characters="v."
        )),
        "created_at": st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=("N",),
            whitelist_characters="-T:Z"
        )),
    }),
)

# 生成有效的关键词列表（0-6 个关键词）
keywords_strategy = st.lists(
    st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    )),
    min_size=0,
    max_size=6,
)

# 生成有效的页码范围 (start, end)，确保 start <= end
page_range_strategy = st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
).filter(lambda t: t[0] <= t[1])

# 生成有效的分块索引列表（至少 1 个索引，索引值非负）
chunk_indices_strategy = st.lists(
    st.integers(min_value=0, max_value=10000),
    min_size=1,
    max_size=50,
)

# 生成有效的 SemanticGroup 对象
semantic_group_strategy = st.builds(
    SemanticGroup,
    group_id=st.from_regex(r"group-[0-9]{1,4}", fullmatch=True),
    chunk_indices=chunk_indices_strategy,
    char_count=st.integers(min_value=0, max_value=100000),
    summary=st.text(min_size=0, max_size=80),
    digest=st.text(min_size=0, max_size=1000),
    full_text=st.text(min_size=1, max_size=5000),
    keywords=keywords_strategy,
    page_range=page_range_strategy,
    summary_status=summary_status_strategy,
    llm_meta=llm_meta_strategy,
)

# 生成有效的 SemanticGroup 列表
semantic_group_list_strategy = st.lists(
    semantic_group_strategy,
    min_size=0,
    max_size=10,
)


# ---- Property 3: 意群数据 JSON round-trip ----

class TestProperty3_SemanticGroupJsonRoundTrip:
    """
    **Feature: chatpdf-rag-optimization, Property 3: 意群数据 JSON round-trip**

    **Validates: Requirements 1.5, 1.6, 1.7**

    属性描述：For any 有效的 SemanticGroup 列表，将其序列化为 JSON
    再反序列化应产生等价的对象列表（所有字段值相同）。
    """

    @given(group=semantic_group_strategy)
    @settings(max_examples=100)
    def test_单个意群_to_dict_from_dict_往返一致(self, group: SemanticGroup):
        """
        **Validates: Requirements 1.5, 1.6**

        对任意有效的 SemanticGroup，to_dict() 后 from_dict() 应还原为等价对象。
        验证所有字段值在序列化/反序列化后保持不变。
        """
        # 序列化为字典
        d = group.to_dict()

        # 确保字典可以被 JSON 序列化（验证 JSON 兼容性）
        json_str = json.dumps(d, ensure_ascii=False)
        d_from_json = json.loads(json_str)

        # 从字典还原
        restored = SemanticGroup.from_dict(d_from_json)

        # 验证所有字段值相同
        assert restored.group_id == group.group_id, \
            f"group_id 不一致: {restored.group_id} != {group.group_id}"
        assert restored.chunk_indices == group.chunk_indices, \
            f"chunk_indices 不一致: {restored.chunk_indices} != {group.chunk_indices}"
        assert restored.char_count == group.char_count, \
            f"char_count 不一致: {restored.char_count} != {group.char_count}"
        assert restored.summary == group.summary, \
            f"summary 不一致: {restored.summary!r} != {group.summary!r}"
        assert restored.digest == group.digest, \
            f"digest 不一致: {restored.digest!r} != {group.digest!r}"
        assert restored.full_text == group.full_text, \
            f"full_text 不一致: {restored.full_text!r} != {group.full_text!r}"
        assert restored.keywords == group.keywords, \
            f"keywords 不一致: {restored.keywords} != {group.keywords}"
        assert restored.page_range == group.page_range, \
            f"page_range 不一致: {restored.page_range} != {group.page_range}"
        assert restored.summary_status == group.summary_status, \
            f"summary_status 不一致: {restored.summary_status} != {group.summary_status}"
        assert restored.llm_meta == group.llm_meta, \
            f"llm_meta 不一致: {restored.llm_meta} != {group.llm_meta}"

    @given(groups=semantic_group_list_strategy)
    @settings(max_examples=100)
    def test_意群列表_save_load_往返一致(self, groups: list):
        """
        **Validates: Requirements 1.5, 1.6, 1.7**

        对任意有效的 SemanticGroup 列表，save_groups() 后 load_groups()
        应还原为等价列表（所有意群的所有字段值相同）。
        """
        service = SemanticGroupService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_id = "test-roundtrip"

            # 保存意群列表
            service.save_groups(doc_id, groups, tmp_dir)

            # 加载意群列表
            loaded = service.load_groups(doc_id, tmp_dir)

            # 验证加载成功
            assert loaded is not None, "load_groups 返回 None，加载失败"

            # 验证列表长度一致
            assert len(loaded) == len(groups), \
                f"意群数量不一致: 加载 {len(loaded)} 个，原始 {len(groups)} 个"

            # 逐个验证每个意群的所有字段
            for i, (original, restored) in enumerate(zip(groups, loaded)):
                assert restored.group_id == original.group_id, \
                    f"意群 {i}: group_id 不一致"
                assert restored.chunk_indices == original.chunk_indices, \
                    f"意群 {i}: chunk_indices 不一致"
                assert restored.char_count == original.char_count, \
                    f"意群 {i}: char_count 不一致"
                assert restored.summary == original.summary, \
                    f"意群 {i}: summary 不一致"
                assert restored.digest == original.digest, \
                    f"意群 {i}: digest 不一致"
                assert restored.full_text == original.full_text, \
                    f"意群 {i}: full_text 不一致"
                assert restored.keywords == original.keywords, \
                    f"意群 {i}: keywords 不一致"
                assert restored.page_range == original.page_range, \
                    f"意群 {i}: page_range 不一致"
                assert restored.summary_status == original.summary_status, \
                    f"意群 {i}: summary_status 不一致"
                assert restored.llm_meta == original.llm_meta, \
                    f"意群 {i}: llm_meta 不一致"



# ---- Property 8: Token 估算语言感知 ----

# 纯中文字符策略：从 CJK 统一表意文字基本区 (U+4E00 - U+9FFF) 中生成
# 使用 integers + map 避免 hypothesis 新版 characters() 的参数限制
chinese_char_strategy = st.integers(
    min_value=0x4E00, max_value=0x9FFF
).map(chr)

# 纯中文文本策略：长度 >= 2
chinese_text_strategy = st.lists(
    chinese_char_strategy,
    min_size=2,
    max_size=200,
).map("".join)

# 纯英文字母策略（仅 ASCII 字母）
english_char_strategy = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
)

# 纯英文文本策略：长度 >= 2
english_text_strategy = st.lists(
    english_char_strategy,
    min_size=2,
    max_size=200,
).map("".join)


class TestProperty8_TokenEstimateLanguageAware:
    """
    **Feature: chatpdf-rag-optimization, Property 8: Token 估算语言感知**

    **Validates: Requirements 3.4**

    属性描述：For any 文本字符串，estimate_tokens 对纯中文文本的估算值
    应大于对等长度纯英文文本的估算值（因为中文 token 密度更高）。
    具体地，纯中文文本的估算应接近 len(text) / 1.5，
    纯英文文本应接近 len(text) / 4。
    """

    @given(chinese_text=chinese_text_strategy)
    @settings(max_examples=100)
    def test_中文估算大于等长英文估算(self, chinese_text: str):
        """
        **Validates: Requirements 3.4**

        对任意长度 >= 2 的纯中文文本，生成等长度的纯英文文本，
        验证中文的 Token 估算值严格大于英文的 Token 估算值。
        """
        manager = TokenBudgetManager()

        # 生成等长度的纯英文文本
        text_len = len(chinese_text)
        english_text = "a" * text_len

        chinese_tokens = manager.estimate_tokens(chinese_text)
        english_tokens = manager.estimate_tokens(english_text)

        # 中文 token 密度更高（1.5 字符/token vs 4 字符/token），
        # 所以同等长度的中文文本应产生更多 token
        assert chinese_tokens > english_tokens, (
            f"长度 {text_len} 的纯中文文本估算 Token ({chinese_tokens}) "
            f"应大于等长纯英文文本估算 Token ({english_tokens})"
        )

    @given(chinese_text=chinese_text_strategy)
    @settings(max_examples=100)
    def test_纯中文估算接近_len除以1点5(self, chinese_text: str):
        """
        **Validates: Requirements 3.4**

        对任意纯中文文本，estimate_tokens 的结果应等于
        math.ceil(len(text) / 1.5)，即中文按 1.5 字符/token 估算。
        """
        manager = TokenBudgetManager()

        text_len = len(chinese_text)
        estimated = manager.estimate_tokens(chinese_text)
        expected = math.ceil(text_len / 1.5)

        assert estimated == expected, (
            f"纯中文文本（长度 {text_len}）估算 Token 为 {estimated}，"
            f"期望值为 ceil({text_len}/1.5) = {expected}"
        )

    @given(english_text=english_text_strategy)
    @settings(max_examples=100)
    def test_纯英文估算接近_len除以4(self, english_text: str):
        """
        **Validates: Requirements 3.4**

        对任意纯英文文本，estimate_tokens 的结果应等于
        math.ceil(len(text) / 4)，即英文按 4 字符/token 估算。
        """
        manager = TokenBudgetManager()

        text_len = len(english_text)
        estimated = manager.estimate_tokens(english_text)
        expected = math.ceil(text_len / 4)

        assert estimated == expected, (
            f"纯英文文本（长度 {text_len}）估算 Token 为 {estimated}，"
            f"期望值为 ceil({text_len}/4) = {expected}"
        )


# ---- Property 7: Token 预算不变量 ----

# 粒度策略：随机选择 full/digest/summary
granularity_strategy = st.sampled_from(["full", "digest", "summary"])

# 生成带初始粒度分配的意群列表项
# 每个项为 {"group": SemanticGroup, "granularity": str, "tokens": 0}
# tokens 字段初始为 0，由 fit_within_budget 重新计算
semantic_group_with_granularity_strategy = st.builds(
    lambda group, granularity: {
        "group": group,
        "granularity": granularity,
        "tokens": 0,
    },
    group=semantic_group_strategy,
    granularity=granularity_strategy,
)

# 生成意群列表（带初始粒度分配）
groups_with_granularity_list_strategy = st.lists(
    semantic_group_with_granularity_strategy,
    min_size=0,
    max_size=10,
)

# Token 预算值策略：正整数，范围覆盖从极小到较大的预算
token_budget_strategy = st.integers(min_value=1, max_value=50000)


class TestProperty7_TokenBudgetInvariant:
    """
    **Feature: chatpdf-rag-optimization, Property 7: Token 预算不变量**

    **Validates: Requirements 3.2, 3.3**

    属性描述：For any 意群列表和 Token 预算值，Token_Budget_Manager
    返回的结果中所有意群的累计 Token 数不应超过预算值。
    """

    @given(
        groups=groups_with_granularity_list_strategy,
        budget=token_budget_strategy,
    )
    @settings(max_examples=100)
    def test_fit_within_budget_累计Token不超过预算(
        self, groups: list, budget: int
    ):
        """
        **Validates: Requirements 3.2, 3.3**

        对任意意群列表和正整数 Token 预算值，fit_within_budget 返回的
        结果中所有意群的累计 Token 数不应超过预算值。
        """
        manager = TokenBudgetManager()

        # 调用 fit_within_budget，使用随机生成的预算值
        result = manager.fit_within_budget(groups, max_tokens=budget)

        # 计算结果中所有意群的累计 Token 数
        total_tokens = sum(item["tokens"] for item in result)

        # 核心不变量：累计 Token 数不应超过预算值
        assert total_tokens <= budget, (
            f"累计 Token 数 ({total_tokens}) 超过预算 ({budget})。"
            f"返回了 {len(result)} 个意群，"
            f"各意群 Token: {[item['tokens'] for item in result]}"
        )

    @given(
        groups=groups_with_granularity_list_strategy,
        budget=token_budget_strategy,
    )
    @settings(max_examples=100)
    def test_fit_within_budget_返回Token与estimate一致(
        self, groups: list, budget: int
    ):
        """
        **Validates: Requirements 3.2, 3.3**

        对任意意群列表和预算值，fit_within_budget 返回的每个意群的
        tokens 字段应与使用 estimate_tokens 对该粒度文本的估算值一致。
        这确保了 Token 计数的准确性，从而保证预算不变量的可靠性。
        """
        manager = TokenBudgetManager()

        result = manager.fit_within_budget(groups, max_tokens=budget)

        # 粒度到文本属性的映射
        granularity_text_attr = {
            "full": "full_text",
            "digest": "digest",
            "summary": "summary",
        }

        for item in result:
            group = item["group"]
            granularity = item["granularity"]
            reported_tokens = item["tokens"]

            # 获取对应粒度的文本
            text_attr = granularity_text_attr[granularity]
            text = getattr(group, text_attr, "")

            # 验证 tokens 字段与 estimate_tokens 一致
            expected_tokens = manager.estimate_tokens(text)
            assert reported_tokens == expected_tokens, (
                f"意群 {group.group_id} 的 tokens ({reported_tokens}) "
                f"与 estimate_tokens 估算值 ({expected_tokens}) 不一致，"
                f"粒度: {granularity}"
            )

    @given(
        groups=groups_with_granularity_list_strategy,
        budget=token_budget_strategy,
    )
    @settings(max_examples=100)
    def test_fit_within_budget_粒度只降不升(
        self, groups: list, budget: int
    ):
        """
        **Validates: Requirements 3.2, 3.3**

        对任意意群列表和预算值，fit_within_budget 返回的每个意群的
        粒度级别应 ≤ 输入的粒度级别（即只降级不升级）。
        粒度顺序：full(0) > digest(1) > summary(2)。
        """
        manager = TokenBudgetManager()

        # 构建输入粒度映射（按 group_id + 位置索引）
        granularity_order = {"full": 0, "digest": 1, "summary": 2}

        result = manager.fit_within_budget(groups, max_tokens=budget)

        # 结果中的意群顺序与输入一致（只可能更少），逐个比较
        result_idx = 0
        for input_item in groups:
            if result_idx >= len(result):
                break
            result_item = result[result_idx]
            # 通过 group 对象引用匹配（同一对象）
            if result_item["group"] is input_item["group"]:
                input_level = granularity_order.get(input_item["granularity"], 0)
                output_level = granularity_order.get(result_item["granularity"], 0)
                assert output_level >= input_level, (
                    f"意群 {result_item['group'].group_id} 的粒度从 "
                    f"{input_item['granularity']}({input_level}) 升级为 "
                    f"{result_item['granularity']}({output_level})，"
                    f"违反了只降不升的规则"
                )
                result_idx += 1


# ---- Property 4: 查询分类输出有效性 ----

# 导入粒度选择器和查询分析器
from services.granularity_selector import GranularitySelector, QUERY_TYPE_MAPPING
from services.query_analyzer import analyze_query_type

# 随机查询字符串策略：包含 ASCII 文本和中文字符的混合
query_string_strategy = st.text(
    min_size=0,
    max_size=200,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
    ),
)

# 有效的查询类型集合
VALID_QUERY_TYPES = {"overview", "extraction", "analytical", "specific"}


class TestProperty4_QueryClassificationValidity:
    """
    **Feature: chatpdf-rag-optimization, Property 4: 查询分类输出有效性**

    **Validates: Requirements 2.1**

    属性描述：For any 字符串查询，Query_Analyzer 的分类结果必须是
    "overview"、"extraction"、"analytical" 或 "specific" 四种类型之一。
    """

    @given(query=query_string_strategy)
    @settings(max_examples=100)
    def test_查询分类结果必须是四种有效类型之一(self, query: str):
        """
        **Validates: Requirements 2.1**

        对任意字符串查询，analyze_query_type 返回的分类结果
        必须是 "overview"、"extraction"、"analytical" 或 "specific" 之一。
        """
        # 调用查询分析器
        result = analyze_query_type(query)

        # 验证返回值是四种有效类型之一
        assert result in VALID_QUERY_TYPES, (
            f"查询 {query!r} 的分类结果 {result!r} "
            f"不在有效类型集合 {VALID_QUERY_TYPES} 中"
        )

    @given(query=query_string_strategy)
    @settings(max_examples=100)
    def test_通过GranularitySelector_select获取的query_type也有效(self, query: str):
        """
        **Validates: Requirements 2.1**

        对任意字符串查询，通过 GranularitySelector.select 获取的
        query_type 也必须是四种有效类型之一。
        """
        selector = GranularitySelector()

        # 使用空意群列表调用 select（查询分类不依赖意群内容）
        selection = selector.select(query, groups=[], max_tokens=8000)

        # 验证 query_type 是有效类型
        assert selection.query_type in VALID_QUERY_TYPES, (
            f"查询 {query!r} 通过 GranularitySelector.select 获取的 "
            f"query_type {selection.query_type!r} "
            f"不在有效类型集合 {VALID_QUERY_TYPES} 中"
        )


# ---- Property 5: 粒度选择规则一致性 ----

# 期望的查询类型到 (粒度, max_groups) 的映射
EXPECTED_MAPPING = {
    "overview": ("summary", 10),
    "extraction": ("full", 3),
    "analytical": ("digest", 5),
    "specific": ("digest", 5),
}


class TestProperty5_GranularitySelectionConsistency:
    """
    **Feature: chatpdf-rag-optimization, Property 5: 粒度选择规则一致性**

    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**

    属性描述：For any 查询和对应的查询类型，Granularity_Selector 返回的
    粒度和 max_groups 应符合以下映射：
    overview→(summary, 10)、extraction→(full, 3)、
    analytical→(digest, 5)、specific→(digest, 5)。
    """

    @given(query=query_string_strategy)
    @settings(max_examples=100)
    def test_粒度选择结果符合查询类型映射规则(self, query: str):
        """
        **Validates: Requirements 2.2, 2.3, 2.4, 2.5**

        对任意查询字符串，GranularitySelector.select 返回的
        granularity 和 max_groups 应与查询类型的映射规则一致。
        """
        selector = GranularitySelector()

        # 调用粒度选择器
        selection = selector.select(query, groups=[], max_tokens=8000)

        # 获取查询类型
        query_type = selection.query_type

        # 确保查询类型有效（Property 4 已验证）
        assert query_type in EXPECTED_MAPPING, (
            f"查询类型 {query_type!r} 不在期望映射中"
        )

        # 获取期望的粒度和 max_groups
        expected_granularity, expected_max_groups = EXPECTED_MAPPING[query_type]

        # 验证粒度一致
        assert selection.granularity == expected_granularity, (
            f"查询 {query!r}（类型: {query_type}）的粒度为 "
            f"{selection.granularity!r}，期望为 {expected_granularity!r}"
        )

        # 验证 max_groups 一致
        assert selection.max_groups == expected_max_groups, (
            f"查询 {query!r}（类型: {query_type}）的 max_groups 为 "
            f"{selection.max_groups}，期望为 {expected_max_groups}"
        )


# ---- Property 6: 混合粒度位置规则 ----


class TestProperty6_MixedGranularityPositionRule:
    """
    **Feature: chatpdf-rag-optimization, Property 6: 混合粒度位置规则**

    **Validates: Requirements 3.1**

    属性描述：For any 包含 N 个（N≥1）排序意群的列表，混合粒度分配应满足：
    第 1 个意群使用 full 粒度，第 2-3 个使用 digest 粒度，
    第 4 个及之后使用 summary 粒度（Token 预算充足时）。
    """

    @given(
        groups=st.lists(semantic_group_strategy, min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_混合粒度按位置规则分配(self, groups: list):
        """
        **Validates: Requirements 3.1**

        对任意包含 N 个（N≥1）排序意群的列表，select_mixed 返回的
        粒度分配应满足位置规则：
        - 第 1 个（rank=0）→ full
        - 第 2-3 个（rank=1,2）→ digest
        - 第 4 个及之后（rank>=3）→ summary
        """
        selector = GranularitySelector()

        # 调用混合粒度分配
        result = selector.select_mixed(
            query="测试查询",
            ranked_groups=groups,
            max_tokens=8000,
        )

        # 验证返回列表长度与输入一致
        assert len(result) == len(groups), (
            f"返回列表长度 {len(result)} 与输入长度 {len(groups)} 不一致"
        )

        # 逐个验证粒度分配
        for rank, item in enumerate(result):
            if rank == 0:
                expected = "full"
            elif rank <= 2:
                expected = "digest"
            else:
                expected = "summary"

            assert item["granularity"] == expected, (
                f"排名第 {rank + 1} 的意群（{item['group'].group_id}）"
                f"粒度为 {item['granularity']!r}，期望为 {expected!r}"
            )

    @given(
        groups=st.lists(semantic_group_strategy, min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_混合粒度返回的group对象与输入一致(self, groups: list):
        """
        **Validates: Requirements 3.1**

        对任意排序意群列表，select_mixed 返回的每个 group 对象
        应与输入列表中对应位置的对象相同（保持顺序和引用一致）。
        """
        selector = GranularitySelector()

        result = selector.select_mixed(
            query="测试查询",
            ranked_groups=groups,
            max_tokens=8000,
        )

        # 验证每个位置的 group 对象引用一致
        for i, (input_group, result_item) in enumerate(zip(groups, result)):
            assert result_item["group"] is input_group, (
                f"位置 {i} 的 group 对象引用不一致"
            )

    def test_空列表返回空结果(self):
        """
        **Validates: Requirements 3.1**

        空的排序意群列表应返回空结果列表。
        """
        selector = GranularitySelector()

        result = selector.select_mixed(
            query="测试查询",
            ranked_groups=[],
            max_tokens=8000,
        )

        assert result == [], f"空输入应返回空列表，实际返回 {result}"


# ---- Property 1: 意群聚合字符数约束 ----

# 普通文本分块策略：不包含标题、表格、代码块标记
# 避免以编号模式（如 "1." "1.1"）、Markdown 标题（"# "）、
# 全大写字母行、表格行（含 |）、代码块标记（```）开头
def _is_safe_chunk(text: str) -> bool:
    """检查文本是否不包含硬边界标记（标题/表格/代码块）"""
    if not text:
        return True
    first_line = text.split("\n", 1)[0].strip()
    if not first_line:
        return True
    service = SemanticGroupService()
    # 排除标题行和表格/代码块边界
    if service._is_heading_line(first_line):
        return False
    if service._is_table_or_code_boundary(first_line):
        return False
    return True


# 安全字符集：小写字母、数字、中文字符、空格、标点（不含 | # `）
_safe_alphabet = st.characters(
    whitelist_categories=("Ll", "N"),
    whitelist_characters="，。、；：？！ 　"
)

# 生成不包含硬边界标记的普通文本分块
# 使用小写字母开头确保不会被识别为标题
safe_chunk_strategy = st.text(
    min_size=1,
    max_size=3000,
    alphabet=_safe_alphabet,
).filter(_is_safe_chunk)

# 生成非空的安全分块列表
safe_chunk_list_strategy = st.lists(
    safe_chunk_strategy,
    min_size=1,
    max_size=30,
)

# 聚合参数策略：target_chars, min_chars, max_chars
# 约束：min_chars <= target_chars <= max_chars
@st.composite
def aggregation_params_strategy(draw):
    """生成合理的聚合参数 (target_chars, min_chars, max_chars)"""
    min_chars = draw(st.integers(min_value=100, max_value=3000))
    target_chars = draw(st.integers(min_value=min_chars, max_value=min_chars + 5000))
    max_chars = draw(st.integers(min_value=target_chars, max_value=target_chars + 3000))
    return (target_chars, min_chars, max_chars)


class TestProperty1_AggregateChunksCharConstraint:
    """
    **Feature: chatpdf-rag-optimization, Property 1: 意群聚合字符数约束**

    **Validates: Requirements 1.1**

    属性描述：对任意非空分块列表（所有分块在同一页且不包含标题/表格/代码块边界），
    聚合后的每个候选意群（除最后一个外）的 char_count 应 >= min_chars 且 <= max_chars。
    """

    @given(
        chunks=safe_chunk_list_strategy,
        params=aggregation_params_strategy(),
    )
    @settings(max_examples=100)
    def test_非最后意群的char_count不超过max_chars(
        self, chunks: list, params: tuple
    ):
        """
        **Validates: Requirements 1.1**

        对任意非空分块列表（同一页、无硬边界），聚合后除最后一个外，
        每个候选意群的 char_count 应 <= max_chars。
        """
        target_chars, min_chars, max_chars = params

        service = SemanticGroupService()

        # 所有分块使用相同页码，避免页面边界干扰
        chunk_pages = [1] * len(chunks)

        candidates = service._aggregate_chunks(
            chunks, chunk_pages,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        # 至少应有一个候选意群（输入非空）
        assert len(candidates) >= 1, "非空分块列表应至少产生一个候选意群"

        # 除最后一个外，每个候选意群的 char_count 应 <= max_chars
        for i, candidate in enumerate(candidates[:-1]):
            assert candidate["char_count"] <= max_chars, (
                f"第 {i} 个候选意群（非最后一个）的 char_count "
                f"({candidate['char_count']}) 超过 max_chars ({max_chars})"
            )

    @given(
        chunks=safe_chunk_list_strategy,
        params=aggregation_params_strategy(),
    )
    @settings(max_examples=100)
    def test_非最后意群的char_count不低于min_chars_无硬边界(
        self, chunks: list, params: tuple
    ):
        """
        **Validates: Requirements 1.1**

        对任意非空分块列表（同一页、无硬边界），聚合后除最后一个外，
        每个候选意群的 char_count 应 >= min_chars。
        由于所有分块在同一页且不包含标题/表格/代码块标记，
        不存在硬边界导致的提前切分。
        """
        target_chars, min_chars, max_chars = params

        service = SemanticGroupService()

        # 所有分块使用相同页码
        chunk_pages = [1] * len(chunks)

        candidates = service._aggregate_chunks(
            chunks, chunk_pages,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        # 如果只有一个候选意群，则它是最后一个，跳过 min_chars 检查
        if len(candidates) <= 1:
            return

        # 除最后一个外，每个候选意群的 char_count 应 >= min_chars
        for i, candidate in enumerate(candidates[:-1]):
            assert candidate["char_count"] >= min_chars, (
                f"第 {i} 个候选意群（非最后一个）的 char_count "
                f"({candidate['char_count']}) 低于 min_chars ({min_chars})，"
                f"但不存在硬边界（同一页、无标题/表格/代码块标记）"
            )

    @given(
        chunks=safe_chunk_list_strategy,
        params=aggregation_params_strategy(),
    )
    @settings(max_examples=100)
    def test_所有分块索引被完整覆盖(
        self, chunks: list, params: tuple
    ):
        """
        **Validates: Requirements 1.1**

        对任意非空分块列表，聚合后所有候选意群的 chunk_indices
        应完整覆盖所有输入分块索引（0 到 len(chunks)-1），且不重叠。
        """
        target_chars, min_chars, max_chars = params

        service = SemanticGroupService()
        chunk_pages = [1] * len(chunks)

        candidates = service._aggregate_chunks(
            chunks, chunk_pages,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        # 收集所有候选意群的分块索引
        all_indices = []
        for candidate in candidates:
            all_indices.extend(candidate["chunk_indices"])

        # 验证完整覆盖且不重叠
        expected_indices = list(range(len(chunks)))
        assert sorted(all_indices) == expected_indices, (
            f"分块索引不完整或有重叠: "
            f"实际 {sorted(all_indices)}，期望 {expected_indices}"
        )


# ---- Property 2: 意群结构完整性 ----

import asyncio
from unittest.mock import AsyncMock, patch


# 生成用于 Property 2 的分块列表（可以包含任意文本）
property2_chunk_strategy = st.text(
    min_size=1,
    max_size=2000,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
    ),
)

property2_chunk_list_strategy = st.lists(
    property2_chunk_strategy,
    min_size=1,
    max_size=15,
)


class TestProperty2_SemanticGroupStructuralIntegrity:
    """
    **Feature: chatpdf-rag-optimization, Property 2: 意群结构完整性**

    **Validates: Requirements 1.3**

    属性描述：For any 生成的 Semantic_Group，其 summary 字段长度应 ≤ 80 字符，
    digest 字段长度应 ≤ 1000 字符，full_text 字段应非空，
    且 keywords 应为列表类型。
    """

    @given(chunks=property2_chunk_list_strategy)
    @settings(max_examples=100)
    def test_generate_groups_结构完整性(self, chunks: list):
        """
        **Validates: Requirements 1.3**

        对任意非空分块列表，generate_groups 返回的每个 SemanticGroup 应满足：
        - summary 长度 ≤ 80 字符
        - digest 长度 ≤ 1000 字符
        - full_text 非空
        - keywords 为列表类型
        """
        service = SemanticGroupService(api_key="", model="test")
        chunk_pages = [1] * len(chunks)

        # mock _generate_summary：返回截断后的文本和 "ok" 状态
        async def mock_generate_summary(text: str, max_length: int):
            # 模拟 LLM 返回：截断到 max_length 以内
            truncated = text[:max_length] if len(text) > max_length else text
            return truncated, "ok"

        # mock _extract_keywords：返回固定的关键词列表
        async def mock_extract_keywords(text: str):
            return ["关键词1", "关键词2", "关键词3"]

        # 使用 patch 替换 LLM 相关方法
        with patch.object(
            service, "_generate_summary",
            side_effect=mock_generate_summary
        ), patch.object(
            service, "_extract_keywords",
            side_effect=mock_extract_keywords
        ):
            # 运行异步方法（兼容 Python 3.10+）
            groups = asyncio.run(
                service.generate_groups(
                    chunks, chunk_pages,
                    target_chars=5000,
                    min_chars=2500,
                    max_chars=6000,
                )
            )

        # 至少应有一个意群（输入非空）
        assert len(groups) >= 1, "非空分块列表应至少产生一个意群"

        for i, group in enumerate(groups):
            # summary 长度 ≤ 80 字符
            assert len(group.summary) <= 80, (
                f"意群 {group.group_id} 的 summary 长度 "
                f"({len(group.summary)}) 超过 80 字符"
            )

            # digest 长度 ≤ 1000 字符
            assert len(group.digest) <= 1000, (
                f"意群 {group.group_id} 的 digest 长度 "
                f"({len(group.digest)}) 超过 1000 字符"
            )

            # full_text 非空
            assert len(group.full_text) > 0, (
                f"意群 {group.group_id} 的 full_text 为空"
            )

            # keywords 为列表类型
            assert isinstance(group.keywords, list), (
                f"意群 {group.group_id} 的 keywords 类型为 "
                f"{type(group.keywords).__name__}，期望为 list"
            )

    @given(chunks=property2_chunk_list_strategy)
    @settings(max_examples=100)
    def test_generate_groups_降级时结构仍完整(self, chunks: list):
        """
        **Validates: Requirements 1.3**

        当 LLM 不可用（api_key 为空）时，generate_groups 降级为文本截断，
        返回的每个 SemanticGroup 仍应满足结构完整性约束。
        """
        # 不设置 api_key，触发降级逻辑
        service = SemanticGroupService(api_key="", model="test")
        chunk_pages = [1] * len(chunks)

        # 直接调用 generate_groups，不 mock，让降级逻辑生效（兼容 Python 3.10+）
        groups = asyncio.run(
            service.generate_groups(
                chunks, chunk_pages,
                target_chars=5000,
                min_chars=2500,
                max_chars=6000,
            )
        )

        assert len(groups) >= 1, "非空分块列表应至少产生一个意群"

        for i, group in enumerate(groups):
            # summary 长度 ≤ 80 字符
            assert len(group.summary) <= 80, (
                f"降级模式下，意群 {group.group_id} 的 summary 长度 "
                f"({len(group.summary)}) 超过 80 字符"
            )

            # digest 长度 ≤ 1000 字符
            assert len(group.digest) <= 1000, (
                f"降级模式下，意群 {group.group_id} 的 digest 长度 "
                f"({len(group.digest)}) 超过 1000 字符"
            )

            # full_text 非空
            assert len(group.full_text) > 0, (
                f"降级模式下，意群 {group.group_id} 的 full_text 为空"
            )

            # keywords 为列表类型
            assert isinstance(group.keywords, list), (
                f"降级模式下，意群 {group.group_id} 的 keywords 类型为 "
                f"{type(group.keywords).__name__}，期望为 list"
            )


# ---- Property 9: 上下文格式完整性 ----

from services.context_builder import ContextBuilder

# 生成带粒度分配的意群选择项策略
# 每个选择项为 {"group": SemanticGroup, "granularity": str, "tokens": int}
context_selection_strategy = st.builds(
    lambda group, granularity: {
        "group": group,
        "granularity": granularity,
        "tokens": 0,
    },
    group=semantic_group_strategy,
    granularity=st.sampled_from(["full", "digest", "summary"]),
)

# 非空的选择项列表策略（Property 9 要求非空）
non_empty_selections_strategy = st.lists(
    context_selection_strategy,
    min_size=1,
    max_size=10,
)


class TestProperty9_ContextFormatCompleteness:
    """
    **Feature: chatpdf-rag-optimization, Property 9: 上下文格式完整性**

    **Validates: Requirements 3.5**

    属性描述：For any 非空的粒度选择结果列表，构建的上下文字符串应包含
    每个意群的 group_id 标识、粒度级别标注文本和页码范围。
    """

    @given(selections=non_empty_selections_strategy)
    @settings(max_examples=100)
    def test_上下文包含每个意群的group_id(self, selections: list):
        """
        **Validates: Requirements 3.5**

        对任意非空的粒度选择结果列表，build_context 返回的上下文字符串
        应包含每个意群的 group_id 标识。
        """
        builder = ContextBuilder()
        context, citations = builder.build_context(selections)

        # 验证上下文非空
        assert len(context) > 0, "非空选择列表应产生非空上下文字符串"

        # 验证每个意群的 group_id 出现在上下文中
        for selection in selections:
            group_id = selection["group"].group_id
            assert group_id in context, (
                f"上下文字符串中未找到意群标识 {group_id!r}"
            )

    @given(selections=non_empty_selections_strategy)
    @settings(max_examples=100)
    def test_上下文包含每个意群的粒度级别标注(self, selections: list):
        """
        **Validates: Requirements 3.5**

        对任意非空的粒度选择结果列表，build_context 返回的上下文字符串
        应包含每个意群的粒度级别标注文本（full/digest/summary）。
        """
        builder = ContextBuilder()
        context, citations = builder.build_context(selections)

        # 验证每个意群的粒度级别标注出现在上下文中
        for selection in selections:
            granularity = selection["granularity"]
            group_id = selection["group"].group_id
            # 粒度标注应出现在对应意群的头部行中
            # 头部格式：[ref]【group_id - granularity | 页码: start-end】
            expected_fragment = f"{group_id} - {granularity}"
            assert expected_fragment in context, (
                f"上下文字符串中未找到意群 {group_id} 的粒度标注 "
                f"'{expected_fragment}'"
            )

    @given(selections=non_empty_selections_strategy)
    @settings(max_examples=100)
    def test_上下文包含每个意群的页码范围(self, selections: list):
        """
        **Validates: Requirements 3.5**

        对任意非空的粒度选择结果列表，build_context 返回的上下文字符串
        应包含每个意群的页码范围。
        """
        builder = ContextBuilder()
        context, citations = builder.build_context(selections)

        # 验证每个意群的页码范围出现在上下文中
        for selection in selections:
            page_start, page_end = selection["group"].page_range
            expected_page = f"页码: {page_start}-{page_end}"
            group_id = selection["group"].group_id
            assert expected_page in context, (
                f"上下文字符串中未找到意群 {group_id} 的页码范围 "
                f"'{expected_page}'"
            )

    @given(selections=non_empty_selections_strategy)
    @settings(max_examples=100)
    def test_citations列表与selections长度一致(self, selections: list):
        """
        **Validates: Requirements 3.5**

        对任意非空的粒度选择结果列表，build_context 返回的 citations
        列表长度应与输入 selections 长度一致，且每个 citation 包含
        正确的 ref、group_id 和 page_range。
        """
        builder = ContextBuilder()
        context, citations = builder.build_context(selections)

        # 验证 citations 长度一致
        assert len(citations) == len(selections), (
            f"citations 长度 ({len(citations)}) 与 selections 长度 "
            f"({len(selections)}) 不一致"
        )

        # 验证每个 citation 的字段正确
        for idx, (selection, citation) in enumerate(zip(selections, citations)):
            expected_ref = idx + 1
            expected_group_id = selection["group"].group_id
            expected_page_range = list(selection["group"].page_range)

            assert citation["ref"] == expected_ref, (
                f"citation[{idx}] 的 ref ({citation['ref']}) "
                f"应为 {expected_ref}"
            )
            assert citation["group_id"] == expected_group_id, (
                f"citation[{idx}] 的 group_id ({citation['group_id']!r}) "
                f"应为 {expected_group_id!r}"
            )
            assert citation["page_range"] == expected_page_range, (
                f"citation[{idx}] 的 page_range ({citation['page_range']}) "
                f"应为 {expected_page_range}"
            )


# ---- Property 10: 正则搜索结果正确性 ----

import re

from services.advanced_search import AdvancedSearchService

# 安全的正则表达式模式策略：生成简单但有效的正则模式
# 避免生成可能导致灾难性回溯或零宽度匹配的模式
_safe_regex_literals = st.sampled_from([
    r"\d+",           # 数字序列
    r"[a-z]+",        # 小写字母序列
    r"[A-Z]+",        # 大写字母序列
    r"\w+",           # 单词字符序列
    r"[a-zA-Z]+",     # 字母序列
    r"\d{1,3}",       # 1-3 位数字
    r"[0-9]+\.[0-9]+",  # 小数
    r"[a-z]{2,5}",    # 2-5 个小写字母
])

# 包含可匹配内容的文本策略：确保文本非空且包含字母和数字
_text_with_content = st.text(
    min_size=5,
    max_size=2000,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
    ),
).filter(lambda t: any(c.isalnum() for c in t))


class TestProperty10_RegexSearchResultCorrectness:
    """
    **Feature: chatpdf-rag-optimization, Property 10: 正则搜索结果正确性**

    **Validates: Requirements 4.1**

    属性描述：For any 有效正则表达式和非空文本，regex_search 返回的
    每个结果的 match_text 应能被该正则表达式匹配。
    """

    @given(
        pattern=_safe_regex_literals,
        text=_text_with_content,
    )
    @settings(max_examples=100)
    def test_每个结果的match_text能被正则表达式匹配(
        self, pattern: str, text: str
    ):
        """
        **Validates: Requirements 4.1**

        对任意有效正则表达式和非空文本，regex_search 返回的每个结果的
        match_text 应能被该正则表达式完整匹配（即 re.fullmatch 或
        re.search 能在 match_text 中找到匹配）。
        """
        service = AdvancedSearchService()

        results = service.regex_search(pattern, text)

        # 编译正则表达式（与服务使用相同的标志）
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

        for i, result in enumerate(results):
            match_text = result["match_text"]

            # 验证 match_text 非空
            assert len(match_text) > 0, (
                f"结果 {i} 的 match_text 为空，"
                f"模式: {pattern!r}，文本长度: {len(text)}"
            )

            # 验证 match_text 能被正则表达式匹配
            match = regex.search(match_text)
            assert match is not None, (
                f"结果 {i} 的 match_text {match_text!r} "
                f"无法被正则表达式 {pattern!r} 匹配"
            )

    @given(
        pattern=_safe_regex_literals,
        text=_text_with_content,
    )
    @settings(max_examples=100)
    def test_每个结果的match_offset正确指向原文(
        self, pattern: str, text: str
    ):
        """
        **Validates: Requirements 4.1**

        对任意有效正则表达式和非空文本，regex_search 返回的每个结果的
        match_offset 应正确指向原文中的匹配位置，即
        text[match_offset:match_offset+len(match_text)] == match_text。
        """
        service = AdvancedSearchService()

        results = service.regex_search(pattern, text)

        for i, result in enumerate(results):
            match_text = result["match_text"]
            match_offset = result["match_offset"]

            # 验证偏移量在文本范围内
            assert 0 <= match_offset <= len(text) - len(match_text), (
                f"结果 {i} 的 match_offset ({match_offset}) 超出文本范围，"
                f"文本长度: {len(text)}，match_text 长度: {len(match_text)}"
            )

            # 验证偏移量处的文本与 match_text 一致
            actual_text = text[match_offset:match_offset + len(match_text)]
            # 由于使用了 IGNORECASE 标志，需要忽略大小写比较
            assert actual_text.lower() == match_text.lower(), (
                f"结果 {i} 的 match_offset ({match_offset}) 处的文本 "
                f"{actual_text!r} 与 match_text {match_text!r} 不一致"
            )


# ---- Property 11: 布尔搜索结果排序 ----

# 布尔查询策略：生成包含 AND/OR/NOT 操作符的查询
# 使用常见的搜索词项组合，确保查询有意义
_boolean_terms = st.sampled_from([
    "学习", "网络", "方法", "模型", "数据",
    "分析", "系统", "算法", "研究", "技术",
    "deep", "learning", "network", "model", "data",
])

_boolean_operators = st.sampled_from(["AND", "OR", "NOT"])

# 生成布尔查询：2-4 个词项，用操作符连接
@st.composite
def boolean_query_strategy(draw):
    """生成有效的布尔查询表达式"""
    num_terms = draw(st.integers(min_value=1, max_value=3))
    terms = [draw(_boolean_terms) for _ in range(num_terms)]

    if num_terms == 1:
        return terms[0]

    # 用操作符连接词项
    parts = [terms[0]]
    for i in range(1, num_terms):
        op = draw(_boolean_operators)
        parts.append(op)
        parts.append(terms[i])

    return " ".join(parts)


# 包含多个搜索词项的文本策略：确保文本中包含一些常见词项
@st.composite
def text_with_boolean_terms_strategy(draw):
    """生成包含多个可搜索词项的文本"""
    # 基础文本片段，包含常见的搜索词项
    fragments = [
        "深度学习是一种机器学习方法",
        "卷积神经网络用于图像识别",
        "数据分析和模型训练是关键步骤",
        "自然语言处理技术不断发展",
        "强化学习算法在游戏中表现出色",
        "deep learning is a machine learning method",
        "neural network model for data analysis",
        "research on advanced algorithms and systems",
        "技术研究和系统设计需要大量数据",
        "学习方法和网络模型的对比分析",
    ]

    # 随机选择 2-5 个片段组合
    num_fragments = draw(st.integers(min_value=2, max_value=5))
    selected = [draw(st.sampled_from(fragments)) for _ in range(num_fragments)]

    # 用句号连接，确保词项之间有一定距离
    return "。".join(selected)


class TestProperty11_BooleanSearchResultOrdering:
    """
    **Feature: chatpdf-rag-optimization, Property 11: 布尔搜索结果排序**

    **Validates: Requirements 4.4**

    属性描述：For any 布尔查询和文本，boolean_search 返回的结果列表
    应按 score 字段降序排列。
    """

    @given(
        query=boolean_query_strategy(),
        text=text_with_boolean_terms_strategy(),
    )
    @settings(max_examples=100)
    def test_布尔搜索结果按score降序排列(
        self, query: str, text: str
    ):
        """
        **Validates: Requirements 4.4**

        对任意布尔查询和文本，boolean_search 返回的结果列表中，
        每个结果的 score 应 >= 下一个结果的 score（降序排列）。
        """
        service = AdvancedSearchService()

        results = service.boolean_search(query, text)

        # 验证结果按 score 降序排列
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"], (
                f"布尔搜索结果未按 score 降序排列: "
                f"results[{i}].score={results[i]['score']} < "
                f"results[{i + 1}].score={results[i + 1]['score']}，"
                f"查询: {query!r}"
            )

    @given(
        query=boolean_query_strategy(),
        text=text_with_boolean_terms_strategy(),
    )
    @settings(max_examples=100)
    def test_布尔搜索结果score为正数(
        self, query: str, text: str
    ):
        """
        **Validates: Requirements 4.4**

        对任意布尔查询和文本，boolean_search 返回的每个结果的
        score 应为正数（> 0）。
        """
        service = AdvancedSearchService()

        results = service.boolean_search(query, text)

        for i, result in enumerate(results):
            assert result["score"] > 0, (
                f"布尔搜索结果 {i} 的 score ({result['score']}) 应为正数，"
                f"查询: {query!r}"
            )


# ---- Property 12: 生成提示词匹配 ----

from services.preset_service import (
    get_generation_prompt,
    MINDMAP_SYSTEM_PROMPT,
    MERMAID_SYSTEM_PROMPT,
)

# 思维导图关键词策略：从已知的思维导图关键词中选择
_mindmap_keyword_strategy = st.sampled_from(["思维导图", "脑图", "mindmap", "mind map"])

# 流程图关键词策略：从已知的流程图关键词中选择
_flowchart_keyword_strategy = st.sampled_from(["流程图", "flowchart", "flow chart", "mermaid"])

# 前缀/后缀文本策略：生成随机的上下文文本包裹关键词
_surrounding_text_strategy = st.text(
    min_size=0,
    max_size=100,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
        whitelist_characters="，。、；：？！ 　"
    ),
)


@st.composite
def mindmap_query_strategy(draw):
    """生成包含思维导图关键词的查询字符串"""
    prefix = draw(_surrounding_text_strategy)
    keyword = draw(_mindmap_keyword_strategy)
    suffix = draw(_surrounding_text_strategy)
    return f"{prefix}{keyword}{suffix}"


@st.composite
def flowchart_query_strategy(draw):
    """生成包含流程图关键词的查询字符串"""
    prefix = draw(_surrounding_text_strategy)
    keyword = draw(_flowchart_keyword_strategy)
    suffix = draw(_surrounding_text_strategy)
    return f"{prefix}{keyword}{suffix}"


class TestProperty12_GenerationPromptMatching:
    """
    **Feature: chatpdf-rag-optimization, Property 12: 生成提示词匹配**

    **Validates: Requirements 5.3, 5.4**

    属性描述：For any 包含"思维导图"关键词的查询，get_generation_prompt
    应返回非 None 的思维导图提示词；For any 包含"流程图"关键词的查询，
    应返回非 None 的 Mermaid 提示词。
    """

    @given(query=mindmap_query_strategy())
    @settings(max_examples=100)
    def test_包含思维导图关键词的查询返回非None的思维导图提示词(self, query: str):
        """
        **Validates: Requirements 5.3**

        对任意包含思维导图关键词（"思维导图"、"脑图"、"mindmap"、"mind map"）
        的查询字符串，get_generation_prompt 应返回非 None 的思维导图提示词，
        且返回值应等于 MINDMAP_SYSTEM_PROMPT。
        """
        # 过滤掉同时包含流程图关键词的查询，避免关键词冲突
        # （当两类关键词同时存在时，get_generation_prompt 按位置优先匹配）
        query_lower = query.lower()
        _flowchart_kws = ["流程图", "flowchart", "flow chart", "mermaid"]
        assume(not any(kw in query_lower for kw in _flowchart_kws))

        result = get_generation_prompt(query)

        # 核心属性：返回值非 None
        assert result is not None, (
            f"包含思维导图关键词的查询 {query!r} "
            f"应返回非 None 的提示词，但返回了 None"
        )

        # 验证返回的是思维导图提示词
        assert result == MINDMAP_SYSTEM_PROMPT, (
            f"包含思维导图关键词的查询 {query!r} "
            f"应返回 MINDMAP_SYSTEM_PROMPT，但返回了不同的提示词"
        )

    @given(query=flowchart_query_strategy())
    @settings(max_examples=100)
    def test_包含流程图关键词的查询返回非None的Mermaid提示词(self, query: str):
        """
        **Validates: Requirements 5.4**

        对任意包含流程图关键词（"流程图"、"flowchart"、"flow chart"、"mermaid"）
        的查询字符串，get_generation_prompt 应返回非 None 的 Mermaid 提示词，
        且返回值应等于 MERMAID_SYSTEM_PROMPT。
        """
        # 过滤掉同时包含思维导图关键词的查询，避免关键词冲突
        # （当两类关键词同时存在时，get_generation_prompt 按位置优先匹配）
        query_lower = query.lower()
        _mindmap_kws = ["思维导图", "脑图", "mindmap", "mind map"]
        assume(not any(kw in query_lower for kw in _mindmap_kws))

        result = get_generation_prompt(query)

        # 核心属性：返回值非 None
        assert result is not None, (
            f"包含流程图关键词的查询 {query!r} "
            f"应返回非 None 的提示词，但返回了 None"
        )

        # 验证返回的是 Mermaid 提示词
        assert result == MERMAID_SYSTEM_PROMPT, (
            f"包含流程图关键词的查询 {query!r} "
            f"应返回 MERMAID_SYSTEM_PROMPT，但返回了不同的提示词"
        )
