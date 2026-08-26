import pytest

from services.query_analyzer import (
    analyze_evidence_need,
    analyze_query_type,
    get_dynamic_top_k,
    get_retrieval_strategy,
    is_code_implementation_query,
    is_method_implementation_query,
    is_section_explanation_query,
)


def test_method_section_explanation_queries_use_analytical_strategy():
    query = "详细讲解方法部分"

    assert is_section_explanation_query(query) is True
    assert analyze_query_type(query) == "analytical"
    assert get_dynamic_top_k(query) == 16


def test_plain_detail_extraction_queries_remain_extraction():
    query = "请给我具体数据和数值"

    assert is_section_explanation_query(query) is False
    assert analyze_query_type(query) == "extraction"
    assert get_dynamic_top_k(query) == 5


def test_numeric_table_queries_expose_evidence_need_and_higher_top_k():
    query = "ImageNet-LT 上 DiffuLT 相比第二好的方法在 Many/Medium/Few 上分别提升多少？"

    assert "numeric_table" in analyze_evidence_need(query)
    assert analyze_query_type(query) == "extraction"
    assert get_dynamic_top_k(query) == 8


def test_cost_queries_expose_numeric_table_evidence_need_and_higher_top_k():
    query = "这篇论文的额外开销、推理时间和 FLOPs 分别是多少？"

    assert "numeric_table" in analyze_evidence_need(query)
    assert analyze_query_type(query) == "extraction"
    assert get_dynamic_top_k(query) == 8


def test_cost_queries_are_treated_as_numeric_table_queries():
    query = "DiffuLT 的训练时间和额外推理开销分别是多少？"

    strategy = get_retrieval_strategy(query)

    assert "numeric_table" in strategy["evidence_need"]
    assert strategy["query_type"] == "extraction"
    assert strategy["top_k"] == 8


def test_reference_trap_queries_are_detected():
    query = "这篇论文引用了哪些关于扩散模型的经典工作？"

    strategy = get_retrieval_strategy(query)

    assert "reference_trap" in strategy["evidence_need"]


def test_section_explanation_queries_expose_evidence_need():
    query = "请详细讲解方法部分的设计"

    strategy = get_retrieval_strategy(query)

    assert "section_explanation" in strategy["evidence_need"]


def test_reference_meta_queries_expose_reference_meta_evidence_need():
    query = "这篇论文的第一作者和通讯作者分别是谁？他们来自哪个机构？"

    strategy = get_retrieval_strategy(query)

    assert "reference_meta" in strategy["evidence_need"]


def test_comparison_multi_aspect_queries_expose_comparison_evidence_need():
    query = "DiffuLT 与传统重采样方法相比，在多个方面分别有哪些优势和劣势？"

    strategy = get_retrieval_strategy(query)

    assert "comparison_multi_aspect" in strategy["evidence_need"]


@pytest.mark.parametrize(
    "query",
    [
        "这篇论文的训练脚本在仓库哪",
        "这个方法的官方实现开源地址是什么",
        "仓库里的 loss 是怎么写的",
        "哪份配置文件定义了学习率",
        "where is the training script in the official repo",
        "What is the official implementation of this method?",
    ],
)
def test_code_implementation_queries_expose_code_evidence_need(query):
    assert is_code_implementation_query(query) is True
    assert "code_implementation" in analyze_evidence_need(query)


@pytest.mark.parametrize(
    "query",
    [
        # 只是提到 github/链接：属于 reference_trap / reference_meta 的地盘。
        "参考文献里的 GitHub 链接是什么",
        "这篇论文的 arXiv 链接和 DOI 是什么",
        # 纯机制解释题由 section_explanation 负责，不应触发仓库工具。
        "这个方法怎么实现的",
        "请详细讲解方法部分的设计",
        # 表格精确数值必须保住 numeric_table 主路径。
        "表 3 里各方法的准确率分别是多少",
    ],
)
def test_non_code_queries_do_not_claim_code_implementation(query):
    assert "code_implementation" not in analyze_evidence_need(query)


def test_reference_link_question_still_routes_to_reference_needs():
    needs = analyze_evidence_need("参考文献里的 GitHub 链接是什么")

    assert "reference_trap" in needs
    assert "reference_meta" in needs


def test_explicit_table_numeric_query_keeps_table_route_over_code():
    needs = analyze_evidence_need("表 2 里 F1 的数值是多少，源码里是怎么算的")

    assert "numeric_table" in needs
    assert "code_implementation" not in needs


@pytest.mark.parametrize(
    "query",
    [
        "对照源码讲一下",
        "详细讲解实现",
        "代码和论文是否一致",
        "这段代码如何对应论文里的公式",
        "walk me through the implementation",
        "does the code match the paper",
    ],
)
def test_walkthrough_requests_are_code_implementation(query):
    """用户明确要求对照源码讲实现时，不该再等一个制品词才启用仓库工具。"""
    assert is_code_implementation_query(query) is True
    assert "code_implementation" in analyze_evidence_need(query)


@pytest.mark.parametrize(
    "query",
    [
        "这个方法怎么实现的",
        "该模块是如何实现的",
        "How is this method implemented?",
    ],
)
def test_method_implementation_shape_is_recognised_but_not_claimed(query):
    """句型识别归识别，是否读代码由「论文有没有可读仓库」决定。"""
    assert is_method_implementation_query(query) is True
    assert "code_implementation" not in analyze_evidence_need(query)


@pytest.mark.parametrize(
    "query",
    [
        # 参考文献里的链接问题即使换成实现句式也不能抢走 reference 路径。
        "参考文献里的 GitHub 链接是什么",
        "references 部分列出的实现是怎么实现的",
        # 显式表格数值题继续归 numeric_table。
        "表 3 里各方法的准确率分别是多少，这个方法怎么实现的",
        # 纯背景题不属于实现句型。
        "这个方法的动机是什么",
    ],
)
def test_method_implementation_shape_excludes_reference_and_table_queries(query):
    assert is_method_implementation_query(query) is False
