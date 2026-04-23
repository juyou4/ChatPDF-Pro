from services.query_analyzer import (
    analyze_evidence_need,
    analyze_query_type,
    get_dynamic_top_k,
    get_retrieval_strategy,
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
