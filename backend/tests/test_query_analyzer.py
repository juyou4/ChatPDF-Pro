from services.query_analyzer import analyze_query_type, get_dynamic_top_k, is_section_explanation_query


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
