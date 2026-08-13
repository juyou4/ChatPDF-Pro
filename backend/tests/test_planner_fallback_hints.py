"""工具失败显式降级建议（suggested_next_tool）与 planner hint 注入测试。"""

from services.retrieval_agent import (
    RetrievalAgent,
    _compute_planner_hints,
    _format_tool_fallback_hint,
)


def _make_agent() -> RetrievalAgent:
    return RetrievalAgent(api_key="", model="test-model", provider="openai")


def test_format_tool_fallback_hint_dedupes_and_caps():
    suggestions = [
        {"tool": "vector_search", "suggested_next_tool": "keyword_search"},
        {"tool": "vector_search", "suggested_next_tool": "keyword_search"},
        {"tool": "grep", "suggested_next_tool": "search_document"},
        {"tool": "regex_search", "suggested_next_tool": "grep"},
        {"tool": "boolean_search", "suggested_next_tool": "keyword_search"},
    ]
    hint = _format_tool_fallback_hint(suggestions)
    assert "vector_search 失败或无命中，建议改用 keyword_search" in hint
    # 去重后最多保留 3 条
    assert hint.count("建议改用") == 3


def test_format_tool_fallback_hint_skips_self_suggestion():
    assert _format_tool_fallback_hint([{"tool": "grep", "suggested_next_tool": "grep"}]) == ""
    assert _format_tool_fallback_hint([]) == ""
    assert _format_tool_fallback_hint(None) == ""


def test_compute_planner_hints_includes_fallback_after_first_round():
    suggestions = [{"tool": "vector_search", "suggested_next_tool": "keyword_search"}]
    hints = _compute_planner_hints(
        round_idx=1,
        max_rounds=3,
        last_round_calls=[{"tool": "vector_search", "query": "q"}],
        last_round_total_hits=0,
        duplicate_detected=False,
        sufficiency_level="",
        tool_fallback_suggestions=suggestions,
    )
    assert any("建议改用 keyword_search" in hint for hint in hints)


def test_compute_planner_hints_no_fallback_on_first_round():
    suggestions = [{"tool": "vector_search", "suggested_next_tool": "keyword_search"}]
    hints = _compute_planner_hints(
        round_idx=0,
        max_rounds=3,
        last_round_calls=[],
        last_round_total_hits=0,
        duplicate_detected=False,
        sufficiency_level="",
        tool_fallback_suggestions=suggestions,
    )
    assert not any("建议改用" in hint for hint in hints)


def test_tool_issue_uses_static_fallback_mapping():
    agent = _make_agent()
    issue = agent._tool_issue_from_result(
        "vector_search",
        "query",
        {"error": "index broken", "error_code": "retrieval_error"},
        result_count=0,
    )
    assert issue is not None
    assert issue["suggested_next_tool"] == "keyword_search"


def test_tool_issue_prefers_result_suggestion():
    agent = _make_agent()
    issue = agent._tool_issue_from_result(
        "academic_search",
        "query",
        {
            "error": "network down",
            "error_code": "academic_search_failed",
            "suggested_next_tool": "web_search",
        },
        result_count=0,
    )
    assert issue is not None
    assert issue["suggested_next_tool"] == "web_search"


def test_tool_issue_zero_hit_gets_suggestion():
    agent = _make_agent()
    issue = agent._tool_issue_from_result(
        "grep",
        "query",
        {"error_code": "no_relevant_chunks"},
        result_count=0,
    )
    assert issue is not None
    assert issue["suggested_next_tool"] == "search_document"


def test_tool_issue_absent_for_clean_result():
    agent = _make_agent()
    issue = agent._tool_issue_from_result(
        "grep",
        "query",
        {"results": ["hit"]},
        result_count=None,
    )
    assert issue is None
