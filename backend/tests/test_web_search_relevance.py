"""联网搜索相关性优化测试"""
import sys
import os

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.web_search_service import SearchManager
from routes.chat_routes import _build_web_search_query


def test_web_search_rerank_filters_off_topic_results():
    query = "OPDR 物理鲁棒性 对抗攻击 方法"
    results = [
        {
            "title": "OPDR 对抗攻击方法：物理鲁棒性提升",
            "url": "https://example.org/opdr",
            "snippet": "该方法通过全局光照建模提高攻击稳定性。",
        },
        {
            "title": "菊池桃子 - Wikipedia",
            "url": "https://ja.wikipedia.org/wiki/%E8%8F%8A%E6%B1%A0%E6%A1%83%E5%AD%90",
            "snippet": "日本の女優・歌手。",
        },
    ]

    filtered = SearchManager._rerank_and_filter_results(query, results)
    assert filtered
    assert filtered[0]["title"].startswith("OPDR")
    assert all("菊池桃子" not in item["title"] for item in filtered)


def test_web_search_rerank_returns_empty_when_all_results_irrelevant():
    query = "对抗攻击 物理鲁棒性"
    results = [
        {"title": "Celebrity News", "url": "https://example.com/news", "snippet": "latest music and movies"},
        {"title": "Travel Blog", "url": "https://example.com/travel", "snippet": "food and city guides"},
    ]
    filtered = SearchManager._rerank_and_filter_results(query, results)
    assert filtered == []


def test_build_web_search_query_uses_rewritten_query_and_doc_title():
    query = _build_web_search_query(
        base_query="这个方法有什么创新",
        original_question="这个方法有什么创新",
        doc_title="OPDR_paper.pdf",
        selected_text="本文提出OPDR，通过全局光照建模提升物理鲁棒性。",
    )
    assert "OPDR_paper" in query
    assert ".pdf" not in query.lower()
    assert "OPDR" in query


def test_build_web_search_query_skips_reference_like_selected_text():
    query = _build_web_search_query(
        base_query="这个方法是什么",
        original_question="这个方法是什么",
        doc_title="robust_method.pdf",
        selected_text="[1] Momoko Kikuchi. 2021. Example paper. [2] Another Author. 2022.",
    )
    # 不应把参考文献人名注入联网检索 query
    assert "Momoko" not in query
    assert "Kikuchi" not in query
