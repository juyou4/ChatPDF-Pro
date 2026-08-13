"""程序记忆（procedural memory）读写与失效行为测试（P1-3）。"""

import time

from services import procedural_memory_service as pm


def _patch_store(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "_store_dir", lambda: tmp_path / "procedural_memory")


def test_normalize_tool_sequence_dedupes_and_excludes_control():
    tools = ["search_document", "search_document", "fetch", "complete", "fetch", "map"]
    assert pm.normalize_tool_sequence(tools) == ["search_document", "fetch", "map"]
    # "fetch, fetch" 中间隔了 complete，complete 剔除后相邻去重生效
    assert pm.normalize_tool_sequence([]) == []
    assert pm.normalize_tool_sequence(["complete"]) == []


def test_record_and_suggest_requires_two_successes(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    doc_id = "doc-123"
    tools = ["search_document", "fetch"]

    assert pm.record_successful_strategy(doc_id, "numeric_table", tools, question="表3 最大值")
    # 单次成功不足以构成策略
    assert pm.suggest_strategy(doc_id, "numeric_table") == ""

    assert pm.record_successful_strategy(doc_id, "numeric_table", tools, question="表3 最大值")
    hint = pm.suggest_strategy(doc_id, "numeric_table")
    assert "numeric_table" in hint
    assert "search_document → fetch" in hint
    assert "不强制" in hint or "自行判断" in hint


def test_suggest_strategy_isolated_by_query_type_and_doc(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    tools = ["search_document", "fetch"]
    for _ in range(2):
        pm.record_successful_strategy("doc-a", "numeric_table", tools)
    assert pm.suggest_strategy("doc-a", "overview") == ""
    assert pm.suggest_strategy("doc-b", "numeric_table") == ""


def test_expired_entries_are_ignored(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    doc_id = "doc-exp"
    tools = ["grep", "read_around"]
    for _ in range(2):
        pm.record_successful_strategy(doc_id, "general", tools)
    assert pm.suggest_strategy(doc_id, "general") != ""

    # 时间快进 31 天后条目过期
    real_time = time.time
    monkeypatch.setattr(pm.time, "time", lambda: real_time() + 31 * 24 * 3600)
    assert pm.suggest_strategy(doc_id, "general") == ""


def test_strategies_capped_and_sorted_by_count(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    doc_id = "doc-cap"
    winner = ["search_document", "fetch"]
    for _ in range(3):
        pm.record_successful_strategy(doc_id, "general", winner)
    for seq in (["grep"], ["map"], ["visual_search"], ["read_section"]):
        pm.record_successful_strategy(doc_id, "general", seq)

    hint = pm.suggest_strategy(doc_id, "general")
    assert "search_document → fetch" in hint

    path = pm._doc_path(doc_id)
    payload = pm._load_payload(path)
    assert len(payload["strategies"]["general"]) <= 3


def test_record_rejects_empty_and_invalid_doc(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    assert pm.record_successful_strategy("doc-x", "general", []) is False
    assert pm.record_successful_strategy("", "general", ["grep"]) is False
    assert pm.suggest_strategy("", "general") == ""
