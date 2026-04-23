import json
import pickle
from pathlib import Path

from routes import chat_routes
from services.query_analyzer import get_retrieval_strategy


def test_agent_doc_context_loads_chunks_and_groups(tmp_path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / "data"
    vector_store_dir = data_dir / "vector_stores"
    semantic_groups_dir = data_dir / "semantic_groups"
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    semantic_groups_dir.mkdir(parents=True, exist_ok=True)

    doc_id = "doc-1"
    with open(vector_store_dir / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": ["第一段内容", "第二段内容"],
                "embedding_model": "local-minilm",
            },
            f,
        )

    with open(semantic_groups_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "doc_id": doc_id,
                "groups": [
                    {
                        "group_id": "group-0",
                        "chunk_indices": [0, 1],
                        "char_count": 6,
                        "summary": "摘要",
                        "digest": "精要",
                        "full_text": "完整文本",
                        "keywords": ["关键词"],
                        "page_range": [1, 2],
                        "summary_status": "ok",
                        "llm_meta": None,
                    }
                ],
            },
            f,
            ensure_ascii=False,
        )

    monkeypatch.setattr(chat_routes, "_get_project_root", lambda: project_root)

    doc = {
        "data": {
            "full_text": "第一段内容。第二段内容。",
            "pages": [{"page": 1, "content": "第一段内容。"}, {"page": 2, "content": "第二段内容。"}],
        }
    }

    ctx = chat_routes._build_agent_doc_context(
        doc_id,
        doc,
        str(vector_store_dir),
        api_key="test-key",
    )

    assert ctx.doc_id == doc_id
    assert ctx.chunks == ["第一段内容", "第二段内容"]
    assert len(ctx.semantic_groups) == 1
    assert ctx.semantic_groups[0]["group_id"] == "group-0"
    assert ctx.semantic_groups[0]["page_range"] == [1, 2]


def test_agent_doc_context_falls_back_to_paragraphs(tmp_path, monkeypatch):
    project_root = tmp_path
    (project_root / "data" / "vector_stores").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_routes, "_get_project_root", lambda: project_root)

    doc_id = "doc-2"
    doc = {
        "data": {
            "full_text": "第一段。\n\n第二段。",
            "pages": [{"page": 1, "content": "第一段。"}],
        }
    }

    chunks = chat_routes._load_doc_chunks_for_agent(
        doc_id,
        str(project_root / "data" / "vector_stores"),
        doc["data"]["full_text"],
    )

    assert chunks == ["第一段。\n\n第二段。"]
    assert chat_routes._load_doc_semantic_groups_for_agent(doc_id) == []


def test_should_enable_agent_retrieval_only_for_high_value_routes():
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="overview",
        evidence_need=[],
    ) is True
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="specific",
        evidence_need=["section_explanation"],
    ) is True
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="analytical",
        evidence_need=["comparison_multi_aspect"],
    ) is True
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="specific",
        evidence_need=["reference_meta"],
    ) is True
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="extraction",
        evidence_need=["numeric_table"],
    ) is False


def test_should_enable_agent_retrieval_respects_switch_and_selected_text():
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=False,
        selected_text=None,
        query_type="overview",
        evidence_need=[],
    ) is False
    assert chat_routes._should_enable_agent_retrieval(
        enable_agent_retrieval=True,
        selected_text="用户框选内容",
        query_type="overview",
        evidence_need=["section_explanation"],
    ) is False


def test_build_agent_retrieval_gate_reports_match_reason():
    gate = chat_routes._build_agent_retrieval_gate(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="specific",
        evidence_need=["reference_meta"],
    )

    assert gate["enabled"] is True
    assert gate["reason"] == "matched_evidence_need"
    assert gate["matched_query_type"] is None
    assert gate["matched_evidence_need"] == ["reference_meta"]
    assert gate["selected_text_present"] is False


def test_build_agent_retrieval_gate_reports_why_agent_is_disabled():
    gate = chat_routes._build_agent_retrieval_gate(
        enable_agent_retrieval=False,
        selected_text="用户框选内容",
        query_type="overview",
        evidence_need=["section_explanation"],
    )

    assert gate["enabled"] is False
    assert gate["reason"] == "switch_disabled"
    assert gate["matched_query_type"] == "overview"
    assert gate["matched_evidence_need"] == ["section_explanation"]
    assert gate["selected_text_present"] is True


def test_annotate_agent_gate_marks_consistent_enabled_path():
    gate = chat_routes._build_agent_retrieval_gate(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="overview",
        evidence_need=[],
    )

    annotated = chat_routes._annotate_agent_gate(
        gate,
        use_agent=True,
        agent_mode=True,
        search_query_passthrough=True,
    )

    assert annotated["enabled"] is True
    assert annotated["use_agent"] is True
    assert annotated["agent_mode"] is True
    assert annotated["search_query_passthrough"] is True
    assert annotated["consistency_ok"] is True


def test_annotate_agent_gate_marks_consistent_disabled_path():
    gate = chat_routes._build_agent_retrieval_gate(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type="extraction",
        evidence_need=["numeric_table"],
    )

    annotated = chat_routes._annotate_agent_gate(
        gate,
        use_agent=False,
        agent_mode=False,
        search_query_passthrough=False,
    )

    assert annotated["enabled"] is False
    assert annotated["use_agent"] is False
    assert annotated["agent_mode"] is False
    assert annotated["search_query_passthrough"] is False
    assert annotated["consistency_ok"] is True


def test_strategy_and_agent_gate_stay_consistent_for_representative_queries():
    cases = [
        (
            "用三句话概括 DiffuLT 论文解决了什么问题、用了什么方法、取得了什么结果。",
            "overview",
            "matched_query_type",
            True,
        ),
        (
            "请详细解释论文 Method 章节的完整流程，包括每个子模块的输入输出和作用。",
            "section_explanation",
            "matched_evidence_need",
            True,
        ),
        (
            "这篇论文的第一作者和通讯作者分别是谁？他们来自哪个机构？",
            "reference_meta",
            "matched_evidence_need",
            True,
        ),
        (
            "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？",
            "numeric_table",
            "route_not_matched",
            False,
        ),
    ]

    for query, expected_signal, expected_reason, expected_enabled in cases:
        strategy = get_retrieval_strategy(query)
        gate = chat_routes._build_agent_retrieval_gate(
            enable_agent_retrieval=True,
            selected_text=None,
            query_type=strategy["query_type"],
            evidence_need=strategy["evidence_need"],
        )
        annotated = chat_routes._annotate_agent_gate(
            gate,
            use_agent=expected_enabled,
            agent_mode=expected_enabled,
            search_query_passthrough=expected_enabled,
        )

        if expected_signal == "overview":
            assert strategy["query_type"] == "overview"
            assert gate["matched_query_type"] == "overview"
        elif expected_signal == "numeric_table":
            assert "numeric_table" in strategy["evidence_need"]
            assert gate["matched_evidence_need"] == []
        else:
            assert expected_signal in strategy["evidence_need"]
            assert expected_signal in gate["matched_evidence_need"]
        assert gate["reason"] == expected_reason
        assert gate["enabled"] is expected_enabled
        assert annotated["consistency_ok"] is True
