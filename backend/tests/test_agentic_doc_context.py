import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from routes import chat_routes
from services.chat_intent_service import IntentDecision, prepare_chat_intent
from services.embedding_service import RAG_INDEX_VERSION
from services.query_analyzer import get_retrieval_strategy


def _intent(
    *,
    query_type: str,
    evidence_need: Sequence[str] = (),
    **overrides,
) -> IntentDecision:
    """按当前契约构造一个冻结意图。

    ``_build_agent_retrieval_gate`` 现在是纯消费者：语义判定一律读自已冻结的
    ``IntentDecision``，不再接受散装的 query_type / evidence_need 参数。这里先走
    真实构造路径拿到一个合法决策，再只覆盖被测字段，避免手工拼一个可能与真实
    产物不一致的 dataclass。

    与视觉/清单/全文总结相关的字段显式归零，让每个用例只被自己声明的维度影响。
    """
    base = prepare_chat_intent(original_question="用于构造意图的占位问句", enable_agent=True)
    return replace(
        base,
        query_type=query_type,
        evidence_need=tuple(evidence_need),
        modalities=overrides.pop("modalities", ()),
        visual_intent=overrides.pop("visual_intent", False),
        inventory_kinds=overrides.pop("inventory_kinds", ()),
        full_document_summary=overrides.pop("full_document_summary", False),
        **overrides,
    )


def test_agent_doc_context_loads_chunks_and_groups(tmp_path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / "data"
    vector_store_dir = data_dir / "vector_stores"
    semantic_groups_dir = data_dir / "semantic_groups"
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    semantic_groups_dir.mkdir(parents=True, exist_ok=True)

    doc_id = "doc-1"
    # 索引与语义组都必须带上解析身份，否则会被准入闸门当成 stale 产物丢弃：
    # 同一解析代次内的解析器修复也会重建块树，只比对 doc_id 无法证明产物是当前的。
    generation = "gen-test-1"
    source_hash = "hash-test-1"
    with open(vector_store_dir / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": ["第一段内容", "第二段内容"],
                "embedding_model": "local-minilm",
                "index_version": RAG_INDEX_VERSION,
                "parse_generation": generation,
                "document_source_hash": source_hash,
            },
            f,
        )

    with open(semantic_groups_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "doc_id": doc_id,
                "parse_generation": generation,
                "document_source_hash": source_hash,
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
            "parse_manifest": {
                "generation": generation,
                "source_hash": source_hash,
            },
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

    # 语义组还没被加载：准入闸门要求向量索引携带完整的语义代次身份
    # （vector_build_id / embedding_identity_version / embedding_provider /
    # embedding_api_host / vector_dimension），并且语义组必须放在已发布代次的目录
    # 布局里（active/ 清单 + generations/<doc>/<gen>/），而不是根目录的旧式扁平文件。
    # 这份夹具只补齐了解析身份，因此这里断言的是"被安全丢弃"而非"加载成功"。
    # 补一个共享的 published-generation 夹具后，应当改回断言真实加载。
    assert ctx.semantic_groups == []


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


def test_agent_gate_opens_only_for_high_value_routes():
    def enabled(query_type: str, evidence_need: Sequence[str]) -> bool:
        gate = chat_routes._build_agent_retrieval_gate(
            _intent(query_type=query_type, evidence_need=evidence_need),
            enable_agent_retrieval=True,
            selected_text=None,
        )
        return bool(gate["enabled"])

    assert enabled("overview", []) is True
    assert enabled("specific", ["section_explanation"]) is True
    assert enabled("analytical", ["comparison_multi_aspect"]) is True
    assert enabled("specific", ["reference_meta"]) is True
    # 数值表格走确定性检索，Agent 的开放式取材反而会稀释精确证据。
    assert enabled("extraction", ["numeric_table"]) is False


def test_agent_gate_respects_switch_and_selected_text():
    switched_off = chat_routes._build_agent_retrieval_gate(
        _intent(query_type="overview"),
        enable_agent_retrieval=False,
        selected_text=None,
    )
    assert switched_off["enabled"] is False

    with_selection = chat_routes._build_agent_retrieval_gate(
        _intent(query_type="overview", evidence_need=["section_explanation"]),
        enable_agent_retrieval=True,
        selected_text="用户框选内容",
    )
    assert with_selection["enabled"] is False


def test_build_agent_retrieval_gate_reports_match_reason():
    gate = chat_routes._build_agent_retrieval_gate(
        _intent(query_type="specific", evidence_need=["reference_meta"]),
        enable_agent_retrieval=True,
        selected_text=None,
    )

    assert gate["enabled"] is True
    assert gate["reason"] == "matched_evidence_need"
    assert gate["matched_query_type"] is None
    assert gate["matched_evidence_need"] == ["reference_meta"]
    assert gate["selected_text_present"] is False


def test_build_agent_retrieval_gate_reports_why_agent_is_disabled():
    gate = chat_routes._build_agent_retrieval_gate(
        _intent(query_type="overview", evidence_need=["section_explanation"]),
        enable_agent_retrieval=False,
        selected_text="用户框选内容",
    )

    assert gate["enabled"] is False
    assert gate["reason"] == "switch_disabled"
    assert gate["matched_query_type"] == "overview"
    assert gate["matched_evidence_need"] == ["section_explanation"]
    assert gate["selected_text_present"] is True


def test_annotate_agent_gate_marks_consistent_enabled_path():
    gate = chat_routes._build_agent_retrieval_gate(
        _intent(query_type="overview"),
        enable_agent_retrieval=True,
        selected_text=None,
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
        _intent(query_type="extraction", evidence_need=["numeric_table"]),
        enable_agent_retrieval=True,
        selected_text=None,
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
            # 精确表格抽取有确定性检索与视觉核验通路，次要的分析型标签不得绕过它，
            # 因此 gate 给出比 route_not_matched 更具体的拒绝原因。
            "numeric_table_exactness",
            False,
        ),
    ]

    for query, expected_signal, expected_reason, expected_enabled in cases:
        strategy = get_retrieval_strategy(query)
        # 走真实的意图构造路径，顺带守住「策略层与 gate 读的是同一份判定」这条契约。
        intent = prepare_chat_intent(original_question=query, enable_agent=True)
        gate = chat_routes._build_agent_retrieval_gate(
            intent,
            enable_agent_retrieval=True,
            selected_text=None,
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
