"""数值表格视觉校验在聊天路由中的证据边界回归测试。

视觉校验是结构化表格检索后的兜底诊断。只有已确认的结果可以进入
回答上下文和引用；冲突或无法判定的结果必须只保留在诊断中。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes


def _build_request() -> chat_routes.ChatRequest:
    return chat_routes.ChatRequest(
        doc_id="doc-visual-boundary",
        question="Table 1 中 Ours 的 Accuracy 是多少？",
        api_key="test-visual-key",
        model="gpt-4o",
        api_provider="openai",
        custom_params={"numeric_table_visual_background": False},
    )


def _base_table_segment() -> dict:
    return {
        "ref": 1,
        "text": "Table 1: Results\\nMethod | Accuracy\\nOurs | 91.2",
        "chunk_type": "table_row",
        "table_id": "Table 1",
        "table_bundle_id": "bundle-table-1",
        "table_caption": "Table 1: Results",
        "table_header": "Method | Accuracy",
        "numeric_table_exact_context_row_text": "Ours | 91.2",
        "context_id": "bundle-table-1:row:2",
        "evidence_id": "bundle-table-1:row:2",
        "page_range": [3, 3],
        "cell_evidence_units": [
            {"row": "Ours", "column": "Method", "content": "Ours"},
            {"row": "Ours", "column": "Accuracy", "content": "91.2"},
        ],
    }


def _base_retrieval_meta() -> dict:
    table_segment = _base_table_segment()
    return {
        "search_query": "Table 1 中 Ours 的 Accuracy 是多少？",
        "evidence_need": ["numeric_table"],
        "_context_segments": [table_segment],
        "citations": [
            {
                "ref": 1,
                "source_text": table_segment["text"],
                "display_text": table_segment["text"],
                "context_segment_text": table_segment["text"],
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "context_id": "bundle-table-1:row:2",
                "evidence_id": "bundle-table-1:row:2",
                "page_range": [3, 3],
            }
        ],
    }


def _visual_segment(verdict: str) -> dict:
    return {
        "text": (
            "[Numeric Table Visual Verification]\\n"
            "Table ID: Table 1\\n"
            "Matched Row: Ours\\n"
            "Visual Cells: Accuracy = 99.9"
        ),
        "segment_role": "numeric_table_visual_verification",
        "visual_verdict": verdict,
        "visual_cells": {"Accuracy": "99.9"},
        "table_id": "Table 1",
        "table_instance_id": "table-1-page-3",
        "context_id": "table-1-page-3:visual-verification",
        "evidence_id": "table-1-page-3:visual-verification",
        "page_range": [3, 3],
    }


@pytest.mark.asyncio
async def test_confirmed_visual_verification_is_added_to_answer_context_and_citations(monkeypatch):
    retrieval_meta = _base_retrieval_meta()

    async def fake_verify(**_kwargs):
        return _visual_segment("confirmed"), {
            "enabled": True,
            "triggered": True,
            "visual_verdict": "confirmed",
            "status": "confirmed",
        }

    monkeypatch.setattr(chat_routes, "maybe_verify_numeric_table_visual", fake_verify)

    await chat_routes._maybe_add_numeric_table_visual_verification(
        request=_build_request(),
        doc={"data": {}},
        retrieval_meta=retrieval_meta,
        query="Table 1 中 Ours 的 Accuracy 是多少？",
        evidence_need=["numeric_table"],
    )

    context_segments = chat_routes._build_response_context_segments(retrieval_meta)
    assert any(
        segment.get("segment_role") == "numeric_table_visual_verification"
        and "Numeric Table Visual Verification" in segment.get("text", "")
        for segment in context_segments
    )
    assert any(
        "Numeric Table Visual Verification" in citation.get("source_text", "")
        for citation in retrieval_meta["citations"]
    )
    assert retrieval_meta["diagnostics"]["numeric_table_visual_verification"]["visual_verdict"] == "confirmed"


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["conflict", "indeterminate"])
async def test_unconfirmed_visual_verification_stays_out_of_answer_context_and_citations(monkeypatch, verdict):
    retrieval_meta = _base_retrieval_meta()

    async def fake_verify(**_kwargs):
        # Simulate a defensive boundary failure in the verifier: the route must
        # still refuse a non-confirmed segment even when one is returned.
        return _visual_segment(verdict), {
            "enabled": True,
            "triggered": True,
            "visual_verdict": verdict,
            "status": verdict,
            "reason": "visual_and_structured_evidence_disagree",
        }

    monkeypatch.setattr(chat_routes, "maybe_verify_numeric_table_visual", fake_verify)

    await chat_routes._maybe_add_numeric_table_visual_verification(
        request=_build_request(),
        doc={"data": {}},
        retrieval_meta=retrieval_meta,
        query="Table 1 中 Ours 的 Accuracy 是多少？",
        evidence_need=["numeric_table"],
    )

    context_segments = chat_routes._build_response_context_segments(retrieval_meta)
    assert not any(
        segment.get("segment_role") == "numeric_table_visual_verification"
        for segment in retrieval_meta["_context_segments"]
    )
    assert not any(
        "Numeric Table Visual Verification" in citation.get("source_text", "")
        for citation in retrieval_meta["citations"]
    )
    assert any(
        citation.get("evidence_id") == "bundle-table-1:row:2"
        for citation in retrieval_meta["citations"]
    )
    assert not any(
        segment.get("segment_role") == "numeric_table_visual_verification"
        for segment in context_segments
    )
    diagnostics = retrieval_meta["diagnostics"]["numeric_table_visual_verification"]
    assert diagnostics["visual_verdict"] == verdict
    assert diagnostics["reason"] == "visual_and_structured_evidence_disagree"
