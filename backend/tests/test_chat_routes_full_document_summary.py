"""Route-level regressions for the parse-bound full-document summary path.

The renderer and intent unit tests are deliberately not enough here.  These
tests enter the public ``/chat`` and ``/chat/stream`` handlers so an explicit
"each section" request cannot accidentally fall through to ordinary Top-K
retrieval or the Agent loop after a future routing refactor.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes  # noqa: E402


_PARSE_MANIFEST = {
    "route": "mineru",
    "requested_route": "mineru",
    "resolved_route": "mineru",
    "generation": "parse-summary-route-test",
    "source_hash": "summary-route-source-hash",
    "status": "ready",
    "stage": "ready",
    "metadata": {"full_route": True},
}


def _request(question: str) -> chat_routes.ChatRequest:
    return chat_routes.ChatRequest(
        doc_id="doc-summary-route",
        question=question,
        api_key="test-key",
        model="test-model",
        api_provider="openai",
        enable_vector_search=True,
        enable_memory=False,
        enable_glossary=False,
        # An explicit Agent request is intentional: the summary route must
        # still disable it rather than trusting a broad overview whitelist.
        enable_agent_retrieval=True,
        force_agent_retrieval=True,
    )


def _install_parse_ready_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chat_routes.router,
        "documents_store",
        {
            "doc-summary-route": {
                "filename": "summary-route-paper.pdf",
                "data": {
                    "full_text": "已发布的 MinerU 文本。",
                    "pages": [{"page": 1, "content": "已发布的 MinerU 文本。"}],
                    "parse_manifest": dict(_PARSE_MANIFEST),
                },
            }
        },
        raising=False,
    )
    monkeypatch.setattr(chat_routes.router, "vector_store_dir", "", raising=False)
    # The route test owns no on-disk vector artifact.  Keep the actual parse
    # manifest and identity checks active while making this unrelated index
    # admission predicate deterministic.
    monkeypatch.setattr(
        chat_routes,
        "_chat_vector_index_matches_parse",
        lambda *_args, **_kwargs: True,
    )


def _install_summary_route_sentinels(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict,
) -> None:
    async def _contextualize(*, question: str, **_kwargs) -> str:
        # Reproduce the important multi-turn failure mode: a contextualizer
        # shortens the question and drops the words "each section".  The
        # router must retain original-question semantics nevertheless.
        captured["contextualizer_input"] = question
        return "summarize the paper"

    async def _rewrite(*_args, **_kwargs) -> str:
        raise AssertionError("full-document summary must not need retrieval query rewriting")

    async def _unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("full-document summary must not enter Top-K retrieval")

    async def _summary_builder(*, turn_context, **_kwargs) -> dict:
        captured["turn_context"] = turn_context
        return {
            "answer": "## 示例论文\n\n### 章节梳理\n\n- 1. 引言：说明研究问题。",
            "citations": [],
            "coverage": {
                "mode": "reading_outline_full_document",
                "presentation_mode": "section_detail",
                "complete": True,
                "rendered_section_count": 1,
                "visible_section_count": 1,
                "citation_count": 0,
            },
            "outline": {"source": "ai", "provider": "openai", "model": "test-model"},
        }

    def _critic(*_args, **_kwargs) -> dict:
        return {
            "certainty": {"level": "verified"},
            "citation_coverage": {"status": "not_applicable"},
        }

    monkeypatch.setattr(chat_routes, "_maybe_contextualize_intent_query", _contextualize)
    monkeypatch.setattr(chat_routes, "_maybe_rewrite_query", _rewrite)
    monkeypatch.setattr(chat_routes, "should_attempt_llm_clarification", lambda **_kwargs: False)
    monkeypatch.setattr(chat_routes, "vector_context", _unexpected_retrieval)
    monkeypatch.setattr(chat_routes, "_build_full_document_summary_for_turn", _summary_builder)
    monkeypatch.setattr(chat_routes, "postprocess_critic_result", _critic)
    monkeypatch.setattr(
        chat_routes,
        "build_answer_certainty_event",
        lambda certainty: {"type": "answer_certainty", "certainty": certainty},
    )


@pytest.mark.asyncio
async def test_non_stream_each_section_request_uses_parse_bound_summary_route(monkeypatch) -> None:
    captured: dict = {}
    _install_parse_ready_document(monkeypatch)
    _install_summary_route_sentinels(monkeypatch, captured)

    response = await chat_routes.chat_with_pdf(
        _request("Summarize each section of the paper")
    )

    turn = captured["turn_context"]
    assert captured["contextualizer_input"] == "Summarize each section of the paper"
    assert turn.original_question == "Summarize each section of the paper"
    assert turn.effective_question == "summarize the paper"
    assert turn.intent.full_document_summary is True
    assert turn.intent.task == "summarize"
    assert turn.intent.scope == "document"
    assert response["answer"].startswith("## 示例论文")
    assert response["retrieval_meta"]["retrieval_mode"] == "reading_outline_full_document"
    assert response["retrieval_meta"]["agent_gate"]["reason"] == "full_document_summary"
    assert response["retrieval_meta"]["web_search_audit"]["executed"] is False
    assert response["retrieval_meta"]["web_search_audit"]["status"] == "not_requested"
    assert response["intent_decision"]["full_document_summary"] is True


@pytest.mark.asyncio
async def test_stream_each_section_request_emits_summary_progress_and_never_falls_back(monkeypatch) -> None:
    captured: dict = {}
    _install_parse_ready_document(monkeypatch)
    _install_summary_route_sentinels(monkeypatch, captured)

    response = await chat_routes.chat_with_pdf_stream(
        _request("请逐个小节总结这篇论文")
    )
    raw_events: list[str] = []
    async for raw in response.body_iterator:
        raw_events.append(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))

    payloads: list[dict] = []
    for raw in raw_events:
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            value = line.removeprefix("data: ")
            if value != "[DONE]":
                payloads.append(json.loads(value))

    turn = captured["turn_context"]
    assert turn.original_question == "请逐个小节总结这篇论文"
    assert turn.intent.full_document_summary is True
    assert any(
        item.get("type") == "retrieval_progress"
        and item.get("phase") == "full_document_summary"
        for item in payloads
    )
    assert any("### 章节梳理" in str(item.get("content") or "") for item in payloads)
    terminal = next(item for item in reversed(payloads) if item.get("done") is True)
    assert terminal["final_content"].startswith("## 示例论文")
    assert terminal["retrieval_meta"]["retrieval_mode"] == "reading_outline_full_document"
    assert terminal["retrieval_meta"]["full_document_summary"]["presentation_mode"] == "section_detail"
    assert terminal["intent_decision"]["full_document_summary"] is True
