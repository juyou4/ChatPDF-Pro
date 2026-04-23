"""非流式 /chat 路由的错误处理回归测试。"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes


def _build_request(**overrides):
    payload = {
        "doc_id": "doc-1",
        "question": "这段话是什么意思？",
        "api_key": "test-key",
        "model": "test-model",
        "api_provider": "openai",
        "selected_text": "这里是用户框选的一段原文。",
        "enable_vector_search": False,
        "enable_memory": False,
        "enable_glossary": False,
    }
    payload.update(overrides)
    return chat_routes.ChatRequest(**payload)


def _install_minimal_doc_store(monkeypatch):
    monkeypatch.setattr(
        chat_routes.router,
        "documents_store",
        {
            "doc-1": {
                "filename": "demo.pdf",
                "data": {
                    "total_pages": 1,
                    "full_text": "这里是文档全文。",
                    "pages": [{"page": 1, "content": "这里是文档全文。"}],
                },
            }
        },
        raising=False,
    )
    monkeypatch.setattr(chat_routes.router, "vector_store_dir", "", raising=False)


@pytest.mark.asyncio
async def test_chat_with_pdf_surfaces_upstream_ai_error(monkeypatch):
    _install_minimal_doc_store(monkeypatch)

    async def _fake_call_ai_api(*args, **kwargs):
        return {"error": "上游 401: invalid api key"}

    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    with pytest.raises(HTTPException) as exc_info:
        await chat_routes.chat_with_pdf(_build_request())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "AI调用失败: 上游 401: invalid api key"


@pytest.mark.asyncio
async def test_chat_with_pdf_reports_malformed_ai_payload(monkeypatch):
    _install_minimal_doc_store(monkeypatch)

    async def _fake_call_ai_api(*args, **kwargs):
        return {"id": "resp-1"}

    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    with pytest.raises(HTTPException) as exc_info:
        await chat_routes.chat_with_pdf(_build_request(question="请解释这段内容"))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "AI调用失败: AI返回格式无效：缺少choices"


@pytest.mark.asyncio
async def test_chat_with_pdf_preserves_numeric_table_context_segments(monkeypatch):
    _install_minimal_doc_store(monkeypatch)

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "根据用户问题检索到的相关文档片段：\n\nTable 1\nCBDM (τ=1) 5.86 46.6\n\n",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": "Table 1\nCBDM (τ=1) 5.86 46.6"},
                ],
            },
        }

    async def _fake_call_ai_api(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "CBDM (τ=1) 的 FID 是 5.86，准确率是 46.6。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert "numeric_table" in response["retrieval_meta"]["evidence_need"]
    assert any(
        "46.6" in segment.get("text", "")
        for segment in response["retrieval_meta"]["context_segments"]
    )


@pytest.mark.asyncio
async def test_chat_with_pdf_marks_cost_queries_as_numeric_table(monkeypatch):
    _install_minimal_doc_store(monkeypatch)

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT.",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {
                        "ref": 1,
                        "text": "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT.",
                    }
                ],
            },
        }

    async def _fake_call_ai_api(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "训练时间约为 24 hours，且没有额外推理开销。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="DiffuLT 的训练时间和额外推理开销分别是多少？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert "numeric_table" in response["retrieval_meta"]["evidence_need"]
    assert any(
        "24 hours" in segment.get("text", "").lower()
        for segment in response["retrieval_meta"]["context_segments"]
    )


@pytest.mark.asyncio
async def test_chat_with_pdf_keeps_implicit_second_best_query_unexpanded(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        captured["query"] = kwargs.get("query") if "query" in kwargs else args[1]
        return {
            "context": "Table 8 comparator rows",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": "Table 8 comparator rows"},
                ],
            },
        }

    async def _fake_call_ai_api(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "DiffuLT、RIDE(3 experts) 和 cRT 的表格行已检索到。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    question = "在 ImageNet-LT 上，DiffuLT 相比第二好的方法在 Many/Medium/Few 三个子集上分别提升了多少百分点？"
    await chat_routes.chat_with_pdf(
        _build_request(
            question=question,
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert captured["query"] == question


def test_context_segments_from_citations_preserve_exact_table_rows_and_cells():
    citations = [
        {
            "ref": 1,
            "source_text": "Table 1: FID of different generation models.\nCBDM (τ=1) 5.86 46.6",
            "page_range": [4, 4],
            "group_id": "chunk-1",
        },
        {
            "ref": 2,
            "display_text": "AID 45.2 5.78×10^-4",
            "highlight_text": "AID 45.2 5.78×10^-4",
            "page_range": [5, 5],
            "group_id": "chunk-2",
        },
    ]

    segments = chat_routes._build_context_segments_from_citations(citations)

    assert any(
        "CBDM (τ=1)" in segment["text"]
        and "5.86" in segment["text"]
        and "46.6" in segment["text"]
        for segment in segments
    )
    assert any(
        "AID" in segment["text"]
        and "45.2" in segment["text"]
        and "5.78×10^-4" in segment["text"]
        for segment in segments
    )


def test_context_segments_from_citations_prefer_exact_table_row_over_broad_source():
    citations = [
        {
            "ref": 1,
            "source_text": "Broad narrative summary that mentions Table 8 and DiffuLT but omits the exact values.",
            "display_text": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "highlight_text": "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4",
            "chunk_type": "table_row",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
            "page_range": [9, 9],
            "group_id": "table-8-row",
        }
    ]

    segments = chat_routes._build_context_segments_from_citations(citations)

    assert len(segments) == 1
    assert "Table 8: Results on ImageNet-LT." in segments[0]["text"]
    assert "DiffuLT | ResNet-50 | All=56.4 | Many=63.3 | Med.=55.6 | Few=39.4" in segments[0]["text"]
    assert "Broad narrative summary" not in segments[0]["text"]


def test_build_response_context_segments_prefers_numeric_table_citations():
    retrieval_meta = {
        "evidence_need": ["numeric_table"],
        "_context_segments": [
            {
                "ref": 1,
                "text": "Broad narrative summary without the exact numeric row.",
                "page_range": [9, 9],
                "group_id": "group-1",
            }
        ],
        "citations": [
            {
                "ref": 1,
                "display_text": "CBDM (τ=1) 5.86 46.6",
                "highlight_text": "CBDM (τ=1) 5.86 46.6",
                "chunk_type": "table_row",
                "table_caption": "Table 1: FID of different generation models.",
                "table_header": "Model FID Acc.(%)",
                "page_range": [4, 4],
                "group_id": "table-1-row",
            }
        ],
    }

    segments = chat_routes._build_response_context_segments(retrieval_meta)

    assert len(segments) == 1
    assert "CBDM (τ=1) 5.86 46.6" in segments[0]["text"]
    assert "Broad narrative summary" not in segments[0]["text"]


def test_build_response_context_segments_keeps_numeric_table_comparator_bundle():
    retrieval_meta = {
        "evidence_need": ["numeric_table"],
        "search_query": "表 8 中 ResNet-50 的 All 指标上，DiffuLT 分别比 cRT、RIDE(3 experts) 和 ADRW 高多少个百分点？",
        "_context_segments": [],
        "citations": [
            {
                "ref": 1,
                "source_ref": 1,
                "group_id": "table-8",
                "table_id": "Table 8",
                "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
                "table_caption": "Table 8: Results on ImageNet-LT.",
                "page_range": [9, 9],
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "DiffuLT 56.4 63.3 55.6 39.4",
                "display_text": "DiffuLT 56.4 63.3 55.6 39.4",
                "highlight_text": "DiffuLT 56.4 63.3 55.6 39.4",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "group_id": "table-8",
                "table_id": "Table 8",
                "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
                "table_caption": "Table 8: Results on ImageNet-LT.",
                "page_range": [9, 9],
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "cRT 47.3 58.8 44.0 26.1",
                "display_text": "cRT 47.3 58.8 44.0 26.1",
                "highlight_text": "cRT 47.3 58.8 44.0 26.1",
            },
            {
                "ref": 3,
                "source_ref": 3,
                "group_id": "table-8",
                "table_id": "Table 8",
                "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
                "table_caption": "Table 8: Results on ImageNet-LT.",
                "page_range": [9, 9],
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "RIDE(3 experts) 54.9 66.2 51.7 34.9",
                "display_text": "RIDE(3 experts) 54.9 66.2 51.7 34.9",
                "highlight_text": "RIDE(3 experts) 54.9 66.2 51.7 34.9",
            },
            {
                "ref": 4,
                "source_ref": 4,
                "group_id": "table-8",
                "table_id": "Table 8",
                "table_header": "ResNet-10 ResNet-50 All All Many Med. Few",
                "table_caption": "Table 8: Results on ImageNet-LT.",
                "page_range": [9, 9],
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "ADRW 54.1 62.9 52.6 37.1",
                "display_text": "ADRW 54.1 62.9 52.6 37.1",
                "highlight_text": "ADRW 54.1 62.9 52.6 37.1",
            },
        ],
    }

    segments = chat_routes._build_response_context_segments(retrieval_meta)

    assert len(segments) == 1
    assert segments[0]["group_id"] == "table-8"
    assert "Table 8: Results on ImageNet-LT." in segments[0]["text"]
    assert "ResNet-10 ResNet-50 All All Many Med. Few" in segments[0]["text"]
    assert "DiffuLT 56.4 63.3 55.6 39.4" in segments[0]["text"]
    assert "cRT 47.3 58.8 44.0 26.1" in segments[0]["text"]
    assert "RIDE(3 experts) 54.9 66.2 51.7 34.9" in segments[0]["text"]
    assert "ADRW 54.1 62.9 52.6 37.1" in segments[0]["text"]


@pytest.mark.asyncio
async def test_chat_with_pdf_rebuilds_context_segments_from_typed_table_evidence(monkeypatch):
    _install_minimal_doc_store(monkeypatch)

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "根据用户问题检索到的相关文档片段：\n\nTable 1 main results\n\n",
            "retrieval_meta": {
                "citations": [
                    {
                        "ref": 1,
                        "group_id": "chunk-1",
                        "page_range": [4, 4],
                        "highlight_text": "Table 1 main results",
                        "source_text": "Table 1 main results",
                        "display_text": "Table 1 main results",
                        "evidence_units": [
                            {
                                "evidence_unit_type": "table_row",
                                "table_caption": "Table 1: generation results",
                                "table_header": "Method | FID | Acc",
                                "content": "CBDM(τ=1) | 5.86 | 46.6",
                                "cell_evidence_units": [
                                    {"content": "CBDM(τ=1)"},
                                    {"content": "5.86"},
                                    {"content": "46.6"},
                                ],
                            }
                        ],
                    }
                ],
                "_context_segments": [
                    {"ref": 1, "text": "Table 1 main results"},
                ],
            },
        }

    async def _fake_call_ai_api(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "CBDM(τ=1) 的 FID 是 5.86，准确率是 46.6[1]。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert response["retrieval_meta"]["citations"][0]["source_ref"] == 1
    assert any(
        "CBDM(τ=1)" in segment["text"]
        and "5.86" in segment["text"]
        and "46.6" in segment["text"]
        for segment in response["retrieval_meta"]["context_segments"]
    )
