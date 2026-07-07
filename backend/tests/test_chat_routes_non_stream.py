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


@pytest.mark.asyncio
async def test_chat_with_pdf_evidence_selector_prunes_low_relevance_prompt_context(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "Relevant method evidence.\nUnrelated appendix note.\nSupporting experiment evidence.\nAnother relevant detail.",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": "Relevant method evidence for the proposed approach."},
                    {"ref": 2, "text": "Unrelated appendix note about a different dataset."},
                    {"ref": 3, "text": "Supporting experiment evidence for the same approach."},
                    {"ref": 4, "text": "Another relevant detail about the training objective."},
                ],
            },
        }

    async def _fake_score_chunks(question, chunks, **kwargs):
        scored = []
        for chunk in chunks:
            text = chunk["text"]
            score = 0.1 if "Unrelated appendix" in text else 0.8
            scored.append({**chunk, "llm_relevance_score": score})
        return scored

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "核心方法证据已筛选。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    import services.llm_scoring_service as llm_scoring_service

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(llm_scoring_service, "score_chunks", _fake_score_chunks)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="What evidence explains the proposed method?",
            selected_text="",
            enable_vector_search=True,
            custom_params={"enable_evidence_selector": True, "evidence_selector_min_score": 0.35},
        )
    )

    assert "Relevant method evidence" in captured["system_prompt"]
    assert "Supporting experiment evidence" in captured["system_prompt"]
    assert "Unrelated appendix note" not in captured["system_prompt"]
    selector_diag = response["retrieval_meta"]["diagnostics"]["evidence_selector"]
    assert selector_diag["enabled"] is True
    assert selector_diag["removed_count"] >= 1


@pytest.mark.asyncio
async def test_chat_with_pdf_evidence_selector_summarizes_kept_long_segments(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}
    long_relevant = (
        "The proposed method uses a two-stage optimization pipeline. "
        "Unrelated implementation details about logging and UI are repeated here. "
        "The key numerical result is Accuracy 91.2 on the target benchmark. "
    ) * 3

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": long_relevant,
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": long_relevant},
                    {"ref": 2, "text": "A short supporting sentence about Accuracy 91.2."},
                    {"ref": 3, "text": "Another short sentence about the benchmark."},
                ],
            },
        }

    async def _fake_score_chunks(question, chunks, **kwargs):
        return [{**chunk, "llm_relevance_score": 0.9} for chunk in chunks]

    async def _fake_compress_chunk(chunk_text, query, **kwargs):
        assert "Accuracy 91.2" in chunk_text
        return "The key numerical result is Accuracy 91.2 on the target benchmark."

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "Accuracy 是 91.2。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    import services.context_compressor as context_compressor
    import services.llm_scoring_service as llm_scoring_service

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(llm_scoring_service, "score_chunks", _fake_score_chunks)
    monkeypatch.setattr(context_compressor, "compress_chunk", _fake_compress_chunk)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="What accuracy does the method reach?",
            selected_text="",
            enable_vector_search=True,
            custom_params={"enable_evidence_selector": True},
        )
    )

    assert "Accuracy 91.2 on the target benchmark" in captured["system_prompt"]
    assert "logging and UI are repeated" not in captured["system_prompt"]
    selector_diag = response["retrieval_meta"]["diagnostics"]["evidence_selector"]
    assert selector_diag["summary_enabled"] is True
    assert selector_diag["summary_compressed_count"] == 1
    assert selector_diag["summary_chars_saved"] > 0


@pytest.mark.asyncio
async def test_chat_with_pdf_evidence_selector_protects_exact_table_rows(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    exact_row = {
        "ref": 1,
        "text": "Method: Ours; Accuracy: 55.5; FID: 4.7",
        "chunk_type": "table_row",
        "table_caption": "Table 1: Main results.",
        "table_header": "Method | Accuracy | FID",
        "numeric_table_exact_context_row_text": "Method: Ours; Accuracy: 55.5; FID: 4.7",
        "numeric_table_exact_context_caption": "Table 1: Main results.",
        "numeric_table_exact_context_header": "Method | Accuracy | FID",
    }

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "Table 1 row and surrounding notes.",
            "retrieval_meta": {
                "citations": [],
                "evidence_need": ["numeric_table"],
                "_context_segments": [
                    exact_row,
                    {"ref": 2, "text": "Unrelated table note about another method."},
                    {"ref": 3, "text": "Accuracy column explains classification performance."},
                    {"ref": 4, "text": "Irrelevant appendix text."},
                ],
            },
        }

    async def _fake_score_chunks(question, chunks, **kwargs):
        scored = []
        for chunk in chunks:
            text = chunk["text"]
            score = 0.8 if "Accuracy column" in text else 0.0
            scored.append({**chunk, "llm_relevance_score": score})
        return scored

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "Ours 的 Accuracy 是 55.5。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    import services.llm_scoring_service as llm_scoring_service

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(llm_scoring_service, "score_chunks", _fake_score_chunks)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="表 1 中 Ours 的 Accuracy 是多少？",
            selected_text="",
            enable_vector_search=True,
            custom_params={"enable_evidence_selector": True, "evidence_selector_min_score": 0.35},
        )
    )

    assert "Method: Ours; Accuracy: 55.5; FID: 4.7" in captured["system_prompt"]
    selector_diag = response["retrieval_meta"]["diagnostics"]["evidence_selector"]
    assert selector_diag["protected_count"] >= 1
    assert selector_diag["enabled"] is True


@pytest.mark.asyncio
async def test_chat_with_pdf_numeric_regex_locator_augments_prompt(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "The retrieved paragraph mentions LVIS minival but omits the exact AP50 value.",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": "The retrieved paragraph mentions LVIS minival."},
                    {"ref": 2, "text": "The appendix discusses ODinW evaluation."},
                    {"ref": 3, "text": "Another paragraph describes benchmark setup."},
                ],
            },
        }

    def _fake_agent_doc_context(*args, **kwargs):
        return chat_routes.DocContext(
            doc_id="doc-1",
            full_text="The appendix mentions LVIS minival.",
            chunks=[
                "[Structured Table Row Shard]\nLVIS minival | AP: 33.5 | AP50: 52.4",
                "Plain paragraph with LVIS minival only.",
            ],
            pages=[{"page": 7, "content": "The appendix mentions LVIS minival."}],
            chunk_metadata=[
                {
                    "chunk_type": "table_row",
                    "table_row_shard": True,
                    "structured_table_bundle": True,
                    "table_id": "Table 12",
                    "table_caption": "Table 12: ODinW benchmark results.",
                    "table_header": "Dataset | AP | AP50",
                    "numeric_table_exact_context_row_text": "Dataset: LVIS minival; AP: 33.5; AP50: 52.4",
                    "page_range": [7, 7],
                },
                {},
            ],
        )

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "LVIS minival 的 AP50 是 52.4。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "_build_agent_doc_context", _fake_agent_doc_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="表格里 LVIS minival 的 AP50 是 52.4 吗？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert "Dataset: LVIS minival; AP: 33.5; AP50: 52.4" in captured["system_prompt"]
    locator_diag = response["retrieval_meta"]["diagnostics"]["numeric_regex_locator"]
    assert locator_diag["attempted"] is True
    assert locator_diag["added_count"] >= 1


def test_numeric_regex_locator_pattern_ignores_explicit_table_numbers():
    pattern = chat_routes._numeric_regex_locator_pattern(
        "Table 9 的消融实验中，完整使用生成样本后准确率是多少？"
    )

    assert "9" not in pattern


def test_numeric_regex_locator_pattern_ignores_identifier_embedded_numbers():
    pattern = chat_routes._numeric_regex_locator_pattern(
        "Table 1 中 DETR 与 DETR-DC5 在 COCO validation 上的 GFLOPS/FPS、参数量和 AP 分别是多少？"
    )

    assert pattern != r"(?:5\s*%?)"
    assert "DETR" in pattern


def test_numeric_regex_locator_pattern_ignores_condition_numbers():
    pattern = chat_routes._numeric_regex_locator_pattern(
        "Table 7 中 DiffuLT + RIDE (3 experts) 在 CIFAR100-LT r=100 上的 overall 是多少？"
    )

    assert r"100\s*%?" not in pattern
    assert r"3\s*%?" not in pattern
    assert "DiffuLT" in pattern or "RIDE" in pattern


@pytest.mark.asyncio
async def test_chat_with_pdf_numeric_regex_locator_filters_wrong_explicit_table(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "The retrieved paragraph mentions an ablation table but omits the exact value.",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {"ref": 1, "text": "The retrieved paragraph mentions Table 9 ablations."},
                ],
            },
        }

    def _fake_agent_doc_context(*args, **kwargs):
        return chat_routes.DocContext(
            doc_id="doc-1",
            full_text="Table 2 and Table 9 are different tables.",
            chunks=[
                "[Structured Table Row Shard]\nTable 2: Generated samples.\nMethod: Ours; Accuracy: 55.5",
                "Plain paragraph with Table 9 only.",
            ],
            pages=[{"page": 5, "content": "Table 2 and Table 9 are different tables."}],
            chunk_metadata=[
                {
                    "chunk_type": "table_row",
                    "table_row_shard": True,
                    "structured_table_bundle": True,
                    "table_id": "Table 2",
                    "table_caption": "Table 2: Generated samples.",
                    "table_header": "Method | Accuracy",
                    "numeric_table_exact_context_row_text": "Method: Ours; Accuracy: 55.5",
                    "page_range": [5, 5],
                },
                {},
            ],
        )

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "现有上下文未提供该数值。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "_build_agent_doc_context", _fake_agent_doc_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="Table 9 中 Ours 的 Accuracy 是否是 55.5？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert "Method: Ours; Accuracy: 55.5" not in captured["system_prompt"]
    locator_diag = response["retrieval_meta"]["diagnostics"]["numeric_regex_locator"]
    assert locator_diag["attempted"] is True
    assert locator_diag["added_count"] == 0
    assert locator_diag["filtered_count"] >= 1
    assert locator_diag["skipped_reason"] == "explicit_table_mismatch"


@pytest.mark.asyncio
async def test_chat_with_pdf_numeric_regex_locator_prefers_target_row_shard(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": "The retrieved paragraph mentions Table 7 but only has an early row.",
            "retrieval_meta": {
                "citations": [],
                "_context_segments": [
                    {
                        "ref": 1,
                        "text": "Table 7 contains CIFAR100-LT results, but this snippet has the early baseline row.",
                    },
                ],
            },
        }

    def _fake_agent_doc_context(*args, **kwargs):
        return chat_routes.DocContext(
            doc_id="doc-1",
            full_text="Table 7 reports CIFAR100-LT results for several methods.",
            chunks=[
                "[Structured Table Row Shard]\nTable 7: CIFAR100-LT results for DiffuLT.\nMethod: Baseline; CIFAR100-LT overall: 49.7; many: 65.6",
                "[Structured Table Row Shard]\nTable 7: CIFAR100-LT results for DiffuLT.\nMethod: CBDM; CIFAR100-LT overall: 48.9; many: 64.1",
                "[Structured Table Row Shard]\nTable 7: CIFAR100-LT results for DiffuLT.\nMethod: DiffuLT; CIFAR100-LT overall: 51.5; many: 69.0; med: 51.6; few: 29.7",
            ],
            pages=[{"page": 7, "content": "Table 7 reports CIFAR100-LT results."}],
            chunk_metadata=[
                {
                    "chunk_type": "table_row",
                    "table_row_shard": True,
                    "structured_table_bundle": True,
                    "table_id": "Table 7",
                    "table_caption": "Table 7: CIFAR100-LT results for DiffuLT.",
                    "table_header": "Method | CIFAR100-LT overall | many | med | few",
                    "numeric_table_exact_context_row_text": "Method: Baseline; CIFAR100-LT overall: 49.7; many: 65.6",
                    "page_range": [7, 7],
                },
                {
                    "chunk_type": "table_row",
                    "table_row_shard": True,
                    "structured_table_bundle": True,
                    "table_id": "Table 7",
                    "table_caption": "Table 7: CIFAR100-LT results for DiffuLT.",
                    "table_header": "Method | CIFAR100-LT overall | many | med | few",
                    "numeric_table_exact_context_row_text": "Method: CBDM; CIFAR100-LT overall: 48.9; many: 64.1",
                    "page_range": [7, 7],
                },
                {
                    "chunk_type": "table_row",
                    "table_row_shard": True,
                    "structured_table_bundle": True,
                    "table_id": "Table 7",
                    "table_caption": "Table 7: CIFAR100-LT results for DiffuLT.",
                    "table_header": "Method | CIFAR100-LT overall | many | med | few",
                    "numeric_table_exact_context_row_text": "Method: DiffuLT; CIFAR100-LT overall: 51.5; many: 69.0; med: 51.6; few: 29.7",
                    "page_range": [7, 7],
                },
            ],
        )

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "DiffuLT 的 overall 是 51.5。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "_build_agent_doc_context", _fake_agent_doc_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    response = await chat_routes.chat_with_pdf(
        _build_request(
            question="Table 7 中 DiffuLT 在 CIFAR100-LT r=100 上的 overall 是多少？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    assert "Method: DiffuLT; CIFAR100-LT overall: 51.5" in captured["system_prompt"]
    locator_diag = response["retrieval_meta"]["diagnostics"]["numeric_regex_locator"]
    assert locator_diag["attempted"] is True
    assert locator_diag["candidate_count"] >= 3
    assert locator_diag["added_count"] >= 1


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


def test_build_response_context_segments_drops_synthetic_when_original_evidence_exists():
    retrieval_meta = {
        "search_query": "图 2 展示了什么趋势？",
        "evidence_need": ["figure_caption"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "text": "AI generated figure description with a plausible but unverified trend.",
                "synthetic_description": True,
                "chunk_type": "figure",
                "table_id": "Figure 2",
            },
            {
                "ref": 2,
                "text": "Figure 2: Performance increases as the number of examples grows.",
                "chunk_type": "caption",
                "block_type": "caption",
            },
        ],
        "citations": [],
    }

    segments = chat_routes._build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment.get("text", "") for segment in segments)

    assert "AI generated figure description" not in joined
    assert "Figure 2: Performance increases" in joined


@pytest.mark.asyncio
async def test_chat_with_pdf_structured_citation_prompt_omits_synthetic_when_original_exists(monkeypatch):
    _install_minimal_doc_store(monkeypatch)
    captured = {}

    async def _fake_vector_context(*args, **kwargs):
        return {
            "context": (
                "AI generated figure description with a plausible trend.\n"
                "Figure 2: Performance increases as the number of examples grows."
            ),
            "retrieval_meta": {
                "citations": [
                    {
                        "ref": 1,
                        "group_id": "figure-2-synthetic",
                        "page_range": [2, 2],
                        "source_text": "AI generated figure description with a plausible trend.",
                        "highlight_text": "AI generated figure description",
                        "synthetic_description": True,
                    },
                    {
                        "ref": 2,
                        "group_id": "figure-2-caption",
                        "page_range": [2, 2],
                        "source_text": "Figure 2: Performance increases as the number of examples grows.",
                        "highlight_text": "Figure 2: Performance increases",
                        "chunk_type": "caption",
                    },
                ],
                "_context_segments": [
                    {
                        "ref": 1,
                        "text": "AI generated figure description with a plausible trend.",
                        "synthetic_description": True,
                    },
                    {
                        "ref": 2,
                        "text": "Figure 2: Performance increases as the number of examples grows.",
                        "chunk_type": "caption",
                    },
                ],
            },
        }

    async def _fake_call_ai_api(messages, *args, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return {
            "choices": [{"message": {"content": "FINAL ANSWER\n图 2 显示性能随样本数增加而提升[1]。"}}],
            "_used_provider": "openai",
            "_used_model": "test-model",
            "_fallback_used": False,
        }

    monkeypatch.setattr(chat_routes, "vector_context", _fake_vector_context)
    monkeypatch.setattr(chat_routes, "call_ai_api", _fake_call_ai_api)

    await chat_routes.chat_with_pdf(
        _build_request(
            question="图 2 展示了什么趋势？",
            selected_text="",
            enable_vector_search=True,
        )
    )

    prompt = captured["system_prompt"]
    citation_sources = prompt.split("可用的引用来源：", 1)[1]
    assert "figure-2-caption" in citation_sources
    assert "figure-2-synthetic" not in citation_sources


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

    assert len(segments) == 2
    assert segments[0]["segment_role"] == "numeric_evidence_pack"
    assert "CBDM (τ=1) 5.86 46.6" in segments[0]["text"]
    assert segments[1]["chunk_type"] == "table_row"
    assert "CBDM (τ=1) 5.86 46.6" in segments[1]["text"]
    assert "Broad narrative summary" not in " ".join(segment["text"] for segment in segments)


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

    assert len(segments) == 5
    assert segments[0]["segment_role"] == "numeric_evidence_pack"
    assert segments[0]["group_id"] == "table-8"
    assert "Table 8: Results on ImageNet-LT." in segments[0]["text"]
    assert "ResNet-10 ResNet-50 All All Many Med. Few" in segments[0]["text"]
    assert "DiffuLT 56.4 63.3 55.6 39.4" in segments[0]["text"]
    assert "cRT 47.3 58.8 44.0 26.1" in segments[0]["text"]
    assert "RIDE(3 experts) 54.9 66.2 51.7 34.9" in segments[0]["text"]
    assert "ADRW 54.1 62.9 52.6 37.1" in segments[0]["text"]
    assert set(
        segment["numeric_table_exact_context_row_text"]
        for segment in segments[1:]
    ) == {
        "DiffuLT 56.4 63.3 55.6 39.4",
        "cRT 47.3 58.8 44.0 26.1",
        "RIDE(3 experts) 54.9 66.2 51.7 34.9",
        "ADRW 54.1 62.9 52.6 37.1",
    }


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
