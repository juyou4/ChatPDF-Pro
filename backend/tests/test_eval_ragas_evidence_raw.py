import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.eval_ragas import (
    ChatpdfClient,
    CollectedSample,
    QuestionItem,
    _build_request_overrides_manifest,
    _is_non_recoverable_provider_error,
    print_summary_table,
    save_results,
)


def _client() -> ChatpdfClient:
    return ChatpdfClient(
        backend_url="http://127.0.0.1:8000",
        doc_id="doc-1",
        api_key="test-key",
        model="test-model",
        provider="openai",
    )


def test_eval_payload_requests_evidence_raw_by_default():
    payload = _client()._build_payload("测试问题")

    assert payload["include_evidence_raw"] is True


def test_eval_payload_allows_overriding_evidence_raw_flag():
    payload = _client()._build_payload("测试问题", {"include_evidence_raw": False})

    assert payload["include_evidence_raw"] is False


def test_eval_detects_non_recoverable_provider_quota_errors():
    error = 'HTTP 500: {"detail":"AI调用失败: 402: OpenAI兼容API错误: {\\"error\\":{\\"message\\":\\"Insufficient Balance\\"}}"}'

    assert _is_non_recoverable_provider_error(error) is True
    assert _is_non_recoverable_provider_error("HTTP 429: rate limit exceeded") is False


def test_eval_results_persist_evidence_raw(tmp_path):
    output_path = tmp_path / "ragas_result.json"
    sample = CollectedSample(
        question="测试问题",
        answer="测试答案",
        contexts=["上下文"],
        ground_truth=None,
        latency_ms=12.3,
        retrieval_diagnostics={"duplicate_chunk_ratio": 0.1},
        retrieval_evidence_raw={
            "schema_version": 1,
            "source": "chatpdf.retrieval_meta.evidence_raw",
            "metadata": {
                "format": "debug_evidence_bundle",
                "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
            },
            "evidence_need": ["numeric_table"],
            "counts": {"citations": 1, "context_segments": 1},
            "chunks": [{"chunk_type": "table_row", "table_id": "Table 1", "text": "Ours | 55.5"}] * 16,
            "agent_pipeline": {"likely_bottleneck": "context_budget"},
        },
    )

    save_results([sample], {"faithfulness": 1.0}, str(output_path))

    data = json.loads(output_path.read_text(encoding="utf-8"))
    run_evidence = data["summary"]["retrieval_evidence_summary"]
    assert run_evidence["evidence_raw_samples"] == 1
    assert run_evidence["evidence_raw_coverage_ratio"] == 1.0
    assert run_evidence["agent_pipeline_samples"] == 1
    assert run_evidence["numeric_table_evidence_samples"] == 1
    assert run_evidence["bottleneck_counts"] == {"context_budget": 1}
    assert run_evidence["schema_versions"] == ["1"]
    assert run_evidence["sources"] == ["chatpdf.retrieval_meta.evidence_raw"]

    saved = data["samples"][0]
    assert saved["retrieval_diagnostics"]["duplicate_chunk_ratio"] == 0.1
    assert saved["retrieval_evidence_raw"]["counts"]["citations"] == 1
    assert saved["retrieval_evidence_raw"]["agent_pipeline"]["likely_bottleneck"] == "context_budget"
    assert saved["retrieval_evidence_summary"]["schema_version"] == 1
    assert saved["retrieval_evidence_summary"]["source"] == "chatpdf.retrieval_meta.evidence_raw"
    assert saved["retrieval_evidence_summary"]["has_agent_pipeline"] is True
    assert saved["retrieval_evidence_summary"]["likely_bottleneck"] == "context_budget"
    assert saved["retrieval_evidence_summary"]["has_numeric_table_evidence"] is True

    csv_path = output_path.with_suffix(".csv")
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    header = rows[0]
    row = rows[1]
    assert header[header.index("证据版本")] == "证据版本"
    assert row[header.index("证据版本")] == "1"
    assert row[header.index("Agent证据")] == "True"
    assert row[header.index("数值表格证据")] == "True"


def test_eval_results_summarize_evidence_raw_truncation(tmp_path):
    output_path = tmp_path / "ragas_result.json"
    sample = CollectedSample(
        question="测试问题",
        answer="测试答案",
        contexts=["上下文"],
        ground_truth=None,
        latency_ms=12.3,
        retrieval_evidence_raw={
            "schema_version": 1,
            "metadata": {
                "format": "debug_evidence_bundle",
                "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
            },
            "counts": {"citations": 13, "context_segments": 1, "retrieval_context_segments": 20, "chunks": 16},
        },
    )

    save_results([sample], {"faithfulness": 1.0}, str(output_path))

    data = json.loads(output_path.read_text(encoding="utf-8"))
    run_evidence = data["summary"]["retrieval_evidence_summary"]
    assert run_evidence["evidence_raw_samples"] == 1
    assert run_evidence["truncated_evidence_samples"] == 1
    assert run_evidence["truncated_field_counts"] == {"citations": 1, "retrieval_context_segments": 1}

    summary = data["samples"][0]["retrieval_evidence_summary"]
    assert summary["is_truncated"] is True
    assert summary["truncated_fields"] == ["citations", "retrieval_context_segments"]

    csv_path = output_path.with_suffix(".csv")
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    header = rows[0]
    row = rows[1]
    assert row[header.index("证据截断字段")] == "citations|retrieval_context_segments"


def test_print_summary_table_includes_evidence_run_summary(capsys):
    sample = CollectedSample(
        question="测试问题",
        answer="测试答案",
        contexts=["上下文"],
        ground_truth=None,
        latency_ms=12.3,
        retrieval_evidence_raw={
            "schema_version": 1,
            "source": "chatpdf.retrieval_meta.evidence_raw",
            "metadata": {
                "format": "debug_evidence_bundle",
                "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
            },
            "evidence_need": ["numeric_table"],
            "counts": {"citations": 13, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
            "chunks": [{"chunk_type": "table_row", "table_id": "Table 1", "text": "Ours | 55.5"}],
            "agent_pipeline": {"likely_bottleneck": "context_budget"},
        },
    )

    print_summary_table([sample], {"faithfulness": 1.0})

    output = capsys.readouterr().out
    assert "证据链统计" in output
    assert "证据包覆盖" in output
    assert "1/1 (100.0%)" in output
    assert "Agent诊断样本" in output
    assert "数值表格证据样本" in output
    assert "证据截断样本" in output
    assert "citations:1" in output
    assert "Agent瓶颈分布" in output
    assert "context_budget:1" in output


def test_print_summary_table_includes_index_source_summary(capsys):
    samples = [
        CollectedSample(
            question="原生索引问题",
            answer="答案",
            contexts=["上下文1", "上下文2"],
            ground_truth=None,
            latency_ms=100.0,
            index_source="pdf_native",
            retrieval_diagnostics={"numeric_table_hit_quality": 0.5},
            retrieval_evidence_raw={
                "schema_version": 1,
                "source": "chatpdf.retrieval_meta.evidence_raw",
                "metadata": {
                    "format": "debug_evidence_bundle",
                    "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
                },
                "counts": {"citations": 1, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
            },
        ),
        CollectedSample(
            question="MinerU索引问题",
            answer="答案",
            contexts=["上下文"],
            ground_truth=None,
            latency_ms=300.0,
            index_source="mineru",
            retrieval_diagnostics={"numeric_table_hit_quality": 1.0},
            retrieval_evidence_raw={
                "schema_version": 1,
                "source": "chatpdf.retrieval_meta.evidence_raw",
                "metadata": {
                    "format": "debug_evidence_bundle",
                    "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
                },
                "evidence_need": ["numeric_table"],
                "counts": {"citations": 1, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
                "chunks": [{"chunk_type": "table_row", "table_id": "Table 1", "text": "Ours | 55.5"}],
            },
        ),
    ]

    print_summary_table(samples, {"faithfulness": 1.0})

    output = capsys.readouterr().out
    assert "索引来源统计" in output
    assert "pdf_native" in output
    assert "mineru" in output
    assert "数表质量" in output
    assert "证据覆盖" in output
    assert "100.0%" in output


def test_eval_results_group_summary_by_index_source(tmp_path):
    output_path = tmp_path / "ragas_result.json"
    pdf_native = CollectedSample(
        question="原生索引问题",
        answer="答案",
        contexts=["上下文1", "上下文2"],
        ground_truth=None,
        latency_ms=100.0,
        index_source="pdf_native",
        retrieval_diagnostics={"duplicate_chunk_ratio": 0.2, "numeric_table_hit_quality": 0.5},
        retrieval_evidence_raw={
            "schema_version": 1,
            "source": "chatpdf.retrieval_meta.evidence_raw",
            "metadata": {
                "format": "debug_evidence_bundle",
                "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
            },
            "counts": {"citations": 1, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
        },
    )
    mineru = CollectedSample(
        question="MinerU索引问题",
        answer="答案",
        contexts=["上下文"],
        ground_truth=None,
        latency_ms=300.0,
        index_source="mineru",
        retrieval_diagnostics={"duplicate_chunk_ratio": 0.0, "numeric_table_hit_quality": 1.0},
        retrieval_evidence_raw={
            "schema_version": 1,
            "source": "chatpdf.retrieval_meta.evidence_raw",
            "metadata": {
                "format": "debug_evidence_bundle",
                "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
            },
            "evidence_need": ["numeric_table"],
            "counts": {"citations": 1, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
            "chunks": [{"chunk_type": "table_row", "table_id": "Table 1", "text": "Ours | 55.5"}],
        },
    )

    save_results([pdf_native, mineru], {"faithfulness": 1.0}, str(output_path))

    data = json.loads(output_path.read_text(encoding="utf-8"))
    grouped = data["summary"]["by_index_source"]
    assert sorted(grouped) == ["mineru", "pdf_native"]
    assert grouped["pdf_native"]["total_samples"] == 1
    assert grouped["pdf_native"]["avg_contexts"] == 2.0
    assert grouped["pdf_native"]["avg_latency_ms"] == 100.0
    assert grouped["pdf_native"]["avg_duplicate_chunk_ratio"] == 0.2
    assert grouped["pdf_native"]["retrieval_evidence_summary"]["evidence_raw_samples"] == 1
    assert grouped["mineru"]["total_samples"] == 1
    assert grouped["mineru"]["avg_contexts"] == 1.0
    assert grouped["mineru"]["avg_numeric_table_hit_quality"] == 1.0
    assert grouped["mineru"]["retrieval_evidence_summary"]["numeric_table_evidence_samples"] == 1

    csv_path = output_path.with_suffix(".csv")
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    section_idx = rows.index(["按索引来源汇总"])
    header = rows[section_idx + 1]
    grouped_rows = {
        row[0]: row
        for row in rows[section_idx + 2 : section_idx + 4]
    }
    assert header[:4] == ["index_source", "总样本", "有效样本", "错误样本"]
    assert grouped_rows["pdf_native"][header.index("平均片段数")] == "2.0000"
    assert grouped_rows["pdf_native"][header.index("重复率")] == "0.2000"
    assert grouped_rows["mineru"][header.index("平均延迟ms")] == "300.0000"
    assert grouped_rows["mineru"][header.index("数值表格命中质量")] == "1.0000"
    assert grouped_rows["mineru"][header.index("数值表格证据样本")] == "1"


def test_eval_results_classify_error_types(tmp_path, capsys):
    output_path = tmp_path / "ragas_result.json"
    samples = [
        CollectedSample(
            question="模型失败",
            answer="",
            contexts=[],
            ground_truth=None,
            latency_ms=1000.0,
            error="HTTP 429: rate limit exceeded",
        ),
        CollectedSample(
            question="无上下文",
            answer="有答案但没有上下文",
            contexts=[],
            ground_truth=None,
            latency_ms=20.0,
        ),
        CollectedSample(
            question="评测失败",
            answer="",
            contexts=[],
            ground_truth=None,
            latency_ms=50.0,
            error="RAGAS metric evaluator failed",
        ),
    ]

    print_summary_table(samples, {})
    output = capsys.readouterr().out
    assert "错误类型统计" in output
    assert "provider" in output
    assert "retrieval_empty" in output
    assert "metric_evaluator" in output

    save_results(samples, {}, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["summary"]["by_error_type"]["provider"]["samples"] == 1
    assert data["summary"]["by_error_type"]["retrieval_empty"]["without_contexts"] == 1
    assert data["samples"][0]["error_type"] == "provider"
    assert data["samples"][1]["error_type"] == "retrieval_empty"
    assert data["samples"][2]["error_type"] == "metric_evaluator"

    rows = list(csv.reader(output_path.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
    header = rows[0]
    assert "错误类型" in header
    assert rows[1][header.index("错误类型")] == "provider"
    section_idx = rows.index(["按错误类型汇总"])
    assert rows[section_idx + 1][:3] == ["错误类型", "样本", "后端错误"]


def test_eval_results_group_summary_by_question_type(tmp_path, capsys):
    output_path = tmp_path / "ragas_result.json"
    samples = [
        CollectedSample(
            question="数值题",
            answer="答案",
            contexts=["表格上下文", "正文上下文"],
            ground_truth=None,
            latency_ms=100.0,
            question_type="numeric_table",
            retrieval_diagnostics={"duplicate_chunk_ratio": 0.1, "numeric_table_hit_quality": 1.0},
            retrieval_evidence_raw={
                "schema_version": 1,
                "source": "chatpdf.retrieval_meta.evidence_raw",
                "metadata": {
                    "format": "debug_evidence_bundle",
                    "limits": {"citations": 12, "context_segments": 16, "retrieval_context_segments": 16, "chunks": 16},
                },
                "evidence_need": ["numeric_table"],
                "counts": {"citations": 1, "context_segments": 1, "retrieval_context_segments": 1, "chunks": 1},
            },
        ),
        CollectedSample(
            question="综述题",
            answer="",
            contexts=[],
            ground_truth=None,
            latency_ms=30.0,
            question_type="overview",
        ),
    ]

    print_summary_table(samples, {"faithfulness": 1.0})
    output = capsys.readouterr().out
    assert "题型统计" in output
    assert "numeric_table" in output
    assert "overview" in output
    assert "retrieval_empty:1" in output

    save_results(samples, {"faithfulness": 1.0}, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))
    grouped = data["summary"]["by_question_type"]
    assert grouped["numeric_table"]["total_samples"] == 1
    assert grouped["numeric_table"]["avg_contexts"] == 2.0
    assert grouped["numeric_table"]["avg_numeric_table_hit_quality"] == 1.0
    assert grouped["overview"]["error_type_counts"] == {"retrieval_empty": 1}

    rows = list(csv.reader(output_path.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
    section_idx = rows.index(["按题型汇总"])
    header = rows[section_idx + 1]
    grouped_rows = {
        row[0]: row
        for row in rows[section_idx + 2 : section_idx + 4]
    }
    assert header[:4] == ["题型", "总样本", "有效样本", "错误样本"]
    assert grouped_rows["numeric_table"][header.index("平均片段数")] == "2.0000"
    assert grouped_rows["overview"][header.index("错误类型")] == "retrieval_empty:1"


def test_eval_results_csv_writes_ragas_metric_counts(tmp_path):
    output_path = tmp_path / "ragas_result.json"
    samples = [
        CollectedSample(
            question="可评问题",
            answer="答案",
            contexts=["上下文"],
            ground_truth=None,
            latency_ms=10.0,
        ),
        CollectedSample(
            question="跳过问题",
            answer="",
            contexts=[],
            ground_truth=None,
            latency_ms=20.0,
        ),
    ]
    ragas_scores = {
        "faithfulness": 0.75,
        "_meta": {
            "sample_count": 1,
            "metric_stats": {
                "faithfulness": {"valid_count": 1, "total_count": 2, "nan_count": 1},
            },
        },
    }

    save_results(samples, ragas_scores, str(output_path))

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["summary"]["ragas_eval_samples"] == {
        "collected_samples": 2,
        "ragas_ready_samples": 1,
        "ragas_evaluated_samples": 1,
        "ragas_skipped_samples": 1,
    }
    assert data["summary"]["ragas_metric_stats"]["faithfulness"]["nan_count"] == 1

    rows = list(csv.reader(output_path.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
    sample_section_idx = rows.index(["RAGAS样本统计"])
    assert rows[sample_section_idx + 1] == ["收集样本", "可评样本", "实际评估样本", "跳过样本"]
    assert rows[sample_section_idx + 2] == ["2", "1", "1", "1"]
    metric_section_idx = rows.index(["RAGAS指标", "分数", "有效样本", "总样本", "NaN/跳过"])
    metric_row = rows[metric_section_idx + 1]
    assert metric_row == ["忠实性 (Faithfulness)", "0.7500", "1", "2", "1"]


def test_eval_results_writes_request_overrides_hash(tmp_path):
    output_path = tmp_path / "ragas_result.json"
    sample = CollectedSample(
        question="带覆盖参数的问题",
        answer="答案",
        contexts=["上下文"],
        ground_truth=None,
        latency_ms=10.0,
        request_overrides={"doc_id": "doc-1", "top_k": 8, "enable_agent_retrieval": True},
    )

    save_results([sample], {}, str(output_path))

    data = json.loads(output_path.read_text(encoding="utf-8"))
    saved_hash = data["samples"][0]["request_overrides_hash"]
    assert len(saved_hash) == 64

    rows = list(csv.reader(output_path.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
    header = rows[0]
    assert "overrides_hash" in header
    assert rows[1][header.index("overrides_hash")] == saved_hash


def test_request_overrides_manifest_hash_is_stable_and_order_sensitive():
    questions = [
        QuestionItem(
            question="Q1",
            doc_id="doc-1",
            payload_overrides={"doc_id": "doc-1", "top_k": 8, "flags": {"b": 2, "a": 1}},
        ),
        QuestionItem(
            question="Q2",
            doc_id="doc-1",
            payload_overrides={"doc_id": "doc-1", "top_k": 10},
        ),
    ]
    same_questions = [
        QuestionItem(
            question="Q1",
            doc_id="doc-1",
            payload_overrides={"flags": {"a": 1, "b": 2}, "top_k": 8, "doc_id": "doc-1"},
        ),
        QuestionItem(
            question="Q2",
            doc_id="doc-1",
            payload_overrides={"top_k": 10, "doc_id": "doc-1"},
        ),
    ]
    changed_questions = [
        QuestionItem(
            question="Q1",
            doc_id="doc-1",
            payload_overrides={"doc_id": "doc-1", "top_k": 9, "flags": {"b": 2, "a": 1}},
        ),
        questions[1],
    ]

    manifest = _build_request_overrides_manifest(questions)
    same_manifest = _build_request_overrides_manifest(same_questions)
    changed_manifest = _build_request_overrides_manifest(changed_questions)

    assert manifest["overrides_count"] == 2
    assert len(manifest["manifest_hash"]) == 64
    assert manifest["manifest_hash"] == same_manifest["manifest_hash"]
    assert manifest["manifest_hash"] != changed_manifest["manifest_hash"]
    assert all(len(item["overrides_hash"]) == 64 for item in manifest["items"])
