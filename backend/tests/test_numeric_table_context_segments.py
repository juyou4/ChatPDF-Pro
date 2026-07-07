import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routes.chat_routes import (
    _build_agent_detail_citations,
    _build_context_segments_from_citations,
    _build_evidence_raw_debug,
    _build_numeric_table_projected_cells,
    _build_public_retrieval_meta,
    _build_response_context_segments,
    _extract_numeric_table_target_methods,
    _format_numeric_table_context_segments_for_prompt,
    _numeric_regex_locator_pattern,
    _query_rewriter,
    _should_run_numeric_regex_locator,
    _should_run_dataset_frame_locator,
)
from services.query_analyzer import analyze_evidence_need


def test_numeric_table_response_context_keeps_exact_retrieval_row_before_digest():
    retrieval_meta = {
        "search_query": "Table 3 中 Ours 的 All 指标是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "text": "group-47 digest: This section discusses Table 1 and model efficiency.",
                "retrieval_type": "agent_fetch_group",
                "group_id": "group-47",
                "context_id": "group-47:digest",
            },
            {
                "text": "Table 3: Main results. Method | All | Many | Medium | Few\nOurs | 55.5 | 60.1 | 54.2 | 41.8",
                "chunk_type": "table_row",
                "table_id": "Table 3",
                "table_bundle_id": "bundle-table-3",
                "numeric_table_exact_context_row_text": "Ours | 55.5 | 60.1 | 54.2 | 41.8",
                "context_id": "bundle-table-3:rows:2-2",
                "evidence_id": "bundle-table-3:row:2",
            },
        ],
        "citations": [
            {
                "ref": 1,
                "source_text": "group-47 digest: This section discusses Table 1 and model efficiency.",
                "display_text": "group-47 digest: This section discusses Table 1 and model efficiency.",
                "highlight_text": "Table 1 and model efficiency",
                "context_segment_text": "group-47 digest: This section discusses Table 1 and model efficiency.",
                "group_id": "group-47",
                "context_id": "group-47:digest",
                "retrieval_type": "agent_fetch_group",
            }
        ],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments
    assert segments[0]["segment_role"] == "numeric_evidence_pack"
    assert segments[0]["chunk_type"] == "table_row"
    assert "Ours | 55.5" in segments[0]["numeric_table_exact_context_row_text"]
    assert any(
        segment.get("segment_role") != "numeric_evidence_pack"
        and segment.get("numeric_table_exact_context_row_text") == "Ours | 55.5 | 60.1 | 54.2 | 41.8"
        for segment in segments
    )
    assert "group-47 digest" not in " ".join(segment["text"] for segment in segments)
    assert len(segments) <= 3


def test_numeric_table_response_context_packs_same_table_rows_with_budget():
    retrieval_meta = {
        "search_query": "Table 2 中 Baseline 和 Ours 的 Acc 分别是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "Method: Baseline; Acc: 38.3; FID: 10.1",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Ablation results",
                "table_header": "Method | Acc | FID",
                "numeric_table_exact_context_row_text": "Method: Baseline; Acc: 38.3; FID: 10.1",
                "context_id": "bundle-table-2:row:1",
                "evidence_id": "bundle-table-2:row:1",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Method: Ours; Acc: 51.5; FID: 8.7",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Ablation results",
                "table_header": "Method | Acc | FID",
                "numeric_table_exact_context_row_text": "Method: Ours; Acc: 51.5; FID: 8.7",
                "context_id": "bundle-table-2:row:2",
                "evidence_id": "bundle-table-2:row:2",
            },
            {
                "ref": 3,
                "source_ref": 3,
                "text": "Method: Extra; Acc: 49.0; FID: 9.2",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "numeric_table_exact_context_row_text": "Method: Extra; Acc: 49.0; FID: 9.2",
                "context_id": "bundle-table-2:row:3",
                "evidence_id": "bundle-table-2:row:3",
            },
            {
                "ref": 4,
                "source_ref": 4,
                "text": "[Structured Table Bundle]\nTable 2: Ablation results\n[Footnote]\nAll Acc values are percentages.",
                "chunk_type": "table",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Ablation results",
                "table_header": "Method | Acc | FID",
                "table_footnote": "All Acc values are percentages.",
                "context_id": "bundle-table-2",
                "evidence_id": "bundle-table-2",
            },
            {
                "text": "Unrelated discussion about another figure and training settings.",
                "retrieval_type": "agent_fetch_group",
                "group_id": "group-noise",
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert len(segments) == 3
    pack = segments[0]
    assert pack["segment_role"] == "numeric_evidence_pack"
    assert "Table 2: Ablation results" in pack["text"]
    assert "[ref 1] Method: Baseline; Acc: 38.3" in pack["numeric_table_exact_context_row_text"]
    assert "[ref 2] Method: Ours; Acc: 51.5" in pack["numeric_table_exact_context_row_text"]
    assert "[ref 3] Method: Extra" not in pack["numeric_table_exact_context_row_text"]
    assert "Method = Baseline; Acc = 38.3" in pack["numeric_table_projected_cells"]
    assert "Method = Ours; Acc = 51.5" in pack["numeric_table_projected_cells"]
    assert "Answer Cells:" in pack["text"]
    assert pack["text"].find("Answer Cells:") < pack["text"].find("Relevant Rows:")
    assert "Table Footnote: All Acc values are percentages." in pack["text"]
    assert pack["table_footnote"] == "All Acc values are percentages."
    assert segments[1]["numeric_table_exact_context_row_text"] == "Method: Baseline; Acc: 38.3; FID: 10.1"
    assert segments[2]["numeric_table_exact_context_row_text"] == "Method: Ours; Acc: 51.5; FID: 8.7"
    prompt_context = _format_numeric_table_context_segments_for_prompt(segments)
    assert "结构化投影:" in prompt_context
    assert "表注: All Acc values are percentages." in prompt_context
    assert "Unrelated discussion" not in pack["text"]


def test_numeric_table_projected_cells_use_readable_answer_format_for_percent_headers():
    rows = [
        (
            1,
            "omega: 0.3; Acc. (%): 51.5",
            {},
        ),
        (
            2,
            "alpha: 0.1; Acc. (%): 49.7",
            {},
        ),
    ]

    projected = _build_numeric_table_projected_cells(
        rows,
        header="Parameter | Acc. (%)",
        query="Table 10 中 weighted cross-entropy 的 omega 和 AID-biased loss 的 alpha 最优默认设置分别是多少？对应准确率是多少？",
    )

    assert "omega = 0.3" in projected
    assert "alpha = 0.1" in projected
    assert "Acc. (%) = 49.7" in projected
    assert "Acc. (%): 49.7" not in projected


def test_numeric_table_projected_cells_skip_fetch_digest_prose():
    rows = [
        (
            1,
            (
                "table 10. Through iterative adjustments, the optimal performance is achieved when "
                "$\\omega = 0.3$. 【检索证据 | source = fetch | group_id:group-23】 "
                "[ref 6] omega: 0; Acc. (%): 43.3; alpha: 4.0"
            ),
            {},
        ),
        (
            2,
            "omega: 0.3; Acc. (%): 51.5; alpha: 0.1",
            {},
        ),
    ]

    projected = _build_numeric_table_projected_cells(
        rows,
        header="ω | Acc. (%) | α",
        query="Table 10 中 weighted cross-entropy 的 omega 和 AID-biased loss 的 alpha 最优默认设置分别是多少？对应准确率是多少？",
    )

    assert "Through iterative adjustments" not in projected
    assert "43.3" not in projected
    assert "alpha = 4.0" not in projected
    assert "omega = 0.3" in projected
    assert "alpha = 0.1" in projected


def test_numeric_table_response_context_keeps_wide_context_when_explicit_table_unmatched():
    retrieval_meta = {
        "search_query": "Table 9 中 Ours 的 Accuracy 是否是 55.5？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "Table 2: Generated samples.\nMethod: Ours; Accuracy: 55.5",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Generated samples.",
                "table_header": "Method | Accuracy",
                "numeric_table_exact_context_row_text": "Method: Ours; Accuracy: 55.5",
                "context_id": "bundle-table-2:row:1",
                "evidence_id": "bundle-table-2:row:1",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": "The ablation setup for Table 9 is described in the appendix.",
                "retrieval_type": "agent_fetch_group",
                "group_id": "group-table-9",
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment.get("text", "") for segment in segments)

    assert segments
    assert not any(segment.get("segment_role") == "numeric_evidence_pack" for segment in segments)
    assert "Table 2: Generated samples." in joined
    assert "The ablation setup for Table 9" in joined


def test_numeric_table_pack_does_not_trust_table_label_leaked_in_row_text():
    retrieval_meta = {
        "search_query": "Table 9 中 Ours 的 Accuracy 是否是 55.5？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "Method: Ours; Accuracy: 55.5; Note: compared with Table 9 setting",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Generated samples.",
                "table_header": "Method | Accuracy | Note",
                "numeric_table_exact_context_row_text": (
                    "Method: Ours; Accuracy: 55.5; Note: compared with Table 9 setting"
                ),
                "context_id": "bundle-table-2:row:1",
                "evidence_id": "bundle-table-2:row:1",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Table 9 ablation evidence is nearby but not row-extracted.",
                "retrieval_type": "agent_fetch_group",
                "group_id": "group-table-9",
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment.get("text", "") for segment in segments)
    prompt_context = _format_numeric_table_context_segments_for_prompt(segments)

    assert segments
    assert not any(segment.get("segment_role") == "numeric_evidence_pack" for segment in segments)
    assert "Table 2: Generated samples." in prompt_context
    assert "Table 9 ablation evidence" in joined


def test_numeric_table_target_methods_keep_attention_residual_variant_phrase():
    query = "Block AttnRes 的 I/O cost 是多少？"
    hints = _query_rewriter.extract_numeric_table_hints(query)

    methods = _extract_numeric_table_target_methods(query, hints)

    assert "blockattnres" in methods


def test_numeric_table_target_methods_rescue_model_identifier_from_columns():
    query = "Table 1 中 DETR-DC5-R101 的 AP、AP50、AP75、APS、APM、APL 分别是多少？"
    hints = _query_rewriter.extract_numeric_table_hints(query)

    methods = _extract_numeric_table_target_methods(query, hints)

    assert "detrdc5r101" in methods
    assert "ap50" not in methods
    assert "ap75" not in methods
    assert "aps" not in methods


def test_numeric_regex_locator_skips_ordinary_model_scale_question():
    query = "CLIP 的预训练数据规模是多少？它的核心预训练任务是什么？"

    assert _should_run_numeric_regex_locator(query, r"(?:CLIP)", []) is False


def test_numeric_regex_locator_runs_for_explicit_table_question():
    query = "Table 1 中 DETR-DC5-R101 的 AP、AP50、AP75 分别是多少？"

    assert _should_run_numeric_regex_locator(query, r"(?:DETR\-DC5\-R101)", []) is True


def test_numeric_regex_locator_pattern_uses_explicit_table_label():
    query = "Table 10 中 weighted cross-entropy 的 omega 和 AID-biased loss 的 alpha 最优默认设置分别是多少？"

    pattern = _numeric_regex_locator_pattern(query)

    assert "table" in pattern.lower()
    assert "10" in pattern


def test_numeric_regex_locator_skips_experiment_setup_detail_question():
    query = "AdvRoad 的数字攻击实验使用了哪些 victim model 和 backbone？nuScenes 的训练/验证帧数是多少？"

    assert _should_run_numeric_regex_locator(query, r"(?:AdvRoad|nuScenes)", ["numeric_table"]) is False


def test_numeric_regex_locator_runs_for_cost_question_without_explicit_table_need():
    query = "Standard Residuals、Full AttnRes、Block AttnRes 的 typical total I/O 分别是多少？"

    assert _should_run_numeric_regex_locator(query, r"(?:AttnRes)", ["numeric_table"]) is True


def test_evidence_need_does_not_treat_digital_attack_setup_as_numeric_table():
    query = "AdvRoad 的数字攻击实验使用了哪些 victim model 和 backbone？nuScenes 的训练/验证帧数是多少？"

    assert "numeric_table" not in analyze_evidence_need(query)


def test_dataset_frame_locator_only_runs_for_train_validation_frame_questions():
    query = "AdvRoad 的数字攻击实验使用了哪些 victim model 和 backbone？nuScenes 的训练/验证帧数是多少？"

    assert _should_run_dataset_frame_locator(query) is True
    assert _should_run_dataset_frame_locator("CLIP 的预训练数据规模是多少？") is False


def test_evidence_need_keeps_explicit_table_metric_questions_numeric_table():
    query = "Table 2 中 AdvRoad 在 nuScenes 上的 ASR 分别是多少？"

    assert "numeric_table" in analyze_evidence_need(query)


def test_numeric_table_response_context_relaxes_weak_table_need_for_setup_question():
    retrieval_meta = {
        "search_query": "AdvRoad 的数字攻击实验使用了哪些 victim model 和 backbone？nuScenes 的训练/验证帧数是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": (
                    "[Structured Table Row Shard]\n"
                    "Table 2: Digital attack results of adversarial creation attack in the nuScenes dataset.\n"
                    "Model: BEVDet4D-SwinT; Attack: AdvRoad; LPIPS: 0.1370; ASR: 39.1"
                ),
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "numeric_table_exact_context_row_text": "Model: BEVDet4D-SwinT; Attack: AdvRoad; ASR: 39.1",
                "context_id": "table-2-row",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": (
                    "Experiment Experimental Setup Victim Model. BEVDet, BEVDet4D, and BEVFormer are "
                    "selected as victim models. For each detector, ResNet50 and SwinTransformer-Tiny are "
                    "used as image backbone respectively. Dataset. The training and validation set contains "
                    "28,130 and 6,019 frames respectively."
                ),
                "retrieval_type": "agent_fetch_group",
                "context_id": "setup-group",
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment.get("text", "") for segment in segments)

    assert not any(segment.get("segment_role") == "numeric_evidence_pack" for segment in segments)
    assert "BEVDet, BEVDet4D, and BEVFormer" in joined
    assert "28,130 and 6,019 frames" in joined


def test_numeric_table_response_context_verifies_row_method_not_header_only():
    retrieval_meta = {
        "search_query": "Block AttnRes 的 I/O cost 是多少？",
        "evidence_need": ["numeric_table"],
        "_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "Method: Full AttnRes; I/O cost: 24d",
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "table_caption": "Table 1: Training cost comparison.",
                "table_header": "Method | Standard AttnRes | Full AttnRes | Block AttnRes | I/O cost",
                "row_id": "Full AttnRes",
                "numeric_table_exact_context_row_text": "Method: Full AttnRes; I/O cost: 24d",
                "context_id": "bundle-table-1:row:full",
                "evidence_id": "bundle-table-1:row:full",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Method: Block AttnRes; I/O cost: 5.5d",
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "table_caption": "Table 1: Training cost comparison.",
                "table_header": "Method | Standard AttnRes | Full AttnRes | Block AttnRes | I/O cost",
                "row_id": "Block AttnRes",
                "numeric_table_exact_context_row_text": "Method: Block AttnRes; I/O cost: 5.5d",
                "context_id": "bundle-table-1:row:block",
                "evidence_id": "bundle-table-1:row:block",
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    pack = segments[0]

    assert pack["segment_role"] == "numeric_evidence_pack"
    assert "Method: Block AttnRes; I/O cost: 5.5d" in pack["numeric_table_exact_context_row_text"]
    assert "Method: Full AttnRes; I/O cost: 24d" not in pack["numeric_table_exact_context_row_text"]


def test_numeric_table_response_context_does_not_repack_existing_pack():
    retrieval_meta = {
        "search_query": "Table 2 中 Ours 的 Acc 是多少？",
        "evidence_need": ["numeric_table"],
        "_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "[Numeric Table Evidence Pack]\nRelevant Rows:\n[ref 1] Method: Old; Acc: 1.0",
                "segment_role": "numeric_evidence_pack",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "numeric_table_exact_context_row_text": "[ref 1] Method: Old; Acc: 1.0",
                "context_id": "bundle-table-2:old-pack",
                "evidence_id": "bundle-table-2:old-pack",
            }
        ],
        "_retrieval_context_segments": [
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Method: Ours; Acc: 51.5; FID: 8.7",
                "chunk_type": "table_row",
                "table_id": "Table 2",
                "table_bundle_id": "bundle-table-2",
                "table_caption": "Table 2: Ablation results",
                "table_header": "Method | Acc | FID",
                "numeric_table_exact_context_row_text": "Method: Ours; Acc: 51.5; FID: 8.7",
                "context_id": "bundle-table-2:row:2",
                "evidence_id": "bundle-table-2:row:2",
            }
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment["text"] for segment in segments)

    assert joined.count("[Numeric Table Evidence Pack]") == 1
    assert "Method: Old; Acc: 1.0" not in joined
    assert "Method: Ours; Acc: 51.5" in joined
    assert any(
        segment.get("segment_role") != "numeric_evidence_pack"
        and segment.get("numeric_table_exact_context_row_text") == "Method: Ours; Acc: 51.5; FID: 8.7"
        for segment in segments
    )


def test_numeric_table_response_context_does_not_repack_existing_execution_segment():
    retrieval_meta = {
        "search_query": "Table 1 中 Standard Residuals、Full AttnRes、Block AttnRes 的 typical total I/O 分别是多少？",
        "evidence_need": ["numeric_table"],
        "_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": (
                    "[Numeric Table Execution]\n"
                    "Operation: direct lookup on typical total I/O\n"
                    "Standard Residuals: 3d\n"
                    "Full AttnRes: 24d\n"
                    "Block AttnRes: 5.5d"
                ),
                "segment_role": "numeric_table_execution",
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "context_id": "bundle-table-1:numeric_execution",
                "evidence_id": "bundle-table-1:numeric_execution",
                "numeric_table_projected_cells": "Standard Residuals: 3d\nFull AttnRes: 24d\nBlock AttnRes: 5.5d",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": (
                    "Operation | Operation | [Numeric Table Execution] "
                    "Standard Residuals: 3d Full AttnRes: 24d Block AttnRes: 3d"
                ),
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "context_id": "bundle-table-1:numeric_execution:contaminated",
                "evidence_id": "bundle-table-1:numeric_execution:contaminated",
            }
        ],
        "_retrieval_context_segments": [
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Operation: Standard Residuals; Total I/O: 3d\nOperation: Full AttnRes Block; Column 8: 24d\nOperation: Full AttnRes Block; Write: (N/S + 5)d 5.5d",
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "table_caption": "Table 1: Memory access cost per token per layer.",
                "table_header": "Operation | Total I/O | Total I/O",
                "numeric_table_exact_context_row_text": "Operation: Standard Residuals; Total I/O: 3d\nOperation: Full AttnRes Block; Column 8: 24d\nOperation: Full AttnRes Block; Write: (N/S + 5)d 5.5d",
                "context_id": "bundle-table-1:rows:1-10",
                "evidence_id": "bundle-table-1:rows:1-10",
            }
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    execution_segments = [
        segment for segment in segments if segment.get("segment_role") == "numeric_table_execution"
    ]

    assert len(execution_segments) == 1
    assert "Block AttnRes: 5.5d" in execution_segments[0]["text"]
    assert "numeric_execution:numeric_pack" not in execution_segments[0].get("context_id", "")


def test_numeric_table_response_context_adds_deterministic_execution_segment_for_delta():
    def _row(ref, method, value):
        return {
            "ref": ref,
            "source_ref": ref,
            "text": f"{method} | {value}",
            "chunk_type": "table_row",
            "table_id": "Table 8",
            "table_bundle_id": "bundle-table-8",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "Method | ResNet-50 All",
            "row_id": method,
            "numeric_table_exact_context_row_text": f"Method: {method}; ResNet-50 All: {value}",
            "context_id": f"bundle-table-8:row:{ref}",
            "evidence_id": f"bundle-table-8:row:{ref}",
            "cell_evidence_units": [
                {
                    "evidence_unit_type": "table_cell",
                    "header_path": "Method",
                    "col_id": "Method",
                    "content": method,
                },
                {
                    "evidence_unit_type": "table_cell",
                    "header_path": "ResNet-50 All",
                    "col_id": "ResNet-50 All",
                    "content": value,
                },
            ],
        }

    retrieval_meta = {
        "search_query": "表 8 中 ResNet-50 的 All 指标上，DiffuLT 比 cRT 高多少个百分点？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            _row(1, "DiffuLT", "56.4"),
            _row(2, "cRT", "47.3"),
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: difference on ResNet-50 All" in segments[0]["text"]
    assert "Delta: DiffuLT - cRT = 9.1" in segments[0]["text"]
    assert segments[1]["segment_role"] == "numeric_evidence_pack"
    assert any(segment.get("row_id") == "DiffuLT" for segment in segments)
    assert any(segment.get("row_id") == "cRT" for segment in segments)


def test_numeric_table_response_context_adds_direct_execution_for_cost_lookup_text_shard():
    retrieval_meta = {
        "search_query": "Table 1 中 Standard Residuals、Full AttnRes、Block AttnRes 的 typical total I/O 分别是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": (
                    "[Structured Table Row Shard]\n"
                    "Table 1: Memory access cost per token per layer incurred by the residual mechanism under each scheme. "
                    "For AttnRes, both Full and Block variants use the two-phase inference schedule described in Appendix B.\n"
                    "[Header]\nOperation | Operation | | Read | Write | Total I/O | Total I/O\n"
                    "[Rows]\n"
                    "Operation: Standard Residuals; Column 3: Residual Merge; Read: 2d; Write: d; Total I/O: 3d; Total I/O: 3d\n"
                    "Operation: mHC (m streams); Operation: mHC (m streams); Column 3: Full AttnRes Block; Read: Phase 1 (amortized) Phase 2; Write: (N−1)d (S−1)d; Total I/O: d d; Total I/O: (S+N)d; Column 8: 24d\n"
                    "Operation: Phase 1 (amortized) Phase 2; Operation: N/S d; Column 3: Full AttnRes Block; Read: d d; Write: (N/S + 5)d 5.5d"
                ),
                "chunk_type": "table_row",
                "table_id": "Table 1",
                "table_bundle_id": "bundle-table-1",
                "table_caption": "Table 1: Memory access cost per token per layer.",
                "table_header": "Operation | Operation | Read | Write | Total I/O | Total I/O",
                "numeric_table_exact_context_row_text": (
                    "Operation: Standard Residuals; Total I/O: 3d; Total I/O: 3d\n"
                    "Operation: Full AttnRes Block; Total I/O: (S+N)d; Column 8: 24d\n"
                    "Operation: Full AttnRes Block; Write: (N/S + 5)d 5.5d"
                ),
                "context_id": "bundle-table-1:rows:1-10",
                "evidence_id": "bundle-table-1:rows:1-10",
            }
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: direct lookup on typical total I/O" in segments[0]["text"]
    assert "Standard Residuals: 3d" in segments[0]["text"]
    assert "Full AttnRes: 24d" in segments[0]["text"]
    assert "Block AttnRes: 5.5d" in segments[0]["text"]
    assert segments[1]["segment_role"] == "numeric_evidence_pack"


def test_numeric_table_response_context_adds_deterministic_execution_segment_for_maximum():
    def _row(ref, method, acc, fid):
        return {
            "ref": ref,
            "source_ref": ref,
            "text": f"{method} | {acc} | {fid}",
            "chunk_type": "table_row",
            "table_id": "Table 1",
            "table_bundle_id": "bundle-table-1",
            "table_caption": "Table 1: Generated sample quality.",
            "table_header": "Method | Accuracy | FID",
            "row_id": method,
            "numeric_table_exact_context_row_text": f"Method: {method}; Accuracy: {acc}; FID: {fid}",
            "context_id": f"bundle-table-1:row:{ref}",
            "evidence_id": f"bundle-table-1:row:{ref}",
            "cell_evidence_units": [
                {"header_path": "Method", "col_id": "Method", "content": method},
                {"header_path": "Accuracy", "col_id": "Accuracy", "content": acc},
                {"header_path": "FID", "col_id": "FID", "content": fid},
            ],
        }

    retrieval_meta = {
        "search_query": "表 1 中哪种生成模型取得了最高分类准确率？对应的 FID 和准确率分别是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            _row(1, "DDPM", "46.6", "5.86"),
            _row(2, "CBDM(τ=1)", "51.5", "4.71"),
            _row(3, "GAN", "43.2", "8.10"),
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: maximum on Accuracy" in segments[0]["text"]
    assert "Selected Row: CBDM(τ=1) = 51.5" in segments[0]["text"]
    assert "DDPM=46.6" in segments[0]["text"]


def test_numeric_table_response_context_adds_deterministic_execution_segment_for_second_best():
    def _row(ref, method, acc):
        return {
            "ref": ref,
            "source_ref": ref,
            "text": f"{method} | {acc}",
            "chunk_type": "table_row",
            "table_id": "Table 8",
            "table_bundle_id": "bundle-table-8",
            "table_caption": "Table 8: Results on ImageNet-LT.",
            "table_header": "Method | ResNet-50 All",
            "row_id": method,
            "numeric_table_exact_context_row_text": f"Method: {method}; ResNet-50 All: {acc}",
            "context_id": f"bundle-table-8:row:{ref}",
            "evidence_id": f"bundle-table-8:row:{ref}",
            "cell_evidence_units": [
                {"header_path": "Method", "col_id": "Method", "content": method},
                {"header_path": "ResNet-50 All", "col_id": "ResNet-50 All", "content": acc},
            ],
        }

    retrieval_meta = {
        "search_query": "表 8 中 ResNet-50 All 指标第二好的方法是哪一个？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            _row(1, "DiffuLT", "56.4"),
            _row(2, "RIDE(3 experts)", "54.9"),
            _row(3, "cRT", "47.3"),
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: rank 2 on ResNet-50 All" in segments[0]["text"]
    assert "Selected Row: RIDE(3 experts) = 54.9" in segments[0]["text"]
    assert "#1:DiffuLT=56.4" in segments[0]["text"]


def test_numeric_table_response_context_adds_deterministic_execution_segment_for_top_k():
    def _row(ref, method, acc):
        return {
            "ref": ref,
            "source_ref": ref,
            "text": f"{method} | {acc}",
            "chunk_type": "table_row",
            "table_id": "Table 2",
            "table_bundle_id": "bundle-table-2",
            "table_caption": "Table 2: Accuracy comparison.",
            "table_header": "Method | Accuracy",
            "row_id": method,
            "numeric_table_exact_context_row_text": f"Method: {method}; Accuracy: {acc}",
            "context_id": f"bundle-table-2:row:{ref}",
            "evidence_id": f"bundle-table-2:row:{ref}",
            "cell_evidence_units": [
                {"header_path": "Method", "col_id": "Method", "content": method},
                {"header_path": "Accuracy", "col_id": "Accuracy", "content": acc},
            ],
        }

    retrieval_meta = {
        "search_query": "Table 2 Accuracy top 2 methods 是哪些？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            _row(1, "A", "41.0"),
            _row(2, "B", "52.0"),
            _row(3, "C", "48.0"),
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: top 2 on Accuracy" in segments[0]["text"]
    assert "Rank 1: B = 52.0" in segments[0]["text"]
    assert "Rank 2: C = 48.0" in segments[0]["text"]


def test_numeric_table_response_context_adds_deterministic_execution_segment_for_named_rank():
    def _row(ref, method, fid):
        return {
            "ref": ref,
            "source_ref": ref,
            "text": f"{method} | {fid}",
            "chunk_type": "table_row",
            "table_id": "Table 1",
            "table_bundle_id": "bundle-table-1",
            "table_caption": "Table 1: Generated sample quality.",
            "table_header": "Method | FID",
            "row_id": method,
            "numeric_table_exact_context_row_text": f"Method: {method}; FID: {fid}",
            "context_id": f"bundle-table-1:row:{ref}",
            "evidence_id": f"bundle-table-1:row:{ref}",
            "cell_evidence_units": [
                {"header_path": "Method", "col_id": "Method", "content": method},
                {"header_path": "FID", "col_id": "FID", "content": fid},
            ],
        }

    retrieval_meta = {
        "search_query": "表 1 中 CBDM 的 FID 排名第几？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            _row(1, "DDPM", "5.86"),
            _row(2, "CBDM", "4.71"),
            _row(3, "GAN", "8.10"),
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)

    assert segments[0]["segment_role"] == "numeric_table_execution"
    assert "Operation: rank on FID (ascending)" in segments[0]["text"]
    assert "Rank: CBDM = #1 (4.71)" in segments[0]["text"]


def test_numeric_table_response_context_drops_synthetic_when_original_table_evidence_exists():
    retrieval_meta = {
        "search_query": "Table 3 中 Ours 的 All 指标是多少？",
        "evidence_need": ["numeric_table"],
        "_retrieval_context_segments": [
            {
                "ref": 1,
                "source_ref": 1,
                "text": "AI generated table description says Ours has All 55.5.",
                "synthetic_description": True,
                "chunk_type": "table",
                "table_id": "Table 3",
                "table_caption": "Table 3: Main results",
            },
            {
                "ref": 2,
                "source_ref": 2,
                "text": "Ours | 55.5 | 60.1",
                "chunk_type": "table_row",
                "table_id": "Table 3",
                "table_bundle_id": "bundle-table-3",
                "table_caption": "Table 3: Main results",
                "table_header": "Method | All | Many",
                "row_id": "Ours",
                "numeric_table_exact_context_row_text": "Method: Ours; All: 55.5; Many: 60.1",
                "cell_evidence_units": [
                    {"header_path": "Method", "content": "Ours"},
                    {"header_path": "All", "content": "55.5"},
                    {"header_path": "Many", "content": "60.1"},
                ],
            },
        ],
        "citations": [],
    }

    segments = _build_response_context_segments(retrieval_meta)
    joined = "\n".join(segment["text"] for segment in segments)

    assert "AI generated table description" not in joined
    assert "Ours | 55.5 | 60.1" in joined
    assert "All = 55.5" in joined


def test_citation_context_segment_preserves_table_metadata():
    citation = {
        "ref": 2,
        "source_text": "Table 3: Main results. Ours | 55.5 | 60.1",
        "display_text": "Ours | 55.5 | 60.1",
        "highlight_text": "Ours | 55.5",
        "context_segment_text": "Table 3: Main results. Ours | 55.5 | 60.1",
        "chunk_type": "table_row",
        "table_id": "Table 3",
        "table_bundle_id": "bundle-table-3",
        "numeric_table_exact_context_row_text": "Ours | 55.5 | 60.1",
        "numeric_table_exact_context_caption": "Table 3: Main results",
        "table_footnote": "Higher is better.",
    }

    segments = _build_context_segments_from_citations(
        [citation],
        query="Table 3 中 Ours 的 All 指标是多少？",
    )

    assert segments[0]["source_ref"] == 2
    assert segments[0]["chunk_type"] == "table_row"
    assert segments[0]["table_id"] == "Table 3"
    assert segments[0]["numeric_table_exact_context_row_text"] == "Ours | 55.5 | 60.1"
    assert segments[0]["table_footnote"] == "Higher is better."
    assert segments[0]["surrounding_context"]


def test_numeric_table_prompt_serializer_uses_structured_exact_row():
    segments = [
        {
            "ref": 1,
            "text": "Table 3: Main results. Ours | 55.5 | 60.1",
            "page_range": [3, 3],
            "chunk_type": "table_row",
            "table_caption": "Table 3: Main results",
            "table_header": "Method | All | Many",
            "numeric_table_exact_context_row_text": "Ours | 55.5 | 60.1",
            "surrounding_context": "The table compares accuracy across splits.",
        }
    ]

    prompt_context = _format_numeric_table_context_segments_for_prompt(segments)

    assert "数值表格证据" in prompt_context
    assert "表题: Table 3: Main results" in prompt_context
    assert "表头: Method | All | Many" in prompt_context
    assert "精确行: Ours | 55.5 | 60.1" in prompt_context
    assert "邻近说明:" in prompt_context


def test_public_retrieval_meta_adds_evidence_raw_without_private_chunks():
    retrieval_meta = {
        "search_query": "Ours 的 All 指标是多少？",
        "evidence_need": ["numeric_table"],
        "_chunks": [
            {
                "chunk": "very long internal chunk",
                "chunk_type": "table_row",
                "rerank_score": 0.9,
            }
        ],
        "citations": [
            {
                "ref": 1,
                "highlight_text": "Ours | 55.5",
                "alignment_status": "span_matched",
                "start_phrase": "Ours",
                "end_phrase": "55.5",
                "best_ratio": 0.92,
                "chunk_type": "table_row",
                "bbox": [10, 20, 120, 40],
                "page_range": [3, 3],
            }
        ],
    }
    segments = _build_context_segments_from_citations(retrieval_meta["citations"], query=retrieval_meta["search_query"])

    public = _build_public_retrieval_meta(retrieval_meta, segments, include_evidence_raw=True)
    raw = _build_evidence_raw_debug(retrieval_meta, segments)

    assert "_chunks" not in public
    assert "evidence_raw" in public
    assert public["evidence_raw"]["schema_version"] == 1
    assert public["evidence_raw"]["source"] == "chatpdf.retrieval_meta.evidence_raw"
    assert public["evidence_raw"]["metadata"]["format"] == "debug_evidence_bundle"
    assert public["evidence_raw"]["metadata"]["limits"]["citations"] == 12
    assert public["evidence_raw"]["metadata"]["limits"]["text_chars"] == 1200
    assert public["evidence_raw"]["metadata"]["processing_info"] == public["evidence_raw"]["counts"]
    assert public["citations"][0]["citation_anchor"]["bbox"] == [10.0, 20.0, 120.0, 40.0]
    assert public["evidence_raw"]["citations"][0]["alignment_status"] == "span_matched"
    assert public["evidence_raw"]["citations"][0]["start_phrase"] == "Ours"
    assert public["evidence_raw"]["citations"][0]["end_phrase"] == "55.5"
    assert public["evidence_raw"]["citations"][0]["best_ratio"] == 0.92
    assert raw["counts"]["chunks"] == 1
    assert raw["schema_version"] == 1
    assert raw["metadata"]["processing_info"]["chunks"] == 1
    assert raw["chunks"][0]["text"] == "very long internal chunk"


def test_evidence_raw_includes_agent_pipeline_diagnostics():
    retrieval_meta = {
        "search_query": "哪一步导致表格证据没有进入答案？",
        "diagnostics": {
            "retrieval": {
                "successful_tool_calls": 2,
                "zero_result_tool_calls": 1,
                "search_result_count": 18,
                "fetched_group_count": 4,
                "detail_count": 6,
                "source_mix": {"vector": 1, "grep": 1},
                "candidate_pool": {
                    "candidate_count": 18,
                    "selected_count": 6,
                    "pages": [1, 2, 3],
                    "selected_pages": [2],
                    "table_ids": ["Table 1", "Table 2"],
                    "selected_table_ids": ["Table 2"],
                    "by_tool": [
                        {
                            "round": 1,
                            "tool": "vector_search",
                            "query": "table result",
                            "result_count": 10,
                            "candidate_count": 10,
                            "selected_count": 3,
                            "pages": [1, 2],
                            "selected_pages": [2],
                            "table_ids": ["Table 2"],
                        }
                    ],
                },
                "final_external_rerank": {"provider": "jina", "output_count": 5},
            },
            "context_assembly": {
                "token_budget_used": 3980,
                "token_budget_limit": 4000,
                "token_budget_ratio": 0.995,
                "parts_before": 9,
                "parts_after": 5,
                "truncated": True,
                "context_chars": 12000,
            },
            "agent": {
                "fallback_reason": "max_tool_calls_reached",
            },
        },
    }

    raw = _build_evidence_raw_debug(retrieval_meta, [])

    pipeline = raw["agent_pipeline"]
    assert pipeline["likely_bottleneck"] == "context_budget"
    assert pipeline["fallback_reason"] == "max_tool_calls_reached"
    assert pipeline["tool_stage"]["successful_tool_calls"] == 2
    assert pipeline["tool_stage"]["candidate_pool"]["selected_table_ids"] == ["Table 2"]
    assert pipeline["tool_stage"]["candidate_pool"]["by_tool"][0]["tool"] == "vector_search"
    assert pipeline["rerank_stage"]["kept_count"] == 5
    assert pipeline["budget_stage"]["token_budget_ratio"] == 0.995
    assert pipeline["budget_stage"]["parts_after"] == 5


def test_public_retrieval_meta_sanitizes_default_diagnostics():
    retrieval_meta = {
        "search_query": "诊断字段是否默认暴露？",
        "diagnostics": {
            "duplicate_chunk_ratio": 0.25,
            "numeric_table_hit_quality": 0.8,
            "source_mix": {"vector": 3, "bm25": 2},
            "candidate_pool": {"ids": ["chunk-secret"], "selected_count": 1},
            "retrieval": {
                "successful_tool_calls": 2,
                "zero_result_tool_calls": 1,
                "search_result_count": 12,
                "source_mix": {"vector": 1},
                "candidate_pool": {
                    "ids": ["chunk-1", "chunk-2"],
                    "by_tool": [{"tool": "vector_search", "query": "private query"}],
                },
            },
            "context_assembly": {
                "token_budget_used": 1800,
                "token_budget_limit": 2400,
                "parts_before": 8,
                "parts_after": 5,
                "truncated": True,
            },
            "agent": {
                "context_budget": {"limit_tokens": 2400, "after_tokens": 1800},
                "sub_questions": ["子问题 A"],
                "planner_invocation_mode": ["native_tools"],
                "planner_rounds": [{"prompt": "private planner prompt"}],
                "errors": [{"message": "private provider error"}],
                "tool_timings": [{"tool": "vector_search", "elapsed_ms": 12}],
            },
        },
    }

    public = _build_public_retrieval_meta(retrieval_meta, [], include_evidence_raw=False)

    diagnostics = public["diagnostics"]
    assert diagnostics["duplicate_chunk_ratio"] == 0.25
    assert diagnostics["numeric_table_hit_quality"] == 0.8
    assert diagnostics["source_mix"] == {"vector": 3, "bm25": 2}
    assert "candidate_pool" not in diagnostics
    assert diagnostics["retrieval"]["successful_tool_calls"] == 2
    assert "candidate_pool" not in diagnostics["retrieval"]
    assert diagnostics["context_assembly"]["truncated"] is True
    assert diagnostics["agent"]["context_budget"]["after_tokens"] == 1800
    assert diagnostics["agent"]["sub_questions"] == ["子问题 A"]
    assert diagnostics["agent"]["planner_invocation_mode"] == ["native_tools"]
    assert "planner_rounds" not in diagnostics["agent"]
    assert "errors" not in diagnostics["agent"]
    assert "tool_timings" not in diagnostics["agent"]
    assert "evidence_raw" not in public


def test_public_retrieval_meta_sanitizes_public_citation_fields():
    long_source = "A" * 1800
    retrieval_meta = {
        "search_query": "引用字段是否默认暴露？",
        "citations": [
            {
                "ref": 1,
                "display_ref": 1,
                "source_ref": 7,
                "group_id": "group-7",
                "page_range": [2, 2],
                "highlight_text": "关键证据",
                "display_text": "可展示证据",
                "source_text": long_source,
                "context_segment_text": "private context segment",
                "_full_text": "private full text",
                "raw_chunk_text": "private raw chunk",
                "bbox": [10, 20, 120, 40],
                "alignment_status": "span_matched",
            }
        ],
    }

    public = _build_public_retrieval_meta(retrieval_meta, [], include_evidence_raw=True)

    citation = public["citations"][0]
    assert citation["ref"] == 1
    assert citation["source_ref"] == 7
    assert citation["highlight_text"] == "关键证据"
    assert citation["citation_anchor"]["bbox"] == [10.0, 20.0, 120.0, 40.0]
    assert len(citation["source_text"]) <= 1400
    assert "_full_text" not in citation
    assert "context_segment_text" not in citation
    assert "raw_chunk_text" not in citation
    raw_citation = public["evidence_raw"]["citations"][0]
    assert raw_citation["text"] == long_source[:1199] + "…"


def test_public_retrieval_meta_sanitizes_context_segments():
    long_text = "B" * 2600
    retrieval_meta = {
        "search_query": "context segments 是否默认暴露内部字段？",
        "citations": [],
    }
    segments = [
        {
            "ref": 1,
            "source_ref": 3,
            "text": long_text,
            "page_range": [4, 4],
            "chunk_type": "table_row",
            "table_id": "Table 4",
            "numeric_table_exact_context_row_text": "Ours | 55.5 | 60.1",
            "surrounding_context": "邻近说明",
            "bbox": [10, 20, 120, 40],
            "context_segment_text": "private context segment",
            "_full_text": "private full text",
            "raw_chunk_text": "private raw chunk",
            "candidate_pool": {"ids": ["private"]},
        }
    ]

    public = _build_public_retrieval_meta(retrieval_meta, segments, include_evidence_raw=False)

    segment = public["context_segments"][0]
    assert segment["ref"] == 1
    assert segment["source_ref"] == 3
    assert segment["table_id"] == "Table 4"
    assert segment["numeric_table_exact_context_row_text"] == "Ours | 55.5 | 60.1"
    assert segment["bbox"] == [10.0, 20.0, 120.0, 40.0]
    assert len(segment["text"]) <= 2400
    assert "context_segment_text" not in segment
    assert "_full_text" not in segment
    assert "raw_chunk_text" not in segment
    assert "candidate_pool" not in segment


def test_agent_detail_citation_extracts_row_from_structured_row_shard_text():
    detail = {
        "retrieval_type": "agent_search_result",
        "chunk_type": "table_row",
        "table_id": "Table 3",
        "table_bundle_id": "bundle-table-3",
        "context_id": "bundle-table-3:rows:1-3",
        "evidence_id": "bundle-table-3:rows:1-3",
        "text": "\n".join(
            [
                "[Structured Table Row Shard]",
                "Table 3: Main results",
                "",
                "[Header]",
                "Method | All | Many | Medium | Few",
                "",
                "[Rows]",
                "Baseline | 52.1 | 58.0 | 49.2 | 38.0",
                "Ours | 55.5 | 60.1 | 54.2 | 41.8",
            ]
        ),
        "page_range": [3, 3],
    }

    citations = _build_agent_detail_citations(
        [detail],
        query="Table 3 中 Ours 的 All 指标是多少？",
        max_citations=1,
    )

    assert citations
    assert citations[0]["chunk_type"] == "table_row"
    assert citations[0]["numeric_table_exact_context_row_text"] == "Ours | 55.5 | 60.1 | 54.2 | 41.8"
    assert citations[0]["display_text"] == "Ours | 55.5 | 60.1 | 54.2 | 41.8"
