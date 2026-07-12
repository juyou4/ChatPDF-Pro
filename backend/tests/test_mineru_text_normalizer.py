from services.mineru_text_normalizer import (
    normalize_mineru_for_rag,
    validate_mineru_rag_data,
)
from services.embedding_service import (
    _append_structured_table_bundle_chunks,
    _augment_with_numeric_exact_row_search,
    _ensure_structured_table_row_shard_results,
    _resolve_structured_table_row_shard_limit,
    _structured_table_row_text,
)


def _middle_table_block(table_html, bbox, *, lines_deleted=False):
    body = {
        "type": "table_body",
        "bbox": bbox,
        "lines": [{"spans": [{"type": "table", "bbox": bbox, "html": table_html}]}] if table_html else [],
    }
    if lines_deleted:
        body["lines_deleted"] = True
    return {"type": "table", "blocks": [body]}


def test_normalize_mineru_content_list_without_middle_json_filters_artifacts():
    payload = {
        "content_list_json": [
            {"type": "header", "text": "8 S. Liu et al.", "page_idx": 0},
            {"type": "page_number", "text": "8", "page_idx": 0},
            {"type": "page_footnote", "text": "1 This is a footnote", "page_idx": 0},
            {"type": "aside_text", "text": "arXiv:2303.05499", "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "page_idx": 0},
            {"type": "list", "text": "First contribution", "page_idx": 0},
            {"type": "code", "text": "for i in range(n): pass", "page_idx": 1},
            {"type": "equation", "latex": "E = mc^2", "page_idx": 1},
            {
                "type": "image",
                "img_path": "images/foo.jpg",
                "image_caption": "Fig. 1. Model overview",
                "page_idx": 1,
            },
            {
                "type": "chart",
                "chart_path": "charts/bar.png",
                "chart_caption": "Chart 1. Accuracy trend",
                "chart_footnote": "Values are averaged.",
                "page_idx": 1,
            },
        ]
    }

    normalized = normalize_mineru_for_rag(payload, source_hash="hash")

    assert normalized["index_source"] == "mineru"
    assert normalized["source_hash"] == "hash"
    assert len(normalized["pages"]) == 2
    full_text = normalized["full_text"]
    assert "1 Introduction" in full_text
    assert "First contribution" in full_text
    assert "for i in range" in full_text
    assert "$$\nE = mc^2\n$$" in full_text
    assert "Fig. 1. Model overview" in full_text
    assert "Chart 1. Accuracy trend" in full_text
    assert "Values are averaged." in full_text
    assert "images/foo.jpg" not in full_text
    assert "charts/bar.png" not in full_text
    assert "8 S. Liu et al." not in full_text
    assert "arXiv:2303.05499" not in full_text
    assert normalized["quality_report"]["filtered_type_counts"]["header"] == 1
    assert normalized["quality_report"]["filtered_type_counts"]["aside_text"] == 1


def test_table_html_generates_markdown_bundle_and_evidence_units():
    payload = {
        "content_list_json": [
            {
                "type": "table",
                "page_idx": 2,
                "bbox": [100, 200, 900, 420],
                "table_caption": "Table 1: Results",
                "table_footnote": "Higher is better.",
                "table_body": """
                    <table>
                      <tr><td rowspan="2">Model</td><td colspan="2">Score</td></tr>
                      <tr><td>Acc</td><td>F1</td></tr>
                      <tr><td>A</td><td>90</td><td>88</td></tr>
                    </table>
                """,
            }
        ]
    }

    normalized = normalize_mineru_for_rag(payload)
    assert len(normalized["structured_table_bundles"]) == 1

    bundle = normalized["structured_table_bundles"][0]
    body = bundle["table_body_markdown"]
    assert "| Model | Score | Score |" in body
    assert "| Model | Acc | F1 |" in body
    assert "| A | 90 | 88 |" in body
    assert "<td" not in body.lower()
    assert "<tr" not in body.lower()
    assert bundle["source"] == "mineru"
    assert bundle["pages"] == [3]
    assert bundle["page_start"] == 3
    assert len(bundle["evidence_units"]) == 3
    assert bundle["evidence_units"][0]["is_header_row"] is True
    assert len(bundle["evidence_units"][0]["cell_evidence_units"]) == 3
    data_cells = bundle["evidence_units"][2]["cell_evidence_units"]
    assert data_cells[0]["header_path"] == "Model"
    assert data_cells[0]["col_id"] == "Model"
    assert data_cells[1]["header_path"] == "Score Acc"
    assert data_cells[1]["column_header"] == "Score Acc"
    assert data_cells[2]["header_path"] == "Score F1"
    assert data_cells[2]["bbox"] == [100.0, 200.0, 900.0, 420.0]

    geometry_normalized = normalize_mineru_for_rag(payload, page_sizes={3: [600, 800]})
    geometry_bundle = geometry_normalized["structured_table_bundles"][0]
    assert geometry_bundle["raw_bbox"] == [100.0, 200.0, 900.0, 420.0]
    assert geometry_bundle["bbox_coordinate_space"] == "normalized_0_1000"
    assert geometry_bundle["visual_bbox"] == [60.0, 160.0, 540.0, 336.0]
    assert geometry_bundle["evidence_units"][2]["visual_bbox"] == [60.0, 160.0, 540.0, 336.0]

    full_text = normalized["full_text"]
    assert "[TABLE] Table 1: Results" in full_text
    assert "Higher is better." in full_text
    assert "<td" not in full_text.lower()
    assert "<tr" not in full_text.lower()

    ok, failures = validate_mineru_rag_data(normalized, original_full_text="x" * 50)
    assert ok, failures


def test_mineru_uses_explicit_cell_geometry_and_refuses_table_bbox_as_row_crop():
    payload = {
        "content_list_json": [
            {
                "type": "table",
                "page_idx": 0,
                "bbox": [100, 100, 900, 500],
                "table_body": "<table><tr><td>Method</td><td>Score</td></tr><tr><td>A</td><td>90</td></tr></table>",
                "table_cells": [
                    {"row_idx": 0, "col_idx": 0, "bbox": [100, 100, 500, 200]},
                    {"row_idx": 0, "col_idx": 1, "bbox": [500, 100, 900, 200]},
                    {"row_idx": 1, "col_idx": 0, "bbox": [100, 200, 500, 300]},
                    {"row_idx": 1, "col_idx": 1, "bbox": [500, 200, 900, 300]},
                ],
            }
        ]
    }
    normalized = normalize_mineru_for_rag(payload, page_sizes={1: [600, 800]})
    rows = normalized["structured_table_bundles"][0]["evidence_units"]

    assert rows[1]["bbox"] == [100.0, 200.0, 900.0, 300.0]
    assert rows[1]["visual_bbox"] == [60.0, 160.0, 540.0, 240.0]
    assert rows[1]["visual_crop_eligible"] is True
    assert rows[1]["cell_evidence_units"][1]["visual_crop_eligible"] is True

    fallback = normalize_mineru_for_rag({
        "content_list_json": [{
            "type": "table", "page_idx": 0, "bbox": [100, 100, 900, 500],
            "table_body": "<table><tr><td>Method</td><td>Score</td></tr><tr><td>A</td><td>90</td></tr></table>",
        }]
    })
    assert fallback["structured_table_bundles"][0]["evidence_units"][1]["visual_crop_eligible"] is False


def test_mineru_middle_json_enriches_only_table_overview_geometry():
    table_html = "<table><tr><td>Method</td><td>Score</td></tr><tr><td>A</td><td>90</td></tr></table>"
    payload = {
        "content_list_json": [{
            "type": "table",
            "page_idx": 0,
            "table_body": table_html,
        }],
        "middle_json": {
            "pdf_info": [{
                "page_idx": 0,
                "page_size": [2000, 1000],
                "para_blocks": [_middle_table_block(table_html, [200, 100, 1800, 800])],
            }],
        },
    }

    normalized = normalize_mineru_for_rag(payload, page_sizes={1: [600, 800]})
    bundle = normalized["structured_table_bundles"][0]

    assert bundle["bounding_box"] == [100.0, 100.0, 900.0, 800.0]
    assert bundle["visual_bbox"] == [60.0, 80.0, 540.0, 640.0]
    assert bundle["table_geometry_source"] == "middle_json_table"
    assert bundle["row_cell_geometry_available"] is False
    assert all(row["visual_crop_eligible"] is False for row in bundle["evidence_units"])
    assert all(
        cell["visual_crop_eligible"] is False
        for row in bundle["evidence_units"]
        for cell in row["cell_evidence_units"]
    )


def test_mineru_middle_json_matches_same_page_tables_by_html_not_position():
    table_a = "<table><tr><td>Method</td></tr><tr><td>A</td></tr></table>"
    table_b = "<table><tr><td>Method</td></tr><tr><td>B</td></tr></table>"
    payload = {
        "content_list_json": [
            {"type": "table", "page_idx": 0, "table_body": table_b},
            {"type": "table", "page_idx": 0, "table_body": table_a},
        ],
        "middle_json": {
            "pdf_info": [{
                "page_idx": 0,
                "page_size": [1000, 1000],
                "para_blocks": [
                    _middle_table_block(table_a, [50, 100, 450, 400]),
                    _middle_table_block(table_b, [550, 100, 950, 400]),
                ],
            }],
        },
    }

    bundles = normalize_mineru_for_rag(payload)["structured_table_bundles"]

    assert bundles[0]["bounding_box"] == [550.0, 100.0, 950.0, 400.0]
    assert bundles[1]["bounding_box"] == [50.0, 100.0, 450.0, 400.0]


def test_mineru_middle_json_does_not_assign_bbox_to_an_unmatched_table():
    content_table = "<table><tr><td>Method</td></tr><tr><td>A</td></tr></table>"
    other_table = "<table><tr><td>Method</td></tr><tr><td>B</td></tr></table>"
    payload = {
        "content_list_json": [{"type": "table", "page_idx": 0, "table_body": content_table}],
        "middle_json": {
            "pdf_info": [{
                "page_idx": 0,
                "page_size": [1000, 1000],
                "para_blocks": [_middle_table_block(other_table, [100, 100, 900, 500])],
            }],
        },
    }

    normalized = normalize_mineru_for_rag(payload)
    bundle = normalized["structured_table_bundles"][0]

    assert bundle["bounding_box"] == []
    assert bundle["visual_bbox"] == []
    assert any("middle_json_table_geometry_unmatched" in warning for warning in normalized["quality_report"]["warnings"])
    assert all(row["visual_crop_eligible"] is False for row in bundle["evidence_units"])


def test_mineru_middle_json_cross_page_merge_marks_bundle_as_multi_page():
    merged_table = "<table><tr><td>Method</td><td>Score</td></tr><tr><td>A</td><td>90</td></tr><tr><td>B</td><td>91</td></tr></table>"
    payload = {
        "content_list_json": [{"type": "table", "page_idx": 0, "table_body": merged_table}],
        "middle_json": {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [1000, 1000],
                    "para_blocks": [_middle_table_block(merged_table, [100, 100, 900, 900])],
                },
                {
                    "page_idx": 1,
                    "page_size": [1000, 1000],
                    "para_blocks": [_middle_table_block("", [100, 100, 900, 600], lines_deleted=True)],
                },
            ],
        },
    }

    bundle = normalize_mineru_for_rag(payload)["structured_table_bundles"][0]

    assert bundle["pages"] == [1, 2]
    assert bundle["page_start"] == 1
    assert bundle["page_end"] == 2
    assert all(row["visual_crop_eligible"] is False for row in bundle["evidence_units"])


def test_table_quality_gate_allows_escaped_pipe_inside_cells():
    payload = {
        "content_list_json": [
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": "Table 2: Results",
                "table_body": (
                    "<table>"
                    "<tr><td>Model</td><td>|Zero-Shot</td><td>Fine-Tune</td></tr>"
                    "<tr><td>Grounding DINO</td><td>|48.4</td><td>57.2</td></tr>"
                    "</table>"
                ),
            }
        ]
    }

    normalized = normalize_mineru_for_rag(payload)
    body = normalized["structured_table_bundles"][0]["table_body_markdown"]

    assert "\\|Zero-Shot" in body
    assert "\\|48.4" in body
    ok, failures = validate_mineru_rag_data(normalized)
    assert ok, failures


def test_quality_gate_rejects_html_in_full_text():
    normalized = {
        "full_text": "<table><tr><td>bad</td></tr></table>",
        "pages": [{"page": 1, "content": "<td>bad</td>"}],
        "structured_table_bundles": [],
    }

    ok, failures = validate_mineru_rag_data(normalized)

    assert ok is False
    assert "html_tag_in_full_text" in failures


def test_mineru_table_bundle_is_compatible_with_embedding_table_chunks():
    normalized = normalize_mineru_for_rag({
        "content_list_json": [
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": "Table 2: Numeric QA",
                "table_body": "<table><tr><td>Method</td><td>Score</td></tr><tr><td>Ours</td><td>91.2</td></tr></table>",
            }
        ]
    })
    chunks = []
    chunk_headings = []
    chunk_pages = []
    chunk_types = []
    chunk_metadata = []

    _append_structured_table_bundle_chunks(
        "doc-mineru-table",
        chunks,
        chunk_headings,
        chunk_pages,
        chunk_types,
        chunk_metadata,
        normalized["structured_table_bundles"],
    )

    assert len(chunks) == 3
    assert "[Structured Table Bundle]" in chunks[0]
    assert "[Structured Table Exact Row]" in chunks[1]
    assert "[Structured Table Row Shard]" in chunks[2]
    assert "Method: Ours; Score: 91.2" in chunks[1]
    assert chunk_types == ["table", "table_row", "table_row"]
    assert chunk_pages == [1, 1, 1]
    assert chunk_metadata[0]["structured_table_bundle"] is True
    assert chunk_metadata[1]["table_row_slice_kind"] == "exact"
    assert chunk_metadata[1]["numeric_table_exact_context_row_text"] == "Method: Ours; Score: 91.2"
    assert chunk_metadata[2]["table_row_shard"] is True
    assert chunk_metadata[0]["source"] == "mineru"
    assert chunk_metadata[0]["evidence_units"]
    assert chunk_metadata[0]["evidence_units"][1]["row_text"] == "Method: Ours; Score: 91.2"
    assert chunk_metadata[0]["evidence_units"][1]["raw_row_text"] == "Ours | 91.2"


def test_large_structured_table_bundle_adds_row_shards_for_vector_recall():
    row_units = [
        {
            "evidence_unit_type": "table_row",
            "is_header_row": True,
            "row_id": "header",
            "row_text": "Method | Score",
        }
    ]
    body_rows = []
    for index in range(1, 26):
        row_text = f"Method{index} | {index / 10:.1f}"
        body_rows.append(row_text)
        row_units.append({
            "evidence_unit_type": "table_row",
            "row_id": f"Method{index}",
            "row_text": row_text,
            "page": 3,
        })
    bundle = {
        "bundle_id": "table-bundle-1",
        "table_id": "Table 9",
        "table_caption": "Table 9: Long numeric results",
        "table_header": "Method | Score",
        "table_body_markdown": "\n".join(body_rows),
        "page_start": 3,
        "page_end": 4,
        "pages": [3, 4],
        "source": "mineru",
        "evidence_units": row_units,
    }
    chunks = []
    chunk_headings = []
    chunk_pages = []
    chunk_types = []
    chunk_metadata = []

    _append_structured_table_bundle_chunks(
        "doc-long-table",
        chunks,
        chunk_headings,
        chunk_pages,
        chunk_types,
        chunk_metadata,
        [bundle],
    )

    assert len(chunk_types) == 29
    assert chunk_types[0] == "table"
    assert all(chunk_type == "table_row" for chunk_type in chunk_types[1:])
    assert chunk_metadata[0]["structured_table_bundle"] is True
    assert chunk_metadata[0].get("table_row_shard") is None
    assert "Method25 | 2.5" in chunk_metadata[0]["table_body_markdown"]

    exact_row_metadata = chunk_metadata[1:26]
    assert len(exact_row_metadata) == 25
    assert all(metadata["table_row_slice_kind"] == "exact" for metadata in exact_row_metadata)
    assert all(metadata["row_count"] == 1 for metadata in exact_row_metadata)
    assert exact_row_metadata[-1]["numeric_table_exact_context_row_text"] == "Method: Method25; Score: 2.5"

    shard_metadata = chunk_metadata[-3:]
    assert [metadata["row_count"] for metadata in shard_metadata] == [10, 10, 5]
    assert all(metadata["table_row_shard"] is True for metadata in shard_metadata)
    assert all(metadata["table_bundle_id"] == "table-bundle-1" for metadata in shard_metadata)
    assert all(len(metadata["evidence_units"]) <= 10 for metadata in shard_metadata)
    assert "Method: Method25; Score: 2.5" in chunks[-1]
    assert "Method1 | 0.1" not in chunks[-1]
    assert len(chunks[-1]) < len(chunks[0])


def test_structured_table_row_text_binds_headers_for_legacy_raw_rows():
    row = {
        "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
        "row_text": "AttnRes | 63.5 | 34.2 | 48.1 | 72.4",
        "cell_evidence_units": [
            {"content": "AttnRes"},
            {"content": "63.5"},
            {"content": "34.2"},
            {"content": "48.1"},
            {"content": "72.4"},
        ],
    }

    assert _structured_table_row_text(row) == (
        "Method: AttnRes; MMLU: 63.5; GPQA-Diamond: 34.2; "
        "HumanEval: 48.1; C-Eval: 72.4"
    )


def test_structured_table_row_shard_is_promoted_from_same_table_final_context():
    results = [
        {
            "chunk_id": 0,
            "chunk": "[Structured Table Bundle]\nTable 3: Benchmark results.\nBaseline and AttnRes are compared.",
            "chunk_type": "table",
            "structured_table_bundle": True,
            "table_id": "Table 3",
            "table_bundle_id": "mineru:p10:table:128",
            "page": 10,
        },
        {
            "chunk_id": 1,
            "chunk": "Figure 6 discusses validation loss.",
            "chunk_type": "text",
            "page": 9,
        },
    ]
    chunks = [
        results[0]["chunk"],
        results[1]["chunk"],
        (
            "[Structured Table Row Shard]\n"
            "Table 3: Benchmark results.\n"
            "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval\n"
            "Baseline | 60.1 | 32.0 | 45.0 | 70.0\n"
            "AttnRes | 63.5 | 34.2 | 48.1 | 72.4"
        ),
    ]
    chunk_pages = [10, 9, 10]
    chunk_types = ["table", "text", "table_row"]
    chunk_metadata = [
        {
            "structured_table_bundle": True,
            "table_id": "Table 3",
            "table_bundle_id": "mineru:p10:table:128",
            "page": 10,
        },
        {},
        {
            "structured_table_bundle": True,
            "table_row_shard": True,
            "table_row_slice_kind": "shard",
            "parent_table_bundle_id": "mineru:p10:table:128",
            "table_id": "Table 3",
            "table_caption": "Benchmark results",
            "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
            "row_start": 1,
            "row_end": 10,
            "page": 10,
        },
    ]

    final = _ensure_structured_table_row_shard_results(
        results,
        chunks=chunks,
        chunk_pages=chunk_pages,
        chunk_types=chunk_types,
        chunk_metadata=chunk_metadata,
        query="Table 3 中 Baseline 和 AttnRes 在 MMLU、GPQA-Diamond、HumanEval、C-Eval 上分别是多少？",
        top_k=3,
    )

    assert any(item.get("chunk_type") == "table_row" and item.get("table_row_shard") for item in final)
    assert "AttnRes" in final[-1]["chunk"]


def test_structured_table_exact_row_is_promoted_from_same_table_final_context():
    results = [
        {
            "chunk_id": 0,
            "chunk": "[Structured Table Bundle]\nTable 3: Benchmark results.",
            "chunk_type": "table",
            "structured_table_bundle": True,
            "table_id": "Table 3",
            "table_bundle_id": "mineru:p10:table:128",
            "page": 10,
        },
        {
            "chunk_id": 1,
            "chunk": "Figure 6 discusses validation loss.",
            "chunk_type": "text",
            "page": 9,
        },
    ]
    exact_row = (
        "[Structured Table Exact Row]\n"
        "Table 3: Benchmark results.\n\n"
        "[Header]\nMethod | MMLU | GPQA-Diamond | HumanEval | C-Eval\n\n"
        "[Row]\nMethod: AttnRes; MMLU: 63.5; GPQA-Diamond: 34.2; HumanEval: 48.1; C-Eval: 72.4"
    )
    shard = (
        "[Structured Table Row Shard]\n"
        "Table 3: Benchmark results.\n"
        "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval\n"
        "Baseline | 60.1 | 32.0 | 45.0 | 70.0\n"
        "AttnRes | 63.5 | 34.2 | 48.1 | 72.4"
    )

    final = _ensure_structured_table_row_shard_results(
        results,
        chunks=[results[0]["chunk"], results[1]["chunk"], exact_row, shard],
        chunk_pages=[10, 9, 10, 10],
        chunk_types=["table", "text", "table_row", "table_row"],
        chunk_metadata=[
            {
                "structured_table_bundle": True,
                "table_id": "Table 3",
                "table_bundle_id": "mineru:p10:table:128",
                "page": 10,
            },
            {},
            {
                "structured_table_bundle": True,
                "table_row_evidence": True,
                "table_row_slice_kind": "exact",
                "parent_table_bundle_id": "mineru:p10:table:128",
                "table_bundle_id": "mineru:p10:table:128",
                "table_id": "Table 3",
                "table_caption": "Table 3: Benchmark results",
                "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
                "numeric_table_exact_context_row_text": (
                    "Method: AttnRes; MMLU: 63.5; GPQA-Diamond: 34.2; "
                    "HumanEval: 48.1; C-Eval: 72.4"
                ),
                "row_start": 2,
                "row_end": 2,
                "page": 10,
            },
            {
                "structured_table_bundle": True,
                "table_row_shard": True,
                "table_row_slice_kind": "shard",
                "parent_table_bundle_id": "mineru:p10:table:128",
                "table_id": "Table 3",
                "table_caption": "Table 3: Benchmark results",
                "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
                "row_start": 1,
                "row_end": 10,
                "page": 10,
            },
        ],
        query="Table 3 中 AttnRes 的 MMLU 是多少？",
        top_k=3,
    )

    promoted_rows = [item for item in final if item.get("chunk_type") == "table_row"]
    assert promoted_rows
    assert promoted_rows[0]["table_row_slice_kind"] == "exact"
    assert promoted_rows[0]["numeric_table_exact_context_row_text"].startswith("Method: AttnRes")


def test_numeric_exact_row_search_adds_matching_single_row_candidate():
    chunks = [
        "[Structured Table Bundle]\nTable 3: Benchmark results.",
        "Figure 6 discusses validation loss.",
        (
            "[Structured Table Exact Row]\n"
            "Table 3: Benchmark results.\n\n"
            "[Header]\nMethod | MMLU | GPQA-Diamond | HumanEval | C-Eval\n\n"
            "[Row]\nMethod: AttnRes; MMLU: 63.5; GPQA-Diamond: 34.2; HumanEval: 48.1; C-Eval: 72.4"
        ),
        (
            "[Structured Table Exact Row]\n"
            "Table 3: Benchmark results.\n\n"
            "[Header]\nMethod | MMLU | GPQA-Diamond | HumanEval | C-Eval\n\n"
            "[Row]\nMethod: Baseline; MMLU: 60.1; GPQA-Diamond: 32.0; HumanEval: 45.0; C-Eval: 70.0"
        ),
    ]
    metadata = [
        {"structured_table_bundle": True, "table_id": "Table 3", "table_bundle_id": "mineru:p10:table:128"},
        {},
        {
            "structured_table_bundle": True,
            "table_row_evidence": True,
            "table_row_slice_kind": "exact",
            "parent_table_bundle_id": "mineru:p10:table:128",
            "table_bundle_id": "mineru:p10:table:128",
            "table_id": "Table 3",
            "table_caption": "Table 3: Benchmark results",
            "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
            "numeric_table_exact_context_row_text": (
                "Method: AttnRes; MMLU: 63.5; GPQA-Diamond: 34.2; "
                "HumanEval: 48.1; C-Eval: 72.4"
            ),
            "row_start": 2,
            "row_end": 2,
            "page": 10,
        },
        {
            "structured_table_bundle": True,
            "table_row_evidence": True,
            "table_row_slice_kind": "exact",
            "parent_table_bundle_id": "mineru:p10:table:128",
            "table_bundle_id": "mineru:p10:table:128",
            "table_id": "Table 3",
            "table_caption": "Table 3: Benchmark results",
            "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
            "numeric_table_exact_context_row_text": (
                "Method: Baseline; MMLU: 60.1; GPQA-Diamond: 32.0; "
                "HumanEval: 45.0; C-Eval: 70.0"
            ),
            "row_start": 1,
            "row_end": 1,
            "page": 10,
        },
    ]

    augmented = _augment_with_numeric_exact_row_search(
        [
            {
                "chunk_id": 0,
                "chunk": chunks[0],
                "chunk_type": "table",
                "table_id": "Table 3",
                "table_bundle_id": "mineru:p10:table:128",
                "page": 10,
            }
        ],
        chunks=chunks,
        chunk_pages=[10, 9, 10, 10],
        chunk_types=["table", "text", "table_row", "table_row"],
        chunk_metadata=metadata,
        pages=[{"page": 10, "content": "Table 3 page"}],
        page_index=None,
        query="Table 3 中 AttnRes 的 MMLU 是多少？",
        evidence_need=["numeric_table"],
    )

    added_rows = [item for item in augmented if item.get("numeric_exact_search")]
    assert added_rows
    assert added_rows[0]["table_row_slice_kind"] == "exact"
    assert added_rows[0]["numeric_table_exact_context_row_text"].startswith("Method: AttnRes")
    assert added_rows[0]["page"] == 10


def test_numeric_exact_row_search_regex_locator_adds_winner_style_column_rows():
    chunks = [
        "[Structured Table Bundle]\nTable 1: Generated sample quality.",
        (
            "[Structured Table Exact Row]\n"
            "Table 1: Generated sample quality.\n\n"
            "[Header]\nMethod | Accuracy | FID\n\n"
            "[Row]\nMethod: DDPM; Accuracy: 46.6; FID: 5.86"
        ),
        (
            "[Structured Table Exact Row]\n"
            "Table 1: Generated sample quality.\n\n"
            "[Header]\nMethod | Accuracy | FID\n\n"
            "[Row]\nMethod: CBDM; Accuracy: 51.5; FID: 4.71"
        ),
    ]
    metadata = [
        {"structured_table_bundle": True, "table_id": "Table 1", "table_bundle_id": "mineru:p4:table:1"},
        {
            "structured_table_bundle": True,
            "table_row_evidence": True,
            "table_row_slice_kind": "exact",
            "table_id": "Table 1",
            "table_bundle_id": "mineru:p4:table:1",
            "table_caption": "Table 1: Generated sample quality",
            "table_header": "Method | Accuracy | FID",
            "numeric_table_exact_context_row_text": "Method: DDPM; Accuracy: 46.6; FID: 5.86",
            "row_id": "DDPM",
            "page": 4,
        },
        {
            "structured_table_bundle": True,
            "table_row_evidence": True,
            "table_row_slice_kind": "exact",
            "table_id": "Table 1",
            "table_bundle_id": "mineru:p4:table:1",
            "table_caption": "Table 1: Generated sample quality",
            "table_header": "Method | Accuracy | FID",
            "numeric_table_exact_context_row_text": "Method: CBDM; Accuracy: 51.5; FID: 4.71",
            "row_id": "CBDM",
            "page": 4,
        },
    ]

    augmented = _augment_with_numeric_exact_row_search(
        [{"chunk_id": 0, "chunk": chunks[0], "chunk_type": "table", "table_id": "Table 1", "page": 4}],
        chunks=chunks,
        chunk_pages=[4, 4, 4],
        chunk_types=["table", "table_row", "table_row"],
        chunk_metadata=metadata,
        pages=[{"page": 4, "content": "Table 1 page"}],
        page_index=None,
        query="表 1 中哪种生成模型取得了最高分类准确率？",
        evidence_need=["numeric_table"],
    )

    added_rows = [item for item in augmented if item.get("numeric_regex_locator")]
    assert len(added_rows) >= 2
    assert all("regex:column_numeric_row" in item.get("numeric_regex_locator_hits", []) for item in added_rows)
    assert {item.get("row_id") for item in added_rows} >= {"DDPM", "CBDM"}


def test_structured_table_row_shard_keeps_explanatory_text_when_replacing_full_context():
    table = {
        "chunk_id": 0,
        "chunk": "[Structured Table Bundle]\nTable 3: Benchmark results.\nBaseline and AttnRes are compared.",
        "chunk_type": "table",
        "structured_table_bundle": True,
        "table_id": "Table 3",
        "table_bundle_id": "mineru:p10:table:128",
        "page": 10,
    }
    explanatory = {
        "chunk_id": 1,
        "chunk": "The benchmark evaluates reasoning and coding performance across multiple datasets.",
        "chunk_type": "text",
        "page": 10,
    }
    off_topic = {
        "chunk_id": 2,
        "chunk": "Appendix implementation notes mention the UI demo and unrelated settings.",
        "chunk_type": "text",
        "page": 2,
    }
    shard = (
        "[Structured Table Row Shard]\n"
        "Table 3: Benchmark results.\n"
        "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval\n"
        "Baseline | 60.1 | 32.0 | 45.0 | 70.0\n"
        "AttnRes | 63.5 | 34.2 | 48.1 | 72.4"
    )

    final = _ensure_structured_table_row_shard_results(
        [table, explanatory, off_topic],
        chunks=[table["chunk"], explanatory["chunk"], off_topic["chunk"], shard],
        chunk_pages=[10, 10, 2, 10],
        chunk_types=["table", "text", "text", "table_row"],
        chunk_metadata=[
            {"structured_table_bundle": True, "table_id": "Table 3", "table_bundle_id": "mineru:p10:table:128", "page": 10},
            {},
            {},
            {
                "structured_table_bundle": True,
                "table_row_shard": True,
                "table_row_slice_kind": "shard",
                "parent_table_bundle_id": "mineru:p10:table:128",
                "table_id": "Table 3",
                "table_caption": "Table 3: Benchmark results",
                "table_header": "Method | MMLU | GPQA-Diamond | HumanEval | C-Eval",
                "row_start": 1,
                "row_end": 10,
                "page": 10,
            },
        ],
        query="Table 3 中 Baseline 和 AttnRes 在 MMLU、GPQA-Diamond、HumanEval、C-Eval 上分别是多少？",
        top_k=3,
    )

    chunks = [item["chunk"] for item in final]
    assert any(item.get("chunk_type") == "table_row" and item.get("table_row_shard") for item in final)
    assert explanatory["chunk"] in chunks
    assert off_topic["chunk"] not in chunks


def test_structured_table_row_shard_limit_expands_for_comparison_queries():
    assert _resolve_structured_table_row_shard_limit(
        "Table 3 中 Baseline 和 AttnRes 相比 Other 在 MMLU、HumanEval 上分别是多少？",
        default_limit=1,
    ) >= 3
