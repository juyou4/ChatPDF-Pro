"""结构化 table bundle 保留链路回归测试"""

import os
import pickle
import sys
from unittest.mock import patch

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.embedding_service as embedding_service
from services.odl_parser_service import _attach_structured_bundles_to_pages, _build_structured_table_bundles
from services.table_aware_service import (
    bind_nearest_table_captions,
    build_structured_table_bundle,
    extract_table_caption_candidates_from_text_dict,
    merge_pdf_native_structured_table_bundles,
)


EMBED_DIM = 16


def _make_table(rows):
    return {
        "type": "table",
        "rows": [
            {
                "type": "table row",
                "row number": row_idx + 1,
                "cells": [
                    {
                        "type": "table cell",
                        "row number": row_idx + 1,
                        "column number": col_idx + 1,
                        "content": cell,
                    }
                    for col_idx, cell in enumerate(row)
                ],
            }
            for row_idx, row in enumerate(rows)
        ],
    }


def _make_evidence_units(bundle_key="id:100", table_id="Table 7", table_caption="Table 7: Main results", source_id=100, page=1):
    return [
        {
            "evidence_unit_id": f"{bundle_key}::table_row::source:{source_id}::r1",
            "evidence_unit_type": "table_row",
            "table_bundle_id": bundle_key,
            "table_id": table_id,
            "table_caption": table_caption,
            "source_id": source_id,
            "page": page,
            "row_idx": 1,
            "bbox": [1, 2, 3, 4],
            "content": "Method | All",
            "cell_count": 2,
            "cell_evidence_units": [
                {
                    "evidence_unit_id": f"{bundle_key}::table_cell::source:{source_id}::r1::c1",
                    "evidence_unit_type": "table_cell",
                    "table_bundle_id": bundle_key,
                    "table_id": table_id,
                    "table_caption": table_caption,
                    "source_id": source_id,
                    "page": page,
                    "row_idx": 1,
                    "col_idx": 1,
                    "col_id": "Method",
                    "column_header": "Method",
                    "header_path": "Method",
                    "row_span": 1,
                    "col_span": 1,
                    "bbox": [1, 2, 2, 4],
                    "content": "Method",
                    "source": "odl",
                },
                {
                    "evidence_unit_id": f"{bundle_key}::table_cell::source:{source_id}::r1::c2",
                    "evidence_unit_type": "table_cell",
                    "table_bundle_id": bundle_key,
                    "table_id": table_id,
                    "table_caption": table_caption,
                    "source_id": source_id,
                    "page": page,
                    "row_idx": 1,
                    "col_idx": 2,
                    "col_id": "All",
                    "column_header": "All",
                    "header_path": "All",
                    "row_span": 1,
                    "col_span": 1,
                    "bbox": [2, 2, 3, 4],
                    "content": "All",
                    "source": "odl",
                },
            ],
            "source": "odl",
        }
    ]


def _fake_embed_fn(texts):
    vectors = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    for idx, text in enumerate(texts):
        vectors[idx, idx % EMBED_DIM] = 1.0
        vectors[idx, (idx + 1) % EMBED_DIM] = float(max(len(text), 1))
    faiss.normalize_L2(vectors)
    return vectors


def test_odl_structured_table_bundles_merge_cross_page_chain():
    elements = [
        {
            "type": "caption",
            "id": 10,
            "page number": 1,
            "linked content id": 100,
            "content": "Table 7: Main results",
        },
        {
            **_make_table(
                [
                    ["Method", "All"],
                    ["Baseline", "52.1"],
                ]
            ),
            "id": 100,
            "page number": 1,
            "number of rows": 2,
            "number of columns": 2,
            "next table id": 101,
        },
        {
            **_make_table(
                [
                    ["Method", "All"],
                    ["Ours", "55.5"],
                ]
            ),
            "id": 101,
            "page number": 2,
            "number of rows": 2,
            "number of columns": 2,
            "previous table id": 100,
        },
    ]

    bundles = _build_structured_table_bundles(elements)

    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["table_id"] == "Table 7"
    assert bundle["table_caption"] == "Table 7: Main results"
    assert bundle["pages"] == [1, 2]
    assert bundle["page_start"] == 1
    assert bundle["page_end"] == 2
    assert bundle["source_ids"] == [100, 101]
    assert "Baseline" in bundle["table_body_markdown"]
    assert "Ours" in bundle["table_body_markdown"]


def test_odl_structured_table_bundles_include_typed_evidence_units():
    elements = [
        {
            "type": "caption",
            "id": 10,
            "page number": 1,
            "linked content id": 100,
            "content": "Table 7: Main results",
        },
        {
            "type": "table",
            "id": 100,
            "page number": 1,
            "bounding box": [1, 2, 3, 4],
            "number of rows": 2,
            "number of columns": 2,
            "rows": [
                {
                    "type": "table row",
                    "row number": 1,
                    "bounding box": [1, 2, 3, 4],
                    "cells": [
                        {
                            "type": "table cell",
                            "row number": 1,
                            "column number": 1,
                            "row span": 1,
                            "column span": 1,
                            "bounding box": [1, 2, 2, 4],
                            "content": "Method",
                        },
                        {
                            "type": "table cell",
                            "row number": 1,
                            "column number": 2,
                            "row span": 1,
                            "column span": 1,
                            "bounding box": [2, 2, 3, 4],
                            "content": "All",
                        },
                    ],
                },
            ],
        },
    ]

    bundle = _build_structured_table_bundles(elements)[0]
    evidence_units = bundle["evidence_units"]

    assert len(evidence_units) == 1
    row_unit = evidence_units[0]
    assert row_unit["evidence_unit_type"] == "table_row"
    assert row_unit["evidence_unit_id"] == "id:100::table_row::source:100::r1"
    assert row_unit["row_idx"] == 1
    assert row_unit["bbox"] == [1, 2, 3, 4]
    assert row_unit["content"] == "Method | All"
    assert row_unit["cell_evidence_units"][0]["evidence_unit_type"] == "table_cell"
    assert row_unit["cell_evidence_units"][0]["evidence_unit_id"] == "id:100::table_cell::source:100::r1::c1"
    assert row_unit["cell_evidence_units"][0]["col_idx"] == 1
    assert row_unit["cell_evidence_units"][0]["col_id"] == "Method"
    assert row_unit["cell_evidence_units"][0]["column_header"] == "Method"
    assert row_unit["cell_evidence_units"][0]["header_path"] == "Method"
    assert row_unit["cell_evidence_units"][0]["bbox"] == [1, 2, 2, 4]
    assert row_unit["cell_evidence_units"][1]["content"] == "All"
    assert row_unit["cell_evidence_units"][1]["header_path"] == "All"
    assert row_unit["cell_evidence_units"][1]["col_span"] == 1


def test_odl_table_bbox_is_not_reused_as_a_target_row_crop():
    elements = [{
        "type": "table",
        "id": 100,
        "page number": 1,
        "bounding box": [0, 0, 600, 800],
        "rows": [{
            "type": "table row",
            "row number": 1,
            "cells": [
                {"type": "table cell", "row number": 1, "column number": 1, "content": "Method"},
                {"type": "table cell", "row number": 1, "column number": 2, "content": "Score"},
            ],
        }],
    }]

    bundle = _build_structured_table_bundles(elements, page_sizes={1: [600, 800]})[0]
    row = bundle["evidence_units"][0]

    assert bundle["visual_bbox"] == [0.0, 0.0, 600.0, 800.0]
    assert bundle["row_cell_geometry_available"] is False
    assert row["bbox"] == [0, 0, 600, 800]
    assert row["visual_crop_eligible"] is False
    assert all(cell["visual_crop_eligible"] is False for cell in row["cell_evidence_units"])


def test_odl_complete_explicit_cell_geometry_can_form_a_row_crop():
    elements = [{
        "type": "table",
        "id": 101,
        "page number": 1,
        "bounding box": [0, 0, 600, 800],
        "rows": [{
            "type": "table row",
            "row number": 1,
            "cells": [
                {"type": "table cell", "row number": 1, "column number": 1, "bounding box": [0, 700, 300, 800], "content": "Method"},
                {"type": "table cell", "row number": 1, "column number": 2, "bounding box": [300, 700, 600, 800], "content": "Score"},
            ],
        }],
    }]

    bundle = _build_structured_table_bundles(elements, page_sizes={1: [600, 800]})[0]
    row = bundle["evidence_units"][0]

    assert bundle["row_cell_geometry_available"] is True
    assert row["bbox"] == [0, 700, 600, 800]
    assert row["visual_crop_eligible"] is True
    assert all(cell["visual_crop_eligible"] is True for cell in row["cell_evidence_units"])


def test_odl_structured_table_bundles_expand_spans_into_header_paths():
    elements = [
        {
            "type": "caption",
            "id": 10,
            "page number": 1,
            "linked content id": 100,
            "content": "Table 8: Results on ImageNet-LT.",
        },
        {
            "type": "table",
            "id": 100,
            "page number": 1,
            "bounding box": [1, 2, 8, 9],
            "rows": [
                {
                    "type": "table row",
                    "row number": 1,
                    "cells": [
                        {"type": "table cell", "row number": 1, "column number": 1, "content": ""},
                        {
                            "type": "table cell",
                            "row number": 1,
                            "column number": 2,
                            "column span": 2,
                            "content": "ResNet-10",
                        },
                        {
                            "type": "table cell",
                            "row number": 1,
                            "column number": 4,
                            "column span": 2,
                            "content": "ResNet-50",
                        },
                    ],
                },
                {
                    "type": "table row",
                    "row number": 2,
                    "cells": [
                        {"type": "table cell", "row number": 2, "column number": 1, "content": "Method"},
                        {"type": "table cell", "row number": 2, "column number": 2, "content": "All"},
                        {"type": "table cell", "row number": 2, "column number": 3, "content": "Many"},
                        {"type": "table cell", "row number": 2, "column number": 4, "content": "All"},
                        {"type": "table cell", "row number": 2, "column number": 5, "content": "Many"},
                    ],
                },
                {
                    "type": "table row",
                    "row number": 3,
                    "cells": [
                        {"type": "table cell", "row number": 3, "column number": 1, "content": "DiffuLT"},
                        {"type": "table cell", "row number": 3, "column number": 2, "content": "56.4"},
                        {"type": "table cell", "row number": 3, "column number": 3, "content": "63.3"},
                        {"type": "table cell", "row number": 3, "column number": 4, "content": "55.6"},
                        {"type": "table cell", "row number": 3, "column number": 5, "content": "39.4"},
                    ],
                },
            ],
        },
    ]

    bundle = _build_structured_table_bundles(elements)[0]

    assert bundle["table_header"] == "Method | ResNet-10 All | ResNet-10 Many | ResNet-50 All | ResNet-50 Many"
    data_row = bundle["evidence_units"][2]
    assert data_row["row_id"] == "DiffuLT"
    cells = data_row["cell_evidence_units"]
    assert cells[1]["header_path"] == "ResNet-10 All"
    assert cells[2]["header_path"] == "ResNet-10 Many"
    assert cells[3]["header_path"] == "ResNet-50 All"
    assert "ResNet-50 | ResNet-50" in bundle["table_body_markdown"]


def test_attach_structured_table_bundles_to_pages_injects_bundle_text():
    elements = [
        {
            "type": "caption",
            "id": 10,
            "page number": 1,
            "linked content id": 100,
            "content": "Table 7: Main results",
        },
        {
            **_make_table(
                [
                    ["Method", "All"],
                    ["Ours", "55.5"],
                ]
            ),
            "id": 100,
            "page number": 1,
            "number of rows": 2,
            "number of columns": 2,
        },
    ]
    bundles = _build_structured_table_bundles(elements)
    pages = [
        {
            "page": 1,
            "text": "引言段落\n\n| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
            "content": "引言段落\n\n| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
            "source": "odl",
        }
    ]

    updated_pages, full_text = _attach_structured_bundles_to_pages(pages, bundles)

    assert updated_pages[0]["table_bundles"][0]["table_id"] == "Table 7"
    assert "[Structured Table Bundle]" in updated_pages[0]["text"]
    assert "Table 7: Main results" in updated_pages[0]["text"]
    assert "[Structured Table Bundle]" in full_text


def test_pdf_native_table_info_builds_structured_bundle_with_typed_rows():
    bundle = build_structured_table_bundle(
        {
            "markdown": "| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
            "data": [["Method", "All"], ["Ours", "55.5"]],
            "bbox": (10, 20, 200, 120),
            "caption": "Table 7: Main results",
            "page": 3,
            "source": "pymupdf_table",
        },
        index_in_doc=2,
    )

    assert bundle["source"] == "pymupdf_table"
    assert bundle["table_id"] == "Table 7"
    assert bundle["table_caption"] == "Table 7: Main results"
    assert bundle["table_header"] == "Method | All"
    assert bundle["pages"] == [3]
    assert bundle["bounding_box"] == [10.0, 20.0, 200.0, 120.0]
    assert bundle["evidence_units"][0]["is_header_row"] is True
    assert bundle["evidence_units"][1]["content"] == "Ours | 55.5"
    assert bundle["evidence_units"][1]["cell_evidence_units"][1]["content"] == "55.5"
    assert "[Structured Table Bundle]" in bundle["bundle_text"]


def test_pdf_native_table_bundle_flattens_multilevel_headers_into_cell_schema():
    bundle = build_structured_table_bundle(
        {
            "markdown": "",
            "data": [
                ["", "ResNet-10", "", "ResNet-50", ""],
                ["Method", "All", "Many", "All", "Many"],
                ["DiffuLT", "56.4", "63.3", "55.6", "39.4"],
            ],
            "bbox": (10, 20, 300, 160),
            "caption": "Table 8: Results on ImageNet-LT.",
            "page": 9,
            "source": "pymupdf_table",
        },
        index_in_doc=1,
    )

    assert bundle["table_header"] == "Method | ResNet-10 All | ResNet-10 Many | ResNet-50 All | ResNet-50 Many"
    data_row = bundle["evidence_units"][2]
    assert data_row["row_id"] == "DiffuLT"
    cells = data_row["cell_evidence_units"]
    assert cells[0]["header_path"] == "Method"
    assert cells[1]["header_path"] == "ResNet-10 All"
    assert cells[3]["header_path"] == "ResNet-50 All"
    assert cells[3]["col_id"] == "ResNet-50 All"
    assert cells[3]["row_number"] == 3


def test_pdf_native_table_bundle_uses_cell_level_bboxes_when_available():
    bundle = build_structured_table_bundle(
        {
            "markdown": "",
            "data": [["Method", "All"], ["Ours", "55.5"]],
            "bbox": (10, 20, 210, 120),
            "cell_bboxes": [
                [[10, 20, 110, 50], [110, 20, 210, 50]],
                [[10, 50, 110, 90], [110, 50, 210, 90]],
            ],
            "caption": "Table 7: Main results",
            "page": 3,
            "source": "pymupdf_table",
        },
        index_in_doc=1,
    )

    row = bundle["evidence_units"][1]
    assert row["bbox"] == [10.0, 50.0, 210.0, 90.0]
    assert row["bounding_box"] == [10.0, 50.0, 210.0, 90.0]
    assert row["cell_evidence_units"][0]["bbox"] == [10.0, 50.0, 110.0, 90.0]
    assert row["cell_evidence_units"][1]["bbox"] == [110.0, 50.0, 210.0, 90.0]
    assert row["cell_evidence_units"][1]["header_path"] == "All"


def test_pdf_native_structured_bundles_merge_uncaptioned_next_page_continuation():
    first = build_structured_table_bundle(
        {
            "markdown": "",
            "data": [["Method", "All"], ["Baseline", "52.1"]],
            "bbox": (10, 80, 220, 730),
            "caption": "Table 7: Main results",
            "page": 1,
            "source": "pymupdf_table",
        },
        index_in_doc=1,
    )
    second = build_structured_table_bundle(
        {
            "markdown": "",
            "data": [["Method", "All"], ["Ours", "55.5"]],
            "bbox": (12, 40, 222, 160),
            "caption": "",
            "page": 2,
            "source": "pymupdf_table",
        },
        index_in_doc=1,
    )

    merged = merge_pdf_native_structured_table_bundles([first, second])

    assert len(merged) == 1
    bundle = merged[0]
    assert bundle["table_id"] == "Table 7"
    assert bundle["table_caption"] == "Table 7: Main results"
    assert bundle["pages"] == [1, 2]
    assert bundle["page_start"] == 1
    assert bundle["page_end"] == 2
    assert "Baseline" in bundle["table_body_markdown"]
    assert "Ours" in bundle["table_body_markdown"]
    assert bundle["table_body_markdown"].count("| Method | All |") == 1
    rows = bundle["evidence_units"]
    assert [row["row_id"] for row in rows] == ["Method", "Baseline", "Ours"]
    assert rows[-1]["table_bundle_id"] == bundle["bundle_id"]
    assert rows[-1]["cell_evidence_units"][1]["header_path"] == "All"


def test_extract_table_caption_candidates_from_text_dict_uses_line_bbox():
    text_dict = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {"text": "Table 12: ODinW benchmark results", "bbox": [40, 310, 360, 326]},
                        ]
                    },
                    {
                        "spans": [
                            {"text": "ordinary paragraph", "bbox": [40, 340, 260, 356]},
                        ]
                    },
                ]
            }
        ]
    }

    candidates = extract_table_caption_candidates_from_text_dict(text_dict, page_num=4)

    assert len(candidates) == 1
    assert candidates[0]["table_id"] == "Table 12"
    assert candidates[0]["caption"] == "Table 12: ODinW benchmark results"
    assert candidates[0]["bbox"] == [40.0, 310.0, 360.0, 326.0]


def test_pdf_native_nearest_caption_binding_handles_below_table_caption():
    bundle = build_structured_table_bundle(
        {
            "markdown": "",
            "data": [["Dataset", "AP50"], ["LVIS minival", "52.4"]],
            "bbox": (40, 120, 360, 300),
            "caption": "",
            "page": 4,
            "source": "pymupdf_table",
        },
        index_in_doc=1,
    )
    candidates = [
        {
            "caption": "Table 12: ODinW benchmark results",
            "table_id": "Table 12",
            "page": 4,
            "bbox": [44, 310, 355, 328],
        }
    ]

    bound = bind_nearest_table_captions([bundle], candidates)

    assert len(bound) == 1
    table = bound[0]
    assert table["table_id"] == "Table 12"
    assert table["table_caption"] == "Table 12: ODinW benchmark results"
    assert "Table 12: ODinW benchmark results" in table["bundle_text"]
    assert table["evidence_units"][1]["table_id"] == "Table 12"
    assert table["evidence_units"][1]["table_caption"] == "Table 12: ODinW benchmark results"
    assert table["evidence_units"][1]["cell_evidence_units"][1]["table_id"] == "Table 12"


def test_build_vector_index_appends_structured_table_bundle_chunks(tmp_path, monkeypatch):
    doc_id = "structured-bundle-doc"
    monkeypatch.setattr(
        embedding_service,
        "resolve_model_id",
        lambda embedding_model_id: (embedding_model_id, {}),
    )
    monkeypatch.setattr(
        embedding_service,
        "structure_aware_split_with_context",
        lambda text, chunk_size, chunk_overlap: [("正文段落", "引言")],
    )
    monkeypatch.setattr(
        embedding_service,
        "get_embedding_function",
        lambda *args, **kwargs: _fake_embed_fn,
    )
    monkeypatch.setattr(
        embedding_service,
        "_build_semantic_group_index_async",
        lambda **kwargs: None,
    )

    embedding_service.build_vector_index(
        doc_id=doc_id,
        text="正文段落",
        vector_store_dir=str(tmp_path),
        embedding_model_id="local-minilm",
        pages=[{"page": 1, "text": "正文段落"}],
        structured_table_bundles=[
            {
                "bundle_id": "id:42",
                "table_id": "Table 7",
                "table_caption": "Table 7: Main results",
                "table_header": "Method | All",
                "table_body_markdown": "| Method | All |\n| --- | --- |\n| Ours | 55.5 |",
                "html_table": "<table><tr><th>Method</th><th>All</th></tr></table>",
                "page_start": 1,
                "page_end": 2,
                "pages": [1, 2],
                "source_ids": [42],
                "source": "odl",
                "evidence_units": _make_evidence_units(
                    bundle_key="id:42",
                    table_id="Table 7",
                    table_caption="Table 7: Main results",
                    source_id=42,
                    page=1,
                ),
            }
        ],
    )

    with open(tmp_path / f"{doc_id}.pkl", "rb") as f:
        data = pickle.load(f)

    table_indices = [
        idx for idx, chunk_type in enumerate(data["chunk_types"])
        if chunk_type == "table" and data["chunk_metadata"][idx].get("structured_table_bundle")
    ]
    assert table_indices
    table_idx = table_indices[-1]
    assert data["chunk_metadata"][table_idx]["table_id"] == "Table 7"
    assert data["chunk_metadata"][table_idx]["page_range"] == [1, 2]
    assert data["chunk_metadata"][table_idx]["html_table"].startswith("<table>")
    assert data["chunk_metadata"][table_idx]["evidence_units"][0]["evidence_unit_type"] == "table_row"
    assert data["chunk_metadata"][table_idx]["evidence_units"][0]["cell_evidence_units"][0]["col_idx"] == 1
    assert "[Structured Table Bundle]" in data["chunks"][table_idx]
    assert "Table 7: Main results" in data["chunks"][table_idx]
    assert any(
        metadata.get("table_row_slice_kind") == "exact"
        and metadata.get("numeric_table_exact_context_row_text")
        for metadata in data["chunk_metadata"]
    )
    exact_row = next(
        metadata
        for metadata in data["chunk_metadata"]
        if metadata.get("table_row_slice_kind") == "exact"
    )
    assert exact_row["cell_evidence_units"][1]["header_path"] == "All"
    assert exact_row["cell_evidence_units"][1]["col_id"] == "All"


def test_search_document_chunks_hydrates_structured_table_bundle_metadata(tmp_path, monkeypatch):
    doc_id = "structured-bundle-search"
    index = faiss.IndexFlatIP(EMBED_DIM)
    vector = np.zeros((1, EMBED_DIM), dtype="float32")
    vector[0, 0] = 1.0
    faiss.normalize_L2(vector)
    index.add(vector)
    faiss.write_index(index, str(tmp_path / f"{doc_id}.index"))

    with open(tmp_path / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": [
                    "[Structured Table Bundle]\n\nTable 7: Main results\n\n[Body]\n| Method | All |\n| --- | --- |\n| Ours | 55.5 |"
                ],
                "embedding_model": "local-minilm",
                "chunk_headings": ["Table 7: Main results"],
                "chunk_pages": [2],
                "chunk_types": ["table"],
                "chunk_metadata": [
                    {
                        "structured_table_bundle": True,
                        "table_bundle_id": "id:42",
                        "table_id": "Table 7",
                        "table_caption": "Table 7: Main results",
                        "table_header": "Method | All",
                        "page_range": [2, 3],
                        "html_table": "<table><tr><th>Method</th><th>All</th></tr></table>",
                        "evidence_units": _make_evidence_units(
                            bundle_key="id:42",
                            table_id="Table 7",
                            table_caption="Table 7: Main results",
                            source_id=42,
                            page=2,
                        ),
                    }
                ],
                "parent_chunks": [],
                "child_to_parent": {},
            },
            f,
        )

    monkeypatch.setattr(
        embedding_service,
        "get_embedding_function",
        lambda *args, **kwargs: (lambda texts: vector.copy()),
    )

    with patch("services.embedding_service._query_vector_cache") as mock_cache, \
         patch("services.embedding_service._merge_with_group_search", side_effect=lambda **kwargs: kwargs["chunk_results"]), \
         patch("services.embedding_service._augment_with_table_chunks", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.embedding_service._unified_post_clean", side_effect=lambda results, *_args, **_kwargs: results), \
         patch("services.chunk_expander.expand_context_chunks", side_effect=lambda results, *_args, **_kwargs: results):
        mock_cache.get.return_value = None
        results, _timings = embedding_service.search_document_chunks(
            doc_id=doc_id,
            query="普通查询",
            vector_store_dir=str(tmp_path),
            pages=[{"page": 2, "text": "Table 7 page text"}],
            top_k=1,
            use_hybrid=False,
            use_rerank=False,
        )

    assert results
    top = results[0]
    assert top["table_id"] == "Table 7"
    assert top["table_caption"] == "Table 7: Main results"
    assert top["page_range"] == [2, 3]
    assert top["structured_table_bundle"] is True
    assert top["evidence_units"][0]["evidence_unit_type"] == "table_row"
    assert top["evidence_units"][0]["cell_evidence_units"][1]["content"] == "All"


def test_expand_numeric_table_evidence_units_uses_typed_evidence_units(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "_analyze_evidence_need",
        lambda _query: ["numeric_table"],
    )
    monkeypatch.setattr(
        embedding_service._query_rewriter_singleton,
        "extract_numeric_table_hints",
        lambda _query: {
            "comparison": False,
            "methods": ["Ours"],
            "columns": ["All"],
            "table_labels": ["Table 7"],
            "datasets": [],
            "backbones": [],
        },
    )
    monkeypatch.setattr(
        embedding_service,
        "_build_query_focused_table_row",
        lambda unit, _hints: {
            "text": unit.get("row_text", ""),
            "matched_backbone": "",
            "resolved_columns": ["All"],
            "column_coverage": 1,
        },
    )
    monkeypatch.setattr(
        embedding_service,
        "_compute_lexical_evidence_score",
        lambda _query, _text: 0.95,
    )
    monkeypatch.setattr(
        embedding_service,
        "_numeric_table_sort_bonus",
        lambda _unit, _query, _hints: 0.0,
    )

    results = [
        {
            "chunk": "[Structured Table Bundle]\n\nTable 7: Main results",
            "raw_chunk_text": "[Structured Table Bundle]\n\nTable 7: Main results",
            "page": 1,
            "chunk_type": "table",
            "similarity": 0.62,
            "table_id": "Table 7",
            "table_caption": "Table 7: Main results",
            "table_header": "Method | All",
            "evidence_units": [
                {
                    "evidence_unit_id": "id:42::table_row::source:42::r1",
                    "evidence_unit_type": "table_row",
                    "table_id": "Table 7",
                    "table_caption": "Table 7: Main results",
                    "table_header": "Method | All",
                    "page": 1,
                    "row_number": 1,
                    "row_id": "Method",
                    "row_text": "Method | All",
                    "row_numbers": "All",
                    "is_header_row": True,
                    "cell_evidence_units": [],
                },
                {
                    "evidence_unit_id": "id:42::table_row::source:42::r2",
                    "evidence_unit_type": "table_row",
                    "table_id": "Table 7",
                    "table_caption": "Table 7: Main results",
                    "table_header": "Method | All",
                    "page": 1,
                    "row_number": 2,
                    "row_id": "Ours",
                    "row_text": "Ours | 55.5",
                    "row_numbers": "55.5",
                    "bounding_box": [1, 2, 3, 4],
                    "cell_evidence_unit_ids": [
                        "id:42::table_cell::source:42::r2::c1",
                        "id:42::table_cell::source:42::r2::c2",
                    ],
                    "cell_evidence_units": [
                        {
                            "evidence_unit_id": "id:42::table_cell::source:42::r2::c1",
                            "content": "Ours",
                        },
                        {
                            "evidence_unit_id": "id:42::table_cell::source:42::r2::c2",
                            "content": "55.5",
                        },
                    ],
                },
            ],
        }
    ]

    expanded = embedding_service._expand_numeric_table_evidence_units(
        results,
        "What is the All score of Ours in Table 7?",
    )

    table_rows = [item for item in expanded if item.get("table_row_evidence")]
    assert table_rows
    top = table_rows[0]
    assert top["chunk"] == "Ours | 55.5"
    assert top["evidence_unit_id"] == "id:42::table_row::source:42::r2"
    assert top["cell_evidence_unit_ids"] == [
        "id:42::table_cell::source:42::r2::c1",
        "id:42::table_cell::source:42::r2::c2",
    ]
    assert top["table_row_bbox"] == [1, 2, 3, 4]
