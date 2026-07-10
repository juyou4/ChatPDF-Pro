import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.table_source_selector import (
    MIXED_TABLE_RAG_INDEX_SOURCE,
    build_mixed_table_rag_data,
)


def _row(row_id: str, values: list[str]) -> dict:
    cells = [
        {"header_path": "Method", "content": row_id},
        *[
            {"header_path": f"Metric {idx}", "content": value}
            for idx, value in enumerate(values, 1)
        ],
    ]
    return {
        "evidence_unit_type": "table_row",
        "row_id": row_id,
        "row_text": f"Method: {row_id}; " + "; ".join(
            f"Metric {idx}: {value}" for idx, value in enumerate(values, 1)
        ),
        "cell_evidence_units": cells,
        "cell_count": len(cells),
    }


def _bundle(table_id: str, caption: str, rows: list[dict], *, source: str) -> dict:
    return {
        "bundle_id": f"{source}:{table_id}",
        "table_id": table_id,
        "table_caption": caption,
        "table_header": "Method | Metric 1 | Metric 2",
        "page_start": 1,
        "evidence_units": rows,
    }


def _doc(*bundles: dict) -> dict:
    return {
        "data": {
            "full_text": "paper text",
            "pages": [{"page": 1, "content": "paper text"}],
            "structured_table_bundles": list(bundles),
        }
    }


def test_table_selector_aligns_by_table_number_and_fuzzy_caption():
    result = build_mixed_table_rag_data(
        {
            "pdf_native": _doc(
                _bundle(
                    "Table 5",
                    "Table 5: Training efficiency of different models with ResNet-50.",
                    [_row("A", ["10.0"])],
                    source="pdf_native",
                )
            ),
            "mineru_pipeline": _doc(
                _bundle(
                    "Table 5",
                    "Table 5: Training efficieny of diferent models with ResNet-50.",
                    [_row("A", ["10.0"]), _row("B", ["20.0"])],
                    source="mineru_pipeline",
                )
            ),
            "mineru_vlm": _doc(
                _bundle(
                    "Table 5",
                    "Table 5: Training efficiency of different models with ResNet-50.",
                    [_row("A", ["10.0"]), _row("B", ["20.0"]), _row("C", ["30.0"])],
                    source="mineru_vlm",
                )
            ),
        }
    )

    assert result["index_source"] == MIXED_TABLE_RAG_INDEX_SOURCE
    assert len(result["structured_table_bundles"]) == 1
    decision = result["table_selection_decisions"][0]
    assert decision["align_key"].startswith("table-5:")
    assert len(decision["candidates"]) == 3
    assert decision["selected_source"] == "mineru_vlm"

    selected = result["structured_table_bundles"][0]
    assert selected["selected_source"] == "mineru_vlm"
    assert selected["selection_reason"]
    assert selected["evidence_units"][0]["selected_source"] == "mineru_vlm"
    assert selected["evidence_units"][0]["cell_evidence_units"][0]["selected_source"] == "mineru_vlm"


def test_table_selector_splits_same_table_number_when_caption_is_different():
    result = build_mixed_table_rag_data(
        {
            "pdf_native": _doc(
                _bundle(
                    "Table 1",
                    "Table 1: Accuracy on COCO.",
                    [_row("A", ["10.0"])],
                    source="pdf_native",
                )
            ),
            "mineru_pipeline": _doc(
                _bundle(
                    "Table 1",
                    "Table 1: Latency ablation on edge devices.",
                    [_row("A", ["10.0"])],
                    source="mineru_pipeline",
                )
            ),
            "mineru_vlm": _doc(
                _bundle(
                    "Table 1",
                    "Table 1: Accuracy on COCO.",
                    [_row("A", ["10.0"]), _row("B", ["20.0"])],
                    source="mineru_vlm",
                )
            ),
        }
    )

    decisions = result["table_selection_decisions"]
    assert len(decisions) == 2
    assert sorted(len(decision["candidates"]) for decision in decisions) == [1, 2]
    assert {decision["table_ref"] for decision in decisions} == {"table-1"}
