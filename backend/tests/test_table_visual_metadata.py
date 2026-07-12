import copy

from routes.document_routes import _clean_structured_table_bundle
from services import embedding_service
from services.table_visual_metadata import (
    build_table_visual_metadata,
    derive_table_instance_id,
    derive_table_source_hash,
)


def _bundle(*, page: int, bbox: list[float]) -> dict:
    return {
        # 故意固定 source 标识，确保本测试验证的是页码/坐标，而非 parser ID。
        "bundle_id": "mineru:stable-table-source",
        "table_id": "Table 3",
        "table_caption": "Table 3: Ablation results",
        "table_header": "Method | mAP",
        "table_body_markdown": "| Method | mAP |\n| --- | --- |\n| Ours | 51.2 |",
        "page_start": page,
        "page_end": page,
        "pages": [page],
        "bounding_box": bbox,
        "bounding_boxes": [bbox],
        "source": "mineru",
        "evidence_units": [
            {
                "evidence_unit_type": "table_row",
                "row_number": 2,
                "row_id": "Ours",
                "row_text": "Ours | 51.2",
                "cell_evidence_units": [
                    {"header_path": "Method", "content": "Ours"},
                    {"header_path": "mAP", "content": "51.2"},
                ],
            }
        ],
    }


def test_duplicate_table_labels_on_different_pages_or_boxes_have_distinct_ids():
    first = _bundle(page=3, bbox=[20, 40, 300, 220])
    same_label_other_page = _bundle(page=8, bbox=[20, 40, 300, 220])
    same_label_other_box = _bundle(page=3, bbox=[24, 44, 304, 224])

    identifiers = {
        derive_table_instance_id(first),
        derive_table_instance_id(same_label_other_page),
        derive_table_instance_id(same_label_other_box),
    }
    hashes = {
        derive_table_source_hash(first),
        derive_table_source_hash(same_label_other_page),
        derive_table_source_hash(same_label_other_box),
    }

    assert len(identifiers) == 3
    assert len(hashes) == 3


def test_equivalent_table_representation_keeps_a_stable_identity():
    canonical = _bundle(page=3, bbox=[20, 40, 300, 220])
    formatted = copy.deepcopy(canonical)
    formatted["table_caption"] = "  Table 3:   Ablation\nresults  "
    formatted["table_body_markdown"] = "| Method | mAP |  \n| --- | --- |\n| Ours | 51.2 |"
    formatted["bounding_box"] = [20.0004, 40.0004, 300.0004, 220.0004]
    formatted["bounding_boxes"] = [[20.0004, 40.0004, 300.0004, 220.0004]]
    formatted["source_ids"] = ["b", "a"]
    canonical["source_ids"] = ["a", "b"]

    assert derive_table_instance_id(formatted) == derive_table_instance_id(canonical)
    assert derive_table_source_hash(formatted) == derive_table_source_hash(canonical)


def test_metadata_helpers_and_document_cleaning_do_not_mutate_parser_bundle():
    raw = _bundle(page=3, bbox=[20, 40, 300, 220])
    original = copy.deepcopy(raw)

    metadata = build_table_visual_metadata(raw)
    cleaned = _clean_structured_table_bundle(raw)

    assert raw == original
    assert "table_instance_id" not in raw
    assert "table_source_hash" not in raw
    assert cleaned["table_instance_id"] == metadata["table_instance_id"]
    assert cleaned["table_source_hash"] == metadata["table_source_hash"]


def test_embedding_sanitizer_and_row_chunks_propagate_visual_metadata_without_mutation():
    raw = _bundle(page=3, bbox=[20, 40, 300, 220])
    original = copy.deepcopy(raw)

    sanitized = embedding_service._sanitize_structured_table_bundle(raw)
    chunks: list[str] = []
    headings: list[str] = []
    pages: list[int] = []
    types: list[str] = []
    metadata: list[dict] = []
    embedding_service._append_structured_table_bundle_chunks(
        "visual-metadata-test",
        chunks,
        headings,
        pages,
        types,
        metadata,
        [raw],
    )

    assert raw == original
    assert sanitized["table_instance_id"]
    assert sanitized["table_source_hash"]
    table_metadata = [item for item in metadata if item.get("structured_table_bundle")]
    assert table_metadata
    assert all(item["table_instance_id"] == sanitized["table_instance_id"] for item in table_metadata)
    assert all(item["table_source_hash"] == sanitized["table_source_hash"] for item in table_metadata)
    exact_row = next(item for item in metadata if item.get("table_row_slice_kind") == "exact")
    assert exact_row["evidence_units"][0]["table_instance_id"] == sanitized["table_instance_id"]


def test_legacy_page_bbox_and_markdown_fields_keep_identity_across_cleaning_and_indexing():
    raw = _bundle(page=4, bbox=[11, 22, 333, 444])
    raw["table_markdown"] = raw.pop("table_body_markdown")
    raw["page"] = raw.pop("page_start")
    raw.pop("page_end")
    raw.pop("pages")
    raw["bbox"] = raw.pop("bounding_box")
    raw.pop("bounding_boxes")

    cleaned = _clean_structured_table_bundle(raw)
    sanitized = embedding_service._sanitize_structured_table_bundle(raw)

    assert cleaned["table_instance_id"] == sanitized["table_instance_id"]
    assert cleaned["table_source_hash"] == sanitized["table_source_hash"]
