from services.retrieval_tools import DocContext, _ensure_table_result_selected, execute_tool


def test_table_row_shard_is_promoted_when_selected_only_has_whole_table():
    query = "Table 12 中 LVIS minival 的 AP50 数值是多少？"
    selected = [
        {
            "chunk_id": "paragraph-1",
            "chunk": "Grounding DINO compares object detection results across several datasets.",
            "chunk_type": "text",
            "similarity": 0.88,
        },
        {
            "chunk_id": "table-12",
            "chunk": "[Structured Table Bundle]\nTable 12: ODinW benchmark results.\nDataset | AP | AP50\n...",
            "chunk_type": "table",
            "structured_table_bundle": True,
            "table_id": "Table 12",
            "table_caption": "ODinW benchmark results",
            "similarity": 0.81,
        },
    ]
    candidates = [
        *selected,
        {
            "chunk_id": "table-12-rows-11-20",
            "chunk": "[Structured Table Row Shard]\nTable 12: ODinW benchmark results.\nDataset | AP | AP50\nLVIS minival | 33.5 | 52.4",
            "chunk_type": "table_row",
            "table_row_shard": True,
            "structured_table_bundle": True,
            "parent_table_bundle_id": "table-12",
            "table_id": "Table 12",
            "table_caption": "ODinW benchmark results",
            "row_text": "LVIS minival | 33.5 | 52.4",
            "similarity": 0.62,
        },
    ]

    final = _ensure_table_result_selected(query, selected, candidates, limit=3)

    assert [item["chunk_id"] for item in final] == [
        "paragraph-1",
        "table-12",
        "table-12-rows-11-20",
    ]


def test_table_row_shard_replaces_last_slot_when_limit_is_full():
    query = "Table 12 中 LVIS minival 的 AP50 数值是多少？"
    selected = [
        {
            "chunk_id": "paragraph-1",
            "chunk": "The paper studies open-set object detection.",
            "chunk_type": "text",
            "similarity": 0.9,
        },
        {
            "chunk_id": "table-12",
            "chunk": "[Structured Table Bundle]\nTable 12: ODinW benchmark results.",
            "chunk_type": "table",
            "structured_table_bundle": True,
            "table_id": "Table 12",
            "similarity": 0.82,
        },
        {
            "chunk_id": "paragraph-2",
            "chunk": "Additional discussion without the requested metric.",
            "chunk_type": "text",
            "similarity": 0.78,
        },
    ]
    row_shard = {
        "chunk_id": "table-12-rows-11-20",
        "chunk": "[Structured Table Row Shard]\nLVIS minival | 33.5 | 52.4",
        "chunk_type": "table_row",
        "table_row_shard": True,
        "table_id": "Table 12",
        "row_text": "LVIS minival | 33.5 | 52.4",
        "similarity": 0.6,
    }

    final = _ensure_table_result_selected(query, selected, [*selected, row_shard], limit=3)

    assert [item["chunk_id"] for item in final] == [
        "paragraph-1",
        "table-12",
        "table-12-rows-11-20",
    ]


def test_table_selection_keeps_existing_row_evidence_unchanged():
    query = "Table 12 中 LVIS minival 的 AP50 数值是多少？"
    selected = [
        {
            "chunk_id": "table-12-rows-11-20",
            "chunk": "[Structured Table Row Shard]\nLVIS minival | 33.5 | 52.4",
            "chunk_type": "table_row",
            "table_row_shard": True,
            "row_text": "LVIS minival | 33.5 | 52.4",
            "similarity": 0.7,
        },
        {
            "chunk_id": "paragraph-1",
            "chunk": "Background text.",
            "chunk_type": "text",
            "similarity": 0.6,
        },
    ]
    candidates = [
        *selected,
        {
            "chunk_id": "table-12",
            "chunk": "[Structured Table Bundle]\nTable 12: ODinW benchmark results.",
            "chunk_type": "table",
            "structured_table_bundle": True,
            "similarity": 0.9,
        },
    ]

    final = _ensure_table_result_selected(query, selected, candidates, limit=2)

    assert final == selected


def test_regex_search_prefers_structured_table_metadata_rows():
    ctx = DocContext(
        doc_id="doc-1",
        full_text="The appendix mentions LVIS minival but omits the AP50 value.",
        chunks=[
            "[Structured Table Row Shard]\nLVIS minival | AP: 33.5 | AP50: 52.4",
            "Plain paragraph with LVIS minival only.",
        ],
        pages=[{"page": 1, "content": "The appendix mentions LVIS minival."}],
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

    result = execute_tool(
        "regex_search",
        {"pattern": r"AP50[:：]?\s*52\.4", "limit": 3},
        ctx,
    )

    assert result["result_count"] >= 1
    assert "结构化表格 1 个" in result["summary"]
    assert "Dataset: LVIS minival; AP: 33.5; AP50: 52.4" in result["results"][0]
    assert result["chunk_meta"][0]["numeric_regex_locator"] is True
    assert result["chunk_meta"][0]["table_id"] == "Table 12"
