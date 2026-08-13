from services.document_parse_artifact import (
    DOCUMENT_PARSE_ARTIFACT_VERSION,
    artifact_reference,
    build_document_parse_artifact,
    derive_table_geometry_capabilities,
    persist_document_parse_artifact,
)


def test_parse_artifact_is_versioned_and_atomically_persisted(tmp_path):
    artifact = build_document_parse_artifact(
        doc_id="doc-1",
        provider="mineru",
        provider_version="mineru-rag-v1",
        pages=[{"page": 1, "content": "text"}],
        tables=[{"bundle_id": "table-1"}],
        capabilities={"per_page_text": True, "structured_tables": True},
        raw_ref="mineru_results/doc-1.json",
    )

    path = persist_document_parse_artifact(tmp_path, artifact)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    assert artifact_reference(tmp_path, path).startswith("parse_artifacts/doc-1/mineru/")
    # 断言随常量走：这条测试守的是「产物带版本号」，不是钉死某个具体版本值。
    assert artifact["schema_version"] == DOCUMENT_PARSE_ARTIFACT_VERSION
    assert artifact["capabilities"]["structured_tables"] is True


def test_table_geometry_capabilities_keep_overview_and_exact_crops_separate():
    capabilities = derive_table_geometry_capabilities([
        {
            "visual_bbox": [10, 20, 100, 200],
            "evidence_units": [{
                "visual_crop_eligible": False,
                "visual_bbox": [10, 20, 100, 200],
                "cell_evidence_units": [{
                    "visual_crop_eligible": False,
                    "visual_bbox": [10, 20, 40, 50],
                }],
            }],
        },
        {
            "visual_bbox": [20, 30, 200, 300],
            "evidence_units": [{
                "visual_crop_eligible": True,
                "visual_bbox": [20, 80, 200, 120],
                "cell_evidence_units": [{
                    "visual_crop_eligible": True,
                    "visual_bbox": [20, 80, 80, 120],
                }],
            }],
        },
    ])

    assert capabilities == {
        "table_geometry": True,
        "table_overview_geometry": True,
        "table_row_geometry": True,
        "table_cell_geometry": True,
    }
