from services.document_parse_artifact import (
    artifact_reference,
    build_document_parse_artifact,
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
    assert artifact["schema_version"] == 1
    assert artifact["capabilities"]["structured_tables"] is True
