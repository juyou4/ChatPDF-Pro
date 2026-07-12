from services import embedding_service
from services import rag_config


def test_semantic_group_builder_returns_disabled_contract(monkeypatch):
    class DisabledConfig:
        enable_semantic_groups = False

    monkeypatch.setattr(rag_config, "RAGConfig", DisabledConfig)

    result = embedding_service._build_semantic_group_index(
        "doc-disabled",
        ["chunk"],
        [{"page": 1, "content": "chunk"}],
        lambda values: values,
        raise_on_error=True,
    )

    assert result == {"status": "disabled", "group_count": 0, "paths": []}
