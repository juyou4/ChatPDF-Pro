from pathlib import Path

from services.semantic_group_store import (
    active_manifest_path,
    deactivate_generation,
    publish_generation,
    semantic_group_paths,
)


def _write_staged_generation(root: Path, doc_id: str, *, marker: str) -> Path:
    stage = root / f"stage-{marker}"
    stage.mkdir(parents=True)
    (stage / f"{doc_id}.json").write_text(marker, encoding="utf-8")
    (stage / f"{doc_id}_groups.index").write_bytes(marker.encode("utf-8"))
    (stage / f"{doc_id}_groups.pkl").write_bytes(marker.encode("utf-8"))
    return stage


def test_publish_generation_switches_readers_only_after_manifest_write(tmp_path: Path):
    root = tmp_path / "semantic_groups"
    doc_id = "doc-with-safe-id"
    legacy = semantic_group_paths(root, doc_id)
    for path in legacy.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy")

    stage = _write_staged_generation(root, doc_id, marker="new")
    published = publish_generation(root, doc_id, stage, source_hash="hash")

    active = semantic_group_paths(root, doc_id)
    assert active["json"].read_text(encoding="utf-8") == "new"
    assert active_manifest_path(root, doc_id).exists()
    assert published["generation_id"] in str(active["json"])
    assert all(not path.exists() for path in stage.iterdir())


def test_incomplete_staging_does_not_change_active_generation(tmp_path: Path):
    root = tmp_path / "semantic_groups"
    doc_id = "doc-safe"
    first = _write_staged_generation(root, doc_id, marker="first")
    publish_generation(root, doc_id, first)
    before = semantic_group_paths(root, doc_id)["json"].read_text(encoding="utf-8")

    incomplete = root / "incomplete"
    incomplete.mkdir()
    (incomplete / f"{doc_id}.json").write_text("bad", encoding="utf-8")
    try:
        publish_generation(root, doc_id, incomplete)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete generation should be rejected")

    assert semantic_group_paths(root, doc_id)["json"].read_text(encoding="utf-8") == before


def test_deactivate_removes_pointer_but_keeps_generation_history(tmp_path: Path):
    root = tmp_path / "semantic_groups"
    doc_id = "doc-disabled"
    stage = _write_staged_generation(root, doc_id, marker="active")
    publish_generation(root, doc_id, stage)
    active_path = semantic_group_paths(root, doc_id)["json"]

    result = deactivate_generation(root, doc_id)

    assert result["deactivated"] is True
    assert active_path.exists()
    assert not active_manifest_path(root, doc_id).exists()
