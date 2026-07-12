import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from services.semantic_group_store import (
    active_manifest_path,
    deactivate_generation,
    publish_generation,
    semantic_group_paths,
    validate_semantic_group_artifacts,
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


def test_validation_requires_readable_and_consistent_semantic_artifacts(tmp_path: Path):
    doc_id = "doc-validated"
    stage = tmp_path / "stage"
    stage.mkdir()
    paths = {
        "json": stage / f"{doc_id}.json",
        "index": stage / f"{doc_id}_groups.index",
        "pkl": stage / f"{doc_id}_groups.pkl",
    }
    paths["json"].write_text(
        json.dumps({"doc_id": doc_id, "groups": [{"group_id": "g-1"}]}),
        encoding="utf-8",
    )
    with open(paths["pkl"], "wb") as handle:
        pickle.dump({"digest_texts": ["digest"], "group_ids": ["g-1"]}, handle)
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[0.1, 0.2]], dtype="float32"))
    faiss.write_index(index, str(paths["index"]))

    assert validate_semantic_group_artifacts(paths, doc_id)["valid"] is True

    paths["index"].write_bytes(b"not-a-faiss-index")
    result = validate_semantic_group_artifacts(paths, doc_id)
    assert result["valid"] is False
    assert any(error.startswith("groups_faiss_unreadable") for error in result["errors"])
