"""Atomic storage helpers for per-document semantic-group generations.

The legacy layout stores three active files directly under ``semantic_groups``.
That makes a multi-file replacement observable half way through.  New writers
first create a complete generation and then atomically replace a small active
manifest.  Readers keep supporting the legacy layout for existing documents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any


_ARTIFACT_NAMES = {
    "json": "{doc_id}.json",
    "index": "{doc_id}_groups.index",
    "pkl": "{doc_id}_groups.pkl",
}


def _doc_key(doc_id: str) -> str:
    """Return a filesystem-safe, stable key without trusting a document id."""
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", str(doc_id or "document")).strip("_")
    readable = (readable or "document")[:48]
    digest = hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def semantic_group_paths(root: Path | str, doc_id: str) -> dict[str, Path]:
    """Resolve active generation paths, falling back to the legacy flat files."""
    root = Path(root)
    manifest_path = root / "active" / f"{_doc_key(doc_id)}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generation = str(manifest.get("generation_id") or "")
        generation_dir = root / "generations" / _doc_key(doc_id) / generation
        if generation and generation_dir.is_dir():
            paths = {kind: generation_dir / name.format(doc_id=doc_id) for kind, name in _ARTIFACT_NAMES.items()}
            if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
                return paths
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {kind: root / name.format(doc_id=doc_id) for kind, name in _ARTIFACT_NAMES.items()}


def active_manifest_path(root: Path | str, doc_id: str) -> Path:
    return Path(root) / "active" / f"{_doc_key(doc_id)}.json"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def publish_generation(
    root: Path | str,
    doc_id: str,
    staged_dir: Path | str,
    *,
    source_hash: str = "",
    transaction_id: str = "",
) -> dict[str, Any]:
    """Publish a fully validated staging directory by changing one manifest."""
    root = Path(root)
    staged_dir = Path(staged_dir)
    expected = {kind: staged_dir / name.format(doc_id=doc_id) for kind, name in _ARTIFACT_NAMES.items()}
    missing = [str(path) for path in expected.values() if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"semantic group generation is incomplete: {missing}")

    generation_id = uuid.uuid4().hex
    generation_dir = root / "generations" / _doc_key(doc_id) / generation_id
    generation_dir.mkdir(parents=True, exist_ok=False)
    for path in expected.values():
        shutil.move(str(path), str(generation_dir / path.name))

    manifest = {
        "schema_version": 1,
        "doc_id": doc_id,
        "generation_id": generation_id,
        "source_hash": source_hash,
        "transaction_id": transaction_id,
        "artifacts": {kind: path.name for kind, path in expected.items()},
    }
    _atomic_write_json(active_manifest_path(root, doc_id), manifest)
    return {
        "generation_id": generation_id,
        "manifest": str(active_manifest_path(root, doc_id)),
        "paths": {kind: str(generation_dir / path.name) for kind, path in expected.items()},
    }


def deactivate_generation(root: Path | str, doc_id: str) -> dict[str, Any]:
    """Make semantic groups unavailable for a document without deleting history."""
    root = Path(root)
    manifest_path = active_manifest_path(root, doc_id)
    existed = manifest_path.exists()
    if existed:
        manifest_path.unlink()
    removed = []
    for path in {kind: root / name.format(doc_id=doc_id) for kind, name in _ARTIFACT_NAMES.items()}.values():
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return {"deactivated": existed or bool(removed), "removed": removed}
