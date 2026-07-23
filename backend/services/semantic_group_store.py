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
import pickle
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


def _normalize_identity_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_semantic_identity(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    provider = str(data.get("embedding_provider") or "").strip()
    return {
        "parse_generation": str(data.get("parse_generation") or data.get("transaction_id") or "").strip(),
        "document_source_hash": str(data.get("document_source_hash") or data.get("source_hash") or "").strip(),
        "vector_build_id": str(data.get("vector_build_id") or "").strip(),
        "embedding_identity_version": _normalize_identity_int(data.get("embedding_identity_version")),
        "embedding_model": str(data.get("embedding_model") or "").strip(),
        "embedding_provider": provider,
        "embedding_api_host": "" if provider == "local" else str(data.get("embedding_api_host") or "").strip(),
        "vector_dimension": _normalize_identity_int(data.get("vector_dimension")),
    }


def _semantic_identity_complete(identity: dict[str, Any]) -> bool:
    return bool(
        identity.get("parse_generation")
        and identity.get("document_source_hash")
        and identity.get("vector_build_id")
        and int(identity.get("embedding_identity_version") or 0) > 0
        and identity.get("embedding_model")
        and identity.get("embedding_provider")
        and (
            identity.get("embedding_provider") == "local"
            or identity.get("embedding_api_host")
        )
        and int(identity.get("vector_dimension") or 0) > 0
    )


def _semantic_identity_declared(identity: dict[str, Any]) -> bool:
    return bool(
        identity.get("parse_generation")
        or identity.get("document_source_hash")
        or identity.get("vector_build_id")
        or int(identity.get("embedding_identity_version") or 0) > 0
        or identity.get("embedding_model")
        or identity.get("embedding_provider")
        or identity.get("embedding_api_host")
        or int(identity.get("vector_dimension") or 0) > 0
    )


def _semantic_identity_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        str(left.get(key) or "") == str(right.get(key) or "")
        for key in (
            "parse_generation",
            "document_source_hash",
            "vector_build_id",
            "embedding_identity_version",
            "embedding_model",
            "embedding_provider",
            "embedding_api_host",
            "vector_dimension",
        )
    )


def _apply_expected_identity_checks(
    errors: list[str],
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    if not expected:
        return
    if not _semantic_identity_complete(actual):
        errors.append(f"{label}_identity_incomplete")
        return
    if not _semantic_identity_matches(actual, expected):
        errors.append(f"{label}_identity_mismatch")


def validate_semantic_group_artifacts(
    paths: dict[str, Path],
    doc_id: str,
    *,
    expected_identity: dict[str, Any] | None = None,
    expected_vector_dimension: int | None = None,
) -> dict[str, Any]:
    """Verify that a semantic generation is loadable before it becomes active."""
    errors: list[str] = []
    normalized_expected = _normalize_semantic_identity(expected_identity) if expected_identity else None
    try:
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(groups, list) or not groups:
            errors.append("groups_json_empty")
        if isinstance(payload, dict) and payload.get("doc_id") not in (None, "", doc_id):
            errors.append("groups_json_doc_id_mismatch")
        json_identity = _normalize_semantic_identity(payload)
    except Exception as exc:
        groups = []
        payload = {}
        json_identity = {}
        errors.append(f"groups_json_unreadable:{exc}")

    try:
        with open(paths["pkl"], "rb") as handle:
            metadata = pickle.load(handle)
        digest_texts = metadata.get("digest_texts") if isinstance(metadata, dict) else None
        group_ids = metadata.get("group_ids") if isinstance(metadata, dict) else None
        if not isinstance(digest_texts, list) or not isinstance(group_ids, list):
            errors.append("groups_pkl_invalid_shape")
        elif not digest_texts or len(digest_texts) != len(group_ids):
            errors.append("groups_pkl_count_mismatch")
        elif groups and len(groups) != len(group_ids):
            errors.append("groups_json_pkl_count_mismatch")
        pkl_identity = _normalize_semantic_identity(metadata)
    except Exception as exc:
        group_ids = []
        metadata = {}
        pkl_identity = {}
        errors.append(f"groups_pkl_unreadable:{exc}")

    if normalized_expected or _semantic_identity_declared(json_identity) or _semantic_identity_declared(pkl_identity):
        if not _semantic_identity_complete(json_identity):
            errors.append("groups_json_identity_incomplete")
        if not _semantic_identity_complete(pkl_identity):
            errors.append("groups_pkl_identity_incomplete")
        if _semantic_identity_complete(json_identity) and _semantic_identity_complete(pkl_identity):
            if not _semantic_identity_matches(json_identity, pkl_identity):
                errors.append("groups_json_pkl_identity_mismatch")

    _apply_expected_identity_checks(
        errors,
        label="groups_json",
        actual=json_identity,
        expected=normalized_expected,
    )
    _apply_expected_identity_checks(
        errors,
        label="groups_pkl",
        actual=pkl_identity,
        expected=normalized_expected,
    )

    try:
        import faiss

        index = faiss.read_index(str(paths["index"]))
        if int(index.ntotal) != len(group_ids):
            errors.append("groups_faiss_count_mismatch")
        faiss_dimension = int(index.d)
        if normalized_expected and int(normalized_expected.get("vector_dimension") or 0) != faiss_dimension:
            errors.append("groups_faiss_dimension_identity_mismatch")
        if expected_vector_dimension and int(expected_vector_dimension) != faiss_dimension:
            errors.append("groups_faiss_dimension_expected_mismatch")
        if _semantic_identity_declared(pkl_identity) and int(pkl_identity.get("vector_dimension") or 0) != faiss_dimension:
            errors.append("groups_faiss_dimension_pkl_mismatch")
    except Exception as exc:
        faiss_dimension = 0
        errors.append(f"groups_faiss_unreadable:{exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "group_count": len(group_ids),
        "semantic_identity": pkl_identity or json_identity,
        "vector_dimension": faiss_dimension,
        "paths": {key: str(value) for key, value in paths.items()},
    }


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
    semantic_identity: dict[str, Any] | None = None,
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

    normalized_identity = _normalize_semantic_identity(semantic_identity)
    manifest = {
        "schema_version": 2,
        "doc_id": doc_id,
        "generation_id": generation_id,
        "source_hash": source_hash,
        "transaction_id": transaction_id,
        "parse_generation": transaction_id,
        "document_source_hash": source_hash,
        "artifacts": {kind: path.name for kind, path in expected.items()},
    }
    if normalized_identity:
        manifest.update({
            "vector_build_id": normalized_identity.get("vector_build_id") or "",
            "embedding_identity_version": int(normalized_identity.get("embedding_identity_version") or 0),
            "embedding_model": normalized_identity.get("embedding_model") or "",
            "embedding_provider": normalized_identity.get("embedding_provider") or "",
            "embedding_api_host": normalized_identity.get("embedding_api_host") or "",
            "vector_dimension": int(normalized_identity.get("vector_dimension") or 0),
        })
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
