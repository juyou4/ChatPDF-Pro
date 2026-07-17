"""Persistent generation fence for document-scoped AI caches."""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


AI_CACHE_MANIFEST_VERSION = 1
LEGACY_AI_CACHE_GENERATION = "legacy"


class AICacheStateError(RuntimeError):
    """Raised when the cache-generation manifest cannot be trusted."""


def get_ai_cache_manifest_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "ai_cache_manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ai_cache_manifest_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_ai_cache_manifest_dir(data_dir) / f"{doc_id}.json"


def load_ai_cache_generation(data_dir: Path | str, doc_id: str) -> str:
    path = get_ai_cache_manifest_path(data_dir, doc_id)
    if not path.exists():
        return LEGACY_AI_CACHE_GENERATION

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise AICacheStateError(f"AI 缓存代际文件不可读: {path}") from exc

    if not isinstance(payload, dict):
        raise AICacheStateError(f"AI 缓存代际文件格式无效: {path}")
    if payload.get("version") != AI_CACHE_MANIFEST_VERSION:
        raise AICacheStateError(f"AI 缓存代际版本不受支持: {path}")
    if str(payload.get("doc_id") or "") != str(doc_id):
        raise AICacheStateError(f"AI 缓存代际文档身份不匹配: {path}")

    generation = str(payload.get("generation") or "").strip()
    if not generation:
        raise AICacheStateError(f"AI 缓存代际为空: {path}")
    return generation


def rotate_ai_cache_generation(data_dir: Path | str, doc_id: str) -> str:
    generation = uuid.uuid4().hex
    path = get_ai_cache_manifest_path(data_dir, doc_id)
    payload: dict[str, Any] = {
        "version": AI_CACHE_MANIFEST_VERSION,
        "doc_id": str(doc_id),
        "generation": generation,
        "updated_at": time.time(),
    }
    _atomic_write_manifest(path, payload)
    return generation


def _atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
