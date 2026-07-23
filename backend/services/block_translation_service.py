"""Block-level translation cache and generation."""
from __future__ import annotations

import asyncio
import os
import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from services.ai_cache_state import LEGACY_AI_CACHE_GENERATION
from services.chat_service import call_ai_api
from services.glossary_service import build_glossary_prompt

logger = logging.getLogger(__name__)

TRANSLATION_CACHE_VERSION = 3
TRANSLATION_PROMPT_VERSION = 5
MAX_BLOCKS_PER_REQUEST = 24
MAX_BLOCK_CHARS = 1800
TRANSLATION_CONCURRENCY = 8
MAX_TRANSLATION_CONCURRENCY = 16


def _bounded_env_int(name: str, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        return default


GLOBAL_TRANSLATION_CONCURRENCY = _bounded_env_int("CHATPDF_GLOBAL_TRANSLATION_CONCURRENCY", 16, 64)
TRANSLATION_TASK_BATCH_SIZE = _bounded_env_int("CHATPDF_TRANSLATION_TASK_BATCH_SIZE", 48, 256)
TABLE_BLOCK_TYPES = {"table"}

_GLOBAL_TRANSLATION_SEMAPHORE = asyncio.Semaphore(GLOBAL_TRANSLATION_CONCURRENCY)

_RE_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_RE_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")
_RE_LATEX_COMMAND = re.compile(r"\\[A-Za-z]{2,}")
_RE_EQUATION_SIGNAL = re.compile(r"(?:[_^]=?|[A-Za-z]\s*=|\\(?:frac|sum|prod|int|sqrt|begin|end)\b)")
_RE_METADATA_DUMP_SIGNAL = re.compile(
    r"(?s)^\s*[\{\[]?['\"]?(?:error|_used_provider|_used_model|_usage_meta|fallback_used|raw_usage|completion_tokens|prompt_tokens)['\"]?\s*[:=]"
)
_PROTECTED_MATH_PATTERNS = (
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\\begin\{(?P<env>[^{}]+)\}.*?\\end\{(?P=env)\}", re.DOTALL),
    re.compile(r"(?<!\\)\$(?!\$)(?:\\.|[^$])*?(?<!\\)\$", re.DOTALL),
)

TranslationCacheWriter = Callable[[dict[str, Any]], Awaitable[None] | None]

_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}


def get_translation_cache_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "block_translations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_translation_cache_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_translation_cache_dir(data_dir) / f"{doc_id}.json"


def load_translation_cache(data_dir: Path | str, doc_id: str) -> dict[str, Any]:
    path = get_translation_cache_path(data_dir, doc_id)
    return _load_translation_cache_path(path, doc_id)


def _load_translation_cache_path(path: Path, doc_id: str) -> dict[str, Any]:
    if not path.exists():
        return _new_translation_cache(doc_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            not isinstance(data, dict)
            or data.get("version") != TRANSLATION_CACHE_VERSION
            or str(data.get("doc_id") or "") != str(doc_id)
        ):
            return _new_translation_cache(doc_id)
        data.setdefault("items", {})
        if not isinstance(data["items"], dict):
            data["items"] = {}
        data.setdefault("outcomes", {})
        if not isinstance(data["outcomes"], dict):
            data["outcomes"] = {}
        return data
    except Exception as exc:
        logger.warning("[BlockTranslation] Failed to load %s: %s", path, exc)
        return _new_translation_cache(doc_id)


def save_translation_cache(data_dir: Path | str, doc_id: str, cache: dict[str, Any]) -> None:
    """Atomically publish an envelope, merging concurrent writes for one document.

    Callers that publish under a document-level parse-generation lock can pass
    this function through ``translate_blocks(cache_writer=...)``.  The service
    still owns the file-level lock so overlapping translation batches cannot
    discard each other's completed blocks.
    """
    path = get_translation_cache_path(data_dir, doc_id)
    lock = _get_translation_cache_lock(path)
    candidate = _normalise_translation_cache(cache, doc_id)
    live_items = cache.get("items") if isinstance(cache.get("items"), dict) else None
    try:
        with lock:
            existing = _load_translation_cache_path(path, doc_id)
            merged = _merge_translation_caches(existing, candidate, doc_id)
            _atomic_write_translation_cache(path, merged)
            # Keep the in-flight request's view coherent after a reload/merge.
            cache.clear()
            cache.update(merged)
            if live_items is not None:
                merged_items = dict(cache.get("items") or {})
                live_items.clear()
                live_items.update(merged_items)
                cache["items"] = live_items
    except Exception:
        logger.exception("[BlockTranslation] Failed to save %s", path)
        raise


def _new_translation_cache(doc_id: str, identity: dict[str, str] | None = None) -> dict[str, Any]:
    identity = identity or {}
    return {
        "version": TRANSLATION_CACHE_VERSION,
        "doc_id": str(doc_id),
        "parser_route": str(identity.get("parser_route") or ""),
        "parse_generation": str(identity.get("parse_generation") or ""),
        "document_source_hash": str(identity.get("document_source_hash") or ""),
        "block_index_hash": str(identity.get("block_index_hash") or ""),
        "ai_cache_generation": str(
            identity.get("ai_cache_generation") or LEGACY_AI_CACHE_GENERATION
        ),
        "items": {},
        "outcomes": {},
    }


def _normalise_translation_cache(cache: dict[str, Any], doc_id: str) -> dict[str, Any]:
    source = cache if isinstance(cache, dict) else {}
    normalised = _new_translation_cache(doc_id, source)
    normalised["items"] = dict(source.get("items") or {})
    normalised["outcomes"] = dict(source.get("outcomes") or {})
    if isinstance(source.get("last_call"), dict):
        normalised["last_call"] = dict(source["last_call"])
    return normalised


def _get_translation_cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _merge_translation_caches(existing: dict[str, Any], candidate: dict[str, Any], doc_id: str) -> dict[str, Any]:
    candidate = _normalise_translation_cache(candidate, doc_id)
    if not _cache_matches_identity(existing, _cache_identity_from_envelope(candidate)):
        return candidate

    merged = _normalise_translation_cache(existing, doc_id)
    merged.update({
        "parser_route": candidate["parser_route"],
        "parse_generation": candidate["parse_generation"],
        "document_source_hash": candidate["document_source_hash"],
        "block_index_hash": candidate["block_index_hash"],
        "ai_cache_generation": candidate["ai_cache_generation"],
    })
    merged["items"].update(candidate["items"])
    for cache_key, outcome in candidate["outcomes"].items():
        if not isinstance(outcome, dict):
            continue
        existing_outcome = merged["outcomes"].get(cache_key)
        if (
            not isinstance(existing_outcome, dict)
            or float(outcome.get("updated_at") or 0) >= float(existing_outcome.get("updated_at") or 0)
        ):
            merged["outcomes"][cache_key] = dict(outcome)
    if "last_call" in candidate:
        merged["last_call"] = candidate["last_call"]
    return merged


def _atomic_write_translation_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _translation_cache_identity(
    block_index: dict[str, Any],
    *,
    ai_cache_generation: str = LEGACY_AI_CACHE_GENERATION,
) -> dict[str, str]:
    block_index_hash = _block_index_hash(block_index)
    source = str(block_index.get("source") or "").strip().lower()
    parser_route = str(block_index.get("parser_route") or "").strip().lower()
    if parser_route not in {"local", "mineru"}:
        parser_route = "mineru" if source == "mineru_vlm" else "local"
    # Pre-manifest documents retain a deterministic legacy identity. This keeps
    # their cache usable while still making it impossible to reuse it after the
    # block structure changes or a modern parse identity is published.
    return {
        "parser_route": parser_route,
        "parse_generation": str(block_index.get("parse_generation") or f"legacy-{block_index_hash[:24]}"),
        "document_source_hash": str(block_index.get("document_source_hash") or f"legacy-{block_index_hash}"),
        "block_index_hash": block_index_hash,
        "ai_cache_generation": str(ai_cache_generation or LEGACY_AI_CACHE_GENERATION),
    }


def _cache_identity_from_envelope(cache: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(cache.get(key) or "")
        for key in (
            "parser_route",
            "parse_generation",
            "document_source_hash",
            "block_index_hash",
            "ai_cache_generation",
        )
    }


def _cache_matches_identity(cache: dict[str, Any], identity: dict[str, str]) -> bool:
    if not isinstance(cache, dict):
        return False
    return all(
        str(cache.get(key) or "") == str(identity.get(key) or "")
        for key in (
            "parser_route",
            "parse_generation",
            "document_source_hash",
            "block_index_hash",
            "ai_cache_generation",
        )
    )


def _block_index_hash(block_index: dict[str, Any]) -> str:
    pages: list[dict[str, Any]] = []
    for page in block_index.get("pages", []) or []:
        page_number = int(page.get("page") or 1)
        blocks: list[dict[str, Any]] = []
        for raw_block in page.get("blocks", []) or []:
            block = dict(raw_block) if isinstance(raw_block, dict) else {}
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            text = str(block.get("text") or "").strip()
            blocks.append({
                "block_id": block_id,
                "type": _normalise_block_type(block),
                "text": text,
                "translation_mode": _get_translation_mode(block, text),
            })
        pages.append({"page": page_number, "blocks": blocks})
    payload = json.dumps(pages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_block_type(block: dict[str, Any]) -> str:
    return str(block.get("type") or "paragraph").strip().lower() or "paragraph"


def _block_signature(block: dict[str, Any]) -> str:
    text = str(block.get("text") or "").strip()
    payload = {
        "block_id": str(block.get("block_id") or ""),
        "page": block.get("page"),
        "type": _normalise_block_type(block),
        "text": text,
        "translation_mode": _get_translation_mode(block, text),
    }
    serialised = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _record_matches_block(record: Any, block: dict[str, Any], target_lang: str) -> bool:
    if not _is_valid_translation_record(record):
        return False
    text = str(block.get("text") or "").strip()
    return (
        str(record.get("target_lang") or "") == _normalize_target_lang(target_lang)
        and record.get("source_hash") == _source_hash(text)
        and record.get("prompt_version") == TRANSLATION_PROMPT_VERSION
        and record.get("block_type") == _normalise_block_type(block)
        and record.get("translation_mode") == _get_translation_mode(block, text)
        and record.get("block_signature") == _block_signature(block)
    )


async def _write_translation_cache(cache_writer: TranslationCacheWriter, cache: dict[str, Any]) -> None:
    outcome = cache_writer(cache)
    if inspect.isawaitable(outcome):
        await outcome


def flatten_blocks(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            item = dict(block)
            item["page"] = page_num
            result[block_id] = item
    return result


def get_cached_translations(
    *,
    data_dir: Path | str,
    doc_id: str,
    block_index: dict[str, Any],
    target_lang: str = "zh",
    provider: str | None = None,
    model: str | None = None,
    ai_cache_generation: str = LEGACY_AI_CACHE_GENERATION,
) -> dict[str, Any]:
    cache = load_translation_cache(data_dir, doc_id)
    block_map = flatten_blocks(block_index)
    cache_identity = _translation_cache_identity(
        block_index,
        ai_cache_generation=ai_cache_generation,
    )
    normalized_target = _normalize_target_lang(target_lang)
    if not _cache_matches_identity(cache, cache_identity):
        return {
            "doc_id": doc_id,
            "target_lang": normalized_target,
            "items": {},
            "failed_block_ids": [],
            "skipped_block_ids": [],
            "status": "empty",
        }
    items: dict[str, Any] = {}
    for block_id, block in block_map.items():
        cache_key = _cache_key(block_id, normalized_target)
        cached = cache.get("items", {}).get(cache_key)
        if not _record_matches_block(cached, block, normalized_target):
            continue
        if provider is not None and str(cached.get("provider") or "") != str(provider):
            continue
        if model is not None and str(cached.get("model") or "") != str(model):
            continue
        items[block_id] = cached

    failed_block_ids: list[str] = []
    skipped_block_ids: list[str] = []
    for outcome in (cache.get("outcomes") or {}).values():
        if not isinstance(outcome, dict):
            continue
        status = str(outcome.get("status") or "").strip().lower()
        if status not in {"failed", "skipped"}:
            continue
        if str(outcome.get("target_lang") or "") != normalized_target:
            continue
        if provider is not None and str(outcome.get("provider") or "") != str(provider):
            continue
        if model is not None and str(outcome.get("model") or "") != str(model):
            continue
        block_id = str(outcome.get("block_id") or "").strip()
        if not block_id:
            continue
        block = block_map.get(block_id)
        expected_signature = str(outcome.get("block_signature") or "")
        if block is not None and expected_signature and expected_signature != _block_signature(block):
            continue
        target = failed_block_ids if status == "failed" else skipped_block_ids
        if block_id not in target:
            target.append(block_id)

    if failed_block_ids or skipped_block_ids:
        cache_status = "partial" if items else ("failed" if failed_block_ids else "skipped")
    else:
        cache_status = "cached" if items else "empty"
    return {
        "doc_id": doc_id,
        "target_lang": normalized_target,
        "items": items,
        "failed_block_ids": failed_block_ids,
        "skipped_block_ids": skipped_block_ids,
        "status": cache_status,
    }


def _translation_outcome(
    *,
    block_id: str,
    target_lang: str,
    status: str,
    provider: str,
    model: str,
    reason: str = "",
    block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "block_id": str(block_id),
        "target_lang": target_lang,
        "status": status,
        "reason": reason,
        "provider": provider,
        "model": model,
        "block_signature": _block_signature(block) if block is not None else "",
        "updated_at": time.time(),
    }


async def translate_blocks(
    *,
    data_dir: Path | str,
    doc_id: str,
    block_index: dict[str, Any],
    block_ids: list[str],
    target_lang: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    force: bool = False,
    max_blocks: int | None = MAX_BLOCKS_PER_REQUEST,
    concurrency: int = TRANSLATION_CONCURRENCY,
    cache_writer: TranslationCacheWriter | None = None,
    ai_cache_generation: str = LEGACY_AI_CACHE_GENERATION,
) -> dict[str, Any]:
    normalized_target = _normalize_target_lang(target_lang)
    unique_ids = list(dict.fromkeys(str(block_id) for block_id in block_ids if str(block_id).strip()))
    if not unique_ids:
        return {
            "doc_id": doc_id,
            "target_lang": normalized_target,
            "items": {},
            "failed_block_ids": [],
            "skipped_block_ids": [],
            "status": "empty",
        }
    if max_blocks is not None and len(unique_ids) > max_blocks:
        unique_ids = unique_ids[:max_blocks]

    block_map = flatten_blocks(block_index)
    cache_identity = _translation_cache_identity(
        block_index,
        ai_cache_generation=ai_cache_generation,
    )
    cache = load_translation_cache(data_dir, doc_id)
    if not _cache_matches_identity(cache, cache_identity):
        cache = _new_translation_cache(doc_id, cache_identity)
    if cache_writer is None:
        def cache_writer(envelope: dict[str, Any]) -> None:
            save_translation_cache(data_dir, doc_id, envelope)

    cache_items = cache.setdefault("items", {})
    cache_outcomes = cache.setdefault("outcomes", {})
    cache["last_call"] = {
        "purpose": "block_translation",
        "provider": provider,
        "model": model,
        "target_lang": normalized_target,
        "requested_count": len(unique_ids),
        "force": bool(force),
        "updated_at": time.time(),
    }
    logger.info(
        "[AI-Audit] purpose=block_translation doc=%s provider=%s model=%s requested=%s force=%s",
        doc_id,
        provider,
        model,
        len(unique_ids),
        bool(force),
    )

    result_items: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    skipped_block_ids: list[str] = []
    outcomes_changed = False

    for block_id in unique_ids:
        block = block_map.get(block_id)
        cache_key = _cache_key(block_id, normalized_target)
        if not block:
            skipped_block_ids.append(block_id)
            cache_outcomes[cache_key] = _translation_outcome(
                block_id=block_id,
                target_lang=normalized_target,
                status="skipped",
                provider=provider,
                model=model,
                reason="block_not_found",
            )
            outcomes_changed = True
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            skipped_block_ids.append(block_id)
            cache_outcomes[cache_key] = _translation_outcome(
                block_id=block_id,
                target_lang=normalized_target,
                status="skipped",
                provider=provider,
                model=model,
                reason="empty_block_text",
                block=block,
            )
            outcomes_changed = True
            continue
        source_hash = _source_hash(text)
        block_type = _normalise_block_type(block)
        translation_mode = _get_translation_mode(block, text)
        cached = cache_items.get(cache_key)
        if (
            not force
            and _record_matches_block(cached, block, normalized_target)
            and cached.get("model") == model
            and cached.get("provider") == provider
        ):
            result_items[block_id] = cached
            if cache_outcomes.pop(cache_key, None) is not None:
                outcomes_changed = True
            continue
        if cached and not _is_valid_translation_record(cached):
            cache_items.pop(cache_key, None)
        missing.append({
            "block_id": block_id,
            "type": block_type,
            "page": block.get("page"),
            "text": text,
            "source_hash": source_hash,
            "translation_mode": translation_mode,
            "block_signature": _block_signature(block),
        })

    if missing:
        missing_by_id = {block["block_id"]: block for block in missing}
        completed_block_ids: set[str] = set()

        async def persist_generated_item(item: dict[str, dict[str, str]]) -> None:
            if not item:
                return
            now = time.time()
            changed = False
            for block_id, payload in item.items():
                block = missing_by_id.get(block_id)
                if not block:
                    continue
                translation = str(payload.get("translation") or "").strip()
                if not _is_valid_translation_text(translation):
                    continue
                record = {
                    "block_id": block_id,
                    "target_lang": normalized_target,
                    "translation": translation,
                    "summary": "",
                    "source_hash": block["source_hash"],
                    "block_type": block["type"],
                    "model": model,
                    "provider": provider,
                    "prompt_version": TRANSLATION_PROMPT_VERSION,
                    "translation_mode": block["translation_mode"],
                    "block_signature": block["block_signature"],
                    "created_at": now,
                }
                cache_items[_cache_key(block_id, normalized_target)] = record
                cache_outcomes.pop(_cache_key(block_id, normalized_target), None)
                result_items[block_id] = record
                completed_block_ids.add(block_id)
                changed = True
            if changed:
                await _write_translation_cache(cache_writer, cache)

        await _generate_translations(
            blocks=missing,
            target_lang=normalized_target,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            concurrency=concurrency,
            on_item=persist_generated_item,
        )
        logger.info(
            "[AI-Audit] purpose=block_translation doc=%s provider=%s model=%s status=success generated=%s failed=%s",
            doc_id,
            provider,
            model,
            len(completed_block_ids),
            len(missing) - len(completed_block_ids),
        )
        failed_block_ids = [
            block["block_id"]
            for block in missing
            if block["block_id"] not in completed_block_ids
        ]
        for block_id in failed_block_ids:
            cache_outcomes[_cache_key(block_id, normalized_target)] = _translation_outcome(
                block_id=block_id,
                target_lang=normalized_target,
                status="failed",
                provider=provider,
                model=model,
                reason="model_returned_no_valid_translation",
                block=block_map.get(block_id),
            )
        await _write_translation_cache(cache_writer, cache)
    else:
        failed_block_ids = []
        logger.info(
            "[AI-Audit] purpose=block_translation doc=%s provider=%s model=%s status=cache_hit count=%s",
            doc_id,
            provider,
            model,
            len(result_items),
        )
        if outcomes_changed:
            await _write_translation_cache(cache_writer, cache)

    if failed_block_ids or skipped_block_ids:
        status = "partial" if result_items else ("failed" if failed_block_ids else "skipped")
    else:
        status = "completed" if result_items else "empty"
    return {
        "doc_id": doc_id,
        "target_lang": normalized_target,
        "items": result_items,
        "failed_block_ids": failed_block_ids,
        "skipped_block_ids": skipped_block_ids,
        "status": status,
    }


async def _generate_translations(
    *,
    blocks: list[dict[str, Any]],
    target_lang: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    concurrency: int = TRANSLATION_CONCURRENCY,
    on_item: Callable[[dict[str, dict[str, str]]], Awaitable[None]] | None = None,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    safe_concurrency = max(1, min(int(concurrency or TRANSLATION_CONCURRENCY), MAX_TRANSLATION_CONCURRENCY))
    semaphore = asyncio.Semaphore(safe_concurrency)

    async def translate_one(block: dict[str, Any]) -> dict[str, dict[str, str]]:
        async with semaphore:
            async with _GLOBAL_TRANSLATION_SEMAPHORE:
                try:
                    return await _generate_single_plain_translation(
                        block=block,
                        target_lang=target_lang,
                        api_key=api_key,
                        model=model,
                        provider=provider,
                        endpoint=endpoint,
                    )
                except Exception as exc:
                    logger.warning(
                        "[BlockTranslation] block %s failed: %s",
                        block.get("block_id"),
                        exc,
                    )
                    return {}

    for offset in range(0, len(blocks), TRANSLATION_TASK_BATCH_SIZE):
        tasks = [
            asyncio.create_task(translate_one(block))
            for block in blocks[offset:offset + TRANSLATION_TASK_BATCH_SIZE]
        ]
        try:
            for task in asyncio.as_completed(tasks):
                item = await task
                result.update(item)
                if on_item:
                    await on_item(item)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    return result


async def _generate_single_plain_translation(
    *,
    block: dict[str, Any],
    target_lang: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, dict[str, str]]:
    """Last-resort single-block translation without JSON output."""
    block_id = str(block.get("block_id") or "")
    source = str(block.get("text") or "").strip()
    if not block_id or not source:
        return {}
    translation_mode = str(block.get("translation_mode") or "plain")

    if translation_mode == "preserve":
        return {
            block_id: {
                "translation": source,
                "summary": "",
                "translation_mode": "preserve",
            }
        }

    translated_segments: list[str] = []
    for segment in _split_translation_source(source):
        prefix, content, suffix = _split_outer_whitespace(segment)
        if not content:
            translated_segments.append(segment)
            continue
        translated = await _translate_source_segment(
            source=content,
            translation_mode=translation_mode,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )
        translated_segments.append(f"{prefix}{translated}{suffix}")

    translation = "".join(translated_segments).strip()
    if not _is_valid_translation_text(translation):
        raise RuntimeError("模型返回了无效译文")
    return {
        block_id: {
            "translation": translation,
            "summary": "",
            "translation_mode": translation_mode,
        }
    }


async def _translate_source_segment(
    *,
    source: str,
    translation_mode: str,
    target_lang: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> str:
    glossary = build_glossary_prompt(source, "中文" if target_lang == "zh" else target_lang)
    system_prompt = (
        "你是学术论文翻译助手。只输出最终译文，不要解释。"
        "保留公式、变量、引用编号、LaTeX 代码和专有名词的必要原形。"
        "不要把数学表达式翻译成中文；行内公式必须用 $...$ 包裹，块级公式必须用 $$...$$ 包裹。"
        "如果原文公式缺少 LaTeX 分隔符，也要在译文中补上对应的 $ 分隔符。"
    )
    if translation_mode == "formula_guard":
        system_prompt += (
            "当前文本包含较多公式或符号：只翻译自然语言部分，所有公式、变量、下标、上标、单位、引用编号必须原样保留。"
            "不要重排、合并或改写任何公式。"
        )

    response = await call_ai_api(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"目标语言：{target_lang}\n"
                    f"{glossary}\n\n"
                    "请翻译下面这个论文文本块，只输出译文：\n"
                    f"{source}"
                ),
            },
        ],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=_estimate_translation_max_tokens(source),
        temperature=0.2,
        purpose="translation",
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))

    content = _extract_content(response)
    if not content.strip():
        raise RuntimeError("模型未返回译文正文")

    translation = _clean_translation_text(content)
    if not _is_valid_translation_text(translation):
        raise RuntimeError("模型返回了无效译文")
    return translation


def _split_translation_source(source: str, max_chars: int = MAX_BLOCK_CHARS) -> list[str]:
    """Split long text at safe boundaries without cutting protected math spans."""
    text = str(source or "")
    if not text:
        return []
    safe_max = max(1, int(max_chars or MAX_BLOCK_CHARS))
    if len(text) <= safe_max:
        return [text]

    protected_spans = _protected_math_spans(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        tentative = min(len(text), start + safe_max)
        if tentative >= len(text):
            chunks.append(text[start:])
            break

        containing_span = next(
            ((span_start, span_end) for span_start, span_end in protected_spans
             if span_start < tentative < span_end),
            None,
        )
        search_limit = containing_span[0] if containing_span and containing_span[0] > start else tentative
        cut = _find_translation_cut(text, start, search_limit, protected_spans, safe_max)
        if cut <= start:
            if containing_span and containing_span[0] <= start:
                cut = containing_span[1]
            else:
                cut = tentative
        chunks.append(text[start:cut])
        start = cut

    return chunks


def _protected_math_spans(text: str) -> list[tuple[int, int]]:
    spans = sorted(
        (match.start(), match.end())
        for pattern in _PROTECTED_MATH_PATTERNS
        for match in pattern.finditer(text)
        if match.end() > match.start()
    )
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _find_translation_cut(
    text: str,
    start: int,
    limit: int,
    protected_spans: list[tuple[int, int]],
    max_chars: int,
) -> int:
    if limit <= start:
        return start
    lower_bound = start + max(1, min(max_chars // 2, limit - start))

    def cut_is_safe(position: int) -> bool:
        return not any(span_start < position < span_end for span_start, span_end in protected_spans)

    boundary_checks = (
        lambda position: text[max(start, position - 2):position] in {"\n\n", "\r\n"},
        lambda position: text[position - 1] in "\n\r",
        lambda position: text[position - 1] in ".!?。！？;；",
        lambda position: text[position - 1].isspace(),
    )
    for matches_boundary in boundary_checks:
        for position in range(limit, lower_bound - 1, -1):
            if cut_is_safe(position) and matches_boundary(position):
                return position
    return limit if cut_is_safe(limit) else start


def _split_outer_whitespace(value: str) -> tuple[str, str, str]:
    leading_length = len(value) - len(value.lstrip())
    trailing_length = len(value) - len(value.rstrip())
    content_end = len(value) - trailing_length if trailing_length else len(value)
    return value[:leading_length], value[leading_length:content_end], value[content_end:]


def _get_translation_mode(block: dict[str, Any], text: str) -> str:
    declared_mode = str(block.get("translation_mode") or "").strip().lower()
    if declared_mode in {"plain", "preserve", "formula_guard"}:
        return declared_mode
    block_type = str(block.get("type") or "paragraph").lower()
    if block_type in TABLE_BLOCK_TYPES or _looks_like_markdown_table(text):
        return "preserve"
    if _looks_formula_dense(text):
        return "formula_guard"
    return "plain"


def _estimate_translation_max_tokens(source: str) -> int:
    text = str(source or "")
    # 英文论文翻译成中文通常不会超过原字符数对应 token 的 1.2-1.5 倍；
    # 短标题/图注不再给 600 token 的大上限，减少供应商调度延迟。
    return max(160, min(3072, int(len(text) * 1.45) + 96))


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    table_rows = sum(1 for line in lines if _RE_MARKDOWN_TABLE_ROW.match(line))
    has_separator = any(_RE_MARKDOWN_TABLE_SEPARATOR.match(line) for line in lines)
    return table_rows >= 2 and has_separator


def _looks_formula_dense(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return False
    latex_hits = len(_RE_LATEX_COMMAND.findall(value))
    equation_hits = len(_RE_EQUATION_SIGNAL.findall(value))
    symbol_count = sum(1 for ch in compact if ch in "_^=+-*/\\{}[]()<>≤≥≈∑∏√")
    symbol_ratio = symbol_count / max(1, len(compact))
    word_count = len(re.findall(r"[A-Za-z]{2,}", value))
    if latex_hits >= 2 or equation_hits >= 2:
        return True
    return symbol_ratio >= 0.16 and word_count <= 80


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(message.get("content"), str):
                return message["content"]
        return ""
    return str(response or "")


def _is_valid_translation_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return _is_valid_translation_text(record.get("translation"))


def _is_valid_translation_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _RE_METADATA_DUMP_SIGNAL.search(text):
        return False
    if "_used_provider" in text and "_usage_meta" in text:
        return False
    if "completion_tokens" in text and "prompt_tokens" in text and "translation" not in text[:80].lower():
        return False
    return True


def _strip_markdown_fence(content: str) -> str:
    text = (content or "").strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    if lines and lines[0].strip().lower() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


def _clean_translation_text(content: str) -> str:
    text = _strip_markdown_fence(content)
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)^.*?</think>", "", text).strip()

    prefix_patterns = [
        r"^(?:翻译|译文|译文如下|翻译如下|中文翻译|结果|Translation|Translated text)\s*[:：]\s*",
        r"^以下是(?:该段|文本|内容)?(?:的)?(?:中文)?译文\s*[:：]\s*",
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    quote_pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』")}
    if len(text) >= 2 and (text[0], text[-1]) in quote_pairs:
        text = text[1:-1].strip()

    return text


def _source_hash(text: Any) -> str:
    value = str(text or "")
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _cache_key(block_id: str, target_lang: str) -> str:
    return f"{target_lang}:{block_id}"


def _normalize_target_lang(target_lang: str) -> str:
    normalized = (target_lang or "zh").strip().lower()
    if normalized in {"zh-cn", "cn", "chinese", "中文"}:
        return "zh"
    return normalized or "zh"
