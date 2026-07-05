"""Block-level translation cache and generation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from services.chat_service import call_ai_api
from services.glossary_service import build_glossary_prompt

logger = logging.getLogger(__name__)

TRANSLATION_CACHE_VERSION = 1
TRANSLATION_PROMPT_VERSION = 4
MAX_BLOCKS_PER_REQUEST = 24
MAX_BLOCK_CHARS = 1800
TRANSLATION_CONCURRENCY = 5
MAX_TRANSLATION_CONCURRENCY = 8
TABLE_BLOCK_TYPES = {"table"}

_RE_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_RE_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")
_RE_LATEX_COMMAND = re.compile(r"\\[A-Za-z]{2,}")
_RE_EQUATION_SIGNAL = re.compile(r"(?:[_^]=?|[A-Za-z]\s*=|\\(?:frac|sum|prod|int|sqrt|begin|end)\b)")


def get_translation_cache_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "block_translations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_translation_cache_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_translation_cache_dir(data_dir) / f"{doc_id}.json"


def load_translation_cache(data_dir: Path | str, doc_id: str) -> dict[str, Any]:
    path = get_translation_cache_path(data_dir, doc_id)
    if not path.exists():
        return {"version": TRANSLATION_CACHE_VERSION, "doc_id": doc_id, "items": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != TRANSLATION_CACHE_VERSION:
            return {"version": TRANSLATION_CACHE_VERSION, "doc_id": doc_id, "items": {}}
        data.setdefault("items", {})
        return data
    except Exception as exc:
        logger.warning("[BlockTranslation] Failed to load %s: %s", path, exc)
        return {"version": TRANSLATION_CACHE_VERSION, "doc_id": doc_id, "items": {}}


def save_translation_cache(data_dir: Path | str, doc_id: str, cache: dict[str, Any]) -> None:
    path = get_translation_cache_path(data_dir, doc_id)
    cache["version"] = TRANSLATION_CACHE_VERSION
    cache["doc_id"] = doc_id
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[BlockTranslation] Failed to save %s: %s", path, exc)


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
) -> dict[str, Any]:
    cache = load_translation_cache(data_dir, doc_id)
    block_map = flatten_blocks(block_index)
    items: dict[str, Any] = {}
    for block_id, block in block_map.items():
        cache_key = _cache_key(block_id, target_lang)
        cached = cache.get("items", {}).get(cache_key)
        if not cached:
            continue
        if cached.get("source_hash") != _source_hash(block.get("text", "")):
            continue
        if cached.get("prompt_version") != TRANSLATION_PROMPT_VERSION:
            continue
        items[block_id] = cached
    return {
        "doc_id": doc_id,
        "target_lang": target_lang,
        "items": items,
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
) -> dict[str, Any]:
    normalized_target = _normalize_target_lang(target_lang)
    unique_ids = list(dict.fromkeys(str(block_id) for block_id in block_ids if str(block_id).strip()))
    if not unique_ids:
        return {"doc_id": doc_id, "target_lang": normalized_target, "items": {}}
    if max_blocks is not None and len(unique_ids) > max_blocks:
        unique_ids = unique_ids[:max_blocks]

    block_map = flatten_blocks(block_index)
    cache = load_translation_cache(data_dir, doc_id)
    cache_items = cache.setdefault("items", {})

    result_items: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []

    for block_id in unique_ids:
        block = block_map.get(block_id)
        if not block:
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        source_hash = _source_hash(text)
        cache_key = _cache_key(block_id, normalized_target)
        cached = cache_items.get(cache_key)
        if (
            not force
            and cached
            and cached.get("source_hash") == source_hash
            and cached.get("prompt_version") == TRANSLATION_PROMPT_VERSION
            and cached.get("model") == model
            and cached.get("provider") == provider
        ):
            result_items[block_id] = cached
            continue
        missing.append({
            "block_id": block_id,
            "type": block.get("type") or "paragraph",
            "page": block.get("page"),
            "text": text[:MAX_BLOCK_CHARS],
            "source_hash": source_hash,
            "translation_mode": _get_translation_mode(block, text),
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
                if not translation:
                    continue
                record = {
                    "block_id": block_id,
                    "target_lang": normalized_target,
                    "translation": translation,
                    "summary": "",
                    "source_hash": block["source_hash"],
                    "model": model,
                    "provider": provider,
                    "prompt_version": TRANSLATION_PROMPT_VERSION,
                    "translation_mode": payload.get("translation_mode") or block.get("translation_mode") or "plain",
                    "created_at": now,
                }
                cache_items[_cache_key(block_id, normalized_target)] = record
                result_items[block_id] = record
                completed_block_ids.add(block_id)
                changed = True
            if changed:
                save_translation_cache(data_dir, doc_id, cache)

        try:
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
        except asyncio.CancelledError:
            save_translation_cache(data_dir, doc_id, cache)
            raise

        failed_block_ids = [
            block["block_id"]
            for block in missing
            if block["block_id"] not in completed_block_ids
        ]
        save_translation_cache(data_dir, doc_id, cache)
    else:
        failed_block_ids = []

    return {
        "doc_id": doc_id,
        "target_lang": normalized_target,
        "items": result_items,
        "failed_block_ids": failed_block_ids,
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

    tasks = [asyncio.create_task(translate_one(block)) for block in blocks]
    try:
        for task in asyncio.as_completed(tasks):
            item = await task
            result.update(item)
            if on_item:
                await on_item(item)
    except asyncio.CancelledError:
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
        max_tokens=max(600, min(4096, len(source) * 2)),
        temperature=0.2,
        purpose="translation",
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))

    translation = _clean_translation_text(_extract_content(response))
    return {
        block_id: {
            "translation": translation or source,
            "summary": "",
            "translation_mode": translation_mode,
        }
    }


def _get_translation_mode(block: dict[str, Any], text: str) -> str:
    block_type = str(block.get("type") or "paragraph").lower()
    if block_type in TABLE_BLOCK_TYPES or _looks_like_markdown_table(text):
        return "preserve"
    if _looks_formula_dense(text):
        return "formula_guard"
    return "plain"


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
    return str(response or "")


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
