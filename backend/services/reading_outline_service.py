"""AI structured reading outline with evidence block bindings."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from services.chat_service import call_ai_api

logger = logging.getLogger(__name__)

READING_OUTLINE_VERSION = 1
MAX_PROMPT_BLOCKS = 96
MAX_BLOCK_TEXT = 700

_KEYWORD_GROUPS = {
    "background": [
        "abstract", "introduction", "background", "motivation", "problem",
        "challenge", "limitation", "gap", "existing", "previous",
    ],
    "innovation": [
        "contribution", "we propose", "we present", "novel", "innovation",
        "framework", "pipeline", "method", "approach", "in short",
    ],
    "experiment": [
        "experiment", "evaluation", "result", "comparison", "ablation",
        "dataset", "setup", "metric", "table", "figure",
    ],
    "conclusion": [
        "conclusion", "discussion", "future", "summary", "value",
    ],
}


def get_reading_outline_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "reading_outlines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reading_outline_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_reading_outline_dir(data_dir) / f"{doc_id}.json"


def load_reading_outline(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_reading_outline_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != READING_OUTLINE_VERSION:
            return None
        return data
    except Exception as exc:
        logger.warning("[ReadingOutline] Failed to load %s: %s", path, exc)
        return None


def save_reading_outline(data_dir: Path | str, doc_id: str, outline: dict[str, Any]) -> None:
    path = get_reading_outline_path(data_dir, doc_id)
    outline["version"] = READING_OUTLINE_VERSION
    outline["doc_id"] = doc_id
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[ReadingOutline] Failed to save %s: %s", path, exc)


async def get_or_create_reading_outline(
    *,
    data_dir: Path | str,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Return cached AI outline, generate one when credentials are available.

    If no model credentials are available, returns a deterministic fallback with
    the same evidence-bound schema, so the UI remains usable.
    """
    source_hash = _source_hash(block_index)
    provider_lower = (provider or "").lower()
    can_call_model = bool(api_key) or provider_lower in {"local", "ollama"}

    cached = None if force else load_reading_outline(data_dir, doc_id)
    if cached and cached.get("source_hash") == source_hash:
        if cached.get("source") == "ai" or not can_call_model:
            return cached

    fallback = build_fallback_reading_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash,
    )
    if not can_call_model:
        return fallback

    try:
        generated = await _generate_ai_outline(
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )
        outline = _normalize_outline(
            raw=generated,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            source_hash=source_hash,
            source="ai",
            model=model,
            provider=provider,
        )
        if not outline.get("items"):
            raise ValueError("empty outline items")
        save_reading_outline(data_dir, doc_id, outline)
        return outline
    except Exception as exc:
        logger.warning("[ReadingOutline] AI generation failed for %s: %s", doc_id, exc)
        fallback.setdefault("meta", {})["generation_error"] = str(exc)
        return fallback


def build_fallback_reading_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str | None = None,
) -> dict[str, Any]:
    block_map = _flatten_blocks(block_index)
    ordered = list(block_map.values())
    title_block = _pick_title_block(ordered)
    title = _clean_title(title_block.get("text") if title_block else "") or _filename_title(doc)

    background = _pick_blocks(ordered, "background", limit=3)
    innovations = _pick_blocks(ordered, "innovation", limit=5)
    experiments = _pick_blocks(ordered, "experiment", limit=6)
    conclusions = _pick_blocks(ordered, "conclusion", limit=2)

    if not background:
        background = _first_paragraphs(ordered, start_page=1, limit=2)
    if not innovations:
        innovations = _first_paragraphs(ordered, start_page=1, limit=4, skip_ids={b["block_id"] for b in background})
    if not experiments:
        experiments = [b for b in ordered if b.get("type") in {"caption", "figure", "table"}][:6]
    if not conclusions:
        conclusions = [b for b in reversed(ordered) if b.get("type") == "paragraph"][:2]

    heading_children = _fallback_heading_children(ordered, limit=5)
    experiment_children_source = _fallback_experiment_children(ordered, limit=6)

    def node(
        node_id: str,
        node_type: str,
        node_title: str,
        blocks: list[dict[str, Any]],
        summary: str = "",
        children: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        evidence = [b["block_id"] for b in blocks if b.get("block_id")]
        text = summary or _summarize_blocks(blocks)
        return {
            "id": node_id,
            "type": node_type,
            "title": node_title,
            "summary": text,
            "evidence_block_ids": evidence[:4],
            "children": children or [],
        }

    innovation_children = [
        node(f"innovation_{idx + 1}", "innovation", _fallback_child_title(block, f"核心创新点 {idx + 1}"), [block])
        for idx, block in enumerate(heading_children)
    ]
    experiment_children = [
        node(f"experiment_{idx + 1}", "experiment", _fallback_child_title(block, f"实验结果 {idx + 1}"), [block])
        for idx, block in enumerate(experiment_children_source)
    ]

    raw = {
        "title": title,
        "items": [
            node("paper_title", "title", "论文标题", [title_block] if title_block else [], title),
            node("background", "background", "研究背景与问题", background),
            node("innovations", "innovation_group", "核心创新点", innovations, children=innovation_children),
            node("experiments", "experiment_group", "实验结果", experiments, children=experiment_children),
            node("conclusion", "conclusion", "结论与价值", conclusions),
        ],
    }
    return _normalize_outline(
        raw=raw,
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        source_hash=source_hash or _source_hash(block_index),
        source="fallback",
        model="",
        provider="",
    )


async def _generate_ai_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, Any]:
    blocks = _select_prompt_blocks(block_index)
    if not blocks:
        return {}

    system = (
        "你是学术论文精读助手，负责生成可定位到原文的中文结构化阅读大纲。"
        "所有要点必须依据提供的 block_id，不允许编造证据 ID。"
    )
    user = (
        "请根据论文块生成完整 AI 结构化解析。只输出 JSON，不要 Markdown。\n"
        "目标结构必须包含：论文标题、研究背景与问题、核心创新点(4-5条)、"
        "实验结果(按子实验/图表/消融分点)、结论与价值。\n\n"
        "JSON 格式：\n"
        "{\n"
        "  \"title\": \"论文英文标题\",\n"
        "  \"items\": [\n"
        "    {\"id\":\"paper_title\",\"type\":\"title\",\"title\":\"论文标题\",\"summary\":\"中文说明\","
        "\"evidence_block_ids\":[\"p1_b0\"],\"children\":[]},\n"
        "    {\"id\":\"background\",\"type\":\"background\",\"title\":\"研究背景与问题\",\"summary\":\"中文总结\","
        "\"evidence_block_ids\":[\"...\"],\"children\":[...]}\n"
        "  ]\n"
        "}\n\n"
        "要求：\n"
        "1. 每个叶子节点 evidence_block_ids 选 1-3 个最相关 block_id；分组节点可选其子节点证据合集。\n"
        "2. title 用简短中文，summary 用 1-3 句中文。\n"
        "3. 实验结果要尽量绑定 Figure/Table caption 或结果段落。\n"
        "4. 不要把 arXiv 水印、页码、单纯指标数字当作大纲节点。\n\n"
        f"文档名：{doc.get('filename', doc_id)}\n"
        f"可用 blocks：{json.dumps(blocks, ensure_ascii=False)}"
    )
    response = await call_ai_api(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=4200,
        temperature=0.2,
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))

    content = _extract_content(response)
    return _parse_json_object(content)


def _normalize_outline(
    *,
    raw: dict[str, Any],
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    source_hash: str,
    source: str,
    model: str,
    provider: str,
) -> dict[str, Any]:
    block_map = _flatten_blocks(block_index)
    used_ids: set[str] = set()
    fallback_title = _filename_title(doc)
    title = _limit(raw.get("title") or fallback_title, 180)

    def normalize_node(node: dict[str, Any], idx_path: str, inherited_type: str = "note") -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        raw_id = str(node.get("id") or f"node_{idx_path}").strip()
        node_id = _slug_id(raw_id) or f"node_{idx_path}"
        base_id = node_id
        suffix = 2
        while node_id in used_ids:
            node_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(node_id)

        node_type = _limit(node.get("type") or inherited_type, 40)
        node_title = _limit(node.get("title") or node.get("label") or "阅读要点", 120)
        summary = _limit(node.get("summary") or node.get("description") or "", 900)
        evidence_ids = _valid_block_ids(node.get("evidence_block_ids") or node.get("block_ids"), block_map)
        children = []
        for child_idx, child in enumerate(node.get("children") or []):
            child_node = normalize_node(child, f"{idx_path}_{child_idx + 1}", node_type)
            if child_node:
                children.append(child_node)

        if not evidence_ids and children:
            for child in children:
                for block_id in child.get("evidence_block_ids", []):
                    if block_id not in evidence_ids:
                        evidence_ids.append(block_id)
        if not evidence_ids:
            evidence_ids = _fallback_evidence(node_title, summary, node_type, block_map)

        pages = sorted({int(block_map[bid]["page"]) for bid in evidence_ids if bid in block_map})
        first_block = evidence_ids[0] if evidence_ids else None
        page = pages[0] if pages else 1

        return {
            "id": node_id,
            "type": node_type,
            "title": node_title,
            "summary": summary or _summarize_blocks([block_map[bid] for bid in evidence_ids if bid in block_map]),
            "evidence_block_ids": evidence_ids[:8],
            "evidence": {
                "block_ids": evidence_ids[:8],
                "pages": pages,
                "primary_page": page,
            },
            "page": page,
            "first_block": first_block,
            "children": children,
        }

    items = []
    for idx, item in enumerate(raw.get("items") or []):
        normalized = normalize_node(item, str(idx + 1))
        if normalized:
            items.append(normalized)

    return {
        "version": READING_OUTLINE_VERSION,
        "doc_id": doc_id,
        "source": source,
        "source_hash": source_hash,
        "title": title,
        "items": items,
        "flat_items": _flatten_outline_items(items),
        "created_at": time.time(),
        "model": model,
        "provider": provider,
        "meta": {
            "block_count": len(block_map),
            "page_count": len(block_index.get("pages", []) or []),
        },
    }


def _flatten_blocks(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            text = str(block.get("text") or "").strip()
            item = dict(block)
            item["page"] = page_num
            item["text"] = text
            result[block_id] = item
    return result


def _select_prompt_blocks(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    block_map = _flatten_blocks(block_index)
    ordered = list(block_map.values())
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(block: dict[str, Any]) -> None:
        block_id = block.get("block_id")
        text = str(block.get("text") or "").strip()
        if not block_id or block_id in seen or not text or _is_noise_text(text):
            return
        seen.add(block_id)
        selected.append({
            "block_id": block_id,
            "page": block.get("page"),
            "type": block.get("type") or "paragraph",
            "text": _limit(text, MAX_BLOCK_TEXT),
        })

    for block in ordered:
        if block.get("type") == "heading":
            add(block)
    for block in ordered:
        if int(block.get("page") or 1) <= 2 and block.get("type") in {"paragraph", "caption", "figure", "table"}:
            add(block)
    for group in _KEYWORD_GROUPS.values():
        for block in ordered:
            text = str(block.get("text") or "").lower()
            if any(keyword in text for keyword in group):
                add(block)
    for block in ordered:
        if block.get("type") in {"caption", "figure", "table"}:
            add(block)
    if len(selected) < 24:
        for block in ordered:
            if block.get("type") == "paragraph":
                add(block)
            if len(selected) >= 24:
                break
    return selected[:MAX_PROMPT_BLOCKS]


def _pick_title_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    first_page = [b for b in blocks if int(b.get("page") or 1) == 1 and b.get("text")]
    candidates = [
        b for b in first_page
        if 18 <= len(str(b.get("text"))) <= 220 and not _is_noise_text(str(b.get("text")))
    ]
    headings = [b for b in candidates if b.get("type") == "heading"]
    return (headings or candidates or first_page or [None])[0]


def _pick_blocks(blocks: list[dict[str, Any]], group: str, limit: int) -> list[dict[str, Any]]:
    keywords = _KEYWORD_GROUPS.get(group, [])
    result = []
    for block in blocks:
        text = str(block.get("text") or "")
        if not text or _is_noise_text(text):
            continue
        lower = text.lower()
        if any(keyword in lower for keyword in keywords):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _first_paragraphs(
    blocks: list[dict[str, Any]],
    *,
    start_page: int,
    limit: int,
    skip_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    skip_ids = skip_ids or set()
    result = []
    for block in blocks:
        if block.get("block_id") in skip_ids:
            continue
        if int(block.get("page") or 1) < start_page:
            continue
        text = str(block.get("text") or "")
        if block.get("type") == "paragraph" and len(text) > 80 and not _is_noise_text(text):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_evidence(
    title: str,
    summary: str,
    node_type: str,
    block_map: dict[str, dict[str, Any]],
) -> list[str]:
    query = f"{title} {summary} {node_type}".lower()
    keywords = re.findall(r"[A-Za-z][A-Za-z-]{3,}|[\u4e00-\u9fff]{2,}", query)
    scored = []
    for block in block_map.values():
        text = str(block.get("text") or "")
        if not text or _is_noise_text(text):
            continue
        lower = text.lower()
        score = sum(1 for keyword in keywords if keyword.lower() in lower)
        if node_type in {"experiment", "experiment_group"} and block.get("type") in {"caption", "figure", "table"}:
            score += 2
        if node_type in {"title", "background"} and int(block.get("page") or 1) <= 2:
            score += 1
        if score:
            scored.append((score, int(block.get("page") or 1), block["block_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:3]]


def _valid_block_ids(value: Any, block_map: dict[str, dict[str, Any]]) -> list[str]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    result = []
    for item in raw:
        block_id = str(item or "").strip()
        if block_id and block_id in block_map and block_id not in result:
            result.append(block_id)
    return result


def _flatten_outline_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], level: int) -> None:
        for node in nodes:
            item = {k: v for k, v in node.items() if k != "children"}
            item["level"] = level
            flat.append(item)
            walk(node.get("children") or [], level + 1)

    walk(items, 1)
    return flat


def _summarize_blocks(blocks: list[dict[str, Any]]) -> str:
    text = " ".join(str(block.get("text") or "").strip() for block in blocks if block)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = " ".join(text.split())
    return _limit(text, 220)


def _fallback_heading_children(blocks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if block.get("type") == "heading" and text and not _is_noise_text(text):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_experiment_children(blocks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text or _is_noise_text(text):
            continue
        block_type = block.get("type")
        lower = text.lower()
        if block_type == "heading" and any(keyword in lower for keyword in _KEYWORD_GROUPS["experiment"]):
            result.append(block)
        elif block_type in {"caption", "figure", "table"}:
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_child_title(block: dict[str, Any], fallback: str) -> str:
    text = str(block.get("text") or "").strip()
    if not text:
        return fallback
    if block.get("type") == "heading":
        return _limit(_clean_title(text), 60)
    if block.get("type") in {"caption", "figure", "table"}:
        return _limit(text, 60)
    return fallback


def _short_cn_title(block: dict[str, Any], fallback: str) -> str:
    text = str(block.get("text") or "").strip()
    if not text:
        return fallback
    if block.get("type") in {"caption", "figure", "table"}:
        return _limit(text, 34)
    sentence = re.split(r"[。.;；\n]", text, maxsplit=1)[0].strip()
    return _limit(sentence or text, 34)


def _clean_title(text: Any) -> str:
    title = " ".join(str(text or "").split())
    title = re.sub(r"^(title|paper title)\s*[:：]\s*", "", title, flags=re.IGNORECASE)
    return _limit(title, 180)


def _filename_title(doc: dict[str, Any]) -> str:
    name = str(doc.get("filename") or "当前论文")
    return re.sub(r"\.[A-Za-z0-9]+$", "", name).strip() or "当前论文"


def _is_noise_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if len(stripped) < 2:
        return True
    lower = stripped.lower()
    if "arxiv:" in lower:
        return True
    if re.fullmatch(r"[\d\s.,:/()%+\-=×x]+", stripped):
        return True
    alnum = re.sub(r"\s+", "", stripped)
    if alnum and sum(ch.isdigit() for ch in alnum) / max(len(alnum), 1) > 0.55:
        return True
    return False


def _source_hash(block_index: dict[str, Any]) -> str:
    parts = []
    for block in _flatten_blocks(block_index).values():
        parts.append(f"{block.get('block_id')}:{block.get('text')}")
    payload = "\n".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _slug_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip()).strip("_").lower()
    return text[:64]


def _limit(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."
