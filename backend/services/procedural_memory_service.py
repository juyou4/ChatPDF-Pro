"""程序记忆（procedural memory）：按题型记录 agent 的成功检索策略。

参考 mem0 的 ``MemoryType.PROCEDURAL`` 与 ragflow 的 ``rank_memory`` 思路，
但刻意不走 MemoryService 的用户事实记忆管线：策略统计不含用户内容，
无需 LLM 仲裁与 embedding，轻量 JSON 落盘即可。

防坏策略固化的约束：
- 只记录正反馈（evidence 完成状态为 answered 的轮次）；
- 条目带失效期（30 天），读取时过滤过期项；
- 注入的 hint 明确标注"可参考、不强制"，planner 仍可自由规划。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_MAX_STRATEGIES_PER_TYPE = 3
_MAX_TOOL_SEQUENCE_LENGTH = 8
_STRATEGY_TTL_SECONDS = 30 * 24 * 3600
_MAX_QUESTION_PREVIEW = 80

_lock = threading.Lock()

# 控制性工具不构成检索策略。
_EXCLUDED_TOOLS = {"complete"}


def _store_dir() -> Path:
    from runtime_mode import runtime

    return Path(runtime.data_dir) / "procedural_memory"


def _sanitize_doc_id(doc_id: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]", "_", str(doc_id or "").strip())
    return text[:120]


def _doc_path(doc_id: str) -> Optional[Path]:
    safe_id = _sanitize_doc_id(doc_id)
    if not safe_id:
        return None
    return _store_dir() / f"{safe_id}.json"


def _normalize_query_type(query_type: str) -> str:
    text = re.sub(r"[^a-z0-9_]", "", str(query_type or "").strip().lower())
    return text or "general"


def normalize_tool_sequence(tool_sequence: Any) -> List[str]:
    """压缩相邻重复并过滤控制性工具，得到可比较的策略序列。"""
    normalized: List[str] = []
    for item in tool_sequence or []:
        tool = str(item or "").strip()
        if not tool or tool in _EXCLUDED_TOOLS:
            continue
        if normalized and normalized[-1] == tool:
            continue
        normalized.append(tool)
        if len(normalized) >= _MAX_TOOL_SEQUENCE_LENGTH:
            break
    return normalized


def _load_payload(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[ProceduralMemory] 读取失败，按空库处理: %s", exc)
    return {"schema_version": _SCHEMA_VERSION, "strategies": {}}


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _fresh_entries(entries: Any, now: float) -> List[dict]:
    fresh: List[dict] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            updated_at = float(entry.get("updated_at") or 0)
        except (TypeError, ValueError):
            updated_at = 0.0
        if now - updated_at > _STRATEGY_TTL_SECONDS:
            continue
        tools = [str(tool) for tool in (entry.get("tools") or []) if str(tool).strip()]
        if not tools:
            continue
        fresh.append({
            "tools": tools[:_MAX_TOOL_SEQUENCE_LENGTH],
            "count": max(1, int(entry.get("count") or 1)),
            "updated_at": updated_at,
            "question_preview": str(entry.get("question_preview") or "")[:_MAX_QUESTION_PREVIEW],
        })
    return fresh


def record_successful_strategy(
    doc_id: str,
    query_type: str,
    tool_sequence: Any,
    question: str = "",
) -> bool:
    """记录一次 evidence 侧判定成功的工具序列；失败静默返回 False。"""
    tools = normalize_tool_sequence(tool_sequence)
    if not tools:
        return False
    path = _doc_path(doc_id)
    if path is None:
        return False
    normalized_type = _normalize_query_type(query_type)
    preview = re.sub(r"\s+", " ", str(question or "")).strip()[:_MAX_QUESTION_PREVIEW]
    now = time.time()
    try:
        with _lock:
            payload = _load_payload(path)
            strategies = payload.get("strategies")
            if not isinstance(strategies, dict):
                strategies = {}
            entries = _fresh_entries(strategies.get(normalized_type), now)
            for entry in entries:
                if entry["tools"] == tools:
                    entry["count"] += 1
                    entry["updated_at"] = now
                    if preview:
                        entry["question_preview"] = preview
                    break
            else:
                entries.insert(0, {
                    "tools": tools,
                    "count": 1,
                    "updated_at": now,
                    "question_preview": preview,
                })
            entries.sort(key=lambda item: (-int(item.get("count") or 0), -float(item.get("updated_at") or 0)))
            strategies[normalized_type] = entries[:_MAX_STRATEGIES_PER_TYPE]
            payload["schema_version"] = _SCHEMA_VERSION
            payload["strategies"] = strategies
            _write_payload(path, payload)
        return True
    except Exception as exc:
        logger.warning("[ProceduralMemory] 写入失败（忽略，不影响检索）: %s", exc)
        return False


def suggest_strategy(doc_id: str, query_type: str) -> str:
    """返回同文档同题型的历史成功策略 hint 文本；无可用策略时返回空串。"""
    path = _doc_path(doc_id)
    if path is None or not path.exists():
        return ""
    normalized_type = _normalize_query_type(query_type)
    try:
        with _lock:
            payload = _load_payload(path)
        strategies = payload.get("strategies")
        entries = _fresh_entries(
            strategies.get(normalized_type) if isinstance(strategies, dict) else [],
            time.time(),
        )
        if not entries:
            return ""
        best = entries[0]
        # 单次成功还不足以构成"策略"，避免偶然序列被放大。
        if int(best.get("count") or 0) < 2:
            return ""
        sequence_text = " → ".join(best["tools"])
        return (
            f"📎 本文档同类问题（{normalized_type}）历史成功策略：{sequence_text}"
            "；可优先参考，但仍按当前问题自行判断"
        )
    except Exception as exc:
        logger.warning("[ProceduralMemory] 读取失败（忽略）: %s", exc)
        return ""


__all__ = [
    "normalize_tool_sequence",
    "record_successful_strategy",
    "suggest_strategy",
]
