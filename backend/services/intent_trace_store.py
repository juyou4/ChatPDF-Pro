"""意图 trace 的本地落盘：只追加、按天滚动的 JSONL。

纯旁路设施，不参与任何判定：
- 开关关闭时不产生任何文件 IO；
- 任何异常一律静默吞掉并记 debug 日志，绝不允许影响聊天主链路；
- trace 本身只带问题的 hash 与前 40 字符预览，问题原文不落盘。
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

_TRACE_DIR_NAME = "intent_traces"
_CORRECTION_DIR_NAME = "intent_corrections"
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_ROLL_INDEX = 999
_TRACE_WRITE_LOCK = threading.RLock()


def _trace_dir() -> Path:
    """沿用项目既有的数据目录约定：desktop 走 AppData，server 走项目根 data/。"""
    from runtime_mode import runtime

    return Path(runtime.data_dir) / _TRACE_DIR_NAME


def _is_full(path: Path) -> bool:
    try:
        return path.stat().st_size >= _MAX_FILE_BYTES
    except OSError:
        # 文件不存在或不可读都按"没满"处理，交给后续 open 决定成败。
        return False


def _trace_file(directory: Path, day: str, *, stem: str = "intent_trace") -> Path:
    """当天主文件写满 50MB 后顺延到 .1/.2/... 分片。"""
    base = directory / f"{stem}_{day}.jsonl"
    if not _is_full(base):
        return base
    for index in range(1, _MAX_ROLL_INDEX + 1):
        candidate = directory / f"{stem}_{day}.{index}.jsonl"
        if not _is_full(candidate):
            return candidate
    return directory / f"{stem}_{day}.{_MAX_ROLL_INDEX}.jsonl"


def _trace_files(directory: Path, *, pattern: str = "intent_trace_*.jsonl") -> list[Path]:
    return sorted(
        (
            path for path in directory.glob(pattern)
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
    )


def _prune_traces(
    directory: Path,
    *,
    now: datetime,
    max_total_bytes: int,
    retention_days: int,
    pattern: str = "intent_trace_*.jsonl",
) -> None:
    """Bound trace retention without ever touching non-trace runtime data."""
    files = _trace_files(directory, pattern=pattern)
    if retention_days > 0:
        cutoff = now - timedelta(days=retention_days)
        retained: list[Path] = []
        for path in files:
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                continue
            if modified < cutoff:
                try:
                    path.unlink()
                except OSError:
                    retained.append(path)
            else:
                retained.append(path)
        files = retained

    limit = max(0, int(max_total_bytes or 0))
    if not limit:
        return
    sized: list[tuple[Path, int]] = []
    total = 0
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        sized.append((path, size))
        total += size
    for path, size in sized:
        if total <= limit:
            break
        try:
            path.unlink()
            total -= size
        except OSError:
            continue


def append_intent_trace(trace: dict) -> bool:
    """追加一条 trace，返回是否真的写了盘。永不抛异常。"""
    try:
        from config import settings

        if not bool(getattr(settings, "intent_trace_enabled", False)):
            return False
        if not isinstance(trace, dict) or not trace:
            return False
        now = datetime.now()
        record = {"ts": now.isoformat(timespec="seconds"), **trace}
        if not bool(getattr(settings, "intent_trace_include_question_preview", False)):
            # Hash + route labels are enough to aggregate regressions. Keep the
            # optional readable preview out of disk unless a developer turns it
            # on deliberately for a bounded debugging session.
            record.pop("question_preview", None)
        serialized = json.dumps(record, ensure_ascii=False) + "\n"
        with _TRACE_WRITE_LOCK:
            directory = _trace_dir()
            directory.mkdir(parents=True, exist_ok=True)
            _prune_traces(
                directory,
                now=now,
                max_total_bytes=int(getattr(settings, "intent_trace_max_total_bytes", 0) or 0),
                retention_days=max(0, int(getattr(settings, "intent_trace_retention_days", 0) or 0)),
            )
            path = _trace_file(directory, now.strftime("%Y-%m-%d"))
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(serialized)
        return True
    except Exception as exc:
        logger.debug(f"[IntentTrace] 写盘失败，已忽略: {exc}")
        return False


def append_intent_correction(correction: dict) -> bool:
    """Persist an explicit user verdict without retaining question text.

    Unlike passive traces, this is initiated by a user action and therefore is
    not gated by ``intent_trace_enabled``. It remains bounded by the same
    retention and size settings and never participates in online routing.
    """
    try:
        from config import settings
        from runtime_mode import runtime

        if not isinstance(correction, dict):
            return False
        intent_id = str(correction.get("intent_id") or "").strip()[:64]
        verdict = str(correction.get("verdict") or "").strip().lower()
        if not intent_id or verdict not in {"correct", "incorrect"}:
            return False
        now = datetime.now()
        safe_record = {
            "event_version": "intent_correction_v1",
            "ts": now.isoformat(timespec="seconds"),
            "intent_id": intent_id,
            "intent_version": str(correction.get("intent_version") or "")[:32],
            "verdict": verdict,
            "predicted_task": str(correction.get("predicted_task") or "")[:40],
            "predicted_scope": str(correction.get("predicted_scope") or "")[:40],
            "predicted_is_ambiguous": bool(correction.get("predicted_is_ambiguous")),
            "decision_strength": max(
                0.0,
                min(1.0, float(correction.get("decision_strength") or 0.0)),
            ),
            "corrected_task": str(correction.get("corrected_task") or "")[:40] or None,
            "corrected_scope": str(correction.get("corrected_scope") or "")[:40] or None,
            "corrected_is_ambiguous": correction.get("corrected_is_ambiguous"),
        }
        safe_record["event_id"] = hashlib.sha256(
            json.dumps(safe_record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        serialized = json.dumps(safe_record, ensure_ascii=False) + "\n"
        with _TRACE_WRITE_LOCK:
            directory = Path(runtime.data_dir) / _CORRECTION_DIR_NAME
            directory.mkdir(parents=True, exist_ok=True)
            _prune_traces(
                directory,
                now=now,
                max_total_bytes=int(getattr(settings, "intent_trace_max_total_bytes", 0) or 0),
                retention_days=max(0, int(getattr(settings, "intent_trace_retention_days", 0) or 0)),
                pattern="intent_correction_*.jsonl",
            )
            path = _trace_file(directory, now.strftime("%Y-%m-%d"), stem="intent_correction")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(serialized)
        return True
    except Exception as exc:
        logger.debug(f"[IntentCorrection] 写盘失败，已忽略: {exc}")
        return False
