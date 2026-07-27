"""意图 trace 的本地落盘：只追加、按天滚动的 JSONL。

纯旁路设施，不参与任何判定：
- 开关关闭时不产生任何文件 IO；
- 任何异常一律静默吞掉并记 debug 日志，绝不允许影响聊天主链路；
- trace 本身只带问题的 hash 与前 40 字符预览，问题原文不落盘。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TRACE_DIR_NAME = "intent_traces"
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_ROLL_INDEX = 999


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


def _trace_file(directory: Path, day: str) -> Path:
    """当天主文件写满 50MB 后顺延到 .1/.2/... 分片。"""
    base = directory / f"intent_trace_{day}.jsonl"
    if not _is_full(base):
        return base
    for index in range(1, _MAX_ROLL_INDEX + 1):
        candidate = directory / f"intent_trace_{day}.{index}.jsonl"
        if not _is_full(candidate):
            return candidate
    return directory / f"intent_trace_{day}.{_MAX_ROLL_INDEX}.jsonl"


def append_intent_trace(trace: dict) -> bool:
    """追加一条 trace，返回是否真的写了盘。永不抛异常。"""
    try:
        from config import settings

        if not bool(getattr(settings, "intent_trace_enabled", False)):
            return False
        if not isinstance(trace, dict) or not trace:
            return False
        now = datetime.now()
        directory = _trace_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # ts 属于存储信封而非 trace 契约，因此只在写盘时补上。
        record = {"ts": now.isoformat(timespec="seconds"), **trace}
        path = _trace_file(directory, now.strftime("%Y-%m-%d"))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        logger.debug(f"[IntentTrace] 写盘失败，已忽略: {exc}")
        return False
