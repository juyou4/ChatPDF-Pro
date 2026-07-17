"""Shared execution controls for bounded visual-enrichment tasks.

Parser routes remain outside this module.  It owns only the operational
contract shared by figure descriptions and post-retrieval table checks:
concurrency, timeout, retry, task state, and per-document request budgets.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar


T = TypeVar("T")
_MAX_TASK_RECORDS = 1024
_MAX_BUDGET_SCOPES = 512
_MAX_RESULT_RECORDS = 256
_DEFAULT_BUDGET_TTL_S = 6 * 60 * 60
_TASK_LOCK = threading.Lock()
_TASK_STATUS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_BUDGET_LEDGER: OrderedDict[str, dict[str, Any]] = OrderedDict()
_TASK_RESULTS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_CACHE_MISS = object()


class VisualTaskTimeoutError(TimeoutError):
    """A visual task exhausted its bounded retry window."""


class VisualTaskBudgetExceeded(RuntimeError):
    """A document generation exhausted its visual-request budget."""


class VisualTaskUpstreamError(RuntimeError):
    """视觉模型网关以结构化 error 响应报告的可重试错误。"""


@dataclass(frozen=True)
class VisualTaskPolicy:
    timeout_seconds: float = 45.0
    max_retries: int = 1
    retry_delay_seconds: float = 0.4
    concurrency: int = 2
    document_budget: int = 16
    budget_units: int = 1
    budget_ttl_seconds: float = _DEFAULT_BUDGET_TTL_S
    cache_ttl_seconds: float = 0.0

    def normalized(self) -> "VisualTaskPolicy":
        return VisualTaskPolicy(
            timeout_seconds=max(5.0, min(180.0, float(self.timeout_seconds))),
            max_retries=max(0, min(3, int(self.max_retries))),
            retry_delay_seconds=max(0.0, min(10.0, float(self.retry_delay_seconds))),
            concurrency=max(1, min(8, int(self.concurrency))),
            document_budget=max(0, min(1000, int(self.document_budget))),
            budget_units=max(1, min(100, int(self.budget_units))),
            budget_ttl_seconds=max(60.0, min(86400.0, float(self.budget_ttl_seconds))),
            cache_ttl_seconds=max(0.0, min(86400.0, float(self.cache_ttl_seconds))),
        )


def build_visual_task_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return f"visual_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def get_visual_task_status(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        record = _TASK_STATUS.get(str(task_id or ""))
        return dict(record) if isinstance(record, dict) else {}


def visual_budget_status(document_id: str, parse_generation: str = "") -> dict[str, Any]:
    scope = _budget_scope(document_id, parse_generation)
    with _TASK_LOCK:
        _prune_ledgers_locked(time.time())
        record = _BUDGET_LEDGER.get(scope)
        return dict(record) if isinstance(record, dict) else {
            "scope": scope,
            "used": 0,
        }


async def execute_visual_task(
    *,
    task_id: str,
    document_id: str,
    parse_generation: str,
    purpose: str,
    operation: Callable[[], Awaitable[T]],
    policy: VisualTaskPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> T:
    """Execute one visual operation under the shared operational policy."""
    resolved = (policy or VisualTaskPolicy()).normalized()
    safe_task_id = str(task_id or "").strip() or build_visual_task_id({
        "document_id": document_id,
        "parse_generation": parse_generation,
        "purpose": purpose,
        "created_at": time.time_ns(),
    })
    base = {
        "document_id": str(document_id or ""),
        "parse_generation": str(parse_generation or ""),
        "purpose": str(purpose or "visual_enrichment"),
        "metadata": _safe_metadata(metadata),
    }
    cached_result = _get_cached_result(safe_task_id, resolved.cache_ttl_seconds)
    if cached_result is not _CACHE_MISS:
        _set_task_status(
            safe_task_id,
            state="succeeded",
            attempts=0,
            cache_hit=True,
            completed_at=time.time(),
            **base,
        )
        return cached_result

    _set_task_status(
        safe_task_id,
        state="queued",
        attempts=0,
        cache_hit=False,
        **base,
    )

    semaphore = _get_semaphore(resolved.concurrency)
    async with semaphore:
        last_error: BaseException | None = None
        for attempt in range(resolved.max_retries + 1):
            if not _reserve_budget(
                document_id=document_id,
                parse_generation=parse_generation,
                limit=resolved.document_budget,
                units=resolved.budget_units,
                ttl_seconds=resolved.budget_ttl_seconds,
            ):
                error = VisualTaskBudgetExceeded(
                    f"visual budget exhausted for document generation ({resolved.document_budget})"
                )
                _set_task_status(
                    safe_task_id,
                    state="budget_exhausted",
                    attempts=attempt,
                    error=str(error),
                    completed_at=time.time(),
                    **base,
                )
                raise error

            _set_task_status(
                safe_task_id,
                state="running",
                attempts=attempt + 1,
                started_at=time.time(),
                **base,
            )
            try:
                result = await asyncio.wait_for(operation(), timeout=resolved.timeout_seconds)
            except asyncio.CancelledError:
                _set_task_status(
                    safe_task_id,
                    state="cancelled",
                    attempts=attempt + 1,
                    completed_at=time.time(),
                    **base,
                )
                raise
            except asyncio.TimeoutError as exc:
                last_error = VisualTaskTimeoutError(
                    f"visual task timed out after {resolved.timeout_seconds:.0f}s"
                )
                if attempt >= resolved.max_retries:
                    _set_task_status(
                        safe_task_id,
                        state="timed_out",
                        attempts=attempt + 1,
                        error=str(last_error),
                        completed_at=time.time(),
                        **base,
                    )
                    raise last_error from exc
            except Exception as exc:
                last_error = exc
                if attempt >= resolved.max_retries:
                    _set_task_status(
                        safe_task_id,
                        state="failed",
                        attempts=attempt + 1,
                        error=f"{type(exc).__name__}: {exc}",
                        completed_at=time.time(),
                        **base,
                    )
                    raise
            else:
                response_error = _visual_response_error(result)
                if response_error:
                    last_error = VisualTaskUpstreamError(response_error)
                    if attempt >= resolved.max_retries:
                        _set_task_status(
                            safe_task_id,
                            state="failed",
                            attempts=attempt + 1,
                            error=str(last_error),
                            completed_at=time.time(),
                            **base,
                        )
                        raise last_error
                else:
                    _set_task_status(
                        safe_task_id,
                        state="succeeded",
                        attempts=attempt + 1,
                        completed_at=time.time(),
                        **base,
                    )
                    _cache_result(
                        safe_task_id,
                        result,
                        ttl_seconds=resolved.cache_ttl_seconds,
                    )
                    return result

            if resolved.retry_delay_seconds:
                await asyncio.sleep(resolved.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(str(last_error or "visual task failed"))


def _visual_response_error(result: Any) -> str:
    """识别 AI 网关常用的 ``{"error": ...}`` 失败返回。"""
    if not isinstance(result, dict) or not result.get("error"):
        return ""
    error = result.get("error")
    if isinstance(error, (dict, list)):
        return json.dumps(error, ensure_ascii=False, sort_keys=True, default=str)
    return str(error)


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    allowed = {
        "provider",
        "model",
        "source",
        "page",
        "bbox_hash",
        "route",
        "document_source_hash",
        "prompt_version",
    }
    return {key: metadata.get(key) for key in allowed if metadata.get(key) not in (None, "")}


def _get_cached_result(task_id: str, ttl_seconds: float) -> Any:
    if ttl_seconds <= 0:
        return _CACHE_MISS
    now = time.time()
    with _TASK_LOCK:
        record = _TASK_RESULTS.get(task_id)
        if not isinstance(record, dict):
            return _CACHE_MISS
        if float(record.get("expires_at") or 0) <= now:
            _TASK_RESULTS.pop(task_id, None)
            return _CACHE_MISS
        _TASK_RESULTS.move_to_end(task_id)
        try:
            return copy.deepcopy(record.get("result"))
        except Exception:
            _TASK_RESULTS.pop(task_id, None)
            return _CACHE_MISS


def _cache_result(task_id: str, result: Any, *, ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    try:
        cached = copy.deepcopy(result)
    except Exception:
        return
    now = time.time()
    with _TASK_LOCK:
        expired = [
            key
            for key, record in _TASK_RESULTS.items()
            if float(record.get("expires_at") or 0) <= now
        ]
        for key in expired:
            _TASK_RESULTS.pop(key, None)
        _TASK_RESULTS[task_id] = {
            "result": cached,
            "created_at": now,
            "expires_at": now + ttl_seconds,
        }
        _TASK_RESULTS.move_to_end(task_id)
        while len(_TASK_RESULTS) > _MAX_RESULT_RECORDS:
            _TASK_RESULTS.popitem(last=False)


def _set_task_status(task_id: str, *, state: str, attempts: int, **fields: Any) -> None:
    now = time.time()
    with _TASK_LOCK:
        previous = _TASK_STATUS.get(task_id) or {}
        record = {
            "created_at": previous.get("created_at", now),
            **previous,
            **fields,
            "task_id": task_id,
            "state": state,
            "attempts": int(attempts),
            "updated_at": now,
        }
        if state == "succeeded":
            record.pop("error", None)
        _TASK_STATUS[task_id] = record
        _TASK_STATUS.move_to_end(task_id)
        while len(_TASK_STATUS) > _MAX_TASK_RECORDS:
            _TASK_STATUS.popitem(last=False)


def _get_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = (id(loop), int(limit))
    semaphore = _SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _SEMAPHORES[key] = semaphore
    return semaphore


def _budget_scope(document_id: str, parse_generation: str) -> str:
    return f"{str(document_id or 'unknown')}:{str(parse_generation or 'legacy')}"


def _reserve_budget(
    *,
    document_id: str,
    parse_generation: str,
    limit: int,
    units: int,
    ttl_seconds: float,
) -> bool:
    if limit <= 0:
        return False
    now = time.time()
    scope = _budget_scope(document_id, parse_generation)
    with _TASK_LOCK:
        _prune_ledgers_locked(now)
        current = _BUDGET_LEDGER.get(scope) or {
            "scope": scope,
            "used": 0,
            "created_at": now,
        }
        if int(current.get("used") or 0) + units > limit:
            current.update({"limit": limit, "updated_at": now, "expires_at": now + ttl_seconds})
            _BUDGET_LEDGER[scope] = current
            _BUDGET_LEDGER.move_to_end(scope)
            return False
        current.update({
            "used": int(current.get("used") or 0) + units,
            "limit": limit,
            "updated_at": now,
            "expires_at": now + ttl_seconds,
        })
        _BUDGET_LEDGER[scope] = current
        _BUDGET_LEDGER.move_to_end(scope)
        while len(_BUDGET_LEDGER) > _MAX_BUDGET_SCOPES:
            _BUDGET_LEDGER.popitem(last=False)
        return True


def _prune_ledgers_locked(now: float) -> None:
    expired = [
        scope for scope, record in _BUDGET_LEDGER.items()
        if float(record.get("expires_at") or 0) <= now
    ]
    for scope in expired:
        _BUDGET_LEDGER.pop(scope, None)
