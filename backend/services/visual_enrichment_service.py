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

from services.completion_outcome import resolve_completion_outcome


T = TypeVar("T")
_MAX_TASK_RECORDS = 1024
_MAX_BUDGET_SCOPES = 512
_MAX_RESULT_RECORDS = 256
_DEFAULT_BUDGET_TTL_S = 6 * 60 * 60
_TASK_LOCK = threading.Lock()
_TASK_STATUS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_BUDGET_LEDGER: OrderedDict[str, dict[str, Any]] = OrderedDict()
_TASK_RESULTS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_DOCUMENT_EPOCHS: dict[str, int] = {}
_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_CACHE_MISS = object()


class VisualTaskTimeoutError(TimeoutError):
    """A visual task exhausted its bounded retry window."""


class VisualTaskBudgetExceeded(RuntimeError):
    """A document generation exhausted its visual-request budget."""


class VisualTaskUpstreamError(RuntimeError):
    """视觉模型网关以结构化 error 响应报告的可重试错误。"""


class VisualTaskInvalidResponseError(VisualTaskUpstreamError):
    """视觉模型返回空内容或无法消费的结构化响应。"""

    def __init__(self, code: str, message: str):
        self.code = str(code or "invalid_visual_response")
        super().__init__(f"{self.code}: {str(message or 'invalid visual response')}")


class VisualTaskInvalidatedError(RuntimeError):
    """A document-scoped visual task was invalidated by an explicit reset."""

    def __init__(self, document_id: str):
        self.code = "visual_task_invalidated"
        self.document_id = str(document_id or "")
        super().__init__("visual task invalidated by document state reset")


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


def reset_visual_document_state(document_id: str) -> dict[str, Any]:
    safe_document_id = str(document_id or "").strip()
    if not safe_document_id:
        return {
            "document_id": "",
            "document_epoch": 0,
            "status_records_cleared": 0,
            "budget_records_cleared": 0,
            "result_records_cleared": 0,
        }

    with _TASK_LOCK:
        next_epoch = int(_DOCUMENT_EPOCHS.get(safe_document_id, 0)) + 1
        _DOCUMENT_EPOCHS[safe_document_id] = next_epoch

        status_keys = [
            key for key, record in _TASK_STATUS.items()
            if isinstance(record, dict) and str(record.get("document_id") or "") == safe_document_id
        ]
        for key in status_keys:
            _TASK_STATUS.pop(key, None)

        result_keys = [
            key for key, record in _TASK_RESULTS.items()
            if isinstance(record, dict) and str(record.get("document_id") or "") == safe_document_id
        ]
        for key in result_keys:
            _TASK_RESULTS.pop(key, None)

        scope_prefix = f"{safe_document_id}:"
        budget_keys = [scope for scope in _BUDGET_LEDGER if scope.startswith(scope_prefix)]
        for scope in budget_keys:
            _BUDGET_LEDGER.pop(scope, None)

    return {
        "document_id": safe_document_id,
        "document_epoch": next_epoch,
        "status_records_cleared": len(status_keys),
        "budget_records_cleared": len(budget_keys),
        "result_records_cleared": len(result_keys),
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
    document_epoch = _document_epoch(document_id)
    safe_task_id = str(task_id or "").strip() or build_visual_task_id({
        "document_id": document_id,
        "parse_generation": parse_generation,
        "purpose": purpose,
        "created_at": time.time_ns(),
    })
    base = {
        "parse_generation": str(parse_generation or ""),
        "purpose": str(purpose or "visual_enrichment"),
        "metadata": _safe_metadata(metadata),
    }
    cached_result = _get_cached_result(
        safe_task_id,
        resolved.cache_ttl_seconds,
        document_id=document_id,
        document_epoch=document_epoch,
    )
    rejected_cache_failure: dict[str, str] | None = None
    if cached_result is not _CACHE_MISS:
        rejected_cache_failure = _visual_response_failure(cached_result)
        if rejected_cache_failure is None:
            _set_task_status(
                safe_task_id,
                state="succeeded",
                attempts=0,
                cache_hit=True,
                completed_at=time.time(),
                document_id=document_id,
                document_epoch=document_epoch,
                **base,
            )
            _raise_if_document_invalidated(document_id, document_epoch)
            return cached_result
        _discard_cached_result(safe_task_id)

    _set_task_status(
        safe_task_id,
        state="queued",
        attempts=0,
        cache_hit=False,
        cache_rejected_code=(rejected_cache_failure or {}).get("code", ""),
        document_id=document_id,
        document_epoch=document_epoch,
        **base,
    )

    semaphore = _get_semaphore(resolved.concurrency)
    async with semaphore:
        last_error: BaseException | None = None
        for attempt in range(resolved.max_retries + 1):
            _raise_if_document_invalidated(document_id, document_epoch)
            budget_reserved = _reserve_budget(
                document_id=document_id,
                parse_generation=parse_generation,
                limit=resolved.document_budget,
                units=resolved.budget_units,
                ttl_seconds=resolved.budget_ttl_seconds,
                document_epoch=document_epoch,
            )
            if budget_reserved is None:
                raise VisualTaskInvalidatedError(document_id)
            if not budget_reserved:
                error = VisualTaskBudgetExceeded(
                    f"visual budget exhausted for document generation ({resolved.document_budget})"
                )
                _set_task_status(
                    safe_task_id,
                    state="budget_exhausted",
                    attempts=attempt,
                    error=str(error),
                    error_code="visual_budget_exhausted",
                    failure={
                        "kind": "budget",
                        "code": "visual_budget_exhausted",
                        "retryable": True,
                    },
                    completed_at=time.time(),
                    document_id=document_id,
                    document_epoch=document_epoch,
                    **base,
                )
                raise error

            _set_task_status(
                safe_task_id,
                state="running",
                attempts=attempt + 1,
                started_at=time.time(),
                document_id=document_id,
                document_epoch=document_epoch,
                **base,
            )
            try:
                result = await asyncio.wait_for(operation(), timeout=resolved.timeout_seconds)
            except asyncio.CancelledError:
                _raise_if_document_invalidated(document_id, document_epoch)
                _set_task_status(
                    safe_task_id,
                    state="cancelled",
                    attempts=attempt + 1,
                    error="visual task cancelled",
                    error_code="visual_task_cancelled",
                    failure={
                        "kind": "cancelled",
                        "code": "visual_task_cancelled",
                        "retryable": True,
                    },
                    completed_at=time.time(),
                    document_id=document_id,
                    document_epoch=document_epoch,
                    **base,
                )
                raise
            except asyncio.TimeoutError as exc:
                last_error = VisualTaskTimeoutError(
                    f"visual task timed out after {resolved.timeout_seconds:.0f}s"
                )
                if attempt >= resolved.max_retries:
                    _raise_if_document_invalidated(document_id, document_epoch)
                    _set_task_status(
                        safe_task_id,
                        state="timed_out",
                        attempts=attempt + 1,
                        error=str(last_error),
                        error_code="visual_timeout",
                        failure={
                            "kind": "timeout",
                            "code": "visual_timeout",
                            "retryable": True,
                        },
                        completed_at=time.time(),
                        document_id=document_id,
                        document_epoch=document_epoch,
                        **base,
                    )
                    raise last_error from exc
            except Exception as exc:
                last_error = exc
                if attempt >= resolved.max_retries:
                    error_code = str(getattr(exc, "code", "") or "visual_operation_failed")
                    _raise_if_document_invalidated(document_id, document_epoch)
                    _set_task_status(
                        safe_task_id,
                        state="failed",
                        attempts=attempt + 1,
                        error=f"{type(exc).__name__}: {exc}",
                        error_code=error_code,
                        failure={
                            "kind": "operation",
                            "code": error_code,
                            "retryable": True,
                        },
                        completed_at=time.time(),
                        document_id=document_id,
                        document_epoch=document_epoch,
                        **base,
                    )
                    raise
            else:
                _raise_if_document_invalidated(document_id, document_epoch)
                response_failure = _visual_response_failure(result)
                if response_failure:
                    if response_failure["code"] == "visual_upstream_error":
                        last_error = VisualTaskUpstreamError(response_failure["message"])
                        failure_kind = "upstream_error"
                    else:
                        last_error = VisualTaskInvalidResponseError(
                            response_failure["code"],
                            response_failure["message"],
                        )
                        failure_kind = "invalid_response"
                    if attempt >= resolved.max_retries:
                        _set_task_status(
                            safe_task_id,
                            state="failed",
                            attempts=attempt + 1,
                            error=str(last_error),
                            error_code=response_failure["code"],
                            failure={
                                "kind": failure_kind,
                                "code": response_failure["code"],
                            "retryable": True,
                        },
                        completed_at=time.time(),
                        document_id=document_id,
                        document_epoch=document_epoch,
                        **base,
                    )
                    raise last_error
                else:
                    _set_task_status(
                        safe_task_id,
                        state="succeeded",
                        attempts=attempt + 1,
                        completed_at=time.time(),
                        document_id=document_id,
                        document_epoch=document_epoch,
                        **base,
                    )
                    _cache_result(
                        safe_task_id,
                        result,
                        ttl_seconds=resolved.cache_ttl_seconds,
                        document_id=document_id,
                        document_epoch=document_epoch,
                    )
                    _raise_if_document_invalidated(document_id, document_epoch)
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


def _visual_response_failure(result: Any) -> dict[str, str] | None:
    """校验通用视觉任务结果，拒绝无法被下游稳定消费的伪成功。"""
    if result is None:
        return {
            "code": "empty_visual_response",
            "message": "visual model returned no response",
        }

    response_error = _visual_response_error(result)
    if response_error:
        return {
            "code": "visual_upstream_error",
            "message": response_error,
        }

    if isinstance(result, dict):
        if "choices" in result or any(
            key in result
            for key in ("finish_reason", "stop_reason", "finishReason", "done_reason")
        ):
            outcome = resolve_completion_outcome(result)
            if not outcome.publishable:
                return {
                    "code": (
                        "visual_output_truncated"
                        if outcome.truncated
                        else "visual_output_incomplete"
                    ),
                    "message": (
                        "visual model output is incomplete "
                        f"(finish_reason={outcome.finish_reason or outcome.status.value})"
                    ),
                }
        if not result:
            return {
                "code": "invalid_visual_schema",
                "message": "visual model returned an empty object",
            }

        # OpenAI-compatible raw responses must contain a consumable first choice.
        if "choices" in result:
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices:
                return {
                    "code": "empty_visual_choices",
                    "message": "visual model returned no choices",
                }
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return {
                    "code": "invalid_visual_schema",
                    "message": "visual model returned an invalid first choice",
                }
            message = first_choice.get("message")
            payload: Any = None
            if isinstance(message, dict):
                payload = message.get("parsed")
                if payload in (None, "", [], {}):
                    payload = message.get("content")
            if payload in (None, "", [], {}):
                payload = first_choice.get("text")
            return _visual_schema_failure(payload)

        # Some gateways expose a direct content field instead of choices.
        if "content" in result or "output_text" in result:
            payload = result.get("content")
            if payload in (None, "", [], {}):
                payload = result.get("output_text")
            return _visual_schema_failure(payload)

        raw_metadata_keys = {
            "id",
            "object",
            "created",
            "created_at",
            "model",
            "usage",
            "_used_provider",
            "_used_model",
        }
        if set(result).issubset(raw_metadata_keys):
            return {
                "code": "empty_visual_content",
                "message": "visual model response contained metadata but no content",
            }

        # Operations may normalize and validate the model response before returning.
        # Preserve that existing contract for non-empty business dictionaries.
        return None

    if isinstance(result, (str, bytes, bytearray)):
        return _visual_schema_failure(result)
    if isinstance(result, (list, tuple)):
        if result:
            return None
        return {
            "code": "invalid_visual_schema",
            "message": "visual task returned an empty collection",
        }
    return {
        "code": "invalid_visual_schema",
        "message": f"visual task returned unsupported type {type(result).__name__}",
    }


def _visual_schema_failure(payload: Any) -> dict[str, str] | None:
    """确认模型正文能形成非空 JSON 对象，不记录原始正文以免泄露内容。"""
    if isinstance(payload, dict):
        if payload:
            return None
        return {
            "code": "invalid_visual_schema",
            "message": "visual model returned an empty schema",
        }

    if isinstance(payload, (list, tuple)):
        text_parts: list[str] = []
        for part in payload:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("text") not in (None, ""):
                text_parts.append(str(part.get("text")))
        payload = "\n".join(text_parts)

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError:
            payload = ""
    text = str(payload or "").strip()
    if not text:
        return {
            "code": "empty_visual_content",
            "message": "visual model returned empty content",
        }

    candidates = [text]
    object_start = text.find("{")
    object_end = text.rfind("}") + 1
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start:object_end])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value:
            return None
    return {
        "code": "invalid_visual_schema",
        "message": "visual model content did not contain a non-empty JSON object",
    }


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


def _get_cached_result(
    task_id: str,
    ttl_seconds: float,
    *,
    document_id: str = "",
    document_epoch: int | None = None,
) -> Any:
    if ttl_seconds <= 0:
        return _CACHE_MISS
    now = time.time()
    with _TASK_LOCK:
        record = _TASK_RESULTS.get(task_id)
        if not isinstance(record, dict):
            return _CACHE_MISS
        if document_id and str(record.get("document_id") or "") != str(document_id or ""):
            _TASK_RESULTS.pop(task_id, None)
            return _CACHE_MISS
        if document_epoch is not None and not _document_epoch_matches_locked(document_id, document_epoch):
            _TASK_RESULTS.pop(task_id, None)
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


def _discard_cached_result(task_id: str) -> None:
    with _TASK_LOCK:
        _TASK_RESULTS.pop(task_id, None)


def _cache_result(
    task_id: str,
    result: Any,
    *,
    ttl_seconds: float,
    document_id: str = "",
    document_epoch: int | None = None,
) -> None:
    if ttl_seconds <= 0 or _visual_response_failure(result) is not None:
        return
    try:
        cached = copy.deepcopy(result)
    except Exception:
        return
    now = time.time()
    with _TASK_LOCK:
        if document_epoch is not None and not _document_epoch_matches_locked(document_id, document_epoch):
            return
        expired = [
            key
            for key, record in _TASK_RESULTS.items()
            if float(record.get("expires_at") or 0) <= now
        ]
        for key in expired:
            _TASK_RESULTS.pop(key, None)
        _TASK_RESULTS[task_id] = {
            "result": cached,
            "document_id": str(document_id or ""),
            "document_epoch": int(document_epoch or 0),
            "created_at": now,
            "expires_at": now + ttl_seconds,
        }
        _TASK_RESULTS.move_to_end(task_id)
        while len(_TASK_RESULTS) > _MAX_RESULT_RECORDS:
            _TASK_RESULTS.popitem(last=False)


def _set_task_status(
    task_id: str,
    *,
    state: str,
    attempts: int,
    document_id: str = "",
    document_epoch: int | None = None,
    **fields: Any,
) -> None:
    now = time.time()
    with _TASK_LOCK:
        if document_epoch is not None and not _document_epoch_matches_locked(document_id, document_epoch):
            return
        extra_fields = dict(fields)
        extra_fields["document_id"] = str(document_id or "")
        extra_fields["document_epoch"] = int(document_epoch or 0)
        previous = _TASK_STATUS.get(task_id) or {}
        record = {
            "created_at": previous.get("created_at", now),
            **previous,
            **extra_fields,
            "task_id": task_id,
            "state": state,
            "attempts": int(attempts),
            "updated_at": now,
        }
        if state in {"queued", "running", "succeeded"}:
            record.pop("error", None)
            record.pop("error_code", None)
            record.pop("failure", None)
        if state in {"queued", "running"}:
            record.pop("completed_at", None)
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
    document_epoch: int | None = None,
) -> bool | None:
    if limit <= 0:
        return False
    now = time.time()
    scope = _budget_scope(document_id, parse_generation)
    with _TASK_LOCK:
        if document_epoch is not None and not _document_epoch_matches_locked(document_id, document_epoch):
            return None
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


def _document_epoch(document_id: str) -> int:
    safe_document_id = str(document_id or "").strip()
    if not safe_document_id:
        return 0
    with _TASK_LOCK:
        return int(_DOCUMENT_EPOCHS.get(safe_document_id, 0))


def _document_epoch_matches_locked(document_id: str, expected_epoch: int | None) -> bool:
    if expected_epoch is None:
        return True
    safe_document_id = str(document_id or "").strip()
    if not safe_document_id:
        return expected_epoch in (None, 0)
    return int(_DOCUMENT_EPOCHS.get(safe_document_id, 0)) == int(expected_epoch)


def _raise_if_document_invalidated(document_id: str, expected_epoch: int | None) -> None:
    if expected_epoch is None:
        return
    with _TASK_LOCK:
        if _document_epoch_matches_locked(document_id, expected_epoch):
            return
    raise VisualTaskInvalidatedError(document_id)
