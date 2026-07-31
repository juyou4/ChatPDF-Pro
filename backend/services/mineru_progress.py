"""Normalize MinerU task progress without inventing unsupported remote detail.

MinerU's public API normally exposes lifecycle states instead of a page-level
percentage.  The helpers below preserve a percentage when a Worker does expose
one, and otherwise provide a bounded estimate for the client UI.  Consumers can
therefore distinguish an authoritative remote value from an activity estimate.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping


_PERCENT_KEYS = (
    "progress_percent",
    "progress_percentage",
    "percentage",
    "percent",
    "progress",
)
_PAGE_PROGRESS_PAIRS = (
    ("completed_pages", "total_pages"),
    ("processed_pages", "total_pages"),
    ("parsed_pages", "total_pages"),
    ("finished_pages", "total_pages"),
)
_STAGE_ESTIMATES = {
    "queued": 2,
    "waiting_for_slot": 4,
    "waiting_for_document_lock": 7,
    "requesting_upload": 10,
    "uploading": 18,
    "resuming": 20,
    "resuming_result_download": 80,
    "downloading": 82,
    "retrying_download": 82,
    "building_index": 84,
    "preparing_rag_index": 88,
    "building_rag_index": 89,
    "rebuilding_rag_index": 89,
    "building_vector_index": 89,
    "validating_vector_index": 93,
    "preparing_semantic_index": 94,
    "building_semantic_index": 95,
    "validating_semantic_index": 97,
    "publishing_rag_index": 98,
    "awaiting_rag_index": 96,
}

# MinerU does not expose a trustworthy end-to-end percentage for local index
# construction. These windows move only inside the currently observed stage;
# actual stage transitions remain the authoritative progress signal.
_STAGE_ESTIMATE_WINDOWS = {
    "building_index": (84, 87, 18.0),
    "building_rag_index": (89, 92, 35.0),
    "rebuilding_rag_index": (89, 92, 35.0),
    "building_vector_index": (89, 92, 35.0),
    "building_semantic_index": (95, 96, 30.0),
    "publishing_rag_index": (98, 99, 12.0),
}


def _as_percentage(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        has_percent_suffix = raw.endswith("%")
        if has_percent_suffix:
            raw = raw[:-1].strip()
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
        if not has_percent_suffix and 0 <= number <= 1:
            number *= 100
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if 0 <= number <= 1:
            number *= 100
    if not math.isfinite(number) or number < 0 or number > 100:
        return None
    return number


def _as_positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def extract_remote_mineru_progress(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract optional remote progress fields from official or Worker payloads.

    The official response currently only guarantees ``state``.  This accepts a
    few common Worker extensions so newer proxies can surface real progress
    without requiring another API-contract migration.
    """
    if not isinstance(payload, Mapping):
        return {}

    candidates: list[Mapping[str, Any]] = [payload]
    for key in ("data", "progress"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)

    for candidate in candidates:
        for key in _PERCENT_KEYS:
            percent = _as_percentage(candidate.get(key))
            if percent is not None:
                return {
                    "remote_progress_percent": round(percent, 1),
                    "remote_progress_source": "remote_percent",
                }

    for candidate in candidates:
        for completed_key, total_key in _PAGE_PROGRESS_PAIRS:
            completed = _as_positive_number(candidate.get(completed_key))
            total = _as_positive_number(candidate.get(total_key))
            if completed is None or total is None or total <= 0:
                continue
            percent = min(100.0, max(0.0, completed / total * 100))
            return {
                "remote_progress_percent": round(percent, 1),
                "remote_progress_source": "remote_pages",
                "remote_pages_completed": int(completed),
                "remote_pages_total": int(total),
            }
    return {}


def _elapsed_seconds(value: Any, *, now: datetime | None = None) -> int | None:
    if not value:
        return None
    try:
        started = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    current = now
    if current is None:
        current = datetime.now(tz=started.tzinfo) if started.tzinfo else datetime.now()
    elif started.tzinfo and current.tzinfo is None:
        current = current.replace(tzinfo=started.tzinfo)
    elif not started.tzinfo and current.tzinfo:
        current = current.replace(tzinfo=None)
    return max(0, int((current - started).total_seconds()))


def _estimated_stage_percent(
    stage: str,
    source: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> int:
    window = _STAGE_ESTIMATE_WINDOWS.get(stage)
    if not window:
        return int(_STAGE_ESTIMATES.get(stage, 22))

    floor, ceiling, time_constant = window
    stage_elapsed = _elapsed_seconds(
        source.get("stage_started_at") or source.get("updated_at"),
        now=now,
    )
    if stage_elapsed is None or stage_elapsed <= 0:
        return floor
    estimate = floor + (ceiling - floor) * (1 - math.exp(-stage_elapsed / time_constant))
    return min(ceiling, max(floor, round(estimate)))


def derive_mineru_progress(task: Mapping[str, Any] | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Return a client-safe progress contract for a persisted MinerU task."""
    source = task if isinstance(task, Mapping) else {}
    status = str(source.get("status") or "").strip().lower()
    stage = str(source.get("stage") or "").strip().lower()
    elapsed = _elapsed_seconds(source.get("started_at") or source.get("created_at"), now=now)

    if status in {"ready", "partial_ready"}:
        return {
            "percent": 100,
            "estimated": False,
            "source": "completed",
            "stage": stage,
            "elapsed_seconds": elapsed,
        }
    if status in {"failed", "cancelled"}:
        return {
            "percent": None,
            "estimated": False,
            "source": "terminal",
            "stage": stage,
            "elapsed_seconds": elapsed,
        }

    remote_percent = _as_percentage(source.get("remote_progress_percent"))
    if remote_percent is not None and stage == "polling":
        # Upload/download/indexing account for the remaining 24% of the task.
        percent = round(24 + remote_percent * 0.54)
        return {
            "percent": min(78, max(24, percent)),
            "remote_percent": round(remote_percent),
            "estimated": False,
            "source": str(source.get("remote_progress_source") or "remote_percent"),
            "stage": stage,
            "elapsed_seconds": elapsed,
        }

    if stage == "polling":
        try:
            attempt = max(0, int(source.get("poll_attempt") or 0))
        except (TypeError, ValueError):
            attempt = 0
        # The estimate moves early to acknowledge liveness, then asymptotically
        # approaches the hand-off point instead of claiming remote completion.
        percent = round(25 + 50 * (1 - math.exp(-attempt / 23)))
    else:
        percent = (
            _estimated_stage_percent(stage, source, now=now)
            if stage in _STAGE_ESTIMATE_WINDOWS
            else _STAGE_ESTIMATES.get(stage, 22 if status == "running" else 2)
        )

    return {
        "percent": min(99, max(0, int(percent))),
        "estimated": True,
        "source": "estimated",
        "stage": stage,
        "elapsed_seconds": elapsed,
    }
