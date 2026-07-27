"""Bounded helpers for context fallbacks that still cover a full document."""

from __future__ import annotations


def sample_document_text(
    text: str,
    *,
    max_chars: int,
    max_segments: int = 8,
) -> str:
    """Return a bounded sample distributed from the start through the tail.

    This is only for fallback paths without reliable page records.  It keeps
    the configured context budget while avoiding the misleading behavior of
    treating a document prefix as a whole-document summary.
    """
    source = str(text or "")
    budget = max(0, int(max_chars or 0))
    if not source or budget <= 0:
        return ""
    if len(source) <= budget:
        return source

    marker = "\n...[document sampled]...\n"
    segment_count = min(max(2, int(max_segments or 2)), max(2, budget // 512))
    separator_chars = len(marker) * (segment_count - 1)
    available = budget - separator_chars
    if available < segment_count:
        # A very small caller budget cannot carry stable sampling markers.
        return source[:budget]

    segment_size = max(1, available // segment_count)
    max_start = max(0, len(source) - segment_size)
    starts = [
        round(max_start * index / (segment_count - 1))
        for index in range(segment_count)
    ]
    sampled = marker.join(source[start:start + segment_size] for start in starts)
    return sampled[:budget]
