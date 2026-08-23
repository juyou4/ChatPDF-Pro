"""从 content 流里拆出思考标签。

部分 OpenAI 兼容网关不返回 reasoning_content，而是把思考写进
`<think>...</think>` 这类标签。流式时标签常被拆成 `<thi` / `nk>`，
必须先缓冲半截标记，不能把 `<` 或 `F` 一类前缀当成正文。
"""
from __future__ import annotations

from dataclasses import dataclass

# 长标签优先，避免 `<think>` 抢先吃掉 `<thinking>`。
_THINK_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<seed:think>", "</seed:think>"),
    ("###Thinking", "###Response"),
    ("<thinking>", "</thinking>"),
    ("<thought>", "</thought>"),
    ("◁think▷", "◁/think▷"),
    ("<think>", "</think>"),
)
_OPEN_TAGS = tuple(sorted((pair[0] for pair in _THINK_TAG_PAIRS), key=len, reverse=True))
_CLOSE_BY_OPEN = {open_tag: close_tag for open_tag, close_tag in _THINK_TAG_PAIRS}


@dataclass(frozen=True)
class ThinkSplitDelta:
    reasoning: str = ""
    content: str = ""


def _potential_start_index(text: str, tag: str) -> int | None:
    if not text or not tag:
        return None
    direct = text.find(tag)
    if direct != -1:
        return direct
    for index in range(len(text) - 1, -1, -1):
        suffix = text[index:]
        if tag.startswith(suffix):
            return index
    return None


def _earliest_potential_start(text: str, tags: tuple[str, ...]) -> int | None:
    earliest = None
    for tag in tags:
        index = _potential_start_index(text, tag)
        if index is not None:
            earliest = index if earliest is None else min(earliest, index)
    return earliest


def _full_tag_at(text: str, start: int, tags: tuple[str, ...]) -> str | None:
    rest = text[start:]
    for tag in tags:
        if rest.startswith(tag):
            return tag
    return None


class StreamingThinkSplitter:
    """跨 chunk 累积，把思考标签内外的文本拆成 reasoning / content。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._close_tag = ""

    def feed(self, text: str) -> ThinkSplitDelta:
        if not text:
            return ThinkSplitDelta()
        self._buffer += text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while self._buffer:
            tags = (self._close_tag,) if self._inside and self._close_tag else _OPEN_TAGS
            start = _earliest_potential_start(self._buffer, tags)
            if start is None:
                chunk = self._buffer
                self._buffer = ""
                if self._inside:
                    reasoning_parts.append(chunk)
                else:
                    content_parts.append(chunk)
                break

            before = self._buffer[:start]
            if before:
                if self._inside:
                    reasoning_parts.append(before)
                else:
                    content_parts.append(before)

            matched = _full_tag_at(self._buffer, start, tags)
            if matched is None:
                self._buffer = self._buffer[start:]
                break

            self._buffer = self._buffer[start + len(matched):]
            if self._inside:
                self._inside = False
                self._close_tag = ""
            else:
                self._inside = True
                self._close_tag = _CLOSE_BY_OPEN.get(matched, "")
        return ThinkSplitDelta("".join(reasoning_parts), "".join(content_parts))

    def flush(self) -> ThinkSplitDelta:
        leftover = self._buffer
        inside = self._inside
        self._buffer = ""
        self._inside = False
        self._close_tag = ""
        if not leftover:
            return ThinkSplitDelta()
        if inside:
            return ThinkSplitDelta(reasoning=leftover)
        return ThinkSplitDelta(content=leftover)


def apply_stream_think_split(
    splitter: StreamingThinkSplitter,
    content: str,
    reasoning: str,
) -> tuple[str, str]:
    """把本 chunk 的 content 拆进思考/正文，并拼到已有 reasoning 上。"""
    if not isinstance(content, str) or not content:
        return (content if isinstance(content, str) else ""), reasoning or ""
    delta = splitter.feed(content)
    return delta.content, f"{reasoning or ''}{delta.reasoning}"


def split_think_tags(text: str) -> tuple[str, str]:
    """一次性拆分。返回 (reasoning, visible_content)。"""
    splitter = StreamingThinkSplitter()
    delta = splitter.feed(text or "")
    flushed = splitter.flush()
    return (
        f"{delta.reasoning}{flushed.reasoning}",
        f"{delta.content}{flushed.content}",
    )
