"""Shared request semantics for the complete-document summary projection.

The reading outline has two legitimate chat projections:

* a thematic synthesis for a normal "summarize the paper" request;
* a structural, chapter-by-chapter view only when the reader explicitly asks
  for every chapter or subsection.

Keeping this recognition in one small dependency-free module is important:
the intent router decides whether to enter the parse-bound full-document
pipeline, while the renderer decides whether that pipeline should reveal the
structural appendix.  Two drifting regexes previously made the renderer
capable of a chapter view that the chat route could never reach.
"""
from __future__ import annotations

import re
from typing import Any


# Chinese requests deliberately require a summary/explanation verb after the
# chapter phrase.  A navigation request such as "按章节跳转" must remain a
# normal scoped interaction rather than being mistaken for a complete summary.
_ZH_SECTION_DETAIL_SUMMARY_RE = re.compile(
    r"(?:按|分|逐|每|各)\s*(?:个|一)?\s*(?:章节?|小节)\s*"
    r"(?:梳理|总结|概览|详解|讲解|介绍|说明|回顾|阅读)|"
    r"(?:章节?|小节)\s*(?:梳理|总结|概览|详解|讲解|介绍|说明|回顾)",
    re.IGNORECASE,
)

# The English alternatives distinguish a document-wide instruction from a
# local request such as "summarize section 3" or "the methods section":
# a singular section needs each/every/all; otherwise the target must be plural
# or use the explicit chapter-by-chapter / section-wise construction.
_EN_SECTION_DETAIL_SUMMARY_RE = re.compile(
    r"(?:summari[sz]e|outline|review|go\s+through|walk\s+(?:me\s+)?through|break\s+down)\s+"
    r"(?:"
    r"(?:each|every|all)\s+(?:of\s+the\s+)?(?:sections?|chapters?|subsections?)\b|"
    r"(?:of\s+the\s+|the\s+)?(?:sections|chapters|subsections)\b"
    r")|"
    r"(?:each|every|all)\s+(?:section|chapter|subsection)\b|"
    r"(?:section|chapter|subsection)\s*(?:-|–|—|\s)+by\s*(?:-|–|—|\s)+"
    r"(?:section|chapter|subsection)\b|"
    r"(?:section|chapter|subsection)(?:\s*[-–—]\s*|\s*)wise"
    r"(?:\s+(?:summary|summari[sz]ation|review|outline))?",
    re.IGNORECASE,
)


def is_full_document_section_summary_request(question: Any = "") -> bool:
    """Return whether a request asks for *all* document sections.

    This signal carries summary semantics as well as presentation semantics.
    It is intentionally false for a named/numerical local section, which
    should continue through ordinary retrieval instead of loading a full
    parse-bound reading outline.
    """

    text = str(question or "").strip()
    return bool(
        text
        and (
            _ZH_SECTION_DETAIL_SUMMARY_RE.search(text)
            or _EN_SECTION_DETAIL_SUMMARY_RE.search(text)
        )
    )
