"""扉页噪声（编辑元信息、无逗号作者行）不应被当成正文章节标题。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.document_block_roles import (  # noqa: E402
    OUTLINE_EXCLUDED_ROLES,
    ROLE_AUTHOR,
    ROLE_HEADING,
    ROLE_PUBLICATION_HEADER,
    classify_block_role,
    classify_front_matter_text,
)


def _role(text: str) -> str:
    decision = classify_front_matter_text(text)
    return str(decision["role"]) if decision else ""


def test_editorial_labels_are_publication_headers() -> None:
    assert _role("Academic Editor: Gerardo Flores") == ROLE_PUBLICATION_HEADER
    assert _role("Corresponding Editor: Jane Roe") == ROLE_PUBLICATION_HEADER
    assert _role("Editor-in-Chief: A. B. Carter") == ROLE_PUBLICATION_HEADER


def test_received_accepted_published_lines_are_publication_headers() -> None:
    assert _role("Received: 12 May 2023") == ROLE_PUBLICATION_HEADER
    assert (
        _role("Received: 1 January 2023; Revised: 5 February 2023; Accepted: 8 February 2023")
        == ROLE_PUBLICATION_HEADER
    )
    assert (
        _role(
            "Received April 3, 2019, accepted April 22, 2019, "
            "date of publication May 1, 2019"
        )
        == ROLE_PUBLICATION_HEADER
    )


def test_editorial_patterns_do_not_swallow_body_text() -> None:
    assert _role("Received signals are aligned before the encoder runs.") == ""
    assert _role("3. Accepted Practices in Robust Optimization") == ""


def test_comma_less_author_byline_is_an_author_line() -> None:
    assert _role("Zhen Wang Ling Chen") == ROLE_AUTHOR
    assert _role("John A. Smith Maria Del Rio") == ROLE_AUTHOR


def test_real_section_titles_are_not_author_lines() -> None:
    for title in (
        "4.3 Main Results",
        "Main Results",
        "Attention Residuals",
        "Attention Residual Network Design",
        "Related Work And Background",
        "Experimental Setup",
    ):
        assert _role(title) == "", title


def test_heading_typed_author_line_is_demoted() -> None:
    decision = classify_block_role(
        {"type": "heading", "page": 1, "text": "Zhen Wang Ling Chen"}
    )
    assert decision["role"] == ROLE_AUTHOR
    assert decision["role"] in OUTLINE_EXCLUDED_ROLES


def test_heading_typed_affiliation_line_is_demoted() -> None:
    decision = classify_block_role({
        "type": "heading",
        "page": 1,
        "text": "Department of Computer Science, Tsinghua University",
    })
    assert decision["role"] in OUTLINE_EXCLUDED_ROLES


def test_heading_typed_editorial_line_is_demoted() -> None:
    decision = classify_block_role(
        {"type": "heading", "page": 1, "text": "Academic Editor: Gerardo Flores"}
    )
    assert decision["role"] == ROLE_PUBLICATION_HEADER


def test_real_headings_stay_headings() -> None:
    for text, page in (
        ("4.3 Main Results", 6),
        ("Attention Residuals", 4),
        ("Lab Setup", 5),
        ("3. Method", 3),
    ):
        decision = classify_block_role({"type": "heading", "page": page, "text": text})
        assert decision["role"] == ROLE_HEADING, text
