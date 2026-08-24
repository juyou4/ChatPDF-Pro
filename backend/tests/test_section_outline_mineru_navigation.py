"""MinerU 章节树已可导航时，不应再为 structure_degraded 去打 AI。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.section_outline_service import (
    _mineru_structure_is_navigation_ready,
    get_or_create_section_outline,
)


def _mineru_block_index(*, degraded: bool, fallback_only: bool = False) -> dict:
    outline = (
        [{"title": "全文", "level": 1, "page": 1, "first_block": "p1_b0", "section_id": "s1", "source": "fallback"}]
        if fallback_only
        else [
            {"title": "1. Introduction", "level": 1, "page": 1, "first_block": "p1_b1", "section_id": "s1", "source": "heading"},
            {"title": "2. Method", "level": 1, "page": 2, "first_block": "p2_b1", "section_id": "s2", "source": "heading"},
            {"title": "3. Conclusion", "level": 1, "page": 3, "first_block": "p3_b1", "section_id": "s3", "source": "heading"},
        ]
    )
    pages = (
        [{"page": 1, "blocks": [{"block_id": "p1_b0", "type": "paragraph", "text": "全文"}]}]
        if fallback_only
        else [
            {"page": 1, "blocks": [{"block_id": "p1_b1", "type": "heading", "text": "1. Introduction", "level": 1}]},
            {"page": 2, "blocks": [{"block_id": "p2_b1", "type": "heading", "text": "2. Method", "level": 1}]},
            {"page": 3, "blocks": [{"block_id": "p3_b1", "type": "heading", "text": "3. Conclusion", "level": 1}]},
        ]
    )
    return {
        "source": "mineru_vlm",
        "version": 1,
        "mineru_meta": {
            "structure_degraded": degraded,
            "structure_version": 13,
            "outline_is_fallback_only": fallback_only,
            "flat_structure_without_headings": False,
        },
        "outline": outline,
        "pages": pages,
    }


def _doc(doc_id: str) -> dict:
    return {
        "filename": "vldrag.pdf",
        "data": {
            "parse_manifest": {
                "generation": "parse-test",
                "source_hash": "src-test",
                "resolved_route": "mineru",
                "status": "ready",
            }
        },
    }


def test_navigation_ready_when_degraded_but_outline_is_complete():
    block_index = _mineru_block_index(degraded=True)
    outline = {"items": block_index["outline"]}
    assert _mineru_structure_is_navigation_ready(block_index, outline) is True


def test_navigation_not_ready_when_outline_is_fallback_only():
    block_index = _mineru_block_index(degraded=True, fallback_only=True)
    outline = {"items": block_index["outline"]}
    assert _mineru_structure_is_navigation_ready(block_index, outline) is False


def test_get_or_create_uses_mineru_tree_instead_of_ai_when_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    doc_id = "doc-vldrag"
    block_index = _mineru_block_index(degraded=True)

    async def _should_not_call_ai(**_kwargs):
        raise AssertionError("quality-ok MinerU outline must not call the LLM")

    monkeypatch.setattr(
        "services.section_outline_service._generate_ai_section_outline",
        _should_not_call_ai,
    )

    result = asyncio.run(
        get_or_create_section_outline(
            data_dir=tmp_path,
            doc_id=doc_id,
            doc=_doc(doc_id),
            block_index=block_index,
            api_key="sk-test",
            model="deepseek-v4-flash-vision-exp",
            provider="deepseek",
        )
    )

    assert result["source"] == "mineru"
    assert result["meta"].get("structure_recovered") is True
    titles = [item["title"] for item in result.get("flat_items") or []]
    assert "1. Introduction" in titles
    assert "2. Method" in titles
