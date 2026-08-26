"""MinerU 把带虚线的多栏 Figure 拆成 (a)(b)(c) 时，应按跨栏总标题合并。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.figure_extraction import (
    build_mineru_logical_figures_from_block_index,
    _figure_blocks_from_mineru_block_index,
)


def _block_index(pages: list[dict]) -> dict:
    return {
        "source": "mineru_vlm",
        "version": 1,
        "pages": pages,
    }


def _page1_attention_residuals() -> dict:
    return {
        "page": 1,
        "blocks": [
            {
                "block_id": "p1_b6",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [91.0, 434.0, 184.0, 669.0],
                "text": "Figure",
            },
            {
                "block_id": "p1_b7",
                "type": "caption",
                "mineru_type": "image_caption",
                "bbox": [94.0, 672.0, 178.0, 682.0],
                "text": "(a) Standard Residuals",
            },
            {
                "block_id": "p1_b8",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [218.0, 434.0, 323.0, 669.0],
                "text": "Figure",
            },
            {
                "block_id": "p1_b9",
                "type": "caption",
                "mineru_type": "image_caption",
                "bbox": [219.0, 671.0, 323.0, 682.0],
                "text": "(b) Full Attention Residuals",
            },
            {
                "block_id": "p1_b10",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [332.0, 434.0, 523.0, 669.0],
                "text": "Figure",
            },
            {
                "block_id": "p1_b11",
                "type": "caption",
                "mineru_type": "image_caption",
                "bbox": [399.0, 671.0, 507.0, 682.0],
                "text": "(c) Block Attention Residuals",
            },
            {
                "block_id": "p1_b12",
                "type": "caption",
                "mineru_type": "image_caption",
                "bbox": [68.0, 688.0, 542.0, 720.0],
                "text": (
                    "Figure 1: Overview of Attention Residuals. "
                    "(a) Standard Residuals: skip connections. "
                    "(b) Full AttnRes. (c) Block AttnRes."
                ),
            },
            {
                "block_id": "visual_vlm_panel_c",
                "type": "caption",
                "mineru_type": None,
                "bbox": [332.0, 669.0, 523.0, 697.2],
                "text": "图1:块注意力残差（Block Attention Residuals）架构示意图",
            },
        ],
    }


def test_figure_one_panels_merge_under_spanning_caption():
    figures = build_mineru_logical_figures_from_block_index(
        _block_index([_page1_attention_residuals()])
    )
    page1 = [item for item in figures if item.page_idx == 0]
    assert len(page1) == 1
    merged = page1[0]
    assert merged.figure_index == "Figure 1"
    assert merged.caption_text.startswith("Figure 1: Overview of Attention Residuals")
    assert merged.source_metadata.get("merge_count") == 3
    assert merged.source_metadata.get("merged_from") == ["p1_b6", "p1_b8", "p1_b10"]
    assert len(merged.panel_bboxes_page_pts) == 3
    assert merged.body_bbox_page_pts[0] == 91.0
    assert merged.body_bbox_page_pts[2] == 523.0


def test_visual_vlm_caption_does_not_keep_figure_one_split():
    blocks = _figure_blocks_from_mineru_block_index(
        _block_index([_page1_attention_residuals()])
    )
    assert len(blocks) == 1
    assert "块注意力残差" not in (blocks[0].caption_text or "")


def test_figure_five_three_charts_merge_under_shared_caption():
    page = {
        "page": 10,
        "blocks": [
            {
                "block_id": "p10_b0",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [72.0, 72.0, 223.0, 237.0],
                "text": "Figure",
            },
            {
                "block_id": "p10_b1",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [230.0, 72.0, 378.0, 232.0],
                "text": "Figure",
            },
            {
                "block_id": "p10_b2",
                "type": "caption",
                "mineru_type": "chart_caption",
                "bbox": [271.0, 227.0, 345.0, 237.0],
                "text": "Transformer Block Index",
            },
            {
                "block_id": "p10_b3",
                "type": "figure",
                "mineru_type": "image",
                "bbox": [384.0, 72.0, 529.0, 237.0],
                "text": "Figure",
            },
            {
                "block_id": "p10_b4",
                "type": "caption",
                "mineru_type": "chart_caption",
                "bbox": [68.0, 243.0, 541.0, 266.0],
                "text": "Figure 5: Training dynamics of Baseline and Block AttnRes. (a) (b) (c)",
            },
        ],
    }
    figures = build_mineru_logical_figures_from_block_index(_block_index([page]))
    page10 = [item for item in figures if item.page_idx == 9]
    assert len(page10) == 1
    assert page10[0].figure_index == "Figure 5"
    assert page10[0].source_metadata.get("merge_count") == 3
