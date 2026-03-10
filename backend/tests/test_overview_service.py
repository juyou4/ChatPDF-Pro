import asyncio

import pytest

from services.overview_service import (
    OverviewData,
    PaperSummary,
    SpeedReadContent,
    OVERVIEW_CACHE_VERSION,
    _build_caption_band_bbox,
    _build_document_excerpt,
    _build_figure_clip_bbox,
    _enrich_figures_with_pdf_geometry,
    _extract_figures_for_overview,
    _get_cache_key,
    get_or_create_overview,
    overview_inflight,
)


def test_extract_figures_for_overview_keeps_grouped_image_bboxes_and_label():
    images = [
        {
            "id": "page2_img1",
            "data": "data:image/png;base64,aaa",
            "page": 2,
            "bbox": [120, 80, 250, 170],
        },
        {
            "id": "page2_img2",
            "data": "data:image/png;base64,bbb",
            "page": 2,
            "bbox": [260, 120, 360, 220],
        },
        {
            "id": "page3_img1",
            "data": "data:image/png;base64,ccc",
            "page": 3,
            "bbox": [90, 100, 210, 210],
        },
    ]
    pages = [
        {"content": "page 1"},
        {"content": "figure two page content"},
        {"content": "figure three page content"},
    ]
    figures = [
        {
            "figure_id": "fig-2",
            "number": "2",
            "label": "Figure 2: The AdvRoad framework.",
            "page": 2,
            "image_ids": ["page2_img1", "page2_img2"],
            "figure_bbox": [96, 40, 390, 250],
        },
        {
            "figure_id": "fig-3",
            "number": "3",
            "label": "Figure 3: Attack results.",
            "page": 3,
            "image_ids": ["page3_img1"],
            "figure_bbox": [70, 60, 240, 235],
        },
    ]

    result = _extract_figures_for_overview(images, pages, "standard", figures)

    assert len(result) == 2
    assert result[0]["figure_id"] == "fig-2"
    assert result[0]["figure_label"] == "Figure 2: The AdvRoad framework."
    assert result[0]["image_data_list"] == ["data:image/png;base64,aaa", "data:image/png;base64,bbb"]
    assert result[0]["image_bboxes"] == [[120, 80, 250, 170], [260, 120, 360, 220]]
    assert result[0]["figure_bbox"] == [96, 40, 390, 250]
    assert result[0]["page_content_snippet"] == "figure two page content"


def test_extract_figures_for_overview_falls_back_to_page_images_and_dedupes_labels():
    images = [
        {
            "id": "page3_img1",
            "data": "data:image/png;base64,aaa",
            "page": 3,
            "bbox": [80, 60, 180, 140],
        },
        {
            "id": "page3_img2",
            "data": "data:image/png;base64,bbb",
            "page": 3,
            "bbox": [220, 160, 380, 280],
        },
    ]
    pages = [
        {"content": "page 1"},
        {"content": "page 2"},
        {"content": "figure two page"},
    ]
    figures = [
        {
            "figure_id": "fig-2-a",
            "number": "2",
            "label": "Fig. 2 presents the framework in text body",
            "page": 3,
            "image_ids": [],
            "figure_bbox": [60, 44, 400, 310],
        },
        {
            "figure_id": "fig-2-b",
            "number": "2",
            "label": "Figure 2: The AdvRoad framework.",
            "page": 3,
            "image_ids": [],
            "figure_bbox": [72, 50, 392, 298],
        },
    ]

    result = _extract_figures_for_overview(images, pages, "standard", figures)

    assert len(result) == 1
    assert result[0]["figure_label"] == "Figure 2: The AdvRoad framework."
    assert result[0]["image_data_list"] == ["data:image/png;base64,aaa", "data:image/png;base64,bbb"]
    assert result[0]["image_bboxes"] == [[80, 60, 180, 140], [220, 160, 380, 280]]
    assert result[0]["figure_bbox"] == [72, 50, 392, 298]


def test_build_figure_clip_bbox_expands_multi_image_group_to_more_complete_figure():
    clip_bbox = _build_figure_clip_bbox(
        image_bboxes=[
            [120, 100, 240, 180],
            [255, 140, 355, 220],
        ],
        page_width=600,
        page_height=800,
    )

    assert clip_bbox is not None
    assert clip_bbox[0] < 120
    assert clip_bbox[1] < 100
    assert clip_bbox[2] > 355
    assert clip_bbox[3] > 220
    assert clip_bbox[2] - clip_bbox[0] >= 600 * 0.58
    assert clip_bbox[3] - clip_bbox[1] >= 800 * 0.22


def test_build_caption_band_bbox_uses_previous_caption_as_upper_bound():
    clip_bbox = _build_caption_band_bbox(
        caption_bbox=[80, 420, 300, 438],
        previous_caption_bbox=[70, 210, 310, 226],
        page_width=600,
        page_height=800,
    )

    assert clip_bbox is not None
    assert clip_bbox[0] == 600 * 0.04
    assert clip_bbox[2] == 600 * 0.96
    assert clip_bbox[1] == 226 + 6.0
    assert clip_bbox[3] == 420 - 4.0


def test_enrich_figures_with_pdf_geometry_for_legacy_documents(monkeypatch):
    stored = [
        {
            "figure_id": "fig-2",
            "number": "2",
            "label": "Fig. 2 framework mention in body text",
            "page": 3,
            "image_ids": [],
        }
    ]
    recovered = [
        {
            "figure_id": "fig-2",
            "number": "2",
            "label": "Figure 2: The AdvRoad framework.",
            "page": 3,
            "bbox": [72, 420, 332, 438],
            "caption_bbox": [72, 420, 332, 438],
            "page_width": 612,
            "page_height": 792,
            "image_ids": [],
        }
    ]

    monkeypatch.setattr(
        "services.overview_service._load_figures_from_pdf",
        lambda pdf_url: recovered,
    )

    enriched = _enrich_figures_with_pdf_geometry("/uploads/demo.pdf", stored)

    assert enriched[0]["label"] == "Figure 2: The AdvRoad framework."
    assert enriched[0]["bbox"] == [72, 420, 332, 438]
    assert enriched[0]["caption_bbox"] == [72, 420, 332, 438]
    assert enriched[0]["page_width"] == 612
    assert enriched[0]["page_height"] == 792


def test_get_cache_key_uses_cache_version_prefix():
    cache_key = _get_cache_key("doc-123", "standard")

    assert cache_key == f"{OVERVIEW_CACHE_VERSION}_doc-123_standard"


def test_build_document_excerpt_limits_chars_and_keeps_multiple_sections():
    text = "A" * 4000 + "B" * 4000 + "C" * 4000

    excerpt = _build_document_excerpt(text, "standard")

    assert len(excerpt) < len(text)
    assert "【开头节选】" in excerpt
    assert "【中段节选】" in excerpt
    assert "【结尾节选】" in excerpt
    assert "A" * 50 in excerpt
    assert "B" * 50 in excerpt
    assert "C" * 50 in excerpt


@pytest.mark.asyncio
async def test_get_or_create_overview_reuses_same_inflight_generation(monkeypatch):
    overview_inflight.clear()
    call_count = 0

    async def fake_cached(*args, **kwargs):
        return None

    async def fake_get_document_text(doc_id):
        return "document text"

    async def fake_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return OverviewData(
            doc_id="doc-1",
            title="demo.pdf",
            depth="standard",
            full_text_summary="summary",
            terminology=[],
            speed_read=SpeedReadContent(method="", experiment_design="", problems_solved=""),
            key_figures=[],
            paper_summary=PaperSummary(strengths="", innovations="", future_work=""),
            created_at=123.0,
        )

    monkeypatch.setattr("services.overview_service.get_cached_overview", fake_cached)
    monkeypatch.setattr("services.overview_service.get_document_text", fake_get_document_text)
    monkeypatch.setattr("services.overview_service.generate_overview_content", fake_generate)

    result1, result2 = await asyncio.gather(
        get_or_create_overview("doc-1", "standard"),
        get_or_create_overview("doc-1", "standard"),
    )

    assert call_count == 1
    assert result1.full_text_summary == "summary"
    assert result2.full_text_summary == "summary"
