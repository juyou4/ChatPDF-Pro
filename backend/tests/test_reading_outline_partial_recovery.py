"""精读部分降级：TLS 重试、定性机制句可用、旧空洞缓存可补洞。"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reading_outline_service import (  # noqa: E402
    READING_OUTLINE_REPAIR_POLICY_VERSION,
    _blocking_reading_outline_quality_issues,
    _build_reading_section_skeleton,
    _call_ai_api_with_transient_retry,
    _canonical_brief_section_result,
    _flatten_skeleton_nodes,
    _incomplete_section_issue,
    _is_transient_reading_llm_error,
    _partial_reading_outline_quality_issues,
    _qualitative_suspect_section_ids,
    _section_result_is_usable,
    _section_study_quality_issues,
    _should_refresh_stale_partial_outline,
)


def _payload() -> dict:
    return {
        "blocks": [
            {
                "block_id": "b1",
                "type": "paragraph",
                "page": 1,
                "text": "We replace fixed residual weights with soft attention.",
            }
        ],
        "table_evidence": [],
        "flow_spine_block_ids": [],
        "allowed_block_ids": ["b1"],
    }


def _mechanism_item(*, repair_kind: str = "") -> dict:
    item = _canonical_brief_section_result(
        {
            "summary": "该方法用软注意力替代固定残差权重，因而实现内容自适应的深度选择。",
            "evidence_block_ids": ["b1"],
            "metric_claims": [],
            "prose_claims": [],
        },
        section_id="intro",
        section_hash="h1",
        payload=_payload(),
    )
    if repair_kind:
        item["repair_kind"] = repair_kind
    return item


_APPENDIX_LETTERS = "ABCDEFGHIJKLM"


def _paper_block_index(*, front_matter_titles: tuple[str, ...] = ()) -> dict:
    """13 章正文 + References + 13 个附录的合成解析索引。"""
    blocks: list[dict[str, Any]] = []
    outline: list[dict[str, Any]] = []

    def add_section(section_id: str, title: str, *, page: int) -> None:
        heading_id = f"h_{section_id}"
        blocks.append({
            "block_id": heading_id,
            "type": "heading",
            "text": title,
            "section_id": section_id,
            "page": page,
        })
        blocks.append({
            "block_id": f"p_{section_id}",
            "type": "paragraph",
            "section_id": section_id,
            "page": page,
            "text": (
                f"{title} 的正文段落：本节给出方法细节、实验设置与对应的证据说明，"
                "内容足够长以便被当作可用的章节证据块。"
            ),
        })
        outline.append({
            "section_id": section_id,
            "title": title,
            "level": 1,
            "page": page,
            "first_block": heading_id,
            "source": "pdf_outline",
        })

    blocks.append({
        "block_id": "title",
        "type": "heading",
        "text": "Evidence Bound Reading Outlines For Scientific Papers",
        "section_id": "title",
        "page": 1,
    })
    for index, junk in enumerate(front_matter_titles):
        add_section(f"front_{index}", junk, page=1)
    for index in range(13):
        add_section(f"body_{index}", f"{index + 1}. Body Chapter {index + 1}", page=index + 2)
    add_section("references", "References", page=16)
    for index, letter in enumerate(_APPENDIX_LETTERS):
        add_section(
            f"appendix_{index}",
            f"Appendix {letter}. Supplementary Material {letter}",
            page=17 + index,
        )
    pages: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        pages.setdefault(int(block["page"]), []).append(block)
    return {
        "source": "pdf",
        "outline": outline,
        "pages": [
            {"page": page, "blocks": page_blocks}
            for page, page_blocks in sorted(pages.items())
        ],
    }


def _outline_with_holes(
    block_index: dict,
    *,
    incomplete: set[str],
) -> dict:
    nodes = _flatten_skeleton_nodes(_build_reading_section_skeleton(block_index))
    flat_items = []
    for node in nodes:
        section_id = str(node.get("source_section_id") or "")
        if not section_id:
            continue
        hole = section_id in incomplete
        flat_items.append({
            "source_section_id": section_id,
            "title": node.get("title"),
            "summary": "" if hole else f"{node.get('title')} 的证据绑定小结。",
            "section_status": "fallback" if hole else "ai",
            "evidence_block_ids": [f"p_{section_id}"],
            "evidence_scope": "section",
            "page_start": node.get("page"),
            "page_end": node.get("page"),
        })
    return {"flat_items": flat_items}


def _summary_issues(issues: list[str]) -> list[str]:
    return [issue for issue in issues if "章节摘要" in issue]


def test_body_widespread_missing_summaries_stay_blocking() -> None:
    block_index = _paper_block_index()
    outline = _outline_with_holes(
        block_index,
        incomplete={f"body_{index}" for index in range(4)},
    )

    issues = _summary_issues(_section_study_quality_issues(outline, block_index))

    assert issues == ["正文章节摘要大面积缺失:4/13"]
    assert _blocking_reading_outline_quality_issues(issues) == issues
    assert _partial_reading_outline_quality_issues(issues) == []


def test_appendix_widespread_missing_summaries_are_retriable_partial() -> None:
    block_index = _paper_block_index()
    outline = _outline_with_holes(
        block_index,
        incomplete={f"appendix_{index}" for index in range(4)},
    )

    issues = _summary_issues(_section_study_quality_issues(outline, block_index))

    # 正文完好、附录稀疏：整篇结果必须保住，只发可重试的 partial。
    assert issues == ["附录章节摘要不完整:4"]
    assert _blocking_reading_outline_quality_issues(issues) == []
    assert _partial_reading_outline_quality_issues(issues) == issues


def test_healthy_body_with_sparse_appendix_never_raises_blocking() -> None:
    block_index = _paper_block_index()
    outline = _outline_with_holes(
        block_index,
        incomplete={f"appendix_{index}" for index in (0, 3, 7, 11)},
    )

    issues = _section_study_quality_issues(outline, block_index)

    assert not _blocking_reading_outline_quality_issues(_summary_issues(issues))


def test_incomplete_issue_labels_respect_the_appendix_policy() -> None:
    expected = {f"section_{index}" for index in range(13)}
    incomplete = sorted(expected)[:4]

    assert (
        _incomplete_section_issue("正文", incomplete, expected)
        == "正文章节摘要大面积缺失:4/13"
    )
    assert (
        _incomplete_section_issue("附录", incomplete, expected, widespread_is_blocking=False)
        == "附录章节摘要不完整:4"
    )
    # 少量缺失在正文侧仍然是 partial。
    assert _incomplete_section_issue("正文", incomplete[:2], expected) == "正文章节摘要不完整:2"


def test_front_matter_titles_are_not_reading_chapters() -> None:
    junk = (
        "Academic Editor: Gerardo Flores",
        "Zhen Wang Ling Chen",
        "Received: 12 May 2023",
    )
    baseline = _flatten_skeleton_nodes(_build_reading_section_skeleton(_paper_block_index()))
    polluted = _flatten_skeleton_nodes(
        _build_reading_section_skeleton(_paper_block_index(front_matter_titles=junk))
    )

    titles = {str(node.get("title") or "") for node in polluted}
    assert not titles & set(junk)
    assert [node.get("source_section_id") for node in polluted] == [
        node.get("source_section_id") for node in baseline
    ]


def test_real_chapter_titles_survive_front_matter_filtering() -> None:
    block_index = _paper_block_index(
        front_matter_titles=("4.3 Main Results", "Attention Residuals", "Lab Setup"),
    )

    titles = {
        str(node.get("title") or "")
        for node in _flatten_skeleton_nodes(_build_reading_section_skeleton(block_index))
    }

    assert {"4.3 Main Results", "Attention Residuals", "Lab Setup"} <= titles


def test_ssl_bad_record_mac_is_transient() -> None:
    assert _is_transient_reading_llm_error(
        "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] tlsv1 alert bad record mac"
    )
    assert _is_transient_reading_llm_error("httpx.ConnectError: connection reset")
    assert not _is_transient_reading_llm_error("模型两次返回的逐章总结格式均不完整")
    assert not _is_transient_reading_llm_error("400 invalid json_object")


def test_first_pass_mechanism_sentence_stays_unusable_for_repair() -> None:
    item = _mechanism_item()
    assert item.get("claim_binding_violations")
    assert not _section_result_is_usable(item)


def test_qualitative_number_free_mechanism_sentence_is_usable() -> None:
    item = _mechanism_item(repair_kind="single_qualitative")
    assert item.get("claim_binding_violations")
    assert _section_result_is_usable(item)


def test_qualitative_numeric_violation_still_unusable() -> None:
    item = _mechanism_item(repair_kind="single_qualitative")
    item["claim_binding_violations"] = [
        {"reason": "unbound_prose_claim", "claim_text": "提升 12.3", "numbers": ["12.3"]},
    ]
    assert not _section_result_is_usable(item)


def test_qualitative_suspects_include_missing_sections() -> None:
    payloads = {
        "intro": {**_payload(), "source_section_id": "intro"},
        "empty": {"blocks": [], "source_section_id": "empty"},
        "ok": {**_payload(), "source_section_id": "ok"},
    }
    results = {
        "ok": {
            "summary": "本节说明软注意力如何替代固定残差权重。",
            "table_claim_violations": [],
            "claim_binding_violations": [],
        },
    }
    assert _qualitative_suspect_section_ids(payloads, results) == ["intro"]


def test_stale_partial_cache_is_refreshed_without_touching_complete() -> None:
    partial = {
        "source": "ai_partial",
        "meta": {"repair_policy_version": "old"},
    }
    complete = {
        "source": "ai",
        "meta": {},
    }
    current_partial = {
        "source": "ai_partial",
        "meta": {"repair_policy_version": READING_OUTLINE_REPAIR_POLICY_VERSION},
    }
    assert _should_refresh_stale_partial_outline(partial, can_call_model=True)
    assert not _should_refresh_stale_partial_outline(complete, can_call_model=True)
    assert not _should_refresh_stale_partial_outline(current_partial, can_call_model=True)
    assert not _should_refresh_stale_partial_outline(partial, can_call_model=False)


@pytest.mark.asyncio
async def test_transient_ssl_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_call_ai_api(**kwargs):
        calls.append("call")
        if len(calls) < 3:
            return {"error": "[SSL: SSLV3_ALERT_BAD_RECORD_MAC]"}
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("services.reading_outline_service.call_ai_api", fake_call_ai_api)
    monkeypatch.setattr("services.reading_outline_service.asyncio.sleep", fake_sleep)

    response = await _call_ai_api_with_transient_retry(
        messages=[],
        api_key="k",
        model="m",
        provider="deepseek",
        endpoint="",
        purpose="reading_outline",
    )
    assert response["choices"][0]["message"]["content"] == '{"ok":true}'
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_non_transient_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_call_ai_api(**kwargs):
        calls.append("call")
        return {"error": "400 invalid json_object"}

    monkeypatch.setattr("services.reading_outline_service.call_ai_api", fake_call_ai_api)

    with pytest.raises(RuntimeError, match="400 invalid json_object"):
        await _call_ai_api_with_transient_retry(
            messages=[],
            api_key="k",
            model="m",
            provider="deepseek",
            endpoint="",
            purpose="reading_outline",
        )
    assert calls == ["call"]
