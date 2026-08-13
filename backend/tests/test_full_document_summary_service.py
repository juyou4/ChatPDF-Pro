"""全文总结展示合同的确定性回归测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.full_document_summary_service import (  # noqa: E402
    FULL_DOCUMENT_SUMMARY_RENDER_VERSION,
    build_full_document_summary,
    should_render_section_details,
)
from services.academic_answer_contract import derive_answer_certainty  # noqa: E402
from services.chat_intent_service import build_chat_turn_context, prepare_chat_intent  # noqa: E402


def _block(block_id: str, page: int, text: str) -> dict:
    return {"block_id": block_id, "page": page, "type": "paragraph", "text": text}


def _fixture() -> tuple[dict, dict]:
    block_index = {
        "pages": [
            {"page": 1, "blocks": [_block("b1", 1, "研究问题与背景证据。"), _block("b2", 1, "方法证据。"), _block("b3", 1, "实验结果证据。"), _block("b4", 1, "结论边界证据。"), _block("b5", 1, "方法细节证据。"), _block("b6", 1, "子章节实验细节。")]},
        ]
    }
    outline = {
        "source": "ai",
        "title": "示例论文",
        "items": [
            {
                "id": "paper_overview",
                "type": "overview",
                "title": "论文要旨",
                "summary": "论文提出一个可验证的文档问答框架。",
                "evidence_block_ids": ["b1"],
            },
            {
                "id": "reading_theme_background",
                "type": "theme_background",
                "title": "研究背景与问题",
                "summary": "现有方法难以同时处理结构与证据。",
                "study": {"findings": []},
                "evidence_block_ids": ["b1"],
            },
            {
                "id": "reading_theme_innovation",
                "type": "theme_innovation",
                "title": "核心方法与创新",
                "summary": "框架把结构解析与证据核验连接起来。",
                "study": {
                    "findings": [
                        "双空间表示：同时保留结构和语义证据。",
                        "证据校验：对关键结论执行来源绑定。",
                    ]
                },
                "evidence_block_ids": ["b2", "b5"],
            },
            {
                "id": "reading_theme_experiment",
                "type": "theme_experiment",
                "title": "关键实验结果",
                "summary": "实验显示混合检索在主要基准上更稳定。",
                "study": {"findings": ["主结果：混合检索优于单一检索。"]},
                "evidence_block_ids": ["b3"],
            },
            {
                "id": "reading_theme_conclusion",
                "type": "theme_conclusion",
                "title": "结论、价值与边界",
                "summary": "方法适合证据敏感的长文档问答，但仍受解析质量影响。",
                "study": {"findings": []},
                "evidence_block_ids": ["b4"],
            },
        ],
        "section_items": [
            {
                "id": "s_method",
                "source_section_id": "s_method",
                "title": "3. Method",
                "summary": "方法章节由子章节组成。",
                "section_kind": "body",
                "section_status": "synthesized",
                "evidence_block_ids": ["b2", "b5"],
                "children": [
                    {
                        "id": "s_method_detail",
                        "source_section_id": "s_method_detail",
                        "title": "3.1 Details",
                        "summary": "方法细节说明。",
                        "section_kind": "body",
                        "section_status": "ai",
                        "evidence_block_ids": ["b5"],
                        "children": [],
                    }
                ],
            },
            {
                "id": "s_result",
                "source_section_id": "s_result",
                "title": "4. Results",
                "summary": "实验结果说明。",
                "section_kind": "body",
                "section_status": "ai",
                "evidence_block_ids": ["b3"],
                "children": [],
            },
        ],
        "meta": {
            "generation_status": "completed",
            "section_coverage": {
                "body_expected": 3,
                "body_summarized": 3,
                "appendix_expected": 0,
                "appendix_summarized": 0,
            },
            "presentation": {"mode": "thematic"},
        },
    }
    return outline, block_index


def test_default_full_document_summary_is_thematic_and_keeps_findings() -> None:
    outline, block_index = _fixture()

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    answer = rendered["answer"]
    coverage = rendered["coverage"]
    assert "核心方法与创新" in answer
    assert "双空间表示" in answer
    assert "证据校验" in answer
    assert "主结果：混合检索优于单一检索" in answer
    assert "### 章节梳理" not in answer
    assert coverage["render_version"] == FULL_DOCUMENT_SUMMARY_RENDER_VERSION
    assert coverage["presentation_mode"] == "thematic"
    assert coverage["visible_section_count"] == 0
    assert coverage["rendered_section_count"] == 3
    assert coverage["body_expected"] == 3
    assert coverage["body_summarized"] == 3
    assert coverage["complete"] is True
    assert coverage["structural_section_count"] == 3
    assert coverage["structural_expected_count"] == 3
    assert coverage["structural_coverage"]["body_expected_ids"] == [
        "s_method",
        "s_method_detail",
        "s_result",
    ]
    assert coverage["structural_coverage"]["body_covered_ids"] == [
        "s_method",
        "s_method_detail",
        "s_result",
    ]
    assert coverage["semantic_quality_status"] == "unknown"


def test_explicit_chapter_request_shows_details_without_synthesized_parent() -> None:
    outline, block_index = _fixture()

    rendered = build_full_document_summary(outline, block_index, question="请按章节梳理这篇论文")

    answer = rendered["answer"]
    assert "### 章节梳理" in answer
    assert "3.1 Details" in answer
    assert "4. Results" in answer
    # The parent is only a child-summary synthesis and should not be printed
    # together with its children.
    assert "3. Method" not in answer
    assert rendered["coverage"]["presentation_mode"] == "section_detail"
    assert rendered["coverage"]["visible_section_count"] == 2
    assert rendered["coverage"]["rendered_section_count"] == 3
    assert rendered["coverage"]["complete"] is True


def test_section_detail_intent_is_narrow() -> None:
    assert should_render_section_details("请按章节总结") is True
    assert should_render_section_details("请逐个小节梳理") is True
    assert should_render_section_details("Summarize each section") is True
    assert should_render_section_details("Give me a chapter-by-chapter summary") is True
    assert should_render_section_details("Please go through the sections") is True
    assert should_render_section_details("Provide a section-wise summary") is True
    assert should_render_section_details("请总结全文的主要内容") is False
    assert should_render_section_details("请概括论文方法和结果") is False
    assert should_render_section_details("Summarize the paper's method and results") is False
    assert should_render_section_details("What does section 3 say?") is False


def test_complete_section_summary_requests_enter_the_parse_bound_summary_route() -> None:
    """Explicit all-section wording is global, unlike one named section."""

    full_requests = [
        "请按章节梳理全文",
        "请逐个小节总结这篇论文",
        "Summarize each section of the paper",
        "Give me a chapter-by-chapter summary of the whole paper",
        "Please go through the sections",
    ]
    for question in full_requests:
        intent = prepare_chat_intent(original_question=question)
        assert intent.task == "summarize"
        assert intent.scope == "document"
        assert intent.full_document_summary is True
        assert "summary_scope:full_document" in intent.matched_rules

    local_requests = [
        "总结第 3 节",
        "summarize section 3",
        "summarize the methods section",
        "总结本文的方法和实验",
        "Summarize the paper's method and results",
        "What does section 3 say?",
    ]
    for question in local_requests:
        intent = prepare_chat_intent(original_question=question)
        assert intent.full_document_summary is False


def test_partial_outline_cannot_become_complete_only_from_a_theme() -> None:
    outline, block_index = _fixture()
    outline["meta"]["generation_status"] = "partial"
    outline["meta"]["section_coverage"]["body_summarized"] = 2

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    assert rendered["coverage"]["presentation_mode"] == "thematic"
    assert rendered["coverage"]["body_summarized"] == 2
    assert rendered["coverage"]["body_complete"] is False
    assert rendered["coverage"]["complete"] is False


def test_fallback_outline_without_themes_keeps_section_guide_available() -> None:
    outline, block_index = _fixture()
    outline["items"] = []
    outline["source"] = "fallback"

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    assert "### 章节梳理" in rendered["answer"]
    assert "3.1 Details" in rendered["answer"]
    assert rendered["coverage"]["presentation_mode"] == "section_detail"
    assert rendered["coverage"]["visible_section_count"] == 2
    assert rendered["coverage"]["complete"] is True


def test_thematic_findings_choose_the_best_authorized_evidence_block() -> None:
    outline, block_index = _fixture()
    innovation = next(item for item in outline["items"] if item["id"] == "reading_theme_innovation")
    innovation["study"]["findings"] = ["方法细节：该框架使用一个专门的可验证模块。"]
    block_index["pages"][0]["blocks"] = [
        _block("b1", 1, "研究问题与背景证据。"),
        _block("b2", 1, "通用方法证据。"),
        _block("b3", 1, "实验结果证据。"),
        _block("b4", 1, "结论边界证据。"),
        _block("b5", 1, "该框架使用一个专门的可验证模块来处理方法细节。"),
        _block("b6", 1, "子章节实验细节。"),
    ]

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    finding_citation = next(
        citation for citation in rendered["citations"]
        if citation["block_id"] == "b5"
    )
    assert f"方法细节：该框架使用一个专门的可验证模块。 [{finding_citation['ref']}]" in rendered["answer"]


def test_finding_evidence_binding_overrides_the_legacy_theme_wide_order() -> None:
    outline, block_index = _fixture()
    innovation = next(item for item in outline["items"] if item["id"] == "reading_theme_innovation")
    finding = "方法细节：该框架使用一个专门的可验证模块。"
    innovation["study"] = {
        "findings": [finding],
        "finding_evidence": [{"text": finding, "evidence_block_ids": ["b5"]}],
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    citation = next(item for item in rendered["citations"] if item["block_id"] == "b5")
    assert f"{finding} [{citation['ref']}]" in rendered["answer"]


def test_certainty_keeps_structural_coverage_when_chat_projection_is_thematic() -> None:
    outline, block_index = _fixture()
    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    certainty = derive_answer_certainty(
        answer=rendered["answer"],
        retrieval_meta={"full_document_summary": rendered["coverage"]},
        answer_mode="full_document_summary",
    )

    summary_meta = certainty["full_document_summary"]
    assert summary_meta["complete"] is True
    assert summary_meta["presentation_mode"] == "thematic"
    assert summary_meta["visible_section_count"] == 0
    assert summary_meta["structural_section_count"] == 3
    assert summary_meta["rendered_section_count"] == 3


def test_semantic_quality_is_forwarded_without_changing_structural_completeness() -> None:
    outline, block_index = _fixture()
    outline["meta"]["semantic_quality"] = {
        "status": "needs_review",
        "missing_slots": ["data_or_setup"],
        "landmark_empty_for_empirical": True,
        "blocking": False,
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")
    certainty = derive_answer_certainty(
        answer=rendered["answer"],
        retrieval_meta={"full_document_summary": rendered["coverage"]},
        answer_mode="full_document_summary",
    )

    assert rendered["coverage"]["semantic_quality_status"] == "needs_review"
    assert rendered["coverage"]["semantic_quality"]["blocking"] is False
    assert rendered["coverage"]["complete"] is True
    assert certainty["full_document_summary"]["semantic_quality_status"] == "needs_review"


def test_old_cache_derives_semantic_quality_without_mutating_outline() -> None:
    outline, block_index = _fixture()
    outline["meta"]["overview_coverage"] = {
        "paper_type": "empirical_method",
        "required_slot_count": 5,
        "covered_slot_count": 4,
        "missing_slots": ["data_or_setup"],
    }
    outline["meta"]["landmark_result_coverage"] = {
        "expected_claim_count": 0,
        "covered_claim_count": 0,
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    semantic_quality = rendered["coverage"]["semantic_quality"]
    assert "semantic_quality" not in outline["meta"]
    assert rendered["coverage"]["semantic_quality_status"] == "needs_review"
    assert semantic_quality["missing_slots"] == ["data_or_setup"]
    assert semantic_quality["landmark_empty_for_empirical"] is True
    assert semantic_quality["blocking"] is False


def test_stale_persisted_ledgers_are_recomputed_with_current_audit_semantics() -> None:
    """A pre-v4.17 ledger must not misreport a cache that is healthy today.

    The old audit filed ``data_or_setup`` under the innovation theme and never
    selected qualitative landmarks, so its persisted verdict is stale.  When
    every theme carries ``source_section_ids`` the response-time diagnostic
    re-runs the current audits over the cached tree instead — without writing.
    """

    outline, block_index = _fixture()
    result_section = next(item for item in outline["section_items"] if item["id"] == "s_result")
    result_section["summary_role"] = "experiment"
    result_section["prose_claims"] = [{
        "claim_text": "混合检索优于单一检索基线。",
        "claim_kind": "comparison",
        "evidence_block_id": "b3",
        "evidence_quote": "混合检索优于单一检索基线。",
        "values": [],
    }]
    theme_sources = {
        "theme_background": ["s_method"],
        "theme_innovation": ["s_method", "s_method_detail"],
        "theme_experiment": ["s_result"],
        "theme_conclusion": ["s_result"],
    }
    for item in outline["items"]:
        sources = theme_sources.get(str(item.get("type") or ""))
        if sources:
            item["source_section_ids"] = sources
    stale_overview = {
        "paper_type": "empirical_method",
        "required_slot_count": 5,
        "covered_slot_count": 4,
        "missing_slots": ["data_or_setup"],
    }
    stale_landmark = {"expected_claim_count": 0, "covered_claim_count": 0}
    outline["meta"]["overview_coverage"] = dict(stale_overview)
    outline["meta"]["landmark_result_coverage"] = dict(stale_landmark)
    outline["meta"]["semantic_quality"] = {
        "status": "needs_review",
        "missing_slots": ["data_or_setup"],
        "blocking": False,
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    quality = rendered["coverage"]["semantic_quality"]
    assert quality["ledgers_recomputed_from_cache"] is True
    assert quality["status"] == "healthy"
    assert quality["missing_slots"] == []
    assert quality["landmark_expected_claim_count"] == 1
    assert quality["landmark_covered_claim_count"] == 1
    assert rendered["coverage"]["semantic_quality_status"] == "healthy"
    # Response-only recomputation: the persisted stale ledgers are untouched.
    assert outline["meta"]["overview_coverage"] == stale_overview
    assert outline["meta"]["landmark_result_coverage"] == stale_landmark
    assert outline["meta"]["semantic_quality"]["status"] == "needs_review"


def test_recomputation_is_skipped_when_a_theme_lacks_source_ids() -> None:
    """Both audits key coverage on theme ``source_section_ids``.

    Recomputing a cache from before that field would judge every slot
    uncovered and make the outline look worse than its own persisted ledger,
    so those shapes must keep the persisted verdict.
    """

    outline, block_index = _fixture()
    outline["items"][2]["source_section_ids"] = ["s_method"]  # only one theme
    outline["meta"]["semantic_quality"] = {
        "status": "healthy",
        "missing_slots": [],
        "issues": [],
        "blocking": False,
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    quality = rendered["coverage"]["semantic_quality"]
    assert "ledgers_recomputed_from_cache" not in quality
    assert quality["status"] == "healthy"


def test_old_cache_recovers_evidence_bound_qualitative_landmarks_without_a_write() -> None:
    outline, block_index = _fixture()
    result_section = next(item for item in outline["section_items"] if item["id"] == "s_result")
    qualitative_claim = "混合检索优于单一检索基线。"
    result_section["prose_claims"] = [{
        "claim_text": qualitative_claim,
        "claim_kind": "comparison",
        "evidence_block_id": "b3",
        "evidence_quote": "混合检索优于单一检索基线。",
    }]
    experiment = next(item for item in outline["items"] if item["type"] == "theme_experiment")
    experiment["source_section_ids"] = ["s_result"]
    experiment["study"]["findings"] = [qualitative_claim]
    outline["meta"]["overview_coverage"] = {
        "paper_type": "empirical_method",
        "required_slot_count": 4,
        "covered_slot_count": 4,
        "missing_slots": [],
    }
    outline["meta"]["landmark_result_coverage"] = {
        "expected_claim_count": 0,
        "covered_claim_count": 0,
    }
    outline["meta"]["semantic_quality"] = {
        "status": "needs_review",
        "issues": ["empirical_landmarks_empty"],
        "landmark_empty_for_empirical": True,
        "blocking": False,
    }

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    quality = rendered["coverage"]["semantic_quality"]
    assert quality["landmark_empty_for_empirical"] is False
    assert quality["landmark_expected_claim_count"] == 1
    assert quality["landmark_covered_claim_count"] == 1
    assert quality["derived_landmark_result_coverage"]["derived_from_cached_prose_claims"] is True
    assert "semantic_quality" in outline["meta"]
    assert outline["meta"]["semantic_quality"]["landmark_empty_for_empirical"] is True


def test_theme_order_is_fixed_even_when_the_cache_is_shuffled() -> None:
    """A reordered cache must not reach the reader as a shuffled narrative."""

    outline, block_index = _fixture()
    by_id = {item["id"]: item for item in outline["items"]}
    outline["items"] = [
        by_id["reading_theme_experiment"],
        by_id["reading_theme_conclusion"],
        {
            "id": "reading_theme_unknown",
            "type": "theme_unknown",
            "title": "未知主题",
            "summary": "一个未来 prompt 版本引入的主题。",
            "evidence_block_ids": ["b6"],
        },
        by_id["paper_overview"],
        by_id["reading_theme_innovation"],
        by_id["reading_theme_background"],
    ]

    answer = build_full_document_summary(outline, block_index, question="请总结全文")["answer"]

    headings = [line.removeprefix("### ") for line in answer.splitlines() if line.startswith("### ")]
    assert headings == [
        "论文要旨",
        "研究背景与问题",
        "核心方法与创新",
        "关键实验结果",
        "结论、价值与边界",
        # An unrecognized theme is kept rather than dropped, but it never
        # displaces the fixed reading order.
        "未知主题",
    ]


def test_renderer_bounds_findings_even_when_the_cache_exceeds_the_budget() -> None:
    outline, block_index = _fixture()
    innovation = next(item for item in outline["items"] if item["type"] == "theme_innovation")
    innovation["study"]["findings"] = [f"方法重点 {index}：机制说明 {index}。" for index in range(6)]
    experiment = next(item for item in outline["items"] if item["type"] == "theme_experiment")
    experiment["study"]["findings"] = [f"实验重点 {index}：结果说明 {index}。" for index in range(8)]

    answer = build_full_document_summary(outline, block_index, question="请总结全文")["answer"]

    assert sum(1 for line in answer.splitlines() if line.startswith("- 方法重点")) == 3
    assert sum(1 for line in answer.splitlines() if line.startswith("- 实验重点")) == 5


def test_single_section_restatement_is_counted_but_kept_in_the_thematic_view() -> None:
    """One echoed point is normal: a section conclusion can be a key result.

    Nothing else shows that conclusion in the thematic view, so dropping it
    would remove information instead of removing duplication.
    """

    outline, block_index = _fixture()
    section = next(item for item in outline["section_items"] if item["id"] == "s_result")
    section["summary"] = "实验在三个公开基准上比较了混合检索与单一检索的表现。"
    experiment = next(item for item in outline["items"] if item["type"] == "theme_experiment")
    experiment["study"]["findings"] = [
        f"4. Results：{section['summary']}",
        "主结果：混合检索在三个基准上均领先。",
    ]

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    answer = rendered["answer"]
    assert "主结果：混合检索在三个基准上均领先。" in answer
    assert section["summary"] in answer
    quality = rendered["coverage"]["semantic_quality"]
    assert quality["section_echo_finding_count"] == 1
    assert quality["themes_restating_sections"] == []
    assert rendered["coverage"]["semantic_quality_status"] == "unknown"
    assert rendered["coverage"]["complete"] is True


def test_section_restatements_are_dropped_when_the_chapter_list_is_rendered() -> None:
    outline, block_index = _fixture()
    section = next(item for item in outline["section_items"] if item["id"] == "s_result")
    section["summary"] = "实验在三个公开基准上比较了混合检索与单一检索的表现。"
    experiment = next(item for item in outline["items"] if item["type"] == "theme_experiment")
    experiment["study"]["findings"] = [
        f"4. Results：{section['summary']}",
        "主结果：混合检索在三个基准上均领先。",
    ]

    rendered = build_full_document_summary(outline, block_index, question="请按章节梳理全文")

    answer = rendered["answer"]
    assert "主结果：混合检索在三个基准上均领先。" in answer
    # The chapter list below prints this section verbatim, so the thematic
    # copy is pure duplication here.
    assert answer.count(section["summary"]) == 1
    assert f"- 4. Results：{section['summary']}" not in answer
    assert rendered["coverage"]["semantic_quality"]["section_echo_finding_count"] == 1


def test_theme_whose_points_all_restate_sections_is_flagged_as_degraded() -> None:
    """This is the shape the fallback payload produces when a theme is missing."""

    outline, block_index = _fixture()
    method = next(item for item in outline["section_items"] if item["id"] == "s_method")
    detail = method["children"][0]
    method["summary"] = "方法章节介绍了整体框架的两个协同组成部分与训练流程。"
    detail["summary"] = "细节小节说明了双空间表示的构造方式与对齐目标函数。"
    innovation = next(item for item in outline["items"] if item["type"] == "theme_innovation")
    innovation["study"]["findings"] = [
        f"3. Method：{method['summary']}",
        f"3.1 Details：{detail['summary']}",
    ]

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    quality = rendered["coverage"]["semantic_quality"]
    assert quality["section_echo_finding_count"] == 2
    assert quality["themes_restating_sections"] == ["theme_innovation"]
    assert "themes_restating_sections:theme_innovation" in quality["issues"]
    assert quality["status"] == "needs_review"
    assert quality["blocking"] is False
    # A presentation defect must never invalidate the parse-bound ledger.
    assert rendered["coverage"]["complete"] is True


def test_theme_without_resolvable_evidence_is_reported_but_still_shown() -> None:
    outline, block_index = _fixture()
    conclusion = next(item for item in outline["items"] if item["type"] == "theme_conclusion")
    conclusion["evidence_block_ids"] = ["missing-block"]

    rendered = build_full_document_summary(outline, block_index, question="请总结全文")

    assert "结论、价值与边界" in rendered["answer"]
    quality = rendered["coverage"]["semantic_quality"]
    assert quality["themes_without_evidence"] == ["theme_conclusion"]
    assert "themes_without_evidence:theme_conclusion" in quality["issues"]
    assert rendered["coverage"]["semantic_quality_status"] == "needs_review"
    assert rendered["coverage"]["complete"] is True


def test_original_question_survives_route_rewrite_for_section_projection() -> None:
    """The chat route must render from original_question, never a stripped query."""

    question = "Summarize each section of the paper"
    intent = prepare_chat_intent(original_question=question)
    turn = build_chat_turn_context(
        original_question=question,
        effective_question="summarize paper",
        intent_question="summarize paper",
        retrieval_query="summary",
        intent=intent,
    )
    outline, block_index = _fixture()

    rendered = build_full_document_summary(
        outline,
        block_index,
        question=turn.original_question,
    )

    assert rendered["coverage"]["presentation_mode"] == "section_detail"
    assert "### 章节梳理" in rendered["answer"]
