"""Regression tests for semantic overview coverage and landmark contracts."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reading_outline_service import (  # noqa: E402
    _audit_landmark_claims,
    _build_overview_coverage_plan,
    _raw_overview_requirement_issues,
    _select_landmark_claims,
)


def _experiment_section() -> dict:
    return {
        "source_section_id": "results",
        "title": "4. Experimental Results",
        "summary_role": "experiment",
        "section_kind": "body",
        "summary": "VLD-RAG 在 Recall、NDCG 和 MRR 上优于单一检索基线。",
        "evidence_block_ids": ["b_results"],
        "metric_claims": [],
        "prose_claims": [
            {
                "claim_text": "VLD-RAG 在 Recall、NDCG 和 MRR 上优于单一检索基线。",
                "claim_kind": "comparison",
                "subject": "VLD-RAG",
                "predicate": "优于",
                "qualifiers": ["Recall、NDCG 和 MRR"],
                "source_subject": "VLD-RAG",
                "source_predicate": "优于",
                "source_qualifiers": ["Recall、NDCG 和 MRR"],
                "evidence_block_id": "b_results",
                "evidence_quote": "VLD-RAG 在 Recall、NDCG 和 MRR 上优于单一检索基线。",
                "values": [],
            }
        ],
    }


def test_qualitative_evidence_bound_result_is_a_landmark() -> None:
    selected = _select_landmark_claims([_experiment_section()])

    assert len(selected) == 1
    landmark = selected[0]
    assert landmark["evidence_kind"] == "prose"
    assert landmark["claim_kind"] == "comparison"
    assert landmark["values"] == []
    assert landmark["evidence_block_id"] == "b_results"


def test_qualitative_landmark_requires_the_actual_bound_claim_in_experiment_theme() -> None:
    section = _experiment_section()
    landmarks = _select_landmark_claims([section])
    raw_overview = {"_landmark_claims": landmarks}
    theme_items = [
        {
            "type": "theme_experiment",
            "summary": "实验表明该框架总体有效。",
            "source_section_ids": [],
            "study": {"findings": []},
        }
    ]

    missing = _audit_landmark_claims(raw_overview, theme_items)
    assert missing["expected_claim_count"] == 1
    assert missing["covered_claim_count"] == 0
    assert missing["claims"][0]["evidence_kind"] == "prose"

    # Inclusion alone is sufficient only for a direct, evidence-bound claim;
    # this covers a stable semantic compression that avoids duplicating the
    # exact source wording in the theme paragraph.
    theme_items[0]["source_section_ids"] = ["results"]
    source_bound = _audit_landmark_claims(raw_overview, theme_items)
    assert source_bound["covered_claim_count"] == 1

    theme_items[0]["study"]["findings"] = [section["prose_claims"][0]["claim_text"]]
    covered = _audit_landmark_claims(raw_overview, theme_items)
    assert covered["covered_claim_count"] == 1
    assert covered["missing_claim_ids"] == []


def test_overview_repair_requests_missing_qualitative_landmark_without_inventing_numbers() -> None:
    section = _experiment_section()
    landmarks = _select_landmark_claims([section])
    overview = {
        "themes": [
            {
                "kind": "experiment",
                "summary": "实验表明该框架总体有效。",
                "points": [],
                # The repair gate is intentionally stricter than the final
                # post-normalization audit: before normalization there is no
                # persisted direct theme-evidence binding yet.
                "source_section_ids": [],
            }
        ]
    }
    issues = _raw_overview_requirement_issues(
        overview,
        {"slots": []},
        landmarks,
    )

    assert issues == [f"landmark:{landmarks[0]['claim_id']}"]


def test_data_and_setup_slot_belongs_to_experiment_theme() -> None:
    sections = [
        {
            "source_section_id": "method",
            "title": "3. Method",
            "summary_role": "method",
            "section_kind": "body",
        },
        {
            "source_section_id": "setup",
            "title": "4. Experimental Setup and Datasets",
            "summary_role": "experiment",
            "section_kind": "body",
        },
        {
            "source_section_id": "results",
            "title": "5. Results",
            "summary_role": "experiment",
            "section_kind": "body",
        },
        {
            "source_section_id": "conclusion",
            "title": "6. Conclusion",
            "summary_role": "conclusion",
            "section_kind": "body",
        },
    ]

    plan = _build_overview_coverage_plan(sections)
    data_slot = next(slot for slot in plan["slots"] if slot["slot"] == "data_or_setup")
    assert data_slot["theme"] == "experiment"
    assert "setup" in data_slot["source_section_ids"]
    assert "results" not in data_slot["source_section_ids"]
    assert data_slot["required"] is True


def test_motivational_robustness_mention_does_not_require_generalization_slot() -> None:
    """Conditional slots become *required* merely by having a candidate.

    A method section motivated by "robustness" or a conclusion praising it is
    not evidence that the paper ran a generalization experiment, so it must
    not create an unsatisfiable required slot.
    """

    sections = [
        {
            "source_section_id": "method",
            "title": "3.4. Multimodal Query Formulation",
            "summary_role": "method",
            "section_kind": "body",
            "summary": "通过提取关键实体来增强下游检索的鲁棒性。",
        },
        {
            "source_section_id": "results",
            "title": "4. Results",
            "summary_role": "experiment",
            "section_kind": "body",
            "summary": "主要结果显示方法优于基线。",
        },
        {
            "source_section_id": "conclusion",
            "title": "5. Conclusion",
            "summary_role": "conclusion",
            "section_kind": "body",
            "summary": "该方法提升检索鲁棒性与问答性能。",
        },
    ]

    plan = _build_overview_coverage_plan(sections)
    assert not any(slot["slot"] == "generalization" for slot in plan["slots"])

    # An explicit generalization section title, or an experiment section whose
    # evidence-bound summary reports it, still creates the slot.
    sections.append({
        "source_section_id": "transfer",
        "title": "4.3. Generalization to Unseen Domains",
        "summary_role": "experiment",
        "section_kind": "body",
        "summary": "在未见过的数据分布上评估迁移性能。",
    })
    plan_with_experiment = _build_overview_coverage_plan(sections)
    generalization = next(
        slot for slot in plan_with_experiment["slots"] if slot["slot"] == "generalization"
    )
    assert generalization["source_section_ids"] == ["transfer"]
    assert generalization["required"] is True


def test_setup_claim_does_not_consume_limited_landmark_result_budget() -> None:
    setup = _experiment_section()
    setup["source_section_id"] = "setup"
    setup["title"] = "4. Experimental Setup and Datasets"
    result = _experiment_section()
    result["source_section_id"] = "results"
    result["title"] = "5. Main Results"

    selected = _select_landmark_claims([setup, result], limit=1)

    assert len(selected) == 1
    assert selected[0]["source_section_id"] == "results"
