"""固定总结质量门禁的回归测试。"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reading_outline_quality_gate import evaluate_reading_outline_quality  # noqa: E402


THRESHOLDS = {
    "version": "test-v1",
    "section_replay_similarity": 0.8,
    "max_section_replay_ratio": 0.4,
    "near_duplicate_similarity": 0.85,
    "min_empirical_landmark_claims": 1,
    "max_review_recommended_sections": 0,
}


def _outline() -> dict:
    return {
        "items": [
            {"type": "overview", "summary": "该研究建立可追溯的多模态文档问答流程。"},
            {"type": "theme_background", "summary": "现有系统难以同时维护结构定位与证据可信度。", "study": {"findings": []}},
            {"type": "theme_innovation", "summary": "方法通过双空间表示协调布局与语义。", "study": {"findings": ["双空间：布局与文本语义分别建模后再融合。"]}},
            {"type": "theme_experiment", "summary": "在基准测试中，该方法整体优于单一检索基线。", "study": {"findings": ["主结果：F1 从 71.2 提升至 76.4。"]}},
            {"type": "theme_conclusion", "summary": "方案适合证据敏感场景，但依赖上游解析质量。", "study": {"findings": []}},
        ],
        "section_items": [
            {
                "source_section_id": "s1",
                "summary": "引言说明多模态文档问答常在布局定位与文本检索之间发生断裂。",
                "children": [],
            },
            {
                "source_section_id": "s2",
                "summary": "方法将页面布局索引与语义索引结合，并用证据绑定限制最终生成。",
                "children": [],
            },
            {
                "source_section_id": "s3",
                "summary": "实验比较多个检索基线，报告了 F1 从 71.2 到 76.4 的提升以及局限。",
                "children": [],
            },
        ],
        "meta": {
            "section_coverage": {
                "body_expected": 3,
                "body_summarized": 3,
                "appendix_expected": 0,
                "appendix_summarized": 0,
            },
            "overview_coverage": {
                "paper_type": "empirical_method",
                "required_slot_count": 4,
                "covered_slot_count": 4,
                "missing_slots": [],
            },
            "landmark_result_coverage": {
                "expected_claim_count": 1,
                "covered_claim_count": 1,
                "missing_claim_ids": [],
            },
            "sampling": {"review_recommended_sections": 0},
        },
    }


def _kinds(report: dict) -> set[str]:
    return {str(issue.get("kind")) for issue in report["issues"]}


def test_complete_thematic_outline_passes_fixed_quality_gate() -> None:
    report = evaluate_reading_outline_quality(_outline(), THRESHOLDS)

    assert report["status"] == "pass"
    assert report["metrics"]["section_replay_ratio"] <= 0.4
    assert report["metrics"]["near_duplicate_pair_count"] == 0


def test_checked_in_empirical_fixture_passes_release_gate() -> None:
    fixture = Path(__file__).parent / "fixtures" / "reading_outline_quality" / "empirical_complete.json"
    outline = json.loads(fixture.read_text(encoding="utf-8"))

    report = evaluate_reading_outline_quality(outline, THRESHOLDS)

    assert report["status"] == "pass"


def test_gate_rejects_section_dump_and_empty_empirical_landmark() -> None:
    outline = _outline()
    themed = [item for item in outline["items"] if item["type"].startswith("theme_")]
    for index, item in enumerate(themed):
        section = outline["section_items"][index % len(outline["section_items"])]
        item["summary"] = section["summary"]
        item["study"] = {"findings": []}
    outline["meta"]["landmark_result_coverage"] = {
        "expected_claim_count": 0,
        "covered_claim_count": 0,
    }

    report = evaluate_reading_outline_quality(outline, THRESHOLDS)

    assert report["status"] == "fail"
    assert "section_replay_ratio" in _kinds(report)
    assert "empirical_landmark_empty_or_insufficient" in _kinds(report)


def test_gate_rejects_duplicate_structure_missing_slot_and_sampling_gap() -> None:
    outline = _outline()
    outline["section_items"].append({
        "source_section_id": "s2",
        "summary": "方法将页面布局索引与语义索引结合，并用证据绑定限制最终生成。",
        "children": [],
    })
    outline["meta"]["section_coverage"]["body_expected"] = 4
    outline["meta"]["overview_coverage"] = {
        "paper_type": "empirical_method",
        "required_slot_count": 4,
        "covered_slot_count": 3,
        "missing_slots": ["data_or_setup"],
    }
    outline["meta"]["sampling"] = {"review_recommended_sections": 2}

    report = evaluate_reading_outline_quality(outline, THRESHOLDS)

    assert report["status"] == "fail"
    assert {"duplicate_section_ids", "section_coverage", "overview_slot_coverage", "sampling_review_recommended"} <= _kinds(report)
