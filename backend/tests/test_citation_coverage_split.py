"""缺引用分流：可补标 / 结论过强 / 主题句不计数。"""

from services.academic_answer_contract import (
    analyze_citation_coverage,
    classify_citation_need,
    fill_supported_citation_markers,
    postprocess_critic_result,
)
from services.citation_alignment_service import align_answer_citations


SCREENSHOT_CLAIM = "该方法解决了以往 patch 式改进成功率低的核心难题。"
TOPIC_SENTENCE = "本文提出了一种新方法并采用了残差连接。"
NUMERIC_CLAIM = "在 ImageNet-LT 上准确率达到 95.2%。"
COMPARE_CLAIM = "该方法优于先前的 patch 式基线。"


def test_classify_splits_overclaim_from_must_cite_and_topic():
    assert classify_citation_need(SCREENSHOT_CLAIM) == "overclaim"
    assert classify_citation_need(TOPIC_SENTENCE) == "none"
    assert classify_citation_need(NUMERIC_CLAIM) == "must_cite"
    assert classify_citation_need(COMPARE_CLAIM) == "must_cite"
    assert classify_citation_need("本节说明了训练流程与残差连接。") == "none"


def test_coverage_does_not_count_topic_or_overclaim_as_missing_citation():
    coverage = analyze_citation_coverage(
        f"{TOPIC_SENTENCE}{SCREENSHOT_CLAIM}{NUMERIC_CLAIM}"
    )
    assert coverage["factual_sentence_count"] == 1
    assert coverage["uncited_factual_count"] == 1
    assert coverage["overclaim_uncited_count"] == 1
    assert coverage["uncited_samples"][0].startswith("在 ImageNet-LT")
    assert coverage["overclaim_samples"][0].startswith("该方法解决了")


def test_postprocess_marks_screenshot_claim_as_overreach_not_missing_citation():
    result = postprocess_critic_result(None, answer=SCREENSHOT_CLAIM)
    types = [item["issue_type"] for item in result["issue_details"]]
    assert "missing_citation" not in types
    assert "overreach" in types
    assert result["citation_risk"] is False
    assert result["overreach_risk"] is True
    assert "补引用" in result["suggestion"] or "弱化" in result["suggestion"]


def test_llm_missing_citation_on_overclaim_is_reclassified():
    result = postprocess_critic_result(
        {
            "score": 7,
            "has_hallucination": False,
            "issue_details": [{
                "text": "缺少引用",
                "issue_type": "missing_citation",
                "claim_span": SCREENSHOT_CLAIM,
                "evidence_refs": [],
            }],
        },
        answer=SCREENSHOT_CLAIM,
    )
    assert result["issue_details"][0]["issue_type"] == "overreach"
    assert result["issue_details"][0]["evidence_refs"] == []


def test_fill_only_attaches_high_support_must_cite():
    citations = [{
        "ref": 3,
        "source_text": "在 ImageNet-LT 上准确率达到 95.2%，优于先前的 patch 式基线。",
    }]
    filled = fill_supported_citation_markers(
        f"{NUMERIC_CLAIM}{SCREENSHOT_CLAIM}",
        citations,
        min_score=0.24,
    )
    assert filled["filled_count"] == 1
    assert "[3]" in filled["answer"]
    assert SCREENSHOT_CLAIM in filled["answer"]
    assert f"{SCREENSHOT_CLAIM[:-1]}[3]。" not in filled["answer"]


def test_alignment_does_not_cite_weak_overclaim():
    citations = [{
        "ref": 1,
        "source_text": "We adopt a residual connection and describe the training setup.",
    }]
    aligned = align_answer_citations(SCREENSHOT_CLAIM, citations, min_score=0.24)
    assert "[1]" not in aligned["answer"]
    assert aligned["diagnostics"]["overclaim_unattached_count"] >= 1


def test_missing_citation_issue_carries_suggested_refs():
    result = postprocess_critic_result(
        None,
        answer=NUMERIC_CLAIM,
        retrieval_meta={
            "citations": [{
                "ref": 2,
                "source_text": "在 ImageNet-LT 上准确率达到 95.2%。",
            }],
        },
    )
    missing = [item for item in result["issue_details"] if item["issue_type"] == "missing_citation"]
    assert missing
    assert 2 in missing[0]["evidence_refs"]
    assert "应对 [2]" in missing[0]["text"]
