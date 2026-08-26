"""精读部分降级：TLS 重试、定性机制句可用、旧空洞缓存可补洞。"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reading_outline_service import (  # noqa: E402
    READING_OUTLINE_REPAIR_POLICY_VERSION,
    _call_ai_api_with_transient_retry,
    _canonical_brief_section_result,
    _is_transient_reading_llm_error,
    _qualitative_suspect_section_ids,
    _section_result_is_usable,
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
