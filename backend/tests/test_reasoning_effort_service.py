"""Provider reasoning capability and payload regression tests."""

import os
import sys
import asyncio

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.reasoning_effort_service as reasoning_service
import services.chat_service as chat_service


@pytest.fixture(autouse=True)
def _isolate_dynamic_capability_declarations(monkeypatch):
    monkeypatch.setattr(reasoning_service, "load_dynamic_models", lambda: {})
    monkeypatch.setattr(reasoning_service, "load_dynamic_providers", lambda: {})


def _apply(provider, model, effort, *, enabled=True, initial=None):
    body = dict(initial or {})
    resolution = reasoning_service.apply_reasoning_to_payload(
        body,
        provider,
        model,
        enable_thinking=enabled,
        requested_effort=effort,
    )
    return body, resolution


def test_qwen_38_uses_budget_protocol_without_openai_effort():
    body, resolution = _apply(
        "aliyun",
        "qwen3.8-max",
        "high",
        initial={"reasoning_effort": "xhigh", "incremental_output": True},
    )

    assert resolution.mode == "qwen_budget"
    assert body == {
        "enable_thinking": True,
        "thinking_budget": 8192,
        "preserve_thinking": True,
    }


def test_qwen_38_off_uses_only_the_native_disable_switch():
    body, resolution = _apply("aliyun", "qwen3.8-flash", "off", enabled=False)

    assert resolution.enabled is False
    assert body == {"enable_thinking": False}


def test_deepseek_v4_keeps_documented_toggle_and_effort_pair():
    body, resolution = _apply("deepseek", "deepseek-v4-pro", "max")

    assert resolution.native_effort == "max"
    assert body == {
        "reasoning_effort": "max",
        "thinking": {"type": "enabled"},
    }


def test_claude_mandatory_thinking_never_sends_disabled():
    body, resolution = _apply("anthropic", "claude-fable-5", "off", enabled=False)

    assert resolution.enabled is True
    assert resolution.effective == "high"
    assert "off" not in resolution.profile.options
    assert body == {"output_config": {"effort": "high"}}


def test_claude_sonnet_5_can_use_native_disable():
    body, resolution = _apply("anthropic", "claude-sonnet-5", "off", enabled=False)

    assert resolution.enabled is False
    assert body == {"thinking": {"type": "disabled"}}


def test_unknown_claude_family_does_not_receive_guessed_thinking_fields():
    body, resolution = _apply("anthropic", "claude-haiku-3", "high")

    assert resolution.mode == "unsupported"
    assert body == {}


def test_grok_46_off_maps_to_reasoning_effort_none():
    body, resolution = _apply("grok", "grok-4.6", "off", enabled=False)

    assert resolution.enabled is False
    assert body == {"reasoning_effort": "none"}


def test_openai_max_is_downgraded_to_documented_compatible_level():
    body, resolution = _apply("openai", "gpt-5.6-sol", "max")

    assert resolution.effective == "xhigh"
    assert resolution.public()["fallback"] is True
    assert body == {"reasoning_effort": "xhigh"}


def test_openai_pro_does_not_expose_unverified_lower_levels():
    body, resolution = _apply("openai", "gpt-5.5-pro", "low")

    assert resolution.profile.options == ("high",)
    assert resolution.effective == "high"
    assert body == {"reasoning_effort": "high"}


def test_minimax_m2_cannot_be_disabled_but_m3_can():
    m2_body, m2_resolution = _apply("minimax", "MiniMax-M2.7", "off", enabled=False)
    m3_body, m3_resolution = _apply("minimax", "MiniMax-M3", "off", enabled=False)

    assert m2_resolution.enabled is True
    assert m2_body == {"reasoning_split": True}
    assert m3_resolution.enabled is False
    assert m3_body == {"thinking": {"type": "disabled"}}


def test_ollama_gpt_oss_uses_string_level_instead_of_boolean():
    body, resolution = _apply("ollama", "gpt-oss:20b", "off", enabled=False)

    assert resolution.enabled is True
    assert resolution.effective == "medium"
    assert body == {"think": "medium"}


def test_siliconflow_uses_its_own_reasoning_fields():
    v4_body, _ = _apply("silicon", "deepseek-ai/DeepSeek-V4-Flash", "max")
    r1_body, _ = _apply("silicon", "deepseek-ai/DeepSeek-R1", "high")

    assert v4_body == {
        "reasoning_effort": "max",
        "enable_thinking": True,
    }
    assert r1_body == {
        "enable_thinking": True,
        "thinking_budget": 8192,
    }
    assert "thinking" not in v4_body
    assert "thinking" not in r1_body


def test_gemini_37_serializes_the_documented_minimal_level():
    body, resolution = _apply("gemini", "gemini-3.7-flash", "minimal")

    assert resolution.effective == "minimal"
    assert resolution.public()["fallback"] is False
    assert body == {
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        }
    }


def test_gemini_37_off_uses_legacy_budget_zero_control():
    body, resolution = _apply("gemini", "gemini-3.7-flash", "off", enabled=False)

    assert resolution.enabled is False
    assert body == {
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
        }
    }


def test_gemini_31_pro_downgrades_an_invalid_medium_level():
    body, resolution = _apply("gemini", "gemini-3.1-pro-preview", "medium")

    assert resolution.effective == "low"
    assert resolution.public()["fallback"] is True
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_gemini_25_flash_caps_the_max_budget_at_its_model_limit():
    body, resolution = _apply("gemini", "gemini-2.5-flash", "max")

    assert resolution.budget_tokens == 24_576
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 24_576}


def test_explicitly_disabled_custom_provider_sends_no_reasoning_fields(monkeypatch):
    monkeypatch.setattr(
        reasoning_service,
        "load_dynamic_providers",
        lambda: {
            "custom-gateway": {
                "type": "openai",
                "supports_reasoning": False,
            }
        },
    )
    body, resolution = _apply(
        "custom-gateway",
        "gpt-5-compatible-name",
        "high",
        initial={"reasoning_effort": "high", "thinking": {"type": "enabled"}},
    )

    assert resolution.mode == "unsupported"
    assert body == {}


def test_reasoning_history_is_preserved_only_for_declared_families():
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "trace",
        "reasoning_details": [{"text": "trace"}],
    }

    minimax = reasoning_service.prepare_reasoning_history_messages(
        [message], "minimax", "MiniMax-M3"
    )[0]
    generic = reasoning_service.prepare_reasoning_history_messages(
        [message], "openai", "gpt-4.1"
    )[0]

    assert minimax["reasoning_content"] == "trace"
    assert minimax["reasoning_details"] == [{"text": "trace"}]
    assert "reasoning_content" not in generic
    assert "reasoning_details" not in generic


def test_chat_service_passes_native_and_extra_fields_to_provider(monkeypatch):
    calls = []

    class CaptureClient:
        async def chat(self, *args, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_service, "_create_provider_client", lambda *args: CaptureClient())

    asyncio.run(chat_service.call_ai_api(
        [{"role": "user", "content": "hello"}],
        "test-key",
        "qwen3.8-max",
        "aliyun",
        enable_thinking=True,
        reasoning_effort="high",
    ))
    asyncio.run(chat_service.call_ai_api(
        [{"role": "user", "content": "hello"}],
        "test-key",
        "deepseek-v4-pro",
        "deepseek",
        enable_thinking=True,
        reasoning_effort="max",
    ))

    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["custom_params"] == {
        "enable_thinking": True,
        "thinking_budget": 8192,
        "preserve_thinking": True,
    }
    assert calls[1]["reasoning_effort"] == "max"
    assert calls[1]["custom_params"] == {"thinking": {"type": "enabled"}}


def test_anthropic_sampling_parameters_are_removed_when_the_api_rejects_them(monkeypatch):
    calls = []

    class CaptureClient:
        async def chat(self, *args, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_service, "_create_provider_client", lambda *args: CaptureClient())

    asyncio.run(chat_service.call_ai_api(
        [{"role": "user", "content": "hello"}],
        "test-key",
        "claude-opus-5",
        "anthropic",
        enable_thinking=False,
        reasoning_effort="off",
        temperature=0.7,
        top_p=0.8,
        custom_params={"temperature": 0.4, "top_p": 0.6},
    ))
    asyncio.run(chat_service.call_ai_api(
        [{"role": "user", "content": "hello"}],
        "test-key",
        "claude-haiku-4-5-20251001",
        "anthropic",
        enable_thinking=True,
        reasoning_effort="medium",
        temperature=0.7,
        top_p=0.8,
    ))

    assert calls[0]["temperature"] is None
    assert calls[0]["top_p"] is None
    assert calls[0]["max_tokens"] is None
    assert "temperature" not in calls[0]["custom_params"]
    assert "top_p" not in calls[0]["custom_params"]
    assert calls[1]["temperature"] is None
    assert calls[1]["top_p"] is None

    _, adaptive_resolution = _apply("anthropic", "claude-opus-5", "high")
    assert reasoning_service.ensure_reasoning_output_budget(
        1_000, adaptive_resolution
    ) == 9_216


def test_openai_reasoning_requests_drop_sampling_parameters(monkeypatch):
    calls = []

    class CaptureClient:
        async def chat(self, *args, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_service, "_create_provider_client", lambda *args: CaptureClient())

    asyncio.run(chat_service.call_ai_api(
        [{"role": "user", "content": "hello"}],
        "test-key",
        "o3",
        "openai",
        enable_thinking=True,
        reasoning_effort="high",
        temperature=0.7,
        top_p=0.8,
    ))

    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["temperature"] is None
    assert calls[0]["top_p"] is None
