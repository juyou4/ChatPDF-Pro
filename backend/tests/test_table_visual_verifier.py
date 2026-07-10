from services.table_visual_verifier import (
    resolve_visual_mode,
    should_verify_numeric_table_visual,
    looks_vision_capable_model,
)
from routes.chat_routes import (
    _resolve_numeric_table_visual_model_params,
    _sanitize_public_diagnostics,
    _should_background_numeric_table_visual_verification,
)


class _RequestStub:
    custom_params = {}
    api_provider = "deepseek"
    model = "deepseek-chat"
    api_key = "chat-key"
    api_host = ""


def test_resolve_visual_mode_aliases():
    assert resolve_visual_mode({}) == "auto"
    assert resolve_visual_mode({"numeric_table_visual_verification": "off"}) == "off"
    assert resolve_visual_mode({"numeric_table_visual_verification": "always"}) == "always"
    assert resolve_visual_mode({"numeric_table_visual_verification": True}) == "always"
    assert resolve_visual_mode({"numeric_table_visual_verification": False}) == "off"
    assert resolve_visual_mode({"numeric_table_visual_verification": "unknown"}) == "auto"


def test_visual_gate_skips_non_vision_model_in_auto():
    should_verify, reasons = should_verify_numeric_table_visual(
        query="Table 4 中 ID D 的 mAP 是多少？",
        segments=[{"text": "Table 4 ID D mAP 0.316", "table_header": "ID | mAP"}],
        mode="auto",
        provider="deepseek",
        model="deepseek-chat",
    )
    assert not should_verify
    assert reasons == ["model_not_vision_capable"]


def test_visual_gate_triggers_on_ambiguous_header_for_vision_model():
    should_verify, reasons = should_verify_numeric_table_visual(
        query="Table 4 中 ID D 的 mAP 是多少？",
        segments=[
            {
                "text": "Table 4 ID D 0.316 0.393",
                "table_header": "ID | Column 2 | Column 3 | mAP | mAP",
                "cell_evidence_units": [
                    {"content": "ID D"},
                    {"content": "0.316"},
                    {"content": "0.393"},
                    {"content": "0.393 0.316 0.312"},
                ],
            }
        ],
        mode="auto",
        provider="openai",
        model="gpt-4o",
    )
    assert should_verify
    assert "header_ambiguous" in reasons
    assert "overpacked_metric_cell" in reasons


def test_visual_gate_always_bypasses_vision_model_guess():
    should_verify, reasons = should_verify_numeric_table_visual(
        query="Table 1 的最高值是多少？",
        segments=[{"text": "Table 1 row A 1.0"}],
        mode="always",
        provider="deepseek",
        model="deepseek-chat",
    )
    assert should_verify
    assert reasons == ["mode_always"]


def test_looks_vision_capable_model_for_common_vl_names():
    assert looks_vision_capable_model("silicon", "Qwen2.5-VL-72B-Instruct")
    assert looks_vision_capable_model("gemini", "gemini-2.5-flash")
    assert looks_vision_capable_model("doubao", "Doubao-Seed-2.1-turbo")
    assert not looks_vision_capable_model("deepseek", "deepseek-chat")


def test_visual_diagnostics_are_publicly_exposed():
    public = _sanitize_public_diagnostics({
        "numeric_table_visual_verification": {
            "enabled": True,
            "mode": "auto",
            "triggered": False,
            "reasons": ["model_not_vision_capable"],
            "skipped_reason": "model_not_vision_capable",
        }
    })
    assert public["numeric_table_visual_verification"]["enabled"] is True
    assert public["numeric_table_visual_verification"]["triggered"] is False
    assert public["numeric_table_visual_verification"]["skipped_reason"] == "model_not_vision_capable"


def test_visual_model_params_can_use_dedicated_env_model(monkeypatch):
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_PROVIDER", "doubao")
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_MODEL", "Doubao-Seed-2.1-turbo")
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_API_KEY", "visual-key")
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_API_HOST", "https://ark.cn-beijing.volces.com/api/v3")
    provider, model, api_key, endpoint = _resolve_numeric_table_visual_model_params(_RequestStub())
    assert provider == "doubao"
    assert model == "Doubao-Seed-2.1-turbo"
    assert api_key == "visual-key"
    assert endpoint == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


def test_visual_background_defaults_to_enabled_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("CHATPDF_TABLE_VISUAL_BACKGROUND", raising=False)
    assert _should_background_numeric_table_visual_verification(_RequestStub()) is True

    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_BACKGROUND", "false")
    assert _should_background_numeric_table_visual_verification(_RequestStub()) is False

    request = _RequestStub()
    request.custom_params = {"numeric_table_visual_background": True}
    assert _should_background_numeric_table_visual_verification(request) is True
