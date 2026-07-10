from services.table_visual_verifier import (
    resolve_visual_mode,
    should_verify_numeric_table_visual,
    looks_vision_capable_model,
)


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
    assert not looks_vision_capable_model("deepseek", "deepseek-chat")
