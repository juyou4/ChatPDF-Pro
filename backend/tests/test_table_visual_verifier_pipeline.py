"""Focused regression tests for post-retrieval visual table verification."""

import asyncio
import json
from pathlib import Path

import pytest

import services.table_visual_verifier as verifier


def _bundle(*, page=3, table_id="Table 1", row_id="Ours", accuracy="91.2", bbox=None):
    return {
        "bundle_id": f"bundle-{table_id}-{page}-{row_id}",
        "table_id": table_id,
        "table_caption": f"{table_id}: Results",
        "table_header": "ID | Accuracy",
        "table_body_markdown": f"| ID | Accuracy |\n| {row_id} | {accuracy} |",
        "page_start": page,
        "page_end": page,
        "pages": [page],
        "bounding_box": bbox or [20, 40, 360, 260],
        "source": "mineru",
        "evidence_units": [
            {
                "evidence_unit_type": "table_row",
                "page": page,
                "row_id": row_id,
                "row_text": f"{row_id} | {accuracy}",
                "bbox": [24, 120, 350, 150],
                "cell_evidence_units": [
                    {"header": "ID", "content": row_id, "bbox": [24, 120, 120, 150]},
                    {"header": "Accuracy", "content": accuracy, "bbox": [120, 120, 230, 150]},
                ],
            }
        ],
    }


def test_candidate_ranking_does_not_guess_between_duplicate_table_labels():
    appendix = _bundle(page=18, row_id="Appendix", accuracy="77.0")
    main = _bundle(page=4, row_id="B", accuracy="91.2")

    ranked = verifier.rank_table_candidates(
        "Table 1 中 ID B 的 Accuracy 是多少？",
        [{
            "text": "Table 1 ID B Accuracy 91.2",
            "table_id": "Table 1",
            "table_bundle_id": main["bundle_id"],
            "page_range": [4, 4],
            "table_header": "ID | Accuracy",
            "cell_evidence_units": main["evidence_units"][0]["cell_evidence_units"],
        }],
        {"structured_table_bundles": [appendix, main]},
    )

    assert ranked[0]["page"] == 4
    assert ranked[0]["selection_score"] > ranked[1]["selection_score"]
    assert ranked[0]["table_instance_id"] != ranked[1]["table_instance_id"]


def test_visual_target_prefers_canonical_visual_bbox_over_parser_bbox():
    bundle = _bundle(bbox=[100, 200, 900, 420])
    bundle.update({
        "bbox_coordinate_space": "normalized_0_1000",
        "visual_bbox": [60, 160, 540, 336],
        "raw_bbox": [100, 200, 900, 420],
    })

    target = verifier._target_from_record(bundle, origin="bundle")

    assert target["bbox"] == [60.0, 160.0, 540.0, 336.0]


def test_schema_and_structured_cell_cross_check_return_all_verdicts():
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target["requested_rows"] = ["ours"]
    target["requested_columns"] = ["Accuracy"]
    target["selected_row"] = target["evidence_units"][0]
    verifier._ensure_target_metadata(target)

    confirmed = verifier._parse_visual_response(json.dumps({
        "table_id": "Table 1",
        "matched_row": "Ours",
        "cells": {"Accuracy": "91.2"},
        "confidence": 0.9,
    }))
    conflict = verifier._parse_visual_response(json.dumps({
        "table_id": "Table 1",
        "matched_row": "Ours",
        "cells": {"Accuracy": "89.0"},
        "confidence": 0.9,
    }))
    indeterminate = verifier._parse_visual_response(json.dumps({
        "table_id": "Table 1",
        "matched_row": "Ours",
        "cells": {"F1": "89.0"},
        "confidence": 0.9,
    }))

    assert confirmed is not None and verifier._evaluate_visual_result(confirmed, target, "ID Ours Accuracy")[0] == "confirmed"
    assert conflict is not None and verifier._evaluate_visual_result(conflict, target, "ID Ours Accuracy")[0] == "conflict"
    assert indeterminate is not None and verifier._evaluate_visual_result(indeterminate, target, "ID Ours Accuracy")[0] == "indeterminate"


def test_low_confidence_visual_result_is_indeterminate_even_when_cells_match():
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy"]})
    target["selected_row"] = target["evidence_units"][0]
    verifier._ensure_target_metadata(target)
    parsed = verifier._parse_visual_response(json.dumps({
        "table_id": "Table 1",
        "matched_row": "Ours",
        "cells": {"Accuracy": "91.2"},
        "confidence": 0.74,
    }))

    assert parsed is not None
    verdict, details = verifier._evaluate_visual_result(parsed, target, "ID Ours Accuracy")
    assert verdict == "indeterminate"
    assert details["reason"] == "confidence_below_threshold"
    assert details["minimum_confidence"] == 0.75


def test_visual_prompt_does_not_include_retrieved_text_values():
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy"]})

    prompt = verifier._build_visual_prompt(
        "Table 1 中 ID Ours 的 Accuracy 是多少？",
        target,
        [{"text": "Retrieved answer candidate: Ours | Accuracy | 91.2"}],
    )

    assert "91.2" not in prompt
    assert "Retrieved answer candidate" not in prompt


def test_confirmed_visual_segment_only_contains_evidence_aligned_cells():
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy"]})
    target["selected_row"] = target["evidence_units"][0]
    verifier._ensure_target_metadata(target)
    parsed = verifier._parse_visual_response(json.dumps({
        "table_id": "Table 1",
        "matched_row": "Ours",
        "cells": {"Accuracy": "91.2", "Unrelated": "should-not-enter-context"},
        "confidence": 0.9,
    }))

    assert parsed is not None
    verdict, details = verifier._evaluate_visual_result(parsed, target, "ID Ours Accuracy")
    assert verdict == "confirmed"
    segment = verifier._build_visual_segment(
        parsed,
        query="ID Ours Accuracy",
        target=target,
        crop_meta=[],
        response={},
        verified_cells=details["verified_cells"],
    )
    assert segment["visual_cells"] == {"Accuracy": "91.2"}
    assert "Unrelated" not in segment["text"]


def test_requested_columns_must_all_be_verified_and_text_capability_must_agree():
    bundle = _bundle()
    bundle["table_header"] = "ID | Accuracy | F1"
    bundle["evidence_units"][0]["cell_evidence_units"].append(
        {"header": "F1", "content": "88.1", "bbox": [230, 120, 330, 150]}
    )
    target = verifier._target_from_record(bundle, origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy", "F1"]})
    target["selected_row"] = target["evidence_units"][0]
    verifier._ensure_target_metadata(target)

    partial = verifier._parse_visual_response(json.dumps({
        "matched_row": "Ours",
        "cells": {"Accuracy": "91.2"},
        "confidence": 0.9,
    }))
    unsupported = verifier._parse_visual_response(json.dumps({
        "matched_row": "Ours",
        "cells": {"Accuracy": "91.2", "F1": "88.1"},
        "confidence": 0.9,
        "supports_text_result": False,
    }))

    assert partial is not None
    assert verifier._evaluate_visual_result(partial, target, "ID Ours Accuracy F1")[0] == "indeterminate"
    assert unsupported is not None
    assert verifier._evaluate_visual_result(unsupported, target, "ID Ours Accuracy F1")[0] == "indeterminate"


def test_cache_key_changes_with_source_prompt_and_model(monkeypatch):
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({
        "table_instance_id": "table-instance-a",
        "table_source_hash": "source-a",
        "requested_rows": ["ours"],
        "requested_columns": ["Accuracy"],
    })
    initial = verifier._cache_key("doc-a", "ID Ours Accuracy", target, "openai", "gpt-4o")
    changed_source = dict(target, table_source_hash="source-b")
    changed_model = verifier._cache_key("doc-a", "ID Ours Accuracy", target, "openai", "gpt-5")
    monkeypatch.setattr(verifier, "_VISUAL_PROMPT_VERSION", verifier._VISUAL_PROMPT_VERSION + 1)
    changed_prompt = verifier._cache_key("doc-a", "ID Ours Accuracy", target, "openai", "gpt-4o")

    assert initial != verifier._cache_key("doc-a", "ID Ours Accuracy", changed_source, "openai", "gpt-4o")
    assert initial != changed_model
    assert initial != changed_prompt


def test_cross_page_table_is_indeterminate_instead_of_being_merged_for_visual_review():
    bundle = _bundle(page=3)
    bundle.update({"page_end": 4, "pages": [3, 4]})
    target = verifier._target_from_record(bundle, origin="bundle")

    assert verifier._unsafe_cross_page_target(target) is True


def test_task_state_is_persisted_and_failed_tasks_enter_cooldown(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_CACHE_DIR", str(tmp_path))
    verifier._VISUAL_CACHE.clear()
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy"]})
    verifier._ensure_target_metadata(target)
    key = verifier._cache_key("doc-task", "ID Ours Accuracy", target, "openai", "gpt-4o")
    request = verifier._request_info("doc-task", "ID Ours Accuracy", target, "openai", "gpt-4o")

    queued = verifier._create_or_update_task(key, request, state="queued", diagnostics={"state": "queued"})
    status = verifier.get_table_visual_verification_status("doc-task", queued["task_id"])
    assert status["state"] == "queued"

    failed = verifier._complete_task(key, request, state="failed", diagnostics={"state": "failed", "error": "upstream"})
    assert verifier._failure_cooldown_active(failed)
    loaded = verifier.get_table_visual_verification_status("doc-task", failed["task_id"])
    assert loaded["state"] == "failed"
    assert loaded["diagnostics"]["state"] == "failed"


def test_stale_running_task_can_be_reclaimed(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_STALE_TASK_S", "30")
    verifier._VISUAL_CACHE.clear()
    target = verifier._target_from_record(_bundle(), origin="bundle")
    target.update({"requested_rows": ["ours"], "requested_columns": ["Accuracy"]})
    verifier._ensure_target_metadata(target)
    key = verifier._cache_key("doc-stale", "ID Ours Accuracy", target, "openai", "gpt-4o")
    request = verifier._request_info("doc-stale", "ID Ours Accuracy", target, "openai", "gpt-4o")
    running = verifier._create_or_update_task(key, request, state="running", diagnostics={"state": "running"})
    running["updated_at"] = 1
    verifier._VISUAL_CACHE[key] = running

    assert verifier._task_is_stale(verifier._load_task_record(key))


@pytest.mark.asyncio
async def test_sync_verification_uses_overview_header_and_target_row_crops(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_CACHE_DIR", str(tmp_path / "cache"))
    verifier._VISUAL_CACHE.clear()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"not-read-because-render-is-mocked")
    captured = {}

    def fake_render(_path, target, **_kwargs):
        captured["target"] = target
        return [
            {"role": "overview", "page": 3, "bbox": [1, 1, 2, 2], "dpi": 180, "width": 20, "height": 20, "image_b64": "a"},
            {"role": "header", "page": 3, "bbox": [1, 1, 2, 2], "dpi": 180, "width": 20, "height": 20, "image_b64": "b"},
            {"role": "target_row", "page": 3, "bbox": [1, 1, 2, 2], "dpi": 180, "width": 20, "height": 20, "image_b64": "c"},
        ]

    async def fake_call(messages, *_args, **_kwargs):
        captured["messages"] = messages
        return {
            "choices": [{"message": {"content": json.dumps({
                "table_id": "Table 1",
                "matched_row": "Ours",
                "cells": {"Accuracy": "91.2"},
                "confidence": 0.9,
            })}}],
            "_used_provider": "openai",
            "_used_model": "gpt-4o",
        }

    monkeypatch.setattr(verifier, "render_table_crops_base64", fake_render)
    monkeypatch.setattr(verifier, "call_ai_api", fake_call)
    bundle = _bundle()
    segment, diagnostics = await verifier.maybe_verify_numeric_table_visual(
        query="Table 1 中 ID Ours 的 Accuracy 是多少？",
        doc_id="doc-crops",
        doc_data={"structured_table_bundles": [bundle]},
        pdf_path=pdf_path,
        segments=[],
        api_key="key",
        model="gpt-4o",
        provider="openai",
        endpoint="",
        custom_params={"numeric_table_visual_verification": "always"},
        background=False,
    )

    assert diagnostics["state"] == "confirmed"
    assert segment["visual_verdict"] == "confirmed"
    assert [crop["role"] for crop in segment["visual_crops"]] == ["overview", "header", "target_row"]
    images = [item for item in captured["messages"][1]["content"] if item.get("type") == "image_url"]
    assert len(images) == 3


@pytest.mark.asyncio
async def test_background_task_persists_queued_then_terminal_status(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHATPDF_TABLE_VISUAL_CACHE_DIR", str(tmp_path / "cache"))
    verifier._VISUAL_CACHE.clear()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"placeholder")

    monkeypatch.setattr(verifier, "render_table_crops_base64", lambda *_args, **_kwargs: [
        {"role": "overview", "page": 3, "bbox": [1, 1, 2, 2], "dpi": 180, "width": 20, "height": 20, "image_b64": "a"},
    ])

    async def fake_call(*_args, **_kwargs):
        return {
            "choices": [{"message": {"content": json.dumps({
                "table_id": "Table 1",
                "matched_row": "Ours",
                "cells": {"Accuracy": "91.2"},
                "confidence": 0.9,
            })}}],
        }

    monkeypatch.setattr(verifier, "call_ai_api", fake_call)
    bundle = _bundle()
    segment, diagnostics = await verifier.maybe_verify_numeric_table_visual(
        query="Table 1 中 ID Ours 的 Accuracy 是多少？",
        doc_id="doc-background",
        doc_data={"structured_table_bundles": [bundle]},
        pdf_path=pdf_path,
        segments=[{"text": "Table 1 ID Ours Accuracy 91.2", "table_header": "ID | Accuracy"}],
        api_key="key",
        model="gpt-4o",
        provider="openai",
        endpoint="",
        custom_params={"numeric_table_visual_verification": "auto"},
        background=True,
    )

    assert segment == {}
    assert diagnostics["state"] == "queued"
    for _ in range(20):
        status = verifier.get_table_visual_verification_status("doc-background", diagnostics["task_id"])
        if status.get("state") in {"confirmed", "conflict", "indeterminate", "failed"}:
            break
        await asyncio.sleep(0.01)
    assert status["state"] == "confirmed"
