from concurrent.futures import ThreadPoolExecutor

import pytest

from routes.document_routes import _merge_odl_pages_with_existing_ocr
from services import ocr_service
from services.ocr_service import Doc2XAdapter, MinerUAdapter, OCRRegistry, get_ocr_service


class _Adapter:
    def __init__(self, name: str):
        self.name = name

    def is_available(self) -> bool:
        return True


def test_auto_registry_selects_local_ocr_without_silent_cloud_fallback():
    registry = OCRRegistry()
    registry.register(_Adapter("mistral"))
    registry.register(_Adapter("mineru"))
    registry.register(_Adapter("tesseract"))

    assert registry.get_adapter("auto").name == "tesseract"
    assert registry.get_adapter("mistral").name == "mistral"
    assert registry.get_adapter("mineru") is None


def test_document_parsers_cannot_be_used_as_page_ocr_providers():
    mineru = MinerUAdapter(worker_url="", token="")
    doc2x = Doc2XAdapter(worker_url="", token="")

    with pytest.raises(RuntimeError, match="不支持逐页 OCR"):
        mineru.ocr_pages(b"pdf", [0, 1])
    with pytest.raises(RuntimeError, match="不支持逐页 OCR"):
        doc2x.ocr_pages(b"pdf", [0, 1])
    with pytest.raises(ValueError, match="文档级解析器"):
        get_ocr_service("mineru")


def test_odl_merge_preserves_successful_ocr_page_content():
    existing = [
        {"page": 1, "content": "native page", "text": "native page", "source": "pdf_native"},
        {"page": 2, "content": "OCR page", "text": "OCR page", "source": "ocr", "ocr_backend": "mistral"},
    ]
    odl = [
        {"page": 1, "content": "clean native", "text": "clean native", "source": "odl"},
        {"page": 2, "content": "weak ODL", "text": "weak ODL", "source": "odl", "table_bundles": [{"bundle_id": "t"}]},
    ]

    merged, preserved = _merge_odl_pages_with_existing_ocr(existing, odl)

    assert preserved is True
    assert merged[0]["content"] == "clean native"
    assert merged[1]["content"] == "OCR page"
    assert merged[1]["ocr_backend"] == "mistral"
    assert merged[1]["table_bundles"] == [{"bundle_id": "t"}]


def test_provider_usage_is_persisted_for_sunset_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_service, "_OCR_PROVIDER_USAGE_PATH", tmp_path / "usage.json")

    ocr_service.record_ocr_provider_use("doc2x")
    ocr_service.record_ocr_provider_use("doc2x")

    usage = ocr_service.get_ocr_provider_usage("doc2x")
    assert usage["count"] == 2
    assert usage["attempt_count"] == 2
    assert usage["success_count"] == 2
    assert usage["operations"]["page_ocr"]["success_count"] == 2
    assert usage["last_used_at"]


def test_provider_usage_tracks_failures_and_actual_fallback_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_service, "_OCR_PROVIDER_USAGE_PATH", tmp_path / "usage.json")

    ocr_service.record_ocr_provider_use("mistral", outcome="failure", operation="page_ocr")
    ocr_service.record_ocr_provider_use("paddleocr", outcome="success", operation="page_ocr", fallback=True)
    ocr_service.record_ocr_provider_use("mineru", outcome="success", operation="document_parse")

    mistral = ocr_service.get_ocr_provider_usage("mistral")
    paddle = ocr_service.get_ocr_provider_usage("paddleocr")
    mineru = ocr_service.get_ocr_provider_usage("mineru")
    assert mistral["attempt_count"] == 1
    assert mistral["failure_count"] == 1
    assert mistral["success_count"] == 0
    assert paddle["fallback_success_count"] == 1
    assert paddle["operations"]["page_ocr"]["fallback_success_count"] == 1
    assert mineru["operations"]["document_parse"]["success_count"] == 1


def test_provider_usage_concurrent_updates_do_not_drop_successes(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_service, "_OCR_PROVIDER_USAGE_PATH", tmp_path / "usage.json")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: ocr_service.record_ocr_provider_use("doc2x"), range(32)))

    usage = ocr_service.get_ocr_provider_usage("doc2x")
    assert usage["count"] == 32
    assert usage["attempt_count"] == 32
    assert usage["success_count"] == 32
