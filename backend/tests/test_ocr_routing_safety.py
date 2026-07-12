from routes.document_routes import _merge_odl_pages_with_existing_ocr
from services import ocr_service
from services.ocr_service import OCRRegistry


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
    assert usage["last_used_at"]
