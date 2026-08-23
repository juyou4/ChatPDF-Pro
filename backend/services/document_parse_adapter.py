"""Document-level parser contract, separate from page OCR adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from services.mineru_text_normalizer import MINERU_MIN_PAGE_COVERAGE


ProgressCallback = Optional[Callable[[dict[str, Any]], None]]


class MinerUQualityError(RuntimeError):
    """Structured quality-gate failure for MinerU publication."""

    def __init__(
        self,
        message: str,
        *,
        quality_status: str = "failed",
        expected_page_count: int = 0,
        covered_page_count: int = 0,
        coverage: float = 0.0,
        failed_pages: list[int] | None = None,
        page_ledger: list[dict[str, Any]] | None = None,
        block_count: int = 0,
        body_text_chars: int = 0,
        structure_degraded: bool = False,
    ) -> None:
        super().__init__(message)
        self.quality_status = str(quality_status or "failed")
        self.expected_page_count = max(0, int(expected_page_count or 0))
        self.covered_page_count = max(0, int(covered_page_count or 0))
        self.coverage = float(coverage or 0.0)
        self.failed_pages = [
            int(page)
            for page in (failed_pages or [])
            if isinstance(page, (int, float, str)) and str(page).strip()
        ]
        self.page_ledger = [
            dict(item)
            for item in (page_ledger or [])
            if isinstance(item, Mapping)
        ]
        self.block_count = max(0, int(block_count or 0))
        self.body_text_chars = max(0, int(body_text_chars or 0))
        self.structure_degraded = bool(structure_degraded)

    def as_status_dict(self) -> dict[str, Any]:
        return {
            "quality_status": self.quality_status,
            "expected_page_count": self.expected_page_count,
            "covered_page_count": self.covered_page_count,
            "coverage": self.coverage,
            "failed_pages": list(self.failed_pages),
            "page_ledger": list(self.page_ledger),
            "block_count": self.block_count,
            "body_text_chars": self.body_text_chars,
            "structure_degraded": self.structure_degraded,
        }


def _coerce_quality_details(normalized: Any) -> dict[str, Any]:
    meta = {}
    if isinstance(normalized, Mapping):
        raw_meta = normalized.get("mineru_meta")
        if isinstance(raw_meta, Mapping):
            meta = dict(raw_meta)

    def _to_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _to_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "quality_status": str(meta.get("quality_status") or "failed"),
        "expected_page_count": _to_int(meta.get("expected_page_count")),
        "covered_page_count": _to_int(meta.get("covered_page_count")),
        "coverage": _to_float(meta.get("coverage")),
        "failed_pages": [
            _to_int(page)
            for page in (meta.get("failed_pages") or [])
            if _to_int(page) > 0
        ],
        "page_ledger": [
            dict(item)
            for item in (meta.get("page_ledger") or [])
            if isinstance(item, Mapping)
        ],
        "block_count": _to_int(meta.get("block_count")),
        "body_text_chars": _to_int(meta.get("body_text_chars")),
        "structure_degraded": bool(meta.get("structure_degraded")),
    }


_QUALITY_ERROR_FIELDS = (
    "quality_status",
    "expected_page_count",
    "covered_page_count",
    "coverage",
    "failed_pages",
    "page_ledger",
    "block_count",
    "body_text_chars",
    "structure_degraded",
)


def _quality_error_kwargs(quality: Mapping[str, Any]) -> dict[str, Any]:
    return {key: quality[key] for key in _QUALITY_ERROR_FIELDS if key in quality}


def collect_mineru_block_validation(normalized: Any) -> dict[str, Any]:
    """Heading/outline diagnostics for jobs and quality_report; never a publish reject."""
    quality = _coerce_quality_details(normalized)
    meta: dict[str, Any] = {}
    if isinstance(normalized, Mapping):
        raw_meta = normalized.get("mineru_meta")
        if isinstance(raw_meta, Mapping):
            meta = dict(raw_meta)

    def _to_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    missing_run_in_paths = [
        str(item).strip()
        for item in (meta.get("missing_run_in_outline_paths") or [])
        if str(item).strip()
    ]
    return {
        **quality,
        "silently_dropped_heading_count": _to_int(meta.get("silently_dropped_heading_count")),
        "raw_text_level_heading_count": _to_int(meta.get("raw_text_level_heading_count")),
        "covered_text_level_heading_count": _to_int(meta.get("covered_text_level_heading_count")),
        "raw_native_heading_count": _to_int(meta.get("raw_native_heading_count")),
        "covered_native_heading_count": _to_int(meta.get("covered_native_heading_count")),
        "raw_run_in_heading_count": _to_int(meta.get("raw_run_in_heading_count")),
        "covered_run_in_heading_count": _to_int(meta.get("covered_run_in_heading_count")),
        "missing_run_in_outline_paths": missing_run_in_paths,
        "outline_heading_count": _to_int(meta.get("outline_heading_count")),
        "emitted_heading_count": _to_int(
            meta.get("emitted_heading_count") or meta.get("emitted_heading_block_count")
        ),
        "outline_is_fallback_only": bool(meta.get("outline_is_fallback_only")),
        "flat_structure_without_headings": bool(meta.get("flat_structure_without_headings")),
    }


def validate_mineru_block_index_quality(normalized: Any) -> None:
    """Fail closed only when pages, blocks, or body text are missing.

    Outline/heading shortfalls stay on ``structure_degraded`` for navigation
    recovery. They must not refuse a parse that MinerU already completed.
    """
    quality = _coerce_quality_details(normalized)
    failed_quality = {**_quality_error_kwargs(quality), "quality_status": "failed"}
    if not isinstance(normalized, Mapping):
        raise MinerUQualityError("MinerU 结果标准化格式无效", **failed_quality)
    pages = normalized.get("pages")
    if not isinstance(pages, list) or not pages:
        raise MinerUQualityError("MinerU 结果没有可发布的页面", **failed_quality)

    block_count = quality["block_count"]
    if block_count <= 0:
        raise MinerUQualityError("MinerU 结果没有可发布的版面块", **failed_quality)

    body_text_chars = quality["body_text_chars"]
    if body_text_chars <= 0:
        raise MinerUQualityError("MinerU 结果没有可发布的正文", **failed_quality)

    expected_page_count = quality["expected_page_count"]
    coverage = quality["coverage"]
    if expected_page_count <= 0:
        # 页数未知时覆盖率的分母是 MinerU 自己返回的页，coverage 恒为 1.0，
        # 全覆盖契约等于被跳过。无法核对就不能发布。
        raise MinerUQualityError("MinerU 无法确认源文档页数，拒绝发布", **failed_quality)
    if coverage < MINERU_MIN_PAGE_COVERAGE:
        raise MinerUQualityError("MinerU 页面覆盖不完整，拒绝发布", **failed_quality)
    if quality["failed_pages"]:
        raise MinerUQualityError("MinerU 存在失败页面，拒绝发布", **failed_quality)
    if quality["quality_status"] != "success":
        raise MinerUQualityError("MinerU 解析质量未达到完整发布标准", **failed_quality)


@dataclass(frozen=True)
class DocumentParseSubmission:
    provider: str
    job_id: str
    data_id: str = ""
    access_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    inline_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class DocumentParseArtifact:
    provider: str
    submission: DocumentParseSubmission
    raw_payload: dict[str, Any]
    normalized: Any


class DocumentParseAdapter(ABC):
    """Required lifecycle for any future whole-document parser service."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def submit(
        self,
        pdf_bytes: bytes,
        *,
        progress_callback: ProgressCallback = None,
        cancel_event: Any = None,
    ) -> DocumentParseSubmission:
        ...

    @abstractmethod
    def poll(
        self,
        submission: DocumentParseSubmission,
        *,
        progress_callback: ProgressCallback = None,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def normalize(
        self,
        submission: DocumentParseSubmission,
        payload: Mapping[str, Any],
        *,
        normalizer: Callable[[dict[str, Any]], Any],
    ) -> DocumentParseArtifact:
        ...

    @abstractmethod
    def publish(
        self,
        artifact: DocumentParseArtifact,
        *,
        publisher: Callable[[DocumentParseArtifact], Any],
    ) -> Any:
        ...

    @abstractmethod
    def invalidate(self, *, invalidator: Callable[[], Any]) -> Any:
        ...


class MinerUDocumentParseAdapter(DocumentParseAdapter):
    """Lifecycle adapter around Worker and official MinerU transports."""

    def __init__(self, transport: Any):
        self.transport = transport

    @property
    def name(self) -> str:
        return "mineru"

    def is_available(self) -> bool:
        return bool(self.transport and self.transport.is_available())

    def submit(
        self,
        pdf_bytes: bytes,
        *,
        progress_callback: ProgressCallback = None,
        cancel_event: Any = None,
    ) -> DocumentParseSubmission:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("MinerU 深度解析已取消")
        submit_document = getattr(self.transport, "submit_document", None)
        if callable(submit_document):
            value = submit_document(pdf_bytes, progress_callback=progress_callback)
            return self._submission(value)

        # Compatibility for injected transports during rollout. Real MinerU
        # transports implement split submit/poll below.
        payload = self.transport.analyze_pdf(
            pdf_bytes,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        return self._submission(payload, inline_payload=dict(payload or {}))

    def poll(
        self,
        submission: DocumentParseSubmission,
        *,
        progress_callback: ProgressCallback = None,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        if submission.inline_payload is not None:
            return dict(submission.inline_payload)
        poll_document = getattr(self.transport, "poll_document", None)
        if callable(poll_document):
            return dict(poll_document(
                {
                    "batch_id": submission.job_id,
                    "data_id": submission.data_id,
                    "access_mode": submission.access_mode,
                    **dict(submission.metadata or {}),
                },
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            ) or {})
        return dict(self.transport.resume_batch(
            submission.job_id,
            data_id=submission.data_id,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        ) or {})

    def normalize(
        self,
        submission: DocumentParseSubmission,
        payload: Mapping[str, Any],
        *,
        normalizer: Callable[[dict[str, Any]], Any],
    ) -> DocumentParseArtifact:
        raw_payload = dict(payload or {})
        normalized = normalizer(raw_payload)
        if normalized is None:
            raise RuntimeError("MinerU 结果标准化失败")
        self._validate_normalized(normalized)
        return DocumentParseArtifact(
            provider=self.name,
            submission=submission,
            raw_payload=raw_payload,
            normalized=normalized,
        )

    def publish(
        self,
        artifact: DocumentParseArtifact,
        *,
        publisher: Callable[[DocumentParseArtifact], Any],
    ) -> Any:
        return publisher(artifact)

    def invalidate(self, *, invalidator: Callable[[], Any]) -> Any:
        return invalidator()

    def cancel_batch(self, batch_id: str, *, data_id: str = "") -> dict:
        cancel = getattr(self.transport, "cancel_batch", None)
        if not callable(cancel):
            return {"attempted": False, "state": "unsupported"}
        return dict(cancel(batch_id, data_id=data_id) or {})

    def _submission(
        self,
        value: Mapping[str, Any] | None,
        *,
        inline_payload: dict[str, Any] | None = None,
    ) -> DocumentParseSubmission:
        data = dict(value or {})
        job_id = str(data.get("batch_id") or data.get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError("MinerU 提交结果缺少 batch_id")
        return DocumentParseSubmission(
            provider=self.name,
            job_id=job_id,
            data_id=str(data.get("data_id") or ""),
            access_mode=str(data.get("access_mode") or ""),
            metadata={
                key: data.get(key)
                for key in ("full_zip_url",)
                if data.get(key) not in (None, "")
            },
            inline_payload=inline_payload,
        )

    @staticmethod
    def _validate_normalized(normalized: Any) -> None:
        validate_mineru_block_index_quality(normalized)
