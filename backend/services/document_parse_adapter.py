"""Document-level parser contract, separate from page OCR adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


ProgressCallback = Optional[Callable[[dict[str, Any]], None]]


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
