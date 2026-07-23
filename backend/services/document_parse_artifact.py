"""Provider-neutral, versioned artifacts for document parsing results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOCUMENT_PARSE_ARTIFACT_VERSION = 2


def derive_table_geometry_capabilities(tables: list[dict] | None) -> dict[str, bool]:
    """Describe geometry that is actually safe for visual table verification.

    A table overview bbox is useful for inspection, but it does not prove where
    a requested row or cell lives.  Keep those capabilities separate so parser
    adapters cannot overstate crop precision.
    """
    has_overview = False
    has_row = False
    has_cell = False
    for table in tables or []:
        if not isinstance(table, dict):
            continue
        if _valid_bbox(table.get("visual_bbox")):
            has_overview = True
        for row in table.get("evidence_units") or []:
            if not isinstance(row, dict):
                continue
            if row.get("visual_crop_eligible") is True and _valid_bbox(row.get("visual_bbox")):
                has_row = True
            for cell in row.get("cell_evidence_units") or []:
                if isinstance(cell, dict) and cell.get("visual_crop_eligible") is True and _valid_bbox(cell.get("visual_bbox")):
                    has_cell = True
    return {
        # Retains the existing capability name for exact table evidence, rather
        # than treating a table outline as a row/cell geometry guarantee.
        "table_geometry": has_row or has_cell,
        "table_overview_geometry": has_overview,
        "table_row_geometry": has_row,
        "table_cell_geometry": has_cell,
    }


def build_document_parse_artifact(
    *,
    doc_id: str,
    provider: str,
    provider_version: str,
    pages: list[dict],
    tables: list[dict],
    figures: list[dict] | None = None,
    warnings: list[str] | None = None,
    capabilities: dict[str, bool] | None = None,
    source_hash: str = "",
    raw_ref: str = "",
) -> dict[str, Any]:
    """Create a canonical artifact without taking ownership of provider raw data."""
    safe_pages = [dict(page) for page in pages or [] if isinstance(page, dict)]
    safe_tables = [dict(table) for table in tables or [] if isinstance(table, dict)]
    safe_figures = [dict(figure) for figure in figures or [] if isinstance(figure, dict)]
    safe_capabilities = dict(capabilities or {})
    if "figures" in safe_capabilities:
        safe_capabilities["figures"] = bool(safe_figures)
    artifact_source_hash = source_hash or _artifact_source_hash(
        provider=provider,
        provider_version=provider_version,
        pages=safe_pages,
        tables=safe_tables,
        figures=safe_figures,
    )
    return {
        "schema_version": DOCUMENT_PARSE_ARTIFACT_VERSION,
        "doc_id": doc_id,
        "source_hash": artifact_source_hash,
        "provider": str(provider or "unknown"),
        "provider_version": str(provider_version or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": safe_capabilities,
        "pages": safe_pages,
        "tables": safe_tables,
        "figures": safe_figures,
        "warnings": [str(warning) for warning in warnings or [] if str(warning).strip()],
        "raw_ref": str(raw_ref or ""),
    }


def persist_document_parse_artifact(data_dir: Path | str, artifact: dict[str, Any]) -> Path:
    """Atomically store an artifact under a provider/source-hash namespace."""
    doc_id = _safe_path_part(artifact.get("doc_id"), "document")
    provider = _safe_path_part(artifact.get("provider"), "unknown")
    source_hash = _safe_path_part(artifact.get("source_hash"), "artifact")
    path = Path(data_dir) / "parse_artifacts" / doc_id / provider / f"{source_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def artifact_reference(data_dir: Path | str, path: Path | str) -> str:
    """Return a portable reference rooted at ChatPDF's data directory."""
    try:
        return str(Path(path).resolve().relative_to(Path(data_dir).resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _artifact_source_hash(
    *,
    provider: str,
    provider_version: str,
    pages: list[dict],
    tables: list[dict],
    figures: list[dict],
) -> str:
    payload = {
        "provider": provider,
        "provider_version": provider_version,
        "pages": pages,
        "tables": tables,
        "figures": figures,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_path_part(value: Any, fallback: str) -> str:
    text = "".join(char for char in str(value or "") if char.isalnum() or char in {"-", "_", "."})
    return text[:128] or fallback


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        x0, y0, x1, y1 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return False
    return x1 > x0 and y1 > y0
