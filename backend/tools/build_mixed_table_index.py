"""Build an offline mixed-table RAG index for evaluation.

Example:
    python tools/build_mixed_table_index.py \
        --doc-id <doc_id> \
        --embedding-model BAAI/bge-m3 \
        --embedding-api-key <key> \
        --embedding-api-host https://api.siliconflow.cn/v1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.embedding_service import (  # noqa: E402
    _build_semantic_group_index,
    get_embedding_function,
)
from services.table_source_selector import (  # noqa: E402
    MIXED_TABLE_RAG_INDEX_SOURCE,
    MIXED_TABLE_SELECTOR_VERSION,
    build_mixed_table_rag_data,
)
from services.vector_service import create_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mixed table-source RAG index for selected documents.")
    parser.add_argument("--doc-id", action="append", default=[], help="Document id. Can be repeated.")
    parser.add_argument("--doc-ids", default="", help="Comma-separated document ids.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-api-key", default=os.getenv("EMBEDDING_API_KEY", ""))
    parser.add_argument("--embedding-api-host", default=os.getenv("EMBEDDING_API_HOST", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--summary-api-key", default="")
    parser.add_argument("--summary-model", default="gpt-4o-mini")
    parser.add_argument("--summary-provider", default="openai")
    parser.add_argument("--summary-api-host", default="")
    parser.add_argument("--base-source", default="mineru_vlm", choices=["mineru_vlm", "mineru_pipeline", "pdf_native"])
    parser.add_argument("--report", default=str(PROJECT_ROOT / "temp" / "mixed_table_selector_report.json"))
    parser.add_argument("--no-activate-doc", action="store_true", help="Do not write mixed data back to data/docs/{doc_id}.json")
    parser.add_argument("--skip-semantic-groups", action="store_true")
    args = parser.parse_args()

    doc_ids = list(args.doc_id)
    if args.doc_ids:
        doc_ids.extend([part.strip() for part in args.doc_ids.split(",") if part.strip()])
    doc_ids = list(dict.fromkeys(doc_ids))
    if not doc_ids:
        parser.error("At least one --doc-id or --doc-ids is required")

    data_dir = Path(args.data_dir)
    docs_dir = data_dir / "docs"
    vector_dir = data_dir / "vector_stores"
    semantic_dir = data_dir / "semantic_groups"
    vector_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "selector_version": MIXED_TABLE_SELECTOR_VERSION,
        "index_source": MIXED_TABLE_RAG_INDEX_SOURCE,
        "documents": {},
    }

    for doc_id in doc_ids:
        print(f"[mixed-table] building {doc_id}")
        sources = _load_source_documents(docs_dir, doc_id)
        mixed = build_mixed_table_rag_data(sources, base_source=args.base_source)
        _backup_vlm_doc_if_needed(docs_dir, doc_id, sources["mineru_vlm"])

        create_index(
            doc_id,
            mixed.get("full_text", ""),
            str(vector_dir),
            args.embedding_model,
            args.embedding_api_key,
            args.embedding_api_host,
            pages=mixed.get("pages") or [],
            structured_table_bundles=mixed.get("structured_table_bundles") or [],
            summary_api_key=args.summary_api_key,
            summary_model=args.summary_model,
            summary_provider=args.summary_provider,
            summary_api_host=args.summary_api_host,
            index_source=MIXED_TABLE_RAG_INDEX_SOURCE,
            index_meta={
                "source_hash": mixed.get("source_hash", ""),
                "normalizer_version": mixed.get("normalizer_version", ""),
                "table_selector_version": MIXED_TABLE_SELECTOR_VERSION,
                "base_source": mixed.get("base_source", ""),
                "selected_table_count": len(mixed.get("structured_table_bundles") or []),
                "selected_source_counts": (mixed.get("quality_report") or {}).get("selected_source_counts", {}),
            },
            build_semantic_groups=False,
        )

        chunks = _load_current_chunks(vector_dir, doc_id)
        if not args.skip_semantic_groups:
            embed_fn = get_embedding_function(args.embedding_model, args.embedding_api_key, args.embedding_api_host)
            _remove_semantic_artifacts(semantic_dir, doc_id)
            _build_semantic_group_index(
                doc_id=doc_id,
                chunks=chunks,
                pages=mixed.get("pages") or [],
                embed_fn=embed_fn,
                api_key=args.summary_api_key,
                model=args.summary_model,
                provider=args.summary_provider,
                endpoint=args.summary_api_host,
            )

        if not args.no_activate_doc:
            _write_active_mixed_doc(docs_dir, doc_id, sources["mineru_vlm"], mixed)

        quality = mixed.get("quality_report") or {}
        report["documents"][doc_id] = {
            "source_hash": mixed.get("source_hash", ""),
            "base_source": mixed.get("base_source", ""),
            "table_count": len(mixed.get("structured_table_bundles") or []),
            "selected_source_counts": quality.get("selected_source_counts", {}),
            "source_table_counts": quality.get("source_table_counts", {}),
            "decisions": quality.get("decisions", []),
            "chunk_count": len(chunks),
        }
        print(f"[mixed-table] {doc_id}: tables={report['documents'][doc_id]['table_count']} chunks={len(chunks)} selected={report['documents'][doc_id]['selected_source_counts']}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mixed-table] report saved: {report_path}")
    return 0


def _load_source_documents(docs_dir: Path, doc_id: str) -> dict[str, dict[str, Any]]:
    vlm_path = docs_dir / f"{doc_id}.mineru_vlm.bak.doc.json"
    if not vlm_path.exists():
        vlm_path = docs_dir / f"{doc_id}.json"
    paths = {
        "mineru_vlm": vlm_path,
        "mineru_pipeline": docs_dir / f"{doc_id}.mineru.bak.doc.json",
        "pdf_native": docs_dir / f"{doc_id}.pdf_native.bak.doc.json",
    }
    missing = [f"{label}:{path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source documents for {doc_id}: {', '.join(missing)}")
    return {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in paths.items()
    }


def _backup_vlm_doc_if_needed(docs_dir: Path, doc_id: str, vlm_doc: dict[str, Any]) -> None:
    path = docs_dir / f"{doc_id}.mineru_vlm.bak.doc.json"
    if path.exists():
        return
    path.write_text(json.dumps(vlm_doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_active_mixed_doc(docs_dir: Path, doc_id: str, base_doc: dict[str, Any], mixed: dict[str, Any]) -> None:
    doc = dict(base_doc)
    data = dict(doc.get("data") or {})
    pages = []
    for page in mixed.get("pages") or []:
        page_copy = dict(page)
        content = str(page_copy.get("content") or page_copy.get("text") or "")
        page_copy["content"] = content
        page_copy["text"] = content
        page_copy["source"] = MIXED_TABLE_RAG_INDEX_SOURCE
        pages.append(page_copy)
    data.update({
        "full_text": mixed.get("full_text", ""),
        "pages": pages,
        "total_pages": len(pages) or data.get("total_pages", 0),
        "structured_table_bundles": mixed.get("structured_table_bundles") or [],
        "structured_table_count": len(mixed.get("structured_table_bundles") or []),
        "rag_index_source": MIXED_TABLE_RAG_INDEX_SOURCE,
        "rag_source_hash": mixed.get("source_hash", ""),
        "rag_normalizer_version": mixed.get("normalizer_version", ""),
        "rag_quality_report": mixed.get("quality_report") or {},
    })
    doc["data"] = data
    (docs_dir / f"{doc_id}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_current_chunks(vector_dir: Path, doc_id: str) -> list[str]:
    pkl_path = vector_dir / f"{doc_id}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    chunks = data.get("chunks") if isinstance(data, dict) else []
    return [str(chunk or "") for chunk in chunks if str(chunk or "").strip()]


def _remove_semantic_artifacts(semantic_dir: Path, doc_id: str) -> None:
    for suffix in (".json", "_groups.index", "_groups.pkl"):
        path = semantic_dir / f"{doc_id}{suffix}"
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
