import glob
import os
import platform
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

# Align storage paths with project root (same as app.py/document_routes)
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/version")
async def get_version():
    return {"version": "2.0.1", "build_time": "2025-11-25 19:30:00", "feature": "native_pdf_url"}


@router.get("/storage_info")
async def get_storage_info():
    uploads_path = str(UPLOADS_DIR.resolve())
    data_path = str(DATA_DIR.resolve())
    docs_path = str(DOCS_DIR.resolve())
    vector_stores_path = str(VECTOR_STORES_DIR.resolve())

    pdf_count = len(glob.glob(os.path.join(uploads_path, "*.pdf")))
    doc_count = len(glob.glob(os.path.join(docs_path, "*.json")))

    return {
        "uploads_dir": uploads_path,
        "data_dir": data_path,
        "docs_dir": docs_path,
        "vector_stores_dir": vector_stores_path,
        "pdf_count": pdf_count,
        "doc_count": doc_count,
        "platform": platform.system()
    }
