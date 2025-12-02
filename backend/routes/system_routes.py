import glob
import os
import platform
from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/version")
async def get_version():
    return {"version": "2.0.1", "build_time": "2025-11-25 19:30:00", "feature": "native_pdf_url"}


@router.get("/storage_info")
async def get_storage_info():
    uploads_path = os.path.abspath("uploads")
    data_path = os.path.abspath("data")
    docs_path = os.path.abspath(os.path.join("data", "docs"))
    vector_stores_path = os.path.abspath(os.path.join("data", "vector_stores"))

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
