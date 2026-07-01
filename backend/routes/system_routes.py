import glob
import os
import platform
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from runtime_mode import runtime

router = APIRouter()

# 目录策略与 app.py/document_routes.py 保持一致：
# - desktop: 使用 Electron 传入的 runtime.data_dir
# - server: 使用项目根目录 data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if runtime.is_desktop:
    DATA_DIR = Path(runtime.data_dir)
else:
    DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
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
