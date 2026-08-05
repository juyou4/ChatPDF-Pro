import glob
import os
import platform
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from runtime_mode import runtime
from services.build_identity import get_public_build_identity

router = APIRouter()

# 目录策略与 app.py/document_routes.py 保持一致：显式 CHATPDF_DATA_DIR 优先；
# 未配置时 desktop 使用 Electron 数据目录，server 使用项目根目录 data/。
DATA_DIR = Path(runtime.data_dir)
UPLOADS_DIR = DATA_DIR / "uploads"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"


@router.get("/health")
async def health_check():
    build = get_public_build_identity()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": build.get("version"),
        "git_short_sha": build.get("git_short_sha"),
        "build_dirty": build.get("build_dirty"),
    }


@router.get("/version")
async def get_version():
    return get_public_build_identity()


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
