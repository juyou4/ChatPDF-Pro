import io
import os
import glob
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import PyPDF2
import pdfplumber
from fastapi import APIRouter, UploadFile, File, HTTPException

from services.vector_service import create_index
from models.model_detector import normalize_embedding_model_id

router = APIRouter()

# Project root (two levels up from routes/) so storage matches app.py
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
UPLOAD_DIR = BASE_DIR / "uploads"

# Legacy paths from the old layout (stored under backend/)
LEGACY_BASE_DIR = Path(__file__).resolve().parents[1]
LEGACY_DATA_DIR = LEGACY_BASE_DIR / "data"
LEGACY_DOCS_DIR = LEGACY_DATA_DIR / "docs"
LEGACY_VECTOR_STORE_DIR = LEGACY_DATA_DIR / "vector_stores"
LEGACY_UPLOAD_DIR = LEGACY_BASE_DIR / "uploads"

documents_store = {}


def save_document(doc_id: str, data: dict):
    try:
        file_path = DOCS_DIR / f"{doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved document {doc_id} to {file_path}")
    except Exception as e:
        print(f"Error saving document {doc_id}: {e}")


def load_documents():
    print("Loading documents from disk...")
    count = 0
    for file_path in glob.glob(str(DOCS_DIR / "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                documents_store[doc_id] = data
                count += 1
        except Exception as e:
            print(f"Error loading document from {file_path}: {e}")
    print(f"Loaded {count} documents.")


def migrate_legacy_storage():
    """Move files from old backend/* paths to project root if needed."""
    migrations = [
        (LEGACY_DOCS_DIR, DOCS_DIR, "*.json"),
        (LEGACY_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.index"),
        (LEGACY_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.pkl"),
        (LEGACY_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
    ]

    for src_dir, dest_dir, pattern in migrations:
        if not src_dir.exists() or src_dir.resolve() == dest_dir.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.glob(pattern):
            dest_file = dest_dir / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)


def generate_doc_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def extract_text_from_pdf(pdf_file):
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "content": text})
            full_text += text + "\n\n"

    reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(reader.pages)

    return {
        "full_text": full_text,
        "total_pages": total_pages,
        "pages": pages
    }


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = "local-minilm",
    embedding_api_key: str = None,
    embedding_api_host: str = None
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    try:
        content = await file.read()
        pdf_file = io.BytesIO(content)

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置或格式不正确（建议使用 provider:model 格式）")
        embedding_model = normalized_model

        extracted_data = extract_text_from_pdf(pdf_file)

        doc_id = generate_doc_id(extracted_data["full_text"])

        pdf_filename = f"{doc_id}.pdf"
        pdf_path = UPLOAD_DIR / pdf_filename
        with open(pdf_path, "wb") as f:
            f.write(content)

        pdf_url = f"/uploads/{pdf_filename}"

        documents_store[doc_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": pdf_url
        }

        save_document(doc_id, documents_store[doc_id])

        create_index(doc_id, extracted_data["full_text"], str(VECTOR_STORE_DIR), embedding_model, embedding_api_key, embedding_api_host)

        return {
            "message": "PDF上传成功",
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted_data["total_pages"],
            "total_chars": len(extracted_data["full_text"]),
            "pdf_url": pdf_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")


@router.get("/document/{doc_id}")
async def get_document(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "upload_time": doc["upload_time"],
        "total_pages": doc["data"]["total_pages"],
        "total_chars": len(doc["data"]["full_text"]),
        "pages": doc["data"]["pages"],
        "pdf_url": doc.get("pdf_url")
    }


# initialize
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
migrate_legacy_storage()
load_documents()
