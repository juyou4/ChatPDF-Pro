"""
ChatPDF backend - main app entry mounting all routers.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models.model_registry import EMBEDDING_MODELS
from models.dynamic_store import load_dynamic_models
from routes.model_provider_routes import router as model_provider_router
from routes.system_routes import router as system_router
from routes.document_routes import router as document_router, documents_store
from routes.search_routes import router as search_router
from routes.chat_routes import router as chat_router
from routes.summary_routes import router as summary_router
from routes.glossary_routes import router as glossary_router
from routes.prompt_pool_routes import router as prompt_pool_router

# Directories (resolve to project root so frontend/backend共用同一份数据)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ChatPDF Pro with Vision API")

# Routers
app.include_router(model_provider_router)
app.include_router(system_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(summary_router)
app.include_router(glossary_router)
app.include_router(prompt_pool_router)

# Inject shared stores/paths to routers that need them
search_router.documents_store = documents_store
chat_router.documents_store = documents_store
summary_router.documents_store = documents_store
search_router.vector_store_dir = str(VECTOR_STORE_DIR)
chat_router.vector_store_dir = str(VECTOR_STORE_DIR)
summary_router.vector_store_dir = str(VECTOR_STORE_DIR)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static for PDFs
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/embedding_models")
async def get_embedding_models(as_list: bool = False):
    """获取可用嵌入模型列表；可返回标准化列表"""
    merged_models = {**EMBEDDING_MODELS, **load_dynamic_models()}
    if not as_list:
        return merged_models

    items = []
    for key, cfg in merged_models.items():
        provider = cfg.get("provider", "openai")
        full_id = key if ":" in key else f"{provider}:{key}"
        items.append({
            "id": key,
            "full_id": full_id,
            "provider": provider,
            "name": cfg.get("name", key),
            "dimension": cfg.get("dimension"),
            "max_tokens": cfg.get("max_tokens"),
            "description": cfg.get("description"),
            "price": cfg.get("price"),
            "base_url": cfg.get("base_url"),
            "embedding_endpoint": cfg.get("embedding_endpoint"),
        })

    return {"models": items}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
