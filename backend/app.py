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
from routes import chat_routes
from routes.summary_routes import router as summary_router
from routes.glossary_routes import router as glossary_router
from routes.prompt_pool_routes import router as prompt_pool_router
from routes.preset_routes import router as preset_router
from routes.memory_routes import router as memory_router
from routes import memory_routes
from services.memory_service import MemoryService
from config import settings

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
app.include_router(preset_router)
app.include_router(memory_router)

# 初始化 MemoryService 单例并注入到 memory_routes
_memory_data_dir = str(DATA_DIR / "memory")
_memory_service = MemoryService(
    data_dir=_memory_data_dir,
    use_sqlite=settings.memory_use_sqlite
)
# 应用配置参数
_memory_service.max_summaries = settings.memory_max_summaries
_memory_service.keyword_threshold = settings.memory_keyword_threshold
# 注入到路由模块
memory_routes.memory_service = _memory_service
chat_routes.memory_service = _memory_service

# 初始化文件监听器（如果启用 Markdown 源文件）
_memory_watcher = None
if settings.memory_enabled:
    try:
        from services.memory_sync import MemoryFileWatcher
        _memory_dir = str(DATA_DIR / "memory" / "memory")
        _memory_watcher = MemoryFileWatcher(_memory_service, _memory_dir)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"初始化记忆文件监听器失败: {e}")

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


@app.on_event("startup")
async def startup_event():
    """应用启动时的事件处理"""
    # 启动记忆文件监听器
    global _memory_watcher
    if _memory_watcher:
        try:
            _memory_watcher.start()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"启动记忆文件监听器失败: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的事件处理"""
    # 停止记忆文件监听器
    global _memory_watcher
    if _memory_watcher:
        try:
            _memory_watcher.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"停止记忆文件监听器失败: {e}")
    
    # 关闭 SQLite 连接（如果使用）
    global _memory_service
    if _memory_service and hasattr(_memory_service.store, 'close'):
        try:
            _memory_service.store.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"关闭 SQLite 连接失败: {e}")


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


def _kill_port(port: int):
    """启动前清理占用指定端口的旧进程"""
    import subprocess, os, signal, sys
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f'netstat -ano | findstr :{port} | findstr LISTENING',
                capture_output=True, text=True, shell=True
            )
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip().split()[-1])
                if pid != os.getpid():
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True,
                                   capture_output=True)
                    print(f"  已清理旧进程 PID={pid}")
        else:
            result = subprocess.run(
                f'lsof -ti:{port}', capture_output=True, text=True, shell=True
            )
            for pid_str in result.stdout.strip().splitlines():
                pid = int(pid_str)
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
                    print(f"  已清理旧进程 PID={pid}")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    _kill_port(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
