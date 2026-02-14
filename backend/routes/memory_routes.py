"""
记忆系统 API 路由

提供记忆数据的 CRUD 接口，包括：
- 用户画像查询
- 文档会话记忆查询
- 记忆条目的增删改
- 记忆系统状态查询
- 清空所有记忆
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/memory")

# 模块级变量，由 app.py 注入 MemoryService 实例
memory_service = None


# ==================== 请求/响应模型 ====================

class MemoryEntryCreate(BaseModel):
    """创建记忆条目的请求体"""
    content: str
    source_type: str = "manual"  # "manual" | "liked"
    doc_id: str | None = None


class MemoryEntryUpdate(BaseModel):
    """更新记忆条目的请求体"""
    content: str


class MemoryEntryResponse(BaseModel):
    """记忆条目响应模型"""
    id: str
    content: str
    source_type: str
    created_at: str
    doc_id: str | None
    importance: float


class MemoryStatusResponse(BaseModel):
    """记忆系统状态响应模型"""
    enabled: bool
    total_entries: int
    index_size: int
    profile_focus_areas: list[str]


# ==================== 辅助函数 ====================

def _get_service():
    """获取 memory_service 实例，未初始化时抛出 500"""
    if memory_service is None:
        raise HTTPException(status_code=500, detail="记忆服务未初始化")
    return memory_service


# ==================== API 路由 ====================

@router.get("/profile")
async def get_profile():
    """获取用户画像数据"""
    svc = _get_service()
    return svc.get_profile()


@router.get("/sessions/{doc_id}")
async def get_session(doc_id: str):
    """获取指定文档的会话记忆"""
    svc = _get_service()
    return svc.get_session(doc_id)


@router.get("/status", response_model=MemoryStatusResponse)
async def get_status():
    """获取记忆系统状态"""
    svc = _get_service()
    return svc.get_status()


@router.post("/entries", response_model=MemoryEntryResponse)
async def add_entry(body: MemoryEntryCreate):
    """添加记忆条目"""
    svc = _get_service()
    entry = svc.add_entry(
        content=body.content,
        source_type=body.source_type,
        doc_id=body.doc_id,
    )
    return MemoryEntryResponse(
        id=entry.id,
        content=entry.content,
        source_type=entry.source_type,
        created_at=entry.created_at,
        doc_id=entry.doc_id,
        importance=entry.importance,
    )


@router.put("/entries/{entry_id}", response_model=MemoryEntryResponse)
async def update_entry(entry_id: str, body: MemoryEntryUpdate):
    """编辑指定记忆条目的内容"""
    svc = _get_service()
    success = svc.update_entry(entry_id, body.content)
    if not success:
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")

    # 从 store 中获取更新后的条目信息
    all_entries = svc.store.get_all_entries()
    for e in all_entries:
        if e.id == entry_id:
            return MemoryEntryResponse(
                id=e.id,
                content=e.content,
                source_type=e.source_type,
                created_at=e.created_at,
                doc_id=e.doc_id,
                importance=e.importance,
            )
    # 理论上不会到这里，因为 update 成功了
    raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    """删除指定记忆条目"""
    svc = _get_service()
    success = svc.delete_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")
    return {"message": f"记忆条目 {entry_id} 已删除"}


@router.delete("/all")
async def clear_all():
    """清空所有记忆数据"""
    svc = _get_service()
    svc.clear_all()
    return {"message": "所有记忆数据已清空"}
