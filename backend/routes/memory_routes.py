"""
记忆系统 API 路由

提供记忆数据的 CRUD 接口，包括：
- 用户画像查询
- 文档会话记忆查询
- 记忆条目的增删改
- 记忆系统状态查询
- 清空所有记忆
"""
import asyncio

from fastapi import APIRouter, HTTPException, Query
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


class MemoryEntryToggle(BaseModel):
    """停用/启用记忆条目的请求体"""
    disabled: bool = True


class MemoryGraphRebuild(BaseModel):
    """LLM 重建图谱的请求体（需要调用方提供模型凭证）"""
    api_key: str
    model: str
    api_provider: str


class MemoryEntryResponse(BaseModel):
    """记忆条目响应模型"""
    id: str
    content: str
    source_type: str
    created_at: str
    doc_id: str | None
    importance: float
    memory_kind: str | None = None
    memory_scope: str | None = None
    status: str | None = None
    title: str | None = None
    summary: str | None = None


class MemoryStatusResponse(BaseModel):
    """记忆系统状态响应模型"""
    enabled: bool
    total_entries: int
    index_size: int
    profile_focus_areas: list[str]
    snapshot_primary: bool = True
    profile_snapshot_exists: bool = False
    session_snapshot_count: int = 0
    event_log_files: int = 0
    last_event_at: str = ""
    dirty: bool = False
    last_sync_at: str = ""
    last_reindex_at: str = ""
    last_reindex_reason: str = ""
    index_version: int = 1
    pending_sync: bool = False
    stored_embedding_model: str = ""
    rebuild_required: bool = False
    rebuild_reason: str = ""
    llm_calls_per_turn: int = 3


# ==================== 辅助函数 ====================

def _get_service():
    """获取 memory_service 实例，未初始化时抛出 500"""
    if memory_service is None:
        raise HTTPException(status_code=500, detail="记忆服务未初始化")
    return memory_service


def _validate_doc_id(svc, doc_id: str | None) -> str | None:
    if doc_id is None:
        return None
    try:
        return svc.validate_doc_id(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
    return svc.get_session(_validate_doc_id(svc, doc_id))


@router.get("/entries")
async def list_entries(
    doc_id: str | None = Query(default=None),
    memory_kind: str | None = Query(default=None),
    memory_scope: str | None = Query(default=None),
    status: str | None = Query(default=None),
    lifecycle: str | None = Query(
        default=None,
        description="retrievable | invalidated | disabled | archived",
    ),
    limit: int | None = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """按条件列出记忆条目，支持生命周期筛选与分页。"""
    svc = _get_service()
    doc_id = _validate_doc_id(svc, doc_id)
    page = svc.list_entries_page(
        doc_id=doc_id,
        memory_kind=memory_kind,
        memory_scope=memory_scope,
        status=status,
        lifecycle=lifecycle,
        limit=limit,
        offset=offset,
    )
    # 保留 entries 键，老前端不受影响
    return {
        "entries": page["items"],
        "total": page["total"],
        "offset": page["offset"],
        "limit": page["limit"],
    }


@router.get("/entries/{entry_id}/history")
async def get_entry_history(
    entry_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """返回单条记忆的演化链（ADD/UPDATE/INVALIDATE/... 全过程）。"""
    svc = _get_service()
    return {"entry_id": entry_id, "history": svc.get_entry_history(entry_id, limit=limit)}


@router.get("/audit")
async def get_recent_audit(
    limit: int = Query(default=50, ge=1, le=200),
    doc_id: str | None = Query(default=None),
    event: str | None = Query(default=None),
):
    """返回最近的记忆变更，用于面板的演化历史时间线。"""
    svc = _get_service()
    doc_id = _validate_doc_id(svc, doc_id)
    return {"events": svc.get_recent_audit(limit=limit, doc_id=doc_id, event=event)}


@router.get("/entries/{entry_id}/trace")
async def get_entry_trace(entry_id: str):
    """返回指定记忆的来源链。"""
    svc = _get_service()
    try:
        return svc.get_entry_trace(entry_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")


@router.get("/graph/{doc_id}")
async def get_graph_summary(doc_id: str):
    """返回指定文档的轻量图谱摘要。"""
    svc = _get_service()
    return svc.get_graph_summary(_validate_doc_id(svc, doc_id))


@router.post("/graph/{doc_id}/rebuild")
async def rebuild_graph(doc_id: str, body: MemoryGraphRebuild):
    """用 LLM 重建指定文档的实体关系图谱。

    这是一次真实的 LLM 调用，所以只放在用户显式触发的入口上；
    GET /graph/{doc_id} 永远只读缓存或走正则降级。
    """
    svc = _get_service()
    doc_id = _validate_doc_id(svc, doc_id)
    # ``rebuild_graph`` uses the synchronous memory LLM wrapper. Running it
    # on the request loop would make that wrapper wait on its own loop and
    # time out. A worker thread has no running event loop, so the call remains
    # synchronous there without blocking other API requests.
    summary = await asyncio.to_thread(
        svc.rebuild_graph,
        doc_id,
        api_key=body.api_key,
        model=body.model,
        api_provider=body.api_provider,
        force=True,
    )
    if summary is None:
        raise HTTPException(
            status_code=422,
            detail="图谱重建未执行：请检查是否启用、凭证是否有效、该文档事实是否少于 2 条",
        )
    return summary


@router.get("/quota")
async def get_quota(doc_id: str | None = Query(default=None)):
    """返回记忆存储配额占用情况。"""
    svc = _get_service()
    return svc.get_quota_status(_validate_doc_id(svc, doc_id))


@router.get("/status", response_model=MemoryStatusResponse)
async def get_status():
    """获取记忆系统状态"""
    svc = _get_service()
    return svc.get_status()


@router.post("/entries", response_model=MemoryEntryResponse)
async def add_entry(body: MemoryEntryCreate):
    """添加记忆条目"""
    svc = _get_service()
    doc_id = _validate_doc_id(svc, body.doc_id)
    entry = svc.add_entry(
        content=body.content,
        source_type=body.source_type,
        doc_id=doc_id,
    )
    return MemoryEntryResponse(
        id=entry.id,
        content=entry.content,
        source_type=entry.source_type,
        created_at=entry.created_at,
        doc_id=entry.doc_id,
        importance=entry.importance,
        memory_kind=entry.memory_kind,
        memory_scope=entry.memory_scope,
        status=entry.status,
        title=entry.title,
        summary=entry.summary,
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
                memory_kind=e.memory_kind,
                memory_scope=e.memory_scope,
                status=e.status,
                title=e.title,
                summary=e.summary,
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


@router.post("/entries/{entry_id}/disable")
async def disable_entry(entry_id: str, body: MemoryEntryToggle):
    """停用/启用一条记忆。停用后不再参与检索，但可随时恢复。"""
    svc = _get_service()
    if not svc.set_entry_disabled(entry_id, body.disabled, actor="user"):
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")
    return {
        "entry_id": entry_id,
        "disabled": body.disabled,
        "message": "记忆已停用" if body.disabled else "记忆已启用",
    }


@router.post("/entries/{entry_id}/revalidate")
async def revalidate_entry(entry_id: str):
    """撤销失效，把被裁决判为过期的记忆放回检索池。"""
    svc = _get_service()
    if not svc.revalidate_entry(entry_id, actor="user"):
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")
    return {"entry_id": entry_id, "message": "记忆已恢复"}


@router.post("/entries/{entry_id}/restore")
async def restore_archived_entry(entry_id: str):
    """把压缩归档的原始记忆恢复为活跃状态（压缩可能摘丢了细节）。"""
    svc = _get_service()
    if not svc.restore_archived_entry(entry_id, actor="user"):
        raise HTTPException(status_code=404, detail=f"记忆条目 {entry_id} 不存在")
    return {"entry_id": entry_id, "message": "已从归档恢复"}


@router.delete("/sessions/{doc_id}")
async def clear_session(doc_id: str):
    """清空指定文档的全部记忆。"""
    svc = _get_service()
    doc_id = _validate_doc_id(svc, doc_id)
    removed = svc.clear_document(doc_id)
    return {"doc_id": doc_id, "removed": removed, "message": f"已删除 {removed} 条记忆"}


@router.delete("/all")
async def clear_all():
    """清空所有记忆数据"""
    svc = _get_service()
    svc.clear_all()
    return {"message": "所有记忆数据已清空"}


@router.post("/rebuild-from-events")
async def rebuild_from_events():
    """从事件日志重建快照与索引。"""
    svc = _get_service()
    return svc.rebuild_from_events()
