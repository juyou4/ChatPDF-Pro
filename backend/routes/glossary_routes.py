"""
术语库 API 路由
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.glossary_service import glossary_service

router = APIRouter(prefix="/glossary", tags=["glossary"])


class CreateSetRequest(BaseModel):
    name: str
    description: str = ""


class AddEntryRequest(BaseModel):
    term: str
    translation: str
    case_sensitive: bool = False
    whole_word: bool = True
    category: str = "general"
    notes: str = ""


class ImportEntriesRequest(BaseModel):
    entries: List[dict]


class ToggleSetRequest(BaseModel):
    enabled: bool


class FindMatchesRequest(BaseModel):
    text: str


@router.get("/sets")
async def get_glossary_sets():
    """获取所有术语集"""
    return {"sets": glossary_service.get_all_sets()}


@router.post("/sets")
async def create_glossary_set(request: CreateSetRequest):
    """创建术语集"""
    glossary_set = glossary_service.create_glossary_set(
        name=request.name,
        description=request.description
    )
    return {
        "message": "术语集创建成功",
        "set_id": glossary_set.id,
        "name": glossary_set.name
    }


@router.get("/sets/{set_id}/entries")
async def get_set_entries(set_id: str):
    """获取术语集的所有条目"""
    entries = glossary_service.get_set_entries(set_id)
    if entries is None:
        raise HTTPException(status_code=404, detail="术语集不存在")
    return {"entries": entries}


@router.post("/sets/{set_id}/entries")
async def add_entry(set_id: str, request: AddEntryRequest):
    """添加术语条目"""
    success = glossary_service.add_entry(
        set_id=set_id,
        term=request.term,
        translation=request.translation,
        case_sensitive=request.case_sensitive,
        whole_word=request.whole_word,
        category=request.category,
        notes=request.notes
    )
    if not success:
        raise HTTPException(status_code=404, detail="术语集不存在")
    return {"message": "术语添加成功"}


@router.post("/sets/{set_id}/import")
async def import_entries(set_id: str, request: ImportEntriesRequest):
    """批量导入术语"""
    count = glossary_service.import_entries(set_id, request.entries)
    if count == 0:
        raise HTTPException(status_code=400, detail="没有有效的术语被导入")
    return {"message": f"成功导入 {count} 条术语"}


@router.put("/sets/{set_id}/toggle")
async def toggle_set(set_id: str, request: ToggleSetRequest):
    """启用/禁用术语集"""
    success = glossary_service.toggle_set(set_id, request.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="术语集不存在")
    return {"message": "术语集状态已更新"}


@router.delete("/sets/{set_id}")
async def delete_set(set_id: str):
    """删除术语集"""
    success = glossary_service.delete_set(set_id)
    if not success:
        raise HTTPException(status_code=404, detail="术语集不存在")
    return {"message": "术语集已删除"}


@router.post("/match")
async def find_matches(request: FindMatchesRequest):
    """在文本中查找匹配的术语"""
    matches = glossary_service.find_matches(request.text)
    return {
        "matches": matches,
        "count": len(matches)
    }


@router.post("/instruction")
async def build_instruction(request: FindMatchesRequest, target_lang: str = "中文"):
    """为文本构建术语库提示词指令"""
    matches = glossary_service.find_matches(request.text)
    if not matches:
        return {"instruction": "", "matches": []}
    
    instruction = glossary_service.build_glossary_instruction(matches, target_lang)
    return {
        "instruction": instruction,
        "matches": matches,
        "count": len(matches)
    }
