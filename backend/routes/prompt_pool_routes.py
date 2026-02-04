"""
提示词池 API 路由
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.prompt_pool_service import prompt_pool_service

router = APIRouter(prefix="/prompts", tags=["prompts"])


class AddPromptRequest(BaseModel):
    name: str
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    category: str = "general"
    tags: List[str] = []


class RecordUsageRequest(BaseModel):
    prompt_id: str
    success: bool
    response_time: float = 0
    error: Optional[str] = None


class TogglePromptRequest(BaseModel):
    selected: bool


@router.get("/")
async def get_all_prompts():
    """获取所有提示词"""
    return {"prompts": prompt_pool_service.get_all_prompts()}


@router.get("/defaults")
async def get_default_prompts():
    """获取默认提示词模板"""
    return {"prompts": prompt_pool_service.get_default_prompts()}


@router.post("/")
async def add_prompt(request: AddPromptRequest):
    """添加提示词"""
    prompt = prompt_pool_service.add_prompt(
        name=request.name,
        system_prompt=request.system_prompt,
        user_prompt_template=request.user_prompt_template,
        description=request.description,
        category=request.category,
        tags=request.tags
    )
    return {
        "message": "提示词添加成功",
        "prompt_id": prompt.id,
        "name": prompt.name
    }


@router.get("/{prompt_id}")
async def get_prompt(prompt_id: str):
    """获取提示词详情"""
    prompt = prompt_pool_service.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="提示词不存在")
    
    return {
        "id": prompt.id,
        "name": prompt.name,
        "system_prompt": prompt.system_prompt,
        "user_prompt_template": prompt.user_prompt_template,
        "description": prompt.description,
        "category": prompt.category,
        "tags": prompt.tags,
        "usage_count": prompt.usage_count,
        "is_active": prompt.is_active,
        "user_selected": prompt.user_selected,
        "health_status": {
            "status": prompt.health_status.status.value,
            "total_requests": prompt.health_status.total_requests,
            "success_count": prompt.health_status.success_count,
            "failure_count": prompt.health_status.failure_count,
            "consecutive_failures": prompt.health_status.consecutive_failures,
            "average_response_time": prompt.health_status.average_response_time
        }
    }


@router.put("/{prompt_id}/toggle")
async def toggle_prompt(prompt_id: str, request: TogglePromptRequest):
    """切换提示词选中状态"""
    success = prompt_pool_service.toggle_prompt(prompt_id, request.selected)
    if not success:
        raise HTTPException(status_code=404, detail="提示词不存在")
    return {"message": "提示词状态已更新"}


@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """删除提示词"""
    success = prompt_pool_service.delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="提示词不存在")
    return {"message": "提示词已删除"}


@router.post("/usage")
async def record_usage(request: RecordUsageRequest):
    """记录提示词使用结果"""
    prompt_pool_service.record_usage(
        prompt_id=request.prompt_id,
        success=request.success,
        response_time=request.response_time,
        error=request.error
    )
    return {"message": "使用记录已保存"}


@router.get("/healthy/select")
async def select_healthy_prompt(exclude_id: Optional[str] = None):
    """选择一个健康的提示词"""
    prompt = prompt_pool_service.select_healthy_prompt(exclude_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="没有可用的健康提示词")
    
    return {
        "id": prompt.id,
        "name": prompt.name,
        "system_prompt": prompt.system_prompt,
        "user_prompt_template": prompt.user_prompt_template
    }


@router.post("/resurrection/check")
async def check_resurrection():
    """检查并复活失活的提示词"""
    prompt_pool_service.check_resurrection()
    return {"message": "复活检查完成"}
