"""
预设问题 API 路由

提供 GET /api/presets 端点，返回预设问题列表。
"""

from fastapi import APIRouter

from services.preset_service import PRESET_QUESTIONS

router = APIRouter()


@router.get("/api/presets")
async def get_presets():
    """获取预设问题列表

    返回预定义的常用问题按钮列表，前端可用于在文档加载完成后
    显示预设问题按钮，方便用户快速发起查询。
    """
    return {"presets": PRESET_QUESTIONS}
