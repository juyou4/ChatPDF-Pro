"""
速览（Overview）服务 - 生成结构化 AI 学术导读
"""
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============ 数据模型 ============

class OverviewDepth(str):
    """速览深度枚举"""
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class TermItem(BaseModel):
    """术语解释项"""
    term: str
    explanation: str


class SpeedReadContent(BaseModel):
    """论文速读内容"""
    method: str
    experiment_design: str
    problems_solved: str


class KeyFigureItem(BaseModel):
    """关键图表项"""
    figure_id: str
    caption: str
    image_base64: Optional[str] = None
    analysis: str


class PaperSummary(BaseModel):
    """论文总结"""
    strengths: str
    innovations: str
    future_work: str


class OverviewData(BaseModel):
    """速览完整数据结构"""
    doc_id: str
    title: str
    depth: str
    full_text_summary: str
    terminology: List[TermItem]
    speed_read: SpeedReadContent
    key_figures: List[KeyFigureItem]
    paper_summary: PaperSummary
    created_at: float


class OverviewTask(BaseModel):
    """异步任务状态"""
    task_id: str
    doc_id: str
    depth: str
    api_key: str = ""
    model: str = "gpt-4o"
    provider: str = "openai"
    endpoint: str = ""
    status: str  # pending, processing, completed, failed
    result: Optional[OverviewData] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


# ============ 配置 ============

# 缓存目录
CACHE_DIR = Path(__file__).parent.parent / "data" / "overviews"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 任务存储（生产环境可替换为 Redis）
overview_tasks: Dict[str, OverviewTask] = {}
overview_cache: Dict[str, OverviewData] = {}

# 深度配置
DEPTH_CONFIG = {
    OverviewDepth.BRIEF: {
        "max_chars_per_card": 150,
        "term_count": 3,
        "figure_count": 2,
    },
    OverviewDepth.STANDARD: {
        "max_chars_per_card": 300,
        "term_count": 5,
        "figure_count": 3,
    },
    OverviewDepth.DETAILED: {
        "max_chars_per_card": 600,
        "term_count": 8,
        "figure_count": 5,
    },
}


# ============ Prompt 模板 ============

def _build_overview_prompt(depth: str) -> str:
    """根据深度构建速览生成 prompt"""
    depth_cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG[OverviewDepth.STANDARD])
    
    prompt = f"""你是一个专业的学术论文导读助手。请根据以下论文内容，生成结构化的学术导读，包含五个部分：

## 【全文概述】
用 50-100 字概括论文的核心贡献、应用场景和主要效果。

## 【术语解释】
列出论文中出现的 {depth_cfg['term_count']} 个关键术语/概念，并给出简短解释（每条 20-40 字）。
格式：术语: 解释

## 【论文速读】
分三块简要说明：
1. 论文方法：核心算法或方法的关键思路
2. 实验设计：数据集、评估指标、对比方法
3. 解决的问题：论文试图解决的具体问题

## 【论文总结】
1. 优点与创新：论文的主要贡献点
2. 未来展望：可能的改进方向或应用场景

请直接输出 JSON 格式，不要包含其他文字。JSON 结构如下：
{{
    "full_text_summary": "全文概述内容",
    "terminology": [{{"term": "术语1", "explanation": "解释1"}}],
    "speed_read": {{
        "method": "方法描述",
        "experiment_design": "实验设计描述",
        "problems_solved": "解决的问题描述"
    }},
    "paper_summary": {{
        "strengths": "优点与创新",
        "innovations": "创新点",
        "future_work": "未来展望"
    }}
}}

论文内容：
"""
    return prompt


# ============ 核心服务 ============

async def get_document_text(doc_id: str) -> Optional[str]:
    """获取文档全文"""
    from routes.document_routes import documents_store
    
    if doc_id not in documents_store:
        return None
    
    doc = documents_store[doc_id]
    return doc.get("data", {}).get("full_text", "")


async def get_document_info(doc_id: str) -> Optional[Dict]:
    """获取文档基本信息"""
    from routes.document_routes import documents_store
    
    if doc_id not in documents_store:
        return None
    
    doc = documents_store[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc.get("filename", "未知文档"),
    }


async def get_document_images_and_pages(doc_id: str) -> tuple:
    """获取文档已提取的图片列表和页面文本。返回 (images, pages)，失败返回 ([], [])。"""
    from routes.document_routes import documents_store
    
    if doc_id not in documents_store:
        return [], []
    
    doc = documents_store[doc_id]
    data = doc.get("data", {})
    images = data.get("images") or []
    pages = data.get("pages") or []
    return images, pages


def _extract_figures_for_overview(
    images: List[Dict],
    pages: List[Dict],
    depth: str,
) -> List[Dict]:
    """
    从文档图片中选取前 N 张作为「关键图表」。
    返回列表，每项为 {"figure_id", "image_data", "page_num", "page_content_snippet"}。
    """
    figure_count = DEPTH_CONFIG.get(depth, DEPTH_CONFIG[OverviewDepth.STANDARD]).get("figure_count", 3)
    # 按页码排序，同页内保持顺序
    sorted_images = sorted(images, key=lambda x: (x.get("page", 0), x.get("id", "")))
    selected = sorted_images[:figure_count]
    
    result = []
    for i, img in enumerate(selected):
        page_num = img.get("page", i + 1)
        data_url = img.get("data", "")
        if not data_url:
            continue
        # 页面文本片段（供多模态模型上下文）
        page_content = ""
        if pages and 1 <= page_num <= len(pages):
            p = pages[page_num - 1]
            page_content = (p.get("content") or "")[:800]
        
        result.append({
            "figure_id": img.get("id", f"fig-{i+1}"),
            "image_data": data_url,
            "page_num": page_num,
            "page_content_snippet": page_content,
        })
    return result


def _extract_content_from_response(response: dict) -> str:
    """从 call_ai_api 返回的原始响应中提取文本 content。"""
    if response.get("content"):
        return response.get("content", "")
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {}) or {}
        return msg.get("content", "") or ""
    return ""


async def _generate_single_figure_analysis(
    figure_id: str,
    figure_index: int,
    image_data_url: str,
    page_content_snippet: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> Optional[KeyFigureItem]:
    """
    调用多模态 LLM 对单张图生成标题与解析。
    返回 KeyFigureItem，失败返回 None。
    """
    from services.chat_service import call_ai_api
    
    prompt = f"""这是一篇学术论文中的第 {figure_index + 1} 张图。该图所在页面的部分文字如下：

{page_content_snippet[:500] if page_content_snippet else "（无正文）"}

请完成两件事（用中文）：
1. 用一句话概括该图的标题（例如「图1: xxx」）。
2. 写一段 2–4 句话的解析，说明该图在论文中的作用（如方法示意、实验对比、架构图等）。

请严格按以下 JSON 格式输出，不要包含其他文字：
{{"caption": "图X: 标题", "analysis": "解析段落"}}
"""
    
    # 多模态消息：先文字后图片（与 chat vision 一致）
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    messages = [
        {"role": "system", "content": "你是学术论文图表分析助手。根据论文图片和上下文，输出指定 JSON 格式。"},
        {"role": "user", "content": user_content},
    ]
    
    try:
        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=1024,
            temperature=0.3,
        )
        
        if isinstance(response, dict) and response.get("error"):
            return None
        
        content = _extract_content_from_response(response)
        if not content:
            return None
        
        # 解析 JSON
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(content[json_start:json_end])
        else:
            data = json.loads(content)
        
        caption = data.get("caption", f"图{figure_index + 1}")
        analysis = data.get("analysis", "")
        
        return KeyFigureItem(
            figure_id=figure_id,
            caption=caption,
            image_base64=image_data_url,
            analysis=analysis,
        )
    except Exception as e:
        logger.warning(f"单张图表解析失败: {e}")
        return None


def _get_cache_key(doc_id: str, depth: str) -> str:
    """生成缓存 key"""
    return f"{doc_id}_{depth}"


def _get_cache_path(doc_id: str, depth: str) -> Path:
    """获取缓存文件路径"""
    key = _get_cache_key(doc_id, depth)
    return CACHE_DIR / f"{key}.json"


async def get_cached_overview(doc_id: str, depth: str) -> Optional[OverviewData]:
    """获取缓存的速览"""
    cache_key = _get_cache_key(doc_id, depth)
    
    # 内存缓存
    if cache_key in overview_cache:
        return overview_cache[cache_key]
    
    # 文件缓存
    cache_path = _get_cache_path(doc_id, depth)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            overview = OverviewData(**data)
            overview_cache[cache_key] = overview
            return overview
        except Exception as e:
            logger.warning(f"读取速览缓存失败: {e}")
    
    return None


async def save_overview_cache(overview: OverviewData):
    """保存速览到缓存"""
    cache_key = _get_cache_key(overview.doc_id, overview.depth)
    
    # 内存缓存
    overview_cache[cache_key] = overview
    
    # 文件缓存
    cache_path = _get_cache_path(overview.doc_id, overview.depth)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(overview.model_dump(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存速览缓存失败: {e}")


async def generate_overview_content(
    doc_id: str,
    depth: str,
    document_text: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
) -> OverviewData:
    """生成速览内容（调用 LLM）"""
    from services.chat_service import call_ai_api
    
    # 获取文档信息
    doc_info = await get_document_info(doc_id)
    title = doc_info.get("filename", "未知文档") if doc_info else "未知文档"
    
    # 构建 prompt
    prompt = _build_overview_prompt(depth)
    full_prompt = f"{prompt}\n\n{document_text[:10000]}"  # 限制输入长度
    
    messages = [
        {"role": "system", "content": "你是一个专业的学术论文导读助手，擅长总结论文核心内容并用简洁易懂的语言解释。"},
        {"role": "user", "content": full_prompt}
    ]
    
    # 调用 LLM
    try:
        response = await call_ai_api(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )
        
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response.get("error"))
        
        content = _extract_content_from_response(response)
        
        # 解析 JSON
        # 尝试提取 JSON 部分
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
        else:
            # 尝试直接解析
            data = json.loads(content)
        
        # 构建返回数据（先不含图表）
        overview = OverviewData(
            doc_id=doc_id,
            title=title,
            depth=depth,
            full_text_summary=data.get("full_text_summary", ""),
            terminology=[TermItem(**t) for t in data.get("terminology", [])],
            speed_read=SpeedReadContent(**data.get("speed_read", {})),
            key_figures=[],
            paper_summary=PaperSummary(**data.get("paper_summary", {})),
            created_at=time.time()
        )
        
        # 关键图表解读：从文档提取图片并用多模态模型生成解析
        try:
            images, pages = await get_document_images_and_pages(doc_id)
            if images:
                figures_to_analyze = _extract_figures_for_overview(images, pages, depth)
                key_figures_list = []
                for i, fig in enumerate(figures_to_analyze):
                    item = await _generate_single_figure_analysis(
                        figure_id=fig["figure_id"],
                        figure_index=i,
                        image_data_url=fig["image_data"],
                        page_content_snippet=fig.get("page_content_snippet", ""),
                        api_key=api_key,
                        model=model,
                        provider=provider,
                        endpoint=endpoint,
                    )
                    if item:
                        key_figures_list.append(item)
                if key_figures_list:
                    overview.key_figures = key_figures_list
        except Exception as e:
            logger.warning(f"关键图表解读跳过: {e}")
        
        # 保存缓存
        await save_overview_cache(overview)
        
        return overview
        
    except Exception as e:
        logger.error(f"生成速览失败: {e}")
        raise


async def create_overview_task(
    doc_id: str,
    depth: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
) -> OverviewTask:
    """创建异步任务"""
    task_id = str(uuid.uuid4())
    
    task = OverviewTask(
        task_id=task_id,
        doc_id=doc_id,
        depth=depth,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="pending",
        created_at=time.time(),
        updated_at=time.time()
    )
    
    overview_tasks[task_id] = task
    
    # 启动异步生成
    asyncio.create_task(_process_overview_task(task_id))
    
    return task


async def _process_overview_task(task_id: str):
    """处理速览生成任务"""
    if task_id not in overview_tasks:
        return
    
    task = overview_tasks[task_id]
    
    try:
        # 更新状态
        task.status = "processing"
        task.updated_at = time.time()
        
        # 检查缓存
        cached = await get_cached_overview(task.doc_id, task.depth)
        if cached:
            task.result = cached
            task.status = "completed"
            task.updated_at = time.time()
            return
        
        # 获取文档内容
        document_text = await get_document_text(task.doc_id)
        if not document_text:
            task.status = "failed"
            task.error = "文档未找到"
            task.updated_at = time.time()
            return
        
        # 生成速览
        result = await generate_overview_content(
            task.doc_id,
            task.depth,
            document_text,
            api_key=task.api_key,
            model=task.model,
            provider=task.provider,
            endpoint=task.endpoint,
        )
        
        task.result = result
        task.status = "completed"
        
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        logger.error(f"速览任务 {task_id} 失败: {e}")
    
    task.updated_at = time.time()


async def get_task_status(task_id: str) -> Optional[OverviewTask]:
    """获取任务状态"""
    return overview_tasks.get(task_id)


# ============ 公开接口 ============

async def get_or_create_overview(
    doc_id: str,
    depth: str = "standard",
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
) -> OverviewData:
    """获取或创建速览（同步接口）"""
    # 先检查缓存
    cached = await get_cached_overview(doc_id, depth)
    if cached:
        return cached
    
    # 创建任务并等待完成
    task = await create_overview_task(doc_id, depth, api_key, model, provider, endpoint)
    
    # 轮询等待完成（超时 120 秒）
    max_wait = 120
    poll_interval = 2
    waited = 0
    
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        
        task_status = await get_task_status(task.task_id)
        if not task_status:
            break
        
        if task_status.status == "completed" and task_status.result:
            return task_status.result
        elif task_status.status == "failed":
            raise RuntimeError(task_status.error or "生成失败")
    
    raise TimeoutError("速览生成超时")
