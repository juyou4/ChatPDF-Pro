"""
速览（Overview）服务 - 生成结构化 AI 学术导读

新增 Figure Pipeline 集成：
- 使用 figure_adapter 进行输入源适配
- 使用 figure_builder 进行 Figure 构建与合并
- 使用 figure_render 进行图像裁剪渲染
- 使用 figure_validation 进行质量门校验
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
import fitz

# 新增：Figure Pipeline 模块
from services.figure_adapter import FigureAdapterFactory, FigureSource
from services.figure_builder import build_logical_figures, select_top_figures
from services.figure_render import render_figure
from services.figure_validation import validate_and_fallback
from schemas.figure_schema import LogicalFigureSchema, OverviewFigureItem
from services.document_parse_state import derive_source_hash, is_parse_prepared, read_parse_manifest
from services.chat_service import call_ai_api
from services.visual_model_service import (
    VisualEnrichmentPolicy,
    VisualModelConfig,
    call_visual_model,
    model_config_identity,
    resolve_visual_enrichment_policy,
    resolve_visual_model_config,
)
from services.visual_enrichment_service import (
    VisualTaskPolicy,
    build_visual_task_id,
    execute_visual_task,
)
from services.visual_document_enrichment_service import recover_risky_local_pages
from services.visual_risk_service import assess_figure_risk, page_text_for_risk
from services.visual_supplement_service import (
    VISUAL_SUPPLEMENT_FIGURE_ANALYSIS_PROMPT_VERSION as FIGURE_ANALYSIS_PROMPT_VERSION,
    VISUAL_SUPPLEMENT_PROMPT_SUITE_IDENTITY,
    build_visual_supplement,
    visual_supplement_revision,
    visual_supplements_are_committed,
)

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
    confidence: Optional[float] = None


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
    figure_meta: Optional[dict] = None
    ai_meta: Optional[dict] = None
    # 文档可以在不改变 ``doc_id`` 的情况下重新解析。缓存内容也保留主
    # 解析 generation，作为文件名之外的第二层校验。
    parse_generation: str = ""
    document_source_hash: str = ""
    text_model_identity: str = ""
    visual_model_identity: str = ""
    visual_supplement_revision: str = ""


class OverviewTask(BaseModel):
    """异步任务状态"""
    task_id: str
    doc_id: str
    depth: str
    api_key: str = ""
    model: str = "gpt-4o"
    provider: str = "openai"
    endpoint: str = ""
    visual_api_key: str = ""
    visual_model: str = ""
    visual_provider: str = ""
    visual_endpoint: str = ""
    visual_enabled: bool = True
    visual_policy_params: dict = Field(default_factory=dict)
    status: str  # pending, processing, completed, failed
    result: Optional[OverviewData] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float
    use_mineru_figures: bool = False
    figure_render_mode: str = "raw"
    parse_generation: str = ""
    document_source_hash: str = ""


# ============ 配置 ============

LEGACY_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "overviews"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "overviews"
try:
    from runtime_mode import runtime
    CACHE_DIR = Path(runtime.data_dir) / "overviews"
except Exception:
    pass
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OVERVIEW_CACHE_VERSION = "v17"

# 任务存储（生产环境可替换为 Redis）
overview_tasks: Dict[str, OverviewTask] = {}
overview_cache: Dict[str, OverviewData] = {}
overview_inflight: Dict[str, asyncio.Task] = {}

VALID_FIGURE_RENDER_MODES = {"raw", "yolo"}


def _normalize_figure_render_mode(mode: str = "raw") -> str:
    """标准化速览图表预览模式。"""
    normalized = (mode or "raw").strip().lower()
    if normalized not in VALID_FIGURE_RENDER_MODES:
        return "raw"
    return normalized

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

OVERVIEW_TEXT_CHAR_LIMITS = {
    OverviewDepth.BRIEF: 3600,
    OverviewDepth.STANDARD: 5600,
    OverviewDepth.DETAILED: 7600,
}

OVERVIEW_OUTPUT_MAX_TOKENS = {
    OverviewDepth.BRIEF: 900,
    OverviewDepth.STANDARD: 1200,
    OverviewDepth.DETAILED: 1600,
}

FIGURE_PAGE_TEXT_LIMIT = 320
FIGURE_ANALYSIS_MAX_TOKENS = 384
FIGURE_RENDER_DPI = 132
FIGURE_CROP_MAX_SIDE = 1280
FIGURE_CROP_JPEG_QUALITY = 72

FIGURE_PATTERNS = [
    r'^图\s*(\d+[a-zA-Z]?)',
    r'^Figure\s+(\d+[a-zA-Z]?)',
    r'^Fig\.?\s+(\d+[a-zA-Z]?)',
]


def _overview_visual_task_policy(document_budget: int | None = None) -> VisualTaskPolicy:
    def _float(name: str, default: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(os.getenv(name, str(default)))))
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(os.getenv(name, str(default)))))
        except (TypeError, ValueError):
            return default

    return VisualTaskPolicy(
        timeout_seconds=_float("CHATPDF_OVERVIEW_VISUAL_TIMEOUT_S", 45.0, 8.0, 120.0),
        max_retries=_int("CHATPDF_OVERVIEW_VISUAL_RETRIES", 1, 0, 3),
        retry_delay_seconds=_float("CHATPDF_OVERVIEW_VISUAL_RETRY_DELAY", 0.4, 0.0, 5.0),
        concurrency=_int("CHATPDF_VISUAL_CONCURRENCY", 2, 1, 8),
        document_budget=(
            max(1, min(1000, int(document_budget)))
            if document_budget is not None
            else _int("CHATPDF_VISUAL_DOCUMENT_BUDGET", 16, 1, 1000)
        ),
    )


def _resolve_overview_visual_policy(
    *,
    provider: str,
    model: str,
    api_key: str,
    endpoint: str,
    visual_provider: str,
    visual_model: str,
    visual_api_key: str,
    visual_endpoint: str,
    visual_enabled: bool,
    visual_policy_params: Optional[dict],
) -> VisualEnrichmentPolicy:
    params = visual_policy_params if isinstance(visual_policy_params, dict) else {}
    return resolve_visual_enrichment_policy(
        strategy=str(params.get("strategy") or "balanced"),
        primary_provider=provider,
        primary_model=model,
        primary_api_key=api_key,
        primary_endpoint=endpoint,
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
        local_visual_provider=str(params.get("local_provider") or ""),
        local_visual_model=str(params.get("local_model") or ""),
        local_visual_api_key=str(params.get("local_api_key") or ""),
        local_visual_endpoint=str(params.get("local_endpoint") or ""),
    )


def _summarize_visual_risk(items: list[dict]) -> dict:
    reason_counts: dict[str, int] = {}
    triggered = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("should_enrich"):
            triggered += 1
        for reason in item.get("reasons") or []:
            key = str(reason or "").strip()
            if key:
                reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "evaluated": len(items),
        "triggered": triggered,
        "reused_text_structure": max(0, len(items) - triggered),
        "reason_counts": reason_counts,
        "items": items[:8],
    }


def _build_document_excerpt(document_text: str, depth: str) -> str:
    """按深度抽取更短但更均衡的文档片段，避免整段长文本直接进模型。"""
    if not document_text:
        return ""

    cleaned = " ".join(document_text.split())
    limit = OVERVIEW_TEXT_CHAR_LIMITS.get(depth, OVERVIEW_TEXT_CHAR_LIMITS[OverviewDepth.STANDARD])
    if len(cleaned) <= limit:
        return cleaned

    head_len = int(limit * 0.55)
    middle_len = int(limit * 0.20)
    tail_len = limit - head_len - middle_len
    middle_start = max(0, len(cleaned) // 2 - middle_len // 2)
    middle_end = middle_start + middle_len

    segments = [
        "【开头节选】\n" + cleaned[:head_len],
        "【中段节选】\n" + cleaned[middle_start:middle_end],
        "【结尾节选】\n" + cleaned[-tail_len:],
    ]
    return "\n\n".join(segment for segment in segments if segment.strip())


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
    """获取文档已提取的图片列表、页面文本、figures 元数据和 pdf_url。"""
    from routes.document_routes import documents_store

    if doc_id not in documents_store:
        return [], [], [], ""

    doc = documents_store[doc_id]
    data = doc.get("data", {})
    images = data.get("images") or []
    pages = data.get("pages") or []
    figures = data.get("figures") or []
    pdf_url = doc.get("pdf_url") or ""
    return images, pages, figures, pdf_url


def _normalize_bbox(bbox: Any) -> Optional[List[float]]:
    """标准化 bbox，过滤无效区域。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _is_positioned_page_bbox(bbox: Any) -> bool:
    """判断 bbox 是否像 PDF 页面坐标，而不是兜底的原图尺寸框。"""
    normalized = _normalize_bbox(bbox)
    if not normalized:
        return False
    x0, y0, _, _ = normalized
    return abs(x0) > 1e-6 or abs(y0) > 1e-6


# ==================== Phase 2: 分层裁切回退链路 ====================

# 裁切安全边距（像素）- 防止坐标轴或图例被切
CROP_PADDING = 15

# 合并子图最大数量（避免Payload Too Large）
MAX_SUB_IMAGES = 4

# 质量评分阈值
QUALITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}


def _apply_crop_padding(bbox: List[float], page_width: float, page_height: float, padding: int = CROP_PADDING) -> List[float]:
    """对 bbox 应用安全边距，防止裁切到坐标轴或图例"""
    if not bbox:
        return bbox
    x0, y0, x1, y1 = bbox
    return [
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    ]


def _expand_bbox_for_fallback(bbox: List[float], page_width: float, page_height: float, ratio: float = 1.3) -> List[float]:
    """扩张 bbox（用于 fallback）"""
    if not bbox:
        return bbox
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    new_width = width * ratio
    new_height = height * ratio
    return [
        max(0.0, center_x - new_width / 2),
        max(0.0, center_y - new_height / 2),
        min(page_width, center_x + new_width / 2),
        min(page_height, center_y + new_height / 2),
    ]


# ==================== Phase 3: 图像质量评分 ====================

def _calculate_image_quality_score(image_data: str) -> dict:
    """
    计算图像质量评分（多维度）

    返回:
        {
            "size_score": 0.0-1.0,
            "entropy_score": 0.0-1.0,
            "edge_score": 0.0-1.0,
            "total_score": 0.0-1.0,
            "reason": "too_small/blank/low_entropy/...",
            "level": "high/medium/low"
        }
    """
    try:
        import io
        from PIL import Image
        import base64
        import math

        # 1. 尺寸评分
        # 去掉 data:image/...;base64, 前缀（如果有）
        b64data = image_data
        if "," in b64data:
            b64data = b64data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        pixels = width * height
        if pixels < 10000:
            size_score = 0.0
            reason = "too_small"
        elif pixels < 50000:
            size_score = 0.3
            reason = "small"
        elif pixels > 5000 * 5000:
            size_score = 0.5  # 太大可能有问题
            reason = "too_large"
        else:
            size_score = 1.0
            reason = "ok"

        # 转换为灰度图计算熵
        gray = img.convert("L")
        pixels_list = list(gray.getdata())

        # 计算熵值
        try:
            from collections import Counter
            pixel_counts = Counter(pixels_list)
            total = len(pixels_list)
            entropy = 0.0
            for count in pixel_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(256)  # 8 bits
            entropy_score = entropy / max_entropy
        except Exception:
            entropy_score = 0.5

        # 3. 边缘密度评分（图表通常有较多线条边缘）
        try:
            from PIL import ImageFilter
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_pixels = list(edges.getdata())
            edge_density = sum(1 for p in edge_pixels if p > 10) / len(edge_pixels)
            edge_score = min(1.0, edge_density * 10)
        except Exception:
            edge_score = 0.5

        # 计算总分
        total_score = (size_score * 0.4 + entropy_score * 0.3 + edge_score * 0.3)

        # 确定等级
        if total_score >= QUALITY_THRESHOLDS["high"]:
            level = "high"
        elif total_score >= QUALITY_THRESHOLDS["medium"]:
            level = "medium"
            if not reason or reason == "ok":
                reason = "medium_quality"
        else:
            level = "low"
            if not reason or reason == "ok":
                reason = "low_quality"

        return {
            "size_score": size_score,
            "entropy_score": entropy_score,
            "edge_score": edge_score,
            "total_score": total_score,
            "reason": reason,
            "level": level,
            "width": width,
            "height": height,
        }
    except Exception as e:
        return {
            "size_score": 0.0,
            "entropy_score": 0.0,
            "edge_score": 0.0,
            "total_score": 0.0,
            "reason": f"error: {str(e)}",
            "level": "low",
        }


def _is_valid_image_for_analysis(image_data: str, min_quality: str = "medium") -> bool:
    """检查图像是否适合用于分析"""
    if not image_data:
        return False

    quality = _calculate_image_quality_score(image_data)
    threshold = QUALITY_THRESHOLDS.get(min_quality, QUALITY_THRESHOLDS["medium"])
    return quality["total_score"] >= threshold


# ==================== Phase 4: 候选打分选图 ====================

def _calculate_figure_selection_score(
    figure: dict,
    quality_score: float,
    existing_figures: list,
    page: int,
) -> float:
    """
    计算 figure 的选择评分

    打分因素:
    - quality_score: 图像质量评分 (权重 30%)
    - structure_score: 子图数量带来的复杂度 (权重 25%)
    - caption_score: caption 信息密度 (权重 20%)
    - novelty_score: 与已选图的差异度 (权重 15%)
    - page_score: 页面位置 (权重 10%)
    """
    # 1. 质量评分 (30%)
    quality_weight = 0.30

    # 2. 结构评分 (25%) - 有子图或多个image的figure信息量更大
    image_ids = figure.get("image_ids", [])
    sub_figures = figure.get("sub_figures", [])
    structure_score = min(1.0, (len(image_ids) + len(sub_figures)) / 4)
    structure_weight = 0.25

    # 3. Caption 评分 (20%)
    caption = figure.get("caption", "")
    caption_score = 0.5  # 默认中等
    if caption:
        # 包含关键词的 caption 更有信息量
        keywords = ["framework", "architecture", "pipeline", "overview", "results", "comparison", "ablation"]
        if any(kw in caption.lower() for kw in keywords):
            caption_score = 1.0
        elif len(caption) > 20:
            caption_score = 0.8
    caption_weight = 0.20

    # 4. 新颖度评分 (15%) - 避免选择与已选图相似的
    novelty_score = 0.5  # 默认
    if existing_figures:
        # 简单策略：同页的 novelty 低，不同页的高
        same_page_count = sum(1 for f in existing_figures if f.get("page") == page)
        novelty_score = max(0.2, 1.0 - same_page_count * 0.3)
    novelty_weight = 0.15

    # 5. 页面位置评分 (10%) - 优先选择前几页的图（通常是总览图）
    page_score = max(0.3, 1.0 - (page - 1) * 0.1)
    page_weight = 0.10

    # 计算总分
    total_score = (
        quality_score * quality_weight +
        structure_score * structure_weight +
        caption_score * caption_weight +
        novelty_score * novelty_weight +
        page_score * page_weight
    )

    return total_score


def _select_best_figures(
    figures: list,
    image_map: dict,
    max_count: int = 3,
) -> list:
    """
    基于多维度评分选择最佳 figures

    返回选中的 figures 列表
    """
    if not figures:
        return []

    if len(figures) <= max_count:
        return figures

    # 为每个 figure 计算评分
    scored_figures = []
    selected_pages = set()

    for fig in figures:
        # 获取图像质量评分
        image_ids = fig.get("image_ids", [])
        if image_ids:
            first_img = image_map.get(image_ids[0])
            if first_img and first_img.get("data"):
                quality = _calculate_image_quality_score(first_img.get("data"))
                quality_score = quality["total_score"]
            else:
                quality_score = 0.5
        else:
            quality_score = 0.5

        # 计算选择评分
        score = _calculate_figure_selection_score(
            fig,
            quality_score,
            scored_figures,
            fig.get("page", 1),
        )

        scored_figures.append({
            "figure": fig,
            "score": score,
            "quality_score": quality_score,
            "page": fig.get("page", 1),
        })

    # 按评分排序
    scored_figures.sort(key=lambda x: x["score"], reverse=True)

    # 选取：优先保证页面分散度
    selected = []
    pages_selected = {}

    for item in scored_figures:
        fig = item["figure"]
        page = item["page"]

        # 同页最多选2个
        page_count = pages_selected.get(page, 0)
        if page_count >= 2:
            continue

        selected.append(fig)
        pages_selected[page] = page_count + 1

        if len(selected) >= max_count:
            break

    # 恢复文档顺序
    selected.sort(key=lambda x: (x.get("page", 0), str(x.get("number", ""))))
    return selected


def _limit_sub_images(image_data_list: list) -> list:
    """
    限制子图数量，避免 Payload Too Large
    """
    if len(image_data_list) <= MAX_SUB_IMAGES:
        return image_data_list

    # 优先保留前面的图片（通常是最重要的）
    return image_data_list[:MAX_SUB_IMAGES]


def _optimize_visual_data_url(image_bytes: bytes) -> Optional[str]:
    """压缩视觉输入，减少传给多模态模型的图片尺寸。"""
    if not image_bytes:
        return None

    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            width, height = img.size
            longest_side = max(width, height)
            if longest_side > FIGURE_CROP_MAX_SIDE:
                scale = FIGURE_CROP_MAX_SIDE / float(longest_side)
                new_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            output = BytesIO()
            img.save(
                output,
                format="JPEG",
                quality=FIGURE_CROP_JPEG_QUALITY,
                optimize=True,
            )
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logger.warning(f"视觉输入压缩失败: {e}")
        return None


def _build_figure_clip_bbox(
    image_bboxes: List[Any],
    page_width: float,
    page_height: float,
) -> Optional[List[float]]:
    """
    根据 figure 下的多个子图 bbox 反推整张 figure 的裁切区域。

    这里不会只取第一张子图，而是对所有匹配图片做并集，并在多子图场景下做额外扩张，
    以覆盖图中的矢量框线、标题框、箭头和说明文字。
    """
    positioned_bboxes = [_normalize_bbox(b) for b in image_bboxes if _is_positioned_page_bbox(b)]
    positioned_bboxes = [b for b in positioned_bboxes if b]
    valid_bboxes = positioned_bboxes or [_normalize_bbox(b) for b in image_bboxes]
    valid_bboxes = [b for b in valid_bboxes if b]
    if not valid_bboxes or page_width <= 0 or page_height <= 0:
        return None

    x0 = min(b[0] for b in valid_bboxes)
    y0 = min(b[1] for b in valid_bboxes)
    x1 = max(b[2] for b in valid_bboxes)
    y1 = max(b[3] for b in valid_bboxes)

    width = x1 - x0
    height = y1 - y0
    multi_image = len(valid_bboxes) > 1

    x_margin = max(24.0, width * 0.18)
    y_margin_top = max(20.0, height * 0.15)
    y_margin_bottom = max(48.0, height * 0.30)

    if multi_image:
        x_margin = max(x_margin, page_width * 0.08)
        y_margin_top = max(y_margin_top, page_height * 0.03)
        y_margin_bottom = max(y_margin_bottom, page_height * 0.08)

    clip_x0 = max(0.0, x0 - x_margin)
    clip_y0 = max(0.0, y0 - y_margin_top)
    clip_x1 = min(page_width, x1 + x_margin)
    clip_y1 = min(page_height, y1 + y_margin_bottom)

    # 多子图往往只覆盖 figure 里的位图局部，需要扩大到更接近整张图的尺度。
    if multi_image:
        # Panels frequently omit the connector area and shared labels between
        # them. Keep enough horizontal context to preserve the composite figure.
        min_width = page_width * 0.58
        min_height = page_height * 0.20
        cur_width = clip_x1 - clip_x0
        cur_height = clip_y1 - clip_y0

        if cur_width < min_width:
            center_x = (clip_x0 + clip_x1) / 2
            half_width = min_width / 2
            clip_x0 = max(0.0, center_x - half_width)
            clip_x1 = min(page_width, center_x + half_width)
            if clip_x1 - clip_x0 < min_width - 1e-6:
                if clip_x0 <= 0:
                    clip_x0 = 0.0
                    clip_x1 = min(page_width, min_width)
                elif clip_x1 >= page_width:
                    clip_x1 = page_width
                    clip_x0 = max(0.0, page_width - min_width)
                else:
                    missing = min_width - (clip_x1 - clip_x0)
                    left_room = clip_x0
                    right_room = page_width - clip_x1
                    grow_left = min(left_room, missing / 2)
                    grow_right = min(right_room, missing - grow_left)
                    clip_x0 = max(0.0, clip_x0 - grow_left)
                    clip_x1 = min(page_width, clip_x1 + grow_right)

        if cur_height < min_height:
            center_y = (clip_y0 + clip_y1) / 2
            half_height = min_height / 2
            clip_y0 = max(0.0, center_y - half_height)
            clip_y1 = min(page_height, center_y + half_height)
            if clip_y1 - clip_y0 < min_height - 1e-6:
                if clip_y0 <= 0:
                    clip_y0 = 0.0
                    clip_y1 = min(page_height, min_height)
                elif clip_y1 >= page_height:
                    clip_y1 = page_height
                    clip_y0 = max(0.0, page_height - min_height)
                else:
                    missing = min_height - (clip_y1 - clip_y0)
                    top_room = clip_y0
                    bottom_room = page_height - clip_y1
                    grow_top = min(top_room, missing / 2)
                    grow_bottom = min(bottom_room, missing - grow_top)
                    clip_y0 = max(0.0, clip_y0 - grow_top)
                    clip_y1 = min(page_height, clip_y1 + grow_bottom)

    if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
        return None
    return [clip_x0, clip_y0, clip_x1, clip_y1]


def _build_caption_band_bbox(
    caption_bbox: Any,
    page_width: float,
    page_height: float,
    previous_caption_bbox: Optional[Any] = None,
) -> Optional[List[float]]:
    """仅根据 caption 位置推断 figure 所在的页面区域。"""
    current = _normalize_bbox(caption_bbox)
    if not current or page_width <= 0 or page_height <= 0:
        return None

    prev = _normalize_bbox(previous_caption_bbox)
    if prev:
        band_top = max(0.0, prev[3] + 6.0)
    else:
        # 无前一个 caption 时，向上回溯合理高度（页面 35% 或 280pt），而非从页顶开始
        default_height = min(page_height * 0.35, 280.0)
        band_top = max(0.0, current[1] - default_height)
    band_bottom = min(page_height, max(band_top + 1.0, current[1] - 4.0))
    if band_bottom <= band_top:
        return None

    min_height = min(page_height, max(page_height * 0.16, 120.0))
    if band_bottom - band_top < min_height:
        band_top = max(0.0, band_bottom - min_height)

    return [
        max(0.0, page_width * 0.04),
        band_top,
        min(page_width, page_width * 0.96),
        band_bottom,
    ]


def _figure_label_rank(label: str) -> tuple:
    """给 figure 标题候选打分，优先保留真正的 caption。"""
    import re

    text = (label or "").strip()
    if not text:
        return (0, 0)

    strong_caption = bool(re.match(r"^(Figure|Fig\.?|图)\s*\d+[a-zA-Z]?\s*[:.]", text, re.IGNORECASE))
    weak_caption = bool(re.match(r"^(Figure|Fig\.?|图)\s*\d+[a-zA-Z]?\b", text, re.IGNORECASE))
    score = 0
    if strong_caption:
        score += 5
    elif weak_caption:
        score += 2
    if ":" in text[:32] or "：" in text[:32]:
        score += 1
    if len(text) <= 120:
        score += 1
    return (score, -len(text))


def _extract_figure_captions_from_text_dict(
    text_dict: Dict[str, Any],
    page_num: int,
    page_width: float,
    page_height: float,
) -> List[Dict[str, Any]]:
    """从 PDF 页面的 text dict 中提取 figure caption。"""
    import re

    figures: List[Dict[str, Any]] = []
    if not text_dict or "blocks" not in text_dict:
        return figures

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            line_bbox = None
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                line_text += text
                span_bbox = span.get("bbox", [0, 0, 0, 0])
                if line_bbox is None:
                    line_bbox = span_bbox
                else:
                    line_bbox = [
                        min(line_bbox[0], span_bbox[0]),
                        min(line_bbox[1], span_bbox[1]),
                        max(line_bbox[2], span_bbox[2]),
                        max(line_bbox[3], span_bbox[3]),
                    ]

            line_text = line_text.strip()
            if not line_text:
                continue

            for pattern in FIGURE_PATTERNS:
                match = re.match(pattern, line_text, re.IGNORECASE)
                if match:
                    figure_num = match.group(1)
                    bbox = line_bbox or [0, 0, 0, 0]
                    figures.append({
                        "figure_id": f"fig-{figure_num}",
                        "number": figure_num,
                        "label": line_text[:120],
                        "page": page_num,
                        "bbox": bbox,
                        "caption_bbox": bbox,
                        "page_width": page_width,
                        "page_height": page_height,
                        "image_ids": [],
                    })
                    break
    return figures


def _load_figures_from_pdf(pdf_url: str) -> List[Dict[str, Any]]:
    """直接从原 PDF 恢复 figure caption 和几何信息。"""
    if not pdf_url:
        return []

    try:
        import fitz
        from routes.document_routes import UPLOAD_DIR
    except Exception as e:
        logger.warning(f"PDF figure 几何恢复初始化失败: {e}")
        return []

    pdf_path = UPLOAD_DIR / pdf_url.split("/")[-1]
    if not pdf_path.exists():
        return []

    pdf_doc = None
    recovered: List[Dict[str, Any]] = []
    try:
        pdf_doc = fitz.open(str(pdf_path))
        for idx in range(len(pdf_doc)):
            page = pdf_doc[idx]
            text_dict = page.get_text("dict")
            recovered.extend(
                _extract_figure_captions_from_text_dict(
                    text_dict=text_dict,
                    page_num=idx + 1,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )
            )
    except Exception as e:
        logger.warning(f"PDF figure 几何恢复失败: {e}")
        return []
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    return _dedupe_figures_metadata(recovered)


def _needs_figure_geometry_recovery(figures: List[Dict[str, Any]]) -> bool:
    if not figures:
        return True
    for fig in figures:
        has_caption_bbox = bool(fig.get("caption_bbox") or fig.get("bbox"))
        has_page_size = bool(fig.get("page_width") and fig.get("page_height"))
        has_geometry = bool(fig.get("figure_bbox")) or (has_caption_bbox and has_page_size)
        if not has_geometry:
            return True
    return False


def _enrich_figures_with_pdf_geometry(pdf_url: str, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为老文档补齐 figure caption 的 bbox 和页面尺寸。"""
    if not _needs_figure_geometry_recovery(figures):
        return figures

    recovered = _load_figures_from_pdf(pdf_url)
    if not recovered:
        return figures
    if not figures:
        return recovered

    recovered_by_key: Dict[tuple, Dict[str, Any]] = {}
    for fig in recovered:
        key = (fig.get("page", 0), str(fig.get("number", "")))
        current = recovered_by_key.get(key)
        if current is None or _figure_label_rank(fig.get("label", "")) > _figure_label_rank(current.get("label", "")):
            recovered_by_key[key] = fig

    enriched: List[Dict[str, Any]] = []
    for fig in figures:
        key = (fig.get("page", 0), str(fig.get("number", "")))
        recovered_fig = recovered_by_key.get(key)
        merged = dict(fig)
        if recovered_fig:
            for field in ("bbox", "caption_bbox", "page_width", "page_height"):
                if not merged.get(field) and recovered_fig.get(field):
                    merged[field] = recovered_fig.get(field)
            if _figure_label_rank(recovered_fig.get("label", "")) > _figure_label_rank(merged.get("label", "")):
                merged["label"] = recovered_fig.get("label", merged.get("label", ""))
        enriched.append(merged)
    return enriched


def _dedupe_figures_metadata(figures: List[Dict]) -> List[Dict]:
    """按页码和 figure 编号去重，尽量保留真正的标题行。"""
    best_by_key: Dict[tuple, Dict] = {}

    for fig in figures or []:
        key = (fig.get("page", 0), str(fig.get("number", "")))
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = fig
            continue

        if _figure_label_rank(fig.get("label", "")) > _figure_label_rank(current.get("label", "")):
            best_by_key[key] = fig

    return sorted(best_by_key.values(), key=lambda x: (x.get("page", 0), str(x.get("number", ""))))


def _render_figure_crop_from_pdf(
    pdf_url: str,
    page_num: int,
    image_bboxes: List[Any],
    figure_bbox: Optional[Any] = None,
    caption_bbox: Optional[Any] = None,
    previous_caption_bbox: Optional[Any] = None,
    figure_render_mode: str = "raw",
) -> Optional[str]:
    """根据 figure 的 bbox 或图片 bbox，从原 PDF 页面渲染整图裁切结果。"""
    if not pdf_url or page_num <= 0:
        return None

    try:
        import fitz
        from routes.document_routes import UPLOAD_DIR
    except Exception as e:
        logger.warning(f"Figure PDF 裁切初始化失败: {e}")
        return None

    pdf_path = UPLOAD_DIR / pdf_url.split("/")[-1]
    if not pdf_path.exists():
        return None

    pdf_doc = None
    try:
        pdf_doc = fitz.open(str(pdf_path))
        if page_num > len(pdf_doc):
            return None

        page = pdf_doc[page_num - 1]
        page_rect = page.rect
        clip_bbox = _normalize_bbox(figure_bbox)
        if not clip_bbox:
            clip_bbox = _build_caption_band_bbox(
                caption_bbox=caption_bbox,
                page_width=page_rect.width,
                page_height=page_rect.height,
                previous_caption_bbox=previous_caption_bbox,
            )
        if not clip_bbox:
            clip_bbox = _build_figure_clip_bbox(image_bboxes, page_rect.width, page_rect.height)
        if not clip_bbox:
            return None

        if _normalize_figure_render_mode(figure_render_mode) == "yolo":
            # 使用 DocLayout-YOLO 收紧到纯图像区域（与 figure_render 一致）
            try:
                from services.figure_render import _tighten_bbox_to_images
                tight = _tighten_bbox_to_images(page, list(clip_bbox), mode="yolo")
                if tight:
                    clip_bbox = tight
            except Exception as e:
                logger.debug(f"YOLO tighten failed in old path: {e}")

        clip = fitz.Rect(*clip_bbox)
        pix = page.get_pixmap(dpi=FIGURE_RENDER_DPI, clip=clip, annots=False)
        if pix.width <= 10 or pix.height <= 10:
            return None

        optimized = _optimize_visual_data_url(pix.tobytes("png"))
        if optimized:
            return optimized

        img_bytes = pix.tobytes("jpeg")
        img_base64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.warning(f"Figure PDF 裁切失败: {e}")
        return None
    finally:
        if pdf_doc is not None:
            pdf_doc.close()


def _extract_figures_for_overview(
    images: List[Dict],
    pages: List[Dict],
    depth: str,
    figures: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    从文档中选取关键图表进行解读。
    优先使用 figures 元数据（按 figure 标题分组），若无 figures 则回退到按单张图片选取。
    返回列表，每项为 figure 元数据，供后续整图裁切和图表解读使用。
    """
    figure_count = DEPTH_CONFIG.get(depth, DEPTH_CONFIG[OverviewDepth.STANDARD]).get("figure_count", 3)

    # 构建 image_id -> image 元数据 的映射
    image_map = {img.get("id", ""): img for img in images if img.get("id") and img.get("data")}
    page_to_images: Dict[int, List[Dict]] = {}
    for img in images:
        page_num = img.get("page", 0)
        if page_num <= 0 or not img.get("data"):
            continue
        page_to_images.setdefault(page_num, []).append(img)
    for page_num in page_to_images:
        page_to_images[page_num] = sorted(
            page_to_images[page_num],
            key=lambda img: (
                (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[1],
                (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[0],
                img.get("id", ""),
            ),
        )

    # 如果有 figures 元数据，优先使用
    if figures:
        sorted_figures = _dedupe_figures_metadata(figures)
        selected_figures = _select_best_figures(sorted_figures, image_map, figure_count)
        previous_caption_bbox_by_ref: Dict[int, Optional[Any]] = {}
        previous_by_page: Dict[int, Optional[Any]] = {}
        for fig in sorted_figures:
            page_num = fig.get("page", 1)
            previous_caption_bbox_by_ref[id(fig)] = previous_by_page.get(page_num)
            previous_by_page[page_num] = fig.get("caption_bbox") or fig.get("bbox")

        result = []
        for fig in selected_figures:
            page_num = fig.get("page", 1)
            image_ids = fig.get("image_ids", [])

            image_items = [image_map.get(img_id) for img_id in image_ids if image_map.get(img_id)]
            if not image_items:
                # 某些 PDF 的 figure 匹配会漏掉 image_ids，此时至少退化到“该页全部图片”
                image_items = page_to_images.get(page_num, [])

            image_data_list = [img.get("data", "") for img in image_items if img.get("data")]
            image_bboxes = [img.get("bbox") for img in image_items if _is_positioned_page_bbox(img.get("bbox"))]

            caption_bbox = fig.get("caption_bbox") or fig.get("bbox")
            has_pdf_crop_source = bool(fig.get("figure_bbox") or caption_bbox)
            if not image_data_list and not has_pdf_crop_source:
                continue

            # 页面文本片段
            page_content = ""
            if pages and 1 <= page_num <= len(pages):
                p = pages[page_num - 1]
                page_content = (p.get("content") or "")[:FIGURE_PAGE_TEXT_LIMIT]

            result.append({
                "figure_id": fig.get("figure_id", f"fig-{fig.get('number', '')}"),
                "image_data_list": image_data_list,
                "image_bboxes": image_bboxes,
                "figure_bbox": fig.get("figure_bbox"),
                "caption_bbox": caption_bbox,
                "previous_caption_bbox": previous_caption_bbox_by_ref.get(id(fig)),
                "page_num": page_num,
                "page_content_snippet": page_content,
                "figure_label": fig.get("label", ""),
                "caption": fig.get("caption") or fig.get("label", ""),
            })

        if result:
            return result

    # 回退：按单张图片选取（旧策略）
    sorted_images = sorted(images, key=lambda x: (x.get("page", 0), x.get("id", "")))
    selected = sorted_images[:figure_count]

    result = []
    for i, img in enumerate(selected):
        page_num = img.get("page", i + 1)
        data_url = img.get("data", "")
        if not data_url:
            continue

        page_content = ""
        if pages and 1 <= page_num <= len(pages):
            p = pages[page_num - 1]
            page_content = (p.get("content") or "")[:FIGURE_PAGE_TEXT_LIMIT]

        result.append({
            "figure_id": img.get("id", f"fig-{i+1}"),
            "image_data_list": [data_url],
            "image_bboxes": [img.get("bbox")] if img.get("bbox") else [],
            "figure_bbox": img.get("bbox"),
            "caption_bbox": None,
            "previous_caption_bbox": None,
            "page_num": page_num,
            "page_content_snippet": page_content,
            "figure_label": "",
        })
    return result


# ============ 新增：Figure Pipeline 集成函数 ============

def _extract_figures_via_pipeline(
    pdf_doc,
    doc_data: dict,
    depth: str,
    page_width: float = 612,
    page_height: float = 792
) -> List[LogicalFigureSchema]:
    """
    使用新的 Figure Pipeline 提取 figures
    
    流程：
    1. 获取 OCR 结果 (figures + images)
    2. 尝试 MinerU Adapter
    3. 尝试 PDF Native Adapter
    4. Fallback to image list
    
    Returns:
        List[LogicalFigureSchema]: 标准化的 Figure 列表
    """
    # 获取文档数据
    ocr_result = doc_data.get("ocr_result", {})
    figures = ocr_result.get("figures", [])
    images = ocr_result.get("images", [])
    
    if not figures and not images:
        logger.info("No figures or images found in document")
        return []
    
    # 尝试不同 Adapter
    adapter_results: List = []
    
    # 1. 尝试 MinerU Adapter
    if figures:
        try:
            mineru_adapter = FigureAdapterFactory.get_adapter(FigureSource.MINERU)
            mineru_blocks = mineru_adapter.parse(
                "",  # pdf_path not needed for this adapter
                {"figures": figures, "images": images},
                page_width,
                page_height
            )
            if mineru_blocks:
                adapter_results.extend(mineru_blocks)
                logger.info(f"MineruAdapter: got {len(mineru_blocks)} blocks")
        except Exception as e:
            logger.warning(f"MineruAdapter failed: {e}")
    
    # 2. 如果 MinerU 没有结果，尝试 PDF Native Adapter
    if not adapter_results and figures:
        try:
            pdf_adapter = FigureAdapterFactory.get_adapter(FigureSource.PDF_NATIVE)
            pdf_blocks = pdf_adapter.parse(
                "",
                {"figures": figures, "images": images},
                page_width,
                page_height
            )
            if pdf_blocks:
                adapter_results.extend(pdf_blocks)
                logger.info(f"PDFFigureAdapter: got {len(pdf_blocks)} blocks")
        except Exception as e:
            logger.warning(f"PDFFigureAdapter failed: {e}")
    
    # 3. 如果都没有结果，使用 Fallback Adapter
    if not adapter_results and images:
        try:
            fallback_adapter = FigureAdapterFactory.get_adapter(FigureSource.FALLBACK)
            fallback_blocks = fallback_adapter.parse(
                "",
                {"figures": figures, "images": images},
                page_width,
                page_height
            )
            if fallback_blocks:
                adapter_results.extend(fallback_blocks)
                logger.info(f"FallbackFigureAdapter: got {len(fallback_blocks)} blocks")
        except Exception as e:
            logger.warning(f"FallbackFigureAdapter failed: {e}")
    
    if not adapter_results:
        return []
    
    # 4. 使用 Figure Builder 构建 Logical Figures
    logical_figures = build_logical_figures(
        adapter_results,
        page_width,
        page_height
    )
    
    # 5. 选取 top N
    selected = select_top_figures(logical_figures, depth)
    
    return selected


def _render_figures_with_pipeline(
    pdf_doc,
    figures: List[LogicalFigureSchema],
    figure_render_mode: str = "raw",
) -> List[Dict]:
    """
    使用 Figure Render + Validation 渲染 figures
    
    Returns:
        List[Dict]: 包含 image_base64 和渲染结果的 dict 列表
    """
    results = []
    render_mode = _normalize_figure_render_mode(figure_render_mode)
    
    for figure in figures:
        # MinerU 与视觉兜底已经提供图主体 bbox，不再用另一套检测二次裁剪。
        # 兼容旧客户端传入 yolo 模式，但不让它改写主解析给出的坐标。
        effective_render_mode = (
            "raw" if str(figure.source or "") in {"mineru", "yolo"} else render_mode
        )
        # 渲染
        render_result = validate_and_fallback(
            figure,
            pdf_doc,
            render_figure,
            render_kwargs={"render_mode": effective_render_mode},
        )
        
        results.append({
            "figure": figure,
            "render_result": render_result,
            "display_image_base64": render_result.display_image_base64 if render_result.success else None,
            "model_image_base64": render_result.model_image_base64 if render_result.success else None,
        })
    
    return results


def _build_key_figure_from_rendered(
    figure_data: Dict,
    index: int,
    reason: str = "analysis_unavailable",
) -> Optional[KeyFigureItem]:
    """图表已裁剪但多模态解析失败时，保留图片和图注给前端展示。"""
    figure = figure_data.get("figure")
    render_result = figure_data.get("render_result")

    if not figure or not render_result or not render_result.success:
        return None

    image_b64 = render_result.display_image_base64 or render_result.model_image_base64
    if not image_b64:
        return None

    caption = (
        (figure.caption_text or "").strip()
        or (figure.figure_index or "").strip()
        or f"图表 {index + 1}"
    )
    image_data = f"data:image/jpeg;base64,{image_b64}"

    if reason == "risk_not_triggered":
        analysis = caption
    elif reason == "analysis_unavailable":
        analysis = "图表已识别并裁剪完成。当前模型未返回可用的图表解析，可切换支持图片输入的模型后重新生成速览。"
    else:
        analysis = "图表已识别并裁剪完成，暂未生成详细解析。"

    return KeyFigureItem(
        figure_id=figure.figure_id,
        caption=caption,
        image_base64=image_data,
        analysis=analysis,
    )


def _build_key_figure_from_legacy_crop(
    fig: Dict,
    index: int,
    display_image_data: Optional[str],
    reason: str = "analysis_unavailable",
) -> Optional[KeyFigureItem]:
    """旧图表路径的展示兜底：PDF 裁剪成功时不要丢图。"""
    if not display_image_data:
        return None

    caption = (
        (fig.get("caption") or "").strip()
        or (fig.get("figure_label") or "").strip()
        or f"图表 {index + 1}"
    )

    return KeyFigureItem(
        figure_id=fig.get("figure_id", f"fig-{index + 1}"),
        caption=caption,
        image_base64=display_image_data,
        analysis=(
            caption
            if reason == "risk_not_triggered"
            else "图表已识别并裁剪完成。当前模型未返回可用的图表解析，可切换支持图片输入的模型后重新生成速览。"
        ),
    )


async def _generate_figure_analysis_via_pipeline(
    figure_data: Dict,
    api_key: str = "",
    model: str = "",
    provider: str = "",
    endpoint: str = "",
    visual_config: VisualModelConfig | None = None,
    document_id: str = "",
    parse_generation: str = "",
    visual_document_budget: int | None = None,
) -> Optional[KeyFigureItem]:
    """
    使用新 pipeline 生成的 figure 数据进行 LLM 分析
    
    Args:
        figure_data: _render_figures_with_pipeline 返回的 dict
        api_key, model, provider, endpoint: LLM 调用参数
        
    Returns:
        KeyFigureItem: 分析结果
    """
    figure = figure_data.get("figure")
    render_result = figure_data.get("render_result")
    
    if not figure or not render_result or not render_result.success:
        return None
    
    # display_image = 当前图表来源给出的展示区域（给用户看）
    # model_image  = 完整区域（给 LLM 分析，包含上下文）
    display_b64 = render_result.display_image_base64
    model_b64 = render_result.model_image_base64 or display_b64
    if not display_b64 and not model_b64:
        return None
    
    display_data = f"data:image/jpeg;base64,{display_b64 or model_b64}"
    caption_text = (figure.caption_text or "").strip()
    
    # 准备数据给 _generate_single_figure_analysis
    figure_info = {
        "figure_id": figure.figure_id,
        "figure_index": figure.page_idx,
        "image_data_list": [f"data:image/jpeg;base64,{model_b64}"],
        "figure_label": figure.figure_index or "",
        "page_content_snippet": "",
        "display_image_data": display_data,
        "caption": caption_text,
    }
    
    # 调用现有的分析函数
    result = await _generate_single_figure_analysis(
        figure_id=figure_info["figure_id"],
        figure_index=figure_info["figure_index"],
        image_data_list=figure_info["image_data_list"],
        figure_label=figure_info["figure_label"],
        page_content_snippet=figure_info.get("page_content_snippet", ""),
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        display_image_data=figure_info.get("display_image_data"),
        caption=figure_info.get("caption"),
        visual_config=visual_config,
        document_id=document_id,
        parse_generation=parse_generation,
        visual_document_budget=visual_document_budget,
    )
    
    return result


# ============ 原有函数保留，标记为旧版 ============

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
    image_data_list: List[str],
    figure_label: str,
    page_content_snippet: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    display_image_data: Optional[str] = None,
    caption: Optional[str] = None,
    sub_figures: Optional[List[dict]] = None,
    visual_config: VisualModelConfig | None = None,
    document_id: str = "",
    parse_generation: str = "",
    visual_document_budget: int | None = None,
) -> Optional[KeyFigureItem]:
    """
    调用多模态 LLM 对一个 figure（可能包含多张子图）生成标题与解析。
    返回 KeyFigureItem，失败返回 None。

    增强版（Phase 5）：
    - 支持 caption 上下文
    - 支持子图组提示
    """
    if not image_data_list:
        if display_image_data:
            image_data_list = [display_image_data]
        else:
            return None

    resolved_visual_config = visual_config or resolve_visual_model_config(
        primary_provider=provider,
        primary_model=model,
        primary_api_key=api_key,
        primary_endpoint=endpoint,
    )
    # 图表理解是可选增强。没有可用的 VLM 时保留原始裁剪图，由上层走
    # fallback item；绝不能让文字速览或主解析链路失败。
    if not resolved_visual_config.can_call:
        return None

    img_count = len(image_data_list)
    subfig_hint = f"（共 {img_count} 张子图）" if img_count > 1 else ""
    label_hint = f"已知图号/标题线索：{figure_label}" if figure_label else "图号/标题线索：未提供"
    fallback_images = _limit_sub_images(image_data_list)
    visual_inputs = [display_image_data] if display_image_data else fallback_images
    sent_image_count = len(visual_inputs)

    # Phase 5 增强：Caption 上下文
    caption_context = ""
    if caption:
        caption_context = f"\n图表官方图注：{caption}"

    # Phase 5 增强：子图整体解读提示
    subfigure_group_hint = ""
    if sub_figures and len(sub_figures) > 1:
        subfigure_group_hint = " 注意：这些子图属于同一Figure，请整体解读。"

    page_hint = f"\n页面文字：{page_content_snippet[:200]}" if page_content_snippet else ""
    prompt = f"""论文第{figure_index + 1}个图表。{label_hint}{caption_context}{page_hint}

请用中文输出JSON：1)caption:一句话标题(图X:xxx格式) 2)analysis:2-3句解析(图表类型、关键内容、趋势结论，看不清说"部分细节不可辨认")
{{"caption":"...","analysis":"..."}}

"""

    # 多模态消息：先文字后图片
    user_content = [{"type": "text", "text": prompt}]
    for img_data in visual_inputs:
        user_content.append({"type": "image_url", "image_url": {"url": img_data, "detail": "low"}})

    messages = [
        {"role": "system", "content": "你是学术论文图表分析助手。根据论文图片和上下文，输出指定 JSON 格式。"},
        {"role": "user", "content": user_content},
    ]

    try:
        async def _invoke_visual_model() -> Any:
            return await call_visual_model(
                messages=messages,
                config=resolved_visual_config,
                max_tokens=FIGURE_ANALYSIS_MAX_TOKENS,
                temperature=0.3,
                purpose="figure_analysis",
            )

        task_document_id = str(document_id or f"overview:{figure_id}")
        response = await execute_visual_task(
            task_id=build_visual_task_id({
                "document_id": task_document_id,
                "parse_generation": parse_generation,
                "purpose": "figure_description",
                "figure_id": figure_id,
                "visual_model_identity": resolved_visual_config.identity,
                "prompt_version": FIGURE_ANALYSIS_PROMPT_VERSION,
            }),
            document_id=task_document_id,
            parse_generation=parse_generation,
            purpose="figure_description",
            operation=_invoke_visual_model,
            policy=_overview_visual_task_policy(visual_document_budget),
            metadata={
                "provider": resolved_visual_config.provider,
                "model": resolved_visual_config.model,
                "source": resolved_visual_config.source,
                "page": figure_index + 1,
                "prompt_version": FIGURE_ANALYSIS_PROMPT_VERSION,
            },
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

        # 返回第一张图片作为展示缩略图
        return KeyFigureItem(
            figure_id=figure_id,
            caption=caption,
            image_base64=display_image_data or image_data_list[0],
            analysis=analysis,
        )
    except Exception as e:
        logger.warning(f"单张图表解析失败: {e}")
        return None


def _fallback_parse_cache_identity(doc_id: str) -> tuple[str, str]:
    """为没有内存文档的调用方返回稳定的解析身份。

    正常 API 请求始终从 ``parse_manifest`` 获取身份。确定性的回退值保留
    运维脚本和独立测试的行为，同时避免未知文档共用一个缓存槽。
    """
    source_hash = derive_source_hash({"overview_cache_doc_id": str(doc_id or "")})
    return f"legacy-{source_hash[:24]}", source_hash


async def _get_document_parse_cache_identity(doc_id: str) -> tuple[str, str]:
    """读取拥有当前速览缓存的主解析 generation。"""
    try:
        from routes.document_routes import documents_store

        document = documents_store.get(doc_id)
        if isinstance(document, dict):
            manifest = read_parse_manifest(document, doc_id=doc_id)
            generation = str(manifest.get("generation") or "").strip()
            source_hash = str(manifest.get("source_hash") or "").strip()
            if generation and source_hash:
                return generation, source_hash
    except Exception as exc:
        # 速览生成流程稍后还会读取文档。独立调用方没有路由存储时，不应让
        # 可选缓存查询变成导入期错误。
        logger.warning("读取速览解析身份失败 doc=%s error=%s", doc_id, exc)

    return _fallback_parse_cache_identity(doc_id)


async def _resolve_parse_cache_identity(
    doc_id: str,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> tuple[str, str]:
    """优先使用已捕获的身份，否则读取当前活动身份。"""
    generation = str(parse_generation or "").strip()
    source_hash = str(document_source_hash or "").strip()
    if generation and source_hash:
        return generation, source_hash
    return await _get_document_parse_cache_identity(doc_id)


def _parse_cache_identity_token(
    parse_generation: str = "",
    document_source_hash: str = "",
) -> str:
    """基于完整解析身份生成路径安全的缓存后缀。"""
    payload = json.dumps(
        {
            "generation": str(parse_generation or "").strip(),
            "source_hash": str(document_source_hash or "").strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _overview_text_model_identity(
    provider: str = "",
    model: str = "",
    endpoint: str = "",
    api_key: str = "",
) -> str:
    """Bind textual overview caches to a non-secret model availability state."""
    provider_l = str(provider or "").strip().lower()
    can_call = bool(str(api_key or "").strip()) or provider_l in {"local", "ollama"}
    return model_config_identity(
        provider,
        model,
        endpoint,
        available=can_call,
    )


def _identity_cache_token(value: str, fallback: str) -> str:
    return re.sub(r"[^a-f0-9]", "", str(value or "").lower())[:24] or fallback


def _get_cache_key(
    doc_id: str,
    depth: str,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
    visual_model_identity: str = "",
    text_model_identity: str = "",
) -> str:
    """Generate a cache key bound to parser, text-model, and VLM identities."""
    render_mode = _normalize_figure_render_mode(figure_render_mode)
    identity_token = _parse_cache_identity_token(parse_generation, document_source_hash)
    visual_token = _identity_cache_token(visual_model_identity, "none")
    text_token = _identity_cache_token(text_model_identity, "none")
    prompt_token = hashlib.sha256(
        VISUAL_SUPPLEMENT_PROMPT_SUITE_IDENTITY.encode("utf-8")
    ).hexdigest()[:12]
    return f"{OVERVIEW_CACHE_VERSION}_{doc_id}_{depth}_{render_mode}_{identity_token}_{visual_token}_{text_token}_{prompt_token}"


def _get_cache_path(
    doc_id: str,
    depth: str,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
    visual_model_identity: str = "",
    text_model_identity: str = "",
) -> Path:
    """获取缓存文件路径"""
    key = _get_cache_key(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    return CACHE_DIR / f"{key}.json"


def _get_legacy_cache_path(
    doc_id: str,
    depth: str,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
    visual_model_identity: str = "",
    text_model_identity: str = "",
) -> Path:
    """旧版 backend/data/overviews 缓存路径，用于一次性兼容迁移。"""
    key = _get_cache_key(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    return LEGACY_CACHE_DIR / f"{key}.json"


def _overview_matches_parse_identity(
    overview: OverviewData,
    parse_generation: str,
    document_source_hash: str,
    visual_model_identity: str = "",
    text_model_identity: str = "",
) -> bool:
    """即使文件名意外复用，也拒绝没有匹配 manifest 的旧缓存。"""
    return (
        str(overview.parse_generation or "").strip() == str(parse_generation or "").strip()
        and str(overview.document_source_hash or "").strip()
        == str(document_source_hash or "").strip()
        and str(overview.visual_model_identity or "").strip()
        == str(visual_model_identity or "").strip()
        and str(overview.text_model_identity or "").strip()
        == str(text_model_identity or "").strip()
    )


def _cached_visual_supplement_is_active(
    doc_id: str,
    overview: OverviewData,
    parse_generation: str,
    document_source_hash: str,
    visual_model_identity: str,
) -> bool:
    """Reject a cached VLM overview when another model owns active read blocks.

    Only local-route VLM summaries publish visual blocks.  Switching from
    model A to B changes those active blocks; reopening A must regenerate and
    republish A instead of showing A's overview alongside B's evidence.
    """
    expected_revision = str(overview.visual_supplement_revision or "").strip()
    cache_without_revision_is_valid = not bool(expected_revision)
    try:
        from routes.document_routes import documents_store

        doc = documents_store.get(doc_id)
        if not isinstance(doc, dict):
            return cache_without_revision_is_valid
        data = doc.get("data")
        if not isinstance(data, dict):
            return cache_without_revision_is_valid

        # ``read_parse_manifest`` intentionally derives a compatibility
        # identity for old documents. A visual supplement has stronger
        # publication requirements: it must belong to a durable, explicit
        # parser run, rather than an inferred legacy identity.
        raw_manifest = data.get("parse_manifest")
        if not isinstance(raw_manifest, dict):
            return cache_without_revision_is_valid
        raw_metadata = raw_manifest.get("metadata")
        if isinstance(raw_metadata, dict) and raw_metadata.get("legacy_inferred"):
            return cache_without_revision_is_valid
        migrated_from_legacy = bool(
            isinstance(raw_metadata, dict) and raw_metadata.get("migrated_from_legacy")
        )
        raw_generation = str(raw_manifest.get("generation") or "").strip()
        raw_source_hash = str(raw_manifest.get("source_hash") or "").strip()
        raw_resolved_route = str(raw_manifest.get("resolved_route") or "").strip().lower()
        if (
            not raw_generation
            or not raw_source_hash
            or raw_resolved_route != "local"
            or (
                not migrated_from_legacy
                and (
                    raw_generation.startswith("legacy-")
                    or raw_source_hash.startswith("legacy-")
                    or str(raw_manifest.get("stage") or "").strip().lower() == "legacy_ready"
                )
            )
        ):
            return cache_without_revision_is_valid

        manifest = read_parse_manifest(doc, doc_id=doc_id)
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        if (
            metadata.get("legacy_inferred")
            or not is_parse_prepared(manifest, route="local")
            or str(manifest.get("resolved_route") or "").strip().lower() != "local"
            or str(manifest.get("generation") or "") != raw_generation
            or str(manifest.get("source_hash") or "") != raw_source_hash
        ):
            return cache_without_revision_is_valid
        parse_identity = {
            "parser_route": "local",
            "parse_generation": raw_generation,
            "document_source_hash": raw_source_hash,
        }
        if not visual_supplements_are_committed(data, parse_identity=parse_identity):
            return cache_without_revision_is_valid
        envelope = data.get("visual_supplements")
        if not isinstance(envelope, dict):
            return cache_without_revision_is_valid
        active_revision = visual_supplement_revision(data, parse_identity)
        if not active_revision:
            return cache_without_revision_is_valid
        if not expected_revision:
            # A text-only cache predates the now-published visual evidence.
            # Regenerate it so the overview and shared reading blocks agree.
            return False
        return (
            str(envelope.get("parse_generation") or "") == str(parse_generation or "")
            and str(envelope.get("document_source_hash") or "") == str(document_source_hash or "")
            and str(envelope.get("visual_model_identity") or "") == str(visual_model_identity or "")
            and active_revision == expected_revision
        )
    except Exception as exc:
        logger.debug("[Overview] skipped cached supplement activity check doc=%s: %s", doc_id, exc)
        return cache_without_revision_is_valid


async def clear_overview_cache(
    doc_id: str,
    depth: str,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
    visual_model_identity: str | None = None,
    text_model_identity: str | None = None,
) -> None:
    """Delete overview caches for a parser generation.

    An omitted visual identity intentionally clears every VLM variant.  This is
    used by route changes and manual cache clearing, where selecting only one
    visual model would leave stale alternatives on disk.
    """
    parse_generation, document_source_hash = await _resolve_parse_cache_identity(
        doc_id,
        parse_generation,
        document_source_hash,
    )
    if visual_model_identity is None:
        render_mode = _normalize_figure_render_mode(figure_render_mode)
        parse_token = _parse_cache_identity_token(parse_generation, document_source_hash)
        prefix = f"{OVERVIEW_CACHE_VERSION}_{doc_id}_{depth}_{render_mode}_{parse_token}_"
        for cache_key in [key for key in overview_cache if key.startswith(prefix)]:
            overview_cache.pop(cache_key, None)
        for directory in (CACHE_DIR, LEGACY_CACHE_DIR):
            try:
                for cache_path in directory.glob(f"{prefix}*.json"):
                    cache_path.unlink()
            except Exception as exc:
                logger.warning("删除速览缓存失败 doc=%s: %s", doc_id, exc)
        return

    if text_model_identity is None:
        render_mode = _normalize_figure_render_mode(figure_render_mode)
        parse_token = _parse_cache_identity_token(parse_generation, document_source_hash)
        visual_token = _identity_cache_token(visual_model_identity, "none")
        prefix = f"{OVERVIEW_CACHE_VERSION}_{doc_id}_{depth}_{render_mode}_{parse_token}_{visual_token}_"
        for cache_key in [key for key in overview_cache if key.startswith(prefix)]:
            overview_cache.pop(cache_key, None)
        for directory in (CACHE_DIR, LEGACY_CACHE_DIR):
            try:
                for cache_path in directory.glob(f"{prefix}*.json"):
                    cache_path.unlink()
            except Exception as exc:
                logger.warning("删除速览缓存失败 doc=%s: %s", doc_id, exc)
        return

    cache_key = _get_cache_key(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    overview_cache.pop(cache_key, None)
    cache_path = _get_cache_path(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    try:
        if cache_path.exists():
            cache_path.unlink()
    except Exception as e:
        logger.warning(f"删除速览缓存失败: {e}")


async def get_cached_overview(
    doc_id: str,
    depth: str,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
    visual_model_identity: str = "",
    text_model_identity: str = "",
) -> Optional[OverviewData]:
    """获取与当前 primary parser generation 匹配的速览缓存。"""
    parse_generation, document_source_hash = await _resolve_parse_cache_identity(
        doc_id,
        parse_generation,
        document_source_hash,
    )
    cache_key = _get_cache_key(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )

    # 内存缓存
    cached = overview_cache.get(cache_key)
    if cached:
        if _overview_matches_parse_identity(
            cached,
            parse_generation,
            document_source_hash,
            visual_model_identity,
            text_model_identity,
        ) and _cached_visual_supplement_is_active(
            doc_id,
            cached,
            parse_generation,
            document_source_hash,
            visual_model_identity,
        ):
            return cached
        overview_cache.pop(cache_key, None)

    # 文件缓存。v13 之前的文件没有解析 identity，必须视为不匹配，
    # 不能在 MinerU 发布后迁移进当前 generation。
    cache_path = _get_cache_path(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    legacy_cache_path = _get_legacy_cache_path(
        doc_id,
        depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        visual_model_identity,
        text_model_identity,
    )
    source_path = cache_path if cache_path.exists() else legacy_cache_path
    if source_path.exists():
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            overview = OverviewData(**data)
            if not _overview_matches_parse_identity(
                overview,
                parse_generation,
                document_source_hash,
                visual_model_identity,
                text_model_identity,
            ) or not _cached_visual_supplement_is_active(
                doc_id,
                overview,
                parse_generation,
                document_source_hash,
                visual_model_identity,
            ):
                logger.info(
                    "忽略过期速览缓存 doc=%s depth=%s generation=%s",
                    doc_id,
                    depth,
                    parse_generation,
                )
                return None
            overview_cache[cache_key] = overview
            if source_path == legacy_cache_path:
                await save_overview_cache(
                    overview,
                    parse_generation=parse_generation,
                    document_source_hash=document_source_hash,
                )
            return overview
        except Exception as e:
            logger.warning(f"读取速览缓存失败: {e}")

    return None


async def save_overview_cache(
    overview: OverviewData,
    *,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> None:
    """保存已绑定 primary parser generation 的速览缓存。"""
    parse_generation = str(parse_generation or overview.parse_generation or "").strip()
    document_source_hash = str(
        document_source_hash or overview.document_source_hash or ""
    ).strip()
    if not parse_generation or not document_source_hash:
        # 缺少该元数据的缓存可能是上一路线遗留的本地结果，不能提升为当前
        # MinerU 路线的结果。
        logger.warning("跳过未绑定解析身份的速览缓存 doc=%s", overview.doc_id)
        return

    overview.parse_generation = parse_generation
    overview.document_source_hash = document_source_hash
    figure_render_mode = (overview.figure_meta or {}).get("render_mode", "raw")
    cache_key = _get_cache_key(
        overview.doc_id,
        overview.depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        overview.visual_model_identity,
        overview.text_model_identity,
    )

    # 文件缓存
    cache_path = _get_cache_path(
        overview.doc_id,
        overview.depth,
        figure_render_mode,
        parse_generation,
        document_source_hash,
        overview.visual_model_identity,
        overview.text_model_identity,
    )
    temp_path: str | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(overview.model_dump(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, cache_path)
        overview_cache[cache_key] = overview
    except Exception as e:
        logger.warning(f"保存速览缓存失败: {e}")
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _build_pipeline_visual_supplement(
    figure_data: dict,
    item: KeyFigureItem,
    visual_config: VisualModelConfig,
    render_mode: str,
) -> dict | None:
    figure = figure_data.get("figure") if isinstance(figure_data, dict) else None
    if not figure:
        return None
    bbox = getattr(figure, "full_bbox_page_pts", None) or getattr(figure, "body_bbox_page_pts", None)
    return build_visual_supplement(
        figure_id=str(getattr(figure, "figure_id", "") or item.figure_id),
        page=int(getattr(figure, "page_idx", -1) if getattr(figure, "page_idx", None) is not None else -1) + 1,
        bbox=bbox,
        caption=item.caption,
        analysis=item.analysis,
        visual_model_identity=visual_config.identity,
        provider=visual_config.provider,
        model=visual_config.model,
        render_mode=render_mode,
        purpose="figure_description",
        confidence=item.confidence,
        prompt_version=FIGURE_ANALYSIS_PROMPT_VERSION,
        route="local",
    )


def _build_legacy_visual_supplement(
    figure: dict,
    item: KeyFigureItem,
    visual_config: VisualModelConfig,
    render_mode: str,
) -> dict | None:
    if not isinstance(figure, dict):
        return None
    bbox = (
        figure.get("group_bbox")
        or figure.get("figure_bbox")
        or figure.get("bbox")
        or figure.get("image_bboxes", [None])[0]
    )
    page = figure.get("page_num") or figure.get("page") or 0
    return build_visual_supplement(
        figure_id=str(figure.get("figure_id") or item.figure_id),
        page=int(page or 0),
        bbox=bbox,
        caption=item.caption,
        analysis=item.analysis,
        visual_model_identity=visual_config.identity,
        provider=visual_config.provider,
        model=visual_config.model,
        render_mode=render_mode,
        purpose="figure_description",
        confidence=item.confidence,
        prompt_version=FIGURE_ANALYSIS_PROMPT_VERSION,
        route="local",
    )


async def generate_overview_content(
    doc_id: str,
    depth: str,
    document_text: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    visual_api_key: str = "",
    visual_model: str = "",
    visual_provider: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool = True,
    visual_policy_params: Optional[dict] = None,
    use_mineru_figures: bool = False,
    figure_render_mode: str = "raw",
    parse_generation: str = "",
    document_source_hash: str = "",
) -> OverviewData:
    """生成速览内容（调用 LLM）"""
    figure_render_mode = _normalize_figure_render_mode(figure_render_mode)
    visual_policy = _resolve_overview_visual_policy(
        provider=provider,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
        visual_policy_params=visual_policy_params,
    )
    visual_config = visual_policy.strong_model
    text_model_identity = _overview_text_model_identity(provider, model, endpoint, api_key)
    parse_generation, document_source_hash = await _resolve_parse_cache_identity(
        doc_id,
        parse_generation,
        document_source_hash,
    )
    logger.info(
        "[AI-Audit] purpose=overview doc=%s provider=%s model=%s depth=%s render_mode=%s visual=%s",
        doc_id,
        provider,
        model,
        depth,
        figure_render_mode,
        visual_policy.identity,
    )
    
    # 获取文档信息
    doc_info = await get_document_info(doc_id)
    title = doc_info.get("filename", "未知文档") if doc_info else "未知文档"
    
    # 构建 prompt
    prompt = _build_overview_prompt(depth)
    document_excerpt = _build_document_excerpt(document_text, depth)
    full_prompt = f"{prompt}\n\n{document_excerpt}"
    
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
            max_tokens=OVERVIEW_OUTPUT_MAX_TOKENS.get(
                depth,
                OVERVIEW_OUTPUT_MAX_TOKENS[OverviewDepth.STANDARD],
            ),
            purpose="overview",
        )
        
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response.get("error"))
        
        content = _extract_content_from_response(response)
        if not content or not content.strip():
            raise RuntimeError("模型返回为空，请稍后重试或切换模型")
        
        # 解析 JSON
        # 尝试提取 JSON 部分
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        
        try:
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(content)
        except json.JSONDecodeError as exc:
            preview = content[:240].replace("\n", " ")
            raise RuntimeError(f"模型返回的速览格式不是有效 JSON: {exc}; preview={preview}") from exc
        
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
            created_at=time.time(),
            ai_meta={
                "purpose": "overview",
                "provider": provider,
                "model": model,
                "text_model_identity": text_model_identity,
                "depth": depth,
                "render_mode": figure_render_mode,
                "visual_model": visual_config.public_metadata(),
                "visual_policy": visual_policy.public_metadata(),
            },
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            text_model_identity=text_model_identity,
            visual_model_identity=visual_policy.identity,
        )
        logger.info(
            "[AI-Audit] purpose=overview doc=%s provider=%s model=%s status=success depth=%s",
            doc_id,
            provider,
            model,
            depth,
        )
        
        # 关键图表解读：从文档提取图片并用多模态模型生成解析
        # [T2] 优先使用 figure_extraction 服务的缓存结果
        try:
            from services.figure_extraction import build_logical_figures_for_overview, get_figure_extraction_status
            from routes.document_routes import documents_store

            # 检查文档是否存在
            key_figures_list = []
            visual_supplement_items: list[dict] = []
            visual_risk_assessments: list[dict] = []
            pdf_doc = None
            doc_data: dict = {}

            if doc_id in documents_store:
                doc = documents_store[doc_id]
                doc_data = doc.get("data", {})
                pdf_url = doc.get("pdf_url", "")

                # 检查是否需要强制重建 figure extraction 缓存
                cache_status = get_figure_extraction_status(doc_data)
                force_rebuild = not cache_status.get("has_cache", False)
                # 如果已有 MinerU 数据（上传阶段产生）但缓存来源不匹配，强制重建
                has_mineru_data = bool(doc_data.get("ocr_result", {}).get("figures"))
                if has_mineru_data and cache_status.get("source") != "mineru":
                    force_rebuild = True
                    logger.info("[Overview] 已有 MinerU 数据，强制重建 figure extraction")

                logical_figures = build_logical_figures_for_overview(
                    doc_id,
                    doc,
                    depth,
                    force_rebuild=force_rebuild,
                )
                if logical_figures and pdf_url:
                    try:
                        # 解析 pdf_url 为本地文件路径（使用与 document_routes 一致的 UPLOAD_DIR）
                        from routes.document_routes import UPLOAD_DIR as _upload_dir2
                        _pdf_path = _upload_dir2 / pdf_url.split("/")[-1]
                        pdf_doc = fitz.open(str(_pdf_path))
                        rendered_figures = _render_figures_with_pipeline(
                            pdf_doc,
                            logical_figures,
                            figure_render_mode=figure_render_mode,
                        )

                        analysis_jobs: list[tuple[int, Any, VisualModelConfig]] = []
                        figure_items_by_index: dict[int, KeyFigureItem] = {}
                        for idx, fig_data in enumerate(rendered_figures):
                            figure = fig_data.get("figure") if isinstance(fig_data, dict) else None
                            if not figure:
                                continue
                            assessment = assess_figure_risk(
                                figure,
                                page_text=page_text_for_risk(doc_data, int(figure.page_idx) + 1),
                                threshold=visual_policy.risk_threshold,
                            )
                            selected_visual_config = visual_policy.select(
                                risk_level=assessment.level,
                                purpose="figure_description",
                            )
                            assessment_data = {
                                "figure_id": str(figure.figure_id or ""),
                                **assessment.to_dict(),
                                "selected_model": selected_visual_config.public_metadata(),
                            }
                            visual_risk_assessments.append(assessment_data)
                            fig_data["visual_risk"] = assessment_data
                            if not assessment.should_enrich:
                                fallback_item = _build_key_figure_from_rendered(
                                    fig_data,
                                    idx,
                                    reason="risk_not_triggered",
                                )
                                if fallback_item:
                                    figure_items_by_index[idx] = fallback_item
                                continue
                            if not selected_visual_config.can_call:
                                fallback_item = _build_key_figure_from_rendered(fig_data, idx)
                                if fallback_item:
                                    figure_items_by_index[idx] = fallback_item
                                continue
                            analysis_jobs.append((
                                idx,
                                _generate_figure_analysis_via_pipeline(
                                    fig_data,
                                    api_key,
                                    model,
                                    provider,
                                    endpoint,
                                    visual_config=selected_visual_config,
                                    document_id=doc_id,
                                    parse_generation=parse_generation,
                                    visual_document_budget=visual_policy.document_budget,
                                ),
                                selected_visual_config,
                            ))

                        analysis_results = await asyncio.gather(
                            *(job for _, job, _config in analysis_jobs),
                            return_exceptions=True,
                        ) if analysis_jobs else []

                        for (idx, _job, selected_visual_config), result in zip(analysis_jobs, analysis_results):
                            if isinstance(result, KeyFigureItem):
                                figure_items_by_index[idx] = result
                                supplement = _build_pipeline_visual_supplement(
                                    rendered_figures[idx],
                                    result,
                                    selected_visual_config,
                                    figure_render_mode,
                                )
                                if supplement:
                                    visual_supplement_items.append(supplement)
                            elif isinstance(result, Exception):
                                logger.warning(f"Figure analysis failed: {result}")
                                fallback_item = _build_key_figure_from_rendered(
                                    rendered_figures[idx],
                                    idx,
                                )
                                if fallback_item:
                                    figure_items_by_index[idx] = fallback_item
                            else:
                                fallback_item = _build_key_figure_from_rendered(
                                    rendered_figures[idx],
                                    idx,
                                )
                                if fallback_item:
                                    figure_items_by_index[idx] = fallback_item
                        key_figures_list.extend(
                            figure_items_by_index[idx]
                            for idx in sorted(figure_items_by_index)
                        )
                    except Exception as e:
                        logger.warning(f"New figure pipeline failed: {e}")
                    finally:
                        if pdf_doc:
                            pdf_doc.close()
                            pdf_doc = None

            # 如果新 pipeline 没有产生结果，回退到旧逻辑
            if not key_figures_list:
                # 获取 images, pages, figures 供旧逻辑使用
                images, pages, figures, pdf_url = await get_document_images_and_pages(doc_id)

                figures = _enrich_figures_with_pdf_geometry(pdf_url, figures)
                figures_to_analyze = _extract_figures_for_overview(images, pages, depth, figures)
                logger.info(
                    "速览生成: doc=%s depth=%s excerpt_chars=%s figure_count=%s sub_images=%s",
                    doc_id,
                    depth,
                    len(document_excerpt),
                    len(figures_to_analyze),
                    [len(fig.get("image_data_list", [])) for fig in figures_to_analyze],
                )
                
                for i, fig in enumerate(figures_to_analyze):
                    display_image_data = _render_figure_crop_from_pdf(
                        pdf_url=pdf_url,
                        page_num=fig.get("page_num", 0),
                        image_bboxes=fig.get("image_bboxes", []),
                        figure_bbox=fig.get("figure_bbox"),
                        caption_bbox=fig.get("caption_bbox"),
                        previous_caption_bbox=fig.get("previous_caption_bbox"),
                        figure_render_mode=figure_render_mode,
                    )
                    page_number = int(fig.get("page_num") or fig.get("page") or 0)
                    assessment = assess_figure_risk(
                        fig,
                        page_text=(
                            page_text_for_risk(doc_data, page_number)
                            or str(fig.get("page_content_snippet") or "")
                        ),
                        threshold=visual_policy.risk_threshold,
                    )
                    selected_visual_config = visual_policy.select(
                        risk_level=assessment.level,
                        purpose="figure_description",
                    )
                    visual_risk_assessments.append({
                        "figure_id": str(fig.get("figure_id") or ""),
                        **assessment.to_dict(),
                        "selected_model": selected_visual_config.public_metadata(),
                    })
                    item = None
                    if assessment.should_enrich and selected_visual_config.can_call:
                        item = await _generate_single_figure_analysis(
                            figure_id=fig["figure_id"],
                            figure_index=i,
                            image_data_list=fig.get("image_data_list", []),
                            figure_label=fig.get("figure_label", ""),
                            page_content_snippet=fig.get("page_content_snippet", ""),
                            api_key=api_key,
                            model=model,
                            provider=provider,
                            endpoint=endpoint,
                            display_image_data=display_image_data,
                            caption=fig.get("caption"),
                            sub_figures=fig.get("sub_figures"),
                            visual_config=selected_visual_config,
                            document_id=doc_id,
                            parse_generation=parse_generation,
                            visual_document_budget=visual_policy.document_budget,
                        )
                    if item:
                        key_figures_list.append(item)
                        supplement = _build_legacy_visual_supplement(
                            fig,
                            item,
                            selected_visual_config,
                            figure_render_mode,
                        )
                        if supplement:
                            visual_supplement_items.append(supplement)
                    else:
                        fallback_item = _build_key_figure_from_legacy_crop(
                            fig,
                            i,
                            display_image_data,
                            reason=(
                                "analysis_unavailable"
                                if assessment.should_enrich
                                else "risk_not_triggered"
                            ),
                        )
                        if fallback_item:
                            key_figures_list.append(fallback_item)
            
            if key_figures_list:
                overview.key_figures = key_figures_list

            # 写入 figure_meta
            figure_source = "unknown"
            if doc_id in documents_store:
                meta = documents_store[doc_id].get("data", {}).get("logical_figures_meta", {})
                figure_source = meta.get("source", "unknown")
            if not key_figures_list:
                figure_source = "none"
            overview.figure_meta = {
                "source": figure_source,
                "count": len(key_figures_list),
                "render_mode": figure_render_mode,
                "visual_model": visual_config.public_metadata(),
                "visual_policy": visual_policy.public_metadata(),
                "visual_risk": _summarize_visual_risk(visual_risk_assessments),
            }
            if visual_supplement_items:
                try:
                    from routes.document_routes import publish_visual_supplements

                    publication = publish_visual_supplements(
                        doc_id,
                        parse_generation=parse_generation,
                        document_source_hash=document_source_hash,
                        visual_model_identity=visual_policy.identity,
                        items=visual_supplement_items,
                    )
                    overview.visual_supplement_revision = str(publication.get("revision") or "")
                    overview.figure_meta["visual_supplement_revision"] = overview.visual_supplement_revision
                    overview.figure_meta["visual_supplements_published"] = bool(publication.get("published"))
                except Exception as exc:
                    logger.warning("[Overview] visual supplement publication skipped doc=%s: %s", doc_id, exc)
                
        except Exception as e:
            logger.warning(f"关键图表解读跳过: {e}")
        
        # 保存缓存
        await save_overview_cache(
            overview,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
        
        return overview
        
    except Exception as e:
        logger.error(
            "[AI-Audit] purpose=overview doc=%s provider=%s model=%s status=failed error=%s",
            doc_id,
            provider,
            model,
            e,
        )
        raise


async def build_fallback_overview_content(
    doc_id: str,
    depth: str,
    document_text: str,
    *,
    model: str = "",
    provider: str = "",
    figure_render_mode: str = "raw",
    error: str = "",
    parse_generation: str = "",
    document_source_hash: str = "",
    text_model_identity: str = "",
    visual_model_identity: str = "",
) -> OverviewData:
    """模型不可用时的确定性基础速览，避免前端空白报错。"""
    parse_generation, document_source_hash = await _resolve_parse_cache_identity(
        doc_id,
        parse_generation,
        document_source_hash,
    )
    doc_info = await get_document_info(doc_id)
    title = doc_info.get("filename", "未知文档") if doc_info else "未知文档"
    text = " ".join(str(document_text or "").split())
    if not text:
        text = "文档文本暂不可用。"
    summary = text[:520].rstrip()
    if len(text) > len(summary):
        summary += "..."

    sentences = [item.strip() for item in re.split(r"(?<=[。.!?])\s+|\n+", text) if item.strip()]
    first = sentences[0] if sentences else summary
    second = sentences[1] if len(sentences) > 1 else summary

    overview = OverviewData(
        doc_id=doc_id,
        title=title,
        depth=depth,
        full_text_summary=summary,
        terminology=[],
        speed_read=SpeedReadContent(
            method=first[:220],
            experiment_design="模型暂未生成实验设计速览；请检查模型连接后点击重新生成。",
            problems_solved=second[:220],
        ),
        key_figures=[],
        paper_summary=PaperSummary(
            strengths="当前展示基础速览，尚未完成 AI 深度总结。",
            innovations="模型连接恢复后可重新生成完整创新点分析。",
            future_work="建议稍后重试，或切换更稳定的模型服务商。",
        ),
        created_at=time.time(),
        figure_meta={
            "source": "fallback",
            "count": 0,
            "render_mode": _normalize_figure_render_mode(figure_render_mode),
            "generation_error": error,
        },
        ai_meta={
            "purpose": "overview",
            "provider": provider,
            "model": model,
            "text_model_identity": text_model_identity,
            "depth": depth,
            "render_mode": _normalize_figure_render_mode(figure_render_mode),
            "fallback": True,
            "generation_error": error,
            "visual_model_identity": visual_model_identity,
        },
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
        text_model_identity=text_model_identity,
        visual_model_identity=visual_model_identity,
    )
    await save_overview_cache(
        overview,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )
    return overview


async def create_overview_task(
    doc_id: str,
    depth: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    visual_api_key: str = "",
    visual_model: str = "",
    visual_provider: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool = True,
    visual_policy_params: Optional[dict] = None,
    use_mineru_figures: bool = False,
    figure_render_mode: str = "raw",
) -> OverviewTask:
    """创建异步任务"""
    task_id = str(uuid.uuid4())
    parse_generation, document_source_hash = await _get_document_parse_cache_identity(doc_id)
    
    task = OverviewTask(
        task_id=task_id,
        doc_id=doc_id,
        depth=depth,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        visual_api_key=visual_api_key,
        visual_model=visual_model,
        visual_provider=visual_provider,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
        visual_policy_params=dict(visual_policy_params or {}),
        status="pending",
        created_at=time.time(),
        updated_at=time.time(),
        use_mineru_figures=use_mineru_figures,
        figure_render_mode=_normalize_figure_render_mode(figure_render_mode),
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
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
        _use_mineru = getattr(task, 'use_mineru_figures', False)
        _render_mode = _normalize_figure_render_mode(getattr(task, 'figure_render_mode', 'raw'))
        visual_policy = _resolve_overview_visual_policy(
            provider=task.provider,
            model=task.model,
            api_key=task.api_key,
            endpoint=task.endpoint,
            visual_provider=getattr(task, "visual_provider", ""),
            visual_model=getattr(task, "visual_model", ""),
            visual_api_key=getattr(task, "visual_api_key", ""),
            visual_endpoint=getattr(task, "visual_endpoint", ""),
            visual_enabled=getattr(task, "visual_enabled", True),
            visual_policy_params=getattr(task, "visual_policy_params", {}),
        )
        text_model_identity = _overview_text_model_identity(
            task.provider,
            task.model,
            task.endpoint,
            task.api_key,
        )
        parse_generation, document_source_hash = await _resolve_parse_cache_identity(
            task.doc_id,
            getattr(task, "parse_generation", ""),
            getattr(task, "document_source_hash", ""),
        )
        task.parse_generation = parse_generation
        task.document_source_hash = document_source_hash
        cached = await get_cached_overview(
            task.doc_id,
            task.depth,
            _render_mode,
            parse_generation,
            document_source_hash,
            visual_policy.identity,
            text_model_identity,
        )
        if cached:
            if _use_mineru and (cached.figure_meta or {}).get("source") != "mineru":
                logger.info(f"[Overview] task: 缓存非 MinerU，跳过")
            else:
                task.result = cached
                task.status = "completed"
                task.updated_at = time.time()
                return
        
        result = await _generate_or_wait_overview(
            task.doc_id,
            task.depth,
            api_key=task.api_key,
            model=task.model,
            provider=task.provider,
            endpoint=task.endpoint,
            visual_api_key=getattr(task, "visual_api_key", ""),
            visual_model=getattr(task, "visual_model", ""),
            visual_provider=getattr(task, "visual_provider", ""),
            visual_endpoint=getattr(task, "visual_endpoint", ""),
            visual_enabled=getattr(task, "visual_enabled", True),
            visual_policy_params=getattr(task, "visual_policy_params", {}),
            use_mineru_figures=getattr(task, 'use_mineru_figures', False),
            figure_render_mode=_render_mode,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
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

async def _generate_or_wait_overview(
    doc_id: str,
    depth: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    visual_api_key: str = "",
    visual_model: str = "",
    visual_provider: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool = True,
    visual_policy_params: Optional[dict] = None,
    use_mineru_figures: bool = False,
    figure_render_mode: str = "raw",
    force: bool = False,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> OverviewData:
    """相同 doc/depth 的 overview 只生成一次，其余请求直接复用。"""
    render_mode = _normalize_figure_render_mode(figure_render_mode)
    visual_policy = _resolve_overview_visual_policy(
        provider=provider,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
        visual_policy_params=visual_policy_params,
    )
    text_model_identity = _overview_text_model_identity(provider, model, endpoint, api_key)
    parse_generation, document_source_hash = await _resolve_parse_cache_identity(
        doc_id,
        parse_generation,
        document_source_hash,
    )
    cache_key = _get_cache_key(
        doc_id,
        depth,
        render_mode,
        parse_generation,
        document_source_hash,
        visual_policy.identity,
        text_model_identity,
    )

    cached = await get_cached_overview(
        doc_id,
        depth,
        render_mode,
        parse_generation,
        document_source_hash,
        visual_policy.identity,
        text_model_identity,
    )
    if cached and not force:
        if use_mineru_figures and (cached.figure_meta or {}).get("source") != "mineru":
            logger.info(f"[Overview] _generate_or_wait: 缓存非 MinerU，跳过")
        else:
            return cached

    inflight = overview_inflight.get(cache_key)
    if inflight and not force:
        return await asyncio.shield(inflight)

    async def _runner() -> OverviewData:
        document_text = await get_document_text(doc_id)
        page_recovery_revision = ""
        page_recovery_diagnostics: dict[str, Any] = {}
        try:
            from routes.document_routes import (
                _resolve_document_pdf_path,
                documents_store,
                publish_visual_supplements,
            )

            active_doc = documents_store.get(doc_id)
            if isinstance(active_doc, dict):
                page_recovery = await recover_risky_local_pages(
                    doc_id=doc_id,
                    doc=active_doc,
                    pdf_path=_resolve_document_pdf_path(active_doc),
                    visual_policy=visual_policy,
                )
                recovered_text = str(page_recovery.get("text") or "").strip()
                recovery_items = page_recovery.get("items") or []
                page_recovery_revision = str(
                    page_recovery.get("visual_supplement_revision") or ""
                )
                page_recovery_diagnostics = dict(page_recovery.get("diagnostics") or {})
                recovery_generation = str(
                    page_recovery.get("parse_generation") or parse_generation
                )
                recovery_source_hash = str(
                    page_recovery.get("document_source_hash") or document_source_hash
                )
                recovery_is_published = not recovery_items
                if recovery_items:
                    publication = publish_visual_supplements(
                        doc_id,
                        parse_generation=recovery_generation,
                        document_source_hash=recovery_source_hash,
                        visual_model_identity=visual_policy.identity,
                        items=recovery_items,
                    )
                    page_recovery_revision = str(publication.get("revision") or "")
                    recovery_is_published = bool(
                        publication.get("published") or publication.get("committed")
                    )

                current_doc = documents_store.get(doc_id)
                current_manifest = (
                    read_parse_manifest(current_doc, doc_id=doc_id)
                    if isinstance(current_doc, dict)
                    else {}
                )
                recovery_identity_is_current = bool(
                    recovery_is_published
                    and str(current_manifest.get("resolved_route") or "").strip().lower() == "local"
                    and str(current_manifest.get("generation") or "") == recovery_generation
                    and str(current_manifest.get("source_hash") or "") == recovery_source_hash
                )
                if recovered_text and recovery_identity_is_current:
                    document_text = "\n\n".join(
                        part for part in (str(document_text or "").strip(), recovered_text) if part
                    )
                elif recovered_text:
                    page_recovery_revision = ""
                    page_recovery_diagnostics["discarded_reason"] = "parse_identity_changed"
        except Exception as exc:
            # 页级视觉只是 local 的兜底增强，单页或整批失败都不能阻断已有文本速览。
            logger.warning("[Overview] 页级视觉恢复跳过 doc=%s: %s", doc_id, exc)
        if not document_text:
            raise RuntimeError("文档未找到")

        result = await generate_overview_content(
            doc_id,
            depth,
            document_text,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            visual_api_key=visual_api_key,
            visual_model=visual_model,
            visual_provider=visual_provider,
            visual_endpoint=visual_endpoint,
            visual_enabled=visual_enabled,
            visual_policy_params=visual_policy_params,
            use_mineru_figures=use_mineru_figures,
            figure_render_mode=render_mode,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
        if page_recovery_diagnostics:
            result.ai_meta = dict(result.ai_meta or {})
            result.ai_meta["page_visual_recovery"] = page_recovery_diagnostics
        if page_recovery_revision and not result.visual_supplement_revision:
            result.visual_supplement_revision = page_recovery_revision
            result.figure_meta = dict(result.figure_meta or {})
            result.figure_meta["visual_supplement_revision"] = page_recovery_revision
        if page_recovery_revision or page_recovery_diagnostics:
            await save_overview_cache(
                result,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )
        return result

    task = asyncio.create_task(_runner())
    overview_inflight[cache_key] = task
    try:
        return await asyncio.shield(task)
    finally:
        if overview_inflight.get(cache_key) is task:
            overview_inflight.pop(cache_key, None)


async def get_or_create_overview(
    doc_id: str,
    depth: str = "standard",
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    visual_api_key: str = "",
    visual_model: str = "",
    visual_provider: str = "",
    visual_endpoint: str = "",
    visual_enabled: bool = True,
    visual_policy_params: Optional[dict] = None,
    use_mineru_figures: bool = False,
    figure_render_mode: str = "raw",
    force: bool = False,
) -> OverviewData:
    """获取或创建速览（同步接口）"""
    render_mode = _normalize_figure_render_mode(figure_render_mode)
    visual_policy = _resolve_overview_visual_policy(
        provider=provider,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        visual_provider=visual_provider,
        visual_model=visual_model,
        visual_api_key=visual_api_key,
        visual_endpoint=visual_endpoint,
        visual_enabled=visual_enabled,
        visual_policy_params=visual_policy_params,
    )
    text_model_identity = _overview_text_model_identity(provider, model, endpoint, api_key)
    parse_generation, document_source_hash = await _get_document_parse_cache_identity(doc_id)
    cached = await get_cached_overview(
        doc_id,
        depth,
        render_mode,
        parse_generation,
        document_source_hash,
        visual_policy.identity,
        text_model_identity,
    )
    if cached and not force:
        logger.info(
            "[AI-Audit] purpose=overview doc=%s provider=%s model=%s status=cache_hit depth=%s",
            doc_id,
            (cached.ai_meta or {}).get("provider", ""),
            (cached.ai_meta or {}).get("model", ""),
            depth,
        )
        return cached

    try:
        return await asyncio.wait_for(
            _generate_or_wait_overview(
                doc_id,
                depth,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                visual_api_key=visual_api_key,
                visual_model=visual_model,
                visual_provider=visual_provider,
                visual_endpoint=visual_endpoint,
                visual_enabled=visual_enabled,
                visual_policy_params=visual_policy_params,
                use_mineru_figures=use_mineru_figures,
                figure_render_mode=render_mode,
                force=force,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            ),
            timeout=180,
        )
    except asyncio.TimeoutError as e:
        if cached:
            logger.warning("[Overview] 重新生成超时，返回旧缓存: doc=%s depth=%s", doc_id, depth)
            logger.info(
                "[AI-Audit] purpose=overview doc=%s provider=%s model=%s status=cache_hit_after_failure depth=%s",
                doc_id,
                (cached.ai_meta or {}).get("provider", ""),
                (cached.ai_meta or {}).get("model", ""),
                depth,
            )
            return cached
        document_text = await get_document_text(doc_id)
        return await build_fallback_overview_content(
            doc_id,
            depth,
            document_text,
            model=model,
            provider=provider,
            figure_render_mode=render_mode,
            error="速览生成超时",
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            text_model_identity=text_model_identity,
            visual_model_identity=visual_policy.identity,
        )
    except Exception as e:
        if cached:
            logger.warning("[Overview] 重新生成失败，返回旧缓存: doc=%s depth=%s error=%s", doc_id, depth, e)
            logger.info(
                "[AI-Audit] purpose=overview doc=%s provider=%s model=%s status=cache_hit_after_failure depth=%s",
                doc_id,
                (cached.ai_meta or {}).get("provider", ""),
                (cached.ai_meta or {}).get("model", ""),
                depth,
            )
            return cached
        document_text = await get_document_text(doc_id)
        return await build_fallback_overview_content(
            doc_id,
            depth,
            document_text,
            model=model,
            provider=provider,
            figure_render_mode=render_mode,
            error=str(e),
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
            text_model_identity=text_model_identity,
            visual_model_identity=visual_policy.identity,
        )
