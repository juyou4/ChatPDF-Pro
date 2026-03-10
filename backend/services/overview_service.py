"""
速览（Overview）服务 - 生成结构化 AI 学术导读
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from io import BytesIO
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
OVERVIEW_CACHE_VERSION = "v4"

# 任务存储（生产环境可替换为 Redis）
overview_tasks: Dict[str, OverviewTask] = {}
overview_cache: Dict[str, OverviewData] = {}
overview_inflight: Dict[str, asyncio.Task] = {}

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
        min_width = page_width * 0.68
        min_height = page_height * 0.26
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
    band_top = max(0.0, (prev[3] + 6.0) if prev else 0.0)
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

            if not image_data_list:
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
                "caption_bbox": fig.get("caption_bbox") or fig.get("bbox"),
                "previous_caption_bbox": previous_caption_bbox_by_ref.get(id(fig)),
                "page_num": page_num,
                "page_content_snippet": page_content,
                "figure_label": fig.get("label", ""),
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
) -> Optional[KeyFigureItem]:
    """
    调用多模态 LLM 对一个 figure（可能包含多张子图）生成标题与解析。
    返回 KeyFigureItem，失败返回 None。

    增强版（Phase 5）：
    - 支持 caption 上下文
    - 支持子图组提示
    """
    if not image_data_list:
        return None

    from services.chat_service import call_ai_api

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
        subfigure_group_hint = """
【重要】这些图片属于同一个 Figure group（可能包含 a, b, c 等子图）。
请先分别简述各子图的内容，再总结它们共同说明的整体结论。
不要把它们视为互不相关的独立图片。"""

    prompt = f"""这是一篇学术论文中的第 {figure_index + 1} 个关键图表{subfig_hint}。
{label_hint}{caption_context}{subfigure_group_hint}
本次实际提供给你分析的视觉输入数量：{sent_image_count}。

该图所在页面的部分文字如下：

{page_content_snippet[:FIGURE_PAGE_TEXT_LIMIT] if page_content_snippet else "（无正文）"}

请完成两件事（用中文）：
1. 用一句话概括该图标题；如果能识别图号，请保持为「图X: xxx」格式。
2. 写一段 2-4 句话的解析，优先说明：
   - 这是架构图、流程图、实验对比图、可视化结果图还是别的图表；
   - 图中的关键模块、子图关系、输入输出或对比对象；
   - 如果是曲线图/柱状图/统计图，请指出坐标轴、图例、主要趋势和论文结论；
   - 如果局部文字看不清，请明确说“部分细节不可辨认”，不要臆测。

请严格按以下 JSON 格式输出，不要包含其他文字：
{{"caption": "图X: 标题", "analysis": "解析段落"}}
"""

    # 多模态消息：先文字后图片
    user_content = [{"type": "text", "text": prompt}]
    for img_data in visual_inputs:
        user_content.append({"type": "image_url", "image_url": {"url": img_data}})

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
            max_tokens=FIGURE_ANALYSIS_MAX_TOKENS,
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


def _get_cache_key(doc_id: str, depth: str) -> str:
    """生成缓存 key"""
    return f"{OVERVIEW_CACHE_VERSION}_{doc_id}_{depth}"


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
            images, pages, figures, pdf_url = await get_document_images_and_pages(doc_id)
            figures = _enrich_figures_with_pdf_geometry(pdf_url, figures)
            if images:
                figures_to_analyze = _extract_figures_for_overview(images, pages, depth, figures)
                logger.info(
                    "速览生成: doc=%s depth=%s excerpt_chars=%s figure_count=%s sub_images=%s",
                    doc_id,
                    depth,
                    len(document_excerpt),
                    len(figures_to_analyze),
                    [len(fig.get("image_data_list", [])) for fig in figures_to_analyze],
                )
                key_figures_list = []
                for i, fig in enumerate(figures_to_analyze):
                    display_image_data = _render_figure_crop_from_pdf(
                        pdf_url=pdf_url,
                        page_num=fig.get("page_num", 0),
                        image_bboxes=fig.get("image_bboxes", []),
                        figure_bbox=fig.get("figure_bbox"),
                        caption_bbox=fig.get("caption_bbox"),
                        previous_caption_bbox=fig.get("previous_caption_bbox"),
                    )
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
        
        result = await _generate_or_wait_overview(
            task.doc_id,
            task.depth,
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

async def _generate_or_wait_overview(
    doc_id: str,
    depth: str,
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
) -> OverviewData:
    """相同 doc/depth 的 overview 只生成一次，其余请求直接复用。"""
    cache_key = _get_cache_key(doc_id, depth)

    cached = await get_cached_overview(doc_id, depth)
    if cached:
        return cached

    inflight = overview_inflight.get(cache_key)
    if inflight:
        return await asyncio.shield(inflight)

    async def _runner() -> OverviewData:
        document_text = await get_document_text(doc_id)
        if not document_text:
            raise RuntimeError("文档未找到")

        return await generate_overview_content(
            doc_id,
            depth,
            document_text,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )

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
) -> OverviewData:
    """获取或创建速览（同步接口）"""
    # 先检查缓存
    cached = await get_cached_overview(doc_id, depth)
    if cached:
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
            ),
            timeout=120,
        )
    except asyncio.TimeoutError as e:
        raise TimeoutError("速览生成超时") from e
