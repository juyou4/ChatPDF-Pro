"""
Figure Render - 图像裁剪与渲染服务

使用 PyMuPDF 按区域直接渲染图像。

核心函数：
- crop_figure_image: 使用 clip 直接渲染
- generate_display_and_model_images: 同时生成展示图和分析图
"""
import base64
import logging
from typing import Tuple, Optional, List
import fitz  # PyMuPDF

from schemas.figure_schema import LogicalFigureSchema, RenderResult, FigureImageOutput

logger = logging.getLogger(__name__)

# 页级 YOLO 缓存: key=page_number → List[fitz.Rect] (ImageBody bboxes in page pts)
# 避免同页多 figure 重复跑 YOLO（CPU ~2-5s/次）
_yolo_page_cache: dict = {}


# 渲染配置
RENDER_CONFIG = {
    # 展示图：JPEG, 中等分辨率（web 显示 ~600px 宽，150 DPI 足够）
    "display_dpi": 150,
    "display_format": "jpeg",
    "display_jpg_quality": 90,
    
    # 分析图：JPEG, 低分辨率（给 LLM，detail:low 只用 512px）
    "model_dpi": 120,
    "model_format": "jpeg",
    "model_jpg_quality": 65,
    
    # 裁剪边距 (points)
    "padding": 5,
    
    # 最小尺寸阈值
    "min_width": 50,
    "min_height": 50,
}


def crop_figure_image(
    pdf_doc,
    page_idx: int,
    bbox_page_pts: List[float],
    dpi: int = 150,
    output_format: str = "png",
    jpg_quality: int = 85
) -> Tuple[bytes, int, int]:
    """使用 PyMuPDF clip 直接渲染
    
    Args:
        pdf_doc: PyMuPDF 文档对象
        page_idx: 页码 (0-indexed)
        bbox_page_pts: page_points 坐标系的 bbox [x0, y0, x1, y1]
        dpi: 渲染 DPI
        output_format: 输出格式 "png" 或 "jpeg"
        jpg_quality: JPEG 质量 (1-100)
        
    Returns:
        (image_bytes, width, height)
    """
    if not pdf_doc or page_idx < 0 or page_idx >= len(pdf_doc):
        raise ValueError(f"Invalid page_idx: {page_idx}")
    
    if not bbox_page_pts or len(bbox_page_pts) != 4:
        raise ValueError(f"Invalid bbox: {bbox_page_pts}")
    
    page = pdf_doc[page_idx]
    
    # 创建 clip 区域，与页面求交
    clip_rect = fitz.Rect(bbox_page_pts)
    clip_rect = clip_rect.intersect(page.rect)
    
    if clip_rect.is_empty or clip_rect.width <= 0 or clip_rect.height <= 0:
        raise ValueError(f"Empty clip rect: {clip_rect}")
    
    # 渲染
    pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
    
    # 转换为字节
    if output_format == "jpeg":
        try:
            img_bytes = pix.tobytes("jpeg", quality=jpg_quality)
        except TypeError:
            img_bytes = pix.tobytes("jpeg")
    else:
        img_bytes = pix.tobytes("png")
    
    return img_bytes, pix.width, pix.height


def generate_display_and_model_images(
    pdf_doc,
    figure: LogicalFigureSchema
) -> FigureImageOutput:
    """为一个 Figure 生成展示图和分析图
    
    Args:
        pdf_doc: PyMuPDF 文档对象
        figure: LogicalFigureSchema
        
    Returns:
        FigureImageOutput: 包含两类图像
    """
    cfg = RENDER_CONFIG
    page_idx = figure.page_idx
    
    # 优先使用 body_bbox_page_pts 作为展示图
    display_bbox = figure.body_bbox_page_pts
    
    # 如果没有 body bbox，回退到 full bbox
    if not display_bbox:
        display_bbox = figure.full_bbox_page_pts
    
    # 如果都没有，无法渲染
    if not display_bbox:
        raise ValueError(f"No bbox available for figure {figure.figure_id}")
    
    # 借鉴 MinerU ImageBody 思想：收紧到纯图像区域
    page = pdf_doc[page_idx]
    tight_bbox = _tighten_bbox_to_images(page, display_bbox)
    if tight_bbox:
        display_bbox = tight_bbox
    
    # 添加边距
    padded_display_bbox = _add_padding(display_bbox, cfg["padding"])
    
    # 渲染展示图
    try:
        display_bytes, display_width, display_height = crop_figure_image(
            pdf_doc,
            page_idx,
            padded_display_bbox,
            dpi=cfg["display_dpi"],
            output_format=cfg["display_format"]
        )
    except Exception as e:
        logger.warning(f"Failed to render display image for {figure.figure_id}: {e}")
        display_bytes, display_width, display_height = b"", 0, 0
    
    # 渲染分析图：优先使用 full_bbox
    model_bbox = figure.full_bbox_page_pts or figure.body_bbox_page_pts
    
    if model_bbox:
        padded_model_bbox = _add_padding(model_bbox, cfg["padding"])
        
        try:
            model_bytes, model_width, model_height = crop_figure_image(
                pdf_doc,
                page_idx,
                padded_model_bbox,
                dpi=cfg["model_dpi"],
                output_format=cfg["model_format"],
                jpg_quality=cfg["model_jpg_quality"]
            )
        except Exception as e:
            logger.warning(f"Failed to render model image for {figure.figure_id}: {e}")
            model_bytes, model_width, model_height = b"", 0, 0
    else:
        model_bytes, model_width, model_height = b"", 0, 0
    
    # Base64 编码
    display_base64 = base64.b64encode(display_bytes).decode("ascii") if display_bytes else ""
    model_base64 = base64.b64encode(model_bytes).decode("ascii") if model_bytes else ""
    
    return FigureImageOutput(
        display_image_base64=display_base64,
        display_width=display_width,
        display_height=display_height,
        model_image_base64=model_base64,
        model_width=model_width,
        model_height=model_height
    )


def render_figure(
    pdf_doc,
    figure: LogicalFigureSchema,
    padding: int = 5
) -> RenderResult:
    """渲染单个 Figure，返回 RenderResult
    
    这是主要入口函数，供 overview_service 调用。
    """
    cfg = RENDER_CONFIG
    page_idx = figure.page_idx
    
    # 获取渲染 bbox
    render_bbox = figure.body_bbox_page_pts or figure.full_bbox_page_pts
    
    if not render_bbox:
        return RenderResult(
            success=False,
            error_message=f"No bbox available for figure {figure.figure_id}"
        )
    
    # 借鉴 MinerU ImageBody 思想：收紧到纯图像区域
    # 用 full_bbox 作为搜索范围（覆盖所有子图），body_bbox 可能只覆盖部分子图
    page = pdf_doc[page_idx]
    search_bbox = figure.full_bbox_page_pts or render_bbox
    tight_bbox = _tighten_bbox_to_images(page, search_bbox)
    display_bbox = tight_bbox if tight_bbox else render_bbox
    
    # 添加边距
    padded_bbox = _add_padding(display_bbox, padding)
    
    # 渲染展示图
    try:
        display_bytes, display_width, display_height = crop_figure_image(
            pdf_doc,
            page_idx,
            padded_bbox,
            dpi=cfg["display_dpi"],
            output_format=cfg["display_format"],
            jpg_quality=cfg.get("display_jpg_quality", 90)
        )
    except Exception as e:
        return RenderResult(
            success=False,
            error_message=f"Failed to render display image: {str(e)}"
        )
    
    # 渲染分析图
    model_bytes = b""
    model_width, model_height = 0, 0
    
    if figure.full_bbox_page_pts:
        model_bbox = _add_padding(figure.full_bbox_page_pts, padding)
        try:
            model_bytes, model_width, model_height = crop_figure_image(
                pdf_doc,
                page_idx,
                model_bbox,
                dpi=cfg["model_dpi"],
                output_format=cfg["model_format"],
                jpg_quality=cfg["model_jpg_quality"]
            )
        except Exception as e:
            logger.warning(f"Failed to render model image: {e}")
    
    # Base64 编码
    display_base64 = base64.b64encode(display_bytes).decode("ascii")
    model_base64 = base64.b64encode(model_bytes).decode("ascii") if model_bytes else ""
    
    return RenderResult(
        success=True,
        display_image_base64=display_base64,
        display_width=display_width,
        display_height=display_height,
        model_image_base64=model_base64,
        model_width=model_width,
        model_height=model_height
    )


def _tighten_bbox_to_images(
    page,
    bbox_page_pts: List[float],
    min_coverage: float = 0.05
) -> Optional[List[float]]:
    """收紧 bbox 到纯图像区域

    优先使用 DocLayout-YOLO 模型检测 ImageBody（与 MinerU 同源），
    若模型不可用则回退到 PyMuPDF get_image_info()。

    Args:
        page: PyMuPDF Page 对象
        bbox_page_pts: 原始 figure bbox [x0, y0, x1, y1]
        min_coverage: 图片与 figure bbox 交集面积占图片面积的最小比例

    Returns:
        收紧后的 image-only bbox，或 None（无法收紧时回退原始 bbox）
    """
    if not bbox_page_pts or len(bbox_page_pts) != 4:
        return None

    # 优先尝试 DocLayout-YOLO
    yolo_result = _tighten_bbox_with_layout_model(page, bbox_page_pts)
    if yolo_result is not None:
        return yolo_result

    # 回退到 get_image_info
    return _tighten_bbox_with_image_info(page, bbox_page_pts, min_coverage)


def _get_page_body_rects(page) -> Optional[List]:
    """获取页面所有 ImageBody 的 fitz.Rect 列表（带页级缓存）

    首次调用渲染页面 + YOLO 推理，后续同页直接返回缓存。
    """
    global _yolo_page_cache

    page_num = page.number
    if page_num in _yolo_page_cache:
        return _yolo_page_cache[page_num]

    try:
        from services.layout_service import is_available, get_image_body_bboxes, pixel_bbox_to_page_pts
        if not is_available():
            return None
    except ImportError:
        return None

    try:
        pix = page.get_pixmap(dpi=144)
        img_data = pix.tobytes("png")

        from PIL import Image as PILImage
        import io
        page_image = PILImage.open(io.BytesIO(img_data))

        page_w_px, page_h_px = page_image.size
        page_w_pts = page.rect.width
        page_h_pts = page.rect.height

        body_bboxes_px = get_image_body_bboxes(page_image, conf=0.15)
        if not body_bboxes_px:
            _yolo_page_cache[page_num] = []
            return []

        rects = []
        for bbox_px in body_bboxes_px:
            bbox_pts = pixel_bbox_to_page_pts(
                bbox_px, page_w_px, page_h_px, page_w_pts, page_h_pts
            )
            rects.append(fitz.Rect(bbox_pts))

        # 缓存上限
        if len(_yolo_page_cache) >= 30:
            _yolo_page_cache.clear()
        _yolo_page_cache[page_num] = rects
        return rects

    except Exception as e:
        logger.warning(f"[LayoutModel] Page detection failed: {e}")
        return None


def _tighten_bbox_with_layout_model(
    page,
    bbox_page_pts: List[float],
) -> Optional[List[float]]:
    """使用 DocLayout-YOLO 检测 ImageBody 区域（与 MinerU 同源模型）

    采用 y-band 聚类策略：先找与 figure bbox 重叠的 ImageBody，
    再扩展到同一水平带（y-range 重叠）的所有 ImageBody，
    从而正确处理多子图并排的 Figure（如 Figure 1 的 a/b/c）。

    使用页级缓存，同页多 figure 只推理一次。
    """
    fig_rect = fitz.Rect(bbox_page_pts)
    if fig_rect.is_empty:
        return None

    all_body_rects = _get_page_body_rects(page)
    if not all_body_rects:
        return None

    # 第一步：找与 figure bbox 直接重叠的 ImageBody
    seed_rects = []
    for body_rect in all_body_rects:
        isect = fig_rect.intersect(body_rect)
        if isect.is_empty:
            continue
        body_area = body_rect.width * body_rect.height
        if body_area > 0 and (isect.width * isect.height) / body_area >= 0.05:
            seed_rects.append(body_rect)

    if not seed_rects:
        return None

    # 第二步：y-band 扩展 —— 找同一水平带的其他 ImageBody（处理并排子图）
    seed_y0 = min(r.y0 for r in seed_rects)
    seed_y1 = max(r.y1 for r in seed_rects)
    y_tolerance = (seed_y1 - seed_y0) * 0.3

    matched_rects = list(seed_rects)
    for body_rect in all_body_rects:
        if body_rect in seed_rects:
            continue
        if body_rect.y1 >= seed_y0 - y_tolerance and body_rect.y0 <= seed_y1 + y_tolerance:
            matched_rects.append(body_rect)

    # 合并所有匹配的 ImageBody bbox
    union = matched_rects[0]
    for r in matched_rects[1:]:
        union = union | r

    union = union.intersect(page.rect)
    if union.is_empty:
        return None

    logger.info(
        f"[LayoutModel] Tightened bbox: {[round(v,1) for v in [union.x0, union.y0, union.x1, union.y1]]} "
        f"({len(seed_rects)} seed + {len(matched_rects) - len(seed_rects)} expanded)"
    )
    return [union.x0, union.y0, union.x1, union.y1]


def _tighten_bbox_with_image_info(
    page,
    bbox_page_pts: List[float],
    min_coverage: float = 0.05
) -> Optional[List[float]]:
    """回退方案：使用 PyMuPDF get_image_info() 收紧 bbox"""
    fig_rect = fitz.Rect(bbox_page_pts)
    if fig_rect.is_empty:
        return None

    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:
        return None

    matched_rects = []
    for info in image_infos:
        img_bbox = info.get("bbox")
        if not img_bbox or len(img_bbox) != 4:
            continue
        img_rect = fitz.Rect(img_bbox)
        if img_rect.is_empty or img_rect.width < 20 or img_rect.height < 20:
            continue

        isect = fig_rect.intersect(img_rect)
        if isect.is_empty:
            continue

        img_area = img_rect.width * img_rect.height
        if img_area > 0 and (isect.width * isect.height) / img_area >= min_coverage:
            matched_rects.append(img_rect)

    if not matched_rects:
        return None

    union = matched_rects[0]
    for r in matched_rects[1:]:
        union = union | r

    union = union.intersect(page.rect)
    if union.is_empty:
        return None

    return [union.x0, union.y0, union.x1, union.y1]


def _add_padding(bbox: List[float], padding: float) -> List[float]:
    """给 bbox 添加边距"""
    if not bbox or len(bbox) != 4:
        return bbox
    
    x0, y0, x1, y1 = bbox
    
    return [
        max(0, x0 - padding),
        max(0, y0 - padding),
        x1 + padding,
        y1 + padding
    ]


__all__ = [
    "crop_figure_image",
    "generate_display_and_model_images",
    "render_figure",
    "RENDER_CONFIG",
]
