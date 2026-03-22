"""
Figure Schema - 标准化 Logical Figure 数据结构定义

本模块定义后端统一使用的 Figure 数据结构，包括:
- LogicalFigureSchema: 标准化 Figure 输出
- FigureImageOutput: 图像渲染结果
- OverviewFigureItem: Overview 展示项
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
from enum import Enum
from pydantic import BaseModel


class FigureSource(str, Enum):
    """Figure 来源枚举"""
    MINERU = "mineru"
    PDF_NATIVE = "pdf_native"
    FALLBACK = "fallback"


class BBoxCoordinateSpace(str, Enum):
    """坐标空间类型枚举"""
    NORMALIZED_0_1000 = "normalized_0_1000"
    PAGE_POINTS = "page_points"
    RENDER_PIXELS = "render_pixels"
    UNKNOWN = "unknown"


class LogicalFigureSchema(BaseModel):
    """标准化输出：一个完整的 Figure
    
    后端所有 Figure 处理流程统一使用此 Schema，
    统一坐标系统（page_points），统一输出格式。
    """
    figure_id: str                           # 唯一标识，例如 "fig_p3_01"
    page_idx: int                             # 0-indexed 页码
    figure_index: Optional[str] = None       # "Figure 1" / "图2" / None
    caption_text: str = ""                   # 图注文本，可为空

    # 展示用：尽量干净，只含图主体
    body_bbox_page_pts: Optional[List[float]] = None   # [x0, y0, x1, y1]

    # 分析用：图主体 + caption + 可选 footnote
    full_bbox_page_pts: Optional[List[float]] = None   # [x0, y0, x1, y1]

    # 可选：子图列表 (a)(b)(c)(d) 等
    panel_bboxes_page_pts: List[List[float]] = field(default_factory=list)

    # 来源
    source: str = "unknown"                  # mineru / pdf_native / fallback
    confidence: float = 0.0

    # 调试与追踪
    source_metadata: Dict = field(default_factory=dict)

    # 质量门结果
    validation_status: Optional[Dict] = None

    class Config:
        use_enum_values = True


class FigureImageOutput(BaseModel):
    """一个 Figure 输出的两类图像"""
    display_image_base64: str     # PNG, 高分辨率，展示用
    display_width: int
    display_height: int
    model_image_base64: str      # JPEG, 中等分辨率，分析用
    model_width: int
    model_height: int


class OverviewFigureItem(BaseModel):
    """给 overview 用的最终 figure 项
    
    前端直接消费此数据结构进行展示。
    """
    figure_id: str
    figure_index: str
    caption: str
    image_base64: str            # 展示图（前端直接用）
    analysis: str                # LLM 分析结果
    source: str                  # 来源标识
    confidence: float = 0.0


@dataclass
class FigureBlock:
    """内部使用的 Figure 块结构
    
    在 Builder 阶段使用，尚未转换为 Schema。
    """
    figure_id: str
    page_idx: int
    figure_index: Optional[str] = None
    caption_text: str = ""
    
    # 原始 bbox 列表（可能多个子图）
    raw_bboxes: List[List[float]] = field(default_factory=list)
    
    # 标准化后的 bbox
    body_bbox_page_pts: Optional[List[float]] = None
    full_bbox_page_pts: Optional[List[float]] = None
    
    # 子图
    panel_bboxes_page_pts: List[List[float]] = field(default_factory=list)
    
    # 来源
    source: str = "unknown"
    confidence: float = 0.0
    
    # 原始数据
    source_metadata: Dict = field(default_factory=dict)


@dataclass
class RenderResult:
    """渲染结果"""
    success: bool
    display_image_base64: Optional[str] = None
    display_width: Optional[int] = None
    display_height: Optional[int] = None
    model_image_base64: Optional[str] = None
    model_width: Optional[int] = None
    model_height: Optional[int] = None
    error_message: Optional[str] = None
    validation_status: Optional[Dict] = None
    fallback_applied: Optional[str] = None


@dataclass
class ValidationResult:
    """质量门校验结果"""
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, any] = field(default_factory=dict)
    error_message: Optional[str] = None


# 导出所有类型
__all__ = [
    "FigureSource",
    "BBoxCoordinateSpace",
    "LogicalFigureSchema",
    "FigureImageOutput",
    "OverviewFigureItem",
    "FigureBlock",
    "RenderResult",
    "ValidationResult",
]
