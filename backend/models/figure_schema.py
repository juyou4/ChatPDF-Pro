"""
Figure 数据模型定义

用于统一 ChartPDF 图表解读功能中的数据结构，
避免 routes 和 service 各自猜字段的问题。
"""
from typing import Optional, List, Any
from pydantic import BaseModel


class FigureItem(BaseModel):
    """单个 Figure 条目"""
    raw_number: str  # "1a"
    base_number: str  # "1"
    sub_id: Optional[str] = None  # "a", "b" or None
    display_label: str  # "Figure 1a"
    page: int
    caption: Optional[str] = None  # 原始caption文本
    bbox: Optional[List[float]] = None
    image_ids: List[str] = []


class FigureGroup(BaseModel):
    """FigureGroup - 核心数据单元"""
    group_id: str  # "fig-1"
    base_number: str  # "1"
    page: int
    caption: Optional[str] = None  # 主caption，优先用Figure 1的
    sub_figures: List[FigureItem] = []
    image_ids: List[str] = []  # 去重后的image ids
    group_bbox: Optional[List[float]] = None  # 联合bbox
    sort_key: tuple = (0, 0)  # (page, figure_index)
    is_synthetic: bool = False  # 是否为无显式主图时自动生成
    label: str = ""  # 显示标签，如 "Figure 1"


class ImageQualityScore(BaseModel):
    """图像质量评分"""
    size_score: float = 0.0  # 尺寸是否合适
    entropy_score: float = 0.0  # 信息熵
    edge_score: float = 0.0  # 边缘密度
    ocr_density_score: float = 0.0  # OCR文本密度
    layout_score: float = 0.0  # 面板结构评分

    total_score: float = 0.0
    reason: str = ""  # too_small / blank / low_entropy / ...

    def get_level(self) -> str:
        """获取质量等级"""
        if self.total_score >= 0.7:
            return "high"
        elif self.total_score >= 0.4:
            return "medium"
        else:
            return "low"


class CropResult(BaseModel):
    """裁切结果"""
    image_data: str = ""
    fallback_level: int = 0  # 1-5, 0表示未裁切
    quality_score: ImageQualityScore = ImageQualityScore()
    crop_bbox: Optional[List[float]] = None


# 质量评分阈值配置
QUALITY_THRESHOLDS = {
    "high": 0.7,  # 可信，直接使用
    "medium": 0.4,  # 可用但建议fallback比较
    "low": 0.0,  # 强制走下一层fallback
}

# 合并子图最大数量（避免Payload Too Large）
MAX_SUB_IMAGES = 4

# 裁切安全边距（像素）
CROP_PADDING = 15
