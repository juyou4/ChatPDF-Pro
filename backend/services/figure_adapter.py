"""
Figure Adapter - 输入源适配层

负责将不同输入源（MinerU、PDF 原生）的结果转换为标准的 LogicalFigureSchema。

核心职责：
1. 识别图块
2. 提取或绑定 caption
3. 输出标准化的 FigureBlock 或 LogicalFigureSchema
4. 坐标转换：归一化坐标 -> page_points
"""
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

from schemas.figure_schema import (
    FigureSource,
    BBoxCoordinateSpace,
    LogicalFigureSchema,
    FigureBlock,
)

logger = logging.getLogger(__name__)


class FigureAdapter(ABC):
    """Figure 适配器基类"""

    def __init__(self):
        self._coordinate_space: BBoxCoordinateSpace = BBoxCoordinateSpace.UNKNOWN

    @abstractmethod
    def parse(
        self,
        pdf_path: str,
        ocr_result: dict,
        page_width: float = 612,
        page_height: float = 792
    ) -> List[FigureBlock]:
        """解析输入源，返回标准化 FigureBlock 列表
        
        Args:
            pdf_path: PDF 文件路径
            ocr_result: OCR/布局分析结果
            page_width: 页面宽度 (points)
            page_height: 页面高度 (points)
            
        Returns:
            List[FigureBlock]: 标准化后的 Figure 块列表
        """
        pass

    @abstractmethod
    def get_coordinate_space(self) -> BBoxCoordinateSpace:
        """声明坐标系类型"""
        return self._coordinate_space

    def _normalize_bbox(
        self,
        bbox: Optional[List[float]],
        page_width: float,
        page_height: float,
        source_space: BBoxCoordinateSpace
    ) -> Optional[List[float]]:
        """将任意坐标系的 bbox 转换为 page_points
        
        Args:
            bbox: 原始 bbox [x0, y0, x1, y1]
            page_width: 页面宽度
            page_height: 页面高度
            source_space: 原始坐标空间
            
        Returns:
            转换后的 page_points bbox 或 None
        """
        if not bbox or len(bbox) != 4:
            return None

        x0, y0, x1, y1 = bbox

        if source_space == BBoxCoordinateSpace.PAGE_POINTS:
            return [x0, y0, x1, y1]

        elif source_space == BBoxCoordinateSpace.NORMALIZED_0_1000:
            # 归一化坐标 (0-1000) -> page_points
            return [
                x0 / 1000 * page_width,
                y0 / 1000 * page_height,
                x1 / 1000 * page_width,
                y1 / 1000 * page_height
            ]

        elif source_space == BBoxCoordinateSpace.RENDER_PIXELS:
            # 渲染像素坐标 - 需要 DPI 信息，这里暂时不处理
            logger.warning("Render pixels coordinate space requires DPI info")
            return None

        return None


class MineruFigureAdapter(FigureAdapter):
    """MinerU 布局分析结果适配器
    
    MinerU 返回的 JSON 包含:
    - figures: 图表信息列表
    - images: 图片信息列表
    - layout: 布局块信息
    """

    def __init__(self):
        super().__init__()
        self._coordinate_space = BBoxCoordinateSpace.PAGE_POINTS

    def get_coordinate_space(self) -> BBoxCoordinateSpace:
        return BBoxCoordinateSpace.PAGE_POINTS

    def parse(
        self,
        pdf_path: str,
        ocr_result: dict,
        page_width: float = 612,
        page_height: float = 792
    ) -> List[FigureBlock]:
        """解析 MinerU 结果
        
        ocr_result 期望包含:
        - figures: 图表列表
        - images: 图片列表
        - page_width, page_height: 页面尺寸
        """
        figure_blocks = []
        
        # 获取 figures 列表
        figures = ocr_result.get("figures", [])
        if not figures:
            logger.info("MineruFigureAdapter: No figures found in ocr_result")
            return figure_blocks

        for idx, fig in enumerate(figures):
            page_idx = fig.get("page", 0)
            if isinstance(page_idx, str):
                try:
                    page_idx = int(page_idx)
                except (ValueError, TypeError):
                    page_idx = 0

            # 提取 caption
            caption_text = fig.get("caption", "") or fig.get("text", "") or ""
            
            # 提取 figure index/label
            figure_index = fig.get("label") or fig.get("figure_id") or None
            if not figure_index:
                # 尝试从 caption 提取
                figure_index = self._extract_figure_index(caption_text)

            # 获取 bbox
            raw_bbox = fig.get("bbox") or fig.get("figure_bbox")
            if not raw_bbox:
                # 尝试从 images 构建
                image_ids = fig.get("image_ids", [])
                images = ocr_result.get("images", [])
                raw_bboxes = [
                    img.get("bbox")
                    for img in images
                    if img.get("id") in image_ids and img.get("bbox")
                ]
                if raw_bboxes:
                    raw_bbox = self._merge_bboxes(raw_bboxes)

            # 坐标转换
            body_bbox = self._normalize_bbox(
                raw_bbox, page_width, page_height,
                self.get_coordinate_space()
            )

            # 构建 FigureBlock
            block = FigureBlock(
                figure_id=fig.get("figure_id", f"mineru_fig_{page_idx}_{idx}"),
                page_idx=page_idx,
                figure_index=figure_index,
                caption_text=caption_text,
                raw_bboxes=[raw_bbox] if raw_bbox else [],
                body_bbox_page_pts=body_bbox,
                source=FigureSource.MINERU,
                confidence=fig.get("confidence", 0.8),
                source_metadata={
                    "original_figure": fig,
                    "adapter": "mineru"
                }
            )
            figure_blocks.append(block)

        logger.info(f"MineruFigureAdapter: Parsed {len(figure_blocks)} figures")
        return figure_blocks

    def _extract_figure_index(self, caption: str) -> Optional[str]:
        """从 caption 提取 figure 编号"""
        if not caption:
            return None
        
        # 匹配 Figure 1, Fig. 1, 图 1 等模式
        patterns = [
            r'^图\s*(\d+[a-zA-Z]?)',
            r'^Figure\s+(\d+[a-zA-Z]?)',
            r'^Fig\.?\s+(\d+[a-zA-Z]?)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None

    def _merge_bboxes(self, bboxes: List[List[float]]) -> Optional[List[float]]:
        """合并多个 bbox 为一个外接矩形"""
        if not bboxes:
            return None
        
        valid_bboxes = [b for b in bboxes if b and len(b) == 4]
        if not valid_bboxes:
            return None
        
        x0 = min(b[0] for b in valid_bboxes)
        y0 = min(b[1] for b in valid_bboxes)
        x1 = max(b[2] for b in valid_bboxes)
        y1 = max(b[3] for b in valid_bboxes)
        
        return [x0, y0, x1, y1]


class PDFFigureAdapter(FigureAdapter):
    """PDF 原生结构适配器
    
    复用现有的 _match_figures_with_images 逻辑，
    将其结果转换为标准的 FigureBlock。
    """

    def __init__(self):
        super().__init__()
        self._coordinate_space = BBoxCoordinateSpace.PAGE_POINTS

    def get_coordinate_space(self) -> BBoxCoordinateSpace:
        return BBoxCoordinateSpace.PAGE_POINTS

    def parse(
        self,
        pdf_path: str,
        ocr_result: dict,
        page_width: float = 612,
        page_height: float = 792
    ) -> List[FigureBlock]:
        """解析 PDF 原生结构
        
        期望 ocr_result 包含:
        - figures: figure 标题信息（上传阶段已含 group_bbox / image_ids）
        - images: 图片信息列表
        
        优先使用上传阶段预计算的 group_bbox（已合并全部子图），
        仅当 figures 缺少 group_bbox 时才回退到简化匹配。
        """
        figure_blocks = []
        
        figures = ocr_result.get("figures", [])
        images = ocr_result.get("images", [])
        
        if not figures:
            logger.info("PDFFigureAdapter: No figures found in ocr_result")
            return figure_blocks

        # 检查 figures 是否已有上传阶段预计算的 group_bbox
        has_precomputed = any(fig.get("group_bbox") for fig in figures)
        
        if has_precomputed:
            # 直接使用上传阶段的匹配结果（更准确，已分组合并子图）
            matched_figures = figures
            logger.info(f"PDFFigureAdapter: Using precomputed group_bbox from upload stage")
        else:
            # 回退到简化匹配
            matched_figures = self._match_figures_with_images(figures, images)
        
        for idx, matched in enumerate(matched_figures):
            page_idx = matched.get("page", 1) - 1  # 转为 0-indexed
            
            # 获取 group bbox（优先 group_bbox，回退到 caption_bbox）
            group_bbox = matched.get("group_bbox") or matched.get("figure_bbox")
            
            # 获取 caption
            caption_text = matched.get("caption", "")
            
            # 获取 figure index
            figure_index = matched.get("label", "")
            
            # 构建 FigureBlock
            block = FigureBlock(
                figure_id=matched.get("figure_id", f"pdf_native_fig_{page_idx}_{idx}"),
                page_idx=page_idx,
                figure_index=figure_index,
                caption_text=caption_text,
                raw_bboxes=[group_bbox] if group_bbox else [],
                body_bbox_page_pts=group_bbox,
                source=FigureSource.PDF_NATIVE,
                confidence=0.7,
                source_metadata={
                    "original_figure": matched,
                    "adapter": "pdf_native",
                    "image_ids": matched.get("image_ids", [])
                }
            )
            figure_blocks.append(block)

        logger.info(f"PDFFigureAdapter: Parsed {len(figure_blocks)} figures")
        return figure_blocks

    def _match_figures_with_images(self, figures: list, images: list) -> list:
        """将 figure 标题与同页的图片进行空间匹配
        
        这是从 document_routes.py 提取的核心逻辑的简化版本。
        """
        if not figures:
            return []

        # 按页分组图片
        page_to_images: Dict[int, List[Dict]] = {}
        for img in images:
            p = img.get("page", 1)
            if p not in page_to_images:
                page_to_images[p] = []
            page_to_images[p].append(img)

        # 按 Y 坐标排序图片
        for page_num in page_to_images:
            page_to_images[page_num] = sorted(
                page_to_images[page_num],
                key=lambda img: (
                    (img.get("bbox") or [0, 0, 0, 0])[1],
                    (img.get("bbox") or [0, 0, 0, 0])[0],
                )
            )

        result = []
        
        for fig in figures:
            page = fig.get("page", 1)
            page_images = page_to_images.get(page, [])
            
            # 获取 figure bbox
            fig_bbox = fig.get("caption_bbox") or fig.get("bbox")
            if not fig_bbox:
                continue
            
            # 匹配 caption 上方或下方的图片
            caption_top = fig_bbox[1] if len(fig_bbox) >= 4 else 0
            caption_bottom = fig_bbox[3] if len(fig_bbox) >= 4 else 0
            
            matched_images = []
            for img in page_images:
                img_bbox = img.get("bbox")
                if not img_bbox or len(img_bbox) < 4:
                    continue
                
                img_top = img_bbox[1]
                img_bottom = img_bbox[3]
                # 图片在 caption 下方一定范围内（caption 在图上方）
                below_caption = caption_bottom - 20 <= img_top <= caption_bottom + 250
                # 图片在 caption 上方一定范围内（caption 在图下方）
                above_caption = caption_top - 250 <= img_bottom <= caption_top + 20
                if below_caption or above_caption:
                    matched_images.append(img)
            
            # 合并匹配图片的 bbox + caption bbox 得到完整 figure 区域
            image_bbox = self._merge_image_bboxes(matched_images)
            all_bboxes = [b for b in [image_bbox, fig_bbox] if b and len(b) == 4]
            group_bbox = self._merge_bboxes(all_bboxes) if all_bboxes else None
            
            result.append({
                "figure_id": fig.get("figure_id", f"fig_{page}_{len(result)}"),
                "page": page,
                "label": fig.get("label", ""),
                "caption": fig.get("caption", ""),
                "group_bbox": group_bbox,
                "image_ids": [img.get("id") for img in matched_images]
            })
        
        return result

    def _merge_image_bboxes(self, images: List[Dict]) -> Optional[List[float]]:
        """合并图片列表的 bbox"""
        if not images:
            return None
        
        bboxes = [img.get("bbox") for img in images if img.get("bbox")]
        if not bboxes:
            return None
        
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        return [x0, y0, x1, y1]


class FallbackFigureAdapter(FigureAdapter):
    """兜底适配器 - 直接使用图片列表
    
    当其他适配器无法工作时，使用此适配器。
    将每张图片作为一个独立的 Figure。
    """

    def __init__(self):
        super().__init__()
        self._coordinate_space = BBoxCoordinateSpace.PAGE_POINTS

    def get_coordinate_space(self) -> BBoxCoordinateSpace:
        return BBoxCoordinateSpace.PAGE_POINTS

    def parse(
        self,
        pdf_path: str,
        ocr_result: dict,
        page_width: float = 612,
        page_height: float = 792
    ) -> List[FigureBlock]:
        """将每张图片作为一个 Figure"""
        figure_blocks = []
        
        images = ocr_result.get("images", [])
        
        for idx, img in enumerate(images):
            page_idx = img.get("page", 1) - 1  # 转为 0-indexed
            
            bbox = img.get("bbox")
            if not bbox:
                continue
            
            block = FigureBlock(
                figure_id=img.get("id", f"fallback_img_{page_idx}_{idx}"),
                page_idx=page_idx,
                figure_index=None,
                caption_text="",
                raw_bboxes=[bbox],
                body_bbox_page_pts=bbox,
                source=FigureSource.FALLBACK,
                confidence=0.3,
                source_metadata={
                    "original_image": img,
                    "adapter": "fallback"
                }
            )
            figure_blocks.append(block)

        logger.info(f"FallbackFigureAdapter: Parsed {len(figure_blocks)} figures")
        return figure_blocks


# 适配器工厂
class FigureAdapterFactory:
    """Figure 适配器工厂"""

    _adapters = {
        FigureSource.MINERU: MineruFigureAdapter,
        FigureSource.PDF_NATIVE: PDFFigureAdapter,
        FigureSource.FALLBACK: FallbackFigureAdapter,
    }

    @classmethod
    def get_adapter(cls, source: FigureSource) -> FigureAdapter:
        """获取指定类型的适配器"""
        adapter_class = cls._adapters.get(source)
        if not adapter_class:
            logger.warning(f"Unknown adapter source: {source}, using FallbackFigureAdapter")
            return FallbackFigureAdapter()
        return adapter_class()

    @classmethod
    def get_all_adapters(cls) -> List[FigureAdapter]:
        """获取所有可用适配器"""
        return [adapter_class() for adapter_class in cls._adapters.values()]


__all__ = [
    "FigureAdapter",
    "MineruFigureAdapter",
    "PDFFigureAdapter",
    "FallbackFigureAdapter",
    "FigureAdapterFactory",
]
