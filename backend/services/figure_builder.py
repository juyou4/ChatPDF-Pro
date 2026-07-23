"""
Figure Builder - Figure 构建与合并逻辑

负责：
1. 一级绑定接收 - 接收 Adapter 输出的 FigureBlock
2. 二级 merge - 检查合并条件，合并属于同一 Figure 的块
3. body/full bbox 构建
4. confidence 计算

核心原则：
- 一个 Logical Figure = 一个展示项
- 合并条件：同页 + caption 相似 + 空间相邻 + 无正文隔断
"""
import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

from schemas.figure_schema import (
    FigureSource,
    LogicalFigureSchema,
    FigureBlock,
)

logger = logging.getLogger(__name__)


# 合并配置
MERGE_CONFIG = {
    # 同一页内的块才可能合并
    "same_page_only": True,
    
    # Caption 相似度阈值 (0-1)
    "caption_similarity_threshold": 0.7,
    
    # 空间相邻阈值 (points)
    "spatial_distance_threshold": 50,
    
    # 中间正文隔断检测阈值
    "text_block_threshold": 30,
    
    # 合并后宽高比合理性范围
    "min_aspect_ratio": 0.1,
    "max_aspect_ratio": 10.0,
    
    # 合并后面积上限 (相对于页面)
    "max_area_ratio": 0.8,
}


def normalize_caption(caption: str) -> str:
    """归一化 caption 用于比较
    
    移除:
    - 数字编号 (Figure 1 -> Figure)
    - 空格
    - 大小写
    """
    if not caption:
        return ""
    
    # 移除前缀编号
    patterns = [
        r'^(图|Figure|Fig\.?)\s*\d+[a-zA-Z]?\s*[:：]?\s*',
    ]
    
    normalized = caption
    for pattern in patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # 移除空白字符并转小写
    normalized = re.sub(r'\s+', '', normalized).lower()
    
    return normalized


def caption_similarity(caption1: str, caption2: str) -> float:
    """计算两个 caption 的相似度
    
    Returns:
        0-1 之间的相似度分数
    """
    if not caption1 or not caption2:
        return 0.0
    
    norm1 = normalize_caption(caption1)
    norm2 = normalize_caption(caption2)
    
    if norm1 == norm2:
        return 1.0
    
    # 包含关系
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    
    # SequenceMatcher 模糊匹配
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio


def spatial_distance(bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> float:
    """计算两个 bbox 之间的边缘最小距离
    
    使用边缘到边缘的最小距离（而非中心点距离），
    对相邻但尺寸较大的子图更合理。重叠时返回 0。
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return float('inf')
    
    # 水平方向间距：如果不重叠则为正值
    dx = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
    # 垂直方向间距
    dy = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
    
    return (dx ** 2 + dy ** 2) ** 0.5


def merge_bboxes(bboxes: List[List[float]]) -> List[float]:
    """合并多个 bbox 为一个外接矩形
    
    Args:
        bboxes: bbox 列表 [x0, y0, x1, y1]
        
    Returns:
        合并后的 bbox
    """
    if not bboxes:
        return [0, 0, 0, 0]
    
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return [0, 0, 0, 0]
    
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    
    return [x0, y0, x1, y1]


def compute_aspect_ratio(bbox: List[float]) -> float:
    """计算 bbox 的宽高比"""
    if len(bbox) != 4:
        return 0.0
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if height <= 0:
        return 0.0
    
    return width / height


def compute_area(bbox: List[float]) -> float:
    """计算 bbox 的面积"""
    if len(bbox) != 4:
        return 0.0
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    return width * height


def should_merge(block1: FigureBlock, block2: FigureBlock, page_width: float, page_height: float) -> bool:
    """判断两个 FigureBlock 是否应该合并
    
    合并条件：
    1. 同页
    2. Caption 相同或相似
    3. 空间相邻
    4. 合并后 bbox 合理
    """
    cfg = MERGE_CONFIG
    
    # 条件1: 同页
    if block1.page_idx != block2.page_idx:
        return False
    
    # 条件2: Caption 相似
    if block1.caption_text or block2.caption_text:
        sim = caption_similarity(block1.caption_text, block2.caption_text)
        if sim < cfg["caption_similarity_threshold"]:
            # 但如果是子图关系 (1a, 1b)，也允许合并
            if not _is_subfigure_relationship(block1.figure_index, block2.figure_index):
                return False
    
    # 条件3: 空间相邻
    bbox1 = block1.body_bbox_page_pts
    bbox2 = block2.body_bbox_page_pts
    
    if not bbox1 or not bbox2:
        return False
    
    dist = spatial_distance(bbox1, bbox2)
    # 子图关系允许更大的空间距离（同一 Figure 的子图可能间距较大）
    is_subfig = _is_subfigure_relationship(block1.figure_index, block2.figure_index)
    threshold = cfg["spatial_distance_threshold"] * 3 if is_subfig else cfg["spatial_distance_threshold"]
    if dist > threshold:
        return False
    
    # 条件4: 合并后 bbox 合理性
    merged_bbox = merge_bboxes([bbox1, bbox2])
    aspect_ratio = compute_aspect_ratio(merged_bbox)
    area = compute_area(merged_bbox)
    page_area = page_width * page_height
    
    if aspect_ratio < cfg["min_aspect_ratio"] or aspect_ratio > cfg["max_aspect_ratio"]:
        return False
    
    if area / page_area > cfg["max_area_ratio"]:
        return False
    
    return True


def _is_subfigure_relationship(idx1: Optional[str], idx2: Optional[str]) -> bool:
    """判断两个 figure index 是否为子图关系
    
    例如: Figure 1a 和 Figure 1b, 图1(a) 和 图1(b)
    """
    if not idx1 or not idx2:
        return False
    
    # 提取基础编号（支持多种格式）
    patterns = [
        r'^(图|Figure|Fig\.?)\s*(\d+)\s*[a-zA-Z(（]?',
        r'^(\d+)\s*[a-zA-Z(（]',
    ]
    
    base1 = None
    base2 = None
    
    for pattern in patterns:
        if not base1:
            m1 = re.match(pattern, idx1, re.IGNORECASE)
            if m1:
                base1 = m1.group(2) if m1.lastindex >= 2 else m1.group(1)
        if not base2:
            m2 = re.match(pattern, idx2, re.IGNORECASE)
            if m2:
                base2 = m2.group(2) if m2.lastindex >= 2 else m2.group(1)
    
    if base1 and base2 and base1 == base2:
        return True
    
    return False


def group_by_caption(figure_blocks: List[FigureBlock]) -> Dict[str, List[FigureBlock]]:
    """按 caption 归一化分组（一级分组）
    
    Returns:
        Dict[caption_key -> List[FigureBlock]]
    """
    groups: Dict[str, List[FigureBlock]] = {}
    
    for block in figure_blocks:
        # 使用归一化 caption 作为 key
        key = normalize_caption(block.caption_text)
        
        # 如果没有 caption，使用 figure_index
        if not key and block.figure_index:
            key = normalize_caption(block.figure_index)
        
        # 如果还是没有，使用唯一 ID
        if not key:
            key = f"no_caption_{block.figure_id}"
        
        if key not in groups:
            groups[key] = []
        groups[key].append(block)
    
    return groups


def merge_blocks(blocks: List[FigureBlock], page_width: float, page_height: float) -> List[FigureBlock]:
    """合并符合条件的 blocks
    
    使用 Union-Find 算法进行合并
    """
    if len(blocks) <= 1:
        return blocks
    
    # Union-Find
    parent = list(range(len(blocks)))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 检查每对 block 是否应该合并
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if should_merge(blocks[i], blocks[j], page_width, page_height):
                union(i, j)
    
    # 按 parent 分组
    groups: Dict[int, List[FigureBlock]] = {}
    for i in range(len(blocks)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(blocks[i])
    
    # 合并每个 group
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_merge_group(group))
    
    return merged


def _merge_group(blocks: List[FigureBlock]) -> FigureBlock:
    """合并一组 blocks 为一个"""
    if len(blocks) == 1:
        return blocks[0]
    
    # 使用第一个 block 作为基础
    merged = FigureBlock(
        figure_id=blocks[0].figure_id,
        page_idx=blocks[0].page_idx,
        figure_index=blocks[0].figure_index,
        caption_text=blocks[0].caption_text,
        raw_bboxes=[],
        panel_bboxes_page_pts=[],
        source=blocks[0].source,
        confidence=0.0,
        source_metadata={}
    )
    
    # 收集所有 bbox
    all_bboxes = []
    for block in blocks:
        if block.body_bbox_page_pts:
            all_bboxes.append(block.body_bbox_page_pts)
        merged.raw_bboxes.extend(block.raw_bboxes)
        merged.panel_bboxes_page_pts.extend(block.panel_bboxes_page_pts)
    
    # 合并所有 bbox
    merged.body_bbox_page_pts = merge_bboxes(all_bboxes)
    
    # 计算置信度（取最高）
    merged.confidence = max(b.confidence for b in blocks)
    
    # 合并 source_metadata
    merged.source_metadata = {
        "merged_from": [b.figure_id for b in blocks],
        "merge_count": len(blocks),
    }
    
    return merged


def build_body_bbox(block: FigureBlock) -> Optional[List[float]]:
    """构建 body bbox - 只包含图主体
    
    策略：
    - 如果有多个子图 panel_bboxes，取 union
    - 否则使用 body_bbox_page_pts
    """
    if block.panel_bboxes_page_pts:
        bbox = merge_bboxes(block.panel_bboxes_page_pts)
        try:
            bottom_padding = max(
                0.0,
                float(block.source_metadata.get("panel_label_bottom_padding") or 0.0),
            )
        except (TypeError, ValueError):
            bottom_padding = 0.0
        if bottom_padding:
            bbox[3] += bottom_padding
        return bbox
    
    return block.body_bbox_page_pts


def build_full_bbox(block: FigureBlock, caption_bbox: Optional[List[float]] = None) -> Optional[List[float]]:
    """构建 full bbox - 图主体 + caption
    
    策略：
    - body_bbox + caption 区域的 union
    """
    body = build_body_bbox(block)
    if not body:
        return None
    
    if caption_bbox:
        return merge_bboxes([body, caption_bbox])
    
    # 如果没有 caption bbox，尝试从 caption 文本位置推断
    # 这里简化处理，直接返回 body
    return body


def compute_confidence(block: FigureBlock) -> float:
    """计算 FigureBlock 的置信度
    
    考虑因素：
    - 来源可靠性 (mineru > pdf_native > fallback)
    - 是否有 caption
    - bbox 完整性
    """
    # 来源基础分数
    source_scores = {
        FigureSource.MINERU: 0.9,
        FigureSource.YOLO: 0.82,
        "yolo": 0.82,
        FigureSource.PDF_NATIVE: 0.7,
        FigureSource.FALLBACK: 0.3,
    }
    
    base_score = source_scores.get(block.source, 0.5)
    
    # 有 caption 加分
    if block.caption_text:
        base_score += 0.05
    
    # 有 figure_index 加分
    if block.figure_index:
        base_score += 0.05
    
    # bbox 完整加分
    if block.body_bbox_page_pts:
        base_score += 0.1
    
    # 合并来源加分
    if block.source_metadata.get("merge_count", 1) > 1:
        base_score += 0.1
    
    return min(base_score, 1.0)


def build_logical_figures(
    figure_blocks: List[FigureBlock],
    page_width: float = 612,
    page_height: float = 792
) -> List[LogicalFigureSchema]:
    """构建逻辑 Figure 列表
    
    主流程：
    1. 按 caption 归一化分组（一级）
    2. 检查合并条件，二级合并
    3. 构建 body / full bbox
    4. 计算置信度
    5. 转换为 Schema
    """
    if not figure_blocks:
        return []
    
    # 1. 按 caption 归一化分组
    caption_groups = group_by_caption(figure_blocks)
    
    # 2. 每组内检查合并条件
    merged_blocks: List[FigureBlock] = []
    for caption_key, blocks in caption_groups.items():
        merged = merge_blocks(blocks, page_width, page_height)
        merged_blocks.extend(merged)
    
    # 3. 构建 body / full bbox 和计算置信度
    for block in merged_blocks:
        block.body_bbox_page_pts = build_body_bbox(block)
        block.full_bbox_page_pts = build_full_bbox(block)
        block.confidence = compute_confidence(block)
    
    # 4. 转换为 Schema
    result = []
    for block in merged_blocks:
        schema = LogicalFigureSchema(
            figure_id=block.figure_id,
            page_idx=block.page_idx,
            figure_index=block.figure_index,
            caption_text=block.caption_text,
            body_bbox_page_pts=block.body_bbox_page_pts,
            full_bbox_page_pts=block.full_bbox_page_pts,
            panel_bboxes_page_pts=block.panel_bboxes_page_pts,
            source=block.source,
            confidence=block.confidence,
            source_metadata=block.source_metadata,
        )
        result.append(schema)
    
    # 5. 按页码和位置排序
    result.sort(key=lambda x: (x.page_idx, x.body_bbox_page_pts[1] if x.body_bbox_page_pts else 0))
    
    logger.info(f"Built {len(result)} logical figures from {len(figure_blocks)} blocks")
    return result


def select_top_figures(
    figures: List[LogicalFigureSchema],
    depth: str = "standard",
    max_count: Optional[int] = None
) -> List[LogicalFigureSchema]:
    """选取 top N figures 用于 overview
    
    策略：
    1. 按置信度排序
    2. 优先选择有 caption 的
    3. 跨页分布均匀
    """
    # 深度配置
    depth_config = {
        "brief": 2,
        "standard": 3,
        "detailed": 5,
    }
    
    if max_count is None:
        max_count = depth_config.get(depth, 3)
    
    if len(figures) <= max_count:
        return figures
    
    # 按文档顺序排序（页码 + y 坐标），与学术论文阅读顺序一致
    sorted_figures = sorted(
        figures,
        key=lambda x: (
            x.page_idx,
            x.body_bbox_page_pts[1] if x.body_bbox_page_pts else 0
        )
    )
    
    # 按文档顺序依次选择，每页最多 2 个
    selected: List[LogicalFigureSchema] = []
    page_selected: Dict[int, int] = {}
    
    for fig in sorted_figures:
        if len(selected) >= max_count:
            break
        
        page = fig.page_idx
        count = page_selected.get(page, 0)
        
        if count < 2:
            selected.append(fig)
            page_selected[page] = count + 1
    
    return selected


__all__ = [
    "build_logical_figures",
    "select_top_figures",
    "group_by_caption",
    "merge_blocks",
    "build_body_bbox",
    "build_full_bbox",
    "compute_confidence",
]
