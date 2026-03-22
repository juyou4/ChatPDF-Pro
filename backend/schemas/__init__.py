"""
Schemas 模块 - 数据结构定义

导出所有 Schema 类供其他模块使用。
"""
from .figure_schema import (
    FigureSource,
    BBoxCoordinateSpace,
    LogicalFigureSchema,
    FigureImageOutput,
    OverviewFigureItem,
    FigureBlock,
    RenderResult,
    ValidationResult,
)

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
