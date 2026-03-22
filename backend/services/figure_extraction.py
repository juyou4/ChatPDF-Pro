"""
Figure Extraction Service - 统一入口

为速览构建标准化的 LogicalFigure 列表，支持：
- 缓存复用
- MinerU 优先
- PDF-native fallback

本模块是 figure 提取的唯一入口，overview_service 只消费这里的结果。
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

import fitz

from schemas.figure_schema import LogicalFigureSchema, FigureSource
from services.figure_adapter import FigureAdapterFactory
from services.figure_builder import build_logical_figures, select_top_figures

logger = logging.getLogger(__name__)

# Schema 版本，用于缓存失效判断
SCHEMA_VERSION = "1.0"


def build_logical_figures_for_overview(
    doc_id: str,
    doc_record: dict,
    depth: str = "standard",
    force_rebuild: bool = False,
) -> List[LogicalFigureSchema]:
    """
    为速览构建标准化 Figure 列表

    职责：
    1. 缓存命中判断（检查 logical_figures_status）
    2. provider 选择（MinerU / PDF-native）
    3. 调用 provider 获取 figure 数据
    4. 适配为 LogicalFigureSchema
    5. 质量校验，不合格则 fallback
    6. 结果写回 doc["data"]["logical_figures"] + meta + status

    参数：
    - doc_id: 文档 ID
    - doc_record: 文档记录，包含 data 字段
    - depth: 速览深度 brief/standard/detailed
    - force_rebuild: 强制重新构建，忽略缓存

    返回：
    - List[LogicalFigureSchema]: 标准化的 figure 列表
    """
    doc_data = doc_record.get("data", {})

    # ========== 1. 缓存命中判断 ==========
    if not force_rebuild:
        status = doc_data.get("logical_figures_status", {})
        meta = doc_data.get("logical_figures_meta", {})

        if status.get("state") == "done":
            # 检查 schema 版本
            if meta.get("schema_version") == SCHEMA_VERSION:
                cached = doc_data.get("logical_figures", [])
                if cached:
                    logger.info(
                        f"[FigureExtraction] Cache hit for doc {doc_id}: "
                        f"{len(cached)} figures"
                    )
                    return [LogicalFigureSchema(**fig) for fig in cached]

    # ========== 2. 获取文档信息 ==========
    pdf_url = doc_record.get("pdf_url")
    images = doc_data.get("images", [])
    figures = doc_data.get("figures", [])  # 上传阶段提取的 figure 标题

    # 如果没有图片，直接返回空
    if not images:
        logger.info(f"[FigureExtraction] No images found for doc {doc_id}")
        _update_status(doc_data, "done", provider="none", count=0)
        return []

    # ========== 3. 打开 PDF 获取尺寸 ==========
    pdf_doc = None
    page_width = 612.0
    page_height = 792.0

    if pdf_url:
        try:
            from routes.document_routes import UPLOAD_DIR as _upload_dir
            pdf_path = str(_upload_dir / pdf_url.split("/")[-1])
            pdf_doc = fitz.open(pdf_path)
            if pdf_doc.page_count > 0:
                first_page = pdf_doc[0]
                page_width = first_page.rect.width
                page_height = first_page.rect.height
        except Exception as e:
            logger.warning(f"[FigureExtraction] Failed to open PDF: {e}")

    # ========== 4. 尝试使用 Adapter 构建 ==========
    source = "pdf_native"  # 默认来源
    fallback_used = False
    adapter_results = []

    # 准备 Adapter 输入
    adapter_input = {
        "figures": figures,
        "images": images,
    }

    # 4.1 尝试 MinerU Adapter（如果上传阶段有 MinerU 结果）
    # 注意：当前上传阶段不会自动调用 MinerU 获取 figure 数据
    # 这里检查 ocr_result 中是否有 MinerU 格式的数据
    mineru_figures = doc_data.get("ocr_result", {}).get("figures", [])
    if mineru_figures:
        try:
            mineru_adapter = FigureAdapterFactory.get_adapter(FigureSource.MINERU)
            mineru_blocks = mineru_adapter.parse(
                "",
                {"figures": mineru_figures, "images": images},
                page_width,
                page_height
            )
            if mineru_blocks:
                adapter_results.extend(mineru_blocks)
                source = "mineru"
                logger.info(f"[FigureExtraction] MinerU Adapter: got {len(mineru_blocks)} blocks")
        except Exception as e:
            logger.warning(f"[FigureExtraction] MinerU Adapter failed: {e}")
            fallback_used = True

    # 4.2 如果没有 MinerU 结果，使用 PDF Native Adapter
    if not adapter_results and figures:
        try:
            pdf_adapter = FigureAdapterFactory.get_adapter(FigureSource.PDF_NATIVE)
            pdf_blocks = pdf_adapter.parse(
                "",
                adapter_input,
                page_width,
                page_height
            )
            if pdf_blocks:
                adapter_results.extend(pdf_blocks)
                source = "pdf_native"
                logger.info(f"[FigureExtraction] PDF Native Adapter: got {len(pdf_blocks)} blocks")
        except Exception as e:
            logger.warning(f"[FigureExtraction] PDF Native Adapter failed: {e}")
            fallback_used = True

    # 4.3 Fallback: 使用图片列表
    if not adapter_results and images:
        try:
            fallback_adapter = FigureAdapterFactory.get_adapter(FigureSource.FALLBACK)
            fallback_blocks = fallback_adapter.parse(
                "",
                adapter_input,
                page_width,
                page_height
            )
            if fallback_blocks:
                adapter_results.extend(fallback_blocks)
                source = "fallback"
                logger.info(f"[FigureExtraction] Fallback Adapter: got {len(fallback_blocks)} blocks")
        except Exception as e:
            logger.warning(f"[FigureExtraction] Fallback Adapter failed: {e}")

    # ========== 5. 构建 Logical Figures ==========
    if not adapter_results:
        logger.info(f"[FigureExtraction] No figure blocks for doc {doc_id}")
        _update_status(doc_data, "done", provider=source, count=0)
        return []

    # 使用 figure_builder 构建
    logical_figures = build_logical_figures(
        adapter_results,
        page_width,
        page_height
    )

    # 选取 top N
    selected = select_top_figures(logical_figures, depth)

    # ========== 6. 质量校验（简单版）==========
    # 检查是否有 caption
    valid_count = sum(1 for fig in selected if fig.caption_text)
    if valid_count == 0 and selected:
        logger.info("[FigureExtraction] No captions found, using raw selection")
        # 保留结果，但不认为是高质量

    # ========== 7. 写回缓存 ==========
    # 转换 为 dict 列表用于存储
    figures_dict = [fig.model_dump() for fig in selected]

    doc_data["logical_figures"] = figures_dict
    doc_data["logical_figures_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fallback_used": fallback_used,
        "count": len(selected),
    }
    doc_data["logical_figures_status"] = {
        "state": "done",
        "error": None,
        "provider": source,
    }

    # 同步更新到 documents_store（如果可用）
    try:
        from routes.document_routes import documents_store
        if doc_id in documents_store:
            documents_store[doc_id]["data"] = doc_data
    except Exception as e:
        logger.warning(f"[FigureExtraction] Failed to update documents_store: {e}")

    logger.info(
        f"[FigureExtraction] Built {len(selected)} figures for doc {doc_id}, "
        f"source={source}, fallback={fallback_used}"
    )

    if pdf_doc:
        pdf_doc.close()

    return selected


def _update_status(
    doc_data: dict,
    state: str,
    provider: str,
    count: int,
    error: Optional[str] = None,
) -> None:
    """更新 figure 状态"""
    doc_data["logical_figures_status"] = {
        "state": state,
        "error": error,
        "provider": provider,
    }
    doc_data["logical_figures_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": provider,
        "fallback_used": False,
        "count": count,
    }
    doc_data["logical_figures"] = []


def get_figure_extraction_status(doc_data: dict) -> dict:
    """
    获取 figure 提取状态，供 overview_service 使用

    返回：
    - state: "done" | "failed" | "running" | "none"
    - count: figure 数量
    - source: 来源
    - has_cache: 是否有可用缓存
    """
    status = doc_data.get("logical_figures_status", {})
    meta = doc_data.get("logical_figures_meta", {})

    state = status.get("state", "none")
    count = meta.get("count", 0)
    source = status.get("provider", "none")

    # 判断是否有可用缓存
    has_cache = (
        state == "done"
        and meta.get("schema_version") == SCHEMA_VERSION
        and count > 0
    )

    return {
        "state": state,
        "count": count,
        "source": source,
        "has_cache": has_cache,
        "error": status.get("error"),
    }
