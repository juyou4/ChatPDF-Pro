"""
Figure Extraction Service - 统一入口

为速览构建标准化的 LogicalFigure 列表，支持：
- 缓存复用
- MinerU 优先
- PDF-native fallback

本模块是 figure 提取的唯一入口，overview_service 只消费这里的结果。
"""
import logging
import io
import re
from datetime import datetime, timezone
from typing import List, Optional

import fitz

from schemas.figure_schema import LogicalFigureSchema, FigureSource, FigureBlock
from services.document_parse_state import read_parse_manifest
from services.figure_adapter import FigureAdapterFactory
from services.figure_builder import build_logical_figures, select_top_figures

logger = logging.getLogger(__name__)

# Schema 版本，用于缓存失效判断。1.4 起缓存完整 Logical Figure 集，
# 不再把 brief/standard/detailed 的 Top-N 结果作为文档级缓存。
# 1.6：按跨栏 Figure N 总标题合并 MinerU 拆开的 (a)(b)(c) 子图。
SCHEMA_VERSION = "1.6"
MINERU_FIGURE_GROUPING_VERSION = "span-caption-v1"
MINERU_PANEL_LABEL_BOTTOM_PADDING = 18.0
MAX_MINERU_PANEL_GROUP_SIZE = 8


def _cache_matches_parse_identity(meta: dict, manifest: dict) -> bool:
    """A figure cache belongs to one parser generation, not just one doc id."""
    expected_generation = str((manifest or {}).get("generation") or "").strip()
    expected_source_hash = str((manifest or {}).get("source_hash") or "").strip()
    if not expected_generation and not expected_source_hash:
        return True
    return (
        str((meta or {}).get("parse_generation") or "").strip() == expected_generation
        and str((meta or {}).get("document_source_hash") or "").strip() == expected_source_hash
    )


def build_logical_figures_for_overview(
    doc_id: str,
    doc_record: dict,
    depth: str = "standard",
    force_rebuild: bool = False,
    reference: str = "",
) -> List[LogicalFigureSchema]:
    """
    为速览构建标准化 Figure 列表

    职责：
    1. 缓存命中判断（检查 logical_figures_status）
    2. provider 选择（MinerU / PDF-native / 本地视觉兜底）
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
    parse_manifest = read_parse_manifest(doc_record, doc_id=doc_id)
    resolved_route = str(parse_manifest.get("resolved_route") or "").strip().lower()
    is_mineru_route = resolved_route == "mineru"
    parse_metadata = (
        parse_manifest.get("metadata")
        if isinstance(parse_manifest.get("metadata"), dict)
        else {}
    )
    has_explicit_parse_state = bool(
        isinstance(doc_data.get("parse_manifest"), dict)
        or isinstance(doc_data.get("document_parse_state"), dict)
    )
    allow_legacy_mineru = bool(
        parse_metadata.get("legacy_inferred") and not has_explicit_parse_state
    )
    allow_mineru_sources = is_mineru_route or allow_legacy_mineru

    # A new MinerU publication persists this exact logical-figure geometry.
    # Reusing it here keeps overview, Agent attachments and visual retrieval on
    # the same panel grouping instead of independently rebuilding figures.
    if is_mineru_route and not allow_legacy_mineru:
        try:
            from services.mineru_visual_asset_service import (
                active_mineru_visual_asset_envelope,
                logical_figures_from_mineru_visual_assets,
            )

            persisted_assets = active_mineru_visual_asset_envelope(
                doc_data,
                parse_identity=parse_manifest,
            )
            persisted_figures = logical_figures_from_mineru_visual_assets(persisted_assets)
            if (
                persisted_figures
                and str((persisted_assets or {}).get("grouping_version") or "")
                == MINERU_FIGURE_GROUPING_VERSION
            ):
                figures_dict = [figure.model_dump() for figure in persisted_figures]
                doc_data["logical_figures"] = figures_dict
                doc_data["logical_figures_meta"] = {
                    "schema_version": SCHEMA_VERSION,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "source": "mineru_visual_assets",
                    "fallback_used": False,
                    "count": len(figures_dict),
                    "selection_version": "dynamic-depth-v1",
                    "parse_generation": str(parse_manifest.get("generation") or ""),
                    "document_source_hash": str(parse_manifest.get("source_hash") or ""),
                    "parser_route": resolved_route,
                    "visual_asset_revision": str(persisted_assets.get("revision") or ""),
                    "grouping_version": MINERU_FIGURE_GROUPING_VERSION,
                }
                doc_data["logical_figures_status"] = {
                    "state": "done",
                    "error": None,
                    "provider": "mineru_visual_assets",
                }
                if reference:
                    return [
                        figure for figure in persisted_figures
                        if _logical_figure_matches_reference(figure, reference)
                    ][:1]
                return select_top_figures(persisted_figures, depth)
        except Exception as exc:
            logger.warning("[FigureExtraction] MinerU visual asset reuse failed: %s", exc)

    # ========== 1. 缓存命中判断 ==========
    if not force_rebuild:
        status = doc_data.get("logical_figures_status", {})
        meta = doc_data.get("logical_figures_meta", {})

        if status.get("state") == "done":
            # 检查 schema 版本
            cached_source = str(meta.get("source") or "").strip().lower()
            cached_is_mineru = cached_source in {
                "mineru",
                "mineru_deep_parse",
                "mineru_visual_assets",
            }
            cache_matches_route = (
                cached_is_mineru
                if is_mineru_route
                else (not cached_is_mineru or allow_legacy_mineru)
            )
            cache_matches_identity = _cache_matches_parse_identity(meta, parse_manifest)
            if meta.get("schema_version") == SCHEMA_VERSION and cache_matches_route and cache_matches_identity:
                cached = doc_data.get("logical_figures", [])
                if cached:
                    cached_figures = [LogicalFigureSchema(**fig) for fig in cached]
                    if reference:
                        matched = [
                            figure for figure in cached_figures
                            if _logical_figure_matches_reference(figure, reference)
                        ]
                        if matched:
                            return matched[:1]
                    else:
                        logger.info(
                            f"[FigureExtraction] Cache hit for doc {doc_id}: "
                            f"{len(cached)} complete figures"
                        )
                        return select_top_figures(cached_figures, depth)

    # ========== 2. 获取文档信息 ==========
    pdf_url = doc_record.get("pdf_url")
    images = doc_data.get("images", [])
    figures = doc_data.get("figures", [])  # 上传阶段提取的 figure 标题

    # MinerU 全程路线发布后，结构化块索引是图表定位的第一来源。
    # 先读取它再判断本地 images/figures，避免纯 MinerU 文档被提前判为空。
    if is_mineru_route and not allow_legacy_mineru:
        deep_parse_figures = _load_deep_parse_figures(
            doc_id,
            parse_manifest=parse_manifest,
        )
    elif allow_mineru_sources:
        # Legacy documents have no durable generation identity to fence.
        deep_parse_figures = _load_deep_parse_figures(doc_id)
    else:
        deep_parse_figures = []

    # 没有任何结构信号且也没有原始 PDF 时，视觉兜底无从运行。
    if not deep_parse_figures and not images and not figures and not pdf_url:
        logger.info(f"[FigureExtraction] No images and no figures found for doc {doc_id}")
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
    source = (
        "mineru_deep_parse"
        if deep_parse_figures
        else ("mineru" if is_mineru_route else "pdf_native")
    )
    fallback_used = False
    adapter_results = list(deep_parse_figures)

    # 准备 Adapter 输入
    adapter_input = {
        "figures": figures,
        "images": images,
    }

    if deep_parse_figures:
        logger.info(
            f"[FigureExtraction] MinerU deep-parse cache: got {len(deep_parse_figures)} blocks"
        )

    # 4.1 尝试 MinerU Adapter（如果上传阶段有 MinerU 结果）
    # 注意：当前上传阶段不会自动调用 MinerU 获取 figure 数据
    # 这里检查 ocr_result 中是否有 MinerU 格式的数据
    mineru_figures = (
        doc_data.get("ocr_result", {}).get("figures", [])
        if allow_mineru_sources and not adapter_results
        else []
    )
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

    # 4.2 本地路线优先使用 PDF 原生图片与图注结构。
    if not is_mineru_route and not adapter_results and figures and images:
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

    # 4.3 主解析与 PDF 原生结构均无结果时，才尝试本地视觉模型。
    if not is_mineru_route and not adapter_results and pdf_doc is not None:
        try:
            yolo_blocks = _build_figures_from_yolo_layout(pdf_doc, figures)
            if yolo_blocks:
                adapter_results.extend(yolo_blocks)
                source = "yolo"
                logger.info(f"[FigureExtraction] Visual fallback: got {len(yolo_blocks)} blocks")
        except Exception as e:
            logger.warning(f"[FigureExtraction] Visual fallback failed: {e}")
            fallback_used = True

    # 4.4 Caption-only fallback: 矢量图 PDF 没有光栅图片但有 figure 标题
    if not is_mineru_route and not adapter_results and not images and figures:
        try:
            caption_blocks = _build_figures_from_captions(
                figures, page_width, page_height
            )
            if caption_blocks:
                adapter_results.extend(caption_blocks)
                source = "caption_only"
                logger.info(
                    f"[FigureExtraction] Caption-only fallback: "
                    f"got {len(caption_blocks)} blocks from {len(figures)} captions"
                )
        except Exception as e:
            logger.warning(f"[FigureExtraction] Caption-only fallback failed: {e}")
            fallback_used = True

    # 4.5 Fallback: 使用图片列表
    if not is_mineru_route and not adapter_results and images:
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
        if pdf_doc:
            pdf_doc.close()
        return []

    # 使用 figure_builder 构建
    logical_figures = build_logical_figures(
        adapter_results,
        page_width,
        page_height
    )

    # 图号查询必须从完整结构源精确匹配，不能受速览 Top-N 缓存限制。
    selected = (
        [
            figure for figure in logical_figures
            if _logical_figure_matches_reference(figure, reference)
        ][:1]
        if reference
        else select_top_figures(logical_figures, depth)
    )

    if reference:
        if pdf_doc:
            pdf_doc.close()
        return selected

    # ========== 6. 质量校验（简单版）==========
    # 检查是否有 caption
    valid_count = sum(1 for fig in selected if fig.caption_text)
    if valid_count == 0 and selected:
        logger.info("[FigureExtraction] No captions found, using raw selection")
        # 保留结果，但不认为是高质量

    # ========== 7. 写回缓存 ==========
    # 缓存完整集合；brief/standard/detailed 只影响本次返回的 Top-N。
    # 否则先生成 brief 会永久把 detailed 锁成两张图。
    figures_dict = [fig.model_dump() for fig in logical_figures]

    figures_meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fallback_used": fallback_used,
        "count": len(figures_dict),
        "selection_version": "dynamic-depth-v1",
        "parse_generation": str(parse_manifest.get("generation") or ""),
        "document_source_hash": str(parse_manifest.get("source_hash") or ""),
        "parser_route": resolved_route,
        "grouping_version": MINERU_FIGURE_GROUPING_VERSION,
    }
    figures_status = {
        "state": "done",
        "error": None,
        "provider": source,
    }

    # 发布到当前代际的内存缓存。服务层不能将开始时拿到的 doc_data
    # 快照整块写回 documents_store，否则晚到的旧速览任务会覆盖新解析。
    published = False
    active_doc_is_input = False
    try:
        from routes.document_routes import documents_store, publish_logical_figure_cache

        active_doc_is_input = documents_store.get(doc_id) is doc_record
        published = publish_logical_figure_cache(
            doc_id,
            parse_generation=str(parse_manifest.get("generation") or ""),
            document_source_hash=str(parse_manifest.get("source_hash") or ""),
            parser_route=resolved_route,
            figures=figures_dict,
            metadata=figures_meta,
            status=figures_status,
        )
    except Exception as e:
        logger.warning(f"[FigureExtraction] Failed to publish logical figure cache: {e}")

    # Direct service callers and unit tests may intentionally supply a detached
    # record. Preserve that local result, but never mutate an active record
    # after a guarded publication was rejected.
    if published or not active_doc_is_input:
        doc_data["logical_figures"] = figures_dict
        doc_data["logical_figures_meta"] = figures_meta
        doc_data["logical_figures_status"] = figures_status

    logger.info(
        f"[FigureExtraction] Built {len(figures_dict)} complete figures for doc {doc_id}, "
        f"source={source}, fallback={fallback_used}"
    )

    if pdf_doc:
        pdf_doc.close()

    return selected


def _logical_figure_matches_reference(
    figure: LogicalFigureSchema,
    reference: str,
) -> bool:
    match = re.search(
        r"(?:\b(?:figure|fig\.?)\s*|图\s*)"
        r"([a-z]?\d+(?:[.-]\d+)?(?:[a-z]|\s*\([a-z]\))?)(?![a-z0-9])",
        str(reference or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    target = re.sub(r"[^a-z0-9.-]+", "", match.group(1).lower()).lstrip("0") or "0"
    text = " ".join((
        str(figure.figure_index or ""),
        str(figure.caption_text or ""),
        str(figure.figure_id or ""),
    ))
    candidates = re.findall(
        r"(?:\b(?:figure|fig\.?)\s*|图\s*|fig[_-]?)"
        r"([a-z]?\d+(?:[.-]\d+)?(?:[a-z]|\s*\([a-z]\))?)(?![a-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    return any(
        (re.sub(r"[^a-z0-9.-]+", "", value.lower()).lstrip("0") or "0") == target
        for value in candidates
    )


def _load_deep_parse_figures(
    doc_id: str,
    *,
    parse_manifest: Optional[dict] = None,
) -> List[FigureBlock]:
    """从已重建的 MinerU 深度解析块索引中提取 figure/table，并配对邻近 caption。

    刻意不重新解析 mineru_results 的原始 middle_json/content_list_json：那条
    payload 的 bbox 坐标系因文档而异（可能是 page points，也可能是本文档特有的
    0-1000 归一化坐标——取决于 MinerU 是否在结果里带了 width/height），
    `mineru_block_index_service` 已经按页推断并做了正确缩放（悬浮翻译锚点用的
    就是这份数据，已验证坐标准确）。这里直接复用它转换好的 block bbox，避免
    重新实现一遍缩放逻辑、重蹈坐标错位的覆辙。
    """
    try:
        from routes.document_routes import DATA_DIR
        from services.block_index_service import load_block_index
        from services.mineru_block_index_service import MINERU_BLOCK_INDEX_SOURCE
    except Exception as exc:
        logger.debug("[FigureExtraction] deep-parse figure imports unavailable: %s", exc)
        return []

    block_index = load_block_index(DATA_DIR, doc_id)
    if not isinstance(block_index, dict) or block_index.get("source") != MINERU_BLOCK_INDEX_SOURCE:
        return []

    # A document id can be reused for a new parse generation.  This helper is
    # called by the overview path, where a stale MinerU index would otherwise
    # look structurally valid and leak figures from the prior document state.
    # Legacy callers retain the historical behavior because they have no
    # generation fence to enforce.
    manifest = parse_manifest if isinstance(parse_manifest, dict) else {}
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    if manifest and not metadata.get("legacy_inferred"):
        expected_route = str(manifest.get("resolved_route") or "").strip().lower()
        expected_generation = str(manifest.get("generation") or "").strip()
        expected_source_hash = str(manifest.get("source_hash") or "").strip()
        actual_route = str(block_index.get("parser_route") or "").strip().lower()
        actual_generation = str(block_index.get("parse_generation") or "").strip()
        actual_source_hash = str(block_index.get("document_source_hash") or "").strip()
        if (
            expected_route != "mineru"
            or not expected_generation
            or not expected_source_hash
            or (actual_route, actual_generation, actual_source_hash)
            != (expected_route, expected_generation, expected_source_hash)
        ):
            logger.warning(
                "[FigureExtraction] Ignore stale MinerU block index doc=%s expected=%s/%s/%s actual=%s/%s/%s",
                doc_id,
                expected_route,
                expected_generation,
                expected_source_hash,
                actual_route,
                actual_generation,
                actual_source_hash,
            )
            return []

    return _figure_blocks_from_mineru_block_index(block_index)


def build_mineru_logical_figures_from_block_index(
    block_index: dict,
) -> List[LogicalFigureSchema]:
    """Build canonical logical figures from an already validated MinerU index.

    This is deliberately block-index-only so parser publication, overview and
    reusable visual assets share one geometry interpretation.
    """
    if not isinstance(block_index, dict) or block_index.get("source") != "mineru_vlm":
        return []
    return build_logical_figures(_figure_blocks_from_mineru_block_index(block_index))


def _figure_blocks_from_mineru_block_index(block_index: dict) -> List[FigureBlock]:
    """Extract MinerU body/caption blocks after their coordinates are normalized."""
    blocks: List[FigureBlock] = []
    for page in block_index.get("pages") or []:
        page_num = int(page.get("page") or 1)
        page_blocks = page.get("blocks") or []

        body_blocks = [b for b in page_blocks if b.get("type") in ("figure", "table")]
        if not body_blocks:
            continue

        caption_candidates = [
            {
                "bbox": _normalize_bbox(b.get("bbox")),
                "kind": "table" if str(b.get("mineru_type") or "").startswith("table") else "figure",
                "caption": str(b.get("text") or "").strip(),
                "source": "mineru_caption",
                "block_id": str(b.get("block_id") or ""),
                "mineru_type": str(b.get("mineru_type") or ""),
            }
            for b in page_blocks
            if b.get("type") == "caption" and _normalize_bbox(b.get("bbox"))
        ]
        pdf_captions = [
            caption for caption in caption_candidates
            if not _is_generated_visual_caption(caption)
        ]

        page_figure_blocks: List[FigureBlock] = []
        for idx, body in enumerate(body_blocks):
            body_bbox = _normalize_bbox(body.get("bbox"))
            if not body_bbox:
                continue
            kind = "table" if body.get("type") == "table" else "figure"
            caption = _match_caption_to_body(body_bbox, kind, pdf_captions)
            caption_bbox = caption.get("bbox") if caption else None
            caption_text = (caption or {}).get("caption") or ""
            if not caption_text and kind == "figure":
                # MinerU 的 content_list.json 格式没有独立 caption 块，figure 的
                # caption 文本直接内嵌在块自己的 text 字段里（_convert_mineru_item
                # 找不到文本时才会回退成占位符 "Figure"）。table 的 text 字段是
                # HTML 表体和 caption 拼接后的原始文本，不适合直接当 caption 用，
                # 表格解读留给 RAG/table_aware_service 处理，这里不取。
                body_text = str(body.get("text") or "").strip()
                if body_text and body_text != "Figure":
                    caption_text = body_text
            full_bbox = _merge_bboxes([body_bbox, caption_bbox]) if caption_bbox else body_bbox

            figure_block = FigureBlock(
                figure_id=body.get("block_id") or f"mineru_deep_{kind}_p{page_num}_{idx}",
                page_idx=page_num - 1,
                figure_index=_caption_display_label(caption_text),
                caption_text=caption_text,
                raw_bboxes=[body_bbox],
                body_bbox_page_pts=body_bbox,
                full_bbox_page_pts=full_bbox,
                source=FigureSource.MINERU,
                confidence=0.85,
                source_metadata={
                    "adapter": "mineru_deep_parse",
                    "block_id": body.get("block_id"),
                    "mineru_type": body.get("mineru_type"),
                },
            )
            if kind == "figure":
                page_figure_blocks.append(figure_block)
            else:
                blocks.append(figure_block)

        blocks.extend(
            _group_mineru_panel_figures(
                _group_mineru_figures_by_spanning_caption(page_figure_blocks, pdf_captions)
            )
        )

    return blocks


_PANEL_ONLY_RE = re.compile(r"^\s*\(?\s*([a-z])\s*\)?[.)]?\s*$", re.IGNORECASE)
_PANEL_LABEL_RE = re.compile(
    r"^\s*\(?\s*([a-z])\s*\)?[.)]?(?:\s+.+)?$",
    re.IGNORECASE | re.DOTALL,
)
_PANEL_FIGURE_CAPTION_RE = re.compile(
    r"^\s*\(?\s*[a-z]\s*\)?[.)]?\s*((?:fig(?:ure)?\.?|图)\s*\d+[a-z]?(?:\s*[:：.])?.*)$",
    re.IGNORECASE | re.DOTALL,
)
_FIGURE_CAPTION_RE = re.compile(
    r"(?:^|\s)((?:fig(?:ure)?\.?|图)\s*\d+[a-z]?(?:\s*[:：.])?.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _panel_only_caption(text: str) -> bool:
    return bool(_PANEL_ONLY_RE.fullmatch(" ".join(str(text or "").split())))


def _is_orphan_panel_caption(text: str) -> bool:
    """Recognize a panel label without accepting a second figure caption.

    MinerU frequently emits blank image blocks or labels such as ``(a) Encoder``.
    Those can belong to a neighbouring canonical figure caption.  A block that
    itself contains ``Figure N`` remains an anchor, never an orphan, so nearby
    independent figures are not silently merged.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return True
    if _canonical_figure_caption(value):
        return False
    return bool(_PANEL_LABEL_RE.fullmatch(value))


def _canonical_figure_caption(text: str) -> str:
    value = " ".join(str(text or "").split())
    match = _PANEL_FIGURE_CAPTION_RE.match(value) or _FIGURE_CAPTION_RE.search(value)
    return match.group(1).strip() if match else ""


def _bbox_overlap_ratio(first: list[float], second: list[float], axis: int) -> float:
    start = max(first[axis], second[axis])
    end = min(first[axis + 2], second[axis + 2])
    first_size = first[axis + 2] - first[axis]
    second_size = second[axis + 2] - second[axis]
    return max(0.0, end - start) / max(1.0, min(first_size, second_size))


def _panel_bboxes_are_neighbors(first: list[float], second: list[float]) -> bool:
    horizontal_gap = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical_gap = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    horizontal_aligned = _bbox_overlap_ratio(first, second, 1) >= 0.20
    vertical_aligned = _bbox_overlap_ratio(first, second, 0) >= 0.20
    first_width, second_width = first[2] - first[0], second[2] - second[0]
    first_height, second_height = first[3] - first[1], second[3] - second[1]

    # A fixed 60pt gap is too permissive for small diagrams.  Bound the gap by
    # both an absolute ceiling and the adjacent panel dimensions.
    if horizontal_aligned and vertical_gap == 0:
        allowed_gap = min(60.0, max(18.0, min(first_width, second_width) * 0.70))
        return horizontal_gap <= allowed_gap
    if vertical_aligned and horizontal_gap == 0:
        allowed_gap = min(60.0, max(18.0, min(first_height, second_height) * 0.45))
        return vertical_gap <= allowed_gap
    return False


def _is_generated_visual_caption(caption: dict) -> bool:
    """Ignore VLM overlay captions when pairing MinerU image bodies.

    Those texts often repeat ``图1`` on a single panel and would otherwise beat
    the real spanning PDF caption, leaving Figure 1 split in the overview.
    """
    token = " ".join(
        str(caption.get(key) or "")
        for key in ("block_id", "mineru_type", "source")
    ).lower()
    return "visual_vlm" in token


def _horizontal_overlap_ratio(body: list[float], caption: list[float]) -> float:
    overlap = max(0.0, min(body[2], caption[2]) - max(body[0], caption[0]))
    return overlap / max(body[2] - body[0], 1.0)


def _caption_row_member_indexes(
    figures: List[FigureBlock],
    caption_bbox: list[float],
) -> list[int]:
    """Bodies sitting in the row immediately above a spanning caption."""
    candidates: list[tuple[int, float]] = []
    for index, figure in enumerate(figures):
        body = figure.body_bbox_page_pts
        if not body:
            continue
        gap = caption_bbox[1] - body[3]
        if gap < -24.0 or gap > 220.0:
            continue
        if _horizontal_overlap_ratio(body, caption_bbox) < 0.25:
            continue
        candidates.append((index, gap))
    if not candidates:
        return []
    min_gap = min(gap for _, gap in candidates)
    return [index for index, gap in candidates if gap <= min_gap + 40.0]


def _merge_panel_figure_group(
    members: List[FigureBlock],
    *,
    caption_text: str,
    caption_bbox: Optional[list[float]] = None,
    extra_meta: Optional[dict] = None,
) -> FigureBlock:
    members = sorted(
        members,
        key=lambda item: (
            (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[1],
            (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[0],
        ),
    )
    anchor = members[0]
    panel_bboxes = [
        list(member.body_bbox_page_pts)
        for member in members
        if member.body_bbox_page_pts
    ]
    grouped_body_bbox = _merge_bboxes(panel_bboxes)
    grouped_body_bbox[3] += MINERU_PANEL_LABEL_BOTTOM_PADDING
    full_bboxes = [
        member.full_bbox_page_pts or member.body_bbox_page_pts
        for member in members
        if member.full_bbox_page_pts or member.body_bbox_page_pts
    ]
    if caption_bbox:
        full_bboxes.append(caption_bbox)
    canonical = _canonical_figure_caption(caption_text) or " ".join(str(caption_text or "").split())
    metadata = {
        **anchor.source_metadata,
        "panel_group": True,
        "panel_label_bottom_padding": MINERU_PANEL_LABEL_BOTTOM_PADDING,
        "merged_from": [member.figure_id for member in members],
        "merge_count": len(members),
        **(extra_meta or {}),
    }
    return FigureBlock(
        figure_id=anchor.figure_id,
        page_idx=anchor.page_idx,
        figure_index=_caption_display_label(canonical),
        caption_text=canonical,
        raw_bboxes=panel_bboxes,
        body_bbox_page_pts=grouped_body_bbox,
        full_bbox_page_pts=_merge_bboxes([*full_bboxes, grouped_body_bbox]),
        panel_bboxes_page_pts=panel_bboxes,
        source=anchor.source,
        confidence=max(member.confidence for member in members),
        source_metadata=metadata,
    )


def _group_mineru_figures_by_spanning_caption(
    figures: List[FigureBlock],
    captions: list[dict],
) -> List[FigureBlock]:
    """Merge bodies that share one page-level ``Figure N`` / ``图N`` caption.

    MinerU often splits a dashed multi-panel figure into adjacent image
    blocks and attaches ``(a)`` / ``(b)`` labels to each crop.  The official
    caption still spans the whole row; use that geometry as the grouping key.
    """
    if len(figures) < 2:
        return figures

    remaining = set(range(len(figures)))
    spanning: list[tuple[float, dict, list[int]]] = []
    for caption in captions or []:
        text = str(caption.get("caption") or "").strip()
        bbox = caption.get("bbox")
        if not text or not bbox or not _canonical_figure_caption(text):
            continue
        members = _caption_row_member_indexes(figures, bbox)
        if len(members) >= 2:
            spanning.append((bbox[2] - bbox[0], caption, members))

    spanning.sort(key=lambda item: -item[0])
    grouped: List[FigureBlock] = []
    for _, caption, members in spanning:
        member_indexes = [index for index in members if index in remaining]
        if len(member_indexes) < 2:
            continue
        grouped.append(
            _merge_panel_figure_group(
                [figures[index] for index in member_indexes],
                caption_text=str(caption.get("caption") or ""),
                caption_bbox=caption.get("bbox"),
                extra_meta={"spanning_caption": True},
            )
        )
        remaining.difference_update(member_indexes)

    grouped.extend(figures[index] for index in sorted(remaining))
    grouped.sort(key=lambda item: (
        item.page_idx,
        (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[1],
        (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[0],
    ))
    return grouped


def _group_mineru_panel_figures(figures: List[FigureBlock]) -> List[FigureBlock]:
    """把 MinerU 分拆的同页子图归组，完整 caption 所在块作为组锚点。"""
    if len(figures) < 2:
        return figures

    remaining = set(range(len(figures)))
    canonical_anchor_indexes = {
        index
        for index, figure in enumerate(figures)
        if _canonical_figure_caption(figure.caption_text) and figure.body_bbox_page_pts
    }
    grouped: List[FigureBlock] = []
    for anchor_index, anchor in enumerate(figures):
        if anchor_index not in remaining:
            continue
        canonical_caption = _canonical_figure_caption(anchor.caption_text)
        if not canonical_caption or not anchor.body_bbox_page_pts:
            continue

        member_indexes = {anchor_index}
        changed = True
        while changed:
            changed = False
            if len(member_indexes) >= MAX_MINERU_PANEL_GROUP_SIZE:
                break
            candidate_indexes = sorted(
                remaining - member_indexes,
                key=lambda index: (
                    (figures[index].body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[1],
                    (figures[index].body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[0],
                ),
            )
            for candidate_index in candidate_indexes:
                candidate = figures[candidate_index]
                if (
                    not candidate.body_bbox_page_pts
                    or not _is_orphan_panel_caption(candidate.caption_text)
                ):
                    continue
                # Do not attach a blank/labelled block when it is also adjacent
                # to another canonical caption.  Keeping that ambiguity split is
                # safer than merging two independent figures into one overview.
                if any(
                    other_anchor_index != anchor_index
                    and _panel_bboxes_are_neighbors(
                        figures[other_anchor_index].body_bbox_page_pts,
                        candidate.body_bbox_page_pts,
                    )
                    for other_anchor_index in canonical_anchor_indexes
                ):
                    continue
                if any(
                    _panel_bboxes_are_neighbors(
                        figures[member_index].body_bbox_page_pts,
                        candidate.body_bbox_page_pts,
                    )
                    for member_index in member_indexes
                    if figures[member_index].body_bbox_page_pts
                ):
                    member_indexes.add(candidate_index)
                    changed = True
                    if len(member_indexes) >= MAX_MINERU_PANEL_GROUP_SIZE:
                        break

        if len(member_indexes) < 2:
            continue
        members = [figures[index] for index in member_indexes]
        grouped.append(
            _merge_panel_figure_group(
                members,
                caption_text=canonical_caption,
                extra_meta={"neighbor_group": True},
            )
        )
        remaining.difference_update(member_indexes)

    grouped.extend(figures[index] for index in sorted(remaining))
    grouped.sort(key=lambda item: (
        item.page_idx,
        (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[1],
        (item.body_bbox_page_pts or [0.0, 0.0, 0.0, 0.0])[0],
    ))
    return grouped


def _build_figures_from_yolo_layout(pdf_doc, figures: list) -> List[FigureBlock]:
    """用 DocLayout-YOLO 的 ImageBody/TableBody 直接构建候选图表。

    render_mode=yolo 时调用。它与 caption-only 的区别是：body bbox 来自版面模型，
    caption 只用于命名和补充上下文，不再用“从图注往上猜一块区域”的弱规则。
    """
    try:
        from PIL import Image as PILImage
        from services.layout_service import detect_layout, is_available, pixel_bbox_to_page_pts
    except Exception as exc:
        logger.debug("[FigureExtraction] YOLO layout unavailable: %s", exc)
        return []

    if not is_available():
        return []

    caption_pages = {
        int(fig.get("page") or 0)
        for fig in figures or []
        if int(fig.get("page") or 0) > 0
    }
    pages_to_scan = sorted(caption_pages) if caption_pages else list(range(1, min(len(pdf_doc), 12) + 1))

    caption_candidates = _caption_candidates_by_page(figures)
    blocks: List[FigureBlock] = []

    for page_num in pages_to_scan:
        if page_num <= 0 or page_num > len(pdf_doc):
            continue
        page = pdf_doc[page_num - 1]
        try:
            pix = page.get_pixmap(dpi=144)
            page_image = PILImage.open(io.BytesIO(pix.tobytes("png")))
            detections = detect_layout(page_image, conf=0.12)
        except Exception as exc:
            logger.debug("[FigureExtraction] YOLO page %s failed: %s", page_num, exc)
            continue

        page_width_pts = page.rect.width
        page_height_pts = page.rect.height
        yolo_captions = []
        body_detections = []
        for det in detections or []:
            category = det.get("category_name")
            bbox = det.get("bbox")
            if category not in {"ImageBody", "TableBody", "ImageCaption", "TableCaption"}:
                continue
            bbox_pts = pixel_bbox_to_page_pts(
                bbox,
                page_image.width,
                page_image.height,
                page_width_pts,
                page_height_pts,
            )
            if category in {"ImageCaption", "TableCaption"}:
                yolo_captions.append({
                    "bbox": bbox_pts,
                    "kind": "table" if category == "TableCaption" else "figure",
                    "caption": "",
                    "display_label": "Table" if category == "TableCaption" else "Figure",
                    "source": "yolo_caption",
                    "score": float(det.get("score") or 0.0),
                })
            else:
                body_detections.append({
                    "bbox": bbox_pts,
                    "kind": "table" if category == "TableBody" else "figure",
                    "category": category,
                    "score": float(det.get("score") or 0.0),
                })

        page_captions = caption_candidates.get(page_num, []) + yolo_captions
        used_body_keys: set[tuple[int, int, int, int]] = set()
        for idx, body in enumerate(sorted(body_detections, key=lambda item: (item["bbox"][1], item["bbox"][0])), start=1):
            body_bbox = _normalize_bbox(body.get("bbox"))
            if not body_bbox:
                continue
            body_key = tuple(round(v) for v in body_bbox)
            if body_key in used_body_keys:
                continue
            used_body_keys.add(body_key)

            caption = _match_caption_to_body(body_bbox, body["kind"], page_captions)
            caption_bbox = _normalize_bbox(caption.get("bbox")) if caption else None
            caption_text = (caption or {}).get("caption") or ""
            display_label = (caption or {}).get("display_label") or _fallback_yolo_label(body["kind"], page_num, idx)
            figure_id = _stable_yolo_figure_id(body["kind"], display_label, page_num, idx)
            full_bbox = _merge_bboxes([body_bbox, caption_bbox]) if caption_bbox else body_bbox

            blocks.append(FigureBlock(
                figure_id=figure_id,
                page_idx=page_num - 1,
                figure_index=display_label,
                caption_text=caption_text or display_label,
                raw_bboxes=[body_bbox],
                body_bbox_page_pts=body_bbox,
                full_bbox_page_pts=full_bbox,
                source="yolo",
                confidence=max(0.0, min(float(body.get("score") or 0.0), 1.0)),
                source_metadata={
                    "adapter": "doclayout_yolo",
                    "category": body["category"],
                    "layout_score": body.get("score"),
                    "caption_bbox": caption_bbox,
                    "caption_source": (caption or {}).get("source"),
                    "page_width": page_width_pts,
                    "page_height": page_height_pts,
                },
            ))

    return blocks


def _caption_candidates_by_page(figures: list) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for fig in figures or []:
        page_num = int(fig.get("page") or 0)
        bbox = _normalize_bbox(fig.get("caption_bbox") or fig.get("bbox"))
        if page_num <= 0 or not bbox:
            continue
        caption = str(fig.get("caption") or fig.get("label") or "").strip()
        result.setdefault(page_num, []).append({
            "bbox": bbox,
            "kind": _caption_kind(caption),
            "caption": caption,
            "display_label": fig.get("display_label") or _caption_display_label(caption),
            "source": "pdf_text_caption",
            "score": 0.8,
        })
    return result


def _match_caption_to_body(body_bbox: list[float], body_kind: str, captions: list[dict]) -> Optional[dict]:
    best: tuple[float, dict] | None = None
    body_width = max(body_bbox[2] - body_bbox[0], 1.0)

    for caption in captions or []:
        cap_bbox = _normalize_bbox(caption.get("bbox"))
        if not cap_bbox:
            continue
        kind_penalty = 0.0 if caption.get("kind") == body_kind else 80.0
        overlap = max(0.0, min(body_bbox[2], cap_bbox[2]) - max(body_bbox[0], cap_bbox[0]))
        overlap_ratio = overlap / max(min(body_width, cap_bbox[2] - cap_bbox[0]), 1.0)
        if overlap_ratio < 0.08:
            continue

        above_gap = body_bbox[1] - cap_bbox[3]
        below_gap = cap_bbox[1] - body_bbox[3]
        candidates: list[tuple[float, str]] = []
        if -12.0 <= above_gap <= 160.0:
            candidates.append((max(0.0, above_gap), "above"))
        if -12.0 <= below_gap <= 180.0:
            candidates.append((max(0.0, below_gap), "below"))
        if not candidates:
            continue

        gap, position = min(candidates, key=lambda item: item[0])
        orientation_penalty = 0.0
        if body_kind == "table" and position != "above":
            orientation_penalty = 30.0
        if body_kind == "figure" and position != "below":
            orientation_penalty = 20.0

        score = gap + kind_penalty + orientation_penalty - overlap_ratio * 20.0
        if best is None or score < best[0]:
            best = (score, caption)

    return best[1] if best else None


def _caption_kind(text: str) -> str:
    return "table" if re.match(r"^\s*(table|表)\b", str(text or ""), re.IGNORECASE) else "figure"


def _caption_display_label(text: str) -> str:
    match = re.match(r"^\s*((?:fig(?:ure)?\.?|图|table|表)\s*[0-9ivxlcdm]+[a-z]?)", str(text or ""), re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _fallback_yolo_label(kind: str, page_num: int, index: int) -> str:
    prefix = "Table" if kind == "table" else "Figure"
    return f"{prefix} p{page_num}-{index}"


def _stable_yolo_figure_id(kind: str, display_label: str, page_num: int, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", display_label or "").strip("_").lower()
    return f"yolo_{kind}_p{page_num}_{slug or 'item'}_{index}"


def _normalize_bbox(bbox: object) -> Optional[list[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _merge_bboxes(bboxes: list[Optional[list[float]]]) -> list[float]:
    valid = [bbox for bbox in bboxes if bbox]
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def _build_figures_from_captions(
    figures: list,
    page_width: float,
    page_height: float,
) -> List[FigureBlock]:
    """从 figure 标题（caption）构建 FigureBlock，用于矢量图 PDF。

    当 PDF 没有嵌入光栅图片（常见于 LaTeX TikZ/PGF 绘制的论文）时，
    利用已检测到的 caption bbox 推断 figure body 区域。
    策略：figure 主体在 caption 上方，宽度取页面左右 4% 边距内。
    """

    blocks: List[FigureBlock] = []

    # 按页分组并排序，以便计算相邻 caption 间距
    page_groups: dict = {}
    for fig in figures:
        p = fig.get("page", 1)
        page_groups.setdefault(p, []).append(fig)

    for page_num, page_figs in page_groups.items():
        # 按 caption y 坐标排序
        page_figs.sort(key=lambda f: (f.get("caption_bbox") or f.get("bbox") or [0, 0, 0, 0])[1])

        for i, fig in enumerate(page_figs):
            caption_bbox = fig.get("caption_bbox") or fig.get("bbox")
            if not caption_bbox or len(caption_bbox) != 4:
                continue

            caption_y0 = caption_bbox[1]

            # 估算 body 区域上边界
            if i > 0:
                prev_bbox = page_figs[i - 1].get("caption_bbox") or page_figs[i - 1].get("bbox") or [0, 0, 0, 0]
                # 从上一个 caption 底部开始
                body_top = max(0.0, prev_bbox[3] + 6.0)
            else:
                # 第一个 figure：向上回溯页面 35% 或 280pt
                default_height = min(page_height * 0.35, 280.0)
                body_top = max(0.0, caption_y0 - default_height)

            body_bottom = max(body_top + 1.0, caption_y0 - 4.0)
            if body_bottom <= body_top:
                continue

            # 保证最小高度
            min_height = min(page_height, max(page_height * 0.16, 120.0))
            if body_bottom - body_top < min_height:
                body_top = max(0.0, body_bottom - min_height)

            body_bbox = [
                max(0.0, page_width * 0.04),
                body_top,
                min(page_width, page_width * 0.96),
                body_bottom,
            ]

            # full_bbox = body + caption
            full_bbox = [
                min(body_bbox[0], caption_bbox[0]),
                min(body_bbox[1], caption_bbox[1]),
                max(body_bbox[2], caption_bbox[2]),
                max(body_bbox[3], caption_bbox[3]),
            ]

            block = FigureBlock(
                figure_id=fig.get("figure_id", f"caption_fig_p{page_num}_{i}"),
                page_idx=page_num - 1,  # 转为 0-indexed
                figure_index=fig.get("display_label") or fig.get("label", ""),
                caption_text=fig.get("caption", ""),
                raw_bboxes=[body_bbox],
                body_bbox_page_pts=body_bbox,
                full_bbox_page_pts=full_bbox,
                source="caption_only",
                confidence=0.5,
                source_metadata={
                    "adapter": "caption_only",
                    "caption_bbox": caption_bbox,
                    "page_width": page_width,
                    "page_height": page_height,
                },
            )
            blocks.append(block)

    logger.info(f"[FigureExtraction] Built {len(blocks)} caption-only figure blocks")
    return blocks


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
