"""视觉资产检索适配器合同。

视觉检索器只能给出当前请求内资产的 ``asset_id`` 排序。返回值不能携带
证据内容、坐标、模型配置或解析身份；这些字段必须由调用方从当前的
``modal_asset_index`` 重新水合并做公开字段白名单处理。
"""
from __future__ import annotations

import copy
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from services.modal_asset_service import search_modal_assets

logger = logging.getLogger(__name__)

DEFAULT_VISUAL_RETRIEVER_ID = "asset_text_v1"
_MAX_ASSET_ID_LENGTH = 240
_SAFE_RETRIEVER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


@dataclass(frozen=True)
class VisualRetrieverRequest:
    """已规范化的视觉检索请求，不含模型或解析控制参数。"""

    query: str = ""
    reference: str = ""
    page: int = 0
    kinds: tuple[str, ...] = ()
    limit: int = 5


@dataclass(frozen=True)
class VisualRetrieverScope:
    """检索器必须原样返回的固定解析身份。"""

    route: str = ""
    generation: str = ""
    source_hash: str = ""
    revision: str = ""
    index_id: str = ""


@dataclass(frozen=True)
class VisualRetrievalResult:
    """检索器唯一允许返回的结果：按优先级排序的资产 ID。"""

    asset_ids: tuple[str, ...] = ()
    scope: VisualRetrieverScope = VisualRetrieverScope()


class VisualRetriever(Protocol):
    """可插拔视觉检索器的最小合同。"""

    retriever_id: str

    def retrieve(
        self,
        *,
        request: VisualRetrieverRequest,
        scope: VisualRetrieverScope,
        asset_index: Mapping[str, Any],
    ) -> VisualRetrievalResult:
        """Return ranked canonical asset IDs only."""


@dataclass(frozen=True)
class VisualRetrieverExecution:
    """调用端可安全公开的检索执行状态。"""

    asset_ids: tuple[str, ...]
    retriever_id: str
    status: str = "ok"
    fallback_reason: str = ""
    rejected_asset_count: int = 0

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.retriever_id,
            "status": self.status,
        }
        if self.fallback_reason:
            result["fallback_reason"] = self.fallback_reason
        if self.rejected_asset_count:
            result["rejected_asset_count"] = self.rejected_asset_count
        return result


class DeterministicAssetRetriever:
    """现有 asset-text 排序的适配器，作为默认且零额外模型成本的实现。"""

    retriever_id = DEFAULT_VISUAL_RETRIEVER_ID

    def retrieve(
        self,
        *,
        request: VisualRetrieverRequest,
        scope: VisualRetrieverScope,
        asset_index: Mapping[str, Any],
    ) -> VisualRetrievalResult:
        assets = search_modal_assets(
            dict(asset_index),
            query=request.query,
            reference=request.reference,
            page=request.page,
            kinds=list(request.kinds) or None,
            limit=request.limit,
        )
        return VisualRetrievalResult(
            scope=scope,
            asset_ids=tuple(
                str(asset.get("asset_id") or "").strip()
                for asset in assets
                if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
            )
        )


DEFAULT_VISUAL_RETRIEVER = DeterministicAssetRetriever()


def execute_visual_retriever(
    retriever: VisualRetriever | None,
    *,
    request: VisualRetrieverRequest,
    modal_asset_index: Mapping[str, Any] | None,
) -> VisualRetrieverExecution:
    """执行检索器并把结果限制为当前索引中存在的资产 ID。

    外部适配器得到的是深拷贝后的最小文本投影，不能影响请求中的规范
    索引。任何异常、异步返回值或非法返回类型均确定性回退到默认实现。
    """
    canonical_ids = _canonical_asset_ids(modal_asset_index)
    scope = _scope_from_index(modal_asset_index)
    active_retriever = retriever if _supports_visual_retriever(retriever) else DEFAULT_VISUAL_RETRIEVER
    requested_id = _safe_retriever_id(getattr(active_retriever, "retriever_id", ""))
    is_default = active_retriever is DEFAULT_VISUAL_RETRIEVER

    try:
        result = active_retriever.retrieve(
            request=request,
            scope=scope,
            asset_index=_retriever_index_snapshot(modal_asset_index),
        )
        if inspect.isawaitable(result):
            _close_awaitable_if_possible(result)
            raise TypeError("async_visual_retriever_not_supported")
        asset_ids, rejected = _validated_result_ids(
            result,
            scope=scope,
            canonical_ids=canonical_ids,
            limit=request.limit,
        )
        if rejected:
            raise ValueError("invalid_visual_retriever_asset_ids")
        return VisualRetrieverExecution(
            asset_ids=asset_ids,
            retriever_id=requested_id,
            rejected_asset_count=rejected,
        )
    except Exception:
        if is_default:
            logger.warning("[VisualRetriever] 默认视觉资产检索失败，返回空结果", exc_info=True)
            return VisualRetrieverExecution(
                asset_ids=(),
                retriever_id=DEFAULT_VISUAL_RETRIEVER_ID,
                status="failed_closed",
                fallback_reason="default_retriever_failed",
            )

        logger.warning("[VisualRetriever] 适配器执行失败，回退 asset_text_v1", exc_info=True)
        fallback = _run_default_retriever(
            request=request,
            modal_asset_index=modal_asset_index,
            canonical_ids=canonical_ids,
            scope=scope,
        )
        return VisualRetrieverExecution(
            asset_ids=fallback,
            retriever_id=DEFAULT_VISUAL_RETRIEVER_ID,
            status="fallback",
            fallback_reason="retriever_failed",
        )


def deterministic_ranked_assets(
    *,
    request: VisualRetrieverRequest,
    modal_asset_index: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """返回可信 asset-text 排序副本，仅用于内部评分保持。"""
    if not isinstance(modal_asset_index, Mapping):
        return []
    try:
        assets = search_modal_assets(
            dict(modal_asset_index),
            query=request.query,
            reference=request.reference,
            page=request.page,
            kinds=list(request.kinds) or None,
            limit=request.limit,
        )
    except Exception:
        logger.warning("[VisualRetriever] 生成默认可信排序失败", exc_info=True)
        return []
    return [copy.deepcopy(asset) for asset in assets if isinstance(asset, dict)]


def _run_default_retriever(
    *,
    request: VisualRetrieverRequest,
    modal_asset_index: Mapping[str, Any] | None,
    canonical_ids: set[str],
    scope: VisualRetrieverScope,
) -> tuple[str, ...]:
    try:
        result = DEFAULT_VISUAL_RETRIEVER.retrieve(
            request=request,
            scope=scope,
            asset_index=_retriever_index_snapshot(modal_asset_index),
        )
        asset_ids, _ = _validated_result_ids(
            result,
            scope=scope,
            canonical_ids=canonical_ids,
            limit=request.limit,
        )
        return asset_ids
    except Exception:
        logger.warning("[VisualRetriever] 默认回退检索失败", exc_info=True)
        return ()


def _supports_visual_retriever(value: Any) -> bool:
    return callable(getattr(value, "retrieve", None))


def _safe_retriever_id(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if _SAFE_RETRIEVER_ID_RE.fullmatch(candidate) else "custom"


def _canonical_asset_ids(index: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(index, Mapping):
        return set()
    assets = index.get("assets")
    if not isinstance(assets, list):
        return set()
    result: set[str] = set()
    for asset in assets:
        if not isinstance(asset, Mapping):
            continue
        asset_id = str(asset.get("asset_id") or "").strip()
        if asset_id and len(asset_id) <= _MAX_ASSET_ID_LENGTH:
            result.add(asset_id)
    return result


def _scope_from_index(index: Mapping[str, Any] | None) -> VisualRetrieverScope:
    if not isinstance(index, Mapping):
        return VisualRetrieverScope()
    return VisualRetrieverScope(
        route=_scope_text(index.get("parser_route") or index.get("route"), 32).lower(),
        generation=_scope_text(index.get("parse_generation") or index.get("generation"), 160),
        source_hash=_scope_text(index.get("document_source_hash") or index.get("source_hash"), 256),
        revision=_scope_text(index.get("visual_supplement_revision") or index.get("revision"), 160),
        index_id=_scope_text(index.get("index_id"), 160),
    )


def _scope_text(value: Any, limit: int) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return " ".join(str(value).split())[:limit]


def _validated_result_ids(
    result: Any,
    *,
    scope: VisualRetrieverScope,
    canonical_ids: set[str],
    limit: int,
) -> tuple[tuple[str, ...], int]:
    if not isinstance(result, VisualRetrievalResult):
        raise TypeError("invalid_visual_retriever_result")
    if result.scope != scope:
        raise ValueError("visual_retriever_scope_mismatch")
    if not isinstance(result.asset_ids, (tuple, list)):
        raise TypeError("invalid_visual_retriever_asset_ids")
    if len(result.asset_ids) > 32:
        raise ValueError("visual_retriever_asset_id_limit")
    accepted: list[str] = []
    rejected = 0
    bounded_limit = max(1, min(int(limit or 1), 8))
    for raw_id in result.asset_ids:
        if not isinstance(raw_id, str):
            rejected += 1
            continue
        asset_id = raw_id.strip()
        if (
            not asset_id
            or len(asset_id) > _MAX_ASSET_ID_LENGTH
            or asset_id not in canonical_ids
            or asset_id in accepted
        ):
            rejected += 1
            continue
        accepted.append(asset_id)
    return tuple(accepted[:bounded_limit]), rejected


def _retriever_index_snapshot(index: Mapping[str, Any] | None) -> dict[str, Any]:
    """向可插拔检索器提供最小、可变但隔离的 asset-text 视图。"""
    if not isinstance(index, Mapping):
        return {"assets": []}

    snapshot = {"assets": []}
    assets = index.get("assets")
    if not isinstance(assets, list):
        return snapshot

    for raw_asset in assets:
        if not isinstance(raw_asset, Mapping):
            continue
        asset_id = _safe_snapshot_text(raw_asset.get("asset_id"), _MAX_ASSET_ID_LENGTH)
        if not asset_id:
            continue
        item = {
            "asset_id": asset_id,
            "kind": _safe_snapshot_text(raw_asset.get("kind"), 80),
            "page": _safe_snapshot_page(raw_asset.get("page")),
            "bbox": _safe_snapshot_bbox(raw_asset.get("bbox") or raw_asset.get("figure_bbox")),
            "caption": _safe_snapshot_text(raw_asset.get("caption"), 8000),
            "description": _safe_snapshot_text(raw_asset.get("description"), 8000),
            "text": _safe_snapshot_text(raw_asset.get("text"), 8000),
            "figure_id": _safe_snapshot_text(raw_asset.get("figure_id"), 240),
            "section_id": _safe_snapshot_text(raw_asset.get("section_id"), 160),
            "references": _safe_snapshot_references(raw_asset.get("references")),
            "confidence": _safe_snapshot_confidence(raw_asset.get("confidence")),
            "visual_provenance": ["enrichment"] if raw_asset.get("visual_provenance") else [],
        }
        snapshot["assets"].append(item)
    return copy.deepcopy(snapshot)


def _safe_snapshot_text(value: Any, limit: int) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return " ".join(str(value or "").split())[:limit]


def _safe_snapshot_page(value: Any) -> int:
    try:
        return max(0, min(int(value or 0), 1_000_000))
    except (TypeError, ValueError):
        return 0


def _safe_snapshot_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence != confidence or abs(confidence) == float("inf"):
        return None
    return max(0.0, min(1.0, confidence))


def _safe_snapshot_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return []
    if any(number != number or abs(number) == float("inf") for number in (x0, y0, x1, y1)):
        return []
    if x1 <= x0 or y1 <= y0:
        return []
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _safe_snapshot_references(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        reference = _safe_snapshot_text(item, 160)
        if reference and reference not in result:
            result.append(reference)
    return result[:32]


def _close_awaitable_if_possible(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
