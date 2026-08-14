from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.block_index_service import ensure_block_index, load_block_index
from services.ai_cache_state import load_ai_cache_generation
from services.document_parse_state import is_parse_prepared, read_parse_manifest
from services.delta_brief_service import build_delta_brief
from services.reference_extraction_service import (
    REFERENCE_EXTRACTION_VERSION,
    extract_references,
    recommend_by_coupling,
    references_to_bibtex,
)
from services.reading_outline_service import (
    get_or_create_reading_outline,
    load_reading_outline,
    save_reading_outline,
)


router = APIRouter()

# 对照组上限：对比是逐点做的，成本随对照篇数线性增长，且超过三篇后简报本身
# 就不再适合阅读。宁可让用户显式挑选，也不默认拉全库。
MAX_DELTA_BRIEF_BASELINES = 3


def _require_document_parse_ready(doc_id: str, doc: dict) -> dict:
    """主解析器尚未完成时，不允许总结临时文本。"""
    manifest = read_parse_manifest(doc or {}, doc_id=doc_id)
    if is_parse_prepared(manifest):
        return manifest

    route = str(manifest.get("requested_route") or manifest.get("route") or "auto")
    stage = str(manifest.get("stage") or "")
    if route == "mineru" and stage == "awaiting_rag_index":
        detail = "MinerU 已完成版面解析，正在等待问答索引发布"
    elif route == "mineru":
        detail = "当前文档正在按 MinerU 全程解析，完成前不能生成总结"
    else:
        detail = "当前文档解析尚未完成，请稍后重试"
    raise HTTPException(status_code=409, detail=detail)


class DeltaBriefRequest(BaseModel):
    doc_id: str
    baseline_doc_ids: list[str] = []


class ReferenceCouplingRequest(BaseModel):
    doc_id: str
    # 候选范围留空表示与库内全部已解析文档比对。
    candidate_doc_ids: list[str] = []
    limit: int = 10
    min_shared: int = 2


class SummaryRequest(BaseModel):
    doc_id: str
    api_key: Optional[str] = None
    model: str
    api_provider: str
    # 兼容旧客户端省略身份字段；服务端仍会绑定并返回当前身份。
    # 新客户端携带身份后，可在模型调用前直接拒绝过期请求。
    parse_generation: Optional[str] = None
    document_source_hash: Optional[str] = None
    source_hash: Optional[str] = None


def _summary_data_dir() -> Path:
    configured = str(getattr(router, "data_dir", "") or "").strip()
    if configured:
        return Path(configured)
    vector_store_dir = str(getattr(router, "vector_store_dir", "") or "").strip()
    if vector_store_dir:
        return Path(vector_store_dir).parent
    from runtime_mode import runtime
    return Path(runtime.data_dir)


def _resolve_document_pdf_path(doc: dict, data_dir: Path) -> Path | None:
    pdf_url = str((doc or {}).get("pdf_url") or "").strip()
    pdf_name = pdf_url.split("?")[0].replace("\\", "/").rsplit("/", 1)[-1]
    if not pdf_name:
        return None
    pdf_path = data_dir / "uploads" / pdf_name
    return pdf_path if pdf_path.exists() else None


def _manifest_identity(manifest: dict) -> dict[str, str]:
    generation = str(manifest.get("generation") or "").strip()
    source_hash = str(manifest.get("source_hash") or "").strip()
    if not generation or not source_hash:
        raise HTTPException(status_code=409, detail="当前文档解析身份不完整，请重新解析后再试")
    return {
        "parse_generation": generation,
        "document_source_hash": source_hash,
    }


def _bind_requested_parse_identity(request: SummaryRequest, manifest: dict) -> dict[str, str]:
    current = _manifest_identity(manifest)
    requested_generation = str(request.parse_generation or "").strip()
    requested_document_hash = str(request.document_source_hash or "").strip()
    requested_source_hash = str(request.source_hash or "").strip()
    if requested_document_hash and requested_source_hash and requested_document_hash != requested_source_hash:
        raise HTTPException(status_code=409, detail="摘要请求携带了互相冲突的文档来源身份")
    requested_hash = requested_document_hash or requested_source_hash
    if bool(requested_generation) != bool(requested_hash):
        raise HTTPException(
            status_code=409,
            detail="摘要请求必须同时携带 parse_generation 和 source_hash",
        )
    if requested_generation and (
        requested_generation != current["parse_generation"]
        or requested_hash != current["document_source_hash"]
    ):
        raise HTTPException(status_code=409, detail="文档解析结果已更新，请重新生成总结")
    return current


def _require_parse_identity_current(doc_id: str, expected: dict[str, str]) -> dict:
    store = getattr(router, "documents_store", None)
    doc = store.get(doc_id) if isinstance(store, dict) else None
    if not isinstance(doc, dict):
        raise HTTPException(status_code=404, detail="文档未找到")
    manifest = _require_document_parse_ready(doc_id, doc)
    if _manifest_identity(manifest) != expected:
        raise HTTPException(
            status_code=409,
            detail="文档解析结果已在摘要生成期间更新，请重新生成总结",
        )
    return doc


def _require_bound_block_index(
    block_index: dict[str, Any],
    manifest: dict,
    expected: dict[str, str],
) -> None:
    index_generation = str(block_index.get("parse_generation") or "").strip()
    index_source_hash = str(block_index.get("document_source_hash") or "").strip()
    legacy = bool((manifest.get("metadata") or {}).get("legacy_inferred"))
    if not legacy and (not index_generation or not index_source_hash):
        raise HTTPException(status_code=409, detail="当前阅读块索引缺少解析身份，请重新生成索引")
    if index_generation and index_generation != expected["parse_generation"]:
        raise HTTPException(status_code=409, detail="当前阅读块索引属于旧的解析代际")
    if index_source_hash and index_source_hash != expected["document_source_hash"]:
        raise HTTPException(status_code=409, detail="当前阅读块索引与文档来源不一致")

    has_text = any(
        str(block.get("text") or "").strip()
        for page in block_index.get("pages") or []
        if isinstance(page, dict)
        for block in page.get("blocks") or []
        if isinstance(block, dict)
    )
    if not has_text:
        raise HTTPException(status_code=409, detail="当前解析结果没有可用于生成总结的正文")


def _legacy_summary_from_outline(outline: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in outline.get("items") or []:
        if not isinstance(item, dict) or str(item.get("type") or "") == "title":
            continue
        title = str(item.get("title") or "").strip()
        content = str(item.get("summary") or "").strip()
        if not content:
            continue
        lines.append(f"• {title}：{content}" if title else f"• {content}")
    return "\n".join(lines).strip()


def _legacy_questions_from_outline(outline: dict[str, Any]) -> list[str]:
    questions: list[str] = []

    def visit(items: Any) -> None:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            title = str(item.get("title") or "").strip(" ：:。")
            if title and item_type != "title":
                question = f"文档如何阐述{title}？"
                if question not in questions:
                    questions.append(question)
            if len(questions) < 5:
                visit(item.get("children"))
            if len(questions) >= 5:
                return

    visit(outline.get("items"))
    return questions[:5]


@router.post("/summary")
async def generate_summary(request: SummaryRequest):
    store = getattr(router, "documents_store", None)
    if not isinstance(store, dict):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = store[request.doc_id]
    manifest = _require_document_parse_ready(request.doc_id, doc)
    parse_identity = _bind_requested_parse_identity(request, manifest)
    data_dir = _summary_data_dir()
    ai_cache_generation = load_ai_cache_generation(data_dir, request.doc_id)
    try:
        block_index = ensure_block_index(
            doc_id=request.doc_id,
            doc=doc,
            data_dir=data_dir,
            pdf_path=_resolve_document_pdf_path(doc, data_dir),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"阅读块索引尚未就绪: {exc}") from exc
    _require_bound_block_index(block_index, manifest, parse_identity)

    def persist_if_current(outline: dict[str, Any]) -> None:
        _require_parse_identity_current(request.doc_id, parse_identity)
        if load_ai_cache_generation(data_dir, request.doc_id) != ai_cache_generation:
            raise HTTPException(status_code=409, detail="总结缓存已失效，请基于当前文档状态重新生成")
        if (
            str(outline.get("parse_generation") or "") != parse_identity["parse_generation"]
            or str(outline.get("document_source_hash") or "")
            != parse_identity["document_source_hash"]
        ):
            raise HTTPException(status_code=409, detail="阅读总结结果不属于当前解析代际")
        save_reading_outline(data_dir, request.doc_id, outline)

    try:
        outline = await get_or_create_reading_outline(
            data_dir=data_dir,
            doc_id=request.doc_id,
            doc=doc,
            block_index=block_index,
            api_key=str(request.api_key or ""),
            model=request.model,
            provider=request.api_provider,
            force=False,
            cache_writer=persist_if_current,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"摘要生成失败: {exc}") from exc

    _require_parse_identity_current(request.doc_id, parse_identity)
    summary = _legacy_summary_from_outline(outline)
    if not summary:
        raise HTTPException(status_code=503, detail="摘要服务未返回有效正文，请稍后重试")

    source = str(outline.get("source") or "fallback").strip().lower()
    degraded = source != "ai"
    meta = outline.get("meta") if isinstance(outline.get("meta"), dict) else {}
    degraded_reason = str(meta.get("generation_error") or "").strip()
    if degraded and not degraded_reason:
        degraded_reason = "当前返回基于阅读块索引的确定性摘要，模型摘要尚未生成"

    # 保留旧响应字段，同时显式返回降级状态和解析身份。
    # 确定性兜底可以继续使用，但不能伪装成模型成功响应。
    return {
        "summary": summary,
        "suggested_questions": _legacy_questions_from_outline(outline),
        "doc_id": request.doc_id,
        "timestamp": datetime.now().isoformat(),
        "used_provider": outline.get("provider") or meta.get("provider"),
        "used_model": outline.get("model") or meta.get("model"),
        "fallback_used": degraded,
        "status": "degraded" if degraded else "completed",
        "degraded": degraded,
        "degraded_reason": degraded_reason if degraded else "",
        "parse_generation": parse_identity["parse_generation"],
        "source_hash": parse_identity["document_source_hash"],
        "document_source_hash": parse_identity["document_source_hash"],
        "block_source_hash": outline.get("source_hash") or "",
    }


def _outline_title(doc_id: str, doc: Any, outline: dict[str, Any]) -> str:
    title = str((outline or {}).get("title") or "").strip()
    if title:
        return title
    if isinstance(doc, dict):
        metadata = doc.get("paper_metadata") if isinstance(doc.get("paper_metadata"), dict) else {}
        return str(metadata.get("title") or doc.get("filename") or doc_id).strip() or doc_id
    return doc_id


@router.post("/delta-brief")
async def generate_delta_brief(request: DeltaBriefRequest):
    """相对若干对照论文，给出这篇论文的真实增量。

    只读已有的阅读大纲，不触发解析、不调用模型、不联网：拿不到大纲的文档
    直接报告为不可比较，而不是临时生成一份质量未知的替代品。
    """
    store = getattr(router, "documents_store", None)
    if not isinstance(store, dict):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")

    data_dir = _summary_data_dir()
    primary_outline = load_reading_outline(data_dir, request.doc_id)
    if not isinstance(primary_outline, dict):
        raise HTTPException(
            status_code=409,
            detail="当前文档还没有可用的阅读总结，请先生成总结再做增量对比",
        )

    baselines: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for baseline_id in request.baseline_doc_ids[:MAX_DELTA_BRIEF_BASELINES]:
        baseline_id = str(baseline_id or "").strip()
        if not baseline_id or baseline_id == request.doc_id:
            continue
        if baseline_id not in store:
            skipped.append({"doc_id": baseline_id, "reason": "document_not_found"})
            continue
        outline = load_reading_outline(data_dir, baseline_id)
        if not isinstance(outline, dict):
            # 对照论文没有大纲就跳过并说明，不静默降级成"没有增量"。
            skipped.append({"doc_id": baseline_id, "reason": "reading_outline_missing"})
            continue
        baselines.append({
            "doc_id": baseline_id,
            "title": _outline_title(baseline_id, store.get(baseline_id), outline),
            "outline": outline,
        })

    brief = build_delta_brief(
        {
            "doc_id": request.doc_id,
            "title": _outline_title(request.doc_id, store.get(request.doc_id), primary_outline),
            "outline": primary_outline,
        },
        baselines,
    )
    brief["skipped_baselines"] = skipped
    brief["timestamp"] = datetime.now().isoformat()
    return brief


def _document_references(doc_id: str, doc: Any) -> list:
    """取一篇文档的参考文献，拿不到块索引就返回空。

    只读已发布的块索引缓存，不走 ensure_block_index：推荐是附加能力，遍历
    整库时不应该顺带触发一轮重建。
    """
    if not isinstance(doc, dict):
        return []
    try:
        block_index = load_block_index(_summary_data_dir(), doc_id)
    except Exception:
        return []
    blocks = (block_index or {}).get("blocks") if isinstance(block_index, dict) else None
    if not isinstance(blocks, list):
        return []
    return extract_references(blocks)


@router.post("/reference-coupling")
async def recommend_reference_coupling(request: ReferenceCouplingRequest):
    """按共享参考文献推荐库内相关论文。

    完全本地：不联网、不调用模型。共享参考文献是作者自己给出的主题信号，比
    摘要相似度更难被表述差异干扰，但只有 DOI/arXiv/长标题能作为可靠身份，
    解析不出的条目会被丢弃而不是猜。
    """
    store = getattr(router, "documents_store", None)
    if not isinstance(store, dict):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")

    target = _document_references(request.doc_id, store.get(request.doc_id))
    if not target:
        return {
            "doc_id": request.doc_id,
            "version": REFERENCE_EXTRACTION_VERSION,
            "reference_count": 0,
            "recommendations": [],
            # 与"没有相关论文"区分开：这是取不到参考文献，不是比对后没结果。
            "status": "no_references_extracted",
            "timestamp": datetime.now().isoformat(),
        }

    candidate_ids = [
        str(doc_id) for doc_id in (request.candidate_doc_ids or store.keys())
        if str(doc_id) != request.doc_id and str(doc_id) in store
    ]
    corpus = {}
    for candidate_id in candidate_ids:
        references = _document_references(candidate_id, store.get(candidate_id))
        if references:
            corpus[candidate_id] = references

    ranked = recommend_by_coupling(
        target,
        corpus,
        limit=max(1, min(50, int(request.limit or 10))),
        min_shared=max(1, int(request.min_shared or 2)),
    )
    for row in ranked:
        candidate = store.get(row["doc_id"])
        row["title"] = _outline_title(row["doc_id"], candidate, {})

    return {
        "doc_id": request.doc_id,
        "version": REFERENCE_EXTRACTION_VERSION,
        "reference_count": len(target),
        "compared_document_count": len(corpus),
        "recommendations": ranked,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/references/bibtex")
async def export_references_bibtex(request: ReferenceCouplingRequest):
    """导出这篇文档参考文献的 BibTeX。

    只写解析出来的字段，不臆造作者或期刊——缺字段可以在文献管理器里补，
    编造的字段会被当成真的。
    """
    store = getattr(router, "documents_store", None)
    if not isinstance(store, dict):
        raise HTTPException(status_code=500, detail="文档存储未初始化")
    if request.doc_id not in store:
        raise HTTPException(status_code=404, detail="文档未找到")

    references = _document_references(request.doc_id, store.get(request.doc_id))
    return {
        "doc_id": request.doc_id,
        "version": REFERENCE_EXTRACTION_VERSION,
        "reference_count": len(references),
        "bibtex": references_to_bibtex(references),
        "references": [reference.to_dict() for reference in references],
        "timestamp": datetime.now().isoformat(),
    }
