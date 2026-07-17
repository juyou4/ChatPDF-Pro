"""按风险触发的页级与图号级视觉补充，不改变主解析路线。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz

from services.document_parse_state import read_parse_manifest
from services.figure_extraction import build_logical_figures_for_overview
from services.figure_render import render_figure
from services.figure_validation import validate_and_fallback
from services.visual_enrichment_service import (
    VisualTaskPolicy,
    build_visual_task_id,
    execute_visual_task,
)
from services.visual_model_service import (
    VisualEnrichmentPolicy,
    VisualModelConfig,
    call_visual_model,
)
from services.visual_risk_service import (
    assess_figure_risk,
    assess_page_risk,
    extract_figure_reference,
    page_text_for_risk,
)
from services.visual_supplement_service import (
    VISUAL_SUPPLEMENT_FIGURE_ON_DEMAND_PROMPT_VERSION as FIGURE_ON_DEMAND_PROMPT_VERSION,
    VISUAL_SUPPLEMENT_PAGE_RECOVERY_PROMPT_VERSION as PAGE_RECOVERY_PROMPT_VERSION,
    build_visual_supplement,
    committed_visual_evidence_for_document,
    visual_supplement_revision,
)


_PAGE_LIMITS = {"privacy": 4, "balanced": 6, "quality": 12}


def _task_policy(policy: VisualEnrichmentPolicy) -> VisualTaskPolicy:
    return VisualTaskPolicy(
        timeout_seconds=60.0,
        max_retries=1,
        retry_delay_seconds=0.4,
        concurrency=2,
        document_budget=policy.document_budget,
        cache_ttl_seconds=6 * 60 * 60,
    )


def _response_content(response: Any) -> str:
    if not isinstance(response, dict):
        return str(response or "").strip()
    if response.get("content"):
        return str(response.get("content") or "").strip()
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "").strip()
    return ""


def _response_json(response: Any) -> dict[str, Any]:
    content = _response_content(response)
    if not content:
        return {}
    start = content.find("{")
    end = content.rfind("}") + 1
    candidate = content[start:end] if start >= 0 and end > start else content
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"text": content, "analysis": content, "confidence": 0.5}


def _confidence(value: Any, fallback: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def _visual_evidence_model_is_current(
    item: dict[str, Any],
    policy: VisualEnrichmentPolicy,
) -> bool:
    visual_model = item.get("visual_model")
    item_identity = str(
        visual_model.get("identity") if isinstance(visual_model, dict) else ""
    ).strip()
    if not item_identity:
        return False
    current_identities = {
        config.identity
        for config in (policy.strong_model, policy.local_model)
        if config.can_call
    }
    return item_identity in current_identities


def _pixmap_looks_blank(pix: Any) -> bool:
    """Use a bounded pixel sample to avoid spending VLM budget on blank pages."""
    try:
        channels = max(1, int(pix.n))
        samples = memoryview(pix.samples)
        pixel_count = len(samples) // channels
    except (AttributeError, TypeError, ValueError):
        return False
    if pixel_count <= 0:
        return True

    step = max(1, pixel_count // 4096)
    luminance: list[float] = []
    color_channels = min(3, channels)
    for pixel_index in range(0, pixel_count, step):
        offset = pixel_index * channels
        values = samples[offset:offset + color_channels]
        if not values:
            continue
        luminance.append(sum(values) / len(values))
    if not luminance:
        return True

    dynamic_range = max(luminance) - min(luminance)
    if dynamic_range < 8.0:
        return True
    mean = sum(luminance) / len(luminance)
    variance = sum((value - mean) ** 2 for value in luminance) / len(luminance)
    return dynamic_range < 20.0 and variance < 4.0


async def _analyze_image(
    *,
    image_url: str,
    prompt: str,
    system_prompt: str,
    config: VisualModelConfig,
    purpose: str,
    prompt_version: str,
    document_id: str,
    route: str,
    parse_generation: str,
    document_source_hash: str,
    page: int,
    bbox_hash: str,
    policy: VisualEnrichmentPolicy,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]

    async def operation() -> Any:
        return await call_visual_model(
            messages=messages,
            config=config,
            purpose=purpose,
            max_tokens=1200,
            temperature=0,
        )

    response = await execute_visual_task(
        task_id=build_visual_task_id({
            "document_id": document_id,
            "route": route,
            "parse_generation": parse_generation,
            "document_source_hash": document_source_hash,
            "purpose": purpose,
            "page": page,
            "bbox_hash": bbox_hash,
            "model": config.identity,
            "prompt_version": prompt_version,
        }),
        document_id=document_id,
        parse_generation=parse_generation,
        purpose=purpose,
        operation=operation,
        policy=_task_policy(policy),
        metadata={
            "provider": config.provider,
            "model": config.model,
            "source": config.source,
            "page": page,
            "bbox_hash": bbox_hash,
            "route": route,
            "document_source_hash": document_source_hash,
            "prompt_version": prompt_version,
        },
    )
    return _response_json(response)


async def recover_risky_local_pages(
    *,
    doc_id: str,
    doc: dict[str, Any],
    pdf_path: Path | None,
    visual_policy: VisualEnrichmentPolicy,
) -> dict[str, Any]:
    """为 local 文档的空白、极低文本或乱码页生成限额补充块。"""
    manifest = read_parse_manifest(doc, doc_id=doc_id)
    route = str(manifest.get("resolved_route") or "").strip().lower()
    generation = str(manifest.get("generation") or "").strip()
    source_hash = str(manifest.get("source_hash") or "").strip()
    if route != "local":
        return {"items": [], "text": "", "diagnostics": {"skipped_reason": "non_local_route"}}

    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    parse_identity = {
        "parser_route": route,
        "parse_generation": generation,
        "document_source_hash": source_hash,
    }
    envelope = data.get("visual_supplements")
    reuse_existing = bool(
        isinstance(envelope, dict)
        and str(envelope.get("visual_model_identity") or "") == visual_policy.identity
    )
    existing = committed_visual_evidence_for_document(doc, limit=64) if reuse_existing else []
    existing_revision = (
        visual_supplement_revision(data, parse_identity)
        if reuse_existing
        else ""
    )
    existing_pages = {
        int(item.get("page") or 0)
        for item in existing
        if str(item.get("purpose") or "") == "scan_region_recognition"
        and _visual_evidence_model_is_current(item, visual_policy)
    }
    existing_text = [
        str(item.get("text") or "").strip()
        for item in existing
        if str(item.get("purpose") or "") == "scan_region_recognition"
        and _visual_evidence_model_is_current(item, visual_policy)
        and str(item.get("text") or "").strip()
    ]
    if not pdf_path or not Path(pdf_path).exists():
        return {
            "items": [],
            "text": "\n\n".join(existing_text),
            "visual_supplement_revision": existing_revision,
            "diagnostics": {"skipped_reason": "missing_pdf"},
        }

    pdf_doc = fitz.open(str(pdf_path))
    try:
        candidates: list[dict[str, Any]] = []
        for page_index in range(pdf_doc.page_count):
            page_number = page_index + 1
            if page_number in existing_pages:
                continue
            page = pdf_doc[page_index]
            bbox = [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
            assessment = assess_page_risk(
                page=page_number,
                bbox=bbox,
                page_text=page_text_for_risk(data, page_number),
                threshold=visual_policy.risk_threshold,
            )
            if assessment.should_enrich:
                candidates.append({
                    "page_index": page_index,
                    "page": page_number,
                    "bbox": bbox,
                    "assessment": assessment,
                })

        candidate_count = len(candidates)
        page_limit = min(
            visual_policy.document_budget,
            _PAGE_LIMITS[visual_policy.normalized_strategy],
        )
        jobs: list[tuple[dict[str, Any], VisualModelConfig, str]] = []
        blank_pages_skipped = 0
        render_failed_pages = 0
        for candidate in candidates:
            if len(jobs) >= page_limit:
                break
            config = visual_policy.select(
                risk_level=candidate["assessment"].level,
                purpose="scan_region_recognition",
            )
            if not config.can_call:
                continue
            try:
                page = pdf_doc[candidate["page_index"]]
                pix = page.get_pixmap(dpi=144, alpha=False, annots=False)
                if _pixmap_looks_blank(pix):
                    blank_pages_skipped += 1
                    continue
                image_b64 = base64.b64encode(
                    pix.pil_tobytes(format="JPEG", quality=78)
                ).decode("ascii")
            except Exception:
                render_failed_pages += 1
                continue
            image_url = f"data:image/jpeg;base64,{image_b64}"
            jobs.append((candidate, config, image_url))

        results = await asyncio.gather(*(
            _analyze_image(
                image_url=image_url,
                prompt=(
                    f"这是 PDF 第 {candidate['page']} 页。请只依据可见内容输出 JSON："
                    '{"text":"忠实转写的关键文字，最多1200字","summary":"一句话页面说明",'
                    '"confidence":0.0}。看不清的内容不要猜测；表格、公式和图示只描述可确认信息。'
                ),
                system_prompt="你是谨慎的 PDF 页面视觉识别助手，只转写和概括图中明确可见的内容。",
                config=config,
                purpose="scan_region_recognition",
                prompt_version=PAGE_RECOVERY_PROMPT_VERSION,
                document_id=doc_id,
                route=route,
                parse_generation=generation,
                document_source_hash=source_hash,
                page=candidate["page"],
                bbox_hash=candidate["assessment"].bbox_hash,
                policy=visual_policy,
            )
            for candidate, config, image_url in jobs
        ), return_exceptions=True) if jobs else []

        items: list[dict[str, Any]] = []
        failures = 0
        for (candidate, config, _image_url), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures += 1
                continue
            text = " ".join(str(result.get("text") or "").split())
            summary = " ".join(str(result.get("summary") or "").split())
            if not text and not summary:
                failures += 1
                continue
            item = build_visual_supplement(
                figure_id=f"page-{candidate['page']}",
                page=candidate["page"],
                bbox=candidate["bbox"],
                caption=summary or f"第 {candidate['page']} 页视觉补充",
                analysis=text or summary,
                visual_model_identity=config.identity,
                provider=config.provider,
                model=config.model,
                render_mode="page",
                purpose="scan_region_recognition",
                confidence=_confidence(result.get("confidence")),
                prompt_version=PAGE_RECOVERY_PROMPT_VERSION,
            )
            if item:
                item["visual_risk"] = candidate["assessment"].to_dict()
                items.append(item)

        recovered_text = [*existing_text, *(str(item.get("text") or "") for item in items)]
        return {
            "items": items,
            "text": "\n\n".join(part for part in recovered_text if part),
            "parse_generation": generation,
            "document_source_hash": source_hash,
            "visual_supplement_revision": existing_revision,
            "diagnostics": {
                "candidate_pages": candidate_count,
                "requested_pages": len(jobs),
                "completed_pages": len(items),
                "failed_pages": failures,
                "render_failed_pages": render_failed_pages,
                "blank_pages_skipped": blank_pages_skipped,
                "page_limit": page_limit,
            },
        }
    finally:
        pdf_doc.close()


def _normalized_figure_number(value: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "", str(value or "").lower()).lstrip("0") or "0"


def _figure_matches_reference(figure: Any, number: str) -> bool:
    target = _normalized_figure_number(number)
    text = " ".join((
        str(getattr(figure, "figure_index", "") or ""),
        str(getattr(figure, "caption_text", "") or ""),
        str(getattr(figure, "figure_id", "") or ""),
    ))
    matches = re.findall(
        r"(?:\b(?:figure|fig\.?)\s*|图\s*|fig[_-]?)"
        r"([a-z]?\d+(?:[.-]\d+)?(?:[a-z]|\s*\([a-z]\))?)(?![a-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    return any(_normalized_figure_number(match) == target for match in matches)


def _text_matches_reference(text: str, number: str) -> bool:
    holder = type("FigureReferenceText", (), {
        "figure_index": text,
        "caption_text": text,
        "figure_id": text,
    })()
    return _figure_matches_reference(holder, number)


def _has_substantive_reference_evidence(text: str, label: str) -> bool:
    normalized_label = re.sub(
        r"[\s.:：。_\-()（）\[\]]+",
        "",
        str(label or "").lower(),
    )
    if not normalized_label:
        return False
    for line in str(text or "").splitlines():
        normalized_line = re.sub(r"[\s.:：。_\-()（）\[\]]+", "", line.lower())
        if normalized_label not in normalized_line:
            continue
        remainder = normalized_line.replace(normalized_label, "", 1)
        if len(remainder) >= 80:
            return True
    return False


async def enrich_referenced_figure(
    *,
    doc_id: str,
    doc: dict[str, Any],
    pdf_path: Path | None,
    query: str,
    text_evidence: str,
    visual_policy: VisualEnrichmentPolicy,
) -> dict[str, Any]:
    """普通问答明确点名图号且检索无证据时，按需解读对应图。"""
    reference = extract_figure_reference(query)
    if not reference:
        return {"item": None, "diagnostics": {"skipped_reason": "no_figure_reference"}}
    if _has_substantive_reference_evidence(text_evidence, reference["label"]):
        return {"item": None, "diagnostics": {"skipped_reason": "substantive_figure_evidence_present"}}
    manifest = read_parse_manifest(doc, doc_id=doc_id)
    generation = str(manifest.get("generation") or "").strip()
    route = str(manifest.get("resolved_route") or "").strip().lower()
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    envelope = data.get("visual_supplements") if isinstance(data, dict) else None
    if (
        route == "local"
        and isinstance(envelope, dict)
        and str(envelope.get("visual_model_identity") or "") == visual_policy.identity
    ):
        for item in committed_visual_evidence_for_document(doc, limit=64):
            combined = f"{item.get('figure_id', '')} {item.get('caption', '')}"
            if (
                not _text_matches_reference(combined, reference["number"])
                or not _visual_evidence_model_is_current(item, visual_policy)
            ):
                continue
            return {
                "item": item,
                "parse_generation": generation,
                "document_source_hash": str(manifest.get("source_hash") or ""),
                "route": route,
                "visual_model_identity": visual_policy.identity,
                "diagnostics": {
                    "triggered": False,
                    "reused_committed": True,
                    "reference": reference,
                    "visual_supplement_revision": str(
                        item.get("visual_supplement_revision") or ""
                    ),
                },
            }

    if not pdf_path or not Path(pdf_path).exists():
        return {"item": None, "diagnostics": {"skipped_reason": "missing_pdf"}}

    figures = build_logical_figures_for_overview(
        doc_id,
        doc,
        depth="detailed",
        reference=reference["label"],
    )
    figure = next(
        (candidate for candidate in figures if _figure_matches_reference(candidate, reference["number"])),
        None,
    )
    if figure is None:
        return {"item": None, "diagnostics": {"skipped_reason": "figure_not_located"}}

    assessment = assess_figure_risk(
        figure,
        page_text=page_text_for_risk(data, int(figure.page_idx) + 1),
        query=query,
        text_evidence=text_evidence,
        threshold=visual_policy.risk_threshold,
    )
    if not assessment.should_enrich:
        return {"item": None, "diagnostics": {"skipped_reason": "risk_not_triggered", "visual_risk": assessment.to_dict()}}
    config = visual_policy.select(risk_level=assessment.level, purpose="figure_description")
    if not config.can_call:
        return {"item": None, "diagnostics": {"skipped_reason": "visual_model_unavailable", "visual_risk": assessment.to_dict()}}

    pdf_doc = fitz.open(str(pdf_path))
    try:
        render_result = validate_and_fallback(
            figure,
            pdf_doc,
            render_figure,
            render_kwargs={"render_mode": "raw"},
        )
    finally:
        pdf_doc.close()
    if not render_result.success:
        return {"item": None, "diagnostics": {"skipped_reason": "render_failed", "visual_risk": assessment.to_dict()}}

    image_b64 = render_result.model_image_base64 or render_result.display_image_base64
    image_mime = "image/jpeg" if render_result.model_image_base64 else "image/png"
    result = await _analyze_image(
        image_url=f"data:{image_mime};base64,{image_b64}",
        prompt=(
            f"请解读 {reference['label']}。已有图注：{figure.caption_text or '无'}。"
            '输出 JSON：{"caption":"一句话图题","analysis":"2-4句可核验的图表内容与趋势",'
            '"confidence":0.0}。看不清的数值不要猜测。'
        ),
        system_prompt="你是谨慎的学术图表分析助手，只陈述图片中可确认的信息。",
        config=config,
        purpose="figure_description",
        prompt_version=FIGURE_ON_DEMAND_PROMPT_VERSION,
        document_id=doc_id,
        route=route,
        parse_generation=generation,
        document_source_hash=str(manifest.get("source_hash") or ""),
        page=int(figure.page_idx) + 1,
        bbox_hash=assessment.bbox_hash,
        policy=visual_policy,
    )
    analysis = " ".join(str(result.get("analysis") or result.get("text") or "").split())
    caption = " ".join(str(result.get("caption") or figure.caption_text or reference["label"]).split())
    if not analysis:
        return {"item": None, "diagnostics": {"skipped_reason": "empty_visual_result", "visual_risk": assessment.to_dict()}}

    bbox = figure.full_bbox_page_pts or figure.body_bbox_page_pts or []
    if route == "local":
        item = build_visual_supplement(
            figure_id=str(figure.figure_id or reference["label"]),
            page=int(figure.page_idx) + 1,
            bbox=bbox,
            caption=caption,
            analysis=analysis,
            visual_model_identity=config.identity,
            provider=config.provider,
            model=config.model,
            render_mode="raw",
            purpose="figure_description",
            confidence=_confidence(result.get("confidence")),
            prompt_version=FIGURE_ON_DEMAND_PROMPT_VERSION,
        )
    else:
        digest = hashlib.sha256(
            f"{doc_id}:{generation}:{figure.figure_id}:{config.identity}".encode("utf-8")
        ).hexdigest()[:18]
        item = {
            "id": f"visual_runtime_{digest}",
            "figure_id": str(figure.figure_id or ""),
            "page": int(figure.page_idx) + 1,
            "bbox": list(bbox),
            "bbox_hash": assessment.bbox_hash,
            "caption": caption,
            "analysis": analysis,
            "text": f"{caption}. {analysis}",
            "source": "visual_vlm",
            "route": route or "mineru",
            "block_type": "visual_enrichment",
            "purpose": "figure_description",
            "confidence": _confidence(result.get("confidence")),
            "provider": config.provider,
            "model": config.model,
            "prompt_version": FIGURE_ON_DEMAND_PROMPT_VERSION,
            "visual_model": {"identity": config.identity, **config.public_metadata()},
        }
    if item:
        item["visual_risk"] = assessment.to_dict()
    return {
        "item": item,
        "parse_generation": generation,
        "document_source_hash": str(manifest.get("source_hash") or ""),
        "route": route,
        "visual_model_identity": visual_policy.identity,
        "diagnostics": {
            "triggered": True,
            "reference": reference,
            "visual_risk": assessment.to_dict(),
            "visual_model": config.public_metadata(),
        },
    }
