"""AI structured reading outline with evidence block bindings."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.chat_service import call_ai_api
from services.document_block_roles import (
    FRONT_MATTER_ROLES,
    ROLE_KEYWORDS,
    annotate_block_role,
    block_is_keyword_excluded,
    block_is_outline_excluded,
    classify_block_role,
)
from services.document_parse_state import read_parse_manifest
from services.keyword_extractor import KeywordExtractor
from services.structured_json import (
    StructuredJSONError,
    parse_json_object,
    structured_json_request_params,
)

logger = logging.getLogger(__name__)
_fallback_keyword_extractor = KeywordExtractor()

READING_OUTLINE_VERSION = 4
READING_OUTLINE_PROMPT_VERSION = "reading-outline-v4.13"
READING_OUTLINE_CONTRACT = "thematic-quick-study-v3"
READING_OUTLINE_DETAIL_LEVEL = "thematic-quick-study"
SECTION_BATCH_SIZE = 5
SECTION_BATCH_CONCURRENCY = 2
MAX_SECTION_PROMPT_BLOCKS = 8
# 长章节动态采样：按块数 35% 采样，上限 16 块；短章节仍然全量。
MAX_SECTION_PROMPT_BLOCKS_LONG = 16
SECTION_SAMPLING_RATIO_PCT = 35
# 单批选中块预算：长章节采样变多后，防止固定 5 章/批把 prompt 撑爆。
SECTION_BATCH_BLOCK_BUDGET = 30
SECTION_BATCH_TABLE_SECTION_LIMIT = 2
SECTION_OUTPUT_MIN_TOKENS = 1200
SECTION_OUTPUT_TOKENS_PER_SECTION = 520
SECTION_OUTPUT_MAX_TOKENS = 4096
SECTION_OUTPUT_CLAIM_RISK_TOKENS = 160
OVERVIEW_OUTPUT_MIN_TOKENS = 1800
OVERVIEW_OUTPUT_MAX_TOKENS = 3200
LONG_SECTION_REVIEW_COVERAGE_PCT = 60.0
MAX_SECTION_REVIEW_BLOCKS = 12
MAX_SECTION_REVIEW_SECTIONS = 2
MAX_SECTION_BLOCK_TEXT = 800
MAX_SECTION_SUMMARY_CHARS = 90
MAX_STUDY_FIELD_CHARS = 180
MAX_FINDING_CHARS = 160
MAX_SECTION_FINDINGS = 3
MAX_OVERVIEW_SUMMARY_CHARS = 90
MAX_THEME_SUMMARY_CHARS = 130
MAX_THEME_POINT_LABEL_CHARS = 14
MAX_THEME_POINT_TEXT_CHARS = 80
MAX_SECTION_TABLE_BUNDLES = 4
MAX_SECTION_TABLE_ROWS = 40
MAX_TABLE_ROW_TEXT_CHARS = 600
FLOW_SECTION_TITLE_PATTERN = re.compile(
    r"(?:method|approach|framework|architecture|pipeline|training|generation|adaptation|algorithm|"
    r"方法|框架|架构|流程|训练|生成|适配|算法)",
    re.IGNORECASE,
)
BACKGROUND_SECTION_TITLE_PATTERN = re.compile(
    r"(?:abstract|introduction|background|motivation|related\s+work|prior\s+work|"
    r"摘要|引言|背景|动机|相关工作)",
    re.IGNORECASE,
)
EXPERIMENT_SECTION_TITLE_PATTERN = re.compile(
    r"(?:experiment|evaluation|result|analysis|ablation|benchmark|robustness|generalization|"
    r"实验|评估|结果|分析|消融|基准|鲁棒|泛化)",
    re.IGNORECASE,
)
CONCLUSION_SECTION_TITLE_PATTERN = re.compile(
    r"(?:conclusion|discussion|limitation|future|impact|结论|讨论|局限|限制|未来|影响)",
    re.IGNORECASE,
)
FLOW_MARKER_PATTERN = re.compile(
    r"(?:\b(?:stage|step|phase)\s*(?:[-:]?\s*)?(?:\d+|[ivx]+|one|two|three|first|second|third)\b|"
    r"\b(?:first|second|third)\s+(?:stage|step|phase)\b|"
    r"第[一二三四五六七八九十\d]+(?:阶段|步)|阶段[一二三四五六七八九十\d]+|"
    r"(?:首先|其次|然后|最后).{0,36}(?:训练|优化|冻结|更新))",
    re.IGNORECASE,
)
PROSE_CLAIM_RISK_PATTERN = re.compile(
    r"(?:优于|高于|低于|领先|提升|下降|接近随机|所有|均|最高|最低|"
    r"证明|导致|因而|outperform|improv|decreas|increase|random|\ball\b|\bbest\b|\bworst\b|cause)",
    re.IGNORECASE,
)
PROSE_CLAIM_QUALIFIER_PATTERNS = (
    (
        "zero_shot",
        re.compile(r"zero[- ]shot|零样本", re.IGNORECASE),
        re.compile(r"zero[- ]shot|零样本", re.IGNORECASE),
    ),
    (
        "supervised",
        re.compile(r"supervised|labeled\s+data|监督|有标签", re.IGNORECASE),
        re.compile(r"supervised|监督|有标签", re.IGNORECASE),
    ),
    (
        "linear_classifier",
        re.compile(r"linear\s+(?:probe|classifier)|logistic\s+regression|线性(?:探测|分类)|逻辑回归", re.IGNORECASE),
        re.compile(r"linear\s+(?:probe|classifier)|logistic\s+regression|线性(?:探测|分类)|逻辑回归", re.IGNORECASE),
    ),
    (
        "compute_efficiency",
        re.compile(r"compute\s+efficien|计算效率|算力效率", re.IGNORECASE),
        re.compile(r"compute\s+efficien|计算效率|算力效率", re.IGNORECASE),
    ),
    (
        "distribution_shift",
        re.compile(r"distribution\s+shift|out[- ]of[- ]distribution|分布偏移|分布外", re.IGNORECASE),
        re.compile(r"distribution\s+shift|out[- ]of[- ]distribution|分布偏移|分布外", re.IGNORECASE),
    ),
    (
        "percentage_point",
        re.compile(r"percentage\s+points?|百分点", re.IGNORECASE),
        re.compile(r"percentage\s+points?|百分点", re.IGNORECASE),
    ),
    (
        "negation_or_decrease",
        re.compile(r"\b(?:not|no|without|cannot|failed?)\b|不|未|无|不能|无法|下降|降低", re.IGNORECASE),
        re.compile(r"\b(?:not|no|without|cannot|failed?)\b|不|未|无|不能|无法|下降|降低", re.IGNORECASE),
    ),
    (
        "uncertainty_or_limit",
        re.compile(r"\b(?:may|might|likely|potentially|only|slightly|approximately)\b|可能|或许|有限|仅|轻微|约", re.IGNORECASE),
        re.compile(r"\b(?:may|might|likely|potentially|only|slightly|approximately)\b|可能|或许|有限|仅|轻微|约", re.IGNORECASE),
    ),
)
THEME_SPECS = (
    ("background", "研究背景与问题", 0),
    ("innovation", "核心方法与创新", 3),
    ("experiment", "关键实验结果", 5),
    ("conclusion", "结论、价值与边界", 0),
)
OVERVIEW_COVERAGE_SLOT_SPECS = (
    (
        "problem",
        "研究问题与动机",
        "background",
        re.compile(r"abstract|introduction|background|motivation|problem|摘要|引言|背景|动机|问题", re.IGNORECASE),
    ),
    (
        "mechanism",
        "核心机制与推理路径",
        "innovation",
        re.compile(r"method|approach|framework|architecture|model|training|inference|zero[- ]shot|方法|框架|架构|模型|训练|推理|零样本", re.IGNORECASE),
    ),
    (
        "data_or_setup",
        "数据构建或基准定义",
        "innovation",
        re.compile(r"dataset|data collection|pretraining data|benchmark construction|数据集|数据采集|预训练数据|基准构建", re.IGNORECASE),
    ),
    (
        "main_result",
        "标志性主结果",
        "experiment",
        re.compile(r"main result|result|evaluation|experiment|performance|zero[- ]shot|transfer|主结果|结果|评估|实验|性能|迁移", re.IGNORECASE),
    ),
    (
        "comparison",
        "关键基线或现有工作对比",
        "experiment",
        re.compile(r"comparison|baseline|state of the art|current work|versus|对比|基线|现有工作", re.IGNORECASE),
    ),
    (
        "ablation",
        "消融与组件贡献",
        "experiment",
        re.compile(r"ablation|component|sensitivity|消融|组件|敏感性", re.IGNORECASE),
    ),
    (
        "generalization",
        "泛化、鲁棒性或真实场景",
        "experiment",
        re.compile(r"generalization|robustness|distribution shift|transfer|physical|real[- ]world|泛化|鲁棒|分布偏移|迁移|物理|真实场景", re.IGNORECASE),
    ),
    (
        "theory",
        "理论结论与成立条件",
        "innovation",
        re.compile(r"theorem|proof|lemma|proposition|analysis|定理|证明|引理|命题|理论分析", re.IGNORECASE),
    ),
    (
        "taxonomy",
        "分类体系与主要脉络",
        "innovation",
        re.compile(r"survey|review|taxonomy|categorization|综述|回顾|分类体系|分类法", re.IGNORECASE),
    ),
    (
        "boundary",
        "结论、限制与适用边界",
        "conclusion",
        re.compile(r"conclusion|discussion|limitation|failure|bias|ethic|future|结论|讨论|局限|限制|失败|偏差|伦理|未来", re.IGNORECASE),
    ),
)
THEME_SUMMARY_PLACEHOLDERS = {
    "全文要旨",
    "背景与问题",
    "方法总括",
    "结果总括",
    "结论价值与边界",
    "请填写",
}
MISSING_SECTION_SUMMARY = "该小节未提取到独立正文。"
LEGACY_MISSING_SECTION_SUMMARY = "当前解析结果未提取到可供总结的正文。"
MAX_PROMPT_BLOCKS = 96
MAX_BLOCK_TEXT = 700
MAX_PROMPT_HEADINGS = 36
MAX_SECTION_CONTENT_BLOCKS = 3
MAX_PROMPT_RESULT_BLOCKS = 12
# Failed model calls are short-lived cache hints, never completed results.
FAILED_GENERATION_COOLDOWN_SECONDS = 60.0

_KEYWORD_GROUPS = {
    "background": [
        "abstract", "introduction", "background", "motivation", "problem",
        "challenge", "limitation", "gap", "existing", "previous",
    ],
    "innovation": [
        "contribution", "we propose", "we present", "novel", "innovation",
        "framework", "pipeline", "method", "approach", "in short",
    ],
    "experiment": [
        "experiment", "evaluation", "result", "comparison", "ablation",
        "dataset", "setup", "metric", "table", "figure",
    ],
    "conclusion": [
        "conclusion", "discussion", "future", "summary", "value",
    ],
}


def get_reading_outline_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "reading_outlines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reading_outline_path(data_dir: Path | str, doc_id: str) -> Path:
    return get_reading_outline_dir(data_dir) / f"{doc_id}.json"


def load_reading_outline(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_reading_outline_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != READING_OUTLINE_VERSION:
            return None
        return data
    except Exception as exc:
        logger.warning("[ReadingOutline] Failed to load %s: %s", path, exc)
        return None


def save_reading_outline(data_dir: Path | str, doc_id: str, outline: dict[str, Any]) -> None:
    if not _has_parse_identity(outline):
        logger.warning("[ReadingOutline] Skip cache without parse identity for %s", doc_id)
        return
    path = get_reading_outline_path(data_dir, doc_id)
    outline["version"] = READING_OUTLINE_VERSION
    outline["doc_id"] = doc_id
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(outline, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    except Exception as exc:
        logger.warning("[ReadingOutline] Failed to save %s: %s", path, exc)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


async def get_or_create_reading_outline(
    *,
    data_dir: Path | str,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    api_key: str = "",
    model: str = "gpt-4o",
    provider: str = "openai",
    endpoint: str = "",
    force: bool = False,
    cache_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return cached AI outline, generate one when credentials are available.

    If no model credentials are available, returns a deterministic fallback with
    the same evidence-bound schema, so the UI remains usable.
    """
    structured_table_bundles = _document_structured_table_bundles(doc)
    source_hash = _source_hash(block_index, structured_table_bundles)
    parse_generation, document_source_hash = _parse_identity(doc_id, doc)
    provider_lower = (provider or "").lower()
    can_call_model = bool(api_key) or provider_lower in {"local", "ollama"}

    cached = load_reading_outline(data_dir, doc_id)
    healthy_cached: dict[str, Any] | None = None
    if (
        cached
        and _matches_parse_identity(cached, parse_generation, document_source_hash)
        and cached.get("source_hash") == source_hash
        and cached.get("prompt_version") == READING_OUTLINE_PROMPT_VERSION
    ):
        cache_repaired = _repair_cached_summary_tree(cached)
        if cache_repaired:
            try:
                save_reading_outline(data_dir, doc_id, cached)
            except Exception as exc:
                logger.warning("[ReadingOutline] Failed to persist summary cache repair doc=%s: %s", doc_id, exc)
        if _is_ai_outline_source(cached.get("source")):
            quality_issues = _reading_outline_quality_issues(cached, block_index)
            blocking_quality_issues = _blocking_reading_outline_quality_issues(quality_issues)
            model_matches = _matches_ai_generation(cached, provider=provider, model=model)
            if not blocking_quality_issues:
                healthy_cached = cached
            if not blocking_quality_issues and (not can_call_model or (not force and model_matches)):
                logger.info(
                    "[AI-Audit] purpose=reading_outline doc=%s provider=%s model=%s status=%s",
                    doc_id,
                    cached.get("provider") or "",
                    cached.get("model") or "",
                    "partial_cache_hit" if quality_issues else "cache_hit",
                )
                return cached
            logger.info(
                "[ReadingOutline] Ignore AI cache doc=%s model_match=%s blocking_quality=%s partial_quality=%s",
                doc_id,
                model_matches,
                blocking_quality_issues,
                _partial_reading_outline_quality_issues(quality_issues),
            )
        elif not can_call_model or (not force and _matches_failed_generation(cached, provider=provider, model=model)):
            return cached
    if cached:
        logger.info(
            "[ReadingOutline] Ignore stale cache doc=%s parse_match=%s source_match=%s",
            doc_id,
            _matches_parse_identity(cached, parse_generation, document_source_hash),
            cached.get("source_hash") == source_hash,
        )

    fallback = build_fallback_reading_outline(
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        structured_table_bundles=structured_table_bundles,
        source_hash=source_hash,
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )
    if not can_call_model:
        return fallback

    try:
        logger.info(
            "[AI-Audit] purpose=reading_outline doc=%s provider=%s model=%s status=start",
            doc_id,
            provider,
            model,
        )
        quality_feedback: list[str] = []
        outline: dict[str, Any] = {}
        retry_cache = None if force else cached
        for quality_attempt in range(2):
            generated = await _generate_ai_outline(
                doc_id=doc_id,
                doc=doc,
                block_index=block_index,
                structured_table_bundles=structured_table_bundles,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                quality_feedback=quality_feedback,
                cached_outline=retry_cache,
            )
            outline = _normalize_outline(
                raw=generated,
                doc_id=doc_id,
                doc=doc,
                block_index=block_index,
                structured_table_bundles=structured_table_bundles,
                source_hash=source_hash,
                source="ai",
                model=model,
                provider=provider,
                parse_generation=parse_generation,
                document_source_hash=document_source_hash,
            )
            quality_feedback = _reading_outline_quality_issues(outline, block_index)
            if not quality_feedback:
                break
            if quality_attempt == 0:
                retry_cache = outline
                logger.info(
                    "[ReadingOutline] Low-quality result from %s/%s; retrying once: %s",
                    provider,
                    model,
                    quality_feedback,
                )
        blocking_quality_issues = _blocking_reading_outline_quality_issues(quality_feedback)
        partial_quality_issues = _partial_reading_outline_quality_issues(quality_feedback)
        if blocking_quality_issues:
            raise ValueError("low-quality reading outline: " + "; ".join(blocking_quality_issues))
        outline_meta = outline.setdefault("meta", {})
        if partial_quality_issues:
            # A single section may exhaust its retry budget while the rest of
            # the document is evidence-bound and useful. Preserve that work;
            # the UI exposes an explicit regenerate action for a later retry.
            outline["source"] = "ai_partial"
            outline_meta["generation_status"] = "partial"
            outline_meta["partial_quality_issues"] = partial_quality_issues
            outline_meta["retryable"] = True
        else:
            outline_meta["generation_status"] = "completed"
            outline_meta["partial_quality_issues"] = []
            outline_meta["retryable"] = False
    except Exception as exc:
        logger.warning(
            "[AI-Audit] purpose=reading_outline doc=%s provider=%s model=%s status=failed error=%s",
            doc_id,
            provider,
            model,
            exc,
        )
        if healthy_cached is not None:
            logger.warning(
                "[ReadingOutline] Refresh failed for %s; preserving verified AI cache: %s",
                doc_id,
                exc,
            )
            return healthy_cached
        fallback.setdefault("meta", {})["generation_error"] = str(exc)
        fallback.setdefault("meta", {})["generation_error_at"] = time.time()
        fallback.setdefault("meta", {})["provider"] = provider
        fallback.setdefault("meta", {})["model"] = model
        writer = cache_writer or (lambda value: save_reading_outline(data_dir, doc_id, value))
        writer(fallback)
        return fallback

    # A route can inject a generation fence here.  Keep it outside the model
    # failure handler so a stale result cannot be reported as a valid fallback.
    writer = cache_writer or (lambda value: save_reading_outline(data_dir, doc_id, value))
    writer(outline)
    logger.info(
        "[AI-Audit] purpose=reading_outline doc=%s provider=%s model=%s status=success",
        doc_id,
        provider,
        model,
    )
    return outline


def _matches_failed_generation(
    cached: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> bool:
    """相同模型的自动生成失败缓存可复用，手动 force 会在调用前绕过。"""
    if cached.get("prompt_version") != READING_OUTLINE_PROMPT_VERSION:
        return False
    meta = cached.get("meta") if isinstance(cached.get("meta"), dict) else {}
    if not str(meta.get("generation_error") or "").strip():
        return False
    if (
        str(meta.get("provider") or "").strip().lower() != str(provider or "").strip().lower()
        or str(meta.get("model") or "").strip() != str(model or "").strip()
    ):
        return False
    try:
        age = time.time() - float(meta.get("generation_error_at"))
    except (TypeError, ValueError):
        return False
    return 0 <= age < FAILED_GENERATION_COOLDOWN_SECONDS


def _is_ai_outline_source(value: Any) -> bool:
    return str(value or "").strip().lower() in {"ai", "ai_partial"}


def _partial_reading_outline_quality_issues(issues: list[str]) -> list[str]:
    """Return retriable per-section issues without discarding valid chapters."""
    return [
        issue for issue in issues
        if str(issue).startswith(("正文章节摘要不完整:", "附录章节摘要不完整:"))
    ]


def _blocking_reading_outline_quality_issues(issues: list[str]) -> list[str]:
    partial = set(_partial_reading_outline_quality_issues(issues))
    return [issue for issue in issues if issue not in partial]


def _matches_ai_generation(
    cached: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> bool:
    return (
        cached.get("prompt_version") == READING_OUTLINE_PROMPT_VERSION
        and str(cached.get("provider") or "").strip().lower()
        == str(provider or "").strip().lower()
        and str(cached.get("model") or "").strip() == str(model or "").strip()
    )


def build_fallback_reading_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None = None,
    source_hash: str | None = None,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> dict[str, Any]:
    skeleton = _build_reading_section_skeleton(block_index)
    payloads = _prepare_section_payloads(skeleton, block_index, structured_table_bundles)
    raw = _build_section_outline_raw(
        skeleton=skeleton,
        payloads=payloads,
        results={},
        block_index=block_index,
    )
    raw["title"] = _paper_title(doc, block_index)
    raw["generation_stats"] = {
        "generated_sections": 0,
        "reused_sections": 0,
        "fallback_sections": sum(1 for payload in payloads.values() if payload.get("blocks")),
    }
    return _normalize_outline(
        raw=raw,
        doc_id=doc_id,
        doc=doc,
        block_index=block_index,
        structured_table_bundles=structured_table_bundles,
        source_hash=source_hash or _source_hash(block_index, structured_table_bundles),
        source="fallback",
        model="",
        provider="",
        parse_generation=parse_generation,
        document_source_hash=document_source_hash,
    )


def _section_table_row_count(section: dict[str, Any]) -> int:
    return sum(
        len(bundle.get("rows") or [])
        for bundle in section.get("table_evidence") or []
        if isinstance(bundle, dict)
    )


def _section_output_token_budget(sections: list[dict[str, Any]]) -> int:
    """Estimate JSON output cost from the contract, not from paper length alone."""
    section_count = max(1, len(sections))
    base = max(SECTION_OUTPUT_MIN_TOKENS, SECTION_OUTPUT_TOKENS_PER_SECTION * section_count)
    table_sections = sum(1 for section in sections if _section_table_row_count(section) > 0)
    table_rows = sum(min(_section_table_row_count(section), 40) for section in sections)
    flow_labels = sum(len(section.get("flow_labels") or []) for section in sections)
    claim_risk_sections = sum(
        1
        for section in sections
        if any(
            PROSE_CLAIM_RISK_PATTERN.search(str(block.get("text") or ""))
            or _numeric_tokens(block.get("text"))
            for block in section.get("blocks") or []
            if isinstance(block, dict)
        )
    )
    complexity_reserve = (
        180 * table_sections
        + min(720, 18 * table_rows)
        + min(320, 48 * flow_labels)
        + min(640, SECTION_OUTPUT_CLAIM_RISK_TOKENS * claim_risk_sections)
    )
    return min(SECTION_OUTPUT_MAX_TOKENS, base + complexity_reserve)


def _overview_output_token_budget(digest: list[dict[str, Any]]) -> int:
    validated_claims = sum(
        len(item.get("metric_claims") or []) + len(item.get("prose_claims") or [])
        for item in digest
        if isinstance(item, dict)
    )
    return min(
        OVERVIEW_OUTPUT_MAX_TOKENS,
        OVERVIEW_OUTPUT_MIN_TOKENS + min(1400, validated_claims * 90),
    )


def _pack_section_batches(sections: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Preserve order while bounding input size and high-risk table output complexity."""
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_weight = 0
    current_table_sections = 0
    for section in sections:
        table_rows = _section_table_row_count(section)
        is_table_section = table_rows > 0
        weight = max(1, len(section.get("blocks") or []) + (table_rows + 7) // 8)
        if current_batch and (
            current_weight + weight > SECTION_BATCH_BLOCK_BUDGET
            or len(current_batch) >= SECTION_BATCH_SIZE
            or (
                is_table_section
                and current_table_sections >= SECTION_BATCH_TABLE_SECTION_LIMIT
            )
        ):
            batches.append(current_batch)
            current_batch = []
            current_weight = 0
            current_table_sections = 0
        current_batch.append(section)
        current_weight += weight
        current_table_sections += int(is_table_section)
    if current_batch:
        batches.append(current_batch)
    return batches


def _table_row_lookup(table_evidence: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for bundle in table_evidence or []:
        bundle_id = str(bundle.get("bundle_id") or "")
        for row in bundle.get("rows") or []:
            if not isinstance(row, dict):
                continue
            evidence_unit_id = str(row.get("evidence_unit_id") or "").strip()
            if not evidence_unit_id:
                continue
            rows[evidence_unit_id] = {
                **row,
                "bundle_id": bundle_id,
                "table_id": str(bundle.get("table_id") or ""),
                "caption": str(bundle.get("caption") or ""),
                "header": str(bundle.get("header") or ""),
                "row_count": int(bundle.get("row_count") or len(bundle.get("rows") or [])),
                "rows_truncated": bool(bundle.get("rows_truncated")),
            }
    return rows


def _numeric_value(value: str) -> float | None:
    normalized = str(value or "").replace(",", "").replace("%", "").strip()
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def _row_column_text(row: dict[str, Any], column: int) -> str:
    for cell in row.get("cells") or []:
        try:
            if int(cell.get("column") or 0) == column:
                return str(cell.get("text") or "")
        except (TypeError, ValueError):
            continue
    return ""


def _metric_claim_fragments(claim_text: str) -> list[str]:
    fragments = [
        value.strip()
        for value in re.split(
            r"(?:[，；;。！？]|(?<!\d),(?!\d)|(?<!\d)\.(?!\d)|(?<!\d)!(?!\d)|"
            r"\b(?:vs\.?|versus|than)\b|高于|低于|优于|不及|超过|落后于)",
            str(claim_text or ""),
            flags=re.IGNORECASE,
        )
        if value.strip()
    ]
    return fragments or [str(claim_text or "")]


def _claim_supports_row_binding(
    claim_text: str,
    *,
    subject: str,
    values: set[str],
    peer_subjects: set[str],
    peer_values: set[str],
) -> bool:
    subject_key = _normalized_match_text(subject)
    if not subject_key or not values:
        return False
    for fragment in _metric_claim_fragments(claim_text):
        fragment_key = _normalized_match_text(fragment)
        fragment_values = _numeric_tokens(fragment)
        if subject_key not in fragment_key or not values.issubset(fragment_values):
            continue
        if any(
            peer_key and peer_key in fragment_key
            for peer_key in (_normalized_match_text(value) for value in peer_subjects)
        ):
            continue
        if fragment_values & peer_values:
            continue
        return True
    return False


def _normalize_metric_claims(
    value: Any,
    table_evidence: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_claims = value if isinstance(value, list) else []
    row_lookup = _table_row_lookup(table_evidence)
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue
        claim_text = " ".join(str(raw.get("claim_text") or "").split())
        kind = str(raw.get("claim_kind") or "value").strip().lower()
        if kind not in {"value", "maximum", "minimum", "range", "comparison"}:
            kind = "value"
        try:
            value_column = int(raw.get("value_column") or 0)
        except (TypeError, ValueError):
            value_column = 0
        reason = ""
        raw_bindings = raw.get("row_bindings") if isinstance(raw.get("row_bindings"), list) else []
        normalized_bindings: list[dict[str, Any]] = []
        for binding in raw_bindings:
            if not isinstance(binding, dict):
                reason = "invalid_row_binding"
                break
            row_id = str(binding.get("table_evidence_unit_id") or "").strip()
            row = row_lookup.get(row_id)
            binding_subject = str(binding.get("subject") or "").strip()
            binding_values = _numeric_tokens(binding.get("value"))
            row_entities_value = binding.get("row_entities") if isinstance(binding.get("row_entities"), list) else []
            row_entities = list(dict.fromkeys([
                binding_subject,
                *(str(entity).strip() for entity in row_entities_value),
            ]))
            row_entities = [entity for entity in row_entities if entity]
            if not row or not binding_subject or not binding_values or value_column <= 0:
                reason = "missing_row_binding_fields"
                break
            row_key = _normalized_match_text(row.get("row_text"))
            if any(_normalized_match_text(entity) not in row_key for entity in row_entities):
                reason = "binding_entity_not_in_row"
                break
            if not binding_values.issubset(_numeric_tokens(_row_column_text(row, value_column))):
                reason = "binding_value_not_in_row_column"
                break
            normalized_bindings.append({
                "subject": binding_subject,
                "row_entities": row_entities,
                "value": sorted(binding_values)[0],
                "values": sorted(binding_values),
                "table_evidence_unit_id": row_id,
                "row_text": str(row.get("row_text") or ""),
            })

        value_numbers = {
            number
            for binding in normalized_bindings
            for number in binding.get("values") or []
        }
        binding_subjects = [str(binding.get("subject") or "") for binding in normalized_bindings]
        for binding in normalized_bindings:
            binding_values = set(binding.get("values") or [])
            if not _claim_supports_row_binding(
                claim_text,
                subject=str(binding.get("subject") or ""),
                values=binding_values,
                peer_subjects=set(binding_subjects) - {str(binding.get("subject") or "")},
                peer_values=value_numbers - binding_values,
            ):
                reason = "claim_text_row_binding_mismatch"
                break

        declared_values: set[str] = set()
        raw_values = raw.get("values") if isinstance(raw.get("values"), list) else []
        if raw.get("value") not in {None, ""}:
            raw_values = [raw.get("value"), *raw_values]
        for raw_value in raw_values:
            declared_values.update(_numeric_tokens(raw_value))
        if not reason and declared_values and declared_values != value_numbers:
            reason = "declared_values_do_not_match_bindings"

        subjects_value = raw.get("subjects") if isinstance(raw.get("subjects"), list) else []
        subjects = [str(subject).strip() for subject in subjects_value if str(subject).strip()]
        subject = str(raw.get("subject") or "").strip()
        if subject and subject not in subjects:
            subjects.insert(0, subject)
        if not subjects:
            subjects = list(dict.fromkeys(binding_subjects))

        row_ids = list(dict.fromkeys(
            str(binding.get("table_evidence_unit_id") or "")
            for binding in normalized_bindings
            if binding.get("table_evidence_unit_id")
        ))
        rows = [row_lookup[row_id] for row_id in row_ids]
        if not reason and (not claim_text or not rows or not subjects or not value_numbers):
            reason = "missing_claim_fields"
        row_texts = [str(row.get("row_text") or "") for row in rows]
        if not reason:
            for required_subject in subjects:
                subject_key = _normalized_match_text(required_subject)
                if (
                    not subject_key
                    or subject_key not in _normalized_match_text(claim_text)
                    or not any(subject_key in _normalized_match_text(text) for text in row_texts)
                ):
                    reason = "subject_not_in_cited_rows"
                    break
        if not reason and _numeric_tokens(claim_text) != value_numbers:
            reason = "claim_numbers_not_declared"

        if not reason and kind in {"maximum", "minimum", "range"}:
            if value_column <= 0 or len({str(row.get("bundle_id") or "") for row in rows}) != 1:
                reason = "aggregate_scope_missing"
            elif any(bool(row.get("rows_truncated")) for row in rows):
                reason = "aggregate_scope_incomplete"
            else:
                scope_values = raw.get("scope_entities") if isinstance(raw.get("scope_entities"), list) else []
                scope_entities = [str(entity).strip() for entity in scope_values if str(entity).strip()]
                scope_keys = [_normalized_match_text(entity) for entity in scope_entities]
                bundle_id = str(rows[0].get("bundle_id") or "")
                candidates: list[float] = []
                if not scope_keys:
                    reason = "aggregate_scope_missing"
                for candidate in row_lookup.values():
                    if reason:
                        break
                    if str(candidate.get("bundle_id") or "") != bundle_id:
                        continue
                    candidate_key = _normalized_match_text(candidate.get("row_text"))
                    if not all(scope_key in candidate_key for scope_key in scope_keys):
                        continue
                    for token in _numeric_tokens(_row_column_text(candidate, value_column)):
                        numeric = _numeric_value(token)
                        if numeric is not None:
                            candidates.append(numeric)
                claimed_values = sorted(
                    numeric for numeric in (_numeric_value(token) for token in value_numbers)
                    if numeric is not None
                )
                if not reason:
                    if len(candidates) < 2 or not claimed_values:
                        reason = "aggregate_values_missing"
                    elif kind == "maximum" and claimed_values[-1] != max(candidates):
                        reason = "maximum_mismatch"
                    elif kind == "minimum" and claimed_values[0] != min(candidates):
                        reason = "minimum_mismatch"
                    elif kind == "range" and (
                        len(claimed_values) < 2
                        or claimed_values[0] != min(candidates)
                        or claimed_values[-1] != max(candidates)
                    ):
                        reason = "range_mismatch"
        if not reason and kind == "comparison" and (
            len(set(binding_subjects)) < 2 or len(normalized_bindings) < 2
        ):
            reason = "comparison_scope_missing"

        if reason:
            rejected.append({
                "claim_text": claim_text,
                "reason": reason,
                "table_evidence_unit_ids": row_ids,
            })
            continue
        valid.append({
            "claim_text": _limit_claim_text(claim_text, 180),
            "subject": subject or subjects[0],
            "subjects": subjects,
            "context": _limit(raw.get("context") or "", 100),
            "metric": _limit(raw.get("metric") or "", 100),
            "values": sorted(value_numbers),
            "claim_kind": kind,
            "value_column": value_column,
            "scope_entities": [
                str(entity).strip()
                for entity in (raw.get("scope_entities") or [])
                if str(entity).strip()
            ] if isinstance(raw.get("scope_entities"), list) else [],
            "table_bundle_id": str(rows[0].get("bundle_id") or ""),
            "table_id": str(rows[0].get("table_id") or ""),
            "table_evidence_unit_ids": row_ids,
            "row_bindings": normalized_bindings,
            "row_texts": row_texts,
            "pages": sorted({int(row.get("page") or 1) for row in rows}),
        })
    return valid, rejected


def _sanitize_table_bound_text(
    text: Any,
    *,
    metric_claims: list[dict[str, Any]],
    table_evidence: list[dict[str, Any]] | None,
) -> tuple[str, list[str]]:
    value = " ".join(str(text or "").split())
    if not value or not table_evidence:
        return value, []
    table_numbers = _numeric_tokens(" ".join(
        str(row.get("row_text") or "")
        for bundle in table_evidence
        for row in bundle.get("rows") or []
    ))
    kept: list[str] = []
    rejected_numbers: list[str] = []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+", value)
        if part.strip()
    ] or [value]
    for sentence in sentences:
        sentence_numbers = _numeric_tokens(sentence)
        table_bound_numbers = sentence_numbers & table_numbers
        if not table_bound_numbers:
            kept.append(sentence)
            continue
        sentence_key = _normalized_match_text(sentence)
        matched = False
        for claim in metric_claims:
            claim_numbers = set(claim.get("values") or [])
            subjects = [str(subject) for subject in claim.get("subjects") or []]
            if (
                table_bound_numbers.issubset(claim_numbers)
                and subjects
                and all(_normalized_match_text(subject) in sentence_key for subject in subjects)
            ):
                matched = True
                break
        if matched:
            kept.append(sentence)
        else:
            rejected_numbers.extend(sorted(table_bound_numbers))
    return " ".join(kept), rejected_numbers


def _normalize_prose_claims(
    value: Any,
    *,
    summary: str,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind high-risk prose claims to an exact source quote and its semantic roles."""
    raw_claims = value if isinstance(value, list) else []
    block_lookup = {
        str(block.get("block_id") or ""): block
        for block in payload.get("blocks") or []
        if isinstance(block, dict) and block.get("block_id")
    }
    summary_key = _normalized_match_text(summary)
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue
        claim_text = " ".join(str(raw.get("claim_text") or "").split())
        subject = " ".join(str(raw.get("subject") or "").split())
        predicate = " ".join(str(raw.get("predicate") or "").split())
        qualifiers = [
            " ".join(str(value).split())
            for value in raw.get("qualifiers") or []
            if str(value).strip()
        ] if isinstance(raw.get("qualifiers"), list) else []
        source_subject = " ".join(str(raw.get("source_subject") or "").split())
        source_predicate = " ".join(str(raw.get("source_predicate") or "").split())
        source_qualifiers = [
            " ".join(str(value).split())
            for value in raw.get("source_qualifiers") or []
            if str(value).strip()
        ] if isinstance(raw.get("source_qualifiers"), list) else []
        block_id = str(raw.get("evidence_block_id") or "").strip()
        quote = " ".join(str(raw.get("evidence_quote") or "").split())
        block = block_lookup.get(block_id)
        reason = ""
        if not all((claim_text, subject, predicate, source_subject, source_predicate, block_id, quote)):
            reason = "missing_prose_claim_fields"
        elif _normalized_match_text(claim_text) not in summary_key:
            reason = "claim_text_not_in_summary"
        elif any(
            _normalized_match_text(value) not in _normalized_match_text(claim_text)
            for value in (subject, predicate, *qualifiers)
        ):
            reason = "summary_semantic_role_mismatch"
        elif not block or _normalized_match_text(quote) not in _normalized_match_text(block.get("text")):
            reason = "evidence_quote_not_in_block"
        elif any(
            _normalized_match_text(value) not in _normalized_match_text(quote)
            for value in (source_subject, source_predicate, *source_qualifiers)
        ):
            reason = "source_semantic_role_mismatch"
        else:
            claim_numbers = _numeric_tokens(claim_text)
            quote_numbers = _evidence_numeric_tokens(quote)
            if claim_numbers and not claim_numbers.issubset(quote_numbers):
                reason = "prose_claim_numbers_not_in_quote"
            else:
                missing_categories = [
                    name
                    for name, source_pattern, summary_pattern in PROSE_CLAIM_QUALIFIER_PATTERNS
                    if source_pattern.search(quote) and not summary_pattern.search(claim_text)
                ]
                if missing_categories:
                    reason = "semantic_qualifier_mismatch:" + ",".join(missing_categories)
        if reason:
            rejected.append({
                "claim_text": claim_text,
                "reason": reason,
                "evidence_block_id": block_id,
            })
            continue
        valid.append({
            "claim_text": _limit_claim_text(claim_text, 180),
            "claim_kind": str(raw.get("claim_kind") or "numeric"),
            "subject": _limit(subject, 100),
            "predicate": _limit(predicate, 100),
            "qualifiers": [_limit(value, 100) for value in qualifiers[:4]],
            "source_subject": _limit(source_subject, 120),
            "source_predicate": _limit(source_predicate, 120),
            "source_qualifiers": [_limit(value, 120) for value in source_qualifiers[:4]],
            "evidence_block_id": block_id,
            "evidence_quote": _limit(quote, 320),
            "values": sorted(_numeric_tokens(claim_text)),
        })
    return valid, rejected


def _sanitize_prose_bound_text(
    text: Any,
    *,
    prose_claims: list[dict[str, Any]],
    metric_claims: list[dict[str, Any]],
    rejected_claims: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    value = " ".join(str(text or "").split())
    if not value:
        return "", []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+", value)
        if part.strip()
    ] or [value]
    kept: list[str] = []
    violations: list[dict[str, Any]] = []
    for sentence in sentences:
        sentence_key = _normalized_match_text(sentence)
        sentence_numbers = _numeric_tokens(sentence)
        is_high_risk = bool(sentence_numbers) or bool(PROSE_CLAIM_RISK_PATTERN.search(sentence))
        if not is_high_risk:
            kept.append(sentence)
            continue
        table_supported = any(
            sentence_numbers
            and sentence_numbers.issubset(set(claim.get("values") or []))
            and all(
                _normalized_match_text(subject) in sentence_key
                for subject in claim.get("subjects") or []
            )
            for claim in metric_claims
        )
        prose_supported = any(
            (
                _normalized_match_text(claim.get("claim_text")) in sentence_key
                or sentence_key in _normalized_match_text(claim.get("claim_text"))
            )
            and sentence_numbers.issubset(set(claim.get("values") or []))
            for claim in prose_claims
        )
        if table_supported or prose_supported:
            kept.append(sentence)
            continue
        rejected = next(
            (
                claim for claim in rejected_claims
                if _normalized_match_text(claim.get("claim_text"))
                and (
                    _normalized_match_text(claim.get("claim_text")) in sentence_key
                    or sentence_key in _normalized_match_text(claim.get("claim_text"))
                )
            ),
            None,
        )
        if rejected is None:
            # Keep an unbound first-pass sentence long enough for the existing
            # numeric precheck to report missing evidence; repair or final
            # usability filtering will still prevent it from being published.
            kept.append(sentence)
        violations.append({
            "reason": str((rejected or {}).get("reason") or "unbound_prose_claim"),
            "claim_text": sentence,
            "numbers": sorted(sentence_numbers),
        })
    return " ".join(kept), violations


def _apply_table_claim_guard(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    guarded = dict(result)
    metric_claims, rejected_claims = _normalize_metric_claims(
        guarded.get("metric_claims"),
        payload.get("table_evidence") or [],
    )
    summary, rejected_numbers = _sanitize_table_bound_text(
        guarded.get("summary"),
        metric_claims=metric_claims,
        table_evidence=payload.get("table_evidence") or [],
    )
    prose_claims, rejected_prose_claims = _normalize_prose_claims(
        guarded.get("prose_claims"),
        summary=summary,
        payload=payload,
    )
    summary, prose_claim_violations = _sanitize_prose_bound_text(
        summary,
        prose_claims=prose_claims,
        metric_claims=metric_claims,
        rejected_claims=rejected_prose_claims,
    )
    guarded["summary"] = summary
    guarded["metric_claims"] = metric_claims
    guarded["prose_claims"] = prose_claims
    guarded["table_evidence_unit_ids"] = list(dict.fromkeys(
        evidence_unit_id
        for claim in metric_claims
        for evidence_unit_id in claim.get("table_evidence_unit_ids") or []
    ))
    violations = list(rejected_claims)
    if rejected_numbers:
        violations.append({"reason": "unbound_summary_numbers", "numbers": sorted(set(rejected_numbers))})
    guarded["table_claim_violations"] = violations
    guarded["claim_binding_violations"] = prose_claim_violations
    return guarded


def _canonical_brief_section_result(
    item: dict[str, Any],
    *,
    section_id: str,
    section_hash: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_ids = item.get("evidence_block_ids")
    if isinstance(evidence_ids, list) and payload:
        evidence_ids = list(dict.fromkeys([
            *(str(value) for value in evidence_ids if str(value).strip()),
            *(
                str(claim.get("evidence_block_id") or "")
                for claim in item.get("prose_claims") or []
                if isinstance(claim, dict) and str(claim.get("evidence_block_id") or "").strip()
            ),
            *(str(value) for value in payload.get("flow_spine_block_ids") or [] if str(value).strip()),
        ]))[:6]
    result = {
        "source_section_id": section_id,
        "translated_title": "",
        "summary": str(item.get("summary") or "").strip(),
        "study": {
            "purpose": "",
            "method_or_argument": "",
            "findings": [],
            "caveats_or_connections": "",
        },
        "evidence_block_ids": evidence_ids if isinstance(evidence_ids, list) else [],
        "metric_claims": item.get("metric_claims") if isinstance(item.get("metric_claims"), list) else [],
        "prose_claims": item.get("prose_claims") if isinstance(item.get("prose_claims"), list) else [],
        "table_evidence_unit_ids": item.get("table_evidence_unit_ids") if isinstance(item.get("table_evidence_unit_ids"), list) else [],
        "table_claim_violations": item.get("table_claim_violations") if isinstance(item.get("table_claim_violations"), list) else [],
        "claim_binding_violations": item.get("claim_binding_violations") if isinstance(item.get("claim_binding_violations"), list) else [],
        "section_hash": section_hash,
        "section_status": "ai",
    }
    return _apply_table_claim_guard(result, payload or {})


def _is_substantive_section_summary(value: Any) -> bool:
    summary = " ".join(str(value or "").split())
    return bool(summary) and summary not in {
        MISSING_SECTION_SUMMARY,
        LEGACY_MISSING_SECTION_SUMMARY,
    }


def _section_result_is_usable(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict) or not _is_substantive_section_summary(item.get("summary")):
        return False
    violations = item.get("table_claim_violations")
    claim_violations = item.get("claim_binding_violations")
    return not (
        (isinstance(violations, list) and violations)
        or (isinstance(claim_violations, list) and claim_violations)
    )


async def _generate_ai_outline(
    *,
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None = None,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    quality_feedback: list[str] | None = None,
    cached_outline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del quality_feedback
    skeleton = _build_reading_section_skeleton(block_index)
    payloads = _prepare_section_payloads(skeleton, block_index, structured_table_bundles)
    if not skeleton or not any(payload.get("blocks") for payload in payloads.values()):
        return {}
    cached_results = _cached_section_results(
        cached_outline,
        provider=provider,
        model=model,
    )
    results: dict[str, dict[str, Any]] = {}
    reused_count = 0
    pending: list[dict[str, Any]] = []
    for section_id, payload in payloads.items():
        if not payload.get("blocks"):
            continue
        cached = cached_results.get(section_id)
        if cached and cached.get("section_hash") == payload.get("section_hash"):
            results[section_id] = {**cached, "section_status": "cached"}
            reused_count += 1
        else:
            pending.append(payload)

    semaphore = asyncio.Semaphore(SECTION_BATCH_CONCURRENCY)

    async def generate_batch(batch: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        async with semaphore:
            warnings: list[str] = []
            try:
                raw = await _generate_section_batch(
                    doc_id=doc_id,
                    doc=doc,
                    sections=batch,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                )
            except Exception as exc:
                titles = ", ".join(str(item.get("title") or item.get("source_section_id")) for item in batch)
                warnings.append(f"章节批次生成失败（{titles}）：{exc}")
                raw = {"sections": []}

            generated: dict[str, dict[str, Any]] = {}
            for item in raw.get("sections") or []:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("source_section_id") or "").strip()
                payload = payloads.get(section_id)
                if not payload or section_id not in {str(section.get("source_section_id")) for section in batch}:
                    continue
                generated[section_id] = _canonical_brief_section_result(
                    item,
                    section_id=section_id,
                    section_hash=payload["section_hash"],
                    payload=payload,
                )

            missing = [item for item in batch if str(item.get("source_section_id")) not in generated]
            for section in missing:
                section_id = str(section.get("source_section_id"))
                try:
                    single = await _generate_section_batch(
                        doc_id=doc_id,
                        doc=doc,
                        sections=[section],
                        api_key=api_key,
                        model=model,
                        provider=provider,
                        endpoint=endpoint,
                    )
                    candidate = next(
                        (
                            value for value in single.get("sections") or []
                            if isinstance(value, dict)
                            and str(value.get("source_section_id") or "") == section_id
                        ),
                        None,
                    )
                    if candidate:
                        generated[section_id] = _canonical_brief_section_result(
                            candidate,
                            section_id=section_id,
                            section_hash=section["section_hash"],
                            payload=section,
                        )
                        continue
                except Exception as exc:
                    warnings.append(f"章节 {section.get('title') or section_id} 单独重试失败：{exc}")
                warnings.append(f"章节 {section.get('title') or section_id} 未返回有效结果，已局部回退")
            return generated, warnings

    batches = _pack_section_batches(pending)
    batch_results = await asyncio.gather(*(generate_batch(batch) for batch in batches)) if batches else []
    warnings: list[str] = []
    generated_count = 0
    for generated, batch_warnings in batch_results:
        results.update(generated)
        generated_count += len(generated)
        warnings.extend(batch_warnings)

    if pending and generated_count == 0 and reused_count == 0:
        reason = warnings[0] if warnings else "模型未返回任何可用章节总结"
        raise RuntimeError(reason)

    block_map_all = _flatten_blocks(block_index)
    initial_sampling_stats = _build_section_sampling_stats(
        payloads=payloads,
        results=results,
        block_map=block_map_all,
    )
    review_candidates = sorted(
        (
            item for item in initial_sampling_stats.get("sections") or []
            if item.get("review_recommended")
            and str(item.get("source_section_id") or "") in results
        ),
        key=lambda item: (
            int(item.get("total_blocks") or 0),
            -float(item.get("block_coverage_pct") or 0),
        ),
        reverse=True,
    )[:MAX_SECTION_REVIEW_SECTIONS]
    supplemental_reviewed_ids: list[str] = []
    supplemental_review_failures: list[str] = []
    for review_info in review_candidates:
        section_id = str(review_info.get("source_section_id") or "")
        payload = payloads.get(section_id)
        review_payload = _build_section_review_payload(payload or {}, block_map_all)
        if not section_id or not review_payload:
            continue
        try:
            review_raw = await _generate_section_batch(
                doc_id=doc_id,
                doc=doc,
                sections=[review_payload],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
            )
            review_item = next(
                (
                    item for item in review_raw.get("sections") or []
                    if isinstance(item, dict)
                    and str(item.get("source_section_id") or "") == section_id
                ),
                None,
            )
            if not review_item:
                raise RuntimeError("补读未返回目标章节")
            candidate = _canonical_brief_section_result(
                review_item,
                section_id=section_id,
                section_hash=str(payload.get("section_hash") or ""),
                payload=review_payload,
            )
            if not _section_result_is_usable(candidate):
                raise RuntimeError("补读结果未通过证据校验")
            candidate["repair_kind"] = "coverage_review"
            results[section_id] = candidate
            supplemental_reviewed_ids.append(section_id)
        except Exception as exc:
            supplemental_review_failures.append(section_id)
            warnings.append(
                f"章节 {payload.get('title') or section_id} 覆盖补读失败，保留首轮结果：{exc}"
            )

    # ── 数字预检：以「全章原文」为证据域抓编造数字的章节，带反馈只重试问题章节 ──

    def collect_missing_numbers(item: dict[str, Any], payload: dict[str, Any]) -> list[str]:
        evidence_text = " ".join(
            str(block_map_all.get(block_id, {}).get("text") or "")
            for block_id in payload.get("allowed_block_ids") or []
        )
        evidence_text = " ".join([
            evidence_text,
            *(
                str(row.get("row_text") or "")
                for bundle in payload.get("table_evidence") or []
                for row in bundle.get("rows") or []
            ),
        ])
        evidence_numbers = _evidence_numeric_tokens(evidence_text)
        texts: list[str] = [str(item.get("summary") or "")]
        study = item.get("study")
        if isinstance(study, dict):
            texts.extend(
                str(study.get(key) or "")
                for key in ("purpose", "method_or_argument", "caveats_or_connections")
            )
            findings = study.get("findings")
            if isinstance(findings, list):
                texts.extend(str(finding or "") for finding in findings)
        missing: set[str] = set()
        for text in texts:
            missing |= _numeric_tokens(text) - evidence_numbers
        return sorted(missing)

    numeric_feedback: dict[str, list[str]] = {}
    table_feedback: dict[str, list[str]] = {}
    flow_feedback: dict[str, list[str]] = {}
    claim_feedback: dict[str, list[str]] = {}
    for section_id, item in results.items():
        if item.get("section_status") != "ai":
            continue
        payload = payloads.get(section_id)
        if not payload:
            continue
        missing_numbers = collect_missing_numbers(item, payload)
        if missing_numbers:
            numeric_feedback[section_id] = missing_numbers
        violations = item.get("table_claim_violations") if isinstance(item.get("table_claim_violations"), list) else []
        reasons = list(dict.fromkeys(
            str(violation.get("reason") or "table_row_binding_failed")
            for violation in violations
            if isinstance(violation, dict)
        ))
        if reasons:
            table_feedback[section_id] = reasons
        claim_violations = (
            item.get("claim_binding_violations")
            if isinstance(item.get("claim_binding_violations"), list)
            else []
        )
        claim_reasons = list(dict.fromkeys(
            str(violation.get("reason") or "prose_claim_binding_failed")
            for violation in claim_violations
            if isinstance(violation, dict)
        ))
        if claim_reasons:
            claim_feedback[section_id] = claim_reasons
        expected_flow_labels = (
            payload.get("flow_labels") or []
            if payload.get("summary_role") == "method"
            else []
        )
        missing_flow_labels = _missing_flow_labels(item.get("summary"), expected_flow_labels)
        if missing_flow_labels:
            flow_feedback[section_id] = missing_flow_labels

    numeric_repaired = 0
    table_repaired = 0
    flow_repaired = 0
    claim_repaired = 0
    repair_ids = (
        set(numeric_feedback)
        | set(table_feedback)
        | set(flow_feedback)
        | set(claim_feedback)
    )
    if repair_ids:
        suspect_payloads = [
            payload for section_id, payload in payloads.items()
            if section_id in repair_ids
        ]
        for chunk in _pack_section_batches(suspect_payloads):
            chunk_ids = {
                str(payload.get("source_section_id"))
                for payload in chunk
            }
            chunk_numeric_feedback = {
                section_id: numeric_feedback[section_id]
                for section_id in chunk_ids
                if section_id in numeric_feedback
            }
            chunk_table_feedback = {
                section_id: table_feedback[section_id]
                for section_id in chunk_ids
                if section_id in table_feedback
            }
            chunk_flow_feedback = {
                section_id: flow_feedback[section_id]
                for section_id in chunk_ids
                if section_id in flow_feedback
            }
            chunk_claim_feedback = {
                section_id: claim_feedback[section_id]
                for section_id in chunk_ids
                if section_id in claim_feedback
            }
            try:
                repaired = await _generate_section_batch(
                    doc_id=doc_id,
                    doc=doc,
                    sections=chunk,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                    numeric_feedback=chunk_numeric_feedback or None,
                    table_feedback=chunk_table_feedback or None,
                    flow_feedback=chunk_flow_feedback or None,
                    claim_feedback=chunk_claim_feedback or None,
                )
            except Exception as exc:
                warnings.append(f"章节证据修复重试失败：{exc}")
                continue
            for item in repaired.get("sections") or []:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("source_section_id") or "").strip()
                payload = payloads.get(section_id)
                if not payload or section_id not in chunk_ids:
                    continue
                if collect_missing_numbers(item, payload):
                    continue  # 重试后仍有幻觉数字，保留旧结果交给 normalize 删句兜底
                candidate = _canonical_brief_section_result(
                    item,
                    section_id=section_id,
                    section_hash=payload["section_hash"],
                    payload=payload,
                )
                if not _section_result_is_usable(candidate):
                    continue
                expected_flow_labels = (
                    payload.get("flow_labels") or []
                    if payload.get("summary_role") == "method"
                    else []
                )
                if _missing_flow_labels(candidate.get("summary"), expected_flow_labels):
                    continue
                results[section_id] = candidate
                if section_id in numeric_feedback:
                    numeric_repaired += 1
                if section_id in table_feedback:
                    table_repaired += 1
                if section_id in flow_feedback:
                    flow_repaired += 1
                if section_id in claim_feedback:
                    claim_repaired += 1
        if numeric_repaired < len(numeric_feedback):
            warnings.append(
                f"数字修复：{len(numeric_feedback)} 章存在证据外数字，重试后修复 {numeric_repaired} 章"
            )
        if table_repaired < len(table_feedback):
            warnings.append(
                f"表格归属修复：{len(table_feedback)} 章存在行列绑定问题，重试后修复 {table_repaired} 章"
            )
        if flow_repaired < len(flow_feedback):
            warnings.append(
                f"流程修复：{len(flow_feedback)} 章缺少命名阶段，重试后修复 {flow_repaired} 章"
            )
        if claim_repaired < len(claim_feedback):
            warnings.append(
                f"正文 Claim 修复：{len(claim_feedback)} 章存在语义绑定问题，重试后修复 {claim_repaired} 章"
            )

    qualitative_suspect_ids = [
        section_id
        for section_id in set(table_feedback) | set(claim_feedback)
        if section_id in payloads and not _section_result_is_usable(results.get(section_id))
    ]
    qualitative_repaired = 0
    for section_id in qualitative_suspect_ids:
        payload = payloads[section_id]
        try:
            candidate = await _generate_single_qualitative_section(
                doc_id=doc_id,
                doc=doc,
                section=payload,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
            )
        except Exception as exc:
            warnings.append(
                f"章节 {payload.get('title') or section_id} 定性救援失败：{exc}"
            )
            continue
        if not _section_result_is_usable(candidate):
            warnings.append(
                f"章节 {payload.get('title') or section_id} 未生成可验证的定性提要，已局部回退"
            )
            continue
        results[section_id] = candidate
        qualitative_repaired += 1

    unusable_result_ids = [
        section_id
        for section_id, item in results.items()
        if not _section_result_is_usable(item)
    ]
    for section_id in unusable_result_ids:
        results.pop(section_id, None)
    if unusable_result_ids:
        warnings.append(
            f"最终校验：{len(unusable_result_ids)} 章未形成可用提要，已标记为局部回退"
        )

    raw = _build_section_outline_raw(
        skeleton=skeleton,
        payloads=payloads,
        results=results,
        block_index=block_index,
    )
    raw["title"] = _paper_title(doc, block_index)
    try:
        raw["overview"] = await _generate_learning_overview(
            doc_id=doc_id,
            doc=doc,
            items=raw.get("items") or [],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
        )
    except Exception as exc:
        warnings.append(f"论文要旨生成失败，已使用章节摘要合成：{exc}")
    raw["generation_warnings"] = warnings
    raw["generation_stats"] = {
        "model_returned_sections": generated_count,
        "generated_sections": sum(
            1 for item in results.values() if item.get("section_status") == "ai"
        ),
        "reused_sections": sum(
            1 for item in results.values() if item.get("section_status") == "cached"
        ),
        "usable_sections": len(results),
        "fallback_sections": sum(
            1 for section_id, payload in payloads.items()
            if payload.get("blocks") and section_id not in results
        ),
    }
    raw["sampling_stats"] = _build_section_sampling_stats(
        payloads=payloads,
        results=results,
        block_map=block_map_all,
    )
    raw["sampling_stats"]["supplemental_review"] = {
        "candidates": [
            str(item.get("source_section_id") or "")
            for item in review_candidates
            if str(item.get("source_section_id") or "")
        ],
        "reviewed_sections": supplemental_reviewed_ids,
        "failed_sections": supplemental_review_failures,
        "max_sections": MAX_SECTION_REVIEW_SECTIONS,
    }
    raw["numeric_repair_stats"] = {
        "suspect_sections": len(numeric_feedback),
        "repaired_sections": numeric_repaired,
    }
    raw["table_claim_repair_stats"] = {
        "table_suspect_sections": len(table_feedback),
        "table_repaired_sections": table_repaired,
    }
    raw["flow_repair_stats"] = {
        "suspect_sections": len(flow_feedback),
        "repaired_sections": flow_repaired,
    }
    raw["claim_binding_repair_stats"] = {
        "suspect_sections": len(claim_feedback),
        "repaired_sections": claim_repaired,
    }
    raw["qualitative_repair_stats"] = {
        "suspect_sections": len(qualitative_suspect_ids),
        "repaired_sections": qualitative_repaired,
    }
    return raw


def _paper_title(doc: dict[str, Any], block_index: dict[str, Any]) -> str:
    ordered = list(_flatten_blocks(block_index).values())
    title_block = _pick_title_block(ordered)
    return _clean_title(title_block.get("text") if title_block else "") or _filename_title(doc)


def _outline_candidates(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    from services.section_outline_service import (
        get_structural_outline_items,
        recover_section_outline_items,
    )

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(get_structural_outline_items(block_index)):
        if not isinstance(item, dict):
            continue
        title = _clean_title(item.get("title"))
        if not title:
            continue
        candidates.append({
            "source_section_id": str(item.get("section_id") or f"outline_{index + 1}"),
            "title": title,
            "level": max(1, min(4, int(item.get("level") or 1))),
            "page": max(1, int(item.get("page") or 1)),
            "first_block": str(item.get("first_block") or ""),
            "outline_source": str(item.get("source") or ""),
        })
    generic_titles = {
        "全文", "全文书签", "full text", "fulltext", "document", "article", "paper",
        "contents", "content",
    }
    only_placeholder = bool(candidates) and all(
        re.sub(r"\s+", " ", str(item.get("title") or "").lower()).strip() in generic_titles
        for item in candidates
    )
    if candidates and not only_placeholder:
        return _resolve_outline_candidate_anchors(candidates, block_index)

    # MinerU 的 content_list 可能把所有章节标题都降成普通 text，block_index
    # 因而只剩一个“全文”占位目录。复用章节导航的确定性恢复器，保证精读与
    # 导航共享同一组真实块锚点，而不是让 Abstract 吞掉整篇文档。
    recovered = recover_section_outline_items(block_index)
    if recovered:
        candidates = [
            {
                "source_section_id": str(item.get("section_id") or f"recovered_{index + 1}"),
                "title": _clean_title(item.get("title")),
                "level": max(1, min(4, int(item.get("level") or 1))),
                "page": max(1, int(item.get("page") or 1)),
                "first_block": str(item.get("first_block") or ""),
                "outline_source": str(item.get("source") or "heading_recovery"),
            }
            for index, item in enumerate(recovered)
            if _clean_title(item.get("title"))
        ]
        return _resolve_outline_candidate_anchors(candidates, block_index)

    for page in block_index.get("pages") or []:
        page_num = max(1, int(page.get("page") or 1))
        for block in page.get("blocks") or []:
            if not isinstance(block, dict) or block.get("type") != "heading":
                continue
            title = _clean_title(block.get("text"))
            if not title or _is_noise_text(title):
                continue
            candidates.append({
                "source_section_id": str(block.get("section_id") or block.get("block_id") or f"heading_{len(candidates) + 1}"),
                "title": title,
                "level": max(1, min(4, int(block.get("level") or 1))),
                "page": page_num,
                "first_block": str(block.get("block_id") or ""),
                "outline_source": "heading",
            })
    return _resolve_outline_candidate_anchors(candidates, block_index)


def _resolve_outline_candidate_anchors(
    candidates: list[dict[str, Any]],
    block_index: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    # The section-outline service already owns title-to-block matching for
    # bookmarks/TOCs whose page-level anchors point at the wrong text block.
    from services.section_outline_service import _resolve_section_anchor

    block_map = _flatten_blocks(block_index)
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        anchor = _resolve_section_anchor(
            title=str(candidate.get("title") or ""),
            requested_page=int(candidate.get("page") or 1),
            raw_block_id=str(candidate.get("first_block") or "") or None,
            evidence_ids=[],
            block_map=block_map,
        )
        resolved = dict(candidate)
        if anchor and anchor in block_map:
            resolved["first_block"] = anchor
            resolved["page"] = int(block_map[anchor].get("page") or resolved.get("page") or 1)
        result.append(resolved)
    return result


def _appendix_heading_level(title: str, *, allow_bare_letter: bool = False) -> int | None:
    normalized = " ".join(str(title or "").split())
    if re.match(r"^(?:appendix|appendices|supplementary\s+material)\b", normalized, re.IGNORECASE):
        match = re.match(
            r"^(?:appendix\s+)?[A-Z](?:\.(\d+(?:\.\d+)*))?(?:[.)]?\s+|$)",
            normalized,
        )
        return 1 + len(match.group(1).split(".")) if match and match.group(1) else 1
    if not allow_bare_letter:
        return None
    match = re.match(r"^[A-Z](?:\.(\d+(?:\.\d+)*))?(?:[.)]?\s+|$)", normalized)
    if not match:
        return None
    return 1 + len(match.group(1).split(".")) if match.group(1) else 1


def _build_level_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []
    for raw in nodes:
        node = {**raw, "children": []}
        level = max(1, int(node.get("level") or 1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            roots.append(node)
        stack.append((level, node))
    return roots


def _synthetic_page_sections(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    page_numbers = [
        max(1, int(page.get("page") or 1))
        for page in block_index.get("pages") or []
        if isinstance(page, dict)
    ]
    if not page_numbers:
        return []
    first_page, last_page = min(page_numbers), max(page_numbers)
    sections = []
    for start in range(first_page, last_page + 1, 6):
        end = min(last_page, start + 5)
        sections.append({
            "id": f"pages_{start}_{end}",
            "source_section_id": f"pages_{start}_{end}",
            "title": f"第 {start}-{end} 页" if start != end else f"第 {start} 页",
            "type": "section",
            "section_kind": "body",
            "level": 1,
            "page": start,
            "page_start": start,
            "page_end": end,
            "first_block": "",
            "children": [],
        })
    return sections


def _inline_abstract_node(
    block_map: dict[str, dict[str, Any]],
    body_nodes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if any(
        re.match(r"^(?:abstract|摘要)\b", str(node.get("title") or "").strip(), re.IGNORECASE)
        for node in body_nodes
    ):
        return None

    ordered = list(block_map.values())
    order_by_id = {
        str(block.get("block_id") or ""): order
        for order, block in enumerate(ordered)
        if block.get("block_id")
    }
    first_body_order = min(
        (
            order_by_id[str(node.get("first_block"))]
            for node in body_nodes
            if str(node.get("first_block") or "") in order_by_id
        ),
        default=len(ordered),
    )
    pattern = re.compile(r"^\s*(abstract|摘要)(?:\s*[.:：—-]\s*|\s+|$)", re.IGNORECASE)
    for order, block in enumerate(ordered):
        if order >= first_body_order:
            break
        if str(block.get("type") or "").lower() not in {"paragraph", "heading"}:
            continue
        match = pattern.match(str(block.get("text") or ""))
        block_id = str(block.get("block_id") or "")
        if not match or not block_id:
            continue
        chinese = match.group(1) == "摘要"
        node = {
            "id": f"section_inline_abstract_{_slug_id(block_id)}",
            "source_section_id": f"inline_abstract_{block_id}",
            "title": "摘要" if chinese else "Abstract",
            "type": "section",
            "section_kind": "body",
            "level": 1,
            "page": max(1, int(block.get("page") or 1)),
            "first_block": block_id,
            "outline_source": "inline_abstract",
        }
        if order > 0:
            previous = ordered[order - 1]
            if (
                str(previous.get("type") or "").lower() in {"caption", "visual_enrichment"}
                and _is_substantive_evidence_block(previous)
                and previous.get("block_id")
            ):
                node["content_start_block"] = str(previous["block_id"])
        return node
    return None


def _build_reading_section_skeleton(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _outline_candidates(block_index)
    if not candidates:
        return _synthetic_page_sections(block_index)

    block_map = _flatten_blocks(block_index)
    title_block = _pick_title_block(list(block_map.values()))
    title_block_id = str((title_block or {}).get("block_id") or "")
    body_nodes: list[dict[str, Any]] = []
    appendix_nodes: list[dict[str, Any]] = []
    references_seen = False
    mineru_appendix_mode = ""
    terminal_body_seen = False
    seen_sections: set[str] = set()
    for index, candidate in enumerate(candidates):
        title = str(candidate.get("title") or "").strip()
        source_section_id = str(candidate.get("source_section_id") or f"section_{index + 1}")
        if not title or source_section_id in seen_sections:
            continue
        if index == 0 and str(candidate.get("first_block") or "") == title_block_id:
            if not re.match(r"^(?:abstract|摘要)\b", title, re.IGNORECASE):
                continue
        if _is_reference_heading(title):
            references_seen = True
            continue
        if _is_acknowledgment_heading(title):
            continue
        if references_seen and _is_post_reference_template_artifact(title):
            continue

        anchor_text = str(
            block_map.get(str(candidate.get("first_block") or ""), {}).get("text") or ""
        ).strip()
        anchor_block = block_map.get(str(candidate.get("first_block") or ""), {})
        is_mineru_heading = str(anchor_block.get("source") or "").strip().lower() == "mineru_vlm"
        explicit_appendix_level = (
            _appendix_heading_level(title, allow_bare_letter=False)
            or _appendix_heading_level(anchor_text, allow_bare_letter=False)
        )
        bare_appendix_level = (
            _appendix_heading_level(title, allow_bare_letter=True)
            or _appendix_heading_level(anchor_text, allow_bare_letter=True)
        )
        is_bare_appendix_heading = (
            bare_appendix_level is not None
            and explicit_appendix_level is None
        )
        if references_seen and is_mineru_heading:
            if not mineru_appendix_mode:
                if explicit_appendix_level is not None:
                    mineru_appendix_mode = "explicit"
                elif is_bare_appendix_heading:
                    mineru_appendix_mode = "bare"
                else:
                    continue
            elif mineru_appendix_mode == "bare" and not is_bare_appendix_heading:
                continue
        mineru_explicit_appendix_child = is_mineru_heading and mineru_appendix_mode == "explicit"
        appendix_context = (
            references_seen
            or bool(appendix_nodes)
            or (terminal_body_seen and bare_appendix_level is not None)
        )
        appendix_level = (
            _appendix_heading_level(title, allow_bare_letter=appendix_context)
            or _appendix_heading_level(anchor_text, allow_bare_letter=appendix_context)
        )
        recovered_appendix_heading = bool(
            appendix_nodes
            and str(candidate.get("outline_source") or "") == "mineru_heading_recovery"
        )
        if (
            references_seen
            and appendix_level is None
            and not recovered_appendix_heading
            and not mineru_explicit_appendix_child
        ):
            continue
        section_kind = "appendix" if appendix_level is not None or appendix_context else "body"
        # The existing outline owns title order and hierarchy. Classification
        # only decides whether coverage is counted as body or appendix.
        level = int(candidate.get("level") or 1)
        if appendix_level is not None:
            level = appendix_level
        elif (recovered_appendix_heading or mineru_explicit_appendix_child) and level <= 1:
            level = 2
        node = {
            "id": f"section_{_slug_id(source_section_id) or index + 1}",
            "source_section_id": source_section_id,
            "title": title,
            "type": "appendix" if section_kind == "appendix" else "section",
            "section_kind": section_kind,
            "level": max(1, min(4, int(level or 1))),
            "page": max(1, int(candidate.get("page") or 1)),
            "first_block": str(candidate.get("first_block") or ""),
            "outline_source": str(candidate.get("outline_source") or ""),
        }
        seen_sections.add(source_section_id)
        (appendix_nodes if section_kind == "appendix" else body_nodes).append(node)
        if section_kind == "body" and re.match(
            r"^(?:\d+(?:\.\d+)*\s+|[IVXLCM]+\.\s+)?(?:conclusions?|结论)\b",
            title,
            re.IGNORECASE,
        ):
            terminal_body_seen = True

    inline_abstract = _inline_abstract_node(block_map, body_nodes)
    if inline_abstract:
        body_nodes.insert(0, inline_abstract)

    if not body_nodes and not appendix_nodes:
        return _synthetic_page_sections(block_index)

    items = _build_level_tree(body_nodes)
    appendix_tree = _build_level_tree(appendix_nodes)
    if appendix_tree:
        items.append({
            "id": "appendices",
            "source_section_id": "",
            "title": "附录",
            "type": "appendix_group",
            "section_kind": "appendix_group",
            "level": 1,
            "page": int(appendix_tree[0].get("page") or 1),
            "first_block": "",
            "children": appendix_tree,
        })
    return items


def _flatten_skeleton_nodes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node.get("source_section_id"):
                result.append(node)
            walk(node.get("children") or [])

    walk(items)
    return result


def _document_structured_table_bundles(doc: dict[str, Any]) -> list[dict[str, Any]]:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    candidates = data.get("structured_table_bundles") or doc.get("structured_table_bundles") or []
    return [bundle for bundle in candidates if isinstance(bundle, dict)]


def _compact_table_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    bundle_id = str(bundle.get("bundle_id") or bundle.get("table_bundle_id") or "").strip()
    table_id = str(bundle.get("table_id") or "").strip()
    caption = " ".join(str(bundle.get("table_caption") or bundle.get("caption") or "").split())
    header = " ".join(str(bundle.get("table_header") or bundle.get("header") or "").split())
    pages: set[int] = set()
    for raw_page in bundle.get("pages") or []:
        try:
            pages.add(max(1, int(raw_page)))
        except (TypeError, ValueError):
            continue
    for field in ("page_start", "page_end", "page", "page_index"):
        try:
            raw_value = bundle.get(field)
            if raw_value is not None:
                page = int(raw_value)
                if field == "page_index":
                    if pages or bundle.get("page_start") is not None:
                        continue
                    page += 1
                pages.add(max(1, page))
        except (TypeError, ValueError):
            continue

    rows: list[dict[str, Any]] = []
    for row_index, unit in enumerate(bundle.get("evidence_units") or []):
        if not isinstance(unit, dict):
            continue
        if str(unit.get("evidence_unit_type") or "").strip().lower() != "table_row":
            continue
        if bool(unit.get("is_header_row")):
            continue
        row_text = " ".join(str(
            unit.get("row_text") or unit.get("raw_row_text") or unit.get("content") or ""
        ).split())
        if not row_text:
            continue
        try:
            page = max(1, int(unit.get("page") or min(pages or {1})))
        except (TypeError, ValueError):
            page = min(pages or {1})
        pages.add(page)
        cells = []
        for cell_index, cell in enumerate(unit.get("cell_evidence_units") or []):
            if not isinstance(cell, dict):
                continue
            try:
                column = max(1, int(cell.get("col_idx") or cell.get("column_number") or cell_index + 1))
            except (TypeError, ValueError):
                column = cell_index + 1
            cell_text = " ".join(str(cell.get("cell_text") or cell.get("content") or "").split())
            cells.append({"column": column, "text": _limit(cell_text, 220)})
        evidence_unit_id = str(unit.get("evidence_unit_id") or "").strip()
        if not evidence_unit_id:
            evidence_unit_id = f"{bundle_id or table_id or 'table'}::row::{row_index + 1}"
        rows.append({
            "evidence_unit_id": evidence_unit_id,
            "table_bundle_id": bundle_id,
            "row_id": str(unit.get("row_id") or "").strip(),
            "row_text": _limit(row_text, MAX_TABLE_ROW_TEXT_CHARS),
            "page": page,
            "cells": sorted(cells, key=lambda cell: int(cell.get("column") or 0)),
        })

    if not rows and isinstance(bundle.get("rows"), list):
        for row_index, raw_row in enumerate(bundle.get("rows") or []):
            if not isinstance(raw_row, dict):
                continue
            row_text = " ".join(str(raw_row.get("row_text") or raw_row.get("content") or "").split())
            if not row_text:
                continue
            try:
                page = max(1, int(raw_row.get("page") or min(pages or {1})))
            except (TypeError, ValueError):
                page = min(pages or {1})
            pages.add(page)
            cells = []
            for cell_index, raw_cell in enumerate(raw_row.get("cells") or []):
                if not isinstance(raw_cell, dict):
                    continue
                try:
                    column = max(1, int(raw_cell.get("column") or raw_cell.get("col_idx") or cell_index + 1))
                except (TypeError, ValueError):
                    column = cell_index + 1
                cells.append({
                    "column": column,
                    "text": _limit(raw_cell.get("text") or raw_cell.get("cell_text") or "", 220),
                })
            rows.append({
                "evidence_unit_id": str(raw_row.get("evidence_unit_id") or f"{bundle_id or table_id or 'table'}::row::{row_index + 1}"),
                "table_bundle_id": bundle_id,
                "row_id": str(raw_row.get("row_id") or "").strip(),
                "row_text": _limit(row_text, MAX_TABLE_ROW_TEXT_CHARS),
                "page": page,
                "cells": sorted(cells, key=lambda cell: int(cell.get("column") or 0)),
            })

    if not rows:
        markdown = str(bundle.get("table_body_markdown") or bundle.get("table_markdown") or "")
        markdown_rows = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
        for row_index, line in enumerate(markdown_rows):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells or all(re.fullmatch(r"[-: ]+", cell or "-") for cell in cells):
                continue
            if row_index == 0 and header:
                continue
            row_text = " | ".join(cells)
            rows.append({
                "evidence_unit_id": f"{bundle_id or table_id or 'table'}::markdown_row::{row_index + 1}",
                "table_bundle_id": bundle_id,
                "row_id": cells[0] if cells else "",
                "row_text": _limit(row_text, MAX_TABLE_ROW_TEXT_CHARS),
                "page": min(pages or {1}),
                "cells": [
                    {"column": column + 1, "text": _limit(cell, 220)}
                    for column, cell in enumerate(cells)
                ],
            })

    return {
        "bundle_id": bundle_id,
        "table_id": table_id,
        "caption": caption,
        "header": header,
        "pages": sorted(pages or {1}),
        "row_count": len(rows),
        "rows": rows,
    }


def _table_bundle_fingerprint(structured_table_bundles: list[dict[str, Any]] | None) -> str:
    compact = [_compact_table_bundle(bundle) for bundle in structured_table_bundles or []]
    payload = [
        {
            "bundle_id": bundle.get("bundle_id"),
            "table_id": bundle.get("table_id"),
            "caption": bundle.get("caption"),
            "header": bundle.get("header"),
            "pages": bundle.get("pages"),
            "rows": [
                {
                    "evidence_unit_id": row.get("evidence_unit_id"),
                    "row_text": row.get("row_text"),
                    "cells": row.get("cells"),
                }
                for row in bundle.get("rows") or []
            ],
        }
        for bundle in compact
        if bundle.get("rows")
    ]
    if not payload:
        return ""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _normalized_match_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _bind_table_bundles_to_sections(
    *,
    nodes: list[dict[str, Any]],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    compact_bundles = [
        compact for compact in (_compact_table_bundle(bundle) for bundle in structured_table_bundles or [])
        if compact.get("rows")
    ]
    if not compact_bundles:
        return {}
    ordered_blocks = list(_flatten_blocks(block_index).values())
    caption_blocks = [
        block for block in ordered_blocks
        if str(block.get("section_id") or "").strip()
        and (
            str(block.get("type") or "").lower() == "caption"
            or (
                str(block.get("type") or "").lower() == "artifact"
                and re.search(r"(?:\btable\s*\d+|表\s*\d+)", str(block.get("text") or ""), re.IGNORECASE)
            )
        )
    ]
    known_section_ids = {str(node.get("source_section_id") or "") for node in nodes}
    result: dict[str, list[dict[str, Any]]] = {}

    for bundle in compact_bundles:
        bundle_pages = {int(page) for page in bundle.get("pages") or []}
        table_id_key = _normalized_match_text(bundle.get("table_id"))
        caption_key = _normalized_match_text(bundle.get("caption"))
        ranked: list[tuple[int, int, str]] = []
        for order, block in enumerate(caption_blocks):
            section_id = str(block.get("section_id") or "")
            if section_id not in known_section_ids:
                continue
            page = int(block.get("page") or 1)
            page_distance = min((abs(page - bundle_page) for bundle_page in bundle_pages), default=10**6)
            # Table numbers repeat between the body and appendix. A remote
            # "Table 2" must never beat the section that actually owns the page.
            if page_distance > 1:
                continue
            block_key = _normalized_match_text(block.get("text"))
            score = 30 if page_distance == 0 else 8
            if table_id_key and table_id_key in block_key:
                score += 8
            if caption_key and (caption_key in block_key or block_key in caption_key):
                score += 12
            elif caption_key and block_key:
                prefix = min(len(caption_key), len(block_key), 80)
                if prefix >= 24 and caption_key[:prefix] == block_key[:prefix]:
                    score += 6
            if score > 0:
                ranked.append((-score, order, section_id))
        ranked.sort()
        section_id = ranked[0][2] if ranked else ""
        if not section_id:
            page_sections: dict[str, tuple[int, int]] = {}
            for order, block in enumerate(ordered_blocks):
                if int(block.get("page") or 1) not in bundle_pages:
                    continue
                candidate_id = str(block.get("section_id") or "")
                if candidate_id not in known_section_ids:
                    continue
                count, _last_order = page_sections.get(candidate_id, (0, -1))
                page_sections[candidate_id] = (count + 1, order)
            if page_sections:
                section_id = max(
                    page_sections,
                    key=lambda candidate_id: (
                        page_sections[candidate_id][0],
                        page_sections[candidate_id][1],
                    ),
                )
        if not section_id:
            page_candidates = []
            for order, node in enumerate(nodes):
                start = int(node.get("page_start") or node.get("page") or 1)
                end = int(node.get("page_end") or start)
                if any(start <= page <= end for page in bundle_pages):
                    page_candidates.append((end - start, -int(node.get("level") or 1), order, str(node.get("source_section_id") or "")))
            if page_candidates:
                page_candidates.sort()
                section_id = page_candidates[0][3]
        if section_id:
            result.setdefault(section_id, []).append(bundle)
    return result


def _dynamic_section_block_limit(total_blocks: int) -> int:
    """长章节按比例放大采样上限：短章全量，长章 35% 采样，封顶 16 块。"""
    if total_blocks <= MAX_SECTION_PROMPT_BLOCKS:
        return total_blocks
    sampled = (total_blocks * SECTION_SAMPLING_RATIO_PCT + 99) // 100
    return max(MAX_SECTION_PROMPT_BLOCKS, min(MAX_SECTION_PROMPT_BLOCKS_LONG, sampled))


def _flow_signal_score(block: dict[str, Any]) -> int:
    text = " ".join(str(block.get("text") or "").replace("ﬁ", "fi").replace("ﬂ", "fl").split())
    if not FLOW_MARKER_PATTERN.search(text):
        return 0
    lower = text.lower()
    score = 4
    score += sum(
        2 for keyword in (
            "train", "optimize", "freeze", "update", "generator", "discriminator", "latent",
            "训练", "优化", "冻结", "更新", "生成器", "判别器", "潜变量", "噪声向量",
        )
        if keyword in lower
    )
    return score


def _motivation_signal_score(block: dict[str, Any]) -> int:
    text = " ".join(str(block.get("text") or "").split())
    lower = text.lower()
    score = 0
    score += 3 * sum(
        1 for keyword in (
            "however", "although", "but ", "limitation", "drawback", "fail", "insufficient",
            "existing", "previous", "prior", "state-of-the-art", "in contrast", "we propose",
            "然而", "但是", "尽管", "局限", "不足", "缺陷", "现有", "已有", "先前", "相比", "提出",
        )
        if keyword in lower
    )
    if re.search(r"\bet\s+al\.|\[[0-9,\- ]+\]|\([A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?,?\s*20\d{2}\)", text):
        score += 3
    # Motivation is usually narrative. Numbers remain usable evidence, but no
    # longer dominate the ranking simply because a paragraph contains metrics.
    score -= min(2, len(_numeric_tokens(text)) // 3)
    return score


def _named_flow_labels(blocks: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    patterns = (
        re.compile(r"\b(?:stage|step|phase)\s*(?:[-:]?\s*)?(?:\d+|[ivx]+|one|two|three)\b", re.IGNORECASE),
        re.compile(r"\b(?:first|second|third)\s+(?:stage|step|phase)\b", re.IGNORECASE),
        re.compile(r"第[一二三四五六七八九十\d]+(?:阶段|步)|阶段[一二三四五六七八九十\d]+"),
    )
    for block in blocks:
        text = " ".join(str(block.get("text") or "").replace("ﬁ", "fi").replace("ﬂ", "fl").split())
        for pattern in patterns:
            for match in pattern.finditer(text):
                label = " ".join(match.group(0).split())
                key = _normalized_match_text(label)
                if label and key not in {_normalized_match_text(value) for value in labels}:
                    labels.append(label)
    return labels


def _canonical_flow_label_key(label: str) -> str:
    lower = " ".join(str(label or "").lower().split())
    number_aliases = {
        "one": "1", "first": "1", "i": "1", "一": "1",
        "two": "2", "second": "2", "ii": "2", "二": "2",
        "three": "3", "third": "3", "iii": "3", "三": "3",
        "four": "4", "fourth": "4", "iv": "4", "四": "4",
    }
    for token, normalized in number_aliases.items():
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:$|[^a-z0-9])", lower):
            return f"flow:{normalized}"
        if token in "一二三四" and token in lower:
            return f"flow:{normalized}"
    match = re.search(r"\d+", lower)
    return f"flow:{int(match.group(0))}" if match else _normalized_match_text(lower)


def _missing_flow_labels(summary: Any, expected_labels: list[str] | None) -> list[str]:
    expected = [str(label) for label in expected_labels or [] if str(label).strip()]
    if not expected:
        return []
    observed = {
        _canonical_flow_label_key(label)
        for label in _named_flow_labels([{"text": str(summary or "")}])
    }
    return [label for label in expected if _canonical_flow_label_key(label) not in observed]


def _select_section_blocks(blocks: list[dict[str, Any]], section_title: str = "") -> list[dict[str, Any]]:
    limit = _dynamic_section_block_limit(len(blocks))
    if len(blocks) <= limit:
        return list(blocks)
    picked: list[dict[str, Any]] = []

    def add(block: dict[str, Any] | None) -> None:
        if block and block not in picked and len(picked) < limit:
            picked.append(block)

    flow_blocks = [block for block in blocks if _flow_signal_score(block) > 0]
    flow_quota = min(len(flow_blocks), max(2, limit // 3))
    for block in _evenly_spaced_blocks(flow_blocks, flow_quota):
        add(block)

    # A named stage often continues in the next paragraph. Keep that local
    # context before allocating the remaining generic sampling lanes.
    block_order = {id(block): index for index, block in enumerate(blocks)}
    for block in list(picked):
        index = block_order.get(id(block), -1)
        if 0 <= index < len(blocks) - 1:
            add(blocks[index + 1])

    # 采样预算随 limit 放大：叙述头部、角色信号块、均匀铺开三路配额同步伸缩。
    head_quota = max(2, limit // 4)
    signal_quota = max(2, limit // 4)
    spread_quota = max(3, limit // 2)

    narrative = [block for block in blocks if block.get("type") in {"paragraph", "visual_enrichment"}]
    signal_score = _motivation_signal_score if BACKGROUND_SECTION_TITLE_PATTERN.search(section_title) else _result_signal_score
    signal_ranked = sorted(blocks, key=signal_score, reverse=True)
    for block in narrative[:head_quota]:
        add(block)
    for block in signal_ranked[:signal_quota]:
        add(block)
    for block in _evenly_spaced_blocks(blocks, spread_quota):
        add(block)
    if narrative and len(picked) < limit:
        add(narrative[-1])
    for block in blocks:
        if len(picked) >= limit:
            break
        add(block)
    order = {str(block.get("block_id") or ""): index for index, block in enumerate(blocks)}
    picked.sort(key=lambda block: order.get(str(block.get("block_id") or ""), 10**9))
    return picked[:limit]


def _select_section_review_blocks(
    all_blocks: list[dict[str, Any]],
    selected_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose a bounded, dispersed second-read lane for a long section."""
    selected_ids = {
        str(block.get("block_id") or "")
        for block in selected_blocks
        if str(block.get("block_id") or "")
    }
    remaining = [
        block for block in all_blocks
        if str(block.get("block_id") or "") not in selected_ids
    ]
    if not remaining:
        return []
    picked: list[dict[str, Any]] = []

    def add(block: dict[str, Any] | None) -> None:
        if not block or block in picked or len(picked) >= MAX_SECTION_REVIEW_BLOCKS:
            return
        picked.append(block)

    # Preserve outlier findings, stage transitions and high-value result text,
    # then use a uniform lane so a long section's middle cannot disappear.
    ranked = sorted(
        remaining,
        key=lambda block: (
            _result_signal_score(block)
            + _flow_signal_score(block)
            + _motivation_signal_score(block)
        ),
        reverse=True,
    )
    for block in ranked[:max(3, MAX_SECTION_REVIEW_BLOCKS // 3)]:
        add(block)
    for block in _evenly_spaced_blocks(remaining, MAX_SECTION_REVIEW_BLOCKS):
        add(block)
    order = {str(block.get("block_id") or ""): index for index, block in enumerate(all_blocks)}
    picked.sort(key=lambda block: order.get(str(block.get("block_id") or ""), 10**9))
    return picked[:MAX_SECTION_REVIEW_BLOCKS]


def _section_hash(
    node: dict[str, Any],
    blocks: list[dict[str, Any]],
    table_evidence: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "source_section_id": node.get("source_section_id"),
        "title": node.get("title"),
        "section_kind": node.get("section_kind"),
        "detail_level": READING_OUTLINE_DETAIL_LEVEL,
        "prompt_version": READING_OUTLINE_PROMPT_VERSION,
        "blocks": [
            {
                "block_id": block.get("block_id"),
                "type": block.get("type"),
                "page": block.get("page"),
                "text": " ".join(str(block.get("text") or "").split()),
            }
            for block in blocks
        ],
        "table_evidence": table_evidence or [],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _prepare_section_payloads(
    skeleton: list[dict[str, Any]],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    block_map = _flatten_blocks(block_index)
    ordered = list(block_map.values())
    order_by_id = {str(block.get("block_id") or ""): index for index, block in enumerate(ordered)}
    nodes = _flatten_skeleton_nodes(skeleton)
    table_evidence_by_section = _bind_table_bundles_to_sections(
        nodes=nodes,
        block_index=block_index,
        structured_table_bundles=structured_table_bundles,
    )
    boundary_nodes = [*_outline_candidates(block_index), *nodes]
    boundary_orders = sorted({
        order_by_id[str(candidate.get("first_block"))]
        for candidate in boundary_nodes
        if str(candidate.get("first_block") or "") in order_by_id
    })
    payloads: dict[str, dict[str, Any]] = {}
    for node in nodes:
        section_id = str(node.get("source_section_id") or "")
        page_start = int(node.get("page_start") or node.get("page") or 1)
        page_end = int(node.get("page_end") or page_start)
        if section_id.startswith("pages_"):
            scope_blocks = [
                block for block in ordered
                if page_start <= int(block.get("page") or 1) <= page_end
            ]
        elif node.get("first_block") in order_by_id:
            anchor_order = order_by_id[str(node.get("first_block"))]
            include_anchor = str(node.get("outline_source") or "") == "inline_abstract"
            content_start = str(node.get("content_start_block") or "")
            start = (
                order_by_id[content_start]
                if content_start in order_by_id
                else anchor_order if include_anchor
                else anchor_order + 1
            )
            next_boundary = next((value for value in boundary_orders if value > anchor_order), len(ordered))
            scope_blocks = ordered[start:next_boundary]
        else:
            scope_blocks = [
                block for block in ordered
                if str(block.get("section_id") or "") == section_id
            ]
        all_blocks = [block for block in scope_blocks if _is_substantive_evidence_block(block)]
        keyword_blocks = [
            block for block in scope_blocks
            if classify_block_role(
                block,
                section_title=str(block.get("section_title") or node.get("title") or ""),
            )["role"] == ROLE_KEYWORDS
        ]
        if all_blocks:
            block_pages = [int(block.get("page") or page_start) for block in all_blocks]
            page_start = min(page_start, min(block_pages))
            page_end = max(page_end, max(block_pages))
        selected = _select_section_blocks(all_blocks, str(node.get("title") or ""))
        selected_flow_blocks = [block for block in selected if _flow_signal_score(block) > 0]
        selected_motivation_blocks = sorted(
            (block for block in selected if _motivation_signal_score(block) > 0),
            key=_motivation_signal_score,
            reverse=True,
        ) if BACKGROUND_SECTION_TITLE_PATTERN.search(str(node.get("title") or "")) else []
        section_title = str(node.get("title") or "")
        summary_role = (
            "background" if BACKGROUND_SECTION_TITLE_PATTERN.search(section_title)
            else "experiment" if EXPERIMENT_SECTION_TITLE_PATTERN.search(section_title)
            else "conclusion" if CONCLUSION_SECTION_TITLE_PATTERN.search(section_title)
            else "method" if FLOW_SECTION_TITLE_PATTERN.search(section_title) or selected_flow_blocks
            else "general"
        )
        table_evidence = []
        for bundle in (table_evidence_by_section.get(section_id) or [])[:MAX_SECTION_TABLE_BUNDLES]:
            all_rows = bundle.get("rows") or []
            table_evidence.append({
                "bundle_id": bundle.get("bundle_id"),
                "table_id": bundle.get("table_id"),
                "caption": bundle.get("caption"),
                "header": bundle.get("header"),
                "pages": bundle.get("pages") or [],
                "row_count": len(all_rows),
                "rows_truncated": len(all_rows) > MAX_SECTION_TABLE_ROWS,
                "rows": all_rows[:MAX_SECTION_TABLE_ROWS],
            })
        allowed_table_evidence_unit_ids = [
            str(row.get("evidence_unit_id") or "")
            for bundle in table_evidence
            for row in bundle.get("rows") or []
            if str(row.get("evidence_unit_id") or "")
        ]
        payloads[section_id] = {
            "source_section_id": section_id,
            "title": node.get("title"),
            "section_kind": node.get("section_kind"),
            "page": node.get("page"),
            "page_start": page_start,
            "page_end": page_end,
            "section_hash": _section_hash(node, all_blocks, table_evidence),
            "allowed_block_ids": [str(block.get("block_id")) for block in all_blocks if block.get("block_id")],
            "table_evidence": table_evidence,
            "allowed_table_evidence_unit_ids": allowed_table_evidence_unit_ids,
            "summary_role": summary_role,
            "flow_spine_block_ids": [
                str(block.get("block_id")) for block in selected_flow_blocks if block.get("block_id")
            ],
            "flow_labels": _named_flow_labels(selected_flow_blocks)[:4],
            "motivation_block_ids": [
                str(block.get("block_id")) for block in selected_motivation_blocks[:4] if block.get("block_id")
            ],
            "keyword_blocks": [
                {
                    "block_id": block.get("block_id"),
                    "page": block.get("page"),
                    "type": block.get("type") or "paragraph",
                    "text": _limit(block.get("text"), MAX_SECTION_BLOCK_TEXT),
                }
                for block in keyword_blocks
            ],
            "blocks": [
                {
                    "block_id": block.get("block_id"),
                    "page": block.get("page"),
                    "type": block.get("type") or "paragraph",
                    "text": _limit(block.get("text"), MAX_SECTION_BLOCK_TEXT),
                }
                for block in selected
            ],
        }
    return payloads


def _build_section_review_payload(
    payload: dict[str, Any],
    block_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Expand one sampled section with a bounded supplementary-read lane."""
    all_blocks = [
        block_map[block_id]
        for block_id in payload.get("allowed_block_ids") or []
        if block_id in block_map
    ]
    selected_ids = {
        str(block.get("block_id") or "")
        for block in payload.get("blocks") or []
        if isinstance(block, dict) and str(block.get("block_id") or "")
    }
    selected_blocks = [
        block for block in all_blocks
        if str(block.get("block_id") or "") in selected_ids
    ]
    supplementary = _select_section_review_blocks(all_blocks, selected_blocks)
    if not supplementary:
        return None
    merged = [*selected_blocks, *supplementary]
    order = {str(block.get("block_id") or ""): index for index, block in enumerate(all_blocks)}
    merged.sort(key=lambda block: order.get(str(block.get("block_id") or ""), 10**9))
    review_payload = dict(payload)
    review_payload["blocks"] = [
        {
            "block_id": block.get("block_id"),
            "page": block.get("page"),
            "type": block.get("type") or "paragraph",
            "text": _limit(block.get("text"), MAX_SECTION_BLOCK_TEXT),
        }
        for block in merged
    ]
    review_payload["review_pass"] = True
    review_payload["supplementary_block_ids"] = [
        str(block.get("block_id") or "")
        for block in supplementary
        if str(block.get("block_id") or "")
    ]
    return review_payload


def _cached_section_results(
    cached_outline: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
) -> dict[str, dict[str, Any]]:
    if not cached_outline or cached_outline.get("prompt_version") != READING_OUTLINE_PROMPT_VERSION:
        return {}
    if (cached_outline.get("meta") or {}).get("detail_level") != READING_OUTLINE_DETAIL_LEVEL:
        return {}
    if not _matches_ai_generation(cached_outline, provider=provider, model=model):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in cached_outline.get("flat_items") or _flatten_outline_items(cached_outline.get("items") or []):
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("source_section_id") or "")
        section_hash = str(item.get("section_hash") or "")
        if not section_id or not section_hash or item.get("type") == "overview":
            continue
        if str(item.get("section_status") or "") not in {"ai", "cached"}:
            continue
        if not _section_result_is_usable(item):
            continue
        result[section_id] = {
            "source_section_id": section_id,
            "translated_title": item.get("translated_title") or "",
            "summary": item.get("summary") or "",
            "study": item.get("study") or {},
            "evidence_block_ids": item.get("evidence_block_ids") or [],
            "metric_claims": item.get("metric_claims") or [],
            "prose_claims": item.get("prose_claims") or [],
            "table_evidence_unit_ids": item.get("table_evidence_unit_ids") or [],
            "table_claim_violations": item.get("table_claim_violations") or [],
            "claim_binding_violations": item.get("claim_binding_violations") or [],
            "section_hash": section_hash,
            "repair_kind": str(item.get("repair_kind") or ""),
        }
    return result


def _fallback_section_keyword_result(
    blocks: list[dict[str, Any]],
    section_title: str = "",
    limit: int = 4,
) -> dict[str, Any]:
    generic = {
        "paper", "section", "study", "method", "methods", "result", "results",
        "analysis", "approach", "experiment", "experiments", "abstract", "introduction",
        "we", "our", "us", "et", "al", "using", "use", "used", "based", "like",
        "given", "real", "shown", "table", "figure", "fig", "present", "presents",
        "provide", "provides", "propose", "proposes", "proposed", "introduce", "introduces",
        "verify", "evaluate", "school", "university", "information", "communications", "engineering",
        "keyword", "keywords", "index", "terms",
        "本文", "本节", "方法", "结果", "提出", "介绍", "使用", "实验", "分析",
    }
    title_terms = {
        str(keyword or "").strip().lower()
        for keyword in _fallback_keyword_extractor.extract_keywords(section_title)
        if str(keyword or "").strip()
    }
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    order: dict[str, int] = {}
    explicit_keywords: list[str] = []
    explicit_seen: set[str] = set()
    source_blocks = 0
    max_source_chars = 0
    for block in blocks:
        if (
            block_is_keyword_excluded(block, section_title=section_title)
            or _is_noise_text(str(block.get("text") or ""))
        ):
            continue
        text = " ".join(str(block.get("text") or "").split())
        if not text:
            continue
        source_blocks += 1
        max_source_chars = max(max_source_chars, len(text))
        role = classify_block_role(
            block,
            section_title=section_title,
        )["role"]
        if role == ROLE_KEYWORDS:
            tail = re.sub(
                r"^\s*(?:keywords?|index\s+terms?|关键词|关键字)\s*[:：]\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            for phrase in re.split(r"\s*[,，;；、·•|]\s*", tail):
                label = phrase.strip(" .:：")
                normalized = label.lower()
                if (
                    2 <= len(label) <= 28
                    and normalized not in generic
                    and normalized not in title_terms
                    and normalized not in explicit_seen
                ):
                    explicit_seen.add(normalized)
                    explicit_keywords.append(label)
        for keyword in _fallback_keyword_extractor.extract_keywords(text):
            label = str(keyword or "").strip()
            normalized = label.lower()
            if (
                not label
                or len(label) > 28
                or normalized in generic
                or normalized in title_terms
                or re.fullmatch(r"[-+]?\d[\d,.%]*", label)
            ):
                continue
            if normalized not in order:
                order[normalized] = len(order)
                labels[normalized] = label
            counts[normalized] = counts.get(normalized, 0) + 1
    ranked = sorted(counts, key=lambda value: (-counts[value], order[value]))
    confidence = 0.98 if explicit_keywords else 0.0
    if ranked and not explicit_keywords:
        confidence = 0.56 if max_source_chars >= 60 else 0.4
        if source_blocks >= 2:
            confidence = max(confidence, 0.7)
        if any(counts[value] >= 2 for value in ranked[:limit]):
            confidence = min(0.9, confidence + 0.12)
    keywords = explicit_keywords[:limit]
    if confidence >= 0.55 and not explicit_keywords and len(keywords) < limit:
        for value in ranked:
            normalized = labels[value].lower()
            if normalized in {item.lower() for item in keywords}:
                continue
            keywords.append(labels[value])
            if len(keywords) >= limit:
                break
    return {
        "keywords": keywords,
        "confidence": round(confidence, 2),
        "source_blocks": source_blocks,
    }


def _fallback_section_keywords(
    blocks: list[dict[str, Any]],
    section_title: str = "",
    limit: int = 4,
) -> list[str]:
    return _fallback_section_keyword_result(blocks, section_title, limit)["keywords"]


def _fallback_section_result(payload: dict[str, Any]) -> dict[str, Any]:
    blocks = payload.get("blocks") or []
    keyword_result = _fallback_section_keyword_result(
        [*(payload.get("keyword_blocks") or []), *blocks],
        str(payload.get("title") or ""),
    )
    evidence_blocks: list[dict[str, Any]] = []
    for block in (
        blocks[0] if blocks else None,
        max(blocks, key=_result_signal_score) if blocks else None,
        blocks[-1] if blocks else None,
    ):
        if block and block not in evidence_blocks:
            evidence_blocks.append(block)
    evidence_ids = [str(block.get("block_id")) for block in evidence_blocks if block.get("block_id")]
    return {
        "source_section_id": payload.get("source_section_id"),
        "translated_title": "",
        "summary": "",
        "keywords": keyword_result["keywords"],
        "keyword_confidence": keyword_result["confidence"],
        "keyword_source_blocks": keyword_result["source_blocks"],
        "study": {"purpose": "", "method_or_argument": "", "findings": [], "caveats_or_connections": ""},
        "evidence_block_ids": evidence_ids,
        "metric_claims": [],
        "prose_claims": [],
        "table_evidence_unit_ids": [],
        "table_claim_violations": [],
        "claim_binding_violations": [],
        "section_hash": payload.get("section_hash"),
        "section_status": "fallback" if blocks else "unavailable",
    }


def _build_section_outline_raw(
    *,
    skeleton: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    block_index: dict[str, Any],
) -> dict[str, Any]:
    del block_index

    def build(node: dict[str, Any]) -> dict[str, Any]:
        section_id = str(node.get("source_section_id") or "")
        payload = payloads.get(section_id) or {}
        result = results.get(section_id) or (_fallback_section_result(payload) if section_id else {})
        return {
            "id": node.get("id"),
            "type": node.get("type") or "section",
            "title": node.get("title"),
            "translated_title": result.get("translated_title") or "",
            "summary": result.get("summary") or "",
            "keywords": result.get("keywords") or [],
            "keyword_confidence": result.get("keyword_confidence") or 0.0,
            "keyword_source_blocks": result.get("keyword_source_blocks") or 0,
            "study": result.get("study") or {},
            "evidence_block_ids": result.get("evidence_block_ids") or [],
            "metric_claims": result.get("metric_claims") or [],
            "prose_claims": result.get("prose_claims") or [],
            "table_evidence_unit_ids": result.get("table_evidence_unit_ids") or [],
            "table_claim_violations": result.get("table_claim_violations") or [],
            "claim_binding_violations": result.get("claim_binding_violations") or [],
            "repair_kind": str(result.get("repair_kind") or ""),
            "summary_role": payload.get("summary_role") or "general",
            "flow_spine_block_ids": payload.get("flow_spine_block_ids") or [],
            "flow_labels": payload.get("flow_labels") or [],
            "source_section_id": section_id,
            "section_kind": node.get("section_kind") or "body",
            "section_hash": payload.get("section_hash") or "",
            "section_status": result.get("section_status") or ("group" if not section_id else "synthesized"),
            "section_page": node.get("page") or 1,
            "section_first_block": node.get("first_block") or "",
            "page_start": payload.get("page_start") or node.get("page") or 1,
            "page_end": payload.get("page_end") or node.get("page") or 1,
            "children": [build(child) for child in node.get("children") or []],
        }

    return {
        "contract": READING_OUTLINE_CONTRACT,
        "detail_level": READING_OUTLINE_DETAIL_LEVEL,
        "items": [build(item) for item in skeleton],
    }


def _structured_response_diagnostic(
    response: Any,
    *,
    content: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    parse_error: Exception | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {}
    message: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if isinstance(choice.get("message"), dict):
                message = choice["message"]
        if isinstance(response.get("usage"), dict):
            usage = response["usage"]

    reasoning = ""
    for candidate in (
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("thinking"),
        response.get("reasoning_content") if isinstance(response, dict) else None,
    ):
        if isinstance(candidate, str) and candidate:
            reasoning = candidate
            break

    completion_details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    error = parse_error.__cause__ if parse_error and parse_error.__cause__ else parse_error
    result = {
        "prompt_chars": sum(len(str(item.get("content") or "")) for item in messages),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_reason": str(choice.get("finish_reason") or response.get("finish_reason") or "")
        if isinstance(response, dict)
        else "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "max_tokens": max_tokens,
        "has_choices": bool(choice),
        "parse_error": type(error).__name__ if error else "",
    }
    for field in ("lineno", "colno", "pos"):
        value = getattr(error, field, None) if error else None
        if isinstance(value, int):
            result[f"parse_error_{field}"] = value
    return result


def _structured_request_custom_params(provider: str, model: str) -> dict[str, Any] | None:
    """Compatibility wrapper for the shared structured-output policy."""
    return structured_json_request_params(provider, model)


async def _call_structured_reading_model(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    max_tokens: int,
    diagnostic_label: str = "reading_outline",
) -> dict[str, Any]:
    custom_params = _structured_request_custom_params(provider, model)
    response = await call_ai_api(
        messages=messages,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=0.1,
        custom_params=custom_params,
        purpose="reading_outline",
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response.get("error"))
    content = _extract_content(response)
    initial_diagnostic: dict[str, Any] = {}
    try:
        return parse_json_object(content, allow_partial=False)
    except StructuredJSONError as exc:
        initial_diagnostic = _structured_response_diagnostic(
            response,
            content=content,
            messages=messages,
            max_tokens=max_tokens,
            parse_error=exc,
        )
        logger.warning(
            "[ReadingOutline] Invalid structured JSON label=%s provider=%s model=%s diagnostic=%s",
            diagnostic_label,
            provider,
            model,
            json.dumps(initial_diagnostic, ensure_ascii=False, separators=(",", ":")),
        )
    retry = await call_ai_api(
        messages=[
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "请修复为完整、严格合法的 JSON；不得改动 source_section_id 或编造 block_id。只输出 JSON。",
            },
        ],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=0,
        custom_params=custom_params,
        purpose="reading_outline",
    )
    if isinstance(retry, dict) and retry.get("error"):
        raise RuntimeError(retry.get("error"))
    retry_content = _extract_content(retry)
    try:
        return parse_json_object(retry_content, allow_partial=False)
    except StructuredJSONError as exc:
        retry_diagnostic = _structured_response_diagnostic(
            retry,
            content=retry_content,
            messages=[
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "请修复为完整、严格合法的 JSON；不得改动 source_section_id 或编造 block_id。只输出 JSON。",
                },
            ],
            max_tokens=max_tokens,
            parse_error=exc,
        )
        safe_diagnostic = {
            "label": diagnostic_label,
            "provider": provider,
            "model": model,
            "initial": initial_diagnostic,
            "repair": retry_diagnostic,
        }
        raise StructuredJSONError(
            "模型两次返回的逐章总结格式均不完整；diagnostic="
            + json.dumps(safe_diagnostic, ensure_ascii=False, separators=(",", ":"))
        ) from exc


async def _generate_section_batch(
    *,
    doc_id: str,
    doc: dict[str, Any],
    sections: list[dict[str, Any]],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
    numeric_feedback: dict[str, list[str]] | None = None,
    table_feedback: dict[str, list[str]] | None = None,
    flow_feedback: dict[str, list[str]] | None = None,
    claim_feedback: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    prompt_sections = [
        {
            "source_section_id": section.get("source_section_id"),
            "title": section.get("title"),
            "section_kind": section.get("section_kind"),
            "summary_role": section.get("summary_role") or "general",
            "page": section.get("page"),
            "blocks": section.get("blocks") or [],
            "table_evidence": section.get("table_evidence") or [],
            "flow_spine_block_ids": section.get("flow_spine_block_ids") or [],
            "flow_labels": section.get("flow_labels") or [],
            "motivation_block_ids": section.get("motivation_block_ids") or [],
            "review_pass": bool(section.get("review_pass")),
            "supplementary_block_ids": section.get("supplementary_block_ids") or [],
        }
        for section in sections
    ]
    system = (
        "你是严谨的学术章节提要助手。你只能填写给定章节，不能删除、重命名、合并或重排章节。"
        "所有事实只能来自相同 source_section_id 下提供的证据块；不得使用模型记忆。"
        "你的任务不是翻译，也不是重写论文，只需说明每个小节最值得记住的一点。"
    )
    user = (
        "请在现成大纲基础上，为每个给定小节生成一句简洁中文提要，只输出 JSON。\n"
        "输出格式：{\"sections\":[{\"source_section_id\":\"原样返回\","
        "\"summary\":\"一句话提要\",\"evidence_block_ids\":[\"1-3个直接正文证据ID\"],"
        "\"metric_claims\":[{\"claim_text\":\"与summary中完全一致的数字结论\","
        "\"subject\":\"数值所属方法或对象\",\"subjects\":[\"比较涉及的对象\"],"
        "\"context\":\"数据集/模型/条件\",\"metric\":\"指标和列\",\"value\":\"数值\","
        "\"values\":[\"区间或比较中的全部数值\"],\"claim_kind\":\"value|maximum|minimum|range|comparison\","
        "\"value_column\":1,\"scope_entities\":[\"最高/最低/区间的完整比较范围\"],"
        "\"row_bindings\":[{\"subject\":\"紧邻该数值的实体\",\"row_entities\":[\"该行中的模型/方法\"],"
        "\"value\":\"该实体对应的数值\",\"table_evidence_unit_id\":\"唯一表格行ID\"}]}],"
        "\"prose_claims\":[{\"claim_text\":\"summary中的完整高风险结论句\","
        "\"claim_kind\":\"numeric|comparison|causal|limitation\",\"subject\":\"摘要中的主语\","
        "\"predicate\":\"摘要中的关系或动作\",\"qualifiers\":[\"摘要中的条件和强度限定\"],"
        "\"source_subject\":\"原文短引文中的主语\",\"source_predicate\":\"原文短引文中的关系或动作\","
        "\"source_qualifiers\":[\"原文短引文中的条件和强度限定\"],\"evidence_block_id\":\"正文证据ID\","
        "\"evidence_quote\":\"证据块中连续出现的原文短引文\"}]}]}。\n"
        "summary 建议35-80个中文字，只保留研究问题、核心做法、关键结论或重要边界中最有价值的信息；"
        "不要翻译标题，不要逐句复述，不要输出多个栏目，不得使用‘本节介绍了/主要讨论了’等空泛开头。\n"
        "摘要/引言概括问题与贡献；相关工作概括差异；方法概括核心机制；实验概括最关键结果或设置；"
        "结论概括结论与限制；附录只说明它补充了什么。每节只写一个紧凑提要。\n"
        "流程规则：summary_role=method 且 flow_labels 非空时，提要必须保留原文的 Stage/Step/Phase 标签，"
        "在一句话内分别说明各阶段训练、优化、冻结或更新的对象，不得把一个阶段的对象写到另一个阶段。\n"
        "动机规则：引言/背景优先依据 motivation_block_ids 概括论文批评的主要既有方法缺陷；"
        "不要把某个组件的局部设计理由冒充全文研究动机。\n"
        "证据规则：数字、比较方向、因果和限定条件必须被所列证据逐字支持；不要引用 heading 作为事实证据；"
        "保留负面结果和适用边界；不得补写、拔高或跨章节借证据。\n"
        "表格规则：table_evidence 已按行保留实体与列关系。summary 中凡使用表格数字，必须同时返回 metric_claim；"
        "每个数值必须有一条 row_binding；subject 与 value 必须在同一短分句中，且同时存在于所引表格行，"
        "value_column 使用 cells 中的列号。比较结论要为双方分别绑定行，不得只列 subjects 和 values。"
        "maximum/minimum/range 只有在同一表格、同一subject、同一value_column 的候选行完整时才能使用；"
        "此时 scope_entities 必须明确给出用于筛选完整比较范围的行实体。"
        "comparison 必须引用双方表格行。无法确认归属就不要写数字、最高、领先或区间，改写为不含数字的稳健结论。\n"
        "正文 Claim 规则：非表格中的数字、比较、因果、最强/最弱、普遍性或能力边界结论必须返回 prose_claim；"
        "claim_text 必须逐字出现在 summary，subject、predicate、qualifiers 必须逐字出现在 claim_text；"
        "evidence_quote 必须是 evidence_block_id 对应正文中的连续原文，source_subject、source_predicate、"
        "source_qualifiers 必须逐字出现在该短引文。零样本、有监督适配、线性分类器、计算效率、分布偏移、"
        "百分点、否定和不确定性等限定条件必须在 claim_text 中保留；无高风险结论时返回空数组。\n"
        f"文档名：{doc.get('filename', doc_id)}\n"
        f"章节与证据：{json.dumps(prompt_sections, ensure_ascii=False)}"
    )
    if any(section.get("review_pass") for section in sections):
        user += (
            "\n补读规则：review_pass=true 的章节含有与首轮不同的分散补充证据。"
            "请重新通读该章节的全部给定 blocks，并让 summary 同时覆盖首轮主线与补充证据中的关键结果、限制或边界；"
            "不得只复述首轮开头段落，也不得因为补读而引入证据外结论。"
        )
    if numeric_feedback:
        feedback_lines = "；".join(
            f"{section_id}: {', '.join(numbers[:8])}"
            for section_id, numbers in numeric_feedback.items()
            if numbers
        )
        user += (
            "\n数字修复：上一版输出中以下数字在对应章节证据里逐字找不到——"
            f"{feedback_lines}。请重写这些章节，只使用证据块中逐字出现的数字；"
            "证据中没有的数字一律不要写，宁可省略数字也不要近似或换算。"
        )
    if table_feedback:
        feedback_lines = "；".join(
            f"{section_id}: {', '.join(reasons[:6])}"
            for section_id, reasons in table_feedback.items()
            if reasons
        )
        user += (
            "\n表格归属修复：上一版存在行列绑定失败——"
            f"{feedback_lines}。本轮这些章节不要再使用表格数值、百分比、最高/最低、领先或区间，"
            "metric_claims 返回空数组；只保留由正文或表格整体含义直接支持的不含数字定性结论。"
        )
    if flow_feedback:
        feedback_lines = "；".join(
            f"{section_id}: 缺少 {', '.join(labels[:4])}"
            for section_id, labels in flow_feedback.items()
            if labels
        )
        user += (
            "\n流程修复：上一版没有完整保留命名阶段——"
            f"{feedback_lines}。请在同一句中保留这些 Stage/Step/Phase 标签，并分别写清训练、冻结、优化或更新的对象；"
            "仍只输出一条紧凑提要，不要扩展成流程翻译。"
        )
    if claim_feedback:
        feedback_lines = "；".join(
            f"{section_id}: {', '.join(reasons[:6])}"
            for section_id, reasons in claim_feedback.items()
            if reasons
        )
        user += (
            "\n正文 Claim 修复：上一版的主语、谓词、限定条件或原文短引文绑定失败——"
            f"{feedback_lines}。请重写这些章节：能完整绑定就补齐 prose_claims；"
            "不能完整绑定就删掉该高风险结论，改写为证据直接支持且不含数字、不作强比较的定性提要。"
        )
    return await _call_structured_reading_model(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=_section_output_token_budget(sections),
        diagnostic_label="sections:" + ",".join(
            str(section.get("source_section_id") or "") for section in sections
        ),
    )


_ARABIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[%％])?"
)
_QUALITATIVE_RANKING_RE = re.compile(
    r"(?:最高|最低|领先|提升幅度|提高幅度|best|worst|highest|lowest|outperform(?:s|ed|ing)?)",
    re.IGNORECASE,
)


def _mask_qualitative_evidence_text(value: Any) -> str:
    return _ARABIC_NUMBER_RE.sub("[数值]", " ".join(str(value or "").split()))


def _qualitative_repair_blocks(section: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = [
        block
        for block in section.get("blocks") or []
        if isinstance(block, dict)
        and str(block.get("type") or "").strip().lower() not in {"table", "table_row", "table_cell"}
        and str(block.get("text") or "").strip()
    ]
    preferred = [
        block
        for block in blocks
        if str(block.get("type") or "paragraph").strip().lower()
        in {"paragraph", "text", "list", "caption"}
    ] or blocks
    result: list[dict[str, Any]] = []
    for block in preferred[:6]:
        masked = _mask_qualitative_evidence_text(block.get("text"))
        if not masked:
            continue
        result.append({
            "block_id": str(block.get("block_id") or ""),
            "type": str(block.get("type") or "paragraph"),
            "page": int(block.get("page") or section.get("page") or 1),
            "text": masked[:MAX_SECTION_BLOCK_TEXT],
        })
    return result


async def _generate_single_qualitative_section(
    *,
    doc_id: str,
    doc: dict[str, Any],
    section: dict[str, Any],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, Any] | None:
    """Recover one evidence-bearing section without reintroducing numeric claims."""
    section_id = str(section.get("source_section_id") or "").strip()
    blocks = _qualitative_repair_blocks(section)
    if not section_id or not blocks:
        return None
    allowed_ids = {str(block.get("block_id") or "") for block in blocks}
    messages = [
        {
            "role": "system",
            "content": (
                "你是严谨的学术章节定性修复助手。只能使用给定章节证据，"
                "不得恢复、猜测或比较被遮蔽的数值。"
            ),
        },
        {
            "role": "user",
            "content": (
                "只修复这一个章节，并只输出严格 JSON："
                "{\"source_section_id\":\"原样返回\",\"summary\":\"一句定性结论\","
                "\"evidence_block_ids\":[\"1-3个证据ID\"],\"metric_claims\":[]}。"
                "summary 必须非空、35-80个中文字，不得出现阿拉伯数字、百分比、区间、最高/最低、领先或提升幅度；"
                "不要写‘本节介绍了’等空话，只说明正文能够直接支持的机制、现象、权衡或边界。"
                f"\n文档名：{doc.get('filename', doc_id)}"
                f"\n章节：{json.dumps({'source_section_id': section_id, 'title': section.get('title'), 'blocks': blocks}, ensure_ascii=False)}"
            ),
        },
    ]
    raw = await _call_structured_reading_model(
        messages=messages,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=SECTION_OUTPUT_MIN_TOKENS,
        diagnostic_label=f"qualitative_section:{section_id}",
    )
    item = raw if isinstance(raw, dict) else {}
    if isinstance(item.get("sections"), list):
        item = next(
            (
                candidate
                for candidate in item["sections"]
                if isinstance(candidate, dict)
                and str(candidate.get("source_section_id") or "") == section_id
            ),
            {},
        )
    if str(item.get("source_section_id") or "") != section_id:
        return None
    summary = " ".join(str(item.get("summary") or "").split())
    if (
        not summary
        or _numeric_tokens(summary)
        or re.search(r"[%％]", summary)
        or _QUALITATIVE_RANKING_RE.search(summary)
    ):
        return None
    evidence_ids = [
        str(value)
        for value in item.get("evidence_block_ids") or []
        if str(value) in allowed_ids
    ][:3]
    if not evidence_ids:
        evidence_ids = [str(block.get("block_id") or "") for block in blocks[:3]]
    candidate = _canonical_brief_section_result(
        {
            "source_section_id": section_id,
            "summary": summary,
            "evidence_block_ids": evidence_ids,
            "metric_claims": [],
        },
        section_id=section_id,
        section_hash=str(section.get("section_hash") or ""),
        payload=section,
    )
    if not candidate.get("summary") or candidate.get("table_claim_violations"):
        return None
    candidate["repair_kind"] = "single_qualitative"
    return candidate


def _raw_section_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("source_section_id"):
            result.append(item)
        result.extend(_raw_section_items(item.get("children") or []))
    return result


def _build_overview_coverage_plan(section_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    sections = [
        item for item in section_summaries
        if item.get("source_section_id") and item.get("section_kind") in {"body", "appendix"}
    ]
    slot_candidates: dict[str, list[str]] = {}
    slot_themes: dict[str, str] = {}
    for slot, _label, theme, pattern in OVERVIEW_COVERAGE_SLOT_SPECS:
        ids: list[str] = []
        for item in sections:
            title = str(item.get("title") or "")
            role = str(item.get("summary_role") or "general")
            role_match = (
                (slot == "problem" and role == "background")
                or (slot == "mechanism" and role == "method")
                or (slot == "main_result" and role == "experiment")
                or (slot == "boundary" and role == "conclusion")
            )
            if role_match or pattern.search(title):
                section_id = str(item.get("source_section_id") or "")
                if section_id and section_id not in ids:
                    ids.append(section_id)
        slot_candidates[slot] = ids
        slot_themes[slot] = theme

    body_ids = [
        str(item.get("source_section_id") or "")
        for item in sections
        if item.get("section_kind") == "body"
    ]
    if not slot_candidates.get("problem") and body_ids:
        slot_candidates["problem"] = body_ids[:1]

    has_method = bool(slot_candidates.get("mechanism"))
    has_results = bool(slot_candidates.get("main_result"))
    has_theory = bool(slot_candidates.get("theory"))
    has_taxonomy = bool(slot_candidates.get("taxonomy"))
    has_data = bool(slot_candidates.get("data_or_setup"))
    if has_taxonomy and not has_results:
        paper_type = "survey_or_review"
    elif has_theory and not has_results:
        paper_type = "theoretical"
    elif has_data and has_results and not has_method:
        paper_type = "dataset_or_benchmark"
    elif has_method and has_results:
        paper_type = "empirical_method"
    elif has_results:
        paper_type = "empirical_study"
    else:
        paper_type = "conceptual_or_position"

    always_required = {"problem", "boundary"}
    type_required = {
        "survey_or_review": {"taxonomy"},
        "theoretical": {"theory"},
        "dataset_or_benchmark": {"data_or_setup", "main_result"},
        "empirical_method": {"mechanism", "main_result"},
        "empirical_study": {"main_result"},
        "conceptual_or_position": set(),
    }[paper_type]
    conditional_required = {"data_or_setup", "comparison", "ablation", "generalization"}
    slots: list[dict[str, Any]] = []
    labels = {slot: label for slot, label, _theme, _pattern in OVERVIEW_COVERAGE_SLOT_SPECS}
    for slot, _label, _theme, _pattern in OVERVIEW_COVERAGE_SLOT_SPECS:
        ids = slot_candidates.get(slot) or []
        if not ids:
            continue
        required = (
            slot in always_required
            or slot in type_required
            or slot in conditional_required
        )
        slots.append({
            "slot": slot,
            "label": labels[slot],
            "theme": slot_themes[slot],
            "required": required,
            "source_section_ids": ids[:8],
        })
    return {
        "paper_type": paper_type,
        "slots": slots,
    }


def _landmark_claim_category(title: str) -> str:
    value = str(title or "")
    for slot in ("ablation", "comparison", "generalization", "main_result"):
        pattern = next(
            spec[3] for spec in OVERVIEW_COVERAGE_SLOT_SPECS if spec[0] == slot
        )
        if pattern.search(value):
            return slot
    return "main_result"


def _select_landmark_claims(
    section_summaries: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in section_summaries:
        title = str(item.get("title") or "")
        role = str(item.get("summary_role") or "general")
        if role != "experiment" and not EXPERIMENT_SECTION_TITLE_PATTERN.search(title):
            continue
        category = _landmark_claim_category(title)
        claims = [
            claim for claim in [
                *(item.get("metric_claims") or []),
                *(item.get("prose_claims") or []),
            ]
            if isinstance(claim, dict) and claim.get("claim_text") and claim.get("values")
        ]
        for claim in claims:
            claim_text = " ".join(str(claim.get("claim_text") or "").split())
            dedupe_key = _normalized_match_text(claim_text)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            subjects = claim.get("subjects") if isinstance(claim.get("subjects"), list) else []
            if not subjects and claim.get("subject"):
                subjects = [claim.get("subject")]
            values = [str(value) for value in claim.get("values") or [] if str(value).strip()]
            score = (
                6
                + (4 if category == "main_result" else 3)
                + (2 if str(claim.get("claim_kind") or "") in {"comparison", "range", "maximum", "minimum"} else 0)
                + min(3, len(values))
            )
            claim_id = hashlib.sha1(
                f"{item.get('source_section_id')}|{claim_text}".encode("utf-8")
            ).hexdigest()[:12]
            candidates.append((score, {
                "claim_id": claim_id,
                "category": category,
                "source_section_id": str(item.get("source_section_id") or ""),
                "title": title,
                "claim_text": claim_text,
                "subjects": [str(subject) for subject in subjects if str(subject).strip()],
                "values": values,
                "pages": claim.get("pages") or [],
            }))
    candidates.sort(key=lambda entry: (-entry[0], entry[1]["source_section_id"]))
    selected: list[dict[str, Any]] = []
    selected_categories: set[str] = set()
    for _score, claim in candidates:
        if claim["category"] in selected_categories:
            continue
        selected.append(claim)
        selected_categories.add(claim["category"])
        if len(selected) >= limit:
            return selected
    for _score, claim in candidates:
        if claim in selected:
            continue
        selected.append(claim)
        if len(selected) >= limit:
            break
    return selected


def _raw_overview_requirement_issues(
    overview: dict[str, Any],
    coverage_plan: dict[str, Any],
    landmark_claims: list[dict[str, Any]],
) -> list[str]:
    themes = {
        str(theme.get("kind") or ""): theme
        for theme in overview.get("themes") or []
        if isinstance(theme, dict)
    }

    def theme_text(theme: dict[str, Any]) -> str:
        return " ".join([
            str(theme.get("summary") or ""),
            *(
                str(point.get("text") or "")
                for point in theme.get("points") or []
                if isinstance(point, dict)
            ),
        ])

    issues: list[str] = []
    for slot in coverage_plan.get("slots") or []:
        if not isinstance(slot, dict) or not slot.get("required"):
            continue
        theme = themes.get(str(slot.get("theme") or "")) or {}
        expected_ids = {str(value) for value in slot.get("source_section_ids") or []}
        actual_ids = {str(value) for value in theme.get("source_section_ids") or []}
        if not theme_text(theme).strip() or not (expected_ids & actual_ids):
            issues.append("slot:" + str(slot.get("slot") or ""))

    experiment_text = theme_text(themes.get("experiment") or {})
    experiment_key = _normalized_match_text(experiment_text)
    experiment_numbers = _numeric_tokens(experiment_text)
    for claim in landmark_claims:
        values = {str(value) for value in claim.get("values") or [] if str(value).strip()}
        subjects = [
            _normalized_match_text(subject)
            for subject in claim.get("subjects") or []
            if _normalized_match_text(subject)
        ]
        covered = bool(values) and values.issubset(experiment_numbers)
        if subjects:
            covered = covered and all(subject in experiment_key for subject in subjects)
        if not covered:
            issues.append("landmark:" + str(claim.get("claim_id") or ""))
    return issues


async def _generate_learning_overview(
    *,
    doc_id: str,
    doc: dict[str, Any],
    items: list[dict[str, Any]],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, Any]:
    section_summaries = [
        item for item in _raw_section_items(items)
        if item.get("section_kind") in {"body", "appendix"}
        and str(item.get("summary") or "").strip()
    ]
    if not section_summaries:
        return {}
    coverage_plan = _build_overview_coverage_plan(section_summaries)
    landmark_claims = _select_landmark_claims(section_summaries)
    digest = [
        {
            "source_section_id": item.get("source_section_id"),
            "title": item.get("title"),
            "section_kind": item.get("section_kind"),
            "summary": item.get("summary"),
            "summary_role": item.get("summary_role") or "general",
            "flow_labels": item.get("flow_labels") or [],
            "metric_claims": [
                {
                    "claim_text": claim.get("claim_text") or "",
                    "subject": claim.get("subject") or "",
                    "context": claim.get("context") or "",
                    "metric": claim.get("metric") or "",
                    "values": claim.get("values") or [],
                    "claim_kind": claim.get("claim_kind") or "value",
                }
                for claim in item.get("metric_claims") or []
                if isinstance(claim, dict)
            ],
            "prose_claims": [
                {
                    "claim_text": claim.get("claim_text") or "",
                    "subject": claim.get("subject") or "",
                    "predicate": claim.get("predicate") or "",
                    "qualifiers": claim.get("qualifiers") or [],
                    "values": claim.get("values") or [],
                }
                for claim in item.get("prose_claims") or []
                if isinstance(claim, dict)
            ],
            "pages": ((item.get("evidence") or {}).get("pages") or []),
        }
        for item in section_summaries
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是学术论文重点提炼助手。逐章摘要已经覆盖全文并完成章节内证据校验；"
                "你只能压缩和归类这些摘要，不得翻译原文、补充常识或增加输入中没有的事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                "把全文整理成适合快速学习的主题式重点导读，只输出严格 JSON："
                "{\"summary\":\"全文要旨\",\"source_section_ids\":[\"章节ID\"],\"themes\":["
                "{\"kind\":\"background\",\"summary\":\"背景与问题\",\"points\":[],\"source_section_ids\":[\"章节ID\"]},"
                "{\"kind\":\"innovation\",\"summary\":\"方法总括\",\"points\":[{\"label\":\"短标签\",\"text\":\"一句解释\"}],\"source_section_ids\":[\"章节ID\"]},"
                "{\"kind\":\"experiment\",\"summary\":\"结果总括\",\"points\":[{\"label\":\"短标签\",\"text\":\"一句结果\"}],\"source_section_ids\":[\"章节ID\"]},"
                "{\"kind\":\"conclusion\",\"summary\":\"结论、价值与边界\",\"points\":[],\"source_section_ids\":[\"章节ID\"]}]}。\n"
                "必须严格保留 background、innovation、experiment、conclusion 四组及该顺序，不得新增主题。"
                "全文要旨只写1句、40-90个中文字；背景与问题写1段、60-130字；"
                "核心方法与创新最多3点；关键实验结果最多5点，优先主结果、关键对比、消融、泛化/鲁棒性和真实场景验证；"
                "结论、价值与边界写1段、60-130字。每个 point 的 label 不超过14字，text 不超过80字。\n"
                "每个 point 必须使用 {label,text} 对象，text 是完整的结论句；禁止输出裸字符串、话题标签或省略号残句。\n"
                "不要按章节逐条复述，不要写研究流程之外的泛泛评价，不要把实验设置当成结果。"
                "innovation 必须覆盖所有 summary_role=method 且含 flow_labels 的章节，保留各 Stage/Step/Phase 标签，"
                "但合并为不超过3个机制重点。"
                "相似结果合并成一点；数字、比较方向、限制条件必须在所引用章节摘要中直接出现。"
                "逐章摘要中的 metric_claims 已通过表格行列校验；关键实验优先保留其中的主结果、基线比较和消融数字。"
                "prose_claims 已通过主语、谓词、限定条件和原文短引文校验，同一数值不得改换主语或省略成立条件。"
                "coverage_plan 根据本文实际章节生成；required=true 的槽位必须在对应主题中明确回答，"
                "不得用实验设置代替结果。landmark_claims 是优先级最高的已验证结果，关键实验主题应优先覆盖且保留数值、对象和条件。"
                "每组 source_section_ids 只列实际支撑该组内容的输入章节ID；附录只在包含关键补充实验或边界时纳入。\n"
                f"文档名：{doc.get('filename', doc_id)}"
                f"\ncoverage_plan：{json.dumps(coverage_plan, ensure_ascii=False)}"
                f"\nlandmark_claims：{json.dumps(landmark_claims, ensure_ascii=False)}"
                f"\n逐章摘要：{json.dumps(digest, ensure_ascii=False)}"
            ),
        },
    ]
    result = await _call_structured_reading_model(
        messages=messages,
        api_key=api_key,
        model=model,
        provider=provider,
        endpoint=endpoint,
        max_tokens=_overview_output_token_budget(digest),
        diagnostic_label="reading_overview",
    )
    requirement_issues = _raw_overview_requirement_issues(
        result,
        coverage_plan,
        landmark_claims,
    )
    if requirement_issues:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
            {
                "role": "user",
                "content": (
                    "上一版遗漏了以下本文确实存在的必答项："
                    + ", ".join(requirement_issues)
                    + "。请仍按原 JSON 合同完整重写一次，只补入逐章摘要和已验证 Claim 直接支持的内容；"
                    "landmark 必须同时保留数值、所属对象和限定条件，不得为凑覆盖而编造。"
                ),
            },
        ]
        repaired = await _call_structured_reading_model(
            messages=repair_messages,
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=_overview_output_token_budget(digest),
            diagnostic_label="reading_overview_coverage_repair",
        )
        repaired_issues = _raw_overview_requirement_issues(
            repaired,
            coverage_plan,
            landmark_claims,
        )
        if len(repaired_issues) < len(requirement_issues):
            result = repaired
    result["_coverage_plan"] = coverage_plan
    result["_landmark_claims"] = landmark_claims
    return result


def _numeric_tokens(text: Any) -> set[str]:
    tokens = set()
    for value in re.findall(
        r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*%)?(?![A-Za-z0-9.])",
        str(text or ""),
    ):
        normalized = re.sub(r"[,%\s]", "", value).lstrip("+")
        if normalized:
            tokens.add(normalized)
    return tokens


_ENGLISH_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def _evidence_numeric_tokens(text: Any) -> set[str]:
    """Accept exact English number words as evidence for digit-form claims."""
    value = str(text or "").lower()
    tokens = _numeric_tokens(value)
    for word, normalized in _ENGLISH_NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", value):
            tokens.add(normalized)
    return tokens


def _significant_numeric_tokens(blocks: list[dict[str, Any]]) -> set[str]:
    """Approximate result-bearing numbers without treating every section number as a finding."""
    result_terms = (
        "result", "outperform", "improve", "accuracy", "performance", "finding",
        "increase", "decrease", "significant", "结果", "提升", "下降", "准确率", "性能",
    )
    result_types = {"caption", "table", "visual_enrichment"}
    tokens: set[str] = set()
    pattern = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*%)?")
    for block in blocks:
        text = " ".join(str(block.get("text") or "").split())
        if not text:
            continue
        lower = text.lower()
        result_context = (
            str(block.get("type") or "").lower() in result_types
            or any(term in lower for term in result_terms)
        )
        for match in pattern.finditer(text):
            raw = match.group(0)
            if not result_context and "%" not in raw and "." not in raw:
                continue
            normalized = re.sub(r"[,%\s]", "", raw).lstrip("+")
            if normalized:
                tokens.add(normalized)
    return tokens


def _section_result_text(result: dict[str, Any] | None) -> str:
    value = result if isinstance(result, dict) else {}
    parts = [str(value.get("summary") or "")]
    study = value.get("study") if isinstance(value.get("study"), dict) else {}
    parts.extend(
        str(study.get(field) or "")
        for field in ("purpose", "method_or_argument", "caveats_or_connections")
    )
    parts.extend(str(item or "") for item in study.get("findings") or [])
    return " ".join(part for part in parts if part)


def _build_section_sampling_stats(
    *,
    payloads: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    block_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    total_blocks = 0
    selected_blocks = 0
    total_numbers = 0
    selected_numbers = 0
    output_numbers = 0

    for section_id, payload in payloads.items():
        allowed_ids = [str(value) for value in payload.get("allowed_block_ids") or [] if value]
        selected = [value for value in payload.get("blocks") or [] if isinstance(value, dict)]
        if not allowed_ids and not selected:
            continue
        full_blocks = [block_map[block_id] for block_id in allowed_ids if block_id in block_map]
        available_numeric = _significant_numeric_tokens(full_blocks)
        selected_numeric = _significant_numeric_tokens(selected) & available_numeric
        summary_numeric = _numeric_tokens(_section_result_text(results.get(section_id))) & available_numeric
        block_count = len(allowed_ids)
        selected_count = len(selected)
        block_coverage = round(selected_count * 100 / block_count, 1) if block_count else 100.0
        numeric_input_coverage = (
            round(len(selected_numeric) * 100 / len(available_numeric), 1)
            if available_numeric else None
        )
        numeric_summary_coverage = (
            round(len(summary_numeric) * 100 / len(available_numeric), 1)
            if available_numeric else None
        )
        sampled = selected_count < block_count
        review_recommended = sampled and (
            block_coverage < LONG_SECTION_REVIEW_COVERAGE_PCT
            or (
                numeric_input_coverage is not None
                and numeric_input_coverage < LONG_SECTION_REVIEW_COVERAGE_PCT
            )
        )
        sections.append({
            "source_section_id": section_id,
            "title": str(payload.get("title") or ""),
            "total_blocks": block_count,
            "selected_blocks": selected_count,
            "block_coverage_pct": block_coverage,
            "significant_numbers": len(available_numeric),
            "selected_significant_numbers": len(selected_numeric),
            "summary_significant_numbers": len(summary_numeric),
            "numeric_input_coverage_pct": numeric_input_coverage,
            "numeric_summary_coverage_pct": numeric_summary_coverage,
            "sampled": sampled,
            "review_recommended": review_recommended,
        })
        total_blocks += block_count
        selected_blocks += selected_count
        total_numbers += len(available_numeric)
        selected_numbers += len(selected_numeric)
        output_numbers += len(summary_numeric)

    sampled_sections = sum(1 for section in sections if section["sampled"])
    return {
        "sections_sampled": sampled_sections,
        "sections_full": len(sections) - sampled_sections,
        "review_recommended_sections": sum(1 for section in sections if section["review_recommended"]),
        "block_coverage_pct": round(selected_blocks * 100 / total_blocks, 1) if total_blocks else 100.0,
        "numeric_input_coverage_pct": (
            round(selected_numbers * 100 / total_numbers, 1) if total_numbers else None
        ),
        "numeric_summary_coverage_pct": (
            round(output_numbers * 100 / total_numbers, 1) if total_numbers else None
        ),
        "sections": sections,
    }


def _sanitize_numeric_claims(text: Any, evidence_text: str) -> tuple[str, list[str]]:
    value = " ".join(str(text or "").split())
    if not value:
        return "", []
    evidence_numbers = _evidence_numeric_tokens(evidence_text)
    kept: list[str] = []
    unsupported: list[str] = []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+", value)
        if part.strip()
    ]
    for sentence in sentences or [value]:
        missing = sorted(_numeric_tokens(sentence) - evidence_numbers)
        if missing:
            unsupported.extend(missing)
        else:
            kept.append(sentence)
    return " ".join(kept), unsupported


def _limit_claim_text(text: Any, max_len: int) -> str:
    """Compact at a sentence or clause boundary without displaying fragments."""
    value = " ".join(str(text or "").split())
    if len(value) <= max_len:
        return value

    terminal_pattern = re.compile(r"(?:[。！？；;]|(?<!\d)[.!?](?!\d))")
    before = [
        match.end() for match in terminal_pattern.finditer(value[:max_len])
        if match.end() >= max(18, int(max_len * 0.45))
    ]
    if before:
        return value[:before[-1]].strip()

    extension_end = min(len(value), max_len + max(16, min(48, max_len // 3)))
    after = [
        match.end() for match in terminal_pattern.finditer(value[:extension_end])
        if match.end() > max_len
    ]
    if after:
        return value[:after[0]].strip()

    raw_cut = value[:max_len]
    clause_matches = list(re.finditer(r"(?:[，；;]|(?<!\d),(?!\d))", raw_cut))
    clause_boundaries = [
        match.end() for match in clause_matches
        if match.end() >= max(18, int(max_len * 0.55))
    ]
    if clause_boundaries:
        return raw_cut[:clause_boundaries[-1]].rstrip(" ，,；;:：")

    word_boundaries = [
        match.start() for match in re.finditer(r"\s+", raw_cut)
        if match.start() >= max(18, int(max_len * 0.7))
    ]
    cut = raw_cut[:word_boundaries[-1]] if word_boundaries else raw_cut
    next_char = value[max_len:max_len + 1]
    dangling_number = re.search(r"(?<![A-Za-z0-9])[-+]?\d[\d,.]*%?$", cut)
    if dangling_number and next_char and next_char in "0123456789.,%":
        cut = cut[:dangling_number.start()]
    return cut.rstrip(" ，,；;:：")


def _sanitize_and_limit_claim(
    text: Any,
    evidence_text: str,
    max_len: int,
) -> tuple[str, list[str]]:
    """Validate numbers both before and after compacting a model claim."""
    cleaned, unsupported = _sanitize_numeric_claims(text, evidence_text)
    limited = _limit_claim_text(cleaned, max_len)
    limited, post_limit_unsupported = _sanitize_numeric_claims(limited, evidence_text)
    return limited, [*unsupported, *post_limit_unsupported]


def _normalize_study_fields(value: Any, evidence_text: str) -> tuple[dict[str, Any], list[str]]:
    raw = value if isinstance(value, dict) else {}
    unsupported: list[str] = []

    def clean(field: str, limit: int = MAX_STUDY_FIELD_CHARS) -> str:
        cleaned, missing = _sanitize_and_limit_claim(raw.get(field), evidence_text, limit)
        unsupported.extend(missing)
        return cleaned

    findings_value = raw.get("findings")
    if isinstance(findings_value, str):
        findings_value = [findings_value]
    findings: list[str] = []
    for finding in findings_value if isinstance(findings_value, list) else []:
        cleaned, missing = _sanitize_and_limit_claim(
            finding,
            evidence_text,
            MAX_FINDING_CHARS,
        )
        unsupported.extend(missing)
        if cleaned:
            if cleaned not in findings:
                findings.append(cleaned)
    return {
        "purpose": clean("purpose"),
        "method_or_argument": clean("method_or_argument"),
        "findings": findings[:MAX_SECTION_FINDINGS],
        "caveats_or_connections": clean("caveats_or_connections"),
    }, unsupported


def _normalize_section_keywords(value: Any) -> list[str]:
    values = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for entry in values:
        keyword = " ".join(str(entry or "").split())
        key = keyword.lower()
        if not keyword or key in seen:
            continue
        seen.add(key)
        result.append(_limit(keyword, 28))
        if len(result) >= 4:
            break
    return result


def _remove_redundant_study_fields(study: dict[str, Any], summary: str) -> dict[str, Any]:
    """Drop model repetitions while retaining genuinely additive study details."""
    summary_key = re.sub(r"\W+", "", str(summary or "")).lower()

    def is_redundant(value: Any) -> bool:
        value_key = re.sub(r"\W+", "", str(value or "")).lower()
        if not value_key or not summary_key:
            return False
        if value_key == summary_key:
            return True
        return min(len(value_key), len(summary_key)) >= 24 and (
            value_key in summary_key or summary_key in value_key
        )

    result = dict(study)
    for field in ("purpose", "method_or_argument", "caveats_or_connections"):
        if is_redundant(result.get(field)):
            result[field] = ""
    result["findings"] = [
        finding for finding in result.get("findings") or []
        if not is_redundant(finding)
    ][:MAX_SECTION_FINDINGS]
    return result


def _strip_missing_summary_prefix(value: Any) -> str:
    text = " ".join(str(value or "").split())
    for prefix in (LEGACY_MISSING_SECTION_SUMMARY, MISSING_SECTION_SUMMARY):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _repair_cached_summary_tree(outline: dict[str, Any]) -> bool:
    """Migrate cached parent summaries that absorbed unavailable child text."""
    items = outline.get("items") if isinstance(outline.get("items"), list) else []
    changed = False

    def visit(node: dict[str, Any]) -> str:
        nonlocal changed
        children = [child for child in node.get("children") or [] if isinstance(child, dict)]
        for child in children:
            visit(child)

        original = " ".join(str(node.get("summary") or "").split())
        cleaned = _strip_missing_summary_prefix(original)
        child_summaries = [
            _strip_missing_summary_prefix(child.get("summary"))
            for child in children
            if child.get("section_status") != "unavailable"
            and _strip_missing_summary_prefix(child.get("summary"))
        ]
        if children and (original.startswith(LEGACY_MISSING_SECTION_SUMMARY) or not cleaned):
            cleaned = _limit_claim_text(
                " ".join(child_summaries[:3]),
                MAX_SECTION_SUMMARY_CHARS,
            ) if child_summaries else MISSING_SECTION_SUMMARY
        elif not cleaned:
            cleaned = MISSING_SECTION_SUMMARY

        if cleaned != original:
            node["summary"] = cleaned
            changed = True
        if children and node.get("section_status") in {"", "group", "synthesized", "unavailable"}:
            next_status = "synthesized" if child_summaries else "unavailable"
            if node.get("section_status") != next_status:
                node["section_status"] = next_status
                changed = True
        return cleaned

    for item in items:
        if isinstance(item, dict):
            visit(item)
    if changed:
        outline["flat_items"] = _flatten_outline_items(items)
        meta = outline.setdefault("meta", {})
        meta["summary_repair"] = "removed_unavailable_child_placeholders"
    return changed


def _theme_candidate_section_ids(
    kind: str,
    normalized_sections: dict[str, dict[str, Any]],
) -> list[str]:
    hints = {
        "background": (
            "abstract", "introduction", "background", "motivation", "problem",
            "challenge", "related work", "prior work", "摘要", "引言", "背景", "问题", "挑战", "相关工作",
        ),
        "innovation": (
            "contribution", "method", "approach", "framework", "architecture",
            "model", "training", "algorithm", "pipeline", "贡献", "方法", "框架", "架构", "模型", "训练", "算法",
        ),
        "experiment": (
            "experiment", "evaluation", "result", "analysis", "ablation",
            "benchmark", "robustness", "generalization", "实验", "评估", "结果", "分析", "消融", "鲁棒", "泛化",
        ),
        "conclusion": (
            "conclusion", "discussion", "limitation", "future", "impact",
            "concluding", "结论", "讨论", "局限", "限制", "未来", "影响",
        ),
    }
    limits = {"background": 5, "innovation": 8, "experiment": 10, "conclusion": 6}
    ranked: list[tuple[int, int, str]] = []
    body_ids: list[str] = []
    for order, (section_id, item) in enumerate(normalized_sections.items()):
        if item.get("section_kind") not in {"body", "appendix"}:
            continue
        summary = str(item.get("summary") or "").strip()
        if not summary or summary in {MISSING_SECTION_SUMMARY, LEGACY_MISSING_SECTION_SUMMARY}:
            continue
        if item.get("section_kind") == "body":
            body_ids.append(section_id)
        title = str(item.get("title") or "").lower()
        category = _node_evidence_category(str(item.get("type") or ""), title)
        score = 50 if category == kind else 0
        if kind == "innovation" and item.get("summary_role") == "method":
            score += 80
        elif kind == "background" and item.get("summary_role") == "background":
            score += 60
        score += 8 * sum(1 for hint in hints.get(kind, ()) if hint in title)
        if item.get("section_kind") == "appendix":
            score -= 6 if kind in {"experiment", "conclusion"} else 18
        if score > 0:
            ranked.append((-score, order, section_id))
    ranked.sort()
    selected = [section_id for _score, _order, section_id in ranked[:limits.get(kind, 6)]]
    if selected:
        return selected
    if not body_ids:
        body_ids = [
            section_id for section_id, item in normalized_sections.items()
            if str(item.get("summary") or "").strip()
        ]
    if kind == "background":
        return body_ids[:2]
    if kind == "conclusion":
        return body_ids[-2:]
    if kind == "innovation":
        start = max(0, len(body_ids) // 3 - 1)
        return body_ids[start:start + min(4, len(body_ids))]
    start = max(0, (len(body_ids) * 2) // 3 - 1)
    return body_ids[start:start + min(5, len(body_ids))]


def _fallback_theme_payload(
    kind: str,
    normalized_sections: dict[str, dict[str, Any]],
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_ids = [
        section_id for section_id in (source_ids or [])
        if section_id in normalized_sections
        and _is_substantive_section_summary(normalized_sections[section_id].get("summary"))
        and normalized_sections[section_id].get("evidence_block_ids")
    ] or _theme_candidate_section_ids(kind, normalized_sections)
    source_items = [normalized_sections[section_id] for section_id in source_ids if section_id in normalized_sections]
    summaries = [
        str(item.get("summary") or "").strip()
        for item in source_items
        if str(item.get("summary") or "").strip()
        not in {MISSING_SECTION_SUMMARY, LEGACY_MISSING_SECTION_SUMMARY}
    ]
    point_limit = next((limit for name, _title, limit in THEME_SPECS if name == kind), 0)
    points = []
    if point_limit:
        for item in source_items[:point_limit]:
            text = str(item.get("summary") or "").strip()
            if not text or text in {MISSING_SECTION_SUMMARY, LEGACY_MISSING_SECTION_SUMMARY}:
                continue
            label = re.sub(r"^\s*(?:[A-Z]|\d+(?:\.\d+)*)[.)：:\-]?\s*", "", str(item.get("title") or "重点"), flags=re.IGNORECASE)
            points.append({"label": _limit(label, 28), "text": text})
    return {
        "kind": kind,
        "summary": " ".join(summaries[:2]),
        "points": points,
        "source_section_ids": source_ids,
    }


def _is_theme_summary_placeholder(value: Any) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE)
    return not normalized or normalized in THEME_SUMMARY_PLACEHOLDERS


def _is_complete_theme_point(label: Any, text: Any) -> bool:
    label_value = " ".join(str(label or "").split())
    text_value = " ".join(str(text or "").split())
    if not label_value or not text_value or re.search(r"(?:\.{3}|…|[:：/])\s*$", text_value):
        return False
    text_key = _normalized_match_text(text_value)
    return bool(text_key) and text_key != _normalized_match_text(label_value)


def _normalize_learning_themes(
    *,
    raw_overview: dict[str, Any],
    normalized_sections: dict[str, dict[str, Any]],
    block_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_themes = raw_overview.get("themes") if isinstance(raw_overview.get("themes"), list) else []
    by_kind = {
        str(theme.get("kind") or "").strip().lower(): theme
        for theme in raw_themes
        if isinstance(theme, dict)
    }
    unsupported_claims: list[dict[str, Any]] = []
    themes: list[dict[str, Any]] = []

    for kind, title, point_limit in THEME_SPECS:
        model_theme = by_kind.get(kind)
        raw_theme = model_theme or _fallback_theme_payload(kind, normalized_sections)
        requested_ids = raw_theme.get("source_section_ids") if isinstance(raw_theme.get("source_section_ids"), list) else []
        source_ids = list(dict.fromkeys(
            str(value) for value in requested_ids if str(value) in normalized_sections
        ))
        usable_source_ids = [
            section_id for section_id in source_ids
            if _is_substantive_section_summary(normalized_sections[section_id].get("summary"))
            and normalized_sections[section_id].get("evidence_block_ids")
        ]
        if usable_source_ids:
            source_ids = usable_source_ids
        else:
            source_ids = _theme_candidate_section_ids(kind, normalized_sections)
        if kind == "innovation":
            required_flow_sections = [
                section_id
                for section_id, item in normalized_sections.items()
                if item.get("summary_role") == "method" and item.get("flow_labels")
            ]
            source_ids = list(dict.fromkeys([*source_ids, *required_flow_sections]))

        evidence_ids: list[str] = []
        for section_id in source_ids:
            for block_id in normalized_sections[section_id].get("evidence_block_ids") or []:
                if block_id in block_map and block_id not in evidence_ids:
                    evidence_ids.append(block_id)
        evidence_ids = evidence_ids[:32]
        evidence_text = " ".join(
            str(block_map[block_id].get("text") or "")
            for block_id in evidence_ids
            if block_id in block_map
        )
        source_metric_claims = [
            claim
            for section_id in source_ids
            for claim in normalized_sections[section_id].get("metric_claims") or []
            if isinstance(claim, dict)
        ]
        table_row_texts = [
            row_text
            for claim in source_metric_claims
            for row_text in claim.get("row_texts") or []
        ]
        if table_row_texts:
            evidence_text = " ".join([evidence_text, *table_row_texts])
        synthetic_table_evidence = [{
            "rows": [
                {"row_text": row_text}
                for row_text in table_row_texts
            ],
        }] if table_row_texts else []

        raw_summary = raw_theme.get("summary")
        guarded_raw_summary, table_rejected_numbers = _sanitize_table_bound_text(
            "" if _is_theme_summary_placeholder(raw_summary) else raw_summary,
            metric_claims=source_metric_claims,
            table_evidence=synthetic_table_evidence,
        )
        summary, unsupported = _sanitize_and_limit_claim(
            guarded_raw_summary,
            evidence_text,
            MAX_THEME_SUMMARY_CHARS,
        )
        unsupported.extend(table_rejected_numbers)
        if not summary:
            fallback = _fallback_theme_payload(kind, normalized_sections, source_ids)
            summary, fallback_unsupported = _sanitize_and_limit_claim(
                fallback.get("summary"),
                evidence_text,
                MAX_THEME_SUMMARY_CHARS,
            )
            unsupported.extend(fallback_unsupported)

        raw_points = (
            raw_theme.get("points")
            if point_limit and isinstance(raw_theme.get("points"), list)
            else []
        )
        findings: list[str] = []
        for point in raw_points:
            # The model contract requires typed points. Bare strings are often
            # topic labels (for example "KITTI 泛化") rather than conclusions.
            if not isinstance(point, dict):
                continue
            raw_label = point.get("label")
            raw_text = point.get("text")
            label, label_unsupported = _sanitize_and_limit_claim(
                raw_label,
                evidence_text,
                MAX_THEME_POINT_LABEL_CHARS,
            )
            text, text_unsupported = _sanitize_and_limit_claim(
                raw_text,
                evidence_text,
                MAX_THEME_POINT_TEXT_CHARS,
            )
            unsupported.extend(label_unsupported)
            unsupported.extend(text_unsupported)
            if not _is_complete_theme_point(label, text):
                continue
            finding = f"{label}：{text}" if label and text else text
            if finding and finding not in findings:
                findings.append(finding)
            if len(findings) >= point_limit:
                break

        if unsupported:
            unsupported_claims.append({
                "source_section_id": f"theme_{kind}",
                "title": title,
                "numbers": sorted(set(unsupported)),
            })
        pages = sorted({
            int(block_map[block_id].get("page") or 1)
            for block_id in evidence_ids
            if block_id in block_map
        })
        page = pages[0] if pages else 1
        themes.append({
            "id": f"reading_theme_{kind}",
            "type": f"theme_{kind}",
            "title": title,
            "translated_title": "",
            "summary": summary,
            "study": {
                "purpose": "",
                "method_or_argument": "",
                "findings": findings,
                "caveats_or_connections": "",
                "learning_path": [],
            },
            "source_section_ids": source_ids,
            "source_section_id": "",
            "section_kind": "theme",
            "section_hash": "",
            "section_status": "ai" if model_theme else "synthesized",
            "evidence_scope": "cross_section",
            "evidence_block_ids": evidence_ids,
            "metric_claims": source_metric_claims,
            "table_evidence_unit_ids": list(dict.fromkeys(
                evidence_unit_id
                for claim in source_metric_claims
                for evidence_unit_id in claim.get("table_evidence_unit_ids") or []
            )),
            "evidence": {"block_ids": evidence_ids, "pages": pages, "primary_page": page},
            "page_start": pages[0] if pages else page,
            "page_end": pages[-1] if pages else page,
            "page": page,
            "first_block": evidence_ids[0] if evidence_ids else None,
            "children": [],
        })
    return themes, unsupported_claims


def _theme_audit_text(item: dict[str, Any]) -> str:
    study = item.get("study") if isinstance(item.get("study"), dict) else {}
    return " ".join([
        str(item.get("summary") or ""),
        *(str(value or "") for value in study.get("findings") or []),
    ])


def _audit_overview_coverage(
    raw_overview: dict[str, Any],
    theme_items: list[dict[str, Any]],
) -> dict[str, Any]:
    plan = raw_overview.get("_coverage_plan") if isinstance(raw_overview.get("_coverage_plan"), dict) else {}
    by_theme = {
        str(item.get("type") or "").removeprefix("theme_"): item
        for item in theme_items
        if isinstance(item, dict)
    }
    audited_slots: list[dict[str, Any]] = []
    for slot in plan.get("slots") or []:
        if not isinstance(slot, dict) or not slot.get("required"):
            continue
        theme = by_theme.get(str(slot.get("theme") or "")) or {}
        expected_ids = {
            str(value) for value in slot.get("source_section_ids") or [] if str(value).strip()
        }
        actual_ids = {
            str(value) for value in theme.get("source_section_ids") or [] if str(value).strip()
        }
        covered = bool(_theme_audit_text(theme).strip()) and bool(expected_ids & actual_ids)
        audited_slots.append({
            "slot": str(slot.get("slot") or ""),
            "label": str(slot.get("label") or ""),
            "theme": str(slot.get("theme") or ""),
            "covered": covered,
            "source_section_ids": sorted(expected_ids & actual_ids),
        })
    covered_count = sum(1 for slot in audited_slots if slot["covered"])
    return {
        "paper_type": str(plan.get("paper_type") or "unknown"),
        "required_slot_count": len(audited_slots),
        "covered_slot_count": covered_count,
        "coverage_ratio": covered_count / len(audited_slots) if audited_slots else 1.0,
        "missing_slots": [slot["slot"] for slot in audited_slots if not slot["covered"]],
        "slots": audited_slots,
    }


def _audit_landmark_claims(
    raw_overview: dict[str, Any],
    theme_items: list[dict[str, Any]],
) -> dict[str, Any]:
    claims = [
        claim for claim in raw_overview.get("_landmark_claims") or []
        if isinstance(claim, dict) and claim.get("claim_id")
    ]
    experiment = next(
        (item for item in theme_items if item.get("type") == "theme_experiment"),
        {},
    )
    text = _theme_audit_text(experiment)
    text_key = _normalized_match_text(text)
    text_numbers = _numeric_tokens(text)
    audited: list[dict[str, Any]] = []
    for claim in claims:
        values = {str(value) for value in claim.get("values") or [] if str(value).strip()}
        subjects = [
            _normalized_match_text(value)
            for value in claim.get("subjects") or []
            if _normalized_match_text(value)
        ]
        covered = bool(values) and values.issubset(text_numbers)
        if subjects:
            covered = covered and all(subject in text_key for subject in subjects)
        audited.append({
            "claim_id": str(claim.get("claim_id") or ""),
            "category": str(claim.get("category") or "main_result"),
            "source_section_id": str(claim.get("source_section_id") or ""),
            "values": sorted(values),
            "covered": covered,
        })
    covered_count = sum(1 for claim in audited if claim["covered"])
    return {
        "expected_claim_count": len(audited),
        "covered_claim_count": covered_count,
        "coverage_ratio": covered_count / len(audited) if audited else 1.0,
        "missing_claim_ids": [claim["claim_id"] for claim in audited if not claim["covered"]],
        "claims": audited,
    }


def _normalize_section_study_outline(
    *,
    raw: dict[str, Any],
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None,
    source_hash: str,
    source: str,
    model: str,
    provider: str,
    parse_generation: str,
    document_source_hash: str,
) -> dict[str, Any]:
    block_map = _flatten_blocks(block_index)
    is_ai_source = _is_ai_outline_source(source)
    skeleton = _build_reading_section_skeleton(block_index)
    payloads = _prepare_section_payloads(skeleton, block_index, structured_table_bundles)
    unsupported_numeric_claims: list[dict[str, Any]] = []
    table_claim_violations: list[dict[str, Any]] = []
    claim_binding_violations: list[dict[str, Any]] = []
    valid_metric_claim_count = 0
    valid_prose_claim_count = 0
    evidence_scope_violations = 0
    model_supported_sections = 0

    def evidence_in_scope(block_id: str, node: dict[str, Any]) -> bool:
        block = block_map.get(block_id)
        if not block or not _is_substantive_evidence_block(block):
            return False
        section_id = str(node.get("source_section_id") or "")
        allowed_ids = set((payloads.get(section_id) or {}).get("allowed_block_ids") or [])
        if allowed_ids:
            return block_id in allowed_ids
        if section_id.startswith("pages_"):
            page = int(block.get("page") or 1)
            return int(node.get("page_start") or 1) <= page <= int(node.get("page_end") or 1)
        return str(block.get("section_id") or "") == section_id

    def normalize_node(node: dict[str, Any]) -> dict[str, Any]:
        nonlocal evidence_scope_violations, model_supported_sections
        nonlocal valid_metric_claim_count, valid_prose_claim_count
        children = [normalize_node(child) for child in node.get("children") or [] if isinstance(child, dict)]
        section_id = str(node.get("source_section_id") or "")
        payload = payloads.get(section_id) or {}
        guarded_node = _apply_table_claim_guard(node, payload)
        metric_claims = guarded_node.get("metric_claims") or []
        prose_claims = guarded_node.get("prose_claims") or []
        valid_metric_claim_count += len(metric_claims)
        valid_prose_claim_count += len(prose_claims)
        for violation in guarded_node.get("table_claim_violations") or []:
            table_claim_violations.append({
                "source_section_id": section_id,
                "title": node.get("title") or "",
                **violation,
            })
        for violation in guarded_node.get("claim_binding_violations") or []:
            claim_binding_violations.append({
                "source_section_id": section_id,
                "title": node.get("title") or "",
                **violation,
            })
        node_type = str(node.get("type") or "section")
        raw_evidence = _valid_block_ids(guarded_node.get("evidence_block_ids"), block_map)
        evidence_ids = [block_id for block_id in raw_evidence if evidence_in_scope(block_id, node)]
        evidence_scope_violations += len(raw_evidence) - len(evidence_ids)
        if evidence_ids:
            model_supported_sections += 1
        if not evidence_ids and section_id:
            evidence_ids = [
                str(block.get("block_id"))
                for block in (payloads.get(section_id) or {}).get("blocks", [])[:4]
                if block.get("block_id") in block_map
            ]

        evidence_scope = "section"
        if not evidence_ids and children:
            evidence_scope = "descendants"
            for child in children:
                for block_id in child.get("evidence_block_ids") or []:
                    if block_id not in evidence_ids:
                        evidence_ids.append(block_id)

        # Claims must be validated against the exact evidence ids persisted in
        # the response, not a larger temporary descendant set.
        evidence_ids = evidence_ids[:10]
        evidence_text = " ".join(str(block_map[block_id].get("text") or "") for block_id in evidence_ids if block_id in block_map)
        table_claim_text = " ".join(
            row_text
            for claim in metric_claims
            for row_text in claim.get("row_texts") or []
        )
        evidence_text = " ".join(part for part in (evidence_text, table_claim_text) if part)
        study, unsupported = _normalize_study_fields(guarded_node.get("study"), evidence_text)
        summary, summary_unsupported = _sanitize_and_limit_claim(
            guarded_node.get("summary"),
            evidence_text,
            MAX_SECTION_SUMMARY_CHARS,
        )
        unsupported.extend(summary_unsupported)
        if not summary:
            parts = [study["purpose"], study["method_or_argument"], *study["findings"], study["caveats_or_connections"]]
            summary = " ".join(part for part in parts if part)
        if summary in {MISSING_SECTION_SUMMARY, LEGACY_MISSING_SECTION_SUMMARY}:
            summary = ""
        has_direct_substantive_summary = _is_substantive_section_summary(summary)
        has_substantive_child_summary = False
        if not summary and children:
            child_summaries = [
                str(child.get("summary") or "")
                for child in children
                if child.get("summary")
                and child.get("section_status") != "unavailable"
                and str(child.get("summary") or "") not in {
                    MISSING_SECTION_SUMMARY,
                    LEGACY_MISSING_SECTION_SUMMARY,
                }
            ]
            has_substantive_child_summary = bool(child_summaries)
            summary = " ".join(child_summaries[:2])
        if not summary and section_id:
            summary = MISSING_SECTION_SUMMARY
        summary, final_summary_unsupported = _sanitize_and_limit_claim(
            summary,
            evidence_text,
            MAX_SECTION_SUMMARY_CHARS,
        )
        unsupported.extend(final_summary_unsupported)
        study = _remove_redundant_study_fields(study, summary)
        if unsupported:
            unsupported_numeric_claims.append({
                "source_section_id": section_id,
                "title": node.get("title") or "",
                "numbers": sorted(set(unsupported)),
            })

        first_block = evidence_ids[0] if evidence_ids else str(node.get("section_first_block") or "")
        pages = sorted({int(block_map[block_id].get("page") or 1) for block_id in evidence_ids if block_id in block_map})
        primary_page = int(block_map.get(first_block, {}).get("page") or node.get("section_page") or 1)
        status = str(node.get("section_status") or "")
        has_final_substantive_summary = _is_substantive_section_summary(summary)
        if children and has_substantive_child_summary and not has_direct_substantive_summary:
            status = "synthesized"
        elif section_id and not has_final_substantive_summary:
            status = "fallback" if evidence_ids else "unavailable"
        elif children and status in {"", "group", "synthesized", "fallback", "unavailable"}:
            status = "synthesized" if has_substantive_child_summary else "unavailable"
        try:
            keyword_confidence = float(node.get("keyword_confidence") or 0.0)
        except (TypeError, ValueError):
            keyword_confidence = 0.0
        try:
            keyword_source_blocks = max(0, int(node.get("keyword_source_blocks") or 0))
        except (TypeError, ValueError):
            keyword_source_blocks = 0
        return {
            "id": str(node.get("id") or f"section_{_slug_id(section_id)}"),
            "type": node_type,
            "title": _limit(node.get("title") or "阅读章节", 180),
            "translated_title": _limit(node.get("translated_title") or "", 160),
            "summary": summary,
            "keywords": _normalize_section_keywords(node.get("keywords")),
            "keyword_confidence": round(max(0.0, min(1.0, keyword_confidence)), 2),
            "keyword_source_blocks": keyword_source_blocks,
            "study": study,
            "source_section_id": section_id,
            "section_kind": str(node.get("section_kind") or "body"),
            "section_hash": str(node.get("section_hash") or ""),
            "section_status": status or "fallback",
            "summary_role": str(node.get("summary_role") or "general"),
            "flow_spine_block_ids": [
                str(value) for value in node.get("flow_spine_block_ids") or [] if str(value).strip()
            ],
            "flow_labels": [str(value) for value in node.get("flow_labels") or [] if str(value).strip()],
            "section_page": int(node.get("section_page") or 1),
            "section_first_block": str(node.get("section_first_block") or ""),
            "page_start": int(node.get("page_start") or node.get("section_page") or 1),
            "page_end": int(node.get("page_end") or node.get("section_page") or 1),
            "evidence_scope": evidence_scope,
            "evidence_block_ids": evidence_ids,
            "metric_claims": metric_claims,
            "prose_claims": prose_claims,
            "table_evidence_unit_ids": guarded_node.get("table_evidence_unit_ids") or [],
            "table_claim_violations": guarded_node.get("table_claim_violations") or [],
            "claim_binding_violations": guarded_node.get("claim_binding_violations") or [],
            "repair_kind": str(node.get("repair_kind") or ""),
            "evidence": {"block_ids": evidence_ids, "pages": pages, "primary_page": primary_page},
            "page": primary_page,
            "first_block": first_block or None,
            "children": children,
        }

    section_items = [normalize_node(item) for item in raw.get("items") or [] if isinstance(item, dict)]
    normalized_sections = {
        str(item.get("source_section_id")): item
        for item in _flatten_outline_items(section_items)
        if item.get("source_section_id")
    }

    raw_overview = raw.get("overview") if isinstance(raw.get("overview"), dict) else {}
    requested_ids = raw_overview.get("source_section_ids") if isinstance(raw_overview.get("source_section_ids"), list) else []
    for raw_theme in raw_overview.get("themes") if isinstance(raw_overview.get("themes"), list) else []:
        if isinstance(raw_theme, dict) and isinstance(raw_theme.get("source_section_ids"), list):
            requested_ids = [*requested_ids, *raw_theme["source_section_ids"]]
    overview_section_ids = [str(value) for value in requested_ids if str(value) in normalized_sections]
    overview_section_ids = list(dict.fromkeys(overview_section_ids))
    if len(overview_section_ids) < 2:
        preferred = [
            section_id for section_id, item in normalized_sections.items()
            if item.get("section_kind") == "body"
            and re.search(r"abstract|introduction|conclusion|摘要|引言|结论", str(item.get("title") or ""), re.IGNORECASE)
        ]
        overview_section_ids = list(dict.fromkeys([*overview_section_ids, *preferred]))[:4]
    if len(overview_section_ids) < 2:
        overview_section_ids = [
            section_id for section_id, item in normalized_sections.items()
            if item.get("section_kind") == "body"
        ][:4]
    overview_evidence: list[str] = []
    for section_id in overview_section_ids:
        for block_id in normalized_sections[section_id].get("evidence_block_ids") or []:
            if block_id not in overview_evidence:
                overview_evidence.append(block_id)
    overview_evidence = overview_evidence[:16]
    overview_evidence_text = " ".join(str(block_map[block_id].get("text") or "") for block_id in overview_evidence if block_id in block_map)
    overview_summary, overview_unsupported = _sanitize_and_limit_claim(
        "" if _is_theme_summary_placeholder(raw_overview.get("summary")) else raw_overview.get("summary"),
        overview_evidence_text,
        MAX_OVERVIEW_SUMMARY_CHARS,
    )
    if not overview_summary:
        if is_ai_source:
            overview_summary = " ".join(
                str(normalized_sections[section_id].get("summary") or "")
                for section_id in overview_section_ids[:3]
            )
        else:
            overview_summary = "当前仅展示原文章节结构和页码范围；配置模型后可生成中文快速精读。"
    overview_summary, final_overview_unsupported = _sanitize_and_limit_claim(
        overview_summary,
        overview_evidence_text,
        MAX_OVERVIEW_SUMMARY_CHARS,
    )
    overview_unsupported.extend(final_overview_unsupported)
    if overview_unsupported:
        unsupported_numeric_claims.append({
            "source_section_id": "overview",
            "title": "论文要旨",
            "numbers": sorted(set(overview_unsupported)),
        })

    overview_pages = sorted({int(block_map[block_id].get("page") or 1) for block_id in overview_evidence if block_id in block_map})
    overview_page = overview_pages[0] if overview_pages else 1
    overview_item = {
        "id": "paper_overview",
        "type": "overview",
        "title": "论文要旨" if is_ai_source else "结构导览",
        "translated_title": "",
        "summary": overview_summary,
        "study": {
            "purpose": "",
            "method_or_argument": "",
            "findings": [],
            "caveats_or_connections": "",
            "learning_path": [],
        },
        "source_section_ids": overview_section_ids,
        "source_section_id": "",
        "section_kind": "overview",
        "section_hash": "",
        "section_status": "ai" if raw_overview else "synthesized",
        "evidence_scope": "cross_section",
        "evidence_block_ids": overview_evidence,
        "evidence": {"block_ids": overview_evidence, "pages": overview_pages, "primary_page": overview_page},
        "page": overview_page,
        "first_block": overview_evidence[0] if overview_evidence else None,
        "children": [],
    }
    theme_items, theme_unsupported_claims = _normalize_learning_themes(
        raw_overview=raw_overview,
        normalized_sections=normalized_sections,
        block_map=block_map,
    )
    unsupported_numeric_claims.extend(theme_unsupported_claims)
    overview_coverage = _audit_overview_coverage(raw_overview, theme_items)
    landmark_result_coverage = _audit_landmark_claims(raw_overview, theme_items)
    items = [overview_item, *theme_items] if is_ai_source else [overview_item, *section_items]
    display_flat_items = _flatten_outline_items(items)
    flat_section_items = _flatten_outline_items(section_items)
    flat_items = [*display_flat_items, *flat_section_items] if is_ai_source else display_flat_items
    section_nodes = [item for item in flat_section_items if item.get("source_section_id")]
    body_nodes = [item for item in section_nodes if item.get("section_kind") == "body"]
    appendix_nodes = [item for item in section_nodes if item.get("section_kind") == "appendix"]

    def countable(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in nodes if item.get("evidence_block_ids")]

    countable_body_nodes = countable(body_nodes)
    countable_appendix_nodes = countable(appendix_nodes)

    def is_covered(item: dict[str, Any]) -> bool:
        return (
            _is_substantive_section_summary(item.get("summary"))
            and bool(item.get("evidence_block_ids"))
            and item.get("section_status") != "unavailable"
        )

    def covered(nodes: list[dict[str, Any]]) -> int:
        return sum(1 for item in nodes if is_covered(item))

    body_summarized = covered(countable_body_nodes)
    appendix_summarized = covered(countable_appendix_nodes)

    section_coverage = {
        "body_expected": len(countable_body_nodes),
        "body_summarized": body_summarized,
        "body_ratio": body_summarized / len(countable_body_nodes) if countable_body_nodes else 1.0,
        "appendix_expected": len(countable_appendix_nodes),
        "appendix_summarized": appendix_summarized,
        "appendix_ratio": (
            appendix_summarized / len(countable_appendix_nodes)
            if countable_appendix_nodes
            else 1.0
        ),
    }
    raw_generation_stats = raw.get("generation_stats") if isinstance(raw.get("generation_stats"), dict) else {}
    countable_section_nodes = [*countable_body_nodes, *countable_appendix_nodes]
    covered_section_nodes = [item for item in countable_section_nodes if is_covered(item)]

    def stat_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    generation_stats = {
        **raw_generation_stats,
        "model_returned_sections": stat_int(
            raw_generation_stats.get(
                "model_returned_sections",
                raw_generation_stats.get("generated_sections"),
            )
        ),
        "generated_sections": sum(
            1 for item in covered_section_nodes if item.get("section_status") == "ai"
        ),
        "reused_sections": sum(
            1 for item in covered_section_nodes if item.get("section_status") == "cached"
        ),
        "usable_sections": len(covered_section_nodes),
        "fallback_sections": len(countable_section_nodes) - len(covered_section_nodes),
    }
    return {
        "version": READING_OUTLINE_VERSION,
        "doc_id": doc_id,
        "source": source,
        "source_hash": source_hash,
        "prompt_version": READING_OUTLINE_PROMPT_VERSION,
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
        "title": _limit(raw.get("title") or _filename_title(doc), 180),
        "items": items,
        "flat_items": flat_items,
        "created_at": time.time(),
        "model": model,
        "provider": provider,
        "section_items": section_items,
        "meta": {
            "contract": READING_OUTLINE_CONTRACT,
            "detail_level": READING_OUTLINE_DETAIL_LEVEL,
            "block_count": len(block_map),
            "page_count": len(block_index.get("pages") or []),
            "section_coverage": section_coverage,
            "generation": generation_stats,
            "generation_warnings": [str(value) for value in raw.get("generation_warnings") or []],
            "unsupported_numeric_claims": unsupported_numeric_claims,
            "sampling": raw.get("sampling_stats") or {},
            "numeric_repair": raw.get("numeric_repair_stats") or {},
            "table_claim_repair": raw.get("table_claim_repair_stats") or {},
            "flow_repair": raw.get("flow_repair_stats") or {},
            "qualitative_repair": raw.get("qualitative_repair_stats") or {},
            "claim_binding_repair": raw.get("claim_binding_repair_stats") or {},
            "table_claims": {
                "valid_claim_count": valid_metric_claim_count,
                "violations": table_claim_violations,
                "table_bundle_hash": _table_bundle_fingerprint(structured_table_bundles),
            },
            "prose_claims": {
                "valid_claim_count": valid_prose_claim_count,
                "violations": claim_binding_violations,
            },
            "overview_coverage": overview_coverage,
            "landmark_result_coverage": landmark_result_coverage,
            "presentation": {
                "mode": "thematic" if is_ai_source else "section_guide",
                "theme_count": len(theme_items) if is_ai_source else 0,
                "sections_retained": len(section_nodes),
            },
            "evidence_quality": {
                "section_count": len(section_nodes),
                "model_supported_section_count": model_supported_sections,
                "evidence_scope_violations": evidence_scope_violations,
            },
        },
    }


def _normalize_outline(
    *,
    raw: dict[str, Any],
    doc_id: str,
    doc: dict[str, Any],
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None = None,
    source_hash: str,
    source: str,
    model: str,
    provider: str,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> dict[str, Any]:
    if raw.get("contract") == READING_OUTLINE_CONTRACT:
        return _normalize_section_study_outline(
            raw=raw,
            doc_id=doc_id,
            doc=doc,
            block_index=block_index,
            structured_table_bundles=structured_table_bundles,
            source_hash=source_hash,
            source=source,
            model=model,
            provider=provider,
            parse_generation=parse_generation,
            document_source_hash=document_source_hash,
        )
    block_map = _flatten_blocks(block_index)
    used_ids: set[str] = set()
    evidence_stats = {
        "leaf_count": 0,
        "model_supported_leaf_count": 0,
        "supported_leaf_count": 0,
        "fallback_relinked_leaf_count": 0,
    }
    fallback_title = _filename_title(doc)
    title = _limit(raw.get("title") or fallback_title, 180)

    def normalize_node(node: dict[str, Any], idx_path: str, inherited_type: str = "note") -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        raw_id = str(node.get("id") or f"node_{idx_path}").strip()
        node_id = _slug_id(raw_id) or f"node_{idx_path}"
        base_id = node_id
        suffix = 2
        while node_id in used_ids:
            node_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(node_id)

        node_type = _limit(node.get("type") or inherited_type, 40)
        node_title = _limit(node.get("title") or node.get("label") or "阅读要点", 120)
        summary = _limit(node.get("summary") or node.get("description") or "", 900)
        raw_evidence_ids = _valid_block_ids(
            node.get("evidence_block_ids") or node.get("block_ids"),
            block_map,
        )
        children = []
        for child_idx, child in enumerate(node.get("children") or []):
            child_node = normalize_node(child, f"{idx_path}_{child_idx + 1}", node_type)
            if child_node:
                children.append(child_node)

        is_title_node = node_type.lower() == "title" or node_id == "paper_title"
        if is_title_node:
            evidence_ids = raw_evidence_ids
        else:
            evidence_ids = _supported_evidence_ids(
                raw_evidence_ids,
                node_type=node_type,
                title=node_title,
                summary=summary,
                block_map=block_map,
            )
        model_supported = bool(evidence_ids)

        if children:
            for child in children:
                for block_id in child.get("evidence_block_ids", []):
                    if block_id not in evidence_ids:
                        evidence_ids.append(block_id)
        fallback_relinked = False
        if not evidence_ids:
            evidence_ids = _fallback_evidence(
                node_title,
                summary,
                node_type,
                block_map,
                anchor_ids=raw_evidence_ids,
            )
            fallback_relinked = bool(evidence_ids)

        is_leaf_claim = not children and not is_title_node
        if is_leaf_claim:
            evidence_stats["leaf_count"] += 1
            if model_supported:
                evidence_stats["model_supported_leaf_count"] += 1
            if evidence_ids:
                evidence_stats["supported_leaf_count"] += 1
            if fallback_relinked:
                evidence_stats["fallback_relinked_leaf_count"] += 1

        evidence_ids = evidence_ids[:8]
        first_block = evidence_ids[0] if evidence_ids else None
        page = int(block_map[first_block]["page"]) if first_block in block_map else 1
        pages = sorted({int(block_map[bid]["page"]) for bid in evidence_ids if bid in block_map})

        return {
            "id": node_id,
            "type": node_type,
            "title": node_title,
            "summary": summary or _summarize_blocks([block_map[bid] for bid in evidence_ids if bid in block_map]),
            "evidence_block_ids": evidence_ids,
            "evidence": {
                "block_ids": evidence_ids,
                "pages": pages,
                "primary_page": page,
            },
            "page": page,
            "first_block": first_block,
            "children": children,
        }

    items = []
    for idx, item in enumerate(raw.get("items") or []):
        normalized = normalize_node(item, str(idx + 1))
        if normalized:
            items.append(normalized)

    return {
        "version": READING_OUTLINE_VERSION,
        "doc_id": doc_id,
        "source": source,
        "source_hash": source_hash,
        "prompt_version": READING_OUTLINE_PROMPT_VERSION,
        "parse_generation": parse_generation,
        "document_source_hash": document_source_hash,
        "title": title,
        "items": items,
        "flat_items": _flatten_outline_items(items),
        "created_at": time.time(),
        "model": model,
        "provider": provider,
        "meta": {
            "block_count": len(block_map),
            "page_count": len(block_index.get("pages", []) or []),
            "evidence_quality": evidence_stats,
        },
    }


def _flatten_blocks(block_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    section_titles: dict[str, str] = {}
    for page in block_index.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            section_id = str(block.get("section_id") or "").strip()
            text = str(block.get("text") or "").strip()
            if section_id and block.get("type") == "heading" and text:
                section_titles.setdefault(section_id, text)

    result: dict[str, dict[str, Any]] = {}
    current_heading = ""
    for page in block_index.get("pages", []) or []:
        page_num = int(page.get("page") or 1)
        for block in page.get("blocks", []) or []:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            text = str(block.get("text") or "").strip()
            item = dict(block)
            if block.get("type") == "heading" and text:
                current_heading = text
            section_id = str(block.get("section_id") or "").strip()
            item["page"] = page_num
            item["text"] = text
            item["section_title"] = section_titles.get(section_id) or current_heading
            result[block_id] = annotate_block_role(
                item,
                section_title=str(item.get("section_title") or ""),
            )
    return result


def _select_prompt_blocks(block_index: dict[str, Any]) -> list[dict[str, Any]]:
    block_map = _flatten_blocks(block_index)
    ordered = list(block_map.values())
    body: list[dict[str, Any]] = []
    for block in ordered:
        if block.get("type") == "heading" and _is_reference_heading(str(block.get("text") or "")):
            break
        body.append(block)
    if not body:
        body = ordered

    sections: list[dict[str, Any]] = []
    preamble: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    order_by_id: dict[str, int] = {}
    for order, block in enumerate(body):
        block_id = str(block.get("block_id") or "")
        if block_id:
            order_by_id[block_id] = order
        if block.get("type") == "heading":
            heading_text = str(block.get("text") or "").strip()
            if not heading_text or _is_noise_text(heading_text):
                continue
            if _is_acknowledgment_heading(heading_text):
                current_section = None
                continue
            current_section = {
                "heading": block,
                "content": [],
                "order": order,
                "priority": 200 if not sections else _section_summary_priority(heading_text),
            }
            sections.append(current_section)
            continue
        if not _is_prompt_content_block(block):
            continue
        if current_section is None:
            preamble.append(block)
        else:
            current_section["content"].append(block)

    ranked_sections = sorted(
        sections,
        key=lambda section: (-int(section["priority"]), int(section["order"])),
    )[:MAX_PROMPT_HEADINGS]
    selected_sections = sorted(ranked_sections, key=lambda section: int(section["order"]))
    for section in selected_sections:
        section["content"] = _pick_section_prompt_content(
            section["content"],
            str(section["heading"].get("text") or ""),
        )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()

    def add(block: dict[str, Any]) -> None:
        block_id = block.get("block_id")
        text = str(block.get("text") or "").strip()
        normalized_text = " ".join(text.lower().split())
        if (
            not block_id
            or block_id in seen_ids
            or not text
            or _is_noise_text(text)
            or normalized_text in seen_text
        ):
            return
        seen_ids.add(block_id)
        seen_text.add(normalized_text)
        selected.append({
            "block_id": block_id,
            "page": block.get("page"),
            "type": block.get("type") or "paragraph",
            "section_title": block.get("section_title") or "",
            "text": _limit(text, MAX_BLOCK_TEXT),
        })

    for section in selected_sections:
        add(section["heading"])
    for block in _evenly_spaced_blocks(preamble, min(4, len(preamble))):
        add(block)
    for content_index in range(min(2, MAX_SECTION_CONTENT_BLOCKS)):
        for section in selected_sections:
            content = section["content"]
            if content_index < len(content):
                add(content[content_index])
            if len(selected) >= MAX_PROMPT_BLOCKS:
                break
        if len(selected) >= MAX_PROMPT_BLOCKS:
            break

    result_blocks = [
        block for block in body
        if block.get("type") in {"caption", "table", "visual_enrichment"}
        and _is_prompt_content_block(block)
    ]
    for block in _evenly_spaced_blocks(result_blocks, MAX_PROMPT_RESULT_BLOCKS):
        add(block)
        if len(selected) >= MAX_PROMPT_BLOCKS:
            break

    for content_index in range(2, MAX_SECTION_CONTENT_BLOCKS):
        for section in selected_sections:
            content = section["content"]
            if content_index < len(content):
                add(content[content_index])
            if len(selected) >= MAX_PROMPT_BLOCKS:
                break
        if len(selected) >= MAX_PROMPT_BLOCKS:
            break

    if len(selected) < MAX_PROMPT_BLOCKS:
        remaining = [
            block for block in body
            if block.get("block_id") not in seen_ids and _is_prompt_content_block(block)
        ]
        for block in _evenly_spaced_blocks(remaining, MAX_PROMPT_BLOCKS - len(selected)):
            add(block)

    selected.sort(key=lambda block: order_by_id.get(str(block.get("block_id") or ""), 10**9))
    return selected[:MAX_PROMPT_BLOCKS]


def _is_reference_heading(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s*)?(references|bibliography)\b", normalized))


def _is_acknowledgment_heading(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s*)?acknowledg(?:e)?ments?\b", normalized))


def _is_post_reference_template_artifact(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    return bool(re.match(r"^\s*(?:[a-z0-9-]+\s+)?(?:paper\s+)?checklist\b", normalized, re.IGNORECASE))

def _is_excluded_summary_section(text: str) -> bool:
    return _is_reference_heading(text) or _is_acknowledgment_heading(text)


def _section_summary_priority(title: str) -> int:
    normalized = " ".join(str(title or "").lower().split())
    high_value = (
        "abstract", "introduction", "background", "motivation", "problem",
        "approach", "method", "model", "training", "pre-training", "dataset",
        "experiment", "evaluation", "result", "analysis", "performance",
        "robustness", "human", "overlap", "limitation", "discussion",
        "broader impact", "social impact", "conclusion",
    )
    return 100 if any(keyword in normalized for keyword in high_value) else 20


def _looks_like_author_or_affiliation(block: dict[str, Any]) -> bool:
    role = classify_block_role(
        block,
        section_title=str(block.get("section_title") or ""),
    )["role"]
    return role in FRONT_MATTER_ROLES


def _is_semantic_figure_block(block: dict[str, Any]) -> bool:
    if block.get("type") != "figure":
        return False
    source = str(block.get("source") or "").strip().lower()
    purpose = str(block.get("purpose") or "").strip().lower()
    text = " ".join(str(block.get("text") or "").split())
    return (
        len(text) >= 80
        and len(re.findall(r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff'\-]*", text)) >= 8
        and (
            source in {"vlm", "mineru_vlm", "visual_enrichment", "vision"}
            or bool(purpose)
            or bool(block.get("visual_enrichment"))
        )
    )


def _is_prompt_content_block(block: dict[str, Any]) -> bool:
    text = " ".join(str(block.get("text") or "").split())
    if (
        not text
        or _is_noise_text(text)
        or block_is_outline_excluded(
            block,
            section_title=str(block.get("section_title") or ""),
        )
        or _is_excluded_summary_section(str(block.get("section_title") or ""))
    ):
        return False
    block_type = str(block.get("type") or "").lower()
    if block_type == "paragraph":
        return len(text) >= 80
    if block_type == "caption":
        return len(text) >= 40
    if block_type == "table":
        return len(text) >= 60
    if block_type == "visual_enrichment":
        return len(text) >= 40
    return _is_semantic_figure_block(block)


def _result_signal_score(block: dict[str, Any]) -> int:
    text = " ".join(str(block.get("text") or "").lower().split())
    score = sum(
        2 for keyword in (
            "result", "outperform", "improve", "accuracy", "performance", "finding",
            "suggest", "show", "demonstrate", "increase", "decrease", "significant",
        )
        if keyword in text
    )
    score += min(4, len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)))
    return score


def _pick_section_prompt_content(
    blocks: list[dict[str, Any]],
    section_title: str,
) -> list[dict[str, Any]]:
    if len(blocks) <= MAX_SECTION_CONTENT_BLOCKS:
        return list(blocks)
    narrative = [
        block for block in blocks
        if block.get("type") in {"paragraph", "visual_enrichment"} or _is_semantic_figure_block(block)
    ]
    visual_results = [block for block in blocks if block.get("type") in {"caption", "table"}]
    picked: list[dict[str, Any]] = []

    def pick(block: dict[str, Any] | None) -> None:
        if block and block not in picked:
            picked.append(block)

    if narrative:
        pick(narrative[0])
        middle = narrative[1:-1]
        if middle:
            pick(max(middle, key=_result_signal_score))
        pick(narrative[-1])
    if visual_results and (
        len(picked) < MAX_SECTION_CONTENT_BLOCKS
        or _section_summary_priority(section_title) >= 100
        and any(keyword in section_title.lower() for keyword in ("experiment", "result", "evaluation"))
    ):
        if len(picked) >= MAX_SECTION_CONTENT_BLOCKS:
            picked[-1] = visual_results[0]
        else:
            pick(visual_results[0])
    for block in blocks:
        if len(picked) >= MAX_SECTION_CONTENT_BLOCKS:
            break
        pick(block)
    order = {str(block.get("block_id") or ""): index for index, block in enumerate(blocks)}
    picked.sort(key=lambda block: order.get(str(block.get("block_id") or ""), 10**9))
    return picked[:MAX_SECTION_CONTENT_BLOCKS]


def _evenly_spaced_blocks(blocks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not blocks:
        return []
    if len(blocks) <= limit:
        return list(blocks)
    if limit == 1:
        return [blocks[len(blocks) // 2]]
    indexes = {
        round(index * (len(blocks) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [blocks[index] for index in sorted(indexes)]


def _pick_title_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    first_page = [b for b in blocks if int(b.get("page") or 1) == 1 and b.get("text")]
    candidates = [
        b for b in first_page
        if 18 <= len(str(b.get("text"))) <= 220 and not _is_noise_text(str(b.get("text")))
    ]
    headings = [b for b in candidates if b.get("type") == "heading"]
    return (headings or candidates or first_page or [None])[0]


def _pick_blocks(blocks: list[dict[str, Any]], group: str, limit: int) -> list[dict[str, Any]]:
    keywords = _KEYWORD_GROUPS.get(group, [])
    result = []
    for block in blocks:
        text = str(block.get("text") or "")
        if not _is_substantive_evidence_block(block):
            continue
        lower = f"{block.get('section_title') or ''} {text}".lower()
        if any(keyword in lower for keyword in keywords):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _first_paragraphs(
    blocks: list[dict[str, Any]],
    *,
    start_page: int,
    limit: int,
    skip_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    skip_ids = skip_ids or set()
    result = []
    for block in blocks:
        if block.get("block_id") in skip_ids:
            continue
        if int(block.get("page") or 1) < start_page:
            continue
        text = str(block.get("text") or "")
        if _is_substantive_evidence_block(block):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_evidence(
    title: str,
    summary: str,
    node_type: str,
    block_map: dict[str, dict[str, Any]],
    anchor_ids: list[str] | None = None,
) -> list[str]:
    category = _node_evidence_category(node_type, title)
    if category == "title":
        title_blocks = [
            block for block in block_map.values()
            if block.get("type") == "heading" and int(block.get("page") or 1) == 1
        ]
        return [str(title_blocks[0]["block_id"])] if title_blocks else []

    anchored: list[str] = []
    for anchor_id in anchor_ids or []:
        anchor = block_map.get(anchor_id) or {}
        if anchor.get("type") != "heading":
            continue
        section_id = str(anchor.get("section_id") or "").strip()
        section_title = str(anchor.get("section_title") or anchor.get("text") or "").strip()
        for block in block_map.values():
            same_section = section_id and str(block.get("section_id") or "").strip() == section_id
            same_title = section_title and str(block.get("section_title") or "").strip() == section_title
            if (
                (same_section or same_title)
                and _is_substantive_evidence_block(block)
                and block.get("block_id") not in anchored
            ):
                anchored.append(str(block["block_id"]))
                break
        if len(anchored) >= 3:
            break
    if anchored:
        return anchored

    candidates = [block for block in block_map.values() if _is_substantive_evidence_block(block)]
    relevant = [
        block for block in candidates
        if _evidence_relevant(block, node_type=node_type, title=title, summary=summary)
    ]
    pool = relevant or candidates
    query = f"{title} {summary} {node_type}".lower()
    keywords = re.findall(r"[A-Za-z][A-Za-z-]{3,}", query)
    keywords.extend(_evidence_category_keywords(category))
    scored = []
    for block in pool:
        lower = f"{block.get('section_title') or ''} {block.get('text') or ''}".lower()
        score = sum(1 for keyword in keywords if keyword.lower() in lower)
        if category and _evidence_relevant(block, node_type=node_type, title=title, summary=summary):
            score += 4
        if category == "experiment" and block.get("type") in {"caption", "table", "visual_enrichment"}:
            score += 2
        if category == "background" and int(block.get("page") or 1) <= 2:
            score += 1
        if score:
            scored.append((score, int(block.get("page") or 1), block["block_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:3]]


def _node_evidence_category(node_type: str, title: str = "") -> str:
    kind = str(node_type or "").lower()
    text = f"{kind} {title}".lower()
    if "title" in kind or "论文标题" in text:
        return "title"
    if any(keyword in kind for keyword in ("experiment", "evaluation", "result", "performance", "analysis")):
        return "experiment"
    if any(keyword in kind for keyword in ("innovation", "contribution", "method", "approach", "training", "model")):
        return "innovation"
    if any(keyword in kind for keyword in ("background", "motivation", "problem")):
        return "background"
    if any(keyword in kind for keyword in ("conclusion", "limitation", "discussion", "impact", "value")):
        return "conclusion"
    if any(keyword in text for keyword in ("背景", "问题", "motivation", "background")):
        return "background"
    if any(keyword in text for keyword in ("实验", "评估", "结果", "性能", "鲁棒", "人类", "重叠")):
        return "experiment"
    if any(keyword in text for keyword in ("创新", "贡献", "方法", "模型", "预训练")):
        return "innovation"
    if any(keyword in text for keyword in ("结论", "局限", "限制", "影响", "价值")):
        return "conclusion"
    return ""


def _evidence_category_keywords(category: str) -> list[str]:
    return {
        "background": ["abstract", "introduction", "background", "motivation", "problem", "challenge", "gap"],
        "innovation": [
            "approach", "method", "model", "training", "pre-training", "dataset",
            "architecture", "objective", "prompt", "we propose", "we present",
        ],
        "experiment": [
            "experiment", "evaluation", "result", "performance", "accuracy", "analysis",
            "robustness", "transfer", "human", "overlap", "table", "figure", "benchmark",
        ],
        "conclusion": ["conclusion", "limitation", "discussion", "broader impact", "social impact", "future work"],
    }.get(category, [])


def _is_substantive_evidence_block(block: dict[str, Any]) -> bool:
    text = " ".join(str(block.get("text") or "").split())
    if (
        not text
        or _is_noise_text(text)
        or block_is_outline_excluded(
            block,
            section_title=str(block.get("section_title") or ""),
        )
    ):
        return False
    block_type = str(block.get("type") or "").lower()
    if block_type == "paragraph":
        return len(text) >= 60
    if block_type == "caption":
        return len(text) >= 40
    if block_type == "table":
        return len(text) >= 60
    if block_type == "visual_enrichment":
        return len(text) >= 40
    return _is_semantic_figure_block(block)


def _evidence_relevant(
    block: dict[str, Any],
    *,
    node_type: str,
    title: str,
    summary: str,
) -> bool:
    if not _is_substantive_evidence_block(block):
        return False
    category = _node_evidence_category(node_type, title)
    if not category or category == "title":
        return True
    haystack = f"{block.get('section_title') or ''} {block.get('text') or ''}".lower()
    if any(keyword in haystack for keyword in _evidence_category_keywords(category)):
        return True
    if category == "background" and int(block.get("page") or 1) <= 2:
        return True
    if category == "experiment" and block.get("type") in {"caption", "table", "visual_enrichment"}:
        return True
    english_claim_terms = re.findall(r"[A-Za-z][A-Za-z-]{3,}", f"{title} {summary}".lower())
    return bool(english_claim_terms and sum(term in haystack for term in english_claim_terms) >= 2)


def _supported_evidence_ids(
    evidence_ids: list[str],
    *,
    node_type: str,
    title: str,
    summary: str,
    block_map: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        block_id for block_id in evidence_ids
        if block_id in block_map
        and _evidence_relevant(
            block_map[block_id],
            node_type=node_type,
            title=title,
            summary=summary,
        )
    ]


def _valid_block_ids(value: Any, block_map: dict[str, dict[str, Any]]) -> list[str]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    result = []
    for item in raw:
        block_id = str(item or "").strip()
        if block_id and block_id in block_map and block_id not in result:
            result.append(block_id)
    return result


def _section_study_quality_issues(
    outline: dict[str, Any],
    block_index: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    block_map = _flatten_blocks(block_index)
    skeleton = _build_reading_section_skeleton(block_index)
    payloads = _prepare_section_payloads(skeleton, block_index)
    flat_items = outline.get("flat_items") or _flatten_outline_items(outline.get("items") or [])
    by_section = {
        str(item.get("source_section_id")): item
        for item in flat_items
        if isinstance(item, dict) and item.get("source_section_id")
    }
    expected_nodes = _flatten_skeleton_nodes(skeleton)
    expected_body = {
        str(node.get("source_section_id")) for node in expected_nodes
        if node.get("section_kind") == "body"
    }
    expected_appendix = {
        str(node.get("source_section_id")) for node in expected_nodes
        if node.get("section_kind") == "appendix"
    }
    missing_body = sorted(expected_body - set(by_section))
    missing_appendix = sorted(expected_appendix - set(by_section))
    if missing_body:
        issues.append(f"正文章节覆盖不足:{len(expected_body) - len(missing_body)}/{len(expected_body)}")
    if missing_appendix:
        issues.append(f"附录章节覆盖不足:{len(expected_appendix) - len(missing_appendix)}/{len(expected_appendix)}")

    def section_incomplete(section_id: str) -> bool:
        payload = payloads.get(section_id) or {}
        item = by_section.get(section_id) or {}
        return bool(payload.get("blocks")) and (
            not _is_substantive_section_summary(item.get("summary"))
            or item.get("section_status") in {"fallback", "unavailable"}
        )

    incomplete_body = [section_id for section_id in expected_body if section_incomplete(section_id)]
    incomplete_appendix = [section_id for section_id in expected_appendix if section_incomplete(section_id)]
    if incomplete_body:
        issues.append(f"正文章节摘要不完整:{len(incomplete_body)}")
    if incomplete_appendix:
        issues.append(f"附录章节摘要不完整:{len(incomplete_appendix)}")

    evidence_errors = 0
    for section_id, item in by_section.items():
        evidence_ids = _valid_block_ids(item.get("evidence_block_ids"), block_map)
        page_start = int(item.get("page_start") or item.get("section_page") or 1)
        page_end = int(item.get("page_end") or item.get("section_page") or 1)
        if item.get("evidence_scope") == "section":
            allowed_ids = set((payloads.get(section_id) or {}).get("allowed_block_ids") or [])
            for block_id in evidence_ids:
                block = block_map.get(block_id) or {}
                if allowed_ids:
                    in_scope = block_id in allowed_ids
                elif section_id.startswith("pages_"):
                    in_scope = page_start <= int(block.get("page") or 1) <= page_end
                else:
                    in_scope = str(block.get("section_id") or "") == section_id
                if not in_scope or not _is_substantive_evidence_block(block):
                    evidence_errors += 1
        if (payloads.get(section_id) or {}).get("blocks") and not evidence_ids:
            evidence_errors += 1
    if evidence_errors:
        issues.append(f"章节证据越界或缺失:{evidence_errors}")

    overview = next((item for item in flat_items if item.get("type") == "overview"), None)
    if not overview or not str(overview.get("summary") or "").strip():
        issues.append("缺少论文要旨")
    elif len(overview.get("source_section_ids") or []) < min(2, len(expected_body)):
        issues.append("论文要旨缺少跨章节来源")
    display_items = outline.get("items") if isinstance(outline.get("items"), list) else []
    themes = {
        str(item.get("type") or "").removeprefix("theme_"): item
        for item in display_items
        if isinstance(item, dict) and str(item.get("type") or "").startswith("theme_")
    }
    missing_themes = [kind for kind, _title, _limit in THEME_SPECS if kind not in themes]
    if missing_themes:
        issues.append("缺少重点主题:" + ",".join(missing_themes))
    empty_themes = [
        kind for kind, item in themes.items()
        if not str(item.get("summary") or "").strip()
        and not ((item.get("study") or {}).get("findings") or [])
    ]
    if empty_themes:
        issues.append("重点主题为空:" + ",".join(empty_themes))
    ungrounded_themes = [
        kind for kind, item in themes.items()
        if not (item.get("source_section_ids") or []) or not (item.get("evidence_block_ids") or [])
    ]
    if ungrounded_themes:
        issues.append("重点主题缺少来源:" + ",".join(ungrounded_themes))
    return issues


def _reading_outline_quality_issues(
    outline: dict[str, Any],
    block_index: dict[str, Any],
) -> list[str]:
    if (outline.get("meta") or {}).get("contract") == READING_OUTLINE_CONTRACT:
        return _section_study_quality_issues(outline, block_index)
    items = outline.get("items") if isinstance(outline.get("items"), list) else []
    flat_items = outline.get("flat_items") or _flatten_outline_items(items)
    issues: list[str] = []
    if len(flat_items) < 5:
        issues.append("总结节点过少")

    categories = {
        _node_evidence_category(str(item.get("type") or ""), str(item.get("title") or ""))
        for item in flat_items
        if isinstance(item, dict)
    }
    missing_categories = [
        category for category in ("background", "innovation", "experiment", "conclusion")
        if category not in categories
    ]
    if missing_categories:
        issues.append("缺少必要部分:" + ",".join(missing_categories))

    block_map = _flatten_blocks(block_index)
    leaf_count = 0
    supported_leaf_count = 0
    page_mismatch_count = 0
    leaf_items: list[dict[str, Any]] = []

    def collect_leaf_items(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            children = node.get("children") if isinstance(node.get("children"), list) else []
            if children:
                collect_leaf_items(children)
            else:
                leaf_items.append(node)

    collect_leaf_items(items)
    for item in leaf_items:
        if _node_evidence_category(str(item.get("type") or ""), str(item.get("title") or "")) == "title":
            continue
        leaf_count += 1
        evidence_ids = _valid_block_ids(
            item.get("evidence_block_ids") or (item.get("evidence") or {}).get("block_ids"),
            block_map,
        )
        if _supported_evidence_ids(
            evidence_ids,
            node_type=str(item.get("type") or ""),
            title=str(item.get("title") or ""),
            summary=str(item.get("summary") or ""),
            block_map=block_map,
        ):
            supported_leaf_count += 1
        first_block = str(item.get("first_block") or "")
        first_page = int(block_map.get(first_block, {}).get("page") or 0)
        primary_page = int((item.get("evidence") or {}).get("primary_page") or item.get("page") or 0)
        if first_block and first_page != primary_page:
            page_mismatch_count += 1

    if leaf_count == 0:
        issues.append("没有可核验的叶子结论")
    elif supported_leaf_count / leaf_count < 0.8:
        issues.append(f"实质证据覆盖不足:{supported_leaf_count}/{leaf_count}")

    evidence_meta = (outline.get("meta") or {}).get("evidence_quality") or {}
    model_leaf_count = int(evidence_meta.get("leaf_count") or 0)
    model_supported_count = int(evidence_meta.get("model_supported_leaf_count") or 0)
    if model_leaf_count != leaf_count:
        issues.append("证据质量统计不完整")
    elif model_leaf_count and model_supported_count / model_leaf_count < 0.7:
        issues.append(f"模型原始证据覆盖不足:{model_supported_count}/{model_leaf_count}")
    if page_mismatch_count:
        issues.append(f"页码与首证据不一致:{page_mismatch_count}")
    return issues


def _flatten_outline_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], level: int) -> None:
        for node in nodes:
            item = {k: v for k, v in node.items() if k != "children"}
            item["level"] = level
            flat.append(item)
            walk(node.get("children") or [], level + 1)

    walk(items, 1)
    return flat


def _summarize_blocks(blocks: list[dict[str, Any]]) -> str:
    text = " ".join(str(block.get("text") or "").strip() for block in blocks if block)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = " ".join(text.split())
    return _limit(text, 220)


def _fallback_heading_children(
    blocks: list[dict[str, Any]],
    limit: int,
    group: str = "",
) -> list[dict[str, Any]]:
    result = []
    keywords = _evidence_category_keywords(group)
    for block in blocks:
        text = str(block.get("text") or "").strip()
        lower = text.lower()
        if (
            block.get("type") == "heading"
            and text
            and not _is_noise_text(text)
            and (not keywords or any(keyword in lower for keyword in keywords))
        ):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_experiment_children(blocks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text or _is_noise_text(text):
            continue
        block_type = block.get("type")
        lower = text.lower()
        if block_type == "heading" and any(keyword in lower for keyword in _KEYWORD_GROUPS["experiment"]):
            result.append(block)
        elif (
            block_type in {"caption", "table", "visual_enrichment", "figure"}
            and _evidence_relevant(
                block,
                node_type="experiment",
                title="实验结果",
                summary="",
            )
        ):
            result.append(block)
        if len(result) >= limit:
            break
    return result


def _fallback_child_title(block: dict[str, Any], fallback: str) -> str:
    text = str(block.get("text") or "").strip()
    if not text:
        return fallback
    if block.get("type") == "heading":
        return _limit(_clean_title(text), 60)
    if block.get("type") in {"caption", "figure", "table"}:
        return _limit(text, 60)
    return fallback


def _short_cn_title(block: dict[str, Any], fallback: str) -> str:
    text = str(block.get("text") or "").strip()
    if not text:
        return fallback
    if block.get("type") in {"caption", "figure", "table"}:
        return _limit(text, 34)
    sentence = re.split(r"[。.;；\n]", text, maxsplit=1)[0].strip()
    return _limit(sentence or text, 34)


def _clean_title(text: Any) -> str:
    title = " ".join(str(text or "").split())
    title = re.sub(r"^(title|paper title)\s*[:：]\s*", "", title, flags=re.IGNORECASE)
    return _limit(title, 180)


def _filename_title(doc: dict[str, Any]) -> str:
    name = str(doc.get("filename") or "当前论文")
    return re.sub(r"\.[A-Za-z0-9]+$", "", name).strip() or "当前论文"


def _is_noise_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if len(stripped) < 2:
        return True
    lower = stripped.lower()
    if "arxiv:" in lower:
        return True
    if re.fullmatch(r"[\d\s.,:/()%+\-=×x]+", stripped):
        return True
    alnum = re.sub(r"\s+", "", stripped)
    if alnum and sum(ch.isdigit() for ch in alnum) / max(len(alnum), 1) > 0.55:
        return True
    return False


def _source_hash(
    block_index: dict[str, Any],
    structured_table_bundles: list[dict[str, Any]] | None = None,
) -> str:
    """Fingerprint the evidence structure, not only the visible text."""
    pages = []
    for page_order, page in enumerate(block_index.get("pages", []) or []):
        if not isinstance(page, dict):
            continue
        blocks = []
        for block_order, block in enumerate(page.get("blocks", []) or []):
            if not isinstance(block, dict):
                continue
            blocks.append({
                "order": block_order,
                "block_id": str(block.get("block_id") or ""),
                "type": str(block.get("type") or ""),
                "text": " ".join(str(block.get("text") or "").split()),
                "level": block.get("level"),
                "section_id": block.get("section_id"),
            })
        pages.append({
            "order": page_order,
            "page": page.get("page"),
            "blocks": blocks,
        })
    payload = {
        "block_index_version": block_index.get("version"),
        "block_index_source": block_index.get("source"),
        "pages": pages,
        "outline": block_index.get("outline") or [],
        "structured_table_hash": _table_bundle_fingerprint(structured_table_bundles),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _parse_identity(doc_id: str, doc: dict[str, Any]) -> tuple[str, str]:
    """Return the active primary parse identity for a cached outline."""
    manifest = read_parse_manifest(doc, doc_id=doc_id)
    return (
        str(manifest.get("generation") or "").strip(),
        str(manifest.get("source_hash") or "").strip(),
    )


def _has_parse_identity(outline: dict[str, Any]) -> bool:
    return bool(
        str(outline.get("parse_generation") or "").strip()
        and str(outline.get("document_source_hash") or "").strip()
    )


def _matches_parse_identity(
    outline: dict[str, Any],
    parse_generation: str,
    document_source_hash: str,
) -> bool:
    return (
        _has_parse_identity(outline)
        and str(outline.get("parse_generation") or "").strip() == parse_generation
        and str(outline.get("document_source_hash") or "").strip() == document_source_hash
    )


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(message.get("content"), str):
                return message["content"]
    return str(response or "")


def _parse_json_object(content: str) -> dict[str, Any]:
    return parse_json_object(content, allow_partial=False)


def _slug_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip()).strip("_").lower()
    return text[:64]


def _limit(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."
