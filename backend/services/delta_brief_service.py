"""增量简报：判断一篇论文相对若干对照论文的真实增量。

与「全文总结」的区别是问题不同：总结回答「这篇讲了什么」，增量简报回答
「相对我已经读过的，这篇多了什么」。论文流水线式产出的背景下，后者才是
稀缺判断。

本模块刻意保持**完全确定性**，不调用模型：

- 判定依据是主题槽位内的词元重合与数字比对，可复现、可测试、零延迟；
- 拿不到证据块的点一律归入 ``undetermined``，绝不猜测——每条判定都必须
  能追溯到双侧的 ``evidence_block_ids``，这是本项目区别于通用工具的根据；
- 阈值沿用 ``multi_doc_fanout_service.group_potential_conflicts`` 的保守
  哲学（要求足量共享词元 + 比例下限），宁可少判也不误判。

对比单位是 ``reading_outline`` 的主题（background / innovation /
experiment / conclusion）。它本身就是按语义组织的，天然适合做配对比较，
也让「方法上的增量」和「实验上的增量」不会互相污染。
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from services.citation_alignment_service import tokenize

DELTA_BRIEF_SCHEMA_VERSION = "delta-brief-v1"

# 主题顺序与 reading_outline 保持一致，保证简报的阅读顺序稳定。
DELTA_THEME_SPECS: tuple[tuple[str, str], ...] = (
    ("theme_background", "研究背景与问题"),
    ("theme_innovation", "核心方法与创新"),
    ("theme_experiment", "关键实验结果"),
    ("theme_conclusion", "结论、价值与边界"),
)

# 判定为「实质重合」所需的词元 Jaccard 下限与共享词元下限。两个条件同时
# 成立才算重合：只共享一两个高频词（例如"模型""实验"）远不足以说明两篇
# 论文讲的是同一件事。
_OVERLAP_RATIO = 0.45
_MIN_SHARED_TOKENS = 2

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _clean(value: Any, limit: int = 400) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _tokens(text: str) -> set[str]:
    """复用引用对齐的 CJK 友好切分。

    自建的「连续中文整串」正则会把一整句中文吞成单个词元，导致任意两句
    中文要么完全相等要么零重合——中文场景下相似度直接失效。这里改用项目
    既有的 ``tokenize``（英文/数字词 + 中文单字与相邻双字），与引用对齐
    使用同一套切分口径。
    """
    return set(tokenize(text, informative_only=True))


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(str(text or "")))


def _finding_points(outline: Mapping[str, Any], theme_type: str) -> list[dict[str, Any]]:
    """取出某主题下的重点，并把每条重点与它自己的证据块对齐。

    ``study.finding_evidence`` 是后来才加入的逐条绑定；更早的缓存只有
    ``study.findings`` 与主题级 ``evidence_block_ids``。两种形状都要能读，
    否则历史文档一进来就整体判为证据缺失。
    """
    items = outline.get("items") if isinstance(outline.get("items"), list) else []
    theme = next(
        (
            item for item in items
            if isinstance(item, dict) and str(item.get("type") or "") == theme_type
        ),
        None,
    )
    if not isinstance(theme, dict):
        return []

    study = theme.get("study") if isinstance(theme.get("study"), dict) else {}
    theme_blocks = [
        str(value).strip()
        for value in (theme.get("evidence_block_ids") or [])
        if str(value).strip()
    ]
    evidence_by_text: dict[str, list[str]] = {}
    for entry in study.get("finding_evidence") or []:
        if not isinstance(entry, dict):
            continue
        key = _clean(entry.get("text"))
        blocks = [str(value).strip() for value in (entry.get("evidence_block_ids") or []) if str(value).strip()]
        if key and blocks:
            evidence_by_text[key] = blocks

    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in study.get("findings") or []:
        text = _clean(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        points.append({
            "text": text,
            "evidence_block_ids": evidence_by_text.get(text) or theme_blocks,
        })

    # 主题只有概括句、没有分条重点时，概括句本身就是可对比的单位。
    if not points:
        summary = _clean(theme.get("summary"))
        if summary:
            points.append({"text": summary, "evidence_block_ids": theme_blocks})
    return points


def _similarity(left: set[str], right: set[str]) -> tuple[float, list[str]]:
    if not left or not right:
        return 0.0, []
    shared = left & right
    union = left | right
    if not union:
        return 0.0, []
    return len(shared) / len(union), sorted(shared)[:8]


def _match_point(
    point_tokens: set[str],
    point_numbers: set[str],
    baselines: Sequence[Mapping[str, Any]],
    theme_type: str,
) -> dict[str, Any] | None:
    """在对照论文的同一主题里找出最相似的一条，判断重合还是冲突。"""
    best: dict[str, Any] | None = None
    for baseline in baselines:
        for candidate in _finding_points(baseline.get("outline") or {}, theme_type):
            ratio, shared = _similarity(point_tokens, _tokens(candidate["text"]))
            if len(shared) < _MIN_SHARED_TOKENS or ratio < _OVERLAP_RATIO:
                continue
            if best is None or ratio > best["ratio"]:
                candidate_numbers = _numbers(candidate["text"])
                best = {
                    "ratio": round(ratio, 4),
                    "shared_terms": shared,
                    "doc_id": str(baseline.get("doc_id") or ""),
                    "title": _clean(baseline.get("title"), 200),
                    "text": candidate["text"],
                    "evidence_block_ids": candidate["evidence_block_ids"],
                    # 共享足够多的词元却给出不同数字，是最值得人工复核的情形；
                    # 这里只做标记，不裁决谁对谁错。
                    "numbers_differ": bool(
                        point_numbers and candidate_numbers and point_numbers != candidate_numbers
                    ),
                    "numbers": sorted(candidate_numbers)[:8],
                }
    return best


def build_delta_brief(
    primary: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """构建增量简报。

    ``primary`` 与 ``baselines`` 的元素形如
    ``{"doc_id": str, "title": str, "outline": dict}``。
    """
    primary_outline = primary.get("outline") if isinstance(primary.get("outline"), Mapping) else {}
    usable_baselines = [
        item for item in (baselines or [])
        if isinstance(item, Mapping) and isinstance(item.get("outline"), Mapping)
        and str(item.get("doc_id") or "") != str(primary.get("doc_id") or "")
    ]

    themes: list[dict[str, Any]] = []
    counters = {"novel": 0, "overlap": 0, "conflict": 0, "undetermined": 0}

    for theme_type, theme_title in DELTA_THEME_SPECS:
        verdicts: list[dict[str, Any]] = []
        for point in _finding_points(primary_outline, theme_type):
            # 证据红线：主侧拿不到证据块就不下判断。宁可显式说不知道，也不
            # 输出一条无法追溯的"增量"。
            if not point["evidence_block_ids"]:
                counters["undetermined"] += 1
                verdicts.append({
                    "type": "undetermined",
                    "text": point["text"],
                    "primary_evidence_block_ids": [],
                    "baseline_evidence": [],
                    "reason": "该结论缺少可追溯的证据块，无法参与对比",
                })
                continue

            point_tokens = _tokens(point["text"])
            point_numbers = _numbers(point["text"])
            match = _match_point(point_tokens, point_numbers, usable_baselines, theme_type)

            if match is None:
                verdict_type = "novel"
                reason = (
                    "对照论文的同一主题下没有实质重合的结论"
                    if usable_baselines
                    else "没有可用的对照论文，仅列出本文结论"
                )
                baseline_evidence: list[dict[str, Any]] = []
            else:
                verdict_type = "conflict" if match["numbers_differ"] else "overlap"
                reason = (
                    f"与对照论文共享关键词 {'、'.join(match['shared_terms'][:4])}，但数值不同，需人工复核"
                    if match["numbers_differ"]
                    else f"与对照论文结论实质重合（共享 {'、'.join(match['shared_terms'][:4])}）"
                )
                baseline_evidence = [{
                    "doc_id": match["doc_id"],
                    "title": match["title"],
                    "text": match["text"],
                    "evidence_block_ids": match["evidence_block_ids"],
                    "shared_terms": match["shared_terms"],
                    "similarity": match["ratio"],
                }]

            counters[verdict_type] += 1
            verdicts.append({
                "type": verdict_type,
                "text": point["text"],
                "primary_evidence_block_ids": point["evidence_block_ids"][:8],
                "baseline_evidence": baseline_evidence,
                "reason": reason,
            })

        if verdicts:
            themes.append({"kind": theme_type, "title": theme_title, "verdicts": verdicts})

    total = sum(counters.values())
    return {
        "schema_version": DELTA_BRIEF_SCHEMA_VERSION,
        "primary": {
            "doc_id": str(primary.get("doc_id") or ""),
            "title": _clean(primary.get("title"), 200),
        },
        "baselines": [
            {"doc_id": str(item.get("doc_id") or ""), "title": _clean(item.get("title"), 200)}
            for item in usable_baselines
        ],
        "themes": themes,
        "coverage": {
            "point_count": total,
            "novel_count": counters["novel"],
            "overlap_count": counters["overlap"],
            "conflict_count": counters["conflict"],
            "undetermined_count": counters["undetermined"],
            "baseline_count": len(usable_baselines),
            # 没有对照论文时结论只能是"本文有什么"，不构成增量判断；
            # 界面据此避免把它说成"相对已读的增量"。
            "comparable": bool(usable_baselines and total > 0),
        },
    }


def iter_theme_types() -> Iterable[str]:
    return (theme_type for theme_type, _title in DELTA_THEME_SPECS)
