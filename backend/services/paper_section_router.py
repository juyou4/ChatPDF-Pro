"""把论文问句对到这篇文档的真实章节标题，而不是只认「方法/贡献」标签。

对照 paperAI 的 get_section / grep_paper(section=) 与 RAGFlow retrieval_by_toc：
先认 facet，再和大纲标题做别名匹配。数值表、引用清单、整篇总结不走这里。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

FACET_HEADING_ALIASES: dict[str, tuple[str, ...]] = {
    "methods": (
        "method", "methods", "methodology", "approach", "our approach",
        "proposed method", "algorithm", "方法", "方法论", "算法", "途径",
    ),
    "results": (
        "result", "results", "experiment", "experiments", "evaluation",
        "experimental results", "实验", "实验结果", "结果", "评估",
    ),
    "limitations": (
        "limitation", "limitations", "discussion", "future work",
        "局限", "局限性", "讨论", "不足", "未来工作",
    ),
    "related_work": (
        "related work", "related works", "prior work", "previous work",
        "related research", "相关工作",
    ),
    "contributions": (
        "contribution", "contributions", "introduction", "引言", "简介",
        "贡献", "创新点",
    ),
    "setup": (
        "experimental setup", "implementation", "implementation details",
        "training details", "实验设置", "实验设计", "实现", "实现细节",
    ),
    "motivation": (
        "introduction", "motivation", "background", "problem",
        "引言", "动机", "背景", "问题",
    ),
    "conclusion": (
        "conclusion", "conclusions", "concluding", "结论", "总结",
    ),
    "architecture": (
        "architecture", "network", "model architecture", "overview of",
        "架构", "网络结构", "模型结构", "整体结构",
    ),
    "dataset": (
        "dataset", "datasets", "benchmark",
        "数据集",
    ),
}

_FACET_QUERY_PATTERNS: dict[str, re.Pattern[str]] = {
    "methods": re.compile(
        r"(?:核心方法|研究方法|提出的方法|本文方法|"
        r"方法(?:是什么|有哪些|是怎样)|"
        r"what\s+(?:is|are)\s+the\s+(?:core\s+)?method)",
        re.IGNORECASE,
    ),
    "results": re.compile(
        r"(?:实验结果|实验发现|主要结果|主要发现|what\s+(?:are|is)\s+the\s+(?:main\s+)?(?:results?|findings?))",
        re.IGNORECASE,
    ),
    "limitations": re.compile(
        r"(?:局限性|不足之处|主要缺陷|局限(?:性|之处)?(?:是什么|有哪些)|what\s+(?:are|is)\s+the\s+(?:main\s+)?limitations?)",
        re.IGNORECASE,
    ),
    "related_work": re.compile(
        r"(?:相关工作|related\s+works?)(?:部分|章节|section)?\s*(?:讲了?什么|介绍|总结|概述|discuss|cover|review|say)",
        re.IGNORECASE,
    ),
    "contributions": re.compile(
        r"(?:主要|核心|关键)(?:贡献|创新点)|贡献(?:是什么|有哪些|有什么)|"
        r"(?:有哪些|有什么)(?:主要|核心|关键)?(?:贡献|创新点)|"
        r"what\s+(?:are|is)\s+the\s+(?:main\s+|key\s+)?contributions?",
        re.IGNORECASE,
    ),
    "setup": re.compile(
        r"(?:实验设置|实验设计|实验配置|实现细节|experimental\s+setup|implementation\s+details?)",
        re.IGNORECASE,
    ),
    "motivation": re.compile(
        r"(?:要解决什么问题|解决了什么问题|研究动机|what\s+(?:problem|motivation))",
        re.IGNORECASE,
    ),
    "conclusion": re.compile(
        r"(?:主要|实验)结论|结论是什么|what\s+(?:are|is)\s+the\s+(?:main\s+)?conclusions?",
        re.IGNORECASE,
    ),
    "architecture": re.compile(
        r"(?:网络结构|模型结构|整体架构)\s*(?:是什么|是怎样|如何)",
        re.IGNORECASE,
    ),
    "dataset": re.compile(
        r"(?:用了?(?:哪些|什么)数据集|数据集是什么|使用了?什么数据|"
        r"what\s+(?:datasets?|data)\s+(?:does|did|is|are)|which\s+datasets?)",
        re.IGNORECASE,
    ),
}

_FACET_BLOCK_RE = re.compile(
    r"(?:表\s*\d+|table\s*\d+|准确率|f1\b|多少|highest|lowest|best\s+result|"
    r"how\s+(?:much|many)|分别是多少|提升多少)",
    re.IGNORECASE,
)
_FIGURE_IDENTITY_RE = re.compile(
    r"(?:图\s*\d+|figure\s*\d+|fig\.?\s*\d+)(?:[^。.?？]{0,24})?"
    r"(?:讲了?什么|是什么|在说|什么意思|shows?|depicts?|illustrates?|mean)",
    re.IGNORECASE,
)
_FORMULA_IDENTITY_RE = re.compile(
    r"(?:公式\s*\d+|eq(?:uation)?\.?\s*\(?\s*\d+|formula\s*\d+)",
    re.IGNORECASE,
)
_STRUCTURE_MAP_RE = re.compile(
    r"(?:论文结构|文章结构|章节安排|章节结构|有哪些章节|目录是什么|"
    r"paper\s+structure|section\s+organization|table\s+of\s+contents|"
    r"outline\s+of\s+(?:this\s+)?paper|how\s+is\s+(?:the\s+)?paper\s+organized)",
    re.IGNORECASE,
)
_OVERVIEW_BLOCK_RE = re.compile(
    r"(?:总结这篇|概括一下全文|简单概括|summarize\s+(?:this\s+)?paper|overview\s+of\s+(?:this\s+)?paper)",
    re.IGNORECASE,
)
_SKIP_HEADING_RE = re.compile(
    r"(?:references?|bibliography|acknowledg|appendix|参考文献|致谢|附录)\b",
    re.IGNORECASE,
)
_HEADING_PREFIX_RE = re.compile(
    r"^(?:section|chapter|第)?\s*[\divxDIVX\d一二三四五六七八九十]+(?:\.\d+)*\.?\s*",
    re.IGNORECASE,
)

_FACET_SUBQUESTION = {
    "methods": ("方法是什么", "What is the method?"),
    "results": ("结果是什么", "What are the results?"),
    "limitations": ("局限性是什么", "What are the limitations?"),
    "related_work": ("相关工作讲了什么", "What does related work discuss?"),
    "contributions": ("贡献是什么", "What are the contributions?"),
    "setup": ("实验设置是什么", "What is the experimental setup?"),
    "motivation": ("要解决什么问题", "What problem does this paper address?"),
    "conclusion": ("结论是什么", "What is the conclusion?"),
    "architecture": ("网络结构是什么", "What is the architecture?"),
    "dataset": ("数据集是什么", "What datasets are used?"),
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_FACET_READ_ORDER = (
    "motivation",
    "contributions",
    "related_work",
    "architecture",
    "methods",
    "setup",
    "dataset",
    "results",
    "limitations",
    "conclusion",
)
_COMPOUND_HINT_RE = re.compile(
    r"(?:分别|以及|、|"
    r"(?:贡献|方法|结果|局限|结论|相关工作|实验设置|数据集).{0,16}"
    r"(?:和|与|以及|、).{0,16}"
    r"(?:贡献|方法|结果|局限|结论|相关工作|实验设置|数据集)|"
    r"(?:contributions?|methods?|results?|limitations?|conclusions?|datasets?).{0,24}"
    r"(?:,|and).{0,24}"
    r"(?:contributions?|methods?|results?|limitations?|conclusions?|datasets?))",
    re.IGNORECASE,
)
_FACET_NOUN_RE: dict[str, re.Pattern[str]] = {
    "methods": re.compile(r"(?:核心方法|研究方法|提出的方法|(?<![各模])方法|methods?\b|approach(?:es)?\b|methodology)", re.IGNORECASE),
    "results": re.compile(r"(?:实验结果|主要结果|(?<![如])结果|findings?\b|results?\b)", re.IGNORECASE),
    "limitations": re.compile(r"(?:局限|不足|limitations?\b)", re.IGNORECASE),
    "related_work": re.compile(r"(?:相关工作|related\s+works?)", re.IGNORECASE),
    "contributions": re.compile(r"(?:贡献|创新点|contributions?\b)", re.IGNORECASE),
    "setup": re.compile(r"(?:实验设置|实验设计|实现细节|experimental\s+setup)", re.IGNORECASE),
    "conclusion": re.compile(r"(?:结论|conclusions?\b)", re.IGNORECASE),
    "architecture": re.compile(r"(?:网络结构|模型结构|整体架构|architectures?\b)", re.IGNORECASE),
    "dataset": re.compile(r"(?:数据集|datasets?\b)", re.IGNORECASE),
}


def _looks_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def normalize_heading(title: str) -> str:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    text = _HEADING_PREFIX_RE.sub("", text)
    return text.lower().strip(" :-—–")


def _ordered_facets(hits: list[str]) -> list[str]:
    rank = {name: idx for idx, name in enumerate(_FACET_READ_ORDER)}
    seen: set[str] = set()
    ordered: list[str] = []
    for facet in sorted(hits, key=lambda item: rank.get(item, 99)):
        if facet in seen:
            continue
        seen.add(facet)
        ordered.append(facet)
    return ordered


def detect_query_facets(query: str) -> list[str]:
    """识别问句要读的论文面。数值/表号问句返回空。"""
    if not query or _FACET_BLOCK_RE.search(query):
        return []
    hits: list[str] = []
    for facet, pattern in _FACET_QUERY_PATTERNS.items():
        if pattern.search(query):
            hits.append(facet)
    if len(hits) < 2 and _COMPOUND_HINT_RE.search(query):
        for facet, noun_re in _FACET_NOUN_RE.items():
            if facet in hits:
                continue
            if noun_re.search(query):
                hits.append(facet)
    return _ordered_facets(hits)


_FIGURE_NUMBER_RE = re.compile(
    r"(?:图\s*\d+|figure\s*\d+|fig\.?\s*\d+)",
    re.IGNORECASE,
)


def is_figure_identity_query(query: str) -> bool:
    if not query or _FACET_BLOCK_RE.search(query):
        return False
    if _FIGURE_IDENTITY_RE.search(query):
        return True
    # 有图号、没有表/数值抽取信号：走结构索引 / visual_search，不当数值表。
    return bool(_FIGURE_NUMBER_RE.search(query))


def is_formula_identity_query(query: str) -> bool:
    if not query or _FACET_BLOCK_RE.search(query):
        return False
    return bool(_FORMULA_IDENTITY_RE.search(query))


def is_structure_map_query(query: str) -> bool:
    if not query or _OVERVIEW_BLOCK_RE.search(query):
        return False
    return bool(_STRUCTURE_MAP_RE.search(query))


def heading_matches_facet(title: str, facet: str) -> bool:
    normalized = normalize_heading(title)
    if not normalized or _SKIP_HEADING_RE.search(normalized):
        return False
    aliases = FACET_HEADING_ALIASES.get(facet) or ()
    # 只做 alias ⊂ 标题。反向匹配会让短标题 "Work"/"Data"/"Net"
    # 误打到 related work / dataset / network。
    return any(alias in normalized for alias in aliases)


def heading_matches_any_facet(title: str, facets: Iterable[str]) -> bool:
    return any(heading_matches_facet(title, facet) for facet in facets)


def outline_entries_from_block_index(block_index: Any) -> list[dict[str, Any]]:
    outline = block_index.get("outline") if isinstance(block_index, dict) else []
    if not isinstance(outline, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw in outline:
        if not isinstance(raw, dict):
            continue
        section_id = str(raw.get("section_id") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not section_id or not title:
            continue
        try:
            level = max(1, min(int(raw.get("level") or 1), 6))
        except (TypeError, ValueError):
            level = 1
        entries.append({"section_id": section_id, "title": title, "level": level})
    return entries


def match_outline_sections(
    query: str,
    outline: Iterable[dict[str, Any]] | None,
    *,
    limit: int = 4,
) -> list[dict[str, str]]:
    """把问句 facet 对到大纲里的真实标题。"""
    facets = detect_query_facets(query)
    if not facets:
        return []
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    ranked = sorted(
        (item for item in (outline or []) if isinstance(item, dict)),
        key=lambda item: (int(item.get("level") or 9), str(item.get("section_id") or "")),
    )
    for entry in ranked:
        section_id = str(entry.get("section_id") or "").strip()
        title = str(entry.get("title") or "").strip()
        if not section_id or section_id in seen:
            continue
        hit_facet = next((facet for facet in facets if heading_matches_facet(title, facet)), "")
        if not hit_facet:
            continue
        seen.add(section_id)
        matches.append({
            "section_id": section_id,
            "title": title,
            "facet": hit_facet,
        })
        if len(matches) >= limit:
            break
    return matches


def rule_decompose_facets(query: str, *, limit: int = 4) -> list[str]:
    """复合身份问句拆成独立子问。单 facet / 数值表不拆。"""
    facets = detect_query_facets(query)
    if len(facets) < 2:
        return []
    chinese = _looks_chinese(query)
    questions: list[str] = []
    seen: set[str] = set()
    for facet in facets[:limit]:
        pair = _FACET_SUBQUESTION.get(facet)
        if not pair:
            continue
        item = pair[0] if chinese else pair[1]
        if item in seen:
            continue
        seen.add(item)
        questions.append(item)
    return questions if len(questions) >= 2 else []
