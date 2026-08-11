"""Canonical intent constraints shared by rewrite, decomposition and Planner.

The user question remains the authority. Downstream model calls may rephrase or
split it, but they may not silently change document entities, numeric locators,
page/figure references, task polarity or scope. This module is deliberately
pure and dependency-free so every model boundary can apply the same contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Sequence


INTENT_CONSTRAINT_SCHEMA_VERSION = "intent_constraints.v1"


_TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summarize", re.compile(r"总结|概述|摘要|概览|summari[sz]e|summary|overview", re.IGNORECASE)),
    ("translate", re.compile(r"翻译|译文|translate|translation", re.IGNORECASE)),
    ("compare", re.compile(r"比较|对比|区别|差异|异同|compare|comparison|contrast|versus|\bvs\.?\b", re.IGNORECASE)),
    ("explain", re.compile(r"解释|说明|为什么|为何|如何|怎么|原理|explain|why|how|mechanism", re.IGNORECASE)),
    ("extract", re.compile(r"提取|找出|给出|列出|数值|extract|list|exact|value|number", re.IGNORECASE)),
    ("inventory", re.compile(r"全部|所有|每个|列出|哪些|多少个|\ball\b|\bevery\b|inventory", re.IGNORECASE)),
    ("figure", re.compile(r"图表|图片|图\s*\d+|figure|fig(?:ure)?|chart|plot", re.IGNORECASE)),
    ("table", re.compile(r"表格|表\s*\d+|table", re.IGNORECASE)),
    ("formula", re.compile(r"公式|方程|损失函数|equation|formula|loss", re.IGNORECASE)),
    ("reference", re.compile(r"参考文献|引用|doi|arxiv|bibliography|reference|citation", re.IGNORECASE)),
    ("limitation", re.compile(r"局限|限制|缺点|不足|limitation|drawback|weakness", re.IGNORECASE)),
)

_NEGATION_RE = re.compile(
    r"不要|不用|无需|别(?:再)?|未|没有|不需要|不(?:要|用|需)|"
    r"\b(?:do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|not|no|without|skip)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[，,。.!！？?；;]|\b(?:but|however|then)\b", re.IGNORECASE)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])")
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*(?:[-_/+][A-Za-z0-9]+)+|[A-Z]{2,}[A-Z0-9]*)\b"
)
_REFERENCE_RE = re.compile(
    r"(?:\b(?:table|fig(?:ure)?|section|appendix|page|eq(?:uation)?)\s*[A-Za-z]?\d+(?:\.\d+)*[A-Za-z]?\b|"
    r"(?:表|图|公式|方程|章节|第)\s*[A-Za-z]?\d+(?:\.\d+)*[A-Za-z]?(?:\s*(?:页|章|节))?)",
    re.IGNORECASE,
)
_PAGE_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"第?\s*(\d+)\s*(?:到|至|[-~～—])\s*(\d+)\s*页"),
    re.compile(r"第?\s*(\d+)\s*页(?!\s*(?:到|至|[-~～—]))"),
    re.compile(r"\bpages?\s*(\d+)\s*(?:to|through|[-~–—])\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpages?\s*(\d+)\b(?!\s*(?:to|through|[-~–—]))", re.IGNORECASE),
)
_QUOTED_ENTITY_RE = re.compile(r"[\"'“”‘’`《》]([^\"'“”‘’`《》]{2,80})[\"'“”‘’`《》]")
_CJK_ENTITY_RE = re.compile(
    r"[A-Za-z0-9\u4e00-\u9fff_-]{1,28}(?:模型|方法|算法|模块|网络|数据集|指标|损失|架构|系统|章节|实验)"
)

_EN_COMPARE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpros\s+and\s+cons\s+of\s+([^,.?]{1,80}?)\s+and\s+([^,.?]{1,80}?)(?:\s+as\s+described|[,.?]|$)", re.IGNORECASE),
    re.compile(r"\b(?:compare|contrast)\s+([^,.?]{1,80}?)\s+(?:and|with|versus|vs\.?)\s+([^,.?]{1,80}?)(?:[,.?]|$)", re.IGNORECASE),
    re.compile(r"\b(?:differences?|distinctions?)\s+between\s+([^,.?]{1,80}?)\s+and\s+([^,.?]{1,80}?)(?:[,.?]|$)", re.IGNORECASE),
    re.compile(r"\b([^,.?]{1,60}?)\s+(?:versus|vs\.?)\s+([^,.?]{1,60}?)(?:[,.?]|$)", re.IGNORECASE),
)
_ZH_COMPARE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:比较|对比)\s*([^，。！？；]{1,28}?)\s*(?:和|与|跟)\s*([^，。！？；]{1,28}?)(?:的|在|有|，|。|！|？|；|$)"),
    re.compile(r"([^，。！？；]{1,28}?)\s*(?:和|与|跟)\s*([^，。！？；]{1,28}?)\s*(?:的)?(?:区别|差异|异同|优缺点|优劣)"),
    re.compile(r"([^，。！？；]{1,28}?)\s*(?:相比|相较于|比起)\s*([^，。！？；]{1,28}?)(?:，|。|！|？|；|$)"),
)

_SCOPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("all", re.compile(r"全部|所有|每个|完整|全文|\b(?:all|every|entire|complete)\b", re.IGNORECASE)),
    ("only", re.compile(r"仅|只|只要|限定|\b(?:only|solely|exclusively)\b", re.IGNORECASE)),
    ("exact", re.compile(r"精确|准确数值|原文|逐字|\b(?:exact|verbatim|precise)\b", re.IGNORECASE)),
    ("exclude", re.compile(r"不包括|排除|除了.+不|\b(?:exclude|excluding|except|without)\b", re.IGNORECASE)),
    ("document", re.compile(r"本文|本论文|该论文|文中|\b(?:this\s+paper|the\s+paper|this\s+document)\b", re.IGNORECASE)),
    ("selected", re.compile(r"选中|划词|这段|该段|\b(?:selected\s+text|this\s+passage)\b", re.IGNORECASE)),
)

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._/-]*")
_LATIN_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "can", "could", "would", "should", "may", "might",
    "of", "in", "on", "at", "by", "for", "to", "from", "with", "within", "about", "as", "into", "over",
    "under", "between", "and", "or", "but", "than", "then", "so", "if", "please", "paper", "document", "author",
    "according", "described", "reported", "used", "using", "proposed", "main", "different", "various", "each", "both",
    "one", "ones",
})
_GENERIC_RETRIEVAL_TOKENS = frozenset({
    "method", "methods", "approach", "approaches", "model", "models", "architecture", "mechanism", "implementation",
    "experiment", "experiments", "result", "results", "performance", "metric", "metrics", "dataset", "datasets",
    "accuracy", "precision", "recall", "comparison", "difference", "differences", "advantage", "advantages",
    "disadvantage", "disadvantages", "limitation", "limitations", "contribution", "contributions", "evidence",
    "definition", "detail", "details", "reason", "reasons", "effect", "effects", "value", "values", "number", "numbers",
    "table", "figure", "section", "page", "formula", "equation", "loss", "training", "evaluation", "ablation",
    "summary", "overview", "explanation", "analysis", "paper", "document", "text", "content", "exact", "related",
    "compare", "contrast", "versus", "explain", "describe", "summarize", "translate", "extract", "list", "show",
    "find", "give", "provide", "identify", "discuss", "evaluate", "why", "how", "only", "all", "every", "without",
    "latest", "official", "source", "sources", "information", "diagram", "chart", "plot", "image", "visual",
})

_TOOL_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "search_document": ("query", "keywords", "exactQuery"),
    "vector_search": ("query",),
    "keyword_search": ("keywords",),
    "grep": ("query",),
    "regex_search": ("pattern",),
    "boolean_search": ("query",),
    "visual_search": ("query", "reference"),
    "web_search": ("query",),
}
_SUBQUESTION_SAFE_TASK_TAGS = frozenset({"explain", "extract", "inventory"})


def _clean_text(value: Any, limit: int = 4000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _normalize_phrase(value: str) -> str:
    text = _clean_text(value, 160).casefold()
    text = re.sub(r"^[\s:：,，.。;；?？!！的]+|[\s:：,，.。;；?？!！]+$", "", text)
    text = re.sub(r"^(?:the|a|an|this|that)\s+", "", text)
    text = re.sub(
        r"\s+(?:on|in|from|within|according\s+to)\s+(?:pages?|sections?|appendix|this\s+paper|the\s+paper)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+as\s+described(?:\s+in\s+this\s+paper)?\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:在|位于)?第?\s*\d+(?:\s*(?:到|至|[-~～—])\s*\d+)?\s*(?:页|章|节).*$", "", text)
    return re.sub(r"\s+", " ", text)


def _normalize_token(value: str) -> str:
    token = str(value or "").casefold().strip("._/-+#")
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _task_tags(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _TASK_PATTERNS if pattern.search(text))


def _task_polarities(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for name, pattern in _TASK_PATTERNS:
        for match in pattern.finditer(text):
            clause_start = 0
            for boundary in _CLAUSE_BOUNDARY_RE.finditer(text, 0, match.start()):
                clause_start = boundary.end()
            prefix = text[clause_start:match.start()]
            prohibited = bool(_NEGATION_RE.search(prefix[-28:]))
            values.append(f"{name}:{'prohibited' if prohibited else 'requested'}")
            break
    return tuple(sorted(set(values)))


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(0).casefold() for match in _NUMBER_RE.finditer(text)}))


def _references(text: str) -> tuple[str, ...]:
    values = {
        re.sub(r"\s+", "", match.group(0)).casefold()
        for match in _REFERENCE_RE.finditer(text)
    }
    return tuple(sorted(values))


def _page_ranges(text: str) -> tuple[tuple[int, int], ...]:
    values: set[tuple[int, int]] = set()
    occupied: list[tuple[int, int]] = []
    for pattern in _PAGE_RANGE_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex == 1 and any(
                match.start() >= start and match.end() <= end
                for start, end in occupied
            ):
                continue
            start = max(1, int(match.group(1)))
            end = max(1, int(match.group(2))) if match.lastindex and match.lastindex >= 2 and match.group(2) else start
            values.add((min(start, end), max(start, end)))
            if match.lastindex and match.lastindex >= 2:
                occupied.append((match.start(), match.end()))
    return tuple(sorted(values))


def _identifiers(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(0).casefold() for match in _IDENTIFIER_RE.finditer(text)}))


def _comparison_objects(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for pattern in (*_EN_COMPARE_PATTERNS, *_ZH_COMPARE_PATTERNS):
        for match in pattern.finditer(text):
            pair = tuple(_normalize_phrase(raw) for raw in match.groups()[:2])
            if any(
                re.match(
                    r"^(?:explain|describe|list|summari[sz]e|analy[sz]e|discuss|why|how)\b",
                    value,
                    re.IGNORECASE,
                )
                or value in {"pros", "cons", "advantages", "disadvantages"}
                for value in pair
            ):
                continue
            for value in pair:
                if value and value not in values:
                    values.append(value)
    return tuple(values)


def _scope_terms(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _SCOPE_PATTERNS if pattern.search(text))


def _content_tokens(text: str) -> tuple[str, ...]:
    tokens = {
        _normalize_token(match.group(0))
        for match in _LATIN_TOKEN_RE.finditer(text)
    }
    tokens = {
        token for token in tokens
        if len(token) >= 2
        and token not in _LATIN_STOPWORDS
        and token not in _GENERIC_RETRIEVAL_TOKENS
    }
    for pattern in (_QUOTED_ENTITY_RE, _CJK_ENTITY_RE):
        for match in pattern.finditer(text):
            value = _normalize_phrase(match.group(1) if pattern is _QUOTED_ENTITY_RE else match.group(0))
            if value:
                tokens.add(value)
    return tuple(sorted(tokens))


def _language_family(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk and latin:
        return "mixed"
    if cjk:
        return "cjk"
    if latin:
        return "latin"
    return "unknown"


def _extract_text_fields(args: dict[str, Any], fields: Iterable[str]) -> str:
    values: list[str] = []
    for field in fields:
        raw = args.get(field)
        if isinstance(raw, (list, tuple)):
            values.extend(_clean_text(item, 320) for item in raw if _clean_text(item, 320))
        elif _clean_text(raw, 800):
            values.append(_clean_text(raw, 800))
    return " ".join(values)


@dataclass(frozen=True)
class ConstraintValidation:
    allowed: bool
    violations: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    introduced: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations": list(self.violations),
            "missing": list(self.missing),
            "introduced": list(self.introduced),
        }


@dataclass(frozen=True)
class IntentConstraintSet:
    source_text: str
    language: str
    task_tags: tuple[str, ...]
    task_polarities: tuple[str, ...]
    numbers: tuple[str, ...]
    page_ranges: tuple[tuple[int, int], ...]
    references: tuple[str, ...]
    identifiers: tuple[str, ...]
    comparison_objects: tuple[str, ...]
    scope_terms: tuple[str, ...]
    content_tokens: tuple[str, ...]
    allowed_context_numbers: tuple[str, ...] = ()
    allowed_context_references: tuple[str, ...] = ()
    allowed_context_identifiers: tuple[str, ...] = ()
    allowed_context_objects: tuple[str, ...] = ()
    allowed_context_tokens: tuple[str, ...] = ()
    allowed_context_phrases: tuple[str, ...] = ()
    has_allowed_context: bool = False
    schema_version: str = INTENT_CONSTRAINT_SCHEMA_VERSION

    @classmethod
    def from_text(
        cls,
        source_text: str,
        *,
        allowed_context: Sequence[str] | None = None,
    ) -> "IntentConstraintSet":
        source = _clean_text(source_text)
        context_items = tuple(
            _clean_text(item, 1200)
            for item in (allowed_context or [])
            if _clean_text(item, 1200)
        )
        context_text = " ".join(context_items)
        return cls(
            source_text=source,
            language=_language_family(source),
            task_tags=_task_tags(source),
            task_polarities=_task_polarities(source),
            numbers=_numbers(source),
            page_ranges=_page_ranges(source),
            references=_references(source),
            identifiers=_identifiers(source),
            comparison_objects=_comparison_objects(source),
            scope_terms=_scope_terms(source),
            content_tokens=_content_tokens(source),
            allowed_context_numbers=_numbers(context_text),
            allowed_context_references=_references(context_text),
            allowed_context_identifiers=_identifiers(context_text),
            allowed_context_objects=_comparison_objects(context_text),
            allowed_context_tokens=_content_tokens(context_text),
            allowed_context_phrases=tuple(
                phrase for phrase in (_normalize_phrase(item) for item in context_items) if phrase
            ),
            has_allowed_context=bool(context_text),
        )

    @property
    def constraint_id(self) -> str:
        payload = self.to_dict(include_source=False)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "ic_" + hashlib.sha256(encoded).hexdigest()[:20]

    def to_dict(self, *, include_source: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "language": self.language,
            "task_tags": list(self.task_tags),
            "task_polarities": list(self.task_polarities),
            "numbers": list(self.numbers),
            "page_ranges": [list(item) for item in self.page_ranges],
            "references": list(self.references),
            "identifiers": list(self.identifiers),
            "comparison_objects": list(self.comparison_objects),
            "scope_terms": list(self.scope_terms),
            "content_tokens": list(self.content_tokens),
        }
        if include_source:
            payload["source_text"] = self.source_text
        payload["constraint_id"] = self.constraint_id if include_source else ""
        if not include_source:
            payload.pop("constraint_id", None)
        return payload

    def prompt_guard(self) -> str:
        protected = {
            "constraint_id": self.constraint_id,
            "task_polarities": list(self.task_polarities),
            "numbers": list(self.numbers),
            "page_ranges": [list(item) for item in self.page_ranges],
            "references": list(self.references),
            "identifiers": list(self.identifiers),
            "comparison_objects": list(self.comparison_objects),
            "scope_terms": list(self.scope_terms),
            "entities": list(self.content_tokens),
        }
        return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))

    def validate_rewrite(self, candidate: str) -> ConstraintValidation:
        return self._validate_text(candidate, require_complete=True, exact_tasks=True)

    def validate_enrichment(self, candidate: str) -> ConstraintValidation:
        """Validate deterministic retrieval enrichment.

        Enrichment may append generic retrieval task words, but every original
        task polarity and every hard entity/locator/scope must survive.
        """
        validation = self._validate_text(
            candidate,
            require_complete=True,
            exact_tasks=False,
            implicit_scope_terms={"selected"} if self.has_allowed_context else None,
            allow_introduced_tasks=True,
        )
        if not validation.allowed:
            return validation
        observed = set(_task_polarities(_clean_text(candidate)))
        missing_polarities = set(self.task_polarities) - observed
        if missing_polarities:
            return ConstraintValidation(
                False,
                ("missing_task_polarity",),
                tuple(sorted(f"polarity:{item}" for item in missing_polarities)),
            )
        return validation

    def validate_subquestions(self, sub_questions: Sequence[str]) -> ConstraintValidation:
        items = [_clean_text(item, 400) for item in sub_questions if _clean_text(item, 400)]
        if not (2 <= len(items) <= 3):
            return ConstraintValidation(False, ("subquestion_count",))
        candidate = " ".join(items)
        if len(candidate) > int(len(self.source_text) * 2.8) + 60:
            return ConstraintValidation(False, ("subquestions_expanded",))
        return self._validate_text(
            candidate,
            require_complete=True,
            exact_tasks=False,
            implicit_scope_terms={"document"},
            allowed_introduced_tasks=_SUBQUESTION_SAFE_TASK_TAGS,
        )

    def validate_tool_arguments(self, tool_name: str, args: dict[str, Any]) -> ConstraintValidation:
        name = str(tool_name or "").strip()
        if not isinstance(args, dict):
            return ConstraintValidation(False, ("tool_args_not_object",))

        text = _extract_text_fields(args, _TOOL_TEXT_FIELDS.get(name, ()))
        if text:
            validation = self._validate_text(
                text,
                require_complete=False,
                exact_tasks=False,
                allow_introduced_tasks=True,
                allow_language_change=True,
            )
            # A generic question such as "Explain the method" has no identity
            # anchor to preserve. Planner terms then narrow retrieval rather
            # than replace a named subject, so do not collapse every new word
            # back to the root question and accidentally deduplicate retries.
            if (
                not validation.allowed
                and set(validation.violations) == {"introduced_entity"}
                and not self._has_hard_tool_identity_anchor()
            ):
                validation = ConstraintValidation(True)
            if not validation.allowed:
                return validation

        if name in {"visual_search", "read_blocks"}:
            try:
                page = max(0, int(args.get("page") or 0))
            except (TypeError, ValueError):
                page = 0
            if page and self.page_ranges:
                if not any(start <= page <= end for start, end in self.page_ranges):
                    return ConstraintValidation(False, ("introduced_page_scope",), introduced=(str(page),))
        return ConstraintValidation(True)

    def _has_hard_tool_identity_anchor(self) -> bool:
        """Whether a tool query must retain a user-specified identity.

        Lowercase retrieval focus words are intentionally not sufficient here:
        they are often generated from a generic question to broaden a search.
        Explicit identifiers, references, numeric locators, and comparison
        operands remain hard constraints.
        """

        return bool(
            self.identifiers
            or self.references
            or self.numbers
            or self.page_ranges
            or self.comparison_objects
        )

    def repair_tool_arguments(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a schema-compatible root-intent fallback for text tools.

        The repair never invents a locator. It keeps bounded execution controls
        (limit/strategy) but replaces model-authored semantic fields with the
        frozen root question and its explicit protected identifiers.
        """
        name = str(tool_name or "").strip()
        if not isinstance(args, dict) or name not in _TOOL_TEXT_FIELDS:
            return None
        repaired = dict(args)
        root = self.source_text[:800]
        protected_terms = list(dict.fromkeys([
            *self.references,
            *self.identifiers,
            *self.comparison_objects,
            *self.content_tokens,
        ]))
        if name == "search_document":
            repaired["query"] = root
            repaired["keywords"] = protected_terms[:16]
            repaired["exactQuery"] = "|".join(protected_terms[:12])[:260]
        elif name == "keyword_search":
            repaired["keywords"] = protected_terms[:16] or [root[:160]]
        elif name == "visual_search":
            repaired["query"] = root
            repaired["reference"] = self.references[0] if self.references else ""
            repaired["page"] = self.page_ranges[0][0] if self.page_ranges else 0
        elif name == "regex_search":
            repaired["pattern"] = re.escape(root[:160])
        else:
            field = _TOOL_TEXT_FIELDS[name][0]
            repaired[field] = root

        validation = self.validate_tool_arguments(name, repaired)
        return repaired if validation.allowed else None

    def _validate_text(
        self,
        candidate: str,
        *,
        require_complete: bool,
        exact_tasks: bool,
        implicit_scope_terms: set[str] | None = None,
        allow_introduced_tasks: bool = False,
        allowed_introduced_tasks: frozenset[str] = frozenset(),
        allow_language_change: bool = False,
    ) -> ConstraintValidation:
        text = _clean_text(candidate)
        if not text:
            return ConstraintValidation(False, ("empty_candidate",))

        violations: list[str] = []
        missing: set[str] = set()
        introduced: set[str] = set()
        candidate_language = _language_family(text)
        if (
            not allow_language_change
            and
            self.language in {"cjk", "latin"}
            and candidate_language in {"cjk", "latin"}
            and candidate_language != self.language
        ):
            violations.append("language_changed")

        candidate_tasks = set(_task_tags(text))
        source_tasks = set(self.task_tags)
        if exact_tasks and candidate_tasks != source_tasks:
            violations.append("task_tags_changed")
            missing.update(f"task:{value}" for value in source_tasks - candidate_tasks)
            introduced.update(f"task:{value}" for value in candidate_tasks - source_tasks)
        elif (
            not allow_introduced_tasks
            and candidate_tasks - source_tasks - set(allowed_introduced_tasks)
        ):
            violations.append("introduced_task")
            introduced.update(
                f"task:{value}"
                for value in candidate_tasks - source_tasks - set(allowed_introduced_tasks)
            )

        candidate_polarities = set(_task_polarities(text))
        source_polarities = set(self.task_polarities)
        if exact_tasks and candidate_polarities != source_polarities:
            violations.append("task_polarity_changed")
            missing.update(f"polarity:{value}" for value in source_polarities - candidate_polarities)
            introduced.update(f"polarity:{value}" for value in candidate_polarities - source_polarities)

        normalized_candidate = _normalize_phrase(text)
        candidate_comparison_objects = set(_comparison_objects(text))
        candidate_comparison_objects.update(
            item for item in self.comparison_objects if item and item in normalized_candidate
        )
        checks = (
            ("number", set(self.numbers), set(_numbers(text)), set(self.allowed_context_numbers)),
            ("reference", set(self.references), set(_references(text)), set(self.allowed_context_references)),
            ("identifier", set(self.identifiers), set(_identifiers(text)), set(self.allowed_context_identifiers)),
            ("comparison_object", set(self.comparison_objects), candidate_comparison_objects, set(self.allowed_context_objects)),
            ("entity", set(self.content_tokens), set(_content_tokens(text)), set(self.allowed_context_tokens)),
            ("scope", set(self.scope_terms), set(_scope_terms(text)), set()),
        )
        for label, required, observed, allowed_context in checks:
            if label == "scope" and implicit_scope_terms:
                required = required - set(implicit_scope_terms)
            if require_complete and required - observed:
                violations.append(f"missing_{label}")
                missing.update(f"{label}:{value}" for value in required - observed)
            extra = observed - required - allowed_context
            if label == "entity":
                extra = {value for value in extra if value not in _GENERIC_RETRIEVAL_TOKENS}
                # Chinese tokenization can join resolved selected text with its
                # neighboring query characters. The exact context phrase is
                # still authorized even when it is embedded in such a token.
                if self.has_allowed_context and self.allowed_context_phrases:
                    extra = {
                        value
                        for value in extra
                        if not any(
                            phrase in value or value in phrase
                            for phrase in self.allowed_context_phrases
                        )
                    }
            if extra:
                violations.append(f"introduced_{label}")
                introduced.update(f"{label}:{value}" for value in extra)

        candidate_pages = set(_page_ranges(text))
        source_pages = set(self.page_ranges)
        if require_complete and source_pages - candidate_pages:
            violations.append("missing_page_range")
            missing.update(f"page:{start}-{end}" for start, end in source_pages - candidate_pages)
        if candidate_pages - source_pages:
            violations.append("introduced_page_range")
            introduced.update(f"page:{start}-{end}" for start, end in candidate_pages - source_pages)

        return ConstraintValidation(
            allowed=not violations,
            violations=tuple(dict.fromkeys(violations)),
            missing=tuple(sorted(missing)),
            introduced=tuple(sorted(introduced)),
        )


def validate_rewrite(
    source: str,
    candidate: str,
    *,
    allowed_context: Sequence[str] | None = None,
) -> ConstraintValidation:
    return IntentConstraintSet.from_text(source, allowed_context=allowed_context).validate_rewrite(candidate)


def validate_subquestions(source: str, sub_questions: Sequence[str]) -> ConstraintValidation:
    return IntentConstraintSet.from_text(source).validate_subquestions(sub_questions)
