"""单篇学术轻量图（Method / Dataset / Metric / Model 实体与关系）。

目标：为 explain / compare / analytical 问句提供「概念关系骨架」，
不替代 GraphRAG 全量索引，也不做跨文档库。

实体类型：
- method / model / dataset / metric / baseline / concept

关系类型（固定 ontology）：
- proposes      本文提出
- uses          使用/采用
- evaluates_on  在…上评估
- outperforms   优于
- compares_with 与…比较
- reports       报告指标

抽取策略：纯本地启发式（标题、文首、大纲、术语共现），默认离线、可缓存。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

ACADEMIC_GRAPH_VERSION = "v1"

ENTITY_TYPES = ("method", "model", "dataset", "metric", "baseline", "concept")
EDGE_TYPES = (
    "proposes",
    "uses",
    "evaluates_on",
    "outperforms",
    "compares_with",
    "reports",
)

_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,6})?)\b")
_PROPER_PHRASE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:[\- ][A-Z][A-Za-z0-9]+){0,4})\b"
)
_METRIC_NAME_RE = re.compile(
    r"\b(accuracy|acc\.?|precision|recall|f1(?:-score)?|bleu|rouge(?:-?[l12n])?|"
    r"mAP|mIoU|IoU|AUC|RMSE|MAE|perplexity|PPL|latency|throughput|"
    r"准确率|精确率|召回率|得分|指标)\b",
    re.IGNORECASE,
)
_DATASET_CUE_RE = re.compile(
    r"\b(ImageNet(?:-?\w+)?|CIFAR-?\d+|COCO|GLUE|SQuAD|MNIST|WikiText|BookCorpus|"
    r"MultiNLI|MNLI|QNLI|QQP|SST-?2|CoLA|RTE|MRPC|STS-B|SuperGLUE|"
    r"LibriSpeech|WMT\d*|Penn Treebank|PTB|MS MARCO|HotpotQA|Natural Questions|"
    r"数据集|基准|benchmark)\b",
    re.IGNORECASE,
)
_METHOD_CUE_RE = re.compile(
    r"(?:propos(?:e|es|ed)|introduc(?:e|es|ed)|present(?:s|ed)?|we\s+(?:propose|present|introduce)|"
    r"提出|引入|本文方法|our\s+(?:method|approach|model|framework))",
    re.IGNORECASE,
)
_USES_CUE_RE = re.compile(
    r"(?:based\s+on|built\s+on|using|uses|with|via|adopt(?:s|ed)?|employ(?:s|ed)?|"
    r"基于|采用|使用|利用)",
    re.IGNORECASE,
)
_EVAL_CUE_RE = re.compile(
    r"(?:evaluat(?:e|ed|ion)|experiment(?:s|al)?|benchmark(?:ed|ing)?|on\s+[A-Z]|"
    r"评估|实验|基准|测试集|验证集)",
    re.IGNORECASE,
)
_OUTPERFORM_CUE_RE = re.compile(
    r"(?:outperform(?:s|ed)?|surpass(?:es|ed)?|better\s+than|superior\s+to|"
    r"improves?\s+over|beats?|高于|优于|超过|提升)",
    re.IGNORECASE,
)
_COMPARE_CUE_RE = re.compile(
    r"(?:compar(?:e|ed|ison)\s+(?:with|to|against)|versus|\bvs\.?\b|相比|对比|对照)",
    re.IGNORECASE,
)

_STOP_ACRONYMS = {
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "CAN", "MAY",
    "ALL", "USE", "NEW", "SET", "GET", "PUT", "PDF", "URL", "HTTP", "HTTPS",
    "JSON", "API", "GPU", "CPU", "NLP", "NLU", "ML", "AI", "ID", "EQ", "FIG",
    "TAB", "SEC", "EQS", "AKA", "ETC", "IEEE", "ACM", "CVPR", "ICLR", "NEURIPS",
    "NIPS", "AAAI", "ACL", "EMNLP", "NAACL", "ICML", "ICCV", "ECCV",
}

_STOP_PHRASES = {
    "Abstract", "Introduction", "Related Work", "Conclusion", "References",
    "Acknowledgement", "Appendix", "Table", "Figure", "Equation", "Method",
    "Experiments", "Results", "Discussion", "Background", "Future Work",
    "This", "Our", "We", "The", "These", "Those", "Their", "Its", "They", "Since",
    "Experimental", "However", "Therefore", "Moreover", "Finally", "Overall",
    "In", "On", "As", "If", "When", "While", "Where", "Which", "What", "Who",
    "Problem", "Definition", "Section", "Chapter", "Paper", "Work", "Model",
    "Approach", "Framework", "System", "Network", "Layer", "Output", "Input",
    "Knowledge", "Information", "Data", "Training", "Testing", "Learning",
    "Usually", "With", "Without", "Using", "Based", "From", "Into", "Over",
    "Under", "After", "Before", "During", "Between", "Among", "Such", "Each",
}

_STOP_NAME_RE = re.compile(
    r"^(?:this|our|we|the|these|those|their|its|they|since|a|an|in|on|of|to|for|and|or|"
    r"if|when|while|where|which|what|who|experimental|however|therefore|moreover|"
    r"finally|overall|paper|work|knowledge|information|data|training|testing|learning)$",
    re.IGNORECASE,
)


@dataclass
class AcademicEntity:
    entity_id: str
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    mentions: int = 1
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcademicEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    evidence: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcademicGraph:
    version: str = ACADEMIC_GRAPH_VERSION
    doc_id: str = ""
    entities: list[AcademicEntity] = field(default_factory=list)
    edges: list[AcademicEdge] = field(default_factory=list)
    parse_generation: str = ""
    document_source_hash: str = ""
    built_at: str = ""
    source: str = "heuristic"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "doc_id": self.doc_id,
            "entities": [e.to_dict() for e in self.entities],
            "edges": [e.to_dict() for e in self.edges],
            "parse_generation": self.parse_generation,
            "document_source_hash": self.document_source_hash,
            "built_at": self.built_at,
            "source": self.source,
            "confidence": self.confidence,
            "entity_count": len(self.entities),
            "edge_count": len(self.edges),
        }


def should_use_academic_graph(
    *,
    task: str = "",
    query_type: str = "",
    evidence_need: Sequence[str] | None = None,
    graph_mode: str = "",
) -> bool:
    """Only inject for explain/compare/analytical-style turns."""
    task_l = str(task or "").strip().lower()
    qtype = str(query_type or "").strip().lower()
    needs = {str(x).strip() for x in (evidence_need or []) if str(x).strip()}
    mode = str(graph_mode or "").strip().lower()
    if task_l in {"explain", "compare"}:
        return True
    if qtype in {"analytical", "overview"} and task_l in {"qa", "explain", "compare", "summarize"}:
        # overview summarize can benefit from method skeleton lightly
        if qtype == "overview" and task_l == "summarize":
            return True
        if qtype == "analytical":
            return True
    if needs & {
        "section_explanation",
        "analysis_explanation",
        "comparison_multi_aspect",
    }:
        return True
    if mode in {"hybrid", "global"} and task_l != "extract":
        return True
    return False


def get_academic_graph_dir(data_dir: Path | str) -> Path:
    path = Path(data_dir) / "academic_graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_academic_graph_path(data_dir: Path | str, doc_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._\-]", "_", str(doc_id or "doc"))[:120]
    return get_academic_graph_dir(data_dir) / f"{safe}.json"


def load_academic_graph(data_dir: Path | str, doc_id: str) -> dict[str, Any] | None:
    path = get_academic_graph_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_academic_graph(data_dir: Path | str, doc_id: str, graph: dict[str, Any]) -> Path:
    path = get_academic_graph_path(data_dir, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _entity_id(name: str, entity_type: str) -> str:
    digest = hashlib.sha1(f"{entity_type}:{name.casefold()}".encode("utf-8")).hexdigest()[:10]
    return f"{entity_type}:{digest}"


def _edge_id(source_id: str, target_id: str, relation: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{relation}|{target_id}".encode("utf-8")).hexdigest()[:10]
    return f"edge:{digest}"


def _normalize_name(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip(" ,.;:()[]{}\"'")
    return text[:80]


def _classify_entity(name: str, context: str = "") -> str:
    raw = _normalize_name(name)
    lower = raw.casefold()
    ctx = f"{raw} {context}"
    if _METRIC_NAME_RE.fullmatch(raw) or _METRIC_NAME_RE.search(raw):
        if len(raw) <= 24:
            return "metric"
    if _DATASET_CUE_RE.search(raw) or _DATASET_CUE_RE.search(ctx[:200]):
        # "GLUE tasks" style
        if re.search(r"dataset|benchmark|glue|imagenet|cifar|coco|squad", lower):
            return "dataset"
        if raw.isupper() or re.search(r"[A-Z]{2,}", raw):
            # could still be model; check dataset cues nearby in context
            if _DATASET_CUE_RE.search(context):
                return "dataset"
    if re.search(r"\b(baseline|vanilla|without|w/?o)\b", lower):
        return "baseline"
    if re.search(r"[-_/]?(BERT|GPT|T5|LSTM|CNN|Transformer|ResNet|ViT)\b", raw, re.I):
        return "model"
    if re.search(r"\b(KD|distillation|attention|projection|framework|approach|algorithm)\b", lower):
        return "method"
    if raw.isupper() and 2 <= len(raw) <= 12:
        # acronyms near propose -> method, near dataset cue -> dataset
        if _METHOD_CUE_RE.search(context):
            return "method"
        if _DATASET_CUE_RE.search(context):
            return "dataset"
        return "model"
    if re.match(r"^[A-Z][A-Za-z0-9]+(?:-[A-Z0-9]+)+$", raw):
        return "method"
    return "concept"


def _collect_outline_texts(outline: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(outline, dict):
        return texts
    title = str(outline.get("title") or "").strip()
    if title:
        texts.append(title)
    items = outline.get("items") or outline.get("sections") or []
    if not isinstance(items, list):
        return texts
    for item in items[:40]:
        if not isinstance(item, dict):
            continue
        for key in ("title", "translated_title", "summary"):
            value = str(item.get(key) or "").strip()
            if value:
                texts.append(value)
        study = item.get("study") if isinstance(item.get("study"), dict) else {}
        for key in ("purpose", "method_or_argument", "caveats_or_connections"):
            value = str(study.get(key) or "").strip()
            if value:
                texts.append(value)
        findings = study.get("findings") or item.get("findings") or []
        if isinstance(findings, list):
            for finding in findings[:5]:
                if isinstance(finding, str) and finding.strip():
                    texts.append(finding.strip())
                elif isinstance(finding, dict):
                    texts.append(str(finding.get("text") or finding.get("label") or "").strip())
    return [t for t in texts if t]


def _sample_text(full_text: str, *, max_chars: int = 12000) -> str:
    text = str(full_text or "")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + "\n...\n" + tail


def _add_entity(
    bucket: dict[str, AcademicEntity],
    name: str,
    entity_type: str,
    *,
    source: str,
    context: str = "",
) -> Optional[AcademicEntity]:
    clean = _normalize_name(name)
    if not clean or len(clean) < 2:
        return None
    if clean.endswith(".pdf") or clean.endswith(".docx"):
        return None
    if clean in _STOP_PHRASES or clean.upper() in _STOP_ACRONYMS:
        return None
    if clean.casefold() in {p.casefold() for p in _STOP_PHRASES}:
        return None
    if _STOP_NAME_RE.match(clean):
        return None
    # Drop bare English function/capitalized stop words of length <= 12 without digits/hyphen.
    if " " not in clean and "-" not in clean and clean.isalpha() and clean[:1].isupper() and clean[1:].islower():
        if clean.casefold() in {
            "this", "our", "experimental", "however", "therefore", "moreover",
            "finally", "overall", "problem", "definition", "section", "chapter",
            "paper", "model", "approach", "framework", "system", "network",
            "layer", "output", "input", "method", "result", "results",
        }:
            return None
    etype = entity_type if entity_type in ENTITY_TYPES else _classify_entity(clean, context)
    # Merge case-insensitively within type family
    key = f"{etype}:{clean.casefold()}"
    existing = bucket.get(key)
    if existing:
        existing.mentions += 1
        if source not in existing.sources:
            existing.sources.append(source)
        if clean != existing.name and clean not in existing.aliases:
            existing.aliases.append(clean)
        return existing
    # Cross-type merge on exact name
    for other_key, other in list(bucket.items()):
        if other.name.casefold() == clean.casefold():
            other.mentions += 1
            if source not in other.sources:
                other.sources.append(source)
            return other
    ent = AcademicEntity(
        entity_id=_entity_id(clean, etype),
        name=clean,
        entity_type=etype,
        aliases=[],
        mentions=1,
        sources=[source],
    )
    bucket[key] = ent
    return ent


def _add_edge(
    edges: dict[str, AcademicEdge],
    source: AcademicEntity | None,
    target: AcademicEntity | None,
    relation: str,
    *,
    evidence: str = "",
    weight: float = 1.0,
) -> None:
    if not source or not target or source.entity_id == target.entity_id:
        return
    if relation not in EDGE_TYPES:
        return
    eid = _edge_id(source.entity_id, target.entity_id, relation)
    existing = edges.get(eid)
    if existing:
        existing.weight = round(existing.weight + weight, 3)
        if evidence and len(evidence) > len(existing.evidence or ""):
            existing.evidence = evidence[:180]
        return
    edges[eid] = AcademicEdge(
        edge_id=eid,
        source_id=source.entity_id,
        target_id=target.entity_id,
        relation=relation,
        evidence=re.sub(r"\s+", " ", evidence or "").strip()[:180],
        weight=float(weight),
    )


def _extract_candidates_from_text(text: str) -> list[tuple[str, str]]:
    """Return list of (name, local_context)."""
    results: list[tuple[str, str]] = []
    if not text:
        return results
    for match in _ACRONYM_RE.finditer(text):
        name = match.group(1)
        if name in _STOP_ACRONYMS:
            continue
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        results.append((name, text[start:end]))
    for match in _PROPER_PHRASE_RE.finditer(text):
        name = match.group(1).strip()
        if name in _STOP_PHRASES or len(name) < 3:
            continue
        if name.upper() in _STOP_ACRONYMS:
            continue
        # Skip pure section headings style single common words already filtered
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        results.append((name, text[start:end]))
    for match in _DATASET_CUE_RE.finditer(text):
        name = match.group(1)
        if name.casefold() in {"dataset", "benchmark", "数据集", "基准"}:
            continue
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        results.append((name, text[start:end]))
    for match in _METRIC_NAME_RE.finditer(text):
        name = match.group(1)
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        results.append((name, text[start:end]))
    return results


def build_academic_graph(
    *,
    doc_id: str = "",
    full_text: str = "",
    pages: Optional[list] = None,
    paper_metadata: Optional[dict] = None,
    outline: Optional[dict] = None,
    parse_generation: str = "",
    document_source_hash: str = "",
) -> AcademicGraph:
    """Build a compact single-document academic entity graph."""
    entities: dict[str, AcademicEntity] = {}
    edges: dict[str, AcademicEdge] = {}

    meta = paper_metadata if isinstance(paper_metadata, dict) else {}
    title = str(meta.get("title") or (outline or {}).get("title") or "").strip()
    if title.lower().endswith((".pdf", ".docx", ".doc")):
        title = re.sub(r"\.(pdf|docx?)$", "", title, flags=re.I).strip()
    paper_method = None
    if title and len(title) >= 3:
        # Title often encodes the proposed method name, e.g. "ALP-KD: ..."
        head = title.split(":")[0].strip()
        method_name = head if 2 <= len(head) <= 40 else title[:60]
        paper_method = _add_entity(
            entities,
            method_name,
            "method",
            source="title",
            context=f"we propose {title}",
        )
        if paper_method and ":" in title and head != title:
            # Keep descriptive subtitle phrases only when informative.
            subtitle = title.split(":", 1)[1].strip()
            if 8 <= len(subtitle) <= 80:
                _add_entity(entities, subtitle, "concept", source="title", context=title)

    # Outline-derived entities
    for text in _collect_outline_texts(outline):
        for name, ctx in _extract_candidates_from_text(text):
            etype = _classify_entity(name, ctx)
            _add_entity(entities, name, etype, source="outline", context=ctx)

    # Body sample
    page_text = ""
    if isinstance(pages, list):
        bits = []
        for page in pages[:4]:
            if isinstance(page, dict):
                bits.append(str(page.get("text") or page.get("content") or ""))
        page_text = "\n".join(bits)
    body = _sample_text(page_text or full_text, max_chars=14000)

    # Sentence-level relation harvest
    sentences = re.split(r"(?<=[。．.!?])\s+|\n+", body)
    for sentence in sentences[:400]:
        sent = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if len(sent) < 20:
            continue
        cands = _extract_candidates_from_text(sent)
        if not cands:
            continue
        local_entities: list[AcademicEntity] = []
        for name, ctx in cands[:8]:
            ent = _add_entity(
                entities,
                name,
                _classify_entity(name, ctx or sent),
                source="body",
                context=sent,
            )
            if ent:
                local_entities.append(ent)
        if len(local_entities) < 2 and not paper_method:
            continue

        methods = [e for e in local_entities if e.entity_type in {"method", "model"}]
        datasets = [e for e in local_entities if e.entity_type == "dataset"]
        metrics = [e for e in local_entities if e.entity_type == "metric"]
        baselines = [e for e in local_entities if e.entity_type == "baseline"]
        models = [e for e in local_entities if e.entity_type == "model"]

        # "we propose X" → paper proposes X (only explicit capture, avoid false PKD links)
        subject = paper_method
        propose_match = re.search(
            r"(?:we\s+)?(?:propos(?:e|es|ed)|introduc(?:e|es|ed)|present(?:s|ed)?)\s+"
            r"([A-Z][A-Za-z0-9][A-Za-z0-9\-]{1,30})",
            sent,
            re.IGNORECASE,
        )
        if propose_match:
            proposed = _add_entity(
                entities,
                propose_match.group(1),
                "method",
                source="body",
                context=sent,
            )
            if proposed:
                subject = proposed
                if paper_method and proposed.entity_id != paper_method.entity_id:
                    _add_edge(edges, paper_method, proposed, "proposes", evidence=sent, weight=1.3)
                elif not paper_method:
                    paper_method = proposed

        actor = subject or paper_method or (methods[0] if methods else None)

        if _USES_CUE_RE.search(sent) and actor:
            for target in methods[1:3] + models[:2]:
                if target.entity_id != actor.entity_id:
                    _add_edge(edges, actor, target, "uses", evidence=sent, weight=0.9)

        if (_EVAL_CUE_RE.search(sent) or datasets) and actor:
            for ds in datasets[:3]:
                _add_edge(edges, actor, ds, "evaluates_on", evidence=sent, weight=1.1)

        if metrics and actor:
            for metric in metrics[:3]:
                _add_edge(edges, actor, metric, "reports", evidence=sent, weight=0.8)

        if _OUTPERFORM_CUE_RE.search(sent) and actor:
            rivals = baselines + [e for e in methods + models if e.entity_id != actor.entity_id]
            for rival in rivals[:3]:
                _add_edge(edges, actor, rival, "outperforms", evidence=sent, weight=1.0)

        if _COMPARE_CUE_RE.search(sent) and actor:
            rivals = [e for e in methods + models + baselines if e.entity_id != actor.entity_id]
            for rival in rivals[:3]:
                _add_edge(edges, actor, rival, "compares_with", evidence=sent, weight=0.7)

    # If we have a paper method and datasets/metrics globally, link weakly
    if paper_method:
        all_entities = list(entities.values())
        for ds in [e for e in all_entities if e.entity_type == "dataset"][:4]:
            _add_edge(edges, paper_method, ds, "evaluates_on", evidence="document-level co-occurrence", weight=0.4)
        for metric in [e for e in all_entities if e.entity_type == "metric"][:4]:
            _add_edge(edges, paper_method, metric, "reports", evidence="document-level co-occurrence", weight=0.3)

    # Rank and cap
    ranked_entities = sorted(
        entities.values(),
        key=lambda e: (-e.mentions, 0 if e.entity_type == "method" else 1, e.name.casefold()),
    )[:36]
    keep_ids = {e.entity_id for e in ranked_entities}
    ranked_edges = [
        edge for edge in edges.values()
        if edge.source_id in keep_ids and edge.target_id in keep_ids
    ]
    ranked_edges.sort(key=lambda e: (-e.weight, e.relation, e.edge_id))
    ranked_edges = ranked_edges[:48]

    confidence = 0.2
    if ranked_entities:
        confidence = 0.45
    if any(e.entity_type == "method" for e in ranked_entities):
        confidence += 0.15
    if ranked_edges:
        confidence += 0.2
    if paper_method:
        confidence += 0.1
    confidence = round(min(0.95, confidence), 3)

    return AcademicGraph(
        version=ACADEMIC_GRAPH_VERSION,
        doc_id=str(doc_id or ""),
        entities=ranked_entities,
        edges=ranked_edges,
        parse_generation=str(parse_generation or "").strip(),
        document_source_hash=str(document_source_hash or "").strip(),
        built_at=datetime.now(timezone.utc).isoformat(),
        source="heuristic",
        confidence=confidence,
    )


def academic_graph_from_dict(raw: Any) -> Optional[AcademicGraph]:
    if not isinstance(raw, dict):
        return None
    entities: list[AcademicEntity] = []
    for item in raw.get("entities") or []:
        if not isinstance(item, dict):
            continue
        entities.append(
            AcademicEntity(
                entity_id=str(item.get("entity_id") or ""),
                name=str(item.get("name") or ""),
                entity_type=str(item.get("entity_type") or "concept"),
                aliases=[str(a) for a in (item.get("aliases") or []) if str(a).strip()],
                mentions=int(item.get("mentions") or 1),
                sources=[str(s) for s in (item.get("sources") or []) if str(s).strip()],
            )
        )
    edges: list[AcademicEdge] = []
    for item in raw.get("edges") or []:
        if not isinstance(item, dict):
            continue
        try:
            weight = float(item.get("weight") or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        edges.append(
            AcademicEdge(
                edge_id=str(item.get("edge_id") or ""),
                source_id=str(item.get("source_id") or ""),
                target_id=str(item.get("target_id") or ""),
                relation=str(item.get("relation") or ""),
                evidence=str(item.get("evidence") or ""),
                weight=weight,
            )
        )
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return AcademicGraph(
        version=str(raw.get("version") or ACADEMIC_GRAPH_VERSION),
        doc_id=str(raw.get("doc_id") or ""),
        entities=entities,
        edges=edges,
        parse_generation=str(raw.get("parse_generation") or ""),
        document_source_hash=str(raw.get("document_source_hash") or ""),
        built_at=str(raw.get("built_at") or ""),
        source=str(raw.get("source") or "cached"),
        confidence=confidence,
    )


def format_academic_graph_context(
    graph: AcademicGraph | dict | None,
    *,
    question: str = "",
    max_entities: int = 12,
    max_edges: int = 12,
) -> str:
    """Compact prompt block for answer / planner context."""
    g = graph if isinstance(graph, AcademicGraph) else academic_graph_from_dict(graph)
    if not g or not g.entities:
        return ""

    q = str(question or "").casefold()
    entities = list(g.entities)

    def _rank(entity: AcademicEntity) -> tuple:
        hit = 1 if q and entity.name.casefold() in q else 0
        alias_hit = 1 if q and any(a.casefold() in q for a in entity.aliases) else 0
        type_rank = {
            "method": 0,
            "model": 1,
            "dataset": 2,
            "metric": 3,
            "baseline": 4,
            "concept": 5,
        }.get(entity.entity_type, 6)
        return (-hit, -alias_hit, type_rank, -entity.mentions, entity.name.casefold())

    entities = sorted(entities, key=_rank)[: max(1, int(max_entities or 12))]
    keep = {e.entity_id for e in entities}
    id_to_name = {e.entity_id: e.name for e in g.entities}

    edges = [e for e in g.edges if e.source_id in keep or e.target_id in keep]
    # Prefer edges touching kept entities
    edges = sorted(edges, key=lambda e: (-e.weight, e.relation))[: max(1, int(max_edges or 12))]

    type_groups: dict[str, list[str]] = {}
    for ent in entities:
        type_groups.setdefault(ent.entity_type, []).append(ent.name)

    type_labels = {
        "method": "方法",
        "model": "模型",
        "dataset": "数据集",
        "metric": "指标",
        "baseline": "基线",
        "concept": "概念",
    }
    rel_labels = {
        "proposes": "提出",
        "uses": "使用",
        "evaluates_on": "评估于",
        "outperforms": "优于",
        "compares_with": "比较",
        "reports": "报告",
    }

    lines = ["【学术概念图 · 单篇轻量】"]
    for etype in ("method", "model", "dataset", "metric", "baseline", "concept"):
        names = type_groups.get(etype) or []
        if not names:
            continue
        lines.append(f"- {type_labels.get(etype, etype)}: " + "、".join(names[:6]))
    if edges:
        lines.append("- 关系:")
        for edge in edges:
            src = id_to_name.get(edge.source_id, edge.source_id)
            tgt = id_to_name.get(edge.target_id, edge.target_id)
            rel = rel_labels.get(edge.relation, edge.relation)
            lines.append(f"  · {src} —{rel}→ {tgt}")
    lines.append("- 仅作结构线索；回答仍须引用原文证据，不可只依据本图编造细节。")
    return "\n".join(lines)


def _identity_from_doc(doc: dict) -> tuple[str, str]:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    generation = str(
        doc.get("parse_generation")
        or data.get("parse_generation")
        or (doc.get("parse_manifest") or {}).get("generation")
        or ""
    ).strip()
    source_hash = str(
        doc.get("document_source_hash")
        or data.get("document_source_hash")
        or data.get("source_hash")
        or (doc.get("parse_manifest") or {}).get("source_hash")
        or ""
    ).strip()
    return generation, source_hash


def _resolve_data_root(data_dir: Path | str | None = None) -> Path:
    """Use an explicit task directory, otherwise the shared runtime data root."""
    if data_dir:
        return Path(data_dir)
    try:
        from runtime_mode import runtime

        return Path(runtime.data_dir)
    except Exception:
        return Path(__file__).resolve().parents[2] / "data"


def _load_outline_for_doc(doc_id: str, data_dir: Path | str | None = None) -> dict | None:
    if not doc_id:
        return None
    root = _resolve_data_root(data_dir)
    # Prefer reading outline, then section outline
    for sub in ("reading_outlines", "section_outlines"):
        path = root / sub / f"{doc_id}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and (payload.get("items") or payload.get("sections")):
                return payload
        except Exception:
            continue
    return None


def ensure_academic_graph(
    doc: dict,
    *,
    doc_id: str = "",
    data_dir: Path | str | None = None,
    force: bool = False,
    outline: Optional[dict] = None,
) -> dict[str, Any]:
    """Ensure doc has academic_graph; cache on doc and optionally on disk."""
    if not isinstance(doc, dict):
        return {}
    resolved_id = str(doc_id or doc.get("doc_id") or "").strip()
    generation, source_hash = _identity_from_doc(doc)

    existing_raw = doc.get("academic_graph")
    existing = academic_graph_from_dict(existing_raw) if isinstance(existing_raw, dict) else None
    if (
        existing
        and not force
        and existing.entities
        and (
            not generation
            or not existing.parse_generation
            or existing.parse_generation == generation
        )
        and (
            not source_hash
            or not existing.document_source_hash
            or existing.document_source_hash == source_hash
        )
    ):
        payload = existing.to_dict()
        doc["academic_graph"] = payload
        return payload

    # Try disk cache
    if resolved_id and not force:
        try:
            root = _resolve_data_root(data_dir)
        except Exception:
            root = _resolve_data_root()
        disk = load_academic_graph(root, resolved_id)
        disk_graph = academic_graph_from_dict(disk) if disk else None
        if (
            disk_graph
            and disk_graph.entities
            and (
                not generation
                or not disk_graph.parse_generation
                or disk_graph.parse_generation == generation
            )
            and (
                not source_hash
                or not disk_graph.document_source_hash
                or disk_graph.document_source_hash == source_hash
            )
        ):
            payload = disk_graph.to_dict()
            doc["academic_graph"] = payload
            return payload

    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    paper_meta = doc.get("paper_metadata") if isinstance(doc.get("paper_metadata"), dict) else {}
    if not paper_meta:
        try:
            from services.paper_metadata_service import ensure_paper_metadata

            paper_meta = ensure_paper_metadata(doc) or {}
        except Exception:
            paper_meta = {}

    outline_payload = outline if isinstance(outline, dict) else _load_outline_for_doc(resolved_id, data_dir)

    graph = build_academic_graph(
        doc_id=resolved_id,
        full_text=str(data.get("full_text") or data.get("text") or ""),
        pages=data.get("pages") if isinstance(data.get("pages"), list) else None,
        paper_metadata=paper_meta,
        outline=outline_payload,
        parse_generation=generation,
        document_source_hash=source_hash,
    )
    payload = graph.to_dict()
    doc["academic_graph"] = payload
    if resolved_id:
        try:
            root = _resolve_data_root(data_dir)
            save_academic_graph(root, resolved_id, payload)
        except Exception:
            logger.debug("[AcademicGraph] disk cache save skipped", exc_info=True)
    return payload
