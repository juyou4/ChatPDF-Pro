"""
引用质量评估脚手架

指标：
- citation_precision_at_1：每句首个引用的平均支撑得分
- unsupported_citation_rate：低支撑引用占比
- topical_relevance：回答与引用片段的平均词重合度

用法：
  python -m tests.eval_citation_quality --input path/to/citation_eval.jsonl

JSONL 每行格式示例：
{"answer":"... [1] ...","citations":[{"ref":1,"highlight_text":"..."}]}
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import Iterable


INLINE_REF_RE = re.compile(r"(?<!!)(?:\[(\d{1,3})\](?!\()|【(\d{1,3})】)")


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    return tokens


def _overlap_ratio(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for t in ta if t in tb)
    return overlap / max(1, len(ta))


def _split_sentences(answer: str) -> list[str]:
    if not answer:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*", answer)
    return [p.strip() for p in parts if p.strip()]


def _extract_refs(text: str) -> list[int]:
    refs = []
    seen = set()
    for m in INLINE_REF_RE.finditer(text or ""):
        ref = int(m.group(1) or m.group(2))
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


@dataclass
class CitationMetrics:
    citation_precision_at_1: float
    unsupported_citation_rate: float
    topical_relevance: float
    sentence_count: int
    citation_count: int


def evaluate_case(answer: str, citations: list[dict], unsupported_threshold: float = 0.08) -> CitationMetrics:
    cmap = {int(c.get("ref")): c for c in citations if c.get("ref") is not None}
    sentences = _split_sentences(answer)

    p1_scores = []
    topical_scores = []
    total_citations = 0
    unsupported = 0

    for sent in sentences:
        refs = _extract_refs(sent)
        if not refs:
            continue

        core = INLINE_REF_RE.sub("", sent).strip()
        scored = []
        for ref in refs:
            c = cmap.get(ref)
            support_text = (c or {}).get("highlight_text", "")
            score = _overlap_ratio(core, support_text)
            scored.append(score)
            total_citations += 1
            if score < unsupported_threshold:
                unsupported += 1

        if scored:
            p1_scores.append(scored[0])
            topical_scores.extend(scored)

    precision_at_1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0.0
    unsupported_rate = unsupported / total_citations if total_citations else 0.0
    topical_relevance = sum(topical_scores) / len(topical_scores) if topical_scores else 0.0
    return CitationMetrics(
        citation_precision_at_1=precision_at_1,
        unsupported_citation_rate=unsupported_rate,
        topical_relevance=topical_relevance,
        sentence_count=len(sentences),
        citation_count=total_citations,
    )


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="评估引用质量指标")
    parser.add_argument("--input", required=True, help="JSONL 数据文件")
    args = parser.parse_args()

    metrics = []
    for item in _iter_jsonl(args.input):
        m = evaluate_case(item.get("answer", ""), item.get("citations", []))
        metrics.append(m)

    if not metrics:
        print("无可评估数据")
        return

    n = len(metrics)
    avg_p1 = sum(m.citation_precision_at_1 for m in metrics) / n
    avg_unsupported = sum(m.unsupported_citation_rate for m in metrics) / n
    avg_topical = sum(m.topical_relevance for m in metrics) / n

    print(f"样本数: {n}")
    print(f"Citation Precision@1: {avg_p1:.4f}")
    print(f"Unsupported Citation Rate: {avg_unsupported:.4f}")
    print(f"Topical Relevance: {avg_topical:.4f}")


if __name__ == "__main__":
    main()
