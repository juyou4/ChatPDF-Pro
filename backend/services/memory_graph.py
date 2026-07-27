"""论文记忆图谱构建

把 llm_distilled / consolidated 记忆里的事实抽成 (主体, 关系, 客体) 三元组，
让"这个方法在哪个数据集上、比哪个基线好多少"这类多跳问题有结构可依。

替代原先只用正则抓 Figure/Table 引用的做法——那套只能表达"某条事实提到了图2"，
表达不了方法与数据集、指标、基线之间的关系。

三个关键约束：
1. **抽取绝不能跑在检索热路径上**。它由后台记忆写入线程按增量阈值触发，
   检索侧只读缓存；缓存没有就退回正则，保证零延迟代价。
2. 解析用正则而不是 eval——模型输出是不可信输入。
3. 同名实体合并并累计 mentions，作为图侧的重要度信号；
   关系失效用软删除（valid=False）而不是物理删除，保留可回溯性。
"""
import hashlib
import logging
import re
from typing import Any, Optional

from services.memory_llm import call_llm_sync, extract_json_object

logger = logging.getLogger(__name__)

# 实体类型；未知类型统一归到 concept
ENTITY_TYPES = {
    "method", "dataset", "metric", "model", "task",
    "figure", "table", "concept", "paper", "author",
}
DEFAULT_ENTITY_TYPE = "concept"

MAX_FACTS_PER_EXTRACTION = 40
MAX_NODES = 40
MAX_EDGES = 80

_EXTRACTION_PROMPT = """你是论文知识图谱抽取器。从给定的事实陈述中抽取实体关系三元组。

实体类型只能用：method（方法/模型架构）、dataset（数据集）、metric（评价指标）、
model（骨干网络）、task（任务）、figure（图）、table（表）、concept（概念）、
paper（论文）、author（作者）。拿不准就用 concept。

关系命名要求（很重要）：
- 用**一般化、无时态**的关系名：用 evaluated_on 而不是 was_evaluated_on，
  用 outperforms 而不是 outperformed_last_year
- 常用关系：proposes / evaluated_on / achieves / outperforms / uses /
  compared_with / reported_in / part_of / measured_by

输出严格 JSON，不要解释、不要代码块：
{"triples": [{"s": "主体名", "s_type": "类型", "r": "关系名", "o": "客体名", "o_type": "类型"}]}

规则：
- 实体名用原文里的写法，保留大小写与连字符（如 ResNet-50、CIFAR-100-LT）
- 同一实体在不同事实里写法不同时，统一成最完整的那个写法
- 事实里没有明确关系就不要编造，宁可少抽
- 最多抽 30 条"""


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label or "")).strip(" 　\t\n.,;:，。；：、")


def _entity_key(label: str) -> str:
    """实体归并键：忽略大小写、空白与连接符差异。"""
    normalized = _normalize_label(label).lower()
    return re.sub(r"[\s_\-]+", "", normalized)


def _label_rank(label: str) -> tuple[int, int]:
    """实体展示名的优先级：更长优先，等长时大写更多的优先。"""
    return (len(label), sum(1 for ch in label if ch.isupper()))


def _normalize_relation(relation: str) -> str:
    """关系名归一：小写下划线，去掉时态化前后缀。"""
    text = _normalize_label(relation).lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"^(?:was|were|is|are|has|have|had)_", "", text)
    text = re.sub(r"_(?:in|on|at|by)$", lambda m: m.group(0), text)
    return text or "related_to"


def _normalize_type(value: str) -> str:
    normalized = _normalize_label(value).lower()
    return normalized if normalized in ENTITY_TYPES else DEFAULT_ENTITY_TYPE


def facts_signature(texts: list[str]) -> str:
    """事实集合的内容签名，用于判断图谱缓存是否过期。"""
    # 先归一化再过滤：纯空白字符串是 truthy，先过滤会让 "  " 混进签名
    normalized = [label for label in (_normalize_label(t) for t in texts) if label]
    joined = "\n".join(sorted(normalized))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


class MemoryGraph:
    """实体-关系图，支持同名合并、mentions 计数与关系软删除。"""

    def __init__(self):
        # key -> {id, type, label, mentions}
        self.nodes: dict[str, dict[str, Any]] = {}
        # (s_key, relation, o_key) -> {source, target, type, mentions, valid}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def upsert_node(self, label: str, node_type: str) -> Optional[str]:
        normalized = _normalize_label(label)
        if not normalized:
            return None
        key = _entity_key(normalized)
        if not key:
            return None
        existing = self.nodes.get(key)
        if existing is None:
            self.nodes[key] = {
                "id": key,
                "type": _normalize_type(node_type),
                "label": normalized,
                "mentions": 1,
            }
            return key
        existing["mentions"] += 1
        # 保留更规范的写法：先看长度（ResNet-50 优于 resnet），
        # 等长时看大写字母数量（ResNet-50 优于 resnet 50）——
        # 论文实体名的大小写是有意义的，不能随第一次出现的写法定死。
        if _label_rank(normalized) > _label_rank(existing["label"]):
            existing["label"] = normalized
        if existing["type"] == DEFAULT_ENTITY_TYPE:
            existing["type"] = _normalize_type(node_type)
        return key

    def upsert_edge(self, s_key: str, relation: str, o_key: str) -> None:
        if not s_key or not o_key or s_key == o_key:
            return
        rel = _normalize_relation(relation)
        edge_key = (s_key, rel, o_key)
        existing = self.edges.get(edge_key)
        if existing is None:
            self.edges[edge_key] = {
                "source": s_key,
                "target": o_key,
                "type": rel,
                "mentions": 1,
                "valid": True,
            }
            return
        existing["mentions"] += 1
        # 重新被提及则"复活"（借鉴 mem0 的关系复活语义）
        existing["valid"] = True

    def invalidate_edge(self, s_key: str, relation: str, o_key: str) -> bool:
        """软删除一条关系：保留历史，只是不再参与查询。"""
        edge = self.edges.get((s_key, _normalize_relation(relation), o_key))
        if edge is None:
            return False
        edge["valid"] = False
        return True

    def add_triple(self, subject: str, s_type: str, relation: str, obj: str, o_type: str) -> None:
        s_key = self.upsert_node(subject, s_type)
        o_key = self.upsert_node(obj, o_type)
        if s_key and o_key:
            self.upsert_edge(s_key, relation, o_key)

    def to_summary(self, doc_id: Optional[str] = None) -> dict[str, Any]:
        """输出与旧接口兼容的摘要结构（额外带 mentions）。"""
        valid_edges = [e for e in self.edges.values() if e.get("valid", True)]
        nodes = sorted(
            self.nodes.values(),
            key=lambda n: (-n["mentions"], n["label"]),
        )
        edges = sorted(valid_edges, key=lambda e: -e["mentions"])
        return {
            "doc_id": doc_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes[:MAX_NODES],
            "edges": edges[:MAX_EDGES],
        }


def extract_triples(
    facts: list[str],
    *,
    api_key: str,
    model: str,
    provider: str,
) -> Optional[list[dict[str, str]]]:
    """用 LLM 从事实里抽三元组；失败返回 None 让调用方降级。"""
    usable = [_normalize_label(f) for f in facts if _normalize_label(f)]
    if not usable:
        return None

    payload = "\n".join(f"- {f}" for f in usable[:MAX_FACTS_PER_EXTRACTION])
    messages = [
        {"role": "system", "content": _EXTRACTION_PROMPT},
        {"role": "user", "content": f"事实列表：\n{payload}"},
    ]

    try:
        response = call_llm_sync(
            messages,
            api_key=api_key,
            model=model,
            provider=provider,
            max_tokens=1200,
        )
    except Exception as exc:
        logger.warning(f"[MemoryGraph] 三元组抽取调用失败: {exc}")
        return None

    parsed = extract_json_object(response or "")
    if not parsed:
        logger.warning("[MemoryGraph] 三元组结果无法解析为 JSON")
        return None

    raw_triples = parsed.get("triples")
    if not isinstance(raw_triples, list):
        return None

    triples: list[dict[str, str]] = []
    for item in raw_triples:
        if not isinstance(item, dict):
            continue
        subject = _normalize_label(item.get("s", ""))
        obj = _normalize_label(item.get("o", ""))
        relation = _normalize_label(item.get("r", ""))
        if not (subject and obj and relation):
            continue
        triples.append({
            "s": subject,
            "s_type": _normalize_type(item.get("s_type", "")),
            "r": relation,
            "o": obj,
            "o_type": _normalize_type(item.get("o_type", "")),
        })
    return triples or None


_FIGURE_RE = re.compile(r"(?:图|Fig(?:ure)?\.?)\s*(\d+)", re.IGNORECASE)
_TABLE_RE = re.compile(r"(?:表|Table)\s*(\d+)", re.IGNORECASE)


def build_regex_graph(entries: list) -> MemoryGraph:
    """LLM 不可用时的降级图谱：仍然只抓 Figure/Table 引用与标签。

    保留这条路径是为了保证图谱功能在没有凭证/调用失败时不至于整个消失。
    """
    graph = MemoryGraph()
    for entry in entries:
        content = getattr(entry, "content", "") or ""
        label = getattr(entry, "title", "") or content[:50]
        fact_label = _normalize_label(label) or "记忆"
        for match in _FIGURE_RE.finditer(content):
            graph.add_triple(fact_label, "concept", "reported_in", f"Figure {match.group(1)}", "figure")
        for match in _TABLE_RE.finditer(content):
            graph.add_triple(fact_label, "concept", "reported_in", f"Table {match.group(1)}", "table")
        for tag in (getattr(entry, "tags", None) or []):
            graph.add_triple(fact_label, "concept", "part_of", tag, "concept")
    return graph


def build_llm_graph(
    facts: list[str],
    *,
    api_key: str,
    model: str,
    provider: str,
) -> Optional[MemoryGraph]:
    """抽三元组并合并成图；抽取失败返回 None。"""
    triples = extract_triples(facts, api_key=api_key, model=model, provider=provider)
    if not triples:
        return None
    graph = MemoryGraph()
    for triple in triples:
        graph.add_triple(
            triple["s"], triple["s_type"], triple["r"], triple["o"], triple["o_type"]
        )
    return graph if graph.nodes else None
