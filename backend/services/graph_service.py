"""GraphRAG 知识图谱服务（基础框架）

参考 kotaemon 的 NanoGraphRAGIndexingPipeline / NanoGraphRAGRetrieverPipeline。

当前实现为基础框架，提供：
1. 基于 LLM 的实体-关系提取
2. 简单的内存图谱存储
3. 基于实体匹配的图谱检索

完整 GraphRAG（nano-graphrag 集成）可在此基础上扩展。
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# LLM 调用缓存（hash → response），避免重复提取（参考 LightRAG）
_llm_cache: dict[str, str] = {}

ENTITY_EXTRACTION_PROMPT = """从以下文本中提取实体和关系。

输出 JSON 格式：
{{"entities": ["实体1", "实体2", ...], "relations": [{{"source": "实体1", "target": "实体2", "relation": "关系描述"}}]}}

只提取最重要的实体（人名、组织、概念、方法、技术等）和它们之间的关系。
每段文本最多提取 10 个实体和 10 条关系。
直接输出 JSON，不要加任何前缀或解释。

文本：
{text}
"""


@dataclass
class GraphNode:
    """图谱节点"""
    name: str
    node_type: str = "entity"
    mentions: int = 1
    source_chunks: list[int] = field(default_factory=list)


@dataclass
class GraphEdge:
    """图谱边"""
    source: str
    target: str
    relation: str
    weight: float = 1.0


class DocumentGraph:
    """文档知识图谱"""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_entity(self, name: str, chunk_idx: int = -1):
        """添加实体节点"""
        normalized = name.strip().lower()
        if not normalized:
            return
        if normalized in self.nodes:
            self.nodes[normalized].mentions += 1
            if chunk_idx >= 0:
                self.nodes[normalized].source_chunks.append(chunk_idx)
        else:
            self.nodes[normalized] = GraphNode(
                name=name.strip(),
                source_chunks=[chunk_idx] if chunk_idx >= 0 else [],
            )

    def add_relation(self, source: str, target: str, relation: str):
        """添加关系边"""
        src = source.strip().lower()
        tgt = target.strip().lower()
        if src and tgt and src != tgt:
            self.edges.append(GraphEdge(
                source=src, target=tgt, relation=relation,
            ))

    def search_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """基于关键词匹配查找相关实体及其关系

        Args:
            query: 用户查询
            top_k: 返回的最大实体数

        Returns:
            [{"entity": str, "mentions": int, "relations": [...], "source_chunks": [...]}]
        """
        query_lower = query.lower()
        query_chars = set(query_lower)

        scored = []
        for key, node in self.nodes.items():
            # 计算查询与实体名的字符重叠度
            name_chars = set(key)
            overlap = len(query_chars & name_chars)
            total = len(query_chars | name_chars) or 1
            score = overlap / total
            # 提升精确子串匹配的分数
            if key in query_lower or query_lower in key:
                score += 0.5
            scored.append((score, key, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, key, node in scored[:top_k]:
            if score < 0.1:
                break
            # 收集该实体的关系
            relations = []
            for edge in self.edges:
                if edge.source == key or edge.target == key:
                    relations.append({
                        "source": self.nodes.get(edge.source, GraphNode(edge.source)).name,
                        "target": self.nodes.get(edge.target, GraphNode(edge.target)).name,
                        "relation": edge.relation,
                    })

            results.append({
                "entity": node.name,
                "mentions": node.mentions,
                "relations": relations[:5],
                "source_chunks": node.source_chunks[:5],
            })

        return results

    def to_context(self, query: str, max_entities: int = 5) -> str:
        """将图谱检索结果转为 LLM 上下文字符串"""
        entities = self.search_entities(query, top_k=max_entities)
        if not entities:
            return ""

        parts = ["## 知识图谱关联信息\n"]
        for e in entities:
            parts.append(f"**{e['entity']}** (出现 {e['mentions']} 次)")
            for r in e["relations"]:
                parts.append(f"  - {r['source']} → {r['relation']} → {r['target']}")

        return "\n".join(parts)

    def stats(self) -> dict:
        """返回图谱统计信息"""
        return {
            "doc_id": self.doc_id,
            "num_entities": len(self.nodes),
            "num_relations": len(self.edges),
            "top_entities": sorted(
                [(n.name, n.mentions) for n in self.nodes.values()],
                key=lambda x: x[1], reverse=True,
            )[:10],
        }

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "doc_id": self.doc_id,
            "nodes": {k: {"name": v.name, "type": v.node_type, "mentions": v.mentions, "chunks": v.source_chunks} for k, v in self.nodes.items()},
            "edges": [{"source": e.source, "target": e.target, "relation": e.relation, "weight": e.weight} for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentGraph":
        """从字典反序列化"""
        graph = cls(data["doc_id"])
        for k, v in data.get("nodes", {}).items():
            graph.nodes[k] = GraphNode(name=v["name"], node_type=v.get("type", "entity"), mentions=v.get("mentions", 1), source_chunks=v.get("chunks", []))
        for e in data.get("edges", []):
            graph.edges.append(GraphEdge(source=e["source"], target=e["target"], relation=e["relation"], weight=e.get("weight", 1.0)))
        return graph

    def save(self, store_dir: str):
        """持久化图谱到 JSON 文件"""
        os.makedirs(store_dir, exist_ok=True)
        path = os.path.join(store_dir, f"{self.doc_id}_graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[GraphRAG] 图谱已保存: {path}")

    @classmethod
    def load(cls, doc_id: str, store_dir: str) -> Optional["DocumentGraph"]:
        """从文件加载图谱"""
        path = os.path.join(store_dir, f"{doc_id}_graph.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph = cls.from_dict(data)
            logger.info(f"[GraphRAG] 图谱已加载: {len(graph.nodes)} 实体, {len(graph.edges)} 关系")
            return graph
        except Exception as e:
            logger.warning(f"[GraphRAG] 加载失败: {e}")
            return None


# 全局图谱缓存
_graph_cache: dict[str, DocumentGraph] = {}


async def build_document_graph(
    doc_id: str,
    chunks: list[str],
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    batch_size: int = 4,
) -> DocumentGraph:
    """为文档构建知识图谱

    Args:
        doc_id: 文档 ID
        chunks: 文档分块文本列表
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点
        batch_size: 每批处理的 chunk 数

    Returns:
        构建完成的 DocumentGraph
    """
    from services.chat_service import call_ai_api
    from services.completion_outcome import require_publishable_completion

    graph = DocumentGraph(doc_id)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for j, chunk_text in enumerate(batch):
            chunk_idx = i + j
            if len(chunk_text) < 20:
                continue

            try:
                prompt = ENTITY_EXTRACTION_PROMPT.format(text=chunk_text[:1500])

                # LLM 缓存：相同输入不重复调用（参考 LightRAG hash 缓存）
                cache_key = hashlib.md5(f"completion-v2:{prompt}".encode()).hexdigest()
                if cache_key in _llm_cache:
                    content = _llm_cache[cache_key]
                else:
                    response = await call_ai_api(
                        messages=[{"role": "user", "content": prompt}],
                        api_key=api_key, model=model,
                        provider=provider, endpoint=endpoint,
                        max_tokens=500, temperature=0.0,
                    )
                    require_publishable_completion(response, operation="document graph extraction")
                    if isinstance(response, str):
                        content = response
                    elif isinstance(response, dict):
                        choices = response.get("choices") or []
                        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
                        content = str(message.get("content") or "") if isinstance(message, dict) else ""
                    else:
                        content = ""
                    if content:
                        _llm_cache[cache_key] = content

                if not content:
                    continue

                # 处理可能的 markdown 代码块
                if "```" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        content = content[start:end]

                parsed = json.loads(content)
                entities = parsed.get("entities", [])
                relations = parsed.get("relations", [])

                for entity in entities:
                    if isinstance(entity, str):
                        graph.add_entity(entity, chunk_idx)

                for rel in relations:
                    if isinstance(rel, dict):
                        graph.add_relation(
                            rel.get("source", ""),
                            rel.get("target", ""),
                            rel.get("relation", ""),
                        )

            except json.JSONDecodeError:
                logger.debug(f"[GraphRAG] chunk {chunk_idx} JSON 解析失败")
            except Exception as e:
                logger.debug(f"[GraphRAG] chunk {chunk_idx} 实体提取失败: {e}")

    _graph_cache[doc_id] = graph
    logger.info(f"[GraphRAG] 图谱构建完成: {graph.stats()}")

    # 自动持久化
    try:
        from config import settings
        data_dir = getattr(settings, 'data_dir', 'data')
        store_dir = os.path.join(data_dir, 'vector_stores')
        graph.save(store_dir)
    except Exception as e:
        logger.debug(f"[GraphRAG] 自动保存失败: {e}")

    return graph


def get_graph(doc_id: str, store_dir: str = "") -> Optional[DocumentGraph]:
    """获取已构建的文档图谱（优先内存缓存，回退磁盘加载）"""
    if doc_id in _graph_cache:
        return _graph_cache[doc_id]
    if store_dir:
        graph = DocumentGraph.load(doc_id, store_dir)
        if graph:
            _graph_cache[doc_id] = graph
            return graph
    return None


def get_graph_context(doc_id: str, query: str) -> str:
    """获取图谱增强的检索上下文"""
    graph = get_graph(doc_id)
    if not graph:
        return ""
    return graph.to_context(query)
