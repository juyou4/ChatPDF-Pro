"""
RAPTOR 层次聚类摘要服务

参考 ragflow raptor.py 实现：
- UMAP 降维 + GMM 聚类，将语义相近的 chunk 聚为一组
- LLM 对每个聚类生成摘要，形成树状层次结构
- 检索时可在不同层次进行，概览类问题用上层摘要，细节问题用底层 chunk

与现有意群系统的关系：
- 现有意群按文档顺序连续聚合（SemanticGroupService）
- RAPTOR 按语义相似度聚类（可跨章节），是意群的增强补充
- 两者产出共存，检索时通过 RRF 融合

依赖：umap-learn, scikit-learn（可选，未安装时自动禁用）
"""
import logging
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 可选依赖检测
try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class RaptorNode:
    """RAPTOR 树节点

    Attributes:
        node_id: 节点唯一标识
        level: 层级（0=原始 chunk，1+=聚类摘要）
        chunk_indices: 包含的原始 chunk 索引
        text: 节点文本（原始 chunk 文本或聚类摘要）
        embedding: 向量嵌入（可选，序列化时不保存）
        children: 子节点 ID 列表
        summary_of: 描述该节点摘要了哪些内容
    """
    node_id: str
    level: int
    chunk_indices: List[int]
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    children: List[str] = field(default_factory=list)
    summary_of: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "RaptorNode":
        data.pop("embedding", None)
        return cls(**data)


def _cluster_embeddings(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    dim: int = 10,
    random_state: int = 42,
) -> List[List[int]]:
    """使用 UMAP 降维 + GMM 聚类分组 chunk

    Args:
        embeddings: (N, D) 嵌入矩阵
        max_clusters: 最大聚类数
        dim: UMAP 降维目标维度
        random_state: 随机种子

    Returns:
        聚类分组列表，每组包含 chunk 索引列表
    """
    if not _HAS_UMAP or not _HAS_SKLEARN:
        logger.warning("[RAPTOR] umap-learn 或 scikit-learn 未安装，回退到简单等分聚类")
        return _simple_partition(len(embeddings), max_clusters)

    n_samples = embeddings.shape[0]
    if n_samples <= max_clusters:
        return [[i] for i in range(n_samples)]

    # UMAP 降维
    target_dim = min(dim, n_samples - 2, embeddings.shape[1])
    if target_dim < 2:
        target_dim = 2

    try:
        reduced = UMAP(
            n_components=target_dim,
            metric="cosine",
            random_state=random_state,
            n_neighbors=min(15, n_samples - 1),
        ).fit_transform(embeddings)
    except Exception as e:
        logger.warning(f"[RAPTOR] UMAP 降维失败: {e}，回退到简单分组")
        return _simple_partition(n_samples, max_clusters)

    # GMM 聚类：自动选择最佳聚类数（BIC 准则）
    best_n, best_bic = 2, float("inf")
    max_k = min(max_clusters, n_samples // 2)

    for k in range(2, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=random_state, covariance_type="full")
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            if bic < best_bic:
                best_bic = bic
                best_n = k
        except Exception:
            continue

    gmm = GaussianMixture(n_components=best_n, random_state=random_state, covariance_type="full")
    labels = gmm.fit_predict(reduced)

    # 按标签分组
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    result = [indices for indices in clusters.values() if indices]
    logger.info(f"[RAPTOR] GMM 聚类: {n_samples} chunks → {len(result)} 组 (BIC 最优 k={best_n})")
    return result


def _simple_partition(n: int, max_groups: int) -> List[List[int]]:
    """简单等分分组（无依赖回退）"""
    group_size = max(1, n // max(1, max_groups))
    groups = []
    for i in range(0, n, group_size):
        groups.append(list(range(i, min(i + group_size, n))))
    return groups


class RaptorService:
    """RAPTOR 层次聚类摘要服务

    构建层次化的文档摘要树：
    Level 0: 原始 chunk
    Level 1: chunk 聚类摘要
    Level 2+: 聚类摘要的再聚类摘要（递归）

    用法：
        service = RaptorService(embed_fn=embed_fn)
        tree = service.build_tree(chunks, api_key=key, model="gpt-4o-mini")
        service.save_tree(doc_id, tree, store_dir)
    """

    def __init__(
        self,
        embed_fn=None,
        max_cluster_size: int = 10,
        max_levels: int = 3,
        summary_max_tokens: int = 300,
    ):
        self.embed_fn = embed_fn
        self.max_cluster_size = max_cluster_size
        self.max_levels = max_levels
        self.summary_max_tokens = summary_max_tokens

    def build_tree(
        self,
        chunks: List[str],
        embeddings: Optional[np.ndarray] = None,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        endpoint: str = "",
    ) -> List[RaptorNode]:
        """构建 RAPTOR 层次树

        Args:
            chunks: 原始文本分块列表
            embeddings: 预计算的嵌入矩阵（可选）
            api_key: LLM API 密钥
            model: 摘要生成模型
            provider: 模型提供商
            endpoint: API 端点

        Returns:
            RaptorNode 列表（所有层级）
        """
        if len(chunks) < 3:
            logger.info("[RAPTOR] chunk 数量不足，跳过层次聚类")
            return []

        t0 = time.perf_counter()
        all_nodes: List[RaptorNode] = []

        # Level 0: 原始 chunk 节点
        level0_nodes = []
        for i, chunk in enumerate(chunks):
            node = RaptorNode(
                node_id=f"L0-{i}",
                level=0,
                chunk_indices=[i],
                text=chunk,
            )
            level0_nodes.append(node)
            all_nodes.append(node)

        # 获取/计算嵌入
        if embeddings is None and self.embed_fn:
            try:
                embeddings = np.array(self.embed_fn(chunks)).astype("float32")
            except Exception as e:
                logger.warning(f"[RAPTOR] 嵌入计算失败: {e}")
                return all_nodes

        if embeddings is None:
            return all_nodes

        current_nodes = level0_nodes
        current_embeddings = embeddings

        for level in range(1, self.max_levels + 1):
            if len(current_nodes) <= 2:
                break

            # 聚类
            clusters = _cluster_embeddings(
                current_embeddings,
                max_clusters=self.max_cluster_size,
            )

            if len(clusters) <= 1:
                break

            # 为每个聚类生成摘要
            next_nodes = []
            next_texts = []

            for ci, cluster_indices in enumerate(clusters):
                if not cluster_indices:
                    continue

                # 合并聚类中的文本
                cluster_texts = [current_nodes[i].text for i in cluster_indices if i < len(current_nodes)]
                merged_text = "\n\n".join(cluster_texts)

                # 收集所有原始 chunk 索引
                all_chunk_indices = []
                child_ids = []
                for i in cluster_indices:
                    if i < len(current_nodes):
                        all_chunk_indices.extend(current_nodes[i].chunk_indices)
                        child_ids.append(current_nodes[i].node_id)

                # LLM 摘要
                summary = self._generate_summary(
                    merged_text,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                )

                node = RaptorNode(
                    node_id=f"L{level}-{ci}",
                    level=level,
                    chunk_indices=sorted(set(all_chunk_indices)),
                    text=summary,
                    children=child_ids,
                    summary_of=f"聚类 {ci}: {len(cluster_indices)} 个 L{level-1} 节点",
                )
                next_nodes.append(node)
                next_texts.append(summary)
                all_nodes.append(node)

            if not next_texts:
                break

            # 计算下一层嵌入
            if self.embed_fn:
                try:
                    current_embeddings = np.array(self.embed_fn(next_texts)).astype("float32")
                except Exception:
                    break
            else:
                break

            current_nodes = next_nodes
            logger.info(f"[RAPTOR] Level {level}: {len(next_nodes)} 个聚类摘要节点")

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"[RAPTOR] 树构建完成: {len(all_nodes)} 个节点, "
            f"{max(n.level for n in all_nodes) + 1} 层, {elapsed}ms"
        )
        return all_nodes

    def _generate_summary(
        self,
        text: str,
        api_key: str,
        model: str,
        provider: str,
        endpoint: str,
        max_input_chars: int = 8000,
    ) -> str:
        """使用 LLM 生成聚类摘要"""
        if not api_key:
            return text[:500] + "..."

        truncated = text[:max_input_chars]

        try:
            import asyncio
            import concurrent.futures
            from services.chat_service import call_ai_api

            messages = [
                {"role": "system", "content": (
                    "你是一个文档摘要助手。请对给定文本生成一段简洁的摘要，"
                    "保留所有关键信息、数据和结论。摘要应当自包含，"
                    "读者无需查看原文即可理解核心内容。请用中文输出。"
                )},
                {"role": "user", "content": f"请摘要以下文本：\n\n{truncated}"},
            ]

            def _run(coro):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, coro).result()
                return asyncio.run(coro)

            result = _run(call_ai_api(
                messages=messages,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=self.summary_max_tokens,
                temperature=0.3,
            ))
            return result.strip() if result else text[:500]

        except Exception as e:
            logger.warning(f"[RAPTOR] 摘要生成失败: {e}")
            return text[:500] + "..."

    def save_tree(self, doc_id: str, nodes: List[RaptorNode], store_dir: str):
        """持久化 RAPTOR 树到 JSON 文件"""
        if not nodes:
            return
        os.makedirs(store_dir, exist_ok=True)
        path = os.path.join(store_dir, f"{doc_id}_raptor.json")
        data = {
            "version": 1,
            "doc_id": doc_id,
            "created_at": time.time(),
            "nodes": [n.to_dict() for n in nodes],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[RAPTOR] 树已保存: {path} ({len(nodes)} 节点)")

    def load_tree(self, doc_id: str, store_dir: str) -> Optional[List[RaptorNode]]:
        """加载 RAPTOR 树"""
        path = os.path.join(store_dir, f"{doc_id}_raptor.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = [RaptorNode.from_dict(n) for n in data.get("nodes", [])]
            logger.info(f"[RAPTOR] 树已加载: {path} ({len(nodes)} 节点)")
            return nodes
        except Exception as e:
            logger.warning(f"[RAPTOR] 加载失败: {e}")
            return None

    def get_summary_nodes(self, nodes: List[RaptorNode], min_level: int = 1) -> List[RaptorNode]:
        """获取指定层级以上的摘要节点（用于概览类检索）"""
        return [n for n in nodes if n.level >= min_level]
