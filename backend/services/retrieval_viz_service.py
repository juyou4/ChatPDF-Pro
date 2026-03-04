"""检索嵌入可视化服务

参考 kotaemon 的 CreateCitationVizPipeline：
1. 将检索到的 chunks + query 做 embedding
2. 降维到 2D（使用 PCA 作为轻量替代 UMAP）
3. 生成 Plotly JSON 格式的散点图数据

可选功能，默认关闭（通过配置启用）。
使用 PCA 而非 UMAP 避免额外依赖。
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def generate_viz_data(
    query: str,
    query_embedding: list[float],
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    retrieved_indices: list[int],
    chunk_pages: Optional[list[int]] = None,
) -> Optional[dict]:
    """生成检索可视化的 Plotly JSON 数据

    Args:
        query: 用户查询文本
        query_embedding: 查询的嵌入向量
        chunk_texts: 所有 chunk 的文本列表
        chunk_embeddings: 所有 chunk 的嵌入向量列表
        retrieved_indices: 被检索命中的 chunk 索引列表
        chunk_pages: 每个 chunk 对应的页码列表（可选）

    Returns:
        Plotly JSON 格式的散点图数据，失败返回 None
    """
    if not chunk_embeddings or not query_embedding:
        return None

    try:
        from sklearn.decomposition import PCA

        # 合并所有嵌入：chunks + query
        all_embeddings = np.array(chunk_embeddings + [query_embedding])

        # PCA 降维到 2D
        if all_embeddings.shape[0] < 3:
            return None

        n_components = min(2, all_embeddings.shape[1], all_embeddings.shape[0])
        pca = PCA(n_components=n_components)
        coords_2d = pca.fit_transform(all_embeddings)

        # 分离 chunk 和 query 坐标
        chunk_coords = coords_2d[:-1]
        query_coord = coords_2d[-1]

        retrieved_set = set(retrieved_indices)

        # 构建 Plotly traces
        # 1. 未检索到的 chunks（蓝色半透明）
        other_x, other_y, other_text = [], [], []
        # 2. 检索到的 chunks（绿色）
        retrieved_x, retrieved_y, retrieved_text = [], [], []

        for i, (x, y) in enumerate(chunk_coords):
            text_preview = chunk_texts[i][:200] if i < len(chunk_texts) else ""
            page_label = f" (P{chunk_pages[i]})" if chunk_pages and i < len(chunk_pages) else ""

            if i in retrieved_set:
                retrieved_x.append(float(x))
                retrieved_y.append(float(y))
                retrieved_text.append(f"[命中] {text_preview}{page_label}")
            else:
                other_x.append(float(x))
                other_y.append(float(y))
                other_text.append(f"{text_preview}{page_label}")

        traces = []

        # 未命中的 chunks
        if other_x:
            traces.append({
                "x": other_x,
                "y": other_y,
                "text": other_text,
                "mode": "markers",
                "type": "scatter",
                "name": "其他文档块",
                "marker": {
                    "color": "rgba(100, 149, 237, 0.3)",
                    "size": 6,
                },
                "hoverinfo": "text",
            })

        # 命中的 chunks
        if retrieved_x:
            traces.append({
                "x": retrieved_x,
                "y": retrieved_y,
                "text": retrieved_text,
                "mode": "markers",
                "type": "scatter",
                "name": "检索命中",
                "marker": {
                    "color": "rgba(34, 197, 94, 0.8)",
                    "size": 10,
                    "symbol": "circle",
                },
                "hoverinfo": "text",
            })

        # Query
        traces.append({
            "x": [float(query_coord[0])],
            "y": [float(query_coord[1])],
            "text": [f"查询: {query[:100]}"],
            "mode": "markers",
            "type": "scatter",
            "name": "用户查询",
            "marker": {
                "color": "rgba(239, 68, 68, 0.9)",
                "size": 14,
                "symbol": "x",
            },
            "hoverinfo": "text",
        })

        layout = {
            "title": "检索嵌入空间可视化",
            "showlegend": True,
            "legend": {"x": 0, "y": 1},
            "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
            "height": 350,
        }

        logger.info(
            f"[RetrievalViz] 生成可视化: {len(chunk_embeddings)} chunks, "
            f"{len(retrieved_indices)} 命中"
        )

        return {"data": traces, "layout": layout}

    except ImportError:
        logger.debug("[RetrievalViz] sklearn 不可用，跳过可视化")
        return None
    except Exception as e:
        logger.warning(f"[RetrievalViz] 可视化生成失败: {e}")
        return None
