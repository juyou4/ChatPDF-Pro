"""GraphRAG 知识图谱增强检索子包

基于 shibing624/ChatPDF 的轻量级 GraphRAG 实现，适配 Chatpdf 后端架构。
提供：
1. LLM 实体-关系提取（含 gleaning 多轮补全）
2. Leiden 社区层级聚类
3. 社区报告生成
4. 实体向量库语义检索
5. Local Query（实体+关系+社区报告+源文本块 四路上下文组装）
6. 持久化存储（NetworkX GraphML + JSON KV + NanoVectorDB）
"""

import asyncio
import threading

from .graphrag import GraphRAG, GraphRAGConfig, BuildProgress
from .base import QueryParam
from ._utils import limit_async_func_call

# 模块级实例注册表：{doc_id: GraphRAG 实例}
# 由 document_routes 的构建端点写入，chat_routes 的融合逻辑与 stats 端点读取。
# 必须集中到这里，否则两个 APIRouter 各自持有一份副本，聊天路径永远找不到实例。
INSTANCES: dict = {}

# 模块级文档构建锁：{doc_id: threading.Lock}
# 防止同一文档并发构建（重复点击 / 多请求同时触发）
BUILD_LOCKS: dict[str, threading.Lock] = {}

# 模块级构建进度注册表：{doc_id: BuildProgress}
# 由构建端点写入，stats/progress 端点读取
BUILD_PROGRESS: dict = {}


def get_build_lock(doc_id: str) -> threading.Lock:
    """获取文档级构建锁（线程安全）"""
    if doc_id not in BUILD_LOCKS:
        BUILD_LOCKS[doc_id] = threading.Lock()
    return BUILD_LOCKS[doc_id]


__all__ = [
    "GraphRAG", "GraphRAGConfig", "BuildProgress", "QueryParam", "limit_async_func_call",
    "INSTANCES", "BUILD_LOCKS", "BUILD_PROGRESS", "get_build_lock",
]
