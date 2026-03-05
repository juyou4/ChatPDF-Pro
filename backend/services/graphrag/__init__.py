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

from .graphrag import GraphRAG, GraphRAGConfig
from .base import QueryParam
from ._utils import limit_async_func_call

__all__ = ["GraphRAG", "GraphRAGConfig", "QueryParam", "limit_async_func_call"]
