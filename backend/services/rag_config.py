"""
RAG 系统配置模块

定义 RAG（检索增强生成）系统的核心配置参数，
包括语义意群、Token 预算和粒度选择等功能的默认设置。
"""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG 系统配置数据类

    包含语义意群生成、Token 预算管理和粒度选择等功能的可配置参数。
    所有参数均提供合理的默认值，可根据实际场景调整。

    Attributes:
        enable_semantic_groups: 是否启用语义意群功能，禁用时回退到分块级别检索
        target_group_chars: 意群目标字符数，聚合分块时的理想大小
        min_group_chars: 意群最小字符数，低于此值时继续聚合（最后一个意群除外）
        max_group_chars: 意群最大字符数，超过此值时强制切分
        max_token_budget: 最大 Token 预算，控制发送给 LLM 的上下文总量
        reserve_for_answer: 预留给回答和系统提示词的 Token 数
        default_granularity: 默认粒度级别，可选 "summary"、"digest" 或 "full"
        relevance_threshold: 检索质量阈值，所有结果相似度低于此值时附加低质量提示
        small_doc_chunk_threshold: 小文档分块数阈值，低于此值时跳过意群级别检索以加速响应
        enable_hyde: 是否启用 HyDE 假设文档嵌入
        enable_query_expansion: 是否启用多查询扩展
        query_expansion_n: 多查询扩展数量
        enable_contextual_chunking: 是否启用上下文增强分块（章节标题注入）
        enable_lost_in_middle_reorder: 是否启用 Lost-in-the-Middle 缓解
        token_budget_ratio: 动态 Token 预算比例（0 表示使用固定 max_token_budget）
    """

    enable_semantic_groups: bool = True       # 是否启用意群功能
    target_group_chars: int = 5000            # 意群目标字符数
    min_group_chars: int = 2500               # 意群最小字符数
    max_group_chars: int = 6000               # 意群最大字符数
    max_token_budget: int = 8000              # 最大 Token 预算
    reserve_for_answer: int = 1500            # 预留给回答和系统提示词的 Token 数
    default_granularity: str = "digest"       # 默认粒度
    relevance_threshold: float = 0.3          # 检索质量阈值（需求 8.2）
    small_doc_chunk_threshold: int = 10       # 小文档分块数阈值，低于此值跳过意群检索（需求 10.3，从 20 降至 10 以提升召回率）

    # ---- RAG 优化开关 ----
    enable_hyde: bool = False                 # HyDE 假设文档嵌入，用 LLM 生成假设答案做检索
    enable_query_expansion: bool = False      # 多查询扩展，LLM 生成多个改写查询合并检索
    query_expansion_n: int = 3               # 多查询扩展数量
    enable_contextual_chunking: bool = False  # 上下文增强分块，chunk 前注入章节标题
    enable_lost_in_middle_reorder: bool = False  # Lost-in-the-Middle 缓解，交替排列上下文
    enable_parent_child_retrieval: bool = False  # Parent-Child 分块：用小 chunk 检索，返回大 parent chunk
    token_budget_ratio: float = 0.0          # 动态 Token 预算比例（0 表示使用固定 max_token_budget）
