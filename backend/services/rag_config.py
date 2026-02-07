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
    """

    enable_semantic_groups: bool = True       # 是否启用意群功能
    target_group_chars: int = 5000            # 意群目标字符数
    min_group_chars: int = 2500               # 意群最小字符数
    max_group_chars: int = 6000               # 意群最大字符数
    max_token_budget: int = 8000              # 最大 Token 预算
    reserve_for_answer: int = 1500            # 预留给回答和系统提示词的 Token 数
    default_granularity: str = "digest"       # 默认粒度
