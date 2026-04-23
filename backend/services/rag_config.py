"""
RAG 系统配置模块

定义 RAG（检索增强生成）系统的核心配置参数，
包括语义意群、Token 预算和粒度选择等功能的默认设置。
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


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
    enable_hyde: bool = True                  # HyDE 假设文档嵌入，用 LLM 生成假设答案做检索
    hyde_query_types: str = "analytical,overview"  # 默认仅对分析/概览类问题启用
    hyde_evidence_allowlist: str = "section_explanation,comparison_multi_aspect"
    hyde_evidence_blocklist: str = "numeric_table,reference_trap,reference_meta"
    enable_query_expansion: bool = True       # 多查询扩展，LLM 生成多个改写查询合并检索
    query_expansion_n: int = 3               # 多查询扩展数量
    query_expansion_query_types: str = "analytical,overview"
    query_expansion_evidence_allowlist: str = "section_explanation,comparison_multi_aspect"
    query_expansion_evidence_blocklist: str = "numeric_table,reference_trap,reference_meta"
    enable_contextual_chunking: bool = False  # 上下文增强分块，chunk 前注入章节标题
    enable_lost_in_middle_reorder: bool = True   # Lost-in-the-Middle 缓解，交替排列上下文
    enable_parent_child_retrieval: bool = True   # Parent-Child 分块：用小 chunk 检索，返回大 parent chunk
    token_budget_ratio: float = 0.0          # 动态 Token 预算比例（0 表示使用固定 max_token_budget）

    # ---- 统一后处理清洗 ----
    enable_post_clean: bool = True            # 启用末端统一清洗：结构黑名单 + 碎片惩罚 + 最小分数过滤
    post_clean_min_score: float = 0.06        # 低于此相似度分数的 chunk 被过滤（至少保留 3 条）
    post_clean_min_keep: int = 3              # 过滤后最少保留条数，防止结果为空

    # ---- 条件 rerank gate ----
    enable_conditional_rerank: bool = True    # 按题型启用 rerank：默认纳入论文证据题型
    conditional_rerank_types: str = "extraction,analytical"  # 触发 rerank 的题型（逗号分隔）
    conditional_rerank_evidence_needs: str = "numeric_table,section_explanation,figure_caption"
    rerank_score_min: float = 0.08            # rerank/evidence gate 后的最低分阈值
    rerank_score_min_keep: int = 2            # 触发阈值过滤后最少保留条数

    # ---- 查询类型候选池扩展比例 ----
    overview_candidate_multiplier: float = 2.5   # overview 题型扩展候选数倍率
    analytical_candidate_multiplier: float = 2.0 # analytical 题型扩展候选数倍率
    extraction_candidate_multiplier: float = 1.5 # extraction 题型扩展候选数倍率

    # ---- Focus Mode（rerank 后句级压缩）----
    enable_focus_mode: bool = False          # 对 rerank 后候选做句级最优支持句压缩
    focus_mode_window_size: int = 2          # 每侧保留的上下文句数
    focus_mode_max_sentences: int = 4        # 每个候选最多保留的支持句数
    focus_mode_min_chars: int = 80           # 文本低于此字符数时跳过压缩（太短无需压缩）

    # ---- Section / path-aware budgeting ----
    enable_path_budget: bool = True          # 按 section/path 路径聚合证据，优先同路径相邻证据
    path_max_singletons: int = 3             # 最多允许无同伴的孤立路径数（超出则降权）


# ==================== Per-request feature flag overrides ====================
# 这些 ContextVar 允许单个请求临时覆盖全局 settings 开关（前端 UI 细化控制）。
# FastAPI 每个请求是独立 contextvars context，设置值不会泄漏到其他请求。
# 入口统一走 apply_request_overrides()。

_numeric_table_override: ContextVar[Optional[bool]] = ContextVar(
    "chatpdf_numeric_table_override", default=None
)
_answer_critic_override: ContextVar[Optional[bool]] = ContextVar(
    "chatpdf_answer_critic_override", default=None
)
_llm_query_rewrite_override: ContextVar[Optional[bool]] = ContextVar(
    "chatpdf_llm_query_rewrite_override", default=None
)
_bm25_synonyms_override: ContextVar[Optional[bool]] = ContextVar(
    "chatpdf_bm25_synonyms_override", default=None
)


def apply_request_overrides(
    *,
    numeric_table: Optional[bool] = None,
    answer_critic: Optional[bool] = None,
    llm_query_rewrite: Optional[bool] = None,
    bm25_synonyms: Optional[bool] = None,
) -> None:
    """在请求入口一次性设置 per-request feature flag 覆盖。

    仅在传入非 None 时生效；None 表示"跟随全局 settings"。
    """
    if numeric_table is not None:
        _numeric_table_override.set(bool(numeric_table))
    if answer_critic is not None:
        _answer_critic_override.set(bool(answer_critic))
    if llm_query_rewrite is not None:
        _llm_query_rewrite_override.set(bool(llm_query_rewrite))
    if bm25_synonyms is not None:
        _bm25_synonyms_override.set(bool(bm25_synonyms))


def should_apply_numeric_table_specialization() -> bool:
    """numeric_table 专项检索增强的统一开关入口。

    优先级：per-request ContextVar override > ``config.settings.enable_numeric_table_specialization``。
    关闭后：
    - embedding_service 中的 ``_apply_numeric_table_same_bundle_hard_gate`` 等
      专项 gate / 证据槽逻辑全部跳过
    - chat_routes 中 ``_supplement_numeric_table_citations`` 等专项引文补充
      逻辑全部跳过
    走通用主链路（向量 + BM25 + 常规 rerank + 通用引文对齐）。

    读取时做异常兜底，任何异常退化为开启状态，避免破坏现有行为。
    """
    override = _numeric_table_override.get()
    if override is not None:
        return override
    try:
        from config import settings
        return bool(getattr(settings, "enable_numeric_table_specialization", True))
    except Exception:
        return True


def should_enable_answer_critic() -> bool:
    """答案自审开关的统一入口（per-request 可覆盖）。"""
    override = _answer_critic_override.get()
    if override is not None:
        return override
    try:
        from config import settings
        return bool(getattr(settings, "enable_answer_critic", False))
    except Exception:
        return False


def should_enable_llm_query_rewrite() -> bool:
    """LLM 查询改写开关的统一入口（per-request 可覆盖）。"""
    override = _llm_query_rewrite_override.get()
    if override is not None:
        return override
    try:
        from config import settings
        return bool(getattr(settings, "enable_llm_query_rewrite", True))
    except Exception:
        return True


def should_expand_bm25_synonyms() -> bool:
    """BM25 同义词扩展开关的统一入口（per-request 可覆盖）。"""
    override = _bm25_synonyms_override.get()
    if override is not None:
        return override
    try:
        from config import settings
        return bool(getattr(settings, "bm25_expand_synonyms", True))
    except Exception:
        return True
