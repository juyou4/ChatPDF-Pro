"""
检索可观测性模块

提供检索过程的结构化日志记录和 API 响应元数据转换功能。
用于追踪每次检索的关键决策信息，帮助定位"答非所问"等质量问题的根因。

核心组件：
- RetrievalTrace: 单次检索的完整追踪数据类，包含查询类型、命中数、
  RRF 融合结果、Token 使用情况和降级信息
- RetrievalLogger: 日志记录器，负责将追踪信息写入结构化日志，
  并转换为 API 响应中的 retrieval_meta 字段
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """单次检索的完整追踪信息

    记录从查询分析到上下文构建的全链路关键数据，
    用于检索质量分析和问题定位。

    Attributes:
        query: 用户原始查询文本
        query_type: 查询分类结果（overview/extraction/analytical/specific）
        query_confidence: 查询分类的置信度（0.0 ~ 1.0）
        chunk_hits: 分块级别索引的命中数
        group_hits: 意群级别索引的命中数
        rrf_top_k: RRF 融合后的 topK 结果列表，
            每项格式: {"group_id": str, "rank": int, "source": str}
        token_budget: 总 Token 预算
        token_reserved: 预留给回答和系统提示词的 Token 数
        token_used: 实际使用的 Token 数
        granularity_assignments: 各意群的粒度分配列表，
            每项格式: {"group_id": str, "granularity": str}
        fallback_type: 降级类型，None 表示未降级，
            可选值: "llm_failed" | "index_missing" | "groups_disabled"
        fallback_detail: 降级详情描述
        citations: 引文映射列表，
            每项格式: {"ref": int, "group_id": str, "page_range": [int, int]}
        timing: 各阶段耗时字典（毫秒），
            可能包含: vector_search_ms, bm25_search_ms, rerank_ms, total_ms 等
    """

    query: str
    query_type: str
    query_confidence: float = 0.0
    chunk_hits: int = 0
    group_hits: int = 0
    rrf_top_k: List[dict] = field(default_factory=list)
    token_budget: int = 0
    token_reserved: int = 0
    token_used: int = 0
    granularity_assignments: List[dict] = field(default_factory=list)
    fallback_type: Optional[str] = None
    fallback_detail: Optional[str] = None
    citations: List[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)  # 各阶段耗时（毫秒）（需求 10.2）


class RetrievalLogger:
    """检索日志记录器

    负责将 RetrievalTrace 追踪信息写入结构化日志，
    并转换为 API 响应中的 retrieval_meta 字段。

    日志记录包含：
    - 查询类型和置信度
    - 检索来源（chunk/group 命中数）
    - RRF 融合后 topK 的意群标识和排名
    - Token 预算使用情况
    - 降级原因（如有）

    retrieval_meta 字段包含：
    - query_type: 查询类型
    - granularities: 使用的粒度列表
    - token_used: Token 使用量
    - fallback: 降级信息（如有）
    - citations: 引文映射列表
    """

    def log_trace(self, trace: RetrievalTrace) -> None:
        """记录结构化检索日志

        将检索追踪信息以结构化格式写入日志，便于后续分析和问题定位。
        包含查询信息、检索命中、Token 使用和降级信息等关键数据。

        Args:
            trace: 检索追踪数据
        """
        # 基础检索信息
        logger.info(
            "检索追踪 | 查询类型: %s (置信度: %.2f) | "
            "命中: chunk=%d, group=%d | "
            "Token: 预算=%d, 预留=%d, 已用=%d",
            trace.query_type,
            trace.query_confidence,
            trace.chunk_hits,
            trace.group_hits,
            trace.token_budget,
            trace.token_reserved,
            trace.token_used,
        )

        # RRF 融合结果
        if trace.rrf_top_k:
            top_k_summary = ", ".join(
                f"{item.get('group_id', '?')}(rank={item.get('rank', '?')}, "
                f"source={item.get('source', '?')})"
                for item in trace.rrf_top_k
            )
            logger.info("RRF topK: %s", top_k_summary)

        # 粒度分配
        if trace.granularity_assignments:
            assignments_summary = ", ".join(
                f"{item.get('group_id', '?')}={item.get('granularity', '?')}"
                for item in trace.granularity_assignments
            )
            logger.info("粒度分配: %s", assignments_summary)

        # 降级信息（需求 8.2：降级回退时在日志中标记降级类型和策略差异）
        if trace.fallback_type:
            logger.warning(
                "检索降级 | 类型: %s | 详情: %s",
                trace.fallback_type,
                trace.fallback_detail or "无",
            )

        # 耗时信息（需求 10.2）
        if trace.timing:
            logger.info("检索耗时: %s", trace.timing)

        # 记录原始查询（DEBUG 级别，避免日志过大）
        logger.debug("原始查询: %s", trace.query)

    def to_retrieval_meta(self, trace: RetrievalTrace) -> dict:
        """转换为 API 响应中的 retrieval_meta 字段

        将检索追踪信息转换为前端可用的元数据格式，
        供前端调试和引文追踪使用。

        Args:
            trace: 检索追踪数据

        Returns:
            retrieval_meta 字典，包含以下字段：
            - query_type (str): 查询类型
            - granularities (List[str]): 使用的粒度级别列表（去重）
            - token_used (int): 实际使用的 Token 数
            - fallback (Optional[dict]): 降级信息，None 表示未降级，
                格式: {"type": str, "detail": str}
            - citations (List[dict]): 引文映射列表，
                每项格式: {"ref": int, "group_id": str, "page_range": [int, int]}
        """
        # 提取去重后的粒度列表，保持出现顺序
        granularities = []
        seen = set()
        for assignment in trace.granularity_assignments:
            granularity = assignment.get("granularity", "")
            if granularity and granularity not in seen:
                granularities.append(granularity)
                seen.add(granularity)

        # 构建降级信息
        fallback = None
        if trace.fallback_type:
            fallback = {
                "type": trace.fallback_type,
                "detail": trace.fallback_detail or "",
            }

        return {
            "query_type": trace.query_type,
            "granularities": granularities,
            "token_used": trace.token_used,
            "fallback": fallback,
            "citations": trace.citations,
            "timing": trace.timing,
        }
