"""
粒度选择器模块

根据用户查询类型智能选择最佳的内容粒度级别，支持基础粒度选择和混合粒度分配。

基础粒度选择规则：
- overview（概览）→ summary 粒度，最多 10 个意群
- extraction（提取）→ full 粒度，最多 3 个意群
- analytical（分析）→ digest 粒度，最多 5 个意群
- specific（具体）→ digest 粒度，最多 5 个意群

混合粒度分配规则（按排名位置）：
- 第 1 名 → full（全文）
- 第 2-3 名 → digest（精要）
- 第 4 名及之后 → summary（摘要）
"""

import logging
from dataclasses import dataclass
from typing import List

from services.query_analyzer import analyze_query_type, QueryType
from services.semantic_group_service import SemanticGroup

logger = logging.getLogger(__name__)

# 查询类型到粒度和最大意群数的映射规则
QUERY_TYPE_MAPPING: dict[str, tuple[str, int]] = {
    "overview": ("summary", 10),
    "extraction": ("full", 3),
    "analytical": ("digest", 5),
    "specific": ("digest", 5),
}

# 查询类型到选择理由的映射
QUERY_TYPE_REASONING: dict[str, str] = {
    "overview": "概览性查询：使用摘要粒度覆盖更多意群，提供全面视角",
    "extraction": "提取性查询：使用全文粒度确保精确内容，限制意群数量",
    "analytical": "分析性查询：使用精要粒度平衡细节与覆盖范围",
    "specific": "具体性查询：使用精要粒度提供适中的上下文细节",
}


@dataclass
class GranularitySelection:
    """粒度选择结果

    Attributes:
        granularity: 选择的粒度级别 ('summary' | 'digest' | 'full')
        max_groups: 最大返回意群数
        query_type: 查询类型 ('overview' | 'extraction' | 'analytical' | 'specific')
        reasoning: 选择理由说明
    """

    granularity: str
    max_groups: int
    query_type: str
    reasoning: str


class GranularitySelector:
    """粒度选择器

    根据用户查询类型智能选择最佳的内容粒度级别。
    复用现有 query_analyzer.py 的 analyze_query_type 函数进行查询分类，
    然后根据映射规则返回对应的粒度和最大意群数。
    """

    def select(
        self,
        query: str,
        groups: List[SemanticGroup],
        max_tokens: int = 8000,
    ) -> GranularitySelection:
        """根据查询类型选择基础粒度

        使用 analyze_query_type 分析查询意图，然后根据映射规则
        返回对应的粒度级别和最大意群数。

        映射规则：
        - overview → (summary, 10)
        - extraction → (full, 3)
        - analytical → (digest, 5)
        - specific → (digest, 5)

        Args:
            query: 用户查询文本
            groups: 可用的语义意群列表
            max_tokens: 最大 Token 预算（预留给后续 TokenBudgetManager 使用）

        Returns:
            GranularitySelection 粒度选择结果
        """
        # 使用现有查询分析器分类查询类型
        query_type: QueryType = analyze_query_type(query)

        # 根据映射规则获取粒度和最大意群数
        granularity, max_groups = QUERY_TYPE_MAPPING[query_type]

        # 获取选择理由
        reasoning = QUERY_TYPE_REASONING[query_type]

        logger.info(
            f"粒度选择: query_type={query_type}, "
            f"granularity={granularity}, max_groups={max_groups}"
        )

        return GranularitySelection(
            granularity=granularity,
            max_groups=max_groups,
            query_type=query_type,
            reasoning=reasoning,
        )

    def select_mixed(
        self,
        query: str,
        ranked_groups: List[SemanticGroup],
        max_tokens: int = 8000,
    ) -> List[dict]:
        """为排序后的意群列表分配混合粒度

        按排名位置分配不同粒度级别，最相关的内容获得最详细的表示：
        - 第 1 名（排名最高）→ full（全文）
        - 第 2-3 名 → digest（精要）
        - 第 4 名及之后 → summary（摘要）

        tokens 字段初始为 0，由后续 TokenBudgetManager 填充。

        Args:
            query: 用户查询文本
            ranked_groups: 按相关性排序的语义意群列表（第一个最相关）
            max_tokens: 最大 Token 预算（预留给后续 TokenBudgetManager 使用）

        Returns:
            混合粒度分配列表: [{"group": SemanticGroup, "granularity": str, "tokens": int}]
        """
        if not ranked_groups:
            return []

        result: List[dict] = []

        for rank, group in enumerate(ranked_groups):
            # 根据排名位置分配粒度
            if rank == 0:
                # 第 1 名：全文粒度
                granularity = "full"
            elif rank <= 2:
                # 第 2-3 名（rank=1, 2）：精要粒度
                granularity = "digest"
            else:
                # 第 4 名及之后：摘要粒度
                granularity = "summary"

            result.append({
                "group": group,
                "granularity": granularity,
                "tokens": 0,  # 初始为 0，由后续 TokenBudgetManager 填充
            })

        logger.info(
            f"混合粒度分配: 共 {len(result)} 个意群, "
            f"分配详情: {[(r['group'].group_id, r['granularity']) for r in result]}"
        )

        return result
