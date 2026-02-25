"""
上下文构建器模块

将粒度选择结果组装为格式化的上下文字符串，供 LLM 使用。
每个意群的上下文包含引用编号、意群标识、粒度级别、页码范围、关键词和内容文本。

格式示例：
[1]【group-0 - full | 页码: 1-3】
关键词: 机器学习, 深度学习, 神经网络
内容:
这是意群的完整文本内容...

返回值包含格式化的上下文字符串和引文映射列表（citations），
引文映射用于后续引文追踪功能。
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 粒度级别的中文标注映射
GRANULARITY_LABELS = {
    "full": "full",
    "digest": "digest",
    "summary": "summary",
}

# 粒度对应的文本属性名映射
GRANULARITY_TEXT_ATTR = {
    "full": "full_text",
    "digest": "digest",
    "summary": "summary",
}


# ---- 模块级单例 ----
from services.rag_config import RAGConfig as _RAGConfig
_rag_config = _RAGConfig()


class ContextBuilder:
    """上下文构建器

    将粒度选择结果（意群 + 粒度级别）组装为格式化的上下文字符串，
    包含引用编号、意群标识、粒度级别标注、页码范围和关键词。

    同时生成引文映射列表（citations），用于后续引文追踪功能（任务 11）。
    """

    def build_context(
        self,
        selections: List[dict],
        group_best_chunks: dict = None,
        query: str = "",
        selected_text: str = "",
    ) -> Tuple[str, List[dict]]:
        """将粒度选择结果组装为格式化上下文字符串

        对每个选择项，生成如下格式的上下文块：
        [ref]【group_id - granularity | 页码: start-end】
        关键词: kw1, kw2, kw3
        内容:
        <文本内容>

        Args:
            selections: 粒度选择结果列表，每项格式为：
                {
                    "group": SemanticGroup,  # 意群对象
                    "granularity": str,      # 粒度: "full" | "digest" | "summary"
                    "tokens": int            # Token 数
                }
            group_best_chunks: 可选，group_id -> 最佳匹配 chunk 文本的映射，
                用于生成更精确的引用高亮文本。如果未提供，回退到取文本前100字符。
            query: 用户查询文本，用于从 chunk 中提取与查询最相关的片段作为 highlight_text。
            selected_text: 用户框选的文本，关键词也纳入高亮匹配范围。

        Returns:
            (context_string, citations) 元组
            - context_string: 格式化的上下文字符串
            - citations: 引文映射列表，每项格式为：
                {"ref": int, "group_id": str, "page_range": [int, int], "highlight_text": str}
        """
        if not selections:
            return "", []

        # Lost-in-the-Middle 缓解：交替排列，最相关内容在开头和结尾
        if _rag_config.enable_lost_in_middle_reorder and len(selections) > 2:
            selections = self._reorder_lost_in_middle(selections)

        context_parts = []
        citations = []

        for idx, selection in enumerate(selections):
            group = selection["group"]
            granularity = selection.get("granularity", "full")

            # 引用编号从 1 开始
            ref_num = idx + 1

            # 获取意群标识
            group_id = group.group_id

            # 获取粒度级别标注
            granularity_label = GRANULARITY_LABELS.get(granularity, granularity)

            # 获取页码范围
            page_start, page_end = group.page_range

            # 获取对应粒度的文本内容
            text_attr = GRANULARITY_TEXT_ATTR.get(granularity, "full_text")
            text = getattr(group, text_attr, "")

            # 获取关键词列表
            keywords = group.keywords if group.keywords else []

            # 构建格式化的上下文块
            # 头部：[引用编号]【意群标识 - 粒度级别 | 页码: 起始-结束】
            header = f"[{ref_num}]【{group_id} - {granularity_label} | 页码: {page_start}-{page_end}】"

            # 关键词行
            keywords_line = f"关键词: {', '.join(keywords)}" if keywords else ""

            # 组装上下文块
            parts = [header]
            if keywords_line:
                parts.append(keywords_line)
            parts.append("内容:")
            parts.append(text)

            context_parts.append("\n".join(parts))

            # 构建引文映射（包含高亮文本片段，用于前端定位高亮）
            # 优先使用实际匹配的 chunk 文本中与查询最相关的片段
            if group_best_chunks and group_id in group_best_chunks:
                best_chunk = group_best_chunks[group_id]
                # 从 chunk 中提取与查询最相关的片段（而非简单截取前N字符）
                highlight_text = self._extract_relevant_snippet(
                    best_chunk, query, max_len=200, selected_text=selected_text
                ) if best_chunk else ""
            else:
                highlight_text = self._extract_relevant_snippet(
                    text, query, max_len=150, selected_text=selected_text
                ) if text else ""
            citations.append({
                "ref": ref_num,
                "group_id": group_id,
                "page_range": [page_start, page_end],
                "highlight_text": highlight_text,
            })

        # 用双换行分隔各意群的上下文块
        context_string = "\n\n".join(context_parts)

        logger.info(
            f"上下文构建完成: {len(selections)} 个意群, "
            f"总长度 {len(context_string)} 字符"
        )

        return context_string, citations

    def _extract_relevant_snippet(
        self,
        text: str,
        query: str,
        max_len: int = 200,
        selected_text: str = "",
    ) -> str:
        """从文本中提取与查询最相关的片段

        策略：
        1. 将查询和 selected_text 合并后拆分为关键词
        2. 在文本中找到关键词命中密度最高的窗口
        3. 返回该窗口对应的原始文本片段

        如果没有关键词命中，回退到取文本前 max_len 字符。

        Args:
            text: 源文本（chunk 或意群文本）
            query: 用户查询文本
            max_len: 返回片段的最大字符数
            selected_text: 用户框选的文本，关键词也纳入匹配范围

        Returns:
            与查询最相关的文本片段
        """
        if not text:
            return ""
        if not query or len(text) <= max_len:
            return text[:max_len].strip()

        # 合并 query 和 selected_text 的关键词
        combined_source = query
        if selected_text:
            combined_source = f"{query} {selected_text[:100]}"

        # 提取关键词（去除停用词和短词）
        terms = [
            t for t in re.split(r'[\s,;，。；、？！?!：:""''""]+', combined_source.lower())
            if t and len(t) >= 2
        ]
        if not terms:
            return text[:max_len].strip()

        text_lower = text.lower()

        # 找到所有关键词在文本中的命中位置
        hit_positions = []
        for term in terms:
            start = 0
            while True:
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                hit_positions.append(idx)
                start = idx + len(term)

        if not hit_positions:
            # 没有关键词命中，回退到前 max_len 字符
            return text[:max_len].strip()

        hit_positions.sort()

        # 滑动窗口：找到命中密度最高的 max_len 字符窗口
        best_start = 0
        best_count = 0

        for pos in hit_positions:
            # 窗口起始点：以当前命中位置为中心，向前偏移一半窗口
            window_start = max(0, pos - max_len // 3)
            window_end = window_start + max_len

            # 统计窗口内的命中数
            count = sum(1 for p in hit_positions if window_start <= p < window_end)
            if count > best_count:
                best_count = count
                best_start = window_start

        # 调整到句子边界（尽量不截断句子）
        # 向前找到最近的句子起始（句号、换行等之后）
        if best_start > 0:
            # 在 best_start 前30字符范围内找句子边界
            search_start = max(0, best_start - 30)
            boundary_chars = '。\n.！？!?；;'
            best_boundary = best_start
            for i in range(best_start - 1, search_start - 1, -1):
                if text[i] in boundary_chars:
                    best_boundary = i + 1
                    break
            best_start = best_boundary

        snippet = text[best_start:best_start + max_len].strip()
        return snippet

    @staticmethod
    def _reorder_lost_in_middle(selections: List[dict]) -> List[dict]:
        """Lost-in-the-Middle 缓解：交替排列上下文

        LLM 对长上下文中间的信息容易忽略。将最相关的内容放在开头和结尾，
        次相关的放在中间。

        输入顺序（按相关性降序）：#1, #2, #3, #4, #5, #6
        输出顺序：#1, #3, #5, #6, #4, #2
        即奇数位→开头侧，偶数位→结尾侧（逆序）

        Args:
            selections: 按相关性降序排列的选择列表

        Returns:
            重新排列的选择列表
        """
        if len(selections) <= 2:
            return selections

        head = []  # 开头侧（奇数位：0, 2, 4, ...）
        tail = []  # 结尾侧（偶数位：1, 3, 5, ...）

        for i, item in enumerate(selections):
            if i % 2 == 0:
                head.append(item)
            else:
                tail.append(item)

        # 结尾侧逆序，使第2相关的排在最后
        tail.reverse()

        reordered = head + tail
        logger.info(
            f"Lost-in-the-Middle 重排: {len(selections)} 个意群, "
            f"开头侧={len(head)}, 结尾侧={len(tail)}"
        )
        return reordered

    def build_citation_prompt(self, citations: List[dict]) -> str:
        """生成引文指示提示词，指导 LLM 在回答中使用 [1] [2] 等编号标注引用来源

        根据 citations 列表生成一段系统提示词，告知 LLM 可用的引用编号及其
        对应的来源信息，并要求 LLM 在回答中使用这些编号标注引用来源。

        Args:
            citations: 引文映射列表，每项格式为：
                {"ref": int, "group_id": str, "page_range": [int, int]}

        Returns:
            引文指示提示词字符串。如果 citations 为空，返回空字符串。
        """
        if not citations:
            return ""

        # 构建可用引用编号列表描述
        ref_descriptions = []
        for citation in citations:
            ref_num = citation["ref"]
            group_id = citation["group_id"]
            page_range = citation["page_range"]
            ref_descriptions.append(
                f"[{ref_num}] 来源: {group_id}，页码: {page_range[0]}-{page_range[1]}"
            )

        refs_text = "\n".join(ref_descriptions)

        # 生成引文指示提示词
        prompt = (
            "请在回答中使用引用编号标注信息来源。"
            "当你引用或参考上下文中的内容时，请在相关文字后标注对应的引用编号，"
            "格式为 [编号]，例如 [1]、[2]。\n"
            "\n"
            "可用的引用来源：\n"
            f"{refs_text}\n"
            "\n"
            "注意：\n"
            "- 只能使用“可用的引用来源”里出现过的编号，禁止创造新编号\n"
            "- 禁止改写编号含义：同一个编号必须始终对应同一个来源\n"
            "- 每段引用的内容都应标注来源编号\n"
            "- 可以同时引用多个来源，如 [1][2]\n"
            "- 如果信息来自你的通用知识而非上下文，则无需标注编号\n"
            "- 只引用与用户问题直接相关的来源，不要为了引用而引用不相关的内容\n"
            "- 如果某个来源与用户问题无关，请完全忽略它，不要在回答中提及\n"
            "- 宁可少引用，也不要引用不相关的来源\n"
            "- 输出前请自检：若出现不在列表内的编号，必须删除该编号"
        )

        logger.info(f"引文指示提示词生成完成: {len(citations)} 个引用来源")

        return prompt
