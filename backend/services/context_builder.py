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


class ContextBuilder:
    """上下文构建器

    将粒度选择结果（意群 + 粒度级别）组装为格式化的上下文字符串，
    包含引用编号、意群标识、粒度级别标注、页码范围和关键词。

    同时生成引文映射列表（citations），用于后续引文追踪功能（任务 11）。
    """

    def build_context(
        self,
        selections: List[dict],
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

        Returns:
            (context_string, citations) 元组
            - context_string: 格式化的上下文字符串
            - citations: 引文映射列表，每项格式为：
                {"ref": int, "group_id": str, "page_range": [int, int]}
        """
        if not selections:
            return "", []

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
            # 取文本前 100 字符作为高亮锚点文本（更短的片段匹配率更高）
            highlight_text = text[:100].strip() if text else ""
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
            "- 每段引用的内容都应标注来源编号\n"
            "- 可以同时引用多个来源，如 [1][2]\n"
            "- 如果信息来自你的通用知识而非上下文，则无需标注编号\n"
            "- 只引用与问题直接相关的来源，不要为了引用而引用不相关的内容\n"
            "- 如果某个来源与问题无关，请不要引用它"
        )

        logger.info(f"引文指示提示词生成完成: {len(citations)} 个引用来源")

        return prompt
