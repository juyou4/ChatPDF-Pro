"""
预设问题与生成提示词服务

提供预设问题列表和生成类查询（思维导图、流程图）的系统提示词。
根据用户查询关键词自动匹配对应的生成提示词。
"""

from typing import List, Optional


# ---- 预设问题列表 ----

PRESET_QUESTIONS: List[dict] = [
    {"id": "summarize", "label": "总结本文", "query": "请总结本文的主要内容"},
    {"id": "formulas", "label": "关键公式", "query": "本文有哪些关键公式？请列出并解释"},
    {"id": "methods", "label": "研究方法", "query": "本文使用了什么研究方法？有什么发现？"},
    {"id": "mindmap", "label": "生成思维导图🧠", "query": "生成思维导图"},
    {"id": "flowchart", "label": "生成流程图🔄", "query": "生成流程图"},
]


# ---- Mermaid 流程图系统提示词 ----

MERMAID_SYSTEM_PROMPT: str = """你是一个专业的文档分析助手。请根据文档内容生成一个 Mermaid 流程图。

要求：
1. 使用有效的 Mermaid 语法（graph TD 或 flowchart TD）
2. 节点文本使用中文，简洁明了
3. 用箭头（-->）表示流程方向和逻辑关系
4. 合理使用子图（subgraph）对相关步骤进行分组
5. 确保流程图结构清晰、层次分明
6. 将输出包裹在 ```mermaid 代码块中

示例格式：
```mermaid
graph TD
    A[开始] --> B[步骤1]
    B --> C{判断条件}
    C -->|是| D[步骤2]
    C -->|否| E[步骤3]
    D --> F[结束]
    E --> F
```

请基于文档内容，提取核心流程并生成流程图。"""


# ---- 思维导图系统提示词 ----

MINDMAP_SYSTEM_PROMPT: str = """你是一个专业的文档分析助手。请根据文档内容生成一个结构化的思维导图。

要求：
1. 使用 Markdown 层级结构表示思维导图
2. 用 # 表示中心主题，## 表示主要分支，### 表示子分支，以此类推
3. 每个节点的文本简洁明了（不超过 20 字）
4. 主要分支控制在 3-6 个
5. 每个主要分支下的子节点控制在 2-5 个
6. 覆盖文档的核心内容和关键概念

示例格式：
# 文档主题
## 第一部分
### 要点 1
### 要点 2
## 第二部分
### 要点 3
### 要点 4
## 第三部分
### 要点 5

请基于文档内容，提取核心概念和层级关系，生成思维导图。"""


# ---- 思维导图关键词列表 ----

_MINDMAP_KEYWORDS: List[str] = [
    "思维导图", "脑图", "mindmap", "mind map",
]

# ---- 流程图关键词列表 ----

_FLOWCHART_KEYWORDS: List[str] = [
    "流程图", "flowchart", "flow chart", "mermaid",
]


def get_generation_prompt(query: str) -> Optional[str]:
    """
    根据查询内容返回对应的生成提示词（流程图/思维导图），无匹配返回 None。

    匹配规则：
    - 查找思维导图和流程图关键词在查询中首次出现的位置
    - 如果两种关键词都存在，优先匹配先出现的那个
    - 仅包含思维导图关键词 → 返回 MINDMAP_SYSTEM_PROMPT
    - 仅包含流程图关键词 → 返回 MERMAID_SYSTEM_PROMPT
    - 无匹配 → 返回 None

    Args:
        query: 用户查询文本

    Returns:
        对应的系统提示词字符串，或 None（无匹配时）
    """
    if not query:
        return None

    query_lower = query.lower()

    # 查找思维导图关键词的最早出现位置
    mindmap_pos = _find_earliest_keyword_position(query_lower, _MINDMAP_KEYWORDS)

    # 查找流程图关键词的最早出现位置
    flowchart_pos = _find_earliest_keyword_position(query_lower, _FLOWCHART_KEYWORDS)

    # 两种关键词都未找到
    if mindmap_pos == -1 and flowchart_pos == -1:
        return None

    # 仅找到思维导图关键词
    if mindmap_pos != -1 and flowchart_pos == -1:
        return MINDMAP_SYSTEM_PROMPT

    # 仅找到流程图关键词
    if flowchart_pos != -1 and mindmap_pos == -1:
        return MERMAID_SYSTEM_PROMPT

    # 两种关键词都找到，优先匹配先出现的
    if flowchart_pos <= mindmap_pos:
        return MERMAID_SYSTEM_PROMPT
    else:
        return MINDMAP_SYSTEM_PROMPT


def _find_earliest_keyword_position(text: str, keywords: List[str]) -> int:
    """
    在文本中查找关键词列表中最早出现的位置。

    Args:
        text: 待搜索的文本（应已转为小写）
        keywords: 关键词列表

    Returns:
        最早出现的位置索引，未找到返回 -1
    """
    earliest = -1
    for keyword in keywords:
        pos = text.find(keyword)
        if pos != -1 and (earliest == -1 or pos < earliest):
            earliest = pos
    return earliest
