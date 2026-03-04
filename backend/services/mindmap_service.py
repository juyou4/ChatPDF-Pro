"""思维导图生成服务

参考 kotaemon 的 CreateMindmapPipeline：
1. LLM 生成 PlantUML 格式的思维导图
2. convert_uml_to_markdown() 转为 Markdown heading 格式
3. 前端使用 markmap 渲染
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MINDMAP_PROMPT = """你是 MapGPT，一个专门生成思维导图的 AI 助手。

请根据以下问题和上下文内容，生成一个结构清晰的思维导图。
使用 PlantUML mindmap 格式输出。

格式要求：
@startmindmap
* 中心主题
** 分支1
*** 子项1.1
*** 子项1.2
** 分支2
*** 子项2.1
@endmindmap

注意：
- 使用与用户相同的语言
- 中心主题应该是问题的核心概念
- 分支应涵盖上下文中的关键信息
- 保持层级不超过 4 层
- 每个分支不超过 5 个子项
- 只输出 @startmindmap 和 @endmindmap 之间的内容

问题：{question}

上下文：
{context}
"""


def convert_uml_to_markdown(uml_text: str) -> str:
    """将 PlantUML mindmap 格式转为 Markdown heading 格式

    PlantUML: * 一级 → # 一级
              ** 二级 → ## 二级
              *** 三级 → ### 三级

    Args:
        uml_text: PlantUML mindmap 文本

    Returns:
        Markdown heading 格式文本
    """
    # 提取 @startmindmap 和 @endmindmap 之间的内容
    match = re.search(
        r"@startmindmap\s*(.*?)\s*@endmindmap",
        uml_text,
        re.DOTALL | re.IGNORECASE,
    )
    content = match.group(1).strip() if match else uml_text.strip()

    lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # 计算 * 的数量
        star_match = re.match(r"^(\*+)\s*(.*)", stripped)
        if star_match:
            level = len(star_match.group(1))
            text = star_match.group(2).strip()
            # * → #, ** → ##, etc.
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(stripped)

    return "\n".join(lines)


async def generate_mindmap(
    question: str,
    context: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> Optional[str]:
    """生成思维导图 Markdown

    Args:
        question: 用户问题
        context: 文档上下文（截断到前 2000 字）
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点

    Returns:
        Markdown heading 格式的思维导图文本，失败返回 None
    """
    if not api_key or not context:
        return None

    try:
        from services.chat_service import call_ai_api

        prompt = MINDMAP_PROMPT.format(
            question=question,
            context=context[:2000],
        )

        response = await call_ai_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=800,
            temperature=0.3,
        )

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                logger.warning(f"[Mindmap] LLM 调用失败: {response['error']}")
                return None
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        content = content.strip()
        if not content:
            return None

        markdown = convert_uml_to_markdown(content)
        if markdown and len(markdown) > 10:
            logger.info(f"[Mindmap] 生成成功, {len(markdown)} 字符")
            return markdown

        return None

    except Exception as e:
        logger.warning(f"[Mindmap] 生成失败: {e}")
        return None
