"""
答案自审服务 — 检测幻觉与上下文一致性

参考 PaperBanana critic_agent.py 的多轮审查策略：
- 流式回答结束后，用 cheap model 做一轮自审
- 检查答案与检索上下文的一致性
- 检查是否有幻觉（答案中包含上下文未提及的事实性声明）
- 生成置信度评分和可选警告

配置：默认关闭（增加延迟），通过 config.enable_answer_critic 启用
"""
import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def critique_answer(
    question: str,
    answer: str,
    context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    endpoint: str = "",
    timeout: float = 10.0,
) -> Optional[dict]:
    """对 LLM 回答进行自审，检测幻觉和不一致

    Args:
        question: 用户原始问题
        answer: LLM 生成的回答
        context: 检索到的上下文文本
        api_key: API 密钥
        model: 审查用模型（建议 cheap model）
        provider: 模型提供商
        endpoint: API 端点
        timeout: 超时时间（秒）

    Returns:
        审查结果字典:
        {
            "score": 0-10,          # 整体可信度
            "has_hallucination": bool,
            "issues": ["..."],      # 发现的问题列表
            "suggestion": "..."     # 简短建议
        }
        超时或失败返回 None
    """
    if not api_key or not answer or not context:
        return None

    # 截断避免超长
    answer_truncated = answer[:3000]
    context_truncated = context[:6000]

    system_prompt = (
        "You are an answer quality auditor. Evaluate if the answer is faithful to the given context.\n"
        "Output ONLY a JSON object with these fields:\n"
        "- score: integer 0-10 (10=perfectly grounded, 0=completely hallucinated)\n"
        "- has_hallucination: boolean\n"
        "- issues: array of short strings describing problems found (empty if none)\n"
        "- suggestion: one short sentence of advice (empty string if answer is fine)\n"
        "No explanation outside the JSON."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context (retrieved from document):\n{context_truncated}\n\n"
        f"Answer to evaluate:\n{answer_truncated}\n\n"
        "Evaluate faithfulness and output JSON:"
    )

    try:
        from services.chat_service import call_ai_api

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await asyncio.wait_for(
            call_ai_api(
                messages=messages,
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=200,
                temperature=0.0,
            ),
            timeout=timeout,
        )

        if not result:
            return None

        # 解析 JSON
        text = result.strip()
        # 提取 JSON 对象
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(text)

        # 验证必需字段
        score = int(parsed.get("score", 5))
        score = max(0, min(10, score))

        critique = {
            "score": score,
            "has_hallucination": bool(parsed.get("has_hallucination", False)),
            "issues": parsed.get("issues", [])[:5],  # 最多 5 个问题
            "suggestion": str(parsed.get("suggestion", ""))[:200],
        }

        logger.info(
            f"[Critic] 自审完成: score={score}/10, "
            f"hallucination={critique['has_hallucination']}, "
            f"issues={len(critique['issues'])}"
        )
        return critique

    except asyncio.TimeoutError:
        logger.warning(f"[Critic] 自审超时({timeout}s)")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"[Critic] 自审结果解析失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[Critic] 自审失败: {e}")
        return None
