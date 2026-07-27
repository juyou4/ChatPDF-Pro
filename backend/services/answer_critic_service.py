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
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 解析失败（模型已应答但格式不对）重试一次；超时不重试——超时说明模型慢，
# 再等一个完整超时窗口只会让用户多等，收益极低。
_MAX_PARSE_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 0.5

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)

# issue 分类，供前端分色与后续自动修复路由使用。
_ISSUE_TYPES = frozenset({
    "hallucination",        # 上下文完全不支持的论断
    "unsupported_number",   # 数值与证据不符或凭空出现
    "missing_citation",     # 事实句缺 [n]
    "wrong_citation",       # 引用编号指向不支撑该句的证据
    "overreach",            # 结论强度超出证据支持范围
    "other",
})
_DEFAULT_ISSUE_TYPE = "other"
_MAX_ISSUES = 5


class _CriticResponseError(ValueError):
    """带稳定错误码的自审响应错误，便于日志聚合和问题定位。"""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _stringify_upstream_error(error: object) -> str:
    if isinstance(error, str):
        return error.strip()
    if isinstance(error, dict):
        for key in ("message", "detail"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(error, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return str(error or "").strip()


def _extract_response_text(response: object) -> str:
    """从 OpenAI 兼容响应中提取非空正文，并兼容历史字符串 mock。"""
    if isinstance(response, str):
        text = response.strip()
        if not text:
            raise _CriticResponseError("empty_content", "AI 返回正文为空")
        return text

    if not isinstance(response, dict):
        raise _CriticResponseError("invalid_response", "AI 返回不是对象")

    upstream_error = _stringify_upstream_error(response.get("error"))
    if upstream_error:
        raise _CriticResponseError("upstream_error", upstream_error)

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _CriticResponseError("missing_choices", "AI 返回缺少 choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise _CriticResponseError("invalid_choice", "AI 返回 choices[0] 不是对象")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise _CriticResponseError("missing_message", "AI 返回缺少 message")

    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                text_parts.append(part_text.strip())
        text = "\n".join(text_parts).strip()
    else:
        text = ""

    if not text:
        raise _CriticResponseError("empty_content", "AI 返回 message.content 为空")
    return text


def _first_balanced_json_object(text: str) -> str:
    """提取第一个括号平衡的 JSON 对象。

    比贪婪的 `\\{[\\s\\S]*\\}` 可靠：贪婪匹配会从首个 `{` 一直吃到最后一个 `}`，
    模型输出多个对象或 JSON 后跟带花括号的说明文字时会拼出脏内容。字符串内的
    花括号也在此正确跳过。截断（括号未闭合）时返回空串，交由调用方判定失败。
    """
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def _normalize_evidence_refs(raw: object) -> list[int]:
    if not isinstance(raw, (list, tuple)):
        return []
    refs: list[int] = []
    for item in raw:
        try:
            ref = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= ref <= 999 and ref not in refs:
            refs.append(ref)
    return refs[:4]


def _normalize_issues(raw: object, answer: str = "") -> list[dict]:
    """把 issues 归一为结构化对象，同时兼容模型返回纯字符串的情况。

    claim_span 会与答案正文核对：模型有时会「引用」一段答案里并不存在的文字，
    那样的锚点会让前端定位到错误位置，不如丢弃。
    """
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    answer_text = str(answer or "")
    normalized: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or item.get("issue") or item.get("description") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)

        issue_type = str(item.get("issue_type") or item.get("type") or "").strip().lower()
        if issue_type not in _ISSUE_TYPES:
            issue_type = _DEFAULT_ISSUE_TYPE

        claim_span = str(item.get("claim_span") or item.get("span") or "").strip()
        if claim_span and answer_text and claim_span not in answer_text:
            claim_span = ""

        normalized.append({
            "text": text[:200],
            "issue_type": issue_type,
            "claim_span": claim_span[:160],
            "evidence_refs": _normalize_evidence_refs(item.get("evidence_refs")),
        })
        if len(normalized) >= _MAX_ISSUES:
            break
    return normalized


def _parse_critic_json(text: str) -> dict:
    """解析自审响应，容忍 markdown 围栏与 JSON 前后的多余文字。"""
    stripped = str(text or "").strip()
    fence = _JSON_FENCE_RE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()

    candidates = [stripped]
    block = _first_balanced_json_object(stripped)
    if block and block != stripped:
        candidates.append(block)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise _CriticResponseError("invalid_json", "自审响应无法解析为 JSON 对象")


async def critique_answer(
    question: str,
    answer: str,
    context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    endpoint: str = "",
    timeout: float = 10.0,
    evidence_brief: str = "",
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
        evidence_brief: 检索侧证据强度简报（见 build_critic_evidence_brief）。
            只看「答案 + 上下文」时，证据很弱但行文通顺的回答容易被放过。

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
        "You are an academic PDF answer auditor. Evaluate faithfulness to the given context.\n"
        "Also check academic citation discipline:\n"
        "- Factual claims (numbers, methods, comparisons, causal statements) should end with [n] citations.\n"
        "- If the answer correctly refuses with insufficient evidence, that is NOT a hallucination.\n"
        "- Invented numbers, methods, or paper claims not supported by context ARE hallucinations.\n"
        "Output ONLY a JSON object with these fields:\n"
        "- score: integer 0-10 (10=perfectly grounded, 0=completely hallucinated)\n"
        "- has_hallucination: boolean\n"
        "- issues: array of objects, each with:\n"
        "    * text: short description of the problem\n"
        f"    * issue_type: one of {sorted(_ISSUE_TYPES)}\n"
        "    * claim_span: the exact substring COPIED VERBATIM from the answer that is\n"
        "      problematic (empty string if it applies to the whole answer)\n"
        "    * evidence_refs: array of integers, the [n] numbers that should support it\n"
        "- suggestion: one short sentence of advice (empty string if answer is fine)\n"
        "- missing_citations: boolean (true if important factual sentences lack [n])\n"
        "Example issue: {\"text\": \"参数量 70B 在上下文中未出现\", \"issue_type\": \"unsupported_number\", "
        "\"claim_span\": \"该模型参数量为 70B\", \"evidence_refs\": []}\n"
        "claim_span must be an exact copy from the answer so the UI can locate it; "
        "do not paraphrase or shorten it.\n"
        "No explanation outside the JSON."
    )

    brief = str(evidence_brief or "").strip()
    user_prompt = (
        f"Question: {question}\n\n"
        + (f"{brief[:800]}\n\n" if brief else "")
        + f"Context (retrieved from document):\n{context_truncated}\n\n"
        + f"Answer to evaluate:\n{answer_truncated}\n\n"
        + "Evaluate faithfulness, citation coverage on factual claims, and output JSON:"
    )

    try:
        from services.chat_service import call_ai_api

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed: Optional[dict] = None
        for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
            result = await asyncio.wait_for(
                call_ai_api(
                    messages=messages,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                    # 200 tokens 装不下 score + issues 数组 + suggestion 的完整 JSON，
                    # 截断会让整次自审静默失败。
                    max_tokens=500,
                    temperature=0.0,
                ),
                timeout=timeout,
            )
            text = _extract_response_text(result)
            try:
                parsed = _parse_critic_json(text)
                break
            except _CriticResponseError as parse_error:
                if attempt >= _MAX_PARSE_ATTEMPTS:
                    raise
                logger.warning(
                    "[Critic] 自审响应解析失败，重试 %s/%s: %s",
                    attempt,
                    _MAX_PARSE_ATTEMPTS,
                    parse_error.detail,
                    extra={
                        "critic_error_code": parse_error.code,
                        "critic_error_detail": parse_error.detail,
                    },
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

        if not isinstance(parsed, dict):
            raise _CriticResponseError(
                "invalid_payload", "自审 JSON 根节点不是对象"
            )

        # 验证必需字段
        try:
            score = int(parsed.get("score", 5))
        except (TypeError, ValueError):
            score = 5
        score = max(0, min(10, score))

        issue_details = _normalize_issues(parsed.get("issues"), answer=answer_truncated)
        critique = {
            "score": score,
            "has_hallucination": bool(parsed.get("has_hallucination", False)),
            # issues 保留纯文本形态（既有前端契约），结构化数据放在 issue_details。
            "issues": [item["text"] for item in issue_details],
            "issue_details": issue_details,
            "suggestion": str(parsed.get("suggestion", ""))[:200],
            # 缺引用提示文案统一由 postprocess_critic_result 生成（它有精确计数和
            # 原句样例），此处只作为布尔信号上报，避免两层各拼一条近似文案。
            "missing_citations": bool(parsed.get("missing_citations", False)),
        }

        logger.info(
            f"[Critic] 自审完成: score={score}/10, "
            f"hallucination={critique['has_hallucination']}, "
            f"issues={len(critique['issues'])}, "
            f"missing_citations={critique['missing_citations']}"
        )
        return critique

    except asyncio.TimeoutError:
        logger.warning(
            "[Critic] 自审超时(%ss)",
            timeout,
            extra={
                "critic_error_code": "timeout",
                "critic_error_detail": f"timeout={timeout}s",
            },
        )
        return None
    except _CriticResponseError as e:
        logger.warning(
            "[Critic] 自审响应无效: code=%s, detail=%s",
            e.code,
            e.detail,
            extra={
                "critic_error_code": e.code,
                "critic_error_detail": e.detail,
            },
        )
        return None
    except json.JSONDecodeError as e:
        logger.warning(
            "[Critic] 自审结果解析失败: %s",
            e,
            extra={
                "critic_error_code": "invalid_json",
                "critic_error_detail": str(e),
            },
        )
        return None
    except Exception as e:
        logger.warning(
            "[Critic] 自审失败: %s",
            e,
            extra={
                "critic_error_code": "request_failed",
                "critic_error_detail": str(e),
            },
        )
        return None
