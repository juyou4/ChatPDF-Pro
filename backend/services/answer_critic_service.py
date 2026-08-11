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
import hashlib
import json
import logging
import re
from typing import Any, Optional

from services.completion_outcome import (
    IncompleteCompletionError,
    require_publishable_completion,
)

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
_MAX_CLAIM_VERIFIER_CANDIDATES = 8
_CLAIM_VERDICTS = frozenset({"supported", "unsupported", "contradicted", "uncertain"})
_ANSWER_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_ANSWER_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?%?(?![\w])")
_REPAIRABLE_ISSUE_TYPES = frozenset({
    "hallucination",
    "unsupported_number",
    "missing_citation",
    "wrong_citation",
    "overreach",
})


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


def _normalize_claim_verifier_candidates(claims: object) -> list[dict[str, Any]]:
    """Keep a bounded, ID-addressable verifier payload.

    The verifier is deliberately not allowed to discover new evidence.  Every
    evidence ID and excerpt must already have survived the deterministic
    summary guards before it reaches this function.
    """
    if not isinstance(claims, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    for raw in claims:
        if not isinstance(raw, dict):
            continue
        claim_id = str(raw.get("claim_id") or "").strip()
        claim_text = " ".join(str(raw.get("claim_text") or "").split())
        if not claim_id or not claim_text or claim_id in seen_claim_ids:
            continue
        evidence: list[dict[str, str]] = []
        seen_evidence_ids: set[str] = set()
        for item in raw.get("evidence") if isinstance(raw.get("evidence"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id") or "").strip()
            text = " ".join(str(item.get("text") or "").split())
            if not evidence_id or not text or evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            evidence.append({"evidence_id": evidence_id, "text": text[:900]})
            if len(evidence) >= 4:
                break
        if not evidence:
            continue
        seen_claim_ids.add(claim_id)
        normalized.append({
            "claim_id": claim_id,
            "claim_kind": str(raw.get("claim_kind") or "claim")[:40],
            "source_section_id": str(raw.get("source_section_id") or "")[:120],
            "claim_text": claim_text[:240],
            "evidence": evidence,
        })
        if len(normalized) >= _MAX_CLAIM_VERIFIER_CANDIDATES:
            break
    return normalized


def _normalize_claim_verdicts(
    raw: object,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Authorize verifier output against the exact input claim/evidence IDs."""
    candidate_by_id = {
        str(candidate["claim_id"]): candidate
        for candidate in candidates
        if candidate.get("claim_id")
    }
    raw_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id") or "").strip()
            if claim_id in candidate_by_id and claim_id not in raw_by_id:
                raw_by_id[claim_id] = item

    verdicts: list[dict[str, Any]] = []
    for claim_id, candidate in candidate_by_id.items():
        item = raw_by_id.get(claim_id) or {}
        status = str(item.get("status") or "uncertain").strip().lower()
        if status not in _CLAIM_VERDICTS:
            status = "uncertain"
        allowed_evidence_ids = {
            str(evidence.get("evidence_id") or "")
            for evidence in candidate.get("evidence") or []
            if evidence.get("evidence_id")
        }
        raw_evidence_ids = (
            item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        )
        evidence_ids = list(dict.fromkeys(
            str(value).strip()
            for value in raw_evidence_ids
            if str(value).strip() in allowed_evidence_ids
        ))[:4]
        reason = " ".join(str(item.get("reason") or "").split())[:240]
        if not item:
            reason = "verifier_omitted_claim"
        elif status in {"supported", "contradicted"} and not evidence_ids:
            original_status = status
            status = "uncertain"
            reason = f"{original_status}_without_authorized_evidence"
        verdicts.append({
            "claim_id": claim_id,
            "status": status,
            "reason": reason,
            "evidence_ids": evidence_ids,
        })
    return verdicts


async def critique_evidence_claims(
    claims: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    provider: str = "openai",
    endpoint: str = "",
    timeout: float = 18.0,
) -> Optional[dict[str, Any]]:
    """Independently check a small set of already-bound high-value claims.

    This pass never retrieves new evidence and never authorizes an unknown
    citation.  ``contradicted`` is intentionally distinct from ``unsupported``:
    the former means an attached excerpt points in the opposite direction,
    while the latter means the excerpt does not establish the claim.
    """
    candidates = _normalize_claim_verifier_candidates(claims)
    if (
        not candidates
        or (not api_key and str(provider or "").lower() not in {"local", "ollama"})
    ):
        return None

    system_prompt = (
        "You are an independent academic claim verifier. Check each claim only against "
        "the evidence excerpts attached to that same claim. Preserve subjects, comparison "
        "direction, numbers, scope, uncertainty, negation, and limitation conditions. "
        "Do not use outside knowledge and do not rewrite any claim. Output ONLY JSON: "
        '{"verdicts":[{"claim_id":"...","status":"supported|unsupported|contradicted|uncertain",'
        '"reason":"short reason","evidence_ids":["IDs actually used"]}]}. '
        "Return one verdict for every input claim. A supported or contradicted verdict must cite "
        "at least one evidence_id supplied with that claim. Use contradicted only when the "
        "attached evidence explicitly points in the opposite direction; use unsupported when "
        "it simply does not establish the claim."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Verify these claims:\n" + json.dumps(candidates, ensure_ascii=False),
        },
    ]

    try:
        from services.chat_service import call_ai_api

        parsed: Optional[dict] = None
        for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
            response = await asyncio.wait_for(
                call_ai_api(
                    messages=messages,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                    max_tokens=900,
                    temperature=0.0,
                    purpose="reading_outline_claim_verifier",
                ),
                timeout=timeout,
            )
            try:
                require_publishable_completion(response, operation="claim verifier")
                parsed = _parse_critic_json(_extract_response_text(response))
                break
            except (IncompleteCompletionError, _CriticResponseError):
                if attempt >= _MAX_PARSE_ATTEMPTS:
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

        if not isinstance(parsed, dict):
            raise _CriticResponseError("invalid_payload", "结论核验 JSON 根节点不是对象")
        verdicts = _normalize_claim_verdicts(parsed.get("verdicts"), candidates)
        counts = {
            status: sum(1 for verdict in verdicts if verdict["status"] == status)
            for status in _CLAIM_VERDICTS
        }
        return {
            "status": "completed",
            "candidate_count": len(candidates),
            "supported_count": counts["supported"],
            "unsupported_count": counts["unsupported"],
            "contradicted_count": counts["contradicted"],
            "uncertain_count": counts["uncertain"],
            "verdicts": verdicts,
        }
    except asyncio.TimeoutError:
        logger.warning("[ClaimVerifier] 关键结论核验超时(%ss)", timeout)
    except _CriticResponseError as exc:
        logger.warning(
            "[ClaimVerifier] 关键结论核验响应无效: code=%s detail=%s",
            exc.code,
            exc.detail,
        )
    except Exception as exc:
        logger.warning("[ClaimVerifier] 关键结论核验失败: %s", exc)
    return None


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
        "- For structure, architecture, interaction, or mechanism questions, distinguish high-level topology from missing layer-level implementation parameters. "
        "Flag an answer as overreach if it says the paper gives no structure while the context gives a topology or pipeline.\n"
        "- A local retrieval miss is not evidence that the whole paper lacks something; require an explicit source statement and citation for document-wide absence claims.\n"
        "- For multi-part questions, flag omitted required subquestions and vague absence claims; any statement that something is not provided must name the concrete missing field.\n"
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
            try:
                require_publishable_completion(result, operation="answer critic")
                text = _extract_response_text(result)
                parsed = _parse_critic_json(text)
                break
            except (IncompleteCompletionError, _CriticResponseError) as parse_error:
                if attempt >= _MAX_PARSE_ATTEMPTS:
                    raise
                logger.warning(
                    "[Critic] 自审响应解析失败，重试 %s/%s: %s",
                    attempt,
                    _MAX_PARSE_ATTEMPTS,
                    getattr(parse_error, "detail", str(parse_error)),
                    extra={
                        "critic_error_code": getattr(parse_error, "code", "incomplete_completion"),
                        "critic_error_detail": getattr(parse_error, "detail", str(parse_error)),
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


def _repair_issue_payload(critic: object) -> list[dict[str, Any]]:
    if not isinstance(critic, dict):
        return []
    details = critic.get("issue_details")
    if not isinstance(details, list):
        details = []
    payload: list[dict[str, Any]] = []
    for item in details[:5]:
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("issue_type") or "other").strip().lower()
        if issue_type not in _REPAIRABLE_ISSUE_TYPES:
            continue
        payload.append({
            "issue_type": issue_type,
            "text": str(item.get("text") or "")[:200],
            "claim_span": str(item.get("claim_span") or "")[:160],
            "evidence_refs": _normalize_evidence_refs(item.get("evidence_refs")),
        })
    return payload


def _numbers_without_citations(text: str) -> set[str]:
    without_refs = _ANSWER_CITATION_RE.sub("", str(text or ""))
    return {match.group(0) for match in _ANSWER_NUMBER_RE.finditer(without_refs)}


async def repair_answer_once(
    *,
    question: str,
    answer: str,
    context: str,
    critic: dict,
    allowed_citation_refs: list[int],
    api_key: str,
    model: str,
    provider: str = "openai",
    endpoint: str = "",
    timeout: float = 14.0,
) -> tuple[str, dict[str, Any]]:
    """Attempt one evidence-locked answer repair without any retrieval access."""
    original = str(answer or "").strip()
    evidence = str(context or "").strip()[:7000]
    issues = _repair_issue_payload(critic)
    allowed_refs = sorted({int(value) for value in allowed_citation_refs if int(value) > 0})[:64]
    diagnostics: dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "attempt_count": 0,
        "retrieval_call_count": 0,
        "validation": "not_started",
        "allowed_citation_refs": allowed_refs,
        "evidence_hash": hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:20]
        if evidence
        else "",
    }
    if not original or not evidence or not issues:
        diagnostics["validation"] = "missing_repair_input"
        return original, diagnostics
    if not api_key and str(provider or "").strip().lower() not in {"local", "ollama"}:
        diagnostics["validation"] = "missing_api_key"
        return original, diagnostics

    diagnostics["attempted"] = True
    diagnostics["attempt_count"] = 1
    system_prompt = (
        "You repair one academic PDF answer using ONLY the supplied evidence. "
        "You have no tools and must not use outside knowledge. Remove or weaken unsupported "
        "claims, preserve the user's language and requested scope, and keep supported details. "
        "Use citation markers only from ALLOWED_REFS and place them after supported factual claims. "
        "If evidence is insufficient, say that the current evidence is insufficient instead of "
        "inventing an answer. Output only the repaired answer, with no analysis or JSON."
    )
    user_prompt = (
        f"QUESTION:\n{str(question or '')[:1200]}\n\n"
        f"ALLOWED_REFS: {allowed_refs}\n\n"
        f"CRITIC_ISSUES:\n{json.dumps(issues, ensure_ascii=False)}\n\n"
        f"AUTHORIZED_EVIDENCE:\n{evidence}\n\n"
        f"ORIGINAL_ANSWER:\n{original[:3500]}\n\n"
        "Return the repaired answer only."
    )
    try:
        from services.chat_service import call_ai_api

        response = await asyncio.wait_for(
            call_ai_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=1400,
                temperature=0.0,
                purpose="answer_critic_repair",
            ),
            timeout=timeout,
        )
        require_publishable_completion(response, operation="answer critic repair")
        candidate = _extract_response_text(response).strip()
    except asyncio.TimeoutError:
        diagnostics["validation"] = "timeout"
        return original, diagnostics
    except Exception as exc:
        diagnostics["validation"] = "request_failed"
        diagnostics["error"] = str(exc)[:160]
        return original, diagnostics

    if not candidate:
        diagnostics["validation"] = "empty_answer"
        return original, diagnostics
    if len(candidate) > max(1200, int(len(original) * 1.35) + 300):
        diagnostics["validation"] = "answer_expanded"
        return original, diagnostics
    allowed_ref_set = set(allowed_refs)
    unauthorized_refs = sorted({
        int(value)
        for value in _ANSWER_CITATION_RE.findall(candidate)
        if int(value) not in allowed_ref_set
    })
    if unauthorized_refs:
        diagnostics["validation"] = "unauthorized_citation"
        diagnostics["unauthorized_citation_refs"] = unauthorized_refs
        return original, diagnostics
    introduced_numbers = sorted(
        _numbers_without_citations(candidate) - _numbers_without_citations(evidence)
    )
    if introduced_numbers:
        diagnostics["validation"] = "number_not_in_evidence"
        diagnostics["introduced_numbers"] = introduced_numbers[:12]
        return original, diagnostics
    retained_spans = [
        item["claim_span"]
        for item in issues
        if item.get("claim_span")
        and item.get("issue_type") in {"hallucination", "unsupported_number", "wrong_citation", "overreach"}
        and item["claim_span"] in candidate
    ]
    if retained_spans:
        diagnostics["validation"] = "flagged_claim_retained"
        diagnostics["retained_claim_spans"] = retained_spans[:5]
        return original, diagnostics
    if candidate == original:
        diagnostics["validation"] = "unchanged"
        return original, diagnostics

    diagnostics["accepted"] = True
    diagnostics["validation"] = "passed"
    diagnostics["answer_hash"] = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:20]
    return candidate, diagnostics
