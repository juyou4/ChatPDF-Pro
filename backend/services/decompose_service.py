"""复杂问题分解服务

参考 kotaemon 的 DecomposeQuestionPipeline，使用 LLM 将复杂问题
拆分为最多 3 个子问题，每个子问题独立检索后合并结果。

阶段 3.1（降级版）里本模块是**规则兜底侧**：LLM 不可用 / 未产出
sub_questions 时的降级路径，同时也是 ``tests/eval_intent.py`` 唯一能离线
测到的部分（评测直接调 ``should_decompose(query)`` 这个纯函数）。
"""

import json
import logging
import re
from typing import Optional

from services.intent_constraints import IntentConstraintSet

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You split a user question about ONE document into independently retrievable sub-questions.

Return [] unless the question asks for two or more things that must be looked up SEPARATELY.

Return [] — these are ONE question, not several:
- One quantity measured under several conditions ("the accuracy on different datasets", "模型在不同数据集上的准确率"). A word that merely MODIFIES a noun — "different" / "various" / "不同" / "各" — is NOT a split signal.
- Asking for the CAUSE or mechanism of a difference ("why does this gap appear", "这种差异是怎么产生的"). The difference is the topic, not two targets.
- Any single lookup of a number, name, setting, or yes/no fact.

Split ONLY when the question names two or more DISTINCT targets to contrast ("compare X and Y", "X 和 Y 的区别"),
or chains two or more DISTINCT tasks ("summarize A and then list B", "先…再…").

Hard constraints — violating ANY ONE means return [] instead:
1. At most 3 items. Fewer is better. Never pad the list.
2. Every item must be answerable from the same single document.
3. The union of the items must mean exactly what the original asks — no more, no less.
   Do NOT expand, define, generalize, or reinterpret. Do NOT introduce any entity, model,
   metric, dataset, section, table, figure or number that is not written in the original question.
4. Copy every proper noun, model name, metric name, table/figure number and numeral EXACTLY
   as written. Do not translate, expand or normalize them.
5. Write every item in the SAME LANGUAGE as the user's question.
6. If you are not confident, return []. Returning [] is always safe.

Frozen intent constraints (all sub-questions together must preserve these and
must not introduce new entities, numbers, locators or scope):
{constraints}

Output ONLY JSON, no markdown fences and no explanation:
{{"needs_decompose": true, "sub_questions": ["...", "..."]}}
or
{{"needs_decompose": false, "sub_questions": []}}

User question: {question}
"""

# ---------------------------------------------------------------------------
# 规则触发器（阶段 3.1 降级版）
#
# 设计原则：**只有当问句本身能指认出两个可分别检索的对象、或两个独立任务时
# 才判 True。** 旧实现把「不同 / 差异 / 区别 / 相同」当作裸触发词，于是
#   - 「模型在不同数据集上的准确率是多少？」（「不同」是定语，答案在同一张表里）
#   - 「结果里的这种差异是怎么产生的？」（「差异」是被解释的话题，不是两个对象）
# 这两类单问题被判成要拆分。修法不是往词表里继续打补丁，而是把这批词从
# **裸触发词**降级为**必须带句法上下文**才生效的复合式：
#   比较词做谓词 / 带疑问词 / 前面挂着两个并列成分  → 是比较意图
#   比较词做定语修饰后续名词、或被指示词限定成话题名词 → 不是比较意图
#
# 英文侧一律用 \b token 边界（同 services/query_analyzer.py:_contains_terms 的
# 范式），杜绝 'code' ⊂ 'encoder' 那类子串误命中；'contrast' 也因此不会命中
# 'contrastive learning'。
# ---------------------------------------------------------------------------

# 早退门：过短的问句不值得拆。保持与改造前一致的 10 字符阈值——它不在本轮
# 7 条失败样本里，评测集也没有 6~9 字符的样本，贸然放宽等于拿线上流量做实验。
_MIN_QUESTION_LENGTH = 10

# 「比较」在中文里既是动词（compare）也是程度副词（rather / quite）。
# 副词用法的判别式：比较 + 一个形容词 + 句末 / 语气词 / 标点，后面不接名词性成分。
# 命中即在匹配前抹成中性字符，于是
#   「这个方法比较好用吗？」→ 抹掉，不触发
#   「比较大模型和小模型」→ 「大」后面接「模型」，不抹，正常触发
#   「比较大小」→ 「大」后面接「小」，不抹，正常触发
_ZH_ADVERBIAL_COMPARE = re.compile(
    r"比较(?="
    r"(?:[好差高低快慢大小多少强弱长短难易贵新旧]|容易|困难|简单|复杂|麻烦|明显|"
    r"重要|常见|普遍|清楚|合理|准确|稳定|理想|突出|流行)"
    r"(?:用)?(?:一点|一些|一下)?"
    r"(?:[吗呢吧了的？?。！!，,、；;：:]|$)"
    r")"
)
_ADVERBIAL_PLACEHOLDER = "〇"

# --- 1. 中文：无歧义的显式比较 / 并列指示词（可裸触发） --------------------
# 「对比」排除「对比学习」（contrastive learning）与「对比度」这两个固定名词。
_ZH_EXPLICIT_TRIGGERS = re.compile(
    r"比较"
    r"|对比(?!学习|度)"
    r"|优缺点|优点和缺点|优点与缺点|优势和劣势|优势与劣势|优劣"
    r"|异同"
    r"|分别|各自"
    r"|以及"
)

# --- 2. 中文：不同 / 差异 / 区别 必须带句法上下文才算比较意图 --------------
_ZH_PAIR_TRIGGERS = re.compile(
    # 疑问词 + 比较名词：「有什么区别」「有何不同」「有哪些差异」
    r"有(?:什么|哪些|何|啥|多大|多少)\s*(?:不同|差异|区别|差别|异同|相同|一样|出入)"
    r"|(?:是否|有没有)\s*(?:相同|不同|一致|一样)"
    # 比较名词带谓词性后缀：「区别在于」「不同点」「差异之处」「差别体现在」
    r"|(?:不同|相同|差异|区别|差别)(?:点|之处|在于|体现在|表现在|主要在)"
    # 显式比较介词：「与 X 相比」「相比之下」「比起」「较之」「相较于」
    r"|(?:相比|相较|比起|较之)"
    # 两个并列对象 + 比较名词：「X 和 Y 的区别 / 对比 / 优劣」
    # 「和」后面紧跟「的」时不算并列（挡住「加权求和的不同实现」这类）。
    r"|[^，。？！；、]{1,16}[和与跟]\s*(?!的)[^，。？！；、]{1,16}?\s*(?:的)?\s*"
    r"(?:区别|差异|不同|异同|对比|优劣|优缺点|差别)"
    # 两个并列对象 + 关系：「X 和 Y（之间）的关系」
    r"|[^，。？！；、]{1,16}[和与跟]\s*(?!的)[^，。？！；、]{1,16}?\s*(?:之间)?的关系"
    # 显式两个对象：「这两种方案的区别」「两者哪个更好」
    r"|两(?:者|种|个|类|组|款|方)[^，。？！]{0,12}"
    r"(?:区别|差异|不同|异同|对比|比较|优劣|优缺点|哪个|哪种|哪一)"
    # 前者 / 后者对举
    r"|前者[^。？！]{0,16}后者|后者[^。？！]{0,16}前者"
    # 复数指代 + 比较名词（两个对象由上文给出）：「它们的区别」「这些方法的差异」
    # 刻意不含「不同」，否则「各个数据集上的不同表现」会误命中。
    r"|(?:它们|他们|她们|这些|那些|两者|双方)[^，。？！]{0,8}"
    r"(?:区别|差异|异同|对比|差别|优劣)"
)

# --- 3. 顺序 / 任务链复合请求（中英） --------------------------------------
# 判别关键：「先…再…」既可能是**要求助手做两件事**（先总结，再说明），
# 也可能是**描述方法本身的流程**（这个方法先训练再微调）。区分手段是要求
# 「先」之后出现一个**任务动词**——描述流程的句子里出现的是领域动词
# （训练 / 编码 / 采样），不会是「总结 / 说明 / 列出」。
_TASK_VERB_ZH = (
    r"(?:总结|概括|归纳|梳理|说明|解释|阐述|介绍|列举|列出|列|给出|分析|讨论|评价|"
    r"评估|比较|对比|告诉|描述|指出|翻译|提取|整理|找出|展开|谈|讲|说)"
)
# 英文任务动词只收明确的祈使动词。刻意排除 state / note / answer / report 这类
# 同形名词（"and state transitions" / "and answer quality" 会误命中）。
_TASK_VERB_EN = (
    r"(?:list|explain|describe|summari[sz]e|analy[sz]e|discuss|evaluate|compare|"
    r"contrast|outline|elaborate|justify|identify|extract|highlight|provide|"
    r"give|show|tell|mention)"
)

_SEQUENTIAL_TRIGGERS = re.compile(
    r"先[^。？！?]{0,8}?" + _TASK_VERB_ZH + r"[^。？！?]{0,30}?(?:再|然后|接着|其次|之后|最后)"
    r"|一方面[^。？！?]{0,30}?另一方面"
    r"|\band\s+then\s+(?:please\s+)?(?:also\s+)?" + _TASK_VERB_EN + r"\b"
    r"|\bfirst\b[^.?!]{0,60}?\bthen\b\s*(?:please\s+)?" + _TASK_VERB_EN + r"\b",
    re.IGNORECASE,
)

# --- 4. 英文触发词（这里才是 re.IGNORECASE 第一次真正起作用的地方） --------
_EN_TRIGGERS = re.compile(
    r"\bcompare[sd]?\b|\bcomparing\b|\bcomparison[s]?\b"
    r"|\bcontrast(?:s|ed|ing)?\b"
    r"|\bversus\b|\bvs\.?(?![a-z0-9])"
    r"|\bdifference[s]?\s+between\b|\bdistinction[s]?\s+between\b"
    # 谓词性的 different from / differ from 才算比较；
    # 定语性的 "different datasets" 一律不算（对应中文的「不同数据集」）。
    r"|\bdifferent\s+from\b|\bdiffers?\s+from\b|\bdiffered\s+from\b"
    r"|\bdistinguish(?:es|ed)?\s+between\b"
    r"|\bpros\s+and\s+cons\b"
    r"|\badvantages?\s+and\s+disadvantages\b"
    r"|\bstrengths?\s+and\s+weakness(?:es)?\b"
    r"|\bbenefits?\s+and\s+(?:drawbacks?|limitations?)\b"
    r"|\btrade-?offs?\b"
    r"|\brespectively\b"
    r"|\beach\s+of\s+the\s+two\b"
    # 任务链：「... and explain ...」「... and list ...」
    r"|\band\s+(?:also\s+)?" + _TASK_VERB_EN + r"\b",
    re.IGNORECASE,
)

_TRIGGER_GROUPS = (
    ("zh_explicit", _ZH_EXPLICIT_TRIGGERS),
    ("zh_pair", _ZH_PAIR_TRIGGERS),
    ("sequential", _SEQUENTIAL_TRIGGERS),
    ("en", _EN_TRIGGERS),
)


def _decompose_probe(question: str) -> str:
    """规范化待匹配文本：去空白 + 抹掉副词性的「比较」。"""
    text = str(question or "").strip()
    if not text:
        return ""
    return _ZH_ADVERBIAL_COMPARE.sub(_ADVERBIAL_PLACEHOLDER, text)


def matched_decompose_rule(question: str) -> Optional[str]:
    """返回命中的规则组名，用于日志 / 排障；未命中返回 None。

    与 ``should_decompose`` 共用同一套正则，不会出现两者结论不一致的情况。
    """
    probe = _decompose_probe(question)
    if len(probe) < _MIN_QUESTION_LENGTH:
        return None
    for name, pattern in _TRIGGER_GROUPS:
        if pattern.search(probe):
            return name
    return None


def should_decompose(question: str) -> bool:
    """快速判断是否值得尝试问题分解（纯规则，不调 LLM）。

    判 True 的充分条件（任一）：
      1. 中文显式比较 / 并列指示词：比较、对比、优缺点、异同、分别、各自、以及
      2. 中文比较复合式：有什么区别、区别在于、与 X 相比、X 和 Y 的区别、
         这两种…的区别、前者…后者、它们的差异
      3. 顺序 / 任务链：先<任务动词>…再…、and then <verb>、first … then <verb>
      4. 英文比较词：compare、differences between、different from、pros and cons、
         trade-offs、respectively、… and explain/list/…

    刻意**不**触发的情形（这三类是本轮修的假阳性）：
      - 「不同 / different」做定语修饰名词：「不同数据集上的准确率」
      - 问差异的成因：「这种差异是怎么产生的」（只有一个话题，没有两个对象）
      - 「比较」做程度副词：「这个方法比较好用吗」

    Args:
        question: 用户原始问题

    Returns:
        True 如果问题可能需要分解
    """
    return matched_decompose_rule(question) is not None


def _is_numeric_table_question(question: str) -> bool:
    """表格 / 数值类问句判定，用于把它们挡在 LLM 分解链之外。

    红线：numeric_table 走确定性路径，一个 token 都不许进 LLM 链。
    任何导入 / 判定异常都按「不是」处理（fail-open，不阻断主流程）。
    """
    text = str(question or "").strip()
    if not text:
        return False
    try:
        from services.query_analyzer import (
            _is_numeric_table_cost_query,
            analyze_evidence_need,
        )

        if _is_numeric_table_cost_query(text.lower()):
            return True
        return "numeric_table" in (analyze_evidence_need(text) or [])
    except Exception as exc:  # pragma: no cover - 依赖缺失时不阻断
        logger.debug(f"[Decompose] numeric_table 判定跳过: {exc}")
        return False


def _extract_json_payload(content: str) -> object:
    """从 LLM 文本中提取 JSON，兼容 markdown 代码块和前后解释文本。"""
    text = str(content or "").strip()
    if not text:
        return {}

    # 推理模型会在正文前挂一段 <think>…</think>，先剥掉再找 JSON。
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

    candidates = [text]
    for start_token, end_token in (("{", "}"), ("[", "]")):
        start = text.find(start_token)
        end = text.rfind(end_token)
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return {}


def _normalize_decompose_payload(parsed: object) -> tuple[bool, list[str]]:
    if isinstance(parsed, list):
        subs = parsed
        needs_decompose = bool(subs)
    elif isinstance(parsed, dict):
        subs = (
            parsed.get("sub_questions")
            or parsed.get("subQuestions")
            or parsed.get("sub_queries")
            or parsed.get("subqueries")
            or parsed.get("questions")
            or []
        )
        needs_value = (
            parsed.get("needs_decompose")
            if "needs_decompose" in parsed
            else parsed.get("needsDecompose")
        )
        if needs_value is None:
            needs_decompose = bool(subs)
        elif isinstance(needs_value, bool):
            needs_decompose = needs_value
        elif isinstance(needs_value, str):
            needs_decompose = needs_value.strip().lower() in {"true", "yes", "1"}
        else:
            needs_decompose = bool(needs_value)
    else:
        return False, []

    if not isinstance(subs, list):
        return False, []
    normalized = [q.strip() for q in subs if isinstance(q, str) and q.strip()]
    return needs_decompose and bool(normalized), normalized[:3]


async def decompose_question(
    question: str,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
) -> list[str]:
    """使用 LLM 将复杂问题分解为子问题

    Args:
        question: 用户原始问题
        api_key: LLM API 密钥
        model: LLM 模型
        provider: LLM 提供商
        endpoint: API 端点

    Returns:
        子问题列表，不需要分解时返回空列表
    """
    if not should_decompose(question):
        return []

    if not api_key:
        return []

    # 硬短路：表格 / 数值类问句一个 token 都不进 LLM 分解链。
    if _is_numeric_table_question(question):
        logger.debug("[Decompose] numeric_table 问句，跳过 LLM 分解")
        return []

    try:
        from services.chat_service import call_ai_api

        constraints = IntentConstraintSet.from_text(question)
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            constraints=constraints.prompt_guard(),
        )

        response = await call_ai_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=200,
            temperature=0.3,
        )

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                return []
            content = response.get("content", "")
            choices = response.get("choices", [])
            if not content and choices:
                content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response) if response else ""

        content = content.strip()
        if not content:
            return []

        parsed = _extract_json_payload(content)
        needs_decompose, subs = _normalize_decompose_payload(parsed)
        if not needs_decompose:
            return []

        if subs:
            validation = constraints.validate_subquestions(subs)
            if not validation.allowed:
                logger.warning(
                    "[Decompose] 拒绝违反意图约束的分解 constraint_id=%s violations=%s",
                    constraints.constraint_id,
                    list(validation.violations),
                )
                return []
            logger.info(f"[Decompose] 问题分解为 {len(subs)} 个子问题: {subs}")
            return subs

        return []

    except json.JSONDecodeError:
        logger.debug(f"[Decompose] JSON 解析失败: {content[:200]}")
        return []
    except Exception as e:
        logger.warning(f"[Decompose] 问题分解失败: {e}")
        return []
