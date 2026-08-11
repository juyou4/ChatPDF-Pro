"""学术论文阅读的回答合同（paper-qa 风格，适配单篇 PDF）。

职责：
1. 按 intent.task 生成科学文体指令
2. 拒答哨兵（cannot answer）与句级引用约束
3. 确定性标签 Certain / Partial / Unsure / Refused
4. 对 critic 结果做学术向后处理（缺引用 claim 检测）

不负责检索；只约束「怎么写答案」与「怎么标可信度」。
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from services.memory_quality import find_unscoped_document_absence_claims
# paper-qa 哨兵；中文产品默认用中文句，英文问句可匹配英文。
CANNOT_ANSWER_PHRASE_ZH = "根据文档内容无法回答此问题"
CANNOT_ANSWER_PHRASE_EN = "I cannot answer"

_CANNOT_ANSWER_RE = re.compile(
    r"(?:根据文档内容无法回答此问题|文档中未明确|文档未明确|无法从(?:检索|文档)|"
    r"信息不足|不足以回答|"
    r"\bi\s+cannot\s+answer\b|\binsufficient\s+(?:evidence|information|context)\b|"
    r"\bnot\s+(?:enough|sufficient)\s+(?:evidence|information)\b)",
    re.IGNORECASE,
)

# 与 chat_routes._INLINE_CITATION_PATTERN 保持同一识别标准：半角 [n] 与遗留全角【n】
# 都算合法行内引用；`(?<!!)` 与 `(?!\()` 避免把 markdown 图片/链接引用误计为引用。
_CITATION_RE = re.compile(r"(?<!!)(?:\[(\d{1,3})\](?!\()|【(\d{1,3})】)")
# 中文句末标点后通常没有空格，若强制要求 \s+ 会把整段中文当成一句，事实句统计
# 随之失真（覆盖率恒为 0 或 1）。ASCII 句点仍要求后随空白，避免切开 95.2 这类小数。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+")

# 分句用的成对符号。学术答案里括号补充说明常含句末标点（如「（见表 3。）」），
# 引号里也常包含完整句子，配对未闭合时不应切句。
_PAIR_OPENERS = {
    "「": "」",
    "『": "』",
    "《": "》",
    "“": "”",
    "（": "）",
    "(": ")",
}
# 对称引号无法用栈区分开合，改用开/合切换。TrustRAG 的 cut() 在这里有 bug：
# 它先判断 `char in quote_pairs.keys()`，导致成对的 " 每次都被当作开引号入栈，
# 永远无法闭合。
_SYMMETRIC_QUOTES = frozenset("\"'")
_CJK_SENTENCE_END = frozenset("。！？；\n")
_ASCII_SENTENCE_END = frozenset(".!?")
# 句末标点后紧跟的引用角标属于上一句（「…准确率。[1]」），归错会让该句被判缺引用。
_LEADING_CITATION_RE = re.compile(r"^\s*((?:[\[【]\d{1,3}[\]】]\s*)+)")

# 结构化引文协议行（CITATION LIST 段），不是回答正文，必须排除在事实句统计之外。
_CITATION_PROTOCOL_LINE_RE = re.compile(
    r"^\s*(?:CITATION\s*[\[【]?\d{0,3}[\]】]?|CITATION\s+LIST|FINAL\s+ANSWER|"
    r"START_PHRASE\s*[:：]|END_PHRASE\s*[:：]|//)",
    re.IGNORECASE,
)

# 粗略「事实 claim」信号：含数字、方法/指标词、比较词时更需要引用
_FACTUAL_CLAIM_RE = re.compile(
    r"(?:\d+(?:\.\d+)?%?|"
    r"准确率|精度|召回|提升|下降|优于|低于|高于|表明|证明|提出|采用|达到|"
    r"accuracy|precision|recall|f1|bleu|outperform|propose|achieve|show|demonstrate|"
    r"method|model|dataset|baseline|ablation)",
    re.IGNORECASE,
)

# 注意：`\b` 只能加在拉丁词后面。中文字符在 Unicode 模式下也是 \w，
# 「首先」后面紧跟汉字时不存在词边界，此前整条中文分支只在后接标点时才生效
# （「首先，我们…」命中，「首先我们…」漏掉）。
_TRANSITION_RE = re.compile(
    r"^(?:"
    r"(?:首先|其次|再次|然后|接着|最后|总之|综上|另外|此外|下面|以下|如下|因此|所以|值得注意的是)"
    r"|(?:first|second|third|finally|in\s+summary|overall|therefore|thus|next|"
    r"moreover|furthermore|in\s+addition)\b"
    r")",
    re.IGNORECASE,
)

# 章节/图表/公式等导航性指代：句中出现的数字来自编号而非事实断言。
# 判定事实句前先把它们去掉，否则「第 3 节介绍了整体框架。」会因为含数字被
# 要求标 [n]，制造大量假的「缺少引用」告警。
_STRUCTURAL_REF_RE = re.compile(
    r"第\s*\d+(?:\.\d+)*\s*(?:节|章|部分|小节|页|条|款|项|步|轮)"
    r"|(?:图|表|式|公式|算法|附录|章节)\s*\d+(?:[-.．]\d+)*"
    r"|共\s*\d+\s*(?:章|节|个|部分|点|条|类|步|方面)"
    r"|分\s*\d+\s*(?:点|条|个|方面|步|类|部分)"
    r"|(?:section|sec|figure|fig|table|tab|equation|eq|algorithm|alg|appendix|chapter|step)"
    r"\.?\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)

Certainty = str  # Certain | Partial | Unsure | Refused


def _is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in _CJK_SENTENCE_END:
        return True
    if char == "…":
        # 「……」按一个边界处理，不要切出空片段。
        return index + 1 >= len(text) or text[index + 1] != "…"
    if char in _ASCII_SENTENCE_END:
        # ASCII 句点必须后随空白或行尾，否则会切开 95.2 这类小数与 e.g. 之类缩写。
        return index + 1 >= len(text) or text[index + 1].isspace()
    return False


def _attach_leading_citations(sentences: list[str]) -> list[str]:
    """把落到下一句开头的引用角标移回它真正修饰的那一句。"""
    merged: list[str] = []
    for sentence in sentences:
        match = _LEADING_CITATION_RE.match(sentence) if merged else None
        if match:
            merged[-1] = merged[-1] + match.group(1).strip()
            # 整段只有角标时全部并入上一句，不留下无正文的碎片。
            sentence = sentence[match.end():]
        if sentence.strip():
            merged.append(sentence)
    return merged


def split_sentences(text: str) -> list[str]:
    """引号/括号感知分句。

    纯正则切分会在含引号术语或括号补充说明的答案上切错，把一句拆成两句后，
    带数字的那半句因为不含 [n] 被判成「缺引用的事实句」。这里按字符扫描并
    维护配对栈，配对未闭合时不切句。

    防退化：模型输出常出现不配对的引号或括号，若栈永不闭合就会整段不切，
    覆盖率统计随之失真（正是此前中文分句失效的同类故障）。因此扫描结果若
    明显少于句末标点数，就回退到正则切分。
    """
    raw = str(text or "")
    if not raw.strip():
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    stack: list[str] = []
    open_quotes: set[str] = set()

    for index, char in enumerate(raw):
        buffer.append(char)
        if char in _SYMMETRIC_QUOTES:
            open_quotes.symmetric_difference_update({char})
        elif char in _PAIR_OPENERS:
            stack.append(char)
        elif stack and char == _PAIR_OPENERS[stack[-1]]:
            stack.pop()

        if stack or open_quotes:
            continue
        if _is_sentence_boundary(raw, index):
            sentences.append("".join(buffer))
            buffer = []

    if buffer:
        sentences.append("".join(buffer))

    sentences = _attach_leading_citations(
        [part for part in sentences if part.strip()]
    )

    # 退化判定只看「扫描结束时仍有未闭合的配对」——那才说明边界被吞掉了。
    # 不能用「句末标点数 > 句子数」来判断：括号或引号里合法地含有句末标点时
    # （如「该方法优于基线（详见表 3。）」）本就是一句，会被误判成退化。
    if (stack or open_quotes) and len(sentences) <= 1:
        fallback = [part for part in _SENTENCE_SPLIT_RE.split(raw) if part and part.strip()]
        if len(fallback) > len(sentences):
            return _attach_leading_citations(fallback)
    return sentences


def extract_evidence_signals(retrieval_meta: Optional[dict]) -> dict[str, Any]:
    """从 retrieval_meta 里取出证据侧信号（agent 路径会嵌在 diagnostics.agent 下）。

    确定性标签与 critic prompt 都要读这几个字段，抽出来避免两处各写一遍嵌套查找
    后逐渐漂移。
    """
    meta = retrieval_meta if isinstance(retrieval_meta, dict) else {}
    diagnostics = meta.get("diagnostics")
    if isinstance(diagnostics, dict):
        nested = diagnostics.get("agent")
        agent_diag = nested if isinstance(nested, dict) else diagnostics
    else:
        agent_diag = {}

    def _pick(key: str) -> Optional[dict]:
        value = meta.get(key)
        if not isinstance(value, dict):
            value = agent_diag.get(key) if isinstance(agent_diag, dict) else None
        return value if isinstance(value, dict) else None

    scoring = _pick("evidence_scoring")
    evidence_state = meta.get("agent_evidence_state")
    if not isinstance(evidence_state, dict):
        evidence_state = _pick("evidence_state")
    return {
        "evidence_scoring": scoring,
        "evidence_state": evidence_state,
        "sufficiency": _pick("sufficiency"),
        "max_relevance_score": meta.get("max_relevance_score"),
        "degraded": bool(meta.get("agent_fallback") or meta.get("retrieval_degraded")),
    }


def build_critic_evidence_brief(retrieval_meta: Optional[dict]) -> str:
    """把检索侧已有的证据强度信号压成一段给 critic 的简报。

    借鉴 paper-qa 的思路：证据质量是判断答案可信度的前置输入。只给审查模型
    看「答案 + 上下文」时，证据本身很弱但行文通顺的回答容易被放过。
    """
    signals = extract_evidence_signals(retrieval_meta)
    lines: list[str] = []

    scoring = signals["evidence_scoring"]
    if isinstance(scoring, dict) and scoring.get("applied"):
        lines.append(
            "- Evidence scoring: {high} high-relevance, {mid} mid, {dropped} dropped as irrelevant.".format(
                high=int(scoring.get("high_score_count") or 0),
                mid=int(scoring.get("mid_score_count") or 0),
                dropped=int(scoring.get("dropped_count") or 0),
            )
        )

    state = signals["evidence_state"]
    if isinstance(state, dict) and str(state.get("status") or "").strip():
        lines.append(f"- Retrieval state: {str(state.get('status')).strip()}.")

    sufficiency = signals["sufficiency"]
    if isinstance(sufficiency, dict) and str(sufficiency.get("level") or "").strip():
        lines.append(f"- Evidence sufficiency: {str(sufficiency.get('level')).strip()}.")

    max_rel = signals["max_relevance_score"]
    try:
        if max_rel is not None:
            lines.append(f"- Top retrieval relevance: {float(max_rel):.2f} (0-1 scale).")
    except (TypeError, ValueError):
        pass

    if signals["degraded"]:
        lines.append("- Retrieval ran in degraded/fallback mode; evidence may be incomplete.")

    if not lines:
        return ""
    return (
        "Evidence-side signals from the retrieval stage (use them to calibrate strictness; "
        "weak evidence + confident prose is a strong hallucination signal):\n"
        + "\n".join(lines)
    )


def _coverage_ratio(coverage: Optional[dict]) -> float:
    """取引用覆盖率，缺省视为 1.0。

    不能写成 `coverage.get("coverage") or 1.0`：覆盖率 0.0 是假值，会被当成缺省
    而翻转为 1.0，让「完全没有引用」的答案看起来完美。
    """
    raw = (coverage or {}).get("coverage")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 1.0
    return float(raw)


def is_cannot_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False
    if CANNOT_ANSWER_PHRASE_ZH in text or CANNOT_ANSWER_PHRASE_EN.lower() in text.lower():
        return True
    return bool(_CANNOT_ANSWER_RE.search(text[:400]))


def build_academic_style_prompt(
    *,
    task: str = "qa",
    query_type: str = "",
    answer_detail: str = "standard",
) -> str:
    """Scientific writing style overlay for academic PDF QA."""
    task_key = str(task or "qa").strip().lower()
    detail = str(answer_detail or "standard").strip().lower()
    qtype = str(query_type or "").strip().lower()

    length_hint = {
        "concise": "控制在 200-400 字",
        "detailed": "可用小标题展开，目标 800-1500 字，但仍保持信息密度",
        "standard": "目标 250-700 字，先结论后依据",
    }.get(detail, "目标 250-700 字，先结论后依据")

    task_lines = {
        "summarize": (
            "- 任务形态：总结。先用 2-3 句概括贡献与方法，再列关键要点；"
            "不要扩写成完整综述，也不要引入文档外相关工作。"
        ),
        "explain": (
            "- 任务形态：解释。先定义对象，再说明机制/流程；"
            "公式与符号必须照抄原文，缺失处写「文档未明确说明」。"
        ),
        "compare": (
            "- 任务形态：比较。先对齐比较维度，再逐项给出各方数值或表述；"
            "禁止只给笼统「更好/更差」而不列依据。"
        ),
        "extract": (
            "- 任务形态：抽取。只输出问题要求的事实/数值/条目；"
            "禁止背景铺垫和自由发挥。"
        ),
        "calculate": (
            "- 任务形态：计算。先列出原文中的原始数值，再给出计算步骤与结果；"
            "原始数值必须可追溯到引用。"
        ),
        "translate": (
            "- 任务形态：翻译。忠实于证据文本的语义；不要借翻译补充未出现的解释。"
        ),
        "inventory": (
            "- 任务形态：枚举。按文档顺序完整列出，不要抽样省略。"
        ),
    }.get(
        task_key,
        "- 任务形态：学术问答。只回答问题直接要求的信息，不主动扩展背景、局限或未来工作。",
    )

    if qtype == "overview":
        task_lines += "\n- 本题偏概览：允许覆盖问题/方法/结果三块，但仍须每块有证据支撑。"
    elif qtype == "extraction":
        task_lines += "\n- 本题偏精确抽取：优先短答，禁止综述腔。"

    architecture_coverage_rule = (
        "- 遇到结构、架构、交互或机制类问题：先陈述证据已明确给出的架构级拓扑或流程，"
        "再单列尚未公开的逐层实现参数；不得把“层数、通道、投影、超参等细节未给出”"
        "推导成“整体结构未给出”。\n"
        "- 复合问题要逐项回答。任何“文档未给出/未说明”的判断必须限定到具体字段，"
        "不能用一个缺失细节否定已有的架构级证据。"
    )
    absence_evidence_rule = (
        "- For any document-wide absence claim, distinguish the supplied evidence from the whole paper. "
        "A local retrieval miss is not proof of absence; assert it only when an explicit source statement supports it and cite [n]. "
        "Otherwise state that the current evidence does not cover the field or use the refusal sentence."

    )
    return (
        "【学术论文回答文体（Scientific Answer Contract）】\n"
        "- 用科研论文式简洁书面语：短句、连贯段落，避免口语客套与营销措辞。\n"
        f"- {length_hint}。\n"
        f"{task_lines}\n"
        f"{absence_evidence_rule}\n"
        f"{architecture_coverage_rule}\n"
        f"- 若证据不足，使用固定拒答句：「{CANNOT_ANSWER_PHRASE_ZH}」，并简要说明缺什么；"
        f"英文场景可用 “{CANNOT_ANSWER_PHRASE_EN}”。\n"
        "- 禁止用预训练知识补全文档未给出的数值、超参、作者主张或实验设置。\n"
        "- 每个关键事实声明（数值、因果、方法归属、比较结论）句末使用 [n] 引用；"
        "过渡句与结构提示可不引。"
    )


def build_compact_academic_contract_prompt(
    *,
    agent_mode: bool = False,
    allow_web_evidence: bool = False,
) -> str:
    """Compact contract for agent path (full handbook is skipped there)."""
    focus = (
        "你刚完成多跳检索。请只基于已提供的编号证据回答，不要复述检索过程。"
        if agent_mode
        else "请严格依据检索证据回答。"
    )
    evidence_scope = "已提供的文档证据和联网证据" if allow_web_evidence else "已提供的文档证据"
    citation_rule = (
        "- 文档事实使用 [n]，联网事实使用 [Wn]；不得把网页编号冒充文档编号。\n"
        if allow_web_evidence
        else "- 每个含事实的句子末尾附 [n]；单句最多两个编号，如 [2][5]。\n"
    )
    return (
        "【学术忠实性合同 · 精简版】\n"
        f"- {focus}允许使用的范围是{evidence_scope}。\n"
        "- 结构/架构/机制题先回答证据已给出的架构级拓扑或流程，再列缺失的逐层参数；"
        "不得把“层数、通道、投影或超参未给出”说成“没有结构”。\n"
        "- 复合问题逐项覆盖；任何“未给出”都要明确限定到具体字段。\n"
        "- Do not turn a missing local retrieval hit into a whole-document absence claim; use a current-evidence scope unless an explicit cited statement supports the absence.\n"
        f"- 证据不足时只输出：「{CANNOT_ANSWER_PHRASE_ZH}」+ 一句原因"
        f"（或英文 “{CANNOT_ANSWER_PHRASE_EN}”）。\n"
        "- 数值、方法名、数据集、指标名必须照抄证据，不得估算或改写。\n"
        f"{citation_rule}"
        "- 不要把多个独立事实塞进同一句再共用一个引用。\n"
        "- 不要使用预训练知识填补缺口。"
    )


def strip_citation_protocol_block(answer: str) -> str:
    """剥离结构化引文协议段，只保留回答正文。

    调用方本应传入展示态答案，此处作为防御：万一原始输出漏进来，也不会把
    CITATION LIST 里的协议行统计成「缺引用的事实句」。复用
    citation_service.extract_final_answer，它同时兼容 FINAL ANSWER 在前和
    CITATION LIST 在前两种顺序。
    """
    text = str(answer or "")
    if not text.strip():
        return ""
    try:
        from services.citation_service import extract_final_answer

        text = extract_final_answer(text) or text
    except Exception:
        pass
    return text.strip()


def _is_full_document_summary_mode(
    retrieval_meta: Optional[dict] = None,
    answer_mode: str = "",
) -> bool:
    if str(answer_mode or "").strip().lower() == "full_document_summary":
        return True
    meta = retrieval_meta if isinstance(retrieval_meta, dict) else {}
    summary_meta = meta.get("full_document_summary")
    return isinstance(summary_meta, dict) and bool(summary_meta)


def _full_document_summary_coverage(retrieval_meta: Optional[dict]) -> dict[str, Any]:
    meta = retrieval_meta if isinstance(retrieval_meta, dict) else {}
    value = meta.get("full_document_summary")
    return dict(value) if isinstance(value, dict) else {}


def analyze_citation_coverage(
    answer: str,
    *,
    answer_mode: str = "",
) -> dict[str, Any]:
    """Deterministic check: factual sentences should carry [n]."""
    text = strip_citation_protocol_block(answer)
    if not text:
        return {
            "sentence_count": 0,
            "factual_sentence_count": 0,
            "cited_factual_count": 0,
            "uncited_factual_count": 0,
            "uncited_samples": [],
            "citation_ids": [],
            "coverage": 1.0,
            "answer_mode": answer_mode or "qa",
        }

    if is_cannot_answer(text):
        return {
            "sentence_count": 1,
            "factual_sentence_count": 0,
            "cited_factual_count": 0,
            "uncited_factual_count": 0,
            "uncited_samples": [],
            "citation_ids": [],
            "coverage": 1.0,
            "refused": True,
            "answer_mode": answer_mode or "qa",
        }

    parts = [p.strip() for p in split_sentences(text) if p.strip()]
    if len(parts) <= 1 and len(text) > 80:
        # Fallback split on Chinese commas for run-on answers.
        parts = [p.strip() for p in re.split(r"[。；;]\s*", text) if p.strip()]

    citation_ids = sorted({
        int(match.group(1) or match.group(2))
        for match in _CITATION_RE.finditer(text)
    })
    factual = 0
    cited = 0
    uncited_samples: list[str] = []
    for sentence in parts:
        cleaned = sentence.strip()
        if len(cleaned) < 8:
            continue
        if _CITATION_PROTOCOL_LINE_RE.match(cleaned):
            continue
        if _TRANSITION_RE.match(cleaned):
            continue
        # 先抹掉章节/图表编号再判事实信号：编号里的数字不是事实断言。
        # 「表 3 显示准确率达到 95.2%」抹掉「表 3」后仍有 准确率/95.2，照样算事实句。
        if not _FACTUAL_CLAIM_RE.search(_STRUCTURAL_REF_RE.sub(" ", cleaned)):
            continue
        factual += 1
        if _CITATION_RE.search(cleaned):
            cited += 1
        elif len(uncited_samples) < 3:
            uncited_samples.append(cleaned[:160])

    uncited = max(0, factual - cited)
    coverage = 1.0 if factual == 0 else round(cited / factual, 4)
    return {
        "sentence_count": len(parts),
        "factual_sentence_count": factual,
        "cited_factual_count": cited,
        "uncited_factual_count": uncited,
        "uncited_samples": uncited_samples,
        "citation_ids": citation_ids,
        "coverage": coverage,
        "answer_mode": answer_mode or "qa",
    }


def derive_answer_certainty(
    *,
    answer: str = "",
    retrieval_meta: Optional[dict] = None,
    critic: Optional[dict] = None,
    citation_coverage: Optional[dict] = None,
    answer_mode: str = "",
) -> dict[str, Any]:
    """Derive Certain / Partial / Unsure / Refused for academic reading UX."""
    meta = retrieval_meta if isinstance(retrieval_meta, dict) else {}
    full_document_summary = _is_full_document_summary_mode(meta, answer_mode)
    coverage = citation_coverage or analyze_citation_coverage(
        answer,
        answer_mode="full_document_summary" if full_document_summary else answer_mode,
    )
    critic = critic if isinstance(critic, dict) else {}

    if is_cannot_answer(answer):
        return {
            "label": "Refused",
            "score": 0.2,
            "reasons": ["cannot_answer_sentinel"],
        }

    reasons: list[str] = []
    score = 0.72  # neutral prior for grounded academic answer

    summary_coverage: dict[str, Any] = {}
    if full_document_summary:
        summary_coverage = _full_document_summary_coverage(meta)
        body_expected = max(0, int(summary_coverage.get("body_expected") or 0))
        body_summarized = max(0, int(summary_coverage.get("body_summarized") or 0))
        appendix_expected = max(0, int(summary_coverage.get("appendix_expected") or 0))
        appendix_summarized = max(0, int(summary_coverage.get("appendix_summarized") or 0))
        body_complete = body_expected == 0 or body_summarized >= body_expected
        appendix_complete = appendix_expected == 0 or appendix_summarized >= appendix_expected
        source = str(summary_coverage.get("source") or "").strip().lower()
        if body_complete and appendix_complete:
            score += 0.12
            reasons.append("full_document_section_coverage_complete")
        elif body_expected and body_summarized:
            score -= 0.10
            reasons.append("full_document_section_coverage_partial")
        else:
            score -= 0.20
            reasons.append("full_document_section_coverage_missing")
        if source and not source.startswith("ai"):
            score -= 0.08
            reasons.append("full_document_outline_fallback")

    # Evidence scoring from agent path
    signals = extract_evidence_signals(meta)
    scoring = signals["evidence_scoring"]
    if isinstance(scoring, dict) and scoring.get("applied"):
        high = int(scoring.get("high_score_count") or 0)
        mid = int(scoring.get("mid_score_count") or 0)
        dropped = int(scoring.get("dropped_count") or 0)
        if high >= 2:
            score += 0.12
            reasons.append("high_score_evidence>=2")
        elif high == 1:
            score += 0.05
            reasons.append("high_score_evidence=1")
        if dropped > high + mid:
            score -= 0.08
            reasons.append("many_low_score_dropped")

    evidence_state = signals["evidence_state"]
    if isinstance(evidence_state, dict):
        status = str(evidence_state.get("status") or "")
        if status == "answered":
            score += 0.08
            reasons.append("evidence_state_answered")
        elif status == "insufficient_evidence":
            score -= 0.2
            reasons.append("evidence_state_insufficient")
        elif status == "budget_exhausted":
            score -= 0.12
            reasons.append("evidence_state_budget_exhausted")

    sufficiency = signals["sufficiency"]
    if isinstance(sufficiency, dict):
        level = str(sufficiency.get("level") or "")
        if level == "sufficient":
            score += 0.06
            reasons.append("sufficiency_sufficient")
        elif level == "insufficient":
            score -= 0.15
            reasons.append("sufficiency_insufficient")

    max_rel = signals["max_relevance_score"]
    try:
        if max_rel is not None and float(max_rel) < 0.3:
            score -= 0.15
            reasons.append("low_retrieval_relevance")
    except (TypeError, ValueError):
        pass

    cov = _coverage_ratio(coverage)
    uncited = int(coverage.get("uncited_factual_count") or 0)
    if uncited >= 2 or cov < 0.5:
        score -= 0.18
        reasons.append("weak_citation_coverage")
    elif uncited == 1 or cov < 0.75:
        score -= 0.08
        reasons.append("partial_citation_coverage")
    elif int(coverage.get("factual_sentence_count") or 0) > 0 and cov >= 0.9:
        score += 0.06
        reasons.append("strong_citation_coverage")

    if critic.get("has_hallucination"):
        score -= 0.25
        reasons.append("critic_hallucination")
    try:
        critic_score = critic.get("score")
        if critic_score is not None:
            c = int(critic_score)
            if c <= 4:
                score -= 0.15
                reasons.append("critic_low_score")
            elif c >= 8:
                score += 0.08
                reasons.append("critic_high_score")
    except (TypeError, ValueError):
        pass

    if signals["degraded"]:
        score -= 0.1
        reasons.append("retrieval_degraded_or_fallback")

    score = max(0.0, min(1.0, round(score, 4)))
    summary_ready_for_certain = True
    if full_document_summary:
        summary_source = str(summary_coverage.get("source") or "").strip().lower()
        summary_status = str(summary_coverage.get("generation_status") or "").strip().lower()
        summary_ready_for_certain = bool(
            summary_coverage.get("complete")
            and summary_source.startswith("ai")
            and summary_status not in {"partial", "failed", "unavailable"}
        )
        if not summary_ready_for_certain:
            reasons.append("full_document_summary_not_ready_for_certain")

    if (
        score >= 0.75
        and uncited == 0
        and not critic.get("has_hallucination")
        and summary_ready_for_certain
    ):
        label: Certainty = "Certain"
    elif score >= 0.55:
        label = "Partial"
    else:
        label = "Unsure"

    result = {
        "label": label,
        "score": score,
        "reasons": reasons[:8],
        "citation_coverage": coverage.get("coverage"),
        "uncited_factual_count": uncited,
    }
    if full_document_summary:
        result["full_document_summary"] = {
            key: summary_coverage.get(key)
            for key in (
                "mode",
                "source",
                "generation_status",
                "body_expected",
                "body_summarized",
                "appendix_expected",
                "appendix_summarized",
                "body_complete",
                "appendix_complete",
                "complete",
                "rendered_section_count",
                "citation_count",
                "retryable",
            )
            if key in summary_coverage
        }
    return result


def postprocess_critic_result(
    critique: Optional[dict],
    *,
    answer: str,
    retrieval_meta: Optional[dict] = None,
    answer_mode: str = "",
) -> dict[str, Any]:
    """Merge LLM critic with deterministic academic checks and certainty."""
    base = dict(critique or {}) if isinstance(critique, dict) else {}
    # critic 超时/解析失败时返回 None，此时结论完全来自本地规则。前端需要能区分
    # 两者，否则纯规则分数会被当成模型给出的置信度。
    critic_source = "llm" if isinstance(critique, dict) else "rules_only"
    full_document_summary = _is_full_document_summary_mode(retrieval_meta, answer_mode)
    coverage = analyze_citation_coverage(
        answer,
        answer_mode="full_document_summary" if full_document_summary else answer_mode,
    )

    # issue_details 是结构化形态；critic 返回纯文本时（或历史载荷）向上兼容。
    raw_details = base.get("issue_details")
    if isinstance(raw_details, list) and raw_details:
        issue_details = [dict(item) for item in raw_details if isinstance(item, dict)]
    else:
        raw_issues = base.get("issues") if isinstance(base.get("issues"), list) else []
        issue_details = [
            {"text": str(item).strip(), "issue_type": "other", "claim_span": "", "evidence_refs": []}
            for item in raw_issues
            if str(item).strip()
        ]
    issue_details = issue_details[:5]
    unscoped_absence_claims = find_unscoped_document_absence_claims(answer)
    if unscoped_absence_claims:
        claim_span = unscoped_absence_claims[0][:160]
        msg = (
            "\u7b54\u6848\u5c06\u5c40\u90e8\u68c0\u7d22\u4e0d\u8db3\u5916\u63a8\u4e3a\u6574\u7bc7\u6587\u6863\u7f3a\u5931\u7ed3\u8bba\uff1b"
            "\u5e94\u6539\u4e3a\u201c\u5f53\u524d\u8bc1\u636e\u672a\u8986\u76d6\u8be5\u5b57\u6bb5\u201d\uff0c\u6216\u63d0\u4f9b\u539f\u6587\u660e\u786e\u58f0\u660e\u7684\u5f15\u7528"
        )
        if all(item.get("claim_span") != claim_span for item in issue_details):
            issue_details.append({
                "text": msg,
                "issue_type": "overreach",
                "claim_span": claim_span,
                "evidence_refs": [],
            })

    uncited = int(coverage.get("uncited_factual_count") or 0)
    if uncited > 0 and not coverage.get("refused"):
        sample = coverage.get("uncited_samples") or []
        tip = sample[0] if sample else ""
        msg = (
            f"有 {uncited} 处章节结论未绑定到阅读证据"
            if full_document_summary
            else f"有 {uncited} 处事实陈述缺少 [n] 引用"
        )
        if tip:
            msg += f"：「{tip[:60]}」"
        if all(item.get("text") != msg for item in issue_details):
            issue_details.append({
                "text": msg,
                "issue_type": "missing_citation",
                # 确定性检查知道确切是哪一句，直接作为前端定位锚点。
                "claim_span": str(tip)[:160],
                "evidence_refs": [],
            })

    if is_cannot_answer(answer) and not unscoped_absence_claims:
        # Refused answers are not hallucinations.
        base["has_hallucination"] = False
        if not issue_details:
            issue_details = []

    issues = [str(item.get("text") or "").strip() for item in issue_details]
    issues = [text for text in issues if text]

    # has_hallucination 只表示「LLM 判定答案含无据内容」；引用覆盖不足是另一回事，
    # 单独用 citation_risk 表达。此前把 uncited >= 3 直接升级为幻觉，长答案（事实句
    # 天然更多）会被系统性误报为幻觉。
    has_hallucination = bool(base.get("has_hallucination", False) or unscoped_absence_claims)

    # 引用覆盖风险按比例判定，并要求最小样本量，避免一两句话的短答案被过度惩罚。
    refused = bool(coverage.get("refused"))
    factual = int(coverage.get("factual_sentence_count") or 0)
    cov_ratio = _coverage_ratio(coverage)
    citation_risk_level = "none"
    if not refused and uncited > 0:
        if cov_ratio <= 0.0:
            # 完全没有引用：与样本量无关，直接判高风险。
            citation_risk_level = "high"
        elif factual >= 2:
            if cov_ratio < 0.5:
                citation_risk_level = "high"
            elif cov_ratio < 0.8:
                citation_risk_level = "medium"

    try:
        score = int(base.get("score", 5))
    except (TypeError, ValueError):
        score = 5
    if citation_risk_level == "medium":
        score = min(score, 6)
    elif citation_risk_level == "high":
        score = min(score, 5)
    score = max(0, min(10, score))
    if unscoped_absence_claims:
        score = min(score, 4)

    suggestion = str(base.get("suggestion") or "").strip()
    if unscoped_absence_claims and not suggestion:
        suggestion = "\u8bf7\u5c06\u6574\u7bc7\u6587\u6863\u7684\u7f3a\u5931\u65ad\u8a00\u6539\u4e3a\u5f53\u524d\u8bc1\u636e\u8303\u56f4\uff0c\u6216\u8865\u5145\u660e\u786e\u7684\u539f\u6587\u5f15\u7528\u3002"
    elif uncited > 0 and not suggestion:
        suggestion = (
            "请为章节结论补齐阅读大纲中的证据块绑定。"
            if full_document_summary
            else "请为关键数值与结论句补充 [n] 引用，或改为拒答。"
        )

    certainty = derive_answer_certainty(
        answer=answer,
        retrieval_meta=retrieval_meta,
        critic={
            "has_hallucination": has_hallucination,
            "score": score,
        },
        citation_coverage=coverage,
        answer_mode="full_document_summary" if full_document_summary else answer_mode,
    )

    # reason 只承载「建议」，不镜像 issues[0]。此前 has_hallucination 时把 issues[0]
    # 复制进 reason，前端又分别渲染 reason 与 issues[0]，同一句话会显示两遍。
    reason = suggestion

    return {
        "score": score,
        "has_hallucination": has_hallucination,
        "citation_risk": citation_risk_level != "none",
        "citation_risk_level": citation_risk_level,
        "issues": issues[:5],
        "issue_details": issue_details[:5],
        "suggestion": suggestion[:200],
        "reason": reason[:200],
        "confidence": round(score / 10.0, 3),
        "critic_source": critic_source,
        "missing_citations": bool(base.get("missing_citations", False)),
        "citation_coverage": coverage,
        "certainty": certainty,
        "academic_contract": True,
        "answer_mode": "full_document_summary" if full_document_summary else (answer_mode or "qa"),
    }


def build_answer_certainty_event(certainty: dict[str, Any]) -> dict[str, Any]:
    """SSE/API payload for frontend badge."""
    label = str((certainty or {}).get("label") or "Unsure")
    payload = {
        "type": "answer_certainty",
        "certainty": {
            "label": label,
            "score": float((certainty or {}).get("score") or 0.0),
            "reasons": list((certainty or {}).get("reasons") or [])[:6],
            "citation_coverage": (certainty or {}).get("citation_coverage"),
            "uncited_factual_count": (certainty or {}).get("uncited_factual_count"),
        },
    }
    full_document_summary = (certainty or {}).get("full_document_summary")
    if isinstance(full_document_summary, dict):
        payload["certainty"]["full_document_summary"] = dict(full_document_summary)
    return payload
