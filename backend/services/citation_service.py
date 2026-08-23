"""结构化引文提取与模糊匹配服务

参考 kotaemon 的 AnswerWithInlineCitation 设计：
1. 构建结构化 prompt 要求 LLM 输出 CITATION LIST + FINAL ANSWER
2. 解析 LLM 输出中的 CITATION LIST 提取 start_phrase / end_phrase
3. 用 SequenceMatcher 将 phrase 模糊匹配回原始 chunk 文本
4. 生成精确的高亮文本范围
"""

import re
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

# ── 常量 ──
START_ANSWER = "FINAL ANSWER"
START_CITATION = "CITATION LIST"
_RE_START_ANSWER = re.compile(r"FINAL\s*ANSWER", re.IGNORECASE)
_RE_START_CITATION = re.compile(r"CITATION\s*LIST", re.IGNORECASE)
# 只认独立成行（或行首加冒号）的协议标记。行内提到
# 「按照要求的格式：FINAL ANSWER 然后 CITATION LIST」不能切开正文。
_RE_START_ANSWER_SECTION = re.compile(
    r"(?:^|\n)[ \t]*FINAL[ \t]*ANSWER(?:[ \t]*:[ \t]*(?=\S)|[ \t]*:?[ \t]*(?=\n|$))",
    re.IGNORECASE,
)
_RE_START_CITATION_SECTION = re.compile(
    r"(?:^|\n)[ \t]*CITATION[ \t]*LIST(?:[ \t]*:[ \t]*(?=\S)|[ \t]*:?[ \t]*(?=\n|$))",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"citation[【\[](\d+)[】\]]", re.IGNORECASE)
START_PHRASE_PREFIX = "start_phrase:"
END_PHRASE_PREFIX = "end_phrase:"


@dataclass
class InlineEvidence:
    """一条引文的 start/end phrase 及其引文索引"""
    idx: Optional[int] = None
    start_phrase: Optional[str] = None
    end_phrase: Optional[str] = None


# ═══════════════════════════════════════════════════
# 1. 结构化引文 Prompt 构建
# ═══════════════════════════════════════════════════

def build_structured_citation_prompt(
    citations: list[dict],
    compact: bool = False,
    *,
    include_source_details: bool = True,
) -> str:
    """构建要求 LLM 输出 CITATION LIST + FINAL ANSWER 的结构化提示词

    Args:
        citations: context_builder 生成的引文列表
            [{"ref": 1, "group_id": "...", "page_range": [1,2], ...}, ...]
        include_source_details: 是否在提示中携带动态来源标识。最终回答路径
            只需要安全的编号列表，证据内容由单独消息提供。

    Returns:
        插入 system_prompt 的引文指示字符串
    """
    if not citations:
        return ""

    ref_descriptions = []
    for index, c in enumerate(citations, start=1):
        try:
            ref = int(c.get("ref", index))
        except (TypeError, ValueError):
            ref = index
        if ref <= 0:
            ref = index
        if include_source_details:
            group_id = c.get("group_id") or c.get("source") or c.get("context_id") or "unknown"
            page_text = _format_citation_page_range(c.get("page_range") or c.get("page"))
            ref_descriptions.append(
                f"[{ref}] 来源: {group_id}，页码: {page_text}"
            )
        else:
            ref_descriptions.append(f"[{ref}]")
    refs_text = "\n".join(ref_descriptions)

    if compact:
        prompt = (
            "请使用以下格式回答问题：\n"
            "\n"
            "FINAL ANSWER\n"
            "先在此输出完整回答，引用处使用[编号]标注来源。\n"
            "\n"
            "CITATION LIST\n"
            "CITATION【编号】\n"
            "START_PHRASE: 从证据中原文复制的起始短语\n"
            "END_PHRASE: 从证据中原文复制的结束短语\n"
            "\n"
            f"可用的引用来源：\n{refs_text}\n"
            "\n"
            "注意：\n"
            "- 必须先输出 FINAL ANSWER，再输出 CITATION LIST\n"
            "- reasoning 结束后立即开始 FINAL ANSWER；禁止先生成、草拟或隐藏 CITATION LIST\n"
            "- 只能使用上述列出的编号，禁止创造新编号\n"
            "- START_PHRASE 和 END_PHRASE 必须从上下文中原文复制\n"
            "- 引用编号必须直接支撑它所在句子的具体事实，不能只因为同页、同章节或同主题就引用\n"
            "- 若一个句子包含多个事实且来自不同证据，请拆成多个短句分别标注\n"
            "- FINAL ANSWER 中每个关键事实句都应标注来源编号\n"
            "- 每个句子的引用数量最多 2 个，格式必须为 [编号]\n"
            "- 不同事实若来自不同证据窗口，优先使用不同编号，不要把多个独立事实都标成同一个编号\n"
            "- 如果某句无法在上下文中找到证据，请删除该句的引用而不是强行引用\n"
        )
        logger.info(f"紧凑结构化引文提示词生成完成: {len(citations)} 个引用来源")
        return prompt

    prompt = (
        "请使用以下格式回答问题：\n"
        "\n"
        "FINAL ANSWER\n"
        "在此输出你的完整回答，引用处使用[编号]标注来源。\n"
        "\n"
        "CITATION LIST\n"
        "\n"
        "// 对于你引用的每一处内容，输出一个 CITATION 块\n"
        "CITATION【编号】\n"
        "\n"
        "// 从上下文中精确复制约 6 个词标记引用段落的起止位置\n"
        "// 必须原文复制，不要改写或释义\n"
        "START_PHRASE: 引用段落开头的约6个词\n"
        "END_PHRASE: 引用段落结尾的约6个词\n"
        "\n"
        "示例：\n"
        "FINAL ANSWER\n"
        "固定大小分块是一种传统方法，它按预设大小切分文档而不考虑语义内容，计算效率较高[1]。"
        "然而，这种方法可能导致语义相关内容被割裂，从而影响检索性能[1][2]。\n"
        "\n"
        "CITATION LIST\n"
        "\n"
        "CITATION【1】\n"
        "\n"
        "START_PHRASE: 固定大小分块是一种传统的\n"
        "END_PHRASE: 不会降低最终的检索性能。\n"
        "\n"
        "CITATION【2】\n"
        "\n"
        "START_PHRASE: 固定大小分块器是我们的基准\n"
        "END_PHRASE: 这表明检索质量良好。\n"
        "\n"
        f"可用的引用来源：\n{refs_text}\n"
        "\n"
        "注意：\n"
        "- 必须先输出 FINAL ANSWER，再输出 CITATION LIST\n"
        "- reasoning 结束后立即开始 FINAL ANSWER；禁止先生成、草拟或隐藏 CITATION LIST\n"
        "- 只能使用上述列出的编号，禁止创造新编号\n"
        "- START_PHRASE 和 END_PHRASE 必须从上下文中原文复制\n"
        "- 引用编号必须直接支撑它所在句子的具体事实，不能只因为同页、同章节或同主题就引用\n"
        "- 若一个句子包含多个事实且来自不同证据，请拆成多个短句分别标注\n"
        "- CITATION 块的 START_PHRASE/END_PHRASE 必须包围该句所依赖的真实证据，不要引用无关窗口\n"
        "- FINAL ANSWER 中每个事实性陈述都应标注来源编号\n"
        "- 每个句子的引用数量最多 2 个，格式必须为 [编号]（不要使用【编号】）\n"
        "- 不同事实若来自不同证据窗口，优先使用不同编号，不要把多个独立事实都标成同一个编号\n"
        "- 如果某句无法在上下文中找到证据，请删除该句的引用而不是强行引用\n"
        "- 不允许整段无引用；至少在每个关键结论句后附引用\n"
        "- 如果信息来自通用知识而非上下文，则无需标注\n"
    )

    logger.info(f"结构化引文提示词生成完成: {len(citations)} 个引用来源")
    return prompt


def _format_citation_page_range(value) -> str:
    """格式化 citation page_range，兼容空列表、单页和跨页范围。"""
    if isinstance(value, (list, tuple)):
        pages: list[int] = []
        for item in value:
            try:
                page = int(item)
            except (TypeError, ValueError):
                continue
            if page > 0:
                pages.append(page)
        if not pages:
            return "未知"
        start, end = pages[0], pages[-1]
        return str(start) if start == end else f"{start}-{end}"
    try:
        page = int(value)
    except (TypeError, ValueError):
        return "未知"
    return str(page) if page > 0 else "未知"


# ═══════════════════════════════════════════════════
# 2. CITATION LIST 解析器
# ═══════════════════════════════════════════════════

def parse_citation_list(full_output: str) -> list[InlineEvidence]:
    """从 LLM 完整输出中解析 CITATION LIST

    解析 CITATION【n】 块中的 START_PHRASE / END_PHRASE。

    Args:
        full_output: LLM 的完整输出文本（含 CITATION LIST + FINAL ANSWER）

    Returns:
        InlineEvidence 列表
    """
    citations: list[InlineEvidence] = []
    lines = full_output.split("\n")
    current: Optional[InlineEvidence] = None

    for line in lines:
        stripped = line.strip()

        # 检测 CITATION【n】 开始
        match = CITATION_PATTERN.match(stripped)
        if match:
            # 保存上一个完成的 evidence
            if current:
                citations.append(current)
            try:
                idx = int(match.group(1))
            except ValueError:
                idx = None
            current = InlineEvidence(idx=idx)
            continue

        # 检测 START_PHRASE / END_PHRASE
        lower = stripped.lower()
        if lower.startswith(START_PHRASE_PREFIX):
            phrase = stripped[len(START_PHRASE_PREFIX):].strip()
            if not current:
                current = InlineEvidence()
            current.start_phrase = phrase
        elif lower.startswith(END_PHRASE_PREFIX):
            phrase = stripped[len(END_PHRASE_PREFIX):].strip()
            if not current:
                current = InlineEvidence()
            current.end_phrase = phrase

        # 如果 start 和 end 都有了，该 citation 完整
        if current and current.start_phrase and current.end_phrase:
            citations.append(current)
            current = None

    # 收尾
    if current:
        citations.append(current)

    logger.info(f"解析到 {len(citations)} 条引文")
    return citations


def _ci_split(pattern: re.Pattern, text: str, maxsplit: int = 1):
    """大小写不敏感的 split，返回 (before, after) 或 None。"""
    m = pattern.search(text)
    if not m:
        return None
    return text[:m.start()], text[m.end():]


def _ci_contains(pattern: re.Pattern, text: str) -> bool:
    return pattern.search(text) is not None


def extract_final_answer(full_output: str, *, allow_unmarked: bool = True) -> str:
    """从 LLM 完整输出中提取 FINAL ANSWER 部分

    支持两种顺序：
    - 新格式：FINAL ANSWER → 回答 → CITATION LIST → 引文
    - 旧格式：CITATION LIST → 引文 → FINAL ANSWER → 回答

    标记必须独立成行（或行首 `FINAL ANSWER:`）。行内提及不算。

    如果输出不含任何标记，默认返回原始全文（兼容非结构化输出）。
    流式结构化引文请传 `allow_unmarked=False`，避免把思考草稿当成正文。

    Args:
        full_output: LLM 的完整输出文本
        allow_unmarked: 无行首标记时是否退回全文

    Returns:
        纯回答文本（不含 CITATION LIST）
    """
    parts = _ci_split(_RE_START_ANSWER_SECTION, full_output)
    if parts is not None:
        answer = parts[1].lstrip()
        # 截断尾部 CITATION LIST（新格式：FINAL ANSWER 在前）
        cit_parts = _ci_split(_RE_START_CITATION_SECTION, answer)
        if cit_parts is not None:
            answer = cit_parts[0].rstrip()
        return answer

    # 无 FINAL ANSWER 标记，但可能有 CITATION LIST 在尾部
    cit_parts = _ci_split(_RE_START_CITATION_SECTION, full_output)
    if cit_parts is not None:
        return cit_parts[0].rstrip()

    if not allow_unmarked:
        return ""
    return full_output


# ═══════════════════════════════════════════════════
# 3. 源文模糊匹配（SequenceMatcher）
# ═══════════════════════════════════════════════════

def find_start_end_phrase(
    start_phrase: str,
    end_phrase: str,
    context: str,
    min_length: int = 5,
    max_excerpt_length: int = 160,
) -> tuple[Optional[tuple[int, int]], int]:
    """用 SequenceMatcher 将 start/end phrase 模糊匹配回原始 context

    Args:
        start_phrase: LLM 输出的引用起始短语
        end_phrase: LLM 输出的引用结束短语
        context: 原始上下文文本
        min_length: 最小匹配长度
        max_excerpt_length: 最大摘录长度

    Returns:
        ((start_idx, end_idx), matched_length) 或 (None, 0)
    """
    start_phrase = re.sub(r'\s+', ' ', start_phrase.lower()).strip() if start_phrase else ""
    end_phrase = re.sub(r'\s+', ' ', end_phrase.lower()).strip() if end_phrase else ""
    ctx_lower = context.lower().replace("\n", " ")

    def _find_candidates(phrase: str) -> list[tuple[int, int, int, bool]]:
        if not phrase:
            return []
        words = [re.escape(w) for w in phrase.split() if w]
        candidates: list[tuple[int, int, int, bool]] = []
        if words:
            pattern = re.compile(r'\s+'.join(words))
            for m in pattern.finditer(ctx_lower):
                candidates.append((m.start(), m.end(), m.end() - m.start(), True))
                if len(candidates) >= 8:
                    break
        if candidates:
            return candidates
        match = SequenceMatcher(None, phrase, ctx_lower, autojunk=False).find_longest_match()
        if match.size > max(len(phrase) * 0.35, min_length):
            return [(match.b, match.b + match.size, match.size, False)]
        return []

    def _single_anchor_span(candidates: list[tuple[int, int, int, bool]]) -> tuple[Optional[tuple[int, int]], int]:
        if not candidates:
            return None, 0
        start_idx, end_idx, length, _ = max(candidates, key=lambda item: item[2])
        excerpt_start = max(0, start_idx - 20)
        excerpt_end = min(len(ctx_lower), excerpt_start + max_excerpt_length)
        return (excerpt_start, excerpt_end), length

    start_candidates = _find_candidates(start_phrase)
    end_candidates = _find_candidates(end_phrase)

    best_pair = None
    best_score = None
    for s_start, s_end, s_len, s_exact in start_candidates or [(None, None, 0, False)]:
        for e_start, e_end, e_len, e_exact in end_candidates or [(None, None, 0, False)]:
            if s_start is None and e_start is None:
                continue
            if s_start is None:
                pair_start, pair_end = e_start, e_end
            elif e_start is None:
                pair_start, pair_end = s_start, s_end
            else:
                if e_end <= s_start:
                    continue
                pair_start, pair_end = s_start, e_end
            span_len = pair_end - pair_start
            if span_len <= 0:
                continue
            if span_len > max_excerpt_length * 1.35:
                continue
            exact_bonus = (18 if s_exact else 0) + (18 if e_exact else 0)
            score = exact_bonus + s_len + e_len - span_len * 0.08
            if best_pair is None or score > best_score:
                best_pair = (pair_start, pair_end)
                best_score = score

    if best_pair is not None:
        start_idx, end_idx = best_pair
        if end_idx - start_idx > max_excerpt_length:
            end_idx = start_idx + max_excerpt_length
        return (start_idx, end_idx), end_idx - start_idx

    single_span, single_len = _single_anchor_span(start_candidates)
    if single_span is not None:
        return single_span, single_len

    single_span, single_len = _single_anchor_span(end_candidates)
    if single_span is not None:
        return single_span, single_len

    # 字符级 n-gram 回退（对中文短语效果更好）
    combined = start_phrase + " " + end_phrase if start_phrase and end_phrase else (start_phrase or end_phrase or "")
    if len(combined) >= min_length:
        # 滑动窗口匹配：在 context 中找与 combined 最相似的子串
        ngram_len = min(len(combined), 60)
        best_pos, best_ratio = -1, 0.0
        step = max(1, ngram_len // 3)
        for i in range(0, max(1, len(ctx_lower) - ngram_len + 1), step):
            window = ctx_lower[i:i + ngram_len]
            ratio = SequenceMatcher(None, combined[:ngram_len], window, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
        if best_ratio >= 0.55 and best_pos >= 0:
            end_pos = min(len(ctx_lower), best_pos + max_excerpt_length)
            return (best_pos, end_pos), end_pos - best_pos

    return None, 0


def merge_overlapping_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """合并重叠或相邻的高亮区间（仿 kotaemon 实现）

    Args:
        spans: [(start, end), ...] 可能重叠的区间列表

    Returns:
        合并后的不重叠区间列表，按 start 升序排列
    """
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def match_citations_to_chunks(
    citations: list[InlineEvidence],
    chunks: list[dict],
    context_segments: list[dict] = None,
) -> list[dict]:
    """将解析出的引文匹配回原始文本，生成精确高亮文本

    优先在 context_segments（LLM 实际看到的意群文本）中匹配，
    回退到 chunks（原始小 chunk）级别匹配。

    Args:
        citations: parse_citation_list 解析的引文列表
        chunks: 检索到的 chunk 列表，每个至少含 "text" 字段
            可选字段: "page", "group_id", "ref"
        context_segments: 意群级上下文段列表（可选）
            [{"ref": int, "text": str, "page_range": [...], "group_id": str}, ...]
            每项的 text 是 LLM 实际看到的完整意群文本

    Returns:
        增强后的 citation 列表，每项新增:
        - highlight_text: 精确匹配到的原文片段
        - matched_chunk_idx: 匹配到的 chunk 索引（仅策略3时设置）
    """
    # 构建 ref → segment 元数据映射（包含 text、page_range、group_id）
    segment_map: dict[int, dict] = {}
    if context_segments:
        for seg in context_segments:
            try:
                ref = int(seg.get("ref"))
            except (TypeError, ValueError):
                continue
            if seg.get("text"):
                segment_map[ref] = seg

    enhanced = []

    for evidence in citations:
        if not evidence.start_phrase and not evidence.end_phrase:
            continue

        best_match = None
        best_length = 0
        best_text = ""
        best_chunk_idx = None
        best_seg_ref = None

        try:
            target_ref = int(evidence.idx) if evidence.idx is not None else None
        except (TypeError, ValueError):
            target_ref = None

        # 策略 1：只在该 citation 指向的意群中匹配。以前的全局回退会让
        # ``CITATION [1]`` 的短语落到 [2]，随后调用方却保留 [1] 的其它
        # 元数据，形成无法回放的拼接证据。
        if target_ref is not None and target_ref in segment_map:
            seg = segment_map[target_ref]
            segment_text = seg["text"]
            span, length = find_start_end_phrase(
                evidence.start_phrase or "",
                evidence.end_phrase or "",
                segment_text,
            )
            if span is not None:
                raw_text = segment_text.replace("\n", " ")
                best_match = span
                best_length = length
                best_text = raw_text[span[0]:span[1]]
                best_seg_ref = target_ref

        # 没有可用编号的旧格式才允许全局搜索；有编号但匹配失败时宁可
        # 不提供精确高亮，也不能把另一条 evidence 的身份借给它。
        if not best_match and target_ref is None and segment_map:
            for seg_ref, seg in segment_map.items():
                seg_text = seg["text"]
                span, length = find_start_end_phrase(
                    evidence.start_phrase or "",
                    evidence.end_phrase or "",
                    seg_text,
                )
                if span is not None and length > best_length:
                    best_match = span
                    best_length = length
                    raw_text = seg_text.replace("\n", " ")
                    best_text = raw_text[span[0]:span[1]]
                    best_seg_ref = seg_ref

        # 策略 3：回退到原始 chunk 级别匹配。带编号的 citation 仍只能
        # 匹配同一 ref 的 chunk，避免跨块拼接页码/bbox/group identity。
        if not best_match:
            for ci, chunk in enumerate(chunks):
                if target_ref is not None:
                    try:
                        chunk_ref = int(chunk.get("ref"))
                    except (TypeError, ValueError):
                        continue
                    if chunk_ref != target_ref:
                        continue
                text = chunk.get("text", "") or chunk.get("chunk", "")
                if not text:
                    continue

                span, length = find_start_end_phrase(
                    evidence.start_phrase or "",
                    evidence.end_phrase or "",
                    text,
                )
                if span is not None and length > best_length:
                    best_match = span
                    best_length = length
                    best_chunk_idx = ci
                    raw_text = text.replace("\n", " ")
                    best_text = raw_text[span[0]:span[1]]

        # 计算匹配置信度（0-1）
        match_confidence = 0.0
        if best_match and best_length > 0:
            phrase_len = len((evidence.start_phrase or "") + (evidence.end_phrase or ""))
            if phrase_len > 0:
                match_confidence = min(1.0, best_length / max(phrase_len, 1))
            else:
                match_confidence = 0.5

        entry = {
            "idx": evidence.idx,
            "start_phrase": evidence.start_phrase,
            "end_phrase": evidence.end_phrase,
            "highlight_text": best_text if best_match else None,
            "matched_chunk_idx": best_chunk_idx,
            "match_confidence": round(match_confidence, 3),
            "matched_ref": best_seg_ref if best_seg_ref is not None else target_ref,
        }

        # B4 修复：策略 1/2 成功时从 segment 继承 page_range / group_id
        if best_seg_ref is not None and best_seg_ref in segment_map:
            seg_meta = segment_map[best_seg_ref]
            page_range = seg_meta.get("page_range")
            if page_range and len(page_range) >= 1:
                entry["page"] = page_range[0]
            entry["group_id"] = seg_meta.get("group_id", "")
        elif best_chunk_idx is not None and best_chunk_idx < len(chunks):
            matched = chunks[best_chunk_idx]
            entry["page"] = matched.get("page", 0)
            entry["group_id"] = matched.get("group_id", "")

        enhanced.append(entry)

    segment_matched = sum(1 for e in enhanced if e.get("highlight_text"))
    logger.info(
        f"引文匹配完成: {len(enhanced)}/{len(citations)} 条, "
        f"成功匹配 {segment_matched} 条"
        f"{' (使用意群级匹配)' if segment_map else ''}"
    )
    return enhanced
