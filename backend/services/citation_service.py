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

def build_structured_citation_prompt(citations: list[dict]) -> str:
    """构建要求 LLM 输出 CITATION LIST + FINAL ANSWER 的结构化提示词

    Args:
        citations: context_builder 生成的引文列表
            [{"ref": 1, "group_id": "...", "page_range": [1,2], ...}, ...]

    Returns:
        插入 system_prompt 的引文指示字符串
    """
    if not citations:
        return ""

    ref_descriptions = []
    for c in citations:
        ref_descriptions.append(
            f"[{c['ref']}] 来源: {c['group_id']}，页码: {c['page_range'][0]}-{c['page_range'][1]}"
        )
    refs_text = "\n".join(ref_descriptions)

    prompt = (
        "请使用以下格式回答问题：\n"
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
        "FINAL ANSWER\n"
        "在此输出你的完整回答，引用处使用【编号】标注来源。\n"
        "\n"
        "示例：\n"
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
        "FINAL ANSWER\n"
        "固定大小分块是一种传统方法，它按预设大小切分文档而不考虑语义内容，计算效率较高【1】。"
        "然而，这种方法可能导致语义相关内容被割裂，从而影响检索性能【1】【2】。\n"
        "\n"
        f"可用的引用来源：\n{refs_text}\n"
        "\n"
        "注意：\n"
        "- 只能使用上述列出的编号，禁止创造新编号\n"
        "- START_PHRASE 和 END_PHRASE 必须从上下文中原文复制\n"
        "- FINAL ANSWER 中每个事实性陈述都应标注来源编号\n"
        "- 如果信息来自通用知识而非上下文，则无需标注\n"
    )

    logger.info(f"结构化引文提示词生成完成: {len(citations)} 个引用来源")
    return prompt


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


def extract_final_answer(full_output: str) -> str:
    """从 LLM 完整输出中提取 FINAL ANSWER 部分

    如果输出不含 FINAL ANSWER 标记，返回原始全文（兼容非结构化输出）。

    Args:
        full_output: LLM 的完整输出文本

    Returns:
        FINAL ANSWER 之后的回答文本
    """
    if START_ANSWER in full_output:
        parts = full_output.split(START_ANSWER, 1)
        answer = parts[1].lstrip()
        # 如果 CITATION LIST 出现在 FINAL ANSWER 之后（小模型重复输出），截断
        if START_CITATION in answer:
            answer = answer.split(START_CITATION, 1)[0].rstrip()
        return answer

    # 无结构化标记，返回原文
    return full_output


# ═══════════════════════════════════════════════════
# 3. 源文模糊匹配（SequenceMatcher）
# ═══════════════════════════════════════════════════

def find_start_end_phrase(
    start_phrase: str,
    end_phrase: str,
    context: str,
    min_length: int = 5,
    max_excerpt_length: int = 300,
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
    start_phrase = start_phrase.lower() if start_phrase else ""
    end_phrase = end_phrase.lower() if end_phrase else ""
    ctx_lower = context.lower().replace("\n", " ")

    matches = []
    matched_length = 0

    for sentence in [start_phrase, end_phrase]:
        if not sentence:
            continue
        match = SequenceMatcher(
            None, sentence, ctx_lower, autojunk=False
        ).find_longest_match()
        if match.size > max(len(sentence) * 0.35, min_length):
            matches.append((match.b, match.b + match.size))
            matched_length += match.size

    # 如果第二个匹配在第一个之前，只保留第一个
    if len(matches) == 2 and matches[1][0] < matches[0][0]:
        matches = [matches[0]]

    if matches:
        start_idx = min(s for s, _ in matches)
        end_idx = max(e for _, e in matches)

        if end_idx - start_idx > max_excerpt_length:
            end_idx = start_idx + max_excerpt_length

        return (start_idx, end_idx), matched_length

    return None, 0


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
            [{"ref": int, "text": str}, ...]
            每项的 text 是 LLM 实际看到的完整意群文本

    Returns:
        增强后的 citation 列表，每项新增:
        - highlight_text: 精确匹配到的原文片段
        - matched_chunk_idx: 匹配到的 chunk 索引
    """
    # 构建 ref → context_segment 映射
    segment_map = {}
    if context_segments:
        for seg in context_segments:
            if seg.get("ref") is not None and seg.get("text"):
                segment_map[seg["ref"]] = seg["text"]

    enhanced = []

    for evidence in citations:
        if not evidence.start_phrase and not evidence.end_phrase:
            continue

        best_match = None
        best_length = 0
        best_text = ""
        best_chunk_idx = None

        # 策略 1：优先在对应 ref 的意群文本中精准匹配
        if evidence.idx and evidence.idx in segment_map:
            segment_text = segment_map[evidence.idx]
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

        # 策略 2：意群匹配失败时，遍历所有意群文本搜索
        if not best_match and segment_map:
            for seg_ref, seg_text in segment_map.items():
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

        # 策略 3：回退到原始 chunk 级别匹配
        if not best_match:
            for ci, chunk in enumerate(chunks):
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

        entry = {
            "idx": evidence.idx,
            "start_phrase": evidence.start_phrase,
            "end_phrase": evidence.end_phrase,
            "highlight_text": best_text if best_match else None,
            "matched_chunk_idx": best_chunk_idx,
        }

        # 从匹配的 chunk 中继承页码等信息
        if best_chunk_idx is not None and best_chunk_idx < len(chunks):
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
