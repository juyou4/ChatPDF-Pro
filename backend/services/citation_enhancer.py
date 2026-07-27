"""
P3.6 引用增强器（two-pass citation injection）
借鉴 ragflow citation_plus.md + agent_with_tools._gen_citations_async

设计：
- First pass: LLM 正常生成答案（流式输出已完成）
- Gate: 检测原答案的引用覆盖率，< 50% 时触发二次注入
- Second pass: 使用 citation_plus 提示词，让 LLM 在答案上添加 [N] 引文
- Safety: only-add 策略，保留原引用，仅追加缺失的；失败时回退到原答案
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ragflow citation_plus prompt 中文化版
_CITATION_PLUS_PROMPT = """你是一名严谨的引用助手，负责为已写好的答案补全引用编号 [N]。

## 任务
- 仔细阅读下方的"参考资料"列表（每条带编号 ID）
- 仔细阅读"原始答案"
- 在原始答案中**逐句追加**引用编号 [N]，让每个事实声明都能追溯到参考资料
- 不要修改原始答案的文字内容（除非删除明显错误的引用）
- 不要新增段落或解释

## 引用规则
- 引用格式仅限 [N]，不允许 [ID:N]、[N,M]、【N】等其他写法
- 单句最多 2 个引用编号；不同事实应使用不同编号
- 仅当原答案包含的内容能被参考资料中的具体句子、表格行或公式**直接支撑**，才追加引用
- 不要因为参考资料同页、同章节、同主题就追加引用；必须能回答"这条证据是否支持该句事实"
- 若一句话包含多个独立事实且来自不同证据，分别在对应短句后标注引用
- 若某句无法在参考资料中找到，**保留原文不动，不要硬编引用**
- 若原答案已含 [N]，保留；可在末尾追加更多引用
- 通用常识或过渡句无需引用

## 参考资料
{references}

## 原始答案
{answer}

## 输出
请直接输出注入引用后的答案文本（不要带任何前缀如"以下是答案"），保持原文结构和内容。"""


# 中文/英文标点都计数。ASCII 句点须不紧跟在数字后且后随空白，否则 95.2 这类
# 小数会被切成两句，使覆盖率被低估、白白触发二次 LLM 注入。
# （_SELECTOR_SENTENCE_SPLIT_RE 早就有这个守卫，这里此前漏了。）
# 仅作为 split_sentences 不可用时的兜底；正常路径见 _split_answer_sentences。
_SENTENCE_END_RE = re.compile(r"(?:[。！？；;]|(?<!\d)[.!?](?=\s|$))\s*")
_SELECTOR_SENTENCE_SPLIT_RE = re.compile(r"([。！？!?；;]\s*|(?<!\d)\.(?=\s|$)\s*)")
# 与 academic_answer_contract._CITATION_RE 和 chat_routes._INLINE_CITATION_PATTERN
# 保持同一识别标准：遗留的全角【n】也是合法引用。此前只认半角，导致用【n】标注
# 的答案被判成「零引用」，既触发无谓的二次注入，又会在已有【n】的句子上再补一个
# [n]。用字符类而非两个分组，保证本文件里的 findall/group(1) 调用点不受影响。
_CITATION_RE = re.compile(r"(?<!!)[\[【](\d+)[\]】](?!\()")
_BILINGUAL_TERM_ALIASES = {
    "框架": {"framework"},
    "统一": {"unified", "universal"},
    "对抗": {"adversarial"},
    "补丁": {"patch"},
    "攻击": {"attack"},
    "自动驾驶": {"autonomous", "driving"},
    "目标检测": {"object", "detection"},
    "检测器": {"detector", "detection"},
    "视觉": {"visual", "vision"},
    "实例": {"instance", "object"},
    "场景": {"scene"},
    "隐蔽": {"stealthy", "invisible"},
    "伪造": {"fake", "fabrication"},
    "物理": {"physical"},
    "通用": {"universal"},
    "鲁棒": {"robust"},
    "迁移": {"transferable", "transferability"},
    "方法": {"method", "approach"},
    "模型": {"model"},
    "训练": {"training", "train"},
    "样本": {"sample"},
    "生成": {"generation", "generate"},
    "数据集": {"dataset"},
    "准确率": {"accuracy"},
    "实验": {"experiment"},
    "表格": {"table"},
    "公式": {"formula"},
    "扩散": {"diffusion"},
    "分类器": {"classifier"},
}


def _split_answer_sentences(answer: str, *, min_length: int = 5) -> List[str]:
    """与 academic_answer_contract 共用同一套分句。

    本模块此前自己用正则切句，和自审那边的实现结论不一致：引号或括号里合法地
    含有句末标点时（「作者指出「…95.2%。这是最优。」并给出证据 [1]。」），
    这里会切成 3 句、覆盖率算成 0.33，而自审算 1.0。覆盖率是二次注入的触发
    条件，低估就会白白多打一次 LLM。
    """
    text = str(answer or "")
    if not text.strip():
        return []
    try:
        from services.academic_answer_contract import split_sentences

        parts = split_sentences(text)
    except Exception:  # pragma: no cover - 仅依赖异常时兜底
        parts = _SENTENCE_END_RE.split(text)
    return [s.strip() for s in parts if s and len(s.strip()) > min_length]


def estimate_citation_coverage(answer: str) -> Tuple[float, int, int]:
    """估计答案的引用覆盖率（带 [N] 引文的句子比例）

    Returns:
        (coverage, n_cited_sentences, n_total_sentences)
    """
    if not answer or not answer.strip():
        return 0.0, 0, 0

    # 这里刻意不套用 _is_non_factual_sentence：它是为「要不要给这句加引用」
    # 设计的保守谓词（拿不准就跳过），其中「纯文本且 ≤14 字」一条会连
    # 「模型优于 baseline」这类真实结论句一起滤掉。当分母用会把真句子丢光，
    # 极端情况下整段被滤成 0 句、覆盖率反而算成 0.0，触发本不该有的二次调用。
    sentences = _split_answer_sentences(answer)
    if not sentences:
        return 0.0, 0, 0

    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    coverage = cited / len(sentences)
    return coverage, cited, len(sentences)


def _build_evidence_schema(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _collect_text(ch: Dict[str, Any]) -> str:
        fields = (
            "text",
            "context_segment_text",
            "numeric_table_exact_context_row_text",
            "table_row_boundary_text",
            "table_row_raw_text",
            "source_text",
            "display_text",
            "highlight_text",
            "chunk",
            "raw_chunk_text",
        )
        parts: List[str] = []
        seen: set[str] = set()
        for field in fields:
            value = ch.get(field)
            if not isinstance(value, str):
                continue
            normalized = re.sub(r"\s+", " ", value).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(normalized)
        return "\n".join(parts)

    evidence_units: List[Dict[str, Any]] = []
    for idx, ch in enumerate(retrieved_chunks or []):
        if not isinstance(ch, dict):
            continue
        ref = ch.get("ref")
        if ref is None:
            ref = idx + 1
        text = _collect_text(ch)
        if not text:
            continue
        page_range = ch.get("page_range") or [ch.get("page", 0), ch.get("page", 0)]
        evidence_units.append({
            "evidence_id": ch.get("evidence_id") or f"ev-{ref}",
            "ref": ref,
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id", ""),
            "page": page_range[0] if page_range else ch.get("page", 0),
            "page_range": page_range,
            "group_id": ch.get("group_id", ""),
            "modality": ch.get("modality", "text"),
            "text": text,
            "score": ch.get("score", 0.0),
        })
    return evidence_units


def _support_terms(text: str) -> set[str]:
    sample = _CITATION_RE.sub("", text or "").lower()
    sample = re.sub(r"\s+", " ", sample)
    words = set(re.findall(r"[a-z0-9][a-z0-9_\-./%]{2,}", sample))
    chinese = re.findall(r"[\u4e00-\u9fff]", sample)
    grams = {"".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))}
    aliases: set[str] = set()
    for trigger, mapped_terms in _BILINGUAL_TERM_ALIASES.items():
        if trigger in sample:
            aliases.update(mapped_terms)
    return {term for term in (words | grams | aliases) if len(term) >= 2}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text or ""))


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![a-z0-9])\d+(?:[.,]\d+)?%?(?![a-z0-9])", text or ""))


def _is_non_factual_sentence(sentence: str) -> bool:
    text = _CITATION_RE.sub("", sentence or "")
    text = re.sub(r"[*_`#>\-]+", "", text)
    text = re.sub(r"^[-*+\d.、\s]+", "", text).strip()
    normalized = re.sub(r"[：:。！？!?；;，,\s]+$", "", text).strip()
    if not normalized:
        return True
    generic_phrases = (
        "以下是",
        "具体如下",
        "具体而言",
        "正如原文所述",
        "关键依据",
        "关系说明",
        "具体比较",
        "核心步骤和依据",
        "总结如下",
    )
    if any(phrase in normalized for phrase in generic_phrases) and len(normalized) <= 36:
        return True
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9\s/()（）\-]+", normalized) and len(normalized) <= 14:
        return True
    return False


def _normalize_post_sentence_citations(text: str) -> str:
    """把中文句末标点后的引用移回标点前，避免 selector 误判为未引用句。"""
    if not text:
        return text
    return re.sub(
        r"([。！？!?；;])\s*((?:\[\d{1,3}\]\s*)+)",
        lambda m: f"{m.group(2).strip()}{m.group(1)}",
        text,
    )


def _is_negative_absence_claim(sentence: str) -> bool:
    """缺失/未报告声明容易被同主题 evidence 误补引用，selector 只做保守跳过。"""
    text = _CITATION_RE.sub("", sentence or "")
    text = re.sub(r"\s+", "", text)
    if not text:
        return False
    return bool(re.search(
        r"(未|没有|并未|无法|不能).{0,16}(给出|报告|记载|包含|提供|明确|回答|比较|差距|结果|准确率|样本数|类别数)|"
        r"(无|无法)提供|无法回答",
        text,
        re.IGNORECASE,
    ))


def _has_explicit_absence_evidence(evidence_text: str) -> bool:
    """只有证据本身明确写出缺失/未覆盖事实时，才允许给负向声明补引用。"""
    text = re.sub(r"\s+", "", str(evidence_text or "")).lower()
    if not text:
        return False
    return bool(re.search(
        r"未覆盖数据集证据|未报告数值对比|未给出|没有给出|没有报告|未明确记载|未提供|"
        r"not(?:explicitly)?(?:reported|mentioned|provided|given|stated)|notavailable|"
        r"notincluded|no(?:reported)?(?:accuracy|result|comparison)",
        text,
        re.IGNORECASE,
    ))


def _iter_clause_boundaries(text: str):
    depth = 0
    in_latex_math = False
    i = 0
    while i < len(text or ""):
        if text.startswith(("\\(", "\\["), i):
            in_latex_math = True
            i += 2
            continue
        if in_latex_math:
            if text.startswith(("\\)", "\\]"), i):
                in_latex_math = False
                i += 2
                continue
            i += 1
            continue

        ch = text[i]
        if ch in "([{（【｛":
            depth += 1
        elif ch in ")]}）】｝":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "，,：:":
            end = i + 1
            while end < len(text) and text[end].isspace():
                end += 1
            yield i, end
            i = end
            continue
        i += 1


def _clause_spans(text: str) -> List[Tuple[int, int, str]]:
    def _valid_clause(clause_text: str) -> bool:
        stripped = _CITATION_RE.sub("", clause_text).strip()
        if len(stripped) < 6 or _is_non_factual_sentence(clause_text):
            return False
        if clause_text.count("[") != clause_text.count("]"):
            return False
        if clause_text.count("（") != clause_text.count("）"):
            return False
        return True

    spans: List[Tuple[int, int, str]] = []
    start = 0
    for boundary_start, boundary_end in _iter_clause_boundaries(text or ""):
        end = boundary_start
        clause = (text[start:end] or "").strip()
        if _valid_clause(clause):
            spans.append((start, end, clause))
        start = boundary_end
    tail = (text[start:] or "").strip()
    if _valid_clause(tail):
        spans.append((start, len(text or ""), tail))
    if not spans and text and _valid_clause(text):
        spans.append((0, len(text), text.strip()))
    return spans


def _likely_cross_language(sentence: str, evidence_text: str) -> bool:
    return _contains_cjk(sentence) and _contains_latin(evidence_text)


def _citation_support_score(sentence: str, evidence_text: str) -> float:
    clean_sentence = re.sub(r"\s+", "", _CITATION_RE.sub("", sentence or "")).lower()
    clean_evidence = re.sub(r"\s+", "", evidence_text or "").lower()
    if clean_sentence and clean_sentence in clean_evidence:
        return 1.0
    sent_terms = _support_terms(sentence)
    if not sent_terms:
        return 0.0
    ev_terms = _support_terms(evidence_text)
    if not ev_terms:
        return 0.0
    overlap = len(sent_terms & ev_terms)
    base_score = overlap / max(len(sent_terms), 1)
    sent_numbers = _number_tokens(clean_sentence)
    ev_numbers = _number_tokens(clean_evidence)
    if sent_numbers and ev_numbers and sent_numbers - ev_numbers:
        return base_score
    if overlap and _likely_cross_language(sentence, evidence_text):
        return max(base_score, overlap / max(min(len(sent_terms), 12), 1))
    return base_score


def _evaluate_citation_support(answer: str, evidence_units: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence_by_ref: Dict[int, Dict[str, Any]] = {}
    for e in evidence_units:
        try:
            evidence_by_ref[int(e["ref"])] = e
        except (KeyError, TypeError, ValueError):
            continue
    sentences = _split_answer_sentences(answer)
    cited_sentences = 0
    supported = 0
    invalid_refs: list[int] = []
    unsupported_refs: list[int] = []
    for sentence in sentences:
        refs = [int(r) for r in _CITATION_RE.findall(sentence)]
        if not refs:
            continue
        cited_sentences += 1
        sentence_supported = False
        for ref in refs:
            ev = evidence_by_ref.get(ref)
            if not ev:
                invalid_refs.append(ref)
                continue
            if _citation_support_score(sentence, ev.get("text", "")) >= 0.12:
                sentence_supported = True
            else:
                unsupported_refs.append(ref)
        if sentence_supported:
            supported += 1
    return {
        "citation_support_rate": round(supported / cited_sentences, 3) if cited_sentences else 0.0,
        "supported_cited_sentences": supported,
        "total_cited_sentences": cited_sentences,
        "invalid_refs": sorted(set(invalid_refs))[:20],
        "unsupported_refs": sorted(set(unsupported_refs))[:20],
    }


def _evaluate_answer_sentence_selector(answer: str, evidence_units: List[Dict[str, Any]], *, threshold: float = 0.12) -> Dict[str, Any]:
    evidence_by_ref: Dict[int, Dict[str, Any]] = {}
    for e in evidence_units:
        try:
            evidence_by_ref[int(e["ref"])] = e
        except (KeyError, TypeError, ValueError):
            continue
    sentences = _split_answer_sentences(answer)
    sentence_count = len(sentences)
    supported_sentences = 0
    candidate_sentences = 0
    gap_sentences = 0
    uncited_candidate_sentences = 0
    examples: List[Dict[str, Any]] = []
    for sentence in sentences:
        refs = [int(r) for r in _CITATION_RE.findall(sentence)]
        supported_existing = False
        for ref in refs:
            ev = evidence_by_ref.get(ref)
            if ev and _citation_support_score(sentence, ev.get("text", "")) >= threshold:
                supported_existing = True
                break
        candidates: List[Dict[str, Any]] = []
        for ev in evidence_units:
            try:
                ref = int(ev.get("ref"))
            except (TypeError, ValueError):
                continue
            score = _citation_support_score(sentence, ev.get("text", ""))
            if score >= threshold:
                candidates.append({
                    "ref": ref,
                    "score": round(score, 4),
                    "page": ev.get("page"),
                    "group_id": ev.get("group_id", ""),
                })
        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        top_candidates = candidates[:2]
        if supported_existing:
            supported_sentences += 1
        if supported_existing or top_candidates:
            candidate_sentences += 1
        if top_candidates and not supported_existing:
            gap_sentences += 1
            if not refs:
                uncited_candidate_sentences += 1
            if len(examples) < 8:
                examples.append({
                    "sentence": sentence[:180],
                    "existing_refs": refs,
                    "candidate_refs": top_candidates,
                })
    return {
        "sentence_count": sentence_count,
        "supported_citation_sentence_count": supported_sentences,
        "candidate_sentence_count": candidate_sentences,
        "gap_sentence_count": gap_sentences,
        "uncited_candidate_sentence_count": uncited_candidate_sentences,
        "answer_sentence_selector_gap_rate": round(gap_sentences / sentence_count, 3) if sentence_count else 0.0,
        "citation_recall_with_evidence_selector_candidates": round(candidate_sentences / sentence_count, 3) if sentence_count else 0.0,
        "examples": examples,
    }


def _apply_selector_citation_fill(
    answer: str,
    evidence_units: List[Dict[str, Any]],
    *,
    threshold: float = 0.14,
    min_apply_score: float = 0.18,
    strong_threshold: float = 0.22,
    min_margin: float = 0.06,
    negative_absence_threshold: float = 0.28,
    clause_threshold: float = 0.34,
    clause_strong_threshold: float = 0.34,
    max_refs_per_sentence: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    normalized_answer = _normalize_post_sentence_citations(answer or "")
    parts = _SELECTOR_SENTENCE_SPLIT_RE.split(normalized_answer)
    rebuilt_parts: List[str] = []
    added_sentence_count = 0
    added_ref_count = 0
    examples: List[Dict[str, Any]] = []
    for idx in range(0, len(parts), 2):
        body = parts[idx]
        separator = parts[idx + 1] if idx + 1 < len(parts) else ""
        segment = f"{body}{separator}"
        clean_sentence = segment.strip()
        if len(clean_sentence) <= 5 or _is_non_factual_sentence(clean_sentence):
            rebuilt_parts.append(segment)
            continue
        negative_absence_claim = _is_negative_absence_claim(clean_sentence)
        existing_refs = [int(r) for r in _CITATION_RE.findall(clean_sentence)]
        supported_existing = False
        candidate_by_ref: Dict[int, Dict[str, Any]] = {}
        for ev in evidence_units:
            try:
                ref = int(ev.get("ref"))
            except (TypeError, ValueError):
                continue
            score = _citation_support_score(clean_sentence, ev.get("text", ""))
            if ref in existing_refs and score >= threshold:
                supported_existing = True
            if score < threshold:
                continue
            prev = candidate_by_ref.get(ref)
            if prev and prev.get("score", 0.0) >= score:
                continue
            candidate_by_ref[ref] = {
                "ref": ref,
                "score": round(score, 4),
                "explicit_absence_evidence": _has_explicit_absence_evidence(ev.get("text", "")),
                "page": ev.get("page"),
                "group_id": ev.get("group_id", ""),
            }
        candidates = sorted(candidate_by_ref.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        if supported_existing:
            clause_candidate: Optional[Dict[str, Any]] = None
            if len(existing_refs) < max_refs_per_sentence:
                body_text = segment[: len(segment) - len(separator)] if separator else segment
                for start, end, clause in _clause_spans(body_text):
                    if _CITATION_RE.search(clause):
                        continue
                    if _is_negative_absence_claim(clause):
                        continue
                    clause_supported_by_existing = False
                    for ev in evidence_units:
                        try:
                            ref = int(ev.get("ref"))
                        except (TypeError, ValueError):
                            continue
                        if ref not in existing_refs:
                            continue
                        if _citation_support_score(clause, ev.get("text", "")) >= clause_threshold:
                            clause_supported_by_existing = True
                            break
                    if clause_supported_by_existing:
                        continue
                    clause_candidates: List[Dict[str, Any]] = []
                    for ev in evidence_units:
                        try:
                            ref = int(ev.get("ref"))
                        except (TypeError, ValueError):
                            continue
                        if ref in existing_refs:
                            continue
                        score = _citation_support_score(clause, ev.get("text", ""))
                        if score < clause_threshold:
                            continue
                        clause_candidates.append({
                            "ref": ref,
                            "score": round(score, 4),
                            "explicit_absence_evidence": _has_explicit_absence_evidence(ev.get("text", "")),
                            "page": ev.get("page"),
                            "group_id": ev.get("group_id", ""),
                            "start": start,
                            "end": end,
                            "clause": clause[:120],
                        })
                    clause_candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
                    if not clause_candidates:
                        continue
                    top_clause_candidate = clause_candidates[0]
                    second_clause_score = clause_candidates[1].get("score", 0.0) if len(clause_candidates) > 1 else 0.0
                    top_clause_score = top_clause_candidate.get("score", 0.0)
                    if top_clause_score < clause_strong_threshold and (top_clause_score < min_apply_score or (top_clause_score - second_clause_score) < min_margin):
                        continue
                    if clause_candidate is None or top_clause_score > clause_candidate.get("score", 0.0):
                        clause_candidate = top_clause_candidate
            if clause_candidate:
                body_text = segment[: len(segment) - len(separator)] if separator else segment
                insert_at = int(clause_candidate["end"])
                updated_body = body_text[:insert_at] + f"[{clause_candidate['ref']}]" + body_text[insert_at:]
                rebuilt_parts.append(f"{updated_body}{separator}")
                added_sentence_count += 1
                added_ref_count += 1
                if len(examples) < 8:
                    examples.append({
                        "sentence": clean_sentence[:180],
                        "existing_refs": existing_refs,
                        "added_refs": [clause_candidate["ref"]],
                        "candidate_refs": [clause_candidate],
                        "anchor_mode": "clause",
                    })
                continue
            rebuilt_parts.append(segment)
            continue
        if negative_absence_claim:
            absence_candidates = [
                candidate for candidate in candidates
                if candidate.get("explicit_absence_evidence")
                and candidate.get("score", 0.0) >= negative_absence_threshold
            ]
            if absence_candidates:
                top_candidate = absence_candidates[0]
                tail_match = re.search(r"(\s*[。！？.!?；;]+\s*)$", segment)
                if tail_match:
                    updated_segment = segment[:tail_match.start()] + f"[{top_candidate['ref']}]" + tail_match.group(1)
                else:
                    updated_segment = f"{segment}[{top_candidate['ref']}]"
                rebuilt_parts.append(updated_segment)
                added_sentence_count += 1
                added_ref_count += 1
                if len(examples) < 8:
                    examples.append({
                        "sentence": clean_sentence[:180],
                        "existing_refs": existing_refs,
                        "added_refs": [top_candidate["ref"]],
                        "candidate_refs": absence_candidates[:2],
                        "anchor_mode": "negative_absence_sentence",
                    })
                continue
            rebuilt_parts.append(segment)
            continue
        if not candidates:
            rebuilt_parts.append(segment)
            continue
        top_candidate = candidates[0]
        second_score = candidates[1].get("score", 0.0) if len(candidates) > 1 else 0.0
        top_score = top_candidate.get("score", 0.0)
        if top_candidate["ref"] in existing_refs:
            rebuilt_parts.append(segment)
            continue
        if top_score < strong_threshold and (top_score < min_apply_score or (top_score - second_score) < min_margin):
            rebuilt_parts.append(segment)
            continue
        tail_match = re.search(r"(\s*[。！？.!?；;]+\s*)$", segment)
        if tail_match:
            updated_segment = segment[:tail_match.start()] + f"[{top_candidate['ref']}]" + tail_match.group(1)
        else:
            updated_segment = f"{segment}[{top_candidate['ref']}]"
        rebuilt_parts.append(updated_segment)
        added_sentence_count += 1
        added_ref_count += 1
        if len(examples) < 8:
            examples.append({
                "sentence": clean_sentence[:180],
                "existing_refs": existing_refs,
                "added_refs": [top_candidate["ref"]],
                "candidate_refs": candidates[:2],
                "anchor_mode": "sentence",
            })
    return "".join(rebuilt_parts), {
        "applied": added_sentence_count > 0,
        "mode": "selector_clause_top1_only_add",
        "threshold": threshold,
        "min_apply_score": min_apply_score,
        "strong_threshold": strong_threshold,
        "min_margin": min_margin,
        "negative_absence_threshold": negative_absence_threshold,
        "clause_threshold": clause_threshold,
        "clause_strong_threshold": clause_strong_threshold,
        "max_refs_per_sentence": max_refs_per_sentence,
        "added_sentence_count": added_sentence_count,
        "added_ref_count": added_ref_count,
        "examples": examples,
    }


def _strip_unsupported_citations(
    text: str,
    unsupported_refs: List[int],
    invalid_refs: List[int],
) -> str:
    """Phase 2.2：剔除不支持/无效的引用编号，保留文本内容"""
    refs_to_strip = set(unsupported_refs) | set(invalid_refs)
    if not refs_to_strip:
        return text
    def _replacer(m: re.Match) -> str:
        ref_num = int(m.group(1))
        return "" if ref_num in refs_to_strip else m.group(0)
    cleaned = _CITATION_RE.sub(_replacer, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _format_references(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """格式化参考资料块（仿 ragflow context schema）"""
    lines = []
    for ch in retrieved_chunks:
        if not isinstance(ch, dict):
            continue
        ref = ch.get("ref")
        if ref is None:
            continue
        text = (ch.get("text") or ch.get("source_text") or ch.get("chunk") or "").strip()
        if not text:
            continue
        # 截断每条参考资料避免 prompt 过长
        if len(text) > 2000:
            text = text[:2000] + "...(截断)"
        page_range = ch.get("page_range") or [ch.get("page", 0), ch.get("page", 0)]
        page_label = (
            f"p.{page_range[0]}"
            if page_range[0] == page_range[1]
            else f"p.{page_range[0]}-{page_range[1]}"
        )
        group_id = ch.get("group_id", "")
        lines.append(
            f"<context>\nID: {ref} | {page_label} | {group_id}\n└── 内容: {text}\n</context>"
        )
    return "\n\n".join(lines)


def _validate_only_add(
    original: str,
    enhanced: str,
    *,
    max_extra_chars: int = 80,
    max_extra_ratio: float = 0.20,
    min_keep_ratio: float = 0.6,
) -> bool:
    """only-add 安全校验：增强后答案应保留原文核心，仅多出引用编号。

    判断条件：
    - enhanced 长度不能比 original 长出 max_extra_chars 字符或 max_extra_ratio 比例（防止 LLM 重写）
    - enhanced 移除所有引用后，长度应 >= min_keep_ratio * original_clean（防止 LLM 删减重写）
    - 字符差异不应过大（粗略上限 200 字）
    """
    if not enhanced or not enhanced.strip():
        return False
    if len(enhanced) > len(original) + max(max_extra_chars, int(len(original) * max_extra_ratio)):
        return False
    orig_clean = _CITATION_RE.sub("", original).strip()
    enh_clean = _CITATION_RE.sub("", enhanced).strip()
    if not orig_clean:
        return False
    # 反向：增强后不应明显短于原文（防止 LLM 改写丢失内容）
    if len(enh_clean) < int(len(orig_clean) * min_keep_ratio):
        return False
    if abs(len(enh_clean) - len(orig_clean)) > 200:
        return False
    return True


async def enhance_citations(
    answer_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    *,
    api_key: str,
    model: str,
    provider: str,
    endpoint: str = "",
    coverage_threshold: float = 0.5,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """P3.6 二次引用注入

    Args:
        answer_text: First pass 生成的答案
        retrieved_chunks: 参考资料列表（带 ref/text/page_range/group_id）
        api_key/model/provider/endpoint: LLM 配置
        coverage_threshold: 触发阈值，引用覆盖率 < 此值才触发二次注入
        max_tokens: 二次调用的 max_tokens
        temperature: 二次调用温度（默认 0，确保稳定）

    Returns:
        (enhanced_answer, diagnostics)
        diagnostics: {
            "triggered": bool,
            "before_coverage": float,
            "after_coverage": float,
            "reason": str,        # "below_threshold" / "high_coverage_skip" / "no_chunks" / "llm_error" / "validation_failed"
            "elapsed_ms": float,
        }
    """
    diag: Dict[str, Any] = {
        "triggered": False,
        "before_coverage": 0.0,
        "after_coverage": 0.0,
        "reason": "",
        "elapsed_ms": 0.0,
        "evidence_count": 0,
        "evidence_schema": [],
        "selector_fill": {},
    }

    if not answer_text or not answer_text.strip():
        diag["reason"] = "empty_answer"
        return answer_text, diag

    if not retrieved_chunks:
        diag["reason"] = "no_chunks"
        return answer_text, diag

    evidence_units = _build_evidence_schema(retrieved_chunks)
    diag["evidence_count"] = len(evidence_units)
    diag["evidence_schema"] = [
        {k: v for k, v in ev.items() if k != "text"}
        for ev in evidence_units[:20]
    ]

    coverage, cited, total = estimate_citation_coverage(answer_text)
    diag["before_coverage"] = round(coverage, 3)
    diag["before_support"] = _evaluate_citation_support(answer_text, evidence_units)
    diag["before_selector"] = _evaluate_answer_sentence_selector(answer_text, evidence_units)

    if coverage >= coverage_threshold:
        diag["reason"] = "high_coverage_skip"
        diag["after_coverage"] = round(coverage, 3)
        diag["after_support"] = diag["before_support"]
        diag["after_selector"] = diag["before_selector"]
        filled_answer, selector_fill = _apply_selector_citation_fill(answer_text, evidence_units)
        diag["selector_fill"] = selector_fill
        if selector_fill.get("applied"):
            diag["triggered"] = True
            diag["reason"] = "high_coverage_skip_with_selector_fill"
            new_coverage, _, _ = estimate_citation_coverage(filled_answer)
            diag["after_coverage"] = round(new_coverage, 3)
            diag["after_support"] = _evaluate_citation_support(filled_answer, evidence_units)
            diag["after_selector"] = _evaluate_answer_sentence_selector(filled_answer, evidence_units)
            logger.info(
                f"[CitationEnhancer] selector 补引用: +{selector_fill.get('added_ref_count', 0)} refs, "
                f"{selector_fill.get('added_sentence_count', 0)} 句"
            )
            return filled_answer, diag
        logger.info(
            f"[CitationEnhancer] 引用覆盖率充足 ({coverage:.2f} >= {coverage_threshold}), 跳过二次注入 "
            f"({cited}/{total} 句已带引用)"
        )
        return answer_text, diag

    references_str = _format_references(retrieved_chunks)
    if not references_str:
        diag["reason"] = "no_valid_chunks"
        return answer_text, diag

    prompt = _CITATION_PLUS_PROMPT.format(
        references=references_str, answer=answer_text
    )

    try:
        import time
        from services.chat_service import call_ai_api

        started = time.perf_counter()
        response = await call_ai_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        diag["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)

        content = ""
        if isinstance(response, dict):
            if response.get("error"):
                diag["reason"] = f"llm_error: {response.get('error')}"
                logger.warning(f"[CitationEnhancer] LLM 调用失败: {response.get('error')}")
                filled_answer, selector_fill = _apply_selector_citation_fill(answer_text, evidence_units)
                diag["selector_fill"] = selector_fill
                if selector_fill.get("applied"):
                    diag["triggered"] = True
                    diag["reason"] = f"{diag['reason']}_with_selector_fill"
                    diag["after_coverage"] = round(estimate_citation_coverage(filled_answer)[0], 3)
                    diag["after_support"] = _evaluate_citation_support(filled_answer, evidence_units)
                    diag["after_selector"] = _evaluate_answer_sentence_selector(filled_answer, evidence_units)
                    return filled_answer, diag
                return answer_text, diag
            content = response.get("content") or ""
            if not content and "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content", "")
        else:
            content = str(response or "")

        enhanced = (content or "").strip()
        if not enhanced:
            diag["reason"] = "empty_response"
            filled_answer, selector_fill = _apply_selector_citation_fill(answer_text, evidence_units)
            diag["selector_fill"] = selector_fill
            if selector_fill.get("applied"):
                diag["triggered"] = True
                diag["reason"] = "empty_response_with_selector_fill"
                diag["after_coverage"] = round(estimate_citation_coverage(filled_answer)[0], 3)
                diag["after_support"] = _evaluate_citation_support(filled_answer, evidence_units)
                diag["after_selector"] = _evaluate_answer_sentence_selector(filled_answer, evidence_units)
                return filled_answer, diag
            return answer_text, diag

        # only-add 安全校验
        if not _validate_only_add(answer_text, enhanced):
            diag["reason"] = "validation_failed"
            logger.warning(
                f"[CitationEnhancer] 增强答案校验失败 (orig={len(answer_text)}, enhanced={len(enhanced)}), 回退原答案"
            )
            filled_answer, selector_fill = _apply_selector_citation_fill(answer_text, evidence_units)
            diag["selector_fill"] = selector_fill
            if selector_fill.get("applied"):
                diag["triggered"] = True
                diag["reason"] = "validation_failed_with_selector_fill"
                diag["after_coverage"] = round(estimate_citation_coverage(filled_answer)[0], 3)
                diag["after_support"] = _evaluate_citation_support(filled_answer, evidence_units)
                diag["after_selector"] = _evaluate_answer_sentence_selector(filled_answer, evidence_units)
                return filled_answer, diag
            return answer_text, diag

        new_coverage, new_cited, _ = estimate_citation_coverage(enhanced)
        diag["after_coverage"] = round(new_coverage, 3)
        after_sup = _evaluate_citation_support(enhanced, evidence_units)
        diag["after_support"] = after_sup
        diag["after_selector"] = _evaluate_answer_sentence_selector(enhanced, evidence_units)

        # Phase 2.2：低支持率自动降级 - 剔除不支持的引用
        stripped = False
        sup_rate = after_sup.get("citation_support_rate", 1.0)
        bad_refs = (after_sup.get("unsupported_refs") or []) + (after_sup.get("invalid_refs") or [])
        if sup_rate < 0.4 and bad_refs:
            enhanced = _strip_unsupported_citations(
                enhanced,
                after_sup.get("unsupported_refs", []),
                after_sup.get("invalid_refs", []),
            )
            stripped = True
            stripped_sup = _evaluate_citation_support(enhanced, evidence_units)
            diag["stripped_support"] = stripped_sup
            diag["stripped_selector"] = _evaluate_answer_sentence_selector(enhanced, evidence_units)
            new_coverage, new_cited, _ = estimate_citation_coverage(enhanced)
            diag["after_coverage"] = round(new_coverage, 3)
            logger.info(
                f"[CitationEnhancer] 低支持率降级: 剔除 {len(bad_refs)} 个不支持引用, "
                f"support_rate {sup_rate:.2f} → {stripped_sup.get('citation_support_rate', 0):.2f}"
            )
        diag["citations_stripped"] = stripped
        diag["triggered"] = True
        diag["reason"] = "enhanced_and_stripped" if stripped else "enhanced"
        enhanced, selector_fill = _apply_selector_citation_fill(enhanced, evidence_units)
        diag["selector_fill"] = selector_fill
        if selector_fill.get("applied"):
            diag["reason"] = f"{diag['reason']}_with_selector_fill"
            new_coverage, new_cited, _ = estimate_citation_coverage(enhanced)
            diag["after_coverage"] = round(new_coverage, 3)
            diag["after_support"] = _evaluate_citation_support(enhanced, evidence_units)
            diag["after_selector"] = _evaluate_answer_sentence_selector(enhanced, evidence_units)
            logger.info(
                f"[CitationEnhancer] selector 补引用: +{selector_fill.get('added_ref_count', 0)} refs, "
                f"{selector_fill.get('added_sentence_count', 0)} 句"
            )
        logger.info(
            f"[CitationEnhancer] 二次引用注入完成: coverage {coverage:.2f} → {new_coverage:.2f} "
            f"({cited}/{total} → {new_cited}/{total} 句), 耗时 {diag['elapsed_ms']}ms"
        )
        return enhanced, diag

    except Exception as e:
        diag["reason"] = f"exception: {str(e)[:120]}"
        logger.warning(f"[CitationEnhancer] 二次注入异常，回退原答案: {e}")
        filled_answer, selector_fill = _apply_selector_citation_fill(answer_text, evidence_units)
        diag["selector_fill"] = selector_fill
        if selector_fill.get("applied"):
            diag["triggered"] = True
            diag["reason"] = f"{diag['reason']}_with_selector_fill"
            diag["after_coverage"] = round(estimate_citation_coverage(filled_answer)[0], 3)
            diag["after_support"] = _evaluate_citation_support(filled_answer, evidence_units)
            diag["after_selector"] = _evaluate_answer_sentence_selector(filled_answer, evidence_units)
            return filled_answer, diag
        return answer_text, diag
