"""参考文献抽取与引用耦合（本地启发式，默认不联网）。

从已解析的块索引里取出参考文献条目，解析成可比对的标识，并用共享参考文献
（bibliographic coupling）衡量两篇论文的相关度。

设计约束与项目其余部分一致：
- 完全确定性：不调用模型，同样输入必得同样输出
- 默认离线：不请求 Crossref / Semantic Scholar，联网补全另有显式入口
- 标识优先级 DOI > arXiv > 标题指纹，低置信条目宁可丢弃也不参与耦合计算，
  因为一次错误配对会污染整张推荐图
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

REFERENCE_EXTRACTION_VERSION = "reference-extraction-v1"

_DOI_RE = re.compile(r"\b(?:doi[:\s]*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barXiv[:\s]*(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?:^|[^\d])((?:19|20)\d{2})(?:[^\d]|$)")
# 条目前缀："[1]"、"1."、"(1)"
_ENTRY_PREFIX_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s*")
_URL_TAIL_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
# 标题指纹用的停用词：只在缺 DOI/arXiv 时兜底，因此宁严勿宽。
_TITLE_NOISE_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
# 少于这个 token 数的标题不足以作为身份，避免 "Introduction" 之类误配。
_MIN_TITLE_FINGERPRINT_TOKENS = 4


@dataclass
class Reference:
    """一条被解析出来的参考文献。"""

    raw_text: str = ""
    doi: str = ""
    arxiv_id: str = ""
    title: str = ""
    year: int | None = None
    # doi / arxiv / title / none —— none 表示不参与耦合计算
    identity_kind: str = "none"
    identity: str = ""
    block_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normalize_doi(value: str) -> str:
    doi = _clean(value, 300).lower()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:").strip()
    # 句末标点常被正则一起吃进来
    return doi.rstrip(").,;:")


def _title_fingerprint(title: str) -> str:
    """标题指纹：只在没有 DOI/arXiv 时兜底。

    过短的标题不产生指纹——"Introduction"、"Related Work" 这类在参考文献解析
    失败时很常见，一旦当成身份会把互不相干的论文连起来。
    """
    tokens = [token for token in _TITLE_NOISE_RE.split(_clean(title, 400).lower()) if token]
    if len(tokens) < _MIN_TITLE_FINGERPRINT_TOKENS:
        return ""
    joined = " ".join(tokens)
    return "t:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def _guess_title(raw: str) -> str:
    """从条目里猜标题：作者段之后、期刊/年份之前那一段。

    参考文献格式千差万别，这里只做一件有把握的事——按句点切分后取最长的那
    段自然语言。取不准就返回空，让该条目退出耦合计算，而不是硬猜。
    """
    body = _URL_TAIL_RE.sub(" ", _ENTRY_PREFIX_RE.sub("", _clean(raw)))
    body = _DOI_RE.sub(" ", body)
    segments = [segment.strip(" .,;") for segment in re.split(r"(?<=[.。])\s+", body)]
    candidates = [
        segment for segment in segments
        # 作者段特征是大量逗号与缩写点；标题段更长且逗号少。
        if len(segment) >= 20 and segment.count(",") <= max(2, len(segment) // 40)
    ]
    if not candidates:
        return ""
    return _clean(max(candidates, key=len), 400)


def parse_reference_entry(raw_text: str, *, block_id: str = "") -> Reference:
    """把一条参考文献原文解析为结构化标识。"""
    raw = _clean(raw_text)
    reference = Reference(raw_text=raw[:1000], block_id=str(block_id or ""))
    if not raw:
        return reference

    doi_match = _DOI_RE.search(raw)
    if doi_match:
        reference.doi = _normalize_doi(doi_match.group(1))
    arxiv_match = _ARXIV_RE.search(raw)
    if arxiv_match:
        reference.arxiv_id = arxiv_match.group(1)
    year_match = _YEAR_RE.search(raw)
    if year_match:
        reference.year = int(year_match.group(1))
    reference.title = _guess_title(raw)

    # 身份优先级：DOI 最可靠，arXiv 次之，标题指纹仅作兜底。
    if reference.doi:
        reference.identity_kind = "doi"
        reference.identity = f"doi:{reference.doi}"
    elif reference.arxiv_id:
        reference.identity_kind = "arxiv"
        reference.identity = f"arxiv:{reference.arxiv_id}"
    else:
        fingerprint = _title_fingerprint(reference.title)
        if fingerprint:
            reference.identity_kind = "title"
            reference.identity = fingerprint
    return reference


def _split_reference_entries(text: str) -> list[str]:
    """把一整段参考文献切成条目。

    优先按 "[n]" 编号切；没有编号时退回按行切。切不出多条就整体作为一条。
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    numbered = re.split(r"(?=\[\d{1,3}\]\s)", cleaned)
    entries = [item.strip() for item in numbered if item.strip()]
    if len(entries) > 1:
        return entries
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [cleaned]


def extract_references(blocks: Iterable[Mapping[str, Any]]) -> list[Reference]:
    """从块索引里抽出全部参考文献条目。

    只认已被角色分类标成 reference 的块——重新判定一遍会与既有分类产生分歧，
    而那套分类已经处理了标题识别与正文边界。
    """
    references: list[Reference] = []
    seen: set[str] = set()
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        role = str(block.get("role") or block.get("block_role") or "")
        if role != "reference":
            continue
        text = str(block.get("text") or "")
        block_id = str(block.get("block_id") or block.get("id") or "")
        for entry in _split_reference_entries(text):
            reference = parse_reference_entry(entry, block_id=block_id)
            if not reference.identity:
                continue
            if reference.identity in seen:
                continue
            seen.add(reference.identity)
            references.append(reference)
    return references


def reference_identities(references: Sequence[Reference | Mapping[str, Any]]) -> set[str]:
    """取出可用于比对的标识集合。"""
    identities: set[str] = set()
    for reference in references or []:
        if isinstance(reference, Reference):
            identity = reference.identity
        elif isinstance(reference, Mapping):
            identity = str(reference.get("identity") or "")
        else:
            identity = ""
        if identity:
            identities.add(identity)
    return identities


def coupling_strength(
    left: Sequence[Reference | Mapping[str, Any]],
    right: Sequence[Reference | Mapping[str, Any]],
) -> dict[str, Any]:
    """两篇论文的引用耦合强度。

    共享参考文献数是主信号，Jaccard 作为归一化辅助信号：只看绝对数会让参考
    文献很多的综述压过一切，只看 Jaccard 又会让参考文献很少的短文虚高。两个
    都给出来，让调用方按场景取舍。
    """
    left_ids = reference_identities(left)
    right_ids = reference_identities(right)
    shared = left_ids & right_ids
    union = left_ids | right_ids
    return {
        "shared_count": len(shared),
        "jaccard": round(len(shared) / len(union), 4) if union else 0.0,
        "shared_identities": sorted(shared)[:50],
        "left_reference_count": len(left_ids),
        "right_reference_count": len(right_ids),
    }


def recommend_by_coupling(
    target_references: Sequence[Reference | Mapping[str, Any]],
    corpus: Mapping[str, Sequence[Reference | Mapping[str, Any]]],
    *,
    limit: int = 10,
    min_shared: int = 2,
) -> list[dict[str, Any]]:
    """按引用耦合给出推荐。

    ``min_shared`` 默认为 2：只共享一篇参考文献常常只是都引了同一篇经典综述，
    构不成主题相关。排序先看共享数再看 Jaccard，同分时按文档 ID 保证确定性。
    """
    target_ids = reference_identities(target_references)
    if not target_ids:
        return []

    rows: list[dict[str, Any]] = []
    for doc_id, references in (corpus or {}).items():
        strength = coupling_strength(target_references, references)
        if strength["shared_count"] < max(1, int(min_shared)):
            continue
        rows.append({
            "doc_id": str(doc_id),
            "shared_count": strength["shared_count"],
            "jaccard": strength["jaccard"],
            "shared_identities": strength["shared_identities"],
        })

    rows.sort(key=lambda row: (-row["shared_count"], -row["jaccard"], row["doc_id"]))
    return rows[: max(1, int(limit))]


def _bibtex_key(reference: Reference) -> str:
    if reference.doi:
        seed = reference.doi
    elif reference.arxiv_id:
        seed = f"arxiv{reference.arxiv_id}"
    else:
        seed = reference.title or reference.raw_text
    slug = re.sub(r"[^a-z0-9]+", "", str(seed).lower())[:24]
    return slug or "ref" + hashlib.sha1(str(reference.raw_text).encode("utf-8")).hexdigest()[:8]


def to_bibtex(reference: Reference | Mapping[str, Any]) -> str:
    """导出单条 BibTeX。

    只写解析出来的字段，不臆造作者或期刊——缺字段的条目在文献管理器里可以
    补，编造的字段却会被当成真的。
    """
    if isinstance(reference, Mapping):
        reference = Reference(**{
            key: value for key, value in reference.items()
            if key in Reference.__dataclass_fields__
        })
    entry_type = "article"
    fields: list[tuple[str, str]] = []
    if reference.title:
        fields.append(("title", reference.title))
    if reference.year:
        fields.append(("year", str(reference.year)))
    if reference.doi:
        fields.append(("doi", reference.doi))
    if reference.arxiv_id:
        entry_type = "misc"
        fields.append(("eprint", reference.arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if not fields:
        fields.append(("note", reference.raw_text[:300]))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{_bibtex_key(reference)},\n{body}\n}}"


def references_to_bibtex(references: Sequence[Reference | Mapping[str, Any]]) -> str:
    return "\n\n".join(to_bibtex(reference) for reference in references or [])
