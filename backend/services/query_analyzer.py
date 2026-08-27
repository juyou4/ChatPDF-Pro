"""
查询分析服务 - 分析用户查询类型，用于智能调整检索策略
"""
import re
from typing import Literal, Optional

from services.block_inventory_service import detect_inventory_kind
from services.paper_section_router import is_figure_identity_query, is_structure_map_query

QueryType = Literal['overview', 'extraction', 'analytical', 'specific', 'inventory']
EvidenceNeed = Literal[
    'numeric_table',
    'reference_trap',
    'section_explanation',
    'reference_meta',
    'comparison_multi_aspect',
    'analysis_explanation',
    'figure_caption',
    'code_implementation',
]


def _contains_any(query_lower: str, patterns: list[str]) -> bool:
    return any(pattern in query_lower for pattern in patterns)


def _contains_terms(query_lower: str, patterns: list[str]) -> bool:
    """中文按短语匹配，英文/数字术语按 token 边界匹配。"""
    for pattern in patterns:
        normalized = str(pattern or "").strip().lower()
        if not normalized:
            continue
        if re.search(r"[a-z0-9]", normalized):
            if re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", query_lower):
                return True
        elif normalized in query_lower:
            return True
    return False


def _is_numeric_table_cost_query(query_lower: str) -> bool:
    cost_patterns = [
        '额外开销', '计算开销', '推理开销', '推理时间', '训练时间', '耗时',
        'cost', 'costs', 'overhead', 'extra overhead', 'computation cost',
        'flops', 'inference time', 'inference-time', 'training time', 'training-time',
        'latency', 'runtime',
    ]
    return _contains_terms(query_lower, cost_patterns)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


_QUERY_TERM_STOPWORDS = {
    "请", "帮我", "解释", "说明", "总结", "概述", "主要", "内容", "如何", "怎么", "什么",
    "一下", "一下子", "论文", "文章", "文档", "the", "and", "or", "of", "in", "on", "for",
    "to", "a", "an", "is", "are", "was", "were", "this", "that", "paper", "article",
    "document", "what", "how", "why", "does", "do", "did", "please", "explain", "summarize",
    # 纯功能词：对术语抽取和可检索性判定都是噪声，没有任何一个能当检索锚点。
    # 代词
    "it", "its", "they", "them", "their", "theirs", "he", "she", "him", "her", "his",
    "we", "us", "our", "ours", "you", "your", "yours", "me", "my", "mine",
    "these", "those", "one", "ones", "another", "others", "someone", "something",
    # 疑问 / 限定
    "which", "who", "whom", "whose", "when", "where", "whether", "summarise",
    "some", "any", "all", "both", "each", "every", "either", "neither", "other",
    "same", "such", "no", "not", "yes", "none",
    # 助动词 / 系动词
    "be", "been", "being", "am", "has", "have", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # 介词 / 连词 / 副词
    "at", "by", "with", "without", "within", "into", "onto", "from", "as", "but",
    "if", "so", "because", "between", "among", "over", "under", "above", "below",
    "after", "before", "than", "then", "there", "here", "about",
    "also", "too", "very", "really", "just", "still", "again", "more", "most",
    "much", "many", "little", "few",
    # 口水词 / 泛化形容词
    "tell", "say", "said", "let", "know", "think", "want", "need",
    "good", "better", "best", "bad", "worse", "worst", "nice", "great",
    "former", "latter", "ok", "okay", "yeah", "hmm",
}


def _split_query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?", query or "")
    terms: list[str] = []
    for term in raw_terms:
        normalized = term.lower().strip("-_ ")
        if not normalized or normalized in _QUERY_TERM_STOPWORDS:
            continue
        terms.append(normalized)
    return _dedupe_preserve_order(terms)


# ---------------------------------------------------------------------------
# 可检索性闸门：count_content_terms
#
# 判定「这句话离开对话历史还能不能拿去检索」。返回 0 = 问句里一个实义词都没有，
# 这才是真歧义；返回 >= 1 说明有明确检索目标，无论里面出没出现代词。
# 调用方必须按离散的 ``== 0`` 使用，不要折算成分数——分数会重蹈 confidence
# 纯装饰的覆辙。
# ---------------------------------------------------------------------------

# 只在 count_content_terms 里生效：这些词对检索排序还有边际价值
# （"main contribution" / "second stage"），但单独出现时不构成检索目标，
# 所以不放进 _QUERY_TERM_STOPWORDS 去影响 extract_hl_ll_terms。
_CONTENT_TERM_EXTRA_STOPWORDS = {
    "main", "first", "second", "third", "fourth", "next", "last", "previous",
    "final", "following", "part", "parts", "thing", "things", "stuff",
}

# 中文没有词边界，无法像英文那样按 token 过滤。做法是先把功能词整体从原文里
# 挖掉，剩下的连续汉字段才是实义候选。多字词放这里，单字词放 _ZH_FUNCTION_CHARS。
_ZH_FUNCTION_PHRASES = (
    # 指代 / 回指
    "这篇论文", "这篇文章", "整篇论文", "整篇文章", "这个文档", "这份文档",
    "这一部分", "那一部分", "这一块", "那一块", "最后一个", "另外一个",
    "另一个", "上一个", "下一个", "前一个", "后一个",
    "第一个", "第二个", "第三个", "第一点", "第二点",
    "上述", "前者", "后者", "上面", "下面", "前面", "后面", "刚才", "刚刚",
    "他们", "她们", "它们", "我们", "你们", "咱们", "自己",
    "这个", "那个", "这些", "那些", "这里", "那里", "这儿", "那儿",
    "这块", "那块", "这部分", "那部分", "这种", "那种", "这样", "那样",
    "这么", "那么", "这点", "那点", "这条", "那条", "这项", "那项",
    "其中", "其他", "其它", "之前", "之后", "以上", "以下", "上文", "下文",
    # 疑问
    "为什么", "怎么样", "怎么办", "是什么", "做什么", "有没有", "是不是",
    "能不能", "可不可以", "会不会", "哪一个", "哪一些", "哪几个", "哪一点",
    "什么", "怎么", "怎样", "如何", "为何", "多少", "多久", "多大", "多长",
    "哪个", "哪些", "哪里", "哪儿", "几个", "几项", "是否", "谁",
    # 情态 / 口水 / 泛指
    "想知道", "告诉我", "解释一下", "说明一下", "总结一下", "介绍一下",
    "详细一点", "详细说说", "具体说说", "简单说说",
    "帮我", "请问", "麻烦", "一下", "一点", "一些", "有点",
    "可以", "能否", "应该", "需要", "想要", "知道", "觉得", "告诉",
    "说说", "讲讲", "看看", "聊聊", "展开", "继续", "然后", "接着",
    "另外", "此外", "尤其", "其实", "至于", "关于",
    "详细", "具体", "简单", "大概", "大致", "到底", "究竟",
    "还是", "或者", "以及", "并且", "但是", "不过",
    "意思", "内容", "部分", "东西", "情况", "方面", "地方",
    "非常", "比较", "更好", "更多", "更少", "不好", "最好",
    "设计",
)

# 单字功能词。只用于「整段残留全是功能字」的判定，不做逐字删除，
# 所以不会把「多模态」「大模型」这类实义词咬碎。
_ZH_FUNCTION_CHARS = frozenset(
    "的了着过是在和与或及对把被给从向到就都也还又再很太更最比"
    "能会要想请帮让使有没无不为之所以且而并将已该其此本"
    "这那它他她我你您们个条张些种样面里边上下前后"
    "好坏大小多少几哪什么怎吗呢么啊呀吧哦嗯谁"
    "做说讲看用跟同则即等各每另全总只才于"
)

_ZH_FUNCTION_PHRASE_RE = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in sorted(set(_ZH_FUNCTION_PHRASES), key=len, reverse=True)
    )
)

# 英文同形异类：`work` / `mean` / `compare` 在「(助动词) 代词 + 动词」或
# 「祈使动词 + 代词宾语」位置上是动词，不是可检索的名词锚点
# （How does it work? / Compare them.）；但 "the contribution of this work"
# 和 "the mean accuracy" 里它们是名词，必须保留。所以按位置删，不进停用词表。
_EN_LIGHT_VERB_RE = re.compile(
    r"\b(?:"
    r"(?:does|do|did|will|would|can|could|should|shall)\s+"
    r"(?:it|this|that|they|them|these|those|one|ones)\s+"
    r"(?:works?|means?|compares?|matters?|helps?|differs?)"
    r"|(?:it|they|this|that|these|those)\s+"
    r"(?:works|means|compares|matters|helps|differs)"
    r"|(?:compare|contrast|explain|describe|summarize|summarise|tell|show|list)\s+"
    r"(?:it|them|these|those|both|us|me|him|her)"
    r")\b"
)


def count_content_terms(query: str) -> int:
    """返回问句里实义词（非停用词）的个数，中英文都适用。

    ``0`` 的语义是「这句话里没有任何可以拿去检索的东西」——例如「这个怎么样」
    「How good is it?」。只有这种问句才谈得上必须先澄清；只要有一个实义词
    （「其中哪个模块」→ ``模块``），就应当直接检索，哪怕句子里带代词。

    调用方请严格用 ``count_content_terms(q) == 0`` 判定，不要设阈值。
    """
    text = str(query or "").lower()
    if not text.strip():
        return 0

    text = _EN_LIGHT_VERB_RE.sub(" ", text)

    zh_terms: list[str] = []
    for segment in re.findall(r"[一-鿿]+", _ZH_FUNCTION_PHRASE_RE.sub(" ", text)):
        if segment in _QUERY_TERM_STOPWORDS or segment in _CONTENT_TERM_EXTRA_STOPWORDS:
            continue
        if all(char in _ZH_FUNCTION_CHARS for char in segment):
            continue
        zh_terms.append(segment)

    latin_only = re.sub(r"[一-鿿]+", " ", text)
    en_terms = [
        term
        for term in _split_query_terms(latin_only)
        if term not in _CONTENT_TERM_EXTRA_STOPWORDS
    ]

    return len(_dedupe_preserve_order(zh_terms + en_terms))


_DOCUMENT_TERM_SAMPLE_BUDGET = 12_000
_DOCUMENT_TERM_SAMPLE_WINDOWS = 6


def _sample_document_terms(full_text: str, *, max_chars: int = _DOCUMENT_TERM_SAMPLE_BUDGET) -> str:
    """Take a bounded, evenly distributed term-extraction sample.

    Long PDFs often introduce dataset names and method acronyms in later
    chapters.  Sampling only the prefix silently makes those terms invisible
    to query expansion, while sending the entire document is unnecessary for
    this lightweight lexical pass.
    """
    text = str(full_text or "")
    if len(text) <= max_chars:
        return text
    window_count = min(_DOCUMENT_TERM_SAMPLE_WINDOWS, max_chars)
    window_size = max(1, max_chars // window_count)
    if window_count <= 1:
        return text[:max_chars]
    max_start = max(0, len(text) - window_size)
    starts = [round(max_start * index / (window_count - 1)) for index in range(window_count)]
    samples = [text[start:start + window_size] for start in starts]
    return "\n\n".join(samples)


def extract_document_bilingual_terms(
    full_text: str, max_pairs: int = 12
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """从文档全文中提取中英文术语对和关键短语，用于动态查询扩展和证据评分。

    提取策略：
    1. 提取英文专有名词（首字母大写的多词短语、全大写缩写）
    2. 提取中文技术术语（2-4 字的专业词汇）
    3. 基于共现关系建立中英桥接
    4. 提取高频关键短语用于证据评分

    Returns:
        (bridges, key_phrases) 元组：
        - bridges: list of (trigger_term, [expanded_terms])，用于 expand_academic_bilingual_terms
        - key_phrases: list of str，用于证据评分（高频术语列表）
    """
    if not full_text or len(full_text) < 200:
        return [], []

    # 固定预算覆盖全文，避免长文档的术语桥接只停留在摘要/引言。
    sample_text = _sample_document_terms(full_text)

    # 提取英文专有名词和缩写
    en_proper_nouns: dict[str, int] = {}
    en_acronyms: dict[str, int] = {}

    # 首字母大写的多词短语（如 "Neural Network", "Gradient Descent"）
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', sample_text):
        phrase = match.group(1).strip()
        if 5 <= len(phrase) <= 60:
            en_proper_nouns[phrase] = en_proper_nouns.get(phrase, 0) + 1

    # 全大写缩写（如 "CNN", "BERT", "GPT"）
    for match in re.finditer(r'\b([A-Z]{2,6})\b', sample_text):
        acronym = match.group(1).strip()
        if acronym not in {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS', 'HAD', 'CAN', 'MAY', 'ALL', 'USE', 'NEW', 'SET', 'GET', 'PUT'}:
            en_acronyms[acronym] = en_acronyms.get(acronym, 0) + 1

    # 提取中文技术术语（2-4 字，排除常见停用词）
    zh_terms: dict[str, int] = {}
    zh_stopwords = {'这个', '那个', '这些', '那些', '我们', '他们', '可以', '已经', '但是', '然而', '因此', '所以', '如果', '虽然', '或者', '以及', '通过', '使用', '进行', '实现', '提供', '采用', '基于', '一种', '不同', '其中', '由于', '为了', '关于', '以上', '以下', '首先', '其次', '最后'}
    for match in re.finditer(r'[\u4e00-\u9fff]{2,4}', sample_text):
        term = match.group(0)
        if term not in zh_stopwords and len(term) >= 2:
            zh_terms[term] = zh_terms.get(term, 0) + 1

    # 构建桥接：将英文术语作为扩展词，中文术语作为触发词
    bridges = []

    # 英文专有名词桥接（按频率排序）
    if en_proper_nouns:
        sorted_nouns = sorted(en_proper_nouns.items(), key=lambda x: (-x[1], x[0]))[:8]
        noun_list = [n for n, _ in sorted_nouns]
        for zh_term in sorted(zh_terms.keys(), key=lambda x: zh_terms[x], reverse=True)[:6]:
            bridges.append((zh_term, noun_list[:4]))

    # 英文缩写桥接
    if en_acronyms:
        sorted_acronyms = sorted(en_acronyms.items(), key=lambda x: (-x[1], x[0]))[:6]
        acronym_list = [a for a, _ in sorted_acronyms]
        for zh_term in sorted(zh_terms.keys(), key=lambda x: zh_terms[x], reverse=True)[:4]:
            bridges.append((zh_term, acronym_list[:3]))

    # 构建关键短语列表（用于证据评分）
    key_phrases = []

    # 高频英文术语
    for phrase, count in sorted(en_proper_nouns.items(), key=lambda x: -x[1]):
        if count >= 2:
            key_phrases.append(phrase.lower())

    # 高频缩写
    for acronym, count in sorted(en_acronyms.items(), key=lambda x: -x[1]):
        if count >= 2:
            key_phrases.append(acronym.lower())

    # 高频中文术语
    for term, count in sorted(zh_terms.items(), key=lambda x: -x[1]):
        if count >= 3:
            key_phrases.append(term)

    return bridges[:max_pairs], key_phrases[:30]


def expand_academic_bilingual_terms(
    query: str,
    max_terms: int = 18,
    doc_bridges: Optional[list[tuple[str, list[str]]]] = None,
) -> list[str]:
    """扩展查询的中英文学术术语。

    Args:
        query: 用户查询
        max_terms: 最大返回术语数
        doc_bridges: 文档级术语桥接（从文档全文中提取），优先于内置桥接
    """
    query_lower = (query or "").lower()

    # 文档级桥接优先（动态提取，绑定当前文档）
    if doc_bridges:
        doc_expanded: list[str] = []
        for trigger, terms in doc_bridges:
            if trigger in query_lower:
                doc_expanded.extend(terms)
        if doc_expanded:
            return _dedupe_preserve_order(doc_expanded)[:max_terms]

    # 通用学术中英文术语桥接（不绑定特定论文，覆盖常见学术领域）
    bridges = [
        # 基础学术概念
        (("数据集", "dataset"), ["dataset", "benchmark", "experimental results"]),
        (("评估指标", "指标", "metric"), ["metrics", "evaluation metrics", "performance metrics"]),
        (("实验", "设置", "结论", "结果"), ["experiments", "experimental results", "evaluation"]),
        (("准确率", "accuracy", "acc"), ["accuracy", "Accuracy", "Top-1", "Top-5"]),
        (("损失", "loss", "损失函数"), ["loss function", "training loss", "cross-entropy"]),
        (("收敛", "convergence", "训练轮数", "epoch"), ["convergence", "training epochs", "learning curve"]),
        # 模型/架构
        (("骨干", "骨干网络", "backbone", "预训练", "pre-trained"), ["backbone", "pre-trained model", "feature extractor"]),
        (("注意力", "attention", "自注意力", "self-attention"), ["attention mechanism", "self-attention", "multi-head attention"]),
        (("编码器", "解码器", "encoder", "decoder"), ["encoder", "decoder", "encoder-decoder"]),
        (("嵌入", "embedding", "表示学习", "representation"), ["embedding", "representation learning", "feature representation"]),
        # 训练/优化
        (("采样", "优化", "训练优化", "sampling", "optimization"), ["sampling", "optimization", "gradient descent", "backpropagation"]),
        (("学习率", "learning rate", "调度", "scheduler"), ["learning rate", "learning rate scheduler", "warmup"]),
        (("正则化", "regularization", "dropout", "过拟合"), ["regularization", "dropout", "overfitting", "weight decay"]),
        # 检索/文档
        (("检索", "retrieval", "召回", "recall"), ["retrieval", "recall", "ranking"]),
        (("摘要", "summary", "总结"), ["summary", "abstract", "overview"]),
        (("引用", "citation", "参考文献", "reference"), ["citation", "reference", "bibliography"]),
        # 通用中英桥接
        (("方法", "模型", "算法", "框架", "method", "model", "algorithm", "framework"), ["method", "model", "algorithm", "framework", "approach"]),
        (("原理", "机制", "设计", "实现"), ["principle", "mechanism", "design", "implementation"]),
        (("比较", "对比", "区别", "差异"), ["comparison", "difference", "distinction"]),
        (("优势", "劣势", "优缺点"), ["advantage", "disadvantage", "strength", "weakness"]),
    ]
    expanded: list[str] = []
    for triggers, terms in bridges:
        if any(trigger in query_lower for trigger in triggers):
            expanded.extend(terms)
    return _dedupe_preserve_order(expanded)[:max_terms]


def extract_hl_ll_terms(
    query: str,
    max_hl: int = 8,
    max_ll: int = 10,
    doc_bridges: Optional[list[tuple[str, list[str]]]] = None,
) -> dict:
    terms = _split_query_terms(query)
    if not terms:
        q = (query or "").strip()
        bilingual_terms = expand_academic_bilingual_terms(query, doc_bridges=doc_bridges)
        return {
            "high_level": _dedupe_preserve_order(([q] if q else []) + bilingual_terms[:4])[:max_hl],
            "low_level": bilingual_terms[:max_ll],
            "bilingual": bilingual_terms,
        }

    query_lower = (query or "").lower()
    evidence_need = analyze_evidence_need(query)
    preferred = []
    if "numeric_table" in evidence_need:
        preferred.extend(["table", "metric", "dataset", "accuracy", "score", "表格", "数据集", "指标"])
    if "comparison_multi_aspect" in evidence_need:
        preferred.extend(["compare", "comparison", "difference", "vs", "对比", "比较", "差异"])
    if "section_explanation" in evidence_need:
        preferred.extend([
            "method", "design", "implementation", "section", "contribution",
            "limitation", "dataset", "方法", "设计", "实现", "细节",
            "贡献", "结论", "局限", "数据集", "相关工作",
        ])
    if "analysis_explanation" in evidence_need:
        preferred.extend(["reason", "cause", "analysis", "comparison", "原因", "分析", "比较"])

    high_level: list[str] = []
    low_level: list[str] = []
    for term in terms:
        if term in preferred or any(p in term or term in p for p in preferred):
            high_level.append(term)
        elif len(term) >= 4 or re.search(r"\d", term):
            low_level.append(term)
        else:
            high_level.append(term)

    if not high_level:
        high_level = terms[:max_hl]
    if not low_level:
        low_level = [term for term in terms if term not in high_level[:max_hl]]
    if not low_level and query_lower:
        low_level = terms[:max_ll]

    bilingual_terms = expand_academic_bilingual_terms(query)
    return {
        "high_level": _dedupe_preserve_order(high_level + bilingual_terms[:4])[:max_hl],
        "low_level": _dedupe_preserve_order(low_level + bilingual_terms)[:max_ll],
        "bilingual": bilingual_terms,
    }


def is_section_explanation_query(query: str) -> bool:
    if not query:
        return False
    query_lower = query.lower()
    section_scope_patterns = [
        '部分', '章节', 'section', 'chapter',
        # 复数形原来靠裸子串顺带命中（'section' ⊂ 'sections'）。改成 token
        # 边界后必须显式列出，否则 "讲解一下这两个 sections" 会丢掉判定。
        'sections', 'chapters',
    ]
    section_explain_patterns = [
        '讲解', '解释', '说明', '介绍', '原理', '设计', '实现', '细节', '展开',
        '总结', '概括', '概述', '分析',
    ]
    numbered_section = bool(re.search(
        r"(?:第\s*[\d一二三四五六七八九十]+\s*(?:章|节)|\b(?:section|chapter)\s*\d+\b)",
        query_lower,
        re.IGNORECASE,
    ))
    # 裸子串会让 'section' 命中 'intersection'/'cross-section'、
    # 'chapter' 命中 'chapters'，英文一律按 token 边界判定（中文仍按短语）。
    return (
        (_contains_terms(query_lower, section_scope_patterns) or numbered_section)
        and _contains_terms(query_lower, section_explain_patterns)
    )


def is_overview_query(query: str) -> bool:
    if not query:
        return False
    query_lower = query.lower()
    if re.search(
        r"(?:不要|不用|别|不必|无需|do\s+not|don't|not)\s*(?:总结|概括|概述|summary|summarize|overview)",
        query_lower,
    ):
        return False
    overview_patterns = [
        '总结', '概括', '概述', '简述', '大意', '主要内容',
        '讲什么', '关于什么', '整体', '全文',
        '包括什么', '涉及什么',
        'summary', 'summarize', 'summarise', 'overview', 'outline', 'main idea',
        'what is this about', 'what does it cover',
    ]
    # 裸子串匹配会让 'outline' 命中 'outlined'、'summary' 命中 'summaries'，
    # 更糟的是让短英文词落进长单词里。英文一律按 token 边界判定。
    explicit_overview = _contains_terms(query_lower, overview_patterns)
    # 章节/图表/结构范围的问句永远不是整篇概览，这三道否决保持无条件。
    if (
        is_section_explanation_query(query)
        or is_figure_identity_query(query)
        or is_structure_map_query(query)
    ):
        return False
    # facet 否决必须留一个出口：「用三句话概括论文解决了什么问题、用了什么方法、
    # 取得了什么结果」枚举的是这份摘要要覆盖的方面，不是要去某一节里查的目标。
    # 无条件否决会把这类标准三要素概览判成 analytical，取证预算按分析题翻倍。
    if is_paper_facet_identity_query(query) and not (
        explicit_overview
        and _paper_facet_aspect_count(query) >= MIN_OVERVIEW_FACET_ASPECTS
    ):
        return False
    if explicit_overview:
        return True
    whole_document_nouns = [
        '这篇论文', '本文', '整篇论文', '这篇文章', '整篇文章', '该文档', '整个文档',
        'this paper', 'the paper', 'this article', 'the document',
    ]
    if _contains_terms(query_lower, ['介绍', '背景', 'introduce', 'introduction', 'background']):
        return _contains_terms(query_lower, whole_document_nouns)
    if '有哪些' in query_lower:
        return _contains_terms(query_lower, ['主要贡献', '核心贡献', '章节', '主题', '关键点', '主要结论'])
    return False


# ---------------------------------------------------------------------------
# 机制/原理问句的句式模板
#
# 为什么关键词表单独不够用（两个方向都会错）：
#
#   1. false positive —— 裸 ``in`` 让 'how' 落进 'show'/'however'，
#      "Which module does Table 3 show?" 被判成机制解释题，top_k 被推到 16。
#      这半用 ``_contains_terms`` 的 token 边界就能修掉。
#   2. false negative —— 边界化修不了这半。原来的判定是
#      ``explain 标记 AND scope 名词`` 一道封闭 AND，而 scope 是一张死名词表
#      （方法/模型/算法/框架/架构/网络/模块/组件）。
#      "How does the attention mechanism work?"、"How is the loss computed?"、
#      "encoder 的结构是怎么设计的" 的主语一个都不在表里，永远返回 False。
#      继续往名词表里塞词是无底洞：机制问句的主语可以是论文里任意一个自造组件名。
#
# 参照：工作区里 RAGFlow / kotaemon / paper-qa / LightRAG 的问句意图分类全部
# 交给 LLM，唯一的确定性实现是 RAGFlow ``rag/nlp/__init__.py`` 的
# ``re.match(ask_reg, pure_section)`` —— 判的是「疑问词出现在句首」这个句法位置，
# 而不是「句子里含有疑问词」这个词袋事实（它自己那条 ask_reg 没有右边界，
# 带着同款 'how' ⊂ 'however' 的 bug，只是被 re.match 的位置约束压住了）。
# 本模板沿用同一条思路：机制问句的可靠信号是**句式**——疑问副词 + 机制动词，
# 或解释祈使 + 机制名词；中间的主语允许任意跨度，只锚定两端。
#
# 跨度一律写成 ``[^?.!]{0,N}?`` / ``[^？?。！!，,；;、]{0,N}?``：
# 非贪婪 + 显式终止符。用字符类而不是 ``.`` 是为了不让匹配跨过句子/从句边界
# 拼出假命中（"How many? It works." 不该算），顺带避免 ``.*`` 的回溯开销。
# ---------------------------------------------------------------------------

# 主动式：how + 助动词 + <任意主语> + 机制动词。
# 助动词紧跟 how 是关键的一道闸：'how many' / 'how much' 落不进来，
# 所以 "How many parameters does this model have?" 不会被误判成机制题。
_MECH_EN_ACTIVE = (
    r"\bhow\s+(?:do|does|did|can|could|would|will|should)\b"
    r"[^?.!]{0,80}?"
    r"\b(?:work|works|worked|operate|operates|function|functions|"
    r"perform|performs|achieve|achieves|handle|handles|"
    r"compute|computes|process|processes|learn|learns|train|trains|"
    r"generate|generates|produce|produces|decide|decides|"
    r"determine|determines|select|selects|choose|chooses|"
    r"combine|combines|interact|interacts|aggregate|aggregates|"
    r"encode|encodes|decode|decodes|propagate|propagates|"
    r"converge|converges|update|updates|optimize|optimizes)\b"
)
# 被动式：how + 系动词 + <任意主语> + 过去分词。语义与主动式等价
# （"How is the loss computed?" == "How does it compute the loss?"），
# 只写主动式模板等于只修了一半的一半。
_MECH_EN_PASSIVE = (
    r"\bhow\s+(?:is|are|was|were)\b"
    r"[^?.!]{0,80}?"
    r"\b(?:computed|calculated|implemented|trained|derived|constructed|built|"
    r"designed|obtained|performed|handled|produced|generated|encoded|decoded|"
    r"aggregated|initialized|optimized|updated|fused|selected|defined|"
    r"formulated|structured|organized|integrated|applied|normalized|"
    r"sampled|partitioned|parameterized)\b"
)
# 祈使式：解释动词 + <任意宾语> + 机制名词。名词必须是机制类的，
# "Explain the results" 不该进来。
_MECH_EN_IMPERATIVE = (
    r"\b(?:explain|describe|elaborate\s+on|clarify|"
    r"walk\s+(?:me\s+|us\s+)?through|break\s+down)\b"
    r"[^?.!]{0,60}?"
    r"\b(?:mechanism|mechanisms|principle|principles|pipeline|pipelines|"
    r"procedure|procedures|workflow|workflows|rationale|intuition|"
    r"derivation|formulation|inner\s+workings?)\b"
)
# "tell me how the encoder works" / "how it works"：how 与 work 共现且不跨句。
# 依赖 \bhow\b 的右边界挡住 'however'（'however'[3] == 'e' 不是词边界）。
_MECH_EN_HOW_WORKS = r"\bhow\b[^?.!]{0,60}?\bworks?\b"

# 中文：疑问副词 + 机制动词。刻意不收 “做/理解/样”——
# "他们是怎么做的" / "这个怎么样" 是真歧义问句（评测集 zh_true_ambiguous），
# 不是机制解释题。
_MECH_ZH_HOW_VERB = (
    r"(?:如何|怎么|怎样|咋)"
    r"[^？?。！!，,；;、]{0,12}?"
    r"(?:工作|运作|运行|运转|实现|完成|执行|处理|计算|算出|算得|推导|求得|得到|"
    r"训练|微调|推理|生成|产出|设计|构建|搭建|组织|划分|切分|定义|建模|"
    r"融合|聚合|结合|集成|编码|解码|对齐|更新|优化|初始化|采样|筛选|选择|"
    r"起作用|发挥作用)"
)
# 中文机制名词本身就是强信号，不需要再要求 scope 名词。
# 注意都是复合词：单字「原理」「机制」留给下面两条带句式约束的模板，
# 否则 "应该怎么理解注意力机制" 这种含机制字样的普通问句会被无条件吞掉。
_MECH_ZH_NOUN = (
    r"(?:工作原理|运行原理|运作原理|实现原理|底层原理|基本原理|设计原理|数学原理|"
    r"内部机制|工作机制|运行机制|运作机制|实现机制|内部结构|实现细节|技术细节|"
    r"设计思路|设计动机|推导过程|计算过程|训练流程|推理流程)"
)
_MECH_ZH_EXPLAIN_NOUN = (
    r"(?:解释|说明|讲解|讲讲|说说|阐述|介绍|描述|剖析|分析)"
    r"[^？?。！!]{0,12}?"
    r"(?:原理|机制|工作方式|实现方式|运作方式|实现细节|流程|步骤)"
)
_MECH_ZH_NOUN_QUESTION = (
    r"(?:原理|机制|流程|步骤|过程)\s*(?:是|为)?\s*(?:什么|怎样|怎么样|如何|什么样)"
)

_MECHANISM_TEMPLATE_RE = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in (
            _MECH_EN_ACTIVE,
            _MECH_EN_PASSIVE,
            _MECH_EN_IMPERATIVE,
            _MECH_EN_HOW_WORKS,
            _MECH_ZH_HOW_VERB,
            _MECH_ZH_NOUN,
            _MECH_ZH_EXPLAIN_NOUN,
            _MECH_ZH_NOUN_QUESTION,
        )
    ),
    re.IGNORECASE,
)


_METHOD_IDENTITY_ZH_RE = re.compile(
    r"(?:核心方法|研究方法|本文方法|所用方法|提出的方法|"
    r"(?:使用|采用|用了)了?什么(?:研究)?方法|"
    r"方法(?:是什么|有哪些|是怎样))"
)
_METHOD_IDENTITY_EN_RE = re.compile(
    r"(?:what\s+(?:is|are|was)\s+(?:the\s+)?(?:core|main|proposed|overall)?\s*"
    r"(?:method|approach|methodology)|"
    r"what\s+(?:method|approach|methodology)\s+(?:does|did|is)\b)",
    re.IGNORECASE,
)
_METHOD_IDENTITY_BLOCK_RE = re.compile(
    r"(?:表\s*\d+|table\s*\d+|准确率|f1\b|多少|highest|lowest|best\s+result)",
    re.IGNORECASE,
)


def is_method_identity_query(query: str) -> bool:
    """识别“核心方法/研究方法是什么”这类需要读方法节正文的身份问句。

    有意比机制解释更窄：不把「表 4 里的模型是什么」或指标抽取收进来。
    """
    if not query:
        return False
    if _METHOD_IDENTITY_BLOCK_RE.search(query):
        return False
    if is_overview_query(query):
        return False
    return bool(
        _METHOD_IDENTITY_ZH_RE.search(query)
        or _METHOD_IDENTITY_EN_RE.search(query)
    )


_FACET_IDENTITY_BLOCK_RE = re.compile(
    r"(?:表\s*\d+|table\s*\d+|准确率|f1\b|多少|highest|lowest|best\s+result|"
    r"how\s+(?:much|many)|分别是多少|提升多少)",
    re.IGNORECASE,
)
_FACET_IDENTITY_RE = re.compile(
    r"(?:"
    r"(?:主要|核心|关键)(?:贡献|创新点|贡献点)|贡献(?:是什么|有哪些|有什么)|"
    r"(?:有哪些|有什么)(?:主要|核心|关键)?(?:贡献|创新点|贡献点|结论|发现|局限|不足)|"
    r"(?:主要|实验)结论|结论是什么|(?:主要)?发现是什么|"
    r"(?:局限性|不足之处|主要缺陷)|局限(?:性|之处)?(?:是什么|有哪些|在哪)|"
    r"不足(?:之处)?(?:是什么|有哪些)|"
    r"(?:实验设置|实验设计|实验配置|实现细节)|"
    r"(?:用了?(?:哪些|什么)数据集|数据集是什么|使用了?什么数据)|"
    r"(?:相关工作|related\s+works?)(?:部分|章节|section)?\s*"
    r"(?:讲了?什么|介绍|总结|概述|discuss|cover|review|say)|"
    r"(?:要解决什么问题|解决了什么问题|研究动机)|"
    r"(?:网络结构|模型结构|整体架构)\s*(?:是什么|是怎样|如何)|"
    r"(?:实验结果|实验发现)\s*(?:是什么|有哪些|如何|怎么样)|"
    r"experimental\s+setup|implementation\s+details?|"
    r"what\s+(?:datasets?|data)\s+(?:does|did|is|are)\b|"
    r"which\s+datasets?\b|"
    r"what\s+(?:problem|motivation)\b|"
    r"what\s+(?:are|is)\s+(?:the\s+)?(?:main\s+|key\s+|core\s+)?"
    r"(?:contributions?|limitations?|conclusions?|findings?|results)\b"
    r")",
    re.IGNORECASE,
)


def is_paper_facet_identity_query(query: str) -> bool:
    """识别贡献/结论/局限/设置/数据集/相关工作等需要读对应节正文的身份问句。

    与 ``is_method_identity_query`` 并列：不收表格数值，也不把整篇概览收进来。
    """
    if not query:
        return False
    if _FACET_IDENTITY_BLOCK_RE.search(query):
        return False
    return bool(_FACET_IDENTITY_RE.search(query))


# 论文要素的粗分组。这张表只用来数「一句话里点了几个方面」，判断一个带概括词的
# 问句要的是整篇摘要还是某一个要素，因此按语义分组去重，不做细粒度句式识别。
_PAPER_FACET_ASPECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:要?解决(?:了)?[^。？?！!，,、]{0,8}?问题|研究(?:动机|问题)|问题是什么|"
        r"what\s+problem|motivation)",
        r"(?:(?:核心|研究|本文|所用|提出的|主要)方法|方法(?:是什么|有哪些)|"
        r"(?:用|使用|采用|提出)了?[^。？?！!，,、]{0,8}?(?:方法|模型|框架|技术|思路)|"
        r"what\s+(?:method|approach)\b)",
        r"(?:(?:实验|主要|最终)结果|(?:取得|得到|达到)了?[^。？?！!，,、]{0,8}?(?:结果|效果|性能|表现)|"
        r"结果如何|效果如何|what\s+results?\b)",
        r"(?:贡献|创新点|contributions?)",
        r"(?:局限|不足|缺陷|limitations?)",
        r"(?:结论|发现|conclusions?|findings?)",
        r"(?:数据集|实验设置|实验配置|datasets?|experimental\s+setup)",
    )
)
# 问题 + 方法 + 结果是论文概括的标准三要素。要求三个方面而不是两个：
# 「总结这篇论文的贡献和局限」只覆盖论文的一段，仍然该走 facet 取证。
MIN_OVERVIEW_FACET_ASPECTS = 3


def _paper_facet_aspect_count(query: str) -> int:
    """一个问句点到了几个不同的论文要素。"""
    return sum(1 for pattern in _PAPER_FACET_ASPECT_PATTERNS if pattern.search(query or ""))


def is_mechanism_explanation_query(query: str) -> bool:
    """识别论文方法机制类解释问题，用于触发 Agent 多轮取证。

    判定结构是**两条互相独立的成立路径**（照搬同仓 block_inventory_service.py
    的 ``detect_inventory_kinds`` 多分支门），而不是一道封闭的 AND：

    * 分支 A：句式模板命中，**不要求** scope 名词。修 false negative——
      机制问句的主语是开放集合，靠名词表永远追不上。
    * 分支 B：解释标记 + 范围名词，两边都按 ``_contains_terms`` 的
      token 边界判定。修 false positive——裸 ``in`` 会把 'how' 判进 'show'。

    两条都不命中就干净地返回 False，不留模糊词兜底。
    """
    if not query:
        return False
    query_lower = query.lower()

    # 分支 A：句式模板
    if _MECHANISM_TEMPLATE_RE.search(query_lower):
        return True

    # 分支 B：解释标记 + 范围名词
    #
    # 这里刻意**不再**收 'how' / '如何' / '怎么'：疑问词本身不携带机制语义
    # （RAGFlow 的 rmWWW 干脆把它们当检索噪声删掉），把裸 'how' 当强信号的后果
    # 就是 "How many parameters does this model have?" 因为 how + model 双门
    # 都过而被判成机制题。真正的 how 类机制问句由上面的句式模板负责。
    # 同理移除 'what is' / 'what does' / '是什么' / '做什么'：这些是身份/内容
    # 询问标记，"What is the model size?" / "表 4 里的模型是什么" 问的是一个
    # 属性值，不需要跨段落读方法节。
    explain_patterns = [
        '解释', '说明', '讲解', '原理', '机制', '实现', '细节',
        '解决什么', '解决了什么',
        'explain', 'mechanism', 'implementation', 'principle',
    ]
    mechanism_scope_patterns = [
        '方法', '模型', '算法', '框架', '架构', '网络', '模块', '组件',
        'method', 'model', 'algorithm', 'framework', 'architecture', 'network', 'module',
    ]
    return (
        _contains_terms(query_lower, explain_patterns)
        and _contains_terms(query_lower, mechanism_scope_patterns)
    )


def is_analysis_explanation_query(query: str) -> bool:
    """识别需要跨段落论证的原因/对比/成因类论文分析问题。"""
    if not query:
        return False
    query_lower = query.lower()
    analysis_patterns = [
        '为什么', '原因', '成因', '理由', '根本不同', '本质区别',
        '相比', '比较', '对比', '区别', '差异', '如何分析', '影响最大',
        'why', 'because', 'reason', 'compare', 'comparison', 'difference',
        'versus', 'vs', 'ablation', '消融',
        'reasons', 'compared', 'comparisons', 'differences', 'ablations',
    ]
    # 通用学术论文范围词（不绑定特定论文/领域）
    paper_scope_patterns = [
        '方法', '模型', '算法', '框架', '组件', '模块', '样本', '数据',
        '训练', '推理', '实验', '结果', '性能', '指标',
        'method', 'model', 'algorithm', 'framework', 'module', 'sample', 'data',
        'training', 'inference', 'experiment', 'result', 'performance', 'metric',
        # 复数形是原来靠裸子串顺带命中的（'result' ⊂ 'results'）。改成 token
        # 边界后必须显式列出，否则 "Compare the ablation results ..." 会丢掉
        # analysis_explanation。注意不要加 'dataset'——那正是要挡掉的误命中。
        'results', 'experiments', 'models', 'methods', 'metrics',
        'samples', 'modules', 'frameworks', 'algorithms',
    ]
    # 裸子串会让 'data' 命中 'dataset'、'vs' 命中 'vsual' 之类的词内片段，
    # 把「Which dataset did the authors use?」误判成需要跨段落论证的分析题。
    return (
        _contains_terms(query_lower, analysis_patterns)
        and _contains_terms(query_lower, paper_scope_patterns)
    )


# caption 在视觉语言论文里是**训练目标/任务名**（caption language model、image
# caption、图像描述），不是表格标题。裸 \bcaption\b 曾把「Figure 2 中 caption
# language model 与 baseline 的训练效率关系」判成显式表格范围，叠加 baseline 的
# metric 信号后误判为 numeric_table——该类目带 1.0 硬闸，误判会同时把问题挡在
# Agent 之外并按精确抽取取证。真正的表格 caption 提问必然带表号或「表」字，
# 已由上面的表号模式覆盖，这里只负责剔除建模语义的同形词。
_CAPTION_TOKEN_RE = re.compile(r"\bcaptions?\b", re.IGNORECASE)
_CAPTION_MODELING_RE = re.compile(
    r"(?:\bimage\s+captions?\b|\bcaptions?\s+(?:language\s+model|model|generation|"
    r"generator|decoder|encoder|head|loss|objective|branch|task|pretraining|"
    r"supervision)\b|图像描述|图像字幕|看图说话)",
    re.IGNORECASE,
)


def _is_table_caption_scope(query_lower: str) -> bool:
    """判断 caption 是否指向表格标题，而非图像描述类建模术语。"""
    if not _CAPTION_TOKEN_RE.search(query_lower):
        return False
    return not _CAPTION_MODELING_RE.search(query_lower)


_EXPLICIT_TABLE_SCOPE_RE = re.compile(
    r"(?:表格|表\s*\d+|第\s*\d+\s*表|表中|\btable(?:s)?(?:\s*\d+)?\b)",
    re.IGNORECASE,
)


def _has_explicit_table_scope(query_lower: str) -> bool:
    """问句是否显式点名了某张表，而不只是提到数值。"""
    return bool(_EXPLICIT_TABLE_SCOPE_RE.search(query_lower)) or _is_table_caption_scope(query_lower)


# ---------------------------------------------------------------------------
# 代码实现问句
#
# 这一类的下游后果比其它 evidence_need 重：命中后 Agent 会启用论文仓库工具，
# 并且在读到仓库文件之前不允许 final。所以判定刻意收得很窄，只认两条路径：
#
#   * 分支 A：问句里出现明确的仓库/脚本制品词（训练脚本、源码、代码仓库、
#     training script、official implementation ...）。这些词单独出现就已经
#     指向"去看代码"，不需要再要求动作词。
#   * 分支 B：实现/定位类动作（怎么实现、在哪个文件、how to run ...）叠加一个
#     代码范围词（代码、仓库、脚本、loss、config ...）。单独的"怎么实现"是方法
#     机制题，由 section_explanation 负责，不进这里。
#   * 分支 C：用户直接点名要对照讲解（对照源码、详细讲解实现、walk me through
#     the implementation）。这类要求本身就是"读代码再讲"，不需要再叠范围词。
#
# 还有一类刻意留在外面：「这个方法怎么实现的」。它既可能是纯论文机制题，也可能
# 是想看代码，问句本身分辨不出来。``is_method_implementation_query`` 只负责识别
# 句型，是否升级成实现题由调用方结合"论文是否登记了可读仓库"决定
# （见 retrieval_agent._wants_code_implementation）。
#
# 反面样例必须挡住：「参考文献里的 GitHub 链接是什么」「论文的 arXiv URL」——
# 它们只是提到 github/链接，属于 reference_trap / reference_meta 的地盘。
# ---------------------------------------------------------------------------

_CODE_IMPLEMENTATION_BLOCK_RE = re.compile(
    r"(?:参考文献|引用列表|文献列表|bibliography|references?\s+(?:section|list)|"
    r"cited\s+(?:works?|papers?))",
    re.IGNORECASE,
)
_CODE_ARTIFACT_RE = re.compile(
    r"(?:训练脚本|推理脚本|评测脚本|测试脚本|运行脚本|启动脚本|训练代码|推理代码|"
    r"源代码|源码|代码仓库|代码库|代码实现|代码文件|开源代码|开源仓库|开源地址|开源实现|"
    r"仓库地址|仓库代码|仓库文件|仓库里|仓库中|复现代码|官方实现|官方代码|参考实现|"
    r"配置文件|training\s+scripts?|inference\s+scripts?|eval(?:uation)?\s+scripts?|"
    r"source\s+code|code\s?base|code\s+repositor(?:y|ies)|"
    r"(?:github|gitlab|hugging\s?face)\s+repo(?:sitor(?:y|ies))?|"
    r"official\s+(?:implementation|code|repo(?:sitory)?)|reference\s+implementation|"
    r"repo\s+files?|config(?:uration)?\s+files?)",
    re.IGNORECASE,
)
_CODE_ACTION_RE = re.compile(
    r"(?:怎么实现|如何实现|怎样实现|怎么写|如何写|怎么定义|如何定义|怎么跑|如何跑|"
    r"怎么运行|如何运行|怎么复现|如何复现|在哪实现|哪里实现|写在哪|放在哪|定义在哪|"
    r"在哪个文件|哪个文件|哪份文件|哪份配置|哪个脚本|哪个函数|哪个类|哪一行|"
    r"是否一致|是否相符|对得上|对不上|一致吗|相符吗|对应关系|如何对应|怎么对应|"
    r"how\s+(?:to|do\s+i|can\s+i)\s+(?:run|train|reproduce|implement|use)|"
    r"how\s+is\s+[^?.!]{0,40}?implemented|"
    r"(?:match(?:es)?|consistent\s+with|correspond(?:s)?\s+to)\s+the\s+paper|"
    r"where\s+(?:is|are|can\s+i\s+find)|which\s+(?:file|script|config|module|function))",
    re.IGNORECASE,
)
# 分支 C：明说要对照源码讲实现。这些说法自带"去读代码"的意图，单独出现即生效。
_CODE_WALKTHROUGH_RE = re.compile(
    r"(?:对照源码|对照代码|对照仓库|对照实现|结合源码|结合代码|逐行讲解|"
    r"详细讲解实现|讲解实现|讲讲实现|实现细节讲解|讲解一下实现|讲解其实现|"
    r"walk\s+(?:me\s+)?through\s+the\s+(?:implementation|code)|"
    r"explain\s+the\s+(?:implementation|source\s+code)\s+in\s+detail|"
    r"compare\s+the\s+code\s+(?:with|to|against)\s+the\s+paper)",
    re.IGNORECASE,
)
# 「这个方法怎么实现的」句型。命中不等于实现题，还要论文真有可读仓库。
_METHOD_IMPLEMENTATION_RE = re.compile(
    r"(?:(?:方法|模型|模块|机制|算法|框架|网络|流程|策略|损失|目标函数|注意力)"
    r"[^。！？?!]{0,12}?(?:怎么|如何|怎样)(?:实现|实作|落地|做到)"
    r"|(?:怎么|如何|怎样)实现[^。！？?!]{0,12}?(?:方法|模型|模块|机制|算法|框架|网络)"
    r"|how\s+(?:is|are|was|were)\s+[^?.!]{0,40}?\b(?:implemented|realised|realized)\b"
    r"|how\s+(?:do(?:es)?|did)\s+[^?.!]{0,40}?\bimplement\b)",
    re.IGNORECASE,
)
_CODE_SCOPE_RE = re.compile(
    r"(?:代码|源码|仓库|脚本|损失函数|优化器|数据加载|前向传播|反向传播|"
    r"\bcode\b|\brepo(?:sitory)?\b|\bscripts?\b|\bloss\b|\bconfig\b|\bcheckpoint\b|"
    r"\bdataloader\b|\boptimizer\b|\bgithub\b|\bgitlab\b|hugging\s?face)",
    re.IGNORECASE,
)


def is_code_implementation_query(query: str) -> bool:
    """识别"去读公开仓库代码"类问句，用于启用论文仓库工具。"""
    if not query:
        return False
    if _CODE_IMPLEMENTATION_BLOCK_RE.search(query):
        return False
    if _CODE_ARTIFACT_RE.search(query) or _CODE_WALKTHROUGH_RE.search(query):
        return True
    return bool(_CODE_ACTION_RE.search(query) and _CODE_SCOPE_RE.search(query))


def is_method_implementation_query(query: str) -> bool:
    """识别「这个方法怎么实现的」这类句型，本身不足以启用仓库工具。

    调用方必须再确认论文里抽出了可读取的公开仓库；没有仓库的论文继续走纯论文
    机制讲解。显式点名表格的数值题一律排除，不能把 numeric_table 抢走。
    """
    if not query:
        return False
    if _CODE_IMPLEMENTATION_BLOCK_RE.search(query):
        return False
    if not _METHOD_IMPLEMENTATION_RE.search(query):
        return False
    if _has_explicit_table_scope(query.lower()) and "numeric_table" in analyze_evidence_need(query):
        return False
    return True


def analyze_evidence_need(query: str) -> list[EvidenceNeed]:
    """识别学术论文场景下的证据类型需求。"""
    if not query:
        return []

    query_lower = query.lower()
    evidence_need: list[EvidenceNeed] = []
    cost_query = _is_numeric_table_cost_query(query_lower)

    explicit_table_scope = _has_explicit_table_scope(query_lower)
    numeric_table_metric_scope_patterns = [
        'baseline', '基线', 'metric', '指标', '阈值', 'threshold',
        '比率', 'ratio', 'percentage', 'accuracy', 'acc', 'score',
        'f1', 'bleu', 'rouge', 'precision', 'recall', 'asr', 'lpips',
        'fid', 'map', 'ap50', 'ap75',
        # 中文常用指标名：原表只有「指标/阈值/比率」这类元词，
        # 「准确率/召回率/平均分」这些真正出现在问句里的指标名一个都没有，
        # 导致「各方法在 ImageNet 上的 Top-1 准确率分别是多少」判不出需要读表。
        '准确率', '正确率', '精确率', '精准率', '召回率', '错误率', '命中率', '通过率',
        '得分率', '平均分', '平均值', '中位数', '方差', '标准差', '困惑度',
        'top-1', 'top-5', 'top1', 'top5', 'auc', 'mrr', 'ndcg',
        'psnr', 'ssim', 'rmse', 'mae', 'perplexity', 'ppl',
        'metrics', 'scores',
    ]
    numeric_value_request_patterns = [
        '数值', '具体数据', '分别是多少', '是多少', '多少', '百分点', '最高', '最低',
        '排名', '排第几', '第几', '差值', '差距', '提升多少', '下降多少', '高多少', '低多少',
        'what value', 'what score', 'what is', 'what are', 'how much', 'how many', 'highest', 'lowest',
        'best result', 'metric value', 'percentage point',
    ]
    # 「What accuracy does ...」/「What precision and recall does ...」这类
    # 「what + 指标名 + 助动词」结构在上面的短语表里匹配不到（表里只有
    # what is / what are / what value / what score）。这里只补疑问句式，
    # 仍然要求紧跟 what 的是一个真正的指标名，避免 “What method does…” 被拉进来。
    english_metric_question = bool(re.search(
        r"\bwhat\s+(?:[a-z0-9\-]+\s+(?:and|or)\s+)?"
        r"(?:accuracy|precision|recall|f1|f1-score|bleu|rouge|score|scores|"
        r"value|values|number|numbers|percentage|ratio|latency|throughput|speedup|"
        r"auc|mrr|ndcg|psnr|ssim|fid|perplexity|error\s+rate|success\s+rate)\s+"
        r"(?:does|do|did|is|are|was|were|can|could|would|will)\b",
        query_lower,
    ))
    metric_signal = _contains_terms(query_lower, numeric_table_metric_scope_patterns)
    quantitative_request = (
        _contains_terms(query_lower, numeric_value_request_patterns)
        or english_metric_question
    )
    fewshot_columns = sum(
        1 for term in ('many', 'medium', 'few')
        if _contains_terms(query_lower, [term])
    ) >= 2
    # 论文结果表常用 "数据集 + backbone + 子集/数值" 这样的限定，未必会
    # 显式写出 Table 或 Accuracy。例如 "CIFAR100-LT 上 ResNet-32 的
    # Few-shot 子集中哪个方法最好，数值是多少"。这类问题仍必须走数值表
    # 格的精确证据链，否则会被普通语义检索带到叙述段落。
    benchmark_scope = bool(re.search(
        r"\b(?:cifar|imagenet|places|coco|voc|cityscapes|ade20k|"
        r"kinetics|nus[-_]?wide|inaturalist)[a-z0-9_.-]*\b",
        query_lower,
        re.IGNORECASE,
    ))
    backbone_scope = bool(re.search(
        r"\b(?:resnet|resnext|wide[-_]?resnet|wrn|vgg|densenet|"
        r"mobilenet|efficientnet|convnext|swin|deit|vit|regnet|hrnet)"
        r"[a-z0-9_.-]*\b",
        query_lower,
        re.IGNORECASE,
    ))
    benchmark_backbone_scope = benchmark_scope and backbone_scope
    numeric_table_hit = (
        (cost_query and quantitative_request)
        or (fewshot_columns and (quantitative_request or explicit_table_scope or metric_signal))
        or (benchmark_backbone_scope and quantitative_request)
        or (explicit_table_scope and (quantitative_request or metric_signal))
        or (metric_signal and quantitative_request)
    )

    # ---- 表格数值优先级护栏 -------------------------------------------------
    # numeric_table 的判定必须先于 section_explanation 算出来。理由是
    # analyze_query_type 里的分支顺序：`numeric_table + strong_numeric_extraction`
    # 的 extraction 分支排在最前，但「是多少 / how much」这类强抽取词缺席时就落不到
    # 它，紧接着 section_explanation 分支会抢先返回 analytical。
    #
    # 于是「表 2 里的 F1 是怎么计算的」/「How is the F1 score in Table 2 calculated?」
    # 这种**显式点名表格 + 指标名**的问句，一旦上面的机制句式模板把 false negative
    # 修好，就会从 extraction/top_k=8/['numeric_table'] 翻成
    # analytical/top_k=16/['section_explanation','numeric_table']——证据来源明明还是
    # 那张表，取证预算却翻了一倍。numeric_table_guard 是 HARD_GATES 里唯一的 1.0
    # 硬闸，而现有 20 条恰好没覆盖这个句型，分数不会报警但链路已经坏了。
    #
    # 护栏刻意收得很窄：只有**显式表格范围**（表 N / Table N / 表格 / caption）
    # 且确实判出 numeric_table 时才抑制 section_explanation。
    # 「表格 3 的实验设置是怎么设计的」没有指标信号，numeric_table_hit 为假，
    # 走不到这里，仍然正常拿到 section_explanation。
    table_first = explicit_table_scope and numeric_table_hit

    # 追加顺序保持与改动前完全一致（section → analysis → numeric），
    # 只把 numeric 的**计算**提前；下游有按 evidence_need 顺序取首项的消费者。
    # 整篇概览请求不是章节级取证需求。它枚举的要素是摘要要覆盖的方面，不是要去
    # 某一节里查的目标；再申领一次 section_explanation 会把 agent gate 的理由从
    # matched_query_type 挤成 matched_evidence_need，取证预算也按分析题翻倍。
    if (
        not table_first
        and not is_overview_query(query)
        and (
            is_section_explanation_query(query)
            or is_mechanism_explanation_query(query)
            or is_method_identity_query(query)
            or is_paper_facet_identity_query(query)
        )
    ):
        evidence_need.append('section_explanation')

    if is_analysis_explanation_query(query):
        evidence_need.append('analysis_explanation')

    if numeric_table_hit:
        evidence_need.append('numeric_table')

    if is_figure_identity_query(query) and not numeric_table_hit:
        evidence_need.append('figure_caption')

    reference_trap_patterns = [
        '参考文献', 'references', 'bibliography', 'citation', 'cite',
        '引用', '被引用', 'related work', '相关工作',
        'arxiv', 'doi', 'url', '链接', 'github',
    ]
    if _contains_terms(query_lower, reference_trap_patterns):
        evidence_need.append('reference_trap')

    reference_meta_patterns = [
        '第一作者', '通讯作者', '作者是谁', '哪些作者', '作者名单', '作者信息',
        '机构', '单位', 'affiliation', 'institution', 'organization',
        'doi', 'arxiv', 'url', '链接', 'github', '邮箱', 'email',
    ]
    english_author_meta = bool(re.search(
        r"(?:\bwho\s+(?:is|are|were)\b.{0,30}\bauthors?\b|"
        r"\bauthors?\s+(?:list|names?|affiliations?)\b)",
        query_lower,
        re.IGNORECASE,
    ))
    if _contains_terms(query_lower, reference_meta_patterns) or english_author_meta:
        evidence_need.append('reference_meta')

    comparison_patterns = [
        '比较', '对比', '区别', '差异', '相比', '优缺点', '异同',
        'compare', 'comparison', 'difference', 'versus', 'vs',
    ]
    multi_aspect_patterns = [
        '分别', '多个方面', '不同方面', '各方面', '多维度', '维度',
        '优势', '劣势', '联系',
    ]
    if (
        _contains_any(query_lower, comparison_patterns)
        and _contains_any(query_lower, multi_aspect_patterns)
    ):
        evidence_need.append('comparison_multi_aspect')

    # 表格硬闸优先：显式点名表格的精确数值问题即使顺带提到"代码"，也不能被
    # 实现题接管取证路径，否则 numeric_table 的 1.0 硬闸会被仓库文件闸挤掉。
    if not table_first and is_code_implementation_query(query):
        evidence_need.append('code_implementation')

    return _dedupe_preserve_order(evidence_need)  # type: ignore[return-value]


def analyze_query_type(query: str) -> QueryType:
    """
    分析查询类型（支持中英文）
    
    Args:
        query: 用户查询文本
        
    Returns:
        查询类型: 'overview' | 'extraction' | 'analytical' | 'specific' | 'inventory'
    """
    if not query:
        return 'specific'
    
    query_lower = query.lower()
    # A complete, typed enumeration has a different correctness contract from
    # semantic retrieval.  Downstream routes use this signal to enumerate the
    # published block index in page order rather than sampling with Top-K.
    if detect_inventory_kind(query):
        return 'inventory'
    evidence_need = analyze_evidence_need(query)
    strong_numeric_extraction = bool(re.search(
        r"(?:多少|数值|分别是多少|最高|最低|第几|百分点|差值|提升多少|下降多少|"
        r"\b(?:what\s+(?:is|are|value|score)|how\s+(?:much|many)|highest|lowest|best\s+result)\b)",
        query_lower,
        re.IGNORECASE,
    ))
    if 'numeric_table' in evidence_need and strong_numeric_extraction:
        return 'extraction'

    # 概览性问题 - 需要更多上下文，但可以使用摘要
    if is_overview_query(query):
        return 'overview'

    if is_structure_map_query(query):
        return 'analytical'

    if (
        'section_explanation' in evidence_need
        or 'analysis_explanation' in evidence_need
        or 'figure_caption' in evidence_need
    ):
        return 'analytical'

    if 'numeric_table' in evidence_need:
        return 'extraction'
    
    # 分析性问题 - 需要适中上下文和细节
    analytical_patterns = [
        '分析', '解释', '说明', '讲解', '为什么', '怎么', '如何',
        '原因', '理由', '比较', '对比', '区别', '差异', '联系', '关系',
        '优缺点', '利弊', '优势', '劣势', '影响', '作用', '原理', '设计', '实现', '细节',
        'analyze', 'explain', 'why', 'how does', 'compare',
        'difference', 'advantage', 'disadvantage', 'impact',
        'differences', 'advantages', 'disadvantages', 'impacts', 'compared',
    ]
    if _contains_terms(query_lower, analytical_patterns):
        return 'analytical'
    
    # 提取性问题 - 需要精确内容，但数量较少
    extraction_patterns = [
        '具体', '详细', '准确', '精确', '数据', '数值', '数字',
        '步骤', '流程', '过程', '公式', '代码', '原文',
        'specific', 'detail', 'exact', 'data', 'number',
        'step', 'procedure', 'formula', 'code', 'algorithm',
        # P1-A: 学术论文数值/结果类问句
        '取得了什么', '达到了', '准确率', '得分', '评分', '指标',
        'accuracy', 'performance', 'result', 'score', 'f1', 'bleu', 'rouge',
        'precision', 'recall', 'metric', 'benchmark',
        '多少', '几个', '第几', '哪一年', '哪一页',
        '参数', '超参', 'parameter', 'hyperparameter',
        '实验结果', '测试集', '验证集', '数据集上',
        # 同上：复数形原来靠裸子串顺带命中，token 边界化后要显式补齐。
        # 但**不能**补 'results'——评测集把「论文的 results 部分放在附录里了吗」
        # 这类问句定义为 specific，加了会把它重新推回 extraction。
        'metrics', 'numbers', 'steps', 'procedures', 'formulas', 'details',
        'parameters', 'hyperparameters',
    ]
    # 这里必须按 token 边界匹配：裸子串会让 'code' 命中 encoder/decoder/recode、
    # 'data' 命中 database/metadata/dataset、'step' 命中 timestep、
    # 'score' 命中 scoreboard、'result' 命中 results，把纯事实型问句误判成提取型。
    if _contains_terms(query_lower, extraction_patterns):
        return 'extraction'
    
    # 默认：具体性问题
    return 'specific'


def get_dynamic_top_k(
    query: str,
    query_type: QueryType = None,
    evidence_need: list[EvidenceNeed] = None,
) -> int:
    """
    根据问题类型动态调整top_k
    
    Args:
        query: 用户查询文本
        query_type: 查询类型（如果已知）
        
    Returns:
        建议的top_k值
    """
    if query_type is None:
        query_type = analyze_query_type(query)
    if evidence_need is None:
        evidence_need = analyze_evidence_need(query)
    if query_type == 'inventory':
        # This is a routing sentinel, not a request for zero relevant vector
        # chunks.  Consumers that support inventory use the block-index cursor
        # API; generic retrieval intentionally receives no Top-K budget.
        return 0
    if query_type == 'analytical' and (
        'section_explanation' in evidence_need
        or 'analysis_explanation' in evidence_need
        or 'figure_caption' in evidence_need
    ):
        return 16
    
    # 根据问题类型返回不同的top_k
    if query_type == 'overview':
        return 15  # 概览问题需要更多上下文
    elif query_type == 'extraction':
        if 'numeric_table' in evidence_need:
            return 8
        if 'reference_trap' in evidence_need:
            return 6
        return 5   # 提取问题需要精确内容，数量较少
    elif query_type == 'analytical':
        return 12  # 分析问题需要适中上下文
    else:
        return 10  # 默认


def get_retrieval_strategy(query: str) -> dict:
    """
    获取完整的检索策略
    
    Args:
        query: 用户查询文本
        
    Returns:
        包含查询类型、top_k、reasoning的字典
    """
    query_type = analyze_query_type(query)
    evidence_need = analyze_evidence_need(query)
    top_k = get_dynamic_top_k(query, query_type, evidence_need)
    
    reasoning_map = {
        'overview': '概览性问题：返回更多分块以提供全面视角',
        'extraction': '提取性问题：返回少量精确分块以确保信息准确',
        'analytical': '分析性问题：返回适中数量分块以平衡细节与长度',
        'specific': '具体性问题：返回标准数量分块',
        'inventory': '完整结构化枚举：按页面顺序读取发布的块索引，不使用语义 Top-K',
    }
    
    return {
        'query_type': query_type,
        'evidence_need': evidence_need,
        'top_k': top_k,
        'reasoning': reasoning_map.get(query_type, reasoning_map['specific'])
    }
