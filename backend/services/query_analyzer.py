"""
查询分析服务 - 分析用户查询类型，用于智能调整检索策略
"""
import re
from typing import Literal, Optional

QueryType = Literal['overview', 'extraction', 'analytical', 'specific']
EvidenceNeed = Literal[
    'numeric_table',
    'reference_trap',
    'section_explanation',
    'reference_meta',
    'comparison_multi_aspect',
    'analysis_explanation',
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

    # 取前 12000 字符（摘要、引言、相关工作、结论区域）
    sample_text = full_text[:12000]

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
        preferred.extend(["method", "design", "implementation", "section", "方法", "设计", "实现", "细节"])
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
    return (
        (any(p in query_lower for p in section_scope_patterns) or numbered_section)
        and any(p in query_lower for p in section_explain_patterns)
    )


def is_overview_query(query: str) -> bool:
    if not query:
        return False
    if is_section_explanation_query(query):
        return False
    query_lower = query.lower()
    overview_patterns = [
        '总结', '概括', '概述', '简述', '大意', '主要内容',
        '讲什么', '关于什么', '整体', '全文',
        '包括什么', '涉及什么',
        'summary', 'summarize', 'overview', 'outline', 'main idea',
        'what is this about', 'what does it cover',
    ]
    if any(pattern in query_lower for pattern in overview_patterns):
        return True
    whole_document_nouns = [
        '这篇论文', '本文', '整篇论文', '这篇文章', '整篇文章', '该文档', '整个文档',
        'this paper', 'the paper', 'this article', 'the document',
    ]
    if _contains_any(query_lower, ['介绍', '背景', 'introduce', 'introduction', 'background']):
        return _contains_any(query_lower, whole_document_nouns)
    if '有哪些' in query_lower:
        return _contains_any(query_lower, ['主要贡献', '核心贡献', '章节', '主题', '关键点', '主要结论'])
    return False


def is_mechanism_explanation_query(query: str) -> bool:
    """识别论文方法机制类解释问题，用于触发 Agent 多轮取证。"""
    if not query:
        return False
    query_lower = query.lower()
    explain_patterns = [
        '解释', '说明', '讲解', '原理', '机制', '实现', '细节', '如何', '怎么',
        '是什么', '做什么', '解决什么', '解决了什么',
        'explain', 'how', 'what is', 'what does', 'mechanism', 'implementation', 'principle',
    ]
    mechanism_scope_patterns = [
        '方法', '模型', '算法', '框架', '架构', '网络', '模块', '组件',
        'method', 'model', 'algorithm', 'framework', 'architecture', 'network', 'module',
    ]
    return (
        any(p in query_lower for p in explain_patterns)
        and any(p in query_lower for p in mechanism_scope_patterns)
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
        'versus', 'vs', 'ablation',
    ]
    # 通用学术论文范围词（不绑定特定论文/领域）
    paper_scope_patterns = [
        '方法', '模型', '算法', '框架', '组件', '模块', '样本', '数据',
        '训练', '推理', '实验', '结果', '性能', '指标',
        'method', 'model', 'algorithm', 'framework', 'module', 'sample', 'data',
        'training', 'inference', 'experiment', 'result', 'performance', 'metric',
    ]
    return (
        any(p in query_lower for p in analysis_patterns)
        and any(p in query_lower for p in paper_scope_patterns)
    )


def analyze_evidence_need(query: str) -> list[EvidenceNeed]:
    """识别学术论文场景下的证据类型需求。"""
    if not query:
        return []

    query_lower = query.lower()
    evidence_need: list[EvidenceNeed] = []
    cost_query = _is_numeric_table_cost_query(query_lower)

    if not is_overview_query(query) and (
        is_section_explanation_query(query) or is_mechanism_explanation_query(query)
    ):
        evidence_need.append('section_explanation')

    if is_analysis_explanation_query(query):
        evidence_need.append('analysis_explanation')

    explicit_table_scope = bool(re.search(
        r"(?:表格|表\s*\d+|第\s*\d+\s*表|表中|\btable(?:s)?(?:\s*\d+)?\b|\bcaption\b)",
        query_lower,
        re.IGNORECASE,
    ))
    numeric_table_metric_scope_patterns = [
        'baseline', '基线', 'metric', '指标', '阈值', 'threshold',
        '比率', 'ratio', 'percentage', 'accuracy', 'acc', 'score',
        'f1', 'bleu', 'rouge', 'precision', 'recall', 'asr', 'lpips',
        'fid', 'map', 'ap50', 'ap75',
    ]
    numeric_value_request_patterns = [
        '数值', '具体数据', '分别是多少', '是多少', '多少', '百分点', '最高', '最低',
        '排名', '第几', '差值', '差距', '提升多少', '下降多少', '高多少', '低多少',
        'what value', 'what score', 'how much', 'how many', 'highest', 'lowest',
        'best result', 'metric value', 'percentage point',
    ]
    metric_signal = _contains_terms(query_lower, numeric_table_metric_scope_patterns)
    quantitative_request = _contains_terms(query_lower, numeric_value_request_patterns)
    fewshot_columns = sum(
        1 for term in ('many', 'medium', 'few')
        if _contains_terms(query_lower, [term])
    ) >= 2
    if (
        (cost_query and quantitative_request)
        or (fewshot_columns and (quantitative_request or explicit_table_scope or metric_signal))
        or (explicit_table_scope and (quantitative_request or metric_signal))
        or (metric_signal and quantitative_request)
    ):
        evidence_need.append('numeric_table')

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

    return _dedupe_preserve_order(evidence_need)  # type: ignore[return-value]


def analyze_query_type(query: str) -> QueryType:
    """
    分析查询类型（支持中英文）
    
    Args:
        query: 用户查询文本
        
    Returns:
        查询类型: 'overview' | 'extraction' | 'analytical' | 'specific'
    """
    if not query:
        return 'specific'
    
    query_lower = query.lower()
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

    if 'section_explanation' in evidence_need or 'analysis_explanation' in evidence_need:
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
    ]
    if any(p in query_lower for p in analytical_patterns):
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
    ]
    if any(p in query_lower for p in extraction_patterns):
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
    if query_type == 'analytical' and (
        'section_explanation' in evidence_need
        or 'analysis_explanation' in evidence_need
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
        'specific': '具体性问题：返回标准数量分块'
    }
    
    return {
        'query_type': query_type,
        'evidence_need': evidence_need,
        'top_k': top_k,
        'reasoning': reasoning_map[query_type]
    }
