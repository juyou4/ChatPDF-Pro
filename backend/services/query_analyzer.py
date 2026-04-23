"""
查询分析服务 - 分析用户查询类型，用于智能调整检索策略
"""
from typing import Literal

QueryType = Literal['overview', 'extraction', 'analytical', 'specific']
EvidenceNeed = Literal[
    'numeric_table',
    'reference_trap',
    'section_explanation',
    'reference_meta',
    'comparison_multi_aspect',
]


def _contains_any(query_lower: str, patterns: list[str]) -> bool:
    return any(pattern in query_lower for pattern in patterns)


def _is_numeric_table_cost_query(query_lower: str) -> bool:
    cost_patterns = [
        '额外开销', '计算开销', '推理开销', '推理时间', '训练时间', '耗时',
        'cost', 'costs', 'overhead', 'extra overhead', 'computation cost',
        'flops', 'inference time', 'inference-time', 'training time', 'training-time',
        'latency', 'runtime',
    ]
    return _contains_any(query_lower, cost_patterns)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def is_section_explanation_query(query: str) -> bool:
    if not query:
        return False
    query_lower = query.lower()
    section_scope_patterns = [
        '部分', '章节', 'section', 'chapter', 'method', '方法',
    ]
    section_explain_patterns = [
        '讲解', '解释', '说明', '原理', '设计', '实现', '细节', '展开',
    ]
    return (
        any(p in query_lower for p in section_scope_patterns)
        and any(p in query_lower for p in section_explain_patterns)
    )


def analyze_evidence_need(query: str) -> list[EvidenceNeed]:
    """识别学术论文场景下的证据类型需求。"""
    if not query:
        return []

    query_lower = query.lower()
    evidence_need: list[EvidenceNeed] = []
    cost_query = _is_numeric_table_cost_query(query_lower)

    if is_section_explanation_query(query):
        evidence_need.append('section_explanation')

    numeric_table_scope_patterns = [
        '表', '表格', 'table', 'tables', 'caption',
        '数据集', 'dataset', 'baseline', '基线', 'metric', '指标',
    ]
    numeric_table_value_patterns = [
        'accuracy', 'acc', 'score', 'f1', 'bleu', 'rouge',
        '提升', '下降', '差值', '差距', '相比', '对比', '分别', '多少',
        'many', 'medium', 'few', '数值', '数字',
    ]
    if (
        cost_query
        or
        _contains_any(query_lower, ['many', 'medium', 'few'])
        or (
            _contains_any(query_lower, numeric_table_scope_patterns)
            and _contains_any(query_lower, numeric_table_value_patterns)
        )
    ):
        evidence_need.append('numeric_table')

    reference_trap_patterns = [
        '参考文献', 'references', 'bibliography', 'citation', 'cite',
        '引用', '被引用', 'related work', '相关工作',
        '作者', '第一作者', '通讯作者', 'author', 'authors',
        'arxiv', 'doi', 'url', '链接', 'github',
    ]
    if _contains_any(query_lower, reference_trap_patterns):
        evidence_need.append('reference_trap')

    reference_meta_patterns = [
        '第一作者', '通讯作者', '作者', 'author', 'authors',
        '机构', '单位', 'affiliation', 'institution', 'organization',
        'doi', 'arxiv', 'url', '链接', 'github', '邮箱', 'email',
    ]
    if _contains_any(query_lower, reference_meta_patterns):
        evidence_need.append('reference_meta')

    comparison_patterns = [
        '比较', '对比', '区别', '差异', '相比', '优缺点', '异同',
        'compare', 'comparison', 'difference', 'versus', 'vs',
    ]
    multi_aspect_patterns = [
        '分别', '多个方面', '不同方面', '各方面', '多维度', '维度',
        'many', 'medium', 'few', '优势', '劣势', '联系',
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
    if 'numeric_table' in evidence_need:
        return 'extraction'
    if is_section_explanation_query(query):
        return 'analytical'
    
    # 概览性问题 - 需要更多上下文，但可以使用摘要
    overview_patterns = [
        '总结', '概括', '概述', '简述', '大意', '主要内容', 
        '讲什么', '关于什么', '介绍', '背景', '整体', '全文',
        '有哪些', '包括什么', '涉及什么',
        'summary', 'summarize', 'overview', 'outline', 'main idea',
        'what is this about', 'what does it cover',
    ]
    if any(p in query_lower for p in overview_patterns):
        return 'overview'
    
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
    if query_type == 'analytical' and is_section_explanation_query(query):
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
