"""
查询分析服务 - 分析用户查询类型，用于智能调整检索策略
"""
from typing import Literal

QueryType = Literal['overview', 'extraction', 'analytical', 'specific']


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
    ]
    if any(p in query_lower for p in extraction_patterns):
        return 'extraction'
    
    # 默认：具体性问题
    return 'specific'


def get_dynamic_top_k(query: str, query_type: QueryType = None) -> int:
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
    if query_type == 'analytical' and is_section_explanation_query(query):
        return 16
    
    # 根据问题类型返回不同的top_k
    if query_type == 'overview':
        return 15  # 概览问题需要更多上下文
    elif query_type == 'extraction':
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
    top_k = get_dynamic_top_k(query, query_type)
    
    reasoning_map = {
        'overview': '概览性问题：返回更多分块以提供全面视角',
        'extraction': '提取性问题：返回少量精确分块以确保信息准确',
        'analytical': '分析性问题：返回适中数量分块以平衡细节与长度',
        'specific': '具体性问题：返回标准数量分块'
    }
    
    return {
        'query_type': query_type,
        'top_k': top_k,
        'reasoning': reasoning_map[query_type]
    }
