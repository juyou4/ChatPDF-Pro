"""
测试查询分析器
"""
import sys
sys.path.insert(0, 'backend')

from services.query_analyzer import analyze_query_type, get_dynamic_top_k, get_retrieval_strategy

# 测试用例
test_queries = [
    # 概览性问题
    ("总结一下这篇文档", "overview"),
    ("这篇论文主要讲什么", "overview"),
    ("概括一下主要内容", "overview"),
    
    # 提取性问题
    ("具体的实验数据是什么", "extraction"),
    ("详细的步骤是什么", "extraction"),
    ("公式是什么", "extraction"),
    
    # 分析性问题
    ("分析一下这个方法的优缺点", "analytical"),
    ("为什么要这样做", "analytical"),
    ("比较一下两种方法的区别", "analytical"),
    
    # 具体性问题
    ("实验的管线是什么", "specific"),
    ("作者是谁", "specific"),
]

print("=" * 80)
print("查询分析器测试")
print("=" * 80)

for query, expected_type in test_queries:
    strategy = get_retrieval_strategy(query)
    query_type = strategy['query_type']
    top_k = strategy['top_k']
    reasoning = strategy['reasoning']
    
    status = "✓" if query_type == expected_type else "✗"
    
    print(f"\n{status} 查询: {query}")
    print(f"  预期类型: {expected_type}")
    print(f"  实际类型: {query_type}")
    print(f"  top_k: {top_k}")
    print(f"  原因: {reasoning}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
