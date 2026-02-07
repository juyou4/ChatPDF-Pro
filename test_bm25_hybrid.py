"""
测试BM25索引和混合检索
"""
import sys
sys.path.insert(0, 'backend')

from services.bm25_service import BM25Index, bm25_search, _tokenize
from services.hybrid_search import hybrid_search_merge, reciprocal_rank_fusion

# ============================================================
# 1. 测试分词
# ============================================================
print("=" * 60)
print("1. 分词测试")
print("=" * 60)

test_texts = [
    "自动驾驶系统的管线设计",
    "autonomous driving pipeline",
    "混合中英文 mixed text 测试",
]

for text in test_texts:
    tokens = _tokenize(text)
    print(f"  输入: {text}")
    print(f"  分词: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print()

# ============================================================
# 2. 测试BM25索引
# ============================================================
print("=" * 60)
print("2. BM25索引测试")
print("=" * 60)

chunks = [
    "自动驾驶系统通常包含感知、规划和控制三个主要模块。感知模块负责理解周围环境，规划模块负责决策路径，控制模块负责执行驾驶操作。",
    "深度学习在计算机视觉领域取得了巨大成功，特别是在图像分类、目标检测和语义分割等任务上。卷积神经网络是最常用的架构。",
    "实验结果表明，我们提出的方法在KITTI数据集上达到了95.3%的准确率，比基线方法提高了3.2个百分点。具体的实验数据见表3。",
    "本文提出了一种新的端到端自动驾驶管线，将感知和规划统一到一个神经网络中。该管线使用Transformer架构处理多传感器输入。",
    "相关工作方面，Smith等人在2020年提出了类似的方法，但他们的方法需要额外的标注数据。我们的方法通过自监督学习避免了这一问题。",
]

idx = BM25Index()
idx.build(chunks)

queries = [
    ("自动驾驶管线", "应该匹配chunk 0和3"),
    ("实验数据准确率", "应该匹配chunk 2"),
    ("深度学习卷积神经网络", "应该匹配chunk 1"),
    ("Transformer", "应该匹配chunk 3"),
]

for query, expected in queries:
    results = idx.search(query, top_k=3)
    print(f"\n  查询: {query}")
    print(f"  预期: {expected}")
    for i, r in enumerate(results):
        print(f"  #{i+1} (score={r['score']:.3f}): {r['chunk'][:50]}...")

# ============================================================
# 3. 测试混合检索
# ============================================================
print("\n" + "=" * 60)
print("3. 混合检索测试")
print("=" * 60)

# 模拟向量检索结果（语义相似）
vector_results = [
    {"chunk": chunks[3], "similarity": 0.85, "page": 4},  # 端到端管线
    {"chunk": chunks[0], "similarity": 0.72, "page": 1},  # 感知规划控制
    {"chunk": chunks[1], "similarity": 0.65, "page": 2},  # 深度学习
]

# 模拟BM25结果（关键词匹配）
bm25_results = [
    {"chunk": chunks[0], "score": 5.2, "page": 1},   # 感知规划控制（包含"管线"相关词）
    {"chunk": chunks[3], "score": 4.8, "page": 4},   # 端到端管线
    {"chunk": chunks[4], "score": 2.1, "page": 5},   # 相关工作
]

merged = hybrid_search_merge(vector_results, bm25_results, top_k=5)

print(f"\n  向量结果: {len(vector_results)}条")
print(f"  BM25结果: {len(bm25_results)}条")
print(f"  融合结果: {len(merged)}条")

for i, r in enumerate(merged):
    print(f"  #{i+1} (rrf={r.get('rrf_score', 0):.4f}): {r['chunk'][:50]}...")

# ============================================================
# 4. 测试便捷函数
# ============================================================
print("\n" + "=" * 60)
print("4. 便捷函数测试")
print("=" * 60)

results = bm25_search("test_doc", "自动驾驶管线设计", chunks, top_k=3)
print(f"\n  bm25_search结果: {len(results)}条")
for i, r in enumerate(results):
    print(f"  #{i+1} (score={r['score']:.3f}): {r['chunk'][:50]}...")

# 再次调用应该命中缓存
results2 = bm25_search("test_doc", "实验数据", chunks, top_k=3)
print(f"\n  缓存复用测试: {len(results2)}条")
for i, r in enumerate(results2):
    print(f"  #{i+1} (score={r['score']:.3f}): {r['chunk'][:50]}...")

print("\n" + "=" * 60)
print("✅ 所有测试完成")
print("=" * 60)
