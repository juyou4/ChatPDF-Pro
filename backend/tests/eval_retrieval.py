"""
检索质量评估框架

提供自动化的检索质量评估指标，用于衡量 RAG 优化前后的检索效果。
支持 MRR@K, Recall@K, NDCG@K 等标准信息检索指标。

使用方式：
  python -m tests.eval_retrieval --doc-id <doc_id> --vector-dir <path>

或在代码中：
  from tests.eval_retrieval import RetrievalEvaluator
  evaluator = RetrievalEvaluator(doc_id, vector_store_dir)
  report = evaluator.evaluate(test_cases)
"""

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """单个检索测试用例

    Attributes:
        query: 测试查询
        relevant_keywords: 相关结果应包含的关键词列表（任一命中即算相关）
        relevant_chunk_indices: 已知相关的 chunk 索引列表（可选，精确匹配）
        description: 测试用例描述
    """
    query: str
    relevant_keywords: List[str]
    relevant_chunk_indices: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class EvalResult:
    """单个测试用例的评估结果"""
    query: str
    description: str
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    first_relevant_rank: int  # 第一个相关结果的排名（-1 表示未找到）
    total_relevant_found: int
    total_results: int
    latency_ms: float


@dataclass
class EvalReport:
    """评估报告"""
    results: List[EvalResult]
    avg_mrr: float
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_ndcg_at_10: float
    avg_latency_ms: float
    config_snapshot: Dict  # RAG 配置快照


def _is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """判断 chunk 是否与测试用例相关（关键词匹配）"""
    if not chunk_text or not keywords:
        return False
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain @ K"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 因为 log2(1) = 0
    return dcg


def _ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalized DCG @ K"""
    dcg = _dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """检索质量评估器

    对给定文档执行一组测试查询，计算各项检索质量指标。
    """

    def __init__(
        self,
        doc_id: str,
        vector_store_dir: str,
        pages: List[dict] = None,
        api_key: str = None,
    ):
        self.doc_id = doc_id
        self.vector_store_dir = vector_store_dir
        self.pages = pages or []
        self.api_key = api_key

    def evaluate(
        self,
        test_cases: List[TestCase],
        top_k: int = 10,
        candidate_k: int = 30,
        use_rerank: bool = False,
        reranker_model: str = None,
    ) -> EvalReport:
        """执行评估

        Args:
            test_cases: 测试用例列表
            top_k: 检索返回数量
            candidate_k: 候选数量
            use_rerank: 是否启用重排序
            reranker_model: 重排序模型

        Returns:
            评估报告
        """
        from services.embedding_service import search_document_chunks
        from services.rag_config import RAGConfig

        config = RAGConfig()
        results = []

        for tc in test_cases:
            t0 = time.perf_counter()

            try:
                search_results = search_document_chunks(
                    doc_id=self.doc_id,
                    query=tc.query,
                    vector_store_dir=self.vector_store_dir,
                    pages=self.pages,
                    api_key=self.api_key,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    use_rerank=use_rerank,
                    reranker_model=reranker_model,
                )
            except Exception as e:
                logger.error(f"评估查询 '{tc.query}' 失败: {e}")
                search_results = []

            latency = (time.perf_counter() - t0) * 1000

            # 判断每个结果是否相关
            relevances = []
            for item in search_results:
                chunk_text = item.get("chunk", "")
                is_rel = _is_relevant(chunk_text, tc.relevant_keywords)
                relevances.append(1 if is_rel else 0)

            # MRR: 第一个相关结果的倒数排名
            first_rel_rank = -1
            mrr = 0.0
            for i, rel in enumerate(relevances):
                if rel == 1:
                    first_rel_rank = i + 1
                    mrr = 1.0 / first_rel_rank
                    break

            # Recall@K
            total_relevant_found = sum(relevances)
            # 假设相关文档总数至少等于关键词数或已找到数
            estimated_total_relevant = max(len(tc.relevant_keywords), total_relevant_found, 1)
            recall_5 = sum(relevances[:5]) / estimated_total_relevant
            recall_10 = sum(relevances[:10]) / estimated_total_relevant

            # NDCG@10
            ndcg_10 = _ndcg_at_k(relevances, 10)

            results.append(EvalResult(
                query=tc.query,
                description=tc.description,
                mrr=mrr,
                recall_at_5=min(recall_5, 1.0),
                recall_at_10=min(recall_10, 1.0),
                ndcg_at_10=ndcg_10,
                first_relevant_rank=first_rel_rank,
                total_relevant_found=total_relevant_found,
                total_results=len(search_results),
                latency_ms=round(latency, 1),
            ))

        # 计算平均指标
        n = max(len(results), 1)
        report = EvalReport(
            results=results,
            avg_mrr=sum(r.mrr for r in results) / n,
            avg_recall_at_5=sum(r.recall_at_5 for r in results) / n,
            avg_recall_at_10=sum(r.recall_at_10 for r in results) / n,
            avg_ndcg_at_10=sum(r.ndcg_at_10 for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            config_snapshot={
                "enable_hyde": config.enable_hyde,
                "enable_query_expansion": config.enable_query_expansion,
                "enable_contextual_chunking": config.enable_contextual_chunking,
                "enable_lost_in_middle_reorder": config.enable_lost_in_middle_reorder,
                "enable_semantic_groups": config.enable_semantic_groups,
                "use_rerank": use_rerank,
                "top_k": top_k,
                "candidate_k": candidate_k,
            },
        )

        return report

    @staticmethod
    def print_report(report: EvalReport):
        """格式化打印评估报告"""
        print("\n" + "=" * 60)
        print("  检索质量评估报告")
        print("=" * 60)

        print(f"\n{'指标':<20} {'值':>10}")
        print("-" * 32)
        print(f"{'Avg MRR@10':<20} {report.avg_mrr:>10.4f}")
        print(f"{'Avg Recall@5':<20} {report.avg_recall_at_5:>10.4f}")
        print(f"{'Avg Recall@10':<20} {report.avg_recall_at_10:>10.4f}")
        print(f"{'Avg NDCG@10':<20} {report.avg_ndcg_at_10:>10.4f}")
        print(f"{'Avg Latency (ms)':<20} {report.avg_latency_ms:>10.1f}")

        print(f"\n配置: {report.config_snapshot}")

        print(f"\n{'#':<4} {'Query':<30} {'MRR':>6} {'R@5':>6} {'R@10':>6} {'1st':>5} {'ms':>8}")
        print("-" * 70)
        for i, r in enumerate(report.results):
            q = r.query[:28] + ".." if len(r.query) > 30 else r.query
            print(
                f"{i+1:<4} {q:<30} {r.mrr:>6.3f} "
                f"{r.recall_at_5:>6.3f} {r.recall_at_10:>6.3f} "
                f"{r.first_relevant_rank:>5} {r.latency_ms:>8.1f}"
            )
        print("=" * 60)


# ============================================================
# 内置示例测试用例（通用型，适用于大多数学术论文）
# ============================================================
SAMPLE_TEST_CASES = [
    TestCase(
        query="这篇论文的主要贡献是什么",
        relevant_keywords=["贡献", "contribution", "提出", "propose", "方法", "method"],
        description="概览性问题",
    ),
    TestCase(
        query="实验结果如何",
        relevant_keywords=["实验", "experiment", "结果", "result", "准确率", "accuracy", "性能", "performance"],
        description="提取性问题",
    ),
    TestCase(
        query="使用了什么数据集",
        relevant_keywords=["数据集", "dataset", "数据", "data", "训练", "train", "测试", "test"],
        description="具体性问题",
    ),
    TestCase(
        query="相关工作有哪些",
        relevant_keywords=["相关", "related", "工作", "work", "研究", "study", "先前", "previous"],
        description="概览性问题",
    ),
    TestCase(
        query="模型架构是怎样的",
        relevant_keywords=["模型", "model", "架构", "architecture", "网络", "network", "层", "layer"],
        description="分析性问题",
    ),
]


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--doc-id", required=True, help="文档 ID")
    parser.add_argument("--vector-dir", default="data/vector_store", help="向量存储目录")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        doc_id=args.doc_id,
        vector_store_dir=args.vector_dir,
    )
    report = evaluator.evaluate(
        SAMPLE_TEST_CASES,
        top_k=args.top_k,
        use_rerank=args.rerank,
    )
    RetrievalEvaluator.print_report(report)
