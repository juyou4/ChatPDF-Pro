"""引用质量评估脚手架单元测试"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.eval_citation_quality import evaluate_case


def test_evaluate_case_basic_metrics():
    answer = "方法通过全局光照建模提升鲁棒性[1]。实验显示ASR下降[2]。"
    citations = [
        {"ref": 1, "highlight_text": "全局 光照 建模 提升 鲁棒性"},
        {"ref": 2, "highlight_text": "实验 结果 ASR 下降"},
    ]
    m = evaluate_case(answer, citations)
    assert m.citation_count == 2
    assert m.citation_precision_at_1 > 0
    assert m.unsupported_citation_rate < 1


def test_evaluate_case_unsupported_when_text_mismatch():
    answer = "这句和来源完全不相关[1]。"
    citations = [{"ref": 1, "highlight_text": "东京 旅游 酒店"}]
    m = evaluate_case(answer, citations, unsupported_threshold=0.05)
    assert m.citation_count == 1
    assert m.unsupported_citation_rate == 1.0
