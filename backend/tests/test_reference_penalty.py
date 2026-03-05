"""参考文献型片段降权测试"""
import sys
import os

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embedding_service import _is_reference_like_text, _phrase_boost


def test_reference_like_text_detected():
    text = (
        "[1] K. He, X. Zhang, S. Ren, J. Sun. 2016. Deep residual learning.\n"
        "[2] A. Vaswani et al. 2017. Attention is all you need. doi:10.48550/arXiv.1706.03762"
    )
    assert _is_reference_like_text(text) is True


def test_non_reference_text_not_detected():
    text = "本文提出了一个新的损失函数，用于平衡分类准确率与目标物体自然性。"
    assert _is_reference_like_text(text) is False


def test_phrase_boost_penalizes_reference_chunk_for_non_reference_query():
    ref_chunk = {
        "chunk": "[1] Author A. 2021. Method X. [2] Author B. 2022. Method Y.",
        "similarity": 0.8,
        "similarity_percent": 80.0,
    }
    normal_chunk = {
        "chunk": "该方法的总损失由分类损失和物体损失组成。",
        "similarity": 0.7,
        "similarity_percent": 70.0,
    }
    ranked = _phrase_boost([ref_chunk, normal_chunk], "这个损失函数怎么定义")

    assert ranked[0]["chunk"] == normal_chunk["chunk"]
    assert any(item.get("reference_like") for item in ranked)


def test_phrase_boost_keeps_reference_chunk_for_reference_query():
    ref_chunk = {
        "chunk": "[1] Author A. 2021. Method X. [2] Author B. 2022. Method Y.",
        "similarity": 0.8,
        "similarity_percent": 80.0,
    }
    ranked = _phrase_boost([ref_chunk], "这篇论文的参考文献有哪些")

    assert ranked[0]["chunk"] == ref_chunk["chunk"]
    assert not ranked[0].get("reference_like")
