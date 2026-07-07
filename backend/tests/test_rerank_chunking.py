import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.rerank_service import RerankService


def test_rerank_chunks_long_candidate_and_aggregates_best_parent(monkeypatch):
    service = RerankService()
    monkeypatch.setattr(RerankService, "RERANK_CHUNK_CHAR_LIMIT", 120)
    monkeypatch.setattr(RerankService, "RERANK_CHUNK_OVERLAP", 10)
    monkeypatch.setattr(RerankService, "RERANK_MAX_CHUNKS_PER_CANDIDATE", 8)

    def fake_local(query, candidates, model_name):
        for item in candidates:
            text = item.get("rerank_text") or item.get("chunk") or ""
            item["rerank_score"] = 10.0 if "target-value" in text else 1.0
            item["reranked"] = True
        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)

    monkeypatch.setattr(service, "_rerank_local", fake_local)
    long_text = ("noise " * 80) + " target-value " + ("tail " * 80)
    results = service.rerank(
        "target-value",
        [
            {"chunk": "short irrelevant", "similarity": 0.7},
            {"chunk": long_text, "similarity": 0.1},
        ],
        provider="local",
    )

    assert results[0]["chunk"] == long_text
    assert results[0]["rerank_chunked"] is True
    assert results[0]["rerank_score"] == 10.0
    assert results[0]["rerank_chunk_count"] > 1
