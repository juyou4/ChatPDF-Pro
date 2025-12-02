import sys
from pathlib import Path
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import rerank_api_service


def test_cohere_rerank_success(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        class Resp:
            status_code = 200
            def raise_for_status(self): return None
            def json(self):
                return {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]}
        return Resp()

    monkeypatch.setattr(rerank_api_service.httpx, "post", fake_post)
    scores = rerank_api_service.cohere_rerank("q", ["a", "b"], "m", "k")
    assert calls["n"] == 1
    assert scores[0][0] == 0 and scores[0][1] == 0.9


def test_jina_rerank_success(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        class Resp:
            status_code = 200
            def raise_for_status(self): return None
            def json(self):
                return {"results": [{"index": 1, "score": 0.7}, {"index": 0, "score": 0.2}]}
        return Resp()

    monkeypatch.setattr(rerank_api_service.httpx, "post", fake_post)
    scores = rerank_api_service.jina_rerank("q", ["a", "b"], "m", "k")
    assert calls["n"] == 1
    assert scores[0][0] == 1 and scores[0][1] == 0.7
