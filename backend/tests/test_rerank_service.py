import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.rerank_service import rerank_service


def test_rerank_service_without_candidates():
    assert rerank_service.rerank("query", []) == []


def test_rerank_service_with_mock_model(monkeypatch):
    # Mock model.predict to avoid loading actual CrossEncoder
    def fake_get_model(name):
        m = types.SimpleNamespace()
        m.predict = lambda pairs: [i for i, _ in enumerate(pairs)]
        return m

    monkeypatch.setattr(rerank_service, "_get_model", fake_get_model)

    candidates = [
        {"chunk": "a"},
        {"chunk": "b"},
        {"chunk": "c"},
    ]
    ranked = rerank_service.rerank("q", candidates, model_name="mock")
    # Highest score should be last candidate (score=2)
    assert ranked[0]["chunk"] == "c"
    assert ranked[0]["reranked"] is True
