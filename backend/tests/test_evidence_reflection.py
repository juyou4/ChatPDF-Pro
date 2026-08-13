"""检索侧 Decision Gate 反思闸门测试（P0-2）。"""

import asyncio

from services import retrieval_agent as agent_module
from services.retrieval_agent import RetrievalAgent


def _make_agent() -> RetrievalAgent:
    agent = RetrievalAgent(api_key="", model="test-model", provider="openai")
    agent.reflection_enabled = True
    agent.reflection_timeout = 2.0
    return agent


def test_parse_reflection_json_normal():
    parsed = RetrievalAgent._parse_reflection_json(
        '前置说明 {"can_answer": false, "missing_gaps": ["表3 的 mIoU 数值", "  ", "公式 4 定义"], "reason": "缺表格"} 尾巴'
    )
    assert parsed is not None
    assert parsed["can_answer"] is False
    assert parsed["missing_gaps"] == ["表3 的 mIoU 数值", "公式 4 定义"]
    assert parsed["reason"] == "缺表格"


def test_parse_reflection_json_caps_gaps_at_three():
    parsed = RetrievalAgent._parse_reflection_json(
        '{"can_answer": false, "missing_gaps": ["a", "b", "c", "d"], "reason": ""}'
    )
    assert parsed is not None
    assert parsed["missing_gaps"] == ["a", "b", "c"]


def test_parse_reflection_json_invalid():
    assert RetrievalAgent._parse_reflection_json("") is None
    assert RetrievalAgent._parse_reflection_json("no json here") is None
    assert RetrievalAgent._parse_reflection_json("{broken json") is None


def test_reflect_on_evidence_gap_success(monkeypatch):
    agent = _make_agent()

    async def fake_call_ai_api(**kwargs):
        assert kwargs["temperature"] == 0.0
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"can_answer": false, "missing_gaps": ["实验设置章节"], "reason": "缺实验细节"}'
                    }
                }
            ]
        }

    monkeypatch.setattr(agent_module, "call_ai_api", fake_call_ai_api)
    result = asyncio.run(
        agent._reflect_on_evidence_gap("方法效果如何", ["【证据】方法 A 提升 2%"], [])
    )
    assert result is not None
    assert result["missing_gaps"] == ["实验设置章节"]
    record = agent.diagnostics["evidence_reflection"]
    assert record["ok"] is True
    assert record["missing_gaps"] == ["实验设置章节"]


def test_reflect_on_evidence_gap_timeout_returns_none(monkeypatch):
    agent = _make_agent()
    agent.reflection_timeout = 1.0

    async def slow_call_ai_api(**kwargs):
        await asyncio.sleep(5)
        return {}

    monkeypatch.setattr(agent_module, "call_ai_api", slow_call_ai_api)
    result = asyncio.run(agent._reflect_on_evidence_gap("q", ["【证据】内容"], []))
    assert result is None
    record = agent.diagnostics["evidence_reflection"]
    assert record["ok"] is False
    assert "reflection_timeout" in record["error"]


def test_reflect_on_evidence_gap_parse_failure_returns_none(monkeypatch):
    agent = _make_agent()

    async def bad_call_ai_api(**kwargs):
        return {"choices": [{"message": {"content": "我觉得证据不够，但我不输出 JSON"}}]}

    monkeypatch.setattr(agent_module, "call_ai_api", bad_call_ai_api)
    result = asyncio.run(agent._reflect_on_evidence_gap("q", ["【证据】内容"], []))
    assert result is None
    assert agent.diagnostics["evidence_reflection"]["error"] == "reflection_parse_failed"


def test_reflect_on_evidence_gap_skips_without_document_evidence(monkeypatch):
    agent = _make_agent()

    async def should_not_be_called(**kwargs):
        raise AssertionError("不应触发 LLM 调用")

    monkeypatch.setattr(agent_module, "call_ai_api", should_not_be_called)
    assert asyncio.run(agent._reflect_on_evidence_gap("q", [], [])) is None
