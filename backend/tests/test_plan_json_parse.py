"""planner JSON 解析鲁棒性测试（P2：think 块剥离 + 保守修复兜底）。"""

from services.retrieval_agent import RetrievalAgent


def _make_agent() -> RetrievalAgent:
    return RetrievalAgent(api_key="", model="test-model", provider="openai")


_VALID_PLAN = '{"operations": [{"tool": "search_document", "args": {"query": "方法"}}], "final": false, "taskStatus": {"completed": [], "current": "检索", "pending": []}}'


def test_parse_plan_json_plain():
    agent = _make_agent()
    plan = agent._parse_plan_json(_VALID_PLAN)
    assert plan is not None
    assert plan["operations"][0]["tool"] == "search_document"


def test_parse_plan_json_strips_think_block():
    agent = _make_agent()
    content = "<think>我需要检索 {一些带括号的思考} 再输出</think>\n" + _VALID_PLAN
    plan = agent._parse_plan_json(content)
    assert plan is not None
    assert plan["operations"][0]["tool"] == "search_document"


def test_parse_plan_json_repairs_trailing_comma_and_python_literals():
    agent = _make_agent()
    content = (
        '{"operations": [{"tool": "grep", "args": {"query": "mIoU"}},], '
        '"final": False, "taskStatus": {"completed": [], "current": "", "pending": [],}}'
    )
    plan = agent._parse_plan_json(content)
    assert plan is not None
    assert plan["operations"][0]["tool"] == "grep"
    assert agent.diagnostics.get("planner_json_repaired") is True


def test_parse_plan_json_repairs_fullwidth_quotes():
    agent = _make_agent()
    content = '{“operations”: [], “final”: true, “taskStatus”: {“completed”: [], “current”: “”, “pending”: []}}'
    plan = agent._parse_plan_json(content)
    assert plan is not None
    assert plan.get("final") is True


def test_parse_plan_json_markdown_fence_still_works():
    agent = _make_agent()
    content = "说明文字\n```json\n" + _VALID_PLAN + "\n```\n尾注"
    plan = agent._parse_plan_json(content)
    assert plan is not None


def test_parse_plan_json_garbage_returns_none():
    agent = _make_agent()
    assert agent._parse_plan_json("完全不是 JSON 的输出") is None
    assert agent._parse_plan_json("") is None
