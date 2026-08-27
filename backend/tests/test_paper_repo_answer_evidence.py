"""论文仓库代码进入作答的接线与对照讲解合同。

覆盖的断点：仓库文件其实被读到了，但 ``paper_repo_context`` 停在
``retrieval_complete``，既没进 ``retrieval_meta``，也没有对应的证据段，
于是在进回答模型之前被丢掉，回答只能按论文正文猜实现位置。

所有网络访问都被 mock，测试不得触达真实 GitHub 或任何模型服务。
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.chat_routes as chat_routes
from services import agent_retrieval_service
from services.academic_answer_contract import build_compact_academic_contract_prompt

REPO_EVIDENCE = (
    "[论文仓库文件证据]\n"
    "仓库: github:Org-Lab/Proj-Net\n"
    "路径: src/loss.py\n"
    "class FocalLoss(torch.nn.Module):\n"
    "    def forward(self, logits, target):\n"
    "        return logits.mean()\n"
)
DOC_EVIDENCE = "[1] 我们使用 focal loss 抑制易分类样本的梯度贡献。"


# --------------------------------------------------------------------------
# agent_retrieval_service：retrieval_complete → retrieval_meta
# --------------------------------------------------------------------------

class _FakeAgent:
    """只产出一个 retrieval_complete 事件的 Planner 替身。"""

    def __init__(self, *args, **kwargs):
        self.diagnostics = {}
        self._partial_state = {
            "paper_repo_context_parts": [REPO_EVIDENCE],
            "web_search_context_parts": [],
        }

    def snapshot_partial_diagnostics(self, fallback_reason=""):
        return {"retrieval": {}, "context_assembly": {}}

    async def run(self, **kwargs):
        yield {
            "type": "retrieval_complete",
            "context": "",
            "detail": [],
            "search_history": [],
            "task_status": {},
            "diagnostics": {},
            "retrieval_diagnostics": {},
            "web_search_sources": [],
            "web_search_context": "",
            "web_search_reads": [],
            "paper_repo_context": REPO_EVIDENCE,
        }


class _FakeRequest:
    doc_id = "doc-1"
    question = "这个方法怎么实现的"
    selected_text = ""
    embedding_api_key = ""
    use_rerank = False
    reranker_model = ""
    rerank_provider = ""
    rerank_api_key = ""
    rerank_endpoint = ""
    web_search_enabled = False


def _dependencies() -> agent_retrieval_service.AgentRetrievalDependencies:
    return agent_retrieval_service.AgentRetrievalDependencies(
        get_cheap_model_params=lambda request: ("m", "openai", ""),
        build_agent_doc_context=lambda *args, **kwargs: object(),
        merge_retrieval_meta=lambda base, extra: {**(base or {}), **(extra or {})},
        annotate_agent_gate=lambda gate, **kwargs: dict(gate or {}),
        resolve_citation_candidate_limit=lambda **kwargs: 8,
        build_numbered_context_and_citations=lambda *args, **kwargs: ("", []),
        generate_page_level_citations=lambda *args, **kwargs: [],
        build_agent_detail_citations=lambda *args, **kwargs: [],
        primary_key_for_target=lambda request, provider, endpoint: "",
    )


def test_agent_result_paper_repo_context_reaches_retrieval_meta(monkeypatch):
    monkeypatch.setattr(agent_retrieval_service, "RetrievalAgent", _FakeAgent)
    monkeypatch.setattr(
        agent_retrieval_service,
        "_resolve_effective_web_search_mode",
        lambda request, question: "off",
    )
    meta: dict = {}

    _context, retrieval_meta = asyncio.run(
        agent_retrieval_service.run_agent_retrieval_for_context(
            request=_FakeRequest(),
            doc={"filename": "demo.pdf", "data": {"pages": [], "full_text": "正文"}},
            search_query="这个方法怎么实现的",
            query_type="analytical",
            agent_gate={},
            retrieval_meta=meta,
            deps=_dependencies(),
        )
    )

    assert retrieval_meta["paper_repo_context"] == REPO_EVIDENCE.strip()


def test_degraded_agent_result_keeps_paper_repo_context():
    """超时/异常降级同样不能丢掉已经读到的仓库正文。"""
    result = agent_retrieval_service._build_degraded_agent_result(
        _FakeAgent(),
        degraded_to="agent_total_timeout",
    )

    assert result["paper_repo_context"] == REPO_EVIDENCE


# --------------------------------------------------------------------------
# 独立证据槽
# --------------------------------------------------------------------------

def test_paper_repo_context_gets_its_own_evidence_section():
    message = chat_routes._build_untrusted_evidence_message(
        document_context=DOC_EVIDENCE,
        paper_repo_context=REPO_EVIDENCE,
    )

    assert "<<<BEGIN_PAPER_REPO_EVIDENCE>>>" in message
    assert "<<<END_PAPER_REPO_EVIDENCE>>>" in message

    document_block = message.split("<<<BEGIN_DOCUMENT_EVIDENCE>>>")[1].split(
        "<<<END_DOCUMENT_EVIDENCE>>>"
    )[0]
    # 仓库正文绝不能混进文档 [n] 主链所在的证据段。
    assert "FocalLoss" not in document_block
    assert DOC_EVIDENCE in document_block


def test_evidence_message_omits_repo_section_without_repo_context():
    message = chat_routes._build_untrusted_evidence_message(document_context=DOC_EVIDENCE)

    assert "PAPER_REPO_EVIDENCE" not in message


def test_chat_messages_carry_repo_evidence_to_the_model():
    messages = chat_routes._build_chat_messages(
        "system",
        [],
        "这个方法怎么实现的",
        document_context=DOC_EVIDENCE,
        paper_repo_context=REPO_EVIDENCE,
    )

    evidence = "\n".join(str(message.get("content") or "") for message in messages)
    assert "BEGIN_PAPER_REPO_EVIDENCE" in evidence
    assert "FocalLoss" in evidence


def test_paper_repo_context_never_reaches_the_public_payload():
    assert "paper_repo_context" in chat_routes._PUBLIC_RETRIEVAL_META_DENY_KEYS


# --------------------------------------------------------------------------
# 对照讲解合同
# --------------------------------------------------------------------------

@pytest.mark.parametrize("agent_mode", [True, False])
def test_evidence_contract_demands_a_paper_to_code_walkthrough(agent_mode):
    prompt = chat_routes._append_answer_evidence_contract(
        "base",
        web_search_sources=[],
        web_search_context="",
        agent_mode=agent_mode,
        paper_repo_context=REPO_EVIDENCE,
    )

    assert "对照" in prompt
    assert "PAPER_REPO_EVIDENCE" in prompt
    # 只报位置必须被显式禁止，否则模型会拿"实现位于 src/loss.py"当答案。
    assert "禁止" in prompt and "实现位于" in prompt
    assert "文件路径" in prompt and "符号名" in prompt


def test_evidence_contract_stays_document_only_without_repo_evidence():
    prompt = chat_routes._append_answer_evidence_contract(
        "base",
        web_search_sources=[],
        web_search_context="",
        agent_mode=True,
    )

    assert "PAPER_REPO_EVIDENCE" not in prompt
    assert "论文仓库代码证据" not in prompt


def test_evidence_scope_covers_document_web_and_repo_together():
    scope, insufficient = chat_routes._evidence_scope_phrases(
        allow_web_evidence=True,
        allow_repo_evidence=True,
    )

    assert "文档" in scope and "联网" in scope and "仓库代码" in scope
    assert "仓库代码" in insufficient


def test_compact_contract_splits_repo_citations_from_document_numbers():
    prompt = build_compact_academic_contract_prompt(
        agent_mode=True,
        allow_repo_evidence=True,
    )

    assert "论文仓库代码证据" in prompt
    assert "不套用文档 [n] 编号" in prompt
    assert "禁止只回答" in prompt


def test_agent_focus_prompt_requires_the_walkthrough_when_repo_evidence_exists():
    prompt = chat_routes._build_agent_answer_focus_prompt(
        "这个方法怎么实现的",
        query_type="analytical",
        evidence_need=["section_explanation"],
        paper_repo_context=REPO_EVIDENCE,
    )

    assert "对照" in prompt
    assert "PAPER_REPO_EVIDENCE" in prompt
    assert "不算完成回答" in prompt


def test_agent_focus_prompt_reacts_to_code_implementation_need_alone():
    prompt = chat_routes._build_agent_answer_focus_prompt(
        "训练脚本在仓库哪",
        query_type="analytical",
        evidence_need=["code_implementation"],
    )

    assert "对照" in prompt


def test_agent_focus_prompt_stays_empty_for_plain_questions():
    prompt = chat_routes._build_agent_answer_focus_prompt(
        "这篇论文的动机是什么",
        query_type="analytical",
        evidence_need=["section_explanation"],
    )

    assert prompt == ""
