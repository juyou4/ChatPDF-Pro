"""论文仓库工具：抽取、授权边界、目录树检索与只读文件读取。

所有网络访问都被 mock。CI 没有出网权限，也没有 GitHub token，
真实打 GitHub 的测试一律不允许出现在这里。
"""

import asyncio

import pytest

from services import retrieval_tools
from services.paper_repo_service import (
    extract_paper_repositories,
    extract_source_symbols,
    rank_repo_tree_paths,
    sanitize_repo_path,
    sanitize_repo_ref,
)
from services.retrieval_tools import DocContext
from services.retrieval_tool_schemas import TOOL_SCHEMAS, get_tool_spec

PAPER_TEXT = (
    "我们的方法在 ImageNet 上取得了最优结果。\n"
    "Code is available at https://github.com/Org-Lab/Proj-Net .\n"
    "训练与评测脚本同样发布在 https://github.com/Org-Lab/Proj-Net/tree/main/scripts 。\n"
    "另一个实现见 https://gitlab.com/group/mirror-proj 。\n"
    "预训练权重在 https://huggingface.co/hf-org/proj-net 上发布。\n"
    "占位地址不应被登记：https://github.com/your-repo/example 与 "
    "https://github.com/features/actions 。\n"
)
GITHUB_REPO_ID = "paper-repo:github:Org-Lab/Proj-Net"
GITLAB_REPO_ID = "paper-repo:gitlab:group/mirror-proj"

TREE_ENTRIES = [
    {"path": "README.md", "type": "blob", "size": 2048},
    {"path": "requirements.txt", "type": "blob", "size": 120},
    {"path": "src", "type": "tree", "size": 0},
    {"path": "src/loss.py", "type": "blob", "size": 4096},
    {"path": "src/model.py", "type": "blob", "size": 8192},
    {"path": "scripts/train.py", "type": "blob", "size": 6144},
    {"path": "assets/teaser.png", "type": "blob", "size": 900_000},
    {"path": "checkpoints/best.pt", "type": "blob", "size": 400_000_000},
]
LOSS_SOURCE = (
    "import torch\n"
    "\n"
    "\n"
    "class FocalLoss(torch.nn.Module):\n"
    "    def forward(self, logits, target):\n"
    "        return logits.mean()\n"
    "\n"
    "\n"
    "def build_loss(cfg):\n"
    "    return FocalLoss()\n"
)


def _make_ctx(text: str = PAPER_TEXT) -> DocContext:
    return DocContext(
        doc_id="doc-paper-repo",
        full_text=text,
        chunks=[text],
        pages=[{"page": 1, "content": text}],
    )


def _tree_result(**overrides):
    result = {
        "status": "completed",
        "error_code": "",
        "error": "",
        "owner": "Org-Lab",
        "repo": "Proj-Net",
        "ref": "main",
        "entries": list(TREE_ENTRIES),
        "entry_count": len(TREE_ENTRIES),
        "truncated": False,
    }
    result.update(overrides)
    return result


def _blob_result(text: str, **overrides):
    result = {
        "status": "completed",
        "error_code": "",
        "error": "",
        "adapter": "github_public",
        "content_kind": "github_file",
        "text": text,
        "char_count": len(text),
        "truncated": False,
        "content_start": 0,
    }
    result.update(overrides)
    return result


def _forbid_network(monkeypatch):
    """Fail loudly if a tool reaches the GitHub adapters when it must not."""
    calls: list[str] = []

    async def _blocked_source(*args, **kwargs):
        calls.append("read_github_public_source")
        raise AssertionError("list_paper_repos 不得访问网络")

    async def _blocked_tree(*args, **kwargs):
        calls.append("read_github_repo_tree")
        raise AssertionError("list_paper_repos 不得访问网络")

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", _blocked_source)
    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", _blocked_tree)
    return calls


def _chunk_source(chunk: str) -> str:
    header = chunk.splitlines()[0]
    for part in header.split("|"):
        key, _, value = part.partition(":")
        if key.strip().endswith("source"):
            return value.strip()
    return ""


# --------------------------------------------------------------------------
# Schema 注册
# --------------------------------------------------------------------------

def test_paper_repo_tool_schemas_registered():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert {"list_paper_repos", "search_paper_repo", "read_paper_repo"} <= names

    list_spec = get_tool_spec("list_paper_repos")
    assert list_spec["source_family"] == "paper_repo"
    assert list_spec["cost_class"] == "local"
    assert list_spec["concurrency_safe"] is True
    assert list_spec["planner_default"] is True

    search_spec = get_tool_spec("search_paper_repo")
    assert search_spec["source_family"] == "paper_repo_tree"
    assert search_spec["concurrency_safe"] is False

    read_spec = get_tool_spec("read_paper_repo")
    assert read_spec["source_family"] == "paper_repo_file"
    assert read_spec["concurrency_safe"] is False
    assert read_spec["timeout_s"] <= 30.0


def test_read_paper_repo_schema_defaults():
    schema = next(
        item["function"]
        for item in TOOL_SCHEMAS
        if item["function"]["name"] == "read_paper_repo"
    )
    properties = schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["repoId"]
    assert properties["path"]["default"] == ""
    assert properties["cursor"]["default"] == 0
    assert properties["maxChars"]["default"] == 6000


# --------------------------------------------------------------------------
# 抽取
# --------------------------------------------------------------------------

def test_extract_paper_repositories_covers_three_hosts():
    repos = extract_paper_repositories(PAPER_TEXT)
    by_id = {repo["repo_id"]: repo for repo in repos}

    assert GITHUB_REPO_ID in by_id
    assert by_id[GITHUB_REPO_ID]["fetch_supported"] is True
    assert by_id[GITHUB_REPO_ID]["url"] == "https://github.com/Org-Lab/Proj-Net"

    assert by_id[GITLAB_REPO_ID]["host"] == "gitlab"
    assert by_id[GITLAB_REPO_ID]["fetch_supported"] is False

    hf = by_id["paper-repo:huggingface:hf-org/proj-net"]
    assert hf["host"] == "huggingface"
    assert hf["fetch_supported"] is False


def test_extract_paper_repositories_dedupes_and_drops_fake_addresses():
    repos = extract_paper_repositories(PAPER_TEXT)
    repo_ids = [repo["repo_id"] for repo in repos]

    assert len(repo_ids) == len(set(repo_ids))
    assert repo_ids.count(GITHUB_REPO_ID) == 1
    assert "paper-repo:github:your-repo/example" not in repo_ids
    assert "paper-repo:github:features/actions" not in repo_ids


def test_paper_repo_available_requires_a_repo_in_the_document():
    assert _make_ctx("这篇论文没有给出任何公开仓库地址。").paper_repo_available() is False
    assert _make_ctx().paper_repo_available() is True
    assert _make_ctx("见 https://gitlab.com/group/mirror-proj").paper_repo_available() is True


def test_resolve_paper_repo_rejects_unknown_ids_and_urls():
    ctx = _make_ctx()

    assert ctx.resolve_paper_repo(GITHUB_REPO_ID)["owner"] == "Org-Lab"
    assert ctx.resolve_paper_repo("paper-repo:github:attacker/evil") is None
    assert ctx.resolve_paper_repo("https://github.com/attacker/evil") is None


def test_sanitize_repo_path_and_ref_reject_escapes():
    assert sanitize_repo_path("src/loss.py") == "src/loss.py"
    assert sanitize_repo_path("/src/loss.py") == "src/loss.py"
    assert sanitize_repo_path("../../etc/passwd") == ""
    assert sanitize_repo_path("https://evil.example/x") == ""
    assert sanitize_repo_ref("main") == "main"
    assert sanitize_repo_ref("a b; rm -rf /") == ""


# --------------------------------------------------------------------------
# list_paper_repos
# --------------------------------------------------------------------------

def test_list_paper_repos_is_offline_without_bootstrap_query(monkeypatch):
    calls = _forbid_network(monkeypatch)
    ctx = _make_ctx()

    result = asyncio.run(retrieval_tools.execute_async_tool("list_paper_repos", {}, ctx))

    assert calls == []
    assert not result.get("error")
    assert result["result_count"] == 1
    assert _chunk_source(result["results"][0]) == "paper_repo"
    assert GITHUB_REPO_ID in result["results"][0]
    assert result["paper_repo_bootstrap"] == {"skipped": "no_bootstrap_query"}


def test_list_paper_repos_reports_no_fetchable_github(monkeypatch):
    _forbid_network(monkeypatch)
    ctx = _make_ctx("镜像仓库见 https://gitlab.com/group/mirror-proj 。")
    ctx.set_paper_repo_bootstrap_query("训练脚本在哪")

    result = asyncio.run(retrieval_tools.execute_async_tool("list_paper_repos", {}, ctx))

    assert result["paper_repo_bootstrap"] == {"skipped": "no_fetchable_github"}


def test_list_paper_repos_without_any_repo_returns_error(monkeypatch):
    _forbid_network(monkeypatch)
    ctx = _make_ctx("这篇论文没有给出任何公开仓库地址。")

    result = asyncio.run(retrieval_tools.execute_async_tool("list_paper_repos", {}, ctx))

    assert result["error_code"] == "paper_repo_not_found"
    assert result["result_count"] == 0


def test_list_paper_repos_bootstrap_reads_readme_and_guided_file(monkeypatch):
    reads: list[str] = []

    async def fake_tree(owner, repo, **kwargs):
        assert (owner, repo) == ("Org-Lab", "Proj-Net")
        return _tree_result()

    async def fake_source(url, **kwargs):
        reads.append(url)
        if url.endswith("/blob/main/src/loss.py"):
            return _blob_result(LOSS_SOURCE)
        return _blob_result("# Proj-Net\n训练入口见 src/loss.py 与 scripts/train.py。\n")

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)

    ctx = _make_ctx()
    ctx.set_paper_repo_bootstrap_query("loss 是怎么实现的")
    result = asyncio.run(retrieval_tools.execute_async_tool("list_paper_repos", {}, ctx))

    bootstrap = result["paper_repo_bootstrap"]
    assert bootstrap["repo_id"] == GITHUB_REPO_ID
    assert bootstrap["readme"] is True
    assert bootstrap["read_paths"] == ["src/loss.py"]
    assert bootstrap["readme_guided_path"] == "src/loss.py"
    assert bootstrap["search_count"] == 1
    assert bootstrap["read_count"] == 2
    assert bootstrap["symbols"][0]["path"] == "src/loss.py"

    sources = [_chunk_source(chunk) for chunk in result["results"]]
    assert sources.count("paper_repo") == 1
    assert "paper_repo_tree" in sources
    assert sources.count("paper_repo_file") == 2
    assert ctx.paper_repo_read_count() == 2
    assert reads[0] == "https://github.com/Org-Lab/Proj-Net"


# --------------------------------------------------------------------------
# search_paper_repo
# --------------------------------------------------------------------------

def test_search_paper_repo_finds_loss_path_and_caches_tree(monkeypatch):
    tree_calls: list[tuple[str, str]] = []

    async def fake_tree(owner, repo, **kwargs):
        tree_calls.append((owner, repo))
        return _tree_result()

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    ctx = _make_ctx()

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "loss", "limit": 5},
            ctx,
        )
    )

    assert not result.get("error")
    assert "src/loss.py" in result["repo_paths"]
    # 二进制与大文件不进代码搜索目标。
    assert "assets/teaser.png" not in result["repo_paths"]
    assert "checkpoints/best.pt" not in result["repo_paths"]
    assert _chunk_source(result["results"][0]) == "paper_repo_tree"
    assert "read_paper_repo" in result["results"][0]

    asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "train"},
            ctx,
        )
    )
    assert tree_calls == [("Org-Lab", "Proj-Net")]


def test_search_paper_repo_maps_chinese_terms_to_paths(monkeypatch):
    async def fake_tree(owner, repo, **kwargs):
        return _tree_result()

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "训练脚本在哪"},
            _make_ctx(),
        )
    )

    assert "scripts/train.py" in result["repo_paths"]


def test_search_paper_repo_rejects_unregistered_and_non_github(monkeypatch):
    _forbid_network(monkeypatch)
    ctx = _make_ctx()

    unknown = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": "paper-repo:github:attacker/evil", "query": "loss"},
            ctx,
        )
    )
    assert unknown["error_code"] == "paper_repo_not_registered"

    gitlab = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITLAB_REPO_ID, "query": "loss"},
            ctx,
        )
    )
    assert gitlab["error_code"] == "paper_repo_fetch_unsupported"


def test_search_paper_repo_budget_is_bounded(monkeypatch):
    async def fake_tree(owner, repo, **kwargs):
        return _tree_result()

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    ctx = _make_ctx()

    for _ in range(3):
        assert not asyncio.run(
            retrieval_tools.execute_async_tool(
                "search_paper_repo",
                {"repoId": GITHUB_REPO_ID, "query": "loss"},
                ctx,
            )
        ).get("error")

    blocked = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "loss"},
            ctx,
        )
    )
    assert blocked["error_code"] == "paper_repo_search_limit_reached"


# --------------------------------------------------------------------------
# read_paper_repo
# --------------------------------------------------------------------------

def test_read_paper_repo_emits_paper_repo_file_evidence(monkeypatch):
    requested: list[str] = []

    async def fake_source(url, **kwargs):
        requested.append(url)
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    ctx = _make_ctx()

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/loss.py"},
            ctx,
        )
    )

    assert not result.get("error")
    chunk = result["results"][0]
    assert _chunk_source(chunk) == "paper_repo_file"
    assert result["chunk_meta"][0]["source"] == "paper_repo_file"
    assert retrieval_tools._UNTRUSTED_REPO_EVIDENCE_NOTICE in chunk
    assert result["repo_path"] == "src/loss.py"
    assert requested == ["https://github.com/Org-Lab/Proj-Net/blob/HEAD/src/loss.py"]

    symbols = {(item["name"], item["kind"]) for item in result["repo_symbols"]}
    assert ("FocalLoss", "class") in symbols
    assert ("build_loss", "def") in symbols
    assert all(item["line"] > 0 and item["end_line"] >= item["line"] for item in result["repo_symbols"])


def test_read_paper_repo_defaults_to_readme(monkeypatch):
    requested: list[str] = []

    async def fake_source(url, **kwargs):
        requested.append(url)
        return _blob_result("# Proj-Net\n公开实现说明。\n")

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID},
            _make_ctx(),
        )
    )

    assert requested == ["https://github.com/Org-Lab/Proj-Net"]
    assert result["repo_path"] == "README"
    assert result["repo_symbols"] == []


def test_read_paper_repo_rejects_unregistered_gitlab_and_bad_paths(monkeypatch):
    _forbid_network(monkeypatch)
    ctx = _make_ctx()

    unknown = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": "paper-repo:github:attacker/evil", "path": "README.md"},
            ctx,
        )
    )
    assert unknown["error_code"] == "paper_repo_not_registered"

    gitlab = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITLAB_REPO_ID, "path": "README.md"},
            ctx,
        )
    )
    assert gitlab["error_code"] == "paper_repo_fetch_unsupported"

    traversal = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "../../etc/passwd"},
            ctx,
        )
    )
    assert traversal["error_code"] == "repo_path_invalid"


def test_read_paper_repo_budget_stops_at_four_reads(monkeypatch):
    async def fake_source(url, **kwargs):
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    ctx = _make_ctx()

    for index in range(4):
        assert not asyncio.run(
            retrieval_tools.execute_async_tool(
                "read_paper_repo",
                {"repoId": GITHUB_REPO_ID, "path": f"src/module_{index}.py"},
                ctx,
            )
        ).get("error")

    blocked = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/loss.py"},
            ctx,
        )
    )
    assert blocked["error_code"] == "paper_repo_read_limit_reached"
    assert ctx.paper_repo_read_count() == 4


def test_read_paper_repo_reports_adapter_failure(monkeypatch):
    async def failing_source(url, **kwargs):
        return {"status": "failed", "error_code": "http_status", "error": "404", "text": ""}

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", failing_source)

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/missing.py"},
            _make_ctx(),
        )
    )

    assert result["error_code"] == "http_status"
    assert result["result_count"] == 0


def test_paper_repo_tools_are_not_callable_synchronously():
    for tool in ("list_paper_repos", "search_paper_repo", "read_paper_repo"):
        result = retrieval_tools.execute_tool(tool, {}, _make_ctx())
        assert result["error_code"] == "paper_repo_requires_async_executor"


# --------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------

def test_rank_repo_tree_paths_falls_back_to_implementation_hints():
    ranked, strategy = rank_repo_tree_paths(TREE_ENTRIES, "完全不相关的词", limit=4)

    assert strategy == "implementation_hint"
    assert {row["path"] for row in ranked} & {"src/model.py", "scripts/train.py"}


def test_extract_source_symbols_only_handles_source_files():
    assert extract_source_symbols("README.md", "# title\n") == []
    assert extract_source_symbols("notes.txt", LOSS_SOURCE) == []

    symbols = extract_source_symbols("src/loss.py", LOSS_SOURCE)
    assert symbols[0] == {"name": "FocalLoss", "kind": "class", "line": 4, "end_line": 8}


@pytest.mark.parametrize("tool", ["search_paper_repo", "read_paper_repo"])
def test_paper_repo_tools_require_repo_id(tool):
    result = asyncio.run(retrieval_tools.execute_async_tool(tool, {"query": "loss"}, _make_ctx()))

    assert result["error_code"] == "repo_id_required"


# --------------------------------------------------------------------------
# Planner 编排接线
# --------------------------------------------------------------------------

def _make_agent():
    from services.retrieval_agent import RetrievalAgent

    return RetrievalAgent(api_key="", model="test-model", provider="openai")


def _rendered_repo_file_chunk(monkeypatch, ctx: DocContext) -> str:
    async def fake_source(url, **kwargs):
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/loss.py"},
            ctx,
        )
    )
    return result["results"][0]


def test_agent_reads_paper_repo_file_source_from_rendered_chunk(monkeypatch):
    ctx = _make_ctx()
    chunk = _rendered_repo_file_chunk(monkeypatch, ctx)
    agent = _make_agent()

    assert agent._extract_tool_chunk_meta(chunk).get("source") == "paper_repo_file"
    assert agent._has_paper_repo_file_evidence([chunk]) is True


def test_agent_file_gate_opens_only_after_a_repo_file_read(monkeypatch):
    ctx = _make_ctx()
    chunk = _rendered_repo_file_chunk(monkeypatch, ctx)
    agent = _make_agent()
    agent._doc_ctx = ctx
    question = "这篇论文的训练脚本在仓库哪"

    assert agent._wants_code_implementation(question) is True
    assert agent._code_implementation_repo_gap(question, [], []) == "missing_paper_repo_file"
    assert agent._code_implementation_repo_gap(question, [chunk], []) == ""


def test_agent_file_gate_stays_closed_without_a_paper_repo():
    agent = _make_agent()
    agent._doc_ctx = _make_ctx("这篇论文没有给出任何公开仓库地址。")

    assert agent._code_implementation_repo_gap("这篇论文的训练脚本在仓库哪", [], []) == ""


def test_agent_injects_list_then_search_operations():
    agent = _make_agent()
    agent._doc_ctx = _make_ctx()
    base = [{"tool": "search_document", "args": {}}]

    first = agent._ensure_paper_repo_gap_operations(base, "训练脚本在哪", [])
    assert first[0]["tool"] == "list_paper_repos"

    after_list = agent._ensure_paper_repo_gap_operations(
        base,
        "训练脚本在哪",
        [{"tool": "list_paper_repos"}],
    )
    assert after_list[0]["tool"] == "search_paper_repo"
    assert after_list[0]["args"]["repoId"] == GITHUB_REPO_ID


def test_agent_initial_bundle_seeds_list_paper_repos():
    agent = _make_agent()
    agent._doc_ctx = _make_ctx()
    agent._active_tool_schemas = [
        schema
        for schema in TOOL_SCHEMAS
        if schema["function"]["name"] in {"search_document", "list_paper_repos", "complete"}
    ]

    bundle = agent._build_initial_search_bundle("这篇论文的训练脚本在仓库哪")

    assert bundle["tool_names"] == ["search_document", "list_paper_repos"]
