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


def test_extract_drops_huggingface_site_pages_but_keeps_model_cards():
    repos = extract_paper_repositories(
        "See https://huggingface.co/docs/transformers for the library docs.\n"
        "Weights: https://huggingface.co/hf-org/proj-net\n"
        "Dataset: https://huggingface.co/datasets/hf-org/proj-data\n"
    )
    repo_ids = {repo["repo_id"] for repo in repos}

    assert "paper-repo:huggingface:docs/transformers" not in repo_ids
    assert "paper-repo:huggingface:hf-org/proj-net" in repo_ids
    assert "paper-repo:huggingface:datasets/hf-org/proj-data" in repo_ids


def test_extract_keeps_short_generic_owner_names():
    """占位符过滤不能吃掉 org/user/name/test 这类真实但普通的仓库名。"""
    repos = extract_paper_repositories(
        "Our code is available at https://github.com/org/proj .\n"
        "Baseline: https://github.com/user/test-kit\n"
    )
    repo_ids = {repo["repo_id"] for repo in repos}

    assert "paper-repo:github:org/proj" in repo_ids
    assert "paper-repo:github:user/test-kit" in repo_ids


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

    async def fake_source(url, **kwargs):
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
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


def test_search_paper_repo_auto_reads_top_source_paths(monkeypatch):
    """只返回路径列表会让 Planner 停在"实现在 src/loss.py"，必须带上正文。"""
    read_urls: list[str] = []

    async def fake_tree(owner, repo, **kwargs):
        return _tree_result()

    async def fake_source(url, **kwargs):
        read_urls.append(url)
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    ctx = _make_ctx()

    result = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "损失函数和模型结构怎么实现的", "limit": 5},
            ctx,
        )
    )

    assert result["repo_paths"][:2] == ["src/loss.py", "src/model.py"]
    sources = [_chunk_source(chunk) for chunk in result["results"]]
    assert sources[0] == "paper_repo_tree"
    assert sources.count("paper_repo_file") == 2
    assert result["repo_auto_read_paths"] == result["repo_paths"][:2]
    # 自动读取从同一份读预算里扣，不是额外配额。
    assert ctx.paper_repo_read_count() == 2
    assert "FocalLoss" in result["paper_repo_context"]
    assert all("/blob/main/" in url for url in read_urls)

    symbol_paths = {row["path"] for row in result["repo_auto_read_symbols"]}
    assert "src/loss.py" in symbol_paths


def test_search_paper_repo_auto_read_skips_binary_and_respects_budget(monkeypatch):
    """权重和图片永远不读；读预算耗尽时安静退回路径列表。"""
    read_paths: list[str] = []

    async def fake_tree(owner, repo, **kwargs):
        return _tree_result()

    async def fake_source(url, **kwargs):
        read_paths.append(url)
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    ctx = _make_ctx()

    # 每轮命中 2 个源码路径，两轮就把 4 次读预算用完。
    for _ in range(3):
        asyncio.run(
            retrieval_tools.execute_async_tool(
                "search_paper_repo",
                {"repoId": GITHUB_REPO_ID, "query": "损失函数和模型结构怎么实现的"},
                ctx,
            )
        )

    assert ctx.paper_repo_read_count() == retrieval_tools._MAX_PAPER_REPO_READS
    # 目录树里的权重和图片始终没有被读过。
    assert read_paths
    assert not any(url.endswith((".png", ".pt")) for url in read_paths)

    exhausted = asyncio.run(
        retrieval_tools.execute_async_tool(
            "search_paper_repo",
            {"repoId": GITHUB_REPO_ID, "query": "损失函数和模型结构怎么实现的"},
            ctx,
        )
    )
    # 第 4 次连目录检索预算都没了，但仍是干净的失败而不是异常。
    assert exhausted["error_code"] == "paper_repo_search_limit_reached"


def test_search_paper_repo_maps_chinese_terms_to_paths(monkeypatch):
    async def fake_tree(owner, repo, **kwargs):
        return _tree_result()

    async def fake_source(url, **kwargs):
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)

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

    async def fake_source(url, **kwargs):
        return _blob_result(LOSS_SOURCE)

    monkeypatch.setattr(retrieval_tools, "read_github_repo_tree", fake_tree)
    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
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


def test_read_paper_repo_cursor_paging_does_not_skip_content(monkeypatch):
    """证据渲染会裁剪正文，next_cursor 必须按真正进入证据的长度推进。"""
    source = "\n".join(f"line_{index} = compute({index})" for index in range(400))

    async def fake_source(url, max_chars=6000, start_char=0, **kwargs):
        window = source[start_char:start_char + max_chars]
        return _blob_result(
            window,
            truncated=start_char + len(window) < len(source),
            content_start=start_char,
        )

    monkeypatch.setattr(retrieval_tools, "read_github_public_source", fake_source)
    ctx = _make_ctx()

    first = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/long.py", "maxChars": 6000},
            ctx,
        )
    )
    cursor = first["next_cursor"]
    assert cursor and cursor <= retrieval_tools._PAPER_REPO_EVIDENCE_BODY_CHARS
    # 作答通道 PAPER_REPO_EVIDENCE 必须含有第一页窗口末尾；formatted chunk 仍可能被裁到 1500。
    assert source[cursor - 12:cursor].strip() in first["paper_repo_context"]

    second = asyncio.run(
        retrieval_tools.execute_async_tool(
            "read_paper_repo",
            {"repoId": GITHUB_REPO_ID, "path": "src/long.py", "cursor": cursor},
            ctx,
        )
    )
    assert source[cursor:cursor + 12].strip() in second["paper_repo_context"]


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


def test_method_implementation_question_enables_repo_tools_only_with_a_repo():
    """「这个方法怎么实现的」有仓库就读代码，没仓库仍走纯论文机制讲解。"""
    with_repo = _make_agent()
    with_repo._doc_ctx = _make_ctx()

    without_repo = _make_agent()
    without_repo._doc_ctx = _make_ctx("这篇论文没有给出任何公开仓库地址。")

    for question in ("这个方法怎么实现的", "How is this method implemented?"):
        assert with_repo._wants_code_implementation(question) is True
        assert without_repo._wants_code_implementation(question) is False


def test_reference_link_question_never_enables_repo_tools():
    agent = _make_agent()
    agent._doc_ctx = _make_ctx()

    assert agent._wants_code_implementation("参考文献里的 GitHub 链接是什么") is False
    assert agent._wants_code_implementation("这篇论文的动机是什么") is False


def test_paper_method_gap_flags_code_without_paper_evidence(monkeypatch):
    """只有源码、没有论文方法段时无法做对照讲解，需要补论文侧检索。"""
    ctx = _make_ctx()
    chunk = _rendered_repo_file_chunk(monkeypatch, ctx)
    agent = _make_agent()
    agent._doc_ctx = ctx
    question = "这篇论文的训练脚本在仓库哪"

    assert agent._paper_method_evidence_gap(question, [chunk]) is True

    method_section = retrieval_tools._format_tool_chunk(
        "方法部分详述了 focal loss 的定义与推导。" * 40,
        source="vector_search",
        context_id="c1",
        evidence_id="c1",
        chunk_idx="c1",
        chunk_type="text",
    )
    assert agent._paper_method_evidence_gap(question, [chunk, method_section]) is False
    # 还没读到任何仓库文件时由硬闸负责，软提示不参与。
    assert agent._paper_method_evidence_gap(question, []) is False


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
