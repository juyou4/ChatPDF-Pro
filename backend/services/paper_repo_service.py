"""论文仓库登记表：从当前文档正文里抽取公开代码仓库地址。

本模块是纯文本处理，**不访问网络**，也不消费 web_search 的返回结果。
论文仓库的唯一来源是这篇文档自己的正文，因此 Planner 无法把一个外部搜索
命中"洗"成论文仓库，也无法把任意 URL 递给 GitHub 读取层。

抽取结果只有 GitHub 标记 ``fetch_supported``：本轮只对 GitHub 做只读
文件/目录树访问，GitLab 与 Hugging Face 仅登记与展示。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

PAPER_REPO_ID_PREFIX = "paper-repo"
PAPER_REPO_HOSTS = ("github", "gitlab", "huggingface")

MAX_PAPER_REPO_SCAN_CHARS = 80_000
MAX_PAPER_REPOS = 12

_OWNER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,38}"
_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}"

_GITHUB_REPO_RE = re.compile(
    rf"(?:https?://)?(?:www\.)?github\.com/({_OWNER_PATTERN})/({_NAME_PATTERN})",
    re.IGNORECASE,
)
_GITLAB_REPO_RE = re.compile(
    rf"(?:https?://)?(?:www\.)?gitlab\.com/({_OWNER_PATTERN})/({_NAME_PATTERN})",
    re.IGNORECASE,
)
_HUGGINGFACE_REPO_RE = re.compile(
    rf"(?:https?://)?(?:www\.)?huggingface\.co/"
    rf"(?:(models|datasets|spaces)/)?({_OWNER_PATTERN})/({_NAME_PATTERN})",
    re.IGNORECASE,
)

# 明确带标注的仓库行优先排序：论文正文里的 "Code: https://github.com/..."
# 比参考文献里顺带出现的链接更可能是本文自己的实现。
_REPO_LABEL_RE = re.compile(
    r"(?:code(?:base)?|source\s*code|implementation|repositor(?:y|ies)|repo|project\s*page|"
    r"代码|源码|仓库|代码仓库|开源|实现)"
    r"[^\n]{0,20}?"
    r"(?:is\s+)?(?:available\s+at|released\s+at|found\s+at|hosted\s+at|at\s|:|：|在|位于|见)",
    re.IGNORECASE,
)

# GitHub 上这些一级路径不是仓库 owner，而是站点自身的功能页面。
_GITHUB_RESERVED_OWNERS = {
    "about", "account", "apps", "blog", "codespaces", "collections", "contact",
    "customer-stories", "dashboard", "enterprise", "events", "explore", "features",
    "git", "home", "issues", "join", "login", "logout", "marketplace", "new",
    "notifications", "nonprofit", "organizations", "orgs", "pages", "pricing",
    "pulls", "readme", "resources", "search", "security", "sessions", "settings",
    "site", "solutions", "sponsors", "stars", "team", "topics", "trending", "users",
    "watching",
}

# 论文/README 里常见的占位地址。它们语法合法但指向不存在的仓库。
# 这张表只收**明显**是占位符的词：真实项目确实会叫 org/user/name/test，
# 过度过滤会把论文自己的仓库丢掉，代价比留下一个死链接大得多。
_PLACEHOLDER_TOKENS = {
    "anon", "anonymous", "bar", "example", "examples", "foo", "placeholder",
    "project-name", "reponame", "repo-name", "tbd", "todo", "user-name",
    "user_name", "username", "your-name", "your-repo", "your_name", "your_repo",
    "yourname", "yourrepo", "xxx", "xxxx", "xxxxx",
}

_TRAILING_PUNCTUATION = ".,;:!?)]}\"'）】》、，。；："

_NON_SOURCE_EXTENSIONS = {
    ".7z", ".a", ".avi", ".bin", ".bmp", ".bz2", ".class", ".ckpt", ".dll", ".doc",
    ".docx", ".dylib", ".eot", ".exe", ".gif", ".gz", ".h5", ".ico", ".jar", ".jpeg",
    ".jpg", ".lock", ".mo", ".model", ".mov", ".mp3", ".mp4", ".npy", ".npz", ".o",
    ".onnx", ".otf", ".pdf", ".pickle", ".pkl", ".png", ".ppt", ".pptx", ".pt",
    ".pth", ".pyc", ".rar", ".safetensors", ".so", ".svg", ".tar", ".tgz", ".tif",
    ".tiff", ".ttf", ".wav", ".webp", ".whl", ".woff", ".woff2", ".xls", ".xlsx",
    ".xz", ".zip",
}
_IGNORED_PATH_PREFIXES = (
    ".git/", ".github/workflows/", "node_modules/", "__pycache__/", "dist/",
    "build/", "docs/_build/", "site-packages/", "third_party/", "vendor/",
)
_MAX_TREE_BLOB_BYTES = 512_000

# 中文提问命中不了英文文件名，这里把常见问法映射到代码里真正会出现的词根。
_QUERY_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "损失函数": ("loss", "criterion", "objective"),
    "损失": ("loss", "criterion"),
    "训练脚本": ("train", "run", "main"),
    "训练": ("train", "trainer", "training"),
    "微调": ("finetune", "sft", "train"),
    "预训练": ("pretrain", "train"),
    "推理": ("infer", "inference", "predict", "generate"),
    "评测": ("eval", "benchmark", "test"),
    "评估": ("eval", "metric"),
    "指标": ("metric", "eval"),
    "模型": ("model", "net", "network", "modeling"),
    "网络": ("model", "net"),
    "结构": ("model", "arch", "module"),
    "配置": ("config", "cfg", "yaml"),
    "超参": ("config", "hparams", "args"),
    "参数": ("config", "args"),
    "数据集": ("dataset", "data"),
    "数据": ("data", "dataset"),
    "数据加载": ("dataloader", "dataset"),
    "优化器": ("optim", "optimizer"),
    "学习率": ("lr", "scheduler", "optim"),
    "注意力": ("attention", "attn"),
    "编码器": ("encoder",),
    "解码器": ("decoder",),
    "预处理": ("preprocess", "transform"),
    "后处理": ("postprocess",),
    "入口": ("main", "run"),
    "主函数": ("main",),
    "依赖": ("requirements", "setup", "pyproject"),
    "安装": ("readme", "setup", "install"),
}
_QUERY_STOPWORDS = {
    "a", "about", "an", "and", "are", "code", "file", "files", "find", "for", "how",
    "implement", "implementation", "in", "is", "of", "on", "paper", "repo",
    "repository", "source", "that", "the", "this", "to", "什么", "代码", "仓库",
    "论文", "哪里", "哪个", "怎么", "如何", "文件", "实现", "源码",
}
_IMPLEMENTATION_HINT_TOKENS = (
    "train", "main", "model", "loss", "run", "config", "dataset", "engine",
    "trainer", "solver", "network", "modeling", "pipeline",
)

_PYTHON_SYMBOL_RE = re.compile(r"^(\s*)(?:(async)\s+)?(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_JS_SYMBOL_RE = re.compile(
    r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_PYTHON_SUFFIXES = (".py", ".pyi", ".pyx")
_JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


def build_paper_repo_id(host: str, slug: str) -> str:
    """Return the stable repo id shared by the planner, tools and the trace UI."""
    return f"{PAPER_REPO_ID_PREFIX}:{str(host).strip().lower()}:{str(slug).strip()}"


def normalize_paper_repo_id(repo_id: Any) -> str:
    return str(repo_id or "").strip()


def _clean_segment(value: str) -> str:
    segment = str(value or "").strip().strip(_TRAILING_PUNCTUATION)
    if segment.lower().endswith(".git"):
        segment = segment[: -len(".git")]
    return segment.strip(_TRAILING_PUNCTUATION)


def _is_placeholder(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return True
    if normalized in _PLACEHOLDER_TOKENS:
        return True
    return bool(re.fullmatch(r"[<\[{(].*[>\]})]", normalized))


def collect_paper_repo_text(
    full_text: str = "",
    chunks: Sequence[Any] | None = None,
    pages: Sequence[Any] | None = None,
    *,
    max_chars: int = MAX_PAPER_REPO_SCAN_CHARS,
) -> str:
    """Join a bounded slice of the document sources that may carry repo links."""
    budget = max(0, int(max_chars or 0))
    if budget <= 0:
        return ""
    parts: list[str] = []
    remaining = budget

    def _append(value: Any) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        text = str(value or "").strip()
        if not text:
            return
        parts.append(text[:remaining])
        remaining -= min(len(text), remaining)

    _append(full_text)
    for chunk in chunks or []:
        if remaining <= 0:
            break
        _append(chunk if isinstance(chunk, str) else (chunk or {}).get("text") if isinstance(chunk, dict) else "")
    for page in pages or []:
        if remaining <= 0:
            break
        if isinstance(page, dict):
            _append(page.get("content") or page.get("text"))
        else:
            _append(page)
    return "\n".join(parts)


def _labeled_span(text: str, index: int) -> bool:
    """Whether the same line introduces this URL as the paper's own code link."""
    line_start = text.rfind("\n", 0, index) + 1
    return bool(_REPO_LABEL_RE.search(text[line_start:index]))


def _github_candidate(match: re.Match) -> dict | None:
    owner = _clean_segment(match.group(1))
    name = _clean_segment(match.group(2))
    if not owner or not name:
        return None
    if owner.lower() in _GITHUB_RESERVED_OWNERS:
        return None
    if _is_placeholder(owner) or _is_placeholder(name):
        return None
    slug = f"{owner}/{name}"
    return {
        "repo_id": build_paper_repo_id("github", slug),
        "host": "github",
        "owner": owner,
        "name": name,
        "resource": "",
        "url": f"https://github.com/{slug}",
        "fetch_supported": True,
    }


def _gitlab_candidate(match: re.Match) -> dict | None:
    owner = _clean_segment(match.group(1))
    name = _clean_segment(match.group(2))
    if not owner or not name or _is_placeholder(owner) or _is_placeholder(name):
        return None
    slug = f"{owner}/{name}"
    return {
        "repo_id": build_paper_repo_id("gitlab", slug),
        "host": "gitlab",
        "owner": owner,
        "name": name,
        "resource": "",
        # GitLab 本轮只登记与展示，不做只读文件访问。
        "url": f"https://gitlab.com/{slug}",
        "fetch_supported": False,
    }


def _huggingface_candidate(match: re.Match) -> dict | None:
    resource = str(match.group(1) or "").strip().lower()
    owner = _clean_segment(match.group(2))
    name = _clean_segment(match.group(3))
    if not owner or not name or _is_placeholder(owner) or _is_placeholder(name):
        return None
    slug = f"{resource}/{owner}/{name}" if resource else f"{owner}/{name}"
    return {
        "repo_id": build_paper_repo_id("huggingface", slug),
        "host": "huggingface",
        "owner": owner,
        "name": name,
        "resource": resource,
        "url": f"https://huggingface.co/{slug}",
        "fetch_supported": False,
    }


def extract_paper_repositories(
    text: str,
    *,
    max_repos: int = MAX_PAPER_REPOS,
    max_chars: int = MAX_PAPER_REPO_SCAN_CHARS,
) -> list[dict]:
    """Extract the public repositories that literally appear in the paper text."""
    body = str(text or "")[: max(0, int(max_chars or 0))]
    if not body:
        return []
    found: dict[str, dict] = {}
    for pattern, builder in (
        (_GITHUB_REPO_RE, _github_candidate),
        (_GITLAB_REPO_RE, _gitlab_candidate),
        (_HUGGINGFACE_REPO_RE, _huggingface_candidate),
    ):
        for match in pattern.finditer(body):
            candidate = builder(match)
            if candidate is None:
                continue
            repo_id = candidate["repo_id"]
            existing = found.get(repo_id)
            labeled = _labeled_span(body, match.start())
            if existing is None:
                candidate["labeled"] = labeled
                candidate["first_index"] = match.start()
                found[repo_id] = candidate
            elif labeled and not existing.get("labeled"):
                existing["labeled"] = True
    ordered = sorted(
        found.values(),
        key=lambda item: (not item.get("labeled"), int(item.get("first_index") or 0)),
    )
    repositories: list[dict] = []
    for item in ordered[: max(1, int(max_repos or MAX_PAPER_REPOS))]:
        repositories.append({
            "repo_id": item["repo_id"],
            "host": item["host"],
            "owner": item["owner"],
            "name": item["name"],
            "resource": item.get("resource", ""),
            "url": item["url"],
            "fetch_supported": bool(item.get("fetch_supported")),
            "labeled": bool(item.get("labeled")),
        })
    return repositories


def extract_paper_repositories_from_document(
    full_text: str = "",
    chunks: Sequence[Any] | None = None,
    pages: Sequence[Any] | None = None,
    *,
    max_repos: int = MAX_PAPER_REPOS,
    max_chars: int = MAX_PAPER_REPO_SCAN_CHARS,
) -> list[dict]:
    return extract_paper_repositories(
        collect_paper_repo_text(full_text, chunks, pages, max_chars=max_chars),
        max_repos=max_repos,
        max_chars=max_chars,
    )


def sanitize_repo_path(path: Any) -> str:
    """Return a repository-relative path, or an empty string when unsafe."""
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if "://" in raw or raw.startswith("//"):
        return ""
    raw = raw.lstrip("/")
    if len(raw) > 400:
        return ""
    segments: list[str] = []
    for segment in raw.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            return ""
        if not re.fullmatch(r"[A-Za-z0-9._@+-]{1,120}", segment):
            return ""
        segments.append(segment)
    return "/".join(segments)


def sanitize_repo_ref(ref: Any) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    return value if re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", value) else ""


def is_binary_repo_path(path: Any) -> bool:
    normalized = str(path or "").strip().lower()
    if not normalized:
        return True
    return any(normalized.endswith(suffix) for suffix in _NON_SOURCE_EXTENSIONS)


def _is_searchable_entry(entry: dict) -> bool:
    path = str(entry.get("path") or "").strip()
    if not path or str(entry.get("type") or "blob") != "blob":
        return False
    lowered = path.lower()
    if any(lowered.startswith(prefix) for prefix in _IGNORED_PATH_PREFIXES):
        return False
    if is_binary_repo_path(lowered):
        return False
    try:
        size = int(entry.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    return size <= _MAX_TREE_BLOB_BYTES


def repo_query_terms(query: Any) -> list[str]:
    """Turn a natural-language question into path keywords the tree can match."""
    text = str(query or "").strip().lower()
    if not text:
        return []
    terms: list[str] = []
    for alias, expansions in _QUERY_KEYWORD_ALIASES.items():
        if alias in text:
            terms.extend(expansions)
    for token in re.findall(r"[a-z0-9_]{2,}", text):
        if token in _QUERY_STOPWORDS or token.isdigit():
            continue
        terms.append(token)
    ordered: list[str] = []
    for term in terms:
        if term and term not in ordered:
            ordered.append(term)
    return ordered[:12]


def _path_score(path: str, terms: Sequence[str]) -> float:
    lowered = path.lower()
    basename = lowered.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    score = 0.0
    for term in terms:
        if term == stem:
            score += 4.0
        elif term in basename:
            score += 2.5
        elif term in lowered:
            score += 1.2
    if score <= 0:
        return 0.0
    if lowered.endswith((".py", ".ipynb")):
        score += 0.6
    elif lowered.endswith((".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".md")):
        score += 0.3
    if "/test" in lowered or lowered.startswith("tests/"):
        score -= 0.8
    score -= 0.08 * lowered.count("/")
    return score


def rank_repo_tree_paths(
    entries: Iterable[dict],
    query: Any,
    *,
    limit: int = 8,
) -> tuple[list[dict], str]:
    """Rank tree blobs against a question.

    Returns the ranked rows and the strategy that produced them: ``query`` when
    the question itself matched a path, ``implementation_hint`` when it did not
    and the generic implementation-file heuristic was used instead.
    """
    candidates = [entry for entry in entries or [] if isinstance(entry, dict) and _is_searchable_entry(entry)]
    if not candidates:
        return [], "empty_tree"
    bounded_limit = max(1, min(24, int(limit or 8)))
    terms = repo_query_terms(query)
    strategy = "query"
    scored = [
        (score, entry)
        for entry in candidates
        if (score := _path_score(str(entry.get("path") or ""), terms)) > 0
    ]
    if not scored:
        strategy = "implementation_hint"
        scored = [
            (score, entry)
            for entry in candidates
            if (score := _path_score(str(entry.get("path") or ""), _IMPLEMENTATION_HINT_TOKENS)) > 0
        ]
    if not scored:
        return [], "no_match"
    scored.sort(key=lambda row: (-row[0], str(row[1].get("path") or "")))
    ranked = [
        {
            "path": str(entry.get("path") or ""),
            "size": int(entry.get("size") or 0) if str(entry.get("size") or "").isdigit() else entry.get("size") or 0,
            "score": round(float(score), 3),
        }
        for score, entry in scored[:bounded_limit]
    ]
    return ranked, strategy


def extract_source_symbols(path: Any, text: Any, *, limit: int = 24) -> list[dict]:
    """Extract def/class anchors so the trace panel can align on real symbols."""
    normalized_path = str(path or "").strip().lower()
    body = str(text or "")
    if not body:
        return []
    if normalized_path.endswith(_PYTHON_SUFFIXES):
        pattern = _PYTHON_SYMBOL_RE
        python_style = True
    elif normalized_path.endswith(_JS_SUFFIXES):
        pattern = _JS_SYMBOL_RE
        python_style = False
    else:
        return []

    lines = body.splitlines()
    raw: list[dict] = []
    for index, line in enumerate(lines, start=1):
        match = pattern.match(line)
        if not match:
            continue
        if python_style:
            indent, _async, keyword, name = match.groups()
        else:
            indent, keyword, name = match.groups()
        raw.append({
            "name": name,
            "kind": "class" if keyword == "class" else "def",
            "line": index,
            "indent": len(indent.expandtabs(4)),
        })
    symbols: list[dict] = []
    for position, item in enumerate(raw):
        end_line = len(lines)
        for follower in raw[position + 1:]:
            if follower["indent"] <= item["indent"]:
                end_line = max(item["line"], follower["line"] - 1)
                break
        symbols.append({
            "name": item["name"],
            "kind": item["kind"],
            "line": item["line"],
            "end_line": end_line,
        })
        if len(symbols) >= max(1, int(limit or 24)):
            break
    return symbols


def readme_referenced_paths(readme_text: Any, entries: Iterable[dict], *, limit: int = 3) -> list[str]:
    """Return tree paths that the README itself points at."""
    text = str(readme_text or "")
    if not text:
        return []
    known = {
        str(entry.get("path") or "")
        for entry in entries or []
        if isinstance(entry, dict) and _is_searchable_entry(entry)
    }
    if not known:
        return []
    mentioned: list[str] = []
    for candidate in re.findall(r"[A-Za-z0-9_./-]{3,120}\.[A-Za-z0-9]{1,8}", text):
        normalized = candidate.strip("./")
        if normalized in known and normalized not in mentioned:
            mentioned.append(normalized)
        if len(mentioned) >= max(1, int(limit or 3)):
            break
    return mentioned


__all__ = [
    "MAX_PAPER_REPOS",
    "MAX_PAPER_REPO_SCAN_CHARS",
    "PAPER_REPO_HOSTS",
    "PAPER_REPO_ID_PREFIX",
    "build_paper_repo_id",
    "collect_paper_repo_text",
    "extract_paper_repositories",
    "extract_paper_repositories_from_document",
    "extract_source_symbols",
    "is_binary_repo_path",
    "normalize_paper_repo_id",
    "rank_repo_tree_paths",
    "readme_referenced_paths",
    "repo_query_terms",
    "sanitize_repo_path",
    "sanitize_repo_ref",
]
