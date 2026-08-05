"""Build and release identity helpers for ChatPDF.

`version.json` is the canonical release metadata. During source runs we can
derive the Git identity from the working tree. During packaged desktop runs the
build pipeline writes `build-info.json`, which is bundled by PyInstaller.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundle_root() -> Path | None:
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(str(sys._MEIPASS))
        return Path(sys.executable).resolve().parent
    return None


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    bundle = _bundle_root()
    if bundle is not None:
        roots.extend([bundle, bundle.parent])
    roots.append(_source_root())
    roots.append(_source_root() / "backend")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _load_first_json(env_name: str, filenames: tuple[str, ...]) -> dict[str, Any] | None:
    env_path = os.getenv(env_name, "").strip()
    if env_path:
        data = _read_json(Path(env_path))
        if data is not None:
            return data

    for root in _candidate_roots():
        for filename in filenames:
            data = _read_json(root / filename)
            if data is not None:
                return data
    return None


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return None


def _source_git_identity() -> dict[str, Any]:
    repo_root = _source_root()
    sha = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if not sha:
        return {}
    return {
        "git_sha": sha,
        "git_short_sha": sha[:12],
        "git_branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or None,
        "build_time": _git(repo_root, "show", "-s", "--format=%cI", "HEAD") or None,
        "build_dirty": _git_dirty(repo_root),
        "build_source": "source-git",
    }


@lru_cache(maxsize=1)
def get_build_identity() -> dict[str, Any]:
    version_meta = _load_first_json("CHATPDF_VERSION_FILE", ("version.json",)) or {}
    build_meta = _load_first_json(
        "CHATPDF_BUILD_INFO_FILE",
        ("build-info.json", "backend/build-info.json"),
    ) or {}

    git_meta = _source_git_identity() if not build_meta else {}

    version = str(version_meta.get("version") or build_meta.get("version") or "0.0.0")
    git_sha = (
        os.getenv("CHATPDF_BUILD_GIT_SHA")
        or build_meta.get("git_sha")
        or git_meta.get("git_sha")
        or ""
    )
    git_short_sha = (
        build_meta.get("git_short_sha")
        or git_meta.get("git_short_sha")
        or (git_sha[:12] if git_sha else "")
    )

    identity = {
        "version": version,
        "schema_version": version_meta.get("schema_version") or 1,
        "release_date": version_meta.get("releaseDate") or version_meta.get("release_date"),
        "changelog": version_meta.get("changelog") or "",
        "github_owner": version_meta.get("githubOwner") or "",
        "github_repo": version_meta.get("githubRepo") or "",
        "git_sha": git_sha,
        "git_short_sha": git_short_sha,
        "git_branch": (
            os.getenv("CHATPDF_BUILD_GIT_BRANCH")
            or build_meta.get("git_branch")
            or git_meta.get("git_branch")
            or ""
        ),
        "build_time": (
            os.getenv("CHATPDF_BUILD_TIME")
            or build_meta.get("build_time")
            or git_meta.get("build_time")
            or ""
        ),
        "build_dirty": build_meta.get("build_dirty", git_meta.get("build_dirty")),
        "build_source": build_meta.get("build_source") or git_meta.get("build_source") or "version-json",
        # Backward-compatible marker retained for old clients that read /version.feature.
        "feature": "native_pdf_url",
    }
    return identity


def get_public_build_identity() -> dict[str, Any]:
    identity = dict(get_build_identity())
    identity["github"] = {
        "owner": identity.pop("github_owner", ""),
        "repo": identity.pop("github_repo", ""),
    }
    return identity
