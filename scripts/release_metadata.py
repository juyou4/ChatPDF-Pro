from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing metadata file: {path}") from None
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return None


def canonical_version() -> str:
    version = str(read_json(ROOT / "version.json").get("version") or "").strip()
    if not version:
        raise SystemExit("version.json does not contain a version")
    return version


def assert_version(path: Path, actual: str, expected: str) -> None:
    if actual != expected:
        raise SystemExit(f"Version mismatch: {path} has {actual!r}, expected {expected!r}")


def check_versions() -> None:
    expected = canonical_version()
    assert_version(ROOT / "frontend/public/version.json", str(read_json(ROOT / "frontend/public/version.json").get("version")), expected)
    assert_version(ROOT / "frontend/package.json", str(read_json(ROOT / "frontend/package.json").get("version")), expected)
    assert_version(ROOT / "electron/package.json", str(read_json(ROOT / "electron/package.json").get("version")), expected)

    for lock_path in (ROOT / "frontend/package-lock.json", ROOT / "electron/package-lock.json"):
        lock = read_json(lock_path)
        assert_version(lock_path, str(lock.get("version")), expected)
        root_pkg = lock.get("packages", {}).get("", {})
        assert_version(lock_path, str(root_pkg.get("version")), expected)


def build_manifest() -> dict[str, Any]:
    version_meta = read_json(ROOT / "version.json")
    sha = git("rev-parse", "--verify", "HEAD")
    return {
        "schema_version": 1,
        "version": canonical_version(),
        "release_date": version_meta.get("releaseDate") or "",
        "changelog": version_meta.get("changelog") or "",
        "git_sha": sha,
        "git_short_sha": sha[:12] if sha else "",
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "build_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_dirty": git_dirty(),
        "build_source": "scripts/release_metadata.py",
    }


def write_build_info() -> dict[str, Any]:
    manifest = build_manifest()
    for path in (
        ROOT / "build-info.json",
        ROOT / "backend/build-info.json",
        ROOT / "frontend/public/build-info.json",
    ):
        write_json(path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and write ChatPDF release metadata.")
    parser.add_argument("--check", action="store_true", help="validate version declarations")
    parser.add_argument("--write-build-info", action="store_true", help="write build-info.json files")
    parser.add_argument("--strict-clean", action="store_true", help="fail if tracked files are dirty")
    args = parser.parse_args()

    if args.check or not args.write_build_info:
        check_versions()

    manifest: dict[str, Any] | None = None
    if args.write_build_info:
        manifest = write_build_info()

    if args.strict_clean:
        dirty = git_dirty()
        if dirty:
            raise SystemExit("Working tree has tracked changes; set aside changes before release build")

    if manifest:
        print(
            "ChatPDF build-info "
            f"v{manifest['version']} "
            f"{manifest.get('git_short_sha') or 'no-git'} "
            f"dirty={manifest.get('build_dirty')}"
        )
    else:
        print(f"ChatPDF metadata check OK: v{canonical_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
