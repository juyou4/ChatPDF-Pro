#!/usr/bin/env python3
"""计算源码启动所需依赖清单的稳定指纹。

启动脚本用这个指纹判断 requirements 或前端 lockfile 是否发生变化。
脚本只依赖 Python 标准库，不读取用户文档、历史记录或其他运行时数据。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUP_FILES: dict[str, tuple[Path, ...]] = {
    "python": (Path("backend/requirements-core.txt"),),
    "frontend": (
        Path("frontend/package.json"),
        Path("frontend/package-lock.json"),
    ),
}


def fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative_path in paths:
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        path = ROOT / relative_path
        if not path.is_file():
            digest.update(b"<missing>")
            digest.update(b"\0")
            continue
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUP_FILES), required=True)
    args = parser.parse_args()
    print(fingerprint(GROUP_FILES[args.group]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
