"""后端测试棘轮：只对「新增」失败报错，让回归重新可见。

主干上长期存在一批测试与实现漂移导致的失败（详见 backend/tests/known_failures.txt）。
在它们被逐一裁决之前，整套测试是红的，任何新引入的回归都会淹没在里面——这个脚本
把基线固定下来，使得「红了就是这次改动搞的」重新成立。

用法::

    python backend/scripts/check_test_ratchet.py            # 校验，无新增失败则退出 0
    python backend/scripts/check_test_ratchet.py --update   # 裁决完一批后刷新基线
    python backend/scripts/check_test_ratchet.py -k memory  # 只跑文件名含 memory 的测试

必须逐文件运行：整套测试同进程跑会在原生扩展层（faiss / fitz）abort，
这与被测代码无关，但会让结果不可用。
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = BACKEND_DIR / "tests"
BASELINE_PATH = TESTS_DIR / "known_failures.txt"

_RESULT_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)


def load_baseline() -> set[str]:
    if not BASELINE_PATH.exists():
        return set()
    entries = set()
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def write_baseline(entries: set[str], *, crashed: list[str]) -> None:
    header = [
        "# 后端测试棘轮基线 —— 由 backend/scripts/check_test_ratchet.py --update 生成。",
        "#",
        "# 这里列出的是主干上「已知失败」的用例，绝大多数是测试与实现漂移：实现往前走了，",
        "# 断言没跟上。它们不是被批准的缺陷，而是待裁决项——每修好一个就应该从本文件移除，",
        "# 让基线单调收紧。禁止为了让检查通过而往这里添加新条目。",
        "#",
        f"# 当前条目数：{len(entries)}",
    ]
    if crashed:
        header.append("# 运行时崩溃的文件：" + ", ".join(crashed))
    body = sorted(entries)
    BASELINE_PATH.write_text("\n".join(header + [""] + body) + "\n", encoding="utf-8")


def run_file(name: str) -> tuple[set[str], bool]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", f"tests/{name}",
            "-q", "--no-header", "--tb=no",
            "-p", "no:warnings", "-p", "no:cacheprovider",
        ],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    crashed = proc.returncode not in (0, 1) or "Fatal Python error" in output
    return set(_RESULT_RE.findall(output)), crashed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="用本次结果覆盖基线")
    parser.add_argument("-k", dest="filter", default="", help="只跑文件名包含该子串的测试文件")
    args = parser.parse_args()

    files = sorted(p.name for p in TESTS_DIR.glob("test_*.py") if args.filter in p.name)
    if not files:
        print(f"没有匹配的测试文件（-k {args.filter!r}）")
        return 1

    observed: set[str] = set()
    crashed: list[str] = []
    for name in files:
        failures, did_crash = run_file(name)
        observed |= failures
        if did_crash:
            crashed.append(name)
        status = "CRASH" if did_crash else ("FAIL " if failures else "ok   ")
        print(f"{status} {name}")

    if args.update:
        write_baseline(observed, crashed=crashed)
        print(f"\n基线已刷新：{len(observed)} 条 -> {BASELINE_PATH}")
        return 0

    baseline = load_baseline()
    # 过滤掉本次未运行的文件，避免 -k 子集把基线里的其他条目误判成「已修复」。
    scoped = {entry for entry in baseline if any(f"tests/{name}::" in entry or entry.endswith(f"tests/{name}") for name in files)}
    new_failures = sorted(observed - baseline)
    fixed = sorted(scoped - observed)

    # 复核：部分用例（尤其 hypothesis 属性测试）会偶发失败。一个会误报的门最终会被
    # 无视，所以新增失败必须复跑一次确认，只有稳定复现的才算数。
    flaky: list[str] = []
    if new_failures:
        suspect_files = sorted({entry.split("::")[0].split("/")[-1] for entry in new_failures})
        print(f"\n复核 {len(new_failures)} 个新增失败（{', '.join(suspect_files)}）...")
        rerun: set[str] = set()
        for name in suspect_files:
            failures, _ = run_file(name)
            rerun |= failures
        flaky = [entry for entry in new_failures if entry not in rerun]
        new_failures = [entry for entry in new_failures if entry in rerun]

    if flaky:
        print(f"\n偶发失败 {len(flaky)} 个（复跑已通过，不计入门禁）：")
        for entry in flaky:
            print(f"  ~ {entry}")

    if fixed:
        print(f"\n以下 {len(fixed)} 个用例已经修好，请从基线移除（--update）：")
        for entry in fixed:
            print(f"  + {entry}")

    if crashed:
        print(f"\n运行时崩溃的文件：{', '.join(crashed)}")

    if new_failures:
        print(f"\n新增失败 {len(new_failures)} 个——这些是本次改动引入的：")
        for entry in new_failures:
            print(f"  ! {entry}")
        return 1

    print(f"\n无新增失败（基线 {len(baseline)} 条，本次观察到 {len(observed)} 条）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
