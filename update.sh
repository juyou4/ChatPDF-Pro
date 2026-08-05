#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

select_python() {
    local candidates=()
    if [ -n "${PYTHON:-}" ]; then
        candidates+=("$PYTHON")
    fi
    candidates+=(
        "python3.12"
        "python3.11"
        "python3.10"
        "$HOME/miniforge3/bin/python"
        "$HOME/miniconda3/bin/python"
        "$HOME/anaconda3/bin/python"
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3"
        "python3"
        "python"
    )

    local candidate resolved
    for candidate in "${candidates[@]}"; do
        if [[ "$candidate" == */* ]]; then
            [ -x "$candidate" ] || continue
            resolved="$candidate"
        else
            resolved="$(command -v "$candidate" 2>/dev/null || true)"
            [ -n "$resolved" ] || continue
        fi
        if "$resolved" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
        then
            PYTHON_CMD="$resolved"
            return 0
        fi
    done
    return 1
}

install_npm_deps() {
    local dir="$1"
    if ! command -v npm >/dev/null 2>&1; then
        echo "[X] 未找到 npm，请先安装 Node.js"
        exit 1
    fi
    if [ -f "$dir/package-lock.json" ]; then
        (cd "$dir" && npm ci)
    else
        (cd "$dir" && npm install)
    fi
}

echo "========================================"
echo "   ChatPDF - 一键升级"
echo "========================================"
echo ""

if ! command -v git >/dev/null 2>&1; then
    echo "[X] 未找到 Git，无法自动升级"
    exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "[X] 当前目录在分支 $CURRENT_BRANCH，一键升级只允许在 main 上执行"
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]; then
    echo "[X] 检测到本地源码改动。为避免覆盖或混合版本，请先提交或暂存这些改动。"
    exit 1
fi

echo "[1/4] 拉取 main 最新代码..."
git fetch origin main
git pull --ff-only origin main

echo ""
echo "[2/4] 检查 Python 环境..."
if ! select_python; then
    echo "[X] 未找到 Python 3.10+，请先安装或设置 PYTHON=/path/to/python"
    exit 1
fi
if ! "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_CMD" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
if ! "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    echo "[X] 当前 Python 缺少 pip：$PYTHON_CMD"
    exit 1
fi

echo "[3/4] 更新后端依赖..."
"$PYTHON_CMD" -m pip install -r backend/requirements.txt

echo ""
echo "[4/4] 更新前端与桌面端依赖..."
install_npm_deps frontend
install_npm_deps electron

"$PYTHON_CMD" scripts/release_metadata.py --check

echo ""
echo "========================================"
echo "升级完成"
echo "========================================"
echo "请关闭所有 ChatPDF 进程后重新运行 ./start.sh"
