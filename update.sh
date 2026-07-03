#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR" || exit 1

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

echo "========================================"
echo "   ChatPDF Pro - 一键升级"
echo "========================================"
echo ""

echo "正在从 GitHub 拉取最新代码..."
if ! git pull origin main; then
    echo "Git 拉取失败，请检查网络或手动执行 git pull"
    exit 1
fi

echo ""
echo "正在检查 Python 环境..."
if ! select_python; then
    echo "未找到 Python 3.10+，请先安装或设置 PYTHON=/path/to/python"
    exit 1
fi

if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    "$PYTHON_CMD" -m ensurepip --upgrade > /dev/null 2>&1
fi

if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    echo "当前 Python 缺少 pip：$PYTHON_CMD"
    exit 1
fi

echo "正在更新后端依赖..."
if ! "$PYTHON_CMD" -m pip install -r backend/requirements.txt; then
    echo "后端依赖更新失败"
    exit 1
fi

echo ""
echo "正在更新前端依赖..."
if ! command -v npm > /dev/null 2>&1; then
    echo "未找到 npm，请先安装 Node.js"
    exit 1
fi

cd frontend || exit 1
if ! npm install; then
    echo "前端依赖更新失败"
    exit 1
fi
cd "$BASE_DIR" || exit 1

echo ""
echo "========================================"
echo "升级完成！"
echo "========================================"
echo ""
echo "提示: 请关闭所有 ChatPDF 进程后重新运行 ./start.sh"
echo ""
