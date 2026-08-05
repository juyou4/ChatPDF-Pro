#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR/backend"

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

echo "检查基础运行时..."
if ! "$PYTHON_CMD" -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >/dev/null 2>&1; then
    "$PYTHON_CMD" -m pip install -r requirements-core.txt
fi

echo "API 地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
exec "$PYTHON_CMD" app.py
