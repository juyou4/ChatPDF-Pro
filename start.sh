#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR" || exit 1

APP_VERSION="$(grep -E '"version"' version.json 2>/dev/null | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
if [ -z "$APP_VERSION" ]; then
    APP_VERSION="3.0.2"
fi

# 颜色和样式定义
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 清屏
clear

# 打印 Banner
echo -e "${BLUE}${BOLD}"
cat << EOF
  ╔═══════════════════════════════════════╗
  ║                                       ║
EOF
printf "  ║%-39s║\n" "     ChatPDF Pro v${APP_VERSION}"
cat << EOF
  ║     智能文档助手                      ║
  ║                                       ║
  ╚═══════════════════════════════════════╝
EOF
echo -e "${NC}"

# 进度显示函数
show_progress() {
    echo -ne "${BLUE}  ▶${NC} $1"
}

show_success() {
    echo -e "\r${GREEN}  ✓${NC} $1"
}

show_error() {
    echo -e "\r${RED}  ✗${NC} $1"
}

command_exists() {
    command -v "$1" > /dev/null 2>&1
}

python_version() {
    "$1" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>/dev/null
}

python_is_supported() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

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

        if python_is_supported "$resolved"; then
            PYTHON_CMD="$resolved"
            return 0
        fi
    done
    return 1
}

node_is_supported() {
    node - <<'NODE' >/dev/null 2>&1
const parts = process.versions.node.split('.').map(Number);
const [major, minor, patch] = parts;
const ok =
  (major === 20 && (minor > 19 || (minor === 19 && patch >= 0))) ||
  (major === 22 && (minor > 12 || (minor === 12 && patch >= 0))) ||
  major > 22;
process.exit(ok ? 0 : 1);
NODE
}

cleanup_backend() {
    if [ -n "${BACKEND_PID:-}" ] && ps -p "$BACKEND_PID" > /dev/null 2>&1; then
        kill "$BACKEND_PID" 2>/dev/null
    fi
}

# ==================== 自动更新 ====================
show_progress "检查代码更新..."

# 获取当前分支名
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

# 只在main分支时自动更新，其他分支跳过
if [ "$CURRENT_BRANCH" = "main" ]; then
    git pull origin main > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        show_success "代码已更新到最新版本"
    else
        show_success "已是最新版本 (或更新跳过)"
    fi
else
    if [ -n "$CURRENT_BRANCH" ]; then
        show_success "当前在分支 $CURRENT_BRANCH (跳过自动更新)"
    else
        show_success "跳过更新检查"
    fi
fi

# ==================== 环境检查 ====================
show_progress "检查运行环境..."

# 检查 Python
if ! select_python; then
    show_error "未找到 Python 3.10+，请先安装或设置 PYTHON=/path/to/python"
    exit 1
fi

# 检查 Node.js
if ! command_exists node; then
    show_error "未找到 Node.js，请先安装"
    exit 1
fi

if ! command_exists npm; then
    show_error "未找到 npm，请先安装 Node.js/npm"
    exit 1
fi

if ! node_is_supported; then
    show_error "Node.js 版本不兼容，当前 $(node --version)，需要 ^20.19.0 或 >=22.12.0"
    exit 1
fi

if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    show_progress "安装 pip..."
    "$PYTHON_CMD" -m ensurepip --upgrade > /dev/null 2>&1
fi

if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    show_error "当前 Python 缺少 pip：$PYTHON_CMD"
    exit 1
fi

show_success "环境检查通过 (Python $(python_version "$PYTHON_CMD"), Node $(node --version))"

# ==================== 清理旧进程 ====================
show_progress "清理旧进程..."

# 清理端口 8000
PORT_PIDS=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    echo "$PORT_PIDS" | xargs kill -9 2>/dev/null
fi
pkill -f "python.*backend/app.py" 2>/dev/null
find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

show_success "清理完成"

# ==================== 基础运行时 ====================
show_progress "检查基础运行时..."

# MinerU 是默认解析路线。只确保通用后端、预览和检索依赖；
# 本地 OCR / ODL / YOLO 由用户选择“本地解析”后按需安装。
if ! "$PYTHON_CMD" -c "import fastapi, uvicorn, fitz, pdfplumber, faiss, langchain, openai, sentence_transformers" > /dev/null 2>&1; then
    show_progress "首次安装基础运行时..."
    if ! "$PYTHON_CMD" -m pip install -q -r backend/requirements-core.txt; then
        show_error "基础运行时安装失败，请检查 Python、网络或 requirements-core.txt"
        exit 1
    fi
fi

show_success "基础运行时已就绪"
echo -e "${BLUE}  i${NC} 本地解析组件将在选择本地路线时按需准备"
# 前端依赖
cd frontend
if [ ! -d "node_modules" ]; then
    show_progress "首次运行，安装前端依赖 (需要1-2分钟)..."
    npm install --silent > /dev/null 2>&1
fi

# 确保 rehype-raw 已安装（Blur Reveal 效果依赖）
npm list rehype-raw > /dev/null 2>&1 || npm install rehype-raw --silent > /dev/null 2>&1

cd ..

show_success "依赖检查完成"

# ==================== 启动服务 ====================
show_progress "启动后端服务..."
BACKEND_LOG="$BASE_DIR/backend/backend_startup.log"
nohup "$PYTHON_CMD" backend/app.py > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

trap cleanup_backend EXIT INT TERM

# 检查后端是否成功启动
BACKEND_READY=0
for i in $(seq 1 90); do
    if command_exists curl && curl -fsS --max-time 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
        BACKEND_READY=1
        break
    fi
    if ! ps -p "$BACKEND_PID" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [ "$BACKEND_READY" = "1" ]; then
    show_success "后端服务启动成功 (PID: $BACKEND_PID)"
else
    show_error "后端启动失败或超时"
    if [ -f "$BACKEND_LOG" ]; then
        echo ""
        echo -e "${YELLOW}  后端错误日志:${NC}"
        tail -80 "$BACKEND_LOG"
    fi
    exit 1
fi

show_progress "启动前端服务..."
cd frontend

# 延迟打开浏览器（等待前端服务完全启动）
(sleep 3 && "$PYTHON_CMD" -m webbrowser http://localhost:3000 2>/dev/null || \
 open http://localhost:3000 2>/dev/null || \
 xdg-open http://localhost:3000 2>/dev/null) &

echo ""
echo -e "${GREEN}${BOLD}  🎉 ChatPDF Pro 已启动！${NC}"
echo ""
echo -e "  ${BLUE}访问地址:${NC} ${BOLD}http://localhost:3000${NC}"
echo -e "  ${BLUE}后端API:${NC}  ${BOLD}http://127.0.0.1:8000${NC}"
echo ""
echo -e "  ${YELLOW}提示:${NC} 浏览器将自动打开，按 ${BOLD}Ctrl+C${NC} 停止服务"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 启动前端（前台运行，按 Ctrl+C 停止后会清理后端）
npm run dev

# ==================== 清理 ====================
echo ""
show_progress "正在停止服务..."
cleanup_backend
show_success "已停止所有服务"
