#!/usr/bin/env bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR" || exit 1

APP_VERSION="$(grep -E '"version"' version.json 2>/dev/null | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
if [ -z "$APP_VERSION" ]; then
    APP_VERSION="3.0.2"
fi

# 颜色和样式定义；NO_COLOR 可用于关闭 ANSI 颜色。
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD='\033[1m'
    ACCENT='\033[38;2;201;103;77m'
    GREEN='\033[38;2;84;150;111m'
    YELLOW='\033[38;2;190;139;63m'
    RED='\033[38;2;194;82;82m'
    MUTED='\033[38;2;137;132;129m'
    NC='\033[0m'
else
    BOLD=''
    ACCENT=''
    GREEN=''
    YELLOW=''
    RED=''
    MUTED=''
    NC=''
fi

if [ -t 1 ] && [ -n "${TERM:-}" ]; then
    clear
fi

printf '\n'
printf "  ${ACCENT}${BOLD}ChatPDF${NC}\n"
printf "  ${MUTED}本地文档工作区  /  v%s${NC}\n" "$APP_VERSION"
printf "  ${MUTED}------------------------------------------------------------${NC}\n\n"

show_progress() {
    printf "\n  ${ACCENT}>${NC} ${BOLD}%s${NC}\n" "$1"
}

show_success() {
    printf "     ${GREEN}done${NC}  %s\n" "$1"
}

show_info() {
    printf "     ${MUTED}note${NC}  %s\n" "$1"
}

show_error() {
    printf "     ${RED}error${NC} %s\n" "$1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
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
const [major, minor, patch] = process.versions.node.split('.').map(Number);
const ok =
  (major === 20 && (minor > 19 || (minor === 19 && patch >= 0))) ||
  (major === 22 && (minor > 12 || (minor === 12 && patch >= 0))) ||
  major > 22;
process.exit(ok ? 0 : 1);
NODE
}

kill_port() {
    local port="$1" pids=""
    if command_exists lsof; then
        pids="$(lsof -ti TCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command_exists fuser; then
        pids="$(fuser "$port"/tcp 2>/dev/null || true)"
    fi
    local pid
    for pid in $pids; do
        kill -9 "$pid" 2>/dev/null || true
    done
}

CLEANED_UP=0
cleanup_services() {
    if [ "$CLEANED_UP" = "1" ]; then
        return
    fi
    CLEANED_UP=1
    if [ -n "${BACKEND_PID:-}" ] && ps -p "$BACKEND_PID" >/dev/null 2>&1; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    for port in 3000 8000 8001 8002 8003 8004 8005; do
        kill_port "$port"
    done
}

# ==================== 自动更新 ====================
show_progress "检查代码更新"
CURRENT_BRANCH=""
if command_exists git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [ "$CURRENT_BRANCH" = "main" ]; then
        if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
            show_info "检测到本地代码改动，为避免覆盖已跳过自动更新"
        elif git pull --ff-only origin main >/dev/null 2>&1; then
            show_success "代码已更新到 main 最新版本"
        else
            show_info "自动更新未完成，将继续使用当前版本"
        fi
    elif [ -n "$CURRENT_BRANCH" ]; then
        show_info "当前在分支 $CURRENT_BRANCH（仅 main 自动更新）"
    else
        show_info "无法识别当前分支，跳过自动更新"
    fi
else
    show_info "当前目录不是 Git 工作区，跳过自动更新"
fi

# ==================== 环境检查 ====================
show_progress "检查运行环境"
if ! select_python; then
    show_error "未找到 Python 3.10+，请先安装或设置 PYTHON=/path/to/python"
    exit 1
fi
if ! command_exists node; then
    show_error "未找到 Node.js，请先安装 Node.js 20.19+ 或 22.12+"
    exit 1
fi
if ! command_exists npm; then
    show_error "未找到 npm，请重新安装包含 npm 的 Node.js"
    exit 1
fi
if ! node_is_supported; then
    show_error "Node.js 版本不兼容，当前 $(node --version)，需要 20.19+、22.12+ 或更新版本"
    exit 1
fi

if ! "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    show_info "当前 Python 缺少 pip，正在安装"
    "$PYTHON_CMD" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
if ! "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    show_error "当前 Python 缺少 pip：$PYTHON_CMD"
    exit 1
fi
show_success "环境检查通过（Python $(python_version "$PYTHON_CMD")，Node $(node --version)）"

# ==================== 清理旧进程 ====================
show_progress "清理旧进程"
for port in 3000 8000 8001 8002 8003 8004 8005; do
    kill_port "$port"
done
show_success "旧服务端口已释放"

# ==================== 基础运行时 ====================
show_progress "检查基础运行时"
if ! "$PYTHON_CMD" -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >/dev/null 2>&1; then
    show_info "首次运行，正在安装基础运行时"
    if ! "$PYTHON_CMD" -m pip install -q -r backend/requirements-core.txt; then
        show_error "基础运行时安装失败，请检查 Python、网络或 requirements-core.txt"
        exit 1
    fi
fi
show_success "基础运行时已就绪"
show_info "本地解析组件将在选择本地路线时按需准备"

if [ ! -d "frontend/node_modules" ]; then
    show_info "首次运行，正在安装前端依赖（约 1-2 分钟）"
    if ! (cd frontend && npm install --silent >/dev/null 2>&1); then
        show_error "前端依赖安装失败，请检查 Node.js、npm 和网络"
        exit 1
    fi
fi
if [ ! -d "frontend/node_modules/rehype-raw" ]; then
    if ! (cd frontend && npm install rehype-raw --silent >/dev/null 2>&1); then
        show_error "rehype-raw 安装失败，请检查 npm 和网络"
        exit 1
    fi
fi
show_success "前端依赖已就绪"

# ==================== 启动服务 ====================
show_progress "启动后端服务"
BACKEND_LOG="$BASE_DIR/backend/backend_startup.log"
nohup "$PYTHON_CMD" backend/app.py >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

BACKEND_READY=0
WAIT_COUNT=0
while [ "$WAIT_COUNT" -lt 90 ]; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if "$PYTHON_CMD" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" >/dev/null 2>&1; then
        BACKEND_READY=1
        break
    fi
    if ! ps -p "$BACKEND_PID" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [ "$BACKEND_READY" = "1" ]; then
    show_success "后端服务启动成功（PID: $BACKEND_PID）"
else
    show_error "后端启动失败或超时"
    if [ -f "$BACKEND_LOG" ]; then
        printf '\n'
        printf "     ${YELLOW}log${NC}   后端错误日志\n"
        tail -80 "$BACKEND_LOG"
    fi
    exit 1
fi

show_progress "启动前端服务"
(sleep 3 && "$PYTHON_CMD" -m webbrowser http://localhost:3000 2>/dev/null || \
 open http://localhost:3000 2>/dev/null || \
 xdg-open http://localhost:3000 2>/dev/null) &

printf '\n'
printf "  ${GREEN}${BOLD}Ready${NC}  ChatPDF 正在运行\n"
printf "  ${MUTED}------------------------------------------------------------${NC}\n"
printf "  ${MUTED}Web${NC}     ${BOLD}http://localhost:3000${NC}\n"
printf "  ${MUTED}API${NC}     ${BOLD}http://127.0.0.1:8000${NC}\n"
printf '\n'
printf "  ${MUTED}浏览器将自动打开。按 ${BOLD}Ctrl+C${NC}${MUTED} 停止所有服务。${NC}\n\n"

cd frontend || exit 1
npm run dev
FRONTEND_EXIT=$?
cd "$BASE_DIR" || exit 1

printf '\n'
show_progress "正在停止服务"
cleanup_services
trap - EXIT INT TERM
show_success "已停止所有服务"
if [ "$FRONTEND_EXIT" -ne 0 ]; then
    show_error "前端服务已异常退出（代码 $FRONTEND_EXIT）"
fi
exit "$FRONTEND_EXIT"
