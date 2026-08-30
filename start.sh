#!/usr/bin/env bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR" || exit 1

# 禁止 Git 在启动过程中等待交互式凭据输入；网络异常应尽快反馈并继续使用当前代码。
export GIT_TERMINAL_PROMPT=0

read_app_version() {
    APP_VERSION="$(grep -E '"version"' version.json 2>/dev/null | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
    if [ -z "$APP_VERSION" ]; then
        APP_VERSION="unknown"
    fi
}

read_app_version

# 颜色和样式定义；NO_COLOR 可用于关闭 ANSI 颜色。
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD='\033[1m'
    ACCENT='\033[38;2;217;122;93m'
    GREEN='\033[38;2;90;148;112m'
    YELLOW='\033[38;2;190;139;63m'
    RED='\033[38;2;194;82;82m'
    MUTED='\033[38;2;142;134;128m'
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

STEP_INDEX=0

print_header() {
    printf '\n'
    printf "  ${ACCENT}${BOLD}ChatPDF${NC}  ${MUTED}本地文档工作区${NC}\n"
    printf "  ${MUTED}v%s  ·  后端 8000  ·  前端 3000${NC}\n" "$APP_VERSION"
    printf "  ${MUTED}────────────────────────────────────────${NC}\n\n"
}

show_progress() {
    STEP_INDEX=$((STEP_INDEX + 1))
    printf "\n  ${ACCENT}%02d${NC}  ${BOLD}%s${NC}\n" "$STEP_INDEX" "$1"
}

show_success() {
    printf "     ${GREEN}完成${NC}  %s\n" "$1"
}

show_info() {
    printf "     ${MUTED}说明${NC}  %s\n" "$1"
}

show_error() {
    printf "     ${RED}出错${NC}  %s\n" "$1"
}

show_ready() {
    printf '\n'
    printf "  ${GREEN}${BOLD}就绪${NC}  ChatPDF 正在运行\n"
    printf "  ${MUTED}────────────────────────────────────────${NC}\n"
    printf "  ${MUTED}前端${NC}   ${BOLD}http://localhost:3000${NC}\n"
    printf "  ${MUTED}后端${NC}   ${BOLD}http://127.0.0.1:8000${NC}\n"
    printf '\n'
    printf "  ${MUTED}浏览器将自动打开。按 ${BOLD}Ctrl+C${NC}${MUTED} 停止全部服务。${NC}\n\n"
}

if [ -t 1 ] && [ -n "${TERM:-}" ]; then
    clear
fi

print_header

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

update_from_origin() {
    show_progress "检查代码更新"

    if [ "${CHATPDF_SKIP_UPDATE:-0}" = "1" ]; then
        show_info "已通过 CHATPDF_SKIP_UPDATE=1 跳过更新检查"
        return 0
    fi
    if ! command_exists git || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        show_info "当前目录不是 Git 工作区，跳过自动更新"
        return 0
    fi

    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [ "$current_branch" != "main" ]; then
        if [ -n "$current_branch" ]; then
            show_info "当前在分支 $current_branch（仅 main 自动更新）"
        else
            show_info "无法识别当前分支，跳过自动更新"
        fi
        return 0
    fi

    # 只更新远端跟踪引用，不会覆盖本地改动；这样即使工作树脏，也能告诉用户远端落后多少提交。
    local fetch_output="" fetched=0 attempt
    for attempt in 1 2; do
        if fetch_output="$(git -c http.connectTimeout=10 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=15 fetch --prune origin main 2>&1)"; then
            fetched=1
            break
        fi
        if [ "$attempt" -lt 2 ]; then
            show_info "上游更新连接失败，2 秒后重试（$attempt/2）"
            sleep 2
        fi
    done
    if [ "$fetched" != "1" ]; then
        show_info "无法检查上游更新，将继续使用当前版本"
        [ -n "$fetch_output" ] && printf "     ${YELLOW}Git${NC}  %s\n" "$fetch_output"
        return 0
    fi

    local divergence local_only remote_only
    divergence="$(git rev-list --left-right --count HEAD...origin/main 2>/dev/null || true)"
    local_only=0
    remote_only=0
    if [ -n "$divergence" ]; then
        read -r local_only remote_only <<<"${divergence//$'\t'/ }"
    fi
    if [[ ! "$local_only" =~ ^[0-9]+$ || ! "$remote_only" =~ ^[0-9]+$ ]]; then
        show_info "无法比较本地与 origin/main 的提交，跳过自动合并"
        return 0
    fi

    local current_sha remote_sha
    current_sha="$(git rev-parse --short HEAD 2>/dev/null || true)"
    remote_sha="$(git rev-parse --short origin/main 2>/dev/null || true)"

    if [ "$remote_only" -eq 0 ] && [ "$local_only" -eq 0 ]; then
        show_success "代码已是最新（$current_sha）"
        return 0
    fi

    local working_tree
    working_tree="$(git status --porcelain=v1 --untracked-files=normal 2>/dev/null || true)"
    if [ "$remote_only" -gt 0 ] && [ "$local_only" -eq 0 ]; then
        if [ -n "$working_tree" ]; then
            show_info "上游有 $remote_only 个新提交（$remote_sha），本地有未提交改动，已跳过合并"
            return 0
        fi

        local merge_output=""
        if merge_output="$(git merge --ff-only origin/main 2>&1)"; then
            read_app_version
            current_sha="$(git rev-parse --short HEAD 2>/dev/null || true)"
            show_success "代码已更新到 origin/main（$current_sha）"
        else
            show_info "自动合并失败，将继续使用当前版本（本地 $current_sha，远端 $remote_sha）"
            [ -n "$merge_output" ] && printf "     ${YELLOW}Git${NC}  %s\n" "$merge_output"
        fi
        return 0
    fi

    if [ "$local_only" -gt 0 ] && [ "$remote_only" -eq 0 ]; then
        show_info "本地版本领先 origin/main $local_only 个提交，跳过自动覆盖（$current_sha）"
    else
        show_info "本地与 origin/main 已分叉（本地 +$local_only，远端 +$remote_only），跳过自动合并"
    fi
}

dependency_fingerprint() {
    "$PYTHON_CMD" "$BASE_DIR/scripts/startup_dependency_fingerprint.py" --group "$1" 2>/dev/null || true
}

read_dependency_stamp() {
    local stamp_path="$1" value=""
    if [ -f "$stamp_path" ]; then
        IFS= read -r value < "$stamp_path" || true
    fi
    printf '%s' "$value"
}

write_dependency_stamp() {
    local stamp_path="$1" fingerprint="$2"
    mkdir -p "$BASE_DIR/data"
    if ! printf '%s\n' "$fingerprint" >"$stamp_path" 2>/dev/null; then
        show_info "无法写入依赖指纹，下次启动会再次校验依赖"
    fi
}

sync_python_dependencies() {
    local fingerprint stored stamp_path need_sync=0
    stamp_path="$BASE_DIR/data/.startup-python-deps.sha256"
    fingerprint="$(dependency_fingerprint python)"
    stored="$(read_dependency_stamp "$stamp_path")"
    if [ -z "$fingerprint" ] || [ "$fingerprint" != "$stored" ]; then
        need_sync=1
    fi
    if ! "$PYTHON_CMD" -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >/dev/null 2>&1; then
        need_sync=1
    fi

    if [ "$need_sync" = "1" ]; then
        show_info "后端依赖清单有变化或尚未安装，正在同步 requirements-core.txt"
        if ! "$PYTHON_CMD" -m pip install --disable-pip-version-check -q -r backend/requirements-core.txt; then
            show_error "基础运行时安装失败，请检查 Python、网络或 requirements-core.txt"
            return 1
        fi
        [ -n "$fingerprint" ] && write_dependency_stamp "$stamp_path" "$fingerprint"
    fi
    return 0
}

sync_frontend_dependencies() {
    local fingerprint stored stamp_path need_sync=0
    stamp_path="$BASE_DIR/data/.startup-frontend-deps.sha256"
    fingerprint="$(dependency_fingerprint frontend)"
    stored="$(read_dependency_stamp "$stamp_path")"
    if [ -z "$fingerprint" ] || [ "$fingerprint" != "$stored" ]; then
        need_sync=1
    fi
    if [ ! -x "frontend/node_modules/.bin/vite" ] || [ ! -d "frontend/node_modules/rehype-raw" ]; then
        need_sync=1
    fi

    if [ "$need_sync" = "1" ]; then
        if [ -f "frontend/package-lock.json" ]; then
            show_info "前端依赖清单有变化或尚未安装，正在按 package-lock.json 同步"
            if ! (cd frontend && npm ci --silent); then
                show_error "前端依赖安装失败，请检查 Node.js、npm 和网络"
                return 1
            fi
        else
            show_info "未找到 package-lock.json，正在安装前端依赖"
            if ! (cd frontend && npm install --silent); then
                show_error "前端依赖安装失败，请检查 Node.js、npm 和网络"
                return 1
            fi
        fi
        if [ ! -x "frontend/node_modules/.bin/vite" ] || [ ! -d "frontend/node_modules/rehype-raw" ]; then
            show_error "前端依赖安装后仍缺少 Vite 或 rehype-raw"
            return 1
        fi
        [ -n "$fingerprint" ] && write_dependency_stamp "$stamp_path" "$fingerprint"
    fi
    return 0
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
    # 兼容旧版本残留的根目录 PID 文件，并将新文件固定放在被忽略的 data/ 下。
    rm -f "$BASE_DIR/backend.pid" "${BACKEND_PID_FILE:-$BASE_DIR/data/.backend.pid}"
    for port in 3000 8000 8001 8002 8003 8004 8005; do
        kill_port "$port"
    done
}

# ==================== 自动更新 ====================
update_from_origin
# git merge 可能刚刚更新了 version.json；重新读取，避免标题落后一轮。
read_app_version
show_info "当前代码版本 v$APP_VERSION"

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
if ! sync_python_dependencies; then
    exit 1
fi
show_success "基础运行时已就绪"
show_info "本地解析组件将在选择本地路线时按需准备"

# package.json/package-lock.json 变更时才重建前端依赖，避免每次启动都重新安装。
if ! sync_frontend_dependencies; then
    exit 1
fi
show_success "前端依赖已就绪"

# ==================== 启动服务 ====================
show_progress "启动后端服务"
mkdir -p "$BASE_DIR/data/logs"
BACKEND_LOG="$BASE_DIR/data/logs/backend_startup.log"
nohup "$PYTHON_CMD" backend/app.py >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
BACKEND_PID_FILE="$BASE_DIR/data/.backend.pid"
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
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
    if [ "$WAIT_COUNT" -eq 10 ]; then
        show_info "后端正在加载检索与文档服务"
    elif [ "$WAIT_COUNT" -eq 30 ]; then
        show_info "后端仍在初始化，请稍候"
    fi
    sleep 1
done

if [ "$BACKEND_READY" = "1" ]; then
    show_success "后端服务启动成功（PID: $BACKEND_PID）"
else
    show_error "后端启动失败或超时"
    if [ -f "$BACKEND_LOG" ]; then
        printf '\n'
        printf "     ${YELLOW}日志${NC}  后端错误日志\n"
        tail -80 "$BACKEND_LOG"
    fi
    exit 1
fi

show_progress "启动前端服务"
(sleep 3 && "$PYTHON_CMD" -m webbrowser http://localhost:3000 2>/dev/null || \
 open http://localhost:3000 2>/dev/null || \
 xdg-open http://localhost:3000 2>/dev/null) &

show_ready

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
