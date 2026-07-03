#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR" || exit 1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

kill_pid_file() {
    local pid_file="$1"
    local label="$2"
    if [ ! -f "$pid_file" ]; then
        return
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid" 2>/dev/null
        echo -e "${GREEN}已停止 ${label} (PID: ${pid})${NC}"
    fi
    rm -f "$pid_file"
}

kill_port() {
    local port="$1"
    local label="$2"
    local pids
    pids="$(lsof -ti :"$port" 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        return
    fi

    echo "$pids" | xargs kill 2>/dev/null
    sleep 1

    local remaining
    remaining="$(lsof -ti :"$port" 2>/dev/null || true)"
    if [ -n "$remaining" ]; then
        echo "$remaining" | xargs kill -9 2>/dev/null
    fi
    echo -e "${GREEN}已清理 ${label} 端口 ${port}${NC}"
}

echo -e "${YELLOW}正在停止 ChatPDF Pro 服务...${NC}"

kill_pid_file "backend.pid" "后端服务"
kill_pid_file "frontend.pid" "前端服务"

pkill -f "python.*backend/app.py" 2>/dev/null

kill_port 8000 "后端"
kill_port 3000 "前端"

echo -e "${GREEN}ChatPDF Pro 已停止${NC}"
