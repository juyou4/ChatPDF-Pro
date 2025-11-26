#!/bin/bash

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
cat << "EOF"
  ╔═══════════════════════════════════════╗
  ║                                       ║
  ║     ChatPDF Pro v2.0.2               ║
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

# ==================== 自动更新 ====================
show_progress "检查代码更新..."

# 获取当前分支名
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# 只在main分支时自动更新，其他分支跳过
if [ "$CURRENT_BRANCH" = "main" ]; then
    git pull origin main > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        show_success "代码已更新到最新版本"
    else
        show_success "已是最新版本 (或更新跳过)"
    fi
else
    show_success "当前在分支 $CURRENT_BRANCH (跳过自动更新)"
fi

# ==================== 环境检查 ====================
show_progress "检查运行环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    show_error "未找到 Python3，请先安装"
    exit 1
fi

# 检查 Node.js
if ! command -v node &> /dev/null; then
    show_error "未找到 Node.js，请先安装"
    exit 1
fi

show_success "环境检查通过"

# ==================== 清理旧进程 ====================
show_progress "清理旧进程..."

# 清理端口 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null
pkill -f "python.*backend/app.py" 2>/dev/null
find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

show_success "清理完成"

# ==================== 安装依赖 ====================
show_progress "检查依赖..."

# 后端依赖（静默安装）
pip3 install -q -r backend/requirements.txt 2>&1 | grep -i "error" || true

# 前端依赖
cd frontend
if [ ! -d "node_modules" ]; then
    show_progress "首次运行，安装前端依赖 (需要1-2分钟)..."
    npm install --silent > /dev/null 2>&1
fi
cd ..

show_success "依赖检查完成"

# ==================== 启动服务 ====================
show_progress "启动后端服务..."
nohup python3 backend/app.py > /dev/null 2>&1 &
BACKEND_PID=$!
sleep 2

# 检查后端是否成功启动
if ps -p $BACKEND_PID > /dev/null; then
    show_success "后端服务启动成功 (PID: $BACKEND_PID)"
else
    show_error "后端启动失败"
    exit 1
fi

show_progress "启动前端服务..."
cd frontend

# 延迟打开浏览器（等待前端服务完全启动）
(sleep 3 && python3 -m webbrowser http://localhost:3000 2>/dev/null || \
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

# 启动前端（过滤大部分输出，只保留关键信息）
npm run dev 2>&1 | grep -E "Local:|Network:|ready in|error|Error|ERROR" || npm run dev

# ==================== 清理 ====================
echo ""
show_progress "正在停止服务..."
kill $BACKEND_PID 2>/dev/null
show_success "已停止所有服务"
