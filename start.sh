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
  ║     ChatPDF Pro v2.0.2                ║
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

# ==================== 安装 OCR 依赖 ====================
show_progress "检查 OCR 依赖..."

# 检查 pdf2image 是否已安装
if ! python3 -c "import pdf2image" 2>/dev/null; then
    show_progress "安装 OCR Python 库..."
    pip3 install -q pdf2image pytesseract pillow 2>/dev/null
fi

# OCR 工具目录
OCR_DIR="$(pwd)/ocr_tools"

# 检查并安装 Tesseract
install_tesseract() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            show_progress "安装 Tesseract (Homebrew)..."
            brew install tesseract tesseract-lang poppler 2>/dev/null
            return $?
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            show_progress "安装 Tesseract (apt)..."
            sudo apt-get update -qq
            sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng poppler-utils 2>/dev/null
            return $?
        elif command -v yum &> /dev/null; then
            show_progress "安装 Tesseract (yum)..."
            sudo yum install -y -q tesseract tesseract-langpack-chi_sim poppler-utils 2>/dev/null
            return $?
        elif command -v pacman &> /dev/null; then
            show_progress "安装 Tesseract (pacman)..."
            sudo pacman -S --noconfirm tesseract tesseract-data-chi_sim tesseract-data-eng poppler 2>/dev/null
            return $?
        fi
    fi
    return 1
}

# 检查 Tesseract
if ! command -v tesseract &> /dev/null; then
    install_tesseract
    if [ $? -eq 0 ]; then
        show_success "Tesseract 安装成功"
    else
        echo -e "${YELLOW}  [!] Tesseract 自动安装失败，扫描版PDF将无法识别${NC}"
        echo -e "${YELLOW}  [!] 请手动安装: brew install tesseract (macOS) 或 apt install tesseract-ocr (Linux)${NC}"
    fi
else
    show_success "Tesseract 已安装"
fi

# 检查 Poppler
if ! command -v pdftoppm &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        brew install poppler 2>/dev/null
    fi
fi

show_success "OCR 依赖检查完成"

# ==================== DocLayout-YOLO 模型 ====================
show_progress "检查图表检测模型..."

MODEL_DIR="$(pwd)/backend/models"
MODEL_FILE="$MODEL_DIR/doclayout_yolo_docstructbench_imgsz1280.pt"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    show_success "DocLayout-YOLO 模型已存在"
else
    show_progress "下载 DocLayout-YOLO 模型 (~30MB)..."
    # 优先尝试 HF 镜像（国内加速）
    HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='opendatalab/PDF-Extract-Kit-1.0', filename='models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt')
shutil.copy2(p, '$MODEL_FILE')
" > /dev/null 2>&1
    if [ -f "$MODEL_FILE" ]; then
        show_success "DocLayout-YOLO 模型下载成功"
    else
        echo -e "${YELLOW}  [!] 模型自动下载失败（可能是网络问题）${NC}"
        echo ""
        echo -e "${YELLOW}  ┌─ 手动下载方法 ──────────────────────────────────────┐${NC}"
        echo -e "${YELLOW}  │                                                       │${NC}"
        echo -e "${YELLOW}  │  1. 打开 ModelScope（国内推荐）:                     │${NC}"
        echo -e "${YELLOW}  │     https://modelscope.cn/models/                    │${NC}"
        echo -e "${YELLOW}  │     opendatalab/PDF-Extract-Kit                      │${NC}"
        echo -e "${YELLOW}  │                                                       │${NC}"
        echo -e "${YELLOW}  │  2. 下载文件:                                        │${NC}"
        echo -e "${YELLOW}  │     models/Layout/YOLO/                              │${NC}"
        echo -e "${YELLOW}  │     doclayout_yolo_docstructbench_imgsz1280_2501.pt  │${NC}"
        echo -e "${YELLOW}  │                                                       │${NC}"
        echo -e "${YELLOW}  │  3. 重命名并放到:                                    │${NC}"
        echo -e "${YELLOW}  │     backend/models/                                  │${NC}"
        echo -e "${YELLOW}  │     doclayout_yolo_docstructbench_imgsz1280.pt       │${NC}"
        echo -e "${YELLOW}  │                                                       │${NC}"
        echo -e "${YELLOW}  │  不影响正常使用，图表解读将退化为基础模式             │${NC}"
        echo -e "${YELLOW}  └───────────────────────────────────────────────────────┘${NC}"
        echo ""
    fi
fi

# ==================== ODL 去脏解析器 ====================
show_progress "检查 OpenDataLoader PDF 解析器..."

# 检查并安装 opendataloader_pdf Python 包
if ! python3 -c "import opendataloader_pdf" 2>/dev/null; then
    show_progress "安装 opendataloader-pdf..."
    pip3 install -q opendataloader-pdf 2>/dev/null
    if python3 -c "import opendataloader_pdf" 2>/dev/null; then
        show_success "opendataloader-pdf 安装成功"
    else
        echo -e "${YELLOW}  [!] opendataloader-pdf 安装失败，将使用 pdfplumber 解析${NC}"
    fi
else
    show_success "opendataloader-pdf 已安装"
fi

# 检查 Java（ODL 运行所需）
if ! command -v java &> /dev/null; then
    # 尝试自动安装 Java
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        show_progress "安装 Java (Homebrew)..."
        brew install --cask temurin 2>/dev/null || brew install openjdk 2>/dev/null
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            show_progress "安装 Java (apt)..."
            sudo apt-get install -y -qq default-jre-headless 2>/dev/null
        elif command -v yum &> /dev/null; then
            show_progress "安装 Java (yum)..."
            sudo yum install -y -q java-11-openjdk-headless 2>/dev/null
        elif command -v pacman &> /dev/null; then
            show_progress "安装 Java (pacman)..."
            sudo pacman -S --noconfirm jre-openjdk-headless 2>/dev/null
        fi
    fi
    # 再次检查
    if ! command -v java &> /dev/null; then
        echo -e "${YELLOW}  [!] Java 未安装，ODL 去脏功能将自动降级为 pdfplumber${NC}"
        echo -e "${YELLOW}  [!] 如需完整去脏功能，请安装 Java 11+: https://adoptium.net${NC}"
    else
        show_success "Java 已安装，ODL 去脏功能已启用"
    fi
else
    show_success "Java 已安装，ODL 去脏功能已启用"
fi

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
