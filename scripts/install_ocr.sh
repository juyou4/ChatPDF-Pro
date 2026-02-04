#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║     ChatPDF OCR 依赖安装              ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

# 检测操作系统
OS_TYPE=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
else
    OS_TYPE="unknown"
fi

echo -e "${BLUE}  ▶${NC} 检测到操作系统: $OS_TYPE"
echo ""

# 安装 Python OCR 库
echo -e "${BLUE}  ▶${NC} 安装 Python OCR 库..."
pip3 install pdf2image pytesseract pillow

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓${NC} Python OCR 库安装完成"
else
    echo -e "${RED}  ✗${NC} Python 库安装失败"
    exit 1
fi

echo ""

# 安装系统依赖
echo -e "${BLUE}  ▶${NC} 安装系统 OCR 依赖..."

if [ "$OS_TYPE" == "macos" ]; then
    # macOS: 使用 Homebrew
    if command -v brew &> /dev/null; then
        echo -e "${BLUE}  ▶${NC} 使用 Homebrew 安装 Tesseract 和 Poppler..."
        brew install tesseract tesseract-lang poppler
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓${NC} Tesseract 和 Poppler 安装成功"
        else
            echo -e "${RED}  ✗${NC} 安装失败"
        fi
    else
        echo -e "${YELLOW}  [!] 未找到 Homebrew，请先安装:${NC}"
        echo "      /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo "      然后运行: brew install tesseract tesseract-lang poppler"
    fi

elif [ "$OS_TYPE" == "linux" ]; then
    # Linux: 使用 apt
    if command -v apt-get &> /dev/null; then
        echo -e "${BLUE}  ▶${NC} 使用 apt 安装 Tesseract 和 Poppler..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng poppler-utils
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓${NC} Tesseract 和 Poppler 安装成功"
        else
            echo -e "${RED}  ✗${NC} 安装失败"
        fi
    elif command -v yum &> /dev/null; then
        echo -e "${BLUE}  ▶${NC} 使用 yum 安装 Tesseract 和 Poppler..."
        sudo yum install -y tesseract tesseract-langpack-chi_sim poppler-utils
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓${NC} Tesseract 和 Poppler 安装成功"
        else
            echo -e "${RED}  ✗${NC} 安装失败"
        fi
    else
        echo -e "${YELLOW}  [!] 未找到包管理器，请手动安装 tesseract-ocr 和 poppler-utils${NC}"
    fi
else
    echo -e "${YELLOW}  [!] 未知操作系统，请手动安装 Tesseract OCR 和 Poppler${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 验证安装
echo -e "${BLUE}  ▶${NC} 验证 OCR 安装..."

# 检查 Tesseract
if command -v tesseract &> /dev/null; then
    echo -e "${GREEN}  ✓${NC} Tesseract 已安装: $(tesseract --version 2>&1 | head -1)"
else
    echo -e "${RED}  ✗${NC} Tesseract 未安装"
fi

# 检查 Poppler
if command -v pdftoppm &> /dev/null; then
    echo -e "${GREEN}  ✓${NC} Poppler 已安装"
else
    echo -e "${RED}  ✗${NC} Poppler 未安装"
fi

# 检查 Python 库
python3 -c "from pdf2image import convert_from_path; import pytesseract; print('  ✓ Python OCR 库导入成功')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗${NC} Python OCR 库导入失败"
fi

echo ""
echo -e "${GREEN}  OCR 安装完成！${NC}"
echo ""
