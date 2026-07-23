#!/bin/bash
set -e

echo "启动 ChatPDF 后端服务..."

cd "$(dirname "$0")/../backend"

if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "检查基础运行时..."
if ! python -c "import fastapi, uvicorn, fitz, pdfplumber, faiss, langchain, openai, sentence_transformers" > /dev/null 2>&1; then
    echo "安装基础运行时..."
    python -m pip install -r requirements-core.txt
fi

echo "基础运行时已就绪；本地解析组件会在应用内按需安装"
echo "API 地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
exec python app.py