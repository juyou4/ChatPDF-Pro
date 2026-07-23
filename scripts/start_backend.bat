@echo off
chcp 65001 >nul
echo 🚀 启动 ChatPDF 后端服务...

cd /d "%~dp0..\backend"

REM 清理占用 8000 端口的旧进程
echo 🧹 清理旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo ✓ 端口清理完成

REM 检查虚拟环境
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
echo 🔧 激活虚拟环境...
call venv\Scripts\activate.bat

REM 仅安装基础运行时；本地解析组件由应用内的按需安装器准备。
echo 📥 检查基础运行时...
python -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >nul 2>&1
if errorlevel 1 (
    pip install -r requirements-core.txt
    if errorlevel 1 (
        echo ❌ 基础运行时安装失败
        exit /b 1
    )
)
echo ✓ 基础运行时已就绪；本地解析组件按需安装
REM 启动服务
echo ✨ 启动服务...
echo 🌐 API地址: http://localhost:8000
echo 📚 API文档: http://localhost:8000/docs
echo.
python app.py

pause
