@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "BACKEND_DIR=%~dp0..\backend"
cd /d "%BACKEND_DIR%"

echo 启动 ChatPDF 后端服务

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)

call :SELECT_PYTHON
if errorlevel 1 (
    echo [X] 未找到 Python 3.10+
    pause
    exit /b 1
)

!PYTHON_CMD! -m pip --version >nul 2>&1
if errorlevel 1 (
    !PYTHON_CMD! -m ensurepip --upgrade >nul 2>&1
)
!PYTHON_CMD! -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [X] 当前 Python 缺少 pip
    pause
    exit /b 1
)

echo 检查基础运行时...
!PYTHON_CMD! -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >nul 2>&1
if errorlevel 1 (
    !PYTHON_CMD! -m pip install -r requirements-core.txt
    if errorlevel 1 (
        echo [X] 基础运行时安装失败
        pause
        exit /b 1
    )
)

echo API 地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
!PYTHON_CMD! app.py
pause
exit /b 0

:SELECT_PYTHON
set "PYTHON_CMD="
if defined PYTHON (
    %PYTHON% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=%PYTHON%"
        exit /b 0
    )
)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)
exit /b 1
