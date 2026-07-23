@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

cls

:: 打印 Banner
echo.
echo   ╔═══════════════════════════════════════╗
echo   ║                                       ║
echo   ║     ChatPDF Pro v3.0                  ║
echo   ║     智能文档助手                      ║
echo   ║                                       ║
echo   ╚═══════════════════════════════════════╝
echo.

:: ==================== 自动更新 ====================
echo   [▶] 检查代码更新...

:: 获取当前分支名
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%i"

:: 只在main分支时自动更新
if "%CURRENT_BRANCH%"=="main" (
    git pull origin main >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [✓] 代码已更新到最新版本
    ) else (
        echo   [✓] 已是最新版本 ^(或更新跳过^)
    )
) else (
    if defined CURRENT_BRANCH (
        echo   [✓] 当前在分支 %CURRENT_BRANCH% ^(跳过自动更新^)
    ) else (
        echo   [✓] 跳过更新检查
    )
)

:: ==================== 环境检查 ====================
echo   [▶] 检查运行环境...

where python >nul 2>&1
if errorlevel 1 goto NOPY

where node >nul 2>&1
if errorlevel 1 goto NONODE

echo   [✓] 环境检查通过

:: ==================== 清理旧进程 ====================
echo   [▶] 清理旧进程...

:: 清理 Web 前端、开发后端和桌面后端常用端口
for %%p in (3000 8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING 2^>nul') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

:: 保留 Python 字节码缓存，避免每次启动重新编译全部后端模块。

echo   [✓] 清理完成

:: ==================== 基础运行时 ====================
echo   [▶] 检查基础运行时...

:: MinerU 是默认解析路线。只确保通用后端、预览和检索依赖；
:: 本地 OCR / ODL / YOLO 由用户选择“本地解析”后按需安装。
python -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >nul 2>&1
if errorlevel 1 (
    echo   [▶] 首次安装基础运行时...
    python -m pip install -q -r backend\requirements-core.txt >nul 2>&1
    if errorlevel 1 goto PIPFAIL
)

echo   [✓] 基础运行时已就绪
echo   [i] 本地解析组件将在选择本地路线时按需准备
:: 前端依赖
if not exist "frontend\node_modules" (
    echo   [▶] 首次运行，安装前端依赖 ^(需要1-2分钟^)...
    pushd frontend
    call npm install --silent >nul 2>&1
    popd
    if errorlevel 1 goto NPMFAIL
)

:: 只检查目录，避免每次启动运行一次完整 npm 依赖树解析。
if not exist "frontend\node_modules\rehype-raw" (
    pushd frontend
    call npm install rehype-raw --silent >nul 2>&1
    popd
    if errorlevel 1 goto NPMFAIL
)

echo   [✓] 依赖检查完成

:: ==================== 启动服务 ====================
echo   [▶] 启动后端服务...

:: 启动后端（后台运行，先切到 backend 目录确保模块导入正确）
pushd backend
if exist "backend_startup.log" del /q "backend_startup.log" >nul 2>&1
start "" /B python app.py >backend_startup.log 2>&1
popd

:: 等待后端启动（最多60秒）；发现明确异常时立即退出，不再空等。
set "wait_ok=0"
set "wait_count=0"
:WAIT_BACKEND
curl -fsS --max-time 2 http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto BACK_HEALTHY

if exist "backend\backend_startup.log" (
    findstr /C:"Traceback (most recent call last)" /C:"Application startup failed" /C:"error while attempting to bind" "backend\backend_startup.log" >nul 2>&1
    if not errorlevel 1 goto BACK_OK
)

set /a wait_count+=1
if !wait_count! equ 10 echo   [i] 后端正在加载检索与文档服务...
if !wait_count! equ 30 echo   [i] 后端仍在初始化，请稍候...
if !wait_count! geq 60 goto BACK_OK
timeout /t 1 /nobreak >nul
goto WAIT_BACKEND

:BACK_HEALTHY
set "wait_ok=1"
:BACK_OK

if "!wait_ok!"=="0" (
    echo   [✗] 后端启动超时，错误日志:
    if exist "backend\backend_startup.log" type "backend\backend_startup.log"
    goto BACKFAIL
)
echo   [✓] 后端服务启动成功

echo   [▶] 启动前端服务...
echo.
echo   🎉 ChatPDF Pro 已启动！
echo.
echo     访问地址: http://localhost:3000
echo     后端API:  http://127.0.0.1:8000
echo.
echo     提示: 浏览器将自动打开，关闭此窗口将停止所有服务
echo.
echo   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

:: 延迟打开浏览器
start "" /B timeout /t 3 /nobreak >nul 2>&1 && start "" http://localhost:3000

:: 启动前端（前台运行，保持窗口）
cd frontend
npm run dev

:: ==================== 清理 ====================
echo.
echo   [▶] 正在停止服务...

:: 清理 Web 前端、开发后端和桌面后端常用端口
for %%p in (3000 8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING 2^>nul') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo   [✓] 已停止所有服务
pause
exit /b

:: ==================== 错误处理 ====================
:NOPY
echo   [✗] 未找到 Python，请先安装 Python 3.8+
echo.
pause
exit /b 1

:NONODE
echo   [✗] 未找到 Node.js，请先安装
echo.
pause
exit /b 1

:PIPFAIL
echo   [✗] 基础运行时安装失败，请检查 Python、网络或 requirements-core.txt
echo.
pause
exit /b 1

:NPMFAIL
echo   [✗] 前端依赖安装失败
echo.
pause
exit /b 1

:BACKFAIL
for %%p in (3000 8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING 2^>nul') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)
echo   [✗] 后端启动失败，请检查错误信息
echo.
pause
exit /b 1
