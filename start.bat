@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

set "APP_VERSION=unknown"
for /f "tokens=2 delims=:," %%v in ('findstr /i /c:"version" version.json 2^>nul ^| findstr /v /i /c:"schema_version"') do set "APP_VERSION=%%~v"
set "APP_VERSION=!APP_VERSION: =!"
set "APP_VERSION=!APP_VERSION:"=!"

set "ACCENT="
set "SUCCESS="
set "DANGER="
set "MUTED="
set "BOLD="
set "RESET="
set "STEP_INDEX=0"
if not defined NO_COLOR (
    for /f "delims=#" %%e in ('"prompt #$E# & for %%e in (1) do rem"') do set "ESC=%%e"
    set "ACCENT=!ESC![38;2;217;122;93m"
    set "SUCCESS=!ESC![38;2;90;148;112m"
    set "DANGER=!ESC![38;2;194;82;82m"
    set "MUTED=!ESC![38;2;142;134;128m"
    set "BOLD=!ESC![1m"
    set "RESET=!ESC![0m"
)

title ChatPDF
cls
call :PRINT_HEADER

:: ==================== 自动更新 ====================
call :PRINT_STEP "检查代码更新"
set "CURRENT_BRANCH="
where git >nul 2>&1
if errorlevel 1 (
    call :PRINT_INFO "未找到 Git，跳过自动更新"
) else (
    for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%i"
    if /i "!CURRENT_BRANCH!"=="main" (
        set "HAS_TRACKED_CHANGES="
        for /f "delims=" %%i in ('git status --porcelain --untracked-files=normal 2^>nul') do set "HAS_TRACKED_CHANGES=1"
        if defined HAS_TRACKED_CHANGES (
            call :PRINT_INFO "检测到本地代码改动，为避免覆盖已跳过自动更新"
        ) else (
            git pull --ff-only origin main >nul 2>&1
            if !errorlevel! equ 0 (
                call :PRINT_SUCCESS "代码已更新到 main 最新版本"
            ) else (
                call :PRINT_INFO "自动更新未完成，将继续使用当前版本"
            )
        )
    ) else (
        if defined CURRENT_BRANCH (
            call :PRINT_INFO "当前在分支 !CURRENT_BRANCH!（仅 main 自动更新）"
        ) else (
            call :PRINT_INFO "当前目录不是 Git 工作区，跳过自动更新"
        )
    )
)

:: ==================== 环境检查 ====================
call :PRINT_STEP "检查运行环境"
call :SELECT_PYTHON
if errorlevel 1 goto NOPY

where node >nul 2>&1
if errorlevel 1 goto NONODE
where npm >nul 2>&1
if errorlevel 1 goto NONPM

node -e "const [a,b,c]=process.versions.node.split('.').map(Number);const ok=(a===20&&(b>19||(b===19&&c>=0)))||(a===22&&(b>12||(b===12&&c>=0)))||a>22;process.exit(ok?0:1)" >nul 2>&1
if errorlevel 1 goto BADNODE

!PYTHON_CMD! -m pip --version >nul 2>&1
if errorlevel 1 (
    call :PRINT_INFO "当前 Python 缺少 pip，正在安装"
    !PYTHON_CMD! -m ensurepip --upgrade >nul 2>&1
)
!PYTHON_CMD! -m pip --version >nul 2>&1
if errorlevel 1 goto NOPIP

for /f "tokens=*" %%v in ('!PYTHON_CMD! -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "PYTHON_VERSION=%%v"
for /f "tokens=*" %%v in ('node --version') do set "NODE_VERSION=%%v"
call :PRINT_SUCCESS "环境检查通过（Python !PYTHON_VERSION!，Node !NODE_VERSION!）"

:: ==================== 清理旧进程 ====================
call :PRINT_STEP "清理旧进程"
call :KILL_SERVICE_PORTS
call :PRINT_SUCCESS "旧服务端口已释放"

:: ==================== 基础运行时 ====================
call :PRINT_STEP "检查基础运行时"
!PYTHON_CMD! -c "import importlib.util as u,sys; names=('fastapi','uvicorn','fitz','pdfplumber','faiss','langchain','openai','sentence_transformers'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >nul 2>&1
if errorlevel 1 (
    call :PRINT_INFO "首次运行，正在安装基础运行时"
    !PYTHON_CMD! -m pip install -q -r backend\requirements-core.txt >nul 2>&1
    if errorlevel 1 goto PIPFAIL
)
call :PRINT_SUCCESS "基础运行时已就绪"
call :PRINT_INFO "本地解析组件将在选择本地路线时按需准备"

:: 前端依赖。只看 node_modules 目录不够：半残安装会留下包、丢掉 .bin，
:: 随后 npm run dev 会变成 Windows 找不到 vite。
if not exist "frontend\node_modules\.bin\vite.cmd" (
    if exist "frontend\node_modules" (
        call :PRINT_INFO "前端命令入口缺失，正在重装依赖"
    ) else (
        call :PRINT_INFO "首次运行，正在安装前端依赖（约 1-2 分钟）"
    )
    pushd frontend
    call npm install --silent >nul 2>&1
    set "NPM_EXIT=!errorlevel!"
    popd
    if not "!NPM_EXIT!"=="0" goto NPMFAIL
)
if not exist "frontend\node_modules\rehype-raw" (
    pushd frontend
    call npm install rehype-raw --silent >nul 2>&1
    set "NPM_EXIT=!errorlevel!"
    popd
    if not "!NPM_EXIT!"=="0" goto NPMFAIL
)
if not exist "frontend\node_modules\.bin\vite.cmd" goto NPMFAIL
call :PRINT_SUCCESS "前端依赖已就绪"

:: ==================== 启动服务 ====================
call :PRINT_STEP "启动后端服务"
set "STARTUP_LOG_DIR=%BASE_DIR%data\logs"
if not exist "!STARTUP_LOG_DIR!" mkdir "!STARTUP_LOG_DIR!" >nul 2>&1
set "BACKEND_STARTUP_LOG=!STARTUP_LOG_DIR!\backend_startup.log"
if exist "!BACKEND_STARTUP_LOG!" del /q "!BACKEND_STARTUP_LOG!" >nul 2>&1
pushd backend
start "" /B !PYTHON_CMD! app.py >"!BACKEND_STARTUP_LOG!" 2>&1
popd

:: 等待后端启动（最多 60 秒）；发现明确异常时立即退出。
set "WAIT_OK=0"
set "WAIT_COUNT=0"
:WAIT_BACKEND
!PYTHON_CMD! -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" >nul 2>&1
if not errorlevel 1 goto BACK_HEALTHY

if exist "!BACKEND_STARTUP_LOG!" (
    findstr /C:"Traceback (most recent call last)" /C:"Application startup failed" /C:"error while attempting to bind" "!BACKEND_STARTUP_LOG!" >nul 2>&1
    if not errorlevel 1 goto BACK_CHECK_DONE
)

set /a WAIT_COUNT+=1
if !WAIT_COUNT! equ 10 call :PRINT_INFO "后端正在加载检索与文档服务"
if !WAIT_COUNT! equ 30 call :PRINT_INFO "后端仍在初始化，请稍候"
if !WAIT_COUNT! geq 60 goto BACK_CHECK_DONE
timeout /t 1 /nobreak >nul
goto WAIT_BACKEND

:BACK_HEALTHY
set "WAIT_OK=1"
:BACK_CHECK_DONE
if "!WAIT_OK!"=="0" (
    call :PRINT_ERROR "后端启动失败或超时，错误日志如下"
    if exist "!BACKEND_STARTUP_LOG!" type "!BACKEND_STARTUP_LOG!"
    goto BACKFAIL
)
call :PRINT_SUCCESS "后端服务启动成功"

call :PRINT_STEP "启动前端服务"
call :PRINT_READY

:: 真正延迟 3 秒后再打开浏览器；Start-Process 不阻塞当前窗口。
start "" /B powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:3000'" >nul 2>&1

:: npm.cmd 必须通过 call 执行，否则控制流不会回到清理段。
pushd frontend
call npm run dev
set "FRONTEND_EXIT=!errorlevel!"
popd

:: ==================== 清理 ====================
echo.
call :PRINT_STEP "正在停止服务"
call :KILL_SERVICE_PORTS
call :PRINT_SUCCESS "已停止所有服务"
if not "!FRONTEND_EXIT!"=="0" call :PRINT_ERROR "前端服务已异常退出（代码 !FRONTEND_EXIT!）"
pause
exit /b !FRONTEND_EXIT!

:: ==================== 错误处理 ====================
:NOPY
call :PRINT_ERROR "未找到 Python 3.10+，请先安装或配置 py/python"
goto ERROR_EXIT

:NONODE
call :PRINT_ERROR "未找到 Node.js，请先安装 Node.js 20.19+ 或 22.12+"
goto ERROR_EXIT

:NONPM
call :PRINT_ERROR "未找到 npm，请重新安装包含 npm 的 Node.js"
goto ERROR_EXIT

:BADNODE
for /f "tokens=*" %%v in ('node --version 2^>nul') do set "NODE_VERSION=%%v"
call :PRINT_ERROR "Node.js !NODE_VERSION! 不兼容，需要 20.19+、22.12+ 或更新版本"
goto ERROR_EXIT

:NOPIP
call :PRINT_ERROR "当前 Python 缺少 pip，且自动安装失败"
goto ERROR_EXIT

:PIPFAIL
call :PRINT_ERROR "基础运行时安装失败，请检查 Python、网络或 requirements-core.txt"
goto ERROR_EXIT

:NPMFAIL
call :PRINT_ERROR "前端依赖安装失败，请检查 Node.js、npm 和网络"
goto ERROR_EXIT

:BACKFAIL
call :KILL_SERVICE_PORTS
call :PRINT_ERROR "后端启动失败，请检查上方错误信息"
goto ERROR_EXIT

:ERROR_EXIT
echo.
pause
exit /b 1

:: ==================== 运行时组件 ====================
:SELECT_PYTHON
set "PYTHON_CMD="
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

:KILL_SERVICE_PORTS
for %%p in (3000 8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING 2^>nul') do taskkill /F /PID %%a >nul 2>&1
)
exit /b 0

:: ==================== 界面组件 ====================
:PRINT_HEADER
echo.
echo   !ACCENT!!BOLD!ChatPDF!RESET!  !MUTED!本地文档工作区!RESET!
echo   !MUTED!v!APP_VERSION!  ·  后端 8000  ·  前端 3000!RESET!
echo   !MUTED!────────────────────────────────────────!RESET!
echo.
exit /b 0

:PRINT_STEP
set /a STEP_INDEX+=1
set "STEP_LABEL=0!STEP_INDEX!"
set "STEP_LABEL=!STEP_LABEL:~-2!"
echo.
echo   !ACCENT!!STEP_LABEL!!RESET!  !BOLD!%~1!RESET!
exit /b 0

:PRINT_SUCCESS
echo      !SUCCESS!完成!RESET!  %~1
exit /b 0

:PRINT_INFO
echo      !MUTED!说明!RESET!  %~1
exit /b 0

:PRINT_ERROR
echo      !DANGER!出错!RESET!  %~1
exit /b 0

:PRINT_READY
echo.
echo   !SUCCESS!!BOLD!就绪!RESET!  ChatPDF 正在运行
echo   !MUTED!────────────────────────────────────────!RESET!
echo   !MUTED!前端!RESET!   !BOLD!http://localhost:3000!RESET!
echo   !MUTED!后端!RESET!   !BOLD!http://127.0.0.1:8000!RESET!
echo.
echo   !MUTED!浏览器将自动打开。关闭此窗口会停止全部服务。!RESET!
echo.
exit /b 0
