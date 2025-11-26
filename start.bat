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
echo   ║     ChatPDF Pro v2.0.2                ║
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

:: 清理端口 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: 清理 Python 缓存
for /d /r backend %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

echo   [✓] 清理完成

:: ==================== 安装依赖 ====================
echo   [▶] 检查依赖...

:: 后端依赖
python -m pip install -q -r backend/requirements.txt >nul 2>&1
if errorlevel 1 (
    echo   [!] 后端依赖安装出现警告，尝试继续...
)

:: 前端依赖
if not exist "frontend\node_modules" (
    echo   [▶] 首次运行，安装前端依赖 ^(需要1-2分钟^)...
    pushd frontend
    call npm install --silent >nul 2>&1
    popd
    if errorlevel 1 goto NPMFAIL
)

echo   [✓] 依赖检查完成

:: ==================== 启动服务 ====================
echo   [▶] 启动后端服务...

:: 启动后端（后台运行）
start "" /B python backend/app.py >nul 2>&1

:: 等待后端启动（最多20秒）
set "wait_ok=0"
for /l %%i in (1,1,20) do (
    netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
    if not errorlevel 1 (
        set "wait_ok=1"
        goto BACK_OK
    )
    timeout /t 1 /nobreak >nul
)
:BACK_OK

if "%wait_ok%"=="0" goto BACKFAIL
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

:: 清理端口 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
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

:NPMFAIL
echo   [✗] 前端依赖安装失败
echo.
pause
exit /b 1

:BACKFAIL
echo   [✗] 后端启动失败，请检查错误信息
echo.
pause
exit /b 1
