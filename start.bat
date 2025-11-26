@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
title ChatPDF Pro 启动器
cls

:: ==================== Banner ====================
echo.
echo   ╔═══════════════════════════════════════╗
echo   ║                                       ║
echo   ║     ChatPDF Pro v2.0.2                ║
echo   ║     智能文档助手                      ║
echo   ║                                       ║
echo   ╚═══════════════════════════════════════╝
echo.

:: ==================== 自动更新 ====================
echo   [*] 检查代码更新...

git pull origin main >nul 2>&1
if %errorlevel% equ 0 (
    echo   [✓] 代码已更新到最新版本
) else (
    echo   [✓] 已是最新版本 ^(或更新跳过^)
)
echo.

:: ==================== 环境检查 ====================
echo   [*] 检查运行环境...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [✗] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [✗] 未找到 Node.js，请先安装
    pause
    exit /b
)

echo   [✓] 环境检查通过
echo.

:: ==================== 清理旧进程 ====================
echo   [*] 清理旧进程...

:: 清理端口 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: 清理 Python 缓存
for /r backend %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

echo   [✓] 清理完成
echo.

:: ==================== 安装依赖 ====================
echo   [*] 检查依赖...

:: 后端依赖（静默安装）
pip install -q -r backend\requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo   [✗] 后端依赖安装失败
    pause
    exit /b
)

:: 前端依赖
cd frontend
if not exist "node_modules\" (
    echo   [*] 首次运行，安装前端依赖 (需要1-2分钟)...
    call npm install --silent >nul 2>&1
)
cd ..

echo   [✓] 依赖检查完成
echo.

:: ==================== 启动后端 ====================
echo   [*] 启动后端服务...

start /B python backend\app.py >nul 2>&1
timeout /t 2 /nobreak >nul

:: 检查端口 8000 是否开启
netstat -ano | findstr :8000 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo   [✓] 后端服务启动成功
) else (
    echo   [✗] 后端启动失败
    pause
    exit /b
)

:: ==================== 启动前端 ====================
echo   [*] 启动前端服务...
echo.
echo   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo   🎉 ChatPDF Pro 已启动！
echo.
echo   访问地址: http://localhost:3000
echo   后端API:  http://127.0.0.1:8000
echo.
echo   提示: 浏览器将自动打开，按 Ctrl+C 停止服务
echo.
echo   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

:: 延迟打开浏览器
start /B timeout /t 3 /nobreak >nul && start http://localhost:3000

:: 启动前端（过滤输出）
cd frontend
call npm run dev

:: ==================== 清理 ====================
echo.
echo   [*] 正在停止服务...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq backend*" >nul 2>&1
echo   [✓] 已停止所有服务
pause
