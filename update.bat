@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

echo ========================================
echo   ChatPDF - 一键升级
echo ========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [X] 未找到 Git，无法自动升级
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%i"
if /i not "!CURRENT_BRANCH!"=="main" (
    echo [X] 当前目录在分支 !CURRENT_BRANCH!，一键升级只允许在 main 上执行
    echo     请切换到 main 工作树，或使用 E:\Project\run-chatpdf-main.bat 启动 main 版本。
    pause
    exit /b 1
)

set "HAS_TRACKED_CHANGES="
for /f "delims=" %%i in ('git status --porcelain --untracked-files=normal 2^>nul') do set "HAS_TRACKED_CHANGES=1"
if defined HAS_TRACKED_CHANGES (
    echo [X] 检测到本地源码改动。为避免覆盖或混合版本，请先提交或暂存这些改动。
    pause
    exit /b 1
)

echo [1/4] 拉取 main 最新代码...
git fetch origin main
if errorlevel 1 (
    echo [X] Git fetch 失败，请检查网络
    pause
    exit /b 1
)

git pull --ff-only origin main
if errorlevel 1 (
    echo [X] Git pull 失败：本地 main 不能快进到远端 main
    pause
    exit /b 1
)

echo.
echo [2/4] 检查 Python 环境...
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

echo [3/4] 更新后端依赖...
!PYTHON_CMD! -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [X] 后端依赖更新失败
    pause
    exit /b 1
)

echo.
echo [4/4] 更新前端与桌面端依赖...
call :INSTALL_NPM_DEPS frontend
if errorlevel 1 goto NPMFAIL
call :INSTALL_NPM_DEPS electron
if errorlevel 1 goto NPMFAIL

!PYTHON_CMD! scripts\release_metadata.py --check
if errorlevel 1 (
    echo [X] 版本元数据校验失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 升级完成
echo ========================================
echo 请关闭所有 ChatPDF 窗口后重新运行 start.bat
echo.
pause
exit /b 0

:NPMFAIL
echo [X] npm 依赖更新失败
pause
exit /b 1

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

:INSTALL_NPM_DEPS
where npm >nul 2>&1
if errorlevel 1 exit /b 1
pushd "%~1"
if exist "package-lock.json" (
    call npm ci
) else (
    call npm install
)
set "NPM_EXIT=!errorlevel!"
popd
exit /b !NPM_EXIT!
