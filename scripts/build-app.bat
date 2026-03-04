@echo off
setlocal

echo.
echo [ChatPDF] Frontend + Electron build
echo.

set "PROJECT_DIR=%~dp0.."

echo [1/3] Build frontend...
cd /d "%PROJECT_DIR%\frontend"
if errorlevel 1 (
  echo [X] Cannot enter frontend directory
  exit /b 1
)

call npm install --silent
if errorlevel 1 (
  echo [X] npm install failed in frontend
  exit /b 1
)

call npm run build
if errorlevel 1 (
  echo [X] Frontend build failed
  exit /b 1
)

echo [2/3] Build electron sources...
cd /d "%PROJECT_DIR%\electron"
if errorlevel 1 (
  echo [X] Cannot enter electron directory
  exit /b 1
)

call npm install --silent
if errorlevel 1 (
  echo [X] npm install failed in electron
  exit /b 1
)

call npm run build
if errorlevel 1 (
  echo [X] Electron TypeScript build failed
  exit /b 1
)

echo [3/3] Package app (Windows)...
call npm run package:win
if errorlevel 1 (
  echo [X] Packaging failed
  exit /b 1
)

echo.
echo [OK] App build completed. Output: electron\release\
exit /b 0
