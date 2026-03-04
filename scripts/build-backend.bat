@echo off
setlocal

echo.
echo [ChatPDF] Backend build (PyInstaller)
echo.

set "BACKEND_DIR=%~dp0..\backend"
cd /d "%BACKEND_DIR%"
if errorlevel 1 (
  echo [X] Cannot enter backend directory: %BACKEND_DIR%
  exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
  echo [X] Python not found. Please install Python 3.10+.
  exit /b 1
)

echo [1/3] Install desktop dependencies...
pip install -r requirements-desktop.txt -q
if errorlevel 1 (
  echo [X] Failed to install requirements-desktop.txt
  exit /b 1
)

pip install pyinstaller -q
if errorlevel 1 (
  echo [X] Failed to install pyinstaller
  exit /b 1
)

echo [2/3] Build backend...
pyinstaller chatpdf.spec --noconfirm --clean
if errorlevel 1 (
  echo [X] PyInstaller build failed
  exit /b 1
)

if exist "dist\chatpdf-backend\desktop_entry.exe" (
  echo [3/3] Build success: backend\dist\chatpdf-backend\
) else (
  echo [X] Build output missing: dist\chatpdf-backend\desktop_entry.exe
  exit /b 1
)

echo.
echo [OK] Backend build completed.
exit /b 0
