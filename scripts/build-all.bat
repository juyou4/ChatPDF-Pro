@echo off
setlocal

echo.
echo [ChatPDF] Full build pipeline
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "PYTHON_EXE=python"
if defined PYTHON set "PYTHON_EXE=%PYTHON%"

echo [Step 1/3] Write build metadata...
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [X] Python 3.10+ not found. Set PYTHON to the interpreter used by ChatPDF.
  exit /b 1
)
"%PYTHON_EXE%" "%SCRIPT_DIR%release_metadata.py" --write-build-info --check
if errorlevel 1 (
  echo [X] Release metadata validation failed
  exit /b 1
)

echo [Step 2/3] Build backend...
call "%SCRIPT_DIR%build-backend.bat"
if errorlevel 1 (
  echo [X] Backend build failed
  exit /b 1
)

echo.
echo [Step 3/3] Build frontend and electron package...
call "%SCRIPT_DIR%build-app.bat"
if errorlevel 1 (
  echo [X] App build failed
  exit /b 1
)

echo.
echo [OK] Build complete. Installer: electron\release\
exit /b 0
