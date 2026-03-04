@echo off
setlocal

echo.
echo [ChatPDF] Full build pipeline
echo.

set "SCRIPT_DIR=%~dp0"

echo [Step 1/2] Build backend...
call "%SCRIPT_DIR%build-backend.bat"
if errorlevel 1 (
  echo [X] Backend build failed
  exit /b 1
)

echo.
echo [Step 2/2] Build frontend and electron package...
call "%SCRIPT_DIR%build-app.bat"
if errorlevel 1 (
  echo [X] App build failed
  exit /b 1
)

echo.
echo [OK] Build complete. Installer: electron\release\
exit /b 0
