@echo off
chcp 65001 >nul
echo.
echo   ╔═══════════════════════════════════════╗
echo   ║     ChatPDF OCR 依赖安装              ║
echo   ╚═══════════════════════════════════════╝
echo.

:: 安装 Python OCR 库
echo   [▶] 安装 Python OCR 库...
python -m pip install pdf2image pytesseract pillow
if errorlevel 1 (
    echo   [✗] Python 库安装失败
    pause
    exit /b 1
)
echo   [✓] Python OCR 库安装完成

:: 检查 Tesseract
echo.
echo   [▶] 检查 Tesseract-OCR...
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ╔═══════════════════════════════════════════════════════════╗
    echo   ║  Tesseract-OCR 未安装！请手动安装：                       ║
    echo   ║                                                           ║
    echo   ║  1. 下载地址:                                             ║
    echo   ║     https://github.com/UB-Mannheim/tesseract/wiki         ║
    echo   ║                                                           ║
    echo   ║  2. 安装时勾选 "Chinese Simplified" 语言包                ║
    echo   ║                                                           ║
    echo   ║  3. 安装完成后，将 Tesseract 添加到系统 PATH:             ║
    echo   ║     默认路径: C:\Program Files\Tesseract-OCR              ║
    echo   ║                                                           ║
    echo   ║  4. 重新运行此脚本验证安装                                ║
    echo   ╚═══════════════════════════════════════════════════════════╝
    echo.
    
    :: 尝试打开下载页面
    echo   是否打开 Tesseract 下载页面？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        start "" "https://github.com/UB-Mannheim/tesseract/wiki"
    )
) else (
    echo   [✓] Tesseract-OCR 已安装
    tesseract --version 2>nul | findstr /i "tesseract"
)

:: 安装 poppler (pdf2image 依赖)
echo.
echo   [▶] 检查 Poppler...
where pdftoppm >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ╔═══════════════════════════════════════════════════════════╗
    echo   ║  Poppler 未安装！pdf2image 需要它来转换 PDF               ║
    echo   ║                                                           ║
    echo   ║  安装方法:                                                ║
    echo   ║  1. 下载: https://github.com/oschwartz10612/poppler-windows/releases ║
    echo   ║  2. 解压到 C:\Program Files\poppler                       ║
    echo   ║  3. 将 C:\Program Files\poppler\bin 添加到系统 PATH       ║
    echo   ╚═══════════════════════════════════════════════════════════╝
    echo.
    
    echo   是否打开 Poppler 下载页面？ (Y/N)
    set /p choice2=
    if /i "%choice2%"=="Y" (
        start "" "https://github.com/oschwartz10612/poppler-windows/releases"
    )
) else (
    echo   [✓] Poppler 已安装
)

echo.
echo   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

:: 验证安装
echo   [▶] 验证 OCR 安装...
python -c "from pdf2image import convert_from_path; import pytesseract; print('  [✓] OCR 库导入成功')" 2>nul
if errorlevel 1 (
    echo   [✗] OCR 库导入失败，请检查安装
) else (
    echo   [✓] OCR 安装验证通过！
)

echo.
pause
