# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件 - ChatPDF 桌面后端打包配置

使用方式：
  cd backend
  pip install -r requirements-desktop.txt
  pip install pyinstaller
  pyinstaller chatpdf.spec

目标：onedir 模式，体积 ≤ 250MB
"""

import sys
import os

block_cipher = None

# 项目根目录
backend_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['desktop_entry.py'],
    pathex=[backend_dir],
    binaries=[],
    datas=[
        # pdfminer 资源文件
        # tiktoken 数据文件（如果使用）
    ],
    hiddenimports=[
        # FastAPI / Starlette
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',

        # Pydantic
        'pydantic',
        'pydantic_settings',
        'pydantic.deprecated.decorator',

        # FAISS
        'faiss',
        'faiss.swigfaiss',

        # PDF 处理
        'pdfplumber',
        'pdfminer',
        'pdfminer.high_level',
        'pdfminer.layout',
        'fitz',  # PyMuPDF

        # LangChain
        'langchain',
        'langchain.text_splitter',
        'langchain_community',
        'langchain_core',

        # AI SDK
        'openai',
        'anthropic',
        'httpx',
        'httpx._transports',
        'httpx._transports.default',

        # 数据处理
        'numpy',
        'numpy.core',

        # 其他
        'multipart',
        'python_multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除重量级 ML 依赖（桌面模式不需要）
        'torch', 'torchvision', 'torchaudio',
        'sentence_transformers', 'transformers', 'tokenizers',
        'huggingface_hub', 'safetensors',

        # 排除 Anaconda 附带的科学计算/可视化库
        'matplotlib', 'scipy', 'pandas', 'sklearn', 'scikit-learn',
        'cv2', 'opencv',                      # OpenCV (~95MB)
        'llvmlite', 'numba',                  # LLVM/Numba (~65MB)
        'bokeh', 'panel', 'holoviews',        # 可视化 (~100MB)
        'plotly', 'altair', 'xarray',
        'statsmodels', 'patsy',
        'skimage', 'scikit_image',            # scikit-image (~10MB)
        'astropy',                            # 天文学 (~12MB)

        # 排除 AWS/Google/Playwright 等大型 SDK
        'botocore', 'boto3', 'aiobotocore',   # AWS (~81MB)
        'googleapiclient', 'google.cloud',    # Google API (~90MB)
        'google.auth', 'google.api_core',
        'playwright',                         # 浏览器自动化 (~87MB)
        'grpc', 'grpcio',                     # gRPC (~10MB)
        'pyarrow', 'arrow',                   # Apache Arrow (~16MB)

        # 排除 NLTK 数据和非核心 NLP
        'nltk',                               # NLTK + data (~93MB)

        # 排除文档生成/国际化
        'sphinx', 'docutils',                 # Sphinx (~9MB)
        'babel',                              # Babel i18n (~28MB)
        'nbformat', 'nbconvert',
        'intake',

        # 排除测试/开发工具
        'pytest', 'IPython', 'jupyter', 'notebook',
        'dask', 'distributed',

        # 排除 GUI 库
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',

        # 排除其他不需要的
        'sqlalchemy',                         # 项目不使用 ORM
        'h5py',                               # HDF5
        'zmq',                                # ZeroMQ
        'win32com', 'pythoncom', 'pywintypes',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='desktop_entry',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 需要控制台输出日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='chatpdf-backend',
)
