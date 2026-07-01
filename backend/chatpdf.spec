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

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

backend_dir = os.path.dirname(os.path.abspath(SPEC))

project_hiddenimports = [
    # 路由入口
    'routes.chat_routes',
    'routes.document_routes',
    'routes.feedback_routes',
    'routes.glossary_routes',
    'routes.memory_routes',
    'routes.model_provider_routes',
    'routes.preset_routes',
    'routes.prompt_pool_routes',
    'routes.search_routes',
    'routes.summary_routes',
    'routes.system_routes',

    # Agentic RAG / 检索链路
    'services.agent_retrieval_service',
    'services.retrieval_agent',
    'services.retrieval_tools',
    'services.retrieval_tool_schemas',
    'services.tree_decomposition_retrieval',
    'services.citation_enhancer',
    'services.formula_text',
    'services.grep_service',
    'services.bm25_service',
    'services.advanced_search',
    'services.query_analyzer',
    'services.query_rewriter',
    'services.query_expander',
    'services.query_simplifier',
    'services.semantic_group_service',
    'services.granularity_selector',
    'services.token_budget',
    'services.context_builder',
    'services.context_injector',
    'services.retrieval_logger',
    'services.chunk_expander',
    'services.hybrid_search',
    'services.rerank_service',
    'services.rerank_api_service',
    'services.vector_service',
    'services.embedding_service',
    'services.table_service',
    'services.table_aware_service',
    'services.web_search_service',
    'services.web_search_reranker',

    # 文档上传 / OCR / 概览链路
    'services.url_loader_service',
    'services.multi_format_loader',
    'services.ocr_service',
    'services.odl_parser_service',
    'services.overview_service',
    'services.figure_adapter',
    'services.figure_builder',
    'services.figure_extraction',
    'services.figure_render',
    'services.figure_validation',

    # 业务服务
    'services.chat_service',
    'services.glossary_service',
    'services.preset_service',
    'services.prompt_pool_service',
    'services.memory_service',
    'services.memory_store',
    'services.memory_store_sqlite',
    'services.memory_cache',
    'services.memory_index',
    'services.memory_retriever',
    'services.memory_sync',
    'services.keyword_extractor',
    'services.active_pool',
    'services.memory_compressor',
    'services.memory_tagger',
    'services.followup_service',
    'services.conv_name_service',
    'services.decompose_service',
    'services.mindmap_service',
    'services.answer_critic_service',
    'services.citation_service',

    # GraphRAG 子包存在包级导入顺序问题，避免 collect_submodules 直接导入包。
    'services.graphrag.graphrag',
    'services.graphrag.base',
    'services.graphrag._op',
    'services.graphrag._storage',
    'services.graphrag._utils',
    'services.graphrag.nano_vectordb',
    'services.graphrag.prompt',

    # Provider / model 注册
    *collect_submodules('providers'),
    *collect_submodules('models'),
    *collect_submodules('middleware'),
    *collect_submodules('schemas'),
    *collect_submodules('utils'),
]

a = Analysis(
    ['desktop_entry.py'],
    pathex=[backend_dir],
    binaries=[],
    datas=[
        # pdfminer 资源文件
        *collect_data_files('tiktoken', include_py_files=False),
    ],
    hiddenimports=[
        *project_hiddenimports,

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
        # Gemini provider uses direct REST calls through httpx.
        'httpx',
        'httpx._transports',
        'httpx._transports.default',
        'ddgs',

        # 数据处理
        'numpy',
        'numpy.core',
        'scipy',
        'sklearn',
        'sklearn.metrics',
        'sklearn.metrics.cluster',

        # GraphRAG
        'networkx',
        'graspologic',
        'graspologic.partition',
        'graspologic.utils',
        'tiktoken',

        # 中文检索
        'jieba',

        # 其他
        'multipart',
        'python_multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除重量级 ML / OCR 布局模型依赖（桌面模式不内置本地模型）
        'torch', 'torchvision', 'torchaudio',
        'sentence_transformers', 'transformers', 'tokenizers',
        'huggingface_hub', 'safetensors',
        'doclayout_yolo', 'ultralytics', 'lapx',

        # 排除 Anaconda 附带的科学计算/可视化库
        'matplotlib', 'pandas',
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

# 过滤运行时用户数据，防止打包泄露个人论文缓存
_user_data_prefixes = ('data/overviews', 'data/cache', 'data/graphrag')
clean_datas = [d for d in a.datas if not any(d[0].startswith(p) for p in _user_data_prefixes)]

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    clean_datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='chatpdf-backend',
)
