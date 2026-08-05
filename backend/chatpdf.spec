# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件 - ChatPDF 桌面后端打包配置

使用方式：
  cd backend
  pip install -r requirements-desktop.txt
  pip install pyinstaller
  pyinstaller chatpdf.spec

目标：onedir 模式。含 CPU 版 torch / DocLayout-YOLO 运行库（已剥离 CUDA GPU 库），
不内置 YOLO 权重（桌面端在设置页按需下载）。
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# DocLayout-YOLO must never install missing dependencies during a build or at runtime.
# The desktop artifact is intentionally reproducible and only uses bundled packages.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

block_cipher = None

backend_dir = os.path.dirname(os.path.abspath(SPEC))
project_dir = os.path.dirname(backend_dir)


def _metadata_datas():
    datas = []
    version_file = os.path.join(project_dir, 'version.json')
    if os.path.exists(version_file):
        datas.append((version_file, '.'))
    for build_info in (
        os.path.join(project_dir, 'build-info.json'),
        os.path.join(backend_dir, 'build-info.json'),
    ):
        if os.path.exists(build_info):
            datas.append((build_info, '.'))
            break
    return datas

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
        *_metadata_datas(),
        # pdfminer 资源文件
        *collect_data_files('tiktoken', include_py_files=False),
        # DocLayout-YOLO 权重不内置到安装包；桌面端在设置页下载到用户数据目录或手动指定。
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

        # 速览图表 YOLO 预览模式
        'torch',
        'torchvision',
        'doclayout_yolo',
        'doclayout_yolo.engine.model',
        'doclayout_yolo.engine.predictor',
        'doclayout_yolo.engine.results',
        'doclayout_yolo.models.yolov10.model',
        'doclayout_yolo.models.yolov10.predict',
        'doclayout_yolo.nn.tasks',
        'doclayout_yolo.nn.modules',
        'doclayout_yolo.utils',
        'doclayout_yolo.utils.ops',
        'doclayout_yolo.utils.torch_utils',
        'huggingface_hub',
        'cv2',

        # 其他
        'multipart',
        'python_multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除未使用的重量级 ML 依赖；DocLayout-YOLO 预览模式依赖 torch/torchvision 运行库，保留。
        'torchaudio',
        'sentence_transformers', 'transformers', 'tokenizers',
        'safetensors',
        'ultralytics', 'lapx',

        # 排除 Anaconda 附带但当前桌面包未使用的可视化/分析库
        'opencv',                             # 保留 cv2 给 DocLayout-YOLO；仅排除旧命名占位
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

# 剥离 CUDA / cuDNN 运行库：桌面端 DocLayout-YOLO 仅做 CPU 推理
# （_get_device() 默认 CPU），避免把 CUDA 版 torch 的 ~2.5GB GPU 库打进安装包。
import re as _re
_CUDA_LIB_RE = _re.compile(
    r"(cudnn|cublas|cudart|cufft|curand|cusolver|cusparse|nvrtc|nvtx|"
    r"nvtoolsext|nccl|cupti|cusparselt|torch_cuda|c10_cuda|caffe2_nvrtc)",
    _re.IGNORECASE,
)
_before = len(a.binaries)
a.binaries = [b for b in a.binaries if not _CUDA_LIB_RE.search(os.path.basename(b[0]))]
print(f"[chatpdf.spec] Stripped {_before - len(a.binaries)} CUDA binaries (CPU-only desktop build)")

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

# 不把任何可变的运行时数据打入安装包。桌面端会在首次启动时由
# CHATPDF_DATA_DIR 指向用户应用数据目录；这里的过滤用于阻止未来的 hook、
# --add-data 参数或残留构建产物意外携带本地论文、对话、密钥或日志。
_private_runtime_roots = {
    'data',
    'uploads',
    'logs',
    'cache',
    'history',
    'memory',
    'vector_stores',
    'semantic_groups',
    'overviews',
    'parse',
    # Never ship development fixtures or evaluation/course material.
    'test',
    'tests',
    'fixture',
    'fixtures',
    'course',
    'courses',
    'eval',
    'evaluation',
    'evaluations',
    '课程',
}

_private_runtime_extensions = {
    '.pdf',
    '.db',
    '.sqlite',
    '.sqlite3',
    '.faiss',
    '.pkl',
    '.pickle',
    '.log',
}

_private_runtime_basenames = {
    'online_ocr_config.json',
    'ocr_provider_usage.json',
    'chat_history.json',
    'history.json',
}


def _is_private_runtime_data(toc_entry):
    destination = str(toc_entry[0]).replace('\\', '/').lstrip('./')
    parts = [part.lower() for part in destination.split('/') if part]
    filename = parts[-1] if parts else ''
    return (
        any(part in _private_runtime_roots for part in parts)
        or Path(filename).suffix.lower() in _private_runtime_extensions
        or filename in _private_runtime_basenames
        or filename == '.env'
        or filename.startswith('.env.')
        or 'api_key' in filename
        or 'apikey' in filename
    )


clean_datas = [d for d in a.datas if not _is_private_runtime_data(d)]

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
