"""
OCR Service for PDF text extraction
Supports both local Tesseract OCR and cloud OCR APIs
"""
import io
import os
import re
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path


# ============================================================
# 数据模型与抽象基类（适配器模式）
# ============================================================

@dataclass
class PageOCRResult:
    """单页 OCR 结果"""
    page_number: int              # 页码（从 1 开始）
    text: str                     # OCR 提取的文本
    success: bool                 # 是否成功
    error: Optional[str] = None   # 错误信息（失败时）


@dataclass
class OCRResult:
    """批量 OCR 结果"""
    pages: List[PageOCRResult]                                # 各页结果
    failed_pages: List[int] = field(default_factory=list)     # 失败页码列表
    errors: Dict[int, str] = field(default_factory=dict)      # 页码 -> 错误信息
    backend: str = ""                                         # 使用的后端名称


class BaseOCRAdapter(ABC):
    """OCR 适配器抽象基类，定义所有 OCR 后端的统一接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称标识"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检测该 OCR 后端是否可用"""
        ...

    @abstractmethod
    def ocr_image(self, image) -> str:
        """对单张图片执行 OCR，返回文本"""
        ...

    @abstractmethod
    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 OCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        ...

import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 在线 OCR 配置管理函数
# ============================================================

# 配置文件路径：相对于 Chatpdf 项目根目录
_ONLINE_OCR_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "online_ocr_config.json"

# 各在线 OCR 提供商的环境变量名映射
# 注意：token_mode、enable_ocr、enable_formula、enable_table 等字段
# 不通过环境变量配置，而是从配置文件加载
_ENV_VAR_MAP = {
    "mistral": {
        "api_key": "CHATPDF_MISTRAL_OCR_API_KEY",
        "base_url": "CHATPDF_MISTRAL_OCR_BASE_URL",
    },
    "mineru": {
        "worker_url": "CHATPDF_MINERU_WORKER_URL",
        "auth_key": "CHATPDF_MINERU_AUTH_KEY",
        "token": "CHATPDF_MINERU_TOKEN",
    },
    "doc2x": {
        "worker_url": "CHATPDF_DOC2X_WORKER_URL",
        "auth_key": "CHATPDF_DOC2X_AUTH_KEY",
        "token": "CHATPDF_DOC2X_TOKEN",
    },
}

# 各在线 OCR 提供商的默认配置
_DEFAULT_CONFIG = {
    "mistral": {
        "api_key": "",
        "base_url": "https://api.mistral.ai",
    },
    "mineru": {
        "worker_url": "",
        "auth_key": "",
        "token_mode": "frontend",
        "token": "",
        "enable_ocr": True,
        "enable_formula": True,
        "enable_table": True,
    },
    "doc2x": {
        "worker_url": "",
        "auth_key": "",
        "token_mode": "frontend",
        "token": "",
    },
}


def _load_online_ocr_config(provider: str) -> dict:
    """
    加载在线 OCR 配置，优先级：环境变量 > 配置文件 > 默认值

    对于在 _ENV_VAR_MAP 中定义的字段，按上述优先级加载。
    对于不在 _ENV_VAR_MAP 中的字段（如 token_mode、enable_ocr、enable_formula、
    enable_table），仅从配置文件和默认值加载（不通过环境变量配置）。

    参数:
        provider: 服务提供商名称（如 "mistral"、"mineru"、"doc2x"）
    返回:
        配置字典，包含该提供商的所有配置字段
    """
    # 获取默认配置
    defaults = _DEFAULT_CONFIG.get(provider, {"api_key": "", "base_url": ""})
    result = dict(defaults)

    # 获取该提供商的环境变量映射（用于区分哪些字段可通过环境变量配置）
    env_map = _ENV_VAR_MAP.get(provider, {})

    # 第二优先级：从配置文件加载
    try:
        if _ONLINE_OCR_CONFIG_PATH.exists():
            with open(_ONLINE_OCR_CONFIG_PATH, "r", encoding="utf-8") as f:
                all_config = json.load(f)
            if provider in all_config and isinstance(all_config[provider], dict):
                provider_config = all_config[provider]
                for key in result:
                    if key in provider_config:
                        value = provider_config[key]
                        # 对于字符串类型字段，仅覆盖非空值
                        # 对于布尔类型字段，直接覆盖
                        if isinstance(value, bool):
                            result[key] = value
                        elif value:
                            result[key] = value
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.error(f"读取在线 OCR 配置文件失败: {e}")

    # 第一优先级：从环境变量加载（仅覆盖 _ENV_VAR_MAP 中定义的字段）
    for field_name, env_var_name in env_map.items():
        env_value = os.environ.get(env_var_name, "")
        if env_value:
            result[field_name] = env_value

    return result


def _save_online_ocr_config(provider: str, config: dict) -> None:
    """
    保存在线 OCR 配置到本地文件

    参数:
        provider: 服务提供商名称
        config: 配置字典，包含 api_key 和/或 base_url
    """
    try:
        # 确保目录存在
        _ONLINE_OCR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有配置（如果存在）
        all_config: dict = {}
        if _ONLINE_OCR_CONFIG_PATH.exists():
            try:
                with open(_ONLINE_OCR_CONFIG_PATH, "r", encoding="utf-8") as f:
                    all_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                # 配置文件损坏，使用空字典重新开始
                logger.warning("在线 OCR 配置文件损坏，将重新创建")
                all_config = {}

        # 更新指定提供商的配置
        if provider not in all_config:
            all_config[provider] = {}
        all_config[provider].update(config)

        # 写入配置文件
        with open(_ONLINE_OCR_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(all_config, f, ensure_ascii=False, indent=2)

        logger.info(f"在线 OCR 配置已保存: provider={provider}")
    except (IOError, OSError) as e:
        logger.error(f"保存在线 OCR 配置文件失败: {e}")
        raise


def _mask_api_key(api_key: str) -> str:
    """
    脱敏 API Key，仅显示前 4 位和后 4 位

    参数:
        api_key: 原始 API Key
    返回:
        脱敏后的字符串，如 "sk-x...xxxx"
    """
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "..." + api_key[-4:]

# OCR 可用性标志
TESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

# 自动检测本地 OCR 工具路径
def _find_local_ocr_tools():
    """查找本地安装的 OCR 工具"""
    base_dir = Path(__file__).resolve().parents[2]  # Chatpdf 根目录
    ocr_dir = base_dir / "ocr_tools"
    
    # Tesseract 路径
    tesseract_paths = [
        ocr_dir / "tesseract" / "tesseract.exe",  # Windows 本地安装
        ocr_dir / "tesseract" / "tesseract",  # Linux/Mac 本地安装
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),  # Windows 默认
        Path("/usr/bin/tesseract"),  # Linux 默认
        Path("/usr/local/bin/tesseract"),  # macOS Homebrew
        Path("/opt/homebrew/bin/tesseract"),  # macOS M1 Homebrew
    ]
    
    for path in tesseract_paths:
        if path.exists():
            return str(path.parent)
    
    return None

# 设置 Tesseract 路径
_tesseract_path = _find_local_ocr_tools()
if _tesseract_path:
    os.environ["PATH"] = _tesseract_path + os.pathsep + os.environ.get("PATH", "")

# Poppler 路径
def _find_poppler():
    """
    查找 Poppler 路径

    增强逻辑：
    - 搜索失败时记录所有已搜索路径到日志
    - 检测 ocr_tools/poppler/ 目录存在但无可执行文件的情况并记录警告
    """
    base_dir = Path(__file__).resolve().parents[2]
    ocr_dir = base_dir / "ocr_tools"

    poppler_paths = [
        ocr_dir / "poppler" / "Library" / "bin",  # Windows 本地安装
        Path("/usr/bin"),  # Linux
        Path("/usr/local/bin"),  # macOS Homebrew
        Path("/opt/homebrew/bin"),  # macOS M1 Homebrew
    ]

    # 检测 ocr_tools/poppler/ 目录是否存在但内部无可执行文件
    poppler_dir = ocr_dir / "poppler"
    if poppler_dir.exists() and poppler_dir.is_dir():
        # 检查该目录及其子目录中是否有 pdftoppm 可执行文件
        has_executable = False
        for sub_path in poppler_paths:
            # 仅检查属于 poppler_dir 子路径的搜索路径
            try:
                sub_path.relative_to(poppler_dir)
            except ValueError:
                continue
            if (sub_path / "pdftoppm.exe").exists() or (sub_path / "pdftoppm").exists():
                has_executable = True
                break

        if not has_executable:
            logger.warning(
                f"Poppler 目录 '{poppler_dir}' 存在但未找到可执行文件（pdftoppm），"
                "该目录可能为空或未正确安装。"
                "请确保 Poppler 已正确解压到该目录中。"
            )

    for path in poppler_paths:
        if (path / "pdftoppm.exe").exists() or (path / "pdftoppm").exists():
            return str(path)

    # 搜索失败，记录所有已搜索路径到日志
    searched_paths_str = "\n  ".join(str(p) for p in poppler_paths)
    logger.warning(
        f"未找到 Poppler 可执行文件（pdftoppm），已搜索以下路径:\n  {searched_paths_str}\n"
        "请安装 Poppler 以启用 PDF 转图像功能。\n"
        "安装指引:\n"
        "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
        "  - macOS: brew install poppler\n"
        "  - Linux: sudo apt-get install poppler-utils"
    )
    return None

_poppler_path = _find_poppler()
if _poppler_path:
    os.environ["PATH"] = _poppler_path + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
    # 设置 Tesseract 命令路径
    if _tesseract_path:
        tesseract_cmd = os.path.join(_tesseract_path, "tesseract.exe" if os.name == "nt" else "tesseract")
        if os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_bytes
    # 设置 Poppler 路径
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 具体适配器实现
# ============================================================

class TesseractAdapter(BaseOCRAdapter):
    """Tesseract OCR 适配器"""

    def __init__(self, lang: str = "chi_sim+eng"):
        """
        初始化 Tesseract 适配器

        参数:
            lang: Tesseract 语言代码（默认中文+英文）
        """
        self._lang = lang

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "tesseract"

    def is_available(self) -> bool:
        """检测 Tesseract 和 pdf2image 是否可用"""
        return TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE

    def ocr_image(self, image) -> str:
        """
        使用 Tesseract 对单张图片执行 OCR

        参数:
            image: PIL Image 对象
        返回:
            OCR 提取的文本
        """
        if not TESSERACT_AVAILABLE:
            return ""
        return pytesseract.image_to_string(image, lang=self._lang)

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 Tesseract OCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 未安装，请运行: pip install pdf2image")

        poppler_path = _find_poppler()
        if poppler_path is None:
            raise RuntimeError(
                "Poppler 未找到，无法将 PDF 转换为图像。\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )

        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # pdf2image 的 first_page/last_page 从 1 开始，page_numbers 从 0 开始
            pdf2image_page = page_num + 1
            try:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=pdf2image_page,
                    last_page=pdf2image_page,
                    poppler_path=poppler_path
                )
                if images:
                    raw_text = self.ocr_image(images[0])
                    text = clean_ocr_text(raw_text)
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text=text,
                        success=True
                    ))
                else:
                    # 未能转换出图像
                    failed_pages.append(pdf2image_page)
                    errors[pdf2image_page] = "PDF 页面转换为图像失败：未生成图像"
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text="",
                        success=False,
                        error="PDF 页面转换为图像失败：未生成图像"
                    ))
            except Exception as e:
                # 单页错误隔离：捕获异常，记录错误，继续处理下一页
                error_msg = f"页面 {pdf2image_page} OCR 失败: {str(e)}"
                logger.error(error_msg)
                failed_pages.append(pdf2image_page)
                errors[pdf2image_page] = str(e)
                pages_result.append(PageOCRResult(
                    page_number=pdf2image_page,
                    text="",
                    success=False,
                    error=str(e)
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name
        )


class PaddleOCRAdapter(BaseOCRAdapter):
    """PaddleOCR 适配器"""

    def __init__(self):
        """初始化 PaddleOCR 适配器"""
        self._paddle_ocr = None

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "paddleocr"

    def is_available(self) -> bool:
        """检测 PaddleOCR 和 pdf2image 是否可用"""
        return PADDLEOCR_AVAILABLE and PDF2IMAGE_AVAILABLE

    def _get_paddle_ocr(self):
        """延迟加载 PaddleOCR 实例"""
        if self._paddle_ocr is None and PADDLEOCR_AVAILABLE:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                show_log=False,
                use_gpu=False
            )
        return self._paddle_ocr

    def ocr_image(self, image) -> str:
        """
        使用 PaddleOCR 对单张图片执行 OCR

        参数:
            image: PIL Image 对象
        返回:
            OCR 提取的文本
        """
        if not PADDLEOCR_AVAILABLE:
            return ""
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return ""
        import numpy as np
        img_array = np.array(image)
        result = ocr.ocr(img_array, cls=True)

        if not result or not result[0]:
            return ""

        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                lines.append(text)
        return '\n'.join(lines)

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 PaddleOCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 未安装，请运行: pip install pdf2image")

        poppler_path = _find_poppler()
        if poppler_path is None:
            raise RuntimeError(
                "Poppler 未找到，无法将 PDF 转换为图像。\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )

        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # pdf2image 的 first_page/last_page 从 1 开始，page_numbers 从 0 开始
            pdf2image_page = page_num + 1
            try:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=pdf2image_page,
                    last_page=pdf2image_page,
                    poppler_path=poppler_path
                )
                if images:
                    raw_text = self.ocr_image(images[0])
                    text = clean_ocr_text(raw_text)
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text=text,
                        success=True
                    ))
                else:
                    # 未能转换出图像
                    failed_pages.append(pdf2image_page)
                    errors[pdf2image_page] = "PDF 页面转换为图像失败：未生成图像"
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text="",
                        success=False,
                        error="PDF 页面转换为图像失败：未生成图像"
                    ))
            except Exception as e:
                # 单页错误隔离：捕获异常，记录错误，继续处理下一页
                error_msg = f"页面 {pdf2image_page} OCR 失败: {str(e)}"
                logger.error(error_msg)
                failed_pages.append(pdf2image_page)
                errors[pdf2image_page] = str(e)
                pages_result.append(PageOCRResult(
                    page_number=pdf2image_page,
                    text="",
                    success=False,
                    error=str(e)
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name
        )


class WorkerOCRAdapter(BaseOCRAdapter):
    """Worker 代理模式 OCR 适配器基类，封装 Worker URL、Auth Key、Token Mode 等公共逻辑"""

    def __init__(self, worker_url: str, auth_key: str = "",
                 token: str = "", token_mode: str = "frontend"):
        """
        初始化 Worker 代理适配器

        参数:
            worker_url: Cloudflare Worker 代理服务地址
            auth_key: Worker 端的认证密钥（对应 Worker 的 AUTH_SECRET 环境变量）
            token: 各 OCR 服务的 API Token
            token_mode: Token 传递模式，"frontend"（前端透传）或 "worker"（Worker 配置）
        """
        self._worker_url = worker_url.rstrip("/") if worker_url else ""
        self._auth_key = auth_key
        self._token = token
        self._token_mode = token_mode  # "frontend" 或 "worker"

    def is_available(self) -> bool:
        """Worker URL 已配置且 Token 可用（worker 模式或 frontend 模式有 token）"""
        if not self._worker_url:
            return False
        if self._token_mode == "frontend":
            return bool(self._token)
        return True  # worker 模式不需要前端提供 token

    def ocr_image(self, image) -> str:
        """在线 OCR 不支持单图模式，返回空字符串"""
        return ""

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        子类需覆盖此方法实现具体的 OCR 处理逻辑

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        raise NotImplementedError("子类必须实现 ocr_pages 方法")

    def _build_headers(self) -> dict:
        """
        构建请求头：Auth Key + Token（根据 token_mode）

        返回:
            请求头字典，包含 X-Auth-Key（如有）。
            子类覆盖以添加特定的 Token 头。
        """
        headers = {}
        if self._auth_key:
            headers["X-Auth-Key"] = self._auth_key
        # 子类覆盖以添加特定的 Token 头
        return headers

    def _check_worker_response(self, response, step: str) -> None:
        """
        检查 Worker 响应状态码

        参数:
            response: httpx.Response 对象
            step: 当前步骤描述（用于错误消息）
        异常:
            RuntimeError: 当响应状态码非成功时抛出，401/403 特殊提示认证失败
        """
        if response.is_success:
            return
        status_code = response.status_code
        try:
            error_detail = response.text
        except Exception:
            error_detail = "未知错误"
        if status_code in (401, 403):
            raise RuntimeError(f"{self.name} {step}失败: 认证失败 (HTTP {status_code})")
        raise RuntimeError(f"{self.name} {step}失败 (HTTP {status_code}): {error_detail}")


# ============================================================
# 模块级工具函数
# ============================================================

def _markdown_to_text(markdown: str) -> str:
    """
    将 Markdown 内容转换为纯文本，清理 Markdown 标记

    从 MistralAdapter 中提取为模块级函数，供所有适配器复用。

    参数:
        markdown: Markdown 格式的文本
    返回:
        纯文本字符串
    """
    if not markdown:
        return ""

    text = markdown

    # 移除图片标记 ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 移除链接标记 [text](url)，保留链接文本
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 移除标题标记 # ## ### 等
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 移除粗体标记 **text** 或 __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # 移除斜体标记 *text* 或 _text_（注意不要误匹配粗体）
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)
    # 移除行内代码标记 `code`
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # 移除代码块标记 ```...```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除水平分割线 --- 或 *** 或 ___
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 移除无序列表标记 - 或 * 或 +（行首）
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    # 移除有序列表标记 1. 2. 等（行首）
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    # 移除引用标记 >
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # 移除删除线标记 ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================
# MinerU OCR 适配器
# ============================================================

class MinerUAdapter(WorkerOCRAdapter):
    """MinerU OCR 适配器，通过 Worker 代理处理 PDF"""

    def __init__(self, worker_url: str, auth_key: str = "",
                 token: str = "", token_mode: str = "frontend",
                 enable_ocr: bool = True, enable_formula: bool = True,
                 enable_table: bool = True):
        """
        初始化 MinerU OCR 适配器

        参数:
            worker_url: Cloudflare Worker 代理服务地址
            auth_key: Worker 端的认证密钥
            token: MinerU API Token
            token_mode: Token 传递模式，"frontend" 或 "worker"
            enable_ocr: 是否启用 OCR 识别
            enable_formula: 是否启用公式识别
            enable_table: 是否启用表格识别
        """
        super().__init__(worker_url, auth_key, token, token_mode)
        self._enable_ocr = enable_ocr
        self._enable_formula = enable_formula
        self._enable_table = enable_table

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "mineru"

    def _build_headers(self) -> dict:
        """
        构建请求头：在基类基础上添加 X-MinerU-Key（frontend 模式）

        返回:
            请求头字典
        """
        headers = super()._build_headers()
        if self._token_mode == "frontend" and self._token:
            headers["X-MinerU-Key"] = self._token
        return headers

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        通过 MinerU Worker 代理处理 PDF，提取文本

        流程：上传 PDF → 轮询结果 → 下载 ZIP → 解压 → 提取 Markdown → 转文本

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率（MinerU 忽略此参数）
        返回:
            OCRResult 包含各页结果和错误信息
        """
        import httpx

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                # 步骤 1：上传 PDF
                logger.info("MinerU OCR: 开始上传 PDF 文件...")
                batch_id = self._upload_pdf(client, pdf_bytes)
                logger.info(f"MinerU OCR: 上传成功，batch_id={batch_id}")

                # 步骤 2：轮询结果
                logger.info("MinerU OCR: 开始轮询处理结果...")
                full_zip_url = self._poll_result(client, batch_id)
                logger.info("MinerU OCR: 处理完成，开始下载结果...")

                # 步骤 3：下载 ZIP 并解压提取 full.md
                markdown_content = self._download_and_extract(client, full_zip_url)
                logger.info("MinerU OCR: 结果下载并解压成功")

                # 步骤 4：将 Markdown 转换为纯文本
                text = _markdown_to_text(markdown_content)
                text = clean_ocr_text(text)

                # 步骤 5：构建 OCRResult（MinerU 返回整个文档的文本，按页分配）
                pages_result: List[PageOCRResult] = []
                for page_num in page_numbers:
                    display_page = page_num + 1  # 用于显示的页码（从 1 开始）
                    pages_result.append(PageOCRResult(
                        page_number=display_page,
                        text=text,
                        success=True,
                    ))

                return OCRResult(
                    pages=pages_result,
                    failed_pages=[],
                    errors={},
                    backend=self.name,
                )

        except httpx.TimeoutException as e:
            logger.error(f"MinerU OCR: 网络连接超时: {e}")
            error_msg = f"MinerU OCR 网络连接超时: {e}"
        except RuntimeError as e:
            logger.error(f"MinerU OCR: {e}")
            error_msg = str(e)
        except Exception as e:
            logger.error(f"MinerU OCR: 未知错误: {e}")
            error_msg = f"MinerU OCR 处理失败: {e}"

        # 错误时返回包含错误信息的 OCRResult
        pages_result = []
        failed_pages = []
        errors = {}
        for page_num in page_numbers:
            display_page = page_num + 1
            failed_pages.append(display_page)
            errors[display_page] = error_msg
            pages_result.append(PageOCRResult(
                page_number=display_page,
                text="",
                success=False,
                error=error_msg,
            ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    def _upload_pdf(self, client, pdf_bytes: bytes) -> str:
        """
        上传 PDF 到 MinerU Worker 代理

        参数:
            client: httpx.Client 实例
            pdf_bytes: PDF 原始字节
        返回:
            batch_id 批次标识
        异常:
            RuntimeError: 上传失败时抛出
        """
        headers = self._build_headers()
        response = client.post(
            f"{self._worker_url}/mineru/upload",
            headers=headers,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            data={
                "is_ocr": str(self._enable_ocr).lower(),
                "enable_formula": str(self._enable_formula).lower(),
                "enable_table": str(self._enable_table).lower(),
            },
        )
        self._check_worker_response(response, "上传 PDF")
        data = response.json()
        batch_id = data.get("batch_id")
        if not batch_id:
            raise RuntimeError("MinerU 上传成功但未返回 batch_id")
        return batch_id

    def _poll_result(self, client, batch_id: str) -> str:
        """
        轮询 MinerU 处理结果

        参数:
            client: httpx.Client 实例
            batch_id: 批次标识
        返回:
            full_zip_url ZIP 文件下载地址
        异常:
            RuntimeError: 处理失败或超时时抛出
        """
        headers = self._build_headers()
        max_attempts = 100
        poll_interval = 3  # 秒

        for attempt in range(max_attempts):
            response = client.get(
                f"{self._worker_url}/mineru/result/{batch_id}",
                headers=headers,
            )
            self._check_worker_response(response, "轮询结果")
            data = response.json()
            state = data.get("state", "")

            if state == "done":
                full_zip_url = data.get("full_zip_url")
                if not full_zip_url:
                    raise RuntimeError("MinerU 处理完成但未返回 full_zip_url")
                return full_zip_url
            elif state == "failed":
                error_msg = data.get("error", "未知错误")
                raise RuntimeError(f"MinerU 处理失败: {error_msg}")

            # 继续等待
            logger.debug(
                f"MinerU OCR: 轮询中 ({attempt + 1}/{max_attempts})，"
                f"当前状态: {state}"
            )
            time.sleep(poll_interval)

        raise RuntimeError(
            f"MinerU 处理超时: 已轮询 {max_attempts} 次（共 {max_attempts * poll_interval} 秒）"
        )

    def _download_and_extract(self, client, zip_url: str) -> str:
        """
        下载 ZIP 文件并解压提取 full.md 内容

        参数:
            client: httpx.Client 实例
            zip_url: ZIP 文件下载地址
        返回:
            full.md 文件的文本内容
        异常:
            RuntimeError: 下载失败、解压失败或未找到 full.md 时抛出
        """
        headers = self._build_headers()
        response = client.get(zip_url, headers=headers)
        self._check_worker_response(response, "下载 ZIP")

        try:
            zip_data = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_data, "r") as zf:
                # 查找 full.md 文件
                full_md_path = None
                for name in zf.namelist():
                    if name.endswith("full.md"):
                        full_md_path = name
                        break

                if full_md_path is None:
                    raise RuntimeError(
                        "MinerU ZIP 中未找到 full.md 文件，"
                        f"ZIP 包含: {zf.namelist()}"
                    )

                content = zf.read(full_md_path).decode("utf-8")
                return content
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"MinerU ZIP 文件解压失败: {e}")


# ============================================================
# Doc2X OCR 适配器
# ============================================================

class Doc2XAdapter(WorkerOCRAdapter):
    """Doc2X OCR 适配器，通过 Worker 代理处理 PDF"""

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "doc2x"

    def _build_headers(self) -> dict:
        """
        构建请求头：在基类基础上添加 X-Doc2X-Key（frontend 模式）

        返回:
            请求头字典
        """
        headers = super()._build_headers()
        if self._token_mode == "frontend" and self._token:
            headers["X-Doc2X-Key"] = self._token
        return headers

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        通过 Doc2X Worker 代理处理 PDF，提取文本

        流程：上传 PDF → 轮询状态 → 获取 Markdown → 转文本

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率（Doc2X 忽略此参数）
        返回:
            OCRResult 包含各页结果和错误信息
        """
        import httpx

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                # 步骤 1：上传 PDF
                logger.info("Doc2X OCR: 开始上传 PDF 文件...")
                uid = self._upload_pdf(client, pdf_bytes)
                logger.info(f"Doc2X OCR: 上传成功，uid={uid}")

                # 步骤 2：轮询状态，完成时直接获取 Markdown
                logger.info("Doc2X OCR: 开始轮询处理状态...")
                markdown_content = self._poll_status(client, uid)
                logger.info("Doc2X OCR: 处理完成，已获取 Markdown 内容")

                # 步骤 3：将 Markdown 转换为纯文本
                text = _markdown_to_text(markdown_content)
                text = clean_ocr_text(text)

                # 步骤 4：构建 OCRResult（Doc2X 返回整个文档的文本，按页分配）
                pages_result: List[PageOCRResult] = []
                for page_num in page_numbers:
                    display_page = page_num + 1  # 用于显示的页码（从 1 开始）
                    pages_result.append(PageOCRResult(
                        page_number=display_page,
                        text=text,
                        success=True,
                    ))

                return OCRResult(
                    pages=pages_result,
                    failed_pages=[],
                    errors={},
                    backend=self.name,
                )

        except httpx.TimeoutException as e:
            logger.error(f"Doc2X OCR: 网络连接超时: {e}")
            error_msg = f"Doc2X OCR 网络连接超时: {e}"
        except RuntimeError as e:
            logger.error(f"Doc2X OCR: {e}")
            error_msg = str(e)
        except Exception as e:
            logger.error(f"Doc2X OCR: 未知错误: {e}")
            error_msg = f"Doc2X OCR 处理失败: {e}"

        # 错误时返回包含错误信息的 OCRResult
        pages_result = []
        failed_pages = []
        errors = {}
        for page_num in page_numbers:
            display_page = page_num + 1
            failed_pages.append(display_page)
            errors[display_page] = error_msg
            pages_result.append(PageOCRResult(
                page_number=display_page,
                text="",
                success=False,
                error=error_msg,
            ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    def _upload_pdf(self, client, pdf_bytes: bytes) -> str:
        """
        上传 PDF 到 Doc2X Worker 代理

        参数:
            client: httpx.Client 实例
            pdf_bytes: PDF 原始字节
        返回:
            uid 任务标识
        异常:
            RuntimeError: 上传失败时抛出
        """
        headers = self._build_headers()
        response = client.post(
            f"{self._worker_url}/doc2x/upload",
            headers=headers,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            data={
                "ocr": "true",
                "formula_mode": "dollar",
            },
        )
        self._check_worker_response(response, "上传 PDF")
        data = response.json()
        uid = data.get("uid")
        if not uid:
            raise RuntimeError("Doc2X 上传成功但未返回 uid")
        return uid

    def _poll_status(self, client, uid: str) -> str:
        """
        轮询 Doc2X 处理状态，完成时返回 Markdown 内容

        参数:
            client: httpx.Client 实例
            uid: 任务标识
        返回:
            Markdown 内容字符串
        异常:
            RuntimeError: 处理失败或超时时抛出
        """
        headers = self._build_headers()
        max_attempts = 100
        poll_interval = 3  # 秒

        for attempt in range(max_attempts):
            response = client.get(
                f"{self._worker_url}/doc2x/status/{uid}",
                headers=headers,
            )
            self._check_worker_response(response, "轮询状态")
            data = response.json()
            state = data.get("state", "")

            if state == "done":
                markdown = data.get("markdown", "")
                return markdown
            elif state == "failed":
                error_msg = data.get("error", "未知错误")
                raise RuntimeError(f"Doc2X 处理失败: {error_msg}")

            # 继续等待
            logger.debug(
                f"Doc2X OCR: 轮询中 ({attempt + 1}/{max_attempts})，"
                f"当前状态: {state}"
            )
            time.sleep(poll_interval)

        raise RuntimeError(
            f"Doc2X 处理超时: 已轮询 {max_attempts} 次（共 {max_attempts * poll_interval} 秒）"
        )


class MistralAdapter(BaseOCRAdapter):
    """Mistral OCR 在线适配器，通过 Mistral API 执行 PDF OCR"""

    def __init__(self, api_key: str, base_url: str = "https://api.mistral.ai"):
        """
        初始化 Mistral OCR 适配器

        参数:
            api_key: Mistral API Key
            base_url: Mistral API 基础 URL（默认 https://api.mistral.ai）
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "mistral"

    def is_available(self) -> bool:
        """API Key 已配置则视为可用"""
        return bool(self._api_key)

    def ocr_image(self, image) -> str:
        """在线 OCR 不支持单图模式，返回空字符串"""
        return ""

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        调用 Mistral OCR API 处理整个 PDF，
        然后从结果中提取指定页码的文本。

        流程：
        1. 上传 PDF → 获取 file_id
        2. 获取签名 URL
        3. 调用 OCR 接口
        4. 解析结果，提取指定页码文本
        5. 清理远程文件（失败不影响结果）

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率（在线 OCR 忽略此参数）
        返回:
            OCRResult 包含各页结果和错误信息
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        file_id = None

        try:
            # 使用 httpx 同步客户端，设置合理的超时时间
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                # 步骤 1：上传 PDF 到 Mistral
                logger.info("Mistral OCR: 开始上传 PDF 文件...")
                upload_resp = client.post(
                    f"{self._base_url}/v1/files",
                    headers=headers,
                    files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
                    data={"purpose": "ocr"},
                )
                self._check_http_error(upload_resp, "上传 PDF")
                upload_data = upload_resp.json()
                file_id = upload_data.get("id")
                if not file_id:
                    raise RuntimeError("Mistral OCR: 上传成功但未返回文件 ID")
                logger.info(f"Mistral OCR: 文件上传成功，file_id={file_id}")

                # 步骤 2：获取签名 URL
                logger.info("Mistral OCR: 获取签名 URL...")
                url_resp = client.get(
                    f"{self._base_url}/v1/files/{file_id}/url",
                    headers=headers,
                )
                self._check_http_error(url_resp, "获取签名 URL")
                url_data = url_resp.json()
                signed_url = url_data.get("url")
                if not signed_url:
                    raise RuntimeError("Mistral OCR: 获取的签名 URL 格式不正确")
                logger.info("Mistral OCR: 签名 URL 获取成功")

                # 步骤 3：调用 OCR 接口
                logger.info("Mistral OCR: 开始 OCR 处理...")
                ocr_resp = client.post(
                    f"{self._base_url}/v1/ocr",
                    headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
                    json={
                        "model": "mistral-ocr-latest",
                        "document": {"type": "document_url", "document_url": signed_url},
                        "include_image_base64": False,
                    },
                )
                self._check_http_error(ocr_resp, "OCR 处理")
                ocr_data = ocr_resp.json()
                if not ocr_data or "pages" not in ocr_data:
                    raise RuntimeError("Mistral OCR: OCR 处理成功但返回的数据格式不正确")
                logger.info(f"Mistral OCR: OCR 处理完成，共 {len(ocr_data['pages'])} 页")

                # 步骤 4：解析结果，提取指定页码文本
                result = self._parse_ocr_response(ocr_data, page_numbers)

                # 步骤 5：清理远程文件（失败不影响结果）
                self._delete_file(client, headers, file_id)

                return result

        except httpx.TimeoutException as e:
            logger.error(f"Mistral OCR: 网络连接超时: {e}")
            raise RuntimeError(f"Mistral OCR 网络连接超时: {e}") from e
        except httpx.HTTPError as e:
            # httpx 的其他网络错误（非 HTTP 状态码错误）
            logger.error(f"Mistral OCR: 网络错误: {e}")
            raise RuntimeError(f"Mistral OCR 网络错误: {e}") from e

    def _check_http_error(self, response, step: str) -> None:
        """
        检查 HTTP 响应状态码，对 401/403 做特殊处理

        参数:
            response: httpx.Response 对象
            step: 当前步骤描述（用于错误消息）
        """
        if response.is_success:
            return

        status_code = response.status_code
        try:
            error_detail = response.text
        except Exception:
            error_detail = response.reason_phrase or "未知错误"

        if status_code in (401, 403):
            raise RuntimeError(
                f"Mistral OCR {step}失败: API Key 无效或已过期 "
                f"(HTTP {status_code})"
            )

        raise RuntimeError(
            f"Mistral OCR {step}失败 (HTTP {status_code}): {error_detail}"
        )

    def _parse_ocr_response(
        self, ocr_data: dict, page_numbers: List[int]
    ) -> OCRResult:
        """
        解析 Mistral OCR 响应，提取指定页码的文本

        参数:
            ocr_data: Mistral OCR API 返回的 JSON 数据
            page_numbers: 需要提取的页码列表（从 0 开始）
        返回:
            OCRResult 包含各页结果
        """
        api_pages = ocr_data.get("pages", [])
        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # page_numbers 从 0 开始，Mistral API 返回的 pages 列表也是从 0 开始索引
            display_page = page_num + 1  # 用于显示的页码（从 1 开始）

            if page_num < len(api_pages):
                page_data = api_pages[page_num]
                markdown_content = page_data.get("markdown", "")
                # 将 Markdown 转换为纯文本
                text = self._markdown_to_text(markdown_content)
                text = clean_ocr_text(text)
                pages_result.append(PageOCRResult(
                    page_number=display_page,
                    text=text,
                    success=True,
                ))
            else:
                # 请求的页码超出 OCR 结果范围
                error_msg = f"页码 {display_page} 超出 OCR 结果范围（共 {len(api_pages)} 页）"
                logger.warning(f"Mistral OCR: {error_msg}")
                failed_pages.append(display_page)
                errors[display_page] = error_msg
                pages_result.append(PageOCRResult(
                    page_number=display_page,
                    text="",
                    success=False,
                    error=error_msg,
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    @staticmethod
    def _markdown_to_text(markdown: str) -> str:
        """
        将 Markdown 内容转换为纯文本，清理 Markdown 标记

        委托给模块级函数 _markdown_to_text()，保持向后兼容。

        参数:
            markdown: Markdown 格式的文本
        返回:
            纯文本字符串
        """
        return _markdown_to_text(markdown)

    def _delete_file(self, client, headers: dict, file_id: str) -> None:
        """
        删除 Mistral 服务器上的临时文件，失败不影响 OCR 结果

        参数:
            client: httpx.Client 实例
            headers: 请求头（包含 Authorization）
            file_id: 要删除的文件 ID
        """
        try:
            resp = client.delete(
                f"{self._base_url}/v1/files/{file_id}",
                headers=headers,
            )
            if resp.is_success:
                logger.info(f"Mistral OCR: 远程文件已清理，file_id={file_id}")
            else:
                logger.warning(
                    f"Mistral OCR: 远程文件清理失败 (HTTP {resp.status_code})，"
                    f"file_id={file_id}"
                )
        except Exception as e:
            logger.warning(f"Mistral OCR: 远程文件清理异常: {e}，file_id={file_id}")


# ============================================================
# OCR 适配器注册表
# ============================================================

class OCRRegistry:
    """OCR 适配器注册表，管理可用适配器的注册与查找"""

    def __init__(self):
        self._adapters: Dict[str, BaseOCRAdapter] = {}

    def register(self, adapter: BaseOCRAdapter) -> None:
        """
        注册一个适配器（仅当 is_available() 为 True 时）

        参数:
            adapter: 要注册的 OCR 适配器实例
        """
        if adapter.is_available():
            self._adapters[adapter.name] = adapter
            logger.info(f"OCR 适配器已注册: {adapter.name}")
        else:
            logger.debug(f"OCR 适配器 {adapter.name} 不可用，跳过注册")

    def get_adapter(self, name: str = "auto") -> Optional[BaseOCRAdapter]:
        """
        获取适配器

        参数:
            name: 适配器名称，"auto" 时按优先级返回第一个可用适配器
        返回:
            适配器实例，无可用适配器时返回 None
        """
        if name == "auto":
            # 优先级：mistral > mineru > doc2x > paddleocr > tesseract（在线 OCR 优先）
            for key in ["mistral", "mineru", "doc2x", "paddleocr", "tesseract"]:
                if key in self._adapters:
                    logger.debug(f"自动选择 OCR 适配器: {key}")
                    return self._adapters[key]
            # 无任何可用适配器，记录可用性检测结果
            logger.warning(
                "无可用的 OCR 适配器。已检测的后端: "
                f"mistral={'已注册' if 'mistral' in self._adapters else '不可用'}, "
                f"mineru={'已注册' if 'mineru' in self._adapters else '不可用'}, "
                f"doc2x={'已注册' if 'doc2x' in self._adapters else '不可用'}, "
                f"paddleocr={'已注册' if 'paddleocr' in self._adapters else '不可用'}, "
                f"tesseract={'已注册' if 'tesseract' in self._adapters else '不可用'}"
            )
            return None
        return self._adapters.get(name)

    # 在线 OCR 适配器名称集合，用于回退时排除
    _ONLINE_ADAPTERS = {"mistral", "mineru", "doc2x"}

    def get_local_adapter(self, exclude: Optional[List[str]] = None) -> Optional[BaseOCRAdapter]:
        """
        获取本地 OCR 适配器（排除在线适配器和指定的适配器）

        用于在线 OCR 失败时回退到本地引擎。
        按优先级 paddleocr > tesseract 返回第一个可用的本地适配器。

        参数:
            exclude: 需要额外排除的适配器名称列表
        返回:
            本地适配器实例，无可用本地适配器时返回 None
        """
        exclude_set = set(self._ONLINE_ADAPTERS)
        if exclude:
            exclude_set.update(exclude)

        # 本地适配器优先级：paddleocr > tesseract
        for key in ["paddleocr", "tesseract"]:
            if key in self._adapters and key not in exclude_set:
                logger.debug(f"选择本地 OCR 适配器: {key}")
                return self._adapters[key]

        logger.warning("无可用的本地 OCR 适配器用于回退")
        return None

    def list_available(self) -> Dict[str, bool]:
        """
        列出所有已注册适配器的可用状态

        返回:
            字典，键为适配器名称，值为 True（已注册即可用）
        """
        return {name: True for name in self._adapters}


def is_ocr_available() -> dict:
    """
    检查可用的 OCR 后端

    使用全局 OCRRegistry 注册表查询已注册的适配器，
    同时保持向后兼容的返回格式。
    """
    available = _ocr_registry.list_available()
    return {
        "tesseract": available.get("tesseract", False),
        "paddleocr": available.get("paddleocr", False),
        "mistral": available.get("mistral", False),
        "mineru": available.get("mineru", False),
        "doc2x": available.get("doc2x", False),
        "any": len(available) > 0
    }


def detect_pdf_quality(text: str, page_count: int) -> Tuple[bool, str]:
    """
    Detect if PDF text extraction quality is poor and needs OCR
    Returns: (needs_ocr, reason)
    """
    if not text or len(text.strip()) < 100:
        return True, "文本内容过少，可能是扫描版PDF"
    
    # Check for garbled text (high ratio of replacement characters or control chars)
    garbled_chars = len(re.findall(r'[�□■◆●○◇△▽▲▼\x00-\x08\x0b\x0c\x0e-\x1f]', text))
    total_chars = len(text)
    
    if total_chars > 0 and garbled_chars / total_chars > 0.05:
        return True, "检测到大量乱码字符，可能是编码问题"
    
    # Check average chars per page (academic papers typically have 2000-4000 chars/page)
    avg_chars_per_page = total_chars / max(page_count, 1)
    if avg_chars_per_page < 200:
        return True, f"每页平均字符数过少({avg_chars_per_page:.0f})，可能是图片型PDF"
    
    # Check for meaningful content (not just whitespace and numbers)
    meaningful_text = re.sub(r'[\s\d\W]+', '', text)
    if len(meaningful_text) < total_chars * 0.3:
        return True, "有效文本内容比例过低"
    
    return False, "文本提取质量正常"


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR output text"""
    if not text:
        return ""
    
    # Fix common OCR errors
    text = re.sub(r'[|l](?=[a-z])', 'I', text)  # |ower -> lower
    text = re.sub(r'(?<=[a-z])[0O](?=[a-z])', 'o', text)  # g0od -> good
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove isolated single characters (OCR noise)
    text = re.sub(r'\n[a-zA-Z]\n', '\n', text)
    
    return text.strip()


class OCRService:
    """OCR service with multiple backend support"""
    
    def __init__(self, backend: str = "auto", lang: str = "chi_sim+eng"):
        """
        Initialize OCR service
        
        Args:
            backend: "tesseract", "paddleocr", or "auto"
            lang: Language code for Tesseract (chi_sim+eng for Chinese+English)
        """
        self.backend = backend
        self.lang = lang
        self._paddle_ocr = None
        
        if backend == "auto":
            if PADDLEOCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                self.backend = "paddleocr"
            elif TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
                self.backend = "tesseract"
            else:
                self.backend = "none"
    
    def _get_paddle_ocr(self):
        """Lazy load PaddleOCR instance"""
        if self._paddle_ocr is None and PADDLEOCR_AVAILABLE:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                show_log=False,
                use_gpu=False
            )
        return self._paddle_ocr
    
    def ocr_image(self, image: "Image.Image") -> str:
        """OCR a single image"""
        if self.backend == "tesseract" and TESSERACT_AVAILABLE:
            return pytesseract.image_to_string(image, lang=self.lang)
        
        elif self.backend == "paddleocr" and PADDLEOCR_AVAILABLE:
            ocr = self._get_paddle_ocr()
            import numpy as np
            img_array = np.array(image)
            result = ocr.ocr(img_array, cls=True)
            
            if not result or not result[0]:
                return ""
            
            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    lines.append(text)
            return '\n'.join(lines)
        
        return ""
    
    def ocr_pdf_bytes(self, pdf_bytes: bytes, dpi: int = 200) -> dict:
        """
        OCR a PDF from bytes
        
        Args:
            pdf_bytes: PDF file content as bytes
            dpi: Resolution for PDF to image conversion
            
        Returns:
            dict with full_text, pages, total_pages
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image not installed. Run: pip install pdf2image")
        
        if self.backend == "none":
            raise RuntimeError("No OCR backend available. Install pytesseract or paddleocr")
        
        print(f"Starting OCR with backend: {self.backend}")
        
        # Convert PDF to images (with poppler path if available)
        try:
            if _poppler_path:
                images = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=_poppler_path)
            else:
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
        except Exception as e:
            raise RuntimeError(
                f"PDF转图片失败: {e}\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )
        
        total_pages = len(images)
        
        pages = []
        full_text_parts = []
        
        for i, image in enumerate(images):
            print(f"OCR processing page {i + 1}/{total_pages}...")
            
            page_text = self.ocr_image(image)
            page_text = clean_ocr_text(page_text)
            
            pages.append({
                "page": i + 1,
                "content": page_text
            })
            full_text_parts.append(page_text)
        
        full_text = "\n\n".join(full_text_parts)
        
        return {
            "full_text": full_text,
            "total_pages": total_pages,
            "pages": pages,
            "ocr_used": True,
            "ocr_backend": self.backend
        }


# ============================================================
# 全局 OCR 注册表实例
# ============================================================

# 创建全局注册表并注册可用的适配器
_ocr_registry = OCRRegistry()
_ocr_registry.register(TesseractAdapter())
_ocr_registry.register(PaddleOCRAdapter())

# 注册在线 OCR 适配器：加载 Mistral OCR 配置并注册
_mistral_config = _load_online_ocr_config("mistral")
_ocr_registry.register(MistralAdapter(
    api_key=_mistral_config.get("api_key", ""),
    base_url=_mistral_config.get("base_url", "https://api.mistral.ai"),
))

# 注册在线 OCR 适配器：加载 MinerU OCR 配置并注册
_mineru_config = _load_online_ocr_config("mineru")
_ocr_registry.register(MinerUAdapter(
    worker_url=_mineru_config.get("worker_url", ""),
    auth_key=_mineru_config.get("auth_key", ""),
    token=_mineru_config.get("token", ""),
    token_mode=_mineru_config.get("token_mode", "frontend"),
    enable_ocr=_mineru_config.get("enable_ocr", True),
    enable_formula=_mineru_config.get("enable_formula", True),
    enable_table=_mineru_config.get("enable_table", True),
))

# 注册在线 OCR 适配器：加载 Doc2X OCR 配置并注册
_doc2x_config = _load_online_ocr_config("doc2x")
_ocr_registry.register(Doc2XAdapter(
    worker_url=_doc2x_config.get("worker_url", ""),
    auth_key=_doc2x_config.get("auth_key", ""),
    token=_doc2x_config.get("token", ""),
    token_mode=_doc2x_config.get("token_mode", "frontend"),
))


# 保留旧的全局 OCRService 实例（向后兼容）
_ocr_service: Optional[OCRService] = None


def get_ocr_service(backend: str = "auto") -> OCRService:
    """
    获取 OCR 服务实例

    优先通过 OCRRegistry 注册表获取适配器，
    如果注册表中有可用适配器则使用注册表的结果确定后端，
    否则回退到旧的 OCRService 实例化逻辑。
    保持向后兼容的函数签名。

    参数:
        backend: OCR 后端名称，"auto" 时自动选择
    返回:
        OCRService 实例
    """
    global _ocr_service

    # 通过注册表确定实际可用的后端
    adapter = _ocr_registry.get_adapter(backend)
    if adapter is not None:
        resolved_backend = adapter.name
    else:
        resolved_backend = backend

    if _ocr_service is None or _ocr_service.backend != resolved_backend:
        _ocr_service = OCRService(backend=resolved_backend)
    return _ocr_service


def ocr_pdf(pdf_bytes: bytes, backend: str = "auto", dpi: int = 200) -> dict:
    """
    便捷函数：对 PDF 执行 OCR

    使用全局 OCRRegistry 注册表获取适配器，
    保持向后兼容的函数签名和返回格式。

    参数:
        pdf_bytes: PDF 文件字节内容
        backend: OCR 后端名称，"auto" 时自动选择
        dpi: 图像转换分辨率
    返回:
        包含提取文本数据的字典
    """
    service = get_ocr_service(backend)
    return service.ocr_pdf_bytes(pdf_bytes, dpi=dpi)
