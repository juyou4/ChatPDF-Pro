"""
OCR Service for PDF text extraction
Supports both local Tesseract OCR and cloud OCR APIs
"""
import io
import os
import re
from typing import List, Optional, Tuple
from pathlib import Path

# OCR availability flags
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
    """查找 Poppler 路径"""
    base_dir = Path(__file__).resolve().parents[2]
    ocr_dir = base_dir / "ocr_tools"
    
    poppler_paths = [
        ocr_dir / "poppler" / "Library" / "bin",  # Windows 本地安装
        Path("/usr/bin"),  # Linux
        Path("/usr/local/bin"),  # macOS Homebrew
        Path("/opt/homebrew/bin"),  # macOS M1 Homebrew
    ]
    
    for path in poppler_paths:
        if (path / "pdftoppm.exe").exists() or (path / "pdftoppm").exists():
            return str(path)
    
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


def is_ocr_available() -> dict:
    """Check which OCR backends are available"""
    return {
        "tesseract": TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE,
        "paddleocr": PADDLEOCR_AVAILABLE and PDF2IMAGE_AVAILABLE,
        "any": (TESSERACT_AVAILABLE or PADDLEOCR_AVAILABLE) and PDF2IMAGE_AVAILABLE
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
            raise RuntimeError(f"PDF转图片失败: {e}. 请确保已安装 Poppler")
        
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


# Global OCR service instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service(backend: str = "auto") -> OCRService:
    """Get or create OCR service instance"""
    global _ocr_service
    if _ocr_service is None or _ocr_service.backend != backend:
        _ocr_service = OCRService(backend=backend)
    return _ocr_service


def ocr_pdf(pdf_bytes: bytes, backend: str = "auto", dpi: int = 200) -> dict:
    """
    Convenience function to OCR a PDF
    
    Args:
        pdf_bytes: PDF file content
        backend: OCR backend to use
        dpi: Resolution for conversion
        
    Returns:
        Extracted text data
    """
    service = get_ocr_service(backend)
    return service.ocr_pdf_bytes(pdf_bytes, dpi=dpi)
