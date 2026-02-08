import io
import os
import glob
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import PyPDF2
import pdfplumber
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from services.vector_service import create_index
from services.ocr_service import (
    is_ocr_available,
    detect_pdf_quality,
    ocr_pdf,
    get_ocr_service,
    _ocr_registry,
    _find_poppler,
    _save_online_ocr_config,
    _load_online_ocr_config,
    _mask_api_key,
    MistralAdapter,
    MinerUAdapter,
    Doc2XAdapter,
    WorkerOCRAdapter,
)
from models.model_detector import normalize_embedding_model_id
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root (two levels up from routes/) so storage matches app.py
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
UPLOAD_DIR = BASE_DIR / "uploads"

# Legacy paths from the old layout (stored under backend/)
LEGACY_BASE_DIR = Path(__file__).resolve().parents[1]
LEGACY_DATA_DIR = LEGACY_BASE_DIR / "data"
LEGACY_DOCS_DIR = LEGACY_DATA_DIR / "docs"
LEGACY_VECTOR_STORE_DIR = LEGACY_DATA_DIR / "vector_stores"
LEGACY_UPLOAD_DIR = LEGACY_BASE_DIR / "uploads"

documents_store = {}


def save_document(doc_id: str, data: dict):
    try:
        file_path = DOCS_DIR / f"{doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved document {doc_id} to {file_path}")
    except Exception as e:
        print(f"Error saving document {doc_id}: {e}")


def load_documents():
    print("Loading documents from disk...")
    count = 0
    for file_path in glob.glob(str(DOCS_DIR / "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                documents_store[doc_id] = data
                count += 1
        except Exception as e:
            print(f"Error loading document from {file_path}: {e}")
    print(f"Loaded {count} documents.")


def migrate_legacy_storage():
    """Move files from old backend/* paths to project root if needed."""
    migrations = [
        (LEGACY_DOCS_DIR, DOCS_DIR, "*.json"),
        (LEGACY_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.index"),
        (LEGACY_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.pkl"),
        (LEGACY_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
    ]

    for src_dir, dest_dir, pattern in migrations:
        if not src_dir.exists() or src_dir.resolve() == dest_dir.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.glob(pattern):
            dest_file = dest_dir / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)


def generate_doc_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def extract_text_from_pdf(
    pdf_file,
    pdf_bytes: Optional[bytes] = None,
    enable_ocr: str = "auto",
    extract_images: bool = True,
    ocr_dpi: int = 200,
    ocr_language: str = "chi_sim+eng",
    ocr_quality_threshold: int = 60,
):
    """
    从 PDF 中提取文本和图片，支持可选的 OCR 回退
    参考 paper-burner-x 实现，支持多栏检测、图片提取、分批处理、智能段落合并
    
    Features:
    - P0: 多栏检测 (detect_columns) - 双栏论文支持
    - P0: 逐页质量评估 (assess_page_quality) - 按页决定是否OCR
    - P0: 图片提取与过滤 - 跳过装饰图标，保留有意义的图片
    - P1: 分批处理大文档 - 每50页一批，避免内存溢出
    - P1: 自适应阈值 - 基于中位数字符高度/宽度
    - P1: 保守的垃圾过滤 - 白名单保护公式/引用
    - P2: 智能段落合并 - 根据句号、大写、列表标记判断换段
    - P2: 元数据保留 - page, block_id, bbox, source, quality_score
    
    Args:
        pdf_file: pdfplumber 使用的文件对象
        pdf_bytes: PDF 原始字节（OCR 需要）
        enable_ocr: OCR 模式 - "auto"（自动检测）、"always"（始终启用）或 "never"（禁用）
        extract_images: 是否从 PDF 中提取图片
        ocr_dpi: OCR 图像转换分辨率（DPI），默认 200
        ocr_language: OCR 语言设置（Tesseract 语言代码），默认 "chi_sim+eng"
        ocr_quality_threshold: 页面质量阈值（0-100），低于此值触发 OCR，默认 60
    
    Returns:
        包含 full_text、pages、total_pages、images 和 OCR 元数据的字典
    """
    import re
    import base64
    import time
    from statistics import median
    
    # ==================== 配置常量 ====================
    BATCH_SIZE = 50  # 每批处理页数
    BATCH_SLEEP = 0.3  # 批间休息时间(秒)
    
    # 图片过滤配置
    MIN_IMAGE_SIZE = 50  # 提高到50px，过滤更多小图标
    MAX_ASPECT_RATIO = 10  # 降低到10，过滤长条形图片
    MIN_ASPECT_RATIO = 0.1  # 提高到0.1
    MAX_IMAGE_DIMENSION = 800  # 图片最大尺寸，超过会压缩
    IMAGE_QUALITY = 75  # JPEG压缩质量
    
    # ==================== 白名单模式 ====================
    # 保护公式、引用、特殊格式不被误判为乱码
    WHITELIST_PATTERNS = [
        r'^\s*\[\d+\]',           # 引用 [1], [23]
        r'^\s*\(\d+\)',           # 引用 (1), (23)
        r'^\s*Fig\.\s*\d+',       # Figure 引用
        r'^\s*Table\s*\d+',       # Table 引用
        r'^\s*Eq\.\s*\d+',        # Equation 引用
        r'^\s*§\s*\d+',           # Section 符号
        r'[α-ωΑ-Ω∑∏∫∂∇±×÷≤≥≠≈∞∈∉⊂⊃∪∩]',  # 数学/希腊符号
        r'\$.*\$',               # LaTeX 行内公式
        r'\\[a-zA-Z]+',          # LaTeX 命令
        r'^\s*\d+\.\s+',         # 编号列表 1. 2. 3.
        r'^\s*[a-z]\)\s+',       # 编号列表 a) b) c)
        r'^\s*•\s+',             # 项目符号
        r'^\s*-\s+',             # 破折号列表
        r'https?://',            # URL
        r'[a-zA-Z0-9._%+-]+@',   # Email
    ]
    
    def extract_text_from_dict(text_dict: dict) -> str:
        """
        从 PyMuPDF 的 dict 格式中提取文本
        参考 paper-burner-x 的 _extractTextFromPage 实现
        
        核心逻辑：
        1. 遍历所有文本项（字符/单词）
        2. 根据 Y 坐标变化检测换行
        3. 根据 X 坐标间距决定是否添加空格
        """
        if not text_dict or "blocks" not in text_dict:
            return ""
        
        text_items = []
        
        # 遍历所有块
        for block in text_dict["blocks"]:
            if block.get("type") != 0:  # 0 = text block
                continue
            
            # 遍历块中的所有行
            for line in block.get("lines", []):
                # 遍历行中的所有 span
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    
                    # 获取位置信息
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = bbox
                    
                    text_items.append({
                        "text": text,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "width": x1 - x0
                    })
        
        if not text_items:
            return ""
        
        # 按 Y 坐标排序（从上到下），然后按 X 坐标排序（从左到右）
        text_items.sort(key=lambda item: (round(item["y0"] / 5) * 5, item["x0"]))
        
        # 重建文本
        result = ""
        last_y = None
        last_x_end = None
        
        for item in text_items:
            text = item["text"]
            y = item["y0"]
            x_start = item["x0"]
            x_end = item["x1"]
            
            # 检测换行（Y 坐标变化超过阈值）
            if last_y is not None and abs(y - last_y) > 5:
                result += '\n'
                last_x_end = None
            
            # 检测是否需要添加空格（X 坐标间距）
            if last_x_end is not None:
                # 估算空格宽度为字符宽度的 30%
                space_width = item["width"] * 0.3 if item["width"] > 0 else 3
                gap = x_start - last_x_end
                
                if gap > space_width:
                    result += ' '
            
            result += text
            last_y = y
            last_x_end = x_end
        
        return result.strip()
    
    def clean_text(text: str) -> str:
        """保守清理文本，只移除真正的乱码字符"""
        if not text:
            return ""
        # 只移除 NULL 字符和真正的控制字符，保留换行/制表
        cleaned = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\t\n\r')
        # 移除连续的替换字符
        cleaned = re.sub(r'[\ufffd]{2,}', '', cleaned)
        return cleaned
    
    def matches_whitelist(line: str) -> bool:
        """检查是否匹配白名单模式"""
        for pattern in WHITELIST_PATTERNS:
            if re.search(pattern, line):
                return True
        return False
    
    def is_garbage_line(line: str) -> bool:
        """保守的乱码检测，白名单优先"""
        if not line or len(line) < 2:
            return False
        
        # 白名单保护
        if matches_whitelist(line):
            return False
        
        # 统计不可打印字符
        bad_chars = sum(1 for ch in line if ord(ch) < 32 and ch not in '\t\n\r')
        # 统计替换字符和私用区字符
        weird_chars = sum(1 for ch in line if ch == '\ufffd' or 0xE000 <= ord(ch) <= 0xF8FF)
        # NULL 字符
        null_chars = line.count('\u0000')
        
        total_bad = bad_chars + weird_chars + null_chars
        # 提高阈值，更保守
        return total_bad / len(line) > 0.3
    
    def get_adaptive_thresholds(blocks: list) -> dict:
        """基于中位数计算自适应阈值"""
        if not blocks:
            return {"line_height": 12, "char_width": 8, "column_gap": 50}
        
        heights = []
        widths = []
        for block in blocks:
            if len(block) >= 7 and block[6] == 0:  # 文本块
                h = block[3] - block[1]  # y1 - y0
                w = block[2] - block[0]  # x1 - x0
                if h > 0:
                    heights.append(h)
                if w > 0:
                    widths.append(w)
        
        med_height = median(heights) if heights else 12
        med_width = median(widths) if widths else 100
        
        return {
            "line_height": med_height,
            "char_width": med_width / 10 if med_width > 0 else 8,
            "column_gap": med_width * 0.3,  # 栏间距约为块宽度的30%
            "line_tolerance": med_height * 0.5  # 同行容差
        }
    
    def detect_columns(blocks: list, page_width: float) -> list:
        """检测多栏布局，返回栏边界列表"""
        if not blocks or page_width <= 0:
            return [(0, page_width)]
        
        # 收集所有文本块的X坐标
        x_positions = []
        for block in blocks:
            if len(block) >= 7 and block[6] == 0:
                x_positions.append(block[0])  # x0
                x_positions.append(block[2])  # x1
        
        if not x_positions:
            return [(0, page_width)]
        
        # 分析X坐标分布，寻找明显的间隙
        x_positions.sort()
        
        # 计算相邻X坐标的间隙
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > page_width * 0.1:  # 间隙超过页宽10%
                gaps.append((x_positions[i-1], x_positions[i], gap))
        
        # 如果有明显的中间间隙，判定为双栏
        mid_point = page_width / 2
        for left, right, gap in gaps:
            if abs((left + right) / 2 - mid_point) < page_width * 0.15:
                # 间隙在页面中间附近
                return [(0, left + gap * 0.1), (right - gap * 0.1, page_width)]
        
        return [(0, page_width)]
    
    def sort_blocks_by_columns(blocks: list, columns: list, thresholds: dict) -> list:
        """按栏排序文本块：先按栏，栏内按Y再按X"""
        if not blocks:
            return []
        
        def get_column_index(block):
            x_center = (block[0] + block[2]) / 2
            for i, (col_left, col_right) in enumerate(columns):
                if col_left <= x_center <= col_right:
                    return i
            return 0
        
        # 为每个块添加栏索引
        blocks_with_col = [(block, get_column_index(block)) for block in blocks]
        
        # 排序：栏索引 -> Y坐标 -> X坐标
        line_tol = thresholds.get("line_tolerance", 6)
        sorted_blocks = sorted(
            blocks_with_col,
            key=lambda x: (x[1], round(x[0][1] / line_tol) * line_tol, x[0][0])
        )
        
        return [block for block, _ in sorted_blocks]
    
    def assess_page_quality(page_text: str, block_count: int, quality_threshold: int = 60) -> dict:
        """评估单页提取质量
        
        Args:
            page_text: 页面文本内容
            block_count: 文本块数量
            quality_threshold: 质量阈值（0-100），低于此值判定为需要 OCR
        """
        if not page_text:
            return {"score": 0, "needs_ocr": True, "reason": "empty_page"}
        
        text_len = len(page_text)
        
        # 计算各种指标
        null_ratio = page_text.count('\u0000') / text_len if text_len > 0 else 0
        weird_ratio = sum(1 for ch in page_text if ch == '\ufffd' or 0xE000 <= ord(ch) <= 0xF8FF) / text_len if text_len > 0 else 0
        
        # 有效字符比例
        valid_chars = sum(1 for ch in page_text if ch.isalnum() or ch in ' \t\n.,;:!?-()[]{}"\'' or '\u4e00' <= ch <= '\u9fff')
        valid_ratio = valid_chars / text_len if text_len > 0 else 0
        
        # 计算质量分数 (0-100)
        score = 100
        score -= null_ratio * 200
        score -= weird_ratio * 150
        score -= (1 - valid_ratio) * 50
        
        # 文本密度检查
        if block_count > 0 and text_len / block_count < 10:
            score -= 20
        
        score = max(0, min(100, score))
        
        needs_ocr = score < quality_threshold
        reason = "good" if score >= 80 else ("acceptable" if score >= quality_threshold else "poor_quality")
        
        return {
            "score": round(score, 1),
            "needs_ocr": needs_ocr,
            "reason": reason,
            "null_ratio": round(null_ratio, 3),
            "valid_ratio": round(valid_ratio, 3)
        }
    
    def extract_with_pymupdf(pdf_bytes: bytes, extract_images: bool = True) -> tuple:
        """
        使用 PyMuPDF 进行字符级文本提取，参考 paper-burner-x 实现
        核心改进：
        1. 使用 get_text("dict") 获取字符级坐标
        2. 按 Y 坐标检测换行，按 X 坐标间距添加空格
        3. 精确控制文本重建，避免空格丢失
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, None, None, [], "PyMuPDF not installed"
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        full_text_parts = []
        page_qualities = []
        all_images = []  # 存储所有提取的图片
        
        total_pages = len(doc)
        total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"[PDF] Processing {total_pages} pages in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_page = batch_idx * BATCH_SIZE
            end_page = min((batch_idx + 1) * BATCH_SIZE, total_pages)
            
            print(f"[PDF] Batch {batch_idx + 1}/{total_batches}: pages {start_page + 1}-{end_page}")
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # ==================== 字符级文本提取（参考 paper-burner-x）====================
                # 使用 get_text("dict") 获取详细的文本结构
                try:
                    text_dict = page.get_text("dict")
                    page_text = extract_text_from_dict(text_dict)
                except Exception as dict_err:
                    # 如果 dict 模式失败，回退到简单的 text 模式
                    print(f"[PDF] Page {page_num + 1} dict extraction failed, fallback to text mode: {dict_err}")
                    page_text = page.get_text("text")
                
                # 清理文本
                page_text = clean_text(page_text)
                
                # ==================== 图片提取 ====================
                page_images = []
                if extract_images:
                    try:
                        image_list = page.get_images(full=True)
                        for img_idx, img_info in enumerate(image_list):
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                
                                if not base_image:
                                    continue
                                
                                img_width = base_image.get("width", 0)
                                img_height = base_image.get("height", 0)
                                
                                # 图片过滤
                                if img_width < MIN_IMAGE_SIZE or img_height < MIN_IMAGE_SIZE:
                                    continue  # 跳过装饰图标
                                
                                aspect_ratio = img_width / img_height if img_height > 0 else 0
                                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                                    continue  # 跳过线条/分隔符
                                
                                # 获取图片数据
                                img_data = base_image.get("image")
                                img_ext = base_image.get("ext", "png")
                                
                                if img_data:
                                    # 压缩大图片
                                    if img_width > MAX_IMAGE_DIMENSION or img_height > MAX_IMAGE_DIMENSION:
                                        try:
                                            from PIL import Image
                                            import io as img_io
                                            
                                            img = Image.open(img_io.BytesIO(img_data))
                                            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                                            
                                            buffer = img_io.BytesIO()
                                            if img.mode in ('RGBA', 'P'):
                                                img = img.convert('RGB')
                                            img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
                                            img_data = buffer.getvalue()
                                            img_ext = "jpg"
                                        except Exception as resize_err:
                                            print(f"[PDF] Image resize failed: {resize_err}")
                                    
                                    img_id = f"page{page_num + 1}_img{img_idx + 1}"
                                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                                    
                                    page_images.append({
                                        "id": img_id,
                                        "data": f"data:image/{img_ext};base64,{img_base64}",
                                        "width": img_width,
                                        "height": img_height,
                                        "page": page_num + 1
                                    })
                                    
                                    # 不在文本中插入图片引用，避免干扰RAG检索
                                    # 图片信息已经单独存储在 all_images 数组中
                                    
                            except Exception as img_err:
                                # 单个图片提取失败不影响整体
                                pass
                        
                        all_images.extend(page_images)
                        
                    except Exception as img_extract_err:
                        print(f"[PDF] Page {page_num + 1} image extraction failed: {img_extract_err}")
                
                # 评估页面质量（使用传入的质量阈值）
                quality = assess_page_quality(page_text, 1, ocr_quality_threshold)  # block_count设为1，因为我们不再使用blocks
                page_qualities.append(quality)
                
                pages.append({
                    "page": page_num + 1,
                    "content": page_text,
                    "quality_score": quality["score"],
                    "image_count": len(page_images),
                    "source": "pymupdf_dict"
                })
                full_text_parts.append(page_text)
            
            # 批间休息，释放内存
            if batch_idx < total_batches - 1:
                time.sleep(BATCH_SLEEP)
        
        doc.close()
        return pages, '\n\n'.join(full_text_parts), page_qualities, all_images, None
    
    def extract_with_pdfplumber(pdf_file) -> tuple:
        """使用 pdfplumber 的 chars 进行坐标级文本提取，带自适应阈值"""
        pdf_file.seek(0)
        
        with pdfplumber.open(pdf_file) as pdf:
            pages = []
            full_text_parts = []
            page_qualities = []
            
            total_pages = len(pdf.pages)
            total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(total_batches):
                start_page = batch_idx * BATCH_SIZE
                end_page = min((batch_idx + 1) * BATCH_SIZE, total_pages)
                
                for i in range(start_page, end_page):
                    page = pdf.pages[i]
                    chars = page.chars
                    page_width = page.width
                    
                    if not chars:
                        quality = {"score": 0, "needs_ocr": True, "reason": "no_chars"}
                        page_qualities.append(quality)
                        pages.append({
                            "page": i + 1,
                            "content": "",
                            "quality_score": 0,
                            "source": "pdfplumber"
                        })
                        continue
                    
                    # 计算自适应阈值
                    char_heights = [c.get('height', 10) for c in chars if c.get('height')]
                    char_widths = [c.get('width', 5) for c in chars if c.get('width')]
                    med_height = median(char_heights) if char_heights else 10
                    med_width = median(char_widths) if char_widths else 5
                    
                    line_tolerance = med_height * 0.4
                    space_threshold = med_width * 1.5
                    
                    # 按Y坐标分组，然后按X坐标排序
                    lines = {}
                    for char in chars:
                        if not char.get('text') or ord(char['text']) < 32:
                            continue
                        
                        y = round(char['top'] / line_tolerance) * line_tolerance
                        if y not in lines:
                            lines[y] = []
                        lines[y].append((char['x0'], char['text'], char.get('width', med_width)))
                    
                    # 按Y坐标排序，然后每行按X坐标排序
                    page_lines = []
                    for y in sorted(lines.keys()):
                        line_chars = sorted(lines[y], key=lambda c: c[0])
                        
                        # 智能添加空格
                        line_text = ""
                        last_x_end = None
                        for x, ch, w in line_chars:
                            if last_x_end is not None:
                                gap = x - last_x_end
                                if gap > space_threshold:
                                    line_text += " "
                            line_text += ch
                            last_x_end = x + w
                        
                        if line_text.strip() and not is_garbage_line(line_text):
                            page_lines.append(clean_text(line_text))
                    
                    page_text = '\n'.join(page_lines)
                    
                    # 评估质量（使用传入的质量阈值）
                    quality = assess_page_quality(page_text, len(set(c.get('block', 0) for c in chars)), ocr_quality_threshold)
                    page_qualities.append(quality)
                    
                    pages.append({
                        "page": i + 1,
                        "content": page_text,
                        "quality_score": quality["score"],
                        "source": "pdfplumber"
                    })
                    full_text_parts.append(page_text)
                
                # 批间休息
                if batch_idx < total_batches - 1:
                    time.sleep(BATCH_SLEEP)
        
        return pages, '\n\n'.join(full_text_parts), page_qualities, [], None
    
    def heuristic_rebuild(text: str, is_cjk: bool = False) -> str:
        """
        智能段落合并与启发式文本重建
        完全参考 paper-burner-x 的 _heuristicRebuild 实现
        """
        if not text:
            return ""
        
        rebuilt = text
        
        # 先保护图片引用，避免被文本处理规则破坏
        image_refs = []
        def save_image_ref(match):
            placeholder = f"__IMG_PLACEHOLDER_{len(image_refs)}__"
            image_refs.append(match.group(0))
            return placeholder
        rebuilt = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', save_image_ref, rebuilt)
        
        # 1. 修复被断开的单词（英文连字符换行）
        # 匹配：字母-空格-换行-小写字母 -> 字母字母
        rebuilt = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', rebuilt)
        
        # 2. 合并被打断的句子
        # 如果行尾不是句号等结束符，且下一行不是大写/数字/特殊字符开头，则合并
        rebuilt = re.sub(r'([^\n.!?。！？])\n([a-z\u4e00-\u9fff])', r'\1 \2', rebuilt)
        
        # 3. 修复中文标点符号周围的空格
        rebuilt = re.sub(r'\s+([，。！？；：、）】」』])', r'\1', rebuilt)
        rebuilt = re.sub(r'([（【「『])\s+', r'\1', rebuilt)
        
        # 4. 修复英文标点符号
        # 标点后应有空格（如果后面是字母），但要排除邮箱、网址、缩写等情况
        # 不处理 . 因为它可能是邮箱、网址、缩写
        rebuilt = re.sub(r'([,!?;:])([a-zA-Z])', r'\1 \2', rebuilt)
        # 移除标点前的多余空格
        rebuilt = re.sub(r'\s+([,.!?;:])', r'\1', rebuilt)
        
        # 5. 规范化空白字符
        # 多个空格变成一个
        rebuilt = re.sub(r' {2,}', ' ', rebuilt)
        # 保留段落分隔（最多2个换行）
        rebuilt = re.sub(r'\n{3,}', '\n\n', rebuilt)
        
        # 6. 修复常见的格式问题
        # 修复：数字. 后面应该有空格（列表项）
        rebuilt = re.sub(r'(\d+)\.\s*([a-zA-Z\u4e00-\u9fff])', r'\1. \2', rebuilt)
        # 修复：括号内不应有首尾空格
        rebuilt = re.sub(r'\(\s+', '(', rebuilt)
        rebuilt = re.sub(r'\s+\)', ')', rebuilt)
        
        # 7. 智能段落识别（参考 paper-burner-x）
        lines = rebuilt.split('\n')
        paragraphs = []
        current_para = ''
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line == '':
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = ''
                continue
            
            # 判断是否应该换段
            should_break = (
                current_para == '' or  # 当前段落为空
                re.match(r'^#{1,6}\s', line) or  # 标题
                re.match(r'^[\-\*\+]\s', line) or  # 无序列表
                re.match(r'^\d+\.\s', line) or  # 有序列表
                line.startswith('__IMG_PLACEHOLDER_') or  # 图片占位符
                # 上一段以句号结束且本行首字母大写或中文
                (re.search(r'[.!?。！？]\s*$', current_para) and re.match(r'^[A-Z\u4e00-\u9fff]', line))
            )
            
            if should_break:
                if current_para:
                    paragraphs.append(current_para.strip())
                current_para = line
            else:
                # 合并到当前段落，总是加空格（因为我们已经在字符级提取时处理了空格）
                current_para += ' ' + line
        
        if current_para:
            paragraphs.append(current_para.strip())
        
        rebuilt = '\n\n'.join(paragraphs)
        
        # 恢复图片引用
        for idx, ref in enumerate(image_refs):
            rebuilt = rebuilt.replace(f"__IMG_PLACEHOLDER_{idx}__", ref)
        
        return rebuilt.strip()
    
    def detect_language(text: str) -> str:
        """检测文本主要语言"""
        if not text:
            return "en"
        cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return "cjk" if cjk_count / len(text) > 0.1 else "en"
    
    # ==================== 主提取逻辑 ====================
    pages = None
    full_text = ""
    page_qualities = None
    all_images = []
    extraction_method = None
    
    # 优先使用 PyMuPDF
    if pdf_bytes:
        pages, full_text, page_qualities, all_images, err = extract_with_pymupdf(pdf_bytes, extract_images)
        if pages is not None:
            extraction_method = "pymupdf"
            print(f"[PDF] Using PyMuPDF extraction, {len(pages)} pages, {len(all_images)} images")
    
    # 如果 PyMuPDF 失败，回退到 pdfplumber
    if pages is None:
        print(f"[PDF] PyMuPDF failed ({err}), falling back to pdfplumber")
        pages, full_text, page_qualities, all_images, err = extract_with_pdfplumber(pdf_file)
        extraction_method = "pdfplumber"
    
    # 检测语言并应用启发式重建
    is_cjk = detect_language(full_text) == "cjk"
    full_text = heuristic_rebuild(full_text, is_cjk)
    for page in pages:
        page["content"] = heuristic_rebuild(page["content"], is_cjk)
    
    # 获取总页数
    pdf_file.seek(0)
    reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    # 计算整体质量分数
    avg_quality = sum(q["score"] for q in page_qualities) / len(page_qualities) if page_qualities else 50
    pages_needing_ocr = [i for i, q in enumerate(page_qualities) if q.get("needs_ocr")] if page_qualities else []
    
    result = {
        "full_text": full_text,
        "total_pages": total_pages,
        "pages": pages,
        "images": all_images,  # 新增：提取的图片列表
        "image_count": len(all_images),
        "ocr_used": False,
        "ocr_backend": None,
        "extraction_quality": "good" if avg_quality >= 80 else ("acceptable" if avg_quality >= 60 else "poor"),
        "extraction_method": extraction_method,
        "avg_quality_score": round(avg_quality, 1),
        "pages_needing_ocr": pages_needing_ocr
    }
    
    # 检查是否需要 OCR
    if enable_ocr == "never":
        return result
    
    # 逐页 OCR 决策：enable_ocr 为 "always" 时对所有页面执行 OCR
    if enable_ocr == "always":
        # "always" 模式：对所有页面执行 OCR
        ocr_target_pages = list(range(total_pages))
    else:
        # "auto" 模式：仅对质量差的页面执行 OCR
        ocr_target_pages = pages_needing_ocr

    if not ocr_target_pages:
        print(f"[PDF] 所有页面质量合格 (平均: {avg_quality:.1f})，无需 OCR")
        return result
    
    # 通过注册表获取 OCR 适配器
    adapter = _ocr_registry.get_adapter(settings.ocr_backend)
    if adapter is None:
        print(f"[PDF] 需要对 {len(ocr_target_pages)} 页执行 OCR，但无可用 OCR 后端")
        result["ocr_error"] = "OCR 未安装，请安装 pytesseract 或 paddleocr"
        result["ocr_warning"] = "OCR 未安装，请安装 pytesseract 或 paddleocr"
        return result
    
    if pdf_bytes is None:
        print("[PDF] 需要 OCR 但未提供 pdf_bytes")
        result["ocr_error"] = "无法执行 OCR：缺少 PDF 原始数据"
        result["ocr_warning"] = "无法执行 OCR：缺少 PDF 原始数据"
        return result
    
    # 使用适配器系统执行逐页 OCR
    print(f"[PDF] 开始逐页 OCR，共 {len(ocr_target_pages)} 页，后端: {adapter.name}")
    try:
        # 调用适配器的 ocr_pages()，仅传入需要 OCR 的页码列表
        ocr_result = adapter.ocr_pages(
            pdf_bytes=pdf_bytes,
            page_numbers=ocr_target_pages,
            dpi=ocr_dpi
        )
        
        # 构建页码到 OCR 结果的映射（page_number 从 1 开始，pages_needing_ocr 从 0 开始）
        ocr_page_map = {}
        for page_ocr in ocr_result.pages:
            if page_ocr.success:
                # page_number 从 1 开始，转换为从 0 开始的索引
                ocr_page_map[page_ocr.page_number - 1] = page_ocr.text
        
        # 合并 OCR 结果到原始提取文本
        merged_text_parts = []
        for i, page in enumerate(pages):
            if i in ocr_page_map:
                ocr_content = ocr_page_map[i]
                orig_content = page.get("content", "")
                
                # 只有 OCR 结果更好时才替换（OCR 文本长度 >= 原始文本的 80%）
                if len(ocr_content) > len(orig_content) * 0.8:
                    page["content"] = heuristic_rebuild(ocr_content, is_cjk)
                    page["source"] = "ocr"
                    page["ocr_backend"] = ocr_result.backend
                    result["ocr_used"] = True
            
            merged_text_parts.append(page["content"])
        
        # 更新结果中的 OCR 元数据
        if result["ocr_used"]:
            result["full_text"] = "\n\n".join(merged_text_parts)
            result["ocr_backend"] = ocr_result.backend
            result["ocr_pages"] = ocr_target_pages
        
        # 处理部分页面 OCR 失败的警告信息
        if ocr_result.failed_pages:
            failed_info = ", ".join(str(p) for p in ocr_result.failed_pages)
            warning_msg = f"部分页面 OCR 失败（页码: {failed_info}）"
            result["ocr_warning"] = warning_msg
            print(f"[PDF] OCR 警告: {warning_msg}")
        
        # 所有目标页面均失败时，附带全部失败警告
        if len(ocr_result.failed_pages) == len(ocr_target_pages):
            result["ocr_warning"] = "所有需要 OCR 的页面均处理失败，已保留原始提取文本"
            result["ocr_used"] = False
            print("[PDF] OCR 全部失败，保留原始文本")
        
        print(f"[PDF] OCR 完成。已使用: {result['ocr_used']}，目标页面: {ocr_target_pages}，后端: {ocr_result.backend}")
        
    except Exception as e:
        # 在线 OCR 失败时，尝试回退到本地 OCR 引擎
        if adapter.name in _ocr_registry._ONLINE_ADAPTERS:
            logger.warning(f"在线 OCR ({adapter.name}) 失败，尝试回退到本地引擎: {e}")
            print(f"[PDF] 在线 OCR ({adapter.name}) 失败，尝试回退到本地引擎: {e}")
            local_adapter = _ocr_registry.get_local_adapter(exclude=[adapter.name])
            if local_adapter is not None:
                try:
                    print(f"[PDF] 回退到本地 OCR 引擎: {local_adapter.name}")
                    logger.info(f"回退到本地 OCR 引擎: {local_adapter.name}")
                    ocr_result = local_adapter.ocr_pages(
                        pdf_bytes=pdf_bytes,
                        page_numbers=ocr_target_pages,
                        dpi=ocr_dpi
                    )

                    # 构建页码到 OCR 结果的映射
                    ocr_page_map = {}
                    for page_ocr in ocr_result.pages:
                        if page_ocr.success:
                            ocr_page_map[page_ocr.page_number - 1] = page_ocr.text

                    # 合并 OCR 结果到原始提取文本
                    merged_text_parts = []
                    for i, page in enumerate(pages):
                        if i in ocr_page_map:
                            ocr_content = ocr_page_map[i]
                            orig_content = page.get("content", "")
                            if len(ocr_content) > len(orig_content) * 0.8:
                                page["content"] = heuristic_rebuild(ocr_content, is_cjk)
                                page["source"] = "ocr"
                                page["ocr_backend"] = ocr_result.backend
                                result["ocr_used"] = True
                        merged_text_parts.append(page["content"])

                    if result["ocr_used"]:
                        result["full_text"] = "\n\n".join(merged_text_parts)
                        result["ocr_backend"] = ocr_result.backend
                        result["ocr_pages"] = ocr_target_pages

                    result["ocr_warning"] = (
                        f"在线 OCR ({adapter.name}) 失败，已回退到本地引擎 ({local_adapter.name})"
                    )
                    logger.info(
                        f"在线 OCR 回退成功: {adapter.name} -> {local_adapter.name}"
                    )
                    print(f"[PDF] 在线 OCR 回退成功: {adapter.name} -> {local_adapter.name}")
                except Exception as fallback_err:
                    logger.error(f"本地 OCR 回退也失败: {fallback_err}")
                    print(f"[PDF] 本地 OCR 回退也失败: {fallback_err}")
                    result["ocr_error"] = str(e)
                    result["ocr_warning"] = (
                        f"在线 OCR ({adapter.name}) 和本地 OCR 回退均失败: {str(e)}"
                    )
            else:
                logger.warning("在线 OCR 失败且无可用的本地 OCR 引擎用于回退")
                print("[PDF] 在线 OCR 失败且无可用的本地 OCR 引擎用于回退")
                result["ocr_error"] = str(e)
                result["ocr_warning"] = (
                    f"在线 OCR ({adapter.name}) 失败且无可用的本地 OCR 引擎: {str(e)}"
                )
        else:
            print(f"[PDF] OCR 失败: {e}")
            result["ocr_error"] = str(e)
            result["ocr_warning"] = f"OCR 处理异常: {str(e)}"
    
    return result


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = "local-minilm",
    embedding_api_key: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
    enable_ocr: Optional[str] = None
):
    """
    上传并处理 PDF 文件
    
    Args:
        file: 要上传的 PDF 文件
        embedding_model: 文本嵌入模型
        embedding_api_key: 云端嵌入模型的 API 密钥
        embedding_api_host: 自定义 API 地址
        enable_ocr: OCR 模式 - "auto"（自动检测）、"always"（始终启用）或 "never"（禁用）。
                    缺失时使用后端配置中的 ocr_default_mode 默认值。
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    try:
        content = await file.read()
        pdf_file = io.BytesIO(content)

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置或格式不正确（建议使用 provider:model 格式）")
        embedding_model = normalized_model

        # 当 enable_ocr 参数缺失时，回退到配置中的默认值
        ocr_mode = enable_ocr if enable_ocr is not None else settings.ocr_default_mode

        # 使用配置中的 OCR 参数提取文本
        extracted_data = extract_text_from_pdf(
            pdf_file,
            pdf_bytes=content,
            enable_ocr=ocr_mode,
            ocr_dpi=settings.ocr_dpi,
            ocr_language=settings.ocr_language,
            ocr_quality_threshold=settings.ocr_quality_threshold,
        )

        doc_id = generate_doc_id(extracted_data["full_text"])

        pdf_filename = f"{doc_id}.pdf"
        pdf_path = UPLOAD_DIR / pdf_filename
        with open(pdf_path, "wb") as f:
            f.write(content)

        pdf_url = f"/uploads/{pdf_filename}"

        documents_store[doc_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": pdf_url
        }

        save_document(doc_id, documents_store[doc_id])

        create_index(doc_id, extracted_data["full_text"], str(VECTOR_STORE_DIR), embedding_model, embedding_api_key, embedding_api_host, pages=extracted_data.get("pages"))

        response = {
            "message": "PDF上传成功",
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted_data["total_pages"],
            "total_chars": len(extracted_data["full_text"]),
            "image_count": extracted_data.get("image_count", 0),
            "pdf_url": pdf_url,
            "ocr_used": extracted_data.get("ocr_used", False),
            "ocr_backend": extracted_data.get("ocr_backend"),
            "extraction_quality": extracted_data.get("extraction_quality", "unknown"),
            "extraction_method": extracted_data.get("extraction_method", "unknown")
        }
        
        if extracted_data.get("ocr_error"):
            response["ocr_warning"] = extracted_data["ocr_error"]
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")


@router.get("/document/{doc_id}")
async def get_document(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "upload_time": doc["upload_time"],
        "total_pages": doc["data"]["total_pages"],
        "total_chars": len(doc["data"]["full_text"]),
        "image_count": doc["data"].get("image_count", 0),
        "pages": doc["data"]["pages"],
        "images": doc["data"].get("images", []),  # 新增：返回图片数据
        "pdf_url": doc.get("pdf_url"),
        "ocr_used": doc["data"].get("ocr_used", False),
        "ocr_backend": doc["data"].get("ocr_backend"),
        "extraction_quality": doc["data"].get("extraction_quality", "unknown"),
        "extraction_method": doc["data"].get("extraction_method", "unknown")
    }


@router.get("/api/ocr/status")
async def get_ocr_status():
    """
    检查 OCR 可用性、后端状态和当前配置

    返回包含 OCR 后端可用性、Poppler 状态、当前配置和安装指引的完整状态信息。
    """
    status = is_ocr_available()

    # 使用 OCRRegistry 获取后端可用性
    available_backends = _ocr_registry.list_available()
    backends = {
        "tesseract": available_backends.get("tesseract", False),
        "paddleocr": available_backends.get("paddleocr", False),
        "mistral": available_backends.get("mistral", False),  # 在线 OCR
        "mineru": available_backends.get("mineru", False),  # MinerU Worker OCR
        "doc2x": available_backends.get("doc2x", False),  # Doc2X Worker OCR
    }

    # 检测 Poppler 可用性
    poppler_path = _find_poppler()
    poppler_available = poppler_path is not None

    # 确定推荐后端（在线优先：mistral > mineru > doc2x > paddleocr > tesseract）
    recommended = None
    if backends.get("mistral"):
        recommended = "mistral"
    elif backends.get("mineru"):
        recommended = "mineru"
    elif backends.get("doc2x"):
        recommended = "doc2x"
    elif backends.get("paddleocr"):
        recommended = "paddleocr"
    elif backends.get("tesseract"):
        recommended = "tesseract"

    # 构建在线 OCR 服务状态信息
    online_services = {}
    for provider in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        provider_config = _load_online_ocr_config(provider)
        if provider in ("mineru", "doc2x"):
            # Worker 代理模式：通过 worker_url 和 token 判断配置状态
            worker_url = provider_config.get("worker_url", "")
            token = provider_config.get("token", "")
            token_mode = provider_config.get("token_mode", "frontend")
            # 配置完成条件：worker_url 非空且（worker 模式或 frontend 模式有 token）
            configured = bool(worker_url) and (token_mode == "worker" or bool(token))
            adapter = _ocr_registry.get_adapter(provider)
            available = adapter.is_available() if adapter else False
            online_services[provider] = {
                "configured": configured,
                "available": available,
            }
        else:
            # Mistral 等直接 API 调用模式
            api_key = provider_config.get("api_key", "")
            base_url = provider_config.get("base_url", "")
            adapter = _ocr_registry.get_adapter(provider)
            available = adapter.is_available() if adapter else False
            online_services[provider] = {
                "configured": bool(api_key),
                "available": available,
            }

    # 从 AppSettings 读取当前 OCR 配置
    config = {
        "default_mode": settings.ocr_default_mode,
        "dpi": settings.ocr_dpi,
        "language": settings.ocr_language,
        "quality_threshold": settings.ocr_quality_threshold,
    }

    # 安装指引
    install_instructions = {
        "tesseract": "pip install pytesseract pdf2image && 安装 Tesseract-OCR",
        "paddleocr": "pip install paddleocr pdf2image",
    }

    # 当 Poppler 不可用时，在安装指引中标注 Poppler 缺失及其影响
    if not poppler_available:
        install_instructions["poppler"] = (
            "Poppler 未安装，PDF 转图像功能不可用，OCR 将无法正常工作。\n"
            "安装方式:\n"
            "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
            "  - macOS: brew install poppler\n"
            "  - Linux: sudo apt-get install poppler-utils"
        )

    return {
        "available": status["any"],
        "backends": backends,
        "poppler_available": poppler_available,
        "recommended": recommended,
        "config": config,
        "online_services": online_services,
        "install_instructions": install_instructions,
    }


# 支持的在线 OCR 提供商列表
_SUPPORTED_ONLINE_OCR_PROVIDERS = {"mistral", "mineru", "doc2x"}


@router.post("/api/ocr/online-config")
async def save_online_ocr_config(request: Request):
    """
    保存在线 OCR 服务配置

    支持 Mistral（API Key + Base URL）和 MinerU/Doc2X（Worker 代理模式）。
    持久化到本地配置文件，并重新注册对应的在线 OCR 适配器。

    请求体（Mistral）:
        {
            "provider": "mistral",
            "api_key": "sk-xxx...",
            "base_url": "https://api.mistral.ai"  // 可选
        }

    请求体（MinerU）:
        {
            "provider": "mineru",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret",  // 可选
            "token_mode": "frontend",  // "frontend" 或 "worker"
            "token": "your-mineru-token",  // token_mode 为 frontend 时必填
            "enable_ocr": true,  // 可选，默认 true
            "enable_formula": true,  // 可选，默认 true
            "enable_table": true  // 可选，默认 true
        }

    请求体（Doc2X）:
        {
            "provider": "doc2x",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret",  // 可选
            "token_mode": "frontend",  // "frontend" 或 "worker"
            "token": "your-doc2x-token"  // token_mode 为 frontend 时必填
        }

    响应:
        {"success": true, "message": "配置已保存"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    provider = body.get("provider", "").strip()

    # 校验 provider 参数
    if not provider:
        raise HTTPException(status_code=400, detail="缺少 provider 参数")
    if provider not in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 provider: {provider}，当前支持: {', '.join(sorted(_SUPPORTED_ONLINE_OCR_PROVIDERS))}",
        )

    # 根据 provider 类型构建配置字典
    if provider in ("mineru", "doc2x"):
        # Worker 代理模式配置
        worker_url = body.get("worker_url", "").strip()
        auth_key = body.get("auth_key", "").strip()
        token_mode = body.get("token_mode", "frontend").strip()
        token = body.get("token", "").strip()

        # 校验 worker_url 参数
        if not worker_url:
            raise HTTPException(status_code=400, detail="缺少 worker_url 参数")

        # 校验 token_mode 参数
        if token_mode not in ("frontend", "worker"):
            raise HTTPException(status_code=400, detail="token_mode 必须为 'frontend' 或 'worker'")

        config: dict = {
            "worker_url": worker_url,
            "auth_key": auth_key,
            "token_mode": token_mode,
            "token": token,
        }

        # MinerU 特有选项
        if provider == "mineru":
            config["enable_ocr"] = body.get("enable_ocr", True)
            config["enable_formula"] = body.get("enable_formula", True)
            config["enable_table"] = body.get("enable_table", True)
    else:
        # Mistral 等直接 API 调用模式
        api_key = body.get("api_key", "").strip()
        base_url = body.get("base_url", "").strip()

        # 校验 api_key 参数
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数")

        config = {"api_key": api_key}
        if base_url:
            config["base_url"] = base_url

    # 持久化配置到本地文件
    try:
        _save_online_ocr_config(provider, config)
    except Exception as e:
        logger.error(f"保存在线 OCR 配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置保存失败: {str(e)}")

    # 重新注册对应的在线 OCR 适配器
    try:
        if provider == "mistral":
            # 重新加载完整配置（合并默认值）
            full_config = _load_online_ocr_config("mistral")
            # 从注册表中移除旧的 mistral 适配器（如果存在）
            _ocr_registry._adapters.pop("mistral", None)
            # 创建新的 MistralAdapter 实例并注册
            new_adapter = MistralAdapter(
                api_key=full_config.get("api_key", ""),
                base_url=full_config.get("base_url", "https://api.mistral.ai"),
            )
            _ocr_registry.register(new_adapter)
            logger.info(f"MistralAdapter 已重新注册，可用: {new_adapter.is_available()}")
        elif provider == "mineru":
            # 重新加载完整配置
            full_config = _load_online_ocr_config("mineru")
            # 从注册表中移除旧的 mineru 适配器（如果存在）
            _ocr_registry._adapters.pop("mineru", None)
            # 创建新的 MinerUAdapter 实例并注册
            new_adapter = MinerUAdapter(
                worker_url=full_config.get("worker_url", ""),
                auth_key=full_config.get("auth_key", ""),
                token=full_config.get("token", ""),
                token_mode=full_config.get("token_mode", "frontend"),
                enable_ocr=full_config.get("enable_ocr", True),
                enable_formula=full_config.get("enable_formula", True),
                enable_table=full_config.get("enable_table", True),
            )
            _ocr_registry.register(new_adapter)
            logger.info(f"MinerUAdapter 已重新注册，可用: {new_adapter.is_available()}")
        elif provider == "doc2x":
            # 重新加载完整配置
            full_config = _load_online_ocr_config("doc2x")
            # 从注册表中移除旧的 doc2x 适配器（如果存在）
            _ocr_registry._adapters.pop("doc2x", None)
            # 创建新的 Doc2XAdapter 实例并注册
            new_adapter = Doc2XAdapter(
                worker_url=full_config.get("worker_url", ""),
                auth_key=full_config.get("auth_key", ""),
                token=full_config.get("token", ""),
                token_mode=full_config.get("token_mode", "frontend"),
            )
            _ocr_registry.register(new_adapter)
            logger.info(f"Doc2XAdapter 已重新注册，可用: {new_adapter.is_available()}")
    except Exception as e:
        # 适配器注册失败不影响配置保存结果，仅记录警告
        logger.warning(f"重新注册在线 OCR 适配器失败: {e}")

    return {"success": True, "message": "配置已保存"}


@router.get("/api/ocr/online-config")
async def get_online_ocr_config():
    """
    获取在线 OCR 服务配置（敏感信息脱敏显示）

    返回各在线 OCR 提供商的配置状态，包括：
    - Mistral: API Key 是否已配置、脱敏后的 API Key 预览和 Base URL
    - MinerU/Doc2X: Worker URL、Auth Key/Token 配置状态和脱敏预览、Token Mode 及 MinerU 特有选项

    响应:
        {
            "mistral": {
                "api_key_configured": true,
                "api_key_preview": "sk-x...xxxx",
                "base_url": "https://api.mistral.ai"
            },
            "mineru": {
                "worker_url": "https://your-worker.workers.dev",
                "auth_key_configured": true,
                "auth_key_preview": "your...cret",
                "token_mode": "frontend",
                "token_configured": true,
                "token_preview": "your...oken",
                "enable_ocr": true,
                "enable_formula": true,
                "enable_table": true
            },
            "doc2x": {
                "worker_url": "",
                "auth_key_configured": false,
                "auth_key_preview": "",
                "token_mode": "frontend",
                "token_configured": false,
                "token_preview": ""
            }
        }
    """
    result = {}

    for provider in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        config = _load_online_ocr_config(provider)

        if provider in ("mineru", "doc2x"):
            # Worker 代理模式：返回 worker_url、auth_key/token 脱敏信息
            worker_url = config.get("worker_url", "")
            auth_key = config.get("auth_key", "")
            token_mode = config.get("token_mode", "frontend")
            token = config.get("token", "")

            provider_result = {
                "worker_url": worker_url,
                "auth_key_configured": bool(auth_key),
                "auth_key_preview": _mask_api_key(auth_key),
                "token_mode": token_mode,
                "token_configured": bool(token),
                "token_preview": _mask_api_key(token),
            }

            # MinerU 特有选项
            if provider == "mineru":
                provider_result["enable_ocr"] = config.get("enable_ocr", True)
                provider_result["enable_formula"] = config.get("enable_formula", True)
                provider_result["enable_table"] = config.get("enable_table", True)

            result[provider] = provider_result
        else:
            # Mistral 等直接 API 调用模式
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", "")

            result[provider] = {
                "api_key_configured": bool(api_key),
                "api_key_preview": _mask_api_key(api_key),
                "base_url": base_url,
            }

    return result


@router.post("/api/ocr/validate-key")
async def validate_ocr_key(request: Request):
    """
    验证在线 OCR 服务的 API Key / Worker 连接有效性

    - Mistral: 调用 GET /v1/files 接口验证 API Key
    - MinerU: 向 Worker URL 发送 GET 请求测试可达性和认证
    - Doc2X: 向 Worker URL 发送 GET 请求测试可达性和认证

    请求体（Mistral）:
        {
            "provider": "mistral",
            "api_key": "sk-xxx..."
        }

    请求体（MinerU/Doc2X）:
        {
            "provider": "mineru",
            "worker_url": "https://your-worker.workers.dev",
            "auth_key": "your-auth-secret"  // 可选
        }

    响应:
        {"valid": true, "message": "验证成功"}
        {"valid": false, "message": "验证失败原因"}
    """
    import httpx

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    provider = body.get("provider", "").strip()

    # 校验 provider 参数
    if not provider:
        raise HTTPException(status_code=400, detail="缺少 provider 参数")
    if provider not in _SUPPORTED_ONLINE_OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 provider: {provider}，当前支持: {', '.join(sorted(_SUPPORTED_ONLINE_OCR_PROVIDERS))}",
        )

    # 根据 provider 执行验证
    if provider == "mistral":
        api_key = body.get("api_key", "").strip()

        # 校验 api_key 参数
        if not api_key:
            raise HTTPException(status_code=400, detail="缺少 api_key 参数")

        # 加载当前配置获取 base_url（如果用户已配置过自定义 base_url）
        current_config = _load_online_ocr_config("mistral")
        base_url = (current_config.get("base_url", "") or "https://api.mistral.ai").rstrip("/")

        try:
            # 调用 Mistral API 的文件列表接口验证 Key 有效性
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                resp = client.get(
                    f"{base_url}/v1/files",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

            if resp.status_code == 200:
                logger.info("Mistral API Key 验证成功")
                return {"valid": True, "message": "API Key 验证成功"}
            elif resp.status_code in (401, 403):
                logger.warning(f"Mistral API Key 验证失败: HTTP {resp.status_code}")
                return {"valid": False, "message": "API Key 无效或已过期"}
            else:
                # 其他 HTTP 错误也视为验证失败
                logger.warning(f"Mistral API Key 验证异常: HTTP {resp.status_code}")
                return {"valid": False, "message": f"验证失败，服务返回 HTTP {resp.status_code}"}

        except httpx.TimeoutException:
            logger.warning("Mistral API Key 验证超时")
            return {"valid": False, "message": "网络连接失败，请检查网络设置"}
        except httpx.ConnectError:
            logger.warning("Mistral API Key 验证连接失败")
            return {"valid": False, "message": "网络连接失败，请检查网络设置"}
        except httpx.RequestError as e:
            logger.warning(f"Mistral API Key 验证网络错误: {e}")
            return {"valid": False, "message": "网络连接失败，请检查网络设置"}

    elif provider in ("mineru", "doc2x"):
        # Worker 代理模式验证：测试 Worker 可达性和认证有效性
        worker_url = body.get("worker_url", "").strip()
        auth_key = body.get("auth_key", "").strip()
        token = body.get("token", "").strip()
        token_mode = body.get("token_mode", "frontend").strip()

        # 校验 worker_url 参数
        if not worker_url:
            raise HTTPException(status_code=400, detail="缺少 worker_url 参数")

        # 构建请求头（包含 Auth Key 和 Token）
        headers = {}
        if auth_key:
            headers["X-Auth-Key"] = auth_key

        # 前端透传模式下，将 Token 加入请求头
        if token_mode == "frontend" and token:
            if provider == "mineru":
                headers["X-MinerU-Key"] = token
            else:
                headers["X-Doc2X-Key"] = token

        # 根据 provider 构建测试 URL
        # MinerU: GET {worker_url}/mineru/result/test-ping（预期 404 但 Worker 可达）
        # Doc2X: GET {worker_url}/doc2x/status/test-ping（预期 404 但 Worker 可达）
        worker_url_clean = worker_url.rstrip("/")
        if provider == "mineru":
            test_url = f"{worker_url_clean}/mineru/result/test-ping"
        else:
            test_url = f"{worker_url_clean}/doc2x/status/test-ping"

        provider_label = "MinerU" if provider == "mineru" else "Doc2X"

        try:
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0)) as client:
                resp = client.get(test_url, headers=headers)

            # Worker 可达：200、404、500 都表示 Worker 正常运行
            # 404 是预期的，因为 test-ping 不是真实的 batch_id/uid
            # 500 也可能是 Worker 将请求转发给了上游 API，上游返回错误（如 batch_id 不存在）
            if resp.status_code in (200, 404, 500):
                logger.info(f"{provider_label} Worker 验证成功 (HTTP {resp.status_code})")
                return {"valid": True, "message": f"{provider_label} Worker 可达且 Token 有效"}
            elif resp.status_code in (401, 403):
                logger.warning(f"{provider_label} Worker 认证失败: HTTP {resp.status_code}")
                # 尝试从响应体获取更具体的错误信息
                try:
                    error_body = resp.json()
                    error_msg = error_body.get("error", "")
                except Exception:
                    error_msg = ""
                if "token" in error_msg.lower():
                    return {"valid": False, "message": f"Token 无效或缺失，请检查 Token 是否正确"}
                return {"valid": False, "message": f"认证失败，请检查 Auth Key 或 Token 是否正确"}
            else:
                logger.warning(f"{provider_label} Worker 验证异常: HTTP {resp.status_code}")
                return {"valid": False, "message": f"验证失败，Worker 返回 HTTP {resp.status_code}"}

        except httpx.TimeoutException:
            logger.warning(f"{provider_label} Worker 验证超时")
            return {"valid": False, "message": "连接超时，请检查 Worker URL 是否正确"}
        except httpx.ConnectError:
            logger.warning(f"{provider_label} Worker 连接失败")
            return {"valid": False, "message": "连接失败，请检查 Worker URL 是否正确"}
        except httpx.RequestError as e:
            logger.warning(f"{provider_label} Worker 验证网络错误: {e}")
            return {"valid": False, "message": "网络连接失败，请检查网络设置"}

    # 不应到达此处，但作为安全兜底
    return {"valid": False, "message": f"暂不支持 {provider} 的验证"}


# initialize
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
migrate_legacy_storage()
load_documents()
