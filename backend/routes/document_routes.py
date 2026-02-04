import io
import os
import glob
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import PyPDF2
import pdfplumber
from fastapi import APIRouter, UploadFile, File, HTTPException

from services.vector_service import create_index
from services.ocr_service import (
    is_ocr_available,
    detect_pdf_quality,
    ocr_pdf,
    get_ocr_service
)
from models.model_detector import normalize_embedding_model_id

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


def extract_text_from_pdf(pdf_file, pdf_bytes: Optional[bytes] = None, enable_ocr: str = "auto", extract_images: bool = True):
    """
    Extract text and images from PDF with optional OCR fallback
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
        pdf_file: File-like object for pdfplumber
        pdf_bytes: Raw PDF bytes (needed for OCR)
        enable_ocr: "auto" (detect and use if needed), "always", or "never"
        extract_images: Whether to extract images from PDF
    
    Returns:
        dict with full_text, pages, total_pages, images, and ocr metadata
    """
    import re
    import base64
    import time
    from statistics import median
    
    # ==================== 配置常量 ====================
    BATCH_SIZE = 50  # 每批处理页数
    BATCH_SLEEP = 0.3  # 批间休息时间(秒)
    
    # 图片过滤配置
    MIN_IMAGE_SIZE = 30  # 最小图片尺寸(px)，小于此值视为装饰图标
    MAX_ASPECT_RATIO = 20  # 最大宽高比，超过视为线条/分隔符
    MIN_ASPECT_RATIO = 0.05  # 最小宽高比
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
    
    def assess_page_quality(page_text: str, block_count: int) -> dict:
        """评估单页提取质量"""
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
        
        needs_ocr = score < 60
        reason = "good" if score >= 80 else ("acceptable" if score >= 60 else "poor_quality")
        
        return {
            "score": round(score, 1),
            "needs_ocr": needs_ocr,
            "reason": reason,
            "null_ratio": round(null_ratio, 3),
            "valid_ratio": round(valid_ratio, 3)
        }
    
    def extract_with_pymupdf(pdf_bytes: bytes, extract_images: bool = True) -> tuple:
        """使用 PyMuPDF 进行坐标级文本提取，支持多栏检测和图片提取"""
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
                
                # 获取文本块，包含坐标信息
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]
                
                # 计算自适应阈值
                thresholds = get_adaptive_thresholds(text_blocks)
                
                # 检测多栏布局
                columns = detect_columns(text_blocks, page_width)
                is_multi_column = len(columns) > 1
                
                # 按栏排序
                sorted_blocks = sort_blocks_by_columns(text_blocks, columns, thresholds)
                
                # 提取文本
                page_lines = []
                last_y = None
                current_line = []
                line_tol = thresholds.get("line_tolerance", 6)
                
                for block in sorted_blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                    
                    text = clean_text(text)
                    if not text or is_garbage_line(text):
                        continue
                    
                    y = block[1]
                    
                    # 检测是否换行
                    if last_y is not None and abs(y - last_y) > line_tol:
                        if current_line:
                            page_lines.append(' '.join(current_line))
                            current_line = []
                    
                    current_line.append(text)
                    last_y = y
                
                # 添加最后一行
                if current_line:
                    page_lines.append(' '.join(current_line))
                
                page_text = '\n'.join(page_lines)
                
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
                                    
                                    # 在文本中添加图片引用
                                    page_text += f"\n\n![图片{len(all_images) + len(page_images)}](images/{img_id}.{img_ext})\n"
                                    
                            except Exception as img_err:
                                # 单个图片提取失败不影响整体
                                pass
                        
                        all_images.extend(page_images)
                        
                    except Exception as img_extract_err:
                        print(f"[PDF] Page {page_num + 1} image extraction failed: {img_extract_err}")
                
                # 评估页面质量
                quality = assess_page_quality(page_text, len(text_blocks))
                page_qualities.append(quality)
                
                pages.append({
                    "page": page_num + 1,
                    "content": page_text,
                    "quality_score": quality["score"],
                    "is_multi_column": is_multi_column,
                    "block_count": len(text_blocks),
                    "image_count": len(page_images),
                    "source": "pymupdf"
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
                    
                    # 评估质量
                    quality = assess_page_quality(page_text, len(set(c.get('block', 0) for c in chars)))
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
        参考 paper-burner-x 的 _heuristicRebuild 实现
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
        rebuilt = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', rebuilt)
        
        # 2. 合并被打断的句子（区分中英文）
        if is_cjk:
            # 中文：直接连接，不加空格
            rebuilt = re.sub(r'([^\n.!?。！？])\n([\u4e00-\u9fff])', r'\1\2', rebuilt)
        else:
            # 英文：加空格连接
            rebuilt = re.sub(r'([^\n.!?。！？])\n([a-z])', r'\1 \2', rebuilt)
        
        # 3. 修复中文标点符号周围的空格
        rebuilt = re.sub(r'\s+([，。！？；：、）】」』])', r'\1', rebuilt)
        rebuilt = re.sub(r'([（【「『])\s+', r'\1', rebuilt)
        
        # 4. 修复英文标点符号
        rebuilt = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', rebuilt)
        rebuilt = re.sub(r'\s+([,.!?;:])', r'\1', rebuilt)
        
        # 5. 规范化空白字符
        rebuilt = re.sub(r' {2,}', ' ', rebuilt)
        rebuilt = re.sub(r'\n{3,}', '\n\n', rebuilt)
        
        # 6. 保护列表格式
        rebuilt = re.sub(r'\n(\d+)\.\s*\n', r'\n\1. ', rebuilt)
        rebuilt = re.sub(r'\n([a-z])\)\s*\n', r'\n\1) ', rebuilt)
        
        # 7. 修复括号
        rebuilt = re.sub(r'\(\s+', '(', rebuilt)
        rebuilt = re.sub(r'\s+\)', ')', rebuilt)
        
        # ==================== 智能段落合并 ====================
        # 参考 paper-burner-x 的段落识别逻辑
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
                (re.search(r'[.!?。！？]\s*$', current_para) and  # 上一段以句号结束
                 re.match(r'^[A-Z\u4e00-\u9fff]', line))  # 本行首字母大写或中文
            )
            
            if should_break:
                if current_para:
                    paragraphs.append(current_para.strip())
                current_para = line
            else:
                # 合并到当前段落
                if is_cjk:
                    current_para += line  # 中文不加空格
                else:
                    current_para += ' ' + line  # 英文加空格
        
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
    
    # Check if OCR is needed
    if enable_ocr == "never":
        return result
    
    # 逐页OCR决策
    if not pages_needing_ocr and enable_ocr != "always":
        print(f"[PDF] All pages quality OK (avg: {avg_quality:.1f})")
        return result
    
    # Check OCR availability
    ocr_status = is_ocr_available()
    if not ocr_status["any"]:
        print(f"[PDF] OCR needed for {len(pages_needing_ocr)} pages but no OCR backend available")
        result["ocr_error"] = "OCR未安装，请安装 pytesseract 或 paddleocr"
        return result
    
    if pdf_bytes is None:
        print("[PDF] OCR needed but pdf_bytes not provided")
        result["ocr_error"] = "无法执行OCR：缺少PDF原始数据"
        return result
    
    # 执行逐页OCR
    print(f"[PDF] Starting per-page OCR for {len(pages_needing_ocr)} pages")
    try:
        ocr_result = ocr_pdf(pdf_bytes, backend="auto", dpi=200)
        ocr_pages = ocr_result.get("pages", [])
        
        # 只替换质量差的页面
        merged_text_parts = []
        for i, page in enumerate(pages):
            if i in pages_needing_ocr and i < len(ocr_pages):
                ocr_content = ocr_pages[i].get("content", "")
                orig_content = page.get("content", "")
                
                # 只有OCR结果更好时才替换
                if len(ocr_content) > len(orig_content) * 0.8:
                    page["content"] = heuristic_rebuild(ocr_content, is_cjk)
                    page["source"] = "ocr"
                    page["ocr_backend"] = ocr_result.get("ocr_backend")
                    result["ocr_used"] = True
            
            merged_text_parts.append(page["content"])
        
        if result["ocr_used"]:
            result["full_text"] = "\n\n".join(merged_text_parts)
            result["ocr_backend"] = ocr_result.get("ocr_backend")
            result["ocr_pages"] = pages_needing_ocr
        
        print(f"[PDF] OCR complete. Used: {result['ocr_used']}, Pages: {pages_needing_ocr}")
        
    except Exception as e:
        print(f"[PDF] OCR failed: {e}")
        result["ocr_error"] = str(e)
    
    return result


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = "local-minilm",
    embedding_api_key: Optional[str] = None,
    embedding_api_host: Optional[str] = None,
    enable_ocr: str = "auto"
):
    """
    Upload and process a PDF file
    
    Args:
        file: PDF file to upload
        embedding_model: Model for text embedding
        embedding_api_key: API key for cloud embedding models
        embedding_api_host: Custom API host
        enable_ocr: OCR mode - "auto" (detect), "always", or "never"
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

        # Extract text with OCR support
        extracted_data = extract_text_from_pdf(pdf_file, pdf_bytes=content, enable_ocr=enable_ocr)

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

        create_index(doc_id, extracted_data["full_text"], str(VECTOR_STORE_DIR), embedding_model, embedding_api_key, embedding_api_host)

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


@router.get("/ocr/status")
async def get_ocr_status():
    """Check OCR availability and supported backends"""
    status = is_ocr_available()
    return {
        "available": status["any"],
        "backends": {
            "tesseract": status["tesseract"],
            "paddleocr": status["paddleocr"]
        },
        "recommended": "paddleocr" if status["paddleocr"] else ("tesseract" if status["tesseract"] else None),
        "install_instructions": {
            "tesseract": "pip install pytesseract pdf2image && 安装 Tesseract-OCR",
            "paddleocr": "pip install paddleocr pdf2image"
        }
    }


# initialize
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
migrate_legacy_storage()
load_documents()
