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
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from services.vector_service import create_index
from services.url_loader_service import fetch_url_content
from services.multi_format_loader import is_supported_format, extract_from_file
from runtime_mode import runtime
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

# 目录策略与 app.py 保持一致：
# - desktop: 使用 runtime.data_dir（由 Electron 传入）
# - server: 使用项目根目录 data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if runtime.is_desktop:
    DATA_DIR = Path(runtime.data_dir)
else:
    DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
UPLOAD_DIR = DATA_DIR / "uploads"

# Legacy paths from the old layout (stored under backend/)
LEGACY_BACKEND_DATA_DIR = BACKEND_ROOT / "data"
LEGACY_BACKEND_DOCS_DIR = LEGACY_BACKEND_DATA_DIR / "docs"
LEGACY_BACKEND_VECTOR_STORE_DIR = LEGACY_BACKEND_DATA_DIR / "vector_stores"
LEGACY_BACKEND_UPLOAD_DIR = BACKEND_ROOT / "uploads"
LEGACY_PROJECT_UPLOAD_DIR = PROJECT_ROOT / "uploads"

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
        (LEGACY_BACKEND_DOCS_DIR, DOCS_DIR, "*.json"),
        (LEGACY_BACKEND_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.index"),
        (LEGACY_BACKEND_VECTOR_STORE_DIR, VECTOR_STORE_DIR, "*.pkl"),
        (LEGACY_BACKEND_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
        (LEGACY_PROJECT_UPLOAD_DIR, UPLOAD_DIR, "*.pdf"),
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

    FIGURE_PATTERNS = [
        r'^图\s*(\d+)([a-zA-Z]?)',
        r'^Figure\s+(\d+)([a-zA-Z]?)',
        r'^Fig\.?\s+(\d+)([a-zA-Z]?)',
    ]

    FIGURE_CAPTION_PATTERNS = [
        r'(图\s*\d+[a-zA-Z]?)',
        r'(Figure\s+\d+[a-zA-Z]?)',
        r'(Fig\.?\s+\d+[a-zA-Z]?)',
    ]

    def _parse_figure_number(figure_num: str) -> tuple:
        """
        解析 figure 编号，返回 (base_number, sub_id)
        例如:
            "1" -> ("1", None)
            "1a" -> ("1", "a")
            "1A" -> ("1", "A")
        """
        if not figure_num:
            return ("", None)

        import re
        # 支持 "1a", "1A", "1.1" 等格式
        match = re.match(r'^(\d+)([a-zA-Z]?)$', figure_num.strip())
        if match:
            base = match.group(1)
            sub = match.group(2) if match.group(2) else None
            if sub:
                sub = sub.lower()
            return (base, sub)

        # 尝试直接解析纯数字
        try:
            return (str(int(figure_num)), None)
        except (ValueError, TypeError):
            return (figure_num, None)

    def _extract_figure_captions_from_dict(
        text_dict: dict,
        page_num: int,
        page_width: float = 0,
        page_height: float = 0,
    ) -> list:
        """从 PyMuPDF 的 dict 格式中检测 figure 标题"""
        if not text_dict or "blocks" not in text_dict:
            return []

        import re
        figures = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_bbox = None

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text:
                        line_text += text
                        if line_bbox is None:
                            line_bbox = span.get("bbox")
                        else:
                            cur_bbox = span.get("bbox", [0, 0, 0, 0])
                            line_bbox = [
                                min(line_bbox[0], cur_bbox[0]),
                                min(line_bbox[1], cur_bbox[1]),
                                max(line_bbox[2], cur_bbox[2]),
                                max(line_bbox[3], cur_bbox[3])
                            ]

                line_text = line_text.strip()
                if not line_text:
                    continue

                for pattern in FIGURE_PATTERNS:
                    match = re.match(pattern, line_text, re.IGNORECASE)
                    if match:
                        # 解析 base_number 和 sub_id
                        raw_num = match.group(1)
                        sub_id_raw = match.group(2) if match.group(2) else ""
                        base_number, sub_id = _parse_figure_number(raw_num + sub_id_raw)

                        # 构建 display_label
                        if sub_id:
                            display_label = f"Figure {base_number}{sub_id}"
                        else:
                            display_label = f"Figure {base_number}"

                        figures.append({
                            "figure_number": base_number,  # 主编号，用于分组
                            "raw_number": raw_num + sub_id_raw,  # 原始编号，如 "1a"
                            "base_number": base_number,  # 主编号 "1"
                            "sub_id": sub_id,  # 子图标识 "a" or None
                            "display_label": display_label,
                            "label": line_text[:50],
                            "caption": line_text[:100],  # 保存完整caption
                            "page": page_num,
                            "bbox": line_bbox or [0, 0, 0, 0],
                            "caption_bbox": line_bbox or [0, 0, 0, 0],
                            "page_width": page_width,
                            "page_height": page_height,
                        })
                        break

        return figures

    def _normalize_bbox(bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _merge_bboxes(bboxes):
        valid = [_normalize_bbox(b) for b in bboxes]
        valid = [b for b in valid if b]
        if not valid:
            return None
        return [
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid),
        ]

    def _expand_bbox(bbox, page_width, page_height, x_ratio=0.04, y_ratio=0.03):
        normalized = _normalize_bbox(bbox)
        if not normalized:
            return None
        x0, y0, x1, y1 = normalized
        x_pad = max(12.0, page_width * x_ratio)
        y_pad = max(10.0, page_height * y_ratio)
        return [
            max(0.0, x0 - x_pad),
            max(0.0, y0 - y_pad),
            min(page_width, x1 + x_pad),
            min(page_height, y1 + y_pad),
        ]

    def _build_band_bbox(page_width, page_height, upper_bound, lower_bound):
        if page_width <= 0 or page_height <= 0:
            return None
        y0 = max(0.0, upper_bound)
        y1 = min(page_height, lower_bound)
        if y1 <= y0:
            return None
        min_height = min(page_height, max(page_height * 0.16, 120.0))
        if y1 - y0 < min_height:
            y0 = max(0.0, y1 - min_height)
        return [
            max(0.0, page_width * 0.04),
            y0,
            min(page_width, page_width * 0.96),
            y1,
        ]

    def _group_figures_by_base_number(figures: list) -> list:
        """
        将 figures 按 (page, base_number) 分组
        支持 synthetic parent：当只有子图(1a,1b)没有主图(1)时，自动生成 group

        返回: list of FigureGroup dicts
        """
        from collections import defaultdict

        # 按 (page, base_number) 分组
        groups = defaultdict(lambda: {
            "sub_figures": [],
            "has_parent": False,
            "parent_figure": None
        })

        for fig in figures:
            base = fig.get("base_number", "")
            page = fig.get("page", 1)
            key = (page, base)

            if fig.get("sub_id"):  # 子图，如 "1a" 中的 "a"
                groups[key]["sub_figures"].append(fig)
            else:  # 主图，如 "Figure 1"
                groups[key]["has_parent"] = True
                groups[key]["parent_figure"] = fig

        # 构建最终 group 列表
        result = []
        for (page, base), data in sorted(groups.items()):
            sub_figures = data["sub_figures"]

            if data["has_parent"]:
                # 有主图
                parent = data["parent_figure"]
                group = {
                    "group_id": f"fig-{base}",
                    "base_number": base,
                    "page": page,
                    "caption": parent.get("caption"),
                    "label": parent.get("display_label", f"Figure {base}"),
                    "sub_figures": sub_figures,
                    "is_synthetic": False,
                }
            else:
                # 无显式主图，synthetic parent
                # 合并所有子图的 caption
                merged_caption = "; ".join(
                    s.get("caption", "") for s in sub_figures if s.get("caption")
                )
                # 构建显示标签
                if sub_figures:
                    labels = [s.get("display_label", "") for s in sub_figures]
                    display_label = ", ".join(filter(None, labels))
                else:
                    display_label = f"Figure {base}"

                group = {
                    "group_id": f"fig-{base}",
                    "base_number": base,
                    "page": page,
                    "caption": merged_caption if merged_caption else None,
                    "label": display_label,
                    "sub_figures": sub_figures,
                    "is_synthetic": True,
                }
            result.append(group)

        return result

    def _match_figures_with_images(figures: list, images: list) -> list:
        """
        将 figure 标题与同页的图片进行空间匹配，返回结构化的 figures 列表。

        改进版：采用 group-first 策略
        1. 先将 figures 按 base_number 分组（支持子图 1a,1b 合并到主图 1）
        2. 按 group 匹配图片
        3. 返回增强的 figure 数据（包含 sub_figures, is_synthetic 等）
        """
        if not figures:
            return []

        # Step 1: 将 figures 按 base_number 分组
        figure_groups = _group_figures_by_base_number(figures)

        result = []
        page_to_images = {}
        for img in images:
            p = img.get("page", 1)
            if p not in page_to_images:
                page_to_images[p] = []
            page_to_images[p].append(img)

        for page_num in page_to_images:
            page_to_images[page_num] = sorted(
                page_to_images[page_num],
                key=lambda img: (
                    (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[1],
                    (_normalize_bbox(img.get("bbox")) or [0, 0, 0, 0])[0],
                    img.get("id", ""),
                )
            )

        # region agent log: debug figure-image matching
        debug_entries = []
        # endregion agent log

        # Step 2: 按 group 处理
        for group in figure_groups:
            page = group["page"]
            base_number = group["base_number"]
            page_images = page_to_images.get(page, [])

            # 用于跟踪本页面已经分配给前面 group 的图片，避免重复分配
            already_matched_image_ids = set()

            # 获取该页所有 figure（包括子图）对应的 caption bboxes
            all_captions = []

            # 主图的 caption
            if group.get("caption"):
                # 从原始 figures 中找对应的 caption bbox
                for fig in figures:
                    if fig.get("page") == page and fig.get("base_number") == base_number and not fig.get("sub_id"):
                        all_captions.append(fig)
                        break

            # 子图的 captions
            for sf in group.get("sub_figures", []):
                if sf.get("page") == page:
                    all_captions.append(sf)

            # 按 caption 位置排序
            all_captions = sorted(
                all_captions,
                key=lambda f: (_normalize_bbox(f.get("caption_bbox") or f.get("bbox")) or [0, 0, 0, 0])[1]
            )

            # 遍历每个 caption，收集匹配的 image_ids
            group_image_ids = []
            group_bboxes = []

            for idx, caption_fig in enumerate(all_captions):
                fig_bbox = _normalize_bbox(caption_fig.get("caption_bbox") or caption_fig.get("bbox")) or [0, 0, 0, 0]
                caption_top = fig_bbox[1]
                caption_bottom = fig_bbox[3]
                page_width = caption_fig.get("page_width", 0) or 612
                page_height = caption_fig.get("page_height", 0) or 792

                # 确定搜索窗口
                prev_bottom = 0.0
                if idx > 0:
                    prev_bbox = _normalize_bbox(all_captions[idx - 1].get("caption_bbox") or all_captions[idx - 1].get("bbox"))
                    if prev_bbox:
                        prev_bottom = prev_bbox[3]

                band_top = max(0.0, prev_bottom + 6.0)
                band_bottom = max(band_top + 1.0, caption_top - 4.0)

                matched_images = []
                matched_bboxes = []

                for img in page_images:
                    img_id = img.get("id")
                    if img_id in already_matched_image_ids:
                        continue

                    img_bbox = _normalize_bbox(img.get("bbox")) or [0, 0, 0, 0]
                    img_y0 = img_bbox[1]
                    img_y1 = img_bbox[3]
                    img_center_y = (img_y0 + img_y1) / 2 if img_y1 > img_y0 else img_y0

                    in_window = (
                        img_center_y >= band_top and
                        img_center_y <= caption_bottom + 8.0 and
                        img_y0 <= caption_bottom + 24.0
                    )
                    if in_window:
                        matched_images.append(img_id)
                        matched_bboxes.append(img_bbox)

                    already_matched_image_ids.update(matched_images)

                group_image_ids.extend(matched_images)
                group_bboxes.extend(matched_bboxes)

            # 去重 image_ids
            unique_image_ids = list(dict.fromkeys(group_image_ids))

            # 合并 group_bboxes 生成 group_bbox
            group_bbox = _merge_bboxes(group_bboxes)
            if group_bbox:
                group_bbox = _expand_bbox(group_bbox, page_width or 612, page_height or 792, x_ratio=0.05, y_ratio=0.04)

            # 获取 caption bbox（使用第一个 caption 的位置）
            primary_caption_bbox = None
            if all_captions:
                primary_caption_bbox = all_captions[0].get("caption_bbox") or all_captions[0].get("bbox")

            # 构建返回结果
            result.append({
                "figure_id": group["group_id"],
                "number": base_number,
                "label": group.get("label", f"Figure {base_number}"),
                "caption": group.get("caption"),
                "page": page,
                "image_ids": unique_image_ids,
                "group_bbox": group_bbox,  # 联合 bbox
                "caption_bbox": primary_caption_bbox,
                "sub_figures": group.get("sub_figures", []),
                "is_synthetic": group.get("is_synthetic", False),
                "page_width": page_width or 612,
                "page_height": page_height or 792,
            })

            # region agent log
            try:
                debug_entries.append({
                    "group_id": group["group_id"],
                    "base_number": base_number,
                    "page": page,
                    "is_synthetic": group.get("is_synthetic", False),
                    "sub_figures_count": len(group.get("sub_figures", [])),
                    "matched_image_ids": unique_image_ids,
                    "group_bbox": group_bbox,
                })
            except Exception:
                pass
            # endregion agent log

        # region agent log: write debug log to NDJSON file
        if debug_entries:
            try:
                import json as _json
                import time as _time

                log_record = {
                    "id": f"log_{int(_time.time() * 1000)}",
                    "timestamp": int(_time.time() * 1000),
                    "location": "routes/document_routes.py:_match_figures_with_images",
                    "message": "figure-group matching debug (enhanced)",
                    "data": {
                        "figures_count": len(figures),
                        "images_count": len(images),
                        "groups_count": len(figure_groups),
                        "entries": debug_entries,
                    },
                    "runId": "initial",
                    "hypothesisId": "H1-H3",
                }

                with open(r"e:\Project\.cursor\debug.log", "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(log_record, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # endregion agent log

        return result

    def extract_with_pymupdf(pdf_bytes: bytes, extract_images: bool = True) -> tuple:
        """
        使用 PyMuPDF 进行字符级文本提取，参考 paper-burner-x 实现
        核心改进：
        1. 使用 get_text("dict") 获取字符级坐标
        2. 按 Y 坐标检测换行，按 X 坐标间距添加空格
        3. 精确控制文本重建，避免空格丢失
        4. 检测 figure 标题（图1 / Figure 1 等）并与图片关联
        返回: (pages, full_text, page_qualities, all_images, figures, error)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, None, None, [], [], "PyMuPDF not installed"
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        full_text_parts = []
        page_qualities = []
        all_images = []  # 存储所有提取的图片
        all_figures = []  # 存储所有检测到的 figure 标题
        
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

                # ==================== Figure 标题检测 ====================
                page_figures = _extract_figure_captions_from_dict(
                    text_dict,
                    page_num + 1,
                    page_width,
                    page_height,
                )
                all_figures.extend(page_figures)

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

                                    img_bbox = None
                                    try:
                                        img_rects = page.get_image_rects(xref)
                                        if img_rects:
                                            rect = img_rects[0]
                                            img_bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                                    except Exception:
                                        pass

                                    page_images.append({
                                        "id": img_id,
                                        "data": f"data:image/{img_ext};base64,{img_base64}",
                                        "width": img_width,
                                        "height": img_height,
                                        "page": page_num + 1,
                                        "bbox": img_bbox
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

        # 基于图片位置匹配 figure 标题，生成 figures 元数据
        figures = _match_figures_with_images(all_figures, all_images)

        return pages, '\n\n'.join(full_text_parts), page_qualities, all_images, figures, None
    
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
    figures = []
    if pdf_bytes:
        pages, full_text, page_qualities, all_images, figures, err = extract_with_pymupdf(pdf_bytes, extract_images)
        if pages is not None:
            extraction_method = "pymupdf"
            print(f"[PDF] Using PyMuPDF extraction, {len(pages)} pages, {len(all_images)} images, {len(figures)} figures")

    # 如果 PyMuPDF 失败，回退到 pdfplumber
    if pages is None:
        print(f"[PDF] PyMuPDF failed ({err}), falling back to pdfplumber")
        pages, full_text, page_qualities, all_images, err = extract_with_pdfplumber(pdf_file)
        extraction_method = "pdfplumber"
        figures = []  # pdfplumber 暂不提取 figures
    
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
        "figures": figures,  # 新增：检测到的 figure 标题列表
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
    embedding_model: str = Form("local-minilm"),
    embedding_api_key: Optional[str] = Form(None),
    embedding_api_host: Optional[str] = Form(None),
    enable_ocr: Optional[str] = Form(None)
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
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_multi_format = is_supported_format(file.filename)

    if not is_pdf and not is_multi_format:
        supported = "PDF, DOCX, XLSX, TXT, MD, CSV"
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持: {supported}")

    try:
        content = await file.read()

        # 多格式文档处理（非 PDF）
        if is_multi_format and not is_pdf:
            import tempfile
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                normalized_model = normalize_embedding_model_id(embedding_model)
                if not normalized_model:
                    raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置")
                embedding_model = normalized_model

                extracted_data = extract_from_file(tmp_path, file.filename)
                doc_id = generate_doc_id(extracted_data["full_text"])

                documents_store[doc_id] = {
                    "filename": file.filename,
                    "upload_time": datetime.now().isoformat(),
                    "data": extracted_data,
                    "pdf_url": None,
                }
                save_document(doc_id, documents_store[doc_id])
                create_index(
                    doc_id, extracted_data["full_text"], str(VECTOR_STORE_DIR),
                    embedding_model, embedding_api_key, embedding_api_host,
                    pages=extracted_data.get("pages"),
                )
                return {
                    "message": "文档上传成功",
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "total_pages": extracted_data["total_pages"],
                    "total_chars": len(extracted_data["full_text"]),
                    "source_type": extracted_data.get("source_type", "unknown"),
                }
            finally:
                os.unlink(tmp_path)

        pdf_file = io.BytesIO(content)

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置或格式不正确（建议使用 provider:model 格式）")
        embedding_model = normalized_model

        # 桌面模式下本地模型不可用，提前拦截
        if runtime.is_desktop and ('local' in embedding_model.lower().split(':')[0] or embedding_model in ('local-minilm',)):
            raise HTTPException(
                status_code=400,
                detail="桌面版不支持本地 Embedding 模型，请在设置中选择远程 Embedding 服务（如 OpenAI、硅基流动等）并配置 API Key"
            )

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

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"PDF处理失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")


@router.post("/documents/url")
async def import_url(
    request: Request,
):
    """将网页 URL 转为文档并索引到向量库

    请求体 JSON:
        url: 目标网页 URL
        embedding_model: 文本嵌入模型
        embedding_api_key: 云端嵌入模型的 API 密钥（可选）
        embedding_api_host: 自定义 API 地址（可选）
    """
    try:
        body = await request.json()
        url = body.get("url", "").strip()
        embedding_model = body.get("embedding_model", "local-minilm")
        embedding_api_key = body.get("embedding_api_key")
        embedding_api_host = body.get("embedding_api_host")

        if not url:
            raise HTTPException(status_code=400, detail="URL 不能为空")

        normalized_model = normalize_embedding_model_id(embedding_model)
        if not normalized_model:
            raise HTTPException(status_code=400, detail=f"Embedding模型 '{embedding_model}' 未配置")
        embedding_model = normalized_model

        # 抓取网页内容
        result = await fetch_url_content(url)
        title = result["title"]
        content = result["content"]

        if not content or len(content) < 10:
            raise HTTPException(status_code=400, detail="网页内容为空或过短")

        doc_id = generate_doc_id(content)

        # 构建与 PDF 文档兼容的数据结构
        extracted_data = {
            "full_text": content,
            "total_pages": 1,
            "pages": [{"page": 1, "text": content}],
            "source_type": "url",
            "source_url": url,
        }

        documents_store[doc_id] = {
            "filename": f"🌐 {title[:60]}",
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": None,
        }

        save_document(doc_id, documents_store[doc_id])

        create_index(
            doc_id, content, str(VECTOR_STORE_DIR),
            embedding_model, embedding_api_key, embedding_api_host,
            pages=extracted_data["pages"],
        )

        return {
            "message": "URL 导入成功",
            "doc_id": doc_id,
            "filename": f"🌐 {title[:60]}",
            "title": title,
            "url": url,
            "total_chars": len(content),
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 导入失败: {str(e)}")


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


@router.get("/document/{doc_id}/thumbnail/{page}")
async def get_page_thumbnail(doc_id: str, page: int):
    """按需生成 PDF 页面缩略图

    使用 pymupdf 渲染指定页面为 40dpi 缩略图，返回 base64 编码的 JPEG。
    单页缩略图约 5-15KB。

    Args:
        doc_id: 文档 ID
        page: 页码（1-indexed）
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    pdf_url = doc.get("pdf_url")
    if not pdf_url:
        raise HTTPException(status_code=400, detail="该文档无 PDF 文件（可能是 URL 导入的文档）")

    pdf_path = UPLOAD_DIR / pdf_url.split("/")[-1]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 文件不存在")

    try:
        import fitz
        import base64

        pdf_doc = fitz.open(str(pdf_path))
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            raise HTTPException(status_code=400, detail=f"页码超出范围 (1-{len(pdf_doc)})")

        pdf_page = pdf_doc[page - 1]
        # 40dpi 缩略图，体积小且足够预览
        pix = pdf_page.get_pixmap(dpi=40)
        img_bytes = pix.tobytes("jpeg")
        pdf_doc.close()

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {
            "doc_id": doc_id,
            "page": page,
            "thumbnail": f"data:image/jpeg;base64,{b64}",
            "width": pix.width,
            "height": pix.height,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缩略图生成失败: {str(e)}")


@router.post("/document/{doc_id}/graphrag/build")
async def build_graphrag_index(doc_id: str, request: Request):
    """为文档构建 GraphRAG 知识图谱索引

    请求体 JSON:
        api_key: LLM API 密钥
        model: LLM 模型名
        api_provider: LLM 提供商
        api_host: LLM API 地址（可选）
        embedding_model: Embedding 模型名（可选）
        embedding_api_key: Embedding API 密钥（可选）
        embedding_api_host: Embedding API 地址（可选）
    """
    if not settings.enable_graphrag:
        raise HTTPException(status_code=400, detail="GraphRAG 未启用，请在配置中设置 enable_graphrag=true")

    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    full_text = doc.get("data", {}).get("full_text", "")
    if not full_text or len(full_text) < 50:
        raise HTTPException(status_code=400, detail="文档内容过短，无法构建知识图谱")

    try:
        body = await request.json()
        api_key = body.get("api_key", "")
        model = body.get("model", "")
        provider = body.get("api_provider", "")
        api_host = body.get("api_host", "")
        embedding_model = body.get("embedding_model", "")
        embedding_api_key = body.get("embedding_api_key", "")
        embedding_api_host = body.get("embedding_api_host", "")

        if not api_key or not model:
            raise HTTPException(status_code=400, detail="GraphRAG 构建需要 api_key 和 model")

        from services.graphrag import GraphRAG, GraphRAGConfig

        # 解析 endpoint
        endpoint = ""
        if api_host:
            host = api_host.strip().rstrip('/')
            endpoint = f"{host}/chat/completions" if not host.endswith('/chat/completions') else host

        # 解析 embedding endpoint
        embed_endpoint = ""
        if embedding_api_host:
            host = embedding_api_host.strip().rstrip('/')
            embed_endpoint = f"{host}/v1" if not host.endswith('/v1') else host

        working_dir = os.path.join(settings.graphrag_working_dir, doc_id)

        config = GraphRAGConfig(
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            embedding_api_key=embedding_api_key or api_key,
            embedding_model=embedding_model or model,
            embedding_provider=provider,
            embedding_endpoint=embed_endpoint or endpoint.replace("/chat/completions", ""),
        )

        rag = GraphRAG(
            working_dir=working_dir,
            config=config,
            chunk_token_size=settings.graphrag_chunk_token_size,
            entity_extract_max_gleaning=settings.graphrag_max_gleaning,
            best_model_max_async=settings.graphrag_max_async,
            cheap_model_max_async=settings.graphrag_max_async,
        )

        await rag.ainsert(full_text)

        stats = rag.stats()
        # 缓存实例以便查询时复用
        if not hasattr(router, "_graphrag_instances"):
            router._graphrag_instances = {}
        router._graphrag_instances[doc_id] = rag

        return {
            "message": "GraphRAG 索引构建完成",
            "doc_id": doc_id,
            "stats": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GraphRAG] 构建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GraphRAG 构建失败: {str(e)}")


@router.get("/document/{doc_id}/graphrag/stats")
async def get_graphrag_stats(doc_id: str):
    """获取文档的 GraphRAG 索引统计信息"""
    if not hasattr(router, "_graphrag_instances") or doc_id not in router._graphrag_instances:
        raise HTTPException(status_code=404, detail="该文档未构建 GraphRAG 索引")
    rag = router._graphrag_instances[doc_id]
    return {"doc_id": doc_id, "stats": rag.stats()}


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


# ============ 速览（Overview）API ============

from services.overview_service import (
    get_or_create_overview,
    create_overview_task,
    get_task_status,
    OverviewDepth,
)
from models.dynamic_store import load_dynamic_providers
from models.provider_registry import PROVIDER_CONFIG


def _get_overview_provider_endpoint(provider_id: str, api_host: str = "") -> str:
    """按优先级解析速览使用的聊天端点。"""
    if api_host and api_host.strip():
        host = api_host.strip().rstrip("/")
        if host.endswith("/chat/completions"):
            return host
        return f"{host}/chat/completions"

    dynamic = load_dynamic_providers()
    if provider_id in dynamic:
        return dynamic[provider_id].get("endpoint", "")

    return PROVIDER_CONFIG.get(provider_id, {}).get("endpoint", "")


def _resolve_overview_runtime_params(
    request: Request,
    api_key: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    api_host: Optional[str],
):
    resolved_provider = (provider or request.headers.get("X-ChatPDF-Provider") or "openai").strip() or "openai"
    resolved_model = (model or request.headers.get("X-ChatPDF-Model") or "gpt-4o").strip() or "gpt-4o"
    resolved_api_key = (api_key or request.headers.get("X-ChatPDF-Api-Key") or "").strip()
    resolved_api_host = (api_host or request.headers.get("X-ChatPDF-Api-Host") or "").strip()
    return resolved_api_key, resolved_model, resolved_provider, resolved_api_host


@router.post("/documents/{doc_id}/overview")
async def create_overview(
    request: Request,
    doc_id: str,
    depth: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
):
    """
    触发速览生成

    Args:
        doc_id: 文档 ID
        depth: 速览深度 brief(简介) / standard(标准) / detailed(详细)
        api_key: API Key（可选，默认使用配置）
        model: 模型名称（可选，默认 gpt-4o）
        provider: 模型提供商（可选，默认 openai）

    Returns:
        task_id: 任务 ID，用于轮询状态
        status: 任务状态
    """
    # 验证文档存在
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    # 验证深度参数
    valid_depths = [OverviewDepth.BRIEF, OverviewDepth.STANDARD, OverviewDepth.DETAILED]
    if depth not in valid_depths:
        depth = OverviewDepth.STANDARD

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )

    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    task = await create_overview_task(
        doc_id,
        depth,
        api_key,
        model,
        provider,
        _get_overview_provider_endpoint(provider, api_host),
    )

    return {
        "task_id": task.task_id,
        "doc_id": doc_id,
        "depth": depth,
        "status": task.status
    }


@router.get("/documents/{doc_id}/overview/tasks/{task_id}")
async def get_overview_task_status(doc_id: str, task_id: str):
    """
    获取速览生成任务状态
    
    Args:
        doc_id: 文档 ID
        task_id: 任务 ID
    
    Returns:
        status: 任务状态 (pending/processing/completed/failed)
        result: 完成后返回速览数据
        error: 失败时返回错误信息
    """
    task = await get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    if task.doc_id != doc_id:
        raise HTTPException(status_code=400, detail="任务与文档不匹配")
    
    response = {
        "task_id": task.task_id,
        "doc_id": task.doc_id,
        "depth": task.depth,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }
    
    if task.status == "completed" and task.result:
        response["result"] = task.result.model_dump()
    elif task.status == "failed":
        response["error"] = task.error
    
    return response


@router.get("/documents/{doc_id}/overview")
async def get_overview(
    request: Request,
    doc_id: str,
    depth: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_host: Optional[str] = None,
):
    """
    获取速览（同步接口）

    如果速览未生成，会自动创建任务并等待完成。

    Args:
        doc_id: 文档 ID
        depth: 速览深度 brief(简介) / standard(标准) / detailed(详细)
        api_key: API Key（可选，默认使用配置）
        model: 模型名称（可选，默认 gpt-4o）
        provider: 模型提供商（可选，默认 openai）

    Returns:
        速览数据
    """
    # 验证文档存在
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    # 验证深度参数
    valid_depths = [OverviewDepth.BRIEF, OverviewDepth.STANDARD, OverviewDepth.DETAILED]
    if depth not in valid_depths:
        depth = OverviewDepth.STANDARD

    api_key, model, provider, api_host = _resolve_overview_runtime_params(
        request,
        api_key,
        model,
        provider,
        api_host,
    )

    # 如果没传 api_key，从当前模型配置中获取（和对话逻辑一致）
    if not api_key:
        merged = {**PROVIDER_CONFIG, **load_dynamic_providers()}
        prov = merged.get(provider, {})
        api_key = (prov.get("api_key") or "").strip()

    try:
        overview = await get_or_create_overview(
            doc_id,
            depth,
            api_key,
            model,
            provider,
            _get_overview_provider_endpoint(provider, api_host),
        )
        return overview.model_dump()
    except TimeoutError:
        raise HTTPException(status_code=408, detail="速览生成超时，请稍后重试")
    except Exception as e:
        logger.error(f"获取速览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取速览失败: {str(e)}")
