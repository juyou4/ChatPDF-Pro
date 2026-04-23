"""
表格感知服务 — PDF 表格检测与 Markdown 转换

参考 ragflow rag/nlp/__init__.py 的 tokenize_table 和 attach_media_context 策略：
- 使用 PyMuPDF find_tables() 检测页面中的表格区域
- 将表格转换为结构化 Markdown 格式
- 替换原始页面文本中的表格区域，保留上下文
- 自动检测表格标题（Table X / 表 X）并作为前缀

设计：
- 表格转 Markdown 后，structure_aware_split 的 _find_protected_regions 会自动识别
  并将其作为受保护区域，不会被分块切割
- 表格 Markdown 前缀包含 [TABLE] 标记，便于检索时识别
"""
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _table_to_markdown(table_data: list, caption: str = "") -> str:
    """将表格数据转换为 Markdown 格式

    Args:
        table_data: PyMuPDF find_tables() 返回的二维列表
                    每个元素为一行，每行为一个单元格列表
        caption: 可选的表格标题

    Returns:
        Markdown 格式的表格字符串
    """
    if not table_data or not table_data[0]:
        return ""

    rows = []
    for row in table_data:
        # 清理单元格内容：去除换行、多余空白
        cells = []
        for cell in row:
            cell_text = str(cell) if cell is not None else ""
            cell_text = cell_text.replace("\n", " ").replace("|", "\\|").strip()
            cells.append(cell_text)
        rows.append(cells)

    if not rows:
        return ""

    # 统一列数（取最大列数）
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # 构建 Markdown 表格
    lines = []

    if caption:
        lines.append(f"[TABLE] {caption}")
        lines.append("")

    # 表头
    lines.append("| " + " | ".join(rows[0]) + " |")
    # 分隔行
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # 数据行
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _find_table_caption(page_text: str, table_bbox: tuple, page_height: float) -> str:
    """检测表格附近的标题文本（Table X / 表 X）

    搜索策略：在表格上方 3 行内查找 Table/表 编号模式

    Args:
        page_text: 页面完整文本
        table_bbox: 表格的 (x0, y0, x1, y1) 坐标
        page_height: 页面高度

    Returns:
        检测到的标题字符串，未找到返回空字符串
    """
    # 从页面文本中查找所有 Table/表 标题
    caption_pattern = re.compile(
        r'(?:Table|TABLE|表)\s*\.?\s*(\d+(?:\.\d+)?)\s*[.:：]?\s*(.*?)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(caption_pattern.finditer(page_text))
    if not matches:
        return ""

    # 取最接近表格 bbox 上方的标题
    # 简单策略：返回文本中最后一个在表格区域之前的 Table 标题
    table_y0 = table_bbox[1] if table_bbox else 0
    best_match = None

    for m in matches:
        # 粗略估算：匹配位置在文本中的比例 ≈ 在页面中的 Y 位置比例
        text_ratio = m.start() / max(len(page_text), 1)
        est_y = text_ratio * page_height

        if est_y <= table_y0 + 20:  # 允许小偏差
            best_match = m

    if best_match:
        num = best_match.group(1)
        desc = best_match.group(2).strip()
        if desc:
            return f"Table {num}: {desc}"
        return f"Table {num}"

    return ""


def extract_tables_from_page(page, page_text: str, page_num: int) -> List[dict]:
    """从 PyMuPDF 页面对象中提取表格并转换为 Markdown

    Args:
        page: PyMuPDF 页面对象 (fitz.Page)
        page_text: 该页的原始文本
        page_num: 页码（1-indexed）

    Returns:
        表格信息列表，每项包含:
        - markdown: Markdown 格式的表格
        - bbox: 表格坐标 (x0, y0, x1, y1)
        - caption: 表格标题
        - page: 页码
    """
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.debug(f"[Table] 页面 {page_num} 表格检测失败: {e}")
        return []

    if not tables or not tables.tables:
        return []

    results = []
    page_height = page.rect.height

    for i, table in enumerate(tables.tables):
        try:
            # 提取表格数据
            data = table.extract()
            if not data or len(data) < 2:
                # 少于 2 行的不算有效表格
                continue

            # 检查表格是否有足够内容（过滤空表格）
            non_empty_cells = sum(
                1 for row in data for cell in row
                if cell is not None and str(cell).strip()
            )
            total_cells = sum(len(row) for row in data)
            if total_cells == 0 or non_empty_cells / total_cells < 0.3:
                continue

            bbox = table.bbox  # (x0, y0, x1, y1)

            # 检测表格标题
            caption = _find_table_caption(page_text, bbox, page_height)

            # 转换为 Markdown
            md = _table_to_markdown(data, caption)
            if md:
                results.append({
                    "markdown": md,
                    "bbox": bbox,
                    "caption": caption,
                    "page": page_num,
                    "rows": len(data),
                    "cols": len(data[0]) if data else 0,
                })

        except Exception as e:
            logger.debug(f"[Table] 页面 {page_num} 表格 {i} 提取失败: {e}")
            continue

    if results:
        logger.info(f"[Table] 页面 {page_num} 检测到 {len(results)} 个表格")

    return results


def inject_tables_into_text(page_text: str, tables: List[dict]) -> str:
    """将检测到的 Markdown 表格注入页面文本末尾

    策略：在页面文本末尾追加表格的 Markdown 格式。
    不尝试替换原文中的表格区域（因为坐标→文本位置映射不精确），
    而是追加到末尾，让 structure_aware_split 的表格保护机制处理。

    Args:
        page_text: 原始页面文本
        tables: extract_tables_from_page 返回的表格列表

    Returns:
        注入表格后的页面文本
    """
    if not tables:
        return page_text

    parts = [page_text.rstrip()]

    for t in tables:
        md = t["markdown"]
        # 避免重复：如果原文已包含 Markdown 表格（| 分隔），跳过
        first_data_line = ""
        for line in md.split("\n"):
            if line.startswith("|") and "---" not in line:
                first_data_line = line
                break
        if first_data_line and first_data_line in page_text:
            continue

        parts.append("")
        parts.append(md)

    return "\n".join(parts)
