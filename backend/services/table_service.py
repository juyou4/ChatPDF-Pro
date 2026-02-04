"""
表格保护服务 - 参考 paper-burner-x 实现
在文本处理/翻译前保护 Markdown 表格结构，处理后恢复
"""
import re
from typing import Dict, Tuple, List


class TableProtector:
    """Markdown 表格保护器"""
    
    def __init__(self):
        self.placeholder_prefix = "__TABLE_PLACEHOLDER_"
        self.placeholder_suffix = "__"
        
    def protect_tables(self, markdown: str) -> Tuple[str, Dict[str, str]]:
        """
        将 Markdown 表格替换为占位符
        
        Args:
            markdown: 原始 Markdown 文本
            
        Returns:
            (处理后的文本, 占位符到表格内容的映射)
        """
        if not markdown:
            return "", {}
        
        # 标准化换行符
        text = markdown.replace('\r\n', '\n')
        
        # 表格正则
        table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
        table_sep_pattern = re.compile(r'^\s*\|[\s\-:]+\|\s*$')
        
        lines = text.split('\n')
        table_ranges = []  # [(start, end), ...]
        
        # 查找表格标题行
        table_titles = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith(('TABLE', 'Table', '表')) and 
                i < len(lines) - 1 and 
                table_row_pattern.match(lines[i + 1])):
                table_titles[i] = line
        
        # 扫描表格范围
        in_table = False
        table_start = -1
        min_table_rows = 3
        
        for i, line in enumerate(lines):
            is_table_row = table_row_pattern.match(line) is not None
            
            if not in_table and is_table_row:
                table_start = i
                in_table = True
            elif in_table and not is_table_row:
                if i - table_start >= min_table_rows:
                    # 检查是否有表格标题
                    title_idx = table_start - 1
                    if title_idx in table_titles:
                        table_ranges.append((title_idx, i - 1))
                        del table_titles[title_idx]
                    else:
                        table_ranges.append((table_start, i - 1))
                in_table = False
                table_start = -1
        
        # 处理文档末尾的表格
        if in_table and len(lines) - table_start >= min_table_rows:
            title_idx = table_start - 1
            if title_idx in table_titles:
                table_ranges.append((title_idx, len(lines) - 1))
            else:
                table_ranges.append((table_start, len(lines) - 1))
        
        # 合并相邻表格
        if len(table_ranges) > 1:
            merged = [table_ranges[0]]
            for current in table_ranges[1:]:
                last = merged[-1]
                # 如果两个表格间隔小于等于2行，考虑合并
                if current[0] - last[1] <= 2:
                    middle_lines = lines[last[1] + 1:current[0]]
                    # 检查中间是否都是空行或注释
                    if all(not l.strip() or l.strip().startswith(('^', '*', '[^')) 
                           for l in middle_lines):
                        merged[-1] = (last[0], current[1])
                    else:
                        merged.append(current)
                else:
                    merged.append(current)
            table_ranges = merged
        
        # 从后向前替换，避免索引偏移
        placeholders = {}
        for idx, (start, end) in enumerate(reversed(table_ranges)):
            table_content = '\n'.join(lines[start:end + 1])
            placeholder = f"{self.placeholder_prefix}{idx}{self.placeholder_suffix}"
            placeholders[placeholder] = table_content
            
            # 替换
            lines = lines[:start] + [placeholder] + lines[end + 1:]
        
        return '\n'.join(lines), placeholders
    
    def restore_tables(self, text: str, placeholders: Dict[str, str]) -> str:
        """
        恢复表格占位符为原始内容
        
        Args:
            text: 包含占位符的文本
            placeholders: 占位符到表格内容的映射
            
        Returns:
            恢复后的文本
        """
        result = text
        for placeholder, content in placeholders.items():
            result = result.replace(placeholder, content)
        return result
    
    def fix_table_format(self, table_content: str) -> str:
        """
        修复表格格式问题
        
        Args:
            table_content: 表格 Markdown 内容
            
        Returns:
            修复后的表格内容
        """
        lines = [l.strip() for l in table_content.split('\n') if l.strip()]
        
        if len(lines) < 3:
            return table_content
        
        # 分析表头
        header = lines[0]
        header_cells = [c.strip() for c in header.split('|') if c.strip()]
        col_count = len(header_cells)
        
        # 检查分隔行
        separator = lines[1]
        if '-' not in separator or '|' not in separator:
            # 创建正确的分隔行
            new_sep = '|' + ' --- |' * col_count
            lines[1] = new_sep
        else:
            sep_cells = [c.strip() for c in separator.split('|') if c.strip()]
            if len(sep_cells) != col_count:
                new_sep = '|' + ' --- |' * col_count
                lines[1] = new_sep
        
        # 修复数据行
        for i in range(2, len(lines)):
            data_cells = [c.strip() for c in lines[i].split('|') if c.strip()]
            if len(data_cells) != col_count:
                # 补齐或截断
                while len(data_cells) < col_count:
                    data_cells.append('')
                data_cells = data_cells[:col_count]
                lines[i] = '| ' + ' | '.join(data_cells) + ' |'
        
        return '\n'.join(lines)


# 全局实例
table_protector = TableProtector()


def protect_markdown_tables(text: str) -> Tuple[str, Dict[str, str]]:
    """保护 Markdown 表格"""
    return table_protector.protect_tables(text)


def restore_markdown_tables(text: str, placeholders: Dict[str, str]) -> str:
    """恢复 Markdown 表格"""
    return table_protector.restore_tables(text, placeholders)
