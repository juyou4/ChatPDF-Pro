# PDF文本提取问题分析报告

## 问题现象
用户反馈：AI回答"无法确定实验的管线是什么"，说明提取的PDF内容不完整或质量差。

## 根本原因

### 1. 空格丢失问题 ❌ **严重**
**现象**：
- 原始PDF: `"Modern autonomous driving (AD) systems"`
- 提取结果: `"Modernautonomousdriving(AD)systems"`

**根因**：
PyMuPDF的`get_text("blocks")`返回的每个block是一个**完整的文本段落**（可能包含多行），但我们的代码错误地将其当作"单行文本块"处理：

```python
# 错误的处理方式
for block in sorted_blocks:
    text = block[4].strip()  # block[4]已经是完整段落，包含换行符
    current_line.append(text)  # 把段落当作"行内块"添加
    
page_lines.append(' '.join(current_line))  # 用空格连接"块"
```

**问题**：
- Block内部的换行符`\n`被保留
- 但是当我们用`' '.join(current_line)`连接多个blocks时，会在blocks之间添加空格
- 然后`heuristic_rebuild`函数会移除block内部的换行符，导致单词粘连

### 2. 文本块排序逻辑问题 ⚠️ **中等**
当前的排序逻辑：
```python
sorted_blocks = sorted(
    blocks_with_col,
    key=lambda x: (x[1], round(x[0][1] / line_tol) * line_tol, x[0][0])
)
```

这个逻辑假设：
1. 先按栏索引排序
2. 栏内按Y坐标排序（同一行的blocks会被分组）
3. 同行内按X坐标排序

**问题**：PyMuPDF的blocks已经是段落级别的，不是字符级别的。我们不应该按"行"来排序和合并。

### 3. 段落合并逻辑过于激进 ⚠️ **中等**
`heuristic_rebuild`函数会合并不应该换段的行：
```python
if should_break:
    # 换段
else:
    if is_cjk:
        current_para += line  # 中文不加空格
    else:
        current_para += ' ' + line  # 英文加空格
```

但是这个逻辑没有考虑到：
- Block内部已经有换行符
- 某些换行是有意义的（比如列表、公式）

## 解决方案

### 方案A：使用PyMuPDF的`get_text("text")`（推荐） ⭐
最简单的方案：直接使用PyMuPDF的文本提取，它会自动处理空格和换行。

```python
page_text = page.get_text("text")
```

**优点**：
- 简单可靠
- PyMuPDF已经处理好了空格和换行
- 性能好

**缺点**：
- 失去了坐标信息（但我们的多栏检测可能本来就有问题）
- 无法精确控制文本顺序

### 方案B：修复当前的blocks处理逻辑
保留当前的坐标级提取，但修复空格问题：

1. **不要把block当作"行内块"**：
```python
for block in sorted_blocks:
    text = block[4].strip()
    if not text or is_garbage_line(text):
        continue
    
    # 直接添加整个block的文本，不要尝试"合并行"
    page_lines.append(clean_text(text))
```

2. **简化heuristic_rebuild**：
- 只处理明显的断词问题（连字符换行）
- 不要过度合并段落

### 方案C：使用`get_text("dict")`获取更细粒度的控制
使用PyMuPDF的字典模式，获取字符级别的坐标：

```python
page_dict = page.get_text("dict")
# 处理spans和chars
```

**优点**：
- 最精确的控制
- 可以实现真正的多栏检测

**缺点**：
- 复杂度高
- 性能开销大

## 推荐实施方案

**短期修复（立即实施）**：方案A
- 用`get_text("text")`替换当前的blocks处理
- 保留图片提取逻辑
- 简化`heuristic_rebuild`

**长期优化（后续迭代）**：方案C
- 实现真正的字符级坐标提取
- 参考paper-burner-x的实现
- 支持复杂的多栏布局

## 其他发现的问题

### 1. 图片引用格式问题
当前代码：
```python
page_text += f"\n\n![图片{len(all_images) + len(page_images)}](images/{img_id}.{img_ext})\n"
```

这会在文本中插入Markdown格式的图片引用，但：
- 图片数据是base64编码的，不是文件路径
- 这个引用在RAG检索时可能会干扰

**建议**：
- 将图片信息单独存储
- 在文本中只保留`[图片X]`这样的简单标记

### 2. 质量评估阈值可能过于严格
```python
needs_ocr = score < 60
```

对于学术论文，即使有一些公式符号，质量分数也可能低于60，但实际上不需要OCR。

**建议**：
- 降低阈值到40-50
- 或者增加"白名单"检测（如果包含大量LaTeX符号，不触发OCR）

## 测试建议

1. 对比测试不同提取方法的效果
2. 使用多个PDF样本（中文、英文、双栏、单栏）
3. 检查RAG检索的准确性（而不只是文本提取的完整性）
