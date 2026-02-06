#!/usr/bin/env python3
"""
测试新的PDF提取逻辑
"""
import sys
sys.path.insert(0, 'backend')

from routes.document_routes import extract_text_from_pdf
import io

# 读取PDF文件
pdf_path = "uploads/de76f55cf4e8c26fd836caa3e30086d1.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

pdf_file = io.BytesIO(pdf_bytes)

print("=" * 80)
print("测试新的PDF提取逻辑")
print("=" * 80)

# 提取文本
result = extract_text_from_pdf(pdf_file, pdf_bytes=pdf_bytes, enable_ocr="never", extract_images=False)

print(f"\n提取方法: {result.get('extraction_method')}")
print(f"总页数: {result.get('total_pages')}")
print(f"平均质量分数: {result.get('avg_quality_score')}")
print(f"提取质量: {result.get('extraction_quality')}")

print("\n" + "=" * 80)
print("第1页内容（前800字符）:")
print("=" * 80)
page1_content = result['pages'][0]['content']
print(page1_content[:800])

print("\n" + "=" * 80)
print("检查关键问题:")
print("=" * 80)

# 检查空格
if "Modern autonomous" in page1_content:
    print("✓ 空格正常: 'Modern autonomous' 找到")
else:
    print("❌ 空格丢失: 'Modern autonomous' 未找到")

if "Modernautonomous" in page1_content:
    print("❌ 发现粘连: 'Modernautonomous'")
else:
    print("✓ 无粘连问题")

# 检查完整性 - 查找关键章节
full_text = result['full_text']

sections = [
    "Introduction",
    "Related Work",
    "Problem Definition",
    "Proposed AdvRoad Framework",
    "Road-Style Adversary Generation",
    "Scenario-Associated Adaptation",
    "Experiment",
    "Experimental Setup"
]

print("\n章节完整性检查:")
for section in sections:
    if section in full_text:
        print(f"  ✓ {section}")
    else:
        print(f"  ❌ {section} - 未找到")

# 统计信息
print(f"\n总字符数: {len(full_text)}")
print(f"总单词数（估算）: {len(full_text.split())}")
print(f"总行数: {len(full_text.split(chr(10)))}")
