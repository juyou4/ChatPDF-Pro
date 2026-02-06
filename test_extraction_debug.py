#!/usr/bin/env python3
"""
测试PDF提取的详细调试脚本
"""
import fitz
import json

pdf_path = "uploads/de76f55cf4e8c26fd836caa3e30086d1.pdf"

doc = fitz.open(pdf_path)
page = doc[0]

print("=" * 80)
print("第1页原始文本块提取（前10个块）:")
print("=" * 80)

blocks = page.get_text("blocks")
text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]

for i, block in enumerate(text_blocks[:10]):
    x0, y0, x1, y1, text, block_no, block_type = block[:7]
    print(f"\nBlock {i}:")
    print(f"  Position: ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f})")
    print(f"  Text: {repr(text[:100])}")

print("\n" + "=" * 80)
print("检查已保存的JSON文件:")
print("=" * 80)

with open("data/docs/de76f55cf4e8c26fd836caa3e30086d1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
page1_content = data["data"]["pages"][0]["content"]
print(f"\n第1页提取的内容（前500字符）:")
print(repr(page1_content[:500]))

print("\n" + "=" * 80)
print("分析问题:")
print("=" * 80)

# 检查空格丢失
if "Modernautonomous" in page1_content:
    print("❌ 发现问题: 单词之间缺少空格 (Modernautonomous)")
elif "Modern autonomous" in page1_content:
    print("✓ 空格正常")

# 检查换行处理
lines = page1_content.split('\n')
print(f"\n总行数: {len(lines)}")
print(f"前5行:")
for i, line in enumerate(lines[:5]):
    print(f"  {i+1}: {repr(line[:80])}")

doc.close()
