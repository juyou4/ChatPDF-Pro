#!/usr/bin/env python3
"""
重新上传PDF并保存，测试完整流程
"""
import sys
sys.path.insert(0, 'backend')

from routes.document_routes import extract_text_from_pdf, generate_doc_id, save_document
import io
import json
from datetime import datetime

# 读取PDF文件
pdf_path = "uploads/de76f55cf4e8c26fd836caa3e30086d1.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

pdf_file = io.BytesIO(pdf_bytes)

print("重新提取PDF...")
result = extract_text_from_pdf(pdf_file, pdf_bytes=pdf_bytes, enable_ocr="never", extract_images=True)

doc_id = generate_doc_id(result["full_text"])
print(f"文档ID: {doc_id}")

# 保存文档
doc_data = {
    "filename": "AdvRoad.pdf",
    "upload_time": datetime.now().isoformat(),
    "data": result,
    "pdf_url": f"/uploads/{doc_id}.pdf"
}

save_document(doc_id, doc_data)

print(f"\n✓ 文档已保存到: data/docs/{doc_id}.json")
print(f"提取质量: {result['extraction_quality']}")
print(f"平均质量分数: {result['avg_quality_score']}")
print(f"总字符数: {len(result['full_text'])}")
print(f"图片数量: {result['image_count']}")

# 显示前500字符
print("\n前500字符:")
print(result['full_text'][:500])
