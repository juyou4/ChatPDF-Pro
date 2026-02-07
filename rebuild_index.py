#!/usr/bin/env python3
"""
重新构建向量索引
"""
import sys
sys.path.insert(0, 'backend')

from services.embedding_service import build_vector_index
import json

doc_id = "a06bec46742c11712a3c952c6f5a6694"

# 读取文档
with open(f'data/docs/{doc_id}.json', 'r', encoding='utf-8') as f:
    doc_data = json.load(f)

full_text = doc_data['data']['full_text']

print(f"文档ID: {doc_id}")
print(f"文本长度: {len(full_text)} 字符")
print(f"\n重新构建向量索引...")

build_vector_index(
    doc_id=doc_id,
    text=full_text,
    vector_store_dir="data/vector_stores",
    embedding_model_id="local-minilm"
)

print("\n✓ 向量索引重建完成")

# 检查新的分块
import pickle
chunks_path = f"data/vector_stores/{doc_id}.pkl"
with open(chunks_path, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
sizes = [len(chunk) for chunk in chunks]

print(f"\n新的分块统计:")
print(f"  总分块数: {len(chunks)}")
print(f"  平均大小: {sum(sizes) // len(sizes)} 字符")
print(f"  最小: {min(sizes)} 字符")
print(f"  最大: {max(sizes)} 字符")
print(f"\n前2个分块预览:")
for i, chunk in enumerate(chunks[:2]):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:200])
    print("...")
