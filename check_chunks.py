#!/usr/bin/env python3
"""
检查向量索引的分块情况
"""
import pickle
import os

doc_id = "a06bec46742c11712a3c952c6f5a6694"
chunks_path = f"data/vector_stores/{doc_id}.pkl"

if os.path.exists(chunks_path):
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        chunks = data["chunks"]
        embedding_model = data.get("embedding_model", "unknown")
    else:
        chunks = data
        embedding_model = "unknown"
    
    print(f"Embedding Model: {embedding_model}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"\n分块大小分布:")
    
    sizes = [len(chunk) for chunk in chunks]
    print(f"  最小: {min(sizes)} 字符")
    print(f"  最大: {max(sizes)} 字符")
    print(f"  平均: {sum(sizes) // len(sizes)} 字符")
    
    print(f"\n前3个分块:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:300])
        print("...")
    
    print(f"\n最后1个分块:")
    print(f"\n--- Chunk {len(chunks)} ({len(chunks[-1])} chars) ---")
    print(chunks[-1][:300])
    print("...")
    
    # 检查是否包含关键章节
    full_text = "\n\n".join(chunks)
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
    
    print(f"\n章节覆盖检查:")
    for section in sections:
        if section in full_text:
            print(f"  ✓ {section}")
        else:
            print(f"  ❌ {section}")
else:
    print(f"向量索引文件不存在: {chunks_path}")
