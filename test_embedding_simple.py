"""
简化版测试 - 只测试核心功能
跳过多语言模型以避免下载超时
"""

import sys
import os

# 添加backend路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_core_functionality():
    """测试核心功能"""
    print("=" * 70)
    print("ChatPDF - 向量检索和嵌入模型核心功能测试")
    print("=" * 70)
    
    all_passed = True
    
    # 测试1: 本地嵌入模型
    print("\n[测试 1/3] 测试本地免费嵌入模型 (all-MiniLM-L6-v2)")
    print("-" * 70)
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "all-MiniLM-L6-v2"
        print(f"正在加载模型: {model_name}")
        print("注意: 首次运行会自动下载模型 (~80MB)...\n")
        
        model = SentenceTransformer(model_name)
        print("✓ 模型加载成功!")
        
        # 测试嵌入生成
        test_texts = [
            "机器学习是人工智能的核心技术",
            "深度学习使用神经网络",
            "自然语言处理帮助理解文本"
        ]
        
        embeddings = model.encode(test_texts)
        print(f"✓ 嵌入向量生成成功!")
        print(f"  - 文本数量: {len(test_texts)}")
        print(f"  - 嵌入维度: {embeddings.shape[1]}")
        print(f"  - 向量形状: {embeddings.shape}")
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        all_passed = False
    
    # 测试2: FAISS向量检索
    print("\n[测试 2/3] 测试FAISS向量检索功能")
    print("-" * 70)
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # 准备测试文档
        documents = [
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行训练",
            "自然语言处理用于理解和生成人类语言",
            "计算机视觉帮助机器理解图像",
            "今天天气很好,适合出去散步"
        ]
        
        print(f"准备了 {len(documents)} 个测试文档")
        
        # 加载模型并生成嵌入
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = model.encode(documents)
        print(f"✓ 文档嵌入生成完成 (维度: {doc_embeddings.shape[1]})")
        
        # 创建FAISS索引
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings.astype('float32'))
        print(f"✓ FAISS索引创建成功 (包含 {index.ntotal} 个向量)")
        
        # 测试检索
        query = "什么是深度学习?"
        print(f"\n查询: '{query}'")
        
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding.astype('float32'), 3)
        
        print(f"✓ 检索到最相关的3个文档:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"  {i+1}. [相似度距离: {dist:.4f}] {documents[idx]}")
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试3: 应用的完整流程
    print("\n[测试 3/3] 测试应用的完整索引和检索流程")
    print("-" * 70)
    try:
        from app import build_vector_index, get_relevant_context, EMBEDDING_MODELS
        import tempfile
        import shutil
        
        # 显示可用模型
        print("可用的嵌入模型:")
        for model_id, config in EMBEDDING_MODELS.items():
            provider_icon = "💻" if config['provider'] == 'local' else "☁️"
            print(f"  {provider_icon} {model_id}: {config['name']}")
            print(f"     维度: {config['dimension']}, 价格: {config['price']}")
        
        # 创建临时目录
        test_dir = tempfile.mkdtemp(prefix="chatpdf_test_")
        
        # 临时修改全局路径
        import app
        original_vector_dir = app.VECTOR_STORE_DIR
        app.VECTOR_STORE_DIR = test_dir
        
        try:
            # 准备测试文档
            test_doc_id = "test_doc_001"
            test_text = """
            人工智能(AI)是计算机科学的一个重要分支。机器学习是实现人工智能的核心方法。
            深度学习是机器学习的一个子领域,它使用多层神经网络来学习数据的复杂表示。
            自然语言处理(NLP)使计算机能够理解、解释和生成人类语言。
            计算机视觉让机器能够从图像和视频中提取、分析和理解信息。
            强化学习让智能体通过与环境的交互来学习最优决策策略。
            迁移学习可以将在一个任务上学到的知识应用到另一个相关任务上。
            """
            
            print(f"\n正在为测试文档构建向量索引...")
            build_vector_index(test_doc_id, test_text, embedding_model_id="local-minilm")
            
            # 验证文件
            index_path = os.path.join(test_dir, f"{test_doc_id}.index")
            chunks_path = os.path.join(test_dir, f"{test_doc_id}.pkl")
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                print(f"✓ 索引文件创建成功")
            else:
                raise Exception("索引文件未创建")
            
            # 测试检索
            queries = [
                "什么是深度学习?",
                "NLP的作用是什么?",
            ]
            
            print(f"\n测试向量检索:")
            for query in queries:
                print(f"\n  查询: '{query}'")
                context = get_relevant_context(test_doc_id, query, top_k=2)
                
                if context:
                    # 只显示前150个字符
                    preview = context.replace('\n', ' ')[:150] + "..."
                    print(f"  ✓ 检索结果: {preview}")
                else:
                    raise Exception(f"未能检索到内容")
            
            print(f"\n✓ 完整流程测试成功!")
            
        finally:
            # 清理
            app.VECTOR_STORE_DIR = original_vector_dir
            shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 总结
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过!")
        print("=" * 70)
        print("\n✓ 本地免费嵌入模型工作正常")
        print("✓ FAISS向量检索功能正常")
        print("✓ 应用的索引构建和检索流程正常")
        print("\n您可以在ChatPDF中使用向量检索功能来提高问答的准确性!")
        print("在设置中启用'向量检索'选项即可使用。")
    else:
        print("⚠ 部分测试失败")
        print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    try:
        success = test_core_functionality()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
