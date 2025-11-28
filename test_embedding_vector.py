"""
测试向量检索和嵌入模型功能
检查向量检索是否正常工作,以及免费嵌入模型是否能自动下载
"""

import sys
import os
import numpy as np
import faiss
import pickle

# 添加backend路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_local_embedding_model():
    """测试本地嵌入模型 - all-MiniLM-L6-v2"""
    print("=" * 60)
    print("测试1: 测试本地嵌入模型加载和嵌入生成")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 测试默认的免费模型
        model_name = "all-MiniLM-L6-v2"
        print(f"\n正在加载模型: {model_name}")
        print("注意: 如果这是第一次运行,模型会自动从Hugging Face下载")
        print("下载大小约为 80-90 MB,请耐心等待...\n")
        
        model = SentenceTransformer(model_name)
        print(f"✓ 模型加载成功!")
        
        # 测试嵌入生成
        test_texts = [
            "这是一个测试文本",
            "This is a test document about machine learning",
            "人工智能和深度学习"
        ]
        
        print(f"\n正在生成 {len(test_texts)} 个文本的嵌入向量...")
        embeddings = model.encode(test_texts)
        
        print(f"✓ 嵌入向量生成成功!")
        print(f"  - 嵌入维度: {embeddings.shape[1]}")
        print(f"  - 向量数量: {embeddings.shape[0]}")
        print(f"  - 数据类型: {embeddings.dtype}")
        
        # 显示向量的前几个值
        print(f"\n第一个向量的前10个值:")
        print(f"  {embeddings[0][:10]}")
        
        return True, embeddings
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_multilingual_embedding_model():
    """测试多语言嵌入模型"""
    print("\n" + "=" * 60)
    print("测试2: 测试多语言嵌入模型")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        print(f"\n正在加载模型: {model_name}")
        print("注意: 如果这是第一次运行,模型会自动下载")
        print("下载大小约为 420 MB,请耐心等待...\n")
        
        model = SentenceTransformer(model_name)
        print(f"✓ 多语言模型加载成功!")
        
        # 测试多语言文本
        test_texts = [
            "人工智能正在改变世界",
            "Artificial intelligence is changing the world",
            "人工知能は世界を変えています"
        ]
        
        print(f"\n正在生成多语言文本的嵌入向量...")
        embeddings = model.encode(test_texts)
        
        print(f"✓ 多语言嵌入生成成功!")
        print(f"  - 嵌入维度: {embeddings.shape[1]}")
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        print(f"\n语义相似度矩阵 (相同意思的不同语言):")
        print(f"  中文-英文: {similarities[0][1]:.4f}")
        print(f"  中文-日文: {similarities[0][2]:.4f}")
        print(f"  英文-日文: {similarities[1][2]:.4f}")
        
        return True, embeddings
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_vector_search():
    """测试FAISS向量检索功能"""
    print("\n" + "=" * 60)
    print("测试3: 测试FAISS向量检索")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 准备测试文档
        documents = [
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行训练",
            "自然语言处理用于理解和生成人类语言",
            "计算机视觉帮助机器理解图像",
            "今天天气很好,适合出去散步",
            "我喜欢吃苹果和香蕉",
        ]
        
        print(f"\n准备了 {len(documents)} 个测试文档")
        
        # 加载模型
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ 模型加载完成")
        
        # 生成文档嵌入
        print("\n正在为文档生成嵌入向量...")
        doc_embeddings = model.encode(documents)
        print(f"✓ 文档嵌入生成完成 (维度: {doc_embeddings.shape[1]})")
        
        # 创建FAISS索引
        print("\n正在创建FAISS索引...")
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings.astype('float32'))
        print(f"✓ FAISS索引创建成功 (索引中有 {index.ntotal} 个向量)")
        
        # 测试检索
        queries = [
            "什么是深度学习?",
            "如何处理文本数据?",
            "水果有哪些?"
        ]
        
        print("\n" + "-" * 60)
        print("开始测试向量检索:")
        print("-" * 60)
        
        for query in queries:
            print(f"\n查询: '{query}'")
            
            # 生成查询嵌入
            query_embedding = model.encode([query])
            
            # 搜索最相关的3个文档
            k = 3
            distances, indices = index.search(query_embedding.astype('float32'), k)
            
            print(f"  最相关的 {k} 个文档:")
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                print(f"    {i+1}. [距离: {dist:.4f}] {documents[idx]}")
        
        print("\n✓ 向量检索测试完成!")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_embedding_function():
    """测试应用中的嵌入函数"""
    print("\n" + "=" * 60)
    print("测试4: 测试应用的嵌入函数接口")
    print("=" * 60)
    
    try:
        # 导入应用代码
        from app import get_embedding_function, EMBEDDING_MODELS
        
        print(f"\n可用的嵌入模型:")
        for model_id, config in EMBEDDING_MODELS.items():
            print(f"  - {model_id}: {config['name']}")
            print(f"    提供商: {config['provider']}, 维度: {config['dimension']}, 价格: {config['price']}")
        
        # 测试本地模型
        print(f"\n测试 'local-minilm' 模型...")
        embed_fn = get_embedding_function("local-minilm")
        
        test_texts = ["测试文本1", "测试文本2", "测试文本3"]
        embeddings = embed_fn(test_texts)
        
        print(f"✓ 嵌入函数工作正常!")
        print(f"  - 输入文本数: {len(test_texts)}")
        print(f"  - 输出向量形状: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_and_search_index():
    """测试完整的索引构建和检索流程"""
    print("\n" + "=" * 60)
    print("测试5: 测试完整的索引构建和检索流程")
    print("=" * 60)
    
    try:
        from app import build_vector_index, get_relevant_context
        import tempfile
        import shutil
        
        # 创建临时目录用于测试
        test_dir = tempfile.mkdtemp(prefix="chatpdf_test_")
        print(f"\n创建临时测试目录: {test_dir}")
        
        # 临时修改全局路径
        import app
        original_vector_dir = app.VECTOR_STORE_DIR
        app.VECTOR_STORE_DIR = test_dir
        
        try:
            # 准备测试文档
            test_doc_id = "test_doc_123"
            test_text = """
            人工智能是计算机科学的一个重要分支。机器学习是人工智能的核心技术之一。
            深度学习是机器学习的一个子领域,它使用多层神经网络来学习数据的表示。
            自然语言处理(NLP)是人工智能的另一个重要应用领域,它使计算机能够理解和生成人类语言。
            计算机视觉使机器能够从图像和视频中提取信息并进行理解。
            强化学习让智能体通过与环境交互来学习最优策略。
            """
            
            # 构建向量索引
            print(f"\n为测试文档构建向量索引...")
            build_vector_index(test_doc_id, test_text, embedding_model_id="local-minilm")
            
            # 验证文件是否创建
            index_path = os.path.join(test_dir, f"{test_doc_id}.index")
            chunks_path = os.path.join(test_dir, f"{test_doc_id}.pkl")
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                print(f"✓ 索引文件创建成功!")
                print(f"  - 索引文件: {index_path}")
                print(f"  - 分块文件: {chunks_path}")
            else:
                print(f"✗ 索引文件未创建")
                return False
            
            # 测试检索
            test_queries = [
                "什么是深度学习?",
                "NLP是什么?",
                "强化学习如何工作?"
            ]
            
            print("\n" + "-" * 60)
            print("测试向量检索:")
            print("-" * 60)
            
            for query in test_queries:
                print(f"\n查询: '{query}'")
                context = get_relevant_context(test_doc_id, query, top_k=2)
                
                if context:
                    print(f"✓ 检索到相关内容:")
                    # 只显示前200个字符
                    preview = context[:200] + "..." if len(context) > 200 else context
                    print(f"  {preview}")
                else:
                    print(f"✗ 未检索到内容")
            
            print("\n✓ 完整流程测试成功!")
            return True
            
        finally:
            # 恢复原始路径并清理临时目录
            app.VECTOR_STORE_DIR = original_vector_dir
            try:
                shutil.rmtree(test_dir)
                print(f"\n清理临时目录: {test_dir}")
            except:
                pass
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ChatPDF - 向量检索和嵌入模型功能测试")
    print("=" * 60)
    print("\n这个测试将验证:")
    print("  1. 本地免费嵌入模型能否正常加载(首次会自动下载)")
    print("  2. 多语言嵌入模型能否正常工作")
    print("  3. FAISS向量检索功能是否正常")
    print("  4. 应用的嵌入函数接口是否正常")
    print("  5. 完整的索引构建和检索流程是否正常")
    print("\n" + "=" * 60)
    
    results = []
    
    # 运行测试
    success1, _ = test_local_embedding_model()
    results.append(("本地嵌入模型", success1))
    
    success2, _ = test_multilingual_embedding_model()
    results.append(("多语言嵌入模型", success2))
    
    success3 = test_vector_search()
    results.append(("FAISS向量检索", success3))
    
    success4 = test_app_embedding_function()
    results.append(("应用嵌入函数", success4))
    
    success5 = test_build_and_search_index()
    results.append(("完整索引流程", success5))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过! 向量检索和嵌入模型功能正常!")
        print("=" * 60)
        print("\n✓ 免费本地嵌入模型已成功下载并可以使用")
        print("✓ 向量检索功能工作正常")
        print("✓ 应用可以正常进行语义搜索")
    else:
        print("\n" + "=" * 60)
        print("⚠ 部分测试失败,请检查错误信息")
        print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
