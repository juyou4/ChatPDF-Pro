# ChatPDF RAG改进路线图

基于对Paper-Burner-X的深度学习，制定分阶段实施计划。

## 当前状态评估

### 已完成 ✓
- [x] 字符级文本提取（空格准确率99%）
- [x] 移除图片引用干扰
- [x] 增大分块大小（281→654字符）
- [x] 增加检索数量（5→10个）

### 当前问题
- ❌ 分块仍然太小（654字符 vs paper-burner-x的5000字符）
- ❌ 单一粒度，无法适应不同问题类型
- ❌ 固定检索策略，缺乏智能性
- ❌ 没有Token预算管理
- ❌ 单一向量检索，缺少BM25补充

## 阶段1：快速改进（1-2天）⭐⭐⭐

### 目标
在不改变架构的前提下，快速提升用户体验。

### 任务

#### 1.1 实现简单的问题类型识别
```python
# Chatpdf/backend/services/query_analyzer.py

def analyze_query_type(query: str) -> str:
    """
    分析查询类型
    Returns: 'overview' | 'extraction' | 'analytical' | 'specific'
    """
    query_lower = query.lower()
    
    # 概览性问题
    overview_patterns = ['总结', '概括', '概述', '简述', '大意', '主要内容', '讲什么', '关于什么']
    if any(p in query_lower for p in overview_patterns):
        return 'overview'
    
    # 提取性问题
    extraction_patterns = ['具体', '详细', '准确', '数据', '数值', '步骤', '流程', '公式', '代码']
    if any(p in query_lower for p in extraction_patterns):
        return 'extraction'
    
    # 分析性问题
    analytical_patterns = ['分析', '解释', '说明', '为什么', '怎么', '原因', '比较', '区别']
    if any(p in query_lower for p in analytical_patterns):
        return 'analytical'
    
    return 'specific'
```

#### 1.2 动态调整top_k
```python
# Chatpdf/backend/services/embedding_service.py

def get_dynamic_top_k(query: str, query_type: str) -> int:
    """根据问题类型动态调整top_k"""
    if query_type == 'overview':
        return 15  # 概览问题需要更多上下文
    elif query_type == 'extraction':
        return 5   # 提取问题需要精确内容
    elif query_type == 'analytical':
        return 10  # 分析问题需要适中上下文
    else:
        return 10  # 默认
```

#### 1.3 增大分块到1200字符
```python
# Chatpdf/backend/services/embedding_service.py

def get_chunk_params(...):
    chunk_size = base_chunk_size  # 改为1200
    if max_ctx:
        chunk_size = min(chunk_size, int(max_ctx * 0.5))  # 提高到50%
        chunk_size = max(1000, min(chunk_size, 2000))  # 1000-2000
    
    return chunk_size, chunk_overlap
```

### 预期效果
- 上下文从6.5K增加到12-18K字符
- 根据问题类型智能调整检索数量
- 更好地适应不同查询场景

### 工作量
- 开发：4小时
- 测试：2小时
- 总计：**1天**

---

## 阶段2：引入BM25索引（3-5天）⭐⭐⭐⭐

### 目标
实现双索引系统，提供更robust的检索能力。

### 任务

#### 2.1 实现BM25索引
```python
# Chatpdf/backend/services/bm25_service.py

from rank_bm25 import BM25Okapi
import jieba

class BM25Index:
    def __init__(self):
        self.indexes = {}  # doc_id -> BM25Okapi
        self.chunks = {}   # doc_id -> List[str]
    
    def build_index(self, doc_id: str, chunks: List[str]):
        """构建BM25索引"""
        # 分词
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
        
        # 构建索引
        bm25 = BM25Okapi(tokenized_chunks)
        
        self.indexes[doc_id] = bm25
        self.chunks[doc_id] = chunks
    
    def search(self, doc_id: str, query: str, top_k: int = 10) -> List[dict]:
        """BM25检索"""
        if doc_id not in self.indexes:
            return []
        
        # 分词
        tokenized_query = list(jieba.cut(query))
        
        # 检索
        scores = self.indexes[doc_id].get_scores(tokenized_query)
        
        # 排序
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[doc_id][idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        
        return results
```

#### 2.2 实现混合检索
```python
# Chatpdf/backend/services/hybrid_search.py

def hybrid_search(
    doc_id: str,
    query: str,
    vector_results: List[dict],
    bm25_results: List[dict],
    alpha: float = 0.5
) -> List[dict]:
    """
    混合检索：融合向量检索和BM25检索结果
    alpha: 向量检索权重（0-1）
    """
    # RRF (Reciprocal Rank Fusion)
    k = 60  # RRF参数
    scores = {}
    
    # 向量检索分数
    for rank, item in enumerate(vector_results):
        chunk_id = item.get('chunk_id', item.get('chunk'))
        scores[chunk_id] = scores.get(chunk_id, 0) + alpha / (k + rank + 1)
    
    # BM25分数
    for rank, item in enumerate(bm25_results):
        chunk_id = item.get('chunk_id', item.get('chunk'))
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 - alpha) / (k + rank + 1)
    
    # 排序
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_chunks[:len(vector_results)]
```

#### 2.3 集成到现有系统
```python
# Chatpdf/backend/routes/chat_routes.py

# 构建索引时同时构建BM25
bm25_index.build_index(doc_id, chunks)

# 检索时使用混合检索
vector_results = vector_search(doc_id, query, top_k=20)
bm25_results = bm25_index.search(doc_id, query, top_k=20)
final_results = hybrid_search(doc_id, query, vector_results, bm25_results)
```

### 预期效果
- 关键词匹配更准确（BM25）
- 语义理解更好（向量）
- 检索鲁棒性提升

### 工作量
- 开发：2天
- 测试：1天
- 总计：**3天**

---

## 阶段3：三层粒度体系（1-2周）⭐⭐⭐⭐⭐

### 目标
实现summary/digest/full三层粒度，根据问题类型智能选择。

### 任务

#### 3.1 生成多粒度内容
```python
# Chatpdf/backend/services/multi_granularity_service.py

async def generate_multi_granularity(chunks: List[str], llm_config: dict) -> List[dict]:
    """
    为每个分块生成三层粒度
    """
    results = []
    
    for i, chunk in enumerate(chunks):
        # Full: 原始文本
        full_text = chunk
        
        # Digest: 1000字精要（使用LLM）
        digest = await generate_summary(chunk, max_length=1000, llm_config=llm_config)
        
        # Summary: 80字摘要（使用LLM）
        summary = await generate_summary(chunk, max_length=80, llm_config=llm_config)
        
        results.append({
            'chunk_id': f'chunk-{i}',
            'full': full_text,
            'digest': digest,
            'summary': summary,
            'char_count': len(full_text)
        })
    
    return results
```

#### 3.2 智能粒度选择
```python
# Chatpdf/backend/services/granularity_selector.py

def select_granularity(query: str, query_type: str, chunks: List[dict]) -> dict:
    """
    智能选择粒度
    """
    if query_type == 'overview':
        return {
            'granularity': 'summary',
            'max_chunks': 15,
            'reasoning': '概览问题使用摘要快速扫描'
        }
    elif query_type == 'extraction':
        return {
            'granularity': 'full',
            'max_chunks': 5,
            'reasoning': '提取问题使用全文确保信息完整'
        }
    elif query_type == 'analytical':
        return {
            'granularity': 'digest',
            'max_chunks': 10,
            'reasoning': '分析问题使用精要平衡细节与长度'
        }
    else:
        return {
            'granularity': 'digest',
            'max_chunks': 10,
            'reasoning': '默认使用精要'
        }
```

#### 3.3 混合粒度检索
```python
def mixed_granularity_retrieval(
    query: str,
    ranked_chunks: List[dict],
    max_tokens: int = 8000
) -> str:
    """
    混合粒度检索：最相关的用full，次相关的用digest，其他用summary
    """
    result = []
    tokens = 0
    
    for i, chunk in enumerate(ranked_chunks):
        # 渐进式粒度
        if i == 0:
            text = chunk['full']  # 最相关：全文
        elif i < 3:
            text = chunk['digest']  # 前3个：精要
        else:
            text = chunk['summary']  # 其他：摘要
        
        # Token检查
        estimated_tokens = len(text) // 2
        if tokens + estimated_tokens > max_tokens:
            # 降级
            if i == 0:
                text = chunk['digest']
            elif i < 3:
                text = chunk['summary']
            else:
                break
        
        tokens += estimated_tokens
        result.append(f"[分块{i+1}]\n{text}")
    
    return '\n\n'.join(result)
```

### 预期效果
- 概览问题：返回15个摘要（约1200字）
- 提取问题：返回5个全文（约6000字）
- 分析问题：返回10个精要（约10000字）
- 智能Token管理，避免超限

### 工作量
- 开发：5天
- 测试：2天
- 总计：**1周**

---

## 阶段4：语义意群（2-4周）⭐⭐⭐⭐⭐

### 目标
实现完整的语义意群系统，达到paper-burner-x水平。

### 任务

#### 4.1 语义意群生成
```python
# Chatpdf/backend/services/semantic_grouper.py

async def create_semantic_groups(
    chunks: List[str],
    target_size: int = 5000,
    min_size: int = 2500,
    max_size: int = 6000,
    llm_config: dict = None
) -> List[dict]:
    """
    生成语义意群
    """
    groups = []
    current_group = []
    current_size = 0
    
    for chunk in chunks:
        # 检查是否是新章节开始
        if is_section_start(chunk) and current_size > min_size:
            # 生成当前意群
            group = await create_group(current_group, llm_config)
            groups.append(group)
            current_group = [chunk]
            current_size = len(chunk)
        else:
            current_group.append(chunk)
            current_size += len(chunk)
            
            # 达到目标大小
            if current_size >= target_size:
                group = await create_group(current_group, llm_config)
                groups.append(group)
                current_group = []
                current_size = 0
    
    # 处理剩余
    if current_group:
        group = await create_group(current_group, llm_config)
        groups.append(group)
    
    return groups

async def create_group(chunks: List[str], llm_config: dict) -> dict:
    """创建单个意群"""
    full_text = '\n\n'.join(chunks)
    
    # 并发生成三层粒度
    summary, digest = await asyncio.gather(
        generate_summary(full_text, 80, llm_config),
        generate_summary(full_text, 1000, llm_config)
    )
    
    return {
        'group_id': generate_id(),
        'chunks': chunks,
        'full_text': full_text,
        'summary': summary,
        'digest': digest,
        'char_count': len(full_text),
        'keywords': extract_keywords(full_text)
    }
```

#### 4.2 意群索引
```python
# 为意群建立索引
vector_index.build_index(doc_id, [g['summary'] for g in groups])
bm25_index.build_index(doc_id, [g['summary'] for g in groups])
```

#### 4.3 意群检索
```python
def retrieve_semantic_groups(
    doc_id: str,
    query: str,
    query_type: str,
    max_tokens: int = 8000
) -> str:
    """
    基于意群的检索
    """
    # 1. 检索相关意群
    vector_results = vector_search(doc_id, query, top_k=20)
    bm25_results = bm25_search(doc_id, query, top_k=20)
    ranked_groups = hybrid_search(vector_results, bm25_results)
    
    # 2. 选择粒度
    granularity_config = select_granularity(query, query_type, ranked_groups)
    
    # 3. 混合粒度检索
    context = mixed_granularity_retrieval(
        query,
        ranked_groups,
        max_tokens=max_tokens
    )
    
    return context
```

### 预期效果
- 语义完整性：不在句子中间切断
- 智能粒度：根据问题类型动态调整
- Token优化：最大化信息密度
- 用户体验：接近paper-burner-x水平

### 工作量
- 开发：10天
- 测试：3天
- 优化：2天
- 总计：**3周**

---

## 阶段5：用户配置与优化（1周）⭐⭐⭐

### 目标
提供用户配置界面，允许用户自定义RAG行为。

### 任务

#### 5.1 配置管理
```python
# Chatpdf/backend/models/rag_config.py

class RAGConfig:
    use_semantic_groups: bool = True
    use_vector_search: bool = True
    use_bm25_search: bool = True
    target_group_size: int = 5000
    max_tokens: int = 8000
    default_granularity: str = 'digest'
    hybrid_alpha: float = 0.5  # 向量检索权重
```

#### 5.2 前端配置界面
```javascript
// 配置对话框
const config = {
    useSemanticGroups: true,
    useVectorSearch: true,
    useBM25: true,
    targetGroupSize: 5000,
    maxTokens: 8000
};
```

#### 5.3 缓存机制
```python
# 缓存意群和索引
cache_key = f"{doc_id}_semantic_groups_v2"
redis.set(cache_key, json.dumps(groups), ex=86400)
```

### 预期效果
- 用户可自定义RAG行为
- 缓存提升性能
- 灵活适应不同场景

### 工作量
- 开发：3天
- 测试：2天
- 总计：**1周**

---

## 总体时间表

| 阶段 | 任务 | 工作量 | 优先级 |
|------|------|--------|--------|
| 阶段1 | 快速改进 | 1天 | ⭐⭐⭐ 立即 |
| 阶段2 | BM25索引 | 3天 | ⭐⭐⭐⭐ 本周 |
| 阶段3 | 三层粒度 | 1周 | ⭐⭐⭐⭐⭐ 下周 |
| 阶段4 | 语义意群 | 3周 | ⭐⭐⭐⭐⭐ 本月 |
| 阶段5 | 用户配置 | 1周 | ⭐⭐⭐ 下月 |

**总计**：约6周完整实施

---

## 依赖和资源

### Python包
```bash
pip install rank-bm25  # BM25索引
pip install jieba      # 中文分词
pip install redis      # 缓存
```

### LLM API
- 摘要生成需要LLM API
- 建议使用流式API减少等待时间
- 考虑成本：每个意群需要2次API调用

### 硬件资源
- 向量索引：需要足够内存
- BM25索引：轻量级，内存占用小
- 缓存：Redis或本地文件

---

## 成功指标

### 定量指标
- 检索准确率：>85%
- 平均响应时间：<3秒
- Token利用率：>70%
- 用户满意度：>4.5/5

### 定性指标
- 用户不再抱怨"片段"问题
- 能够回答复杂的分析性问题
- 概览问题和详细问题都能很好处理

---

## 风险和缓解

### 风险1：LLM API成本
**缓解**：
- 实现缓存机制
- 提供本地模型选项
- 允许用户关闭摘要生成

### 风险2：性能问题
**缓解**：
- 异步处理
- 批量生成
- 增量索引

### 风险3：用户学习成本
**缓解**：
- 提供默认配置
- 智能推荐设置
- 详细文档和教程

---

## 下一步行动

### 立即开始（今天）
1. 实现问题类型识别
2. 动态调整top_k
3. 增大分块到1200字符

### 本周完成
1. 实现BM25索引
2. 实现混合检索
3. 测试和优化

### 本月目标
1. 完成三层粒度体系
2. 开始语义意群开发
3. 用户测试和反馈

让我们开始吧！🚀
