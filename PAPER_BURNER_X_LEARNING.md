# Paper-Burner-X 深度学习：RAG架构设计哲学

## 核心问题分析

我们遇到的三个问题，paper-burner-x是如何从根本上避免的？

### 问题1：空格丢失
**我们的问题**：使用`get_text("blocks")`导致段落级提取，空格丢失

**Paper-Burner-X的方案**：
- 使用PDF.js的`getTextContent()`获取**字符级**坐标
- 按Y坐标检测换行，按X坐标间距添加空格
- 完全控制文本重建过程

**关键代码**：
```javascript
// paper-burner-x/js/process/ocr-adapters/local-adapter.js
_extractTextFromPage(textContent) {
    const items = textContent.items;
    let text = '';
    let lastY = null;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];

        // 检测换行（Y 坐标变化）
        if (lastY !== null && Math.abs(item.transform[5] - lastY) > 5) {
            text += '\n';
        }

        text += item.str;

        // 如果下一个 item 有空格，添加空格
        if (i < items.length - 1) {
            const nextItem = items[i + 1];
            const spaceWidth = item.width * 0.3; // 估算空格宽度
            if (nextItem.transform[4] - item.transform[4] > spaceWidth) {
                text += ' ';
            }
        }

        lastY = item.transform[5];
    }

    return text.trim();
}
```

**我们学到的**：
- ✓ 字符级提取是王道
- ✓ 坐标信息是关键
- ✓ 空格需要智能判断，不能依赖原始数据

---

### 问题2：图片引用干扰RAG
**我们的问题**：在文本中插入大量`![图片X](images/...)`，干扰检索

**Paper-Burner-X的方案**：
- 图片引用**插入到文本末尾**，不打断正文
- 使用**占位符保护**机制，在处理前移除，处理后恢复
- 图片数量严格过滤（宽高比、尺寸）

**关键代码**：
```javascript
// 先保护图片引用
const imageRefs = [];
rebuilt = rebuilt.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match) => {
    const placeholder = `__IMG_PLACEHOLDER_${imageRefs.length}__`;
    imageRefs.append(match);
    return placeholder;
});

// ... 进行文本处理 ...

// 恢复图片引用
imageRefs.forEach((ref, idx) => {
    rebuilt = rebuilt.replace(`__IMG_PLACEHOLDER_${idx}__`, ref);
});
```

**我们学到的**：
- ✓ 图片信息应该单独存储
- ✓ 如果必须在文本中标记，使用简单标记如`[图X]`
- ✓ 使用占位符保护机制避免被文本处理破坏

---

### 问题3：RAG返回片段（最重要！）
**我们的问题**：
- 分块太小（281字符）
- 检索返回太少（5个分块）
- 没有智能粒度选择

**Paper-Burner-X的革命性方案：语义意群（Semantic Groups）**

## Paper-Burner-X的RAG架构核心

### 1. 三层粒度体系 ⭐⭐⭐⭐⭐

```
文档
  ├─ 意群 (Semantic Groups)
  │   ├─ summary (80字) - 快速浏览
  │   ├─ digest (1000字) - 一般问答
  │   └─ full (完整文本) - 精确查找
  │
  └─ 分块 (Chunks)
      └─ enrichedChunks (带元数据)
```

**设计哲学**：
1. **意群**：语义完整的单元（5000字左右）
2. **三种粒度**：根据问题类型动态选择
3. **混合粒度**：不同意群可以使用不同粒度

### 2. 智能粒度选择器 ⭐⭐⭐⭐⭐

```javascript
// paper-burner-x/js/chatbot/core/smart-granularity-selector.js

const QUERY_PATTERNS = {
    overview: [/总结|概括|概述/],      // 使用 summary
    extraction: [/具体|详细|数据/],    // 使用 full
    analytical: [/分析|解释|为什么/]   // 使用 digest
};

function selectGranularity(query, groups) {
    const queryType = analyzeQueryType(query);
    
    if (queryType === 'overview') {
        return { granularity: 'summary', maxGroups: 10 };
    } else if (queryType === 'extraction') {
        return { granularity: 'full', maxGroups: 3 };
    } else {
        return { granularity: 'digest', maxGroups: 5 };
    }
}
```

**关键特性**：
- **问题类型识别**：根据关键词判断用户意图
- **动态粒度**：不同问题使用不同粒度
- **Token估算**：避免超出LLM上下文限制
- **混合粒度**：最相关的用full，次相关的用digest，其他用summary

### 3. 混合粒度检索 ⭐⭐⭐⭐⭐

```javascript
function selectMixedGranularity(query, rankedGroups, options) {
    const result = [];
    let accumulatedTokens = 0;
    const maxTokens = options.maxTokens || 8000;

    for (let i = 0; i < rankedGroups.length; i++) {
        const group = rankedGroups[i];
        
        // 排名越靠前，使用越高粒度
        let granularity;
        if (i === 0) {
            granularity = 'full';  // 最相关：全文
        } else if (i < 3) {
            granularity = 'digest';  // 前3个：精要
        } else {
            granularity = 'summary';  // 其他：摘要
        }
        
        // Token检查
        const tokens = estimateTokenUsage([group], granularity);
        if (accumulatedTokens + tokens > maxTokens) {
            // 降级或停止
            if (granularity === 'full') granularity = 'digest';
            else if (granularity === 'digest') granularity = 'summary';
            else break;
        }
        
        result.push({ group, granularity, tokens });
        accumulatedTokens += tokens;
    }
    
    return result;
}
```

**设计亮点**：
- **渐进式粒度**：最相关的给最多信息
- **Token预算管理**：动态调整避免超限
- **优雅降级**：Token不够时降低粒度而不是丢弃

### 4. 双索引系统 ⭐⭐⭐⭐

```javascript
// 1. BM25索引（轻量级，无需API）
window.SemanticBM25Search.indexChunks(chunks, docId);

// 2. 向量索引（可选，需要API）
if (useVectorSearch) {
    await window.SemanticVectorSearch.indexChunks(chunks, docId);
}
```

**设计哲学**：
- **BM25作为基础**：始终可用，不依赖外部服务
- **向量搜索作为增强**：可选，提供语义理解
- **用户可配置**：根据需求和成本选择

### 5. 意群生成策略 ⭐⭐⭐⭐

```javascript
const result = await window.SemanticGrouper.aggregate(chunks, {
    targetChars: 5000,   // 目标大小
    minChars: 2500,      // 最小大小
    maxChars: 6000,      // 最大大小
    concurrency: 20,     // 并发数
    docContext: docGist, // 文档总览作为背景
    onProgress: (current, total, message) => {
        // 进度回调
    }
});
```

**关键特性**：
- **语义完整性**：不在句子中间切断
- **大小适中**：5000字左右，保持语义完整
- **并发生成**：20个意群同时生成摘要
- **文档总览**：先生成文档总览，作为后续摘要的背景

## 对比分析

### 我们的方案 vs Paper-Burner-X

| 维度 | 我们的方案 | Paper-Burner-X | 差距 |
|------|-----------|----------------|------|
| **文本提取** | 段落级 → 字符级 | 字符级 | ✓ 已学习 |
| **分块大小** | 281字符 → 654字符 | 5000字符（意群） | ❌ 仍然太小 |
| **粒度层次** | 单一粒度 | 三层粒度 | ❌ 缺失 |
| **智能选择** | 固定top_k=10 | 动态粒度+混合检索 | ❌ 缺失 |
| **Token管理** | 无 | 智能预算管理 | ❌ 缺失 |
| **检索策略** | 单一向量检索 | BM25+向量双索引 | ❌ 缺失 |
| **用户体验** | 被动接受 | 主动配置 | ❌ 缺失 |

## 我们应该学习的核心点

### 1. 语义意群概念 ⭐⭐⭐⭐⭐

**为什么重要**：
- 传统分块：机械切割，破坏语义
- 语义意群：保持语义完整性

**实现要点**：
```python
# 伪代码
def create_semantic_groups(chunks, target_size=5000):
    groups = []
    current_group = []
    current_size = 0
    
    for chunk in chunks:
        # 检查是否是新章节/段落开始
        if is_section_start(chunk) and current_size > min_size:
            # 生成当前意群的摘要
            group = {
                'chunks': current_group,
                'full_text': ''.join(current_group),
                'summary': generate_summary(current_group, 80),
                'digest': generate_summary(current_group, 1000)
            }
            groups.append(group)
            current_group = [chunk]
            current_size = len(chunk)
        else:
            current_group.append(chunk)
            current_size += len(chunk)
            
            # 达到目标大小
            if current_size >= target_size:
                group = create_group(current_group)
                groups.append(group)
                current_group = []
                current_size = 0
    
    return groups
```

### 2. 智能粒度选择 ⭐⭐⭐⭐⭐

**为什么重要**：
- 不同问题需要不同详细程度
- 固定粒度无法满足所有场景

**实现要点**：
```python
def select_granularity(query: str, groups: List[dict]) -> dict:
    # 分析问题类型
    if is_overview_query(query):
        return {'granularity': 'summary', 'max_groups': 10}
    elif is_extraction_query(query):
        return {'granularity': 'full', 'max_groups': 3}
    else:
        return {'granularity': 'digest', 'max_groups': 5}

def is_overview_query(query: str) -> bool:
    patterns = ['总结', '概括', '概述', '主要内容']
    return any(p in query for p in patterns)

def is_extraction_query(query: str) -> bool:
    patterns = ['具体', '详细', '数据', '数值', '步骤']
    return any(p in query for p in patterns)
```

### 3. 混合粒度检索 ⭐⭐⭐⭐⭐

**为什么重要**：
- 最相关的内容应该给最多细节
- 次相关的内容给适当细节
- 不相关的内容给简要信息

**实现要点**：
```python
def mixed_granularity_retrieval(query: str, groups: List[dict], max_tokens: int = 8000):
    # 1. 检索相关意群
    ranked_groups = vector_search(query, groups, top_k=20)
    
    # 2. 混合粒度选择
    result = []
    tokens = 0
    
    for i, group in enumerate(ranked_groups):
        # 渐进式粒度
        if i == 0:
            text = group['full_text']  # 最相关：全文
        elif i < 3:
            text = group['digest']  # 前3个：精要
        else:
            text = group['summary']  # 其他：摘要
        
        # Token检查
        estimated_tokens = len(text) // 2
        if tokens + estimated_tokens > max_tokens:
            # 降级
            if i == 0:
                text = group['digest']
            elif i < 3:
                text = group['summary']
            else:
                break
        
        tokens += len(text) // 2
        result.append(text)
    
    return '\n\n'.join(result)
```

### 4. 双索引系统 ⭐⭐⭐⭐

**为什么重要**：
- BM25：关键词匹配，快速准确
- 向量：语义理解，处理同义词
- 结合使用：互补优势

**实现要点**：
```python
def hybrid_search(query: str, groups: List[dict], use_vector: bool = True):
    # 1. BM25检索（始终执行）
    bm25_results = bm25_search(query, groups, top_k=20)
    
    if not use_vector:
        return bm25_results[:10]
    
    # 2. 向量检索（可选）
    vector_results = vector_search(query, groups, top_k=20)
    
    # 3. 融合排序（RRF - Reciprocal Rank Fusion）
    merged = reciprocal_rank_fusion(bm25_results, vector_results)
    
    return merged[:10]
```

### 5. 用户可配置 ⭐⭐⭐

**为什么重要**：
- 不同用户有不同需求
- 不同文档有不同特点
- 给用户选择权

**实现要点**：
```python
# 配置选项
config = {
    'use_semantic_groups': True,  # 是否使用意群
    'use_vector_search': True,    # 是否使用向量搜索
    'target_group_size': 5000,    # 意群目标大小
    'max_tokens': 8000,           # 最大Token数
    'default_granularity': 'digest'  # 默认粒度
}

# 保存到文档级别
doc_configs[doc_id] = config
```

## 实施建议

### 短期（立即实施）✓
1. **增大分块到1000-1500字符** - 已完成
2. **增加top_k到10-15** - 已完成
3. **实现简单的问题类型识别** - 待实施

### 中期（1-2周）
1. **实现三层粒度体系**
   - summary: 使用LLM生成80字摘要
   - digest: 使用LLM生成1000字精要
   - full: 保留完整文本

2. **实现智能粒度选择**
   - 问题类型识别
   - 动态粒度选择
   - Token预算管理

3. **实现BM25索引**
   - 作为向量检索的补充
   - 提供fallback机制

### 长期（1个月+）
1. **实现语义意群**
   - 智能分组算法
   - 并发摘要生成
   - 缓存机制

2. **实现混合粒度检索**
   - 渐进式粒度
   - 优雅降级
   - Token优化

3. **用户配置界面**
   - 意群开关
   - 向量搜索开关
   - 粒度偏好设置

## 关键代码参考

### Paper-Burner-X核心文件
1. `smart-granularity-selector.js` - 智能粒度选择
2. `semantic-groups-manager.js` - 意群管理
3. `local-adapter.js` - PDF文本提取
4. `_heuristicRebuild` - 文本重建

### 我们需要创建的模块
1. `semantic_grouper.py` - 语义意群生成
2. `granularity_selector.py` - 粒度选择器
3. `hybrid_search.py` - 混合检索
4. `bm25_index.py` - BM25索引

## 总结

Paper-Burner-X的核心优势不在于技术复杂度，而在于**设计哲学**：

1. **语义完整性** > 机械分块
2. **智能选择** > 固定策略
3. **混合粒度** > 单一粒度
4. **用户可配置** > 系统决定
5. **优雅降级** > 硬性限制

我们当前的修复只是"治标"，要真正达到paper-burner-x的水平，需要从架构层面重新设计RAG系统。

**最重要的启示**：
> RAG不是简单的"分块+检索+拼接"，而是一个需要深度设计的智能系统。

**下一步行动**：
1. 先实现简单的问题类型识别
2. 逐步引入三层粒度体系
3. 最终实现完整的语义意群系统
