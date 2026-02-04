# ChatPDF 优化总结

基于 paper-burner-x 项目的分析，已实现以下优化：

## 已完成的优化

### 1. PDF 提取优化 (document_routes.py)

| 优化项 | 说明 | 状态 |
|--------|------|------|
| 多栏检测 | `detect_columns()` - 自动识别双栏论文布局 | ✅ |
| 自适应阈值 | `get_adaptive_thresholds()` - 基于中位数字符高度/宽度 | ✅ |
| 逐页质量评估 | `assess_page_quality()` - 按页决定是否需要OCR | ✅ |
| 图片提取 | 从PDF提取图片，过滤装饰图标，压缩大图 | ✅ |
| 分批处理 | 每50页一批，避免内存溢出 | ✅ |
| 智能段落合并 | `heuristic_rebuild()` - 根据句号、大写、列表标记判断换段 | ✅ |
| 保守垃圾过滤 | 白名单保护公式/引用/特殊格式 | ✅ |

### 2. 表格保护服务 (table_service.py)

```python
from services.table_service import protect_markdown_tables, restore_markdown_tables

# 保护表格
text, placeholders = protect_markdown_tables(markdown)
# 处理文本...
# 恢复表格
result = restore_markdown_tables(processed_text, placeholders)
```

功能：
- 识别 Markdown 表格边界
- 替换为占位符保护结构
- 处理后恢复原始表格
- 自动修复表格格式问题

### 3. 术语库服务 (glossary_service.py)

```python
from services.glossary_service import glossary_service, build_glossary_prompt

# 创建术语集
glossary_service.create_glossary_set("学术术语", "机器学习相关术语")

# 添加术语
glossary_service.add_entry(set_id, "neural network", "神经网络")

# 在文本中查找匹配
matches = glossary_service.find_matches(text)

# 构建提示词指令
instruction = build_glossary_prompt(text, target_lang="中文")
```

API 端点：
- `GET /glossary/sets` - 获取所有术语集
- `POST /glossary/sets` - 创建术语集
- `POST /glossary/sets/{id}/entries` - 添加术语
- `POST /glossary/sets/{id}/import` - 批量导入
- `POST /glossary/match` - 查找匹配术语

### 4. 提示词池服务 (prompt_pool_service.py)

```python
from services.prompt_pool_service import prompt_pool_service, get_healthy_prompt

# 添加提示词
prompt_pool_service.add_prompt(
    name="学术翻译",
    system_prompt="你是专业的学术翻译助手...",
    user_prompt_template="请翻译：${content}"
)

# 选择健康的提示词
prompt = get_healthy_prompt()

# 记录使用结果
prompt_pool_service.record_usage(prompt_id, success=True, response_time=1.5)
```

功能：
- 多提示词轮换
- 健康状态监控 (healthy/degraded/deactivated)
- 连续失败自动降级
- 定时自动复活
- 使用统计与成功率追踪

API 端点：
- `GET /prompts/` - 获取所有提示词
- `POST /prompts/` - 添加提示词
- `GET /prompts/healthy/select` - 选择健康提示词
- `POST /prompts/usage` - 记录使用结果

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF Upload                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   extract_text_from_pdf()                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 多栏检测     │  │ 图片提取     │  │ 质量评估     │           │
│  │ detect_cols  │  │ extract_img  │  │ assess_qual  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │           heuristic_rebuild() 智能段落合并        │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 表格保护     │  │ 术语库匹配   │  │ 向量检索     │           │
│  │ table_svc    │  │ glossary_svc │  │ embedding    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Call                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 提示词池     │  │ 健康监控     │  │ 自动降级     │           │
│  │ prompt_pool  │  │ health_mon   │  │ fallback     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 配置说明

### 术语库配置
术语库数据存储在 `data/glossary/` 目录：
- `index.json` - 术语集索引
- `{set_id}.json` - 各术语集数据

### 提示词池配置
提示词数据存储在 `data/prompt_pool/` 目录：
- `prompts.json` - 提示词列表
- `config.json` - 健康管理配置

健康管理配置项：
```json
{
  "max_consecutive_failures": 2,
  "deactivation_enabled": true,
  "resurrection_time_minutes": 15,
  "resurrection_enabled": true,
  "switch_on_failure": true
}
```

## 后续优化建议

1. **Token 估算** - 在分块时估算 token 数，避免超出 LLM 上下文限制
2. **缓存层** - 对常见查询结果进行缓存
3. **异步处理** - 大文档上传使用后台任务处理
4. **多语言 OCR** - 支持更多语言的 OCR 识别
5. **图片理解** - 集成视觉模型理解图表内容
