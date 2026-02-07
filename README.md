# ChatPDF Pro v3.0.0

<div align="center">

![ChatPDF Logo](https://img.shields.io/badge/ChatPDF_Pro-3.0.0-blue?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![React](https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react)](https://reactjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org)

**智能文档助手 - 与 PDF 对话，让知识触手可及** · [English](README_EN.md)

[快速开始](#快速开始) • [核心功能](#核心功能) • [v3.0 新特性](#v30-新特性) • [技术栈](#技术栈) • [配置指南](#配置指南)

</div>

---

## 应用预览

![ChatPDF Pro Screenshot](docs/screenshot.png)

*专业的 PDF 阅读和 AI 对话界面，支持原生 PDF 渲染、对话历史管理、智能文本提取*

### 一键启动界面示例

![ChatPDF Pro One-Click Start](docs/one-click-start.png)

> `start.bat` / `start.sh` 运行时自动检查版本更新、安装依赖并启动前后端，浏览器会自动打开，关闭窗口即停止服务。

---

## v3.0 新特性

### 🧠 语义意群（Semantic Groups）
- **智能聚合** - 将文本分块聚合为约 5000 字符的语义完整单元，不跨页面、标题或表格边界
- **三层粒度** - 每个意群自动生成 summary（80 字摘要）、digest（1000 字精要）和 full（完整文本）三种表示
- **LLM 摘要** - 调用 AI 生成高质量摘要和关键词，失败时自动降级为文本截断

### 🎯 智能粒度选择
- **查询意图识别** - 自动分析用户问题类型（概览/提取/分析/具体）
- **动态粒度匹配** - 概览问题返回更多意群的摘要，细节问题返回少量意群的全文
- **混合粒度检索** - 最相关的意群给全文，次相关的给精要，其余给摘要

### 💰 Token 预算管理
- **语言感知估算** - 中文按 1.5 字符/token、英文按 4 字符/token 精确估算
- **智能降级** - Token 不够时自动降低粒度（full→digest→summary），而非丢弃内容
- **回答预留** - 自动预留 1500 Token 给 LLM 生成回答

### 🔍 高级搜索
- **正则搜索** - 使用 `/regex:` 前缀进行正则表达式精确匹配
- **布尔搜索** - 支持 AND、OR、NOT 逻辑组合搜索
- **上下文片段** - 每个搜索结果附带前后文上下文

### ⚡ 预设问题与可视化
- **一键问答** - 预设"总结本文"、"关键公式"、"研究方法"等常用问题按钮
- **思维导图** - AI 生成 Markdown 层级结构的思维导图
- **流程图** - AI 生成 Mermaid 语法流程图，前端自动渲染为可视化图表

### 📎 引文追踪
- **引用编号** - AI 回答中自动标注 [1] [2] 等引用来源
- **点击跳转** - 点击引用编号直接跳转到 PDF 对应页面
- **来源验证** - 轻松验证 AI 回答的准确性

### 📊 检索可观测性
- **结构化日志** - 记录每次检索的查询类型、命中来源、Token 使用情况
- **降级追踪** - 自动记录降级原因（LLM 失败、索引缺失等）
- **调试信息** - API 响应包含 retrieval_meta 字段，方便排查问题

---

## 核心功能

### PDF 文档处理
- **原生 PDF 渲染** - 基于 PDF.js 的高保真文档显示，支持缩放、翻页、文本选择
- **智能文本提取** - 使用 pdfplumber 进行高质量文本提取，支持复杂布局识别
- **表格识别** - 自动检测和提取 PDF 中的表格内容，转换为结构化文本
- **分页处理** - 逐页提取和索引，支持精确定位到特定页面
- **可调节缩放** - 支持 50%-200% 无级缩放，自动适应阅读习惯

### AI 对话功能
- **多模型支持** - 集成 OpenAI、Anthropic、Google Gemini、Grok、Ollama 等多个 AI 提供商
- **上下文理解** - 基于文档内容进行智能问答，提供准确的引用和解释
- **流式输出** - 支持打字机效果的实时响应，可调节速度或关闭
- **Markdown 渲染** - 完整支持代码高亮、数学公式（KaTeX）、表格、列表等格式
- **Mermaid 渲染** - 自动检测并渲染 Mermaid 流程图代码块
- **对话历史** - 自动保存对话记录，支持切换和删除历史会话（最多保存 50 条）

### 智能检索（v3.0 升级）
- **语义意群** - 将分块聚合为语义完整的意群，提供三层粒度文本
- **双索引检索** - 同时查询分块级别和意群级别的 FAISS 向量索引
- **BM25 + 向量混合** - 关键词匹配与语义理解互补，RRF 融合排序
- **智能粒度选择** - 根据问题类型自动选择最佳内容详细程度
- **Token 预算管理** - 语言感知的上下文长度控制，智能降级而非丢弃

### 视觉分析能力
- **截图功能** - 支持整页截图和区域选择，配合多模态 AI 进行图像分析
- **图表识别** - 利用 GPT-4V、Claude Sonnet 等视觉模型理解图表、公式、示意图
- **多模态问答** - 结合文本和图像内容进行综合分析

### 用户界面
- **治愈系设计** - 蓝白配色的现代化 UI，毛玻璃效果，流畅动画
- **响应式布局** - 可拖拽调整 PDF 预览和对话区域的比例
- **深色模式** - 支持浅色/深色主题切换，适应不同使用环境
- **预设问题栏** - 文档加载后显示常用问题按钮，一键发起查询
- **引文点击跳转** - 点击 AI 回答中的引用编号跳转到 PDF 对应位置
- **划词工具栏** - 可复制，直接用搜索引擎搜索、AI 解读/翻译选中文本
- **键盘快捷键** - Enter 发送消息，Shift+Enter 换行

---

## 快速开始

### 一键启动（推荐）

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

启动脚本会自动：
- 检查并更新到最新版本
- 安装缺失的依赖
- 启动后端服务（端口 8000）
- 启动前端服务（端口 3000）
- 自动打开浏览器

### 手动启动

**后端:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**前端:**
```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000 即可使用。

---

## 配置指南

### API Key 配置

首次使用需要配置 AI 服务商的 API Key：

1. 点击左下角"设置 & API Key"按钮
2. 选择 API Provider（OpenAI、Anthropic、Google 等）
3. 选择对应的模型
4. 输入 API Key
5. 保存设置

配置会自动保存到浏览器 localStorage，下次无需重新输入。

### 支持的 AI 提供商

| 提供商 | 模型示例 | 视觉支持 | 备注 |
|--------|----------|----------|------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-4o Mini | ✓ | 最佳多模态体验 |
| Anthropic | Claude Sonnet 4.5, Claude 3 Opus | ✓ | 长文档理解优秀 |
| Google | Gemini 2.5 Pro, Gemini 2.5 Flash | ✓ | 高性价比 |
| Grok | Grok 4.1, Grok Vision | ✓ | xAI 出品 |
| 通义千问 (DashScope) | qwen-max, qwen-long, qwen-vl | 部分 | 低成本，支持长文本 |
| 火山豆包 | doubao-1.5-pro-256k | 部分 | 国内可用，长文本性价比高 |
| MiniMax | abab6.5-chat / s-chat | ✗ | 国内可用，OpenAI 兼容接口 |
| Ollama | Llama 3, Qwen, Mistral | ✗ | 本地运行，完全免费 |
| 自定义 OpenAI 兼容 | 任意兼容 `chat/completions` 接口 | 取决于后端 | 配置自定义 base_url + API Key |

### 本地模型（Ollama）

无需 API Key，完全本地运行：

1. 安装 Ollama: https://ollama.com/
2. 拉取模型: `ollama pull llama3`
3. 在设置中选择"Local (Ollama)"提供商
4. 开始使用

### 功能开关

在设置中可以启用/禁用以下功能：

- **Vector Search** - 向量检索增强（需要更长的索引时间）
- **Semantic Groups** - 语义意群功能（v3.0 新增，默认启用）
- **Screenshot Analysis** - 截图分析功能（仅视觉模型可用）
- **流式输出速度** - 快速/正常/慢速/关闭
- **自定义搜索引擎** - 选择"自定义"后输入模板 URL，使用 `{query}` 作为搜索词占位符

---

## 技术栈

### 前端
- **构建工具**: Vite 6.0 - 极速开发体验
- **框架**: React 18.3 - 现代化组件开发
- **PDF 渲染**: react-pdf 9.0 + PDF.js 4.8.69
- **样式**: Tailwind CSS 3.4 - 实用优先的 CSS 框架
- **动画**: Framer Motion - 流畅的页面过渡和交互
- **Markdown**: ReactMarkdown + rehype/remark 插件生态
- **流程图**: Mermaid - 代码块自动渲染为可视化图表
- **数学公式**: KaTeX
- **代码高亮**: Highlight.js

### 后端
- **框架**: FastAPI 0.115 - 高性能异步 API
- **PDF 处理**: pdfplumber 0.11 - 高质量文本和表格提取
- **AI 编排**: LangChain 0.3 - 统一的 LLM 接口
- **向量数据库**: FAISS - 高效相似度搜索（分块级 + 意群级双索引）
- **文本嵌入**: Sentence Transformers 3.3
- **检索增强**: BM25 + 向量混合检索，RRF 融合排序
- **语义意群**: 三层粒度体系（summary/digest/full）+ 智能粒度选择
- **Token 管理**: 语言感知估算 + 预算控制 + 智能降级
- **HTTP 客户端**: httpx - 异步 HTTP 请求

### AI SDK
- openai 1.57
- anthropic 0.40
- google-generativeai 0.8

---

## 项目结构

```
ChatPDF/
├── frontend/                    # React 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPDF.jsx          # 主应用组件
│   │   │   ├── PDFViewer.jsx        # PDF 渲染组件
│   │   │   ├── PresetQuestions.jsx   # 预设问题栏
│   │   │   ├── StreamingMarkdown.jsx # Markdown + Mermaid 渲染
│   │   │   └── CitationLink.jsx     # 引文点击跳转
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/                     # FastAPI 后端
│   ├── app.py                   # 主应用和路由
│   ├── services/
│   │   ├── semantic_group_service.py  # 语义意群生成
│   │   ├── granularity_selector.py    # 智能粒度选择
│   │   ├── token_budget.py            # Token 预算管理
│   │   ├── context_builder.py         # 上下文构建 + 引文追踪
│   │   ├── advanced_search.py         # 正则/布尔搜索
│   │   ├── preset_service.py          # 预设问题 + 生成提示词
│   │   ├── retrieval_logger.py        # 检索可观测性
│   │   ├── embedding_service.py       # 向量索引 + 检索
│   │   ├── query_analyzer.py          # 查询意图分析
│   │   ├── bm25_service.py            # BM25 检索
│   │   ├── hybrid_search.py           # 混合检索 + RRF 融合
│   │   └── rag_config.py              # RAG 配置
│   ├── requirements.txt
│   └── tests/                   # 单元测试 + 属性测试
├── data/
│   ├── semantic_groups/         # 意群数据（JSON + FAISS 索引）
│   └── vector_stores/           # 分块向量索引
├── start.sh / start.bat         # 启动脚本
├── THIRD_PARTY_NOTICES.md       # 第三方声明
└── README.md
```

---

## 使用技巧

### 高效阅读
1. **文本选择问答** - 在 PDF 中选择文本后，在对话框中提问可以针对选中内容回答
2. **预设问题** - 文档加载后点击预设按钮快速获取总结、公式、方法等信息
3. **调整布局** - 拖动中间分隔线调整 PDF 和对话区域的比例

### 智能检索
1. **自动粒度** - 问"总结全文"会返回更多意群的摘要，问"具体数据"会返回少量意群的全文
2. **正则搜索** - 输入 `/regex:pattern` 进行精确匹配
3. **布尔搜索** - 使用 `term1 AND term2`、`term1 OR term2`、`NOT term` 组合搜索

### 引文验证
1. AI 回答中的 [1] [2] 等编号对应文档中的具体位置
2. 点击编号直接跳转到 PDF 对应页面
3. 可在 retrieval_meta 中查看详细的检索信息

### 可视化生成
1. 点击"生成思维导图"按钮获取文档结构化概览
2. 点击"生成流程图"按钮获取 Mermaid 可视化流程图
3. 流程图会自动渲染，也可复制 Mermaid 代码到其他工具使用

---

## 常见问题

**Q: PDF 无法显示？**
A: 确保后端服务正常运行（端口 8000），检查浏览器控制台是否有错误信息。

**Q: API 调用失败？**
A: 检查 API Key 是否正确，确认账户有足够额度，查看网络连接是否正常。

**Q: 本地模型无响应？**
A: 确认 Ollama 服务已启动（`ollama serve`），模型已下载（`ollama list`）。

**Q: 语义意群生成失败？**
A: 意群摘要需要调用 LLM API。如果 API 不可用，系统会自动降级为文本截断，不影响基本功能。

**Q: 文本提取质量差？**
A: 对于扫描版 PDF 建议先进行 OCR 处理。pdfplumber 对文字版 PDF 效果最佳。

**Q: 对话历史丢失？**
A: 历史记录保存在浏览器 localStorage，清除浏览器数据会导致丢失。

---

## 更新日志

### v3.0.0 (当前版本)
- 🧠 语义意群系统 - 将分块聚合为语义完整单元，三层粒度体系（summary/digest/full）
- 🎯 智能粒度选择 - 根据查询类型自动匹配最佳内容详细程度
- 💰 Token 预算管理 - 语言感知估算，智能降级而非丢弃
- 🔍 高级搜索 - 正则表达式搜索和布尔逻辑搜索
- ⚡ 预设问题 - 一键发起常用问题，支持思维导图和流程图生成
- 📎 引文追踪 - AI 回答标注引用来源，点击跳转到 PDF 对应位置
- 📊 检索可观测性 - 结构化日志记录，API 响应包含调试信息
- 🔄 意群级向量索引 - 分块 + 意群双索引，RRF 融合检索
- ⬇️ 优雅降级 - 意群功能可配置开关，LLM 不可用时自动降级

### v2.0.3
- 划词工具栏支持拖动和四角缩放
- 搜索引擎可自定义模板 URL
- 修复悬浮工具栏按钮无效等问题

### v2.0.2
- 升级到 pdfplumber 进行更高质量的文本提取
- 新增表格自动识别和格式化
- 实现完整的对话历史管理

### v2.0.0
- 全新蓝白治愈系 UI 设计
- 升级到 Vite 6.0 + React 18.3
- 支持深色模式
- 集成多个 AI 提供商
- 添加截图和视觉分析功能

---

## 致谢

本项目的 RAG 系统优化方案借鉴了 [Paper Burner X](https://github.com/Feather-2/paper-burner-x) 的设计理念（语义意群、三层粒度、智能粒度选择等概念）。Paper Burner X 采用 AGPL-3.0 许可证，版权归 Feather-2 及贡献者所有。ChatPDF 的所有实现代码为独立编写的 Python 代码，未复制其源代码。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star 支持一下！**

Made with ❤️ by ChatPDF Team

</div>
