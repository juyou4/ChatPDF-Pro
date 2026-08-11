# ChatPDF Pro v3.1.0

<div align="center">

![ChatPDF Logo](https://img.shields.io/badge/ChatPDF_Pro-3.1.0-blue?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![React](https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react)](https://reactjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org)

**面向学术论文的本地优先 AI 阅读助手 — 沉浸式阅读 · Agentic RAG · 引用溯源** · [English](README_EN.md)

[快速开始](#快速开始) • [核心功能](#核心功能) • [评测](#ragas-评测) • [技术栈](#技术栈) • [项目结构](#项目结构)

</div>

---

## 项目简介

ChatPDF Pro 是一款面向学术论文与长篇技术文档的本地优先 AI 阅读助手。左侧原生 PDF 阅读器 + 右侧 AI 对话区的双栏布局，配合基于**语义意群 + 三层粒度 + 双索引 RRF** 的检索管线、**Agentic RAG 工具链**以及 **沉浸式阅读**（AI 大纲 · 章节总结 · 悬浮翻译 · 全文预翻译），让模型既能做高层次综述、也能精确定位到某张表格的某一行数值，同时提供逐块母语化阅读体验。对话历史与运行时数据默认保存在本地；如果选择 MinerU、联网模型或联网搜索，相关文档或请求会按配置发送到对应服务。

---

## 应用预览

### 一键速览

<div align="center">
<img src="docs/preview_overview.png" alt="ChatPDF Pro 一键速览" width="880" />
</div>

**左侧**原生 PDF 阅读区（PDF.js 高保真渲染 + 划词工具栏），**右侧**顶部可在「速览 / 对话」两个 Tab 间切换。文档上传后速览自动生成五张结构化卡片：

- **全文概述** — 一段话讲清文档在做什么、为什么做
- **术语解释** — 挑出核心概念并给出 inline 释义
- **论文速读** — 论文方法 / 实验设计 / 解决的问题三段式拆解
- **关键图表解读** — 自动抽取页内图像（支持 MinerU 增强 / PDF 原生 / 矢量图 caption 识别三种源），输出 caption 之外的 AI 解读
- **论文总结** — 优点与创新 + 未来展望

速览卡片和对话 Tab 共用同一文档上下文，点击卡片内的术语或图注可联动回 PDF 定位。

### 对话示例

<div align="center">
<img src="docs/preview_chat.png" alt="ChatPDF Pro 对话示例" width="880" />
</div>

回答中的 `[1]` `[2]` 引文可点击直接跳转到 PDF 对应页；行内 / 块级公式实时渲染；深度思考过程默认折叠；回答下方自动生成追问建议与可选的幻觉自审提示。

### 沉浸式阅读 — AI 总结

<div align="center">
<img src="docs/summary.png" alt="ChatPDF Pro AI 结构化总结" width="880" />
</div>

右侧「解析」Tab 按当前页自动生成 AI 结构化笔记，每条笔记附带证据数量标记；左侧大纲面板以分级树形展示论文标题层级，点击即跳转 PDF 对应位置。

### 沉浸式阅读 — 大纲导航

<div align="center">
<img src="docs/outline.png" alt="ChatPDF Pro 树形大纲导航" width="880" />
</div>

左侧「大纲」Tab 自动识别论文多级标题（Introduction → Motivation → Training → Experiments ...），当前阅读章节高亮显示。右侧解析面板同步展示当前页原文与图注的 AI 解读，图表区域自动提取 caption 并生成翻译。

### 沉浸式阅读 — 悬浮翻译

<div align="center">
<img src="docs/Floating Translation.png" alt="ChatPDF Pro 悬浮翻译" width="880" />
</div>

鼠标悬停 PDF 文本块即弹出逐段中文翻译浮层，翻译结果按块缓存，再次悬停零延迟。右侧解析面板同步展示当前页原文块与翻译状态，红色标注为 AI 补充的结构化提示。支持一键"翻译当页"批量预翻译。

### 独立桌面客户端

基于 Electron 构建的独立桌面应用，使用 PyInstaller 将 Python 后端打包为单可执行文件，开箱即用、无需配置 Python 或 Node.js 环境。

- **CPU-only 精简打包** — 自动剥离 CUDA / cuDNN 运行库（~2.5 GB），典型 Windows 安装包约 500 MB（随 Python 运行库版本略有变化）
- **YOLO 按需下载** — DocLayout-YOLO 权重不内置，在 OCR 设置面板中一键下载或手动指定本地路径
- **安全隔离** — 桌面模式通过 `X-ChatPDF-Token` 认证中间件保护所有 API，自动绑定 `127.0.0.1`
- **隐私安全分发** — 安装包仅包含前端静态资源、Electron 主进程和冻结后的后端运行库；不会携带上传文档、聊天记录、总结/大纲/速览缓存、向量索引、日志、`.env` 或 API Key
- **数据与程序分离** — 桌面端首次运行时在系统用户数据目录创建自己的运行时数据。安装或升级不会删除本机已有的用户数据，也不会将这些数据写回安装包

---

## 核心功能

### PDF 文档处理
- **原生 PDF 渲染** — 基于 PDF.js 的高保真文档显示，支持平滑缩放、翻页、划词与文本选择
- **字符级文本提取** — 主力管线 PyMuPDF `get_text("dict")` + 自适应坐标阈值（换行 / 空格检测），失败时回退到 pdfplumber；启发式修复断行单词与中英标点
- **表格结构化** — PyMuPDF `find_tables()` 检测表格区域并转为带 `[TABLE]` 标记的 Markdown，分块时作为受保护区域不被切割
- **两条明确解析路线** — 上传前选择 MinerU 深度解析或本地解析；本地 Tesseract / PaddleOCR 只作为低质量页的补充，不会悄悄替换主路线

### Agentic RAG 检索

对综述、章节解释、公式框架、多跳证据等复杂问题，自动激活 agent 模式，组合以下工具并把子问题、命中路径和引用候选实时流式推送到前端 Trace 面板（含检索动画与进度可视化）：

| 工具 | 说明 |
|------|------|
| 向量搜索 | 语义意群级 + 分块级双 FAISS 索引 |
| BM25 | 带中英同义词扩展和细粒度分词 |
| grep / regex | 精确字面匹配 |
| boolean search | `AND` / `OR` / `NOT` 组合 |
| map / fetch | 跨章节定位与段落抓取 |
| 树形分解检索 | 递归子问题拆解，逐层回收证据 |

### 智能检索管线 (RAG)
- **语义意群聚合** — 将零散文本块聚合为约 5000 字符的语义完整单元，不跨越页码、标题或表格边界
- **三层粒度** — 每个意群自动生成 Summary（80 字）、Digest（1000 字）和 Full（全文）三种表示，检索时根据用户意图（概览 / 提取 / 精确数据）动态匹配
- **双索引 RRF** — 同时查询分块级和意群级 FAISS 向量索引，结合 BM25 与倒数排序融合进行重排
- **Token 预算控制** — 根据目标模型自动估算 token，超出上下文窗口时智能降级而非强行截断

### 引用溯源增强
- **表格数值锚定** — 数值对比类查询优先选择目标表、目标列、精确行和方法名锚点一致的证据；命中表格行时自动补齐同簇行作为对照上下文
- **公式归一化** — 公式类问题做通用的 LaTeX / OCR 归一化，减少引用漂移
- **答案自审** — 回答结束后用 cheap_model 对照原文片段检测幻觉，命中时展示红色警告横幅

### 沉浸式阅读（v3.1.0 新增）
- **块级索引** — PyMuPDF `get_text("dict")` 逐页提取文本块 bbox，自动识别段落 / 标题 / 表格 / 公式 / caption 类型，为后续大纲与翻译提供坐标基础
- **AI 阅读大纲** — 基于启发式标题检测（字号 / 加粗 / 编号 / 全大写）+ LLM 章节树双路构建，树形分级展示并可点击跳转 PDF 对应位置；当前阅读位置自动高亮为浮动白色卡片
- **AI 章节总结** — 按大纲章节生成摘要卡片，每节显示核心要点与关键发现；支持重新生成、回退本地启发式总结
- **悬浮翻译** — 鼠标悬停/点击 PDF 文本块即显示逐块译文浮层，翻译结果按 `{lang}:{block_id}` 缓存，已翻译的块再次悬停零延迟
- **全文预翻译** — 一键触发全文档批量翻译，后端 `asyncio.Semaphore(5)` 并发控制，单块纯文本调用（非批量 JSON），支持取消、进度条、失败块精确重试
- **翻译容错** — 单块失败不影响其余，返回 `failed_block_ids` 精确定位；provider 层 JSON 解析异常捕获并附 body preview

### 一键速览
- **五卡结构化导读** — 全文概述 / 术语解释 / 论文速读 / 关键图表解读 / 论文总结
- **图表抽取 Adapter 链** — PDF 原生图层（PyMuPDF 取图 + Figure 标题空间匹配）→ 矢量图 caption-only 路径 → MinerU 云端 OCR（通过 Cloudflare Worker 代理）
- **深度可调** — `brief` / `standard` / `detailed` 三档，分别控制每卡字符上限、术语数量和图表数量

### AI 对话能力
- **多模型支持** — OpenAI、Anthropic、Google Gemini、Grok、DeepSeek 以及 Ollama 本地模型
- **精确引文** — 回答中自动生成 `[1]` `[2]` 内联引用，点击即跳转 PDF 对应页
- **深度思考** — ThinkingBlock 实时展示推理过程，支持低 / 中 / 高三档推理强度
- **划词工具栏** — PDF 中选中文本后弹出悬浮工具栏，一键解释、翻译或作为上下文发送给 AI
- **数学公式** — 内置 KaTeX 与 MathJax 双引擎，支持 `$...$` 行内公式和 `\[...\]` 块级公式
- **可视化图表** — 自动渲染 AI 生成的 Mermaid 流程图与思维导图
- **双模型策略** — 查询改写 / 问题分解 / 追问建议 / 答案自审等辅助任务走廉价模型，与主回答模型解耦，节省 40–60% tokens
- **联网搜索** — 允许 AI 作答时获取实时网络信息，回答底部展示带来源链接的引用标签；对已授权来源支持公开网页正文、GitHub README/文件/Issue/PR，以及 YouTube 公开字幕与视频元数据读取

### UI / UX
- **极光弥散背景** — 双层 radial-gradient + CSS 动画慢速漂移，`prefers-reduced-motion` 自适应
- **Agent 检索动画** — 实时进度条、执行中呼吸光效、弹跳点"规划中"提示；检索过程 trace 流式推送到前端
- **Provider 彩色 Logo** — 26 家供应商原色品牌标识，模型切换器一目了然

---

## RAGAS 评测

使用多论文 manifest 做小规模 RAGAS 回归验证，覆盖 baseline 与 holdout 两个 split；回答模型使用 DeepSeek，嵌入评测使用 SiliconFlow `BAAI/bge-m3`。本组数据用于开发回归，不等同于公开基准。

| Split | index_source | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | pdf_native (historical) | 0.617 | 0.645 | 0.697 | 0.813 | - |
| Holdout | pdf_native (historical) | 0.666 | 0.645 | 0.708 | 1.000 | - |
| MinerU A/B A1 | pdf_native | 0.817 | 0.427 | 0.703 | 0.808 | 0.721 |
| MinerU A/B A2 (initial) | mineru | 0.825 | 0.411 | 0.681 | 0.744 | 0.752 |
| MinerU current (answer cells guard, 26Q) | mineru | 0.970 | 0.340 | 0.877 | 0.962 | 0.753 |

> 评测结果必须标注 `index_source`。未标注 `pdf_native` / `mineru` 的 RAGAS 数字只可作历史锚点，不可直接与 MinerU 重建后的结果比较。

MinerU current 使用同一 26 题正式 manifest，不包含 4 题 smoke / probe 样本。`numeric_table` 子集 17 题已通过推荐门槛：Faithfulness 0.973、Context Precision 0.953、Context Recall 1.000、Answer Correctness 0.799，均不低于 A1 `pdf_native` 基线（0.830 / 0.814 / 0.882 / 0.760）。因此深度解析完成后，如果问答索引仍是 `pdf_native`，系统会建议重建问答索引以启用 MinerU 结构化表格证据。Answer Relevancy 在各轮表格实验中都偏低，主要受短数值答案与评测口径影响，不作为本轮推荐门槛。

相对 pre-agent baseline 四项指标均为正向提升。Baseline 分别 +0.36 / +0.24 / +0.63 / +0.52，Holdout 分别 +0.47 / +0.29 / +0.58 / +0.75。

---

## 快速开始

### 方式一：下载桌面客户端（推荐）

从 [Releases](https://github.com/juyou4/ChatPDF-Pro/releases) 页面下载最新 `.exe` 安装包，双击安装即可运行，无需任何环境配置。

### 构建 Windows 桌面安装包

以下流程会先生成构建身份清单，再分别重建前端、冻结 Python 后端，最后生成 x64 NSIS 安装包：

```bash
scripts/build-all.bat
```

默认产物位于 `electron/release/`。安装包文件名会包含版本号与 Git 短 SHA，并生成 `release-manifest.json` 与 `.sha256` 校验文件，便于定位用户实际安装的代码。打包规则会在 PyInstaller 和 Electron 资源复制两个阶段排除 `data/`、`uploads/`、`logs/`、`cache/`、`history/`、`memory/`、向量/语义索引、测试/课程/评测目录、PDF/数据库/序列化产物、`.env`、日志和 API Key 命名文件。不要手动将本机用户数据目录或开发环境的 `data/` 复制到该目录。

版本信息以仓库根目录 `version.json` 为唯一来源；`/version`、`/health`、`/capabilities`、Electron 包版本和前端静态版本都必须与它一致。发布前可运行：

```bash
python scripts/release_metadata.py --check
```

### 方式二：源码运行（Web 模式）

**1. 后端服务（Python 3.10+）**
```bash
cd backend
python -m pip install -r requirements.txt
python app.py
```

**2. 前端服务（Node.js 20.19+，或 22.12+）**
```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:3000` 即可使用。

---

## 使用技巧

### 高效阅读
1. **划词问答** — 在 PDF 中选中文本后，对话框里的提问会自动关联选中内容
2. **预设问题** — 文档加载后点击预设按钮快速获取总结、公式、方法等信息
3. **大纲导航** — 左侧大纲面板显示论文层级结构，点击跳转，当前位置自动高亮
4. **悬浮翻译** — 悬停 PDF 文本块即显示译文，开启"预翻译"后全文译文预加载，悬停零延迟
5. **调整布局** — 拖动中间分隔线调整 PDF 和对话区域的比例

### 智能检索
1. **自动粒度** — 问"总结全文"返回更多意群摘要，问"具体数据"返回少量意群全文
2. **正则搜索** — 输入 `/regex:pattern` 进行精确匹配
3. **布尔搜索** — `term1 AND term2`、`term1 OR term2`、`NOT term`

### 引文验证
1. 回答中的 `[1]` `[2]` 编号对应文档中的具体位置，点击直接跳转
2. Agent 模式下可在 Trace 面板查看子问题拆解和工具调用路径

---

## 技术栈

### 前端
- **核心**: React 18 + Vite 7 + Tailwind CSS
- **PDF 渲染**: react-pdf 9.0 + PDF.js
- **Markdown**: ReactMarkdown + rehype-katex / rehype-mathjax
- **桌面端**: Electron 28 + electron-builder

### 后端
- **框架**: FastAPI 0.115（Uvicorn 异步驱动）
- **PDF 处理**: PyMuPDF 1.24（主）+ pdfplumber 0.11（fallback）
- **向量数据库**: FAISS 1.9
- **检索架构**: 语义意群 + 双索引 RRF + Agentic RAG 工具链
- **多模型 SDK**: OpenAI, Anthropic, Google Generative AI
- **解析与 OCR**: MinerU（Worker 代理或官方直连）/ 本地 Tesseract 与 PaddleOCR 补充

---

## 项目结构

```text
ChatPDF/
├── frontend/                         # React 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPDF.jsx               # 主应用组件
│   │   │   ├── PDFViewer.jsx             # PDF 渲染
│   │   │   ├── AgentTracePanel.jsx       # Agent 子问题 / 工具调用 Trace
│   │   │   ├── DocumentOutline.jsx       # 沉浸式阅读 — 树形大纲导航
│   │   │   ├── ReadingSummaryPanel.jsx   # 沉浸式阅读 — 章节总结卡片
│   │   │   ├── ReadingAnalysisPanel.jsx  # 沉浸式阅读 — 预翻译控制
│   │   │   ├── OverviewPanel.jsx         # 一键速览面板
│   │   │   ├── StreamingMarkdown.jsx     # Markdown + 公式 + Mermaid
│   │   │   ├── ThinkingBlock.jsx         # 深度思考可视化
│   │   │   ├── VirtualMessageList.jsx    # 虚拟化消息列表
│   │   │   └── GlobalSettings.jsx        # 全局设置（含检索调优）
│   │   ├── contexts/                 # React Context（模型 / 参数 / Provider / 阅读设置）
│   │   └── hooks/                    # 消息状态、流式输出、PDF 等 Hooks
│   └── vite.config.js
├── backend/                          # FastAPI 后端
│   ├── app.py                            # 主入口
│   ├── config.py                         # Pydantic 配置（环境变量 + .env）
│   ├── routes/
│   │   ├── chat_routes.py                # 对话 + 流式 SSE
│   │   ├── document_routes.py            # 文档上传 / 解析 / 索引
│   │   └── search_routes.py              # 独立检索接口
│   ├── services/
│   │   ├── retrieval_agent.py            # Agentic RAG 主控
│   │   ├── retrieval_tools.py            # Agent 工具实现
│   │   ├── agent_retrieval_service.py    # Agent 结果整合
│   │   ├── tree_decomposition_retrieval.py # 树形分解检索
│   │   ├── citation_enhancer.py          # 引用溯源增强
│   │   ├── semantic_group_service.py     # 语义意群生成
│   │   ├── hybrid_search.py              # 混合检索 + RRF
│   │   ├── embedding_service.py          # 嵌入计算 + FAISS 索引
│   │   ├── rerank_service.py             # 交叉编码器重排序
│   │   ├── context_builder.py            # 上下文拼接与引文
│   │   ├── chat_service.py               # 对话逻辑与思考流
│   │   ├── query_analyzer.py             # 查询意图分析
│   │   ├── query_rewriter.py             # 多轮指代消解
│   │   ├── overview_service.py           # 速览卡片生成
│   │   ├── layout_service.py             # DocLayout-YOLO 布局检测与资源管理
│   │   ├── block_index_service.py        # 块级索引构建（标题/段落/表格分类）
│   │   ├── reading_outline_service.py    # AI 阅读大纲生成
│   │   ├── section_outline_service.py    # AI 章节大纲 + 总结
│   │   ├── block_translation_service.py  # 逐块翻译 + 缓存 + 并发控制
│   │   └── table_aware_service.py        # 表格结构化检测
│   ├── chatpdf.spec                      # PyInstaller 打包配置（CUDA 剥离 + 数据过滤）
│   └── requirements.txt
├── electron/                         # Electron 桌面端
│   ├── src/main.ts                       # 主进程：窗口管理、后端拉起、IPC
│   └── package.json
├── scripts/                          # 构建脚本
├── start.bat / start.sh              # 一键启动
└── README.md
```

---

## 常见问题

**Q: 桌面客户端启动白屏或报错？**
A: 确保未开启系统代理拦截 localhost，或尝试以管理员身份运行。首次启动时应用会在后台拉起 Python 引擎，可能需要数秒。

**Q: 为什么新安装后仍能看到旧的文档或聊天记录？**
A: 安装包本身不包含任何用户历史。若在同一台电脑、同一系统账户中安装或升级，桌面端会继续使用该账户已有的应用数据目录，因此会显示该机器原先保存的数据；在新的系统账户或没有既有应用数据的机器上会从空白状态开始。

**Q: Web 模式下 PDF 无法显示？**
A: 确认后端服务正常运行（默认端口 8000），检查浏览器控制台有无 CORS 跨域错误。

**Q: API 调用失败或超时？**
A: 检查 API Key 格式是否正确，确认网络环境能访问目标提供商接口（OpenAI 需要海外网络，或配置代理 / 中转 URL）。

**Q: 本地模型（Ollama）连接拒绝？**
A: 确保 Ollama 服务已启动，并在系统环境变量中设置 `OLLAMA_ORIGINS="*"` 以允许跨域请求。

**Q: 数学公式渲染异常？**
A: 在左下角设置面板中切换 KaTeX / MathJax 引擎。KaTeX 渲染快，MathJax 对复杂 LaTeX 嵌套兼容更好。

---

## 更新日志

### v3.1.0（沉浸式阅读 + 翻译管线重构 + UI 焕新）
- **沉浸式阅读** — 块级索引 → AI 阅读大纲（启发式 + LLM 双路）→ 章节总结卡片 → 悬浮翻译 → 全文预翻译，文档打开即构建完整阅读辅助层
- **翻译管线重构** — 废弃批量 JSON 模式，改为单块纯文本 + `asyncio.Semaphore(5)` 并发，每块独立 try/except 互不影响；新增 `failed_block_ids` 返回与精确重试；provider 层 JSON 解析异常捕获
- **树形大纲导航** — DocumentOutline 组件，按字号/加粗/编号多级分层，当前阅读位置浮动白色卡片高亮，已读琥珀标记，点击跳转 PDF 坐标
- **Agent 检索动画** — 实时 trace 流式推送（`setMessages` 浅拷贝刷新），执行中呼吸光效、进度条扫光、弹跳点规划提示
- **极光弥散背景** — 双层 radial-gradient + `aurora-drift` CSS 动画慢速漂移，`prefers-reduced-motion` 自适应
- **Provider 彩色 Logo** — 26 家供应商原色品牌标识（SVG），模型切换器与设置面板显示供应商 logo
- **Provider 模型默认值更新** — systemModels.ts 大幅扩充各供应商最新可用模型列表

### v3.0.2（Agentic RAG 泛化 + 桌面打包优化）
- **Agentic RAG 工具链** — 新增工具化 agent 检索、子问题追踪、树形分解检索，前端新增 AgentTracePanel 展示调用路径
- **引用溯源增强** — 表格数值锚定 + 公式 LaTeX/OCR 归一化，减少引用漂移
- **RAGAS 回归验证** — 多论文 baseline/holdout manifest，四项核心指标均相对 pre-agent baseline 提升
- **数值表格专项** — 针对 "Table N" / 数值对比类查询的专项检索通路（feature flag 可关）
- **BM25 同义词扩展** — 内置中英同义词词典 + 细粒度分词，改善中文长查询召回
- **双模型策略** — 查询改写 / 问题分解 / 追问 / 答案自审等辅助任务独立走廉价模型
- **LLM 查询改写 + 答案自审** — 多轮消解指代；回答后幻觉检测 + 警告横幅
- **受控外部研究适配器** — 外部来源按请求授权、内存缓存和证据身份隔离；JavaScript 页面可通过 `CHATPDF_EXTERNAL_BROWSER_ENABLED=true` 显式启用可选 Playwright 文本渲染 fallback，默认桌面包不携带浏览器运行时
- **请求级 Feature Flag** — 前端 GlobalSettings 新增"检索增强调优"面板，无需重启后端即可三态切换
- **DocLayout-YOLO 资源管理** — OCR 设置面板新增 YOLO 模型一键下载、手动路径配置、重置；权重不再内置于安装包
- **桌面打包瘦身** — PyInstaller spec 自动剥离 CUDA/cuDNN 库，安装包从 ~3 GB 降至 ~440 MB；过滤用户数据防止隐私泄露

### v3.0.1（桌面客户端）
- **桌面客户端发布** — Windows 独立应用，Electron 28 + PyInstaller 打包
- **深度思考增强** — ThinkingBlock 组件，多档位推理可视化与平滑折叠
- **数学引擎迭代** — KaTeX / MathJax 在线切换，修复复杂 LaTeX 嵌套渲染崩溃
- **渲染优化** — StreamingMarkdown 底层改用 DOM Ref 直写，加入虚拟列表

### v3.0.0（RAG 架构重构）
- 语义意群（Semantic Groups）与三级粒度降级策略
- 基于语言字符特性的动态 Token 预算系统
- 意群级 + 块级双 FAISS 向量检索 + RRF 融合

### v2.x
- v2.0.3 — 划词工具栏拖动缩放、搜索引擎自定义模板
- v2.0.2 — pdfplumber 文本提取、表格自动识别、对话历史管理
- v2.0.0 — 蓝白 UI 重构、Vite 6.0 + React 18.3、深色模式、多 AI 提供商

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

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

<div align="center">

**如果这个项目对你有帮助，请给一个 Star 支持一下！**

Made with ❤️ by ChatPDF Team

</div>
