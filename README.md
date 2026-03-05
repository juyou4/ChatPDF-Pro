# ChatPDF Pro v3.0.1

<div align="center">

![ChatPDF Logo](https://img.shields.io/badge/ChatPDF_Pro-3.0.1-blue?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![React](https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react)](https://reactjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org)

**智能文档助手 - 与 PDF 对话，让知识触手可及** · [English](README_EN.md)

[快速开始](#快速开始) • [核心功能](#核心功能) • [v3.0.1 新特性](#v301-新特性) • [技术栈](#技术栈) • [项目结构](#项目结构)

</div>

---

## 应用预览

![ChatPDF Pro Screenshot](docs/screenshot.png)

*专业的 PDF 阅读和 AI 对话界面，支持原生 PDF 渲染、对话历史管理、智能文本提取*

### 独立桌面客户端

基于 Electron 构建的独立桌面应用，采用 PyInstaller 将 Python 后端集成打包为单可执行文件，实现**开箱即用，无需配置任何 Python 或 Node.js 环境**。

---

## v3.0.1 新特性

### 桌面客户端架构 (Electron)
- **独立应用** - 基于 Electron 26 打包的 Windows 桌面客户端，脱离浏览器限制
- **一键安装** - 提供 NSIS 格式的独立安装包，双击即可完成安装
- **内嵌后端** - 使用 PyInstaller 打包 FastAPI 后端，应用启动时由进程管理器自动寻找可用端口并拉起后端服务

### 深度思考模式 (Deep Thinking)
- **推理过程可视化** - 在对话区实时展示 AI 的 ThinkingBlock，支持手动折叠与展开
- **推理强度可调** - 可在对话参数中实时调节低、中、高三档推理强度
- **平滑流式输出** - 思考过程与最终回复均支持基于 RequestAnimationFrame 的平滑逐字渲染

### 数学公式引擎 (Math Rendering)
- **多引擎支持** - 内置 KaTeX 与 MathJax 双引擎，用户可在设置面板中自由切换或关闭
- **单美元符号支持** - 支持 `$...$` 行内公式渲染，规避纯文本与公式冲突
- **LaTeX 括号转换** - 内置平衡匹配算法，自动将 `\[...\]` 和 `\(...\)` 转换为标准的 Markdown 公式语法

### 联网增强与 UI 优化
- **联网搜索** - 允许 AI 在作答时获取实时网络信息，并在回答底部清晰展示带有来源链接的引用标签
- **渲染性能** - 引入虚拟滚动（Virtual List），在包含大量长文本的对话历史中依然保持 60fps 流畅体验
- **DOM 直写** - 流式输出期间跳过 React 状态更新，直接通过 ref 修改 DOM 节点，大幅降低内存和 CPU 占用

---

## 核心功能

### PDF 文档处理
- **原生 PDF 渲染** - 基于 PDF.js 的高保真文档显示，支持平滑缩放、翻页、划词与文本选择
- **多模态提取** - 使用 `pdfplumber` 进行高质量文本提取，支持复杂双栏布局识别
- **结构化解析** - 自动检测并提取 PDF 中的表格内容，转换为易于 LLM 理解的 Markdown 结构

### 智能检索增强 (RAG v3.0)
- **语义意群聚合 (Semantic Groups)** - 将零散的文本块聚合为约 5000 字符的语义完整单元，不跨越页码、标题或表格边界
- **三层粒度体系** - 每个意群自动生成 Summary（80 字）、Digest（1000 字）和 Full（全文）三种粒度表示
- **动态粒度匹配** - 检索时通过大模型判断用户意图（如概览、提取、具体数据），自动选择最优的文本粒度返回
- **Token 预算控制** - 根据目标模型自动估算中英文字符 token，当超出上下文窗口时触发智能降级而非强行截断
- **双索引检索** - 同时查询分块级别和意群级别的 FAISS 向量索引，结合 BM25 算法与 RRF（倒数排序融合）进行重排

### AI 对话能力
- **多模型支持** - 原生支持 OpenAI、Anthropic、Google Gemini、Grok 以及 Ollama 本地模型
- **精确引文** - 回答中自动生成 [1] [2] 格式的内联引用，点击即可使左侧 PDF 视图高亮并平滑滚动到对应页
- **划词工具栏** - 在 PDF 中选中文本后，自动弹出悬浮工具栏，支持一键解释、翻译或作为上下文发送给 AI
- **可视化图表** - 自动解析并渲染 AI 生成的 Mermaid 代码块，适用于流程图与思维导图

---

## 快速开始

### 方式一：下载桌面客户端 (推荐)

直接从 [Releases](https://github.com/juyou4/ChatPDF-Pro/releases) 页面下载最新的 `.exe` 安装包。
安装后双击桌面图标即可运行，无需任何环境配置。

### 方式二：源码运行 (Web 模式)

**1. 后端服务 (Python 3.10+)**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**2. 前端服务 (Node.js 18+)**
```bash
cd frontend
npm install
npm run dev
```
访问 `http://localhost:3000` 即可使用。

---

## 技术栈

### 前端 (Frontend)
- **核心**: React 18 + Vite 6 + Tailwind CSS
- **PDF 渲染**: react-pdf 9.0 + PDF.js
- **UI 动画**: Framer Motion
- **Markdown**: ReactMarkdown + rehype-katex / rehype-mathjax
- **桌面端**: Electron 26 + electron-builder

### 后端 (Backend)
- **框架**: FastAPI 0.115 (Uvicorn 异步驱动)
- **PDF 处理**: pdfplumber
- **向量数据库**: FAISS
- **检索架构**: 语义意群 (Semantic Groups) + 双索引 RRF 融合
- **多模型 SDK**: OpenAI, Anthropic, Google Generative AI

---

## 项目结构

```text
ChatPDF/
├── frontend/                    # React 前端

│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPDF.jsx          # 主应用组件
│   │   │   ├── PDFViewer.jsx        # PDF 渲染组件
│   │   │   ├── StreamingMarkdown.jsx # Markdown + 数学公式 + Mermaid 渲染
│   │   │   ├── ThinkingBlock.jsx    # 深度思考可视化
│   │   │   ├── ChatSettings.jsx     # 对话参数设置面板
│   │   │   ├── VirtualMessageList.jsx # 虚拟化消息列表
│   │   │   ├── PresetQuestions.jsx   # 预设问题栏
│   │   │   └── CitationLink.jsx     # 引文点击跳转
│   │   ├── contexts/
│   │   │   ├── ChatParamsContext.jsx # 对话参数（含数学引擎设置）
│   │   │   ├── GlobalSettingsContext.jsx
│   │   │   └── WebSearchContext.jsx  # 联网搜索状态
│   │   ├── hooks/
│   │   │   ├── useMessageState.js    # 消息状态 + 流式请求
│   │   │   └── useSmoothStream.js    # 平滑流式输出
│   │   └── utils/
│   │       └── processLatexBrackets.js # LaTeX 括号转换
│   ├── package.json
│   └── vite.config.js
├── backend/                     # FastAPI 后端
│   ├── app.py                   # 主应用入口
│   ├── desktop_entry.py         # 桌面端冻结打包入口
│   ├── routes/                  # API 路由
│   ├── services/
│   │   ├── semantic_group_service.py  # 语义意群生成
│   │   ├── hybrid_search.py           # 混合检索 + RRF 融合
│   │   ├── context_builder.py         # 上下文拼接与引文生成
│   │   ├── chat_service.py            # AI 对话逻辑与思考流处理
│   │   ├── web_search_service.py      # 联网搜索引擎服务
│   │   ├── embedding_service.py       # 文本嵌入计算与 FAISS 索引
│   │   └── rerank_service.py          # 交叉编码器重排序
│   └── requirements.txt
├── electron/                    # Electron 桌面端
│   ├── src/main.ts              # 主进程：窗口管理、后端拉起、自动更新
│   └── package.json
├── scripts/                     # 跨平台构建脚本
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

**Q: 桌面客户端启动白屏或报错？**
A: 请确保未开启系统代理拦截 localhost，或尝试右键以管理员身份运行。首次启动时应用会自动在后台拉起 Python 引擎，可能需要数秒钟。

**Q: Web 模式下 PDF 无法显示？**
A: 确保后端服务正常运行（默认端口 8000），检查浏览器控制台是否有 CORS 跨域错误或网络拦截。

**Q: API 调用失败或超时？**
A: 检查 API Key 格式是否正确，并确认您的网络环境能否访问目标提供商的接口（如 OpenAI 需要海外网络，或配置代理/中转 URL）。

**Q: 本地模型（Ollama）连接拒绝？**
A: 请确保后台 Ollama 服务已启动，并在系统环境变量中设置了 `OLLAMA_ORIGINS="*" ` 以允许跨域请求。

**Q: 部分数学公式渲染乱码？**
A: 请在左下角设置面板中切换 KaTeX 与 MathJax 引擎。KaTeX 渲染速度快，而 MathJax 对复杂 LaTeX 嵌套的兼容性更好。

---

## 更新日志

### v3.0.1 (当前版本)
- **桌面客户端发布**: 完整的 Windows 独立应用，基于 Electron 26 与 PyInstaller 打包。
- **深度思考增强**: 引入 ThinkingBlock 组件，实现多档位推理可视化与平滑折叠。
- **数学引擎迭代**: 支持 KaTeX 与 MathJax 在线切换，解决复杂 LaTeX 嵌套导致的渲染崩溃。
- **渲染优化**: 重写 StreamingMarkdown 的底层渲染逻辑，使用 DOM Ref 直写规避 React 调和开销，并加入虚拟列表解决历史会话卡顿。

### v3.0.0
- **RAG 架构重构**: 引入语义意群（Semantic Groups）与三级粒度（Full/Digest/Summary）降级策略。
- **Token 精确计算**: 基于语言字符特性的动态预算系统。
- **双路检索**: 意群级别与块级别的 FAISS 向量检索结合 RRF 融合。

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
