# ChatPDF Pro v2.0.3

<div align="center">

![ChatPDF Logo](https://img.shields.io/badge/ChatPDF_Pro-2.0.3-blue?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![React](https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react)](https://reactjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org)

**智能文档助手 - 与 PDF 对话，让知识触手可及** · [English](README_EN.md)

[快速开始](#快速开始) • [核心功能](#核心功能) • [技术栈](#技术栈) • [配置指南](#配置指南)

</div>

---

## 应用预览

![ChatPDF Pro Screenshot](docs/screenshot.png)

*专业的 PDF 阅读和 AI 对话界面，支持原生 PDF 渲染、对话历史管理、智能文本提取*

> **提示**: 将应用截图保存为 `docs/screenshot.png` 即可显示预览图

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
- **对话历史** - 自动保存对话记录，支持切换和删除历史会话（最多保存 50 条）

### 视觉分析能力
- **截图功能** - 支持整页截图和区域选择，配合多模态 AI 进行图像分析
- **图表识别** - 利用 GPT-4V、Claude Sonnet 等视觉模型理解图表、公式、示意图
- **多模态问答** - 结合文本和图像内容进行综合分析

### 向量检索（可选）
- **语义搜索** - 基于 Sentence Transformers 的向量化文本检索
- **相似度匹配** - 使用 FAISS 快速查找文档中的相关段落
- **智能引用** - 自动定位答案来源的具体页面和段落

### 用户界面
- **治愈系设计** - 蓝白配色的现代化 UI，毛玻璃效果，流畅动画
- **响应式布局** - 可拖拽调整 PDF 预览和对话区域的比例
- **深色模式** - 支持浅色/深色主题切换，适应不同使用环境
- **会话管理** - 侧边栏显示历史对话列表，一键切换或删除
- **划词工具栏** - 可复制，直接用搜索引擎搜索、ai解读/翻译选中文本，支持自定义搜索引擎模板（`{query}` 占位）
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
| 通义千问 (DashScope) | qwen-max, qwen-long, qwen-vl | 部分 | 低成本，支持长文本（视觉需支持的模型） |
| 火山豆包 | doubao-1.5-pro-256k | 部分 | 国内可用，长文本性价比高 |
| MiniMax | abab6.5-chat / s-chat | ✗ | 国内可用，OpenAI 兼容接口 |
| Ollama | Llama 3, Qwen, Mistral | ✗ | 本地运行，完全免费 |
| 自定义 OpenAI 兼容 | 任意兼容 `chat/completions` 接口 | 取决于后端 | 配置自定义 base_url + API Key 即可接入 |

### 本地模型（Ollama）

无需 API Key，完全本地运行：

1. 安装 Ollama: https://ollama.com/
2. 拉取模型: `ollama pull llama3`
3. 在设置中选择"Local (Ollama)"提供商
4. 开始使用

### 功能开关

在设置中可以启用/禁用以下功能：

- **Vector Search** - 向量检索增强（需要更长的索引时间）
- **Screenshot Analysis** - 截图分析功能（仅视觉模型可用）
- **流式输出速度** - 快速/正常/慢速/关闭
- **自定义搜索引擎** - 选择“自定义”后输入模板 URL，使用 `{query}` 作为搜索词占位符（未包含时默认追加 `?q=`）

---

## 技术栈

### 前端
- **构建工具**: Vite 6.0 - 极速开发体验
- **框架**: React 18.3 - 现代化组件开发
- **PDF 渲染**: react-pdf 9.0 + PDF.js 4.8.69
- **样式**: Tailwind CSS 3.4 - 实用优先的 CSS 框架
- **动画**: Framer Motion - 流畅的页面过渡和交互
- **Markdown**: ReactMarkdown + rehype/remark 插件生态
- **数学公式**: KaTeX
- **代码高亮**: Highlight.js

### 后端
- **框架**: FastAPI 0.115 - 高性能异步 API
- **PDF 处理**: pdfplumber 0.11 - 高质量文本和表格提取
- **AI 编排**: LangChain 0.3 - 统一的 LLM 接口
- **向量数据库**: FAISS - 高效相似度搜索
- **文本嵌入**: Sentence Transformers 3.3
- **HTTP 客户端**: httpx - 异步 HTTP 请求

### AI SDK
- openai 1.57
- anthropic 0.40
- google-generativeai 0.8

---

## 项目结构

```
ChatPDF/
├── frontend/               # React 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPDF.jsx      # 主应用组件
│   │   │   └── PDFViewer.jsx    # PDF 渲染组件
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/                # FastAPI 后端
│   ├── app.py              # 主应用和路由
│   ├── requirements.txt
│   └── uploads/            # PDF 文件存储
├── start.sh / start.bat    # 启动脚本
└── README.md
```

---

## 使用技巧

### 高效阅读
1. **文本选择问答** - 在 PDF 中选择文本后，在对话框中提问可以针对选中内容回答
2. **分页查看** - 使用上下翻页按钮浏览文档，问答会基于整个文档内容
3. **调整布局** - 拖动中间分隔线调整 PDF 和对话区域的比例

### 对话管理
1. **新建对话** - 点击"新对话 / 上传PDF"按钮清空当前会话并上传新文档
2. **切换历史** - 在左侧边栏点击历史记录快速切换到之前的对话
3. **删除对话** - 鼠标悬停在历史记录上，点击垃圾桶图标删除

### 视觉分析
1. 点击截图按钮捕获当前 PDF 页面
2. 输入问题（如"解释这个图表的含义"）
3. 发送后 AI 会结合截图内容进行分析

---

## 常见问题

**Q: PDF 无法显示？**
A: 确保后端服务正常运行（端口 8000），检查浏览器控制台是否有错误信息。

**Q: API 调用失败？**
A: 检查 API Key 是否正确，确认账户有足够额度，查看网络连接是否正常。

**Q: 本地模型无响应？**
A: 确认 Ollama 服务已启动（`ollama serve`），模型已下载（`ollama list`）。

**Q: 文本提取质量差？**
A: v2.0.2 已升级到 pdfplumber，对于扫描版 PDF 仍建议先进行 OCR 处理。

**Q: 对话历史丢失？**
A: 历史记录保存在浏览器 localStorage，清除浏览器数据会导致丢失。建议定期导出重要对话。

---

## 更新日志

### v2.0.3 (当前版本)
- 划词工具栏支持拖动和四角缩放，交互区提示更友好
- 搜索引擎可自定义模板 URL，支持 `{query}` 占位符
- 修复悬浮工具栏按钮无效、缩放区域提示缺失等问题

### v2.0.2
- 修复侧边栏菜单按钮位置错误
- 改进 PDF 缩放功能（使用 CSS zoom 替代 transform）
- 实现完整的对话历史管理（加载、切换、删除）
- 升级到 pdfplumber 进行更高质量的文本提取
- 新增表格自动识别和格式化
- 优化新对话按钮逻辑

### v2.0.1
- 修复 PDF 上传后无法预览的问题
- 解决 PDF.js worker 版本不匹配
- 移除重复的工具栏
- 优化启动脚本，添加自动更新和浏览器跳转

### v2.0.0
- 全新蓝白治愈系 UI 设计
- 升级到 Vite 6.0 + React 18.3
- 支持深色模式
- 集成多个 AI 提供商
- 添加截图和视觉分析功能

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

如需贡献代码：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- GitHub Issues: https://github.com/yourusername/ChatPDF/issues
- 项目主页: https://github.com/yourusername/ChatPDF

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star 支持一下！**

Made with ❤️ by ChatPDF Team

</div>
