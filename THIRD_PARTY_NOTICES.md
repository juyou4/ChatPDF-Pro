# 第三方声明（Third-Party Notices）

## 设计灵感来源

### Paper Burner X

ChatPDF 的 RAG 系统优化方案借鉴了 Paper Burner X 项目的设计理念。

- **项目地址**：https://github.com/Feather-2/paper-burner-x
- **许可证**：AGPL-3.0
- **版权**：Copyright (C) 2024-2025 Feather-2 and contributors

**借鉴的设计概念**（非代码）：
- 语义意群（Semantic Groups）——将文本分块聚合为语义完整单元的架构思想
- 三层粒度体系（summary/digest/full）——多粒度文本表示的设计模式
- 智能粒度选择——根据查询类型动态选择内容详细程度的策略
- 混合粒度检索——按相关性排名分配不同粒度的检索方法
- Token 预算管理——语言感知的上下文长度控制策略

**重要说明**：
- ChatPDF 的所有实现代码均为独立编写的 Python 代码
- 未复制、翻译或移植 Paper Burner X 的 JavaScript 源代码
- 借鉴的是公开的设计概念和算法思想，这些不受版权保护（Ideas vs Expression 原则）
- ChatPDF 继续使用 MIT 许可证，与 Paper Burner X 的 AGPL-3.0 许可证无冲突

**技术分析文档**：
- 详细的技术对比分析见 `PAPER_BURNER_X_LEARNING.md`（包含代码片段引用，仅用于学习参考）

---

## 依赖库许可证

ChatPDF 使用的主要开源依赖及其许可证：

### 后端（Python）
| 库 | 许可证 |
|---|--------|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| pdfplumber | MIT |
| PyMuPDF | AGPL-3.0 |
| PyPDF2 | BSD-3-Clause |
| FAISS (faiss-cpu) | MIT |
| Sentence Transformers | Apache-2.0 |
| LangChain | MIT |
| openai | Apache-2.0 |
| hypothesis (测试) | MPL-2.0 |

### 前端（JavaScript）
| 库 | 许可证 |
|---|--------|
| React | MIT |
| Vite | MIT |
| Tailwind CSS | MIT |
| mermaid | MIT |

> 注意：PyMuPDF 使用 AGPL-3.0 许可证。如果 ChatPDF 作为网络服务部署，
> 需要遵守 AGPL-3.0 的源代码公开要求（Section 13）。
