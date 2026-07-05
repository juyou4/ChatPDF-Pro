# 沉浸式论文阅读方案（大纲树 · 双语对照 · 笔记锚定 · 图表拆解 · 三栏布局）

> 对标产品形态：小绿鲸类学术阅读器的五大能力。
> 设计参考：PaperQuay（`E:\Project\PaperQuay`，**AGPL-3.0，只借设计、代码必须独立实现**，上线前在 THIRD_PARTY_NOTICES.md 加条目）。
> 本文所有参考行号均已逐文件核实（2026-07-02）。

---

## 0. 五大能力 × 现状差距总览

| # | 对标能力 | 我们已有的 | 缺的 | 差距等级 |
|---|---------|-----------|------|---------|
| 1 | 智能摘要与大纲 | 速览五卡（概述/术语/速读/图表/总结）已覆盖"要点提炼"全部子项 | **侧边栏大纲树**（层级结构 + 点击跳转） | 小 |
| 2 | 段落级双语对照 | 划词翻译、cheap model 通道、glossary 术语库 | **块级 bbox 索引 + hover/pin 翻译浮层 + 全文预翻译** | 中 |
| 3 | 笔记与原文高亮联动 | 引用跳转已实现"文本匹配→高亮矩形"（PDFViewer.jsx:705-725） | **段落级 AI 笔记生成 + 双向锚定**（笔记↔原文互相高亮） | 中 |
| 4 | 图表步骤化解析 | YOLO 裁图管线 + VLM 图表解读（速览卡）已缓存 | **hover 浮层 + 步骤化结构输出 + mermaid 重绘** | 小 |
| 5 | 三栏布局 | 左侧 320px 侧栏已存在（ChatPDF.jsx:1019，现放会话历史）、右侧速览/对话 Tab | **左栏加"大纲"Tab + 右栏加"解析"Tab + 滚动联动** | 小 |

结论：**没有一项需要从零造轮子**。最重的公共依赖只有一个——块级 bbox 索引（Phase A），其余四个能力都消费它。

---

## 1. 总体架构

```
┌─ 左栏（已有侧栏扩展）─┬─ 中栏 PDF ────────────┬─ 右栏（已有 Tab 扩展）─┐
│ Tab1: 会话历史(现有)   │ react-pdf 渲染         │ Tab1: 速览(现有)        │
│ Tab2: 大纲树(新)       │ + BlockOverlay 层(新)  │ Tab2: 对话(现有)        │
│  - 章节层级            │   - hover 黄色高亮     │ Tab3: 段落解析(新)      │
│  - 当前位置跟随        │   - 点击→翻译浮层      │  - 当前视口段落的       │
│  - 点击跳转            │   - 图表 hover→解读    │    翻译/总结卡片        │
│                        │ + 笔记锚点高亮(扩展)   │  - 卡片↔原文双向高亮   │
└────────────────────────┴───────────────────────┴────────────────────────┘
                    ▲ 全部消费同一份 blocks.json
后端：
  data/blocks/{doc_id}.json        块级索引（段落/标题/图/表 bbox + 章节归属）
  data/translations/{doc_id}.json  块级译文缓存
  data/notes/{doc_id}.json         AI 笔记（要点 + 原文锚点）
```

---

## Phase A：块级索引 + 大纲树 + 三栏骨架（3-4 天，公共地基）

### A1. 后端块索引

**落点**：解析管线已在调 `page.get_text("dict")`（`backend/routes/document_routes.py:1022-1025`），当场就有 block bbox，只是丢弃了。新增 `backend/services/block_index_service.py`：

- 解析时持久化 `data/blocks/{doc_id}.json`：
  ```json
  {
    "version": 1,
    "pages": [{
      "page": 1, "width_pts": 612.0, "height_pts": 792.0,
      "blocks": [
        {"block_id": "p1_b0", "type": "heading|paragraph|figure|table|caption",
         "bbox": [x0, y0, x1, y1], "text": "...", "section_id": "s1"}
      ]
    }],
    "outline": [
      {"section_id": "s1", "title": "1 Introduction", "level": 1,
       "page": 1, "first_block": "p1_b3"}
    ]
  }
  ```
- 块类型合并三个来源：文本块（get_text dict）、`logical_figures`（`figure_extraction.py:42,63` 已缓存 body_bbox/full_bbox）、表格 bbox（PyMuPDF `find_tables()` 已在用）
- **段落聚合**：PyMuPDF 的 raw block 在双栏论文里会切碎/粘连，复用分块管线现有的断行修复启发式（README 描述的"自适应坐标阈值 + 断行单词修复"就在解析函数里）做二次聚合
- **大纲提取**三级降级：`doc.get_toc()`（PDF 书签，出版社 PDF 有）→ 标题启发式（字号/加粗 span 检测，`semantic_group_service.py` 已有标题边界检测逻辑可抽取复用）→ 首轮速览时让 cheap model 顺带输出章节树（成本摊薄为零）
- API：`GET /documents/{doc_id}/blocks`（一次全量，gzip 后 <200KB，无需分页）

### A2. 前端 BlockOverlay

**参考（PaperQuay，设计级）**：
- 透明层结构：`E:\Project\PaperQuay\src\features\pdf\PdfPageOverlay.tsx`（全文 153 行）——每页一个 `pointer-events-none` 绝对定位层，每 block 一个 div，hover 时加 amber 高亮（对标截图黄色区域），bbox 按"原始页尺寸→渲染尺寸"缩放（`bboxToCssStyle`）
- **命中测试在 viewer 级而不是 div 级**：`PdfViewer.tsx:2050-2086`——统一 `pointermove` 监听 + `requestAnimationFrame` 节流，避免每页几十个 div 各自接事件。这是它 2596 行 viewer 里最值得借的性能细节

**落点**：`frontend/src/components/PDFViewer.jsx` 已有 scale 感知的高亮矩形层（`highlightRects`，705-725 行），BlockOverlay 作为兄弟层加入；坐标换算复用引用跳转已踩平的 page points → viewport 逻辑。

### A3. 三栏骨架

- 左栏：现有 320px 侧栏（`ChatPDF.jsx:1019`）顶部加 Tab 切换「会话 / 大纲」；大纲树消费 blocks.json 的 outline 字段
- 滚动跟随：中栏可见页+滚动位置 → 算出当前 section → 大纲树高亮该节（PDFViewer 已有 onPageChange 回调，扩展为含滚动比例）
- 右栏：Tab 行加「解析」占位（Phase B/C 填充）

---

## Phase B：段落级双语对照（2-3 天）

### B1. 翻译浮层（hover/点击即时翻译）

**参考（PaperQuay，设计级）**：
- 浮层交互链：`E:\Project\PaperQuay\src\features\reader\DocumentReaderTab.tsx:1340-1400`——点击 block → 查 `blockTranslations[blockId]` 缓存 → 有则直接弹层（锚定 block 的 clientRect，带 placement 上/下判断），无则显示"点击立即翻译"
- 全文预翻译状态机：`useDocumentTranslation.ts`——进度（completed/total）、可取消（AbortController）、缓存 map、目标语言切换全套状态管理

**落点**：
- 后端：`POST /documents/{doc_id}/blocks/translate`，body `{block_ids: [...], target_lang}`，批量并发（复用 cheap model 通道），写 `data/translations/{doc_id}.json`；全文模式走 SSE 回报进度
- 前端：hover 停留 500ms 或点击触发；浮层组件新建（现有划词工具栏的弹层定位逻辑可参考自家代码）
- 预取策略：视口页 ±1 页的块后台预翻译，hover 时基本必中缓存

### B2. 术语一致性（我们的差异化，PaperQuay 没有）

`glossary_service.py` 已有 Trie 匹配器（`find_matches`，90 行起）。翻译 prompt 注入该文档速览术语卡 + 用户词库的命中条目：

```
术语表（必须按此翻译）：attention residual→注意力残差；...
```

同一篇论文里术语翻译前后一致——这直接对标"专业术语优化"能力，且几乎零成本。

### B3. 右栏"解析"Tab 的对照视图

当前视口内各段落的「译文 + 一句话总结」卡片流，随滚动更新；卡片 hover 时中栏对应 block 高亮（消费 Phase A 的 blockId 通道，反向联动在 Phase C 统一做）。

---

## Phase C：AI 笔记与原文双向锚定（2-3 天）

对标截图里"1. 视觉3D检测重要但易受对抗攻击"这类边注——要点绑定到原文具体句子。

### C1. 笔记生成（带锚点的结构化输出）

- 新增 `POST /documents/{doc_id}/notes/generate?scope=section&section_id=s2`
- prompt 要求输出 JSON：`[{"point": "要点中文", "quote": "原文精确引句(≤30词)"}]`
- **锚点解析复用引用跳转的文本匹配**：`PDFViewer.jsx` 里 highlight 计算（248-585 行）已实现 normalize + startPhrase/endPhrase 匹配到 text layer 矩形——后端只需存 quote，前端用同一套逻辑解析成 rects。**不需要新造 span 定位**
- 写 `data/notes/{doc_id}.json`，速览生成时可顺带跑（成本摊薄）

### C2. 双向联动

- 右栏笔记卡 hover → 中栏 quote 高亮（复用 citation 高亮通道，换个颜色）
- 中栏 block hover → 右栏滚动到关联笔记卡并加边框
- 数据结构上就是 `note.quote ↔ block_id` 的双向 map，Phase A 索引建好后是纯前端工作

---

## Phase D：图表步骤化解析 + hover 浮层（1-2 天）

### D1. 图表 hover 浮层

blocks.json 里 figure/table 块进 BlockOverlay，hover 弹层直接展示**速览已缓存的 AI 图表解读**（`overview_service.py` figure analysis 结果已持久化）+ caption 译文。零新增 LLM 成本。

### D2. 步骤化拆解（升级现有图表解读 prompt）

现有 VLM 调用链（`overview_service.py:1284-1310`，`image_data_list` 格式）不动，只升级 prompt 为结构化输出：

```json
{"overview": "一句话", "stages": [{"name": "阶段名", "desc": "该阶段做什么"}], "data_flow": "A→B→C"}
```

浮层/解析卡按编号步骤渲染。可选加分项：「重绘为流程图」按钮——把 stages 转 mermaid 喂给 `StreamingMarkdown.jsx`（mermaid 渲染现成），架构图秒变可缩放流程图，这是对标产品没有的。

---

## Phase E：打磨（1-2 天，可后置）

- 大纲树键盘导航（↑↓ 换节、Enter 跳转）
- 双语导出：blocks + translations 拼装为对照 Markdown 下载
- 阅读位置记忆（localStorage per doc_id）
- 全部功能挂 GlobalSettings「阅读增强」分区 feature flag，默认渐进开启

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| PyMuPDF block 粒度糙（双栏切碎/粘连） | 复用分块管线断行修复启发式二次聚合；扫描件走 MinerU OCR 时直接用其 content_list block（`ocr_result` 已有该结构） |
| arXiv 论文常无 PDF 书签 | 三级降级：get_toc → 标题启发式 → 速览顺带生成；用 5 篇典型 arXiv 论文做大纲质量验收集 |
| 全文翻译 token 成本 | 视口 ±1 页预取 + blockId 缓存永不重译 + 全文模式显式用户触发（带进度和取消） |
| 浮层遮挡阅读 | hover 500ms 延迟 + 浮层 placement 自动上下翻转 + 全局开关（对标 PaperQuay 的独立开关设计） |
| AGPL 合规 | 只参考交互设计与数据流形态，实现代码全部独立编写；THIRD_PARTY_NOTICES.md 加 PaperQuay 条目（同 paper-burner-x 先例） |

---

## 工期与依赖总表

| Phase | 内容 | 工期 | 依赖 |
|-------|------|------|------|
| A | 块索引 + BlockOverlay + 大纲树 + 三栏骨架 | 3-4 天 | 无 |
| B | 双语对照（浮层 + 预翻译 + 术语一致性） | 2-3 天 | A |
| C | AI 笔记双向锚定 | 2-3 天 | A |
| D | 图表步骤化解析 + hover | 1-2 天 | A |
| E | 打磨 | 1-2 天 | B/C/D |

**合计约 2 周**。B、C、D 在 A 完成后可并行。与 agent 提升方案（`docs/agent-improvement-plan.md`）完全正交：那条线改检索/生成，这条线改阅读交互，唯一交集是 D2 与 agent 方案二共用 VLM 调用链（互不阻塞）。

## 建议的版本切分

- **v3.1.0**：Phase A + B（大纲树 + 双语对照）——对标能力 1/2/5，用户感知最强
- **v3.1.1**：Phase C + D（笔记锚定 + 图表拆解）——对标能力 3/4
