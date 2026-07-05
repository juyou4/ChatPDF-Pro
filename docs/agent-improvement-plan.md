# ChatPDF Pro Agent 能力提升方案

> 基于 2026-07-02 对工作区内 paper-qa / m3docrag / agentic-rag-for-dummies / TrustRAG / RAGFlow / kotaemon 的源码对比分析。
> 所有引用均已逐文件核实，非二手转述。

---

## 0. 现状基线

当前 agent（`backend/services/retrieval_agent.py`，2829 行）的能力边界：

| 能力 | 现状 | 代码位置 |
|------|------|---------|
| 工具集 | 7 个：vector_search / keyword_search / grep / regex_search / boolean_search / fetch / map | `retrieval_tool_schemas.py:13-118` |
| 规划循环 | 最多 3 轮、12 次工具调用、并发 5；支持原生 tool-calling 与 JSON plan 双模式 | `retrieval_agent.py:516-560,1334-1418` |
| 子问题分解 | decompose + 树形递归检索 + 覆盖度追踪 | `tree_decomposition_retrieval.py`、`retrieval_agent.py:230-287` |
| 充分性判断 | **启发式**：≥2000 字符 + ≥2 来源 | `retrieval_agent.py:552-553,1803` |
| 上下文管理 | 16k 阈值压缩、检索去重、child→parent 意群回填 | `retrieval_agent.py:1606,1744` |
| 证据模态 | **纯文本**（图表只能检索到 caption 文字） | — |
| 文档范围 | **单文档**（DocContext 锁死单 doc_id） | `retrieval_tools.py` |
| Web 检索 | 服务存在但**不在 agent 工具层**（仅聊天级开关） | `web_search_service.py:32-64` |

与内部 research loop 的边界：round 49/50 已排定的 numeric_table 修复（P0 exact-evidence packer → P1 comparator packer）**继续按原计划走**，本方案不与其抢占；方案一显式设计了 exact evidence 直通通道以保护 round-48 grounding。

---

## 方案一：证据级 LLM 评分（RCS：Retrieve → Compress → Score）

**优先级 1 · 预估 2-3 天 · 验收指标：RAGAS answer_relevancy / context_precision**

### 问题

当前"证据够不够"的判断是字符数启发式（`retrieval_agent.py:552-553`：`sufficiency_threshold_chars=2000` + `sufficiency_min_sources=2`）。凑够字数 ≠ 证据相关，低相关证据会稀释上下文、拉低 answer_relevancy——这正是内部 loop round 48 以来 answer_relevancy 停在 0.61 的结构性原因之一。

### 参考实现（paper-qa）

paper-qa 的核心竞争力就是这一层，照抄骨架即可：

1. **打分 prompt**：`E:\Project\paper-qa\src\paperqa\prompts.py:107-118`（`summary_json_system_prompt`）
   - 每条证据输出 JSON `{"summary": "...", "relevance_score": 0-10}`
   - 关键细节：**不相关时要求 summary 留空 + score=0**，而不是让模型硬写——这是防幻觉设计
2. **单条证据处理器**：`E:\Project\paper-qa\src\paperqa\core.py:178-397`（`_map_fxn_summary` / `map_fxn_summary`）
   - JSON 解析失败自动带上错误重试一次（`core.py:389-397`）
   - 摘要长度可配（`{summary_length}` 模板变量）
3. **并发调度**：`E:\Project\paper-qa\src\paperqa\docs.py:492-560`（`aget_evidence`）
   - 先取 `evidence_k` 条候选（settings.py:109，默认 10），`gather_with_concurrency` 并发打分，按 score 排序
4. **双阈值设计**：`E:\Project\paper-qa\src\paperqa\settings.py:109,138`
   - `evidence_k`（进打分的候选数）与 `answer_max_sources`（进最终上下文的条数）分离——检索宽、消费窄
5. **打分结果回传 planner**：`E:\Project\paper-qa\src\paperqa\agents\tools.py:217-313`（`GatherEvidence.gather_evidence`）
   - 把 top-n 高分证据摘要直接写进工具返回值，planner 下一轮规划时知道"已经拿到什么、还缺什么"——对齐我们现有的 `_format_uncovered_subquestion_hint`（`retrieval_agent.py:436`）

### 落地设计

- 新增 `backend/services/evidence_scorer.py`：输入 (question, evidence_list)，用 cheap model（复用 `deps.get_cheap_model_params`，`agent_retrieval_service.py:230`）并发打分，并发上限对齐 `max_tool_concurrency=5`
- 接入点 A：`retrieval_agent.py::_merge_tool_result`（1651 行）后挂钩，每轮工具结果合并后打分
- 接入点 B：`_assess_sufficiency`（1803 行）改为 score 加权：`sum(score≥7 的证据) ≥ N` 替代纯字符数
- 分层消费：score<4 丢弃；4-6 只保留摘要文本；≥7 保留原文全文（与现有 token 预算 `max_context_tokens=12000` 联动）

### 保护性约束（必须遵守）

- **exact evidence 直通**：`table_row` / `table_cell` / cost-anchor 命中（round-48 grounding 主链）**不经过打分**，直接进 primary slot——打分只作用于意群/chunk 级证据
- 失败降级：打分调用超时（建议 8s）→ 整体回退现有启发式，不阻塞回答
- 候选 <3 条时跳过打分（省一次 LLM 往返）
- 上线前跑既有 RAGAS numeric_subset manifest，numeric_table 指标不得回退

---

## 方案二：图表证据工具 `fetch_figure`（agent 多模态）

**优先级 2 · 预估 3-4 天 · 验收：图表问答人工评测集**

### 问题

用户问"Figure 3 展示了什么趋势"时，agent 只能 grep 到 caption 文本，图的实际内容不可见。m3docrag 的 README（`E:\Project\m3docrag\README.md:14`）对这个问题定义得很准：*"documents often have important information in visual elements such as figures, but text extraction tools ignore them"*——但它的方案（ColPali 页面视觉向量 + 本地 Qwen2-VL 7B，README:79-86）对桌面产品太重，**只借它的问题定义，不抄实现**。

### 我们的地基（全部现成，这是性价比最高的原因）

| 组件 | 位置 | 状态 |
|------|------|------|
| 图表定位缓存 | `figure_extraction.py:42,63`——`logical_figures` 已持久化在 doc data | ✅ 现成 |
| 裁图渲染 | `figure_render.py::render_figure`（本次 v3.0.2 刚加了 `render_mode` 参数）+ DocLayout-YOLO 收紧 | ✅ 现成 |
| VLM 调用格式 | `overview_service.py:1284-1310`——`_generate_figure_analysis_via_pipeline` 已在用 `image_data_list=[data:image/jpeg;base64,...]` 调多模态模型 | ✅ 现成 |
| provider 图片支持 | `gemini_provider.py:80-81` 已处理 `image_url` content | ⚠️ 仅 Gemini |

### 参考实现

- **多模态打分 prompt**：`E:\Project\paper-qa\src\paperqa\prompts.py:119-135`（`summary_json_multimodal_system_prompt`）
  - 比纯文本版多一个 `used_images` 布尔字段，并有一条注释级细节值得照搬：*即使图片被用于判定"不相关"，`used_images` 也应为 true*——用于事后审计 VLM 是否真的看了图
  - 输出格式与方案一统一（summary + relevance_score），两个方案共用消费端

### 落地设计

1. `retrieval_tool_schemas.py` 新增两个 schema：
   - `list_figures()`：返回该文档所有 figure 的 `figure_id / caption / page`（读 logical_figures 缓存，零成本）
   - `fetch_figure(figure_id, question)`：渲染裁切图 → VLM 按 multimodal prompt 输出针对 question 的 summary + score
2. `retrieval_tools.py` 实现工具体，复用 `_render_figures_with_pipeline` 与 overview 的 VLM 调用链
3. planner prompt（`retrieval_agent.py:414` `_GROUP_TOOLS_TEMPLATE` 同区域）追加使用指引："问题涉及 Figure/图/曲线/架构图/对比图时，先 list_figures 再 fetch_figure"
4. 证据回传带 `page + figure_id`，接入现有引用跳转（citation 点击回 PDF 定位）

### 分期与门槛

- **第一期**：capability gate——仅当当前 provider 支持 vision（Gemini）时暴露这两个工具；不支持时 planner 看不到它们，自然退化为 caption 文本检索
- **第二期**（+2 天）：给 `openai_provider.py` / `anthropic_provider.py` 补 `image_url` content 处理（两家 API 均原生支持，纯工程活）

---

## 方案三：澄清中断（clarification interrupt）

**优先级 3 · 预估 2 天**

### 问题

query 模糊时（"这个方法好在哪"——哪个方法？），agent 直接开跑 3 轮检索，方向错了全浪费。

### 参考实现（agentic-rag-for-dummies）

- **图结构**：`E:\Project\agentic-rag-for-dummies\project\rag_agent\graph.py:36-49`
  - `summarize_history → rewrite_query → (条件路由) → request_clarification 或 agent`
  - 关键：`interrupt_before=["request_clarification"]`（graph.py:49）——中断等用户回答，回答后**重新走 rewrite_query**，不是从头开始
- **路由判断**：`E:\Project\agentic-rag-for-dummies\project\rag_agent\edges.py:6-13`——`questionIsClear=False` → 澄清；True → `Send` 并行子问题
- **结构化输出**：`E:\Project\agentic-rag-for-dummies\project\rag_agent\nodes.py:31-44`（`rewrite_query`）
  - 用 `with_structured_output(QueryAnalysis)` 一次调用同时产出 `is_clear / questions / clarification_needed` 三个字段——**改写和澄清判断合并成一次 LLM 调用**，不加延迟
  - 细节：`clarification_needed` 长度 <10 字符时用兜底文案（nodes.py:43），防模型输出空泛反问

### 落地设计

- **不引入 LangGraph**。ChatPDF 已有 `query_rewriter` + `query_analyzer`，在 `agent_retrieval_service.py` agent 激活前扩展 query_analyzer 的输出结构：加 `is_ambiguous: bool` + `clarify_question: str`（合并进现有那次 cheap model 调用，零新增延迟）
- SSE 新增事件类型 `clarification_request`，前端渲染为提示气泡 + 可点击候选项；用户回答后走现有 query_rewriter 指代消解合并（多轮消解已支持）
- **触发门槛**（防打断体验）：仅 agent 模式 + 判定模糊 + 会话历史无法消解时触发；加 feature flag 进 GlobalSettings"检索增强调优"面板（三态开关基建现成）

---

## 方案四：web_search 进 agent 工具层

**优先级 4 · 预估 1-2 天**

### 问题

"这篇论文的方法后来有没有被改进？"——一半答案在文档内，一半在网上。当前 web 搜索是聊天级开关（要么全开要么全关），planner 无法按需规划"文档内查不到 → 查网络"。

### 参考实现

- **自家服务直接复用**：`web_search_service.py:32-64` `SearchManager.search` 已支持 Tavily/Serper/DDG/Bing/Brave/Exa/SerpAPI/Google CSE/Firecrawl 九个引擎，带自动降级
- **循环形态借 TrustRAG**：`E:\Project\TrustRAG\trustrag\modules\deepresearch\action.py:27-60,262-353`——"生成多条 SERP query → 并发搜索 → 抽取 learning → 递归缩小 breadth/depth"。只借这个形态用于 schema 描述和多 query 展开，不抄其实现（它面向 web deep research，我们只做单跳补充检索）
- **终止时声明确定性借 paper-qa**：`E:\Project\paper-qa\src\paperqa\agents\tools.py:405-440`（`Complete` 工具）——`has_successful_answer` 布尔声明 + `"Certain | Unsure"` 状态回传。我们可在 agent 最终 diagnostics 里加 `evidence_origin: doc/web/mixed`，前端引用样式区分文档引用与 web 引用（web 引用标签 UI 已有）

### 落地设计

- 新工具 `web_search(query, reason)`：schema 描述里写死使用条件——"仅当文档内证据不足，或问题明确涉及文档外信息（后续工作/横向对比/时效性内容）时使用"；`reason` 参数强制 planner 说明为什么需要出网（进 AgentTracePanel，可审计）
- web 证据独立 slot，**不与文档证据混排竞争 token 预算**，不参与 exact citation 主链
- 默认关闭，跟随现有联网搜索总开关；关闭时 planner 看不到该工具

---

## 方案五（远期）：跨文档问答

**优先级 5 · 预估 2-3 周 · 建议排 v3.2**

### 问题

"这三篇论文哪篇的 mIoU 最高？"——文献综述场景完全无法回答。这是与 paper-qa 的产品级差距，也是所有方案里工程量最大的。

### 参考实现（paper-qa）

- **库级检索工具**：`E:\Project\paper-qa\src\paperqa\agents\tools.py:109-215`（`PaperSearch`）——agent 自主决定检索词，对整个文献库检索并把新命中论文数回报给 planner
- **多文档容器**：`E:\Project\paper-qa\src\paperqa\docs.py`（`Docs` 类）——所有文档的 texts 进统一池，靠 `dockey` 溯源；`aget_evidence` 天然跨文档
- **全局状态回传**：`GatherEvidence` 返回值里的 `state.status`（paper count / evidence count / cost）——planner 每轮都知道全局进度，这是它能自主决定"继续搜还是开始答"的关键

### ChatPDF 改造面（为什么贵）

| 层 | 现状 | 需要的改动 |
|----|------|-----------|
| 索引 | FAISS 按 doc_id 分文件 | collection 级合并检索，或 fan-out + RRF 聚合 |
| 工具层 | `DocContext` 单 doc 假设遍布 `retrieval_tools.py` | 全部工具签名接受 `doc_ids: list` |
| 引用 | `[1]` 页内跳转 | 需带文档名前缀 + 跨文档跳转 |
| UI | 单文档会话 | 文献库面板 / 多文档会话模型 |

### 分两步走（降低风险）

1. **同会话多文档 fan-out**：每个 doc 独立跑现有 pipeline，新增聚合层做跨文档比较——不动索引结构，先把产品形态立起来
2. **统一索引**：确认需求后再做 collection 级 FAISS 合并

---

## 总表

| 顺序 | 方案 | 核心参考 | 工作量 | 前置依赖 | 验收 |
|------|------|---------|--------|---------|------|
| 1 | 证据级 LLM 评分 | paper-qa `core.py` / `docs.py` / `prompts.py` | 2-3 天 | 无 | RAGAS answer_relevancy ↑，numeric_table 不回退 |
| 2 | fetch_figure 图表工具 | 自家 overview 管线 + paper-qa multimodal prompt | 3-4 天（+2 天 provider 扩展） | 无（Gemini 先行） | 图表问答人工集 |
| 3 | 澄清中断 | agentic-rag-for-dummies `graph.py` / `nodes.py` | 2 天 | 无 | 模糊 query 首轮命中率 |
| 4 | web_search 工具化 | 自家 SearchManager + TrustRAG deepresearch 形态 | 1-2 天 | 无 | — |
| 5 | 跨文档问答 | paper-qa `PaperSearch` / `Docs` | 2-3 周 | 1-4 稳定后 | 文献综述评测集（需新建） |

方案 1-4 相互独立、可并行；方案 1 与 2 共用打分输出格式，建议 1 先行定格式。全部方案与内部 research loop 的 P0/P1 numeric_table 修复不冲突，方案一的 exact-evidence 直通约束是硬性要求。
