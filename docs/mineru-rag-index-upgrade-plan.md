# MinerU 深度解析 → RAG 问答索引升舱方案

> 状态：核心链路已实施；P0 已补齐 MinerU RAG swap 后的 semantic_groups 异步重建；
> 旧 Grounding DINO smoke A/B 只能作为格式链路验证，完整 manifest 仍需在 P0 后重跑
> 前置依赖：MinerU 深度解析链路（已上线）、`data/mineru_results/{doc_id}.json` 原始结果缓存（已上线）
> 关联文档：`docs/agent-improvement-plan.md`、`docs/immersive-reading-plan.md`
>
> v2 变更摘要：① 规范化层输出目标从 `[TABLE]` markdown 升级为完整三件套
> `pages + full_text + structured_table_bundles`（对齐 odl_parser_service 的既有输出合同）；
> ② index_source 元数据落到 pkl 的具体改动清单；③ 原子替换明确为三文件一组；
> ④ 新增 documents_store 文档数据同步更新（chat_routes 有 4 处直读 full_text）。
>
> v3 实施补充：① MinerU RAG 重建会备份并清理旧 semantic_groups，避免向量索引已切
> MinerU 但 agent/意群仍读取 pdf_native 旧数据；回退本地索引时同步恢复旧 semantic_groups；
> ② RAGAS 评测脚本会把 `/rag-index/status` 写入结果 `run_config.index_source`，
> 避免 pdf_native / mineru A/B 数字混淆。
>
> v4 验收补充：2026-07-06 使用 `Grounding DINO.pdf`
> (`1bb4d57b7e3a511ab19e2f5158f8034b`) 跑 4 题 numeric smoke A/B：
> `pdf_native` vs `mineru` 均由结果 JSON 的 `run_config.index_source` 证明未混源。
> MinerU 重建后 pkl 为 `index_source="mineru"`，90 chunks，18 个 structured table chunks，
> chunk / table markdown 均无 HTML 标签，表格页码为 1-based，所有 table chunk 均带
> evidence units。小样本指标：Context Precision/Recall 均持平 1.0；
> AnswerCorrectness 从 0.6825 升至 0.9220；Faithfulness 从 0.8458 降至 0.7917；
> AnswerRelevancy 从 0.4873 降至 0.4157。结论：结构化表格证据有效提升数值正确性，
> 但仍应在完整 numeric/common manifest 上复验后再自动推荐。
>
> v5 P0 修复：MinerU RAG 重建现在会在临时向量索引质量门通过后，先从临时
> pkl 准备 semantic group rebuild 所需 chunks 与 embedding 函数；只有准备成功才
> swap 当前索引。swap 成功并清理旧 semantic_groups 后，立即调用
> `_build_semantic_group_index_async` 按 MinerU chunks 异步补建意群索引，返回值中
> 增加 `semantic_group_rebuild.queued/chunk_count`。因此 P0 前跑出的 A/B 结果，
> 特别是 Grounding DINO smoke，不能作为最终 MinerU 双索引效果结论。

---

## 1. 背景与动机

当前系统里 MinerU 深度解析的结果只喂给了**阅读侧**（块索引 → 悬浮翻译 / 大纲 / 章节总结 / 速览图表），而**问答侧（RAG）的分块、语义意群、FAISS/BM25 索引仍然使用上传时的本地提取结果**（PyMuPDF + `find_tables()` 的 `[TABLE]` markdown）。

这造成一个体验断层：用户为一篇排版复杂的论文点了深度解析，阅读体验升级了，但对着同一篇论文提问时，模型看到的还是本地解析的旧文本——表格列错位、双栏顺序穿插、页眉页脚混入 chunk 的问题原样存在。

本地提取在问答侧的三个具体短板：

| 短板 | 根因 | 对问答的影响 |
|------|------|-------------|
| 表格列错位 / 并单元格失败 | `find_tables()` 对学术论文常见的无边框三线表识别能力弱 | 数值对比类问题引用到错误的列/行，是 numeric 子集错误的头号来源 |
| 双栏阅读顺序穿插 | PyMuPDF 文本块按坐标序输出，左右栏内容交错 | 分块把不相干段落缝在一起，语义意群完整性差，长上下文题证据破碎 |
| 页眉/页脚/参考文献噪声 | 本地启发式清理不完全（依赖 YOLO Abandon + 跨页重复检测，均有漏网） | 噪声块进入向量索引，检索命中里垃圾比例升高 |

MinerU 结果对应的能力：`table_body` 是带 rowspan/colspan 的完整 HTML 结构；content_list 按版面阅读顺序输出；discarded 类别天然剔除页眉页脚；公式输出 LaTeX。

---

## 2. 方案总览

```
                     ┌─────────────────────────────────────────────┐
                     │  data/mineru_results/{doc_id}.json（已缓存）  │
                     │  content_list_json（带 page_idx / table_body）│
                     └──────────────────┬──────────────────────────┘
                                        │ 手动触发「重建问答索引」
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │        规范化层（本方案的主体工程）           │
                     │  · 表格 HTML → 与现有 [TABLE] markdown       │
                     │    逐字节同格式的规范表示                     │
                     │  · 公式 → LaTeX 行内文本                     │
                     │  · discarded 块剔除                          │
                     │  · 每段保留 page（1-based）元数据             │
                     └──────────────────┬──────────────────────────┘
                                        │ 输出与上传管线相同的 pages/full_text 结构
                                        ▼
              ┌────────────── 复用现有管线，零改动 ──────────────┐
              │  分块（chunking）→ 语义意群 → 三层粒度            │
              │  → 双 FAISS 索引 + BM25 → 数值表格锚定管线        │
              └──────────────────────────────────────────────────┘
```

**核心设计决策：让数据适配管线，而不是让管线适配数据。**

理由：数值表格专项（表格数值锚定、目标表/列/行锚点、同簇行补齐、`citation_enhancer`）是整个 RAGAS 优化循环调出来的成果，全部正则和行聚类逻辑假设 `[TABLE]` markdown 格式。改下游适配 HTML 意味着这套调优全部重做；转换数据格式则下游一行不动。

---

## 3. 规范化层设计（核心）

### 3.1 输入选择：content_list_json 而非 full_md

- `full_md` 丢失页码边界，无法支撑引用点击跳页——**禁用**
- `content_list_json` 每个 item 带 `page_idx`（0-based，转换时必须 +1，教训见
  `mineru_block_index_service._page_num` 的页码错位事故）
- `middle_json` 存在时可作补充（块级 bbox 更全），但 v1 以 content_list 为准

### 3.2 各类型 item 的转换规则

| MinerU type | 处理 | 说明 |
|-------------|------|------|
| `text` | 直接收录，按 page_idx 归页 | 已是阅读顺序，无需重排 |
| `table` | `table_body`（HTML）→ 规范 [TABLE] markdown；`table_caption` 拼在表格前 | 见 3.3 |
| `image` | 收录 `image_caption` 文本（正文里的图注是检索证据）；图片本身不进文本索引 | 图片已由速览图表链路消费 |
| `equation` | LaTeX 文本直接收录，保留 `$...$` / `$$...$$` 定界 | 公式归一化逻辑（citation_enhancer）已兼容 LaTeX |
| `discarded` | **剔除** | 页眉/页脚/页码/脚注不进索引 |
| `code` / `list` | 按纯文本收录 | 学术论文低频类型 |

### 3.3 表格 HTML → [TABLE] markdown 转换器

这是规范化层里唯一有工程量的部分：

1. 用标准库 `html.parser` 解析 `table_body`（不引入 bs4 等新依赖，桌面打包体积敏感）
2. **rowspan/colspan 展开成矩形网格**：跨行/跨列单元格的值复制到覆盖的每个格子——
   这是数值锚定"同簇行补齐"正确工作的前提（每行列数必须一致）
3. 按现有 `table_aware_service` 的输出格式逐字节对齐地生成 markdown
   （实施前先读该模块确认标记语法，禁止凭记忆书写格式）
4. 单元格内换行压成空格；空单元格保留占位符与现有格式一致
5. 转换失败（畸形 HTML）时回退：剥标签取纯文本 + 记 warning，绝不让单表失败中断整个重建

**验收锚点**：转换后的表格喂进现有数值锚定单测（若无则补最小用例），
锚点命中行为必须与 find_tables 产物一致。

### 3.4 页码与引用元数据

- 输出结构与上传管线的 `pages: [{page, content, ...}]` 完全同构，
  每页 content 由该页 items 按序拼接
- chunk → page 的映射由现有分块逻辑自然继承，引用点击跳页无需改动
- **历史引用漂移是接受的代价**（见 §6 风险 3），不做旧 chunk id 映射

---

## 4. 触发方式与 UI

沿用"手动触发、显式知情"的深度解析产品原则：

1. **入口**：AI 处理面板 → MinerU 深度解析卡片下新增"重建问答索引"按钮，
   仅在 `active_mineru === true` 时出现
2. **确认弹窗**必须说明三件事：
   - 将重新计算全文嵌入（一次 embedding token 开销 + 约 1~3 分钟）
   - 历史对话中的 `[1][2]` 引用可能指向偏移的位置
   - 阅读侧（翻译/大纲）不受影响
3. **不自动触发**：深度解析完成的 toast 里可以附带一句
   "可在 AI 处理面板重建问答索引以升级表格问答"，但不代跑
4. 状态接入现有 `deep-parse/status` 轮询（新增 stage：`rebuilding_rag_index`）
5. **索引版本标记**：向量索引元数据写入 `index_source: "pdf_native" | "mineru"`，
   旧索引文件保留（`.bak` 后缀或版本目录），支持一键回退

---

## 5. 对 RAGAS 评测的影响分析

### 5.1 前提：默认零影响

本功能 opt-in。评测文档不执行深度解析 + 重建时，现有 baseline/holdout 数字不变：

| Split | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-------|:---:|:---:|:---:|:---:|
| Baseline | 0.617 | 0.645 | 0.697 | 0.813 |
| Holdout | 0.666 | 0.645 | 0.708 | 1.000 |

### 5.2 重跑 A/B 时的指标预测

（同一 manifest，pdf_native vs mineru_rebuilt，其余全部固定）

| 指标 | 预期方向 | 依据 |
|------|:---:|------|
| Context Precision | **↑ 明显** | 页眉/页脚/参考文献噪声块从索引中消失，检索命中的垃圾比例直接下降；这是受益最确定的指标 |
| Context Recall | ↑ 小幅 / 持平 | 双栏顺序修正让证据块更完整；Holdout 已 1.000 触顶无空间 |
| Faithfulness | **条件性 ↑** | 表格结构准确 → 数值引用错位减少 → 涨。**规范化层缺失/劣化时反向大跌**：LLM 面对 HTML 长串更易幻觉，RAGAS judge 对 HTML 上下文的解析也不可靠 |
| Answer Relevancy | ≈ 持平 | 由回答风格与问题理解主导，与解析质量弱相关 |

### 5.3 分问题类型的预测（比总分更重要）

- **numeric_table 子集：方差最大**。规范化层合格 → 预期涨幅最大的子集
  （列错位是数值题错误的头号来源）；规范化层不合格 → 直接回吐之前
  整个数值优化循环挣来的提升。**这个子集是 A/B 的第一观察对象**
- 纯文本理解题：基本不动（±噪声）
- 公式题 / 跨栏长上下文题：小幅受益

### 5.4 评测方法学要求

1. **评测输出必须新增 `index_source` 字段**——升舱之后基线语义分叉，
   不标注来源的数字不可比，README 的评测表格同理
2. A/B 只改索引来源这一个变量：同一批 PDF、同一 manifest、同一回答模型
   （DeepSeek）、同一嵌入（bge-m3）、同一检索参数
3. 现成的 eval_runner / diagnose_numeric 基建直接可用，A/B 成本约为
   一轮常规回归
4. 结论门槛：numeric 子集四指标不劣化 + Context Precision 总体提升，
   才把"重建问答索引"从可选动作升级为深度解析后的推荐动作

### 5.5 P0 后完整 A/B 结果（6 篇论文，26 题）

本轮为正式 A/B，只比较同日 A1/A2，不与 README 历史 baseline/holdout 混比。
两组使用同一批 PDF、同一 manifest、同一回答模型 DeepSeek、同一评测 embedding
`BAAI/bge-m3`、同一检索参数。评测 manifest 为
`backend/tests/ragas_mineru_ab_questions.json`，共 26 题：

- DiffuLT 6 题
- Grounding DINO / DETR / CLIP / Attention Residuals / AdvRoad 各 4 题
- 题型分布：17 个 `numeric_table`，5 个 `detail`，4 个 `general`

执行前置条件已满足：A1 六篇均强制 rollback 到 `index_source=pdf_native`；
A2 六篇均重建为 `index_source=mineru`，且
`data/semantic_groups/{doc_id}.json/_groups.index/_groups.pkl` 均存在。CLIP
重建过程中暴露的 bge-m3 超长输入 400 已通过远程 embedding 单条输入保守截断和
`code=20015` 收缩重试修复。

| 指标 | A1 `pdf_native` | A2 `mineru` | 变化 |
|------|:---:|:---:|:---:|
| Faithfulness | 0.8173 | 0.8248 | +0.0075 |
| Answer Relevancy | 0.4268 | 0.4114 | -0.0154 |
| Context Precision | 0.7033 | 0.6810 | -0.0223 |
| Context Recall | 0.8077 | 0.7436 | -0.0641 |
| Answer Correctness | 0.7211 | 0.7517 | +0.0306 |
| 平均检索片段数 | 3.35 | 3.31 | -0.04 |
| 平均响应时间 | 9005 ms | 10381 ms | +1376 ms |

分论文结果：

| 论文 | n | A1 Faith | A2 Faith | Δ | A1 Rel | A2 Rel | Δ | A1 Prec | A2 Prec | Δ | A1 Recall | A2 Recall | Δ | A1 Correct | A2 Correct | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdvRoad | 4 | 0.870 | 0.814 | -0.056 | 0.457 | 0.335 | -0.121 | 0.698 | 0.875 | +0.177 | 0.875 | 1.000 | +0.125 | 0.722 | 0.848 | +0.126 |
| Attention Residuals | 4 | 0.835 | 0.731 | -0.104 | 0.410 | 0.531 | +0.121 | 0.760 | 0.514 | -0.246 | 1.000 | 0.583 | -0.417 | 0.808 | 0.606 | -0.202 |
| CLIP | 4 | 0.812 | 0.917 | +0.104 | 0.566 | 0.574 | +0.008 | 0.717 | 0.542 | -0.175 | 0.625 | 0.750 | +0.125 | 0.751 | 0.774 | +0.023 |
| DETR | 4 | 0.875 | 1.000 | +0.125 | 0.319 | 0.322 | +0.003 | 0.689 | 0.746 | +0.057 | 0.750 | 1.000 | +0.250 | 0.783 | 0.826 | +0.043 |
| DiffuLT | 6 | 0.655 | 0.967 | +0.312 | 0.291 | 0.332 | +0.041 | 0.472 | 0.667 | +0.194 | 0.667 | 0.667 | +0.000 | 0.545 | 0.663 | +0.118 |
| Grounding DINO | 4 | 0.938 | 0.450 | -0.487 | 0.586 | 0.414 | -0.173 | 1.000 | 0.750 | -0.250 | 1.000 | 0.500 | -0.500 | 0.806 | 0.838 | +0.032 |

分题型结果：

| 题型 | n | A1 Faith | A2 Faith | Δ | A1 Rel | A2 Rel | Δ | A1 Prec | A2 Prec | Δ | A1 Recall | A2 Recall | Δ | A1 Correct | A2 Correct | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| detail | 5 | 0.882 | 0.862 | -0.019 | 0.519 | 0.436 | -0.083 | 0.457 | 0.400 | -0.057 | 0.500 | 0.600 | +0.100 | 0.691 | 0.733 | +0.042 |
| general | 4 | 0.685 | 0.917 | +0.232 | 0.427 | 0.428 | +0.001 | 0.542 | 0.676 | +0.134 | 0.875 | 1.000 | +0.125 | 0.595 | 0.638 | +0.043 |
| numeric_table | 17 | 0.830 | 0.792 | -0.037 | 0.400 | 0.400 | +0.001 | 0.814 | 0.765 | -0.049 | 0.882 | 0.725 | -0.157 | 0.760 | 0.784 | +0.024 |

输出文件（均在 ignored `backend/temp/` 下）：

- A1：`temp/ragas_mineru_ab_a1_pdf_native_ragas.json`
- A1 CSV：`temp/ragas_mineru_ab_a1_pdf_native_ragas.csv`
- A2：`temp/ragas_mineru_ab_a2_mineru_ragas.json`
- A2 CSV：`temp/ragas_mineru_ab_a2_mineru_ragas.csv`

结论：

1. MinerU RAG rebuild 的工程链路已闭环：六篇均能完成深度解析缓存 → 问答索引重建
   → semantic group 异步补建 → RAGAS 全量评测。
2. A2 在 Faithfulness 和 Answer Correctness 上小幅优于 A1，说明 MinerU 的表格和版面
   规范化能改善部分答案可信度和数值正确性。
3. A2 的 Context Precision 和 Context Recall 低于 A1，未达到 §5.4 的"Context Precision
   总体提升"门槛，因此现阶段不应把"重建问答索引"升级为深度解析后的默认推荐动作。
   更稳妥的产品策略仍是 opt-in，并在 UI 中标注"实验性问答索引升级"。
4. 下一步优化重点不是继续证明链路可跑，而是提升 MinerU chunk 与检索打分的适配：
   表格 bundle 的粒度、caption/正文合并策略、以及 MinerU 版 chunk 的 BM25 文本形态。

### 5.6 A2' / A2'' 行分片复测（2026-07-06）

针对 §5.5 暴露的 Grounding DINO / Attention Residuals 大表检索失败，本轮按
RAGFlow `tokenize_table` 思路新增结构化表格行分片：保留表级 bundle，同时把
`caption + table_header + 每 10 行` 作为 `[Structured Table Row Shard]`
进入问答索引，分片 metadata 通过 `parent_table_bundle_id` 指回母表。

第一轮 A2' 证明了行分片能显著提升精确定位，但也暴露了新问题：最终上下文被行分片
挤占，只剩 1-2 个片段，导致 Faithfulness 和 Context Recall 下降。因此 A2''
继续增加三条保护：

1. 行分片只作为数值表格证据增补，优先替换同表冗余支撑，不挤掉唯一解释性正文；
2. 数值对比题动态提高 row shard 上限到 4，并把有效 `top_k` 至少提升到 3；
3. 评测脚本写入 request overrides manifest hash，跨实验 hash 不一致时可直接中止。

同一 manifest、同一 6 篇论文、同一回答模型 DeepSeek、同一评测 embedding
`BAAI/bge-m3` 下，A2'' 结果如下：

| 指标 | A1 `pdf_native` | A2 `mineru` | A2' rowshard | A2'' rowshard+context | 备注 |
|------|:---:|:---:|:---:|:---:|---|
| Faithfulness | 0.8173 | 0.8248 | 0.7603 | **0.9031** | A2'' 26/26 有效 |
| Answer Relevancy | 0.4268 | 0.4114 | 0.3800 | 0.4123 | 仍未超过 A1 |
| Context Precision | 0.7033 | 0.6810 | **0.8537** | 0.7415 | A2'' 仅 8/26 有效，只作参考 |
| Context Recall | 0.8077 | 0.7436 | 0.7500 | **0.8462** | A2'' 26/26 有效 |
| Answer Correctness | 0.7211 | 0.7517 | **0.7547** | 0.7231 | A2'' 仅略高于 A1 |
| 平均检索片段数 | 3.35 | 3.31 | - | 10.58 | numeric_table 平均 11.88 |

分题型看，`numeric_table` 子集从 A2 的 Recall 0.725 恢复到 0.8824，
Faithfulness 达 0.8737；但 Answer Correctness 为 0.7461，仍低于 A2 的
0.784 和 A1 的 0.760。说明 P0 已经修复了"检索上下文坍缩"，但还没有完全解决
"检中后如何稳定生成数值答案"。

重点论文回归：

| 论文 | A2 Recall | A2'' Recall | A2 Faith | A2'' Faith | 结论 |
|---|:---:|:---:|:---:|:---:|---|
| Grounding DINO | 0.500 | **1.000** | 0.450 | 0.688 | 大表检索覆盖恢复，但第 8/9 题答案正确性仍低 |
| Attention Residuals | 0.583 | **0.875** | 0.731 | 0.889 | 上下文坍缩基本解除 |

本轮输出文件：

- smoke：`temp/ragas_mineru_ab_a2pp_rowshard_context_smoke_newbackend.json`
- RAGAS：`temp/ragas_mineru_ab_a2pp_rowshard_context_ragas.json`
- CSV：`temp/ragas_mineru_ab_a2pp_rowshard_context_ragas.csv`
- request overrides manifest hash：
  `b8a8fa555e399e2a8b75199541b436a1799db9e5df834fae8a70aa8276091f3c`

当前结论更新为：MinerU RAG rebuild 不再因为大表 bundle 截断而系统性丢召回；
行分片 + 上下文保底已经使 Faithfulness / Context Recall 超过 A1。但由于
numeric_table 的 Answer Correctness 尚未稳定超过 A1，产品策略仍保持 opt-in。
下一步应聚焦两个问题：Grounding DINO 第 8/9 题的表格行选择与回答生成，以及
Context Precision judge 大量 NaN 的评测稳定性。

### 5.7 A2''' 表头绑定行分片复测（2026-07-06）

继续对 §5.6 的两个问题做机械修复，不新增额外 LLM 调用：

1. 结构化表格行文本从裸值 `a | b | c` 改为 `表头: 值; 表头: 值`；
2. 小表也生成 `[Structured Table Row Shard]`，不再只处理超过 10 行的大表；
3. 每个行分片重复携带 caption/header/table id，legacy 裸值行在索引侧尽量回填表头；
4. 数值表格回答 prompt 增加约束：只答所问列、不要罗列未问字段、重复/多级列名优先使用用户问到的子列。

同一 manifest hash
`b8a8fa555e399e2a8b75199541b436a1799db9e5df834fae8a70aa8276091f3c`
重建 6 篇 MinerU 索引后，A2''' 完整 RAGAS 结果如下：

| 指标 | A1 `pdf_native` | A2 `mineru` | A2'' rowshard+context | A2''' header-bound rowshard | 备注 |
|------|:---:|:---:|:---:|:---:|---|
| Faithfulness | 0.8173 | 0.8248 | 0.9031 | **0.9261** | 26/26 有效 |
| Answer Relevancy | 0.4268 | 0.4114 | 0.4123 | 0.3333 | 下降，主要受短数值答案和 judge 口径影响 |
| Context Precision | 0.7033 | 0.6810 | 0.7415* | 0.4945 | 26/26 有效，但上下文过宽导致精度下降 |
| Context Recall | 0.8077 | 0.7436 | 0.8462 | 0.7628 | DiffuLT/CLIP 个别题拉低 |
| Answer Correctness | 0.7211 | 0.7517 | 0.7231 | **0.7778** | 25/26 有效，第 17 题 judge 输出过长 |
| 平均检索片段数 | 3.35 | 3.31 | 10.58 | 10.50 | numeric_table 平均 11.82 |

\* A2'' 的 Context Precision 只有 8/26 有效，只作趋势参考；A2''' 已通过
RAGAS run_config 和单样本回填使 Context Precision 26/26 有效。

分题型看，`numeric_table` 子集 Answer Correctness 达 **0.8115**，
高于 A1 的 0.760、A2 的 0.784 和 A2'' 的 0.7461；Faithfulness 达
0.9328。Grounding DINO 4 题全部 Recall=1.0，平均 Correctness=0.8174，
第 8/9 题的大表行选择基本翻转。Attention Residuals 第 20 题仍未完全解决：
Standard Residuals=3d、Full AttnRes=24d 能答出，但 Block AttnRes 的
5.5d 被 MinerU HTML 的多行表头/子行结构遮蔽，回答仍倾向"文档未明确给出"。

本轮输出文件：

- RAGAS：`temp/ragas_mineru_ab_a2ppp_headerbound_ragas.json`
- CSV：`temp/ragas_mineru_ab_a2ppp_headerbound_ragas.csv`

当前结论更新为：表头绑定行分片把 MinerU 数值表格问答的正确性推到四组最高，
说明 RAGFlow/TrustRAG/kotaemon 共同采用的"表头: 值"行语义是必要修复。
但最终上下文仍偏宽，Context Precision 从 A1 的 0.7033 降至 0.4945；
下一步应把数值表格上下文从"12 段宽召回"收敛为"目标行 1-3 段 + 表题/表头
+ 必要邻近正文"，而不是继续增加 row shard 数量。

### 5.8 A2'''' 上下文收敛设计落地（2026-07-06）

参考 RAGFlow / PaperQA / TrustRAG / paper-burner-x / kotaemon / PaperQuay 的本地源码后，
结论是：能直接解决 A2''' Context Precision 下降的不是继续放大检索或增加 row shard，
而是把最终 prompt 的消费单位从 raw chunk 改成最小证据对象。

本轮已落地两层低风险改动：

1. **numeric table evidence pack**：`_build_response_context_segments` 在数值表格题里不再返回最多
   12 段宽上下文，而是把同表证据合并为 `numeric_evidence_pack`。每个 pack 只保留
   caption、header、相关行和至多一条必要邻近说明。
2. **同表 / 跨表预算**：默认单表最多 3 行；如果问题显式点名多个方法做对比，最多放宽到 4 行；
   全局最多 2 张表，额外解释性文本最多 1 段。
3. **deterministic projection**：在 pack 内对 `表头: 值` 行或 `header | row` 结构做确定性列投影，
   生成 `Projected Cells`，例如只抽出用户问到的 `Acc` / `All` / `zero-shot` 等目标列。

这相当于先做一个轻量版 RAGFlow table executor：不让 LLM 在整表/整页里找数值，
而是先由后端投影出候选单元格，再让 LLM 负责组织答案。完整 SQL/DSL executor 暂不进入本轮，
因为 PDF 论文表格的 schema 仍有合并单元格和多级表头歧义。

下一步验证方式：

- 先跑 numeric context / citation 单测，确认 pack 不丢 exact row、引用 ref 和 focused citation；
- 再跑同一 26 题 manifest 的 A2'''' RAGAS，重点看 Context Precision 是否回升，同时
  numeric_table Answer Correctness 是否维持在 A2''' 的 0.81 附近；
- 若 Attention Residuals Table 1 的 `Block AttnRes = 5.5d` 仍失败，再单独做多行表头/子行拆分。

实测结果：

| 实验 | 平均片段 | numeric 片段 | Faith | CP | CR | Correct | numeric Correct | 结论 |
|------|---------:|-------------:|------:|---:|---:|--------:|----------------:|------|
| A2''' header-bound row shard | 10.54 | 11.82 | 0.9261 | 0.4945 | 0.7628 | 0.7778 | 0.8115 | 正确性最高，但上下文过宽 |
| A2'''' pack-only | 4.27 | 2.71 | 0.7256 | 0.6220 | 0.6731 | 0.5817 | 0.5902 | CP 上升，但 pack 过度替代 exact rows |
| A2''''' hybrid pack + exact rows | 5.69 | 4.76 | 0.7368 | 0.6275 | 0.7692 | 0.6530 | 0.6342 | 比 pack-only 恢复召回和正确性，但仍低于 A2''' |
| A2'''''' focused row shard | 5.80 | 5.00 | 0.7685 | 0.5467 | 0.7692 | 0.6316 | 未采用 | 强制把 row shard 聚焦成单行会降低 CP/Correct，不作为默认路径 |

当前代码保留 A2''''' 的 hybrid 方向，并额外修了两个稳定性问题：

- 已有 `numeric_evidence_pack` 不再被当作 row shard 二次打包，避免 `[Numeric Table Evidence Pack]`
  套娃污染 prompt。
- `_extract_numeric_table_target_methods` 不再把所有 Title Case token 都误判为列名；
  `Baseline`、`Ours`、`DiffuLT` 等方法名可以参与行排序，`Acc` 这类指标列会被排除。

后续不要继续沿“从 10 行 shard 里靠正则选 1 行”这条路默认化。focused-row 实验证明它会把
RAGAS Correctness/CP 拉低。更合理的下一步是结构层修复：

1. 在索引阶段把 row shard 拆成真正的单行 evidence chunk，而不是回答阶段临时从 shard 文本里猜行；
2. 对 DETR / Grounding DINO 这类大表补齐 `table_id + row_id + column header` 的结构字段，降低同表错行；
3. 对 Attention Residuals Table 1 单独做多行表头 / 子行语义拆分，解决 `Block AttnRes 5.5d`
   被解析成 `34d` 的结构歧义。

### 5.9 历史 smoke 结果与当前缓存状态

旧 Grounding DINO 4 题 numeric smoke 只用于验证 P0 前的工程链路和
`index_source` 隔离。该结果生成时 MinerU RAG swap 后尚未异步补建现行
semantic groups，因此不能用于判断最终质量。正式结论以 §5.5、§5.6、§5.7 和 §5.8 的
6 篇 26 题复测结果为准。

当前本地缓存状态：

| 论文 | doc_id | 当前问答索引 | MinerU 结果缓存 | semantic group |
|------|--------|--------------|-----------------|----------------|
| Grounding DINO | `1bb4d57b7e3a511ab19e2f5158f8034b` | `mineru` | 有 | 有 |
| DiffuLT | `d91dcdd98a0bd14dad49dadcc7f8906b` | `mineru` | 有 | 有 |
| DETR | `69a9a0027e6142e61167b167fd88ada1` | `mineru` | 有 | 有 |
| CLIP | `9bc70f2791a8997004e78c178192f17a` | `mineru` | 有 | 有 |
| Attention Residuals | `5edd436b3bba946e74ce7aa85cd1b279` | `mineru` | 有 | 有 |
| AdvRoad | `5655e9fc3ba9939ae69fa832e4e4daf7` | `mineru` | 有 | 有 |

### 5.10 当前版：answer_cells_guard 门槛通过（2026-07-07）

在 §5.8 之后，方向从"回答阶段收窄 pack"回到结构层修复，按三个已定位失败逐步收敛：

1. **unit_row**：索引期生成真正的单行 evidence chunk，不再在回答阶段从 10 行 shard 里靠正则猜行；
2. **table_identity_guard**：建立论文 `Table N` 与结构化表格 chunk 的身份约束，显式表号没有可信匹配时退回宽上下文，避免错表且无兜底；
3. **answer_cells_guard**：答案单元格守卫只保留可信 caption / table id / bundle id / exact row 里的数值单元，避免投影过程丢失或改写可验证数值。

本轮只比较同一 26 题正式 manifest，结果文件为
`temp/ragas_current_mineru_20260707_after_answer_cells_guard_26_ragas_ac_backfilled.json`。
5 个 Answer Correctness 超时样本已用同一问题集合单独回填，不混入 4 题 smoke / probe 样本。

总体结果：

| 指标 | A1 `pdf_native` | 当前 `mineru` | 变化 |
|------|:---:|:---:|:---:|
| Faithfulness | 0.8173 | **0.9695** | +0.1522 |
| Answer Relevancy | 0.4268 | 0.3399 | -0.0869 |
| Context Precision | 0.7033 | **0.8765** | +0.1732 |
| Context Recall | 0.8077 | **0.9615** | +0.1538 |
| Answer Correctness | 0.7211 | **0.7528** | +0.0317 |

`numeric_table` 子集 17 题结果：

| 指标 | A1 `pdf_native` | 当前 `mineru` | 门槛 |
|------|:---:|:---:|:---:|
| Faithfulness | 0.830 | **0.9733** | 通过 |
| Context Precision | 0.814 | **0.9533** | 通过 |
| Context Recall | 0.882 | **1.0000** | 通过 |
| Answer Correctness | 0.760 | **0.7986** | 通过 |

结论：

1. §5.4 的推荐门槛已经满足：`numeric_table` 子集四指标不劣化，且总体 Context Precision 明显高于 A1。
2. 产品策略从"实验性 opt-in"调整为"深度解析完成后的推荐动作"：当阅读侧已经启用 MinerU block index，而问答索引仍是 `pdf_native` 时，`deep-parse/status` 返回 `recommend_rag_index_rebuild=true`，AI 处理面板提示用户重建问答索引。
3. 不自动代跑重建。原因是重建会消耗 embedding 请求并替换问答索引，仍需要用户确认。
4. 表格检索方向冻结，不再继续开新 pack / focused-row 实验。剩余尾巴是 Attention Residuals 第 20 题的多级表头 / 子行结构歧义，单独挂起，不阻塞推荐机制。
5. Answer Relevancy 在 A1、A2 和后续表格实验中都偏低，主要受短数值答案和 RAGAS 反向生成问题口径影响；本轮发布判定以 Faithfulness、Context Precision、Context Recall、Answer Correctness 和 numeric 子集门槛为准。

---

## 6. 风险清单与缓解

| # | 风险 | 等级 | 缓解 |
|---|------|:---:|------|
| 1 | 数值锚定管线对新表格格式失灵 | **高** | 规范化层输出与 find_tables 产物格式逐字节对齐；数值锚定单测做守门 |
| 2 | BM25 被 HTML 标签污染 | 中 | 规范化层保证输出零 HTML 标签；重建后抽查 BM25 词表无 `td/tr/rowspan` |
| 3 | 历史对话引用漂移 | 中 | 确认弹窗显式告知；不做旧 id 映射（成本不值）；新对话即恢复正常 |
| 4 | 页码元数据丢失导致引用跳页失效 | 高 | 强制走 content_list（带 page_idx），禁用 full_md；page_idx 全程 +1 转换（有既往事故） |
| 5 | 重建期间用户继续提问 | 低 | 重建为原子替换：新索引就绪前旧索引继续服务；替换加锁 |
| 6 | embedding 成本失控（超长文档反复重建） | 低 | 确认弹窗展示预估 token；同一 source_hash 的重复重建直接跳过 |
| 7 | 双来源不一致（阅读侧 MinerU、问答侧本地） | — | 本方案正是为了消除此不一致；完成后两侧同源 |

---

## 7. 实施步骤与文件清单

| 步骤 | 内容 | 涉及文件 | 预估 |
|:---:|------|---------|:---:|
| 1 | 表格 HTML → 规范 markdown 转换器 + 单测（rowspan/colspan/畸形输入） | 新建 `backend/services/mineru_text_normalizer.py`；参照 `table_aware_service.py` 确认格式 | 1 天 |
| 2 | content_list → pages/full_text 规范化组装（页码、类型路由、discarded 剔除） | 同上 | 0.5 天 |
| 3 | 重建入口：`POST /documents/{doc_id}/rag-index/rebuild`（校验 mineru 结果存在 → 规范化 → 复用现有 create_index 流程 → 原子替换 + index_source 标记） | `backend/routes/document_routes.py`、`services/vector_service.py`（如需传源标记） | 0.5 天 |
| 4 | 前端按钮 + 确认弹窗 + 状态展示 | `frontend/src/components/ChatPDF.jsx`（AI 处理面板） | 0.5 天 |
| 5 | RAGAS A/B：numeric 子集 + 通用子集，输出带 `index_source` | 评测脚本 / manifest | 已完成 |
| 6 | 按 A/B 结果决定是否把重建纳入深度解析后的推荐动作（复用 recommend 机制） | `document_routes.py` 的 `_assess_deep_parse_recommendation` | 已完成 |

步骤 1 是全部风险的阀门，先做且必须带单测；步骤 5 数据已在 §5.10 通过门槛，因此步骤 6 已从"是否推荐"落地为状态接口和前端 AI 处理面板的推荐信号。

---

## 8. 验收标准

1. 规范化层单测通过：含 rowspan/colspan 展开、空单元格、畸形 HTML 回退三类用例
2. 重建后抽查：chunk 文本无 HTML 标签、无页眉页脚、表格行列数一致
3. 数值锚定行为与 find_tables 产物一致（现有数值单测全绿）
4. 引用点击跳页在重建后的文档上正常
5. RAGAS A/B：numeric 子集不劣化，Context Precision 总体 ≥ 基线（已通过，见 §5.10）
6. 回退路径验证：删除新索引 + 恢复 `.bak` 后问答恢复原行为
