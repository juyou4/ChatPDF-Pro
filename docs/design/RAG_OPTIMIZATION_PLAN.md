# Chatpdf RAG 优化方案 v2

> 目标不是“再堆功能”，而是把检索链路整理成可路由、可回退、可验收的工程方案。

## 1. 总结

- 默认主链路保持低延迟，核心目标是“先可用，再增强，再实验”。
- 仓库里已有很多可复用能力，文档必须明确哪些是“已存在待接线”，哪些是“实验性”，哪些才是真新增。
- 本版方案保留现有引文展示和流式输出机制，不改前端大结构。
- Agentic / GraphRAG / RAPTOR 只作为 opt-in 能力，不能默认叠加进主路径。

## 2. 现状映射表

| 模块 | 状态 | 说明 |
|---|---|---|
| `embedding_service.py` | 已存在 / 主链路 | chunk 索引、HyDE、多查询扩展、Parent-Child、BM25/RRF、chunk importance、retrieval progress、timing 已具备 |
| `bm25_service.py` + `synonym_service.py` | 已存在 / 主链路 | 中文 BM25、同义词扩展、细粒度 token 化已具备 |
| `table_aware_service.py` | 已存在 / 解析阶段 | 表格感知解析已接入文档提取链路 |
| `sentence_window_splitter.py` | 已存在 / 待接线 | 句子窗口工具存在，但默认检索路径仍未全面启用 |
| `raptor_service.py` | 已存在 / 实验性 | RAPTOR 聚类与树状摘要能力已实现，默认不启用 |
| `graph_service.py` | 已存在 / 待接线 | GraphRAG 基础框架已存在，但默认检索路由仍需显式接入 |
| `retrieval_agent.py` + `retrieval_tools.py` | 已存在 / 已接线 | Agentic 多轮检索工具链已具备，需保持 opt-in |
| `answer_critic_service.py` | 已存在 / 实验性 | 答案自审可用，但默认关闭 |
| `citation_service.py` | 已存在 / 主链路 | 结构化引文、兜底匹配、答案后处理已在使用 |

## 3. 查询路由矩阵

### 3.1 Overview

- 目标：快速给出全文级总结，不让重检索拖慢首包。
- 默认路径：`fast_overview_context -> numbered citations -> answer generation`
- 默认禁用：HyDE、query expansion、LLM rerank、Agentic、GraphRAG、RAPTOR
- 适用场景：`总结 / 概述 / 主要内容 / overview / abstract`

### 3.2 Extraction

- 目标：精确提取定义、数字、结论、公式、表格。
- 默认路径：`vector search + BM25 + citation alignment`
- 可选增强：table-aware、sentence window、chunk importance、LLM rerank
- 默认禁用：Agentic、RAPTOR、GraphRAG（除非单独打开）

### 3.3 Analytical

- 目标：比较、因果、推理、跨段综合。
- 默认路径：`query analysis -> vector search -> sub-question retrieval -> citation merge`
- 可选增强：HyDE、query expansion、LLM rerank、answer critic
- 默认禁用：Agentic / RAPTOR / GraphRAG 默认不叠加

### 3.4 Specific

- 目标：具体问题、局部定位、关键词命中。
- 默认路径：`BM25 + vector search + selected text fallback`
- 可选增强：chunk importance、sentence window、table-aware
- 默认禁用：重型聚类、图谱、Agentic

## 4. 重型能力互斥规则

- 同一请求最多启用一个重型增强。
- 建议的互斥集合：
  - `HyDE`
  - `query expansion`
  - `LLM rerank`
  - `RAPTOR`
  - `GraphRAG`
  - `Agentic RAG`
- 默认策略：
  - 主路径只允许“低成本增强”
  - 实验能力必须显式开关
  - 任一阶段失败立即回退到稳定主路径

## 5. 索引时 vs 查询时边界

- **索引时完成**
  - 表格感知解析
  - Sentence Window 切分
  - RAPTOR 聚类/摘要
  - GraphRAG 实体关系抽取
  - chunk importance 标注

- **查询时完成**
  - query rewrite
  - query analysis
  - BM25 / vector / RRF 融合
  - rerank
  - 引文对齐
  - Agentic 多轮工具调用

- 原则：
  - 任何会显著拉长首包的计算，不允许默认放在查询时现算。
  - 查询时的重型能力必须带超时和回退。

## 6. 预算与回退

- 查询改写：超时后直接回退原始问题，不阻塞回答。
- 问题分解：默认短超时，超时则跳过分解。
- 向量检索：失败后回退到全文编号段落。
- Rerank：失败后回退原始排序。
- Agentic：任一轮失败后回退到已收集上下文。
- Answer Critic：默认关闭；打开后只做附加诊断，不阻塞回答。
- GraphRAG / RAPTOR：必须可单独关闭。

## 7. 推荐优先级

### P0

- BM25 同义词扩展
- table-aware chunk
- chunk importance 加权
- 引文精确化 / 连续编号收口
- 现有 HyDE / query expansion 的路由收敛

### P1

- 双模型策略
- LLM rerank
- 答案自审

### P2

- RAPTOR
- GraphRAG
- Agentic RAG
- 多引擎 PDF 解析

## 8. 验收标准

- 检索质量：
  - `MRR@10`
  - `Recall@10`
  - `NDCG@10`
- 引文质量：
  - `Citation Precision@1`
  - unsupported citation rate
  - topical relevance
- 性能：
  - `p95` 检索延迟
  - 首包时间
  - 流式首字节响应时间

## 9. 测试计划

- 检索评估：`backend/tests/eval_retrieval.py`
- 引文评估：`backend/tests/eval_citation_quality.py`
- 耗时属性测试：`backend/tests/test_timings_properties.py`
- 回归样本：
  - 概览类
  - 抽取类
  - 分析类
  - 表格类
  - 长文类

- 失败场景覆盖：
  - 无 reranker
  - 无 UMAP / sklearn
  - 无 API key
  - 检索超时
  - query rewrite 超时
  - 表格解析失败

## 10. 当前落地状态

- 已经可用：
  - 主链路检索、引文、流式输出
  - query rewrite / fast overview / answer critic 的基础骨架
  - BM25 同义词、table-aware、chunk importance、GraphRAG / RAPTOR / Agentic 的底层服务
- 仍需继续完善：
  - `sentence_window_splitter` 的默认接线策略
  - GraphRAG 的默认路由接入
  - RAPTOR 的路由开关与缓存策略
  - Agentic / Web Search / GraphRAG 的互斥与预算控制

## 11. 约束

- 默认优先复用现有服务，只有缺口才写真新增。
- 不改变现有引文展示和流式输出 UI。
- 所有实验能力必须有默认关闭、超时和回退。
- 文档是工程实施计划，不是宣传页。
