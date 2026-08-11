# 检索候选决策快照

`retrieval_meta.decision_snapshot` 是一次问答检索的内部诊断快照。它由
`backend/services/retrieval_decision_snapshot.py` 生成，不参与召回、RRF、重排、
阈值过滤或 Token 预算决策，因此打开或关闭诊断不会改变回答结果。

## 记录内容

每个候选有稳定的 `candidate_id`，并带有页码、`chunk_id`、`block_id`、意群、
章节路径和来源类型。`stages` 按实际发生顺序记录：

- `vector_recall`、`bm25_recall`：各召回源的排名和分数
- `rrf`、`semantic_group`：混合与意群融合后的取舍
- `threshold`、`dedupe`：质量阈值、结构噪声和类型策略
- `rerank` / `final_rank`：重排、分数下限和 top-k 裁决
- `page_scope`：意图页码范围裁决
- `token_budget`、`context`：是否进入 LLM 实际上下文
- `citation`、`claim_support`：最终授权引用和 claim 支持状态

每个阶段都包含 `included`、`reason`，可据此解释候选为何进入或退出。一次
回答由 `retrieval_run_id` 标识；流式多查询和重试合并时保留
`retrieval_run_ids`，候选和阶段事件按 ID 幂等去重。

## 解析身份

快照绑定向量索引中的 `route`、`generation`、`source_hash`。完整绑定为
`identity_status=bound`；缺少任意字段为 `unavailable`；请求身份与当前索引不一致
为 `mismatch`。快照不会绕过既有的 MinerU/local 索引准入检查，也不会静默接受旧代索引。

## 隐私边界

快照只保存结构化定位信息和数值。候选原文仅用于在内存中计算 `text_hash`，不会写入
快照；API Key、URL、请求/响应正文、PDF 路径和历史文档内容均不得进入快照。
默认聊天响应不暴露完整候选列表。开发者显式请求 evidence raw 调试包时，仍通过
`sanitize_candidate_snapshot` 输出无原文、无 source hash 的结构化视图。

现有 citation authorization ledger 仍是引用授权的唯一来源。快照只接受最终已授权的
引用和 claim binding 回写，不以“检索到过”或正则匹配代替授权。
