# 引用可信度回归说明

这套引用闸门解决的是“检索到了相关内容，但回答陈述与引用错绑”的问题。`query_relevance` 只用于检索证据准入，`claim_support` 只用于回答完成后的陈述级绑定，两者不能互换。

## 发布链路

```text
混合检索 / Agent 工具结果
  -> provenance authorization（Agent 仍是硬门槛）
  -> query-aware evidence admission
  -> 生成回答
  -> atomic claims
  -> claim_support top-1（确有互补信息才 top-2）
  -> 高风险 claim 选择性 verifier
  -> 受控修复或 fail-closed 改写
  -> 最终 citation_bindings + support_span
```

没有辅助模型时，确定性准入和 claim 对齐仍运行；LLM 证据选择器、claim verifier 和修复器 fail-open，不会把主模型偷偷当成辅助模型，也不会因此追加检索。
本地/无 API Key 的主模型仍会执行确定性准入；Agent 结果则完全绕过普通 selector，只使用本轮工具授权的证据。`unsupported/uncertain` 的保守改写带有无引用闸门，后续对齐不会重新挂回原引用。

## 运行指标

`backend/services/citation_quality_metrics.py` 输出以下运行指标。它们是确定性运营指标，不能替代人工标注集。

| 指标 | 含义 |
| --- | --- |
| `citation_correctness` | 已挂引用的事实陈述中，至少有一条引用达到 claim 支撑阈值的比例 |
| `citation_completeness` | 全部事实陈述中，得到合格引用支撑的比例 |
| `overcitation` | 已挂引用中未达到 claim 支撑阈值的引用比例 |
| `span_precision` | 生成的 `support_span` 能在对应证据原文中精确找到的比例 |
| `contradiction_escape_rate` | verifier 判定为 `contradicted`、但最终仍保留原引用的比例 |

指标会写入 `retrieval_meta.citation_quality_metrics`，claim 级绑定写入 `retrieval_meta.citation_bindings`。前端流式终态只保存后端最终返回的 `final_content` 和 `citation_bindings`。

## 回归命令

```powershell
python -m pytest backend/tests/test_citation_alignment_service.py `
  backend/tests/test_citation_quality_metrics.py `
  backend/tests/test_answer_critic_claim_verifier.py `
  backend/tests/test_citation_relevance.py `
  backend/tests/test_buffered_stream.py -q

cd frontend
npm run check:streaming
```

回归集覆盖反向比较、同表错行、否定反转、方法/限制混淆、同证据复用、多主题精确 span、中英证据、完全无支撑、Agent provenance、无辅助模型 fail-open 和一/二引用规则。

全量后端测试还会触及历史索引/解析测试及本机 `torch/transformers` 导入；若出现既有基线失败或进程级依赖 abort，应以本文件列出的引用定向套件和前端 streaming gate 作为本改动的可重复验证入口，并单独记录环境问题。
