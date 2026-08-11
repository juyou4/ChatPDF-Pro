# ChatPDF 分层评测门槛

评测分成两个层级，避免把联网模型的不确定性混入每个 PR：

## PR / push 本地门槛

`.github/workflows/chatpdf-quality.yml` 只运行可复现的本地检查：

- Provider contract、缓存身份和 fail-open；
- 检索决策快照、引用对齐、Claim Verifier 和任务事件隐私；
- MinerU 文本/结构回归与文档任务状态；
- PDF 跨页选区、流式状态、前端 lint 和生产构建。

这些检查不需要 API Key、PDF、历史聊天记录或外部服务。

## 手动 / nightly RAGAS

`.github/workflows/chatpdf-ragas.yml` 没有 PR 触发器，只在每周定时或手动运行。
仓库不提交评测文档、答案、问题集或密钥；没有受保护的配置文件或 judge secret
时，工作流会明确跳过而不是伪造一个通过结果。

评测结果由 `backend/scripts/check_ragas_gate.py` 检查。门槛文件为
`backend/benchmarks/ragas_gate.v1.json`，当前要求：

- 至少有真实 RAGAS 评估样本；
- 总错误比例不超过 10%；
- required metric 的 `valid_count`、`total_count`、`nan_count` 可解释且无 NaN；
- Faithfulness、Context Precision、Context Recall 和 Answer Correctness 达到绝对最低分；
- 提供 baseline 时，单项回归不得超过允许幅度（默认 0.03）。

Answer Relevancy 只作为 optional metric 告警，因为历史实验已经证明其容易受短数值
答案和 judge 反向生成问题的影响。它不能被缺失或低分伪装成其它指标通过。

## 结果解释

门槛通过只表示该固定问题集在当前索引、模型和评测配置下满足最低要求，不表示
`confidence` 已完成概率校准。意图系统仍使用“判定强度”语义；Brier Score、ECE、
误澄清率和分路线 precision/recall 需要单独的数据集与校准工作。

本地验证命令：

```text
python backend/scripts/check_ragas_gate.py \
  --results backend/temp/ragas_results.json \
  --thresholds backend/benchmarks/ragas_gate.v1.json
```
