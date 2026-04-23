# AGENTS.md - ChatPDF RAG 优化编排策略

## 重要说明
- 本文件只定义编排策略与授权约束，不直接设置真实权限。
- 真实权限来自 `.codex/agents/*.toml` 与 `config.toml`。
- 若本文件与某个 agent 的 TOML 默认行为冲突，以真实 sandbox / approval 配置为准。
- 本文件修改后通常对下一次新任务 / 新会话生效，不保证立即热更新当前会话。

## 顶层路由规则
- 当用户明确说出“让 orchestrator 接管”“进入 RAG 优化流程”“按评测结果自动决定下一步”时，优先使用 `orchestrator`。
- 当用户提到以下任一类任务时，也应优先考虑使用 `orchestrator`：
  - RAGAS 评测结果分析
  - 阶段推进，例如 “当前 B 阶段”
  - 指标下降或异常，例如 `context_precision`、`answer_relevancy`、`faithfulness`
  - “帮我决定下一步修什么”
  - “按现有评测结果继续优化”
- 若用户只是单纯问代码实现细节、查看文件、解释逻辑，而不是进入 RAG 优化流程，则不必默认切到 `orchestrator`。

## orchestrator 的输入与读取规则
- `orchestrator` 接手后，应优先读取：
  - `orchestrator_status.md`
  - 最新的 `ragas_results_*.json`
  - 当前阶段相关说明文件
- 若缺少上述文件，应先报告缺失项，再决定是否继续。
- 不要在未读取状态文件和最近评测结果之前直接开始修复。

## orchestrator 的调度策略
- 当核心问题集中在 `context_precision` 时，优先路线：
  - 先派发 `diagnose-numeric`
  - 再视诊断结果派发 `fix-retrieval`
- 当核心问题集中在 `answer_relevancy` 时，优先路线：
  - 先派发 `diagnose-numeric`
  - 再视诊断结果派发 `fix-prompt`
- 当核心问题集中在 `faithfulness` 时，优先路线：
  - 优先派发 `fix-prompt`
- 修复完成后，优先派发 `eval-runner` 进行最小必要回归验证。
- 若当前流程约定是只跑 `numeric_subset`，则 `eval-runner` 默认只跑 `numeric_subset`，除非用户明确要求全量评测。

## 互斥与节奏控制
- `fix-retrieval` 和 `fix-prompt` 不得在同一轮同时执行。
- 每轮只允许修复一个核心指标，不要并行推进多个优化方向。
- 若诊断结果不足以支持进入修复阶段，应先补诊断，不要强行修改。
- 若单轮修复未改善目标指标，应先总结原因，再决定是否继续下一轮。

## 编排层权限约束
- `orchestrator` 和 `diagnose-numeric` 应被视为只读角色，不应承担代码修改任务。
- `fix-retrieval` 和 `fix-prompt` 应被视为工作区可写角色，但只允许修改其职责范围内文件。
- `eval-runner` 应被视为执行评测脚本的角色，只负责运行评测与生成结果，不负责代码修改。
- 若用户未明确授权更高风险操作，不应把修复任务升级到高权限代理。

## 输出要求
- `orchestrator` 每轮都应输出一份简洁决策摘要，至少包含：
  - 当前判断的主要问题指标
  - 本轮准备派发的代理
  - 为什么选择该代理
  - 本轮完成后的下一步验证方式
- 各 subagent 返回结果时应尽量简洁，避免把大段日志直接回灌给主会话。
- 当流程结束或暂停时，必须给出“当前状态 / 已完成动作 / 下一步建议”三段式总结。

## 禁止事项
- 不要跳过诊断直接大改检索或提示词。
- 不要在同一轮里同时改 retrieval 和 prompt。
- 不要在未查看最新评测结果时继续沿用旧结论。
- 不要让只读角色承担修改任务。
