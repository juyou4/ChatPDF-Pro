# 任务阶段事件账本

ChatPDF 的文档解析和下游 AI 任务仍以现有最新状态记录作为兼容主契约；事件账本只补充可重放的阶段历史，不替换状态机。

## 事件协议

事件按 `task_id` 存储在 `document_jobs/task_event_ledger/` 下，最多保留最近 200 条：

```json
{
  "task_id": "...",
  "sequence": 3,
  "stage": "publish_block_index",
  "status": "running",
  "duration_ms": 820,
  "attempt": 1,
  "error_code": "",
  "degraded_reason": "",
  "route": "mineru",
  "generation": "...",
  "source_hash": "...",
  "timestamp": 1786000000.0
}
```

阶段集合统一为：`queued`、`upload`、`submit_mineru`、`poll`、`download`、`normalize`、`publish_block_index`、`build_rag`、`publish_visual_assets`、`downstream_ai`、`ready`、`failed`、`cancelled`、`restart_recovery`。重复序号和重复事件幂等，乱序写入拒绝。

## 接入点

- MinerU worker 的上传、远端轮询、结果下载、结构整理、阅读块发布、RAG 构建、视觉资产发布和最终状态由 `document_routes._set_deep_parse_status` 追加。
- `overview`、`reading_outline`、`section_outline` 通过 `downstream_task_state` 的创建、转换和重启恢复追加。
- `/documents/{doc_id}/deep-parse/status`、`/documents/{doc_id}/ai-tasks/{purpose}` 和速览任务状态接口只附带脱敏的 `events` 与 `shortfall`。

## 隐私边界

账本只允许任务 ID、阶段、状态、顺序、耗时、尝试次数、诊断代码和解析身份。不会写入 API Key、Token、URL、文件路径、PDF 原文、模型请求/响应正文或完整异常消息。未知的自由文本原因会归一化为 `unclassified`；页面列表只保留整数页码。

## Shortfall

`shortfall` 是结构化的可行动缺口，不是模型解释正文。当前支持解析质量门失败、部分解析、总结 claim 证据不足、下游降级、重启中断和生成失败。它包含类别、代码、阶段、数量、失败页和是否可重试，前端后台任务面板会在展开详情时显示失败阶段、原因类别和已有重试动作。
