# ChatPDF 可靠性与多论文能力实施计划

## 目标与不变量

本轮目标不是更换 ChatPDF 的主架构，而是把已有能力补成可观测、可修复、可评测的闭环。

必须始终保持以下不变量：

- MinerU/local 是文档主解析身份，视觉增强只能增加补充块，不能静默切换主路线。
- 单文档检索继续使用现有混合检索底座，不建立会混淆解析身份的多文档统一大索引。
- 流式回答已经发出的正文不可被 critic 静默替换。
- 外部元数据、期刊信息、被引量、订阅兴趣和新颖性反馈不能参与事实真假判断。
- 所有后台发布都必须绑定当前解析代际与源文件哈希。

## 顺序与验收

### 1. 统一意图约束

状态：已完成。

- `IntentConstraintSet` 冻结任务标签、否定、数字、页码、普通实体和限定范围。
- 查询改写、问题分解、澄清和 Planner 工具参数使用同一约束对象。
- Planner 漂移仅在可以确定修复时回写根意图，否则拒绝漂移参数。

验收：意图回归 180/180；约束、改写、分解与工具参数测试通过。

### 2. 判定强度与校准

状态：已完成。

- 公共字段改为 `decision_strength`，前端展示“判定强度”。
- `/intent/corrections` 只记录判定身份、预测和纠错结果，不保存问题或答案原文。
- 评测输出混淆矩阵、Brier Score 和 ECE。

验收：TP=26、TN=44、FP=0、FN=0，Brier=0.085536，ECE=0.267857。

### 3. Agent 证据增量

状态：已完成。

- 每轮记录唯一证据增量、重复证据、锚点覆盖、子问题查询/证据覆盖和连续无增量轮数。
- 状态使用规范化快照和 SHA-256 哈希，可复放比较。
- 连续两轮无唯一证据和覆盖增量时以 `evidence_saturation` 停止；等待视觉证据时不提前停止。

### 4. 非流式受控修复

状态：已完成。

- critic 检测高风险后，非流式回答最多执行一次同证据、零检索修复。
- 新数字、未授权引用、保留风险 claim 或过度扩写会拒绝修复并保留原答案。
- 流式模式继续只提供诊断，不替换已发送正文。

### 5. 总结视觉预检

状态：已完成。

- 总结前只检查 Methods、Experiments/Results 和 Ablation 的高风险图表，最多 1-4 张，默认 3 张。
- 单张图失败隔离；视觉服务不可用时总结失败开放。
- 补充块发布后重新加载文档、复核解析身份和 block index，再创建总结任务。
- 总结任务身份绑定发布后的 block-index revision。

### 6. 多文档真实检索

状态：已完成。

- 主文档保留当前检索结果，伴随文档分别构造独立 `DocContext` 并调用 `search_document`。
- 每文档限额后合并，不建立统一向量索引。
- 引用使用 `doc:{doc_id}` 命名空间，并从主文档最后一个编号后续排。
- DOI、arXiv ID 或规范化标题用于版本家族去重；显式新版本优先。
- 跨文档共享锚点但数值不同的陈述标记为“潜在冲突”，不自动裁决。
- 点击伴随文档引用时先切换文档，再定位对应页。

### 7. 外部元数据补全

状态：已完成。

- 默认关闭自动联网，可通过 `CHATPDF_ENABLE_PAPER_METADATA_HYDRATION` 开启，或手动调用接口。
- Crossref 与 Semantic Scholar 同层并行，Unpaywall 仅在有 DOI 和联系邮箱时调用。
- 单 provider 失败不会丢弃本地元数据或其他 provider 结果。
- 结果包含字段 provenance、OA 信息和撤稿信号，并绑定解析代际。
- 撤稿信号只生成来源风险提示；“无信号”不表示确认未撤稿。

接口：

- `GET /documents/{doc_id}/paper-metadata/hydration`
- `POST /documents/{doc_id}/paper-metadata/hydration`

### 8. 解析适配器合同

状态：已完成。

- 通用 conformance harness 覆盖 `submit → poll → normalize → publish → invalidate`。
- 验证 provider 身份、submission 稳定性、进度回调、单次发布和单次失效。
- 取消的提交不能进入 transport；质量门失败不能触达 publisher。
- MinerU 分段传输和兼容内联传输均通过同一套合同。

### 9. 隔离论文库

状态：已完成。

- 订阅、兴趣反馈、seen cursor 和 feed 持久化在独立 `paper_library/state.json`。
- 相关性与新颖性分开存储；反馈不保存问题、回答或文档正文。
- 已上传文档可通过 `process-new` 增量扫描。
- 用户显式调用 `refresh` 时，按订阅并行查询 Crossref 与 Semantic Scholar；默认不会在聊天或上传路径联网。
- 同一 work ID 的已处理论文或旧版本不会重复进入 feed。
- 论文库模块未被聊天、RAG、critic 或可信度计算导入。

接口：

- `GET/POST /paper-library/subscriptions`
- `PATCH/DELETE /paper-library/subscriptions/{subscription_id}`
- `POST /paper-library/feedback`
- `POST /paper-library/process-new`
- `POST /paper-library/refresh`
- `GET /paper-library/feed`
- `DELETE /paper-library/data`

## 最终验证门

- 目标后端联合回归：210 passed。
- 解析适配器与 MinerU 结构回归：44 passed（包含在联合覆盖及单独验证中）。
- 意图固定集：180/180，硬门通过，无 baseline regression。
- 目标前端回归：72 passed。
- 前端生产构建：通过，5262 modules transformed。
- FastAPI 路由导入和目标接口合同：通过。
- `git diff --check`：通过；只有工作区既有的 LF/CRLF 提示。
