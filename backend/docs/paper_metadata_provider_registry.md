# 论文元数据 Provider Registry

ChatPDF 的论文元数据补全是来源提示层，不是文档事实判断层。它不会改变
MinerU/local 的解析身份、`generation`、`source_hash` 或 `block_index`，也不会
把期刊等级、被引量、撤稿信号或新颖性评分交给回答 claim verifier 使用。

## Provider 合同

每个来源通过 `ProviderSpec` 注册，包含以下字段：

- `name`、`capabilities`、`supported_fields`
- `requires_key`、`enabled_by_default`
- `timeout`、`rate_limit`、`priority`
- `fetch`、`normalize`
- `provenance`、脱敏 `diagnostics`

`fetch` 只接收短元数据上下文和共用的 `httpx.AsyncClient`；不会把 PDF 原文、
请求正文或凭据写入 provider 记录。`normalize` 只输出标题、作者、年份、DOI、
开放获取提示等有限字段。

## 默认与可选来源

默认 hydration 保留 Crossref 和 Semantic Scholar，并继续保持整体 hydration
默认关闭。提供 Unpaywall 联系邮箱后才会启用 Unpaywall。OpenAlex、arXiv 和
OpenReview 通过以下环境变量显式启用，默认关闭：

```text
CHATPDF_ENABLE_PAPER_METADATA_OPENALEX=false
CHATPDF_ENABLE_PAPER_METADATA_ARXIV=false
CHATPDF_ENABLE_PAPER_METADATA_OPENREVIEW=false
```

可选来源只在基础字段（默认 `title`、`authors`、`year`）仍缺失时运行。来源按
优先级分层并行；同一层内不会因为单个 provider 失败而阻塞其他 provider。

## Fail-open 与诊断

超时、网络错误、解析错误、HTTP 429 限流和 provider 配置错误都转成脱敏的
`providers` 诊断。HTTP 429 会明确标为 `rate_limited`，但仍保持 fail-open：
本地元数据可继续使用。

```json
{
  "status": "failed",
  "error": "ConnectError",
  "provider": {
    "name": "crossref",
    "priority": 20,
    "supported_fields": ["title", "authors", "year"]
  }
}
```

本地启发式元数据始终保留。外部来源全部失败时 hydration 返回
`status=unavailable`，但不会让文档读取、解析或对话失败。

## 缓存身份

hydration 结果包含 `cache_identity`，其摘要绑定：

```text
registry_version + hydration_version + parse_generation + source_hash
+ provider_names + credential_presence + required_fields
```

只记录凭据是否存在，不记录 API Key、邮箱或完整 URL。文档代际、源哈希、provider
开关或所需字段发生变化时，旧 hydration 结果不会命中。路由在复用持久化结果前
会比较该身份；旧版本没有身份时会安全地重新补全。

## 可信度边界

`field_provenance` 只用于展示字段来自本地还是哪个外部来源。撤稿结果保留为
`retraction.evidence` 来源信号，并附带“无信号不等于确认未撤稿”的提示。任何
外部 metadata 都不能覆盖 MinerU 正文块，也不能参与回答事实真伪、引用授权或
claim verifier 的证据判定。

## 测试

`backend/tests/test_paper_metadata_provider_registry.py` 覆盖：

- Provider 合同、优先级和凭据要求；
- 缓存身份绑定与密钥脱敏；
- 字段满足时跳过可选来源；
- 外部失败时保留本地 metadata；
- OpenAlex 可选 provider 的规范化和 provenance。
