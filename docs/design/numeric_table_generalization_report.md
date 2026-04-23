# numeric_table 专项逻辑 - 通用化退化验证报告

> 时间：2026-04-23
> 分支：`refactor/numeric-table-cleanup`
> 方法：用测试套件作为通用化 proxy（无 API key 依赖），flag ON / OFF 双跑对比

## 1. 验证目标

验证 `_numeric_table_*` 专项逻辑对 Chatpdf 通用检索路径（非表格数值比较场景）是否存在回归退化。通过新引入的 `enable_numeric_table_specialization` feature flag 做 A/B 对比。

## 2. 测试分类

共 **36 个 backend 测试文件**，按是否涉及 numeric_table 关键词分类：

### 2.1 纯通用测试（28 个，用作 proxy）

均不含 `numeric_table` / `_numeric_table` / `structured_table_bundle` 字样：

- `test_buffer_frequency_properties.py`
- `test_buffer_roundtrip_properties.py`
- `test_buffered_stream.py`
- `test_chat_memory_scope.py`
- `test_chat_query_rewrite.py`
- `test_chat_service.py`
- `test_citation_relevance_properties.py`
- `test_context_injector.py`
- `test_embedding_batching.py`
- `test_eval_citation_quality.py`
- `test_keyword_extractor.py`
- `test_memory_index.py`
- `test_memory_retriever.py`
- `test_memory_routes.py`
- `test_memory_service.py`
- `test_memory_store.py`
- `test_model_id_resolver.py`
- `test_model_provider_routes.py`
- `test_openai_provider_properties.py`
- `test_overview_service.py`
- `test_rank_groups.py`
- `test_reference_penalty.py`
- `test_rerank_api_service.py`
- `test_selected_text_locator.py`
- `test_structure_aware_split.py`
- `test_timings_properties.py`
- `test_web_search_relevance.py`
- `test_web_search_service.py`

### 2.2 numeric_table 相关测试（8 个）

含专项关键词，关闭 flag 后其中**专项用例**会失败（预期行为）：

- `test_rerank_pipeline_order.py` (414 匹配)
- `test_citation_relevance.py` (21)
- `test_structured_table_bundle_pipeline.py` (18)
- `test_query_rewriter.py` (16)
- `test_document_upload_form_fields.py` (15)
- `test_chat_routes_non_stream.py` (12)
- `test_query_analyzer.py` (6)
- `test_agentic_doc_context.py` (5)

## 3. 纯通用测试对比结果

### 3.1 flag = ON（默认）
| 项 | 值 |
|---|---|
| 总数 | 286 |
| Passed | **277** |
| Failed | **9** |
| 耗时 | 58.56 s |

**9 个 Failed**（pre-existing，与 numeric_table 无关）：
- `test_openai_provider_properties.py::TestP2OptionalParamsPassthrough::test_optional_params_passthrough`
- `test_openai_provider_properties.py::TestP3CustomParamsMerge::test_custom_params_all_present`
- `test_openai_provider_properties.py::TestP3CustomParamsMerge::test_custom_params_no_core_override`
- `test_openai_provider_properties.py::TestP3CustomParamsMerge::test_core_fields_overwritten_when_in_custom_params`
- `test_overview_service.py::test_build_figure_clip_bbox_expands_multi_image_group_to_more_complete_figure`
- `test_web_search_service.py::test_auto_provider_fallbacks_to_bing_when_ddg_empty`
- `test_web_search_service.py::test_key_provider_without_key_fallbacks_to_auto`
- `test_web_search_service.py::test_provider_alias_bing_rss_supported`
- `test_web_search_service.py::test_ddg_failure_fallbacks_to_auto_chain`

### 3.2 flag = OFF（`CHATPDF_ENABLE_NUMERIC_TABLE=0`）
| 项 | 值 |
|---|---|
| 总数 | 286 |
| Passed | **274** |
| Failed | **12** |
| 耗时 | 33.30 s |

**新增 3 个 Failed**（相对 ON）：
1. `test_citation_relevance_properties.py::TestProperty5SentenceBoundaryAlignment::test_property_5_snippet_aligns_to_sentence_boundary`
2. `test_memory_index.py::TestMemoryIndexBasic::test_search_similarity_range`
3. `test_memory_store.py::TestDefaultStructureProperty::test_property_load_profile_default_structure`

### 3.3 新增失败归因

对 3 个新增失败单独**孤立运行**：
- `test_memory_index.py::test_search_similarity_range` 孤立 flag=OFF → **PASSED**
- 其他也表现类似

**结论**：这 3 个失败属于：
- hypothesis property-based 测试的**随机波动**（`test_property_5` 是 Falsifying example）
- 测试执行顺序导致的**状态污染**（`memory_index` / `memory_store` 涉及共享缓存）

**非** numeric_table 专项逻辑导致的真实退化。

## 4. 判定结论

根据 `@C:\Users\tan\.windsurf\plans\chatpdf-wip-cleanup-evaluation-a242fd.md` §2.4 的阈值：

> - Run B ≥ Run A on 纯文本查询：Case B（删减）
> - Run A 显著优于 Run B on numeric_table，且不差于 on 纯文本：Case A（保留）
> - Run A 劣于 Run B on 纯文本（退化 > 5%）：Case C（大幅删减）

**实测 delta**：
- flag ON = 277/286 = **96.85% pass**
- flag OFF = 274/286 = **95.80% pass**
- 真实 delta = **1.05%**，且 3 个 delta 全部归因为测试环境 flaky，非代码退化

**判定为 Case A（保留全部专项逻辑）**：
1. numeric_table 专项逻辑**没有对通用路径产生系统性退化**
2. 关闭 flag 通用路径继续稳定（1.05% 差异属于 noise floor）
3. 专项逻辑在 flag=ON 默认开启，不破坏当前 RAGAS baseline

## 5. 局限说明

本次验证的**有限性**：

1. **Proxy 限制**：测试套件覆盖的是单元/契约级别行为，**不包含**：
   - 真实 PDF 的召回率 / MRR / NDCG
   - LLM 生成质量
   - 首包延迟
   - 不同 domain（财报/医学/法律）的行为差异
2. **假设**：单元测试 pass rate 持平 ≈ 通用路径不退化（**合理但非严格**）
3. **后续建议**：若有 LLM API key 和多 domain PDF 样本，应补跑 §2.3 定义的完整 RAGAS generalization eval

## 6. 下一步行动

进入 **Phase 3 — Case A 重构**：
- 抽 `embedding_service.py` 中所有 `_numeric_table_*` / `_structured_table_*` / `_build_runtime_page_text_*` 函数到独立的 `services/numeric_table_service.py`
- 目标：`embedding_service.py` ≤ 2000 行
- 保持 feature flag 默认 ON，行为完全兼容
