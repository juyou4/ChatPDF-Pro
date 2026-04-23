# fix_summary

## 变更文件
- `backend/services/embedding_service.py`
- `backend/tests/test_rerank_pipeline_order.py`

## 修复点
- sparse structured bundle 回补改为 query-aware，page-text 行抽取保留 generic 解析，再用当前 query 的 numeric-table hints 重新抽一遍并覆盖 row evidence。
- 升级时优先保留更干净的 metadata caption/header，避免回退到 compact no-space 文本。
- 保护 recovered `row_text/content/row_numbers/table_focus_columns` 不被旧的 generic evidence 覆盖。
- 新增 Table 7 few-shot 回归，覆盖 `DiffuLT | Few=29.7` 的 query-shape 恢复。

## 验证
- `python -m pytest backend/tests/test_rerank_pipeline_order.py -q -k "preserves_fewshot_focus or upgrades_sparse_structured_bundle_with_recovered_rows or upgrades_existing_sparse_structured_bundle_result or finalize_without_rerank_keeps_full_comparator_bundle or finalize_without_rerank_dedupes_normalized_explicit_comparator_rows or slot_reservation_prefers_table_rows_for_best_few_query"`
- 结果：`8 passed, 90 deselected`
- `python -m pytest backend/tests/test_rerank_pipeline_order.py -q -k "preserves_fewshot_focus or upgrades_sparse_structured_bundle_with_recovered_rows or upgrades_existing_sparse_structured_bundle_result or does_not_upgrade_dense_structured_bundle or expansion_prefers_structured_bundle_body_rows_for_second_best_queries or structured_bundle_context_text_keeps_multi_row_body_for_second_best_queries or expansion_second_best_query_skips_composite_target_method_rows or finalize_without_rerank_keeps_full_comparator_bundle or finalize_without_rerank_dedupes_normalized_explicit_comparator_rows or slot_reservation_prefers_table_rows_for_best_few_query"`
- 结果：`12 passed, 86 deselected`
