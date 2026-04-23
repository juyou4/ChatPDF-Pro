# PR: numeric_table specialisation cleanup & feature flag

> Branch: `refactor/numeric-table-cleanup`
> Base: `main` (5712aa5)
> Backup: `wip/pre-cleanup-backup` (41b25ad6, full snapshot of pre-cleanup WIP)

## Motivation

Over 50+ local RAGAS iteration rounds produced ~8k lines of uncommitted
retrieval-pipeline changes centred on `numeric_table` (Table 7 /
DiffuLT / "第二好的方法" style queries) plus a large amount of
evaluation artefacts. The WIP mixed valuable work with debug detritus,
had no feature flag, and was impossible to review as a single blob.

This PR:
1. Archives and gitignores the evaluation detritus.
2. Introduces a master feature flag for the numeric_table
   specialisation so we can A/B its effect on generic queries.
3. Validates via a proxy test-suite comparison that the specialisation
   does **not** systematically degrade generic paths.
4. Splits the remaining ~1.5k lines of real code across 10 atomic
   commits that can each be reverted independently.

See `docs/design/numeric_table_generalization_report.md` for the A/B
validation data.

## Highlights

- **New flag**: `config.enable_numeric_table_specialization`
  (env: `CHATPDF_ENABLE_NUMERIC_TABLE`, default `true`)
  Toggled via `services.rag_config.should_apply_numeric_table_specialization()`.
- **Unified entry point**: Every `_numeric_table_*` / `_structured_table_bundle`
  function in `embedding_service.py` and `chat_routes.py` short-circuits
  to a no-op when the flag is off.
- **Archive**: 84 stale RAGAS round CSV/JSON + 33 log files moved to
  `exports/ragas/`; design docs consolidated under `docs/design/`.
- **Atomic commits**: 10 clean commits ordered by dependency so any
  single feature can be reverted without breaking compilation.

## Commit list (in order)

| # | SHA | Subject |
|---|---|---|
| 1 | `5f7b027b` | chore: gitignore evaluation artifacts and archive design docs |
| 2 | `1a5b8865` | feat(services): add scaffolds for synonym, table-aware, answer-critic, raptor, odl-parser |
| 3 | `1551ecde` | feat(config): add feature flags for cheap_model, answer_critic, numeric_table, synonym, query_rewrite |
| 4 | `154cec06` | feat(bm25): wire in synonym expansion for query-time recall |
| 5 | `63d71adc` | feat(query): enhance query_rewriter and query_analyzer with numeric_table awareness |
| 6 | `45bb1b39` | feat(retrieval): numeric_table specialization and generic retrieval improvements |
| 7 | `7b17b550` | feat(chat): agent context builder, cheap-model integration, numeric_table citation support |
| 8 | `c20519e6` | feat(document): structured-table bundle sanitisation and model provider route hardening |
| 9 | `8c55a946` | feat(frontend): streaming state plumbing for numeric_table citations and model defaults |
| 10 | `9e11f381` | chore(docs): add AGENTS.md workspace guide, README updates, DocLayout-YOLO startup wiring |

## Validation

Per `docs/design/numeric_table_generalization_report.md`:

### Flag ON (default)
| Scope | Result |
|---|---|
| `fix_summary.md` baseline (8 regression tests) | **8 passed** |
| 6 numeric-table-adjacent suites | **116 passed / 0 failed** |
| Full backend suite (excl. the huge pipeline-order file) | **396 passed / 11 pre-existing failed** |

The 11 failing tests are unrelated to numeric_table: OpenAI provider
properties, web-search service, overview figure clip, and a
hypothesis-seed-dependent snippet-alignment property. All of them
already failed on the snapshot before cleanup.

### Flag OFF (`CHATPDF_ENABLE_NUMERIC_TABLE=0`)
| Scope | Result |
|---|---|
| 28 generic test files (no numeric_table keyword) | **274 passed / 12 failed** |

Δ vs flag ON = +3 failures; all three are hypothesis flakes or
test-ordering side-effects that disappear when run in isolation.
**No systematic regression in the generic retrieval path.**

## Case decision

The A/B result places us in **Case A — keep all numeric_table
specialisation**. The flag stays default-on; anyone needing to sanity
check a generic-only path can flip it off without touching code.

## Deferred follow-up

`embedding_service.py` is still ~9.4k lines. Extracting the
`_numeric_table_*` / `_structured_table_bundle` family into a dedicated
`services/numeric_table_service.py` is deliberately deferred to a
follow-up PR because:
- it is a purely structural move (no behaviour delta), keeping it
  separate makes review safer;
- function dependencies are dense and a naive move risks import cycles;
- the feature flag already gives us the "one place to disable" property
  we wanted from the split.

Plan doc (`C:\\Users\\tan\\.windsurf\\plans\\chatpdf-wip-cleanup-evaluation-a242fd.md`)
updated accordingly.

## Rollback

Any single commit is safe to `git revert`. If a bigger problem surfaces,
we can reset to `main` and cherry-pick the good commits back; the full
pre-cleanup state is preserved at `wip/pre-cleanup-backup`.

## Files

- New: `backend/services/{synonym,table_aware,answer_critic,raptor,odl_parser}_service.py`
- New: `backend/tests/{test_agentic_doc_context,test_chat_routes_non_stream,test_rerank_pipeline_order,test_structured_table_bundle_pipeline}.py`
- New: `backend/tests/{eval_ragas,build_groups,dump_per_sample,diagnose_agent_routes}.py`
- New: `AGENTS.md`
- New: `docs/design/{RAG_OPTIMIZATION_PLAN,fix_summary,numeric_diagnosis,numeric_table_generalization_report,numeric_table_cleanup_pr_notes}.md`
- Modified: `backend/{config,routes/chat_routes,routes/document_routes,routes/model_provider_routes}.py`
- Modified: `backend/services/{embedding_service,chat_service,rerank_service,hybrid_search,graph_service,context_builder,vector_service,retrieval_agent,query_rewriter,query_analyzer,rag_config,citation_service,bm25_service}.py`
- Modified: `frontend/src/{components/ChatPDF.jsx,config/systemModels.ts,contexts/DefaultsContext.tsx,hooks/useMessageState.js}`
- Modified: `README.md`, `README_EN.md`, `start.bat`, `start.sh`, `.gitignore`

## How to try it locally

```powershell
# Sanity-check with flag ON (default)
python -m pytest backend/tests/test_rerank_pipeline_order.py -q \
  -k "preserves_fewshot_focus or upgrades_sparse_structured_bundle_with_recovered_rows \
  or upgrades_existing_sparse_structured_bundle_result \
  or finalize_without_rerank_keeps_full_comparator_bundle \
  or finalize_without_rerank_dedupes_normalized_explicit_comparator_rows \
  or slot_reservation_prefers_table_rows_for_best_few_query"

# Fallback path with flag OFF
$env:CHATPDF_ENABLE_NUMERIC_TABLE="0"
python -m pytest backend/tests/test_agentic_doc_context.py \
  backend/tests/test_query_rewriter.py -q
```
