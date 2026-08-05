# OCR Provider Sunset Gate

MinerU is the only online document-parse provider exposed by the product. The
old `mistral` and `doc2x` identifiers are retained only as migration tombstones:
their configuration keys can be read so an upgrade does not crash, but they are
not registered, selectable, or accepted by the online configuration/validation
endpoints. They must never be silently re-enabled as a fallback.

`paddleocr` is different: it remains an optional local page-quality supplement.
It is not a document route and must not replace a selected MinerU/local route.

The legacy readers can be removed in a later breaking release only after:

1. usage telemetry shows zero successful requests and zero configured legacy installations for a full observation window;
2. no active document artifact still references a legacy provider;
3. a one-time migration has converted old settings to MinerU or local OCR;
4. a tombstone response has been shipped for at least one release cycle.

Usage telemetry stores `attempt_count`, `success_count`, `failure_count`,
`fallback_success_count`, and per-operation counters (`page_ocr` or
`document_parse`). Legacy counters are conservatively imported as successes so
an upgrade cannot create a false zero-use result.
