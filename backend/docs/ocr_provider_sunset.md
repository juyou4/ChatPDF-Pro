# OCR Provider Sunset Gate

`doc2x` and `paddleocr` are compatibility providers. Their configuration keys
and persisted artifacts must remain readable until this gate is satisfied.

Before removing either provider, record a full release cycle with:

1. zero `success_count` requests and zero configured installations in provider usage telemetry for the full observation window. `count` is retained only as a compatibility alias; do not use attempts or failures as deletion evidence;
2. no active document artifact whose `provider` matches the retiring provider;
3. migration availability: `doc2x -> local_auto` and `paddleocr -> paddleocr_vl` only after a real PaddleOCR-VL adapter is shipped;
4. a tombstone reader that accepts the old config and returns a structured deprecation warning;
5. fixture coverage for old configuration round-trip and existing document reads.

Until then, provider IDs, environment variables, and config readers are retained.
`doc2x` is already excluded from page-level OCR execution. `paddleocr` remains
local-only; it must not be silently substituted with a cloud service.

Usage telemetry stores `attempt_count`, `success_count`, `failure_count`,
`fallback_success_count`, and per-operation counters (`page_ocr` or
`document_parse`). Legacy counters are conservatively imported as successes so
an upgrade cannot create a false zero-use result.
