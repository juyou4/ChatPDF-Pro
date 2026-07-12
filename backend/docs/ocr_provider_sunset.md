# OCR Provider Sunset Gate

`doc2x` and `paddleocr` are compatibility providers. Their configuration keys
and persisted artifacts must remain readable until this gate is satisfied.

Before removing either provider, record a full release cycle with:

1. zero successful requests and zero configured installations in provider usage telemetry;
2. no active document artifact whose `provider` matches the retiring provider;
3. migration availability: `doc2x -> local_auto` and `paddleocr -> paddleocr_vl`;
4. a tombstone reader that accepts the old config and returns a structured deprecation warning;
5. fixture coverage for old configuration round-trip and existing document reads.

Until then, provider IDs, environment variables, and config readers are retained.
`doc2x` is already excluded from page-level OCR execution. `paddleocr` remains
local-only; it must not be silently substituted with a cloud service.
