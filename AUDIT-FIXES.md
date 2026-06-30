# v0.1.0 Audit — Fix Tracker

Remediation of the release blockers from `AUDIT-v0.1.0.md`. Checked items are landed in the working tree; see commit history for details.

## Blockers

- [x] **B1 — Remote methods dead on the wire.** Added `base_url=self.base_url` to both pooled httpx clients (`_ensure_sync_client`, `_ensure_client`); bare-path methods now resolve, absolute `_build_url` paths pass through unchanged. `client.py`
- [x] **B2 — `query()`/`query_async()` default `search_type` divergence.** Standardized the default to **`"hybrid"`** across every unified-query entry point (per maintainer preference — hybrid generally outperforms vector-only): `query`/`query_async`/`query_cursor(_async)`/`query_stream(_async)`/`query_multi_column(_async)` in the ABC, local impl, and remote client; the server unified-query handlers (`search`/multi-column/streaming + `validate_search_params`), `DBManager.search_databases`, the `QueryBuilder` `SearchClause` fallback, and the README signature. Also made `query_async` keyword-only to match `query`; all 3 internal callers already used kwargs. Type-specific endpoints (`/search/vector` etc.) and deliberate per-endpoint defaults (factcheck/MCP, already hybrid) untouched.
- [x] **B3 — `lvdb shell` `list`/`search` call nonexistent methods.** `list`→`db.filter(limit=, offset=)`, `search`→`db.query(query, search_type=, k=limit)`. `cli/_shell.py`
- [ ] **B4 — Reranker silently dropped.** Wire reranker through cursor/stream paths and the remote `query` payload. (deferred — larger change)
- [x] **B5 — Hard-coded `/tmp` breaks backup/migration on Windows.** Replaced 4 `Path("/tmp")` placeholders with `Path(tempfile.gettempdir())`. `cli/_backup.py`, `cli/_migration.py`
- [x] **B6 — `/health` returns 200 on unhealthy.** Now sets HTTP 503 on the failure path via injected `Response`. `routers/health.py`
- [x] **B7 — Contributor docs reference nonexistent plugin API.** Corrected `CLAUDE.md` embedding-provider guide to `_embed_single_batch` + the `embed_batch/embed_async/embed_sync` public surface.

**Validation:** ruff ✓ · black ✓ · mypy ✓ (only a pre-existing unrelated error in `_comparison.py:100`) · full fast suite **1289 passed, 64 deselected**.

## Notes / decisions

- **B2 default = `"hybrid"`**: maintainer chose hybrid as the product default (better general-purpose recall than vector-only). Applied uniformly to all unified-query entry points across library/client/server + README so the sync/async (and local/remote) split cannot reappear. Type-specific endpoints that force a type are unchanged.
