# v0.1.0 Audit — Fix Tracker

Remediation of the release blockers from `AUDIT-v0.1.0.md`. Checked items are landed in the working tree; see commit history for details.

## Blockers

- [x] **B1 — Remote methods dead on the wire.** Added `base_url=self.base_url` to both pooled httpx clients (`_ensure_sync_client`, `_ensure_client`); bare-path methods now resolve, absolute `_build_url` paths pass through unchanged. `client.py`
- [x] **B2 — `query()`/`query_async()` default `search_type` divergence.** Standardized the default to **`"hybrid"`** across every unified-query entry point (per maintainer preference — hybrid generally outperforms vector-only): `query`/`query_async`/`query_cursor(_async)`/`query_stream(_async)`/`query_multi_column(_async)` in the ABC, local impl, and remote client; the server unified-query handlers (`search`/multi-column/streaming + `validate_search_params`), `DBManager.search_databases`, the `QueryBuilder` `SearchClause` fallback, and the README signature. Also made `query_async` keyword-only to match `query`; all 3 internal callers already used kwargs. Type-specific endpoints (`/search/vector` etc.) and deliberate per-endpoint defaults (factcheck/MCP, already hybrid) untouched.
- [x] **B3 — `lvdb shell` `list`/`search` call nonexistent methods.** `list`→`db.filter(limit=, offset=)`, `search`→`db.query(query, search_type=, k=limit)`. `cli/_shell.py`
- [x] **B4 — Reranker silently dropped.** Two-part fix, by design split along "materializing vs streaming":
  - **Materializing paths now rerank end-to-end.** Remote `query`/`query_async` forward `reranker_config` (and the previously-dropped `search_level`) in the payload, and the server `search_handler` reads `reranker_config`/`search_level` and passes them to `db.query()`. A non-serializable `reranker` *instance* passed to a remote DB now raises a clear `ValueError` (only `reranker_config` can cross HTTP). `client.py`, `routers/search.py`
  - **Streaming/cursor paths now fail loudly instead of silently dropping.** `query_cursor(_async)`, and `QueryBuilder.cursor()/.stream()` raise `ValueError` when a reranker is supplied — reranking requires the fully materialized result set, which is incompatible with lazy cursor hydration. Shared message lives in `cursor.py` (`_RERANK_STREAMING_UNSUPPORTED`). `database/_search.py`, `query_builder.py`, `cursor.py`
- [x] **B5 — Hard-coded `/tmp` breaks backup/migration on Windows.** Replaced 4 `Path("/tmp")` placeholders with `Path(tempfile.gettempdir())`. `cli/_backup.py`, `cli/_migration.py`
- [x] **B6 — `/health` returns 200 on unhealthy.** Now sets HTTP 503 on the failure path via injected `Response`. `routers/health.py`
- [x] **B7 — Contributor docs reference nonexistent plugin API.** Corrected `CLAUDE.md` embedding-provider guide to `_embed_single_batch` + the `embed_batch/embed_async/embed_sync` public surface.

**Validation:** ruff ✓ · black ✓ · mypy ✓ (only a pre-existing unrelated error in `_comparison.py:100`) · full fast suite **1305 passed, 64 deselected** (includes 16 new B4 + parity tests).

## Regression guards added

- `tests/test_api_parity.py` — **sync↔async** signature/default parity (`query`/`query_async`, cursor, stream, multi-column) and **local↔remote** signature parity (`query`/`query_async`), plus an explicit `search_type == "hybrid"` default check. These would have caught B2 in CI.
- `tests/test_cursor.py::TestCursorRerankingRejected` + `tests/test_client.py::TestRemoteRerankerWiring` — B4 behavior: streaming rejects rerankers; remote forwards `reranker_config`/`search_level` and rejects reranker instances.

## Notes / decisions

- **B2 default = `"hybrid"`**: maintainer chose hybrid as the product default (better general-purpose recall than vector-only). Applied uniformly to all unified-query entry points across library/client/server + README so the sync/async (and local/remote) split cannot reappear. Type-specific endpoints that force a type are unchanged.
