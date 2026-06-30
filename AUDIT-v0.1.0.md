# LocalVectorDB — Pre-v0.1.0 Consistency & Clarity Audit

Read-only audit of `localvectordb` + `localvectordb_server` (~50k LoC) focused on **API-pattern consistency** and **code clarity**. Conducted via six parallel subsystem audits; per-subsystem reports live in the session scratchpad (`audit-01..06-*.md`). Findings tagged `[verified]` were spot-checked against source by the lead.

Severity: **BLOCKER** = confirmed runtime failure or definitely-broken surface that ships in v0.1.0 · **HIGH** = consumer-visible inconsistency that traps users or breaks the local↔remote promise · **MEDIUM/LOW** = drift & clarity debt.

---

## Release blockers (confirmed broken, not merely inconsistent)

| # | Finding | Location | Evidence |
|---|---------|----------|----------|
| B1 | **~14 RemoteVectorDB methods are dead on the wire.** Tuning/maintenance/system/comparison/factcheck/streaming methods pass bare paths (`f"/api/v1/{name}/..."`) to `_make_request_with_retry`, which forwards `url` unchanged to `client.request()`. But `_ensure_sync_client`/`_ensure_client` build the httpx client with **no `base_url`** → `httpx.UnsupportedProtocol`. Core methods avoid this only because they call `self._build_url()` first. | `client.py` clients at `:607`,`:2221`; bare-path calls at `:3322,3329,…,3738`; `_build_url` at `:669` | `[verified]` |
| B2 | **`query()` and `query_async()` return different results for identical input** — sync defaults `search_type="vector"`, async defaults `"hybrid"`. Root cause is the ABC: `base.py:88-92` (vector) vs `base.py:466-470` (hybrid). Remote client mirrors the split *inverted* (remote async defaults `vector`, keyword-only), so it's wrong three ways. | `_search.py:130` vs `:2114`; `client.py:2952` | `[verified]` |
| B3 | **Interactive `lvdb shell` `list` and `search` are dead on arrival.** Calls `db.list_document_ids()` (defined nowhere) and `db.search()` (only on RemoteVectorDB/QueryBuilder, never `LocalVectorDB`). Errors are swallowed by the REPL catch-all. Non-shell `db list`/`db search` correctly use `db.filter()`/`db.query()`. | `_shell.py:437,496` | `[verified]` |
| B4 | **Reranking silently does nothing on several paths.** `query_cursor()` has a `pass` no-op; `query_stream()` and `QueryBuilder.cursor()/.stream()` don't accept a reranker at all; the remote `query`/`query_async`/`query_stream` signatures take `reranker`/`reranker_config`/`search_level` but never put them in the payload. A configured reranker is dropped with no error. | `_search.py:518-522,723,766`; `query_builder.py:1435,1707`; `client.py:1579-1589,3005-3016` | — |
| B5 | **Hard-coded `/tmp` breaks backup & migration CLI on Windows** (the project's primary platform). Sibling code uses `tempfile`. | `_backup.py:498,598`; `_migration.py:445,524` | — |
| B6 | **`/health` returns HTTP 200 with `{"status":"unhealthy"}`** on failure — breaks both status semantics and the error envelope; load balancers see a healthy node. | `health.py:31-33` | — |
| B7 | **Contributor docs reference a plugin API that doesn't exist.** CLAUDE.md's "Adding New Embedding Provider" says implement `embed_documents()`/`embed_query()`; the real abstract is `_embed_single_batch` and the public surface is `embed_batch/embed_async/embed_sync`. Following the docs fails. | `CLAUDE.md`; `embeddings.py:193,420` | — |

---

## Cross-cutting theme 1 — sync/async divergence (highest-risk pattern)

The sync and async halves of the same method repeatedly disagree on more than threading:

- **Default `search_type`** (B2). `[verified]`
- **Call convention:** `query()` is keyword-only (`*`); `query_async()` accepts positionals — and the remote client flips which side is keyword-only. `_search.py:129` vs `:2112`; `client.py:2952`.
- **Metadata-embedding logic differs:** sync embeds any non-None field via `str()`/`embed_sync`; async only handles TEXT/JSON, requires `.strip()`, uses `embed_batch` → same data, different stored vectors. `_metadata.py:399` vs `:419`.
- **`update_metadata_schema_async` drops pragmas + skips async-init guards** that the sync path applies. `_metadata.py:354-355` vs `_core.py:1042`.
- **Empty-input / missing handling differs sync vs async** across `get`/`delete`/`exists` (one raises, one returns, one builds `IN ()`). `_crud.py` / `_core.py`.
- **Reranker async is fake:** base `rerank_async` just calls the sync body in-thread; only Jina truly overrides — inverted vs embeddings where async is primary. `reranking.py:53-57`.

**Recommendation:** make async methods thin wrappers over a shared core (or vice versa) so defaults/validation/logic cannot drift; add a parity test that asserts identical signatures + defaults for every `x`/`x_async` pair.

## Cross-cutting theme 2 — local↔remote parity (the migration guarantee)

Beyond B1/B2, the `VectorDB()` factory promise ("swap local for remote, same API") leaks:

- **Silent param drop:** remote `query` discards `search_level`/`reranker`/`reranker_config` (B4).
- **Return-shape divergence:** local comparison methods return dataclasses (`DocumentComparisonResult`, `DocumentSimilarityMatrix`); remote returns raw dicts though docstrings still promise the dataclass. `client.py:3573-3592,3631-3646`.
- **Missing params:** remote `nearest_neighbors` lacks `score_threshold`/`filters` the local has.
- **One-sided surface:** local-only `auto_tune`, `list_sqlite_profiles`, `chunk_similarity_matrix`, `visualize_*`; remote-only `fact_check`, `hybrid_query`, `keyword_search`, `database_exists`, `healthy`.
- **Naming:** `get_async_stats` (local) vs `get_stats_async` (remote); neither in the ABC. `update()` is single-doc `doc_id=` while siblings take `ids: Union[str, List[str]]`.
- **Factory advertises `AsyncLocalVectorDB`/`AsyncRemoteVectorDB` it never returns.** `factory.py:56-57,91-93`.

**Recommendation:** add a parity test that introspects both classes against `base.py` and fails on signature/default/return-type mismatch; have the ABC own the canonical signatures.

## Cross-cutting theme 3 — server has no typed request/response layer

- **No Pydantic models on any JSON endpoint.** Every handler is `(db_name, request: Request)` + `await request.json()` + hand-rolled checks (15+ copies of the empty-body check), returning untyped `dict`. OpenAPI body/response schema is essentially empty.
- **Three error-envelope shapes ship together:** `{"error":{…}}` (APIError) vs `{"detail":…}` (auth `HTTPException` + host/proxy middleware) vs inline `{"error":"…"}` (multi-DB/streaming). Clients can't parse uniformly.
- **`_deps` are plain helper calls, not `Depends()`**; `get_config` is dead (handlers read `app.state.config` directly). Auth, by contrast, *is* applied cleanly via `Depends(require_*_permission)`.
- **Duplicated serializers:** `serialize_query_result` ×3, `serialize_document` ×2.
- `APIError.timestamp` is malformed (`isoformat()+"Z"` → `…+00:00Z`) in every error body.

## Cross-cutting theme 4 — vocabulary & convention drift

The same concept is spelled differently depending on where you are:

| Concept | Variants seen |
|---------|---------------|
| metadata filter | `filters` / `where` / `metadata_filters` |
| result limit | `k` / `top_k` / `limit` / `total_k` |
| pagination | `limit+offset` (cap 10000) / `page+limit` (cap 1000) |
| DB-name response key | `name` / `database` |
| total count | `total_results` / `count` / `total_count` |
| URL casing | kebab (`/nearest-neighbors`, `/auto-tune`) vs snake (`/query_builder`, `/incremental_vacuum`) |
| CLI output format | `--json/-j` / `--json` / `--output/-o {table,json,key-only}` / `--format/-f {…}` |
| CLI DB identity | group arg / positional `database_name` / positional `database` / `--database/-d` / `--databases-root` |

Plus: REST uses `POST` for reads/deletes (`POST /documents/delete`, `POST /documents/filter`) and `PUT` for *partial* updates (should be `PATCH`); CLI exit codes are non-uniform (`_tuning.py`/`_mcp.py` print errors then exit 0); CLI short flags are overloaded (`-m`, `-t`, `-l` each mean 2–4 things, `-m` is both a value and a boolean *in the same group*).

**Recommendation:** publish a one-page naming convention (filter key, limit key, pagination, format flag, error envelope) and align before tagging — these are cheap now and breaking later.

## Cross-cutting theme 5 — scoring normalization

`QueryBuilder.semantic_filter` (COSINE) compares a raw `[-1,1]` cosine against a `[0,1]`-validated threshold, while the rest of the codebase maps cosine via `(cos+1)/2`. Same threshold value means different things on different paths. `query_builder.py:230-234,433` vs `_comparison.py:41,51`, `_search.py:98`. Post-rerank `score` is also incomparable across rerankers (sigmoid vs min-max vs raw API vs overlap fraction).

## Cross-cutting theme 6 — plugin families don't share a shape

The three plugin families (embeddings / rerankers / extractors) were clearly designed at different times:

- **Availability/dependency contract exists only for extractors** (`_check_availability`, `required_packages`, `available`, `priority`); embeddings/rerankers surface a missing dep as a bare `ImportError` deep in `_load_model`.
- **`validate_model()` has no consistent contract:** Ollama does a real network check and raises; Google returns `False` on error; Jina/HFInference `return True`; others try-load.
- **Registries diverge three ways:** embeddings/rerankers store *classes* keyed by short names (`"ollama"`) and auto-register in code; ExtractorRegistry stores *instances* keyed by *class name* (`"All2MdExtractor"`), has no factory, and is populated only by entry-point discovery → extractors silently vanish if package metadata is stale (a hazard the project memory already records).
- **Error philosophy splits:** embeddings/rerankers raise; extractors return `ExtractionResult(success=False)`.

## Clarity / duplication debt (representative, not exhaustive)

- `update_metadata_schema` vs `_async`: ~110 near-verbatim lines that have already drifted (theme 1's metadata bug is that drift). `_metadata.py:137-248` / `:250-362`.
- Duplicated hybrid-merge logic across sync collector / async loop / merge helper, already drifting. `_search.py:351,603,661`.
- HTTP embedding providers each repeat an effectively-dead `if client is None:` branch; `_embed_batch_impl` duplicated ~80 lines base↔HTTP-base; reranker retry loops hand-rolled per provider.
- Dead code: factory async classes; `_db.py:1046-1094` unused validators; `_split_into_sections`/`_split_large_section`; OpenAI's unreachable `raise ValueError("Unknown model.")`.
- `QueryBuilder` can't reach two documented capabilities: `return_type="enriched"` (no literal/method) and pure-keyword `vector_weight=0.0` (swallowed by `if vector_weight:`).
- Route collision: top-level `/{db_name}/...` shadows literal routes, so a DB named `search`/`embeddings`/`factcheck` is unreachable; the manager doesn't reserve those names.
- Copy-pasted docstrings (Ollama copied from OpenAI, "embeddding" typo); stale docstrings (`ChunkPosition.to_dict` lists 4 keys, emits 6).

---

## Suggested sequencing for v0.1.0

1. **Fix the blockers B1–B7** — these are broken behavior, not taste. B1/B2/B3 in particular will be the first things a user hits.
2. **Add two parity tests** (sync↔async signature/default parity; local↔remote method parity vs the ABC). These convert themes 1–2 from recurring drift into CI failures.
3. **Pick conventions** (theme 4) and align the cheap ones before the API is public and the names are load-bearing.
4. Defer themes 5–6 + clarity debt to a fast-follow unless time allows — they're real but not user-breaking on day one.
