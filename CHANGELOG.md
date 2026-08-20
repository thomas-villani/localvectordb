# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A persisted per-database default reranker.** Pass `reranker_config` at
  creation (or call `set_default_reranker()` later) and every `query()` applies
  it — library, CLI, MCP and server alike — without per-call ceremony. The
  config is saved in the database's own config table like the embedding
  provider, so it survives reopen. Reranking measured ~9× the effect of any
  first-stage parameter in the retrieval study, but it was previously
  reachable only by passing the full config on every single call.
  - Precedence per call: a `reranker` instance > `reranker_config` >
    `reranker=False` > the persisted default. `reranker=False` (wire form
    `"rerank": false`) disables reranking for one call; combining it with a
    `reranker_config` raises.
  - The default reranker is constructed lazily on the first query that needs
    it and cached per database, so opening a database never loads a model.
  - Cursors and streaming still never rerank, and now ignore the persisted
    default rather than erroring on it; passing an explicit reranker to a
    cursor still raises.
  - Store credentials as environment references (`api_key="$MY_KEY_VAR"`);
    a raw key warns at set time and is redacted from every echo.
- CLI: `lvdb create --reranker-provider/--reranker-model` persists the default;
  `lvdb db <name> search` gains `--rerank/--no-rerank`, `--rerank-provider`,
  `--rerank-model` and `--rerank-k`. Neither surface had any rerank flag.
- HTTP: `POST /databases` accepts a `reranker` block and echoes the resolved
  (redacted) config from both create and `GET /databases/{name}/info`;
  `QueryBody` gains `rerank` for the per-call disable. `RemoteVectorDB` sends
  the block at creation and exposes read-only `get_default_reranker()`.
- MCP `query_database` inherits the database default with no new tool surface.

### Fixed

- **`/query-multi-column` and the global `/search` accepted `reranker_config`
  and `rerank_k` and then silently discarded them** — reranking requested
  through either route simply never happened. Both now forward the parameters,
  which required `query_multi_column()`/`query_multi_column_async()` and
  `search_databases()` to grow the parameters in the first place.
- **Multi-column search would have reranked only its content leg**, merging
  re-scored content hits with un-rescored metadata-field hits into one ranking.
  Reranking now runs exactly once, over the merged pool, with the same
  `rerank_k` over-fetch `query()` uses.
- **A `QueryBuilder` rerank step on a database with a default reranker would
  have reranked twice.** The builder now suppresses the database default for
  its underlying query whenever it carries its own rerank configuration; a
  builder without one inherits the default as usual.

### Security

- A persisted reranker's `base_url` is now gated by the same SSRF policy as
  embedding provider URLs (`embedding.allow_custom_provider_url` plus the host
  allowlist), since the server POSTs to it on every query. Per-request
  `reranker_config.base_url` on `/query` remains ungated — tracked separately.

## [0.1.1] - 2026-08-19

### Fixed

- **Bulk operations on ≥32,767 ids no longer die with `sqlite3.OperationalError:
  too many SQL variables`.** Every `IN (?,?,...)` clause expanded from a
  caller-sized id list — the `upsert()`/`insert()` existing-document pre-checks
  (sync and async), `get()`, `exists()`, `delete()` and their async twins,
  per-document chunk removal, and incremental backup's changed-chunk collection —
  now batches its ids under SQLite's bound-parameter limit (999 before SQLite
  3.32, 32,766 since; batches of 900 clear both). First seen when the tag-triggered
  benchmark workflow ran the tier-2 insert benchmark at its 50,000-document
  scale, which crashed inside `upsert()`'s existing-chunk pre-fetch; search-path
  id lists are bounded by pool sizes and were never affected.
- **JavaScript SDK: the query surface the server already supported is now
  expressible from TypeScript** — `search_level` (hierarchical retrieval was
  unreachable), `reranker_config`/`rerank_k` (cross-encoder reranking), and the
  missing `DocumentScoringMethod` values `"auto"` (the server default) and
  `"percentile"`. The SDK never sent wrong values — these were type-surface
  gaps, and the server rejects unknown fields, so the affected features were
  simply unusable from JS. See `sdk/js/CHANGELOG.md`.
- All nine open CodeQL alerts resolved (quality-level, none security-severity):
  an unexplained bare `except` in section-metadata hydration, NaN checks
  spelled `v != v` in a benchmark reporter, a test override whose signature had
  drifted from `embed_batch`'s, and assorted test hygiene.

### Security

- Bumped the locked (transitive) `cryptography` from 48.0.1 to 50.0.0
  (GHSA: PKCS#7 `EnvelopedData` decryption Bleichenbacher oracle). Pulled in
  via the MCP extra's auth stack (authlib/joserfc) and all2md's Outlook
  extra (msoffcrypto-tool); localvectordb never calls the vulnerable PKCS#7
  path itself.

### Infrastructure

- The JS SDK's typecheck/tests/build now run in PR-time CI; previously they ran
  only inside the tag-triggered npm release workflow, where a failure blocks a
  release already in flight.
- `bump-my-version` now keeps `sdk/js/package.json` in version lock-step (the
  npm tag workflow refuses a mismatch). Caveat documented for PEP 440 rc
  versions, which npm's semver cannot represent.

## [0.1.0] - 2026-08-19

The first stable release. Everything below was developed and measured on top of
`0.1.0rc1`; the retrieval-affecting changes were gated against the committed
BEIR SciFact and Qasper baselines (see `benchmarks/RETRIEVAL_BASELINE.md`) and
the study behind them is written up in the documentation's retrieval study and
lab notebook pages.

### Added

- **Retrieval prefixes for asymmetric embedding models.** Most modern retrieval
  encoders are trained with a different instruction prepended to a stored passage
  than to a search query; embedding both sides identically raises nothing and
  simply ranks worse. Ingestion now embeds with a `document_prefix` and `query()`
  with a `query_prefix`, both resolved automatically from the model name for
  `embeddinggemma`, `nomic-embed-text`, `snowflake-arctic-embed*`,
  `mxbai-embed-large`, `bge-*-en` and `e5-*`/`multilingual-e5`. Model matching
  ignores registry paths and version tags, so `hf.co/google/EmbeddingGemma-300M`
  and `embeddinggemma:300m` resolve alike. Set them yourself for an unrecognised
  model via `embedding_config={"document_prefix": ..., "query_prefix": ...}`, the
  new `lvdb create --document-prefix/--query-prefix` flags, or the `[embedding]`
  `config` table; `""` forces no prefix and `auto_prefix=False` disables the
  lookup. Every embedding call takes a `task` (`"document"` by default, or
  `"query"`), and providers gain `embed_query()`/`embed_query_async()` returning a
  single 1-D vector.

  A model not in the registry gets no prefix — symmetric models such as `bge-m3`,
  `gte-*` and OpenAI's `text-embedding-3` family are correct as-is, and an unknown
  model is assumed symmetric rather than guessed at.
- **Per-side task parameters for Google and Jina**, whose APIs take an explicit
  task instead of a text prefix: `document_task_type`/`query_task_type` and
  `document_task`/`query_task` respectively.
- **A worked visualization gallery in the docs.** The comparison/visualization
  page now shows real output for every plot type — embedding map, clusters,
  similarity heatmap, similarity graph, synteny and chord — over 60 Qasper papers
  encoded with `embeddinggemma:300m`, plus a cross-encoder comparison of the same
  corpus under four models with query overlays. Regenerate with
  `python -m benchmarks.doc_figures`, which reads the existing benchmark
  embedding cache and makes no API calls.
- **Text labels on synteny and chord diagrams.** `plot_synteny(labels_1=,
  labels_2=)` and `plot_chord(labels=)` take one string per chunk — a section
  heading, say — instead of the chunk index, which is what turns a chord from
  "chunk 17 resembles chunk 3" into "the evaluation section restates the
  introduction". Text labels are drawn outside the track and, on a chord, rotated
  to follow the circle. Supported by the interactive renderers too, where they
  additionally appear in each segment's hover text, and reachable through
  `db.visualize_synteny()` / `db.visualize_chord()` with or without
  `interactive=True`.


- **`delimiter` chunking strategy** — split a document on a literal delimiter
  sequence (`"\n\n"` by default), packing the resulting segments into chunks up
  to `chunk_size`. A segment larger than the limit falls back to character-level
  splitting, so no chunk overflows even when the delimiter leaves an over-long
  piece; spans stay contiguous, so `reconstruct_document` is exact. Select it as
  `chunking_method="delimiter"` with a new `chunk_delimiter` constructor
  parameter (persisted per database), on the CLI as
  `lvdb create --chunking-method delimiter --chunk-delimiter '\n---\n'`, and for
  standalone use as `lvdb chunk -M delimiter --delimiter '\n---\n'`. The CLI
  interprets `\n`/`\t`/`\r` escapes in the delimiter. (Local library + CLI;
  remote/server exposure of `chunk_delimiter` is deferred, consistent with the
  other remote-parity gaps.)
- **`lvdb chunk`** — run the position-aware chunkers standalone, with no
  database, embedding, or config, emitting one JSON object per chunk (JSONL).
  Reads files, globs, direct text, or stdin, extracting rich formats
  (PDF/DOCX/HTML/…) to Markdown first, exactly as ingestion does.
- **`LocalVectorDB.grep()`** — lexical, line-oriented content search, distinct
  from the ranked `query()` retrieval pipeline. Literal or regex matching with
  `ignore_case`, `whole_word`, and grep-style context (`context` / `before_context`
  / `after_context`), returning `GrepMatch` objects (document id, 1-based line
  number, column span, and surrounding lines). Scope the scan with `prefix=` (id
  prefix) or `where=` (metadata filter), and cap results with `max_count` (per
  document) / `limit` (total). Exposed on the CLI as `lvdb db <name> grep PATTERN`
  (`-e/--regex`, `-i`, `-w`, `-A/-B/-C`, `--prefix`, `-m`, `-n`, `-j`). Intended
  to sit alongside vector and keyword search for agents that know a precise
  string. Also exposed as the read-only `grep_documents` MCP tool. Local-library
  + CLI + MCP; HTTP/SDK exposure is deferred.
- **`LocalVectorDB.list_prefixes()`** — S3-style navigation of "filesystem-like"
  document ids: treats a delimiter (`/` by default) as a virtual path separator
  and rolls documents up to their immediate children beneath a prefix, returning
  a `PrefixListing` of virtual folders (common prefixes with recursive counts)
  and leaf documents. Makes relative-path document ids (`docs/reports/q1`)
  browsable like folders without any schema change. Exposed as
  `lvdb db <name> ls [PREFIX]` (`-d/--delimiter`, `-j`) and as the read-only
  `list_prefixes` MCP tool. New public types
  `PrefixListing` / `PrefixEntry` / `GrepMatch`.
- **OpenRouter embedding provider** (`provider="openrouter"`,
  `OpenRouterEmbeddings`) — OpenAI-compatible access to OpenRouter's embedding
  models (OpenAI, Google, Mistral, Nvidia, and free options) through a single
  endpoint. Pass the model slug (e.g. `openai/text-embedding-3-small`). The index
  dimension resolves as `requested_dimensions` (also requests API-side Matryoshka
  truncation) → `dimension` (a plain declaration of the native size, no payload
  effect) → a one-off probe request; pass either dimension kwarg to skip the
  probe entirely (e.g. for offline/network-free database creation). Reads
  `OPENROUTER_API_KEY`, supports optional `HTTP-Referer` / `X-Title` attribution
  headers and optional L2 `normalize`.
- **`examples/`** — runnable programs rather than snippets, covered by
  `tests/test_examples.py` so they cannot rot. The first is
  `section_vs_chunk_retrieval.py`, which runs this project's headline retrieval
  comparison (section-level vs chunk-level vs fused) **on your own corpus** and
  reports nDCG@10 / recall@k per mode, so the claim in the README is something
  you can check rather than something you have to believe. Ships with a sample
  corpus and judgments. Requires a real embedding backend and refuses the `mock`
  provider, which cannot measure relevance.
- Document **patch API** for in-place edits — change part of a stored document
  without re-sending the whole content. Exact find/replace with a uniqueness
  requirement (the contract coding agents already handle), plus `splice` /
  `append` / `prepend` ops resolved against the original content, validated
  non-overlapping, and applied atomically. Surfaced across every layer:
  - `LocalVectorDB.patch()` / `patch_async()` and `RemoteVectorDB` equivalents,
    returning `PatchResult(updated, new_hash, ops_applied)`.
  - `PATCH /databases/{db}/documents/{doc_id}` gains additive `ops` +
    `expect_hash` fields (mutually exclusive with `content`); `409 HASH_CONFLICT`
    on a stale precondition, `422 PATCH_FAILED` on an unmatched/ambiguous/
    overlapping op.
  - `patch_document` MCP tool exposing the `old_string`/`new_string` edit
    contract for agents.
  - `lvdb db <name> patch <doc_id> --find/--replace/--append/--prepend/--expect-hash`.
  - JavaScript SDK `database.patch()` with typed `PatchOp` / `PatchOptions`.
- Optional `expect_hash` precondition on patches for optimistic concurrency:
  fail instead of clobbering a concurrent write. New `PatchConflictError` and
  `PatchError` exceptions (mirrored in the JS SDK as `PatchConflictError` /
  `PatchFailedError`).
- `OllamaEmbeddings` gains `num_ctx`, `num_batch`, and `truncate` options
  (settable via `embedding_config`). Ollama's `/api/embed` caps input at
  `n_batch` (default **2048**) regardless of `num_ctx`, silently truncating
  longer inputs — so raising `num_ctx` alone does nothing for embeddings past
  2048. `num_batch` auto-defaults to `num_ctx` so a raised context actually
  takes effect (e.g. embed full 8192-token inputs with a long-context encoder).
- **Keyword and hybrid search at every retrieval level.** FTS5 previously
  indexed chunks only, so `search_level="documents"`, `"sections"`, and
  `"fused"` accepted `search_type` and silently answered vector-only — including
  the *default* `"hybrid"`. The keyword leg measures +0.084 to +0.131 nDCG@10
  across six corpus/encoder pairs, so those levels were quietly leaving the
  single largest retrieval effect we measured on the table. Three fixes, one per
  level:
  - `documents` reads the `documents_fts` index that every database already
    built and maintained (qasper document level: vector 0.2758 → hybrid 0.4037).
  - `sections` gains a new `sections_fts` — contentless where SQLite ≥ 3.43
    allows it, since sections tile their parent document and a text-carrying
    index would roughly double a hierarchical database — written at ingest from
    the document slice, cleaned by an `AFTER DELETE` trigger, and *self-healed*
    on existing databases by a batched, resumable backfill: no rebuild, no
    export/import. Section-level hybrid gains +0.04 to +0.25 nDCG@10 depending
    on corpus and `section_vector_strategy`.
  - `fused` runs each granularity as a hybrid at its native pool width and then
    blends, worth +0.057 to +0.123 at the document target on three of four
    corpora (the blend topology and pool were settled by a measured sweep, the
    weights left unchanged).
  No query is silently downgraded any more, and the "search_type cannot be
  honoured" warning added mid-cycle no longer has anything to warn about.
- **`structure` chunking strategy** — a boundary hierarchy (heading > paragraph
  > line > sentence) that cuts each chunk at the strongest boundary inside the
  token budget rather than at a fixed offset, so chunks end at headings and
  paragraph breaks when one is in reach. Documents still reconstruct
  byte-exactly; `min_fill` guards against sliver chunks and fenced code blocks
  are never split. A `heading_finder` callable can contribute extra cut
  positions for corpora whose headings Markdown detection cannot see (numbered
  contract clauses, say) — precision filtering is deliberately the caller's,
  since a bare `Section \d+` regex fires on every cross-reference and table of
  contents.
- **`openai_compatible` embedding provider** — one adapter for every local
  server speaking OpenAI's `/v1/embeddings`: llama.cpp, LM Studio, vLLM,
  text-embeddings-inference, LocalAI. Point `base_url` at the server and go.
  The database persists `embedding_base_url`, because once a provider can point
  anywhere, provider+model alone no longer identifies a vector space — the same
  model served by different engines pools differently. Rerankers gain the same
  reach: `openai_compatible` and `openrouter` reranker providers for the shared
  `/rerank` wire format, with `score_transform="auto"` squashing raw
  cross-encoder logits so `score_threshold` keeps meaning.
- **`document_scoring_method="percentile"`** returns as an explicit option with
  a single knob (`percentile`, default 90) — the two-percentile blend it once
  was is gone, since the clean order statistic beat it in 19 of 20 measured
  cells. It is a *document-target* aggregator: worth up to +0.02 nDCG@10 when
  rolling chunks up to documents on high-fanout corpora, consistently losing at
  section targets — which is why `"auto"` never selects it.
- **`db.diagnose()` and `lvdb db <name> doctor`** — the retrieval study's
  product conclusion, turned into a feature. Six retrieval knobs have no
  defensible global default because their argmaxes are corpus properties; what a
  built index *can* tell you in seconds is which regime you are in. The report
  covers: encoder coverage of chunk text (counted with the encoder's own
  tokenizer where importable, and labelled "estimated" where it is not), section
  length against the encoder window (long sections are windowed and mean-pooled,
  a regime where the pooled vector measurably weakens), the chunkless-section
  recall ceiling, chunks-per-document/section fanout, and keyword-index health
  for all three FTS tables. Ingest also warns — once per database instance, and
  only when measured coverage drops below 0.80, the calibrated point where
  truncation starts costing relevance rather than clipping redundant tails — so
  a `chunk_size` that silently overruns a small-context encoder is no longer
  silent.

### Changed

- **The default embedding model is now `embeddinggemma`** (was
  `nomic-embed-text`), which measures substantially better in our retrieval
  evaluations at the same 768 dimensions. `ollama pull embeddinggemma` to follow
  the quickstart. This only affects databases created *without* an explicit
  `embedding_model`; existing databases keep the model recorded in their config.

  EmbeddingGemma is markedly prefix-sensitive, which is what motivated the prefix
  support above — it is applied automatically, so no extra configuration is
  needed.

- **Google embeddings now default to asymmetric retrieval task types**
  (`retrieval_document` on ingest, `retrieval_query` on search) instead of
  `semantic_similarity` on both sides. `semantic_similarity` is the wrong setting
  for search, and was silently costing relevance on every Google-backed database.
  Passing a single `task_type` still forces that one task on both sides, which is
  what you want for clustering or classification. Databases built on the old
  default should be re-ingested to move their stored vectors into the new space.

- **A lighter `cli` install extra.** The `lvdb` command now installs with just
  `pip install "localvectordb[cli]"` (click + tomli-w + bcrypt) — enough to
  create, inspect, search, chunk, back up, migrate, and configure databases
  without pulling fastapi/uvicorn/slowapi. The `server` extra now includes `cli`
  and adds the HTTP stack; `lvdb serve` imports it lazily and prints an
  actionable hint if only the `cli` extra is installed.
- **`RepairReport.summary` is now a property**, matching the sibling `healthy`
  property, so `report.summary` returns the human-readable line (previously
  `report.summary()` — a bound method if the parentheses were forgotten). Access
  it without the call.
- **`lvdb serve` only probes for Ollama when the configured embedding provider
  is Ollama.** A server backed by OpenAI/Jina/OpenRouter/etc. no longer requires
  a local Ollama install at startup. `--disable-ollama-check` still overrides the
  probe when it does run.
- **The `lvdb` CLI no longer requires a configuration file to operate on a
  database.** When neither a config file nor `--db-folder` is given, the current
  working directory is used as the database folder, so `lvdb db <name> ...` works
  in any folder that contains a database.
- **`return_type` now defaults to `None`** on `query()`/`query_async()` (local,
  remote, and the `query_database` MCP tool), meaning "the unit `search_level`
  searched": documents for the default chunk search, sections for
  `search_level="sections"`. Every existing default is unchanged — this only
  makes "I want documents" distinguishable from "I didn't say", which is what
  lets `search_level="sections"` honour an explicit `return_type` without
  turning a bare section search into a document search. `RemoteVectorDB` omits
  `return_type` from the request when unset rather than sending `"documents"`,
  so remote and local answer a bare section search in the same unit; the server
  resolves an absent `return_type` the same way and still echoes a concrete
  value.
- `MetadataFieldType.valid_types()` is annotated `Tuple[Type[Any], ...]` rather
  than `Tuple[type, ...]`. Same runtime behaviour and same type-checker result;
  the bare `type` had no documentable target, so Sphinx resolved the rendered
  annotation to the unrelated `MetadataField.type` attribute.
- **`query(search_level="sections"|"documents")` now raises `ValueError` on a
  database created without `hierarchical_embeddings=True`**, instead of silently
  returning chunk-level results. The old behaviour handed back plausible
  wrong-level results, which reads as "the feature does nothing" rather than
  "the feature is switched off". `"fused"` already raised; all three levels are
  now consistent, in `query()` and `query_async()` alike. If you were relying on
  the silent fallthrough, pass `search_level="chunks"` (the default) explicitly.
- `lvdb create --chunking-method` now offers every registered chunker (it was a
  hardcoded list missing `paragraphs` and `code-blocks` — the latter documented
  in the README's own code-repository example but unreachable from the CLI).
  `lvdb db <name> search --search-level` gains `fused`.
- **`DELETE /databases/{name}` is now idempotent.** Deleting an absent database
  returns `200` with `deleted: false` (matching document deletion) instead of
  `404`, so a retried or duplicate delete is no longer an error. The response
  gains a `deleted` boolean for clients that need to distinguish "removed now"
  from "was never there".
- **`query(return_type="sections")` now raises `ValueError` on a non-hierarchical
  database** instead of silently returning chunk-level results — consistent with
  the `search_level="sections"` guard. Create with `hierarchical_embeddings=True`
  (or use `search_level="sections"`) for section results.
- Sub-document range specs (`char_range` / `line_range` / `chunk` in the `get`
  CLI and MCP tool) now reject negative and reversed ranges (e.g. `"5:2"`,
  `"-3:"`) with a clear error instead of silently returning an empty or
  wrong slice.
- The interactive shell's `add` command now routes files through the same
  extraction pipeline as `lvdb db add`, so PDF/DOCX/HTML/… are converted to
  Markdown rather than skipped as "not unicode".
- **The local cross-encoder reranker default is now `BAAI/bge-reranker-base`**
  (was `cross-encoder/ms-marco-MiniLM-L-6-v2`). Our reranking study measured the
  old default as statistically indistinguishable from not reranking at all —
  including on MS MARCO's own home domain — while `bge-reranker-base` gains
  +0.03 nDCG@10 through the identical code path, so the old default silently
  cost more than every first-stage tuning effect in the study combined. The docs
  gain a "Choosing a reranker model" section with the findings that generalise:
  model choice dominates the technique, price predicts nothing, and reranker
  input should never be truncated below 512 tokens.
- **`document_scoring_method` now defaults to `"auto"`**, resolved from
  `search_type`: `"best"` for pure vector search, `"frequency_boost"` (the old
  fixed default) for hybrid and keyword. The frequency multiplier assumes the
  min-max-normalised scale that hybrid fusion produces; on raw vector
  similarities it mostly rewards documents for owning more chunks, and `"best"`
  beats it there on all three corpora measured. Any explicitly passed method is
  used unchanged.
- **Embedding batches are now capped by estimated token volume** (50,000 by
  default) as well as by count. A fixed count made request size scale with
  `chunk_size` — large chunks could exceed the provider timeout, and
  `max_retries` then killed the ingest — and OpenAI's 300,000-token request
  limit can reject a full count-based batch outright. The cap is inert at the
  default configuration; batch grouping cannot change the vectors produced
  (verified bitwise).

### Fixed

- **`plot_similarity_graph(layout="spring")` now runs a force-directed layout.**
  It previously ignored `layout` entirely and always embedded the full
  similarity matrix with MDS, which places every node by its distance to every
  other whether or not an edge is drawn — so thresholded-away similarities still
  dragged nodes around and connected components never visibly grouped. The
  Fruchterman-Reingold layout added here only feels the edges that survive the
  threshold, and takes `gravity`/`spread` to trade label legibility against how
  sharply clusters separate. The previous behaviour is still available as
  `layout="mds"`; an unrecognised layout now raises instead of silently falling
  back.
- **`chunk_labels` now does something in the interactive synteny and chord
  plots.** Both accepted the parameter, documented it, and never drew anything —
  they had no test coverage at all, so nothing caught it.
- **`plot_embedding_map()` no longer drops the category legend when queries are
  overlaid.** Colouring by category and passing `queries=` both called
  `ax.legend()`, and the second call replaced the first, so the category key
  silently vanished from exactly the plot that needed it most.
- **Metadata-field vector search no longer probes for a non-existent
  `embed_query`.** `_metadata_field_search` branched on
  `hasattr(provider, "embed_query")`, which no provider implemented — dead code
  that would have started returning a batch-shaped array the moment one did. It
  now calls the real `embed_query()`, which also gives that path the query prefix
  it was missing.
- **Whitespace-only documents now reconstruct byte-for-byte.** Every
  general-purpose chunker emits a single chunk for whitespace-only input instead
  of dropping it, restoring the reconstruction invariant (truly empty input
  still yields no chunks).
- **`$type: "boolean"` metadata filters now agree between `filter()` and
  `query()`.** SQLite stores booleans as `0`/`1`, so the Python post-filter used
  by `query()` now treats an int `0`/`1` as boolean (matching the SQL `IN (0,1)`
  check `filter()` uses), eliminating a filter/query divergence.
- Oversized `k` no longer over-allocates: vector search clamps `k` to the number
  of stored vectors at the FAISS boundary (FAISS does not clamp it itself), for
  both local and remote callers.
- Hybrid streaming/cursor results no longer drop keyword-only matches whose chunk
  has not been embedded yet (NULL `faiss_id`); such hits now hydrate by row id.
- Embedding reconstruction always returns one row per requested id in order
  (zero-filling any id it cannot reconstruct), preventing score misalignment or
  `IndexError` in deduplication / comparison consumers.
- The SSE streaming endpoint releases its query cursor on client disconnect.
- Cursor batch hydration loops instead of recursing when a batch is fully
  filtered out, so a highly selective filter over a large candidate pool can no
  longer overflow the stack.
- MCP `MCPConfig.from_file` validates `mode` (a typo like `"readonly"` used to
  fail open and permit writes) and reports malformed TOML with a clear error.
- Server error envelopes for database create/load/delete/search failures no
  longer echo the underlying exception text, which could leak filesystem paths;
  the detail is still logged server-side.
- PRAGMA string values are quote-escaped before execution.


- **`POST /databases/{db}/query` returned 500 for a caller's bad arguments.**
  `query()` rejects an unsupported `search_level`/`return_type` pairing — or a
  hierarchical level on a database without `hierarchical_embeddings` — with
  `ValueError`, and nothing mapped `ValueError` to a status, so it reached the
  catch-all handler as `500 INTERNAL_ERROR "An unexpected error occurred"`: the
  caller's mistake billed as a server fault, with the message naming the option
  to change thrown away. These are now `400 VALIDATION_ERROR` carrying the
  explanation. Domain exceptions still map as before — several of them subclass
  `ValueError`, so they pass through ahead of it rather than collapsing into a
  generic 400.
- **`query(search_level="sections")` accepted `return_type` and ignored it**,
  always answering in sections — so `return_type="documents"` silently returned
  the wrong unit, the same class of defect as the silent chunk fallthrough one
  level down. It now rolls section hits up to their parent documents, scoring
  each document by its best-matching section (the roll-up `search_level="fused"`
  already did), over-fetching the section pool so `k` documents stay reachable
  when one document owns several of the top sections.
  `search_level="documents"` with a non-document `return_type`, and
  `search_level="sections"` with `"chunks"`/`"context"`/`"enriched"`, now raise
  `ValueError` instead of being ignored.
- **The `LocalVectorDB` API reference documented none of its 74 methods.** The
  class is assembled from mixins and defines nothing itself, so autodoc needed
  `:inherited-members:` to see anything — but `autodoc_default_options` carried
  an `"inherited-members": False` entry that looked like a harmless restatement
  of the default and in fact overrode the directive (Sphinx replaces a directive
  option with the config default whenever that default is not a string). The
  page rendered the class docstring and stopped. 52 methods were reachable only
  under their `BaseVectorDB` names and 22 — including `repair()`,
  `query_stream()`, `visualize_*()`, `rebuild_hierarchical_embeddings()` and the
  `sqlite_*` tuning calls — appeared nowhere in the docs at all. All 74 are now
  documented, and `tests/test_docs_api_coverage.py` guards both halves of the
  trap, neither of which produces a warning.
- `LocalVectorDB.section_vector_strategy` rendered its description as its type
  ("How sections are represented"), and `upsert_async()` documented a parameter
  named `upsert()`. Napoleon splits a property docstring on its first colon, and
  reads a bare line inside a `Parameters` block as another parameter.
- **Documentation builds clean** (123 Sphinx warnings → 0) and CI now enforces
  that with `sphinx-build -W`, on pull requests as well as pushes. The bulk were
  a numpydoc `Attributes` section and autodoc's `undoc-members` each describing
  the same dataclass field (fixed with `napoleon_use_ivar`), plus the same class
  being documented at both its re-export and its defining module, which left
  autodoc unable to resolve unqualified type references. Also adds intersphinx,
  so `str`/`int`/`Path`/`ndarray` in signatures link to their real docs instead
  of silently rendering as dead text (~230 such references).
- Raw-span section/document embeddings now size their pooling window to the
  encoder's own context (`num_ctx` / `max_input_tokens`) instead of a fixed
  ~24k-char (~8k-token) window. On a small-context encoder (e.g. a 2k-context
  local model) an over-long section is windowed and mean-pooled to represent it
  in full, rather than each 24k window overflowing and being silently truncated.
  The sentence-transformers and local HuggingFace providers now report their
  model's real context for this sizing too — previously they reported none, so
  the fixed fallback window was handed to (and silently truncated by) models
  with far smaller contexts.
- **Every section is now reachable by `return_type="sections"`.** The
  chunk→section roll-up read the single-valued `chunks.section_id`, which
  credits a chunk only to the section holding its midpoint — so any section
  without a midpoint owner was structurally unreturnable at any `k`: ~40% of
  sections at the default `chunk_size` on two measured corpora, including 26%
  of one corpus's *gold* sections. A new `chunk_sections` table records every
  section a chunk's span overlaps (chunks tile the document, so this reaches
  100% of sections by construction); existing databases self-heal on open with
  a single set-based SQL backfill — no re-embedding, no rebuild. Sections
  credited by the same chunk tie exactly and now rank by how much of the chunk
  they hold, which repairs reachability at no ranking cost (measured
  −0.0004/+0.0009 nDCG@10 on one corpus, +0.045/+0.047 on another, against
  midpoint-only). Centroid computation deliberately keeps midpoint ownership,
  which measures better there. The doctor's section-reachability line now
  measures the relation the roll-up actually reads, so it reports ~100% and a
  shortfall means a broken backfill rather than restating chunk geometry.
- **`query_async` now rolls chunk results up to sections exactly as `query()`
  does.** The async chunk-level path converted `return_type="sections"` to
  `"chunks"` and returned chunk results with no warning — sync/async parity is
  a stated contract, and this was the one query shape that broke it. Cursor
  and streaming queries (`query_cursor`, `query_stream`, and their async
  variants) now raise `ValueError` for `return_type="sections"` instead of
  silently answering in chunks: batched hydration cannot roll up (a section's
  best chunk can arrive in any batch), and refusing matches the existing
  `fused` precedent. Use `query()` for section results.
- **Sections owning no chunk had zero *vectors* in the section index** (the
  other half of the reachability defect above, fixed earlier in the cycle).
  Chunk→section attribution credits a chunk to the section holding its
  midpoint, so a section winning no midpoint got a zero vector — which scores
  zero against every query in a normalised index. Not a corner case at the
  defaults: with 500-token chunks over ~1,300-char median sections, 38% of one
  measured corpus's sections owned no chunk and 26% of its *gold* sections were
  unreachable. Such sections now fall back to overlap-based attribution;
  sections that already had a vector are untouched.
- **The OpenAI embedding provider never retried a rate limit.** Well-formed
  OpenAI error bodies (429 included) were re-raised as a bare `RuntimeError`
  with the status code discarded, so the retry classifier — which handles
  429/5xx correctly for httpx exceptions — matched nothing, and any ingest
  large enough to reach the tokens-per-minute limit died mid-run. Provider HTTP
  errors now carry their status and retry with the documented backoff.
- **Section-level results are reproducible run-to-run.** Sections repaired by
  the chunkless-section fix can share identical vectors and therefore tie
  exactly; tied results kept FAISS insertion order, which the threaded ingest
  assigns differently on every build. Section-level sorts now break ties on the
  section id. (Deliberately *not* applied to chunk→document scoring, where the
  incoming order carries BM25 rank and a lexicographic tie-break measurably
  destroys it.)
- **Partly-quoted queries no longer AND-join.** A query mixing a quoted phrase
  with plain words (`how is a "chunk of posts" defined`) required *every* word
  to appear, stopwords included — exactly what the plain-text path OR-joins to
  avoid. On one real corpus 20.6% of queries hit this, and nearly all of them
  got an empty keyword leg. Mixed queries now OR-join; the phrase is still
  matched as a phrase, and a fully-quoted query still binds exact.
- **`return_type="sections"` on a chunk-level search now returns `k` sections.**
  The chunk pool was fetched at exactly `k`, so chunks sharing a section
  collapsed into fewer sections than asked for — 34% of queries got a short
  list on one measured corpus. The pool is now over-fetched before grouping.
- **Embedding and reranking failures name the exception.** A message-less
  `ReadTimeout` — the most common real failure in a bulk ingest — used to
  surface as an error ending in a colon. The exception type is now the
  fallback, and batch-failure logs carry the request's shape (`64 texts,
  112,000 chars, ~32,000 tokens`) and the provider/model/endpoint identity,
  because an embedding failure is almost always about size and that is the one
  thing the exception never says.
- **`OpenAIEmbeddings` honours `base_url`** — it accepted and stored the
  parameter, then hard-coded `api.openai.com` at the call sites, silently
  sending "local server" traffic to OpenAI. Also in that cluster: a database
  now builds its embedding provider once from the *saved* config instead of
  constructor defaults first (opening a non-Ollama database no longer requires
  a running Ollama), a caller-supplied provider/model that conflicts with the
  saved one is reported before the saved one wins, and a persisted
  `embedding_dimension` that no longer matches the provider raises instead of
  being ignored.

### Security

- The server now logs a prominent startup warning when bound to a non-loopback
  interface with API authentication disabled (open read/write access). Defaults
  are unchanged; see the deployment docs for the hardening checklist.
- Bumped `setuptools` to `>= 83.0.0` (Dependabot GHSA — MANIFEST.in exclusion
  bypass via Unicode NFC/NFD collision) and, in the JavaScript SDK, forced
  `esbuild` to `>= 0.28.1` via an npm override (dev-only arbitrary file read in
  `esbuild serve` on Windows).

## [0.1.0rc1] - 2026-07-09

This is the first release candidate. The version published to PyPI is
`0.1.0rc1` (a pre-release); the final `0.1.0` above collected everything that
landed after it. The entries below are the initial feature set as of rc1.

### Added

- Document-first API with automatic position-aware chunking and reconstruction
- SQLite + FAISS dual storage backend for documents, metadata, and vectors
- Unified `query()` interface supporting vector, keyword (FTS5), and hybrid search
- Strongly typed metadata schema with TEXT, INTEGER, REAL, BOOLEAN, DATE, JSON types
- Pluggable embedding providers: Ollama, OpenAI, Google, Jina, HuggingFace, Sentence Transformers
- Pluggable reranker providers: Jina, Sentence Transformers, HuggingFace
- Multiple chunking strategies: sentences, tokens, words, paragraphs, sections, code blocks
- SQL-like query builder for metadata filtering
- FastAPI HTTP server with multi-database management
- API key authentication with permission levels (read-only, read-write)
- Rate limiting, CORS, and security headers middleware
- SSE streaming for query results
- File upload with text extraction via [all2md](https://all2md.readthedocs.io/):
  a single `All2MdExtractor` covering 20+ document formats and 200+ source/text
  formats, emitting Markdown to preserve document structure (headings, tables,
  lists) for better chunk boundaries. The plugin interface (`BaseExtractor`,
  `ExtractorRegistry`, the `localvectordb.file_extractors` entry-point group)
  supports custom extractors.
- Hardened extraction defaults for untrusted uploads (remote fetching and local
  file access disabled, HTML dangerous elements stripped, attachments skipped;
  file-size and ZIP-bomb guards), configurable via the `[extraction]` server
  config section and `LVDB_EXTRACTION_*` environment variables.
- `file-extraction-ocr` extra for OCR of scanned PDFs (Tesseract).
- Section detection and the `sections` chunking strategy ignore Markdown headers
  inside fenced code blocks, so code snippets don't create spurious sections.
- Raw-span section vectors for hierarchical databases: a new
  `section_vector_strategy` option (`"rawspan"` | `"centroid"`) controls how a
  section is represented in the section index. `"rawspan"` embeds the section's
  actual text (window-mean-pooled for over-long spans) instead of averaging its
  chunk vectors, which retrieves better on real, section-structured documents.
  New hierarchical databases default to `"rawspan"`; databases created before this
  option existed keep `"centroid"`, and the resolved value is persisted per
  database. Off by default (requires `hierarchical_embeddings=True`).
- `search_level="fused"` retrieval: blends chunk retrieval with section (raw-span)
  retrieval via relative-score fusion, tunable with a `section_weight` scalar
  (0 = chunk-only, 1 = section-only; default 0.65). Supports `return_type`
  `"documents"` (the measured win) and `"sections"`. Local databases only for now;
  remote/streaming raise a clear error. The default chunk-only retrieval path is
  unchanged.
- Document comparison and nearest-neighbor endpoints
- LLM-based fact-checking module
- Cursor-based pagination for async query results
- Backup and restore with incremental and point-in-time recovery
- Database migration engine and schema versioning
- SQLite tuning profiles for different workloads
- MCP (Model Context Protocol) server integration
- CLI tool (`lvdb`) for database management, server control, and configuration
- Read-only multi-worker read fan-out: a `mmap_index` setting memory-maps the
  FAISS index (`IO_FLAG_MMAP`) so many workers share one page-cached copy instead
  of each loading a private, RAM-resident copy. A memory-mapped database is
  read-only and refuses writes. A shared cachelib/Redis registry coordinates the
  set of database names across workers. The deployment model is single-writer:
  route all writes to one writer process (`mmap_index = false`).
- The FAISS index file is rewritten only when the in-memory index has actually
  changed, so a database that only served reads is never re-persisted (and, under
  read fan-out, never races another worker on the shared index file) on close or
  idle-eviction.
- A hardened `Dockerfile` (pinned base image, dependencies isolated in a virtualenv,
  non-root user, `HEALTHCHECK` against `/api/v1/health`), built and booted in CI on every
  pull request so it cannot drift.
- Comprehensive test suite with 85%+ coverage requirement
- End-to-end release-qualification suite (`scripts/e2e/`) exercising real
  embedding backends (Ollama, Sentence Transformers) and real PDF/DOCX/XLSX/
  HTML/Markdown documents against the library, file ingestion, HTTP server,
  and CLI
- Sphinx documentation with autodoc
- CI/CD pipeline with linting, type checking, security scanning, and tests

### Changed

Breaking HTTP/API contract changes finalized before the v0.1.0 freeze (relevant
to anyone tracking the pre-release):

- **HTTP routes**: all per-database endpoints moved under `/api/v1/databases/{db_name}/...`
  (for example `/api/v1/databases/{db_name}/query`). Global endpoints
  (`/api/v1/databases`, `/api/v1/search`, `/api/v1/embeddings`, `/api/v1/health`,
  `/api/v1/system/resources`, `/api/v1/upload/...`) are unchanged. Database names are
  now namespaced under `/databases/`, so no database names are reserved.
- **Global search**: `POST /api/v1/search` now returns the per-database map under
  `results_by_database` (was `results`).
- **Default `vector_weight` changed from `0.7` to `0.5`** for hybrid search (the default
  `search_type`). This changes hybrid ranking for callers who do not pass `vector_weight`
  explicitly. Once T1.1's relative-score fusion made `vector_weight` an actual blend, an
  even weighting measured better on *both* evaluation corpora — SciFact `frequency_boost`
  nDCG@10 0.6940 → 0.7090 (+2.2% relative) and NFCorpus 0.3298 → 0.3367 (+2.1%) — where it
  is also the best configuration in the entire sweep. Pass `vector_weight=0.7` to restore
  the previous behaviour. Applies to the Python API, HTTP API, MCP server, and the
  `lvdb db <name> search --vector-weight` CLI default.
- **Default server port** changed from `5000` to `8000` (5000 collides with the macOS
  AirPlay Receiver).
- Single-document delete (`DELETE /api/v1/databases/{db_name}/documents/{doc_id}`) is
  idempotent — deleting a missing document succeeds instead of erroring.

### Removed

- Remote/HTTP fact-checking: the `/factcheck` HTTP endpoints and the
  `RemoteVectorDB.fact_check()` client method are removed. Fact-checking ("reverse RAG")
  remains available as a local-only feature via the `FactChecker` class over
  `LocalVectorDB`.

### Fixed

- `server.rate_limit_storage_uri` was defined but never read, so slowapi silently fell
  back to a per-process in-memory store and the effective limit was N× the configured
  one under N workers. It is now passed to the limiter, and a shared store (e.g. Redis)
  enforces one limit across all workers.
- Backups could capture a mutually inconsistent pair of stores. SQLite and the FAISS
  index are copied separately, so a write landing between the two could produce a backup
  whose SQLite rows referenced vectors absent from the copied index (dangling rows,
  which require re-embedding to recover). Passing the live database —
  `BackupManager(path, db=db)` — now holds its write lock and flushes the index for the
  duration of the snapshot. The path-only form is unchanged and is documented as safe
  only for a quiescent or closed database.
- Persisting the index could fail with `PermissionError` on Windows. `os.replace` is the
  final step of writing the index, and it intermittently fails with `[WinError 5] Access
  is denied` when any process holds a transient handle on the target — a virus scanner,
  the search indexer, or simply another process reading the index (a backup copying it,
  a reader worker opening it). The error propagated out of `save()`/`close()`, leaving
  the index unwritten while SQLite had already committed. It is now retried with bounded
  exponential backoff.
- `PATCH /databases/{db}/documents/{doc_id}` conflated "nothing to update" with
  "document not found", inverting both outcomes. `update()` returns `False` for a no-op
  and raises `DocumentNotFoundError` for a missing document, but the route reported the
  no-op as `404 DOCUMENT_NOT_FOUND` — on a document that exists — while the missing
  document raised past the route into the generic 500 branch (`DocumentNotFoundError`
  has no mapping in `standardize_error_response`). A no-op is now `200 {"updated": false}`
  and a missing document is `404 DOCUMENT_NOT_FOUND`.
- `RemoteVectorDB.update()` / `update_async()` swallowed a 404 into a `False` return, so a
  missing document was indistinguishable from "no updates needed" and the remote backend
  diverged from `LocalVectorDB.update()`, which raises `DocumentNotFoundError`. Both now
  raise, and `False` means only "no updates needed". The JavaScript SDK's
  `database.update()` is reconciled the same way (it now throws `DocumentNotFoundError`
  instead of resolving `{updated: false}`). The `update()`/`update_async()` contract is
  now stated on the abstract base so both backends are held to it.
- `RemoteVectorDB.update()` / `update_async()` short-circuited on `if not content and not
  metadata`, so `content=""` (clear a document) and `metadata={}` were silently dropped
  client-side and never reached the server. They now test against `None`.
- The `update_document` MCP tool discarded `update()`'s return value and always reported
  success, so an agent could not distinguish "my edit landed" from "nothing changed". It
  now returns an `updated` flag.

Issues found during pre-release end-to-end qualification with real embedding
providers (the mocked test suite could not catch these):

- Server search endpoints (`/query`, `/search/*`, `/query-multi-column`,
  `/query-builder`, global `/search`) called sync query/embedding paths on the
  event loop, so vector and hybrid search failed with every real embedding
  provider; they now use the async query APIs
- SSE streaming endpoint (`/query/stream`) did not await `query_cursor_async`
  and iterated the cursor incorrectly
- Server-side database creation forwarded unset `api_key`/`base_url` to
  embedding providers that don't accept them, breaking `ollama` database
  creation over HTTP
- `/documents/count` and document listing called `db.count()` with a
  nonexistent `where` keyword and always returned HTTP 500
- Server config, request validation, and CLI rejected every embedding
  provider except `ollama`/`openai`; they now accept any provider registered
  with `EmbeddingRegistry`
- `$contains`/`$not_contains` metadata filters on JSON fields generated SQL
  with two placeholders but bound one parameter, crashing every such filter
- JSON metadata fields were returned as raw serialized strings from
  `get()`/`filter()`, which also broke partial `update()` on any document
  with a JSON-typed field
- `/health` performed an inline Ollama check with a 60-second timeout and
  three retries (minutes-long hangs when Ollama was down); it now uses a
  single 2-second attempt
- Default Ollama base URL changed from `localhost` to `127.0.0.1` (matching
  Ollama's default bind address) to avoid a ~2.5 s IPv6 resolution stall per
  connection on Windows
- README/docs metadata-filter examples used unsupported operator spellings
  (`contains`, `>=`) instead of `$contains`/`$gte`

Pre-release consistency fixes:

- `query(filters=...)`, `query_multi_column(filters=...)`, and
  `nearest_neighbors(filters=...)` silently returned no matches for filter
  fields not in the metadata schema or unsupported operators; they now raise
  `MetadataFilterError` (a `DatabaseError`/`ValueError` subclass) up front,
  matching `filter(where=...)` behavior
- Invalid filter specs over HTTP returned 500 `DATABASE_ERROR`; they now
  return 400 `INVALID_FILTER` (a client error), the Python client raises
  `MetadataFilterError` for it, and clients no longer waste retries on them
- `upsert()` silently dropped metadata fields not in the metadata schema; it
  now logs a warning naming the dropped fields
- `lvdb db <name> <cmd> --help` required the database (and DB folder) to
  exist; the database is now opened lazily on first use so help always works
- A malformed or invalid config file crashed the CLI with a raw traceback; it
  now prints a friendly error and exits with the configuration-error code (2)
- `lvdb db <name> add <file>` assigned generated `doc_N` ids while the
  library's `upsert_from_file()` used the filename stem; the CLI now also
  defaults file inputs to the filename stem (repeated stems in one batch fall
  back to generated ids)

Final pre-release contract hardening (packaging, API, HTTP, and CLI surfaces
frozen for v0.1.0):

- **Packaging**: a base `pip install localvectordb` crashed on import because
  `sqlite_tuning` imported `psutil`, which was only declared in the `[server]`
  and `[benchmark]` extras; `psutil` is now a core dependency. `click` is
  declared explicitly in `[server]`. Importing `localvectordb_server` (and the
  `lvdb` console script) without the `[server]` extra now raises a clear error
  naming the extra instead of a bare `ModuleNotFoundError`.
- **Factory**: `VectorDB(name, "http://...", timeout=...)` raised `TypeError`
  because the remote client's parameter is `request_timeout`; the factory now
  documents and forwards the real remote parameter names.
- **Remote comparison parity**: `RemoteVectorDB.compare_documents_detailed()`
  and `pairwise_similarity_matrix()` returned raw dicts (and the server
  serialized fields the result dataclass never had, dropping the real data);
  they now return the same `DocumentComparisonResult` /
  `DocumentSimilarityMatrix` dataclasses as `LocalVectorDB`. `nearest_neighbors`
  gained the `score_threshold`/`filters` parameters on the remote client and
  server. Removed the remote-only legacy `hybrid_query()`/`keyword_search()`.
- **HTTP contract**: rate-limit (429) responses now use the standard
  `{"error": {...}}` envelope (the stock slowapi body broke the client);
  `query_builder` path is hyphenated (`query-builder`); `PATCH` added to the
  default CORS methods; `DELETE` on a missing database returns 404 instead of
  200; SSE error payloads no longer leak internal exception text.
- **CLI**: failing `tuning`/`maintenance`/`backup verify`/`backup pitr`/
  `migrate`/`db get`/`db delete`/`delete` invocations now exit non-zero;
  machine-output is unified on `--format/-f {table,json}` (with `-j` as a
  shortcut for `--format json`), and `-o/--output` reserved for output files;
  `--help` works
  without a config file and `lvdb serve` falls back to localhost defaults;
  `config init --cors-origins` now persists; and `lvdb db <name> add` errors on
  a path-like argument that does not exist instead of silently storing it as
  text (use `--text` to force literal text).
