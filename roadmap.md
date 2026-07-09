# LocalVectorDB Roadmap

**Status:** pre-v0.1.0 · Last updated 2026-07-09

## Positioning

LocalVectorDB is **agent-native document memory**: a zero-infrastructure library that lets
an agent search a corpus, *read* what it found at document / section / line granularity,
find related material, and check its own output for grounding — offline, in one process.

This is deliberately **not** a bid to be a general-purpose vector store. That market is
commoditized and well-funded: `sqlite-vec` owns SQLite-native vectors, `pgvectorscale`
owns disk-based scale via StreamingDiskANN, LanceDB owns the embedded columnar niche, and
every major engine now ships an embedded mode. Competing there on vectors/sec is a losing
pitch.

What is defensible is the **composition**, and the depth of the agent workflow on top of it:

| Layer | Commodity? | Ours to win? |
|---|---|---|
| SQLite + FAISS storage | Yes — txtai ships the identical architecture | No |
| Hybrid FTS5 + vector search | Yes — RRF hybrid is native nearly everywhere | No |
| Hierarchical doc→section→chunk retrieval | Yes — LlamaIndex `AutoMergingRetriever` (2023) | No |
| Reverse-RAG fact-checking as a capability | Yes — Ragas, DeepEval, TruLens, HHEM, MiniCheck | No |
| Shipping *an* MCP server | Yes — every major vector DB shipped one in 2025 | No |
| **MCP tools that expose sections, portions, provenance, and fact-check** | **No** | **Yes** |
| **Document-first contract: ingest → retrieve → read the actual source span** | **Partly** | **Yes** |
| **The whole workflow, offline, zero infra, one dependency** | **No competitor bundles it** | **Yes** |

The scale ceiling of a flat exact FAISS index (~10⁵–10⁶ vectors, RAM-resident) is
**acceptable and will be stated plainly**. Agent memory over a corpus does not need 100M
vectors. Honesty about the ceiling is worth more than a half-built ANN path.

---

## v0.1.0 — Correctness, durability, honesty

The pre-tag bar is *not* "feature complete." It is: **nothing is silently wrong, and
nothing in the README is untrue.** Every item below is either a data-integrity defect, a
score-semantics change that is free now and breaking forever after, or a false claim.

### Tier 0 — Data integrity (release blockers)

- [ ] **Monotonic FAISS id allocation.** `_core.py:565` derives new ids from `index.ntotal`,
      but `remove_ids` *decrements* `ntotal`. Any delete or update re-issues live ids.
      Confirmed via the public API: after a plain `upsert` of an existing document id, two
      documents share a `faiss_id`; `_search.py:1000` hydrates via `WHERE faiss_id IN (...)`,
      so one vector scores two documents and a query for a document's own text returns a
      *different* document ranked above it.
      Fix with a persisted monotonic counter in database metadata — **not** `MAX(faiss_id)+1`,
      which breaks when `remove_ids` fails and is swallowed.
      Sibling sites with the identical bug: `_metadata.py:63`, `_metadata.py:73`,
      `_ingest.py:1647`, `_ingest.py:1697` (section and document indices).
- [ ] **Dual-store invariant test.** Nothing anywhere asserts `index.ntotal == COUNT(chunks)`.
      For a database split across SQLite and FAISS this is *the* invariant. Add it as a
      fixture-level check, plus an explicit delete-then-search and update-then-search
      correctness test.
- [ ] **Stop swallowing `remove_ids` failures.** `_crud.py:490` catches and logs. HNSW and LSH
      do not support `remove_ids`, so choosing those index types means deletes silently do
      not happen and vectors orphan forever. Either fail loudly or refuse to construct a
      deletable database on an index type that cannot delete.
- [ ] **Atomic index persistence.** `faiss.write_index` writes in place (`_core.py:944`).
      A crash mid-save corrupts the index. Write to a temp file, `fsync`, then rename.
- [ ] **Startup reconcile + repair.** Nothing detects or repairs SQLite↔FAISS divergence.
      Add an integrity check on open (cheap: `ntotal` vs chunk count) and an explicit
      `lvdb db <name> repair` that rebuilds the index from SQLite.
- [ ] **Transactional boundary.** Ingest mutates the in-RAM index *inside* the SQLite
      transaction and a rollback does not undo it (`_ingest.py:1292`); delete commits SQLite
      before touching FAISS (`_crud.py:477` → `:483`). Define and document a single ordering
      with a recovery path for a crash at each step.

### Tier 1 — Silently-wrong retrieval (free to fix now, breaking later)

Every one of these changes score semantics. Pre-1.0 they are free; post-1.0 they are a
migration. **A minimal eval harness lands first** (see Tier 3) so these can be measured
rather than hoped at.

- [ ] **Reciprocal Rank Fusion as the default hybrid.** Today: `w·vector + (1-w)·keyword`
      (`_search.py:543`, `:856`, `:2604`) over two un-normalized, corpus-dependent scales —
      `1/(1+L2)` against `1-exp(bm25_rank)`. A chunk found by only one retriever is
      zero-filled for the other, so a perfect keyword match is structurally capped at `0.3`.
      RRF exists to sidestep scale entirely. Keep weighted-sum as an opt-in.
- [ ] **Rerank over-fetch.** `_search.py:398-410` reranks the *already-truncated* top-k, so a
      cross-encoder can never surface a document first-stage ranked at k+1 — which is the
      entire purpose of reranking. Add `rerank_k` (default ~5×k).
- [ ] **Adaptive over-fetch under selective filters.** Filtering is post-hoc over a fixed
      candidate pool, hard-capped at 100 for hybrid (`_search.py:1232`). `k=10` with a filter
      matching 0.1% of the corpus silently returns near-nothing. Over-fetch and retry, or
      pre-filter with a FAISS `IDSelector`.
- [ ] **Consistent embedding normalization.** `_core.py:875` comments *"Assuming normalized
      embeddings"*; `normalize_L2` appears nowhere in the codebase. Providers disagree —
      Ollama and OpenAI default `normalize=False` (`embeddings.py:462`, `:661`), the rest
      default `True`. With `IndexFlatIP` + Ollama, raw inner products fall outside `[-1,1]`
      and `(d+1)/2` clamps them to `1.0`: top results saturate to a tied score.
- [ ] **Normalize hierarchical centroids; fix the metric mismatch.** Section and document
      vectors are raw `np.mean` centroids (`_ingest.py:1127`, `:1135`), never re-normalized,
      indexed in a hardcoded `IndexFlatL2` (`_core.py:510`) — so if the main index is IP, the
      section similarity conversion at `_search.py:1312` uses the wrong formula.
- [ ] **Prune document scoring from 11 strategies to 3.** Keep `best`, `average`,
      `frequency_boost`. The other eight are overlapping heuristics with unvalidated magic
      constants; `worst` (minimum chunk score) has no plausible retrieval use. Breaking
      change — free today.
- [ ] **Fix `reconstruct_document` for the default chunker.** Chunk spans from the
      `sentences` chunker leave gaps — `(0,89), (90,179)` leaves index 89 uncovered — and the
      function fills a char array by position, so every inter-chunk separator and all trailing
      whitespace becomes `""`. An 8-chunk document silently loses 9 characters. Either make
      it byte-exact or stop calling it "perfect."

### Tier 2 — Honesty (cheap; prevents launch-day damage)

- [ ] **Retract the multi-worker claim** (`README.md:609-618`). Each uvicorn worker loads its
      own FAISS index into RAM and never reloads it — no mtime check, no invalidation, no
      pub/sub. Redis stores a *list of database names*, not vectors or a write log. Worker A's
      upsert is invisible to worker B for up to `db_timeout` (default 1h). Worse, `save()`
      rewrites the whole index guarded only by a *process-local* lock; the cross-process file
      lock is taken on create/load/delete but never on the data path, so concurrent writers
      race and a stale worker can overwrite a newer index. **Document single-writer.**
- [ ] **Remove or wire `rate_limit_storage_uri`.** Defined at `config.py:423`, read nowhere.
      `Limiter` (`app.py:351`) never receives it, so slowapi falls back to per-process memory:
      the effective limit is N× the configured value under N workers, and pointing it at Redis
      does nothing.
- [ ] **Ship a real Dockerfile** (non-root, pinned, healthcheck, CI-built) or delete the
      Docker section. There is no Dockerfile in the repo today.
- [ ] **State the scale ceiling in the README.** Flat exact index, RAM-resident: ~3 GB per 1M
      × 768-dim vectors. Say so.
- [ ] **Backup consistency.** SQLite is snapshotted via the online backup API but FAISS is a
      plain file copy with no shared lock (`backup.py:476-480`), so a concurrent write yields a
      mutually inconsistent backup. Either quiesce writes or document the requirement.

### Tier 3 — Minimal evaluation harness (prerequisite for Tier 1)

Nothing in the repo can currently tell you whether a retrieval change helped. `conftest.py:183`
mocks the FAISS index with `np.random.random()` distances and `np.random.randint()` indices;
`MockEmbeddings` (`embeddings.py:1781`) seeds on a SHA-256 of the text, so semantically related
strings get *orthogonal* vectors. No test in the suite **can** assert that the right document
ranks first.

- [ ] A few hundred labeled `query → relevant doc` pairs (BEIR subset such as SciFact or
      NFCorpus, or a hand-built set over the project's own docs).
- [ ] `recall@k` and `nDCG@10`, reported per configuration.
- [ ] Real-embedding tests in CI using a small cached SentenceTransformers model.
- [ ] Land this **before** the Tier 1 changes, and record a baseline against which RRF,
      rerank over-fetch, and normalization are each measured.

---

## v0.2.0 — Trust: measurement, concurrency, operations

Making the thing provable and operable.

- **Retrieval-quality regression gating.** Promote the Tier 3 harness into CI with a
  threshold. A PR that lowers nDCG@10 fails.
- **`scripts/e2e/` in CI.** It is the only place real retrieval correctness is checked, and
  because it is not in CI it silently drifts from renamed routes and flags.
- **Concurrency and crash-recovery tests.** Today's "concurrency" tests run
  `ThreadPoolExecutor` against a mocked FAISS index and a mocked DB, asserting only "no
  exception." Add real fault injection: kill mid-write, truncate the index, reopen.
- **Enforce single-writer.** An advisory cross-process lock that *refuses* a second writer
  rather than silently corrupting. Read-only workers may attach.
- **Index generation counter + reload.** Cheap cross-worker freshness: bump a counter on
  save, check it on read, reload if stale. Unlocks safe multi-worker *reads*.
- **Observability.** `/ready` (distinct from `/health`, and actually checking `db_manager`),
  Prometheus `/metrics` (query latency histograms, error rate, active DBs), OpenTelemetry
  spans on the existing `request_id`.
- **Per-database API key scoping.** `validate_key_with_permissions` takes no database
  argument (`keymanager.py:528`) — any read-write key works on every database on the server.
- **Re-embedding migration.** `migration.py` handles metadata schema only. Changing the
  embedding model or dimension has no path today; the model is baked in at creation and
  treated as immutable. Add `lvdb db <name> reembed --model ...`.
- **`QueryOptions` dataclass.** `query()` takes 15 keyword params, copy-pasted across five
  methods (`base.py:96-177`), which already caused a documented Liskov drift on
  `query_multi_column` / `query_stream`.

---

## v0.3.0 — The moat: agent-native depth

Where the differentiation actually lives. Research is explicit that the only remaining MCP
novelty is *tools that expose hierarchical retrieval and fact-checking specifically* — a
generic `query(collection, k)` server is a commodity.

- **Provenance-grade results.** Every result carries document id, character span, section
  path, and a stable citation token. An agent should be able to quote and cite without a
  second round-trip.
- **MCP tools for the differentiated surface**, not just search: `get_section`,
  `get_outline`, `get_portion(lines|range|chunk)`, `find_related`, `fact_check`. Read-only
  by default (already correct).
- **Document outline in metadata.** `all2md` already yields structure on extraction — stash
  the outline at ingest so an agent can navigate a document without reading it.
- **Contextual Retrieval** (Anthropic-style chunk prefixing). Independent reproductions show
  5–15% gains (not the headline 35–67%), and it is cheap with prompt caching. Gate on the
  eval harness.
- **True coarse-to-fine hierarchy.** Today section hits and chunk hits are independent code
  paths — it is not actually two-stage. Make section hits constrain the chunk search.
- **Complexity router for agentic retrieval.** Single-shot for simple queries, iterative for
  multi-hop. Iterative retrieval costs +59–97%; route, don't default.

---

## v0.4.0 — Retrieval frontier

Only what has demonstrably won. Gated on the eval harness; anything that does not move
nDCG on a real dataset does not ship.

- **Late-interaction / multi-vector (ColBERT).** The strongest infrastructure signal in the
  space: Qdrant, Weaviate, Milvus, Vespa, Elasticsearch, and LanceDB all added native
  multi-vector within ~18 months. Becoming table stakes.
- **Matryoshka truncate + full-vector rescore.** Providers already expose truncation
  (`embeddings.py:571`); the two-pass rescore is missing.
- **MMR / embedding-space diversity.** The current "diversity" in `query_builder.py:1319` is a
  metadata-field boost, not MMR.
- **Query expansion / HyDE**, behind the eval gate.
- **Learned sparse (SPLADE)** as an optional third retriever. BM25 remains the default —
  SPLADE needs ~10× compute and a GPU.

Explicitly **not** pursuing: semantic chunking. The Vectara/NAACL-2025 critique and a Feb-2026
benchmark both find plain recursive splitting beats it; embedding-model quality dominates the
chunker.

---

## v0.5.0 — Scale, within our lane

Only after the above. The goal is to raise the ceiling for document memory, not to chase
100M-vector SLAs.

- **ANN that actually supports deletes.** IVF with an id map, or HNSW with rebuild-on-delete.
  Today HNSW and LSH are selectable but their deletes silently no-op.
- **int8 / binary quantization with rescoring.** The 32× claims only hold *with* rescoring
  (~95% recall retention).
- **mmap / on-disk index.** `faiss.read_index` currently loads the whole index into RAM with
  no `IO_FLAG_MMAP`.
- **Incremental persistence.** `_save_internal` rewrites the entire index file after every
  `upsert` (`_ingest.py:438`), so bulk-loading 100k documents one call at a time is 100k
  full-index rewrites.

---

## v0.6.0 — Multimodal

- Image embeddings; ColPali/ColQwen-style OCR-free visual document retrieval, now de-facto
  SOTA for PDF-heavy corpora.
- Multimodal extraction wired through the existing `all2md` path.

## v0.7.0 — Knowledge layer

- **Autoclassifier**: sample the latent space, have an LLM name the clusters, write them back
  as metadata.
- **LLM-generated metadata** at ingest.
- **Lightweight graph layer** in the LightRAG/LazyGraphRAG mold. Explicitly *not* Microsoft
  GraphRAG — quality held up but ~$33k indexing cost killed it.
- **Anchor-based chunking** (`etc/anchor-chunking-plan-090925.md`).

## v0.8.0 — 1.0 candidate

- **API freeze** and a written deprecation policy.
- **Namespace the DB object.** ~80 public callables today, including four `visualize_*` and
  ten `sqlite_*` methods hung off the top level. Move to `db.viz.*`, `db.tuning.*`.
- **mypy strict.** Currently `disallow_untyped_defs = false`.
- **Performance regression gating** in CI. `benchmark.yml` is explicitly report-only.
- **Resolved concurrency story**: single-writer + read replicas, or a real write coordinator.
- **Security**: per-database scoping, audit log, secure-by-default config (today
  `require_api_key = False` and `cors_allowed_origins = "*"`).

---

## Deferred / carried forward

From `todo.md`, still open and not yet scheduled above:

- `DatabaseStats` dataclass for `get_stats()` (currently `Dict[str, Any]` on both backends).
- ABC-enforce `query_multi_column[_async]` and `query_stream[_async]` — blocked on
  harmonizing the `return_type` Literals that drifted (`database/base.py:554`).
- Honor upload `extractor_kwargs` on remote without letting clients override hardened
  security defaults.
- Return-shape parity: remote `compare_documents_detailed` / `pairwise_similarity_matrix`
  return raw dicts where local returns dataclasses.
- Async batch `get` ignores `missing_ids` (no `DocumentNotFoundError`), unlike sync.
- Fact-checking over `RemoteVectorDB` (local-only today); expose via REST.
- OpenRouter embedding provider.
- Response-key renames (`name`↔`database`, `total_results`/`count`/`total_count`) — high
  client coupling, low value.

## Non-goals

- Competing with `sqlite-vec`, `pgvectorscale`, or LanceDB on storage or raw scale.
- 100M+ vector, sub-50ms-SLA workloads.
- Being a general-purpose vector database. Storage is an implementation detail; the
  document workflow is the product.
