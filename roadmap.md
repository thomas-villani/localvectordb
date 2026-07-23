# LocalVectorDB Roadmap

This roadmap describes what we plan to build next. It is a direction, not a
commitment: items may move between releases, and priorities shift with feedback.
For what has already shipped, see [CHANGELOG.md](CHANGELOG.md).

## What LocalVectorDB is

LocalVectorDB is **agent-native document memory**: a zero-infrastructure library
that lets an agent search a corpus, *read* what it found at document / section /
line granularity, find related material, and check its own output for grounding —
offline, in one process.

It is deliberately not a general-purpose vector store competing on raw
vectors/sec. The value is the composition and the depth of the agent workflow on
top of it: a document-first contract (ingest → retrieve → read the actual source
span), hierarchical document→section→chunk retrieval, reverse-RAG fact-checking,
and MCP tools that expose sections, portions, provenance, and fact-checking — the
whole workflow, offline, with one dependency.

**Scale ceiling, stated plainly.** The default index is a flat, exact,
RAM-resident FAISS index (roughly 3 GB per 1M × 768-dim vectors), comfortable to
about 10⁵–10⁶ vectors. Agent memory over a corpus does not need 100M vectors, and
we would rather be honest about the ceiling than ship a half-built approximate
path. Raising that ceiling within our lane is planned (see v0.5.0), but chasing
100M-vector, sub-50ms SLAs is not.

---

## v0.2.0 — Trust: measurement, concurrency, operations

Making the thing provable and operable.

- **Retrieval-quality regression gating in CI.** The evaluation harness
  (`benchmarks/eval_retrieval.py`, with `--check`) exists; wire it into CI with a
  threshold so a pull request that lowers nDCG@10 fails.
- **Run the end-to-end suite in CI.** `scripts/e2e/` is where real retrieval
  correctness against live embedding backends is checked, and because it is not
  in CI it silently drifts from renamed routes and flags.
- **Concurrency and crash-recovery tests.** Add real fault injection — kill
  mid-write, truncate the index, reopen — rather than only asserting "no
  exception" against mocked stores.
- **Enforce single-writer.** An advisory cross-process lock that *refuses* a
  second writer rather than risking a corrupt index. Read-only workers may attach.
- **Index generation counter + reload.** Bump a counter on save, check it on read,
  reload if stale — cheap cross-worker freshness that unlocks safe multi-worker
  *reads*.
- **Observability.** A `/ready` endpoint distinct from `/health` (actually
  checking the database manager), Prometheus `/metrics` (query-latency
  histograms, error rate, active databases), and OpenTelemetry spans on the
  existing request id.
- **Per-database API key scoping.** Today a read-write key works on every database
  on the server; scope keys to specific databases.
- **Re-embedding migration.** The embedding model and dimension are baked in at
  database creation and treated as immutable. Add `lvdb db <name> reembed
  --model ...` to change them.
- **`QueryOptions` dataclass.** `query()` takes many keyword parameters,
  copy-pasted across several methods; consolidate them to prevent signature drift.

---

## v0.3.0 — Agent-native depth

Where the differentiation lives: tools that expose hierarchical retrieval and
provenance specifically, not a generic `query(collection, k)` server.

- **Provenance-grade results.** Every result carries a document id, character
  span, and section path (it largely does today); add a stable citation token so
  an agent can quote and cite without a second round-trip.
- **A fuller MCP navigation surface.** Building on the read-only tools already
  shipped (`grep_documents`, `list_prefixes`, portion-aware `get`, `patch`), add
  `get_section`, `get_outline`, and `find_related` so an agent can walk a document
  by structure.
- **Document outline in metadata.** Extraction already yields structure; stash the
  outline at ingest so an agent can navigate a document without reading all of it.
- **Contextual Retrieval** (chunk prefixing). Independent reproductions show
  ~5–15% gains, and it is cheap with prompt caching. Gated on the eval harness.
- **True coarse-to-fine hierarchy.** Section hits and chunk hits are currently
  independent paths blended by score; make section hits actually *constrain* the
  chunk search for genuine two-stage retrieval.
- **Complexity router for agentic retrieval.** Single-shot for simple queries,
  iterative for multi-hop. Iterative retrieval is costly, so route rather than
  default.

---

## v0.4.0 — Retrieval frontier

Only what has demonstrably won, gated on the eval harness: anything that does not
move nDCG on a real dataset does not ship.

- **Late-interaction / multi-vector (ColBERT).** Now table stakes across the major
  engines; the strongest infrastructure signal in the space.
- **Matryoshka truncate + full-vector rescore.** Providers already expose
  truncation; the missing piece is the two-pass rescore.
- **MMR / embedding-space diversity.** Real max-marginal-relevance in embedding
  space, distinct from the current metadata-field diversity boost.
- **Query expansion / HyDE**, behind the eval gate.
- **Learned sparse (SPLADE)** as an optional third retriever. BM25 stays the
  default — SPLADE needs far more compute and a GPU.

Explicitly **not** pursuing semantic chunking: current research finds plain
recursive splitting beats it, and embedding-model quality dominates the chunker.

---

## v0.5.0 — Scale, within our lane

Raising the ceiling for document memory — not chasing 100M-vector SLAs.

- **ANN that actually supports deletes.** IVF with an id map, or HNSW with
  rebuild-on-delete. Today HNSW and LSH are selectable but their deletes no-op.
- **int8 / binary quantization with rescoring.** The large size reductions only
  hold *with* rescoring (~95% recall retention).
- **Incremental persistence.** The index file is rewritten wholesale after a bulk
  upsert; bulk-loading many documents one call at a time is many full-index
  rewrites.

---

## v0.6.0 — Multimodal

- Image embeddings; ColPali/ColQwen-style OCR-free visual document retrieval, now
  a strong option for PDF-heavy corpora.
- Multimodal extraction wired through the existing extraction path.

## v0.7.0 — Knowledge layer

- **Autoclassifier**: sample the latent space, have an LLM name the clusters, and
  write them back as metadata.
- **LLM-generated metadata** at ingest.
- **Lightweight graph layer** in the LightRAG/LazyGraphRAG mold (explicitly *not*
  full GraphRAG-style indexing, whose cost is prohibitive).
- **Anchor-based chunking.**

## v0.8.0 — 1.0 candidate

- **API freeze** and a written deprecation policy.
- **Namespace the database object.** The many top-level `visualize_*` and
  `sqlite_*` methods move to `db.viz.*` / `db.tuning.*`.
- **`mypy --strict`.**
- **Performance regression gating** in CI (currently report-only).
- **A resolved concurrency story**: single-writer plus read replicas, or a real
  write coordinator.
- **Security hardening**: per-database scoping, an audit log, and secure-by-default
  configuration.

---

## Smaller carried-forward items

Not yet scheduled above:

- A `DatabaseStats` dataclass for `get_stats()` (currently a plain dict on both
  backends).
- Honor upload `extractor_kwargs` on the remote backend without letting clients
  override hardened extraction-security defaults.
- Async batch `get` should raise `DocumentNotFoundError` for missing ids, matching
  the sync path.
- Expose `chunk_delimiter` (the delimiter chunking strategy) over the HTTP server
  and remote client, matching the local library.
- Response-key naming consistency across endpoints (low value, high client
  coupling — tracked, not urgent).

## Non-goals

- Competing with `sqlite-vec`, `pgvectorscale`, or LanceDB on storage or raw
  scale.
- 100M+ vector, sub-50ms-SLA workloads.
- Being a general-purpose vector database. Storage is an implementation detail;
  the document workflow is the product.
</content>
