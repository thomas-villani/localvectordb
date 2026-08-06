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

## PRE-RELEASE — defaults to fix before the first cut

Findings from the retrieval study (`experiments/`, §6.31–6.35 of
`span-length-crossover-findings.md`). These are not tuning preferences: each one is a case where the
library **knows** something is wrong and does not say so, or ships a constant that is correct only
for the regime it was tuned on. We have not released, so these are free to change now and expensive
to change later.

Ordered by user harm.

**0a. `search_type="hybrid"` is accepted and SILENTLY IGNORED at section and fused level.**
FTS5 indexes chunks only, so `search_level="sections"` / `"fused"` return vector-only results no
matter what the caller asks for — no warning, no error, and no field in the result recording which
retrieval actually ran. Measured cost of the asymmetry (qasper/egemma, doc-level nDCG@10): BM25 is
worth **+0.0860** to chunks and **+0.0000** to sections and fused. This silently biased every
level comparison in our own study by that amount and reversed two headline conclusions
(`span-length-crossover-findings.md` §6.36). Minimum fix: warn, or reject the argument. Real fix:
give sections and fused a keyword leg — likely the largest single retrieval win available.

**0b. The hybrid default actively HURTS at fine granularity.** BM25's contribution changes sign with
chunk size: on vector-healthy small chunks it dilutes a strong signal (**−0.092** on egemma at
c=219, **−0.067** on MiniLM at c=128), while rescuing badly truncated large ones (**+0.259** /
**+0.182**). A user with small chunks is being degraded by a default they never chose. The shipped
`chunk_size=500` is itself the argmax of a *hybrid* curve; on vector it loses to c=219 by 0.098
(egemma) and to c=128 by 0.167 (MiniLM), where the curve declines monotonically with no interior
optimum at all (§7 of `POST-CONFOUND-SYNTHESIS.md`).

**1. `chunk_size` silently discards text past the encoder's context — and this is the whole ballgame.**
At the shipped `chunk_size=500` on a 256-token encoder (`all-MiniLM-L6-v2`, our own eval default), a
chunk is ~1,767 chars and the encoder reads 896 — **49% of the corpus never enters any vector.** No
error, no warning.

The measured curve (§6.35/1) says this is the *only* chunk-size decision that matters. Retrieval
quality is a **plateau with a cliff**: across an 8× range of chunk sizes that fit inside the
encoder's context, total variation is 0.028 nDCG; once coverage falls to 55%, it drops **0.109** —
four times the entire plateau, and 10× the build-noise floor. Undershooting is nearly free,
overshooting is expensive. Note 500 is the *argmax* on a 2k-context encoder and past the cliff on a
256-token one: the constant is unconditioned, not wrong.

> Both numbers in that paragraph are hybrid-era and coverage-era artifacts. On **vector** search the
> plateau does not exist — the curve declines monotonically over a 14× range (§7 of
> `POST-CONFOUND-SYNTHESIS.md`) — and the "55%" is really **65.4%** once chars/token is measured
> rather than assumed (§8.2). The *direction* of this item survives; its calibration does not.

Fix: warn at DB creation on **measured coverage** — `Σ min(chunk_len, cap) / Σ chunk_len` over the
ingested corpus — *not* on `chunk_size × k > cap`. Those two disagree badly: at one measured rung
the mean-based estimate said "barely truncated" (r_eff 0.86) while 44.9% of text was being discarded,
because the chunk-length distribution is bimodal (median 10,759 > mean 7,229; `CharChunker` leaves
short remainder fragments that drag the mean down). A warning built on the cheap arithmetic would
have stayed silent through the exact failure it exists to catch.

Because the good region is a plateau, this warning *is* the fix — no derived default is needed. Any
value comfortably inside the context is within 0.028 of optimal.

**Two corrections to how this warning must be built (§8 of `POST-CONFOUND-SYNTHESIS.md`, 2026-08-06).**

*Count tokens; do not estimate them.* The chars/token constant we used was 3.5; measured is **4.38**
(MiniLM's own tokenizer) and **~4.2** (embeddinggemma, by bisecting Ollama's real truncation
boundary). Worse, it is not a constant: across chunks of a single corpus it spans **2.00–5.67**, so a
char cap misclassifies **~28–31%** of individual chunks whenever the length distribution sits *on*
the cap — the exact regime a tuning warning fires in — while looking accurate in aggregate. Use the
model's tokenizer where one is reachable; where it isn't (Ollama models have no importable
tokenizer), the warning must say it is estimating rather than print a confident percentage.

*Calibrate the threshold — mild truncation is free.* Coverage does not predict retrieval damage.
Paired builds show **46% of chunks truncated with nDCG unchanged to four decimals** (mean cosine
displacement 0.0027); damage only appears past ~0.01 displacement. So a warning that fires on any
truncation will cry wolf. Fire it on *substantial* loss, and say what it costs, not merely that it
happened.

**2. `_provider_context_tokens` returns `None` for every ollama provider.** It reads only `num_ctx`
and `max_input_tokens`; `OllamaEmbeddings` sets neither by default. `_span_embed` then falls back to
a fixed **24,000-char** window (~6,860 tokens), ~3.3× egemma's real 2,048 — so each rawspan window
overflows and is silently truncated, the exact failure the windowing code was written to prevent.
Ollama exposes the value (`/api/show`, `/api/ps`); sentence-transformers exposes `max_seq_length`.
Pure derivation from the model, no policy. Fix this first — items 1 and 6 depend on knowing the
context. (§6.34/4b. Measured as harmless on Qasper's short sections, 2.0% of text; it would bite hard
on any long-section corpus.)

**3. Batch sizing is count-based, not token-aware — and fails in the direction we are about to
recommend.** `max_batch_size` is a fixed 64 regardless of chunk length, so token volume per request
scales with `chunk_size`: fine at 500, ~112k tokens at 1750, which cannot complete inside the 300s
default `timeout` at the default concurrency. `max_retries=3` then kills the ingest. Users following
our own advice to raise `chunk_size` for a long-context model will hit this. Fix: size batches by
estimated tokens, not count.

**4. Embedding errors stringify to nothing.** `logger.error(f"Error processing batch {n}: {e}")`
prints an empty message for `httpx.ReadTimeout` — an operator hitting a real failure gets silence and
a retry. Log the exception type and request shape (batch size, token estimate) at minimum.

**5. Do not default `search_level="fused"`.** Fusion loses to plain chunk retrieval at **every**
`section_weight` on all five legs (MiniLM, §6.33/1) and on **all five egemma rungs**, by 0.093 to
0.324 — a deficit that *grows* with `chunk_size`, because 0.65 blends a section leg that degrades
monotonically as chunks coarsen (§6.35/4). The constant is mistuned as a function of another
parameter we are telling users to change. `section_weight=0.65` is the exact argmax on Qasper and wrong by −0.05 to −0.18
elsewhere — it is not a bad number, it is an *un-conditioned* one.

**6. Over-fetch when `return_type="sections"`.** `_search` sets `fetch_k == k` with no reranker, so
asking for 10 sections retrieves 10 *chunks*, which collapse into far fewer distinct sections. Worth
+0.008 on Qasper. No policy question.

**7. Ship the diagnostic instead of a prose tuning guide.** `section_weight`, whether hierarchical
embeddings help at all, and gold density are properties of the user's *corpus*, which no static guide
can know. Everything needed is derivable from a built index in seconds — encoder coverage of section
text (46.2% on MiniLM vs 98.0% on egemma, same corpus), chunkless-section rate (**40.1% of all
sections and 26.3% of gold** on Qasper — a hard recall ceiling of 0.737 for chunk→section roll-up),
and chunk-truncation share. A `lvdb doctor` / `db.diagnose()` printing those tells each user which
regime they are in. This is the highest-leverage item on the list: it converts the study into a
feature rather than documentation.

Build the coverage line per item 1's two corrections: count tokens with the model's real tokenizer,
and state uncertainty when there isn't one. A diagnostic that prints a confident wrong percentage is
worse than one that admits it is estimating — ours was understating coverage by up to 16 points.

**Deliberately NOT here: defaulting `chunk_size` from the encoder's context.** The optimum does track
context (§6.35/1 — the slope reverses sign between a 256-tok and a 2k encoder), but two encoders give
a *direction*, not a coefficient: the peak sits at **92% of cap on MiniLM and 25% on egemma**. Those
are not two estimates of one number — they are arbitrary points on **flat regions**, which is also
why a derived default is unnecessary. Ship the coverage warning (item 1); the plateau does the rest.

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
