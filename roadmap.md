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
(`span-length-crossover-findings.md` §6.36).

**Now measured on six leg/encoder pairs: +0.084 to +0.131** — the asymmetry is systematic, not a
qasper quirk, and larger than every vector-side effect in the study.

**Minimum fix DONE** (`d0b0f1a`): `query()` warns once per instance when `search_type` cannot be
honoured at a vector-only level. A warning rather than a rejection, because `hybrid` is the default
and rejecting would break the ordinary `query(text, search_level="sections")` call.

**RESOLVED 2026-08-07 — all three fixes shipped. `_VECTOR_ONLY_LEVELS` is now empty and no query
warns.** It turned out to be three fixes, not one, and only the first was free:

* **Fix A, `documents_fts` (`a4c0918`).** The index already existed, populated and
  trigger-maintained, and no search path read it. Wiring it needed no schema work and no migration.
  qasper document level: vector 0.2758 → hybrid **0.4037**, previously exactly +0.0000.
* **Fix B, `sections_fts` (`1bd45c5`).** `sections` has no `content` column — a section is a
  `start_pos`/`end_pos` slice of its parent — so this needed a new contentless FTS5 table written at
  ingest from the document slice, deleted by an `AFTER DELETE` trigger, plus a self-healing backfill
  for existing databases. Routing through `documents_fts` was never a shortcut: document-level BM25
  is not section-level scoring.
* **Fix C, the `fused` blend.** Could not be a wiring fix at all. `fused` runs two *different*
  fusions, so BM25 had to enter through the blend, which has a free parameter. Settled by a 2-D
  sweep over four corpora (`benchmarks/eval_fused_blend.py`, `--blend-sweep`): two-stage topology,
  native leg pool, existing weights unchanged. `experiments/SYNTHESIS-v2.md` S17.

At the shipped defaults this is worth **+0.057 qasper-doc, +0.058 MAUD, +0.123 MLDR** — matching
S12.8's predictions to three decimals — and **−0.033 on Natural Questions**, taken anyway because
`chunks` already defaults to hybrid and already pays that same penalty. The per-corpus escape hatch
is `vector_weight` (NQ's argmax is 0.90, not 0.50), now documented as a per-level knob.

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

**5. Do not default `search_level="fused"` — but the ORIGINAL REASON IS WITHDRAWN (2026-08-07).**
The old claim was that fusion loses to plain chunk retrieval at *every* `section_weight` on every
leg, by 0.093 to 0.324. That was measured under the 0a confound: fused ran vector-only against a
chunks arm that had BM25. With Fix C shipped, **`fused` is now the best arm at every level on
qasper** (0.4483/0.4516 against chunks 0.4293 and sections 0.4340/0.4368).

What survives is weaker and still enough to keep the default where it is: which level wins remains
**corpus-dependent** — chunks still beats fused on superdocs (0.2924 vs 0.2812) — and `fused` costs
two retrievals instead of one. So it stays opt-in on cost and unpredictability, not on being worse.

`section_weight=0.65` remains *un-conditioned* rather than bad: the measured argmax is 0.65 on
qasper, 0.35 on MAUD, 0.80 on NQ. `vector_weight` is now a second knob with the same property (0.5
qasper/MAUD, 0.9 NQ). Both are documented with those numbers; neither has a defensible global
default, which is the argument for item 7.

**5b. The document aggregator was wrong for vector search — RESOLVED 2026-08-07 (`a8e12fd`).**
`document_scoring_method` defaulted to `frequency_boost` for every search type. It multiplies the best
chunk score by a term growing in the chunk count, which is fine on the min-max-normalised hybrid scale
and wrong on the raw bounded similarity a vector search returns, where it mostly rewards a document
for owning more chunks. The default is now `"auto"`: `best` for `search_type="vector"`,
`frequency_boost` otherwise, with any explicit value passed through untouched. Worth **+0.0226 nDCG@10
on SciFact, +0.0150 on NQ, +0.0084 on qasper** for a default vector query.

Two deliberate non-changes. **Keyword** keeps `frequency_boost`: the same argument plausibly applies,
but that leg was never measured and generalising from an untested one is exactly how the 0a confound
happened. **Sections** keep a plain max and expose no knob — aggregation turned out to be a property
of the *unit being ranked*, not of the corpus, and across 20 measured corpus/target/leg/pool cells a
summing aggregator never lost on a document target and never won on a section target.

Note for whoever gates the next default change: **both retrieval gates were blind to this one.** The
hier gate runs hybrid and the retrieval gate passes explicit methods, so their `+0.0000` proved
inertness for existing callers, not that the new default worked. That needed end-to-end tests
(`tests/test_scoring.py::TestAutoDocumentScoringEndToEnd`).

**5c. A quoted phrase inside a sentence kills the keyword leg — SHIPPED 2026-08-10 (`0cbe243`).**
`handle_phrase_query` AND-joined every term around a quoted phrase, stopwords included — the exact
failure the plain-text branch documents and avoids. It hit **20.6% of MAUD queries**, of which
**99.8% returned zero keyword hits**. Now OR-joins, each phrase surviving as one phrase token.

The fallback variant this item asked for was measured, and it **reframed the defect**:
`phrase_fallback` (strict, degrading to `all_or` only when strict returns nothing) eliminates every
dead leg and gains **+0.0002 (n.s.)**, while `all_or` gains **+0.0022** (p=.028). They differ only on
the 412 queries where the conjunction *did* return rows — so rescuing dead legs is worth ~+0.0012 and
de-conjoining the ones that already worked is worth ~+0.0020. **The problem was the conjunction
ranking badly, not returning nothing.** `all_or` shipped on that evidence.

Trade taken knowingly: a quoted phrase no longer constrains the result set, only ranks it (still
matched as a phrase; a fully-quoted query still binds via the exact-phrase branch). **Both gates are
structurally blind** — SciFact has 0/300 quoted queries — so the evidence is the MAUD number plus 10
tests in `tests/test_keyword_search_semantics.py`. The sanitiser is otherwise injection-safe and
crash-free across 9,333 real queries.

**5d. The hybrid candidate pool is too small — MEASURED, not yet shipped.** `_hybrid_search` sets
`search_k = max(k, min(k * 4, 100))` for both legs, so the default `k=10` retrieves **40**. On qasper
the optimum is **100–200**:

| arm | 40→100 | 40→200 | 100→400 |
|---|---|---|---|
| doc `max` | **+0.0083** p=.004 | **+0.0083** p=.019 | −0.0004 null |
| doc `pctl@0.9` | **+0.0085** p=.016 | **+0.0118** p=.008 | +0.0006 null |
| section `pctl@0.9` | **+0.0078** p=.001 | **+0.0076** p=.004 | +0.0008 null |
| doc **`freq@0.3`** | +0.0004 null | −0.0038 null | **−0.0076 LOSS** |

It is a **hybrid-only** effect: on a vector-only leg widening moves `max` by exactly +0.0000 at every
step. The mechanism is the zero-fill — `_relative_score_fusion` scores a keyword-only chunk **0.0** on
the vector leg, and because the vector band is compressed the chunk just outside the cutoff is nearly
as good as the one inside. **Two conditions before shipping:** it must be measured on the real
`db.query()` path (these are numpy captures), and it has to move *together* with the aggregator,
because the shipped `frequency_boost` degrades at wider pools while `max`/`percentile` gain. qasper
only so far. Cost is 2.5× the candidates through fusion.

**5e. Resurrect `percentile` as an option (not a default).** Removed in `54a9898` — a sound prune
whose one gap was sweeping a single pool width, which is where aggregator differences are smallest.
Re-measured across six pools, three corpora and two encoders: a **document-target** aggregator,
**3 of 4 doc cells positive** (best: NQ doc/vector +0.0201) and **0 of 6 section wins** (two
significant losses). Fanout sets the magnitude of its deviation from `max`; the target unit sets the
sign. Bring it back with **one** parameter — the clean single order statistic beat the shipped
two-percentile blend in **19 of 20 cells**, so `secondary_percentile` and `primary_weight` should not
return with it.

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

**Keyword→vector cascade — measured 2026-08-07, viability is corpus-dependent.** Prefiltering with
`documents_fts` and vector-searching only the survivors is a *different operator* from everything we
have tested: all our fusion is parallel, so it can only add, whereas a cascade makes BM25 a hard
recall gate. Keyword recall@N is therefore a ceiling on the whole system. Measured at document level:
NQ **0.8962 @10 / 0.9716 @100** scanning 0.2% / 1.9% of the corpus, MLDR **0.9300 @10**, but qasper
only **0.5408 @10** and 0.7857 @100 (36% of the corpus), and MAUD 0.5305 @100 (66%). It works where
query and document share vocabulary and fails where they do not.

Treat it as a **latency optimisation that buys speed by capping recall**, not as a quality upgrade,
and only where vector search is actually the bottleneck — with flat FAISS at our current corpus sizes
it is not. The machinery already exists (`_faiss_search_with_selector` + `faiss.IDSelectorBatch`, used
for metadata pushdown today); only a doc-id→faiss-id helper is missing. If it is ever pursued, measure
a **hybrid** stage 1 first: the ceilings above are for a pure-keyword stage 1 and a union of keyword
and vector candidates has a higher one.

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
