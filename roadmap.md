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

## Shipped: v0.1.0 (2026-08-19) and the pre-release retrieval study

The seven-item PRE-RELEASE section that used to live here — measured-default
fixes, the keyword-leg repairs, `db.diagnose()` / `lvdb doctor`, the gate
blind-spot closures — is complete and shipped in v0.1.0. The record lives where
it belongs now: [CHANGELOG.md](CHANGELOG.md) for what changed, the
documentation's *retrieval study* (findings) and *lab notebook* (how the
measurements went wrong and were caught) pages for why, and
`benchmarks/RETRIEVAL_BASELINE.md` + the three gates (`eval_retrieval.py
--dataset all`, `eval_hier_gate`, `eval_extraction.py`) for keeping it true.
The one standing rule worth restating outside those pages: **no retrieval
default changes without moving a gated number on more than one corpus.**

---

## v0.1.x — Post-release review pass and the next experiment slate

v0.1.0 shipped 2026-08-19 (PyPI, npm, GitHub Release with the `.mcpb` bundle).
Before the v0.2.0 trust work starts, a short pass over the surfaces users
actually touch — reviewed by *using* them, not by reading them — plus the
experiment slate the retrieval study left open. These are ordered so that each
review's findings can still change the API cheaply; v0.2.0's freeze-adjacent
work makes them expensive later.

### 1. Agent-driven review — and consolidation — of the MCP server

The MCP surface has accumulated ~18 flat tools (`query_database`,
`get_document`, `grep_documents`, `list_prefixes`, `find_related_documents`,
`patch_document`, schema and lifecycle tools, split read-only/read-write). Two
tracks, review first so the redesign is evidence-based:

**Consolidate into dispatch tools.** Eighteen tool schemas is context-window
tax an agent pays on every session, whether or not it uses them. Restructure
around one or two "master tools" with a `command` argument — CLI-like
semantics (`lvdb query ...`, `lvdb get_document ...`, `lvdb grep ...`), which
agents already know how to drive. Likely shape: a read tool and a write tool
(preserving the existing read-only/read-write mode split at the dispatch
boundary), each documenting its commands the way a CLI documents subcommands.
The per-command help must be reachable *through* the tool (a `help` command or
self-describing errors), because a dispatch tool's schema can no longer carry
every argument's documentation.

**Range retrieval is an HTTP gap, not an MCP gap — close it there.** Measured,
not assumed: MCP `get_document` and the CLI already share the full portion
selector (`document_portions.py`: chunk index/range, `char_range`,
`line_range`, `section`, `outline`). The HTTP API is the surface that only
returns whole documents (`GET /databases/{db}/documents/{id}` takes no portion
parameters), which means the remote client and the JS SDK inherit the gap.
Expose the same five selectors as query parameters on that endpoint and thread
them through `RemoteVectorDB` and the SDK.

**The agent-driven review** runs structured agent sessions (research task,
editing task, corpus-navigation task) against a built corpus and logs friction
rather than opinions:

- **Token economics.** Which tools return more than the agent needed? A `query`
  result an agent immediately follows with `get_document` calls is a signature
  worth counting — it may mean result payloads are too thin (or too fat).
- **Affordance gaps.** What did the agent *try* to do that no tool supported?
  This is the empirical version of v0.3.0's speculative list (`get_section`,
  `get_outline`, `find_related` at the section level) — let the sessions
  confirm or replace it.
- **Argument-shape traps.** Wrong-but-plausible calls (metadata filter syntax,
  `return_type` values) and whether the error message got the agent back on
  track in one step. An MCP error an agent cannot self-correct from is a bug —
  doubly so under a dispatch design, where the error text *is* the docs.
- **Defaults.** `query_database` should be checked against the study's shipped
  defaults — does the MCP path get hybrid + auto scoring + section roll-up
  parity with `db.query()`, or has it drifted?

The deliverable is a findings list with transcripts, then the consolidation
informed by them — same discipline as the retrieval study: capture first,
judge afterwards. Ship the dispatch tools alongside the flat ones for one
release, then retire the flat set before v0.2.0 freezes anything.

### 2. Rerankers are invisible outside the library

Current exposure, measured not assumed: `db.query(reranker=...,
reranker_config=..., rerank_k=...)` works, and the HTTP server passes all three
through. **The CLI and MCP expose none of it** — `lvdb db <name> search` has no
rerank flag at all — and `QueryBuilder.rerank()` is a *different feature*
(recency/metadata post-processing) that collides with the name, which will
mislead exactly the user who goes looking.

The study says this gap matters more than any tuning knob: reranking was worth
~9× every first-stage parameter we tuned (with model choice dominating — ship
bge-base-class or hosted, never MS-MARCO minis, which measured null-to-negative
on our workloads). Plan:

- **A persisted per-database default reranker** (`reranker_config` at creation,
  like the embedding provider), so `query()` applies it without per-call
  ceremony and every surface — CLI, MCP, server — inherits it for free. Per-call
  parameters stay as overrides.
- **CLI flags** on `search` (`--rerank`, `--rerank-model`, `--rerank-k`) for
  one-off use without touching the database config.
- **Naming cleanup** for the `QueryBuilder.rerank()` collision — likely
  `postprocess()` or documenting the distinction loudly; renaming after 1.0
  would be a break, now it is a deprecation alias.
- MCP: decide deliberately whether `query_database` grows a rerank argument or
  simply respects the database default (the default-config route needs no new
  tool surface, which is probably right for agents).

### 3. `lvdb` CLI review and the `db` command restructure

**Restructure `lvdb db <name> <command>` to `lvdb db <command> --db <name>`.**
The dynamic name sitting between two static tokens is the awkward part: help
and shell completion cannot enumerate it, `lvdb db --help` describes a group
you cannot invoke without a name you have to already know, and the pattern
matches no mainstream CLI (resource-as-flag is the common semantics:
`psql -d`, `mysql -D`, `aws --profile`). The verb-first form also unlocks the
real ergonomic win: **a default database** from `LVDB_DB` (env var) or config,
so an agent or a script sets it once and every subsequent call is just
`lvdb db search "query"`. Migration is cheap now and a break after 1.0: keep
`lvdb db <name> <command>` as a hidden deprecated alias for one minor release
(it is unambiguous — a first token that matches no subcommand is a name),
warn, then drop.

The rest is a consistency pass over the broad tree (16 `db` subcommands plus
`chunk`, backup, tuning, auth, serve, mcp), again driven by use — a scripted
session covering create → ingest → search → doctor → backup → compact-someday:

- Flag parity with the library (`search` lacks rerank as above; check
  `--document-scoring`, `--section-weight`, quoted-query behaviour end-to-end).
- Output contract: `--format/-f {table,json}` unification shipped in rc-era
  cleanup; verify no stragglers, and that JSON output is stable enough to pipe
  (agents script the CLI too).
- Exit codes and error text on the failure paths (the rc-era pass fixed the
  worst; re-verify against the current tree).
- `lvdb shell` deserves its own look: it is the closest thing to a REPL demo of
  the product — and under the restructure it is the natural home for the
  "stay scoped to one database" workflow the old positional form served.

### 4. Experiment slate — adaptive retrieval

The study's constant refrain was "no defensible global default" — every knob's
argmax is corpus-dependent. The next tranche of experiments is about making the
*system* adaptive instead of finding better constants. Standing rules: every
arm runs on all four corpora (qasper, NQ, MLDR, MAUD) through the real
`query()` path where possible; both gates (`--dataset all` + hier) before any
src/ change; capture-once-sweep-offline for anything with an LLM in the loop.

- **Per-query `vector_weight` router.** The oracle bound justifies the attempt:
  a perfect router beats the best fixed weight by **+0.051 (section) / +0.060
  (doc)** on qasper — large compared to every constant we tuned. NQ/MLDR/MAUD
  oracle bounds are queued. If the bound holds up, train a cheap router
  (features: query length, IDF profile, quoted-phrase presence, corpus stats —
  no LLM call) and measure how much of the oracle gap a real predictor
  captures. If a corpus's bound is small, that corpus is evidence *against*
  routing and caps the expected win honestly.
- **Query augmentation, keyword and vector legs.** What is already known:
  cross-modal PRF helps NQ only (+0.1010) and hurts everywhere else — the
  refuted version is *unconditional* PRF, not PRF. LLM-side expansion (HyDE,
  keyword synonym injection) is unmeasured here and has a different cost model
  (an LLM call per query). Design: capture expansions once per query set, sweep
  offline; the interesting arm is *routed* augmentation — expand only where a
  cheap signal (short query, low IDF mass, zero keyword hits) predicts benefit,
  since the PRF result shows the per-corpus sign flip is the whole game.
- **Automatic metadata generation** (pull-forward of v0.7.0's LLM-metadata
  item, as an experiment first). At ingest, an LLM writes typed fields
  (doc_type, entities, dates, topics) into the existing metadata schema.
  Machinery exists end-to-end (schema, indexes, filter pushdown); what does not
  exist is an eval that can *score* it — none of the four corpora have
  metadata-dependent relevance. **Build the eval leg first** (same order-of-work
  rule as the multimodal plan): a corpus slice with queries whose answers
  require a filter ("what did X say in 2023"), then measure
  generation+filtering against pure retrieval on identical documents.
- **Automatic metadata filtering** (self-querying). The complement: at query
  time, translate the natural-language query into a `QueryBuilder` filter plus
  a residual text query. The pushdown machinery
  (`_faiss_search_with_selector` + `IDSelectorBatch`) already exists. Two
  measured cautions transfer directly: a wrong filter is a *hard recall gate*
  (the cascade result — BM25 prefiltering capped recall at 0.54 on qasper), so
  the default must be soft (boost, not filter) or verified-then-hard; and the
  gates are blind to it (no gated query needs a filter), so it ships behind the
  new eval leg or not at all.

Sequencing note: items 1–3 are cheap and should land before v0.2.0 freezes
surfaces. Item 4 is a research track that runs alongside; nothing in it ships
without moving a gated number.

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

### `mmap_index=True` is inert, and it is not the way out

Probed 2026-08-13 on a 200k × 768 index (588 MB on disk), fresh process per mode:

```
       flags=0: rss 42.7 -> 643.7 MB (delta +601.0)
  IO_FLAG_MMAP: rss 42.8 -> 643.8 MB (delta +601.0)
```

`_faiss_read_flags` already passes `faiss.IO_FLAG_MMAP`, and `_require_writable`
charges a real price for it — every vector-mutating path raises. It saves nothing.
FAISS honours the flag **only inside `read_InvertedLists`** (an IVF-family path);
`IndexFlat` holds its vectors in a `std::vector<float>` and always allocates.

Aimed at a normally-written IVF index the flag does not degrade, it throws:

```
InvertedListsIOHook::lookup(int): read_InvertedLists:
could not load ArrayInvertedLists as 646f6c69 ("ilod")
```

That message is easy to misread. The file contains `"ilar"` (`ArrayInvertedLists`)
and no `"ilod"` anywhere — the fourcc named in the error is the one the reader
*went looking for*. With `IO_FLAG_MMAP` set, `read_InvertedLists` hands an
ordinary `ilar` payload to `lookup(fourcc("ilod"))->read_ArrayInvertedLists(...)`,
which reinterprets it in place. So **the file format is not the obstacle and
`merge_ondisk` is not a prerequisite** — a normally-written IVF index is mmap-able
as-is wherever that hook exists. What is missing here is the hook itself; this
wheel registers exactly one:

```
registered 1 InvertedListsIOHooks:
6c626c69 ilbl struct faiss::BlockInvertedLists
```

The `ilod` hook self-registers from `OnDiskInvertedLists.cpp`, which is POSIX
(`mmap`/`madvise`) and is not built into the Windows `faiss-cpu` wheel — so
`faiss.OnDiskInvertedLists` is absent and `merge_ondisk` fails on the same name.
FAISS mmap is a **Linux-only deployment option**; check
`faiss.InvertedListsIOHook.print_callbacks()`, not the file, before relying on it.

The rule that generalises: **mmap in FAISS is an interface feature, not a file
feature.** It exists only where storage sits behind the polymorphic
`InvertedLists` interface, which is what lets a mapped backend be swapped in with
`replace_invlists`. A flat index has no such indirection, so it will never honour
the flag on any platform. Either make `mmap_index` honest about that, or delete
it.

SQLite needs nothing here: `sqlite_tuning.py` already sets `mmap_size` (256–512 MB
by profile), and document text plus FTS5 scale to 10⁷ rows on disk fine. **The
vectors were always the whole problem.**

### Quantization is the way out — as a lifecycle step, not an index knob

Same corpus, every candidate wrapped in `IndexIDMap2` so it satisfies the existing
`id_map` requirement:

| index | file | RSS | extrapolated to 10⁷ chunks |
|---|---|---|---|
| `IndexFlat` (today) | 588 MB | +601 MB | **~30 GB** |
| `IVF256,SQ8` | 150 MB | +164 MB | ~8.2 GB |
| `IVF256,PQ96` | 23 MB | +61 MB | **~3.0 GB** |

10⁷ chunks at 768 dims is ~29 GB of raw float32 (~57 GB at 1536 dims). PQ96 puts
that on a laptop with no mmap, no read-only restriction and no platform
dependency; SQ8 is the conservative, near-lossless middle.

The conversion mechanism is confirmed: `reconstruct_n` returned all 200k vectors
in 0.3 s, `faiss.vector_to_array(index.id_map)` gives ids in matching order,
re-add took 13 s, non-contiguous ids survived, and every probe vector retrieved
its own id at rank 1. No re-embedding. That is `repair()`'s existing walk pointed
at a different target index type. **Training is the one real cost — 296 s at a
mere `nlist=1024`**, and 10⁷ chunks wants `nlist` in the 16k–65k range.

Which is why this should not be an index-type knob. Trainable indexes were skipped
originally because `_init_faiss_index` builds against an *empty* database, and
that objection stands. But the shape of the workload is bulk-load-then-read-mostly
— "build knowledge base → retrieve with an LLM" — so the corpus exists in full
before the first query. Keep flat as the ingest default and add compaction as an
explicit, reversible step:

```
lvdb db mydb compact --index IVF16384,PQ96
```

The user never picks a trainable index up front, where there is no data to train
on; they pick it once the corpus exists, which is the only moment the choice is
well-posed. Keep the flat file until the compacted index verifies, and it is an
optimization rather than a migration. It also lands naturally next to
`lvdb doctor`: *"your 4M-chunk database is using 12 GB resident; compacting to
PQ96 would take ~8 minutes and ~1.2 GB."*

**`backup.py` is the trap.** `_create_compatible_base_index` infers index type by
substring-matching the class *name* for `"IP"`/`"L2"`/`"HNSW"` and otherwise falls
back to `IndexFlatL2`, so an IVF/PQ index would silently round-trip to the wrong
type — the one place this corrupts quietly instead of failing loudly. Fix that
before shipping any new index type. The closed `Literal` unions in `config.py`,
`base.py` and `_core.py` plus the if/elif ladder are mechanical by comparison, and
`supports_deletion` / `supports_id_selector` already come out right for IVF.

**The open question is recall, and it is a measurement, not a judgement call.**
IVF is approximate and PQ is lossy, so both cost recall — landing precisely on the
quantity reranking cannot recover (same ceiling as the cascade section above).
`eval_retrieval.py` and the pool-width harness price `nprobe` and PQ width in nDCG
on real corpora; that number decides whether this ships as a default or stays an
opt-in for people who are actually RAM-bound. Do **not** price it on synthetic
Gaussian vectors — isotropic data concentrates into near-ties and flatters or
damns PQ arbitrarily, where real embeddings sit on the correlated low-dimensional
manifold PQ handles well.

### Also in this lane

- **ANN that actually supports deletes.** Today HNSW and LSH are selectable but
  their deletes no-op. IVF with an id map fixes this on the way past.
- **int8 / binary quantization with rescoring.** The large size reductions only
  hold *with* rescoring (~95% recall retention).
- **Incremental persistence.** The index file is rewritten wholesale after a bulk
  upsert; bulk-loading many documents one call at a time is many full-index
  rewrites.

---

## v0.6.0 — Multimodal

The target is scientific literature, where the figure *is* the result and the
text around it is a summary. Everything below is design, not commitment: nothing
here has been measured, because there is no multimodal eval leg yet — and
building that leg is the first item, not the last.

### What already carries over

More than expected. Three pieces of existing machinery do most of the work:

- **Namespaced multi-index is already the architecture.** `_faiss_id_counters`
  keyed by name and `_allocate_faiss_ids("main"|"section"|"document", n)`
  (`database/_core.py:1164`), with three `IndexIDMap2` files persisted
  independently. An `image` namespace is a worn path, not a new one.
- **`column_embeddings` is the precedent for "a vector that is not a chunk"** — a
  sibling table drawing ids from the *main* namespace (`database/_metadata.py:63`)
  with its own hydration query in `_search.py`. An image table is that shape.
- **The `EmbeddingTask` / prefix work generalises.** We already accept that the
  same text embeds differently depending on *role*; modality is a second axis on
  the same idea. `apply_prefix(texts, task)` becomes
  `prepare(items, task, modality)`, which is how the Jina v4 / Voyage / Cohere
  multimodal APIs are shaped anyway.

### Blocking measurement — extraction, 2026-08-13

Measured on 30 cached arXiv cs.CL papers under all2md 1.7.1 *and* 1.12.0:
**zero `Image` nodes reach the AST**, and zero image markers reach the Markdown —
despite defaults `attachment_mode='alt_text'`, `include_image_captions=True`,
`skip_image_extraction=False`. One paper carried **251 embedded rasters** and
produced 0 Image nodes. Figure captions appear in output *only* because they are
ordinary page text.

So `include_image_captions` has never fired in this pipeline, and three
constraints follow:

1. **There is no positional anchor.** With no marker where a figure was, an image
   chunk has no `start_pos` to bind to a document or section. Ingest needs
   `attachment_mode="save"` (relative-path refs) to get bytes *and* an anchor in
   one pass — the current default is the one mode that yields neither.
2. **Figures are a raster/vector mix, and the split is per-paper.** Some papers
   are pure vector (220 draw ops, 0 rasters), others raster-heavy (251). Any
   figure-*extraction* path must handle both; page rasterisation is the only
   approach indifferent to which. That is a cost argument for ColPali that has
   nothing to do with retrieval quality.
3. **Caption provenance is lost by the AST shape.** all2md's `Image` node has no
   `caption` field (`Table` does), so a detected caption is downgraded to
   *fallback* `alt_text`, indistinguishable from a source-supplied attribute. The
   PDF detector is also a 50pt geometric heuristic whose fallback accepts any
   text under 200 chars beginning with a capital — in two-column layouts it can
   return the adjacent column. Filed upstream.

Caption binding is therefore **tiered by source**, and the tier must be recorded,
not assumed: HTML/JATS `<figcaption>` is ground truth, DOCX caption styles are
semi-structured, PDF is a guess. For scientific literature this is often a
*choice* — PMC ships JATS, arXiv ships LaTeX — and the PDF is the worst available
input.

### Architectures

- **A — caption-and-index.** A VLM describes the figure at ingest; the text goes
  through the existing pipeline unchanged and the image is a metadata URI. Zero
  architectural change, the keyword leg works, and every banked tuning result
  transfers. Cannot do image-as-query. **Not a throwaway baseline** — it is the
  text half of B's row, so the ablation falls out of one build.
- **B — shared-space single encoder** (jina-clip / jina-v4 / siglip /
  nomic-vision / voyage-multimodal / cohere v4). One model, one dimension, so
  `_check_saved_embedding_dimension` (`database/_core.py:1390`) keeps holding.
  Images become a chunk *variant belonging to a document*, which is the right unit
  — a figure belongs to a paper, and the measured aggregation result
  ("aggregation tracks the target unit") then applies directly to mixed
  image+text roll-up. **Recommended core.**
- **C — dual encoder, dual index.** B plus a second provider slot and config key;
  keeps the tuned text stack byte-identical. Since B already wants two indices,
  this is a config delta rather than a different design.
- **D — ColPali/ColQwen page-as-image.** Highest ceiling for exactly this corpus —
  tables, equations, multi-column layout, figure/caption binding — all the things
  extraction flattens or drops. But it is multi-vector late interaction
  (~1000 vectors/page, MaxSim), which breaks the 1:1 `faiss_id`↔row invariant that
  `_reconstruct_embeddings_batch`, the dual-store integrity check, `repair`, and
  every aggregator rely on. **This is a second retrieval backend that shares a
  database, not a chunk-level feature**; it should land behind an index-backend
  seam designed now and implemented later. It also shares the multi-vector
  machinery with the v0.4.0 ColBERT item — do not build them twice.

**Split the indices even under a single shared encoder.** CLIP-style joint spaces
have a documented modality gap: text and image embeddings occupy separate cones,
so a text↔image cosine is not on the scale of a text↔text cosine. Hybrid
min-max normalises *within each leg's candidate pool*, so if both modalities sit
in one index and one search, top-k is drawn by raw distance and one modality can
sweep the pool before normalisation happens. Split, and normalisation is per-leg
with a weight knob — which is what `search_level="fused"` and `section_weight`
already do. Unmeasured here; measure the gap before trusting any single-index
result.

### Storage

`store_bytes` is opt-in, and the mode matters more than the flag:
`blob_storage="sidecar" | "sqlite" | "reference"`, **sidecar content-addressed by
default**.

The decisive fact is that a database is already a *directory*
(`{name}.sqlite` + three `.faiss` sidecars), so self-containment is already a
directory property and a `blobs/` sidecar breaks no invariant. Backup then fits
almost for free: `_generate_document_manifest` already diffs incrementals on
content hashes, and a CAS is immutable-by-hash. Two-level hex fanout
(`blobs/ab/cd/<sha256>`) — 100k entries in one directory is painful on NTFS and
worse for the backup copy walk.

- **Reject base64.** +33% before compression, and PNG/JPEG are already
  entropy-coded so lzip buys ~0–2%. It also poisons FTS if it ever reaches an
  indexed column, and it forecloses hash dedup.
- **No live packfile.** Deletes leave holes and appends force rewrites; we already
  have one write-amplification problem (wholesale index rewrite). Pack-on-*export*
  is free — backup already tars the directory.
- **Store the SHA-256 even in `reference` mode**, so drift between a vector and
  the bytes it was built from is detectable by `repair` rather than silent.
- **Cost to pay:** `_create_backup_manifest` SHA-256s every file it walks. The
  blob directory must be excluded and identified by filename, or every backup
  becomes an O(corpus-bytes) rehash.

Dedup is the sleeper win for this corpus — publisher logos, watermarks,
letterheads, and preprint/published duplicates. CAS kills byte-identical copies;
`_filter_similar_chunks_vectorized` with `similarity_threshold` catches the
same figure re-rendered at a different DPI. Both mechanisms already exist. The
counterpart is that `ON DELETE CASCADE` orphans blobs — sweep them in `repair`
(idempotent, and already where dual-store divergence is reconciled) rather than
refcounting, which needs discipline on every delete path.

### Schema sketch

```sql
CREATE TABLE image_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    blob_hash TEXT NOT NULL,   -- CAS key; set in every storage mode
    mime TEXT NOT NULL,
    width INTEGER, height INTEGER, size_bytes INTEGER,
    caption TEXT,              -- text surrogate: FTS'd, reranked, displayed
    caption_source TEXT,       -- figcaption|docx_style|pdf_heuristic|vlm|none
    source_uri TEXT,
    faiss_id INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);
```

A sibling table, not a `modality` column on `chunks`: the text path stays
byte-identical, which matters because the baselines are the asset. The cost is
union logic in `_search.py` hydration and roll-up, and it is worth paying.

`caption` is the load-bearing column. It gives image rows a keyword leg, so they
are not vector-only citizens in a hybrid-by-default database — no new
`_VECTOR_ONLY_LEVELS` entry — and it gives the reranker a string without a new
wire format. `caption_source` is a confidence signal, not bookkeeping: a
`<figcaption>` makes a VLM description redundant, a `pdf_heuristic` makes it
necessary.

### Order of work

1. **The eval leg, first.** `MockEmbeddings` cannot tell whether the right figure
   ranked first. ViDoRe, or a PMC slice: the OA **tgz** packages bundle JATS XML
   *and* figure images with explicit caption bindings, which scores the PDF
   heuristic and gates extraction at the same time. Note PMC is disabled in
   mdparse's `corpus.toml` for good reasons — per-id serves an HTML interstitial
   — so this is real work, not a cache hit.
2. ~~An extraction gate.~~ **Shipped in v0.1.0** (`benchmarks/eval_extraction.py
   --check`, fingerprinting all2md output over committed fixtures — including a
   wrapped-headings PDF that reproduces the 1.7.1→1.12 heading drift). The
   multimodal work extends it with figure-bearing fixtures rather than building
   it.
3. A, then B with split indices. D behind the seam.

**Explicitly not:** a generic "any modality" abstraction. Audio and video have
different chunking, different units, and no corpus here — designing for them buys
nothing and constrains the image path.

---

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
