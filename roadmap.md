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

**Status.** PRE-RELEASE is **complete**: 0a, 0b, 1, 2, 3, 4, 5b, 5c, 5e, 6, 7 all resolved (and 5d
measured, verdict "do not ship"). Items 1 and 7 shipped together as `db.diagnose()` / `lvdb doctor`
(`ad672c2`); 0b was closed as a policy decision — the diagnostic carries it (see its RESOLVED block).

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

**RESOLVED 2026-08-18 — decision: the default stays hybrid; the diagnostic carries the regime.**
Not deferred — decided. Three reasons, none of them new measurement:

* There is no corpus-independent crossover to key a conditional default on. The sign flip depends on
  encoder context vs. chunk size — exactly the "no defensible global default" finding that produced
  the diagnostic in the first place. A `chunk_size < X → vector` rule would just be a new
  unconditioned constant, the same defect this section exists to remove.
* `db.diagnose()` / `lvdb doctor` (`ad672c2`) already reports the regime this item turns on: measured
  encoder coverage tells a user whether their chunks are in the vector-healthy band (where hybrid
  dilutes) or the truncated band (where it rescues). The escape hatches are documented:
  `search_type="vector"` or per-level `vector_weight`.
* A conditional default is a behavior change requiring re-baselining both gates immediately before
  the first release, to ship a heuristic the study showed cannot be made globally right.

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

**RESOLVED 2026-08-17 (`ad672c2`), built to both corrections.** The warning hooks `_save_internal` —
the one chokepoint every mutating path crosses — rather than instrumenting eight ingest sites, reads
the stored per-chunk token counts in a single SQL aggregate (full corpus, no sampling, no model), and
re-checks only when the corpus has doubled, so the scan runs O(log n) times over a database's life.
Threshold 0.80 sits between the two measured anchors: 46%-of-chunks-truncated was free, 65% coverage
cost 0.109. The message states the measured cost and — since the stored counts are cl100k, not the
encoder's tokenizer — says it is estimating and points at `db.diagnose()` for exact numbers. "Warn at
DB creation" was never literally possible (measured coverage needs an ingested corpus); post-ingest,
once per instance, is what the item's own fix specification implies.

**2. `_provider_context_tokens` returns `None` for every ollama provider — RESOLVED 2026-08-14
(`fb7f857`).** It reads only `num_ctx`
and `max_input_tokens`; `OllamaEmbeddings` sets neither by default. `_span_embed` then falls back to
a fixed **24,000-char** window (~6,860 tokens), ~3.3× egemma's real 2,048 — so each rawspan window
overflows and is silently truncated, the exact failure the windowing code was written to prevent.
Ollama exposes the value (`/api/show`, `/api/ps`); sentence-transformers exposes `max_seq_length`.
Pure derivation from the model, no policy. Fix this first — items 1 and 6 depend on knowing the
context. (§6.34/4b. Measured as harmless on Qasper's short sections, 2.0% of text; it would bite hard
on any long-section corpus.)

A new `context_tokens` property, deliberately **separate from `max_input_tokens`** — the latter drives
`HTTPEmbeddingProvider`'s client-side truncation pass, which runs on tiktoken's `cl100k_base` and not
on any tokenizer an Ollama model uses, so reporting the context through it would have bought a correct
window at the price of a wrong truncation. `OllamaEmbeddings` derives it from `/api/show` as the
**minimum of every ceiling**, which is required rather than defensive: `nomic-embed-text` declares
`num_ctx` 8192 against an architectural `context_length` of 2048, so a preference order overstates its
window 4×, and `num_batch` is a real ceiling for embeddings (an encoder embeds its whole input in one
batch) rather than a throughput knob. `_provider_context_tokens` now mins its candidates for the same
reason — it took the first found, so an explicit `num_ctx` could widen a window past what the
architecture can read.

Checked against the one value measured directly: `embeddinggemma:300m` reports 2048 from all three
sources, and bisecting Ollama's real truncation boundary landed on 2048 as well. Every local embedding
model now reports 2048, and the span window drops **24,000 → 6,144 chars**. Both gates are inert by
construction — they run sentence-transformers, where `context_tokens` is `None` and the path reduces
to the old first-of behaviour.

**3. Batch sizing is count-based, not token-aware — RESOLVED 2026-08-14 (`78afdc3`).**
`max_batch_size` was a fixed 64 regardless of chunk length, so token volume per request scaled with
`chunk_size`: fine at 500, ~112k tokens at 1750, which cannot complete inside the 300s default
`timeout` at the default concurrency, after which `max_retries=3` kills the ingest. That is the
direction our own advice points a user with a long-context model. Batches are now capped by estimated
token volume *as well as* by count.

The 50,000-token default is deliberately **inert at the shipped configuration** (64 × 500 is
comfortably under) and binds only where duration is already the failure mode. It is not derived from
throughput, because throughput is not a constant — it varies ~6× between two local models on one box.
The hosted side gains for a different reason: OpenAI's ceiling is 1,000 *texts* against a documented
300,000-token request limit, so a full batch of ordinary chunks is rejected outright today. Counting
texts cannot see that; counting tokens can.

`chars/3.5` is an acceptable estimate **here and not at a truncation boundary** — item 1's "count
tokens, never estimate" rule does not transfer, because a wrong estimate at a batching boundary
changes how text is grouped and never what is sent. Verified rather than argued: the same 120 texts
embedded as one batch and as six produce **bitwise identical** output (max abs delta 0.0). That check
matters because the retrieval gate reuses cached databases and is structurally blind to vector drift.

**4. Embedding errors stringify to nothing — RESOLVED 2026-08-14 (`78afdc3`).**
`logger.error(f"Error processing batch {n}: {e}")` printed an empty message for `httpx.ReadTimeout`,
so an operator hitting a real failure got silence and a retry.

**The item was narrower than the defect.** The log line it named was one site; the same defect sat on
the *raised* error — `EmbeddingError(f"Error retrieving embeddings: {str(e)}")` handed the **caller** a
sentence ending in a colon. `describe_exception()` (in a new `utils.py`, so `reranking.py` need not
import a heavy module for four lines) falls back to the type name, which is the diagnostic here; the
message is the optional part. Applied to **reranking's six identical sites** as well, beyond the item's
literal wording: a cross-encoder is the slowest call in the pipeline, so a message-less timeout is
likelier there, not less.

Batch failure logs now carry the request's shape and identity, because an embedding failure is almost
always about size and that is the one thing the exception never says:

```
Error embedding batch 3/12 (64 texts, 112,000 chars, ~32,000 tokens)
  with ollama 'embeddinggemma:300m' at http://localhost:11434: ReadTimeout
```

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

**5d. The hybrid candidate pool — MEASURED ON THE REAL PATH, and NOT to be shipped.**
`_hybrid_search` fetches `_hybrid_pool_size(k) = max(k, min(k * 4, 100))` for both legs, so the
default `k=10` retrieves **40**. §19.3 measured a ~+0.008 win for widening to 100–200 on numpy
captures. §20 re-ran it through `db.query()` on three corpora. The mechanism replicated exactly on
qasper (numpy predicted +0.0083 at 40→100; the product gave +0.0077, p=.005; `frequency_boost` gained
nothing, as predicted). **The recommendation did not survive the corpus axis.**

The change a user would actually get — `best` @200 vs today's `frequency_boost` @40:

| corpus | n | fanout@40 | baseline | `best` 40→200 | **end-to-end** |
|---|---|---|---|---|---|
| qasper | 882 | 1.78 | 0.4657 | **+0.0099** p=.003 | +0.0067 p=.066 |
| NFCorpus | 323 | 1.03 | 0.3367 | +0.0008 null | +0.0011 null |
| NQ | 2000 | 4.61 | 0.9493 | −0.0007 null | **−0.0030 p=.024 LOSS** |

One win, one null, one significant loss — and even qasper's end-to-end change misses significance,
because switching aggregator costs −0.0032 at the shipped width before the wider pool earns it back.
**The default stays at 40.** Two findings are worth keeping: the gain is an **aggregation** effect and
not candidate recall (NFCorpus's pool is 97% distinct documents; adding 345 more moved +0.0004), and
the tempting "fanout gates it" rule is **refuted** — NQ has the highest fanout and no effect. Cost, if
revisited: 2.2× retrieval latency at pool 200, 4.0× at 400.

**§22 (qasper_full, 2026-08-11) does not overturn the verdict but changes what the open question is.**
Re-measured on the same corpus at 4× the haystack (1,088 papers / 13,503 chunks / 2,940 queries):

| method, 40→P | qasper (275 docs) | qasper_full (1,088 docs) |
|---|---|---|
| `best` @200 | +0.0099 p=.003 | +0.0096 p=.000 |
| `frequency_boost` @200 | +0.0018 **null** | **+0.0069 p=.000** |
| `frequency_boost` @400 | +0.0007 **null** | **+0.0073 p=.001** |

`best` replicates to the third decimal. **`frequency_boost` does not** — and its null on dev is
exactly half of §19.3's rule that *pool width and aggregator must move together*. That rule is a
**small-haystack artifact**: same corpus, same encoder, same code, 813 more distractor documents.
Controlled by scoring dev's identical 882 queries against both indexes (`--query-subset dev`), which
isolates the haystack from the query sample: +0.0018 null → **+0.0099 p=.005**.

Consequence for this item: the change worth evaluating is no longer "switch to `best` and widen"
(which gets *weaker* at scale — +0.0067 p=.066 on dev, +0.0019 p=.346 on full) but **"keep
`frequency_boost`, just widen"**, which is the simpler edit and is the winning arm on both full-corpus
legs. It is still ONE corpus: `frequency_boost` gains nothing on NQ (−0.0015 at 40→200) or NFCorpus
(−0.0002), so **40 remains the default**. **MLDR is still the designed test, and should now be run at
two corpus sizes rather than one** — corpus SIZE was never an axis in §20, which is how a rule derived
on 275 documents survived a three-corpus generalisation check and still failed.

**5e. Resurrect `percentile` as an option (not a default) — RESOLVED 2026-08-14 (`775fc43`).**
Removed in `54a9898` — a sound prune
whose one gap was sweeping a single pool width, which is where aggregator differences are smallest.
Re-measured across six pools, three corpora and two encoders: a **document-target** aggregator,
**3 of 4 doc cells positive** (best: NQ doc/vector +0.0201) and **0 of 6 section wins** (two
significant losses). Fanout sets the magnitude of its deviation from `max`; the target unit sets the
sign. Bring it back with **one** parameter — the clean single order statistic beat the shipped
two-percentile blend in **19 of 20 cells**, so `secondary_percentile` and `primary_weight` should not
return with it.

Shipped exactly that way: one knob (`document_scoring_options={"percentile": 0.9}`), the two dead
parameters ignored if passed, and a test asserting it. Interpolation matches `np.percentile`'s default
because that is what the sweep behind these numbers measured — a different rule is a different
aggregator at small chunk counts. `auto` never selects it, so no existing caller moves. **The bar for
making it a default is unchanged and still unmet**: the wide-pool condition has to hold on more than
one corpus first.

**6. Over-fetch when `return_type="sections"` — RESOLVED 2026-08-14 (`775fc43`), and the item was
wrong on both halves.** `_search` set `fetch_k == k` with no reranker, so asking for 10 sections
retrieved 10 *chunks*, which collapse into far fewer distinct sections.

**The real bug is the count, not the ranking, and it is much larger than the item implies.** The API
silently did not honour `k`:

| corpus | mean returned of 10 | short lists | after |
|---|---|---|---|
| qasper dev | 9.55 | 34.0% | 10.00, 0.0% |
| NQ | **7.88** | **82.6%** | 10.00, 0.0% |

Shipped on that. **The size of the defect tracks chunks-per-section fanout**, which is why NQ — longer
articles, more chunks per heading — is hit five times harder than qasper on the *same* code.

The ranking claim, "worth +0.008 on Qasper, no policy question", does not survive measurement, but the
resolved answer is better than the item's: **both real corpora gain, and only the synthetic leg
loses.** Section-level nDCG@10:

| corpus | before | after | delta | |
|---|---|---|---|---|
| qasper (real) | 0.1765 | 0.1819 | **+0.0054** | |
| NQ (real) | 0.5204 | 0.5315 | **+0.0111** | 95% CI [+0.0057, +0.0166], p=.0001, n=800 paired |
| superdocs (synthetic) | 0.2807 | 0.2760 | −0.0047 | |

The mechanism is real rather than noise: a wider pool can represent a section by a better chunk than
the narrow pool exposed, which makes each score a more accurate estimate of the section's true best
chunk — the narrow pool systematically under-scored any section whose best chunk ranked below `k`. On
superdocs' glued-together sections the resulting reordering happens to hurt, which is the expected
shape for a leg whose gold is a median 825 chars inside a ~24.5k-char section.

**NQ was the right third corpus, and MAUD could not have been.** MAUD has no section target on the
real `query()` path at all — `SectionDetector` is a two-group, line-anchored Markdown regex and finds
essentially no sections in contracts, which is the same reason the hier gate's long-section leg is
synthetic in the first place. NQ is also the *harder* test: its gold is **less** section-shaped than
qasper's (3.6% of golds cover >99% of their owning section, against qasper's 18.0%), so a section-level
win there is not partly circular the way qasper's is. It is the largest of the three deltas anyway.

Measured by scoring one index twice — shipped `_chunk_rollup_pool_size` against it patched to identity
— so corpus, vectors and query path are byte-identical and the delta is the over-fetch alone.

Reuses `_SECTION_ROLLUP_OVERFETCH` rather than adding a second constant: this is the same collapse one
level down (chunk→section instead of section→document), and 5 covers the worst measured
chunks-per-section fanout (1.09 qasper, 3.68 MAUD). It takes the hybrid pool's ceiling instead of the
sibling's unbounded multiply, because this pool feeds the chunk level and hybrid multiplies it again —
a section query now fetches 100 per leg rather than 40, **~2.5× the retrieval work on that path only**.
Document-level nDCG was byte-identical across the before/after.

**Do not try to verify this with `MockEmbeddings`.** A count-based test passes whether or not the
over-fetch is present — mock vectors scatter the top chunks across distinct sections, so the collapse
never occurs. I built exactly that check and it "passed" before the fix; the sibling test
`test_rollup_overfetches_the_section_pool` already documents having been fooled the same way. Both new
tests assert the **pool width handed to the search leg**.

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

**RESOLVED 2026-08-17 (`ad672c2`) — `db.diagnose()` / `lvdb db <name> doctor`.** Reports encoder
coverage of chunk text, section length against the encoder window, the chunkless-section share (as
the hard roll-up recall ceiling it is), chunks-per-document/section fanout, and per-table FTS health.
Token counts are exact only where the encoder's own tokenizer is importable (sentence-transformers,
local HuggingFace, OpenAI's cl100k); everywhere else the report is labelled *estimated* — tiktoken as
the estimator, never a chars/token constant. Two framing choices worth recording: sections are
reported as a **regime** (rawspan windows-and-pools rather than truncates, but the pooled vector
loses 0.25–0.36 nDCG on >8k-token spans, so the summary points long corpora at `centroid`), and the
diagnostic resolves the encoder context itself instead of adding `context_tokens` to
sentence-transformers — doing that would have changed span-embed windowing for ST models and forced a
re-baseline of both gates, which item 2 deliberately avoided. Sampling is evenly strided over id
order, so repeat runs on an unchanged database measure the same rows.

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
2. **An extraction gate.** Both retrieval gates read BEIR corpora as text and
   never touch all2md; the 1.7.1→1.12 bump changed 30/30 PDFs and moved headings
   −7.2% monotone, entirely invisibly. Any multimodal ingest work makes this
   third blind spot worse.
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
