# Retrieval Baseline (T1 reference)

Relevance baseline for `main` at commit `41b824a`, measured by
`benchmarks/eval_retrieval.py`. **Every T1 retrieval change is measured against
this.** A change that does not improve — or at least hold — `ndcg@10` gets
reverted, not shipped.

```bash
# Reproduce (first run downloads SciFact and builds the database, ~6 min)
./.venv/Scripts/python.exe benchmarks/eval_retrieval.py

# Gate a change: non-zero exit if any configuration regressed
./.venv/Scripts/python.exe benchmarks/eval_retrieval.py --check
```

The machine-readable copy is `benchmarks/retrieval_baseline.json` (tracked).
`--check` diffs against it with a tolerance of 0.005 on `ndcg@10`.

This is a *relevance* benchmark and is unrelated to `BASELINE.md`, which records
geometric recall on SIFT-128 and full-stack latency. Neither of those says
anything about whether the right document comes back.

## Setup

| | |
|---|---|
| Dataset | BEIR **SciFact**, test split — 5,183 documents, 300 queries, 339 judgements |
| Relevance | Binary |
| Embeddings | `sentence_transformers` / `all-MiniLM-L6-v2` (384-dim, L2-normalized) |
| Index | `IndexFlatL2` (library default) |
| Chunking | `sentences`, `chunk_size=500`, `chunk_overlap=1` (library defaults) → 5,632 chunks |
| Metrics | `trec_eval` conventions, linear gain. See `benchmarks/metrics.py`. |

### The harness is calibrated, not just plausible

`vector · best` scores **0.6447**. The published BEIR figure for
`all-MiniLM-L6-v2` on SciFact is **0.6455**. Agreement to ~0.001 is the evidence
that the qrels wiring, the metric, and the retrieval path are all correct — a
harness that silently mis-joins query ids to documents would land nowhere near
this. If a future change to chunking or normalization moves this row far from
0.645, suspect the harness before believing the result.

## Baseline

| configuration | recall@1 | recall@5 | recall@10 | **ndcg@10** |
|---|---|---|---|---|
| hybrid vw=0.9 · best | 0.4833 | 0.7406 | 0.7860 | **0.6466** |
| hybrid vw=0.3 · best | 0.4833 | 0.7406 | 0.7860 | 0.6461 |
| hybrid vw=0.5 · best | 0.4833 | 0.7406 | 0.7860 | 0.6461 |
| hybrid vw=0.7 · best | 0.4833 | 0.7406 | 0.7860 | 0.6461 |
| hybrid vw=0.9 · average | 0.4782 | 0.7381 | 0.7858 | 0.6454 |
| hybrid vw=0.3 · average | 0.4782 | 0.7381 | 0.7858 | 0.6450 |
| hybrid vw=0.5 · average | 0.4782 | 0.7381 | 0.7858 | 0.6450 |
| hybrid vw=0.7 · average | 0.4782 | 0.7381 | 0.7858 | 0.6450 |
| vector · best | 0.4800 | 0.7406 | 0.7860 | 0.6447 |
| vector · average | 0.4757 | 0.7381 | 0.7858 | 0.6441 |
| vector · frequency_boost | 0.4456 | 0.7195 | 0.7871 | 0.6221 |
| hybrid vw=0.3 · frequency_boost | 0.4464 | 0.7162 | 0.7871 | 0.6205 |
| hybrid vw=0.5 · frequency_boost | 0.4464 | 0.7162 | 0.7871 | 0.6205 |
| hybrid vw=0.9 · frequency_boost | 0.4431 | 0.7162 | 0.7871 | 0.6197 |
| **hybrid vw=0.7 · frequency_boost** ← library default | 0.4431 | 0.7162 | 0.7871 | **0.6193** |
| keyword · best | 0.0167 | 0.0200 | 0.0200 | 0.0188 |
| keyword · frequency_boost | 0.0167 | 0.0200 | 0.0200 | 0.0188 |
| keyword · average | 0.0167 | 0.0200 | 0.0200 | 0.0183 |

## What this measurement already shows

Three things fall out of the table before a single T1 change is made.

**1. Keyword search is non-functional on natural-language queries.**
`nDCG@10 = 0.019`, near enough to nothing. Two separate defects in
`FTSQuerySanitization` (`src/localvectordb/_filters.py`) split the query set
between them — of 300 SciFact queries, 233 hit the first and 67 hit the second,
and **none takes a working path**.

*The multi-term path joins every term with `AND`* (`:838-852`), so
`"0-dimensional biomaterials show inductive properties."` becomes
`"0-dimensional" AND "biomaterials" AND "show" AND "inductive" AND "properties"`
and matches only documents containing all five words, stopwords included. Probed
directly against the FTS5 table: that expression returns **0 rows**; the same
terms joined with `OR` return 5. BM25 exists precisely to rank partial matches,
and this hands it nothing to rank.

*Ordinary English "and"/"or"/"not" are parsed as FTS5 operators* (`:922-952`).
Dispatch tests `query.upper()`, and the handler uppercases the query — which is
what *creates* the operator, since FTS5 only honours `AND`/`OR`/`NOT` in
uppercase. Adjacent words are then glued into phrases:
`"aspirin does not reduce cardiovascular risk"` becomes
`"ASPIRIN DOES" NOT "REDUCE CARDIOVASCULAR RISK"` — documents containing the
literal bigram *aspirin does* but not the literal phrase *reduce cardiovascular
risk*. The opposite of what was asked.

Tracked as **T1.0**, a release blocker, ahead of the rest of T1.

**2. `vector_weight` is very nearly a no-op**, which follows from (1). Keyword
scores are almost always zero, so `vw * vs + (1-vw) * 0` is a monotonic rescale
of the vector score, and `nDCG` only depends on order. `vw` = 0.3, 0.5 and 0.7
are identical to four decimal places; 0.9 differs by 0.0005. Any user tuning this
knob today is tuning nothing.

**3. The library's default configuration is the worst non-keyword configuration
in the sweep.** `hybrid vw=0.7 · frequency_boost` scores 0.6193 against 0.6466
for `hybrid vw=0.9 · best` — 0.027 nDCG, ~4.4% relative, given away by the
defaults. `frequency_boost` costs ~0.025 nDCG relative to `best` at every
`vector_weight`.

Note that `frequency_boost` has marginally *higher* `recall@10` (0.7871 vs
0.7860) while ranking substantially worse. It rewards documents with several
matching chunks, which surfaces one extra relevant document across the query set
but reorders the top of the list badly. On SciFact this effect is muted anyway:
abstracts average 1.09 chunks, so most documents have exactly one chunk and the
scoring method has nothing to aggregate. **Do not settle T1.6 (prune 11 scoring
strategies to 3) on SciFact alone** — use NFCorpus, whose documents are longer.

## Caveats

- SciFact has binary relevance, so the graded-gain path in `ndcg_at_k` is
  exercised only by NFCorpus (`--dataset nfcorpus`) and by the unit tests.
- 1.09 chunks/document means this dataset barely tests document-scoring
  aggregation or hierarchical retrieval.
- Reranking (`--rerank`) is excluded from the committed baseline: it is slow on
  CPU and, per T1.2, currently reorders an already-truncated top-`k`, so it
  cannot improve recall by construction.
- `--max-docs` / `--max-queries` truncate the corpus and inflate every metric.
  The harness refuses `--save-baseline` when either is set.
