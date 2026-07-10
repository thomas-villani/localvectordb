# Retrieval Baseline (T1 reference)

Relevance baseline for `main` at commit `26a92ce`, measured by
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

Regenerate with `--save-baseline` **from a clean tree**. The harness stamps the
baseline with `git rev-parse HEAD`, suffixed `-dirty` if the tree is not clean; a
`-dirty` stamp in the committed file means the numbers describe code that was
never committed.

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

Both retrieval legs agree with independently published figures on this dataset:

| leg | ours | published | source |
|---|---|---|---|
| `vector · best` | 0.6447 | 0.6455 | BEIR, `all-MiniLM-L6-v2` |
| `keyword · best` | 0.6577 | ~0.665 | BEIR, BM25 (Anserini/Lucene defaults) |

That is the evidence that the qrels wiring, the metric, the tokenizer and the
retrieval path are all correct. A harness that silently mis-joined query ids to
documents would land nowhere near either number, and a wrong join or a wrong
tokenizer could not agree with *both* references at once.

If a future change to chunking or normalization moves the `vector · best` row far
from 0.645, suspect the harness before believing the result.

## Baseline

| configuration | recall@1 | recall@5 | recall@10 | **ndcg@10** |
|---|---|---|---|---|
| hybrid vw=0.7 · best | 0.4957 | 0.7278 | 0.8079 | **0.6597** |
| hybrid vw=0.3 · best | 0.4957 | 0.7311 | 0.8062 | 0.6592 |
| hybrid vw=0.5 · best | 0.4957 | 0.7311 | 0.8062 | 0.6592 |
| hybrid vw=0.9 · best | 0.4957 | 0.7286 | 0.8054 | 0.6590 |
| keyword · best | 0.5183 | 0.7187 | 0.7876 | 0.6577 |
| keyword · frequency_boost | 0.5183 | 0.7187 | 0.7876 | 0.6577 |
| hybrid vw=0.3 · frequency_boost | 0.4857 | 0.7311 | 0.8062 | 0.6542 |
| hybrid vw=0.3 · average | 0.4907 | 0.7253 | 0.7987 | 0.6523 |
| hybrid vw=0.5 · average | 0.4907 | 0.7253 | 0.7987 | 0.6523 |
| hybrid vw=0.9 · average | 0.4773 | 0.7186 | 0.8046 | 0.6497 |
| hybrid vw=0.7 · average | 0.4807 | 0.7111 | 0.8029 | 0.6483 |
| vector · best | 0.4800 | 0.7406 | 0.7860 | 0.6447 |
| vector · average | 0.4757 | 0.7381 | 0.7858 | 0.6441 |
| keyword · average | 0.4933 | 0.7037 | 0.7826 | 0.6427 |
| hybrid vw=0.9 · frequency_boost | 0.4629 | 0.7156 | 0.7998 | 0.6398 |
| **hybrid vw=0.7 · frequency_boost** ← library default | 0.4496 | 0.7181 | 0.8023 | **0.6343** |
| hybrid vw=0.5 · frequency_boost | 0.4379 | 0.7300 | 0.8046 | 0.6299 |
| vector · frequency_boost | 0.4456 | 0.7195 | 0.7871 | 0.6221 |

### What changed since the previous baseline (`41b824a`)

`41b824a` measured keyword search at `ndcg@10 = 0.0188`. It was not mis-tuned; it
was non-functional. Two defects in `FTSQuerySanitization` meant **291 of the 300
judged queries matched literally zero rows**. T1.0 fixed both.

| configuration | `41b824a` | `26a92ce` | Δ |
|---|---|---|---|
| keyword · best | 0.0188 | 0.6577 | +0.6390 |
| keyword · average | 0.0183 | 0.6427 | +0.6244 |
| hybrid vw=0.7 · best | 0.6461 | 0.6597 | +0.0136 |
| hybrid vw=0.7 · frequency_boost ← default | 0.6193 | 0.6343 | +0.0150 |
| vector · best | 0.6447 | 0.6447 | **+0.0000** |

Every vector-only row moved by exactly 0.0000. That is the check that T1.0 changed
the keyword leg and nothing else.

## What this measurement shows

**1. Keyword search is now competitive with dense retrieval, and hybrid beats
both.** `keyword · best` (0.6577) slightly outscores `vector · best` (0.6447) and
has the best `recall@1` of any configuration (0.5183). Do not over-read this:
SciFact is scientific claim verification, where the query shares a great deal of
vocabulary with the target abstract, and BM25 is a famously strong baseline there.
Expect the ordering to flip on a corpus with more paraphrase.

**2. `vector_weight` now does something, but not what its name suggests.** It was
previously inert — every `hybrid` row was identical because the keyword leg was
always empty, making `vw * vs + (1-vw) * 0` a monotonic rescale of the vector
score. The rows now differ. But `vw=0.3` and `vw=0.5` still tie to four decimals,
and that has a cause worth fixing.

`_fts_rank_to_similarity` (`_search.py`) is `1.0 - min(1.0, exp(bm25))`, an
*absolute* transform of raw BM25 that is never normalized per query. On a real
query the top-20 BM25 scores span `-16.6 … -12.4`, and the transform maps all of
them into `[0.999996, 0.99999994]` — a spread of `2.3e-05`. Pure keyword search is
unaffected, because SQLite orders by raw `bm25()` before the transform runs. But
**hybrid fusion receives a near-constant `1.0` for every matched chunk**, so
`vector_weight` behaves as a *presence* weight (chunk matched, or did not) rather
than as a graded blend.

This is T1.1's premise. Until T1.0 landed the transform was almost never reached,
because nothing matched. It is now on the hot path for every query.

**3. The library's default configuration is still one of the worst in the sweep.**
`hybrid vw=0.7 · frequency_boost` scores 0.6343 against 0.6597 for
`hybrid vw=0.7 · best` — 0.025 nDCG, ~4.0% relative, given away by the defaults
alone. `frequency_boost` is the worst scoring method at every `vector_weight`.

It rewards documents with several matching chunks, which costs `recall@1` heavily
(0.4496 vs 0.4957) — the top of the list is where it reorders worst. On the
vector-only rows it does buy a hair of `recall@10` (0.7871 vs 0.7860), the one
place its premise pays; on the hybrid rows even that disappears. On SciFact the
effect is muted anyway: abstracts average 1.09 chunks, so most documents have
exactly one chunk and the scoring method has nothing to aggregate.
**Do not settle T1.6 (prune 11 scoring strategies to 3) on SciFact alone** — use
NFCorpus, whose documents are longer.

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
- Runs are bit-deterministic. Any nonzero delta from `--check` is a real change,
  not noise.
