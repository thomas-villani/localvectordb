# Retrieval Baseline (T1 reference)

Relevance baseline for `main` at commit `a6d2e98`, measured by
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

### T1.6 — document scoring pruned to 3 methods (NFCorpus baseline)

Document scoring was reduced from 11 methods to `best`, `average`, `frequency_boost`.
The other eight (`worst`, `weighted_average`, `harmonic_mean`, `diminishing_returns`,
`statistical`, `robust_mean`, `percentile`, `geometric_mean`) were **measured on
NFCorpus** — the right dataset for this, since SciFact averages 1.09 chunks/document and
barely aggregates — across every search leg and `vector_weight`. None beat the three
keepers by more than 0.0010 (below the 0.005 tolerance), none beat them at the library
default, and `weighted_average` additionally crashes (`ZeroDivisionError`) on any hybrid
document whose chunk scores are all zero-filled. So the deletion cost nothing measurable.

Because `--check` diffs config labels without keying on the dataset, NFCorpus has its own
baseline file, **`benchmarks/retrieval_baseline_nfcorpus.json`** (tracked, 3,633 docs /
323 queries, graded relevance). Gate an NFCorpus change with:

```bash
./.venv/Scripts/python.exe benchmarks/eval_retrieval.py --dataset nfcorpus \
    --check --baseline benchmarks/retrieval_baseline_nfcorpus.json
```

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
| hybrid vw=0.5 · best | 0.5621 | 0.7714 | 0.8410 | **0.7133** |
| **hybrid vw=0.5 · frequency_boost** ← library default | 0.5529 | 0.7714 | 0.8410 | **0.7090** |
| hybrid vw=0.5 · average | 0.5521 | 0.7679 | 0.8402 | 0.7086 |
| hybrid vw=0.3 · frequency_boost | 0.5617 | 0.7539 | 0.8252 | 0.7004 |
| hybrid vw=0.3 · best | 0.5592 | 0.7573 | 0.8252 | 0.6999 |
| hybrid vw=0.3 · average | 0.5458 | 0.7509 | 0.8243 | 0.6942 |
| hybrid vw=0.7 · frequency_boost (the *old* default) | 0.5407 | 0.7596 | 0.8260 | 0.6940 |
| hybrid vw=0.7 · best | 0.5407 | 0.7596 | 0.8260 | 0.6935 |
| hybrid vw=0.7 · average | 0.5357 | 0.7604 | 0.8218 | 0.6924 |
| hybrid vw=0.9 · best | 0.5040 | 0.7417 | 0.8010 | 0.6617 |
| hybrid vw=0.9 · frequency_boost | 0.5040 | 0.7417 | 0.8010 | 0.6614 |
| hybrid vw=0.9 · average | 0.4923 | 0.7392 | 0.8008 | 0.6586 |
| keyword · best | 0.5183 | 0.7187 | 0.7876 | 0.6577 |
| keyword · frequency_boost | 0.5183 | 0.7187 | 0.7876 | 0.6577 |
| vector · best | 0.4800 | 0.7406 | 0.7860 | 0.6447 |
| vector · average | 0.4757 | 0.7381 | 0.7858 | 0.6441 |
| keyword · average | 0.4933 | 0.7037 | 0.7826 | 0.6427 |
| vector · frequency_boost | 0.4456 | 0.7195 | 0.7871 | 0.6221 |

### What changed since the previous baseline (`26a92ce`)

T1.1 replaced hybrid's un-normalized weighted sum with **relative-score fusion**:
each leg is min-max normalized within the query's own candidate pool before being
blended by `vector_weight`. Every `hybrid` row improved. Every `vector` and
`keyword` row moved by exactly `+0.0000` — the check that T1.1 changed the fusion
and nothing else.

| configuration | `26a92ce` | `a6d2e98` | Δ |
|---|---|---|---|
| hybrid vw=0.5 · best (best in sweep) | 0.6592 | 0.7133 | +0.0541 |
| hybrid vw=0.7 · frequency_boost ← default at the time | 0.6343 | 0.6940 | +0.0598 |
| hybrid vw=0.5 · frequency_boost | 0.6299 | 0.7090 | +0.0792 |
| hybrid vw=0.9 · best | 0.6590 | 0.6617 | +0.0028 |
| vector · best | 0.6447 | 0.6447 | **+0.0000** |
| keyword · best | 0.6577 | 0.6577 | **+0.0000** |

## What this measurement shows

**1. Normalizing the legs is worth ~0.06 nDCG. RRF is worth ~0.01.** The two
candidate fixes were measured head to head before either was written. At the default
as it stood then (`vw=0.7 · frequency_boost`; the default has since moved to `vw=0.5`
— see point 2):

| fusion | ndcg@10 | recall@1 |
|---|---|---|
| un-normalized weighted sum (before) | 0.6343 | 0.4496 |
| Reciprocal Rank Fusion, k=60 (canonical) | 0.6439 | 0.4499 |
| Reciprocal Rank Fusion, k=10 (best RRF) | 0.6778 | 0.5163 |
| **relative-score fusion (shipped)** | **0.6940** | **0.5407** |

The best row any family could reach, over every `vector_weight` and every scoring
method: relative-score **0.7133**, RRF **0.6984**, un-normalized sum **0.6604**.
RRF was also rejected on three grounds beyond nDCG: its scores max out near
`1/(60+1)`, so any `score_threshold > 0.02` would silently return nothing; its `k`
is a tuning knob that would be frozen at 1.0, and the canonical 60 is close to the
worst value on this data; and it makes `frequency_boost`'s `min(1.0, …)` clamp
unreachable, silently redefining that scoring method.

**2. `vector_weight` finally behaves like a weight — and the default is now 0.5.**
Under the old sum the keyword leg contributed a near-constant 1.0 to every chunk it
retrieved, so `vector_weight` was asking "did the keyword leg find this at all?"
rather than "how well?". Once fusion normalized the legs, the optimum moved to
`vw=0.5`.

The default was held at 0.7 through T1.1 and T1.6 on purpose: the dense-versus-lexical
tradeoff is corpus-dependent, SciFact is unusually lexical, and tuning a global default
on one dataset is how you overfit a benchmark. **The retune waited for a second
dataset, and NFCorpus agreed** — so `vw=0.5` wins on both, and the default moved:

| dataset | `vw=0.7` (old default) | `vw=0.5` (new default) | Δ |
|---|---|---|---|
| SciFact (`frequency_boost`, 300 q) | 0.6940 | **0.7090** | +0.0150 (+2.2%) |
| NFCorpus (`frequency_boost`, 323 q) | 0.3298 | **0.3367** | +0.0069 (+2.1%) |

On NFCorpus the new default is effectively the best row in the whole sweep (0.3367 vs
0.3370 for the best). **Every baseline row moved `+0.0000` when the default changed** —
the sweep passes `vector_weight` explicitly at each value, so nothing measured here
depends on the default. That is the proof the change moved the default and nothing else;
what it buys is ~2% relative nDCG for every user who never touches the knob.

**3. Hybrid scores are now pool-relative.** They are comparable within one result
set, but not across queries, and not across different `k` (which changes the
candidate pool size). `score_threshold` on a hybrid query now selects by rank
position within the pool rather than by absolute match quality. The best chunk of a
leg always normalizes to 1.0 and the worst to 0.0 — the latter indistinguishable
from a chunk that leg never retrieved. All three are inherent to relative-score
fusion and are documented in `docs/source/document-scoring.rst`.

**4. `frequency_boost` is no longer catastrophic, and its clamp was load-bearing for
the wrong reason.** It now sits within 0.005 of `best` at every `vector_weight`,
where before it gave away up to 0.03. Its `min(1.0, best_score * multiplier)` clamp
turns out to have been *rescuing* the old fusion rather than hurting it: with
saturated keyword scores, `best * multiplier` routinely exceeded 1.0, and the clamp
capped the damage (+0.1024 nDCG at `vw=0.1`). Under relative-score fusion the clamp
now costs 0.0015–0.0026. It is a band-aid over the scale bug and should be revisited
in T1.6 — **but not on SciFact**, whose documents average 1.09 chunks.

## Caveats

- SciFact has binary relevance, so the graded-gain path in `ndcg_at_k` is
  exercised only by NFCorpus (`--dataset nfcorpus`) and by the unit tests.
- 1.09 chunks/document means this dataset barely tests document-scoring
  aggregation or hierarchical retrieval.
- SciFact is scientific claim verification: a query shares a great deal of
  vocabulary with the target abstract, and BM25 is a famously strong baseline
  there. Expect the vector/keyword ordering to shift on a paraphrase-heavy corpus.
- Reranking (`--rerank`) is excluded from the committed baseline: it is slow on
  CPU and, per T1.2, currently reorders an already-truncated top-`k`, so it
  cannot improve recall by construction.
- `--max-docs` / `--max-queries` truncate the corpus and inflate every metric.
  The harness refuses `--save-baseline` when either is set.
- Runs are bit-deterministic. Any nonzero delta from `--check` is a real change,
  not noise.
