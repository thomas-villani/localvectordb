# Performance Baseline (pre-optimization)

Captured with `uv run python -m benchmarks.profile_baseline --scale 5000 --queries 150`
**before** any optimization work, so later changes can be measured against it.

- Platform: `Windows-11-10.0.26200-SP0`, Python 3.12.2
- numpy 2.4.6, faiss 1.14.2
- Corpus: 5,000 generated docs (~100 words each), 187 queries/phase
- Embeddings: **mock** — deliberately isolates library overhead (our Python +
  FAISS + SQLite) from provider network/model latency. These numbers measure
  the code we can actually change; real-provider runs will be embedding-bound.

## Headline numbers

| Phase | total (s) | per op | throughput |
| --- | --- | --- | --- |
| ingest (5000 docs, FTS on) | 10.56 | 2.11 ms/doc | ~474 docs/s |
| query — vector | 1.10 | 5.90 ms/q | ~170 q/s |
| query — keyword | 0.17 | 0.89 ms/q | ~1100 q/s |
| query — hybrid | 1.89 | 10.1 ms/q | ~99 q/s |

Raw `cProfile` dumps: `benchmarks/results/profiles/*.prof` (open with
`uv run snakeviz benchmarks/results/profiles/ingest.prof`).

## Where the time goes

### Ingest (10.56 s)
| Cost | time | % | note |
| --- | --- | --- | --- |
| `sqlite3 .commit` | 4.31 s | **41%** | 5,001 commits — ~one commit per document |
| tiktoken token counting | ~2.3 s | **22%** | `CoreBPE.encode` 50,865×; `count_tokens` re-encodes during chunking |
| `executemany` (chunks/docs) | 1.30 s | 12% | 10,000 calls; already batched, but high volume |
| `execute` (other DML) | 0.31 s | 3% | |
| sentence splitting / regex | ~0.4 s | 4% | |

### Query — hybrid (1.89 s, richest path: runs vector + keyword)
| Cost | time | % | note |
| --- | --- | --- | --- |
| FAISS `IndexIDMap_search` | 0.26 s | 14% | native; `IndexFlatL2` is O(n) — grows with corpus |
| SQLite `execute`+`fetchall` (hydration) | 0.43 s | 23% | |
| **asyncio event-loop spin-up** (`embed_sync`) | ~0.50 s | **26%** | new loop + Windows socket self-pipe **per query** |
| `_get_faiss_metric_type` | 0.14 s | 8% | re-detected per candidate (14,960×) via `hasattr` |
| `_compute_document_scores` | 0.12 s | 6% | Python per-candidate scoring |

## Ranked optimization opportunities (no C++ needed)

These are all in our Python glue or configuration — the native engines are
already fast.

1. **Batch ingest commits.** ~41% of ingest is `commit()` at ~1/doc. Commit
   once per batch (or per pipeline flush) instead of per document. Biggest
   single ingest win.
2. **Cut redundant tokenization in chunking.** `count_tokens` re-encodes text
   repeatedly (`chunking.py:32`, `TokenChunker`/`WordChunker` boundary search).
   Encode once and slice token ids; reuse the encoder. ~22% of ingest.
3. **Stop spinning an event loop per query.** `embed_sync` → `asyncio.run()`
   (`embeddings.py:218`) builds and tears down a loop (+socket pair on Windows)
   on every call. Reuse a persistent loop, or give providers a true sync path.
   ~26% of single-query latency on the local path.
4. **Cache the FAISS metric type.** `_get_faiss_metric_type` (`_core.py:785`) is
   recomputed per candidate. Compute once per search (or memoize on the index)
   and thread `metric_type` into `_distance_to_similarity`. ~8% of query.
5. **Vectorize result scoring.** `_distance_to_similarity` /
   `_compute_document_scores` run per-candidate Python; convert the FAISS
   distance array to similarities with numpy in one shot.
6. **FAISS index type at scale.** `IndexFlatL2` is exact O(n). For large
   corpora, `IndexHNSWFlat` / `IndexIVFFlat` give sub-linear search (recall
   tradeoff). Tier-1 benchmark already supports comparing these.
7. **Batch the per-chunk similarity filter** (`_ingest.py`, when
   `similarity_threshold` is set) — currently one FAISS `search(k=1)` per chunk
   instead of one batched search.

## How to reproduce

```bash
uv run python -m benchmarks.profile_baseline --scale 5000 --queries 150
# wall-clock throughput/latency suite:
uv run python -m benchmarks.run --tier 2 --scales 5000
```

---

# Results after optimization

Measured before/after on a **fixed** environment (faiss 1.11.0, numpy 2.2.6),
toggling only the source under test via `git stash` and verifying the code state
each run. 5,000 docs; ingest reported as best-of-3 to suppress disk/AV noise;
queries as p50 over 300 iterations.

| Metric | Before | After | Speedup |
| --- | --- | --- | --- |
| Ingest (5000 docs) | 12.2 s (411/s) | 3.9 s (1273/s) | **3.1×** |
| Vector query p50 | 3.82 ms | 1.90 ms | **2.0×** |
| Hybrid query p50 | 7.33 ms | 4.84 ms | **1.5×** |
| Keyword query p50 | 0.93 ms | ~1.0 ms | unchanged (FTS-only) |

Changes landed:
1. **Batched ingest commits** — one commit per batch instead of per document
   (the worker holds one connection; doc IDs are reported only after their batch
   commits, so reported == durable).
2. **Chunking fast path** — short documents that fit in one chunk skip
   per-sentence tokenization entirely; `count_tokens` uses `encode_ordinary`.
3. **Persistent per-thread event loop** in `embed_sync` — no more building/tearing
   down an asyncio loop (and Windows socket self-pipe) per query.
4. **Cached FAISS metric type** — detected once per index instead of per candidate.
5. **Vectorized distance→similarity** in the vector-search candidate loop.
6. **Similarity-filter hash set** — fetched once and maintained in memory rather
   than a `SELECT DISTINCT content_hash` full scan per document (also preserves
   cross-document dedup now that commits are batched).

## Index type at scale (already supported)

`faiss_index_type="IndexHNSWFlat"` gives sub-linear search. Measured vector p50
at 25,000 docs:

| Index | p50 | p95 |
| --- | --- | --- |
| IndexFlatL2 (default, exact) | 2.78 ms | 3.68 ms |
| IndexHNSWFlat (ANN) | 1.49 ms | 2.20 ms |

The gap widens with corpus size (Flat is O(n)). **IVF** is not yet offered — it
needs a training lifecycle (train on a sample, tune nlist/nprobe, retrain as the
corpus grows) and is left as a future feature rather than a drop-in.
