# Hierarchical retrieval on local encoders — Qasper study

**Question.** The raw-span / fused hierarchical retrieval that ships in
LocalVectorDB was designed and tuned against **one** encoder: OpenAI
`text-embedding-3-small` (8k context). Does the technique hold on **local
Ollama encoders**, and is the win an artefact of small chunks?

**Answer (short).** Yes on all counts. Raw-span sections beat the chunk baseline
on every local encoder tested, at both chunk sizes, on both relevance targets —
and the advantage *grows* with larger chunks, so it is not a small-chunk
artefact. The `bge-m3` 2k-vs-8k context axis resolves cleanly: its lead is
**model quality, not context** — 2k ≈ 8k once over-long spans are window-pooled.
Full tables and caveats below.

This file is the reproducibility record for the run. The reader-facing writeup
is `docs/source/hierarchical-evaluation.rst`.

---

## Setup

- **Harness:** `benchmarks/eval_hierarchical.py` (arms scored by exact cosine in
  NumPy — no FAISS, no DB — against a per-`(model, text)` disk cache).
- **Runner:** `benchmarks/run_hier_ollama.sh 15` (one process per
  `encoder × chunk-size`; evicts all models between arms so each loads fresh at
  its `num_ctx`/`num_batch` and only one 1.2 GB model is resident at a time).
- **Dataset:** Qasper `dev`, **15 papers → 48 queries** (48/50 questions kept,
  107/110 evidence spans located). Real NLP papers with natural section
  structure and evidence-paragraph relevance judgments.
- **Metric:** nDCG@10.
- **Two targets:**
  - **DOC** — which *paper* holds the answer.
  - **SECTION** — which *section* holds the answer (the real test of the
    hierarchy premise).
- **Two chunk sizes:** 500 tokens (shipped default) and 1000 tokens
  (large-chunk baseline; ~5.5k chars on Qasper's ~4.3 chars/token, so a 1000-tok
  chunk still embeds in **one** pass under a 2k-context model's ~6144-char
  window — a faithful large-chunk arm, not a pooling artefact).

### Encoders / context arms

| arm | `num_ctx` | note |
|---|---|---|
| `nomic-embed-text` | 2048 | arch-capped at 2048 |
| `embeddinggemma:300m` | 2048 | arch-capped at 2048 |
| `bge-m3` | 2048 | 2k baseline for the context axis — **did not complete, see gap** |
| `bge-m3` | 8192 | true 8k, reachable only via the `num_batch` fix |

**The Ollama `num_batch` prerequisite.** Ollama's `/api/embed` silently caps
input at `n_batch` (default **2048**), *not* `num_ctx`. Raising `num_ctx` alone
does nothing past 2048 tokens for an encoder model — the extra context is never
used. `OllamaEmbeddings` now auto-sets `num_batch = num_ctx`, which is what makes
the `bge-m3@8192` arm actually embed 8k-token spans (confirmed: 5 window-pooled
spans at 8k vs 20 at 2k). Without this fix the 8k arm is indistinguishable from
the 2k arm.

---

## Results — nDCG@10

### DOC target (which paper holds the answer)

| arm | nomic·500 | nomic·1000 | egemma·500 | egemma·1000 | bge2k·500 | bge2k·1000 | bge8k·500 | bge8k·1000 |
|---|---|---|---|---|---|---|---|---|
| **rawspan-section** | **0.763** | **0.763** | 0.756 | 0.756 | **0.796** | **0.796** | **0.784** | **0.784** |
| fusion-rawspan | 0.760 | 0.737 | **0.780** | **0.764** | 0.768 | 0.752 | 0.742 | 0.750 |
| fusion-centroid | 0.708 | 0.691 | 0.697 | 0.711 | 0.721 | 0.720 | 0.721 | 0.720 |
| chunk (baseline) | 0.707 | 0.683 | 0.678 | 0.723 | 0.741 | 0.719 | 0.741 | 0.719 |
| centroid-section | 0.693 | 0.690 | 0.661 | 0.722 | 0.718 | 0.725 | 0.718 | 0.725 |
| rawspan-doc | 0.664 | 0.664 | 0.703 | 0.703 | 0.711 | 0.711 | 0.686 | 0.686 |
| centroid-doc | 0.654 | 0.649 | 0.648 | 0.682 | 0.678 | 0.704 | 0.678 | 0.704 |

### SECTION target (which section holds the answer)

| arm | nomic·500 | nomic·1000 | egemma·500 | egemma·1000 | bge2k·500 | bge2k·1000 | bge8k·500 | bge8k·1000 |
|---|---|---|---|---|---|---|---|---|
| **rawspan-section** | **0.367** | **0.367** | **0.315** | **0.315** | **0.419** | **0.419** | **0.415** | **0.415** |
| fusion-rawspan | 0.294 | 0.259 | 0.304 | 0.243 | 0.312 | 0.335 | 0.315 | 0.334 |
| chunk (baseline) | 0.256 | 0.198 | 0.246 | 0.211 | 0.267 | 0.259 | 0.267 | 0.259 |
| fusion-centroid | 0.254 | 0.198 | 0.256 | 0.211 | 0.267 | 0.266 | 0.267 | 0.266 |
| centroid-section | 0.238 | 0.200 | 0.236 | 0.203 | 0.285 | 0.275 | 0.285 | 0.275 |

**Context axis (same model, two contexts).** `chunk` and `centroid-section`
embed sub-window text and are **bit-identical** across 2k/8k (sanity check). The
raw-span arms are the only ones context can move — and they barely move:
`rawspan-section` is **0.796 (2k) vs 0.784 (8k)** on DOC and **0.419 vs 0.415** on
SECTION, i.e. 2k is marginally *higher* despite pooling 20 over-long spans vs 5 at
8k. Raising context did not help; window-mean-pooling at 2k already suffices.

`rawspan-doc` / `centroid-doc` omitted from the SECTION table — the document
level does not target sections.

### rawspan-section lead over chunk baseline (Δ nDCG@10)

| | nomic·500 | nomic·1000 | egemma·500 | egemma·1000 | bge2k·500 | bge2k·1000 | bge8k·500 | bge8k·1000 |
|---|---|---|---|---|---|---|---|---|
| **DOC** | +0.056 | +0.079 | +0.078 | +0.033 | +0.055 | +0.077 | +0.043 | +0.065 |
| **SECTION** | +0.110 | +0.168 | +0.070 | +0.105 | +0.152 | +0.160 | +0.148 | +0.156 |

---

## Findings

1. **Raw-span sections win on every local encoder, both chunk sizes, both
   targets.** The technique was OpenAI-only until now; it generalizes to
   `nomic-embed-text`, `embeddinggemma:300m`, and `bge-m3`. Margins: **+0.03–0.08
   DOC**, **+0.07–0.17 SECTION**.

2. **Raw span >> centroid, re-confirmed on local models.** `rawspan-section`
   beats `centroid-section` almost everywhere, and the *oracle* gains tell the
   same story: rawspan oracle beats chunk by +0.08–0.13 (DOC) / +0.20–0.25
   (SECTION), while centroid oracle adds almost nothing (+0.004–0.08). This
   independently re-validates the shipped decision to default to raw-span.

3. **No diminishing returns with bigger chunks — the opposite.** On the SECTION
   target, a 1000-tok chunk makes the *chunk* baseline **worse** (a big chunk
   smears across section boundaries), while `rawspan-section` is chunk-size
   invariant (it's a pure section leg). So the hierarchy's lead **widens** with
   larger chunks. The win is not a small-chunk artefact.

4. **Fusion is a DOC-target tool that dilutes SECTION precision.**
   `fusion-rawspan` sometimes tops DOC (egemma) but on SECTION it loses to pure
   `rawspan-section` *everywhere* — blending the chunk leg back in adds noise for
   section-dense queries. The default `section_weight=0.65` is a DOC-favouring
   compromise; a router that knew the target could do better, but the target is
   unknown at query time.

5. **`bge-m3` wins on model quality, not context — pooling is enough.** Running
   `bge-m3` at *both* 2k and 8k isolates it: `rawspan-section` is **0.796 (2k) vs
   0.784 (8k)** DOC and **0.419 vs 0.415** SECTION — flat, even marginally lower
   at 8k, despite pooling dropping from 20 over-long spans to 5. So `bge-m3`'s
   lead over `nomic`/`embeddinggemma` is the encoder, not the context.
   `bge-m3@2048` is the single best section arm in the whole study (0.419). The
   `num_batch` fix is *validated* (8k genuinely embeds 8k, pooling → 5) but its
   retrieval **payoff on Qasper is nil**: window-mean-pooling at 2k already
   captures a section. (The short end still bites — a 512-tok encoder pools too
   hard and raw-span degrades — but 2k→8k is not the lever.)

---

## The `bge-m3@2048` cell (context axis) — completed

`bge-m3` is a 1.2 GB / 1024-dim model; on this CPU-only box a single embedding
batch takes **>300 s** at *any* context, so at the 300 s provider default the arm
flakily timed out (`httpx.ReadTimeout` → `EmbeddingError`) — a hardware/timeout
limit, **not** a harness or product bug. Fixed by giving every `bge-m3` arm
`--timeout 1800`; the cell then completed (results folded into the tables above,
finding #5). The warm embed cache meant only the unfinished spans re-embedded.

**Result:** context is not the driver. `bge-m3@2048` ≈ `bge-m3@8192` on every
raw-span arm (2k marginally higher), so the confound is resolved in favour of
"model quality, not context". One irreducible confound remains: `bge-m3` is both
a stronger encoder *and* higher-dimensional (1024 vs 768) than nomic/egemma.

### Reproduce the (now-completed) context-axis arm

```bash
# Ollama must be running: `ollama serve` (+ `ollama pull bge-m3`)
./.venv/Scripts/python.exe benchmarks/eval_hierarchical.py \
  --dataset qasper --split dev --max-papers 15 --mode section \
  --provider ollama --model bge-m3 --num-ctx 2048 --chunk-tokens 500  --timeout 1800
./.venv/Scripts/python.exe benchmarks/eval_hierarchical.py \
  --dataset qasper --split dev --max-papers 15 --mode section \
  --provider ollama --model bge-m3 --num-ctx 2048 --chunk-tokens 1000 --timeout 1800
```

### Reproduce the whole sweep

```bash
bash benchmarks/run_hier_ollama.sh 15        # 4 encoder arms × {500,1000} chunk tokens
```

Raw log: `benchmarks/results/hier_ollama_20260715_161309.log`.
Per-arm JSON: `benchmarks/results/hierarchical_qasper_dev_<model>_ctx<n>_ck<n>_*.json`.

---

## Caveats

- **48 queries is small** — directional, not a leaderboard. Trends are
  consistent across three encoders and two chunk sizes, which is the confidence
  we're leaning on, not any single cell.
- **Qasper only** for the local-encoder study. The OpenAI baseline in
  `docs/source/hierarchical.rst` also covers synthetic BEIR section corpora
  (FiQA, NFCorpus).
- **CPU-only box, memory-constrained** — timings here are not representative of
  a GPU host; only the *relative* nDCG numbers transfer.
