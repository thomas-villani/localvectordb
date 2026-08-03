"""The granularity ladder at DOCUMENT level, on a corpus with no structure.

``span-length-crossover-findings`` §6.21/§6.24 established, inside two rigidly
sectioned corpora, that ``rawspan`` and ``centroid`` are one operator at two
granularities and that finer wins monotonically on long spans. Both corpora were
picked *because* their headings were clean. MLDR-en has **none** (0/800 docs),
so this asks the same question one level up, where headings are not needed:

    doc_rawspan@W  -- the whole document embedded in W-sized windows, mean-pooled
    doc_centroid   -- the mean of the document's chunk vectors
    chunk          -- max-pooled chunk similarities (the strong baseline)

As W shrinks, ``doc_rawspan`` must converge on ``doc_centroid``. If the §6.21
ladder is a property of the operator rather than of contracts and papers, the
same monotone curve appears here.

WHAT THIS ISOLATES THAT MAUD COULD NOT
--------------------------------------
§6.25/1's "structure alignment" residual conflated two things: sentence-aligned
chunk boundaries vs arbitrary char cuts, and cross-section leakage from
``overlap`` attribution. **At document level there is no attribution at all** --
a chunk belongs to exactly one document -- so the leakage term vanishes and the
matched-granularity rung isolates *boundary placement alone*. Note this needs
sentences, not headings, so it survives MLDR's lack of structure.

Cost: each rung re-windows every document, which is a fresh set of window texts
and therefore cache misses -- roughly one pass over the corpus per rung
(~13M tokens, ~$0.27 on text-embedding-3-small). ``--allow-embed`` is required,
matching the rest of the harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks import eval_dual as ed  # noqa: E402
from benchmarks.config import RESULTS_DIR  # noqa: E402
from benchmarks.eval_hierarchical import _chunker, _estimate_tokens, _unit  # noqa: E402
from benchmarks.mldr_data import load_mldr  # noqa: E402

logger = logging.getLogger("eval_docgran")

# Coarse -> fine. `None` means the spec's own setting (token-exact 8191 on
# openai), i.e. the coarsest achievable window for that encoder.
DEFAULT_RUNGS = (None, 24_000, 12_000, 6_000, 3_000, 1_833)


def build_units(bench, chunk_tokens: int):
    """Chunk every document once; return parallel chunk arrays plus doc order."""
    chunker = _chunker(chunk_tokens)
    chunk_texts: List[str] = []
    chunk_doc: List[str] = []
    doc_ids = list(bench.corpus)
    for doc_id in doc_ids:
        for ch in chunker.chunk(bench.corpus[doc_id]):
            chunk_texts.append(ch.content)
            chunk_doc.append(doc_id)
    logger.info(
        "Chunked: %d chunks over %d docs (%.1f/doc)", len(chunk_texts), len(doc_ids), len(chunk_texts) / len(doc_ids)
    )
    return doc_ids, chunk_texts, chunk_doc


def score(units_sorted: Sequence[str], pooled: np.ndarray, qids: Sequence[str], qrels) -> np.ndarray:
    pq, _ = ed.score_arm(units_sorted, pooled, qids, qrels)
    return pq


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai", help="MODEL_POOL key.")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--chunk-tokens", type=int, default=500)
    ap.add_argument("--rungs", default=None, help="Comma-separated window chars; 'none' = the spec's own.")
    ap.add_argument("--allow-embed", action="store_true", help="Permit embedding on cache miss.")
    ap.add_argument("--tag", default="docgran")
    args = ap.parse_args(argv)

    ed._load_experiments_env()
    spec = ed.MODEL_POOL[args.model]
    bench = load_mldr(split=args.split, max_queries=args.max_queries)
    doc_ids, chunk_texts, chunk_doc = build_units(bench, args.chunk_tokens)
    qids = list(bench.queries)
    query_texts = [bench.queries[q] for q in qids]
    doc_texts = [bench.corpus[d] for d in doc_ids]

    rungs: List[Optional[int]] = list(DEFAULT_RUNGS)
    if args.rungs:
        rungs = [None if r.strip().lower() == "none" else int(r) for r in args.rungs.split(",") if r.strip()]

    total_tok = sum(_estimate_tokens(t) for t in doc_texts)
    logger.info(
        "Corpus ~%.1fM est tokens; %d rungs => ~%.1fM doc-embedding tokens (plus chunks)",
        total_tok / 1e6,
        len(rungs),
        total_tok * len(rungs) / 1e6,
    )

    def encoder(window: Optional[int]):
        s = spec if window is None else replace(spec, window_chars=window, window_tokens=None)
        return ed.PrefixedEncoder(s, s.doc_prefix)

    # --- shared arms -------------------------------------------------------
    doc_enc = encoder(None)
    qry_enc = ed.PrefixedEncoder(spec, spec.query_prefix)
    for name, enc, texts in (("chunks", doc_enc, chunk_texts), ("queries", qry_enc, query_texts)):
        _, misses = enc.count_misses(texts)
        if misses and not args.allow_embed:
            raise SystemExit(f"[{name}] {misses} vectors missing; pass --allow-embed to spend.")
    chunk_vecs = _unit(doc_enc.encode(chunk_texts, normalize=False))
    qvecs = _unit(qry_enc.encode(query_texts, normalize=False))

    chunk_pooler = ed.Pooler(chunk_doc, mode="max")
    pq_chunk = score(
        chunk_pooler.units, chunk_pooler.pool((qvecs @ chunk_vecs.T).astype(np.float32)), qids, bench.doc_qrels
    )

    # doc_centroid: mean of the doc's unit-normed chunk vectors, then unit-normed
    row = {d: i for i, d in enumerate(doc_ids)}
    cent = np.zeros((len(doc_ids), chunk_vecs.shape[1]), dtype=np.float32)
    counts = np.zeros(len(doc_ids), dtype=np.int64)
    for i, d in enumerate(chunk_doc):
        cent[row[d]] += chunk_vecs[i]
        counts[row[d]] += 1
    cent = _unit(cent / np.maximum(counts, 1)[:, None])
    doc_pooler = ed.Pooler(doc_ids, mode="max")
    pq_cent = score(doc_pooler.units, doc_pooler.pool((qvecs @ cent.T).astype(np.float32)), qids, bench.doc_qrels)

    out: Dict[str, object] = {
        "corpus": bench.name,
        "model": spec.model,
        "n_docs": len(doc_ids),
        "n_queries": len(qids),
        "chunk_tokens": args.chunk_tokens,
        "arms": {"chunk": float(pq_chunk.mean()), "doc_centroid": float(pq_cent.mean())},
        "rungs": {},
    }
    logger.info("chunk max-pool  nDCG@10 %.4f", pq_chunk.mean())
    logger.info("doc_centroid    nDCG@10 %.4f", pq_cent.mean())

    # --- the ladder --------------------------------------------------------
    for window in rungs:
        enc = encoder(window)
        _, misses = enc.count_misses(doc_texts)
        if misses and not args.allow_embed:
            raise SystemExit(f"[rung {window}] {misses} windows missing; pass --allow-embed.")
        raw = _unit(enc.encode(doc_texts, normalize=False))
        pq = score(doc_pooler.units, doc_pooler.pool((qvecs @ raw.T).astype(np.float32)), qids, bench.doc_qrels)
        b = ed.paired_bootstrap(pq, pq_cent)
        label = "spec (token-exact)" if window is None else f"{window} ch"
        out["rungs"][label] = {  # type: ignore[index]
            "ndcg": float(pq.mean()),
            "vs_centroid": b,
            "pooled_docs": int(enc.n_pooled),
        }
        logger.info(
            "doc_rawspan @%-18s nDCG@10 %.4f   vs centroid %+.4f [%+.4f,%+.4f] p_win %.2f",
            label,
            pq.mean(),
            b["delta"],
            b["ci_lo"],
            b["ci_hi"],
            b["p_win"],
        )

    print("\n" + "=" * 78)
    print(f"{bench.name}  {spec.model}  {len(doc_ids)} docs / {len(qids)} queries")
    print(f"{'arm':26s} {'nDCG@10':>9s}  {'vs doc_centroid':>28s}")
    print(f"{'chunk max-pool':26s} {pq_chunk.mean():9.4f}")
    print(f"{'doc_centroid (finest)':26s} {pq_cent.mean():9.4f}")
    for label, r in out["rungs"].items():  # type: ignore[union-attr]
        b = r["vs_centroid"]
        ci = f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
        print(f"{'doc_rawspan @' + label:26s} {r['ndcg']:9.4f}  {b['delta']:+8.4f} {ci} {b['p_win']:.2f}")
    print("=" * 78)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"docgran_{args.tag}_{bench.name}_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
