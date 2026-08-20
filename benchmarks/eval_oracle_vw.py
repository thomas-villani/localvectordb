"""Oracle per-query vector_weight: the ceiling on every adaptive-weighting idea.

Primer §8.3 item 7. ``vector_weight`` is regime-specific (NQ's argmax is 0.90
against the shipped 0.50), which invites a per-query router: predict the right
weight from the query and capture the spread. Before anyone builds that
classifier, this measures what a PERFECT router would win: score every query at
every weight on a grid, give each query its best weight (chosen by the answer
key — an oracle, not a method), and compare against the best single fixed
weight. The oracle-minus-fixed gap is an upper bound on any router's gain; if
it is small, no router is worth building, however clever.

Capture-once-sweep-offline: one vector pass and one FTS5 pass per corpus, then
every weight is a numpy re-blend. Zero embedding at run time (vectors come from
the shared cache and the run RAISES on a miss).

Both legs ride the shared cell/fusion/roll-up code (`eval_prf.build_cell`,
`eval_section_bm25.blend_arm`), so the numbers sit in the same unit space as
the k1/b/stemming/PRF results — comparable to that whole family, and like them
NOT identical to the src/ `query()` path (no `_hybrid_pool_size` truncation).

Usage:
    ./.venv/Scripts/python.exe benchmarks/eval_oracle_vw.py --dataset qasper --model-key egemma
    ./.venv/Scripts/python.exe benchmarks/eval_oracle_vw.py --dataset nq --model-key openai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_prf import build_cell  # noqa: E402
from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    SEARCH_K,
    VECTOR_WEIGHT,
    blend_arm,
    capture_arm,
    paired,
    score,
    to_ranked,
)

logger = logging.getLogger("oraclevw")

# 0.0 (pure keyword) to 1.0 (pure vector) in 0.05 steps: fine enough that the
# oracle is not starved by grid resolution, coarse enough to stay one numpy
# pass per point. The measured per-corpus argmaxes (0.3 qasper, 0.9 NQ) sit ON
# a coarser 0.1 grid, so 0.05 gives the oracle strictly more room than any
# fixed-weight result it is compared against.
VW_GRID = tuple(round(w * 0.05, 2) for w in range(21))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--level", choices=("chunks", "coarse"), default="chunks")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--per-query-out",
        type=Path,
        default=None,
        help="Dump per-query data for router training: qids/qtexts, the full "
        "per-query x per-weight nDCG matrix per target, and pre-fusion leg "
        "stats. The aggregate --out file is unchanged by this flag.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    cell = build_cell(args)
    uids, texts, udocs = cell.uids, cell.texts, cell.udocs
    qids, qtexts, q_scope = cell.qids, cell.qtexts, cell.q_scope

    vec_cap = [v for v, _ in capture_arm(uids, cell.uv, None, cell.qv, qtexts, udocs, q_scope)]
    fts = FTS(uids, texts, udocs)
    kw_cap = [fts.search(t, SEARCH_K, q_scope[i] if q_scope else None) for i, t in enumerate(qtexts)]
    captured = list(zip(vec_cap, kw_cap, strict=True))

    per_query: Optional[Dict[str, object]] = None
    if args.per_query_out:
        # Per-query leg stats, computed BEFORE fusion and roll-up, so they are
        # target-independent and cost nothing extra. kw scores are raw FTS5
        # bm25 where NEGATIVE is better (the fuse() negation happens later), so
        # "kw_bm25_best" is the MINIMUM. Query-text-only features (length,
        # quoting, IDF profile) are recomputed downstream from qtexts.
        leg: Dict[str, list] = {
            "n_kw_hits": [],
            "kw_bm25_best": [],
            "vec_top1": [],
            "vec_margin": [],
            "vec_top10_mean": [],
        }
        for vec, kw in captured:
            vs = sorted((float(s) for s in vec.values()), reverse=True)
            leg["n_kw_hits"].append(len(kw))
            leg["kw_bm25_best"].append(float(min(kw.values())) if kw else None)
            leg["vec_top1"].append(vs[0] if vs else None)
            leg["vec_margin"].append(vs[0] - vs[1] if len(vs) > 1 else None)
            leg["vec_top10_mean"].append(float(np.mean(vs[:10])) if vs else None)
        per_query = {
            "dataset": args.dataset,
            "level": args.level,
            "model_key": args.model_key,
            "vw_grid": list(VW_GRID),
            "qids": list(qids),
            "qtexts": list(qtexts),
            "leg": leg,
            "targets": {},
        }

    bench = cell.bench
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        # MAUD's document target is unrankable (22 distinct query texts).
        targets = [("section", bench.section_qrels)]

    results: Dict[str, Dict[str, object]] = {}
    for tname, qrels in targets:
        if not qrels:
            continue
        if args.level == "chunks":
            owner = cell.owner_sec if tname == "section" else cell.owner_doc
        else:
            owner = None if tname == "section" else cell.owner_doc
            if args.coarse == "documents" and tname == "section":
                continue

        # queries x weights matrix of per-query nDCG@10
        pq = np.stack(
            [score(to_ranked(blend_arm(captured, owner, vw, use_keyword=True)), qids, qrels) for vw in VW_GRID],
            axis=1,
        )
        means = pq.mean(axis=0)
        best_i = int(means.argmax())
        shipped_i = VW_GRID.index(VECTOR_WEIGHT)
        oracle_pq = pq.max(axis=1)

        # The gap that bounds every router: oracle vs the best FIXED weight
        # (not vs shipped -- a router competes against tuning, not defaults).
        st_fixed = paired(oracle_pq, pq[:, best_i])
        st_shipped = paired(oracle_pq, pq[:, shipped_i])
        moved = float((pq.argmax(axis=1) != best_i).mean())
        gains = oracle_pq - pq[:, best_i]
        big_gain = float((gains > 0.05).mean())

        tag = f"{args.dataset}/{args.level}/{tname}"
        print(f"\n=== {tag} · n={len(qids)} · grid {VW_GRID[0]:g}..{VW_GRID[-1]:g} step 0.05 ===")
        print(f"  shipped vw={VECTOR_WEIGHT:g}        nDCG@10 {means[shipped_i]:.4f}")
        print(f"  best fixed vw={VW_GRID[best_i]:g}     nDCG@10 {means[best_i]:.4f}")
        print(
            f"  ORACLE (per-query)    nDCG@10 {oracle_pq.mean():.4f}"
            f"   vs best fixed {st_fixed['delta']:+.4f} [{st_fixed['ci_lo']:+.4f},{st_fixed['ci_hi']:+.4f}]"
        )
        print(f"                                   vs shipped    {st_shipped['delta']:+.4f}")
        print(
            f"  {moved:.1%} of queries argmax at a non-fixed weight; "
            f"{big_gain:.1%} gain more than 0.05 nDCG from a personal weight"
        )
        if per_query is not None:
            per_query["targets"][tname] = {"ndcg_by_vw": [[float(x) for x in row] for row in pq]}  # type: ignore[index]
        results[tag] = {
            "n_queries": len(qids),
            "vw_grid": list(VW_GRID),
            "mean_by_vw": [float(m) for m in means],
            "shipped_vw": VECTOR_WEIGHT,
            "shipped_mean": float(means[shipped_i]),
            "best_fixed_vw": VW_GRID[best_i],
            "best_fixed_mean": float(means[best_i]),
            "oracle_mean": float(oracle_pq.mean()),
            "oracle_vs_best_fixed": st_fixed,
            "oracle_vs_shipped": st_shipped,
            "share_argmax_moves": moved,
            "share_gain_over_0.05": big_gain,
        }

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.out)
    if per_query is not None:
        args.per_query_out.write_text(json.dumps(per_query), encoding="utf-8")
        logger.info("Wrote %s", args.per_query_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
