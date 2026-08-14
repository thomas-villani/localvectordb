"""`b`, and BM25+ -- the two knobs FTS5's `bm25()` cannot reach. KEYWORD-STRATEGIES §1.2 / §1.5.

WHY THESE TWO TOGETHER. Both change how BM25 normalises for DOCUMENT LENGTH, and
neither is expressible through `bm25()`:

  * `b` scales the length penalty (`norm = 1-b + b*|d|/avgdl`). `bm25()` takes
    per-COLUMN weights, which multiply term frequency and are therefore
    algebraically a `k1` sweep (§1.2); its second argument is silently ignored on
    a one-column table, so `b` is unreachable. Confirmed there, not assumed.
  * BM25+ (Lv & Zhai 2011) adds a floor `idf*delta` per MATCHED term. Its premise
    is that the tf component decays to 0 as `|d|` grows, so a long document that
    CONTAINS a query term can rank below a short one that does not; the floor
    makes presence worth a fixed minimum.

Both now cost almost nothing, because `LexIndex` exists for PRF and `b` touches
only the precomputed `norm` vector -- postings, lengths and idf are independent
of both `k1` and `b`. So one index build serves the whole grid, and a 2-D
`k1` x `b` sweep is affordable where a 1-D one used to look expensive.

THE 2-D GRID IS THE POINT, not a bonus. §1.2 found a genuine interior optimum at
`k1`~0.3 on MAUD sections (+0.0234*) while sweeping `b` at its shipped 0.75.
`k1` and `b` both act on the same denominator `f + k1*(1-b + b*|d|/avgdl)`, so a
"`k1` optimum" measured at one `b` may be a point on a `k1` x `b` RIDGE rather
than an optimum in `k1` at all. Sweeping both separates those, and nothing in
the 1-D result can.

WHERE BM25+ SHOULD BITE, stated before measuring: MLDR, whose documents average
~30k chars. That is the regime the correction was designed for. If it helps
anywhere it should help there, and a null on MLDR is a much stronger result than
a null on qasper.

CONTROLS:

  0. PLUMBING. The grid point (k1=1.2, b=0.75, delta=0) IS shipped BM25, so it
     must reproduce FTS5's `bm25()` through the same fusion and roll-up path.
     Printed as `CONTROL shipped-vs-FTS5`, and it is the same end-to-end shape
     that caught a sign inversion in `eval_prf.py` -- an isolated comparison of
     two orderings is blind to how they are consumed.
  1. CROSS-FILE REGRESSION. The grid contains k1=0.3 and k1=1.2 at b=0.75, which
     is exactly the contrast `eval_bm25_k1.py` measured through FTS5's weight
     trick. Two independent implementations of the same sweep must agree; the
     harness prints that delta next to the published one so a disagreement
     cannot be missed.
  2. MULTIPLICITY, stated rather than corrected. Every grid point is tested
     against the same baseline, so the per-cell star is a NOMINAL 95% interval
     and the argmax of a 20-point grid is an argmax, not a discovery. Treat any
     winner here as a candidate for `eval_heldout_tuning.py`, which is the only
     thing in this study that can tell tuning from fitting.

Zero embedding: vectors come from the `hier_embed` disk cache.

    python benchmarks/eval_bm25_variants.py --dataset mldr --model-key openai
    python benchmarks/eval_bm25_variants.py --dataset maud --model-key openai --level coarse
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_prf import (  # noqa: E402
    B_SHIPPED,
    K1_SHIPPED,
    LexIndex,
    build_cell,
    query_model,
    tokenize_queries,
)
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

logger = logging.getLogger("bm25var")

# `b`=0 turns length normalisation OFF entirely and `b`=1 applies it fully, so
# the grid spans the whole meaningful range rather than a neighbourhood of the
# default. The §1.2 lesson was that the interesting value sat well outside the
# range anyone would have guessed, which is an argument for wide grids.
B_GRID = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)

# Two k1 values, not twelve: 1.2 is shipped and 0.3 is §1.2's measured optimum.
# The question here is whether that optimum MOVES with `b`, which needs only the
# two rows -- a full 12x9 grid would cost 6x more and answer the same question.
K1_GRID = (0.3, K1_SHIPPED)

# delta=1.0 is the value Lv & Zhai recommend; 0.5 and 1.5 bracket it so a null
# at 1.0 can be distinguished from a null everywhere.
DELTA_GRID = (0.5, 1.0, 1.5)


def rank_all(
    index: LexIndex,
    qmodels: Sequence[Dict[str, float]],
    masks: Sequence[Optional[np.ndarray]],
) -> List[Dict[str, float]]:
    """Score every query at the index's CURRENT (k1, b, delta)."""
    return [index.score(qm, SEARCH_K, m) for qm, m in zip(qmodels, masks, strict=True)]


def evaluate(
    kw_caps: Sequence[Dict[str, float]],
    vec_cap: Sequence[Dict[str, float]],
    owner: Optional[Dict[str, List[str]]],
    vw: float,
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Per-query nDCG@10 through the shipped fusion and roll-up path."""
    captured = list(zip(vec_cap, kw_caps, strict=True))
    return score(to_ranked(blend_arm(captured, owner, vw, use_keyword=True)), qids, qrels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--level", choices=("chunks", "coarse"), default="chunks")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument("--b-grid", type=float, nargs="*", default=list(B_GRID))
    p.add_argument("--k1-grid", type=float, nargs="*", default=list(K1_GRID))
    p.add_argument("--delta-grid", type=float, nargs="*", default=list(DELTA_GRID))
    p.add_argument("--delta-k1", type=float, nargs="*", default=[K1_SHIPPED], help="k1 values to apply BM25+ at")
    p.add_argument("--delta-b", type=float, nargs="*", default=[B_SHIPPED], help="b values to apply BM25+ at")
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    cell = build_cell(args)
    uids, texts, udocs = cell.uids, cell.texts, cell.udocs
    qids, qtexts, q_scope = cell.qids, cell.qtexts, cell.q_scope

    index = LexIndex(uids, texts, udocs)
    logger.info("lexical index: %d terms, avgdl %.0f", len(index.term_ids), index.avgdl)

    doc_of_row = np.array(udocs, dtype=object)
    masks: List[Optional[np.ndarray]] = [
        None if q_scope is None else (doc_of_row == q_scope[i]) for i in range(len(qtexts))
    ]
    qmodels = [query_model(t) for t in tokenize_queries(qtexts)]
    vec_cap = [v for v, _ in capture_arm(uids, cell.uv, None, cell.qv, qtexts, udocs, q_scope)]

    # The grid, as (label, k1, b, delta). `base` -- shipped k1/b with no BM25+
    # floor -- goes FIRST so it is the reference every later row is differenced
    # against. It is called `base` and not `shipped` because it is shipped
    # PARAMETERS, not shipped behaviour; CONTROL 0 below measures the remainder.
    grid: List[Tuple[str, float, float, float]] = [("base", K1_SHIPPED, B_SHIPPED, 0.0)]
    for k1 in args.k1_grid:
        for b in args.b_grid:
            if (k1, b, 0.0) == (K1_SHIPPED, B_SHIPPED, 0.0):
                continue
            grid.append((f"k1={k1:g} b={b:g}", k1, b, 0.0))
    # BM25+ at shipped parameters, and -- if asked -- on top of the k1/b argmax.
    # THE COMBINATION IS THE INTERESTING PART. On MAUD sections low k1, low b and
    # a BM25+ floor each buy ~+0.02 alone, while low k1 AND low b together buy
    # +0.0017: they are substitutes, not additive, which is what one underlying
    # mechanism looks like when two knobs both over-correct it. Whether the BM25+
    # floor is a third route to the same place, or something separable, cannot be
    # read off the single-knob rows -- it needs the cross product.
    for d in args.delta_grid:
        for ck1 in args.delta_k1:
            for cb in args.delta_b:
                lab = "BM25+ d={:g}".format(d)
                if (ck1, cb) != (K1_SHIPPED, B_SHIPPED):
                    lab += f" @k1={ck1:g} b={cb:g}"
                grid.append((lab, ck1, cb, d))

    logger.info("%d grid points x %d queries", len(grid), len(qtexts))
    caps: Dict[str, List[Dict[str, float]]] = {}
    for label, k1, b, d in grid:
        caps[label] = rank_all(index.reparam(k1=k1, b=b, delta=d), qmodels, masks)
    # FTS5's own ranking, as an ARM through the same path -- not a side
    # assertion. See CONTROL 0.
    fts = FTS(uids, texts, udocs)
    caps["FTS5"] = [fts.search(t, SEARCH_K, q_scope[i] if q_scope else None) for i, t in enumerate(qtexts)]

    # THE ARM THAT DIAGNOSES A CROSS-FILE DISAGREEMENT. `eval_bm25_k1.py` sweeps
    # k1 through FTS5's column weight (k1_eff = 1.2/w), which is algebraically
    # true BM25 at k1_eff times a document-independent constant -- so it cannot
    # reorder, and it must match `LexIndex` at the same k1. If the two differ,
    # exactly one thing is left to differ: the QUERY MODEL. FTS5 sees shipped
    # sanitisation, where a hyphenated token is an adjacency phrase; LexIndex is
    # positionless. Carrying the weight trick as an arm here turns "two files
    # disagree" into a measurement of WHICH difference is responsible, at the
    # same k1 rather than across cells.
    for k1 in args.k1_grid:
        if k1 <= 0:
            continue
        w = K1_SHIPPED / k1
        idx = FTS(uids, texts, udocs, weight=w)
        caps[f"FTS5 k1eff={k1:g}"] = [
            idx.search(t, SEARCH_K, q_scope[i] if q_scope else None) for i, t in enumerate(qtexts)
        ]

    bench = cell.bench
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    results: Dict[str, Dict[str, float]] = {}
    for tname, qrels in targets:
        if not qrels:
            continue
        if args.level == "chunks":
            owner = cell.owner_sec if tname == "section" else cell.owner_doc
        else:
            owner = None if tname == "section" else cell.owner_doc
            if args.coarse == "documents" and tname == "section":
                continue

        for arm_name, vw in (("keyword", 0.0), ("hybrid", args.vector_weight)):
            pq = {lab: evaluate(caps[lab], vec_cap, owner, vw, qids, qrels) for lab in caps}
            base = pq["base"]
            tag = f"{args.dataset}/{args.level}/{tname}/{arm_name}"
            print(f"\n=== {tag} · vw={vw:.2f} · n={len(qids)} ===")
            print(f"  {'variant':>16} {'nDCG@10':>9} {'delta':>9} {'95% CI':>20}")
            for label, *_ in grid:
                st = paired(pq[label], base)
                ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
                print(f"  {label:>16} {pq[label].mean():>9.4f} {st['delta']:>+9.4f} {ci:>20}{star}")
                results[f"{tag}|{label}"] = {
                    "ndcg@10": float(pq[label].mean()),
                    "delta": st["delta"],
                    "ci_lo": st["ci_lo"],
                    "ci_hi": st["ci_hi"],
                }

            # ---- CONTROL 0: is the grid's origin actually shipped BM25? ----
            # It is shipped PARAMETERS over a bag-of-words query, which is not
            # quite shipped BEHAVIOUR: `LexIndex` has no positions, so a
            # hyphenated query token that src quotes into an adjacency phrase is
            # scored here as independent terms. This gap is therefore not noise
            # to be tolerated under a threshold -- it is the hyphen effect that
            # `eval_hyphen.py` measures directly, showing up as the one thing
            # separating the two scorers. Read it against that file's `split`
            # arm; a value far from it would mean something ELSE also differs.
            gap = float(pq["base"].mean() - pq["FTS5"].mean())
            flag = "" if abs(gap) < 0.01 else "   !! larger than the hyphen effect alone -- something else differs"
            print(
                f"  CONTROL base-vs-FTS5 through the same path: {gap:+.4f}{flag}\n"
                "    (expected non-zero: this scorer is positionless, so it cannot reproduce the"
                " adjacency phrases src builds from hyphenated tokens -- cf. eval_hyphen.py)"
            )

            # ---- CONTROL 1: do the two k1 implementations agree? ----
            # Both rows are differenced against their OWN scorer's baseline --
            # LexIndex row vs `base`, FTS5 row vs `FTS5` -- so the phrase gap
            # cancels out of each delta and what remains is the k1 effect itself.
            # Differencing both against `base` would re-import the very
            # difference this control is trying to isolate.
            for k1 in args.k1_grid:
                mine_lab, theirs_lab = f"k1={k1:g} b={B_SHIPPED:g}", f"FTS5 k1eff={k1:g}"
                if mine_lab not in pq or theirs_lab not in pq:
                    continue
                mine = paired(pq[mine_lab], base)
                theirs = paired(pq[theirs_lab], pq["FTS5"])
                agree = abs(mine["delta"] - theirs["delta"]) < 0.005
                print(
                    f"  CONTROL k1={k1:g}: LexIndex {mine['delta']:+.4f} vs FTS5 weight-trick "
                    f"{theirs['delta']:+.4f}  ({'agree' if agree else 'DISAGREE -- query model, not k1'})"
                )

            # ---- Does the k1 optimum MOVE with b? ----
            for k1 in args.k1_grid:
                # The shipped point has no `k1=... b=...` row of its own, so it
                # enters the argmax under its own label rather than being skipped.
                row = []
                for b in args.b_grid:
                    lab = f"k1={k1:g} b={b:g}"
                    row.append((b, float(pq[lab].mean()) if lab in pq else float(base.mean())))
                best_b, best_v = max(row, key=lambda t: t[1])
                print(f"  argmax over b at k1={k1:g}: b={best_b:g} ({best_v:.4f})")
            print(
                "  NOTE multiplicity: every row is tested against the same baseline, so a star is a "
                "NOMINAL 95% interval and the argmax above is an argmax, not a finding."
            )

    if args.out:
        payload = {"config": vars(args) | {"out": str(args.out)}, "results": results}
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
