"""Mixed hierarchy: keyword at a COARSE level, vector at the FINE level.

Every cross-granularity blend measured so far (`fused`, S17) runs the SAME
``search_type`` on both legs; every cross-level mechanism (the cascade, S18.6)
was keyword-at-documents gating vector-at-documents. Nobody has asked what
happens when the legs are *asymmetric*: BM25 over sections or whole documents,
cosine over chunks. The prior says this is the right way round -- the
granularity law makes the finest vector leg the strongest one, while BM25 is
the leg whose best unit is corpus-dependent (diluted on qasper papers, excellent
on NQ/MLDR documents) -- so the coarse keyword leg gets to be a different
*signal*, not a worse copy of chunk BM25.

Three mechanisms, all built from one chunk-vector capture and one coarse FTS5
index, so the whole sweep is offline numpy and needs NO coarse vectors at all
(which is itself the attraction: a mixed hierarchy is cheaper to build than
`fused`, which embeds every section).

``parallel``
    ``fuse_levels(chunk_leg, coarse_bm25_leg, sw)`` -- the shipped `fused`
    topology with the section leg replaced by a keyword-only leg. The chunk leg
    is vector-only (vw=1.0) or hybrid at vw. No recall ceiling.

``cascade@N``
    Top-N coarse keyword units gate the chunk vector search: cosine is computed
    only over chunks whose parent survived. Keyword recall@N at the target is a
    HARD ceiling, and is reported next to the score so the trade is visible. A
    query with no keyword hits falls back to the ungated chunk search (share
    reported).

``inherit``
    The chunk pool is the vector top-K as shipped, but each chunk's keyword
    score is its PARENT's BM25 rather than its own -- the hybrid fusion at chunk
    level with coarse-context keyword evidence. Pool unchanged; only the keyword
    column moves.

Incumbents: chunk vector-only, chunk hybrid at the best fixed vw (the shipped
default's tuned form), and -- when the coarse vectors happen to be cached --
tuned two-stage ``fused``. READ EVERY ARGMAX AS AN UPPER BOUND: each arm is
tuned on the queries it is scored on, exactly as in eval_fused_blend.py. The
question this file answers is whether the mixed forms have headroom at all, not
what to ship.

Same unit space as eval_oracle_vw / eval_prf / eval_section_bm25 (numpy over
src's own sanitiser and fusion arithmetic; no `_hybrid_pool_size` truncation),
so the numbers sit next to that family and NOT next to `db.query()`.

Usage:
    ./.venv/Scripts/python.exe benchmarks/eval_mixed_hierarchy.py --dataset qasper --model-key egemma
    ./.venv/Scripts/python.exe benchmarks/eval_mixed_hierarchy.py --dataset nq --model-key openai
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_prf import build_cell  # noqa: E402
from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    SEARCH_K,
    VECTOR_WEIGHT,
    blend_arm,
    capture_arm,
    fuse,
    fuse_levels,
    paired,
    rollup,
    score,
    to_ranked,
    vector_top,
)
from benchmarks.metrics import recall_at_k  # noqa: E402

logger = logging.getLogger("mixedhier")

VW_GRID = tuple(round(w * 0.1, 1) for w in range(11))  # 0.0 .. 1.0
# Chunk-leg weights tried inside `parallel`: vector-only, plus three hybrid points
# spanning the measured per-corpus argmaxes (0.3 qasper .. 0.9 NQ).
PARALLEL_VW = (1.0, 0.9, 0.7, 0.5, 0.3)
SW_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
CASCADE_N = (5, 10, 20, 50, 100)

Captured = List[Tuple[Dict[str, float], Dict[str, float]]]


def _ci(st: Dict[str, float], p: bool = True) -> str:
    out = f"{st['delta']:+.4f} [{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
    return out + (f" p={st['p']:.3f}" if p else "")


def _best(pq_by_cfg: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    name = max(pq_by_cfg, key=lambda n: float(pq_by_cfg[n].mean()))
    return name, pq_by_cfg[name]


def _kw_only_leg(kw: Dict[str, float]) -> Dict[str, float]:
    """Raw bm25 (negative-is-better) -> [0,1] positive, via src's own fusion rule at vw=0."""
    return fuse({}, kw, 0.0) if kw else {}


def _coarse_to_target(
    kw_per_query: Sequence[Dict[str, float]], coarse_owner: Optional[Dict[str, Sequence[str]]]
) -> List[Dict[str, float]]:
    out = []
    for kw in kw_per_query:
        leg = _kw_only_leg(kw)
        out.append(rollup(leg, coarse_owner) if coarse_owner is not None else leg)
    return out


def _rank_ceiling(
    coarse_kw: Sequence[Dict[str, float]],
    coarse_owner: Optional[Dict[str, Sequence[str]]],
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    n: int,
) -> float:
    """Recall@n of the coarse keyword leg at the target -- the cascade's hard ceiling."""
    vals = []
    for q, kw in zip(qids, coarse_kw, strict=True):
        top = [u for u, _ in sorted(kw.items(), key=lambda kv: kv[1])[:n]]  # raw bm25: ascending
        if coarse_owner is not None:
            seen: List[str] = []
            for u in top:
                for p in coarse_owner[u]:
                    if p not in seen:
                        seen.append(p)
            top = seen
        vals.append(recall_at_k(top, qrels.get(q, {}), max(len(top), 1)))
    return float(np.mean(vals)) if vals else float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", nargs="+", choices=("sections", "documents"), default=["sections", "documents"])
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument("--fused-control", action="store_true", help="also score tuned fused (needs cached coarse vectors)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    args.level = "chunks"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    cell = build_cell(args)
    units, bench = cell.units, cell.bench
    uids, texts, udocs = cell.uids, cell.texts, cell.udocs
    qids, qtexts, q_scope = cell.qids, cell.qtexts, cell.q_scope

    # --- fine level: chunk vector + chunk BM25 (the incumbent's two legs) ---
    chunk_fts = FTS(uids, texts, udocs)
    chunk_cap: Captured = capture_arm(uids, cell.uv, chunk_fts, cell.qv, qtexts, udocs, q_scope)
    chunk_parent = {
        "sections": dict(zip(uids, units.chunk_section, strict=True)),
        "documents": dict(zip(uids, udocs, strict=True)),
    }

    # --- coarse levels: text only, no vectors ---
    coarse_levels: Dict[str, Dict[str, object]] = {}
    for c in args.coarse:
        if c == "sections":
            cids, ctexts, cdocs = list(units.section_ids), list(units.section_texts), list(units.section_doc)
        else:
            if args.dataset == "maud":
                logger.info("MAUD is scoped per contract; a document-level keyword leg is a no-op. Skipping.")
                continue
            cids = list(bench.corpus)
            ctexts = [bench.corpus[d] for d in cids]
            cdocs = list(cids)
        fts = FTS(cids, ctexts, cdocs)
        # Mirror capture_arm's keyword leg: over-fetch 2x, truncate to SEARCH_K.
        # The cascade needs a wider list, so keep the full over-fetch separately.
        wide = [fts.search(t, max(CASCADE_N) * 2, q_scope[i] if q_scope else None) for i, t in enumerate(qtexts)]
        kw = [dict(list(w.items())[:SEARCH_K]) for w in wide]
        children: Dict[str, np.ndarray] = {}
        for i, parent in enumerate(chunk_parent[c].values()):
            children.setdefault(parent, []).append(i)  # type: ignore[arg-type]
        children = {k: np.asarray(v, dtype=np.int64) for k, v in children.items()}
        owner_doc = {u: [d] for u, d in zip(cids, cdocs, strict=True)}
        coarse_levels[c] = {"kw": kw, "wide": wide, "children": children, "owner_doc": owner_doc, "n_units": len(cids)}
        logger.info(
            "coarse=%s: %d units; %.1f%% of queries have keyword hits",
            c,
            len(cids),
            100 * np.mean([bool(k) for k in kw]),
        )

    # --- optional fused control: needs coarse VECTORS ---
    fused_cells: Dict[str, object] = {}
    if args.fused_control:
        for c in coarse_levels:
            a2 = copy.copy(args)
            a2.level, a2.coarse = "coarse", c
            try:
                fused_cells[c] = build_cell(a2)
            except SystemExit as e:
                logger.info("fused control at coarse=%s skipped: %s", c, e)

    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    results: Dict[str, Dict[str, object]] = {}
    for tname, qrels in targets:
        if not qrels:
            continue
        owner_t = cell.owner_sec if tname == "section" else cell.owner_doc

        def pq_of(per_query: List[Dict[str, float]], qrels: Dict[str, Dict[str, int]] = qrels) -> np.ndarray:
            return score(to_ranked(per_query), qids, qrels)

        # Incumbents.
        base_vec = pq_of(blend_arm(chunk_cap, owner_t, 1.0, use_keyword=False))
        hyb = {f"chunk·hybrid@{vw:g}": pq_of(blend_arm(chunk_cap, owner_t, vw)) for vw in VW_GRID if vw < 1.0}
        hyb_best_name, hyb_best = _best(hyb)
        hyb_shipped = hyb[f"chunk·hybrid@{VECTOR_WEIGHT:g}"]
        print(f"\n=== {args.dataset} / target={tname} · n={len(qids)} ===")
        print(f"  chunk·vector                 {base_vec.mean():.4f}")
        print(f"  chunk·hybrid@{VECTOR_WEIGHT:g} (shipped)    {hyb_shipped.mean():.4f}")
        print(f"  {hyb_best_name:<28} {hyb_best.mean():.4f}   <- INCUMBENT (best fixed)")
        cell_res: Dict[str, object] = {
            "n_queries": len(qids),
            "chunk_vector": float(base_vec.mean()),
            "chunk_hybrid_shipped": float(hyb_shipped.mean()),
            "incumbent": {"name": hyb_best_name, "mean": float(hyb_best.mean())},
            "coarse": {},
        }

        for c, cl in coarse_levels.items():
            kw: List[Dict[str, float]] = cl["kw"]  # type: ignore[assignment]
            wide: List[Dict[str, float]] = cl["wide"]  # type: ignore[assignment]
            children: Dict[str, np.ndarray] = cl["children"]  # type: ignore[assignment]
            # How the coarse unit reaches the target: section->section identity,
            # section->doc via owner_doc, doc->doc identity, doc->section impossible.
            if c == "sections":
                coarse_owner = None if tname == "section" else cl["owner_doc"]
            else:
                coarse_owner = None
            parallel_ok = not (c == "documents" and tname == "section")
            print(f"\n  --- coarse keyword = {c} ({cl['n_units']} units) ---")
            arms: Dict[str, object] = {}

            # 1. parallel
            if parallel_ok:
                coarse_side = _coarse_to_target(kw, coarse_owner)  # type: ignore[arg-type]
                kw_alone = pq_of(coarse_side)
                grid: Dict[str, np.ndarray] = {}
                for vw in PARALLEL_VW:
                    chunk_side = blend_arm(chunk_cap, owner_t, vw, use_keyword=vw < 1.0)
                    for sw in SW_GRID:
                        if sw in (0.0, 1.0):
                            continue  # sw=0 is the chunk leg alone, sw=1 the coarse leg alone
                        grid[f"parallel vw={vw:g} sw={sw:g}"] = pq_of(fuse_levels(chunk_side, coarse_side, sw))
                name, best = _best(grid)
                st = paired(best, hyb_best)
                st_v = paired(best, base_vec)
                # Same mechanism restricted to the vector-only chunk leg: the pure
                # "coarse keyword + fine vector" form, no chunk BM25 anywhere.
                pure = {k: v for k, v in grid.items() if "vw=1 " in k}
                pname, pbest = _best(pure)
                stp = paired(pbest, hyb_best)
                print(f"  coarse-bm25 alone            {kw_alone.mean():.4f}")
                print(f"  {pname:<28} {pbest.mean():.4f}   vs incumbent {_ci(stp)}   (no chunk BM25)")
                print(f"  {name:<28} {best.mean():.4f}   vs incumbent {_ci(st)}   vs chunk·vector {st_v['delta']:+.4f}")
                arms["parallel"] = {
                    "coarse_bm25_alone": float(kw_alone.mean()),
                    "grid": {k: float(v.mean()) for k, v in grid.items()},
                    "best": {"name": name, "mean": float(best.mean()), "vs_incumbent": st, "vs_chunk_vector": st_v},
                    "best_pure": {"name": pname, "mean": float(pbest.mean()), "vs_incumbent": stp},
                }

            # 2. cascade@N -- vector only inside the survivors.
            casc: Dict[str, np.ndarray] = {}
            casc_meta: Dict[str, Dict[str, float]] = {}
            for n in CASCADE_N:
                per_query: List[Dict[str, float]] = []
                fallback = 0
                for qi, w in enumerate(wide):
                    top = [u for u, _ in sorted(w.items(), key=lambda kv: kv[1])[:n]]
                    idx_parts = [children[u] for u in top if u in children]
                    if not idx_parts:
                        fallback += 1
                        per_query.append(rollup(chunk_cap[qi][0], owner_t))
                        continue
                    idx = np.concatenate(idx_parts)
                    sims = cell.uv[idx] @ cell.qv[qi]
                    vec = vector_top(sims, [uids[i] for i in idx], SEARCH_K)
                    per_query.append(rollup(vec, owner_t))
                casc[f"cascade@{n}"] = pq_of(per_query)
                casc_meta[f"cascade@{n}"] = {
                    # A document id can never name a section: no ceiling to report there.
                    "ceiling_recall": (
                        float("nan") if not parallel_ok else _rank_ceiling(wide, coarse_owner, qids, qrels, n)  # type: ignore[arg-type]
                    ),
                    "fallback_share": fallback / len(qids),
                }
            name, best = _best(casc)
            st = paired(best, hyb_best)
            st_v = paired(best, base_vec)
            for k, v in casc.items():
                m = casc_meta[k]
                ceiling, fb = m["ceiling_recall"], m["fallback_share"]
                print(f"  {k:<28} {v.mean():.4f}   ceiling recall {ceiling:.3f}  fallback {fb:.1%}")
            print(f"  best {name:<23} vs incumbent {_ci(st)}   vs chunk·vector {st_v['delta']:+.4f}")
            arms["cascade"] = {
                "grid": {k: {"mean": float(v.mean()), **casc_meta[k]} for k, v in casc.items()},
                "best": {"name": name, "mean": float(best.mean()), "vs_incumbent": st, "vs_chunk_vector": st_v},
            }

            # 3. inherit -- parent's BM25 stands in for the chunk's own.
            parent_of = chunk_parent[c]
            inh_cap: Captured = []
            for (vec, _own_kw), w in zip(chunk_cap, wide, strict=True):
                pk = dict(list(w.items())[:SEARCH_K])
                inh_cap.append((vec, {u: pk[parent_of[u]] for u in vec if parent_of[u] in pk}))
            inh = {f"inherit@{vw:g}": pq_of(blend_arm(inh_cap, owner_t, vw)) for vw in VW_GRID if vw < 1.0}
            name, best = _best(inh)
            st = paired(best, hyb_best)
            st_v = paired(best, base_vec)
            print(f"  {name:<28} {best.mean():.4f}   vs incumbent {_ci(st)}   vs chunk·vector {st_v['delta']:+.4f}")
            arms["inherit"] = {
                "grid": {k: float(v.mean()) for k, v in inh.items()},
                "best": {"name": name, "mean": float(best.mean()), "vs_incumbent": st, "vs_chunk_vector": st_v},
            }

            # 4. tuned fused control (both legs hybrid, coarse VECTORS required).
            fc = fused_cells.get(c)
            if fc is not None and parallel_ok:
                ccap = capture_arm(fc.uids, fc.uv, FTS(fc.uids, fc.texts, fc.udocs), fc.qv, qtexts, fc.udocs, q_scope)  # type: ignore[attr-defined]
                fgrid: Dict[str, np.ndarray] = {}
                for vw in PARALLEL_VW:
                    chunk_side = blend_arm(chunk_cap, owner_t, vw, use_keyword=vw < 1.0)
                    coarse_hs = blend_arm(ccap, coarse_owner, vw, use_keyword=vw < 1.0)  # type: ignore[arg-type]
                    for sw in SW_GRID[1:-1]:
                        fgrid[f"fused vw={vw:g} sw={sw:g}"] = pq_of(fuse_levels(chunk_side, coarse_hs, sw))
                name, best = _best(fgrid)
                st = paired(best, hyb_best)
                print(f"  {name:<28} {best.mean():.4f}   vs incumbent {_ci(st, p=False)}   <- tuned fused control")
                if "parallel" in arms:
                    pb = grid[arms["parallel"]["best"]["name"]]  # type: ignore[index]
                    stf = paired(pb, best)
                    print(f"  parallel(best) vs fused(best)  {_ci(stf)}")
                    arms["parallel"]["vs_fused_control"] = stf  # type: ignore[index]
                arms["fused_control"] = {"best": {"name": name, "mean": float(best.mean()), "vs_incumbent": st}}

            cell_res["coarse"][c] = arms  # type: ignore[index]
        results[f"{args.dataset}/{tname}"] = cell_res

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
