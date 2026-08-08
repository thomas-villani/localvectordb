"""DIAGNOSTIC: which roll-up from child hits to parent unit is actually right?

THE QUESTION, AND THE TRAP IN ASKING IT. It is tempting to open with "everything
here takes the single best child". That is FALSE, and believing it would have
mis-specified this whole experiment. There are two different shipped roll-ups:

  * ``_aggregate_document_scores_with_method`` -- chunk -> document on the
    ordinary ``query()`` path. Already parameterised as ``document_scoring_method``
    over {"best", "average", "frequency_boost"}, and the default is
    **frequency_boost**, not best. So the shipped chunk->document aggregator is
    already coverage-weighted.
  * ``_reduce_to_best_per_key`` / ``_assemble_section_results`` -- max, no knob.
    These carry the ``fused`` level and every chunk -> section roll-up.

So the baseline is target-dependent, and the numpy harnesses (`rollup` in
`eval_section_bm25`, `_rollup_chunks` in `eval_fused_blend`) are pure max
EVERYWHERE. Where they measured a document target they were therefore running an
operator the shipped default does not use. Sizing that gap is the first job here;
it is the same class of instrument/product divergence as the hybrid-default
confound, and it has to be measured rather than assumed small.

WHY THE SHIPPED DEFAULT IS SUSPECT. ``frequency_boost`` is
``min(1.0, best * (1 + (log2(2 + effective_chunks) - 1) * bias))`` with
``bias=0.3``. The clamp is the problem: hybrid scores are min-max normalised into
[0, 1] within the query's own pool, so the best chunk is 1.0 by construction and
the multiplier exceeds 1.3 as soon as a document has two decent chunks. Every
document whose best chunk normalises above ~0.77 therefore lands on exactly 1.0
and ties with all the others there -- ties broken arbitrarily, at the very top of
the ranking. ``freq@0.3_noclip`` is swept beside it to size that specifically.

WHY MAX MIGHT ALSO BE WRONG. Max says relevance is a property of the best passage
and corroboration is worthless. That fits a needle-in-haystack query ("what is
the learning rate?") and not a distributed one ("how does this paper evaluate?").
If corpora differ in which kind of query they carry -- and §12.8 already showed
they differ in what BM25 rescues -- the aggregator is another corpus-dependent
knob rather than a constant.

WHY THE OBVIOUS SWEEP IS NOT ENOUGH. Anything that sums or counts children
rewards parents with MORE children, and long documents have more children. A
"coverage helps" result is indistinguishable from "long documents are more often
gold" unless the length effect is measured separately. `count` is included as a
pure-coverage control for exactly that reason: it ignores scores entirely, so
whatever it earns is the length prior and nothing else. Any aggregator that fails
to beat `count` has not demonstrated that it is using the scores.

THE POOL IS PART OF THE OPERATOR. An aggregator only ever sees the children that
made the retrieved pool, so "mean of the top 3" means something different at pool
40 than at pool 200. Pool width is therefore swept alongside, not fixed -- the
same mistake that briefly invalidated the fused blend sweep before `_truncate`
was added there.

THE VALIDATION THAT MAKES IT TRUSTWORTHY. `max` at the shipped pool must
reproduce `eval_section_bm25`'s published arm exactly, via `--verify`: this
harness captures ONE wide pool and truncates offline, which is only legitimate if
it lands bit-identically on what capturing at the narrow width produces. If it
does not, the run aborts rather than reporting a plausible number.

Zero embedding: every vector comes from the `hier_embed` disk cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    SEARCH_K,
    fuse,
    paired,
    recall,
    score,
    to_ranked,
    vector_top,
)

logger = logging.getLogger("aggregation")

# Capture once at the widest pool any arm needs, then truncate offline. Every
# swept pool must be <= this, and `--verify` proves the truncation is inert.
POOL_MAX = 400
POOLS: Tuple[int, ...] = (10, 20, 40, 100, 200, 400)

# `chunks.section_id` is single-valued and NULL for a chunk whose midpoint falls
# in no detected section, so a naive roll-up mints a literal None parent that
# occupies a rank slot and can never match a qrel. Dropping it is a bug fix, not
# a variant; `--keep-phantom` re-enables the old behaviour to size the effect.
PHANTOM = None


# --------------------------------------------------------------------------
# Aggregators: {child_id: score} + owner map -> {parent_id: score}
# --------------------------------------------------------------------------
def _grouped(scores: Dict[str, float], owner: Dict[str, Sequence[str]], keep_phantom: bool) -> Dict[str, List[float]]:
    """Child scores bucketed under each parent, best-first within a bucket."""
    buckets: Dict[str, List[float]] = {}
    for uid, s in scores.items():
        for parent in owner[uid]:
            if parent is PHANTOM and not keep_phantom:
                continue
            buckets.setdefault(parent, []).append(s)
    for vals in buckets.values():
        vals.sort(reverse=True)
    return buckets


def agg_max(vals: List[float]) -> float:
    return vals[0]


def agg_mean_top(m: int) -> Callable[[List[float]], float]:
    """Mean of the m best children *that exist*.

    Deliberately NOT padded to m with zeros. Padding would make the aggregator a
    function of how many chunks a parent happens to own, which is the length
    prior the `count` control exists to isolate -- it would smuggle that prior
    into every arm instead of keeping it in one.
    """

    def f(vals: List[float]) -> float:
        top = vals[:m]
        return sum(top) / len(top)

    return f


def agg_sum_top(m: int) -> Callable[[List[float]], float]:
    """Sum of the m best children. Length-sensitive BY CONSTRUCTION -- that is the
    point: it is `mean_top` plus the coverage prior, so the pair brackets how much
    of any gain is corroboration and how much is just owning more chunks."""

    def f(vals: List[float]) -> float:
        return sum(vals[:m])

    return f


def agg_lse(tau: float) -> Callable[[List[float]], float]:
    """Log-sum-exp: one knob that spans the whole family.

    ``tau -> inf`` is max, ``tau -> 0`` is mean + log(n)/tau. Sweeping it answers
    whether shipped max sits at a boundary (max really is best) or whether the
    optimum is interior (corroboration is worth something and max is leaving it).
    The max is subtracted first for numerical stability.
    """

    def f(vals: List[float]) -> float:
        top = vals[0]
        return top + math.log(sum(math.exp(tau * (v - top)) for v in vals)) / tau

    return f


def agg_average(vals: List[float]) -> float:
    """src's ``document_scoring_method="average"``: mean over every child in the pool."""
    return sum(vals) / len(vals)


def agg_freq_boost(bias: float, clip: bool) -> Callable[[List[float]], float]:
    """Replica of src's ``frequency_boost`` -- the shipped chunk->document default.

    Transcribed from ``_compute_document_scores`` rather than reimplemented, down
    to the ``best_score == 0`` guard, so a difference between this and the real
    path is a difference in the pool and not in the arithmetic.

    ``clip=False`` is the same operator without the ``min(1.0, ...)``. Ranking is
    invariant to a monotone transform, so the ONLY thing the clamp can do is
    create ties -- which makes the clip/no-clip pair a clean measurement of what
    the clamp costs, with nothing else changing.
    """

    def f(vals: List[float]) -> float:
        best = vals[0]
        weights = [1.0 for _ in vals] if best == 0 else [v / best for v in vals]
        effective = sum(weights)
        multiplier = 1.0 + (math.log2(2 + effective) - 1) * bias
        raw = best * multiplier
        return min(1.0, raw) if clip else raw

    return f


def agg_count(vals: List[float]) -> float:
    """Pure coverage control: how many children this parent has in the pool.

    Ignores every score, so it measures the length prior alone. An aggregator
    that cannot beat this has not shown it is using the scores at all.
    """
    return float(len(vals))


AGGREGATORS: Dict[str, Callable[[List[float]], float]] = {
    # -- shipped operators --
    "max": agg_max,  # = document_scoring_method="best"; _reduce_to_best_per_key
    "average": agg_average,  # = document_scoring_method="average"
    "freq@0.3": agg_freq_boost(0.3, clip=True),  # SHIPPED chunk->document default
    "freq@0.3_noclip": agg_freq_boost(0.3, clip=False),  # isolates the min(1.0) clamp
    "freq@0.1": agg_freq_boost(0.1, clip=False),
    "freq@0.6": agg_freq_boost(0.6, clip=False),
    # -- alternatives --
    "mean@2": agg_mean_top(2),
    "mean@3": agg_mean_top(3),
    "mean@5": agg_mean_top(5),
    "sum@2": agg_sum_top(2),
    "sum@3": agg_sum_top(3),
    "sum@5": agg_sum_top(5),
    "lse@20": agg_lse(20.0),
    "lse@10": agg_lse(10.0),
    "lse@5": agg_lse(5.0),
    # -- control --
    "count": agg_count,
}

# The shipped operator differs BY TARGET, so every paired CI below is quoted
# against the one a user actually gets for that target -- not against a single
# convenient reference. Section roll-up has no knob at all.
SHIPPED_BY_TARGET = {"doc": "freq@0.3", "section": "max"}


def aggregate(
    scores: Dict[str, float],
    owner: Optional[Dict[str, Sequence[str]]],
    how: str,
    keep_phantom: bool,
) -> Dict[str, float]:
    if owner is None:
        return scores
    fn = AGGREGATORS[how]
    return {parent: fn(vals) for parent, vals in _grouped(scores, owner, keep_phantom).items()}


# --------------------------------------------------------------------------
# Capture once at POOL_MAX; truncate per-leg before fusing
# --------------------------------------------------------------------------
def capture_wide(
    unit_ids: Sequence[str],
    unit_vecs: np.ndarray,
    fts: Optional[FTS],
    qvecs: np.ndarray,
    qtexts: Sequence[str],
    unit_docs: Optional[Sequence[str]] = None,
    q_scope: Optional[Sequence[Optional[str]]] = None,
    pool: int = POOL_MAX,
) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
    """``capture_arm`` at an arbitrary pool width. Same code shape, same order.

    Scoping slices the unit matrix per document rather than masking it, for the
    reason ``capture_arm`` documents: masking an object array is a Python loop
    over every unit, per query, per arm.
    """
    by_doc: Dict[str, np.ndarray] = {}
    if q_scope is not None and unit_docs is not None:
        order: Dict[str, List[int]] = {}
        for i, d in enumerate(unit_docs):
            order.setdefault(d, []).append(i)
        by_doc = {d: np.asarray(ix, dtype=np.int64) for d, ix in order.items()}

    out: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for qi, qtext in enumerate(qtexts):
        scope = q_scope[qi] if q_scope is not None else None
        if scope is None:
            sims = unit_vecs @ qvecs[qi]
            ids: Sequence[str] = unit_ids
        else:
            idx = by_doc.get(scope)
            if idx is None or not len(idx):
                out.append(({}, {}))
                continue
            sims = unit_vecs[idx] @ qvecs[qi]
            ids = [unit_ids[i] for i in idx]
        vec = vector_top(sims, ids, pool)
        if fts is None:
            kw: Dict[str, float] = {}
        else:
            kw = fts.search(qtext, pool * 2, scope)
            kw = dict(list(kw.items())[:pool])
        out.append((vec, kw))
    return out


def _truncate(scores: Dict[str, float], pool: int, negate: bool = False) -> Dict[str, float]:
    """Best ``pool`` entries. ``negate`` for raw BM25, where more-negative is better."""
    if len(scores) <= pool:
        return scores
    ranked = sorted(scores.items(), key=lambda kv: (kv[1] if negate else -kv[1]))
    return dict(ranked[:pool])


def blend_at_pool(
    captured: Sequence[Tuple[Dict[str, float], Dict[str, float]]],
    pool: int,
    vector_weight: float,
    *,
    use_keyword: bool,
) -> List[Dict[str, float]]:
    """Truncate each leg to ``pool``, then fuse -- the real path's order.

    Per-leg truncation happens BEFORE fusion because that is where it happens in
    ``_hybrid_search``: each leg is fetched at its own width and cut to it, and
    only then does min-max normalisation see the pool. Truncating after fusing
    would normalise against candidates the shipped code never had.

    THE VECTOR LEG IS RESCALED, AND ONLY THE VECTOR LEG. ``src`` does not hand
    raw cosine to the aggregator: ``_distance_to_similarity`` maps an IP index
    through ``(ip + 1) / 2``, so a chunk at cosine 0.55 reaches
    ``frequency_boost`` as 0.775 -- already at the ``min(1.0, ...)`` clamp once a
    document owns two chunks. Measuring the clamp on raw cosine would understate
    it badly. Hybrid needs no such correction and gets none: ``(x + 1) / 2`` is
    affine, and min-max normalisation is exactly invariant under an affine map,
    so the fused leg is already on src's scale.
    """
    out: List[Dict[str, float]] = []
    for vec, kw in captured:
        v = _truncate(vec, pool)
        if use_keyword and kw:
            out.append(fuse(v, _truncate(kw, pool, negate=True), vector_weight))
        else:
            out.append({key: max(0.0, min(1.0, (s + 1.0) / 2.0)) for key, s in v.items()})
    return out


def verify_inert(
    captured: Sequence[Tuple[Dict[str, float], Dict[str, float]]],
    unit_ids: Sequence[str],
    unit_vecs: np.ndarray,
    fts: Optional[FTS],
    qvecs: np.ndarray,
    qtexts: Sequence[str],
    unit_docs: Optional[Sequence[str]],
    q_scope: Optional[Sequence[Optional[str]]],
) -> None:
    """Capturing wide and truncating must equal capturing narrow. Abort if not.

    This is the whole licence for the capture-once design. Compared on the
    per-leg dicts rather than on nDCG, because two different pools can produce
    the same top-10 and still be different operators everywhere else.
    """
    narrow = capture_wide(unit_ids, unit_vecs, fts, qvecs, qtexts, unit_docs, q_scope, pool=SEARCH_K)
    bad: List[str] = []
    ties_only = 0
    for qi, (nv, nk) in enumerate(narrow):
        for leg, wide, ref, negate in (
            ("vector", _truncate(captured[qi][0], SEARCH_K), nv, False),
            ("keyword", _truncate(captured[qi][1], SEARCH_K, negate=True), nk, True),
        ):
            if not ref and not wide:
                continue
            if set(wide) == set(ref):
                if any(abs(wide[key] - ref[key]) > 1e-9 for key in ref):
                    bad.append(f"q{qi} {leg}: same members, different scores")
                continue
            # A boundary tie is not an infidelity: the differing members all hold
            # the SAME score as the reference pool's cut, so which of them
            # survives is arbitrary in the wide and the narrow capture alike.
            # Distinguish that from a pool that genuinely lost a better
            # candidate, which would be a real fidelity bug.
            sym = set(wide) ^ set(ref)
            edge = max(ref.values()) if negate else min(ref.values())
            if all(abs((wide[key] if key in wide else ref[key]) - edge) <= 1e-9 for key in sym):
                ties_only += 1
            else:
                bad.append(f"q{qi} {leg}: {len(sym)} differing members, NOT explained by a boundary tie")
    if bad:
        raise SystemExit(
            f"VERIFY FAILED: wide-capture+truncate differs from narrow capture on {len(bad)} legs. "
            "The offline pool sweep is not inert, so every number below would be measuring the "
            "harness rather than the aggregator.\n  " + "\n  ".join(bad[:10])
        )
    logger.info(
        "verify: wide-capture+truncate == narrow capture on all %d queries (%d boundary ties)",
        len(narrow),
        ties_only,
    )


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def sweep(
    captured: Sequence[Tuple[Dict[str, float], Dict[str, float]]],
    owner: Optional[Dict[str, Sequence[str]]],
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    vector_weight: float,
    use_keyword: bool,
    pools: Sequence[int],
    keep_phantom: bool,
) -> Dict[str, Any]:
    grid: Dict[str, Dict[str, float]] = {}
    per_query: Dict[str, List[float]] = {}
    fanout: Dict[str, float] = {}
    for pool in pools:
        blended = blend_at_pool(captured, pool, vector_weight, use_keyword=use_keyword)
        # Mean children-per-retrieved-parent: the quantity that decides whether a
        # multi-child aggregator can differ from max at all. At fanout ~1.0 every
        # arm below is the same operator and any spread is noise.
        sizes = [len(v) for s in blended for v in _grouped(s, owner, keep_phantom).values()] if owner else []
        fanout[f"pool={pool}"] = float(np.mean(sizes)) if sizes else 1.0
        for how in AGGREGATORS:
            rolled = [aggregate(s, owner, how, keep_phantom) for s in blended]
            ranked = to_ranked(rolled)
            per_q = score(ranked, qids, qrels)
            cell = f"pool={pool}|{how}"
            grid[cell] = {"ndcg@10": float(per_q.mean()), "recall@10": recall(ranked, qids, qrels)}
            per_query[cell] = [float(v) for v in per_q]
    return {"grid": grid, "per_query": per_query, "fanout": fanout}


# Fixed contender, named BEFORE seeing any number, so it carries an honest CI.
# The per-pool argmax below cannot: it is the best of ~15 correlated arms, and its
# interval is the interval of a maximum, which is biased high by construction.
PREREGISTERED = "sum@2"


def report(tname: str, swept: Dict[str, Any], pools: Sequence[int], shipped: str) -> None:
    grid, per_query, fanout = swept["grid"], swept["per_query"], swept["fanout"]
    print(f"\n=== aggregation sweep · target: {tname} ===  (nDCG@10)")
    print("  " + f"{'aggregator':<16}" + "".join(f"{f'pool={p}':>11}" for p in pools))
    for how in AGGREGATORS:
        row = "".join(f"{grid[f'pool={p}|{how}']['ndcg@10']:>11.4f}" for p in pools)
        mark = "  <- SHIPPED here" if how == shipped else ""
        print(f"  {how:<16}{row}{mark}")
    print("  " + f"{'mean fanout':<16}" + "".join(f"{fanout[f'pool={p}']:>11.2f}" for p in pools))

    # What the min(1.0, ...) clamp costs, with the operator otherwise identical.
    if "freq@0.3" in AGGREGATORS and "freq@0.3_noclip" in AGGREGATORS:
        print("\n--- cost of the frequency_boost min(1.0) clamp (clipped - unclipped) ---")
        for pool in pools:
            clipped = np.asarray(per_query[f"pool={pool}|freq@0.3"])
            unclipped = np.asarray(per_query[f"pool={pool}|freq@0.3_noclip"])
            stat = paired(clipped, unclipped)
            print(
                f"  pool={pool:<4} delta={stat['delta']:+.4f}  "
                f"95% CI [{stat['ci_lo']:+.4f}, {stat['ci_hi']:+.4f}]  p={stat['p']:.3f}"
            )

    print(f"\n--- paired vs SHIPPED ({shipped}), at each pool (10,000 resamples) ---")
    print(f"    '{PREREGISTERED}' was named before the run; 'argmax' is post-hoc over {len(AGGREGATORS)-1} arms")
    print("    and its CI is the CI of a maximum -- read it as a direction, not as a p-value.")
    for pool in pools:
        base = np.asarray(per_query[f"pool={pool}|{shipped}"])
        best, best_stat = None, None
        for how in AGGREGATORS:
            if how == shipped:
                continue
            stat = paired(np.asarray(per_query[f"pool={pool}|{how}"]), base)
            if best_stat is None or stat["delta"] > best_stat["delta"]:
                best, best_stat = how, stat
        assert best is not None and best_stat is not None
        pre = paired(np.asarray(per_query[f"pool={pool}|{PREREGISTERED}"]), base)
        # A CI entirely below zero is a significant LOSS, not a null result --
        # labelling it "indistinguishable" would hide the one direction that
        # should stop a change from shipping.
        if pre["ci_lo"] > 0:
            verdict = "SIGNIFICANT WIN"
        elif pre["ci_hi"] < 0:
            verdict = "SIGNIFICANT LOSS"
        else:
            verdict = "indistinguishable"
        print(
            f"  pool={pool:<4} {PREREGISTERED}: {pre['delta']:+.4f} "
            f"[{pre['ci_lo']:+.4f}, {pre['ci_hi']:+.4f}] p={pre['p']:.3f} {verdict:<18}"
            f" | argmax={best}: {best_stat['delta']:+.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["qasper", "maud", "mldr", "nq"], default="qasper")
    ap.add_argument("--coarse", choices=["sections", "documents"], default="sections")
    ap.add_argument("--model-key", default="egemma")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--vector-weight", type=float, default=0.5)
    ap.add_argument(
        "--search-type",
        choices=["vector", "hybrid", "both"],
        default="both",
        help="Aggregation runs AFTER fusion, so it can interact with it. 'both' is the honest default.",
    )
    ap.add_argument("--rollup", choices=["midpoint", "overlap"], default="midpoint")
    ap.add_argument(
        "--keep-phantom",
        action="store_true",
        help="Keep the None parent minted by chunks with no section (sizes the bug rather than fixing it).",
    )
    ap.add_argument("--pools", type=int, nargs="+", default=list(POOLS))
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--allow-embed", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if max(args.pools) > POOL_MAX:
        raise SystemExit(f"--pools may not exceed POOL_MAX={POOL_MAX}; widen the constant and re-capture.")

    from benchmarks.eval_dual import MODEL_POOL, PrefixedEncoder, _load_experiments_env, load_units

    _load_experiments_env()

    if args.dataset == "maud":
        from benchmarks.maud_data import detect_contract_sections, load_maud

        bench = load_maud(max_contracts=args.max_papers)
        units = load_units(bench, None, detect_contract_sections)
    elif args.dataset == "mldr":
        from benchmarks.mldr_data import load_mldr

        bench = load_mldr(split="dev", max_queries=args.max_papers)
        units = load_units(bench, None)
    elif args.dataset == "nq":
        from benchmarks.nq_data import load_nq

        bench = load_nq(max_queries=args.max_papers)
        units = load_units(bench, None)
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev", max_papers=args.max_papers)
        units = load_units(bench, None)

    spec = MODEL_POOL[args.model_key]
    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)

    if not args.allow_embed:
        miss = doc_enc.count_misses(units.chunk_texts)[1] + qry_enc.count_misses(units.query_texts)[1]
        if miss:
            raise SystemExit(
                f"{miss} vectors are not cached for {spec.model}. This harness is zero-embedding "
                "by default; pass --allow-embed to encode them."
            )

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    cv = unit(doc_enc.encode(units.chunk_texts, normalize=False))
    qv = unit(qry_enc.encode(units.query_texts, normalize=False))
    qids, qtexts = list(units.query_ids), list(units.query_texts)

    chunk_uid = [f"c{i}" for i in range(len(units.chunk_texts))]
    chunk_docs = list(units.chunk_doc)
    fts_chunks = FTS(chunk_uid, units.chunk_texts, chunk_docs)

    if args.rollup == "overlap":
        chunk_to_sec = {u: list(v) for u, v in zip(chunk_uid, units.chunk_sections_all, strict=True)}
    else:
        chunk_to_sec = {u: [v] for u, v in zip(chunk_uid, units.chunk_section, strict=True)}
    chunk_to_doc: Dict[str, Sequence[str]] = {u: [v] for u, v in zip(chunk_uid, chunk_docs, strict=True)}

    n_phantom = sum(1 for parents in chunk_to_sec.values() if any(p is PHANTOM for p in parents))
    logger.info(
        "%d/%d chunks own no section (%.1f%%) -- their roll-up parent is %s",
        n_phantom,
        len(chunk_uid),
        100 * n_phantom / max(len(chunk_uid), 1),
        "KEPT as a phantom rank slot" if args.keep_phantom else "dropped",
    )

    q_scope: Optional[List[Optional[str]]] = None
    if args.dataset == "maud":
        q_scope = [str(q).split("||", 1)[0] for q in qids]

    logger.info("%d chunks, %d queries; capturing at pool=%d", len(cv), len(qids), POOL_MAX)

    targets: List[Tuple[str, Dict[str, Dict[str, int]]]] = [
        ("section", bench.section_qrels),
        ("doc", bench.doc_qrels),
    ]
    if args.dataset == "maud":
        # Every query is scoped to its own contract, so the doc target has one
        # candidate and is trivially perfect -- a number that means nothing.
        targets = [("section", bench.section_qrels)]

    legs = [("vector", 1.0, False), ("hybrid", args.vector_weight, True)]
    if args.search_type != "both":
        legs = [leg for leg in legs if leg[0] == args.search_type]

    config = {key: (str(val) if isinstance(val, Path) else val) for key, val in vars(args).items()}
    results: Dict[str, Any] = {"config": config, "shipped_by_target": SHIPPED_BY_TARGET, "arms": {}}
    for leg_name, vector_weight, use_keyword in legs:
        captured = capture_wide(
            chunk_uid, cv, fts_chunks if use_keyword else None, qv, qtexts, chunk_docs, q_scope, pool=POOL_MAX
        )
        if args.verify:
            verify_inert(
                captured,
                chunk_uid,
                cv,
                fts_chunks if use_keyword else None,
                qv,
                qtexts,
                chunk_docs,
                q_scope,
            )
        for tname, qrels in targets:
            if not qrels:
                continue
            owner = chunk_to_sec if tname == "section" else chunk_to_doc
            swept = sweep(captured, owner, qids, qrels, vector_weight, use_keyword, args.pools, args.keep_phantom)
            report(f"{tname} · {leg_name}", swept, args.pools, SHIPPED_BY_TARGET[tname])
            results["arms"][f"{tname}|{leg_name}"] = swept

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
