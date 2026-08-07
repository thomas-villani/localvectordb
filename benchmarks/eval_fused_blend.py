"""Should ``search_level="fused"`` get a keyword leg, and in what topology?

``fused`` was the last level in ``_VECTOR_ONLY_LEVELS``, which this file emptied.
Fix A (``documents_fts``) and Fix B (``sections_fts``) gave every other level a
keyword leg by *wiring* an index that already existed; ``fused`` could not be
fixed that way. It blends two bounded-similarity legs through
``_two_leg_minmax_fuse``, so adding BM25 is a change to the **blend** -- a
modelling decision with a free parameter -- rather than a wiring change. S12.8
predicted +0.057/+0.058/+0.123 for qasper/MAUD/MLDR; this file measured
+0.0573/+0.0582/+0.1233 and Fix C shipped the two-stage form (S17).

TWO TOPOLOGIES, MEASURED AGAINST EACH OTHER
-------------------------------------------
``two_stage``
    Each leg runs hybrid *within its own granularity* first -- chunk-vector with
    chunk-BM25, section-vector with section-BM25, both through
    ``_relative_score_fusion(vector_weight)`` -- and the two hybrid legs are then
    fused by ``_two_leg_minmax_fuse(section_weight)``. Compositional: it reuses
    the two fusions already shipped and gated, and ``section_weight`` keeps
    meaning exactly what it means today.

``single_stage``
    All four legs are normalized independently at the target unit and blended in
    one step, ``(1-sw)*[vw*cv + (1-vw)*cb] + sw*[vw*sv + (1-vw)*sb]``. Same two
    free parameters, but normalization happens once, at the target, instead of
    twice. This is NOT a strict superset of ``two_stage`` -- the two-stage form normalizes
    within the chunk pool and again after roll-up -- so the pair is a genuine
    A/B, and ``single_stage`` mostly serves as an upper bound on what the extra
    normalization step in ``two_stage`` costs.

WHY THE LEGS ARE CAPTURED ONCE AND BLENDED OFFLINE
--------------------------------------------------
Both blends are pure functions of four ``{key: score}`` dicts, and neither
touches retrieval. So the whole 2-D grid is free: capture the four legs per
query once, at exactly the pool ``_fused_search`` uses, then blend in Python.
``eval_section_weight.py`` re-runs ``db.query()`` once per weight, which is
affordable for a 10-point 1-D sweep and is not for a 63-point 2-D one.

The legs come from ``src/``'s own private methods rather than a reimplementation,
and the arithmetic comes from ``src/``'s own ``_relative_score_fusion`` and
``_two_leg_minmax_fuse``. What is reimplemented here is only the *plumbing*
between them -- roll-up and pool truncation -- and ``--verify`` exists because
that plumbing is exactly where a divergence would hide.

--verify IS THE LOAD-BEARING CHECK
----------------------------------
At ``search_type="vector"`` the keyword legs are empty, so the offline blend must
reproduce ``db.query(search_level="fused", section_weight=w, search_type="vector")``
**exactly** -- same ranked ids, same order, ties included. Run it before reading
any number off this file. It is checked at three weights including 0.0 and 1.0,
because a roll-up or tie-break bug that cancels at w=0.65 need not cancel at the
ends.

READ THE ARGMAX AS AN UPPER BOUND, as in ``eval_section_weight.py``: the best
(vw, sw) is picked on the same queries it is scored on. It bounds what a tuned
blend could buy and is not itself a shippable default. Nothing here licenses
changing ``section_weight=0.65`` or ``vector_weight=0.5``.

Zero API spend against cached DBs; no rebuilds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR, EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.eval_hier_gate import (  # noqa: E402
    K,
    Leg,
    _section_qrel_id,
    build_db,
    qasper_leg,
    superdocs_leg,
)
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from localvectordb.database._search import _relative_score_fusion, _two_leg_minmax_fuse  # noqa: E402

logger = logging.getLogger("eval_fused_blend")

# The blend grid. ``vector_weight`` is the weight on the VECTOR side within each
# leg (1.0 = no BM25, which is today's fused up to the pool-widening noted below);
# ``section_weight`` is the weight on the SECTION leg across granularities, and
# 0.65 is what ships.
VECTOR_WEIGHTS: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0)
SECTION_WEIGHTS: Tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
SHIPPED_SECTION_WEIGHT = 0.65
SHIPPED_VECTOR_WEIGHT = 0.5
STRATEGIES = ("rawspan", "centroid")
TOPOLOGIES = ("two_stage", "single_stage")
TARGETS = ("documents", "sections")

# ``_fused_search`` uses ``pool_k = k * 2`` for both legs, and ``_hybrid_search``
# over-fetches the keyword leg by 2x before truncating to the pool. Mirrored
# rather than re-chosen: a different pool changes min-max normalization, and the
# point of this file is to measure the blend, not the pool.
POOL_MULTIPLIER = 2
KEYWORD_OVERFETCH = 2


# ----------------------------------------------------------------------------
# Leg capture
# ----------------------------------------------------------------------------


def _chunk_to_section_map(db) -> Dict[str, str]:
    """``{doc}:{chunk_index}`` -> ``{doc}:section:{i}``, the SINGLE-VALUED mapping.

    This is ``chunks.section_id`` -- midpoint attribution, the same join
    ``_assemble_section_results`` walks per chunk. Precomputed once because it is
    a property of the index, not of the query, and because doing it per chunk per
    query is what makes ``_assemble_section_results`` the slow part of the fused
    path. 40.1% of sections own no chunk under this mapping (S14.2); that is a
    property of the system under test and is deliberately preserved here.
    """
    out: Dict[str, str] = {}
    with db.connection_pool.get_connection() as conn:
        rows = conn.execute("""
            SELECT c.document_id AS doc_id, c.chunk_index AS chunk_index,
                   s.section_index AS section_index
            FROM chunks c JOIN sections s ON c.section_id = s.id
            """).fetchall()
    for row in rows:
        out[f"{row['doc_id']}:{row['chunk_index']}"] = f"{row['doc_id']}:section:{row['section_index']}"
    return out


def leg_pool_size(pool_k: int, leg_pool: str) -> int:
    """How wide each leg retrieves before the two granularities are fused.

    This is a real fork, not a tuning knob, and it is the one place where the
    swept operator could fail to match a shipped one. ``_fused_search`` fetches
    ``pool_k = k*2`` per leg; ``_hybrid_search`` fetches
    ``search_k = max(k, min(k*4, 100))`` -- and the leg is asked for ``pool_k``
    results, so that is ``max(20, min(80, 100)) = 80`` at k=10 -- before *it*
    fuses. Pool size sets the min-max denominators, so it is part of the fusion
    rule (the same point ``eval_section_bm25`` makes about its own ``SEARCH_K``).

    ``fusion``
        Both legs retrieve at ``pool_k``. Symmetric: the chunk and section legs
        are drawn from equally wide nets, so ``section_weight`` weighs like
        against like.
    ``native``
        Each leg retrieves at the width it would use as a standalone level, then
        the *fused* leg is truncated to ``pool_k`` before roll-up. The chunk leg
        keeps the recall it has at ``search_level="chunks"``, at the cost of the
        two legs no longer being drawn from comparable pools.

    Truncating after per-leg fusion (rather than at retrieval) is what keeps the
    vector-only column identical under both settings, so ``--verify`` is valid
    either way and the two are comparable at every other cell.
    """
    return pool_k if leg_pool == "fusion" else max(pool_k, min(pool_k * 4, 100))


_QUERY_VEC_CACHE: Dict[str, np.ndarray] = {}


def _query_vector(db, query_text: str) -> np.ndarray:
    """Embed a query once per process rather than once per (strategy, target, pool).

    A sweep visits the same query text under both section strategies, both targets
    and both leg pools -- eight times -- and the vector is identical every time
    because the encoder is fixed for the whole run. Keyed on text alone for that
    reason; if this file ever sweeps ACROSS encoders, this cache has to key on the
    model too.
    """
    cached = _QUERY_VEC_CACHE.get(query_text)
    if cached is None:
        cached = np.array(db.embedding_provider.embed_sync([query_text], task="query")[0]).reshape(1, -1)
        _QUERY_VEC_CACHE[query_text] = cached
    # A copy, because FAISS helpers are free to normalize their input in place and
    # the uncached path always handed them a fresh array.
    return cached.copy()


def capture_legs(db, query_text: str, pool_k: int, leg_pool: str = "fusion") -> Dict[str, Dict[str, float]]:
    """The four legs for one query, at the pool ``_fused_search`` would use.

    Returns raw BM25 for the keyword legs (negative, more-negative-better), never
    ``_fts_rank_to_similarity`` output -- ``_relative_score_fusion`` requires the
    raw rank, and normalizing the bounded transform would normalize float noise.
    """
    query_embedding = _query_vector(db, query_text)
    fetch_k = leg_pool_size(pool_k, leg_pool)

    # Chunk vector leg -- the exact call `_fused_search` makes at fetch_k == pool_k.
    chunk_vec_hits = db._vector_search(
        query_text,
        "chunks",
        fetch_k,
        0.0,
        None,
        0,
        None,
        "frequency_boost",
        None,
        "chunks",
        False,
        query_embedding=query_embedding,
    )
    chunk_vec = {r.id: r.score for r in chunk_vec_hits}

    # Chunk keyword leg -- `_hybrid_search` over-fetches then truncates to the pool.
    kw_hits, kw_ranks = db._keyword_chunk_hits(query_text, fetch_k * KEYWORD_OVERFETCH, 0.0, None)
    chunk_bm25 = {r.id: kw_ranks[r.id] for r in kw_hits[:fetch_k] if r.id in kw_ranks}

    section_vec_hits = db._section_level_search(query_embedding, fetch_k, 0.0, None)
    section_vec = {r.id: r.score for r in section_vec_hits}
    section_bm25 = db._keyword_section_hits(query_text, fetch_k, None)

    # Chunk -> document is carried on the result, so no second lookup is needed.
    chunk_doc = {r.id: (r.document_id or r.id) for r in chunk_vec_hits}
    for r in kw_hits[:fetch_k]:
        chunk_doc.setdefault(r.id, r.document_id or r.id)

    return {
        "chunk_vec": chunk_vec,
        "chunk_bm25": chunk_bm25,
        "section_vec": section_vec,
        "section_bm25": section_bm25,
        "chunk_doc": chunk_doc,
    }


# ----------------------------------------------------------------------------
# Offline blending
# ----------------------------------------------------------------------------


def _rollup_chunks(
    chunk_scores: Dict[str, float],
    chunk_doc: Dict[str, str],
    chunk_section: Dict[str, str],
    target: str,
    pool_k: int,
) -> Dict[str, float]:
    """Max chunk score per target unit, mirroring src/'s two roll-up rules.

    ``documents`` mirrors ``_reduce_to_best_per_key``: first-seen insertion order,
    no truncation. ``sections`` mirrors ``_assemble_section_results``: chunks
    whose section is unknown are dropped (that is the 40.1%), and the result is
    sorted by ``(-score, id)`` and truncated to the pool BEFORE fusion. Both
    orderings are load-bearing -- ``_two_leg_minmax_fuse`` breaks ties toward
    whatever the primary leg saw first.
    """
    best: Dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        if target == "documents":
            key = chunk_doc.get(chunk_id)
        else:
            key = chunk_section.get(chunk_id)
        if key is None:
            continue
        prev = best.get(key)
        if prev is None or score > prev:
            best[key] = score
    if target == "sections":
        ordered = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_k]
        return dict(ordered)
    return best


def _sections_to_target(section_scores: Dict[str, float], target: str) -> Dict[str, float]:
    """Section leg at the target unit (identity for sections; max-per-doc for documents)."""
    if target == "sections":
        return section_scores
    best: Dict[str, float] = {}
    for key, score in section_scores.items():
        doc_id = key.rpartition(":section:")[0] or key
        prev = best.get(doc_id)
        if prev is None or score > prev:
            best[doc_id] = score
    return best


def _truncate(scores: Dict[str, float], pool_k: int) -> Dict[str, float]:
    """Best ``pool_k`` keys, mirroring what a leg returns as a standalone level.

    ``_hybrid_search`` fuses its two legs and then truncates the result to its own
    ``search_k`` -- the fused pool is not allowed to stay twice as wide as either
    input. A fused chunk leg has to do the same or ``section_weight`` would be
    weighing a 40-candidate chunk pool against a 20-candidate section one.
    Ties break toward the key seen first, which is the vector leg (see
    ``_relative_score_fusion``'s insertion order).
    """
    if len(scores) <= pool_k:
        return scores
    return dict(sorted(scores.items(), key=lambda kv: -kv[1])[:pool_k])


def blend(
    legs: Dict[str, Dict[str, float]],
    chunk_section: Dict[str, str],
    *,
    topology: str,
    target: str,
    vector_weight: float,
    section_weight: float,
    pool_k: int,
    use_keyword: bool,
) -> List[str]:
    """One grid cell: return the fused ranking of target keys, best first."""
    chunk_bm25 = legs["chunk_bm25"] if use_keyword else {}
    section_bm25 = legs["section_bm25"] if use_keyword else {}

    if topology == "two_stage":
        chunk_leg = _truncate(
            _relative_score_fusion(legs["chunk_vec"], chunk_bm25, vector_weight) if chunk_bm25 else legs["chunk_vec"],
            pool_k,
        )
        section_leg = _truncate(
            (
                _relative_score_fusion(legs["section_vec"], section_bm25, vector_weight)
                if section_bm25
                else legs["section_vec"]
            ),
            pool_k,
        )
        chunk_at_target = _rollup_chunks(chunk_leg, legs["chunk_doc"], chunk_section, target, pool_k)
        section_at_target = _sections_to_target(section_leg, target)
        fused = _two_leg_minmax_fuse(chunk_at_target, section_at_target, section_weight)
    elif topology == "single_stage":
        # Every leg rolled up to the target first, then one blend. The chunk-BM25
        # leg rolls up through the same rule as chunk-vector so the two remain
        # comparable; without that the keyword leg would be scored at a different
        # granularity from the vector leg it is weighted against.
        # Each raw leg is capped at pool_k first, for the reason in _truncate: under
        # --leg-pool native the legs arrive wider than the fusion pool, and an
        # uncapped leg would out-weigh a capped one regardless of section_weight.
        cv = _rollup_chunks(_truncate(legs["chunk_vec"], pool_k), legs["chunk_doc"], chunk_section, target, pool_k)
        sv = _sections_to_target(_truncate(legs["section_vec"], pool_k), target)
        # BM25 is negative-is-better; negate before roll-up so "max" means "best".
        cb = _rollup_chunks(
            _truncate({key: -rank for key, rank in chunk_bm25.items()}, pool_k),
            legs["chunk_doc"],
            chunk_section,
            target,
            pool_k,
        )
        sb = _sections_to_target(_truncate({key: -rank for key, rank in section_bm25.items()}, pool_k), target)
        chunk_side = _two_leg_minmax_fuse(cv, cb, 1.0 - vector_weight) if cb else cv
        section_side = _two_leg_minmax_fuse(sv, sb, 1.0 - vector_weight) if sb else sv
        fused = _two_leg_minmax_fuse(chunk_side, section_side, section_weight)
    else:
        raise ValueError(f"unknown topology {topology!r}")

    return [key for key, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def _scored_queries(bench, target: str) -> List[Tuple[str, str, Dict[str, float]]]:
    """(qid, text, rel) for queries with at least one positive judgement at ``target``."""
    qrels = bench.doc_qrels if target == "documents" else getattr(bench, "section_qrels", {}) or {}
    out = []
    for qid, text in bench.queries.items():
        rel = qrels.get(qid, {})
        if any(r > 0 for r in rel.values()):
            out.append((qid, text, rel))
    return out


def _to_qrel_ids(ranked: Sequence[str], target: str) -> List[str]:
    return list(ranked) if target == "documents" else [_section_qrel_id(r) for r in ranked]


def verify_inert(db, bench, chunk_section: Dict[str, str], pool_k: int, leg_pool: str, n: int = 25) -> None:
    """The offline blend must reproduce src/'s fused ranking exactly on vector.

    Checked at 0.0, 0.65 and 1.0: a roll-up or tie-break bug that happens to
    cancel at the shipped weight need not cancel at the ends, where one leg
    carries the whole ranking.
    """
    queries = _scored_queries(bench, "documents")[:n]
    if not queries:
        raise RuntimeError("no scored queries to verify against")
    for section_weight in (0.0, SHIPPED_SECTION_WEIGHT, 1.0):
        for qid, text, _rel in queries:
            expected = [
                h.id
                for h in db.query(
                    text,
                    search_level="fused",
                    return_type="documents",
                    k=K,
                    search_type="vector",
                    section_weight=section_weight,
                )
            ]
            legs = capture_legs(db, text, pool_k, leg_pool)
            got = blend(
                legs,
                chunk_section,
                topology="two_stage",
                target="documents",
                vector_weight=1.0,
                section_weight=section_weight,
                pool_k=pool_k,
                use_keyword=False,
            )[:K]
            if got != expected:
                raise AssertionError(
                    f"offline blend diverges from src/ at section_weight={section_weight} "
                    f"on qid={qid}:\n  src/    {expected}\n  offline {got}"
                )
    logger.info("verify: offline blend reproduces src/'s fused ranking on %d queries x 3 weights", len(queries))


def sweep_target(
    db,
    bench,
    chunk_section: Dict[str, str],
    target: str,
    pool_k: int,
    collect_per_query: bool,
    leg_pool: str,
) -> Dict[str, Any]:
    """Capture every query's legs once, then score the whole grid off them."""
    queries = _scored_queries(bench, target)
    if not queries:
        return {}
    logger.info("  [%s] capturing legs for %d queries", target, len(queries))
    captured = [(qid, rel, capture_legs(db, text, pool_k, leg_pool)) for qid, text, rel in queries]

    out: Dict[str, Any] = {"n_queries": len(queries), "grid": {}}
    if collect_per_query:
        out["per_query"] = {}

    for topology in TOPOLOGIES:
        for vector_weight in VECTOR_WEIGHTS:
            for section_weight in SECTION_WEIGHTS:
                # vw=1.0 means "no BM25", so the keyword legs are skipped rather
                # than weighted to zero. Weighting them to zero still widens the
                # pool with keyword-only keys at score 0.0, which is a different
                # (and not obviously worse) operator -- but it is not the
                # vector-only baseline this column is meant to be.
                use_keyword = vector_weight < 1.0
                nd, r10 = [], []
                per_q: Dict[str, float] = {}
                for qid, rel, legs in captured:
                    ranked = _to_qrel_ids(
                        blend(
                            legs,
                            chunk_section,
                            topology=topology,
                            target=target,
                            vector_weight=vector_weight,
                            section_weight=section_weight,
                            pool_k=pool_k,
                            use_keyword=use_keyword,
                        )[:K],
                        target,
                    )
                    score = ndcg_at_k(ranked, rel, K)
                    nd.append(score)
                    r10.append(recall_at_k(ranked, rel, K))
                    per_q[qid] = score
                cell = f"{topology}|vw={vector_weight:.2f}|sw={section_weight:.2f}"
                out["grid"][cell] = {
                    "ndcg@10": sum(nd) / len(nd),
                    "recall@10": sum(r10) / len(r10),
                }
                if collect_per_query:
                    out["per_query"][cell] = per_q
    return out


def run_leg(leg: Leg, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    missing = [s for s in STRATEGIES if not _is_cached(leg, args.embedding_model, s)]
    if missing and not args.allow_build:
        logger.warning(
            "[%s] skipping: no cached DB for %s. Re-run with --allow-build to spend the build.",
            leg.name,
            ", ".join(missing),
        )
        return None

    bench = leg.load()
    pool_k = K * POOL_MULTIPLIER
    payload: Dict[str, Any] = {"n_docs": len(bench.corpus), "strategies": {}}

    for strategy in STRATEGIES:
        db = build_db(
            bench,
            leg=leg,
            provider=args.embedding_provider,
            model=args.embedding_model,
            strategy=strategy,
            rebuild=False,
        )
        try:
            if db.count() != len(bench.corpus):
                raise RuntimeError(
                    f"[{leg.name}] DB[{strategy}] holds {db.count()} documents, benchmark has "
                    f"{len(bench.corpus)}. Stale cached index."
                )
            chunk_section = _chunk_to_section_map(db)
            logger.info("[%s/%s] %d chunks carry a section", leg.name, strategy, len(chunk_section))
            if args.verify:
                verify_inert(db, bench, chunk_section, pool_k, args.leg_pool)
            per_strategy: Dict[str, Any] = {}
            for target in args.targets:
                result = sweep_target(db, bench, chunk_section, target, pool_k, args.per_query, args.leg_pool)
                if result:
                    per_strategy[target] = result
            payload["strategies"][strategy] = per_strategy
        finally:
            db.close()
    return payload


def _is_cached(leg: Leg, model: str, strategy: str) -> bool:
    key = f"hiergate__{leg.cache_key}__{model.replace('/', '_').replace(':', '-')}__{strategy}"
    return (DATA_DIR / "db" / f"{key}.complete").exists()


def _summarise(name: str, payload: Dict[str, Any]) -> None:
    for strategy, targets in payload["strategies"].items():
        for target, result in targets.items():
            grid = result["grid"]
            print(f"\n{name} · {strategy} · target={target}  ({result['n_queries']} queries)")
            for topology in TOPOLOGIES:
                baseline_cell = f"{topology}|vw=1.00|sw={SHIPPED_SECTION_WEIGHT:.2f}"
                baseline = grid.get(baseline_cell, {}).get("ndcg@10")
                print(f"  {topology}:  vector-only @ sw=0.65 = {baseline:.4f}" if baseline else f"  {topology}:")
                corner = "vw \\ sw"
                header = "    " + f"{corner:<8}" + "".join(f"{sw:>9.2f}" for sw in SECTION_WEIGHTS)
                print(header)
                for vw in VECTOR_WEIGHTS:
                    row = f"    {vw:<8.2f}"
                    for sw in SECTION_WEIGHTS:
                        row += f"{grid[f'{topology}|vw={vw:.2f}|sw={sw:.2f}']['ndcg@10']:>9.4f}"
                    print(row)
                best_cell, best = max(
                    ((c, v) for c, v in grid.items() if c.startswith(topology)),
                    key=lambda kv: kv[1]["ndcg@10"],
                )
                shipped = grid[f"{topology}|vw={SHIPPED_VECTOR_WEIGHT:.2f}|sw={SHIPPED_SECTION_WEIGHT:.2f}"]["ndcg@10"]
                print(
                    f"    argmax {best_cell.split('|', 1)[1]} -> {best['ndcg@10']:.4f}"
                    + (f" ({best['ndcg@10'] - baseline:+.4f} vs vector-only)" if baseline else "")
                    + f" | shipped weights -> {shipped:.4f}"
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--legs", nargs="+", default=["qasper", "superdocs"], choices=["qasper", "superdocs"])
    p.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--allow-build", action="store_true", help="permit building an uncached leg (slow)")
    p.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="assert the offline blend reproduces src/'s fused ranking on vector (default: on)",
    )
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument(
        "--per-query",
        action="store_true",
        help="retain per-query nDCG so paired_bootstrap.py can attach CIs to the argmax",
    )
    p.add_argument(
        "--leg-pool",
        choices=("fusion", "native"),
        default="fusion",
        help="How wide each leg retrieves before the two granularities are fused. 'fusion' gives both "
        "legs _fused_search's pool_k=k*2 (symmetric); 'native' gives each leg the width it would use "
        "as a standalone level -- _hybrid_search called with k=pool_k uses search_k=80 at k=10 -- and "
        "truncates the "
        "fused leg to pool_k afterwards. Pool size sets the min-max denominators, so this is part of "
        "the fusion rule, not an efficiency knob -- measure it, do not assume it.",
    )
    p.add_argument("--out", default=str(_ROOT / "benchmarks" / "results" / "fused_blend_sweep.json"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    # Load the encoder BEFORE any corpus -- see eval_section_weight.main for why
    # the order is load-bearing (benchmarks/datasets.py shadows HuggingFace's).
    from benchmarks.eval_retrieval import preflight_embedding_model

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    builders = {
        "qasper": lambda: qasper_leg(None),
        "superdocs": lambda: superdocs_leg(3, 32, 200, 0),
    }
    payloads: Dict[str, Any] = {}
    for name in args.legs:
        payload = run_leg(builders[name](), args)
        if payload is not None:
            payloads[name] = payload

    for name, payload in payloads.items():
        _summarise(name, payload)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(),
                "model": args.embedding_model,
                "vector_weights": list(VECTOR_WEIGHTS),
                "section_weights": list(SECTION_WEIGHTS),
                "topologies": list(TOPOLOGIES),
                "shipped_section_weight": SHIPPED_SECTION_WEIGHT,
                "shipped_vector_weight": SHIPPED_VECTOR_WEIGHT,
                "leg_pool": args.leg_pool,
                "legs": payloads,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
