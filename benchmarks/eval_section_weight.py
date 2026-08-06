"""DIAGNOSTIC: is the shipped ``section_weight=0.65`` defensible as a default?

``_search.py`` ships ``section_weight: float = 0.65`` -- nearly two-thirds of the
fused score on the section leg, documented as "tuned on real long docs". Every
density rung in §6.32 had ``fused`` LOSING to plain chunk retrieval, and the
suspected cause is that weight: the harness result the default descends from
(``eval_hierarchical._fuse_best``) searched the best weight **per corpus**, so it
could not lose by construction. That is not the same operator as a fixed 0.65.

WHY THIS IS FREE. ``section_weight`` is a post-hoc scalar in
``_two_leg_minmax_fuse`` -- it re-weights two already-retrieved candidate pools
and never touches ingest. So the whole curve can be swept over **cached** DBs
with no rebuild and no embedding spend beyond re-encoding each query string.

WHAT w=0.0 MEASURES, AND WHY IT IS NOT THE ``chunks`` ARM. Fusion runs its chunk
leg with ``return_type="chunks"`` and rolls up to documents with
``_reduce_to_best_per_key`` (max chunk score per document). The ``chunks`` arm of
the gate uses ``search_level="chunks", return_type="documents"``, which applies
``document_scoring_method="frequency_boost"``. So ``fused @ w=0.0`` and
``chunks`` are two different document-scoring rules over the same chunk hits, and
their difference is a **confound in every fused-vs-chunks comparison made so
far** -- it is attributed here rather than assumed to be zero. Read the gap
between the ``chunks`` row and the ``w=0.00`` column as that confound, and the
shape of the curve from ``w=0.00`` rightward as the actual weight effect.

READ THE ARGMAX AS AN UPPER BOUND. The best weight is picked here *on the same
queries it is scored on*, so it is an oracle, exactly like the harness fusion it
is meant to critique. It bounds what any tuned weight could buy; it is not itself
a shippable number. Nothing here licenses changing 0.65 -- that needs its own
gate run on both legs.

Zero API spend (local sentence-transformers), no rebuilds by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR, EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.eval_hier_gate import Leg, build_db, qasper_leg, superdocs_leg  # noqa: E402
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("eval_section_weight")

K = 10
SHIPPED_WEIGHT = 0.65
WEIGHTS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
STRATEGIES = ("rawspan", "centroid")

# The §6.32 density leg, at its top rung. Included because it is the regime most
# favourable to the section leg (12.5% gold density, sections at their strongest)
# -- if a tuned weight cannot make fusion pay there, it will not pay anywhere.
DENSITY_SECTIONS = 3
DENSITY_PASSAGES = 32
DENSITY_QUERIES = 100
DENSITY_MIN_QUERY_GOLD = 4
DENSITY_SEED = 0


def density_leg(gold: int) -> Leg:
    """One rung of the §6.32 density ladder (all three are already built).

    Sweeping the ladder rather than only its top rung is what separates the two
    ways fusion can lose. The number of gold passages per document is exactly
    what ``frequency_boost`` rewards and ``_reduce_to_best_per_key`` throws away,
    so if the ``chunks``-to-``w=0`` gap is the document roll-up rule, it must grow
    with ``gold``; if it is a fixed property of the fusion path, it will not.
    """
    from benchmarks import beir_data
    from benchmarks.superdocs import build_synthetic_benchmark

    def _load():
        return build_synthetic_benchmark(
            beir_data.load("fiqa"),
            sections_per_doc=DENSITY_SECTIONS,
            passages_per_section=DENSITY_PASSAGES,
            seed=DENSITY_SEED,
            max_queries=DENSITY_QUERIES,
            mode="section",
            min_section_gold=DENSITY_MIN_QUERY_GOLD,
            min_query_gold=DENSITY_MIN_QUERY_GOLD,
            max_section_gold=gold,
        )

    key = f"density_fiqa_s{DENSITY_SECTIONS}p{DENSITY_PASSAGES}q{DENSITY_QUERIES}g{gold}seed{DENSITY_SEED}"
    return Leg(name=f"density_g{gold}", cache_key=key, load=_load)


def _is_cached(leg: Leg, model: str, strategy: str) -> bool:
    key = f"hiergate__{leg.cache_key}__{model.replace('/', '_').replace(':', '-')}__{strategy}"
    return (DATA_DIR / "db" / f"{key}.complete").exists()


def _scored_queries(bench) -> List[Tuple[str, str, Dict[str, float]]]:
    """(qid, text, rel) for queries with at least one positive judgement."""
    out = []
    for qid, text in bench.queries.items():
        rel = bench.doc_qrels.get(qid, {})
        if any(r > 0 for r in rel.values()):
            out.append((qid, text, rel))
    return out


def _score(db, queries, **query_kwargs) -> Dict[str, float]:
    ndcg, r10 = [], []
    for _qid, text, rel in queries:
        ranked = [h.id for h in db.query(text, k=K, return_type="documents", **query_kwargs)]
        ndcg.append(ndcg_at_k(ranked, rel, K))
        r10.append(recall_at_k(ranked, rel, K))
    n = len(queries) or 1
    return {"ndcg@10": sum(ndcg) / n, "recall@10": sum(r10) / n}


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
    queries = _scored_queries(bench)
    logger.info("[%s] %d docs, %d scored queries", leg.name, len(bench.corpus), len(queries))

    dbs = {
        s: build_db(
            bench,
            leg=leg,
            provider=args.embedding_provider,
            model=args.embedding_model,
            strategy=s,
            rebuild=False,
        )
        for s in STRATEGIES
    }
    for strategy, db in dbs.items():
        if db.count() != len(bench.corpus):
            raise RuntimeError(
                f"[{leg.name}] DB[{strategy}] holds {db.count()} documents, benchmark has "
                f"{len(bench.corpus)}. Stale cached index."
            )

    out: Dict[str, Any] = {"n_docs": len(bench.corpus), "n_queries": len(queries), "curves": {}}

    # Reference arms. ``chunks`` reads no section vector, so it is strategy-free;
    # scoring it once off the rawspan DB is not an arbitrary choice of index.
    st = getattr(args, "search_type", "hybrid")
    out["search_type"] = st
    out["chunks"] = _score(dbs["rawspan"], queries, search_level="chunks", search_type=st)
    logger.info("[%s] chunks                  ndcg@10 %.4f", leg.name, out["chunks"]["ndcg@10"])

    for strategy in STRATEGIES:
        db = dbs[strategy]
        out[f"sections · {strategy}"] = _score(db, queries, search_level="sections", search_type=st)
        curve = {}
        for w in WEIGHTS:
            curve[f"{w:.2f}"] = _score(db, queries, search_level="fused", section_weight=w, search_type=st)
            logger.info(
                "[%s] fused · %-8s w=%.2f  ndcg@10 %.4f",
                leg.name,
                strategy,
                w,
                curve[f"{w:.2f}"]["ndcg@10"],
            )
        out["curves"][strategy] = curve
    for db in dbs.values():
        db.close()
    return out


def _summarise(name: str, payload: Dict[str, Any]) -> None:
    chunks = payload["chunks"]["ndcg@10"]
    print(f"\n{name}  ({payload['n_docs']} docs, {payload['n_queries']} queries)")
    print(f"  chunks (frequency_boost roll-up)   ndcg@10 {chunks:.4f}")
    header = "  " + f"{'weight':<10}" + "".join(f"{s:>14}" for s in STRATEGIES)
    print(header)
    for w in WEIGHTS:
        row = f"  {w:<10.2f}"
        for s in STRATEGIES:
            v = payload["curves"][s][f"{w:.2f}"]["ndcg@10"]
            mark = "*" if abs(w - SHIPPED_WEIGHT) < 1e-9 else " "
            row += f"{v:>13.4f}{mark}"
        print(row)
    print(f"  {'sections':<10}" + "".join(f"{payload[f'sections · {s}']['ndcg@10']:>13.4f} " for s in STRATEGIES))

    for s in STRATEGIES:
        curve = payload["curves"][s]
        best_w, best = max(curve.items(), key=lambda kv: kv[1]["ndcg@10"])
        shipped = curve[f"{SHIPPED_WEIGHT:.2f}"]["ndcg@10"]
        zero = curve["0.00"]["ndcg@10"]
        print(
            f"  {s:<9} argmax w={best_w} -> {best['ndcg@10']:.4f} | "
            f"shipped 0.65 -> {shipped:.4f} ({shipped - best['ndcg@10']:+.4f} vs argmax, "
            f"{shipped - chunks:+.4f} vs chunks) | w=0 -> {zero:.4f} ({zero - chunks:+.4f} vs chunks)"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["density_g1", "density_g2", "density_g4", "superdocs", "qasper"],
        choices=["density_g1", "density_g2", "density_g4", "superdocs", "qasper"],
    )
    p.add_argument(
        "--allow-build",
        action="store_true",
        help="permit building a leg whose DB is not cached (slow); off by default",
    )
    p.add_argument("--out", default=str(_ROOT / "benchmarks" / "results" / "section_weight_sweep.json"))
    p.add_argument(
        "--search-type",
        choices=("hybrid", "vector", "keyword"),
        default="hybrid",
        help="Retrieval mechanism. THIS SWEEP IS THE ONE MOST DAMAGED BY THE DEFAULT: it compares "
        "fused against chunks across weights, and under hybrid only chunks has a keyword leg, so "
        "the reference arm carries a free +0.086 at every weight. The published conclusion 'fusion "
        "loses to chunks at every weight on every leg' was measured that way. Use 'vector'.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    # Load the encoder BEFORE any corpus. The order is load-bearing: loading a
    # BEIR corpus first puts a ``datasets`` module in ``sys.modules`` that shadows
    # HuggingFace's (``benchmarks/datasets.py``, reachable because the script's own
    # directory is ``sys.path[0]``), and ``transformers``' lazy import of it then
    # fails -- surfacing as ``LocalVectorDB``'s bare "model is not available".
    from benchmarks.eval_retrieval import preflight_embedding_model

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    builders = {
        "density_g1": lambda: density_leg(1),
        "density_g2": lambda: density_leg(2),
        "density_g4": lambda: density_leg(4),
        "superdocs": lambda: superdocs_leg(3, 32, 200, 0),
        "qasper": lambda: qasper_leg(None),
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
                "search_type": args.search_type,
                "weights": list(WEIGHTS),
                "shipped_weight": SHIPPED_WEIGHT,
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
