"""DIAGNOSTIC: do you need a section VECTOR to rank sections?

Everything measured so far compares the section index to chunk retrieval at
**document** level, where a document rolls up to its best section and the
comparison is structurally blind to per-section reachability (§6.26). The
question that was never asked is the one that decides whether the section FAISS
index earns its existence:

    Rank SECTIONS. Once by searching section vectors, once by searching chunks
    and rolling each hit up to its containing section. Score both against
    ``section_qrels``.

If the roll-up wins, the section index has no reason to exist -- chunk retrieval
already finds the right section, and "sections" is a **return unit**, not a
retrieval level. That would close the hierarchy question directly rather than by
inference from document-level results.

THE FETCH ASYMMETRY, AND WHY BOTH NUMBERS ARE REPORTED. ``_search`` sets
``fetch_k == k`` when no reranker is configured, so ``search_level="chunks",
return_type="sections", k=10`` retrieves **10 chunks** and groups them into *at
most* 10 sections -- usually far fewer, because the top chunks tend to share a
section. The section index meanwhile returns exactly 10 distinct sections. That
is a handicap on the roll-up, not a property of the representation, so:

    fetch x1   what a user gets from the shipped path today
    fetch x10  over-fetch chunks, then take the top 10 sections

The x1 row is the honest report of current behaviour; the x10 row is the fair
test of the representation question. Same split as the shipped-vs-oracle reading
of ``section_weight``, and for the same reason: a default and a capability are
different claims.

Zero API spend, no rebuilds -- every DB this reads is already built.
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
from benchmarks.eval_hier_gate import Leg, _section_qrel_id, build_db, qasper_leg, superdocs_leg  # noqa: E402
from benchmarks.eval_section_weight import density_leg  # noqa: E402
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("eval_section_finder")

K = 10
RECALL_K = (1, 5, 10)
OVERFETCH = (1, 10)
STRATEGIES = ("rawspan", "centroid")


def _is_cached(leg: Leg, model: str, strategy: str) -> bool:
    key = f"hiergate__{leg.cache_key}__{model.replace('/', '_').replace(':', '-')}__{strategy}"
    return (DATA_DIR / "db" / f"{key}.complete").exists()


def _scored_queries(bench) -> List[Tuple[str, str, Dict[str, float]]]:
    """(qid, text, section_rel) for queries with at least one positive SECTION judgement."""
    out = []
    for qid, text in bench.queries.items():
        rel = getattr(bench, "section_qrels", {}).get(qid, {})
        if any(r > 0 for r in rel.values()):
            out.append((qid, text, rel))
    return out


def _score(db, queries, *, fetch: int, **query_kwargs) -> Dict[str, float]:
    """Score section ranking. ``fetch`` over-fetches, then the top K are kept."""
    acc: Dict[str, List[float]] = {f"recall@{k}": [] for k in RECALL_K}
    acc["ndcg@10"] = []
    for _qid, text, rel in queries:
        hits = db.query(text, k=K * fetch, return_type="sections", **query_kwargs)
        ranked = [_section_qrel_id(h.id) for h in hits][:K]
        acc["ndcg@10"].append(ndcg_at_k(ranked, rel, K))
        for k in RECALL_K:
            acc[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
    n = len(queries) or 1
    return {m: sum(v) / n for m, v in acc.items()}


def run_leg(leg: Leg, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    missing = [s for s in STRATEGIES if not _is_cached(leg, args.embedding_model, s)]
    if missing:
        logger.warning("[%s] skipping: no cached DB for %s", leg.name, ", ".join(missing))
        return None

    bench = leg.load()
    queries = _scored_queries(bench)
    if not queries:
        logger.warning("[%s] skipping: no section_qrels", leg.name)
        return None
    logger.info("[%s] %d docs, %d queries with section judgements", leg.name, len(bench.corpus), len(queries))

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
    out: Dict[str, Any] = {"n_docs": len(bench.corpus), "n_queries": len(queries), "arms": {}}

    # The chunk->section roll-up reads no section vector, so it is strategy-free.
    for fetch in OVERFETCH:
        label = f"chunk rollup (fetch x{fetch})"
        out["arms"][label] = _score(dbs["rawspan"], queries, fetch=fetch, search_level="chunks")
        logger.info("[%s] %-28s ndcg@10 %.4f", leg.name, label, out["arms"][label]["ndcg@10"])

    # The section index returns distinct sections already; over-fetching cannot
    # add any, so it is scored once at fetch x1.
    for strategy in STRATEGIES:
        label = f"section index · {strategy}"
        out["arms"][label] = _score(dbs[strategy], queries, fetch=1, search_level="sections")
        logger.info("[%s] %-28s ndcg@10 %.4f", leg.name, label, out["arms"][label]["ndcg@10"])

    for db in dbs.values():
        db.close()
    return out


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
    p.add_argument("--out", default=str(_ROOT / "benchmarks" / "results" / "section_finder.json"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    # Encoder before corpus -- see the note in eval_chunk_size.main.
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
        print(f"\n{name}  ({payload['n_docs']} docs, {payload['n_queries']} queries)  SECTION-level")
        print(f"  {'arm':<30}{'ndcg@10':>9}{'r@1':>8}{'r@5':>8}{'r@10':>8}")
        for label, m in payload["arms"].items():
            print(
                f"  {label:<30}{m['ndcg@10']:>9.4f}{m['recall@1']:>8.4f}" f"{m['recall@5']:>8.4f}{m['recall@10']:>8.4f}"
            )
        best_index = max(payload["arms"][f"section index · {s}"]["ndcg@10"] for s in STRATEGIES)
        for fetch in OVERFETCH:
            rollup = payload["arms"][f"chunk rollup (fetch x{fetch})"]["ndcg@10"]
            verdict = "rollup WINS" if rollup > best_index else "section index wins"
            print(f"  fetch x{fetch}: rollup {rollup:.4f} vs best section index {best_index:.4f} -> {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(),
                "model": args.embedding_model,
                "k": K,
                "overfetch": list(OVERFETCH),
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
