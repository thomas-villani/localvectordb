"""Section-level regression gate for the hierarchical retrieval path.

WHY THIS EXISTS
---------------
``eval_retrieval.py`` -- the gate this project uses for ``src/`` default changes --
contains **zero** occurrences of ``hierarchical``, ``section`` or ``search_level``.
It is BEIR SciFact at chunk level. So every section-level default is currently
ungated: a change to ``section_vector_strategy``, to ``_span_embed`` window
sizing, or to chunk->section attribution passes ``--check`` at +0.0000 while
moving section retrieval by up to 0.36 nDCG (``span-length-crossover-findings``
§6.26). This file closes that hole.

Unlike ``eval_levels.py``, which reimplements the pooling maths in numpy, this
gate drives the **real** ``LocalVectorDB.query()`` path -- ingest, FAISS section
index, ``search_level``, the lot -- so it fails when ``src/`` breaks, not when a
harness diverges.

COVERAGE, STATED HONESTLY
-------------------------
The Qasper leg covers the **short-section** regime only: Qasper sections average
~190 tokens, so a raw-span section vector really is one global embedding and
never window-pools. That is the regime F2 was measured in, and where raw-span is
defensible (egemma +0.0818, §6.23).

It therefore does **NOT** cover the regime the headline finding is about --
sections past ~2k tokens, where raw-span loses 0.25-0.36 nDCG to the centroid.
Catching *that* needs a long-section leg (MAUD section-target, or a third
corpus). Until one lands, a green run here is evidence about short sections and
nothing else. Do not read it as blanket cover for a section-level change.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR, EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("eval_hier_gate")

BASELINE_JSON = _ROOT / "benchmarks" / "hier_gate_baseline.json"
RECALL_K = (1, 5, 10)
K = 10
TOLERANCE = 0.005


@dataclass(frozen=True)
class HierConfig:
    search_level: str
    strategy: str  # section_vector_strategy the DB was built with

    @property
    def label(self) -> str:
        # The chunk leg does not read section vectors, so the strategy is not
        # part of its identity -- labelling it would imply two distinct arms.
        return self.search_level if self.search_level == "chunks" else f"{self.search_level} · {self.strategy}"


def build_configs() -> List[HierConfig]:
    out = [HierConfig("chunks", "centroid")]
    for strategy in ("rawspan", "centroid"):
        for level in ("sections", "fused"):
            out.append(HierConfig(level, strategy))
    return out


def build_db(bench, *, provider: str, model: str, strategy: str, rebuild: bool, max_papers: Optional[int]):
    """Build (or reopen) a hierarchical DB for one section_vector_strategy.

    The strategy is baked in at ingest -- section vectors are written once -- so
    each arm needs its own database, and the key must carry the strategy or the
    two arms silently share an index and the A/B compares a build to itself.

    The key must ALSO carry ``max_papers``. Omitting it made a ``--max-papers 12``
    smoke build get reused by the full 275-paper run: 263 gold documents were
    simply absent from the index and every arm scored at chance (nDCG 0.0357 ~=
    10/275) while looking like a real result. Anything that changes the CONTENT
    of the index belongs in the key.
    """
    from localvectordb import LocalVectorDB

    key = f"hiergate__qasper__{model.replace('/', '_').replace(':', '-')}__{strategy}"
    if max_papers is not None:
        key += f"__max{max_papers}"
    base = DATA_DIR / "db"
    base.mkdir(parents=True, exist_ok=True)
    sentinel = base / f"{key}.complete"

    def _discard() -> None:
        for path in base.glob(f"{key}.*"):
            path.unlink(missing_ok=True)

    if rebuild:
        _discard()

    kwargs: Dict[str, Any] = dict(
        embedding_provider=provider,
        embedding_model=model,
        hierarchical_embeddings=True,
        section_vector_strategy=strategy,
    )
    if sentinel.exists():
        logger.info("Reusing cached DB %s", key)
        return LocalVectorDB(key, base, **kwargs)

    if any(base.glob(f"{key}.*")):
        logger.warning("Discarding incomplete DB %s", key)
        _discard()

    logger.info("Building %s (%d docs) -- slow part", key, len(bench.corpus))
    db = LocalVectorDB(key, base, **kwargs)
    doc_ids = list(bench.corpus)
    slab = 100
    for start in range(0, len(doc_ids), slab):
        batch = doc_ids[start : start + slab]
        db.upsert([bench.corpus[d] for d in batch], ids=batch)
        logger.info("  ingested %d/%d", min(start + slab, len(doc_ids)), len(doc_ids))
    db.save()
    # Confirm the strategy actually persisted; a silently-defaulted DB would make
    # the two arms identical and the whole comparison vacuous.
    got = getattr(db, "section_vector_strategy", None)
    if got != strategy:
        raise RuntimeError(f"DB reports section_vector_strategy={got!r}, expected {strategy!r}")
    sentinel.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return db


def run_config(db, bench, config: HierConfig) -> Dict[str, float]:
    """Score one arm at DOCUMENT level so every arm is measured in one unit."""
    scores: Dict[str, List[float]] = {f"recall@{k}": [] for k in RECALL_K}
    scores["ndcg@10"] = []
    for qid, text in bench.queries.items():
        rel = bench.doc_qrels.get(qid, {})
        if not any(r > 0 for r in rel.values()):
            continue
        hits = db.query(
            text,
            search_level=config.search_level,
            return_type="documents",
            k=K,
        )
        ranked = [h.id for h in hits]
        scores["ndcg@10"].append(ndcg_at_k(ranked, rel, K))
        for k in RECALL_K:
            scores[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
    return {m: (sum(v) / len(v) if v else 0.0) for m, v in scores.items()}


def compare_to_baseline(results: Dict[str, Dict[str, float]], path: Path, tolerance: float) -> int:
    if not path.exists():
        print(f"No baseline at {path}; run with --save-baseline first.", file=sys.stderr)
        return 1
    base = json.loads(path.read_text(encoding="utf-8"))["results"]
    worst, failed = 0.0, []
    for label, metrics in results.items():
        if label not in base:
            print(f"  NEW  {label} (not in baseline)")
            continue
        delta = metrics["ndcg@10"] - base[label]["ndcg@10"]
        worst = min(worst, delta)
        flag = "FAIL" if delta < -tolerance else "ok"
        if delta < -tolerance:
            failed.append(label)
        print(f"  {flag:4s} {label:28s} {metrics['ndcg@10']:.4f}  ({delta:+.4f})")
    print(f"\nworst delta {worst:+.4f} (tolerance {tolerance})")
    return 1 if failed else 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    p.add_argument("--tolerance", type=float, default=TOLERANCE)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    # LocalVectorDB.__init__ turns ANY provider failure into a bare "model is not
    # available" (a cold HF cache included), so surface the real error first.
    from benchmarks.eval_retrieval import preflight_embedding_model
    from benchmarks.qasper_data import load_qasper

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    bench = load_qasper(split="dev", max_papers=args.max_papers)
    logger.info("Qasper dev: %d docs, %d queries", len(bench.corpus), len(bench.queries))

    dbs = {
        s: build_db(
            bench,
            provider=args.embedding_provider,
            model=args.embedding_model,
            strategy=s,
            rebuild=args.rebuild,
            max_papers=args.max_papers,
        )
        for s in ("rawspan", "centroid")
    }

    # Belt and braces on the cache key: assert the index actually holds the corpus
    # being scored. A stale-but-valid database is the failure mode here -- it does
    # not error, it just returns chance-level numbers that read as a real result.
    for strategy, db in dbs.items():
        n = db.count()
        if n != len(bench.corpus):
            raise RuntimeError(
                f"DB[{strategy}] holds {n} documents but the benchmark has "
                f"{len(bench.corpus)}. Stale cached index -- re-run with --rebuild."
            )
    logger.info("Index check: both DBs hold %d documents", len(bench.corpus))

    results: Dict[str, Dict[str, float]] = {}
    for config in build_configs():
        m = run_config(dbs[config.strategy], bench, config)
        results[config.label] = m
        logger.info(
            "%-28s ndcg@10 %.4f  r@1 %.4f  r@10 %.4f",
            config.label,
            m["ndcg@10"],
            m["recall@1"],
            m["recall@10"],
        )

    print("\n" + "=" * 66)
    print(f"{'arm':28s} {'ndcg@10':>9s} {'r@1':>8s} {'r@5':>8s} {'r@10':>8s}")
    for label, m in results.items():
        print(f"{label:28s} {m['ndcg@10']:9.4f} {m['recall@1']:8.4f} {m['recall@5']:8.4f} {m['recall@10']:8.4f}")
    print("=" * 66)

    if args.save_baseline:
        args.baseline.write_text(
            json.dumps(
                {
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "model": args.embedding_model,
                    "dataset": "qasper-dev",
                    "n_docs": len(bench.corpus),
                    "n_queries": len(bench.queries),
                    "coverage": "SHORT sections only (~190 tok mean); no long-span leg yet",
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote baseline {args.baseline}")
    if args.check:
        return compare_to_baseline(results, args.baseline, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
