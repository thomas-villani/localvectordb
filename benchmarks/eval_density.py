"""DIAGNOSTIC: does gold DENSITY rescue the section vector at fixed section length?

The long gate leg is deliberately harsh -- ``mode="point"`` leaves the gold
passage at ~3% of its section, so a section vector is ~97% distractor and its
0.10 against chunks' 0.29 may be an artifact of the setup rather than a fact
about section retrieval. This is the fair test of the hierarchy premise.

DESIGN. Three rungs, identical in every respect except the number of gold
passages placed in the gold section:

    S=3 x P=32 FiQA passages, queries with >= 4 in-corpus golds, mode="section",
    max_section_gold in {1, 2, 4}  ->  gold density 3.1% / 6.2% / 12.5%

Every rung uses ``mode="section"`` -- including the 1-gold rung -- so gold always
lands at the START of a randomly chosen section. Running the 1-gold rung as
``mode="point"`` instead would have varied gold *position* alongside gold count
and confounded the ladder.

WHY THE DENSITY RANGE IS NARROW, AND WHY IT CANNOT BE WIDENED. Density is capped
by how many golds a query actually has. FiQA's median is 2 (max 15), so 12.5% is
near its ceiling at this geometry. NFCorpus has the multiplicity (median 16, mean
38) but is **disqualified**: 89-93% of its placed gold passages are also gold for
another placed query, which would make ``doc_qrels`` systematically false-negative.
The other route to density -- shrinking P -- shortens the section at the same
time, which just re-measures the granularity ladder. So density is separable from
length only over this narrow range, and that is a property of the available
corpora, not a choice.

READ THE SECTION-LEVEL COLUMN. Document-level nDCG rolls a doc up to its best
section (max), so more gold passages give the CHUNK arm more independent shots at
the same document (the §6.13 free-redundancy channel) -- the doc column is
expected to rise with density for reasons that have nothing to do with section
representation. ``ndcg@10_sections`` scores the section itself and is the metric
the hypothesis is actually about.

Zero API spend (local sentence-transformers). ~6 builds, ~40 min on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.eval_hier_gate import Leg, build_configs, build_db, run_config  # noqa: E402

logger = logging.getLogger("eval_density")

SECTIONS = 3
PASSAGES = 32
SEED = 0
MIN_QUERY_GOLD = 4  # eligibility; identical across rungs so the query set matches
GOLD_RUNGS = (1, 2, 4)


def density_leg(gold: int, max_queries: int) -> Leg:
    from benchmarks import beir_data
    from benchmarks.superdocs import build_synthetic_benchmark

    def _load():
        source = beir_data.load("fiqa")
        return build_synthetic_benchmark(
            source,
            sections_per_doc=SECTIONS,
            passages_per_section=PASSAGES,
            seed=SEED,
            max_queries=max_queries,
            mode="section",
            min_section_gold=MIN_QUERY_GOLD,
            min_query_gold=MIN_QUERY_GOLD,
            max_section_gold=gold,
        )

    key = f"density_fiqa_s{SECTIONS}p{PASSAGES}q{max_queries}g{gold}seed{SEED}"
    return Leg(name=f"gold{gold}", cache_key=key, load=_load)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--out", default=str(_ROOT / "benchmarks" / "results" / "density_ladder.json"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configs = build_configs()
    results = {}

    for gold in GOLD_RUNGS:
        leg = density_leg(gold, args.max_queries)
        bench = leg.load()
        density = gold / PASSAGES
        logger.info(
            "[%s] %d docs, %d queries, gold density %.1f%% (%d/%d passages)",
            leg.name,
            len(bench.corpus),
            len(bench.queries),
            100 * density,
            gold,
            PASSAGES,
        )
        # Guard: the whole ladder is void if the rungs do not share a query set.
        results.setdefault("_queries", {})[leg.name] = sorted(bench.queries)

        dbs = {
            s: build_db(
                bench,
                leg=leg,
                provider=args.embedding_provider,
                model=args.embedding_model,
                strategy=s,
                rebuild=args.rebuild,
            )
            for s in ("rawspan", "centroid")
        }
        rung = {"gold_placed": gold, "density": density, "queries": len(bench.queries)}
        for cfg in configs:
            scores = run_config(dbs[cfg.strategy], bench, cfg)
            rung[cfg.label] = scores
            logger.info(
                "  %-24s ndcg@10=%.4f  sec=%s",
                cfg.label,
                scores["ndcg@10"],
                f"{scores['ndcg@10_sections']:.4f}" if "ndcg@10_sections" in scores else "-",
            )
        results[leg.name] = rung
        for db in dbs.values():
            db.close()

    qsets = {name: set(q) for name, q in results.pop("_queries").items()}
    first = next(iter(qsets.values()))
    matched = all(q == first for q in qsets.values())
    results["query_sets_matched"] = matched
    if not matched:
        logger.error("QUERY SETS DIFFER ACROSS RUNGS -- the ladder is not a controlled comparison")

    results["meta"] = {
        "model": args.embedding_model,
        "provider": args.embedding_provider,
        "grid": f"{SECTIONS}x{PASSAGES}",
        "min_query_gold": MIN_QUERY_GOLD,
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'arm':<26}" + "".join(f"{f'{g}g ({100*g/PASSAGES:.1f}%)':>16}" for g in GOLD_RUNGS))
    for cfg in configs:
        row = f"{cfg.label:<26}"
        for g in GOLD_RUNGS:
            row += f"{results[f'gold{g}'][cfg.label]['ndcg@10']:>16.4f}"
        print(row)
    print(f"\n{'section-level nDCG@10':<26}")
    for cfg in configs:
        if cfg.search_level != "sections":
            continue
        row = f"{cfg.label:<26}"
        for g in GOLD_RUNGS:
            row += f"{results[f'gold{g}'][cfg.label].get('ndcg@10_sections', float('nan')):>16.4f}"
        print(row)
    print(f"\nquery sets matched across rungs: {matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
