"""Significance testing for arm-vs-arm comparisons, from retained per-query scores.

WHY. Almost nothing in this project has been significance-tested, and the vector
re-scores made that urgent rather than tidy: removing BM25 shrank several
headline effects to the same order as the 0.011 build-noise floor. An effect
that cannot be distinguished from noise should not be in a conclusion.

WHY *PAIRED*. Two arms are scored on the same queries, so their per-query scores
are strongly correlated -- easy queries are easy for both. An unpaired interval
throws that away and is far too wide; it would call real effects insignificant.
Everything here pairs on the query id.

TWO STATISTICS, because they answer different questions:

* **Paired bootstrap CI** -- resample QUERIES with replacement, recompute the mean
  difference. Answers "how precise is this difference?" and yields an interval to
  quote. The interval, not the point estimate, is the result.
* **Randomisation (Fisher) test** -- under the null the two arms are
  interchangeable per query, so flip the sign of each query's difference at
  random and see how often |mean| exceeds the observed one. Answers "could this
  have arisen by chance?" and needs no distributional assumption. This is the
  standard test for IR effectiveness comparisons.

Both use the SAME paired differences, so they cannot disagree about direction.
When they disagree about significance, believe the interval: a p-value near 0.05
on 100 queries is not a finding.

ON QUERY COUNTS. The superdocs/density legs have 100-200 queries; qasper has 882.
At 100 queries a 95% interval on nDCG@10 is roughly +-0.03 -- wider than several
effects this project has reported to four decimals. That is the point.

Input is whatever ``eval_hier_gate.py --per-query-out`` wrote. Zero embedding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("bootstrap")

SEED = 0
RESAMPLES = 10_000


def paired(a: Dict[str, float], b: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    """Per-query differences a-b over the queries BOTH arms scored.

    The intersection matters. Arms skip queries that have no gold at their level,
    so two arms can report means over different query sets; differencing those
    means compares different populations. Pairing on the shared ids is the only
    honest comparison, and the count is reported so a heavy loss is visible.
    """
    shared = sorted(set(a) & set(b))
    return np.array([a[q] - b[q] for q in shared], dtype=np.float64), shared


def bootstrap_ci(diffs: np.ndarray, resamples: int = RESAMPLES, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(diffs)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def randomisation_p(diffs: np.ndarray, resamples: int = RESAMPLES) -> float:
    """Two-sided paired randomisation test.

    The +1 in numerator and denominator is not cosmetic: it keeps p strictly
    positive, since observing zero more-extreme resamples is evidence that p is
    small, not that it is zero.
    """
    rng = np.random.default_rng(SEED)
    observed = abs(diffs.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (resamples + 1))


def compare(a: Dict[str, float], b: Dict[str, float], label_a: str, label_b: str) -> Dict[str, object]:
    diffs, shared = paired(a, b)
    if len(diffs) < 2:
        return {"arm_a": label_a, "arm_b": label_b, "n": len(diffs), "error": "too few shared queries"}
    lo, hi = bootstrap_ci(diffs)
    return {
        "arm_a": label_a,
        "arm_b": label_b,
        "n": len(shared),
        "n_a_only": len(set(a) - set(b)),
        "n_b_only": len(set(b) - set(a)),
        "mean_a": float(np.mean([a[q] for q in shared])),
        "mean_b": float(np.mean([b[q] for q in shared])),
        "diff": float(diffs.mean()),
        "ci95": [lo, hi],
        "p": randomisation_p(diffs),
        "significant": bool(lo > 0 or hi < 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-query", type=Path, required=True, help="JSON from eval_hier_gate --per-query-out")
    p.add_argument("--metric", default="ndcg@10")
    p.add_argument("--leg", default=None, help="restrict to one leg (default: all)")
    p.add_argument("--baseline-arm", default="chunks", help="arm every other arm is compared against")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = json.loads(args.per_query.read_text(encoding="utf-8"))
    legs = payload["legs"]
    print(f"\n{payload.get('model', '?')}  search_type={payload.get('search_type', '?')}  metric={args.metric}")
    print(f"paired bootstrap, {RESAMPLES} resamples, seed {SEED}\n")

    out: Dict[str, object] = {"source": str(args.per_query), "metric": args.metric, "legs": {}}
    for leg_name, arms in legs.items():
        if args.leg and leg_name != args.leg:
            continue
        base = arms.get(args.baseline_arm, {}).get(args.metric)
        if not base:
            print(f"{leg_name}: no '{args.baseline_arm}' arm with metric {args.metric}; skipped")
            continue
        print(f"{leg_name}  (vs {args.baseline_arm}, n queries in column)")
        print(f"  {'arm':24s} {'n':>5s} {'mean':>8s} {'diff':>9s} {'95% CI':>20s} {'p':>8s}")
        rows = []
        for arm, metrics in arms.items():
            if arm == args.baseline_arm or args.metric not in metrics:
                continue
            r = compare(metrics[args.metric], base, arm, args.baseline_arm)
            rows.append(r)
            if "error" in r:
                print(f"  {arm:24s} {r['n']:>5d}  {r['error']}")
                continue
            star = "*" if r["significant"] else " "
            ci = f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
            print(f"  {arm:24s} {r['n']:>5d} {r['mean_a']:>8.4f} {r['diff']:>+9.4f} {ci:>20s} {r['p']:>8.4f}{star}")
        out["legs"][leg_name] = rows
        print()

    print("* = 95% CI excludes zero. A CI that straddles zero means the arms are not")
    print("  distinguishable on this many queries, whatever the point estimate says.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
