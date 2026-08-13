"""Does corpus-derived tuning survive out-of-sample? (PAPER-OUTLINE C4)

Every argmax in this study is IN-SAMPLE: picked on the queries it is scored on
(SYNTHESIS-v2 §17.7 says so explicitly). That licenses the claim "the optimum is
corpus-dependent" and NOT the claim "you can find your optimum", which is the
actionable one and the one a reviewer will ask for first.

This harness answers the second. For each leg it splits the query set in half,
picks the best arm on half A, and scores that arm on half B against the shipped
default -- so the parameter is chosen without ever seeing the queries it is
graded on.

Three numbers per leg, and the third is what makes it honest:

* **shipped**   -- the constant every user gets today.
* **tuned**     -- argmax on A, evaluated on B. What a `doctor`-style
                   recommendation would actually deliver.
* **oracle**    -- argmax on B itself. The in-sample number this study has been
                   quoting all along, and an upper bound tuning cannot reach.

`capture` = (tuned - shipped) / (oracle - shipped) is the fraction of the
advertised gain that survives contact with unseen queries. A capture near 1.0
means the in-sample tables were honest; near 0 means they were overfitting to
their own query sample.

**Many splits, not one.** A single 50/50 split is one draw, and with ~10 arms the
argmax on half a query set is noisy. Every leg is run over ``--splits`` distinct
partitions and we report the mean plus **win rate** -- the fraction of splits
where tuning beat the shipped constant. A method that wins on average but loses
a third of the time is not a recommendation.

Zero embedding, zero retrieval: reads the per-query nDCG already stored in the
pool-width artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# The arm a user gets today: k*4 capped at 100 (=40 at k=10), and
# `document_scoring_method="auto"` resolving to frequency_boost on hybrid.
SHIPPED_ARM = "pool=40|frequency_boost"

LEGS: List[Tuple[str, str]] = [
    ("qasper dev (275 docs)", "experiments/poolwidth_qasper.json"),
    ("qasper_full (1,088 docs)", "experiments/poolwidth_qasper_full.json"),
    ("MLDR small (1,520 docs)", "experiments/poolwidth_mldr_small.json"),
    ("MLDR big (6,078 docs)", "experiments/poolwidth_mldr.json"),
    ("NQ (1,948 docs)", "experiments/poolwidth_nq.json"),
    ("NFCorpus (3,633 docs)", "experiments/poolwidth_nfcorpus.json"),
]


def split_mask(qids: Sequence[str], seed: int) -> np.ndarray:
    """Deterministic 50/50 partition, stable across machines and reruns.

    Hash-based rather than ``rng.shuffle`` so a given (qid, seed) always lands in
    the same half no matter what else is in the list -- two legs sharing a query
    set therefore get the SAME partition, which is what makes MLDR small and
    MLDR big comparable split-for-split.
    """
    keys = np.array([int(hashlib.sha256(f"{seed}\x00{q}".encode()).hexdigest()[:8], 16) for q in qids])
    return keys < np.median(keys)


def evaluate(path: Path, *, splits: int, exclude: Sequence[str] = ()) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    pq: Dict[str, List[float]] = data["per_query"]
    qids: List[str] = data["qids"]
    if SHIPPED_ARM not in pq:
        raise SystemExit(f"{path}: no {SHIPPED_ARM!r} arm -- cannot measure against what ships")

    # `average` is retained for API compatibility and is never competitive
    # (§18.8). Leaving it in only gives the tuner a way to lose that no real
    # recommender would take.
    arms = [a for a in pq if a not in exclude]
    scores = {a: np.asarray(pq[a], dtype=float) for a in arms}
    shipped = scores[SHIPPED_ARM]

    tuned_deltas, oracle_deltas, picks = [], [], []
    for seed in range(splits):
        a_mask = split_mask(qids, seed)
        b_mask = ~a_mask
        # Tune on A ...
        pick = max(arms, key=lambda arm: scores[arm][a_mask].mean())
        # ... grade on B, which the choice above never saw.
        tuned_deltas.append(float(scores[pick][b_mask].mean() - shipped[b_mask].mean()))
        best_on_b = max(arms, key=lambda arm: scores[arm][b_mask].mean())
        oracle_deltas.append(float(scores[best_on_b][b_mask].mean() - shipped[b_mask].mean()))
        picks.append(pick)

    tuned = np.asarray(tuned_deltas)
    oracle = np.asarray(oracle_deltas)
    # "Fraction of the oracle captured" only means anything when there IS an
    # oracle gain to capture and tuning actually gained. Where tuning LOSES, the
    # ratio is a negative number with a near-zero denominator -- it looks like a
    # measurement (-178%!) and is an artifact. Report NaN and let the caller say
    # "tuning lost" in words.
    meaningful = oracle.mean() > 2e-3 and tuned.mean() > 0
    capture = float(tuned.mean() / oracle.mean()) if meaningful else float("nan")
    modal = max(set(picks), key=picks.count)
    return {
        "n_queries": len(qids),
        "n_arms": len(arms),
        "shipped": float(shipped.mean()),
        "tuned_delta": float(tuned.mean()),
        "tuned_sd": float(tuned.std()),
        "oracle_delta": float(oracle.mean()),
        "capture": capture,
        "win_rate": float((tuned > 0).mean()),
        "modal_pick": modal,
        "modal_pick_share": picks.count(modal) / len(picks),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=int, default=200, help="distinct 50/50 partitions per leg")
    p.add_argument("--keep-average", action="store_true", help="leave the `average` arms in the pool")
    p.add_argument("--out", type=Path, default=Path("experiments/heldout_tuning.json"))
    args = p.parse_args(argv)

    exclude = () if args.keep_average else tuple(f"pool={p_}|average" for p_ in (10, 20, 40, 100, 200, 400))

    print(f"\nHeld-out parameter tuning -- {args.splits} random 50/50 splits per leg")
    print(f"tune on half A, grade on half B, baseline = {SHIPPED_ARM}\n")
    head = f"{'leg':<26}{'n':>6}{'shipped':>9}{'tuned':>10}{'oracle':>10}{'capture':>9}{'win rate':>10}  modal pick"
    print(head)
    print("-" * (len(head) + 12))

    out: Dict[str, object] = {}
    for title, path in LEGS:
        r = evaluate(_ROOT / path, splits=args.splits, exclude=exclude)
        if r is None:
            print(f"{title:<26}{'(artifact missing)':>45}")
            continue
        out[title] = r
        cap = "  n/a" if np.isnan(r["capture"]) else f"{r['capture']:>6.0%}"
        print(
            f"{title:<26}{r['n_queries']:>6}{r['shipped']:>9.4f}"
            f"{r['tuned_delta']:>+10.4f}{r['oracle_delta']:>+10.4f}{cap:>9}"
            f"{r['win_rate']:>10.0%}  {r['modal_pick']} ({r['modal_pick_share']:.0%})"
        )

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
