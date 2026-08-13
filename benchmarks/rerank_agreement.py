"""Do rerankers cluster by training lineage? (PAPER-OUTLINE §9.3)

Motivation. §4.3.1 found that the reranker ranking does NOT transfer between
corpora: among competent models the cross-corpus rank correlation is -0.20, and
one pair inverts *significantly in both directions*. Two explanations survive:

  (a) STRUCTURED -- models genuinely specialise, so which one wins depends on the
      corpus. Then models with shared training lineage should behave alike, and
      the per-query agreement matrix should show family blocks.
  (b) UNSTRUCTURED -- the reshuffle is idiosyncratic (or noise), and agreement
      carries no family signal at all.

Related: NFCorpus is a BEIR dataset WITH A PUBLIC TRAIN SPLIT and is standard
training material; qasper is in no retrieval mixture. If shared training data
drives agreement, clustering should be *stronger* on NFCorpus. That comparison
cannot prove contamination -- only a corpus nobody has published could -- but it
separates structured non-transfer from noise, which is the question that decides
whether "models specialise" is a claim or a story.

THE CONFOUND THIS FILE EXISTS TO REMOVE. Raw per-query nDCG correlates highly
between ANY two rerankers, because most of the variance is *query difficulty*:
easy queries score high for everyone. That number is near-meaningless and would
manufacture a cluster where none exists. So the matrix is computed on
DIFFICULTY-CENTERED residuals -- per query, the across-model mean is subtracted,
leaving only each model's deviation from the consensus. Both are printed, and the
gap between them is the point.

POSITIVE CONTROL, free and built in: the `--max-length 256` MiniLM arm is the
same weights as the 512 arm, so it must land in the top tier of the matrix or
nothing here is readable.

    MEASURED: r=+0.519, rank 3 of 45 -- passes, but NOT first, and the miss is
    a result rather than a defect. It is beaten by voyage-2.5/2.5-lite (+0.550)
    and MiniLM-L6/L12 (+0.528). Halving the token budget churns WHICH queries
    the model gets right about as much as swapping to a different model in the
    same family -- while moving aggregate nDCG@10 by +0.0034. A stable
    aggregate is not evidence of a stable ranker.

READ THE CELLS AGAINST -1/(k-1), NOT ZERO. Residuals sum to zero per query by
construction, so the mean pairwise correlation is forced to about -1/(k-1):
predicted -0.111 at k=10 and -0.200 at k=6, observed -0.108 and -0.191. A cell
at -0.15 is therefore near-average, not "anti-correlated". The permutation test
below is unaffected -- the same constraint holds under the null -- which is why
the inference rests on it and not on the coefficients.

Zero cost: reads artifacts, no API calls, no embedding.

Usage:
    python benchmarks/rerank_agreement.py --dataset qasper
    python benchmarks/rerank_agreement.py --dataset nfcorpus
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_rerankers import check_first_stage, load  # noqa: E402

# Training lineage, not vendor branding: what matters is whether two models
# plausibly share training data. Singletons contribute no same-family pair and
# are carried only for the cross-family baseline.
FAMILY = {
    "cohere/": "cohere",
    "voyageai/": "voyage",
    "cross-encoder/ms-marco": "ms-marco",
    "BAAI/bge": "bge",
    "nvidia/": "nvidia",
}


def family_of(label: str) -> str:
    for prefix, fam in FAMILY.items():
        if label.startswith(prefix):
            return fam
    return "other"


def base_model(label: str) -> str:
    """Strip the `@len256` / `@max5` suffixes -- same weights, different knob."""
    return label.split(" @")[0]


def permutation_test(
    r: np.ndarray, fams: List[str], same: np.ndarray, rng: np.random.Generator, n: int = 10000
) -> Tuple[float, float]:
    """Is same-family agreement higher than cross-family? Shuffle the labels.

    A t-test would be wrong here: the pairs are not independent (every model
    appears in many of them), so the null has to be built by permuting model
    labels rather than by assuming a distribution over pairs.
    """
    iu = np.triu_indices(len(fams), k=1)
    obs = r[iu][same[iu]].mean() - r[iu][~same[iu]].mean()
    fams_arr = np.asarray(fams)
    hits = 0
    for _ in range(n):
        perm = rng.permutation(fams_arr)
        s = perm[:, None] == perm[None, :]
        if not (s[iu].any() and (~s[iu]).any()):
            continue
        if r[iu][s[iu]].mean() - r[iu][~s[iu]].mean() >= obs:
            hits += 1
    return float(obs), (hits + 1) / (n + 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="qasper")
    p.add_argument("--search-type", default="hybrid")
    p.add_argument("--method", default="frequency_boost", choices=["frequency_boost", "best"])
    p.add_argument("--pool", type=int, default=40)
    args = p.parse_args()

    arts = load(args.dataset, args.search_type)
    key = f"pool={args.pool}|{args.method}"
    if len(arts) < 3:
        print(f"need >=3 artifacts, found {len(arts)}")
        return 1

    print(f"\n{args.dataset} / {args.search_type} / {key}  ({len(arts)} arms)\n")
    if not check_first_stage(arts, key):
        print("\n  first stage is not identical -- per-query agreement is not interpretable")
        return 1

    labels = sorted(arts, key=lambda m: -np.asarray(arts[m]["per_query"][f"{key}|rerank"]).mean())
    M = np.vstack([arts[m]["per_query"][f"{key}|rerank"] for m in labels])
    raw = np.corrcoef(M)
    # Remove query difficulty: what is left is each model's deviation from the
    # per-query consensus, i.e. WHERE a model is unusually strong or weak.
    resid = M - M.mean(axis=0, keepdims=True)
    cen = np.corrcoef(resid)

    iu = np.triu_indices(len(labels), k=1)
    print(f"\n  mean pairwise r, raw (query difficulty included): {raw[iu].mean():+.3f}")
    print(f"  mean pairwise r, difficulty-centered:             {cen[iu].mean():+.3f}")
    print("  (the first is mostly 'some queries are easy' and is not evidence of anything)")

    short = [base_model(m).split("/")[-1][:20] + m[len(base_model(m)) :] for m in labels]
    print("\ndifficulty-centered per-query agreement\n")
    print(f"{'':<26}" + "".join(f"{s[:10]:>11}" for s in short))
    for i, s in enumerate(short):
        print(f"{s:<26}" + "".join("         --" if i == j else f"{cen[i, j]:>+11.3f}" for j in range(len(labels))))

    # Positive control: identical weights, different max_length.
    ctl = [
        (i, j)
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if base_model(labels[i]) == base_model(labels[j])
    ]
    print()
    if ctl:
        for i, j in ctl:
            rank = int((cen[iu] > cen[i, j]).sum()) + 1
            print(
                f"  POSITIVE CONTROL {short[i]} vs {short[j]}: r={cen[i, j]:+.3f}, "
                f"rank {rank} of {len(iu[0])} pairs"
            )
    else:
        print("  (no same-weights pair present -- positive control unavailable)")

    # Family clustering, excluding same-weights duplicates from both groups.
    keep = [i for i in range(len(labels)) if base_model(labels[i]) not in {base_model(labels[j]) for j in range(i)}]
    fams = [family_of(labels[i]) for i in keep]
    sub = cen[np.ix_(keep, keep)]
    same = np.asarray(fams)[:, None] == np.asarray(fams)[None, :]
    iu2 = np.triu_indices(len(keep), k=1)
    n_same = int(same[iu2].sum())
    print(f"\n  families: {dict((f, fams.count(f)) for f in sorted(set(fams)))}")
    if n_same == 0 or n_same == len(iu2[0]):
        print("  not enough family structure to test clustering on this corpus")
        return 0
    obs, pval = permutation_test(sub, fams, same, np.random.default_rng(0))
    print(f"  same-family pairs   n={n_same:<3} mean r = {sub[iu2][same[iu2]].mean():+.3f}")
    print(f"  cross-family pairs  n={len(iu2[0]) - n_same:<3} mean r = {sub[iu2][~same[iu2]].mean():+.3f}")
    print(f"  difference {obs:+.3f}, permutation p = {pval:.4f}  (10,000 label shuffles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
