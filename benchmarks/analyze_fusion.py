"""Fusion-miss anatomy and selector bake-off (offline, cache-only).

    ./.venv/Scripts/python.exe benchmarks/analyze_fusion.py --dataset fiqa --mode section
    ./.venv/Scripts/python.exe benchmarks/analyze_fusion.py --dataset qasper

Phase-1 (``eval_hierarchical.py``) showed equal-weight fusion of chunk + a
section-level representation beats chunk-only on real/section data, but *loses*
to the single best arm when one arm dominates (F4). This script dissects that:

1. **Miss anatomy** -- per query, which arm is best, how much equal-weight fusion
   gives up against the best single arm, and whether that regret tracks the
   arm-strength gap, the query length, or arm disagreement.
2. **Selector bake-off** -- three rungs, each scored against the oracle ceiling
   (a perfect per-query arm picker), all on the two shippable arms
   (``chunk`` + ``rawspan-section``; no LLM):
     - **rung 1** best fixed global blend weight (one scalar, no per-query input);
     - **rung 2** unsupervised confidence weighting (per-arm-calibrated top score);
     - **rung 3** a trained query-feature router (single-feature threshold and a
       small logistic regression, both under k-fold cross-validation).

Runs entirely on the embeddings ``eval_hierarchical.py`` already cached, so it
needs no API key and costs nothing. Use identical ``--mode/--sections/...`` to
the cached run or you will force fresh (paid) embeddings.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Running this file as a script puts benchmarks/ on sys.path but not the repo
# root, so ``import benchmarks`` fails. Put the root on first; eval_hierarchical
# then re-fixes the path (root on, benchmarks/ off) at import time.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from benchmarks import beir_data  # noqa: E402
from benchmarks.eval_hierarchical import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    PRIMARY_K,
    BuiltVectors,
    CachedEncoder,
    build_vectors,
    rank_units,
)
from benchmarks.metrics import ndcg_at_k  # noqa: E402
from benchmarks.superdocs import SyntheticBenchmark, build_synthetic_benchmark  # noqa: E402

Scored = Dict[str, List[Tuple[str, float]]]  # qid -> [(unit, score)] descending


# --------------------------------------------------------------------------- #
# Per-query scoring primitives
# --------------------------------------------------------------------------- #
def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    span = (hi - lo) or 1.0
    return (scores - lo) / span


def weighted_fuse_query(chunk_pairs, sec_pairs, sec_weight: float) -> List[str]:
    """Rank one query by ``(1-w)*chunk + w*section`` on min-max-normalised scores.

    ``sec_weight=0`` is chunk-only, ``1`` is section-only, ``0.5`` reproduces the
    equal-weight relative-score fusion of ``eval_hierarchical.fuse``.
    """
    agg: Dict[str, float] = {}
    for pairs, w in ((chunk_pairs, 1.0 - sec_weight), (sec_pairs, sec_weight)):
        if not pairs or w == 0.0:
            continue
        norm = _minmax(np.array([s for _, s in pairs], dtype=np.float64))
        for (doc, _), v in zip(pairs, norm, strict=True):
            agg[doc] = agg.get(doc, 0.0) + w * float(v)
    return [d for d, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


@dataclass
class ArmRuns:
    """Chunk vs one coarse section-level arm, for one target, as scored runs.

    ``coarse_name`` labels the coarse arm (``rawspan-section`` or
    ``summary-section``); ``section`` holds its per-query run. Everything
    downstream compares ``chunk`` against ``section`` generically, so the same
    anatomy + bake-off runs for either coarse representation.
    """

    target: str
    coarse_name: str
    qids: List[str]
    qrels: Dict[str, Dict[str, int]]
    query_texts: Dict[str, str]
    chunk: Scored
    section: Scored


def build_arm_runs(bench: SyntheticBenchmark, built: BuiltVectors, target: str) -> List[ArmRuns]:
    """One ArmRuns per coarse arm present: rawspan-section, and summary-section if built."""
    qids = built.query_ids
    qrels = bench.doc_qrels if target == "doc" else bench.section_qrels
    chunk = rank_units(built.query_vecs, qids, built.chunk, target)
    assert chunk is not None
    coarse_levels = [("rawspan-section", built.rawspan_section)]
    if built.summary_section is not None:
        coarse_levels.append(("summary-section", built.summary_section))
    out: List[ArmRuns] = []
    for name, level in coarse_levels:
        section = rank_units(built.query_vecs, qids, level, target)
        assert section is not None
        out.append(ArmRuns(target, name, qids, qrels, dict(bench.queries), chunk, section))
    return out


def mean_ndcg(runs_by_qid: Dict[str, List[str]], qrels, qids: List[str]) -> float:
    return float(np.mean([ndcg_at_k(runs_by_qid[q], qrels[q], PRIMARY_K) for q in qids]))


# --------------------------------------------------------------------------- #
# Query-time features (everything here is observable without qrels)
# --------------------------------------------------------------------------- #
@dataclass
class QueryFacts:
    qid: str
    chunk_ndcg: float  # outcome (needs qrels) -- used for labels/oracle only
    sec_ndcg: float
    chunk_top: float  # arm confidence signals (query-time observable)
    sec_top: float
    chunk_margin: float  # top1 - top2 within the arm
    sec_margin: float
    qlen: int  # query length in whitespace tokens
    agree: int  # 1 if the two arms agree on the top unit


def _top_and_margin(pairs: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    top_id, top = pairs[0]
    second = pairs[1][1] if len(pairs) > 1 else pairs[0][1]
    return top_id, float(top), float(top - second)


def collect_facts(arms: ArmRuns) -> List[QueryFacts]:
    facts: List[QueryFacts] = []
    for q in arms.qids:
        c_pairs, s_pairs = arms.chunk[q], arms.section[q]
        c_top_id, c_top, c_margin = _top_and_margin(c_pairs)
        s_top_id, s_top, s_margin = _top_and_margin(s_pairs)
        facts.append(
            QueryFacts(
                qid=q,
                chunk_ndcg=ndcg_at_k([d for d, _ in c_pairs], arms.qrels[q], PRIMARY_K),
                sec_ndcg=ndcg_at_k([d for d, _ in s_pairs], arms.qrels[q], PRIMARY_K),
                chunk_top=c_top,
                sec_top=s_top,
                chunk_margin=c_margin,
                sec_margin=s_margin,
                qlen=len(arms.query_texts[q].split()),
                agree=int(c_top_id == s_top_id),
            )
        )
    return facts


def feature_matrix(facts: List[QueryFacts]) -> Tuple[np.ndarray, List[str]]:
    """Query-time features only (no qrels), standardised later per fold.

    ``sec_top - chunk_top`` is the calibrated-margin hypothesis; the rest are the
    cheap secondary signals (per-arm confidence, disagreement, query length).
    """
    names = ["sec_top", "chunk_top", "sec_minus_chunk_top", "sec_margin", "chunk_margin", "disagree", "qlen"]
    rows = [
        [f.sec_top, f.chunk_top, f.sec_top - f.chunk_top, f.sec_margin, f.chunk_margin, 1 - f.agree, float(f.qlen)]
        for f in facts
    ]
    return np.array(rows, dtype=np.float64), names


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def _select_ndcg(facts: List[QueryFacts], choose_section: np.ndarray) -> float:
    """Mean nDCG when ``choose_section[i]`` routes query i to section else chunk."""
    return float(np.mean([f.sec_ndcg if pick else f.chunk_ndcg for f, pick in zip(facts, choose_section, strict=True)]))


def zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / (sd or 1.0)


def confidence_selector(arms: ArmRuns, facts: List[QueryFacts]) -> Tuple[float, float]:
    """Rung 2: route by which arm's top score is higher after per-arm calibration.

    Chunk and section cosines live on different scales, so we standardise each
    arm's top-score distribution first, then (hard) pick the higher z, and (soft)
    blend with a softmax over the two z-scores.
    """
    c_top = np.array([f.chunk_top for f in facts])
    s_top = np.array([f.sec_top for f in facts])
    cz = zscore(s_top, s_top.mean(), s_top.std()) - zscore(c_top, c_top.mean(), c_top.std())
    hard = _select_ndcg(facts, cz > 0)

    # Soft: per-query section weight = sigmoid(zgap); fuse with that weight.
    w = 1.0 / (1.0 + np.exp(-cz))
    ranked = {
        f.qid: weighted_fuse_query(arms.chunk[f.qid], arms.section[f.qid], float(wi))
        for f, wi in zip(facts, w, strict=True)
    }
    soft = mean_ndcg(ranked, arms.qrels, arms.qids)
    return hard, soft


def best_global_weight(arms: ArmRuns, grid: int = 21) -> Tuple[float, float]:
    """Rung 1: the single fixed blend weight that maximises mean nDCG."""
    best_w, best = 0.0, -1.0
    for w in np.linspace(0.0, 1.0, grid):
        ranked = {q: weighted_fuse_query(arms.chunk[q], arms.section[q], float(w)) for q in arms.qids}
        score = mean_ndcg(ranked, arms.qrels, arms.qids)
        if score > best:
            best_w, best = float(w), score
    return best_w, best


def weight_sweep(arms: ArmRuns, weights: Sequence[float]) -> Dict[float, float]:
    """Mean nDCG at each fixed coarse-arm weight -- the table for picking a default."""
    out: Dict[float, float] = {}
    for w in weights:
        ranked = {q: weighted_fuse_query(arms.chunk[q], arms.section[q], float(w)) for q in arms.qids}
        out[float(w)] = mean_ndcg(ranked, arms.qrels, arms.qids)
    return out


def _folds(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def single_feature_selector_cv(facts: List[QueryFacts], X: np.ndarray, names: List[str], k: int = 5):
    """Rung 3a: per feature, learn a threshold on train, route on test.

    Reports each feature's cross-validated selection nDCG, isolating which single
    query-time signal best predicts 'route to section'.
    """
    y_sec = np.array([f.sec_ndcg for f in facts])
    y_chunk = np.array([f.chunk_ndcg for f in facts])
    folds = _folds(len(facts), k)
    out: Dict[str, float] = {}
    for j, name in enumerate(names):
        feat = X[:, j]
        test_scores: List[float] = []
        for fold in folds:
            train = np.setdiff1d(np.arange(len(facts)), fold)
            # Best threshold + direction on train: choose section when feat >(or <) t.
            best_t, best_dir, best_tr = 0.0, 1, -1.0
            cand = np.unique(feat[train])
            for t in cand:
                for direction in (1, -1):
                    pick = (feat[train] * direction) > (t * direction)
                    tr = np.mean(np.where(pick, y_sec[train], y_chunk[train]))
                    if tr > best_tr:
                        best_t, best_dir, best_tr = float(t), direction, float(tr)
            pick_test = (feat[fold] * best_dir) > (best_t * best_dir)
            test_scores.append(float(np.mean(np.where(pick_test, y_sec[fold], y_chunk[fold]))))
        out[name] = float(np.mean(test_scores))
    return out


def _fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.3) -> np.ndarray:
    """Tiny L2-regularised logistic regression (standardised X, bias appended)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + 0.01 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w


def logreg_selector_cv(facts: List[QueryFacts], X: np.ndarray, k: int = 5) -> float:
    """Rung 3b: multivariate logistic router, cross-validated selection nDCG."""
    y_sec = np.array([f.sec_ndcg for f in facts])
    y_chunk = np.array([f.chunk_ndcg for f in facts])
    label = (y_sec > y_chunk).astype(np.float64)  # 1 = section strictly better
    folds = _folds(len(facts), k)
    test_scores: List[float] = []
    for fold in folds:
        train = np.setdiff1d(np.arange(len(facts)), fold)
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1.0
        w = _fit_logreg((X[train] - mu) / sd, label[train])
        Xt = np.hstack([(X[fold] - mu) / sd, np.ones((len(fold), 1))])
        pick = (Xt @ w) > 0.0
        test_scores.append(float(np.mean(np.where(pick, y_sec[fold], y_chunk[fold]))))
    return float(np.mean(test_scores))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyse_target(arms: ArmRuns) -> None:
    facts = collect_facts(arms)
    X, names = feature_matrix(facts)
    n = len(facts)

    chunk_ndcg = np.array([f.chunk_ndcg for f in facts])
    sec_ndcg = np.array([f.sec_ndcg for f in facts])
    best_single = np.maximum(chunk_ndcg, sec_ndcg)  # per-query oracle over the 2 arms

    equal_ranked = {f.qid: weighted_fuse_query(arms.chunk[f.qid], arms.section[f.qid], 0.5) for f in facts}
    equal_ndcg_q = np.array([ndcg_at_k(equal_ranked[f.qid], arms.qrels[f.qid], PRIMARY_K) for f in facts])
    regret = best_single - equal_ndcg_q  # what equal-weight fusion gives up per query

    # Partition by which arm is the per-query winner.
    sec_wins = sec_ndcg > chunk_ndcg
    chunk_wins = chunk_ndcg > sec_ndcg
    ties = ~sec_wins & ~chunk_wins

    print(f"\n===== target: {arms.target.upper()}  |  coarse arm: {arms.coarse_name}  ({n} queries) =====")
    print(
        f"arms (mean nDCG@10):  chunk={chunk_ndcg.mean():.4f}  "
        f"{arms.coarse_name}={sec_ndcg.mean():.4f}  |  oracle(2-arm)={best_single.mean():.4f}"
    )
    print(
        f"per-query winner:  chunk={chunk_wins.sum()} ({chunk_wins.mean():.0%})  "
        f"section={sec_wins.sum()} ({sec_wins.mean():.0%})  tie={ties.sum()} ({ties.mean():.0%})"
    )
    print(
        f"equal-weight fusion regret vs best single arm:  mean={regret.mean():.4f}  "
        f"on section-wins={regret[sec_wins].mean() if sec_wins.any() else 0:.4f}  "
        f"on chunk-wins={regret[chunk_wins].mean() if chunk_wins.any() else 0:.4f}"
    )

    # What does regret track? (outcome gap is diagnostic; the rest are query-time)
    outcome_gap = np.abs(sec_ndcg - chunk_ndcg)
    disagree = np.array([1 - f.agree for f in facts], dtype=np.float64)
    qlen = np.array([f.qlen for f in facts], dtype=np.float64)
    ztop_gap = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() or 1) - (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() or 1)
    print("regret correlates with:")
    print(f"    |outcome gap|       r={_corr(regret, outcome_gap):+.3f}   (why the miss exists)")
    print(f"    arm disagreement    r={_corr(regret, disagree):+.3f}")
    print(f"    query length        r={_corr(regret, qlen):+.3f}")
    print(f"    calibrated top gap  r={_corr(regret, np.abs(ztop_gap)):+.3f}   (what a selector can see)")

    # Selector bake-off.
    always_chunk = chunk_ndcg.mean()
    always_sec = sec_ndcg.mean()
    equal = equal_ndcg_q.mean()
    w_star, global_best = best_global_weight(arms)
    conf_hard, conf_soft = confidence_selector(arms, facts)
    per_feature = single_feature_selector_cv(facts, X, names)
    best_feat = max(per_feature, key=per_feature.get)
    lr_cv = logreg_selector_cv(facts, X)
    oracle = best_single.mean()

    def line(label: str, val: float) -> str:
        return f"    {label:<34}{val:.4f}   ({val - equal:+.4f} vs equal, {val - oracle:+.4f} vs oracle)"

    print("selector bake-off (nDCG@10):")
    print(line("always chunk", always_chunk))
    print(line(f"always {arms.coarse_name}", always_sec))
    print(line("equal-weight fusion (0.5)", equal))
    print(line(f"rung1: best global blend w={w_star:.2f}", global_best))
    print(line("rung2: confidence hard-pick", conf_hard))
    print(line("rung2: confidence soft-blend", conf_soft))
    print(line(f"rung3a: 1-feature [{best_feat}] cv", per_feature[best_feat]))
    print(line("rung3b: logistic router cv", lr_cv))
    print(f"    {'oracle (2-arm ceiling)':<34}{oracle:.4f}")
    print("    per-feature 1-selector cv nDCG:  " + "  ".join(f"{k}={v:.4f}" for k, v in per_feature.items()))

    # Fixed-weight sweep -- the table for choosing a defensible default weight
    # (coarse-arm weight; 0.0 = chunk-only, 1.0 = coarse-only).
    sweep = weight_sweep(arms, [0.0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0])
    print("weight sweep (coarse-arm weight -> nDCG@10):  " + "  ".join(f"{w:.2f}={v:.4f}" for w, v in sweep.items()))


def load_bench(args: argparse.Namespace) -> SyntheticBenchmark:
    if args.dataset == "qasper":
        from benchmarks.qasper_data import load_qasper

        return load_qasper(split=args.split, max_papers=args.max_papers, seed=args.seed)
    source = beir_data.load(args.dataset)
    return build_synthetic_benchmark(
        source,
        sections_per_doc=args.sections,
        passages_per_section=args.passages,
        seed=args.seed,
        max_queries=args.max_queries,
        mode=args.mode,
        min_section_gold=args.min_section_gold,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion-miss anatomy + selector bake-off (cache-only).")
    p.add_argument("--dataset", default="fiqa", choices=("fiqa", "nfcorpus", "scifact", "qasper"))
    p.add_argument("--split", default="dev")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sections", type=int, default=3)
    p.add_argument("--passages", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--mode", default="section", choices=("point", "section"))
    p.add_argument("--min-section-gold", type=int, default=2)
    p.add_argument(
        "--summary",
        action="store_true",
        help="Also fold in the summary-section coarse arm (needs its summaries+embeddings cached).",
    )
    p.add_argument("--summary-model", default="gpt-5.4-nano")
    p.add_argument("--directive", default="retrieval")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bench = load_bench(args)
    print(f"{bench!r}")

    summarizer = None
    if args.summary:
        from benchmarks.summarize import CachedSummarizer

        summarizer = CachedSummarizer(args.summary_model, directive=args.directive)

    encoder = CachedEncoder(args.provider, args.model)
    built = build_vectors(bench, encoder, summarizer=summarizer)
    if encoder.n_embedded:
        print(
            f"WARNING: {encoder.n_embedded} spans were NOT cached and were freshly embedded "
            f"(params differ from the cached run?).",
            file=sys.stderr,
        )
    if summarizer is not None and summarizer.n_called:
        print(f"WARNING: {summarizer.n_called} summaries were NOT cached and were freshly generated.", file=sys.stderr)

    targets = ("doc", "section") if bench.section_qrels else ("doc",)
    for target in targets:
        for arms in build_arm_runs(bench, built, target):
            analyse_target(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
