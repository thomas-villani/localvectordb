"""Relevance metrics for retrieval evaluation.

These follow ``trec_eval`` conventions, which is what BEIR reports via
``pytrec_eval``. Two details matter if you compare numbers to a published table:

* ``ndcg_cut_k`` uses **linear** gain (``rel / log2(rank + 1)``), not the
  ``2**rel - 1`` variant. For binary qrels (SciFact) the two agree; for graded
  qrels (NFCorpus, ``rel in {0, 1, 2}``) they do not.
* The ideal DCG is computed over *all* judged-relevant documents for the query,
  sorted by relevance descending, then truncated at ``k``. A query with more
  than ``k`` relevant documents therefore cannot reach ``nDCG@k == 1.0`` unless
  the top ``k`` are all maximally relevant.

Unjudged documents count as relevance 0, and a query whose ranking is empty
scores 0 rather than being skipped -- retrieving nothing is a failure, not an
absence of evidence.

``benchmarks/../tests/test_retrieval_metrics.py`` pins these against
hand-computed values. Do not "simplify" this module without rerunning it: every
T1 decision is read off these numbers, so a silent metric bug invalidates the
conclusion rather than merely the report.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

# query id -> {document id -> relevance grade}. Grades <= 0 mean "not relevant".
Qrels = Mapping[str, Mapping[str, int]]
# query id -> ranked document ids, best first.
Run = Mapping[str, Sequence[str]]


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` gains, linear gain."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """nDCG@k for one query. ``relevance`` maps document id -> grade."""
    actual = dcg([float(relevance.get(doc_id, 0)) for doc_id in ranked[:k]], k)
    ideal = dcg(sorted((float(r) for r in relevance.values() if r > 0), reverse=True), k)
    return 0.0 if ideal == 0.0 else actual / ideal


def recall_at_k(ranked: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Fraction of this query's relevant documents that appear in the top ``k``."""
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def evaluate(run: Run, qrels: Qrels, k_values: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Mean nDCG@k and recall@k across every query in ``qrels``.

    Queries are driven by ``qrels``, not by ``run``: a query the system failed to
    answer must drag the mean down, so it is scored as an empty ranking.
    """
    if not qrels:
        raise ValueError("qrels is empty; nothing to evaluate against")

    max_k = max(k_values)
    scores: Dict[str, List[float]] = {}
    for query_id, relevance in qrels.items():
        ranked = list(run.get(query_id, ()))
        for k in k_values:
            scores.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, relevance, k))
        scores.setdefault(f"ndcg@{max_k}", []).append(ndcg_at_k(ranked, relevance, max_k))

    return {name: sum(values) / len(values) for name, values in scores.items()}
