"""Pin the retrieval metrics against hand-computed values.

`benchmarks/metrics.py` is the instrument every T1 retrieval decision is read
off. A silent bug there would not merely produce a wrong report -- it would
produce a confidently wrong *conclusion* about whether a fusion change helped.
So the expected values here are worked out by hand, from the definition, rather
than captured from a previous run.
"""

import math

import pytest

from benchmarks.metrics import dcg, evaluate, ndcg_at_k, recall_at_k

pytestmark = pytest.mark.unit

LOG2_3 = math.log2(3)  # the rank-2 discount


class TestDCG:
    def test_rank_one_is_undiscounted(self):
        assert dcg([1.0], 10) == 1.0

    def test_rank_two_is_discounted_by_log2_of_3(self):
        # 1/log2(1+1) + 1/log2(2+1)
        assert dcg([1.0, 1.0], 10) == pytest.approx(1.0 + 1.0 / LOG2_3)

    def test_truncates_at_k(self):
        assert dcg([1.0, 1.0, 1.0], 1) == 1.0

    def test_empty(self):
        assert dcg([], 10) == 0.0


class TestNDCG:
    def test_single_relevant_doc_at_rank_one(self):
        assert ndcg_at_k(["a", "b", "c"], {"a": 1}, 10) == 1.0

    def test_single_relevant_doc_at_rank_two(self):
        # DCG = 1/log2(3); IDCG = 1.0
        assert ndcg_at_k(["b", "a"], {"a": 1}, 10) == pytest.approx(1.0 / LOG2_3)

    def test_unjudged_documents_score_zero(self):
        assert ndcg_at_k(["x", "y"], {"a": 1}, 10) == 0.0

    def test_empty_ranking_scores_zero(self):
        assert ndcg_at_k([], {"a": 1}, 10) == 0.0

    def test_no_relevant_documents_scores_zero(self):
        assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0

    def test_graded_relevance_uses_linear_gain(self):
        # trec_eval convention: gain == grade, NOT 2**grade - 1.
        # ranked ["b", "a"] with grades b=1, a=2.
        #   DCG  = 1/log2(2) + 2/log2(3)
        #   IDCG = 2/log2(2) + 1/log2(3)   (ideal order is a, b)
        expected = (1.0 + 2.0 / LOG2_3) / (2.0 + 1.0 / LOG2_3)
        assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(expected)
        # Guard against the exponential-gain variant, which would give a
        # different number here (and silently shift every NFCorpus result).
        exponential = (1.0 + 3.0 / LOG2_3) / (3.0 + 1.0 / LOG2_3)
        assert not math.isclose(expected, exponential, rel_tol=1e-6)

    def test_ideal_dcg_is_truncated_at_k(self):
        # 3 relevant docs but k=2: retrieving the best 2 must score a perfect 1.0.
        # If IDCG were computed over all 3 relevant docs this would be ~0.765.
        assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1, "c": 1}, 2) == pytest.approx(1.0)

    def test_perfect_graded_ranking_scores_one(self):
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == pytest.approx(1.0)


class TestRecall:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "y"], {"a": 1, "b": 1}, 10) == pytest.approx(0.5)

    def test_respects_the_cutoff(self):
        assert recall_at_k(["x", "a"], {"a": 1}, 1) == 0.0
        assert recall_at_k(["x", "a"], {"a": 1}, 2) == 1.0

    def test_zero_grade_is_not_relevant(self):
        # Denominator counts only grade > 0, so this is 0/0 -> 0.0, not a crash.
        assert recall_at_k(["a"], {"a": 0}, 10) == 0.0

    def test_graded_relevance_is_binarized(self):
        assert recall_at_k(["b"], {"a": 2, "b": 1}, 10) == pytest.approx(0.5)


class TestEvaluate:
    def test_averages_over_queries(self):
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        run = {"q1": ["a"], "q2": ["x", "b"]}
        scores = evaluate(run, qrels, k_values=(1, 2))
        assert scores["recall@1"] == pytest.approx(0.5)  # q1 hits, q2 misses
        assert scores["recall@2"] == pytest.approx(1.0)
        assert scores["ndcg@2"] == pytest.approx((1.0 + 1.0 / LOG2_3) / 2)

    def test_unanswered_query_counts_as_zero_not_skipped(self):
        # The mean is over qrels, not over run. Retrieving nothing for q2 must
        # halve recall, not silently vanish from the denominator.
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        assert evaluate({"q1": ["a"]}, qrels, k_values=(1,))["recall@1"] == pytest.approx(0.5)

    def test_reports_ndcg_at_the_largest_k(self):
        scores = evaluate({"q1": ["a"]}, {"q1": {"a": 1}}, k_values=(1, 5, 10))
        assert set(scores) == {"recall@1", "recall@5", "recall@10", "ndcg@10"}

    def test_empty_qrels_is_an_error(self):
        # Averaging over nothing would report 0.0 and look like a real result.
        with pytest.raises(ValueError, match="qrels is empty"):
            evaluate({}, {})
