"""Direct unit tests for the search scoring statics.

These exercise the pure aggregation/combination math in ``database/_search.py``
without spinning up a database, which the integration query tests only asserted
loosely (``len > 0`` / ``0 <= score <= 1``). They pin the hybrid weighting formula
and the chunk-to-document aggregation methods.
"""

import pytest

from localvectordb.core import QueryResult
from localvectordb.database import LocalVectorDB


def _chunk(doc_id, score, idx=0):
    return QueryResult(id=f"{doc_id}:{idx}", score=score, type="chunk", content=f"chunk {idx}")


def _hit(doc_id, score):
    return QueryResult(id=doc_id, score=score, type="document", content=f"content {doc_id}")


class TestCombineSearchResults:
    """Hybrid score = vector_weight * vec + (1 - vector_weight) * kw."""

    def test_weighted_combination_formula(self):
        vector = [_hit("a", 1.0)]
        keyword = [_hit("a", 0.0)]
        out = LocalVectorDB._combine_search_results(vector, keyword, vector_weight=0.7, k=10, score_threshold=0.0)
        assert out[0].score == pytest.approx(0.7)

    def test_weight_sweep_reorders_results(self):
        # Doc A is a pure vector hit; doc B is a pure keyword hit.
        def run(vw):
            vector = [_hit("a", 1.0), _hit("b", 0.0)]
            keyword = [_hit("a", 0.0), _hit("b", 1.0)]
            out = LocalVectorDB._combine_search_results(vector, keyword, vector_weight=vw, k=10, score_threshold=0.0)
            return [r.id for r in out]

        assert run(0.9)[0] == "a"  # vector-dominant
        assert run(0.1)[0] == "b"  # keyword-dominant

    def test_union_of_ids_with_missing_side(self):
        # Doc only present in keyword results gets vector_score 0.
        vector = [_hit("a", 0.8)]
        keyword = [_hit("b", 0.5)]
        out = LocalVectorDB._combine_search_results(vector, keyword, vector_weight=0.5, k=10, score_threshold=0.0)
        by_id = {r.id: r.score for r in out}
        assert by_id["a"] == pytest.approx(0.4)  # 0.5 * 0.8
        assert by_id["b"] == pytest.approx(0.25)  # 0.5 * 0.5

    def test_score_threshold_filters(self):
        vector = [_hit("a", 1.0), _hit("b", 0.2)]
        keyword = [_hit("a", 0.0), _hit("b", 0.0)]
        out = LocalVectorDB._combine_search_results(vector, keyword, vector_weight=0.5, k=10, score_threshold=0.3)
        ids = {r.id for r in out}
        assert ids == {"a"}  # b's 0.1 combined score is below threshold

    def test_k_limits_results(self):
        vector = [_hit(x, s) for x, s in [("a", 0.9), ("b", 0.8), ("c", 0.7)]]
        out = LocalVectorDB._combine_search_results(vector, [], vector_weight=1.0, k=2, score_threshold=0.0)
        assert [r.id for r in out] == ["a", "b"]


class TestComputeDocumentScores:
    """Chunk-score aggregation into a single per-document score."""

    SCORES = [0.9, 0.5, 0.1]

    def _score_for(self, method):
        doc_groups = {"doc": [_chunk("doc", s, i) for i, s in enumerate(self.SCORES)]}
        results = LocalVectorDB._compute_document_scores(
            method,
            {},
            doc_groups,
            {"doc": "content"},
            {"doc": {}},
        )
        assert len(results) == 1
        return results[0]

    def test_best_worst_average(self):
        assert self._score_for("best").score == pytest.approx(0.9)
        assert self._score_for("worst").score == pytest.approx(0.1)
        assert self._score_for("average").score == pytest.approx(0.5)

    def test_best_ge_average_ge_worst(self):
        best = self._score_for("best").score
        avg = self._score_for("average").score
        worst = self._score_for("worst").score
        assert best >= avg >= worst

    def test_weighted_average_records_weights(self):
        result = self._score_for("weighted_average")
        # Fix #2 regression guard: the key is spelled "weights" (was "primary_wieght"
        # elsewhere); weighted_average records normalized weights in _scoring.
        assert "weights" in result.metadata["_scoring"]
        assert result.metadata["_scoring"]["_aggregation_method"] == "weighted_average"

    def test_results_sorted_by_score_desc(self):
        doc_groups = {
            "low": [_chunk("low", 0.2)],
            "high": [_chunk("high", 0.95)],
            "mid": [_chunk("mid", 0.6)],
        }
        results = LocalVectorDB._compute_document_scores(
            "best", {}, doc_groups, {k: k for k in doc_groups}, {k: {} for k in doc_groups}
        )
        assert [r.id for r in results] == ["high", "mid", "low"]

    def test_documents_without_content_are_skipped(self):
        doc_groups = {"ghost": [_chunk("ghost", 0.9)]}
        results = LocalVectorDB._compute_document_scores("best", {}, doc_groups, {}, {"ghost": {}})
        assert results == []
