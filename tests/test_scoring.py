"""Direct unit tests for the search scoring statics.

These exercise the pure aggregation/combination math in ``database/_search.py``
without spinning up a database, which the integration query tests only asserted
loosely (``len > 0`` / ``0 <= score <= 1``). They pin the hybrid fusion rule and
the chunk-to-document aggregation methods.
"""

import pytest

from localvectordb.core import QueryResult
from localvectordb.database import LocalVectorDB


def _chunk(doc_id, score, idx=0):
    return QueryResult(id=f"{doc_id}:{idx}", score=score, type="chunk", content=f"chunk {idx}")


def _hit(doc_id, score):
    return QueryResult(id=doc_id, score=score, type="document", content=f"content {doc_id}")


class TestCombineSearchResults:
    """Hybrid fusion: min-max each leg within the query, then blend by ``vector_weight``.

    ``keyword_ranks`` are *raw* BM25 (negative; more negative is better), never the
    saturating ``_fts_rank_to_similarity`` output. See ``tests/test_hybrid_fusion.py``.
    """

    def test_each_leg_is_normalized_within_the_query(self):
        # Vector spans [0.2, 0.6]; BM25 spans [-9, -1]. Neither range reaches the other's.
        vector = [_hit("a", 0.6), _hit("b", 0.2)]
        keyword = [_hit("a", 0.0), _hit("b", 0.0)]
        ranks = {"a": -1.0, "b": -9.0}
        out = LocalVectorDB._combine_search_results(
            vector, keyword, ranks, vector_weight=0.5, k=10, score_threshold=0.0
        )
        by_id = {r.id: r.score for r in out}
        # a: best vector (1.0), worst keyword (0.0). b: the mirror image. A tie, at 0.5.
        assert by_id["a"] == pytest.approx(0.5)
        assert by_id["b"] == pytest.approx(0.5)

    def test_weight_sweep_reorders_results(self):
        # Doc A is the stronger vector hit; doc B is the stronger keyword hit.
        def run(vw):
            vector = [_hit("a", 1.0), _hit("b", 0.0)]
            keyword = [_hit("a", 0.0), _hit("b", 0.0)]
            ranks = {"a": -1.0, "b": -9.0}
            out = LocalVectorDB._combine_search_results(
                vector, keyword, ranks, vector_weight=vw, k=10, score_threshold=0.0
            )
            return [r.id for r in out]

        assert run(0.9)[0] == "a"  # vector-dominant
        assert run(0.1)[0] == "b"  # keyword-dominant

    def test_a_lone_candidate_normalizes_to_one(self):
        # A single-member pool has nothing to rank against, so it is the best of what
        # was retrieved. Guards the degenerate branch of _minmax_normalize.
        out = LocalVectorDB._combine_search_results(
            [_hit("a", 0.01)], [_hit("a", 0.0)], {"a": -0.5}, vector_weight=0.7, k=10, score_threshold=0.0
        )
        assert out[0].score == pytest.approx(1.0)

    def test_union_of_ids_with_missing_side(self):
        # A doc retrieved by only one leg scores 0 on the other.
        vector = [_hit("a", 0.8), _hit("c", 0.1)]
        keyword = [_hit("b", 0.0), _hit("d", 0.0)]
        ranks = {"b": -9.0, "d": -1.0}
        out = LocalVectorDB._combine_search_results(
            vector, keyword, ranks, vector_weight=0.5, k=10, score_threshold=0.0
        )
        by_id = {r.id: r.score for r in out}
        assert by_id["a"] == pytest.approx(0.5)  # best vector, absent from keyword
        assert by_id["b"] == pytest.approx(0.5)  # best keyword, absent from vector
        assert by_id["c"] == pytest.approx(0.0)  # worst vector, absent from keyword
        assert by_id["d"] == pytest.approx(0.0)

    def test_score_threshold_filters(self):
        vector = [_hit("a", 1.0), _hit("b", 0.2)]
        keyword = [_hit("a", 0.0), _hit("b", 0.0)]
        ranks = {"a": -9.0, "b": -1.0}
        out = LocalVectorDB._combine_search_results(
            vector, keyword, ranks, vector_weight=0.5, k=10, score_threshold=0.3
        )
        # a normalizes to 1.0 on both legs -> 1.0; b to 0.0 on both -> 0.0.
        assert {r.id for r in out} == {"a"}

    def test_k_limits_results(self):
        vector = [_hit(x, s) for x, s in [("a", 0.9), ("b", 0.8), ("c", 0.7)]]
        out = LocalVectorDB._combine_search_results(vector, [], {}, vector_weight=1.0, k=2, score_threshold=0.0)
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
