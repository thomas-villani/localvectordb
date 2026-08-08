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

    def test_best_and_average(self):
        assert self._score_for("best").score == pytest.approx(0.9)
        assert self._score_for("average").score == pytest.approx(0.5)

    def test_best_ge_average(self):
        best = self._score_for("best").score
        avg = self._score_for("average").score
        assert best >= avg

    def test_frequency_boost_records_metadata(self):
        result = self._score_for("frequency_boost")
        scoring = result.metadata["_scoring"]
        assert scoring["_aggregation_method"] == "frequency_boost"
        assert "effective_chunk_count" in scoring
        assert "frequency_multiplier" in scoring

    def test_unknown_method_raises(self):
        # T1.6 removed the eight heuristic methods; a removed/unknown name must raise
        # rather than silently fall back to 'best'.
        with pytest.raises(ValueError, match="Unknown document_scoring_method"):
            self._score_for("weighted_average")

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


class TestAutoDocumentScoring:
    """``document_scoring_method="auto"`` picks the aggregator from ``search_type``.

    The aggregator that wins depends on the scale the chunk scores arrive on, not
    on the corpus: hybrid/keyword min-max normalise within the query's own pool,
    where ``frequency_boost``'s count multiplier is bounded and helps; vector
    passes a raw bounded similarity through, where the same multiplier mostly
    rewards owning more chunks. Measured on qasper and NQ -- see
    ``DocumentScoringMethod`` for the numbers.
    """

    def test_auto_resolves_by_search_type(self):
        from localvectordb.database._search import _resolve_document_scoring

        assert _resolve_document_scoring("auto", "vector") == "best"
        assert _resolve_document_scoring("auto", "hybrid") == "frequency_boost"
        # Keyword scores are not min-max normalised either, but the vector result
        # was measured and this one was not; it keeps today's behaviour rather
        # than generalising from an untested leg.
        assert _resolve_document_scoring("auto", "keyword") == "frequency_boost"

    @pytest.mark.parametrize("explicit", ["best", "average", "frequency_boost"])
    @pytest.mark.parametrize("search_type", ["vector", "hybrid", "keyword"])
    def test_explicit_choice_is_never_overridden(self, explicit, search_type):
        from localvectordb.database._search import _resolve_document_scoring

        assert _resolve_document_scoring(explicit, search_type) == explicit

    def test_unresolved_auto_reaching_the_scorer_raises(self):
        # Silently treating it as frequency_boost would give the vector path the
        # wrong aggregator and look exactly like a retrieval regression.
        doc_groups = {"d": [_chunk("d", 0.5)]}
        with pytest.raises(ValueError, match="reached the scorer unresolved"):
            LocalVectorDB._compute_document_scores("auto", {}, doc_groups, {"d": "d"}, {"d": {}})


class TestAutoDocumentScoringEndToEnd:
    """The resolver being right does not prove the entry points call it.

    Each public entry point resolves ``auto`` itself, and a missed one surfaces
    as the "reached the scorer unresolved" ValueError rather than as a wrong
    number, so these assert on the recorded ``_aggregation_method`` tag.
    """

    @staticmethod
    def _db(tmp_path):
        from localvectordb.database import LocalVectorDB

        db = LocalVectorDB(
            name="auto_scoring",
            base_path=str(tmp_path),
            embedding_provider="mock",
            embedding_model="mock",
            chunk_size=50,
            chunk_overlap=0,
            enable_fts=True,
        )
        # A many-chunk document beside a one-chunk document: the only shape where
        # a count-sensitive aggregator and a plain max can differ at all.
        db.upsert(
            ["alpha beta gamma delta epsilon zeta eta theta. " * 12, "alpha beta gamma."],
            ids=["many", "one"],
        )
        return db

    @staticmethod
    def _method_of(results):
        return {r.metadata.get("_scoring", {}).get("_aggregation_method") for r in results}

    def test_vector_default_is_best(self, tmp_path):
        db = self._db(tmp_path)
        assert self._method_of(db.query("alpha beta", search_type="vector", k=5)) == {"best"}

    def test_hybrid_default_is_frequency_boost(self, tmp_path):
        db = self._db(tmp_path)
        assert self._method_of(db.query("alpha beta", search_type="hybrid", k=5)) == {"frequency_boost"}

    def test_default_query_matches_explicit_equivalent(self, tmp_path):
        db = self._db(tmp_path)

        def scores(**kw):
            return {r.id: round(r.score, 9) for r in db.query("alpha beta", k=5, **kw)}

        assert scores(search_type="vector") == scores(search_type="vector", document_scoring_method="best")
        assert scores(search_type="hybrid") == scores(search_type="hybrid", document_scoring_method="frequency_boost")

    def test_vector_default_actually_changed(self, tmp_path):
        # Guards the whole point: if "auto" silently kept frequency_boost on the
        # vector path, every assertion above would still pass on a corpus where
        # the two happen to agree. This one fails unless they really differ.
        db = self._db(tmp_path)

        def scores(**kw):
            return {r.id: round(r.score, 9) for r in db.query("alpha beta", search_type="vector", k=5, **kw)}

        assert scores() != scores(document_scoring_method="frequency_boost")

    def test_multi_column_resolves_auto(self, tmp_path):
        # query_multi_column scores merged results itself, so it needs its own
        # resolution rather than inheriting query()'s.
        db = self._db(tmp_path)
        assert self._method_of(db.query_multi_column("alpha beta", search_type="vector", k=5)) == {"best"}
