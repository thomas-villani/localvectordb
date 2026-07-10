"""Regression net for T1.1: hybrid fusion must normalize before it blends.

Hybrid search fused its two legs with an un-normalized weighted sum::

    final = vector_weight * vector_similarity + (1 - vector_weight) * keyword_similarity

The legs are on incompatible scales. The vector leg is a bounded similarity spread
across tenths. The keyword leg went through ``_fts_rank_to_similarity``, which is
``1 - min(1, exp(bm25))`` -- an *absolute* transform of raw BM25, never normalized per
query. Measured on BEIR SciFact, the top-20 BM25 scores for a real query span
``-16.6 ... -12.4`` and that transform maps all of them into ``[0.999996, 0.99999994]``,
a band 2.3e-05 wide. Past about ``-36`` it saturates to *exactly* 1.0 and the
distinction is gone from the float entirely.

So ``vector_weight`` was not a blend. It was a *presence* weight: the keyword term
contributed a near-constant 1.0 to every chunk the keyword leg had retrieved at all.
Pure ``search_type="keyword"`` was unaffected -- SQLite orders by raw ``bm25()`` before
the transform ever runs -- which is exactly why this hid for so long.

The fix is relative-score fusion: min-max each leg within the query's own candidate
pool, then blend. Measured against the un-normalized sum on SciFact at the library
default (``vector_weight=0.7``, ``frequency_boost``), nDCG@10 went 0.6343 -> 0.6940.
Reciprocal Rank Fusion was measured too and scored 0.6439; see
``benchmarks/RETRIEVAL_BASELINE.md``.

The load-bearing invariant these tests pin: **fusion consumes raw BM25, never the
saturating similarity.** ``_keyword_chunk_hits`` returns the raw ranks alongside the
``QueryResult``s for precisely that reason.
"""

import asyncio
import math
import shutil
import tempfile

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.database._search import _minmax_normalize, _relative_score_fusion


# ---------------------------------------------------------------------------
# _minmax_normalize
# ---------------------------------------------------------------------------
class TestMinMaxNormalize:
    def test_empty(self):
        assert _minmax_normalize([]) == []

    def test_spreads_to_the_unit_interval(self):
        assert _minmax_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]

    def test_single_value_is_the_best_of_what_was_retrieved(self):
        assert _minmax_normalize([0.01]) == [1.0]

    def test_identical_values_do_not_divide_by_zero(self):
        assert _minmax_normalize([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]

    def test_negative_values(self):
        assert _minmax_normalize([-9.0, -5.0, -1.0]) == [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# _relative_score_fusion
# ---------------------------------------------------------------------------
class TestRelativeScoreFusion:
    def test_bm25_is_negative_is_better(self):
        """A more negative BM25 is a *better* match and must normalize higher."""
        fused = _relative_score_fusion({}, {"strong": -9.0, "weak": -1.0}, vector_weight=0.0)
        assert fused["strong"] > fused["weak"]
        assert fused["strong"] == pytest.approx(1.0)
        assert fused["weak"] == pytest.approx(0.0)

    def test_vector_weight_blends_rather_than_flags_presence(self):
        """The defect: with a saturated keyword leg, `vector_weight` only asked
        'did the keyword leg retrieve this at all?'. It must now grade."""
        vector = {"a": 1.0, "b": 0.0}
        keyword = {"a": -1.0, "b": -9.0}  # b is the better keyword match
        vector_heavy = _relative_score_fusion(vector, keyword, vector_weight=0.9)
        keyword_heavy = _relative_score_fusion(vector, keyword, vector_weight=0.1)
        assert vector_heavy["a"] > vector_heavy["b"]
        assert keyword_heavy["b"] > keyword_heavy["a"]

    def test_saturating_similarity_would_have_erased_this_distinction(self):
        """The heart of the bug, pinned.

        BM25 -40 and -50 are very different matches. Both map to *exactly* 1.0 under
        `1 - min(1, exp(bm25))` in float64, so a fusion fed the transformed value
        cannot tell them apart. Fed the raw rank, it can.
        """
        saturating = lambda rank: 1.0 - min(1.0, math.exp(rank))  # noqa: E731
        assert saturating(-40.0) == saturating(-50.0) == 1.0  # the information is gone

        fused = _relative_score_fusion({}, {"better": -50.0, "worse": -40.0}, vector_weight=0.0)
        assert fused["better"] > fused["worse"]

    def test_both_legs_contribute_additively(self):
        """Two chunks tied on the vector leg; only one is also a keyword hit."""
        vector = {"top": 1.0, "both": 0.6, "vector_only": 0.6, "tail": 0.2}
        keyword = {"x": -9.0, "both": -5.0, "z": -1.0}
        fused = _relative_score_fusion(vector, keyword, vector_weight=0.5)
        # `both` and `vector_only` normalize identically on the vector leg (0.5), so the
        # keyword leg is the only thing separating them.
        assert fused["both"] > fused["vector_only"]

    def test_worst_of_a_leg_is_indistinguishable_from_absent(self):
        """A caveat of relative-score fusion, pinned so it is not mistaken for a bug.

        `tail` is retrieved by the vector leg but ranks last, so it normalizes to 0.0 --
        exactly what a chunk the leg never retrieved would contribute.
        """
        fused = _relative_score_fusion({"top": 1.0, "tail": 0.2}, {"top": -9.0}, vector_weight=0.5)
        assert fused["tail"] == pytest.approx(0.0)

    def test_missing_leg_contributes_zero(self):
        fused = _relative_score_fusion({"a": 0.9}, {}, vector_weight=0.7)
        assert fused["a"] == pytest.approx(0.7)  # 0.7 * 1.0 + 0.3 * 0.0

    def test_scores_stay_within_the_unit_interval(self):
        """RRF would have broken the `score_threshold` contract; this must not."""
        fused = _relative_score_fusion({"a": 0.9, "b": 0.1}, {"a": -9.0, "b": -1.0}, vector_weight=0.7)
        assert all(0.0 <= score <= 1.0 for score in fused.values())

    def test_vector_leg_first_insertion_order(self):
        """A stable sort downstream breaks ties toward the vector ranking, as before."""
        fused = _relative_score_fusion({"v1": 0.5, "v2": 0.4}, {"k1": -1.0, "v1": -2.0}, vector_weight=0.5)
        assert list(fused) == ["v1", "v2", "k1"]


# ---------------------------------------------------------------------------
# End to end, against a real database
# ---------------------------------------------------------------------------
_DOCS = [
    "Aspirin reduces cardiovascular risk in patients with prior myocardial infarction.",
    "The quick brown fox does not jump over anything at all in the morning.",
    "Statins lower cholesterol and are prescribed to reduce cardiovascular events.",
]


@pytest.fixture
def fusion_db():
    """Real LocalVectorDB, real FTS5, mock embeddings.

    MockEmbeddings seeds np.random on a hash of the text, so the vector leg's ranking
    is arbitrary here. These tests therefore assert on fusion *mechanics* -- score
    ranges, the effect of vector_weight, sync/async agreement -- and never on which
    document is semantically most relevant. Only benchmarks/eval_retrieval.py can
    settle that.
    """
    temp_dir = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="hybrid_fusion",
        base_path=temp_dir,
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    db.upsert(documents=list(_DOCS), ids=[f"doc{i}" for i in range(len(_DOCS))])
    yield db
    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestHybridEndToEnd:
    def test_hybrid_returns_results_with_scores_in_the_unit_interval(self, fusion_db):
        results = fusion_db.query("cardiovascular risk", search_type="hybrid")
        assert results
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_vector_weight_zero_lets_bm25_alone_decide(self, fusion_db):
        results = fusion_db.query("cardiovascular risk", search_type="hybrid", vector_weight=0.0)
        assert [r.id for r in results][:1] == ["doc0"]

    def test_vector_weight_changes_the_ranking(self, fusion_db):
        """It was inert: every vector_weight produced an identical hybrid ranking."""
        scores = {}
        for vw in (0.0, 1.0):
            results = fusion_db.query("cardiovascular risk", search_type="hybrid", vector_weight=vw)
            scores[vw] = [(r.id, round(r.score, 6)) for r in results]
        assert scores[0.0] != scores[1.0]

    def test_cursor_path_also_fuses(self, fusion_db):
        """The cursor path has its own fusion entry point (`_merge_hybrid_candidates`).

        It cannot be compared score-for-score against `query()`: the two use different
        candidate pool sizes, and relative-score fusion is pool-dependent. What must hold
        is that the cursor fuses at all -- with `vector_weight=0.0`, BM25 alone decides.
        """
        with fusion_db.query_cursor(
            "cardiovascular risk", search_type="hybrid", return_type="chunks", k=3, vector_weight=0.0
        ) as cursor:
            streamed = cursor.fetch_all()
        assert streamed
        assert all(0.0 <= r.score <= 1.0 for r in streamed)
        assert streamed[0].document_id == "doc0"

    @pytest.mark.asyncio
    async def test_async_hybrid_agrees_with_sync(self, fusion_db):
        # `query()` calls embed_sync, which refuses to run inside a running event loop.
        sync = await asyncio.to_thread(
            fusion_db.query, "cardiovascular risk", search_type="hybrid", return_type="chunks", k=3
        )
        async_results = await fusion_db.query_async(
            "cardiovascular risk", search_type="hybrid", return_type="chunks", k=3
        )
        assert [r.id for r in sync] == [r.id for r in async_results]
        assert [r.score for r in sync] == pytest.approx([r.score for r in async_results])
