"""The hybrid candidate pool: ``_hybrid_pool_size``.

WHY THIS IS ITS OWN FUNCTION. The width each leg fetches before fusion looks like
a cost knob and is not. With ``return_type="documents"`` the chunk->document
aggregator runs over the *whole* fused pool and only then truncates to ``k``, so
the pool is an argument to the scoring operator: "coverage-weight the children"
means something different over 40 candidates than over 400. Benchmarking found a
real interaction there -- widening the pool helps ``best`` and does nothing for
``frequency_boost`` -- so the width has to be nameable and swept against the real
query path rather than a re-implementation of it.

WHAT THESE TESTS PIN. Two things, and the second is the one that would rot
silently: the arithmetic, and the fact that BOTH hybrid paths (sync and async)
route through this one function. If a future edit re-inlines the expression at
one call site, the sweep would measure a path the product no longer takes.
"""

import shutil
import tempfile
from unittest.mock import patch

import pytest

from localvectordb.database import LocalVectorDB, _search


def _hybrid_pool_size(k: int) -> int:
    """Read through the module so a monkeypatched value is observed."""
    return _search._hybrid_pool_size(k)


class TestPoolArithmetic:
    def test_default_k_fetches_forty(self):
        """The shipped width at the default ``k=10`` -- the number benchmarks quote."""
        assert _hybrid_pool_size(10) == 40

    def test_over_fetches_four_x_for_small_k(self):
        assert _hybrid_pool_size(1) == 4
        assert _hybrid_pool_size(5) == 20

    def test_ceiling_bounds_the_pool(self):
        assert _hybrid_pool_size(25) == 100
        assert _hybrid_pool_size(50) == 100

    def test_never_below_k(self):
        """A large-k request must not be truncated by the ceiling.

        ``fetch_k`` is passed in as ``k`` by the rerank over-fetch, so a ceiling
        that could return less than ``k`` would starve reranking as well as
        silently shrink an ordinary large-k query.
        """
        assert _hybrid_pool_size(200) == 200
        assert _hybrid_pool_size(1000) == 1000

    def test_monotone_in_k(self):
        widths = [_hybrid_pool_size(k) for k in range(1, 300)]
        assert widths == sorted(widths)


@pytest.fixture
def pool_db():
    temp_dir = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="hybrid_pool",
        base_path=temp_dir,
        embedding_provider="mock",
        embedding_model="mock-model",
        chunk_size=500,
        chunk_overlap=0,
    )
    docs = [f"document number {i} discussing subject {i} in detail" for i in range(30)]
    db.upsert(docs, ids=[f"doc{i}" for i in range(30)])
    yield db
    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBothHybridPathsUseIt:
    """A substitution here must reach the real query path, or benchmarks lie."""

    def test_sync_hybrid_fetches_the_returned_width(self, pool_db):
        with patch("localvectordb.database._search._hybrid_pool_size", return_value=17) as fn:
            with patch.object(pool_db, "_vector_search", wraps=pool_db._vector_search) as spy:
                pool_db.query("document subject", k=3, search_type="hybrid")
        assert fn.called, "_hybrid_search did not consult _hybrid_pool_size"
        # positional: (query, return_type, k, ...) -- the third argument is the width.
        assert spy.call_args.args[2] == 17

    @pytest.mark.asyncio
    async def test_async_hybrid_fetches_the_returned_width(self, pool_db):
        with patch("localvectordb.database._search._hybrid_pool_size", return_value=17) as fn:
            with patch.object(
                pool_db,
                "_vector_search_with_embedding_async",
                wraps=pool_db._vector_search_with_embedding_async,
            ) as spy:
                await pool_db.query_async("document subject", k=3, search_type="hybrid")
        assert fn.called, "the async hybrid path did not consult _hybrid_pool_size"
        assert spy.call_args.args[2] == 17
