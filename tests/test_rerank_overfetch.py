"""Reranking over-fetch and score-normalization (T1.2).

Before this change, ``query()`` reranked the *already-truncated* top-``k`` results:
each search leg sorted and returned ``[:k]``, then the reranker reordered those
``k`` and returned ``[:k]`` again. A cross-encoder that never sees the candidates
a leg ranked at ``k+1, k+2, ...`` cannot pull a better match up into the top-``k`` --
which is the entire purpose of reranking. It was a no-op on recall.

The fix over-fetches a larger pool (``rerank_k``, default ``5*k``) whenever a
reranker is configured, reranks that pool, and truncates to ``k``. When no reranker
is configured ``fetch_k == k`` and the search path is byte-for-byte unchanged --
that invariant is what keeps the retrieval baseline stable (reranking is excluded
from it).

The second half pins the score-normalization contract. Every reranker now leaves an
*absolute* ``[0, 1]`` score (comparable across queries, survives ``score_threshold``)
and preserves the model's raw output in ``metadata["rerank_raw_score"]``. In
particular ``HuggingFaceReranker`` no longer uses per-batch min-max, which was
pool-relative (top always 1.0, bottom always 0.0) -- the same defect T1.1 removed
from hybrid fusion.
"""

import math
import shutil
import tempfile
from typing import List, Optional
from unittest.mock import patch

import pytest

from localvectordb.core import QueryResult
from localvectordb.database import LocalVectorDB
from localvectordb.database._search import _RERANK_K_MAX, _resolve_rerank_k
from localvectordb.reranking import HuggingFaceReranker, Reranker


class _RecordingReranker(Reranker):
    """Records how many candidates it was handed on each call; preserves order."""

    def __init__(self) -> None:
        super().__init__("recording")
        self.seen_counts: List[int] = []

    @property
    def provider_name(self) -> str:
        return "recording"

    def validate_model(self) -> bool:
        return True

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        self.seen_counts.append(len(results))
        ranked = list(results)
        return ranked[:top_k] if top_k is not None else ranked


class _MarkerReranker(Reranker):
    """Deterministic: the one result whose content holds ``marker`` scores 1.0."""

    def __init__(self, marker: str) -> None:
        super().__init__("marker")
        self.marker = marker

    @property
    def provider_name(self) -> str:
        return "marker"

    def validate_model(self) -> bool:
        return True

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        for r in results:
            r.metadata = r.metadata or {}
            r.metadata["original_score"] = r.score
            r.score = 1.0 if self.marker in (r.content or "") else 0.0
        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        return ranked[:top_k] if top_k is not None else ranked


# --------------------------------------------------------------------------- #
# _resolve_rerank_k
# --------------------------------------------------------------------------- #
class TestResolveRerankK:
    def test_default_is_five_k(self):
        assert _resolve_rerank_k(None, 10) == 50

    def test_clamped_to_ceiling(self):
        assert _resolve_rerank_k(9999, 10) == _RERANK_K_MAX

    def test_never_below_k(self):
        # A caller asking for fewer candidates than k must not shrink the result set.
        assert _resolve_rerank_k(3, 10) == 10

    def test_large_k_passes_through_even_above_ceiling(self):
        # If k itself exceeds the ceiling, fetch_k == k (no over-fetch, but no shrink).
        assert _resolve_rerank_k(None, 300) == 300


# --------------------------------------------------------------------------- #
# Over-fetch mechanism (real DB, mock embeddings)
# --------------------------------------------------------------------------- #
@pytest.fixture
def many_docs_db():
    temp_dir = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="rerank_overfetch",
        base_path=temp_dir,
        embedding_provider="mock",
        embedding_model="mock-model",
        chunk_size=500,
        chunk_overlap=0,
    )
    docs = [f"document number {i} discussing subject {i} in detail" for i in range(15)]
    ids = [f"doc{i}" for i in range(15)]
    db.upsert(docs, ids=ids)
    yield db
    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestOverfetch:
    def test_reranker_receives_more_than_k(self, many_docs_db):
        rr = _RecordingReranker()
        results = many_docs_db.query("document subject", k=2, search_type="vector", reranker=rr)

        # The reranker saw the over-fetched pool, not just the final k.
        assert rr.seen_counts, "reranker was never invoked"
        assert rr.seen_counts[-1] > 2
        assert rr.seen_counts[-1] <= _resolve_rerank_k(None, 2)  # default 5*k == 10
        # But the caller still gets exactly k back.
        assert len(results) == 2

    def test_larger_rerank_k_widens_the_pool(self, many_docs_db):
        rr = _RecordingReranker()
        many_docs_db.query("document subject", k=1, search_type="vector", reranker=rr, rerank_k=3)
        many_docs_db.query("document subject", k=1, search_type="vector", reranker=rr, rerank_k=15)

        small, large = rr.seen_counts[-2], rr.seen_counts[-1]
        assert large > small, f"rerank_k=15 pool ({large}) not larger than rerank_k=3 pool ({small})"

    def test_overfetch_lets_the_reranker_reach_a_buried_result(self):
        """A doc BM25 ranks last is promotable only when the pool is fetched deep enough."""
        temp_dir = tempfile.mkdtemp()
        db = LocalVectorDB(
            name="rerank_buried",
            base_path=temp_dir,
            embedding_provider="mock",
            embedding_model="mock-model",
            chunk_size=500,
            chunk_overlap=0,
        )
        try:
            # 11 strong keyword matches ("common" twice) + one weak match that also
            # carries the marker the reranker rewards. The weak match ranks last by
            # BM25, so a shallow pool never contains it.
            docs = [f"common common filler text alpha{i}" for i in range(11)]
            ids = [f"strong{i}" for i in range(11)]
            docs.append("common zzmarker lonely tail padding padding padding")
            ids.append("buried")
            db.upsert(docs, ids=ids)

            rr = _MarkerReranker(marker="zzmarker")

            # Shallow pool (rerank_k == k == 1): reranker only sees the BM25 winner,
            # which is a "strong" doc, so the buried marker doc cannot surface.
            shallow = db.query("common", k=1, search_type="keyword", reranker=rr, rerank_k=1)
            assert shallow[0].id != "buried"

            # Deep pool: the reranker sees the buried doc and promotes it to rank 1.
            deep = db.query("common", k=1, search_type="keyword", reranker=rr, rerank_k=12)
            assert deep[0].id == "buried"
        finally:
            db.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# HuggingFace normalization: sigmoid (absolute), not min-max (pool-relative)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, data, ok=True):
        self._data = data
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._data


class _FakeClient:
    response = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        return _FakeClient.response


def _hf_results():
    return [
        QueryResult(id="a", score=0.10, type="chunk", content="doc a"),
        QueryResult(id="b", score=0.20, type="chunk", content="doc b"),
        QueryResult(id="c", score=0.30, type="chunk", content="doc c"),
    ]


class TestHuggingFaceNormalization:
    def _reranker(self):
        return HuggingFaceReranker(model="bge-reranker", api_key="test-key", max_retries=0)

    def test_uses_sigmoid_not_minmax(self):
        # Raw logits. Min-max would force 1.0 / 0.0 / 0.5; sigmoid maps each
        # independently, so the top score is ~0.881, never exactly 1.0.
        _FakeClient.response = _FakeResponse([2.0, -2.0, 0.0])
        with patch("httpx.Client", _FakeClient):
            out = self._reranker().rerank("q", _hf_results())

        top = out[0]
        assert top.id == "a"
        assert top.score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))  # ~0.8808
        assert top.score < 1.0  # the min-max tell would be exactly 1.0
        # Worst logit maps to sigmoid(-2) ~ 0.119, not min-max's 0.0.
        worst = next(r for r in out if r.id == "b")
        assert worst.score == pytest.approx(1.0 / (1.0 + math.exp(2.0)))
        assert worst.score > 0.0

    def test_preserves_raw_score(self):
        _FakeClient.response = _FakeResponse([2.0, -2.0, 0.0])
        with patch("httpx.Client", _FakeClient):
            out = self._reranker().rerank("q", _hf_results())

        top = out[0]
        assert top.metadata["rerank_raw_score"] == pytest.approx(2.0)
        assert top.metadata["original_score"] == pytest.approx(0.10)
