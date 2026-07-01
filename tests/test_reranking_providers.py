"""HTTP-level tests for API reranker providers.

The existing reranking tests only exercise ``MockReranker``; the real providers'
``rerank()`` implementations (payload construction, response parsing, score
normalization, retry/error handling) were never run. These mock httpx to cover
``JinaReranker`` end-to-end without network access.
"""

from unittest.mock import patch

import pytest

from localvectordb.core import QueryResult
from localvectordb.exceptions import RerankerError
from localvectordb.reranking import JinaReranker


def _results():
    return [
        QueryResult(id="a", score=0.10, type="chunk", content="doc a"),
        QueryResult(id="b", score=0.90, type="chunk", content="doc b"),
    ]


class _FakeResponse:
    def __init__(self, data, status_ok=True):
        self._data = data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._data


class _FakeClient:
    """Stands in for httpx.Client; records the last POST and returns a canned response."""

    last_post = {}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        _FakeClient.last_post = {"url": url, "headers": headers, "json": json}
        return _FakeClient.response


class TestJinaRerankerRerank:
    def _reranker(self):
        return JinaReranker(model="jina-reranker-v2", api_key="test-key", max_retries=0)

    def test_reranks_and_reorders_by_relevance(self):
        _FakeClient.response = _FakeResponse(
            {"results": [{"index": 1, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.30}]}
        )
        with patch("httpx.Client", _FakeClient):
            out = self._reranker().rerank("query text", _results())

        # Reordered by the new (relevance) scores, highest first.
        assert [r.id for r in out] == ["b", "a"]
        assert out[0].score == pytest.approx(0.95)
        # Original scores preserved in metadata.
        assert out[0].metadata["original_score"] == pytest.approx(0.90)
        assert out[1].metadata["original_score"] == pytest.approx(0.10)

        # Payload carried the model, query, and the documents' content.
        sent = _FakeClient.last_post["json"]
        assert sent["model"] == "jina-reranker-v2"
        assert sent["query"] == "query text"
        assert sent["documents"] == ["doc a", "doc b"]
        assert _FakeClient.last_post["headers"]["Authorization"] == "Bearer test-key"

    def test_top_k_limits_and_is_sent_as_top_n(self):
        _FakeClient.response = _FakeResponse(
            {"results": [{"index": 1, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.30}]}
        )
        with patch("httpx.Client", _FakeClient):
            out = self._reranker().rerank("q", _results(), top_k=1)

        assert len(out) == 1 and out[0].id == "b"
        assert _FakeClient.last_post["json"]["top_n"] == 1

    def test_empty_results_short_circuit(self):
        # No HTTP call should happen for an empty candidate list.
        with patch("httpx.Client", side_effect=AssertionError("should not call the API")):
            assert self._reranker().rerank("q", []) == []

    def test_api_error_raises_reranker_error(self):
        _FakeClient.response = _FakeResponse({}, status_ok=False)
        with patch("httpx.Client", _FakeClient):
            with pytest.raises(RerankerError):
                self._reranker().rerank("q", _results())

    def test_missing_api_key_raises_on_construction(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JinaReranker(model="jina-reranker-v2")
