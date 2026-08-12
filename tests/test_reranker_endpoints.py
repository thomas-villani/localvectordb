"""Reranking against self-hosted and routed endpoints.

A local embedding stack is only half a local stack if reranking still needs a
hosted API key. These cover the shared /rerank wire format: the request shape is
identical across Jina, vLLM, text-embeddings-inference, Infinity and OpenRouter,
but the servers disagree on how they spell the response and whether their scores
are already probabilities -- and the library's score_threshold depends on the
answer to that last one.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from localvectordb.core import QueryResult
from localvectordb.exceptions import RerankerError
from localvectordb.reranking import (
    JinaReranker,
    OpenAICompatibleReranker,
    OpenRouterReranker,
    RerankerRegistry,
)

VLLM = "http://localhost:8000/v1"


def _results(n: int = 3) -> List[QueryResult]:
    return [
        QueryResult(
            id=f"doc_{i}:0",
            content=f"document {i}",
            score=0.9 - i * 0.1,
            document_id=f"doc_{i}",
            metadata={"index": i},
            type="chunk",
        )
        for i in range(n)
    ]


def _patch_sync(json_body: Any):
    """Patch httpx.Client so rerank() sees json_body, returning the mock client."""
    response = MagicMock()
    response.json.return_value = json_body
    response.raise_for_status = Mock()
    client = MagicMock()
    client.post = Mock(return_value=response)
    patcher = patch("httpx.Client")
    client_class = patcher.start()
    client_class.return_value.__enter__ = Mock(return_value=client)
    client_class.return_value.__exit__ = Mock(return_value=None)
    return patcher, client


@pytest.mark.unit
class TestConstruction:
    def test_base_url_is_required_for_self_hosted(self):
        with pytest.raises(ValueError, match="base_url is required"):
            OpenAICompatibleReranker("bge-reranker-base")

    def test_missing_base_url_names_the_common_servers(self):
        with pytest.raises(ValueError) as exc:
            OpenAICompatibleReranker("bge-reranker-base")
        assert "vLLM" in str(exc.value)

    def test_no_api_key_needed(self):
        """The point of the provider: a local reranker has no credential."""
        assert OpenAICompatibleReranker("bge-reranker-base", base_url=VLLM).api_key is None

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("RERANKER_API_KEY", "sk-local")
        assert OpenAICompatibleReranker("m", base_url=VLLM).api_key == "sk-local"

    def test_endpoint_appends_rerank(self):
        assert OpenAICompatibleReranker("m", base_url=VLLM + "/").endpoint == f"{VLLM}/rerank"

    def test_tei_serves_rerank_at_the_root(self):
        """text-embeddings-inference has no /v1 prefix; base_url must allow that."""
        reranker = OpenAICompatibleReranker("m", base_url="http://localhost:8080")
        assert reranker.endpoint == "http://localhost:8080/rerank"

    def test_unknown_score_transform_rejected(self):
        with pytest.raises(ValueError, match="score_transform"):
            OpenAICompatibleReranker("m", base_url=VLLM, score_transform="minmax")

    def test_registered_for_discovery(self):
        assert RerankerRegistry.get("openai_compatible") is OpenAICompatibleReranker
        assert RerankerRegistry.get("openrouter") is OpenRouterReranker


@pytest.mark.unit
class TestHeaders:
    def test_no_authorization_without_a_key(self):
        headers = OpenAICompatibleReranker("m", base_url=VLLM)._headers()
        assert "Authorization" not in headers

    def test_authorization_with_a_key(self):
        headers = OpenAICompatibleReranker("m", base_url=VLLM, api_key="sk-x")._headers()
        assert headers["Authorization"] == "Bearer sk-x"


@pytest.mark.unit
class TestResponseShapes:
    """Servers disagree on the envelope and the score field name."""

    def _rerank(self, json_body: Any, **kwargs: Any) -> List[QueryResult]:
        patcher, _ = _patch_sync(json_body)
        try:
            reranker = OpenAICompatibleReranker("m", base_url=VLLM, **kwargs)
            return reranker.rerank("query", _results(3))
        finally:
            patcher.stop()

    def test_jina_style_results_with_relevance_score(self):
        out = self._rerank({"results": [{"index": 2, "relevance_score": 0.9}]})
        assert [r.id for r in out] == ["doc_2:0"]
        assert out[0].score == pytest.approx(0.9)

    def test_data_envelope(self):
        out = self._rerank({"data": [{"index": 1, "relevance_score": 0.7}]})
        assert out[0].id == "doc_1:0"

    def test_bare_list_from_text_embeddings_inference(self):
        out = self._rerank([{"index": 0, "score": 0.6}])
        assert out[0].id == "doc_0:0"

    def test_missing_score_field_is_reported(self):
        with pytest.raises(RerankerError, match="relevance_score or score"):
            self._rerank({"results": [{"index": 0}]}, max_retries=0)

    def test_out_of_range_index_is_reported(self):
        with pytest.raises(RerankerError, match="index 99"):
            self._rerank({"results": [{"index": 99, "score": 0.5}]}, max_retries=0)

    def test_unreadable_response_is_reported(self):
        with pytest.raises(RerankerError, match="unreadable"):
            self._rerank("not json at all", max_retries=0)


@pytest.mark.unit
class TestScoreTransform:
    """score_threshold depends on scores being absolute, not pool-relative."""

    def _score(self, raw: float, **kwargs: Any) -> float:
        patcher, _ = _patch_sync({"results": [{"index": 0, "score": raw}]})
        try:
            reranker = OpenAICompatibleReranker("m", base_url=VLLM, **kwargs)
            return reranker.rerank("query", _results(1))[0].score
        finally:
            patcher.stop()

    def test_probability_passes_through_untouched(self):
        assert self._score(0.73) == pytest.approx(0.73)

    def test_logit_is_squashed_into_range(self):
        """A cross-encoder returning logits would otherwise break thresholding."""
        assert self._score(4.0) == pytest.approx(0.982, abs=1e-3)

    def test_negative_logit_is_squashed(self):
        assert self._score(-4.0) == pytest.approx(0.018, abs=1e-3)

    def test_none_leaves_raw_scores_alone(self):
        assert self._score(4.0, score_transform="none") == pytest.approx(4.0)

    def test_sigmoid_forces_the_squash(self):
        assert self._score(0.5, score_transform="sigmoid") == pytest.approx(0.622, abs=1e-3)

    def test_raw_score_is_preserved_in_metadata(self):
        patcher, _ = _patch_sync({"results": [{"index": 0, "score": 4.0}]})
        try:
            reranker = OpenAICompatibleReranker("m", base_url=VLLM)
            out = reranker.rerank("query", _results(1))
        finally:
            patcher.stop()

        assert out[0].metadata["rerank_raw_score"] == 4.0
        assert out[0].metadata["original_score"] == pytest.approx(0.9)


@pytest.mark.unit
class TestRequest:
    def test_posts_to_the_configured_endpoint(self):
        patcher, client = _patch_sync({"results": [{"index": 0, "score": 0.5}]})
        try:
            OpenAICompatibleReranker("bge-reranker-base", base_url=VLLM).rerank("q", _results(2))
        finally:
            patcher.stop()

        assert client.post.call_args.args[0] == f"{VLLM}/rerank"
        payload = client.post.call_args.kwargs["json"]
        assert payload["model"] == "bge-reranker-base"
        assert payload["query"] == "q"
        assert payload["documents"] == ["document 0", "document 1"]

    def test_top_k_is_sent_and_applied(self):
        patcher, client = _patch_sync({"results": [{"index": 0, "score": 0.5}, {"index": 1, "score": 0.9}]})
        try:
            out = OpenAICompatibleReranker("m", base_url=VLLM).rerank("q", _results(2), top_k=1)
        finally:
            patcher.stop()

        assert client.post.call_args.kwargs["json"]["top_n"] == 1
        assert [r.id for r in out] == ["doc_1:0"]

    def test_empty_results_short_circuit_without_a_request(self):
        patcher, client = _patch_sync({"results": []})
        try:
            assert OpenAICompatibleReranker("m", base_url=VLLM).rerank("q", []) == []
        finally:
            patcher.stop()
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_path_uses_the_same_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"results": [{"index": 0, "score": 0.5}]}
        response.raise_for_status = Mock()
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as client_class:
            client_class.return_value.__aenter__ = AsyncMock(return_value=client)
            client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            out = await OpenAICompatibleReranker("m", base_url=VLLM).rerank_async("q", _results(1))

        assert client.post.call_args.args[0] == f"{VLLM}/rerank"
        assert out[0].id == "doc_0:0"


@pytest.mark.unit
class TestOpenRouterReranker:
    def test_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterReranker("some/rerank-model")

    def test_defaults_to_the_openrouter_endpoint(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        assert OpenRouterReranker("some/rerank-model").endpoint == "https://openrouter.ai/api/v1/rerank"

    def test_sends_the_bearer_token(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        assert OpenRouterReranker("m")._headers()["Authorization"] == "Bearer sk-or"


@pytest.mark.unit
class TestJinaRerankerUnchanged:
    """The refactor must not move Jina's behaviour."""

    def test_still_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Jina API key is required"):
            JinaReranker()

    def test_still_targets_api_jina_ai(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-jina")
        assert JinaReranker().endpoint == "https://api.jina.ai/v1/rerank"

    def test_default_model_unchanged(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-jina")
        assert JinaReranker().model == "jina-reranker-v2-base-multilingual"

    def test_provider_name_unchanged(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-jina")
        assert JinaReranker().provider_name == "jina"

    def test_error_message_still_says_jina(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-jina")
        patcher = patch("httpx.Client", side_effect=RuntimeError("boom"))
        patcher.start()
        try:
            with pytest.raises(RerankerError, match="Jina reranking failed"):
                JinaReranker(max_retries=0).rerank("q", _results(1))
        finally:
            patcher.stop()
