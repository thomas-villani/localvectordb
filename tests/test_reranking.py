"""
Tests for localvectordb.reranking module.
"""

from unittest.mock import MagicMock, patch

import pytest

from localvectordb.core import QueryResult
from localvectordb.reranking import (
    HuggingFaceReranker,
    JinaReranker,
    MockReranker,
    Reranker,
    RerankerRegistry,
    SentenceTransformersReranker,
    create_reranker,
    list_rerankers,
)


def _make_results(n=5):
    """Create a list of dummy QueryResult objects."""
    results = []
    for i in range(n):
        results.append(
            QueryResult(
                id=f"doc_{i}:0",
                content=f"document about topic {i} with some words",
                score=0.9 - i * 0.1,
                document_id=f"doc_{i}",
                metadata={"index": i},
                type="chunk",
            )
        )
    return results


@pytest.mark.unit
class TestRerankerABC:
    """Test the Reranker abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            Reranker("test-model")

    def test_abstract_methods_defined(self):
        abstract_methods = Reranker.__abstractmethods__
        assert "rerank" in abstract_methods
        assert "provider_name" in abstract_methods
        assert "validate_model" in abstract_methods


@pytest.mark.unit
class TestMockReranker:
    """Test MockReranker."""

    def test_create(self):
        reranker = MockReranker()
        assert reranker.provider_name == "mock"
        assert reranker.validate_model() is True

    def test_rerank_empty(self):
        reranker = MockReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_word_overlap(self):
        reranker = MockReranker()
        results = [
            QueryResult(
                id="1", content="python programming language", score=0.5, document_id="d1", metadata={}, type="chunk"
            ),
            QueryResult(
                id="2", content="java enterprise development", score=0.8, document_id="d2", metadata={}, type="chunk"
            ),
            QueryResult(
                id="3",
                content="python web development programming",
                score=0.3,
                document_id="d3",
                metadata={},
                type="chunk",
            ),
        ]

        reranked = reranker.rerank("python programming", results)

        # "python programming language" matches 2/2 query words = 1.0
        # "python web development programming" matches 2/2 query words = 1.0
        # "java enterprise development" matches 0/2 query words = 0.0
        assert reranked[0].score >= reranked[-1].score
        assert reranked[-1].content == "java enterprise development"

    def test_rerank_preserves_original_score(self):
        reranker = MockReranker()
        results = _make_results(3)
        original_scores = [r.score for r in results]

        reranked = reranker.rerank("topic", results)

        for r in reranked:
            assert "original_score" in r.metadata
            assert r.metadata["original_score"] in original_scores

    def test_rerank_top_k(self):
        reranker = MockReranker()
        results = _make_results(5)

        reranked = reranker.rerank("topic", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_async(self):
        reranker = MockReranker()
        results = _make_results(3)

        reranked = await reranker.rerank_async("topic", results)

        assert len(reranked) == 3
        for r in reranked:
            assert "original_score" in r.metadata


@pytest.mark.unit
class TestRerankerRegistry:
    """Test RerankerRegistry."""

    def test_list_providers(self):
        providers = RerankerRegistry.list()
        assert "mock" in providers
        assert "jina" in providers
        assert "sentence_transformers" in providers
        assert "huggingface" in providers

    def test_get_mock(self):
        cls = RerankerRegistry.get("mock")
        assert cls is MockReranker

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            RerankerRegistry.get("nonexistent_provider")

    def test_create_reranker_mock(self):
        reranker = RerankerRegistry.create_reranker("mock")
        assert isinstance(reranker, MockReranker)

    def test_create_reranker_with_model(self):
        reranker = RerankerRegistry.create_reranker("mock", "custom-model")
        assert reranker.model == "custom-model"

    def test_convenience_create_reranker(self):
        reranker = create_reranker("mock")
        assert isinstance(reranker, MockReranker)

    def test_convenience_list_rerankers(self):
        providers = list_rerankers()
        assert "mock" in providers


@pytest.mark.unit
class TestJinaReranker:
    """Test JinaReranker initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key is required"):
                JinaReranker(api_key=None)

    def test_env_var_api_key(self):
        with patch.dict("os.environ", {"JINA_API_KEY": "test-key"}):
            reranker = JinaReranker()
            assert reranker.api_key == "test-key"
            assert reranker.provider_name == "jina"

    def test_dollar_env_var(self):
        with patch.dict("os.environ", {"MY_KEY": "custom-key"}):
            reranker = JinaReranker(api_key="$MY_KEY")
            assert reranker.api_key == "custom-key"


@pytest.mark.unit
class TestSentenceTransformersReranker:
    """Test SentenceTransformersReranker."""

    def test_init(self):
        reranker = SentenceTransformersReranker()
        assert reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.provider_name == "sentence_transformers"

    def test_validate_model_import_error(self):
        reranker = SentenceTransformersReranker()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            assert reranker.validate_model() is False


@pytest.mark.unit
class TestHuggingFaceReranker:
    """Test HuggingFaceReranker."""

    def test_init_no_key_ok(self):
        reranker = HuggingFaceReranker()
        assert reranker.provider_name == "huggingface"
        assert reranker.model == "BAAI/bge-reranker-v2-m3"

    def test_env_var_api_key(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf-test-key"}):
            reranker = HuggingFaceReranker()
            assert reranker.api_key == "hf-test-key"


@pytest.mark.unit
class TestCrossEncoderQueryBuilder:
    """Test cross_encoder rerank method in QueryBuilder."""

    def test_rerank_by_model_creates_config(self):
        """Test that rerank_by_model sets the right config."""
        from localvectordb.query_builder import QueryBuilder

        mock_db = MagicMock()
        builder = QueryBuilder(mock_db)
        result = builder.rerank_by_model("mock", "test-model", top_k=5)

        assert result._rerank_config["method"] == "cross_encoder"
        assert result._rerank_config["provider"] == "mock"
        assert result._rerank_config["model"] == "test-model"
        assert result._rerank_config["top_k"] == 5

    def test_rerank_cross_encoder_valid_method(self):
        """Test that cross_encoder is a valid rerank method."""
        from localvectordb.query_builder import QueryBuilder

        mock_db = MagicMock()
        builder = QueryBuilder(mock_db)
        result = builder.rerank("cross_encoder", provider="mock")

        assert result._rerank_config["method"] == "cross_encoder"

    def test_invalid_rerank_method_raises(self):
        """Test that invalid rerank method raises ValueError."""
        from localvectordb.query_builder import QueryBuilder

        mock_db = MagicMock()
        builder = QueryBuilder(mock_db)

        with pytest.raises(ValueError, match="rerank method must be one of"):
            builder.rerank("invalid_method")
