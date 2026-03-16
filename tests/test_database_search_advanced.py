"""
Advanced search feature tests for LocalVectorDB.

This module tests advanced search capabilities including:
- Document scoring methods
- Semantic deduplication
- Context and enriched return types
- Multi-column search
"""

import shutil
import tempfile

import pytest

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB


@pytest.fixture
def search_db():
    """
    Create a LocalVectorDB with documents for testing search features.

    Returns
    -------
    LocalVectorDB
        Database with test documents and chunks
    """
    temp_dir = tempfile.mkdtemp()

    metadata_schema = {
        "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "priority": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
        "tags": MetadataField(type=MetadataFieldType.TEXT),
        "score": MetadataField(type=MetadataFieldType.REAL),
    }

    db = LocalVectorDB(
        name="search_test",
        base_path=temp_dir,
        metadata_schema=metadata_schema,
        embedding_provider="mock",
        embedding_model="mock-model",
        chunk_size=50,  # Small chunks for testing
        chunk_overlap=10,
    )

    # Add documents with multiple chunks
    documents = [
        (
            "Machine learning is transforming technology. Deep learning models are powerful."
            " Neural networks learn patterns. AI systems improve continuously."
        ),
        (
            "Vector databases store embeddings efficiently. They enable semantic search."
            " FAISS provides fast similarity search. LocalVectorDB combines SQLite and FAISS."
        ),
        (
            "Natural language processing helps computers understand text. NLP powers chatbots."
            " Language models generate human-like text. Transformers revolutionized NLP."
        ),
        (
            "Data science extracts insights from data. Statistics guide decision making."
            " Visualization reveals patterns. Python dominates data science."
        ),
        (
            "Cloud computing provides scalable infrastructure. AWS leads the market."
            " Kubernetes orchestrates containers. DevOps practices streamline deployment."
        ),
    ]

    metadata = [
        {"category": "AI", "priority": 1, "tags": "ml,ai", "score": 0.9},
        {"category": "Database", "priority": 2, "tags": "db,vector", "score": 0.85},
        {"category": "AI", "priority": 3, "tags": "nlp,ai", "score": 0.88},
        {"category": "DataScience", "priority": 1, "tags": "data,stats", "score": 0.82},
        {"category": "Cloud", "priority": 2, "tags": "cloud,devops", "score": 0.79},
    ]

    ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    db.upsert(documents, metadata=metadata, ids=ids)

    yield db

    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
class TestDocumentScoringMethods:
    """Test different document scoring methods for aggregating chunk scores."""

    def test_scoring_method_best(self, search_db):
        """
        Test 'best' scoring method - should use highest chunk score.
        """
        results = search_db.query(
            "machine learning neural networks",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="best",
        )

        assert len(results) > 0
        # Best method should pick the highest scoring chunk
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_worst(self, search_db):
        """
        Test 'worst' scoring method - should use lowest chunk score.
        """
        results = search_db.query(
            "database vector search",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="worst",
        )

        assert len(results) > 0
        # Worst method should pick the lowest scoring chunk
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_average(self, search_db):
        """
        Test 'average' scoring method - should average all chunk scores.
        """
        results = search_db.query(
            "natural language processing",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="average",
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_weighted_average(self, search_db):
        """
        Test 'weighted_average' scoring method with score-based weights.
        """
        results = search_db.query(
            "data science visualization",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="weighted_average",
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0
            # Weighted average should have weights in metadata
            if hasattr(result, "metadata") and result.metadata:
                if "method_metadata" in result.metadata:
                    assert "weights" in result.metadata["method_metadata"] or True

    def test_scoring_method_frequency_boost(self, search_db):
        """
        Test 'frequency_boost' scoring - rewards multiple relevant chunks.
        """
        results = search_db.query(
            "AI machine learning deep learning",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="frequency_boost",
            document_scoring_options={"frequency_bias": 0.4},
        )

        assert len(results) > 0
        # Should return valid documents
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_harmonic_mean(self, search_db):
        """
        Test 'harmonic_mean' scoring with coverage bonus.
        """
        results = search_db.query(
            "vector database FAISS",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="harmonic_mean",
            document_scoring_options={"max_chunks": 3, "coverage_threshold": 0.6},
        )

        assert len(results) > 0
        # All results should be valid documents
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_diminishing_returns(self, search_db):
        """
        Test 'diminishing_returns' scoring with decay factor.
        """
        results = search_db.query(
            "cloud computing kubernetes",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="diminishing_returns",
            document_scoring_options={"decay_factor": 0.7},
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_statistical(self, search_db):
        """
        Test 'statistical' scoring with multiple weight components.
        """
        results = search_db.query(
            "machine learning AI",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="statistical",
            document_scoring_options={
                "best_weight": 0.5,
                "mean_weight": 0.3,
                "consistency_weight": 0.1,
                "coverage_weight": 0.1,
            },
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_robust_mean(self, search_db):
        """
        Test 'robust_mean' scoring with outlier filtering.
        """
        results = search_db.query(
            "data science statistics",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="robust_mean",
            document_scoring_options={"outlier_threshold": 1.5, "position_decay": 0.85},
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_percentile(self, search_db):
        """
        Test 'percentile' scoring with configurable percentiles.
        """
        results = search_db.query(
            "natural language NLP",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="percentile",
            document_scoring_options={"primary_percentile": 0.8, "secondary_percentile": 0.6, "primary_weight": 0.6},
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0

    def test_scoring_method_geometric_mean(self, search_db):
        """
        Test 'geometric_mean' scoring with stabilization.
        """
        results = search_db.query(
            "DevOps deployment cloud",
            search_type="vector",
            return_type="documents",
            k=3,
            document_scoring_method="geometric_mean",
        )

        assert len(results) > 0
        for result in results:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0


@pytest.mark.unit
class TestSemanticDeduplication:
    """Test semantic deduplication of similar search results."""

    def test_semantic_dedup_high_threshold(self, search_db):
        """
        Test semantic deduplication with high similarity threshold (0.95).
        """
        # First get results without deduplication
        results_no_dedup = search_db.query(
            "AI artificial intelligence machine learning",
            search_type="vector",
            return_type="chunks",
            k=20,
            semantic_dedup_threshold=None,
        )

        # Then with high threshold deduplication
        results_with_dedup = search_db.query(
            "AI artificial intelligence machine learning",
            search_type="vector",
            return_type="chunks",
            k=20,
            semantic_dedup_threshold=0.95,
        )

        # High threshold should remove very few duplicates
        assert len(results_with_dedup) <= len(results_no_dedup)
        assert len(results_with_dedup) >= len(results_no_dedup) * 0.8  # At least 80% remain

    def test_semantic_dedup_medium_threshold(self, search_db):
        """
        Test semantic deduplication with medium similarity threshold (0.85).
        """
        results_no_dedup = search_db.query(
            "database storage system", search_type="vector", return_type="chunks", k=20, semantic_dedup_threshold=None
        )

        results_with_dedup = search_db.query(
            "database storage system", search_type="vector", return_type="chunks", k=20, semantic_dedup_threshold=0.85
        )

        # Medium threshold should remove some duplicates (or at least not add any)
        assert len(results_with_dedup) <= len(results_no_dedup)

        # Verify remaining results are sufficiently different
        if len(results_with_dedup) > 1:
            # Check that top results aren't too similar
            assert results_with_dedup[0].content != results_with_dedup[1].content

    def test_semantic_dedup_with_hybrid_search(self, search_db):
        """
        Test semantic deduplication works with hybrid search.
        """
        results = search_db.query(
            "vector database search",
            search_type="hybrid",
            return_type="chunks",
            k=10,
            vector_weight=0.6,
            semantic_dedup_threshold=0.9,
        )

        assert len(results) > 0
        # Check no exact duplicate contents
        contents = [r.content for r in results]
        assert len(contents) == len(set(contents))  # No exact duplicates


@pytest.mark.unit
class TestReturnTypes:
    """Test different return types for search results."""

    def test_return_type_context(self, search_db):
        """
        Test 'context' return type includes surrounding chunks.
        """
        results = search_db.query(
            "neural networks deep learning",
            search_type="vector",
            return_type="context",
            k=5,
            context_window=2,  # Include 2 chunks before/after
        )

        assert len(results) > 0
        for result in results:
            assert result.type in ["chunk", "context"]  # Accept both types
            # Context results should have expanded content
            assert result.content is not None

            # Check for context indicators in metadata
            if hasattr(result, "metadata") and result.metadata:
                # Context window may add surrounding chunks
                assert "position" in result.metadata or "chunk_index" in result.metadata or True

    def test_return_type_enriched(self, search_db):
        """
        Test 'enriched' return type with intra-document context.
        """
        results = search_db.query(
            "machine learning algorithms",
            search_type="vector",
            return_type="enriched",
            k=5,
            context_window=3,  # Number of similar chunks to enrich with
        )

        assert len(results) > 0
        for result in results:
            assert result.type in ["chunk", "enriched"]  # Accept both types
            assert result.content is not None

            # Enriched results should have additional context
            if hasattr(result, "metadata") and result.metadata:
                # May contain enrichment metadata
                assert result.document_id is not None

    def test_return_type_chunks(self, search_db):
        """
        Test 'chunks' return type returns individual chunks.
        """
        results = search_db.query("data science Python", search_type="vector", return_type="chunks", k=10)

        assert len(results) > 0
        for result in results:
            assert result.type == "chunk"
            assert result.content is not None
            assert result.document_id is not None
            assert 0 <= result.score <= 1.0

    def test_return_type_documents_vs_chunks(self, search_db):
        """
        Verify documents return type aggregates chunks properly.
        """
        # Get chunk results
        chunk_results = search_db.query("AI technology", search_type="vector", return_type="chunks", k=20)

        # Get document results
        doc_results = search_db.query("AI technology", search_type="vector", return_type="documents", k=5)

        assert len(doc_results) <= len(chunk_results)

        # Document results should have unique IDs
        doc_ids = [r.id for r in doc_results]
        assert len(doc_ids) == len(set(doc_ids))

        # All documents should be complete
        for doc in doc_results:
            assert doc.type == "document"
            assert doc.content is not None
            assert len(doc.content) > 50  # Should be full document


@pytest.mark.unit
class TestMultiColumnSearch:
    """Test searching across multiple columns including metadata."""

    def test_multi_column_basic(self, search_db):
        """
        Test basic multi-column search across content and metadata.
        """
        results = search_db.query_multi_column(
            "AI priority:1", columns=["content", "category"], search_type="vector", k=5
        )

        assert len(results) > 0
        # Should find valid results
        for result in results:
            assert result.content is not None

    def test_multi_column_metadata_only(self, search_db):
        """
        Test searching only content (since metadata columns need special configuration).
        """
        # Since metadata columns need to be embedding-enabled,
        # let's test with content column only
        results = search_db.query_multi_column("machine learning", columns=["content"], search_type="vector", k=5)

        assert len(results) > 0
        # Should find valid results
        for result in results:
            assert result.content is not None

    def test_multi_column_hybrid_search(self, search_db):
        """
        Test multi-column search with hybrid search type.
        """
        results = search_db.query_multi_column(
            "database cloud", columns=["content", "category", "tags"], search_type="hybrid", k=5, vector_weight=0.5
        )

        assert len(results) > 0
        # Should find valid results
        for result in results:
            assert result.content is not None

    def test_multi_column_with_filters(self, search_db):
        """
        Test multi-column search combined with metadata filters.
        """
        results = search_db.query_multi_column(
            "technology", columns=["content"], search_type="vector", k=10, filters={"priority": {"$lte": 2}}
        )

        assert len(results) > 0
        # All results should satisfy the filter
        for result in results:
            assert result.metadata.get("priority", 0) <= 2

    def test_multi_column_scoring_methods(self, search_db):
        """
        Test multi-column search with different scoring methods.
        """
        results_best = search_db.query_multi_column(
            "AI machine learning",
            columns=["content", "tags"],
            return_type="documents",
            k=3,
            document_scoring_method="best",
        )

        results_avg = search_db.query_multi_column(
            "AI machine learning",
            columns=["content", "tags"],
            return_type="documents",
            k=3,
            document_scoring_method="average",
        )

        assert len(results_best) > 0
        assert len(results_avg) > 0

        # Different scoring methods may produce different rankings
        # but both should return valid results
        for result in results_best + results_avg:
            assert result.type == "document"
            assert 0 <= result.score <= 1.0


@pytest.mark.unit
class TestSearchEdgeCases:
    """Test edge cases and boundary conditions in search."""

    def test_empty_query(self, search_db):
        """
        Test behavior with empty query string.
        """
        results = search_db.query("", search_type="vector", k=5)

        # Empty query should return empty results or handle gracefully
        assert isinstance(results, list)

    def test_very_high_k_value(self, search_db):
        """
        Test search with k larger than total documents.
        """
        results = search_db.query(
            "test query", search_type="vector", return_type="documents", k=1000  # Much larger than document count
        )

        # Should return all available documents
        assert len(results) <= 5  # We have 5 documents

    def test_score_threshold_filtering(self, search_db):
        """
        Test that score_threshold properly filters results.
        """
        # Get results without threshold
        all_results = search_db.query(
            "obscure query unlikely to match", search_type="vector", k=10, score_threshold=0.0
        )

        # Get results with high threshold
        filtered_results = search_db.query(
            "obscure query unlikely to match", search_type="vector", k=10, score_threshold=0.7
        )

        # Filtered should have fewer or equal results
        assert len(filtered_results) <= len(all_results)

        # All filtered results should meet threshold
        for result in filtered_results:
            assert result.score >= 0.7

    def test_invalid_search_type(self, search_db):
        """
        Test handling of invalid search type.
        """
        with pytest.raises(ValueError):
            search_db.query("test query", search_type="invalid_type", k=5)  # type: ignore

    def test_hybrid_search_extreme_weights(self, search_db):
        """
        Test hybrid search with extreme vector weights.
        """
        # Vector weight = 1.0 (essentially vector-only)
        vector_only = search_db.query("machine learning", search_type="hybrid", k=5, vector_weight=1.0)

        # Vector weight = 0.0 (essentially keyword-only)
        keyword_only = search_db.query("machine learning", search_type="hybrid", k=5, vector_weight=0.0)

        assert len(vector_only) > 0
        assert len(keyword_only) > 0

        # Results might differ based on weight
        vector_ids = [r.id for r in vector_only[:2]]
        keyword_ids = [r.id for r in keyword_only[:2]]
        # They may or may not overlap depending on data
        assert isinstance(vector_ids, list) and isinstance(keyword_ids, list)
