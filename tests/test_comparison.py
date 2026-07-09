"""Tests for the document comparison mixin."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from conftest import make_faiss_index

from localvectordb.core import (
    ChunkAlignment,
    DocumentComparisonResult,
    DocumentSimilarityMatrix,
    QueryResult,
)
from localvectordb.database import LocalVectorDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(temp_dir, n_docs=3, dim=32):
    """Create a LocalVectorDB with mock embeddings containing *n_docs* documents."""
    with (
        patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_emb,
        patch("localvectordb.chunking.ChunkerFactory.create_chunker") as mock_chunker,
        patch("faiss.IndexFlatL2") as mock_flat,
        patch("faiss.IndexIDMap2") as mock_idmap,
    ):
        provider = Mock()
        provider.validate_model.return_value = True
        provider.get_dimension.return_value = dim
        provider.provider_name = "mock"
        provider.model = "mock-model"
        mock_emb.return_value = provider

        mock_index = Mock()
        mock_index.ntotal = 0
        mock_flat.return_value = mock_index
        # A real index: ``db.close()`` hands this to ``faiss.write_index``, which
        # would never return on a Mock. Nothing here reads it -- these tests go
        # through the ``_reconstruct_embeddings_batch`` stub in ``_seed_docs``.
        mock_idmap.return_value = make_faiss_index(dim)
        mock_chunker.return_value = Mock()

        db = LocalVectorDB(
            name="cmp_test",
            base_path=temp_dir,
            embedding_provider="mock",
            embedding_model="mock-model",
        )
    return db


def _seed_docs(db, n=3, dim=32):
    """Insert synthetic documents directly into the database.

    Each document gets a single chunk whose FAISS vector is a unit vector
    perturbed slightly so they aren't identical.
    """
    rng = np.random.RandomState(42)
    with db.connection_pool.get_connection() as conn:
        for i in range(n):
            doc_id = f"doc_{i}"
            conn.execute(
                "INSERT OR IGNORE INTO documents (id, content, content_hash, created_at, updated_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (doc_id, f"Content for document {i}", f"hash_{i}"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO chunks (document_id, chunk_index, content, faiss_id, "
                "start_pos, end_pos, start_line, start_col, end_line, end_col, tokens, content_hash) "
                "VALUES (?, ?, ?, ?, 0, 100, 1, 1, 1, 100, 10, ?)",
                (doc_id, 0, f"Chunk content {i}", i, f"chunk_hash_{i}"),
            )
        conn.commit()

    # Prepare mock for _reconstruct_embeddings_batch
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)

    def reconstruct(faiss_ids):
        out = []
        for fid in faiss_ids:
            if 0 <= fid < n:
                out.append(vecs[fid])
            else:
                out.append(np.zeros(dim, dtype=np.float32))
        if not out:
            return np.array([]).reshape(0, dim)
        return np.array(out, dtype=np.float32)

    db._reconstruct_embeddings_batch = Mock(side_effect=reconstruct)
    return vecs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareDocuments:
    def test_compare_same_document(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        score = db.compare_documents("doc_0", "doc_0")
        # Self-similarity should be ~1.0
        assert score == pytest.approx(1.0, abs=0.01)
        db.close()

    def test_compare_different_documents(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        score = db.compare_documents("doc_0", "doc_1")
        assert 0.0 <= score <= 1.0
        db.close()

    def test_compare_missing_document_raises(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        with pytest.raises(ValueError, match="nonexistent"):
            db.compare_documents("doc_0", "nonexistent")
        db.close()


@pytest.mark.unit
class TestNearestNeighbors:
    def test_returns_correct_count(self, temp_dir):
        db = _make_db(temp_dir, n_docs=5)
        _seed_docs(db, n=5)
        results = db.nearest_neighbors("doc_0", k=3)
        assert len(results) <= 3
        assert all(isinstance(r, QueryResult) for r in results)
        assert all(r.id != "doc_0" for r in results)
        db.close()

    def test_sorted_by_score_descending(self, temp_dir):
        db = _make_db(temp_dir, n_docs=5)
        _seed_docs(db, n=5)
        results = db.nearest_neighbors("doc_0", k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        db.close()

    def test_score_threshold_filters(self, temp_dir):
        db = _make_db(temp_dir, n_docs=5)
        _seed_docs(db, n=5)
        results = db.nearest_neighbors("doc_0", k=10, score_threshold=0.99)
        # With random vectors, most similarities will be well below 0.99
        assert len(results) <= 4
        db.close()

    def test_result_type_is_document(self, temp_dir):
        db = _make_db(temp_dir, n_docs=3)
        _seed_docs(db, n=3)
        results = db.nearest_neighbors("doc_0", k=2)
        for r in results:
            assert r.type == "document"
        db.close()


@pytest.mark.unit
class TestPairwiseSimilarityMatrix:
    def test_matrix_shape(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        mat = db.pairwise_similarity_matrix()
        assert mat.matrix.shape == (3, 3)
        assert len(mat.doc_ids) == 3
        assert mat.embeddings.shape[0] == 3
        db.close()

    def test_diagonal_is_one(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        mat = db.pairwise_similarity_matrix()
        for i in range(3):
            assert mat.matrix[i, i] == pytest.approx(1.0, abs=0.01)
        db.close()

    def test_symmetry(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        mat = db.pairwise_similarity_matrix()
        np.testing.assert_allclose(mat.matrix, mat.matrix.T, atol=1e-6)
        db.close()

    def test_selected_doc_ids(self, temp_dir):
        db = _make_db(temp_dir, n_docs=5)
        _seed_docs(db, n=5)
        mat = db.pairwise_similarity_matrix(doc_ids=["doc_0", "doc_2"])
        assert mat.matrix.shape == (2, 2)
        assert mat.doc_ids == ["doc_0", "doc_2"]
        db.close()

    def test_empty_database(self, temp_dir):
        db = _make_db(temp_dir, n_docs=0)
        mat = db.pairwise_similarity_matrix()
        assert mat.matrix.shape == (0, 0)
        assert mat.doc_ids == []
        db.close()


@pytest.mark.unit
class TestCompareDocumentsDetailed:
    def test_basic_result_structure(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        result = db.compare_documents_detailed("doc_0", "doc_1")
        assert isinstance(result, DocumentComparisonResult)
        assert result.doc_id_1 == "doc_0"
        assert result.doc_id_2 == "doc_1"
        assert 0.0 <= result.overall_similarity <= 1.0
        assert 0.0 <= result.matched_ratio_1 <= 1.0
        assert 0.0 <= result.matched_ratio_2 <= 1.0
        db.close()

    def test_chunk_alignments_sorted(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        result = db.compare_documents_detailed("doc_0", "doc_1")
        if len(result.chunk_alignments) > 1:
            sims = [a.similarity for a in result.chunk_alignments]
            assert sims == sorted(sims, reverse=True)
        db.close()

    def test_self_comparison_all_matched(self, temp_dir):
        db = _make_db(temp_dir)
        _seed_docs(db)
        result = db.compare_documents_detailed("doc_0", "doc_0", chunk_threshold=0.5)
        assert result.matched_ratio_1 == pytest.approx(1.0)
        assert result.matched_ratio_2 == pytest.approx(1.0)
        assert result.unmatched_chunks_1 == []
        assert result.unmatched_chunks_2 == []
        db.close()


@pytest.mark.unit
class TestDataclasses:
    def test_chunk_alignment(self):
        a = ChunkAlignment(chunk_index_1=0, chunk_index_2=3, similarity=0.85)
        assert a.chunk_index_1 == 0
        assert a.chunk_index_2 == 3
        assert a.similarity == 0.85

    def test_document_similarity_matrix(self):
        m = DocumentSimilarityMatrix(
            matrix=np.eye(2),
            doc_ids=["a", "b"],
            embeddings=np.ones((2, 4)),
        )
        assert m.matrix.shape == (2, 2)
        assert m.doc_ids == ["a", "b"]
