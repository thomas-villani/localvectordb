"""Tests for the visualization module."""

import numpy as np
import pytest

from localvectordb.core import DocumentSimilarityMatrix

# Guard: skip all tests if visualization deps are missing
sklearn = pytest.importorskip("sklearn")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

from localvectordb.visualization import (  # noqa: E402
    build_similarity_graph,
    cluster_embeddings,
    find_optimal_clusters,
    plot_clusters,
    plot_embedding_map,
    plot_similarity_graph,
    plot_similarity_matrix,
    reduce_dimensions,
)
from localvectordb.visualization._dimensionality import project_new_points  # noqa: E402
from localvectordb.visualization.types import ClusterResult, EmbeddingProjection, QueryOverlay  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embeddings():
    rng = np.random.RandomState(0)
    return rng.randn(20, 64).astype(np.float32)


@pytest.fixture
def sample_doc_ids():
    return [f"doc_{i}" for i in range(20)]


@pytest.fixture
def sample_sim_matrix():
    rng = np.random.RandomState(1)
    embs = rng.randn(5, 32).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.maximum(norms, 1e-8)
    matrix = (embs_norm @ embs_norm.T + 1.0) / 2.0
    return DocumentSimilarityMatrix(
        matrix=matrix,
        doc_ids=[f"d{i}" for i in range(5)],
        embeddings=embs,
    )


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReduceDimensions:
    def test_pca_shape(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        assert proj.coordinates.shape == (20, 2)
        assert proj.method == "pca"
        assert len(proj.doc_ids) == 20
        assert proj.explained_variance is not None

    def test_tsne_shape(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="tsne", doc_ids=sample_doc_ids, perplexity=5)
        assert proj.coordinates.shape == (20, 2)
        assert proj.method == "tsne"

    def test_pca_3d(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", n_components=3, doc_ids=sample_doc_ids)
        assert proj.coordinates.shape == (20, 3)

    def test_auto_doc_ids(self, sample_embeddings):
        proj = reduce_dimensions(sample_embeddings, method="pca")
        assert proj.doc_ids == [str(i) for i in range(20)]

    def test_empty_embeddings(self):
        proj = reduce_dimensions(np.array([]).reshape(0, 64), method="pca")
        assert proj.coordinates.shape == (0, 2)

    def test_unknown_method_raises(self, sample_embeddings):
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_dimensions(sample_embeddings, method="umap")

    def test_few_samples_tsne_fallback(self):
        # Only 2 samples: should fall back to PCA
        embs = np.random.randn(2, 10).astype(np.float32)
        proj = reduce_dimensions(embs, method="tsne")
        assert proj.method == "pca"
        assert proj.coordinates.shape[0] == 2


@pytest.mark.unit
class TestProjectNewPoints:
    def test_pca_projection(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        new_pts = np.random.randn(3, 64).astype(np.float32)
        coords = project_new_points(proj, new_pts)
        assert coords.shape == (3, 2)

    def test_tsne_fallback_projection(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="tsne", doc_ids=sample_doc_ids, perplexity=5)
        new_pts = np.random.randn(2, 64).astype(np.float32)
        coords = project_new_points(proj, new_pts)
        assert coords.shape == (2, 2)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClustering:
    def test_kmeans_basic(self, sample_embeddings):
        result = cluster_embeddings(sample_embeddings, n_clusters=3)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3
        assert result.labels.shape == (20,)
        assert set(result.labels).issubset({0, 1, 2})
        assert result.centroids.shape == (3, 64)

    def test_auto_clusters(self, sample_embeddings):
        result = cluster_embeddings(sample_embeddings)
        assert result.n_clusters >= 1

    def test_single_sample(self):
        embs = np.random.randn(1, 10).astype(np.float32)
        result = cluster_embeddings(embs, n_clusters=1)
        assert result.n_clusters == 1
        assert result.labels.shape == (1,)

    def test_empty_embeddings(self):
        result = cluster_embeddings(np.array([]).reshape(0, 10))
        assert result.n_clusters == 0
        assert result.labels.shape == (0,)


@pytest.mark.unit
class TestFindOptimalClusters:
    def test_returns_at_least_one(self, sample_embeddings):
        k = find_optimal_clusters(sample_embeddings)
        assert k >= 1

    def test_too_few_samples(self):
        embs = np.random.randn(2, 10).astype(np.float32)
        k = find_optimal_clusters(embs)
        assert k == 1

    def test_max_k_respected(self, sample_embeddings):
        k = find_optimal_clusters(sample_embeddings, max_k=3)
        assert k <= 3


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlotEmbeddingMap:
    def test_basic_plot(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        fig = plot_embedding_map(proj)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_color_by(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        labels = ["A" if i < 10 else "B" for i in range(20)]
        fig = plot_embedding_map(proj, color_by=labels)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_queries(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        q = QueryOverlay(
            query_text="test query",
            query_embedding=np.random.randn(64).astype(np.float32),
            scores=np.random.rand(20).astype(np.float32),
        )
        fig = plot_embedding_map(proj, queries=[q])
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_projection(self):
        proj = EmbeddingProjection(
            coordinates=np.array([]).reshape(0, 2),
            method="pca",
            doc_ids=[],
        )
        fig = plot_embedding_map(proj)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
class TestPlotSimilarityMatrix:
    def test_basic_heatmap(self, sample_sim_matrix):
        fig = plot_similarity_matrix(sample_sim_matrix)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_matrix(self):
        sm = DocumentSimilarityMatrix(
            matrix=np.array([]).reshape(0, 0),
            doc_ids=[],
            embeddings=np.array([]).reshape(0, 0),
        )
        fig = plot_similarity_matrix(sm)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
class TestPlotClusters:
    def test_basic_clusters(self, sample_embeddings, sample_doc_ids):
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        clusters = cluster_embeddings(sample_embeddings, n_clusters=3)
        fig = plot_clusters(proj, clusters)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSimilarityGraph:
    def test_basic_graph(self, sample_sim_matrix):
        graph = build_similarity_graph(sample_sim_matrix, threshold=0.3)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 5
        for edge in graph["edges"]:
            assert edge["weight"] >= 0.3
            assert "source" in edge
            assert "target" in edge

    def test_high_threshold_fewer_edges(self, sample_sim_matrix):
        g1 = build_similarity_graph(sample_sim_matrix, threshold=0.0)
        g2 = build_similarity_graph(sample_sim_matrix, threshold=0.9)
        assert len(g2["edges"]) <= len(g1["edges"])


@pytest.mark.unit
class TestPlotSimilarityGraph:
    def test_basic_graph_plot(self, sample_sim_matrix):
        fig = plot_similarity_graph(sample_sim_matrix, threshold=0.3)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_graph(self):
        sm = DocumentSimilarityMatrix(
            matrix=np.array([]).reshape(0, 0),
            doc_ids=[],
            embeddings=np.array([]).reshape(0, 0),
        )
        fig = plot_similarity_graph(sm)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
