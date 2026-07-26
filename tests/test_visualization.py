"""Tests for the visualization module."""

import numpy as np
import pytest

from localvectordb.core import ChunkSimilarityMatrix, DocumentSimilarityMatrix

# Guard: skip all tests if visualization deps are missing
sklearn = pytest.importorskip("sklearn")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

from localvectordb.visualization import (  # noqa: E402
    build_similarity_graph,
    cluster_embeddings,
    find_optimal_clusters,
    plot_chord,
    plot_clusters,
    plot_embedding_map,
    plot_similarity_graph,
    plot_similarity_matrix,
    plot_synteny,
    reduce_dimensions,
)
from localvectordb.visualization._dimensionality import project_new_points  # noqa: E402
from localvectordb.visualization._graph import _spring_positions  # noqa: E402
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

    def test_single_sample_tsne_pads_to_n_components(self):
        # H7: a single-document DB collapses PCA to 1 component; coordinates must
        # still be padded to n_components so consumers can index coords[:, 1].
        embs = np.random.randn(1, 10).astype(np.float32)
        proj = reduce_dimensions(embs, method="tsne")
        assert proj.coordinates.shape == (1, 2)

    def test_single_sample_pca_pads_to_n_components(self):
        embs = np.random.randn(1, 10).astype(np.float32)
        proj = reduce_dimensions(embs, method="pca")
        assert proj.coordinates.shape == (1, 2)

    def test_single_sample_pads_3d(self):
        embs = np.random.randn(1, 10).astype(np.float32)
        proj = reduce_dimensions(embs, method="pca", n_components=3)
        assert proj.coordinates.shape == (1, 3)

    def test_embedding_dim_below_n_components_pads(self):
        # An embedding dimension smaller than n_components also under-produces columns.
        embs = np.random.randn(5, 1).astype(np.float32)
        proj = reduce_dimensions(embs, method="pca", n_components=2)
        assert proj.coordinates.shape == (5, 2)


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

    def test_color_by_and_queries_share_one_legend(self, sample_embeddings, sample_doc_ids):
        # A second ax.legend() call replaces the first, so colouring by category
        # *and* overlaying queries used to silently drop the category key.
        proj = reduce_dimensions(sample_embeddings, method="pca", doc_ids=sample_doc_ids)
        labels = ["A" if i < 10 else "B" for i in range(20)]
        q = QueryOverlay(
            query_text="test query",
            query_embedding=np.random.randn(64).astype(np.float32),
            scores=np.random.rand(20).astype(np.float32),
        )
        fig = plot_embedding_map(proj, color_by=labels, queries=[q])
        entries = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
        assert {"A", "B"} <= entries
        assert any(e.startswith("Q: test query") for e in entries)
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

    def test_single_document_default_tsne_does_not_crash(self):
        # H7: the default tsne path on a single-document DB used to IndexError at
        # coords[:, 1]. Padding keeps the whole plot path (dots + query overlay) alive.
        embs = np.random.randn(1, 32).astype(np.float32)
        proj = reduce_dimensions(embs, method="tsne", doc_ids=["only_doc"])
        q = QueryOverlay(
            query_text="q",
            query_embedding=np.random.randn(32).astype(np.float32),
            scores=np.array([0.9], dtype=np.float32),
        )
        fig = plot_embedding_map(proj, queries=[q])
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

    def test_mds_layout(self, sample_sim_matrix):
        fig = plot_similarity_graph(sample_sim_matrix, threshold=0.3, layout="mds")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_unknown_layout_raises(self, sample_sim_matrix):
        with pytest.raises(ValueError, match="Unknown layout"):
            plot_similarity_graph(sample_sim_matrix, layout="circular")


@pytest.mark.unit
class TestSpringPositions:
    """The force-directed layout behind ``layout="spring"``."""

    @staticmethod
    def _two_triangles():
        """Two disconnected triangles plus four isolated nodes."""
        weights = np.zeros((10, 10))
        for a, b in [(0, 1), (1, 2), (0, 2), (5, 6), (6, 7), (5, 7)]:
            weights[a, b] = weights[b, a] = 0.9
        return weights

    def test_finite_positions(self):
        # A zero self-weight against an infinite self-distance produced NaN for
        # every node; the layout must stay finite for any weight matrix.
        pos = _spring_positions(self._two_triangles())
        assert pos.shape == (10, 2)
        assert np.isfinite(pos).all()

    def test_no_edges_is_finite(self):
        pos = _spring_positions(np.zeros((6, 6)))
        assert np.isfinite(pos).all()

    def test_connected_nodes_are_closer_than_unconnected(self):
        pos = _spring_positions(self._two_triangles())
        within = np.linalg.norm(pos[0] - pos[1])
        across = np.linalg.norm(pos[0] - pos[5])
        assert within < across

    def test_gravity_keeps_isolates_in_frame(self):
        # Isolated nodes feel only repulsion; without gravity they accelerate off
        # to the rim and squash the connected structure into a dot.
        pos = _spring_positions(self._two_triangles())
        radii = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        assert radii.max() < 10 * np.median(radii)

    def test_deterministic(self):
        weights = self._two_triangles()
        assert np.allclose(_spring_positions(weights), _spring_positions(weights))

    def test_spread_loosens_the_connected_core(self):
        weights = self._two_triangles()

        def closest_gap(spread):
            pos = _spring_positions(weights, spread=spread)
            extent = float(np.ptp(pos, axis=0).max())
            gaps = [float(np.min(np.linalg.norm(pos[i] - np.delete(pos, i, axis=0), axis=1))) for i in range(len(pos))]
            return min(gaps) / extent

        assert closest_gap(3.0) > closest_gap(1.0)

    def test_tuning_is_reachable_from_plot(self, sample_sim_matrix):
        # gravity/spread used to be swallowed by **kwargs, leaving them unusable.
        tight = plot_similarity_graph(sample_sim_matrix, threshold=0.3, gravity=0.3, spread=1.0)
        loose = plot_similarity_graph(sample_sim_matrix, threshold=0.3, gravity=1.5, spread=3.0)
        assert tight.axes[0].collections[0].get_offsets().data.tolist() != (
            loose.axes[0].collections[0].get_offsets().data.tolist()
        )
        import matplotlib.pyplot as plt

        plt.close(tight)
        plt.close(loose)


# ---------------------------------------------------------------------------
# Synteny diagram
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_doc_chunk_sim():
    """ChunkSimilarityMatrix for two documents with 8 and 6 chunks."""
    rng = np.random.RandomState(42)
    embs1 = rng.randn(8, 32).astype(np.float32)
    embs2 = rng.randn(6, 32).astype(np.float32)
    norms1 = np.linalg.norm(embs1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embs2, axis=1, keepdims=True)
    matrix = (embs1 / np.maximum(norms1, 1e-8)) @ (embs2 / np.maximum(norms2, 1e-8)).T
    matrix = (matrix + 1.0) / 2.0
    return ChunkSimilarityMatrix(
        matrix=matrix,
        doc_id_1="doc_alpha",
        doc_id_2="doc_beta",
        chunk_indices_1=list(range(8)),
        chunk_indices_2=list(range(6)),
    )


@pytest.fixture
def self_chunk_sim():
    """ChunkSimilarityMatrix for self-comparison with 10 chunks."""
    rng = np.random.RandomState(7)
    embs = rng.randn(10, 32).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.maximum(norms, 1e-8)
    matrix = (embs_norm @ embs_norm.T + 1.0) / 2.0
    return ChunkSimilarityMatrix(
        matrix=matrix,
        doc_id_1="doc_self",
        doc_id_2="doc_self",
        chunk_indices_1=list(range(10)),
        chunk_indices_2=list(range(10)),
    )


@pytest.mark.unit
class TestPlotSynteny:
    def test_horizontal(self, cross_doc_chunk_sim):
        fig = plot_synteny(cross_doc_chunk_sim, similarity_threshold=0.4)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_vertical(self, cross_doc_chunk_sim):
        fig = plot_synteny(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            orientation="vertical",
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_labels(self, cross_doc_chunk_sim):
        fig = plot_synteny(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            chunk_labels=True,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_high_threshold_no_ribbons(self, cross_doc_chunk_sim):
        """Threshold of 1.0 should produce a figure with only chunk bars."""
        fig = plot_synteny(cross_doc_chunk_sim, similarity_threshold=1.0)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_title_and_cmap(self, cross_doc_chunk_sim):
        fig = plot_synteny(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            title="Custom Title",
            cmap="plasma",
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_text_labels_are_drawn(self, cross_doc_chunk_sim, orientation):
        fig = plot_synteny(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            orientation=orientation,
            labels_1=[f"Section {i}" for i in range(8)],
            labels_2=[f"Part {j}" for j in range(6)],
        )
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert "Section 3" in drawn
        assert "Part 5" in drawn
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_labels_override_chunk_labels(self, cross_doc_chunk_sim):
        fig = plot_synteny(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            chunk_labels=True,
            labels_1=[f"S{i}" for i in range(8)],
        )
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert "S0" in drawn
        # labels_2 was not given, so the second track falls back to its indices.
        assert "5" in drawn
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_label_length_mismatch_raises(self, cross_doc_chunk_sim):
        with pytest.raises(ValueError, match="expected 8 chunk labels, got 3"):
            plot_synteny(cross_doc_chunk_sim, labels_1=["a", "b", "c"])


@pytest.mark.unit
class TestPlotChord:
    def test_basic_chord(self, self_chunk_sim):
        fig = plot_chord(self_chunk_sim, similarity_threshold=0.4)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_labels(self, self_chunk_sim):
        fig = plot_chord(
            self_chunk_sim,
            similarity_threshold=0.4,
            chunk_labels=True,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_min_chunk_distance(self, self_chunk_sim):
        """Higher min distance should produce fewer chords."""
        fig = plot_chord(
            self_chunk_sim,
            similarity_threshold=0.3,
            min_chunk_distance=5,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_rejects_cross_document(self, cross_doc_chunk_sim):
        with pytest.raises(ValueError, match="self-comparison"):
            plot_chord(cross_doc_chunk_sim)

    def test_text_labels_are_drawn(self, self_chunk_sim):
        n = self_chunk_sim.matrix.shape[0]
        fig = plot_chord(
            self_chunk_sim,
            similarity_threshold=0.4,
            labels=[f"Heading {i}" for i in range(n)],
        )
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert f"Heading {n - 1}" in drawn
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_label_length_mismatch_raises(self, self_chunk_sim):
        with pytest.raises(ValueError, match="chunk labels"):
            plot_chord(self_chunk_sim, labels=["only-one"])


# ---------------------------------------------------------------------------
# Interactive (plotly) ribbon diagrams
#
# These had no coverage at all, which is how `chunk_labels` sat here for so long
# accepted-and-ignored. Assert on rendered annotation text, not just that a
# figure comes back, or a silently-dropped label looks identical to a drawn one.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInteractiveRibbons:
    @staticmethod
    def _annotations(fig):
        return {a.text for a in fig.layout.annotations}

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_synteny_chunk_labels_draw_indices(self, cross_doc_chunk_sim, orientation):
        from localvectordb.visualization import plot_synteny_interactive

        fig = plot_synteny_interactive(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            orientation=orientation,
            chunk_labels=True,
        )
        drawn = self._annotations(fig)
        assert "0" in drawn and "7" in drawn  # doc 1 has 8 chunks
        assert "5" in drawn  # doc 2 has 6

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_synteny_text_labels(self, cross_doc_chunk_sim, orientation):
        from localvectordb.visualization import plot_synteny_interactive

        fig = plot_synteny_interactive(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            orientation=orientation,
            labels_1=[f"Section {i}" for i in range(8)],
            labels_2=[f"Part {j}" for j in range(6)],
        )
        drawn = self._annotations(fig)
        assert "Section 3" in drawn
        assert "Part 5" in drawn

    def test_synteny_labels_reach_hover_text(self, cross_doc_chunk_sim):
        from localvectordb.visualization import plot_synteny_interactive

        fig = plot_synteny_interactive(
            cross_doc_chunk_sim,
            similarity_threshold=0.4,
            labels_1=[f"Section {i}" for i in range(8)],
        )
        hover = [t for trace in fig.data for t in (trace.hovertext or [])]
        assert any("Section 3" in h for h in hover)

    def test_synteny_no_labels_by_default(self, cross_doc_chunk_sim):
        from localvectordb.visualization import plot_synteny_interactive

        fig = plot_synteny_interactive(cross_doc_chunk_sim, similarity_threshold=0.4)
        # Only the two document names should be annotated.
        assert self._annotations(fig) == {"doc_alpha", "doc_beta"}

    def test_synteny_label_length_mismatch_raises(self, cross_doc_chunk_sim):
        from localvectordb.visualization import plot_synteny_interactive

        with pytest.raises(ValueError, match="expected 8 chunk labels, got 2"):
            plot_synteny_interactive(cross_doc_chunk_sim, labels_1=["a", "b"])

    def test_chord_chunk_labels_draw_indices(self, self_chunk_sim):
        from localvectordb.visualization import plot_chord_interactive

        n = self_chunk_sim.matrix.shape[0]
        fig = plot_chord_interactive(self_chunk_sim, similarity_threshold=0.4, chunk_labels=True)
        assert {"0", str(n - 1)} <= self._annotations(fig)

    def test_chord_text_labels(self, self_chunk_sim):
        from localvectordb.visualization import plot_chord_interactive

        n = self_chunk_sim.matrix.shape[0]
        fig = plot_chord_interactive(
            self_chunk_sim,
            similarity_threshold=0.4,
            labels=[f"Heading {i}" for i in range(n)],
        )
        drawn = self._annotations(fig)
        assert f"Heading {n - 1}" in drawn
        hover = [t for trace in fig.data for t in (trace.hovertext or [])]
        assert any("Heading 0" in h for h in hover)

    def test_chord_no_labels_by_default(self, self_chunk_sim):
        from localvectordb.visualization import plot_chord_interactive

        fig = plot_chord_interactive(self_chunk_sim, similarity_threshold=0.4)
        assert self._annotations(fig) == set()

    def test_chord_label_length_mismatch_raises(self, self_chunk_sim):
        from localvectordb.visualization import plot_chord_interactive

        with pytest.raises(ValueError, match="chunk labels"):
            plot_chord_interactive(self_chunk_sim, labels=["only-one"])

    def test_chord_rejects_cross_document(self, cross_doc_chunk_sim):
        from localvectordb.visualization import plot_chord_interactive

        with pytest.raises(ValueError, match="self-comparison"):
            plot_chord_interactive(cross_doc_chunk_sim)

    def test_empty(self):
        csm = ChunkSimilarityMatrix(
            matrix=np.array([]).reshape(0, 0),
            doc_id_1="empty",
            doc_id_2="empty",
            chunk_indices_1=[],
            chunk_indices_2=[],
        )
        fig = plot_chord(csm)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
