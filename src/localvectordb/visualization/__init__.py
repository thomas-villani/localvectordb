# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/__init__.py
"""
Visualization module for LocalVectorDB.

Provides dimensionality reduction, clustering, and plotting utilities
for exploring document embedding spaces.

Optional dependencies:
    - ``scikit-learn`` and ``matplotlib`` (``pip install localvectordb[visualization]``)
    - ``plotly`` for interactive plots (``pip install localvectordb[visualization-interactive]``)
"""

from __future__ import annotations


def _check_visualization_deps() -> None:
    """Raise a friendly error when visualization dependencies are missing."""
    try:
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Visualization features require scikit-learn and matplotlib. "
            "Install them with: pip install localvectordb[visualization]"
        ) from exc


_check_visualization_deps()

# Re-export public API
from localvectordb.visualization._clustering import cluster_embeddings, find_optimal_clusters  # noqa: E402
from localvectordb.visualization._dimensionality import reduce_dimensions  # noqa: E402
from localvectordb.visualization._graph import build_similarity_graph, plot_similarity_graph  # noqa: E402
from localvectordb.visualization._plots import plot_clusters, plot_embedding_map, plot_similarity_matrix  # noqa: E402
from localvectordb.visualization.types import ClusterResult, EmbeddingProjection, QueryOverlay  # noqa: E402

# Interactive (optional, imported lazily)
_PLOTLY_AVAILABLE = False
try:
    import plotly  # noqa: F401

    _PLOTLY_AVAILABLE = True
except ImportError:
    pass


def plot_embedding_map_interactive(*args, **kwargs):
    """Interactive plotly embedding map. Requires ``plotly``."""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Interactive plots require plotly. " "Install with: pip install localvectordb[visualization-interactive]"
        )
    from localvectordb.visualization._interactive import plot_embedding_map_interactive as _impl

    return _impl(*args, **kwargs)


def plot_similarity_matrix_interactive(*args, **kwargs):
    """Interactive plotly similarity heatmap. Requires ``plotly``."""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Interactive plots require plotly. " "Install with: pip install localvectordb[visualization-interactive]"
        )
    from localvectordb.visualization._interactive import plot_similarity_matrix_interactive as _impl

    return _impl(*args, **kwargs)


def plot_clusters_interactive(*args, **kwargs):
    """Interactive plotly cluster plot. Requires ``plotly``."""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Interactive plots require plotly. " "Install with: pip install localvectordb[visualization-interactive]"
        )
    from localvectordb.visualization._interactive import plot_clusters_interactive as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "reduce_dimensions",
    "cluster_embeddings",
    "find_optimal_clusters",
    "plot_embedding_map",
    "plot_similarity_matrix",
    "plot_clusters",
    "plot_similarity_graph",
    "build_similarity_graph",
    "plot_embedding_map_interactive",
    "plot_similarity_matrix_interactive",
    "plot_clusters_interactive",
    "EmbeddingProjection",
    "ClusterResult",
    "QueryOverlay",
]
