# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/_plots.py
"""Static matplotlib plots for embedding visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from localvectordb.core import DocumentSimilarityMatrix
from localvectordb.visualization._dimensionality import project_new_points
from localvectordb.visualization.types import ClusterResult, EmbeddingProjection, QueryOverlay


def plot_embedding_map(
    projection: EmbeddingProjection,
    color_by: Optional[List[str]] = None,
    title: str = "Document Embedding Map",
    save_path: Optional[Union[str, Path]] = None,
    queries: Optional[List[QueryOverlay]] = None,
    figsize: tuple = (10, 8),
    **kwargs,
) -> Figure:
    """Scatter plot of projected document embeddings.

    Parameters
    ----------
    projection : EmbeddingProjection
        Dimensionality-reduced coordinates.
    color_by : list of str, optional
        Category labels for colouring each point.
    title : str
        Plot title.
    save_path : str or Path, optional
        If provided, save figure to this path.
    queries : list of QueryOverlay, optional
        Query overlays to display on the map.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    coords = projection.coordinates

    if coords.shape[0] == 0:
        ax.set_title(title)
        return fig

    # Base dot sizes
    base_size = 40
    sizes = np.full(coords.shape[0], base_size, dtype=float)

    # Scale sizes by query relevance
    if queries:
        max_scores = np.zeros(coords.shape[0])
        for q in queries:
            max_scores = np.maximum(max_scores, q.scores)
        # Scale: base_size to 4x for high scores
        sizes = base_size + max_scores * base_size * 3

    if color_by is not None:
        unique_labels = sorted(set(color_by))
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        colors = [label_to_idx[lbl] for lbl in color_by]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, cmap=cmap, alpha=0.7, edgecolors="w", linewidth=0.5)
        # Legend
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=8, label=lbl)
            for i, lbl in enumerate(unique_labels)
        ]
        ax.legend(handles=handles, title="Category", loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, alpha=0.7, edgecolors="w", linewidth=0.5)

    # Add doc_id labels
    for i, doc_id in enumerate(projection.doc_ids):
        ax.annotate(doc_id, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.6)

    # Overlay query points
    if queries:
        query_markers = ["*", "D", "^", "s", "P", "X"]
        for qi, q in enumerate(queries):
            q_emb = q.query_embedding.reshape(1, -1)
            q_coords = project_new_points(projection, q_emb)
            marker = query_markers[qi % len(query_markers)]
            ax.scatter(
                q_coords[0, 0],
                q_coords[0, 1],
                marker=marker,
                s=200,
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
                label=f"Q: {q.query_text[:30]}",
            )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_similarity_matrix(
    sim_matrix: DocumentSimilarityMatrix,
    title: str = "Document Similarity Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> Figure:
    """Heatmap of pairwise document similarities.

    Parameters
    ----------
    sim_matrix : DocumentSimilarityMatrix
        Similarity matrix to plot.
    title : str
        Plot title.
    save_path : str or Path, optional
        Save path.
    figsize : tuple, optional
        Figure size. Auto-scaled if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(sim_matrix.doc_ids)
    if figsize is None:
        figsize = (max(8, n * 0.6), max(6, n * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    if n == 0:
        ax.set_title(title)
        return fig

    im = ax.imshow(sim_matrix.matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sim_matrix.doc_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sim_matrix.doc_ids, fontsize=8)
    ax.set_title(title)

    # Annotate cells if matrix is small enough
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = sim_matrix.matrix[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_clusters(
    projection: EmbeddingProjection,
    clusters: ClusterResult,
    title: str = "Document Clusters",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    **kwargs,
) -> Figure:
    """Scatter plot of projected embeddings coloured by cluster.

    Parameters
    ----------
    projection : EmbeddingProjection
        Dimensionality-reduced coordinates.
    clusters : ClusterResult
        Cluster assignments.
    title : str
        Plot title.
    save_path : str or Path, optional
        Save path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    coords = projection.coordinates

    if coords.shape[0] == 0:
        ax.set_title(title)
        return fig

    cmap = plt.colormaps.get_cmap("tab10").resampled(clusters.n_clusters)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters.labels,
        cmap=cmap,
        s=50,
        alpha=0.7,
        edgecolors="w",
        linewidth=0.5,
    )

    # Label points
    for i, doc_id in enumerate(projection.doc_ids):
        ax.annotate(doc_id, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.6)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=8, label=f"Cluster {i}")
        for i in range(clusters.n_clusters)
    ]
    ax.legend(handles=handles, title="Cluster", loc="best")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
