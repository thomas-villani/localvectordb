# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Similarity graph construction and visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import MDS

from localvectordb.core import DocumentSimilarityMatrix


def build_similarity_graph(
    sim_matrix: DocumentSimilarityMatrix,
    threshold: float = 0.3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a graph structure from a similarity matrix.

    Parameters
    ----------
    sim_matrix : DocumentSimilarityMatrix
        Pairwise document similarity matrix.
    threshold : float
        Minimum similarity for an edge to be included.

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` where each node is
        ``{"id": str, "index": int}`` and each edge is
        ``{"source": str, "target": str, "weight": float}``.
    """
    nodes = [{"id": doc_id, "index": i} for i, doc_id in enumerate(sim_matrix.doc_ids)]
    edges = []
    n = len(sim_matrix.doc_ids)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim_matrix.matrix[i, j])
            if w >= threshold:
                edges.append(
                    {
                        "source": sim_matrix.doc_ids[i],
                        "target": sim_matrix.doc_ids[j],
                        "weight": w,
                    }
                )
    return {"nodes": nodes, "edges": edges}


def plot_similarity_graph(
    sim_matrix: DocumentSimilarityMatrix,
    threshold: float = 0.3,
    layout: str = "spring",
    title: str = "Document Similarity Graph",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    **kwargs,
) -> Figure:
    """Visualise documents as a similarity graph.

    Nodes represent documents; edges connect documents with similarity above
    *threshold*.  Edge width and opacity are proportional to similarity.

    Layout uses scikit-learn MDS to avoid a ``networkx`` dependency.

    Parameters
    ----------
    sim_matrix : DocumentSimilarityMatrix
        Pairwise similarity matrix.
    threshold : float
        Edge threshold.
    layout : str
        Layout algorithm (``"spring"`` uses MDS on dissimilarity).
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
    n = len(sim_matrix.doc_ids)
    fig, ax = plt.subplots(figsize=figsize)

    if n == 0:
        ax.set_title(title)
        return fig

    # Compute layout via MDS on dissimilarity
    dissimilarity = 1.0 - sim_matrix.matrix
    np.fill_diagonal(dissimilarity, 0.0)
    # Symmetrise just in case
    dissimilarity = (dissimilarity + dissimilarity.T) / 2.0

    if n == 1:
        positions = np.array([[0.0, 0.0]])
    elif n == 2:
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    else:
        mds = MDS(n_components=2, metric="precomputed", random_state=42, normalized_stress="auto", n_init=1)
        positions = mds.fit_transform(dissimilarity)

    # Draw edges
    graph = build_similarity_graph(sim_matrix, threshold=threshold)
    id_to_idx = {doc_id: i for i, doc_id in enumerate(sim_matrix.doc_ids)}
    max_weight = max((e["weight"] for e in graph["edges"]), default=1.0)

    for edge in graph["edges"]:
        i = id_to_idx[edge["source"]]
        j = id_to_idx[edge["target"]]
        w = edge["weight"]
        linewidth = 0.5 + 3.0 * (w / max_weight)
        alpha = 0.2 + 0.6 * (w / max_weight)
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="steelblue",
            linewidth=linewidth,
            alpha=alpha,
        )

    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1], s=100, zorder=5, edgecolors="black", linewidth=0.5)

    for i, doc_id in enumerate(sim_matrix.doc_ids):
        ax.annotate(
            doc_id,
            (positions[i, 0], positions[i, 1]),
            fontsize=8,
            ha="center",
            va="bottom",
            xytext=(0, 8),
            textcoords="offset points",
        )

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
