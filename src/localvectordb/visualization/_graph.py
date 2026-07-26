"""Similarity graph construction and visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import MDS

from localvectordb.core import DocumentSimilarityMatrix

# Spring-layout defaults, shared by the layout function and the plot that calls
# it. Both trade label legibility against how sharply clusters separate; see
# _spring_positions for what moving them does.
SPRING_GRAVITY = 1.0
SPRING_SPREAD = 2.5


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


def _spring_positions(
    weights: np.ndarray,
    iterations: int = 250,
    gravity: float = SPRING_GRAVITY,
    spread: float = SPRING_SPREAD,
    seed: int = 42,
) -> np.ndarray:
    """Fruchterman-Reingold force-directed layout over a weighted adjacency matrix.

    Every node repels every other; connected pairs attract in proportion to
    their edge weight. Unlike an MDS embedding of the full similarity matrix,
    this only feels the edges that survived the threshold, so components
    separate and isolated nodes drift to the rim instead of being pinned by
    similarities the plot never draws.

    Implemented here rather than pulled from ``networkx`` to keep the
    visualization extra at ``scikit-learn`` + ``matplotlib``.

    Parameters
    ----------
    weights : np.ndarray
        (N, N) symmetric non-negative edge weights; zero means no edge.
    iterations : int
        Number of relaxation steps.
    gravity : float
        Strength of a weak pull toward the centre. An unconnected node feels
        only repulsion, so without this it accelerates off to the rim and
        squashes the connected structure into the middle of the plot. Raising it
        also evens out node spacing, which matters when nodes carry text labels.
    spread : float
        Multiplier on the ideal edge length. Textbook Fruchterman-Reingold is
        ``1.0``, which packs a densely connected component tight enough that
        node labels collide; the default loosens it so labels stay readable.

        Both knobs trade legibility against contrast. On an 18-node graph,
        raising them from (0.6, 1.0) to the defaults roughly doubles the closest
        node-to-node gap and grows the connected component from ~1/3 to ~1/2 of
        the frame, while unconnected pairs sit ~3.8x further apart than
        connected ones instead of ~6.3x. Push them much further and the
        clustering the layout exists to show starts to wash out.
    seed : int
        Seed for the initial layout, so the result is reproducible.

    Returns
    -------
    np.ndarray
        (N, 2) positions.
    """
    n = weights.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))

    k = 1.0 / np.sqrt(n)  # ideal edge length for a unit-area canvas
    temperature = 0.1
    cooling = temperature / (iterations + 1)

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]  # (n, n, 2)
        # Floor the distance before dividing: two nodes that land on top of each
        # other would otherwise divide by zero. The diagonal is zeroed out of the
        # force below rather than set to inf, because inf * a zero self-weight is
        # NaN and would poison the whole sum.
        distance = np.clip(np.linalg.norm(delta, axis=-1), 0.01, None)
        direction = delta / distance[..., None]

        # Repulsion k^2/d everywhere, attraction w*d^2/k along edges.
        magnitude = k * k / distance - weights * distance / (k * spread)
        np.fill_diagonal(magnitude, 0.0)  # a node exerts no force on itself
        displacement = (direction * magnitude[..., None]).sum(axis=1)
        displacement -= gravity * (pos - pos.mean(axis=0))

        # Cap each step at the current temperature: large early moves that
        # settle into small ones, which is what keeps the layout from exploding.
        step = np.linalg.norm(displacement, axis=1)
        step = np.clip(step, 1e-9, None)
        pos += displacement / step[:, None] * np.minimum(step, temperature)[:, None]
        temperature -= cooling

    return pos


def plot_similarity_graph(
    sim_matrix: DocumentSimilarityMatrix,
    threshold: float = 0.3,
    layout: str = "spring",
    title: str = "Document Similarity Graph",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    gravity: Optional[float] = None,
    spread: Optional[float] = None,
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
        ``"spring"`` (default) runs a force-directed Fruchterman-Reingold
        layout over the thresholded edges, which groups connected documents and
        pushes unconnected ones apart. ``"mds"`` instead embeds the full
        similarity matrix with multidimensional scaling, placing every node by
        its distance to every other whether or not an edge is drawn.
    title : str
        Plot title.
    save_path : str or Path, optional
        Save path.
    figsize : tuple
        Figure size.
    gravity, spread : float, optional
        Tuning for ``layout="spring"``; see :func:`_spring_positions`. Both trade
        legibility against contrast -- raising them loosens a densely connected
        component so node labels stay readable, at the cost of how sharply
        connected nodes separate from unconnected ones. Ignored for
        ``layout="mds"``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``layout`` is not ``"spring"`` or ``"mds"``.
    """
    if layout not in ("spring", "mds"):
        raise ValueError(f"Unknown layout '{layout}'. Use 'spring' or 'mds'.")

    n = len(sim_matrix.doc_ids)
    fig, ax = plt.subplots(figsize=figsize)

    if n == 0:
        ax.set_title(title)
        return fig

    graph = build_similarity_graph(sim_matrix, threshold=threshold)

    if n == 1:
        positions = np.array([[0.0, 0.0]])
    elif n == 2:
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    elif layout == "spring":
        weights = np.where(sim_matrix.matrix >= threshold, sim_matrix.matrix, 0.0).astype(float)
        weights = (weights + weights.T) / 2.0
        np.fill_diagonal(weights, 0.0)
        positions = _spring_positions(
            weights,
            gravity=SPRING_GRAVITY if gravity is None else gravity,
            spread=SPRING_SPREAD if spread is None else spread,
        )
    else:
        dissimilarity = 1.0 - sim_matrix.matrix
        np.fill_diagonal(dissimilarity, 0.0)
        # Symmetrise just in case
        dissimilarity = (dissimilarity + dissimilarity.T) / 2.0
        mds = MDS(n_components=2, metric="precomputed", random_state=42, normalized_stress="auto", n_init=1)
        positions = mds.fit_transform(dissimilarity)

    # Draw edges
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
