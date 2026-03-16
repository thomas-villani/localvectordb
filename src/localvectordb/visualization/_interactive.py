# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/_interactive.py
"""Interactive plotly-based plots (optional dependency)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from localvectordb.core import DocumentSimilarityMatrix
from localvectordb.visualization._dimensionality import project_new_points
from localvectordb.visualization.types import ClusterResult, EmbeddingProjection, QueryOverlay

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "Interactive plots require plotly. " "Install with: pip install localvectordb[visualization-interactive]"
    ) from exc


def plot_embedding_map_interactive(
    projection: EmbeddingProjection,
    color_by: Optional[List[str]] = None,
    hover_data: Optional[List[str]] = None,
    queries: Optional[List[QueryOverlay]] = None,
    title: str = "Document Embedding Map",
    **kwargs,
) -> go.Figure:
    """Interactive scatter plot of projected embeddings using plotly.

    Parameters
    ----------
    projection : EmbeddingProjection
        Dimensionality-reduced coordinates.
    color_by : list of str, optional
        Category labels for each point.
    hover_data : list of str, optional
        Additional hover text per point.
    queries : list of QueryOverlay, optional
        Query overlays.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    coords = projection.coordinates
    if coords.shape[0] == 0:
        return go.Figure(layout=go.Layout(title=title))

    base_size = 8
    sizes = np.full(coords.shape[0], base_size, dtype=float)

    if queries:
        max_scores = np.zeros(coords.shape[0])
        for q in queries:
            max_scores = np.maximum(max_scores, q.scores)
        sizes = base_size + max_scores * base_size * 3

    fig = go.Figure()

    # Document points
    marker_kwargs = dict(size=sizes, opacity=0.7, line=dict(width=0.5, color="white"))
    if color_by:
        unique_labels = sorted(set(color_by))
        for lbl in unique_labels:
            mask = [i for i, c in enumerate(color_by) if c == lbl]
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers+text",
                    name=lbl,
                    text=[projection.doc_ids[i] for i in mask],
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=sizes[mask], opacity=0.7),
                    hovertext=[hover_data[i] if hover_data else projection.doc_ids[i] for i in mask],
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+text",
                name="Documents",
                text=projection.doc_ids,
                textposition="top center",
                textfont=dict(size=8),
                marker=marker_kwargs,
                hovertext=hover_data or projection.doc_ids,
            )
        )

    # Query overlays
    if queries:
        symbols = ["star", "diamond", "triangle-up", "square", "pentagon", "cross"]
        for qi, q in enumerate(queries):
            q_emb = q.query_embedding.reshape(1, -1)
            q_coords = project_new_points(projection, q_emb)
            fig.add_trace(
                go.Scatter(
                    x=[q_coords[0, 0]],
                    y=[q_coords[0, 1]],
                    mode="markers+text",
                    name=f"Q: {q.query_text[:30]}",
                    text=[q.query_text[:30]],
                    textposition="top center",
                    marker=dict(
                        size=16,
                        symbol=symbols[qi % len(symbols)],
                        line=dict(width=2, color="black"),
                    ),
                )
            )

    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    return fig


def plot_similarity_matrix_interactive(
    sim_matrix: DocumentSimilarityMatrix,
    title: str = "Document Similarity Matrix",
    **kwargs,
) -> go.Figure:
    """Interactive heatmap of pairwise document similarities.

    Parameters
    ----------
    sim_matrix : DocumentSimilarityMatrix
        Similarity matrix.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if len(sim_matrix.doc_ids) == 0:
        return go.Figure(layout=go.Layout(title=title))

    fig = go.Figure(
        data=go.Heatmap(
            z=sim_matrix.matrix,
            x=sim_matrix.doc_ids,
            y=sim_matrix.doc_ids,
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            text=np.round(sim_matrix.matrix, 2).astype(str),
            texttemplate="%{text}",
            hovertemplate="Doc %{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Document", yaxis_title="Document")
    return fig


def plot_clusters_interactive(
    projection: EmbeddingProjection,
    clusters: ClusterResult,
    title: str = "Document Clusters",
    **kwargs,
) -> go.Figure:
    """Interactive scatter plot coloured by cluster.

    Parameters
    ----------
    projection : EmbeddingProjection
        Dimensionality-reduced coordinates.
    clusters : ClusterResult
        Cluster assignments.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    coords = projection.coordinates
    if coords.shape[0] == 0:
        return go.Figure(layout=go.Layout(title=title))

    fig = go.Figure()
    for k in range(clusters.n_clusters):
        mask = clusters.labels == k
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers+text",
                name=f"Cluster {k}",
                text=[projection.doc_ids[i] for i in range(len(projection.doc_ids)) if mask[i]],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=10, opacity=0.7),
            )
        )

    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    return fig
