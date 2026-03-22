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

from localvectordb.core import ChunkSimilarityMatrix, DocumentSimilarityMatrix
from localvectordb.visualization._dimensionality import project_new_points
from localvectordb.visualization._ribbons import _alpha_from_similarity
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


# ------------------------------------------------------------------ #
# Interactive synteny diagram                                          #
# ------------------------------------------------------------------ #


def _rgba_str(r: float, g: float, b: float, a: float) -> str:
    """Convert RGBA floats [0, 1] to a CSS ``rgba()`` string."""
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})"


def plot_synteny_interactive(
    chunk_sim: ChunkSimilarityMatrix,
    similarity_threshold: float = 0.7,
    orientation: str = "horizontal",
    chunk_labels: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Interactive synteny ribbon diagram using Plotly SVG shapes.

    Parameters
    ----------
    chunk_sim : ChunkSimilarityMatrix
        Full chunk-level similarity matrix.
    similarity_threshold : float
        Minimum similarity for a ribbon to be drawn.
    orientation : str
        ``"horizontal"`` or ``"vertical"``.
    chunk_labels : bool
        If ``True``, label each chunk segment with its index.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import matplotlib.pyplot as _plt  # noqa: F811 – for colormap access only

    matrix = chunk_sim.matrix
    n1, n2 = matrix.shape

    if title is None:
        title = f"Synteny: {chunk_sim.doc_id_1} vs {chunk_sim.doc_id_2}"

    fig = go.Figure()
    cmap = _plt.colormaps.get_cmap("viridis")

    if orientation == "horizontal":
        _draw_synteny_horizontal_plotly(fig, chunk_sim, matrix, n1, n2, cmap, similarity_threshold, chunk_labels)
        fig.update_layout(
            xaxis=dict(range=[-0.05, 1.05], visible=False),
            yaxis=dict(range=[-0.15, 1.15], scaleanchor="x", visible=False),
        )
    else:
        _draw_synteny_vertical_plotly(fig, chunk_sim, matrix, n1, n2, cmap, similarity_threshold, chunk_labels)
        fig.update_layout(
            xaxis=dict(range=[-0.15, 1.15], visible=False),
            yaxis=dict(range=[-0.05, 1.05], scaleanchor="x", visible=False),
        )

    fig.update_layout(title=title, showlegend=False)
    return fig


def _draw_synteny_horizontal_plotly(fig, chunk_sim, matrix, n1, n2, cmap, threshold, chunk_labels):
    """Plotly horizontal synteny with SVG path ribbons."""
    bar_height = 0.08
    y_top = 1.0
    y_bottom = 0.0
    width1 = 1.0 / n1 if n1 > 0 else 1.0
    width2 = 1.0 / n2 if n2 > 0 else 1.0

    shapes = []

    # Chunk rectangles
    for i in range(n1):
        x = i * width1
        c = cmap(i / max(n1 - 1, 1))
        shapes.append(
            dict(
                type="rect",
                x0=x,
                y0=y_top - bar_height,
                x1=x + width1,
                y1=y_top,
                fillcolor=_rgba_str(c[0], c[1], c[2], 1.0),
                line=dict(color="white", width=0.5),
            )
        )

    for j in range(n2):
        x = j * width2
        c = cmap(j / max(n2 - 1, 1))
        shapes.append(
            dict(
                type="rect",
                x0=x,
                y0=y_bottom,
                x1=x + width2,
                y1=y_bottom + bar_height,
                fillcolor=_rgba_str(c[0], c[1], c[2], 1.0),
                line=dict(color="white", width=0.5),
            )
        )

    # Ribbons as SVG cubic Bezier paths
    ribbon_top = y_top - bar_height
    ribbon_bottom = y_bottom + bar_height
    y_mid = (ribbon_top + ribbon_bottom) / 2

    for i in range(n1):
        for j in range(n2):
            sim = float(matrix[i, j])
            if sim < threshold:
                continue
            alpha = _alpha_from_similarity(sim, threshold)
            c = cmap(i / max(n1 - 1, 1))
            x1l = i * width1
            x1r = (i + 1) * width1
            x2l = j * width2
            x2r = (j + 1) * width2
            svg = (
                f"M {x1l},{ribbon_top} "
                f"C {x1l},{y_mid} {x2l},{y_mid} {x2l},{ribbon_bottom} "
                f"L {x2r},{ribbon_bottom} "
                f"C {x2r},{y_mid} {x1r},{y_mid} {x1r},{ribbon_top} Z"
            )
            shapes.append(
                dict(
                    type="path",
                    path=svg,
                    fillcolor=_rgba_str(c[0], c[1], c[2], alpha),
                    line=dict(width=0),
                )
            )

    fig.update_layout(shapes=shapes)

    # Hover traces for chunks
    doc1_x = [(i + 0.5) * width1 for i in range(n1)]
    doc1_y = [y_top - bar_height / 2] * n1
    doc1_text = [f"{chunk_sim.doc_id_1} chunk {chunk_sim.chunk_indices_1[i]}" for i in range(n1)]
    fig.add_trace(
        go.Scatter(
            x=doc1_x,
            y=doc1_y,
            mode="markers",
            marker=dict(size=1, opacity=0),
            hovertext=doc1_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    doc2_x = [(j + 0.5) * width2 for j in range(n2)]
    doc2_y = [y_bottom + bar_height / 2] * n2
    doc2_text = [f"{chunk_sim.doc_id_2} chunk {chunk_sim.chunk_indices_2[j]}" for j in range(n2)]
    fig.add_trace(
        go.Scatter(
            x=doc2_x,
            y=doc2_y,
            mode="markers",
            marker=dict(size=1, opacity=0),
            hovertext=doc2_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Labels
    fig.add_annotation(
        x=0.5,
        y=y_top + 0.03,
        text=chunk_sim.doc_id_1,
        showarrow=False,
        font=dict(size=12, color="black"),
    )
    fig.add_annotation(
        x=0.5,
        y=y_bottom - 0.03,
        text=chunk_sim.doc_id_2,
        showarrow=False,
        font=dict(size=12, color="black"),
    )


def _draw_synteny_vertical_plotly(fig, chunk_sim, matrix, n1, n2, cmap, threshold, chunk_labels):
    """Plotly vertical synteny with SVG path ribbons."""
    bar_width = 0.08
    x_left = 0.0
    x_right = 1.0
    height1 = 1.0 / n1 if n1 > 0 else 1.0
    height2 = 1.0 / n2 if n2 > 0 else 1.0

    shapes = []

    for i in range(n1):
        y = 1.0 - (i + 1) * height1
        c = cmap(i / max(n1 - 1, 1))
        shapes.append(
            dict(
                type="rect",
                x0=x_left,
                y0=y,
                x1=x_left + bar_width,
                y1=y + height1,
                fillcolor=_rgba_str(c[0], c[1], c[2], 1.0),
                line=dict(color="white", width=0.5),
            )
        )

    for j in range(n2):
        y = 1.0 - (j + 1) * height2
        c = cmap(j / max(n2 - 1, 1))
        shapes.append(
            dict(
                type="rect",
                x0=x_right - bar_width,
                y0=y,
                x1=x_right,
                y1=y + height2,
                fillcolor=_rgba_str(c[0], c[1], c[2], 1.0),
                line=dict(color="white", width=0.5),
            )
        )

    ribbon_left = x_left + bar_width
    ribbon_right = x_right - bar_width
    x_mid = (ribbon_left + ribbon_right) / 2

    for i in range(n1):
        for j in range(n2):
            sim = float(matrix[i, j])
            if sim < threshold:
                continue
            alpha = _alpha_from_similarity(sim, threshold)
            c = cmap(i / max(n1 - 1, 1))
            y1t = 1.0 - i * height1
            y1b = 1.0 - (i + 1) * height1
            y2t = 1.0 - j * height2
            y2b = 1.0 - (j + 1) * height2
            svg = (
                f"M {ribbon_left},{y1t} "
                f"C {x_mid},{y1t} {x_mid},{y2t} {ribbon_right},{y2t} "
                f"L {ribbon_right},{y2b} "
                f"C {x_mid},{y2b} {x_mid},{y1b} {ribbon_left},{y1b} Z"
            )
            shapes.append(
                dict(
                    type="path",
                    path=svg,
                    fillcolor=_rgba_str(c[0], c[1], c[2], alpha),
                    line=dict(width=0),
                )
            )

    fig.update_layout(shapes=shapes)

    fig.add_annotation(
        x=x_left - 0.03,
        y=0.5,
        text=chunk_sim.doc_id_1,
        showarrow=False,
        textangle=-90,
        font=dict(size=12),
    )
    fig.add_annotation(
        x=x_right + 0.03,
        y=0.5,
        text=chunk_sim.doc_id_2,
        showarrow=False,
        textangle=90,
        font=dict(size=12),
    )


# ------------------------------------------------------------------ #
# Interactive chord diagram                                            #
# ------------------------------------------------------------------ #


def plot_chord_interactive(
    chunk_sim: ChunkSimilarityMatrix,
    similarity_threshold: float = 0.7,
    min_chunk_distance: int = 3,
    chunk_labels: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Interactive chord (Circos-style) diagram using Plotly.

    Parameters
    ----------
    chunk_sim : ChunkSimilarityMatrix
        Chunk self-similarity matrix (``doc_id_1 == doc_id_2``).
    similarity_threshold : float
        Minimum similarity for a chord to be drawn.
    min_chunk_distance : int
        Minimum index distance between chunks.
    chunk_labels : bool
        If ``True``, label each arc segment.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import matplotlib.pyplot as _plt  # noqa: F811

    if chunk_sim.doc_id_1 != chunk_sim.doc_id_2:
        raise ValueError("Chord diagrams require self-comparison. Use plot_synteny_interactive for cross-document.")

    matrix = chunk_sim.matrix
    n = matrix.shape[0]

    if title is None:
        title = f"Self-similarity: {chunk_sim.doc_id_1}"

    fig = go.Figure()
    cmap = _plt.colormaps.get_cmap("viridis")

    if n == 0:
        fig.update_layout(title=title)
        return fig

    outer_r = 1.0
    arc_width = 0.08
    inner_r = outer_r - arc_width
    gap_angle = 2.0 * np.pi * 0.01
    chunk_angle = (2.0 * np.pi - n * gap_angle) / n

    arc_starts = np.empty(n)
    arc_ends = np.empty(n)
    theta = np.pi / 2
    for i in range(n):
        arc_starts[i] = theta
        arc_ends[i] = theta - chunk_angle
        theta -= chunk_angle + gap_angle

    shapes = []

    # Arc segments as SVG paths
    for i in range(n):
        c = cmap(i / max(n - 1, 1))
        angles = np.linspace(arc_starts[i], arc_ends[i], 30)
        outer_pts = " ".join(f"L {outer_r * np.cos(a)},{outer_r * np.sin(a)}" for a in angles[1:])
        inner_pts = " ".join(f"L {inner_r * np.cos(a)},{inner_r * np.sin(a)}" for a in reversed(angles[1:]))
        svg = (
            f"M {outer_r * np.cos(angles[0])},{outer_r * np.sin(angles[0])} "
            f"{outer_pts} "
            f"L {inner_r * np.cos(angles[-1])},{inner_r * np.sin(angles[-1])} "
            f"{inner_pts} Z"
        )
        shapes.append(
            dict(
                type="path",
                path=svg,
                fillcolor=_rgba_str(c[0], c[1], c[2], 1.0),
                line=dict(color="white", width=0.5),
            )
        )

    # Chord ribbons
    ribbon_half = chunk_angle * 0.35
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) < min_chunk_distance:
                continue
            sim = float(matrix[i, j])
            if sim < similarity_threshold:
                continue
            alpha = _alpha_from_similarity(sim, similarity_threshold)
            c = cmap(i / max(n - 1, 1))
            mid_i = (arc_starts[i] + arc_ends[i]) / 2
            mid_j = (arc_starts[j] + arc_ends[j]) / 2

            ti_l = mid_i + ribbon_half
            ti_r = mid_i - ribbon_half
            tj_l = mid_j + ribbon_half
            tj_r = mid_j - ribbon_half

            p_il = (inner_r * np.cos(ti_l), inner_r * np.sin(ti_l))
            p_ir = (inner_r * np.cos(ti_r), inner_r * np.sin(ti_r))
            p_jl = (inner_r * np.cos(tj_l), inner_r * np.sin(tj_l))
            p_jr = (inner_r * np.cos(tj_r), inner_r * np.sin(tj_r))

            svg = (
                f"M {p_il[0]},{p_il[1]} "
                f"Q 0,0 {p_jr[0]},{p_jr[1]} "
                f"L {p_jl[0]},{p_jl[1]} "
                f"Q 0,0 {p_ir[0]},{p_ir[1]} Z"
            )
            shapes.append(
                dict(
                    type="path",
                    path=svg,
                    fillcolor=_rgba_str(c[0], c[1], c[2], alpha),
                    line=dict(width=0),
                )
            )

    fig.update_layout(shapes=shapes)

    # Hover traces at arc midpoints
    mid_angles = [(arc_starts[i] + arc_ends[i]) / 2 for i in range(n)]
    hover_r = (outer_r + inner_r) / 2
    hover_x = [hover_r * np.cos(a) for a in mid_angles]
    hover_y = [hover_r * np.sin(a) for a in mid_angles]
    hover_text = [f"Chunk {chunk_sim.chunk_indices_1[i]}" for i in range(n)]

    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=8, opacity=0),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    margin = outer_r * 0.25
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(range=[-outer_r - margin, outer_r + margin], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-outer_r - margin, outer_r + margin], visible=False),
    )
    return fig
