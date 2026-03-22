# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/_ribbons.py
"""Synteny and chord diagram visualisations for chunk-level document comparison.

Synteny diagrams show two documents as parallel bars of chunk segments with
Bezier ribbons connecting regions of high similarity -- analogous to synteny
plots in comparative genomics.

Chord (Circos-style) diagrams arrange a single document's chunks around a
circle with interior ribbons connecting self-similar regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from localvectordb.core import ChunkSimilarityMatrix

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _alpha_from_similarity(similarity: float, threshold: float) -> float:
    """Map similarity to alpha, scaling [threshold, 1] -> [0.15, 0.85]."""
    if similarity < threshold:
        return 0.0
    t = (similarity - threshold) / max(1.0 - threshold, 1e-8)
    return 0.15 + 0.70 * t


def _make_ribbon_path_horizontal(
    x1_left: float,
    x1_right: float,
    y_top: float,
    x2_left: float,
    x2_right: float,
    y_bottom: float,
) -> MPath:
    """Return a cubic-Bezier ribbon path between two horizontal segments."""
    y_mid = (y_top + y_bottom) / 2.0
    verts = [
        (x1_left, y_top),
        (x1_left, y_mid),
        (x2_left, y_mid),
        (x2_left, y_bottom),
        (x2_right, y_bottom),
        (x2_right, y_mid),
        (x1_right, y_mid),
        (x1_right, y_top),
        (x1_left, y_top),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    return MPath(verts, codes)


def _make_ribbon_path_vertical(
    y1_top: float,
    y1_bottom: float,
    x_left: float,
    y2_top: float,
    y2_bottom: float,
    x_right: float,
) -> MPath:
    """Return a cubic-Bezier ribbon path between two vertical segments."""
    x_mid = (x_left + x_right) / 2.0
    verts = [
        (x_left, y1_top),
        (x_mid, y1_top),
        (x_mid, y2_top),
        (x_right, y2_top),
        (x_right, y2_bottom),
        (x_mid, y2_bottom),
        (x_mid, y1_bottom),
        (x_left, y1_bottom),
        (x_left, y1_top),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    return MPath(verts, codes)


def _make_chord_path(
    inner_r: float,
    theta_i_left: float,
    theta_i_right: float,
    theta_j_left: float,
    theta_j_right: float,
) -> MPath:
    """Return a quadratic-Bezier chord path through the circle interior."""
    p_i_left = (inner_r * np.cos(theta_i_left), inner_r * np.sin(theta_i_left))
    p_i_right = (inner_r * np.cos(theta_i_right), inner_r * np.sin(theta_i_right))
    p_j_left = (inner_r * np.cos(theta_j_left), inner_r * np.sin(theta_j_left))
    p_j_right = (inner_r * np.cos(theta_j_right), inner_r * np.sin(theta_j_right))
    ctrl = (0.0, 0.0)

    verts = [
        p_i_left,
        ctrl,
        p_j_right,
        p_j_left,
        ctrl,
        p_i_right,
        p_i_left,
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.LINETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.CLOSEPOLY,
    ]
    return MPath(verts, codes)


# ------------------------------------------------------------------ #
# Synteny diagram                                                      #
# ------------------------------------------------------------------ #


def plot_synteny(
    chunk_sim: ChunkSimilarityMatrix,
    similarity_threshold: float = 0.7,
    orientation: str = "horizontal",
    chunk_labels: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "viridis",
    **kwargs,
) -> Figure:
    """Synteny ribbon diagram comparing chunks of two documents.

    Two parallel bars represent the documents, with Bezier ribbons connecting
    chunks of high similarity -- analogous to synteny plots in comparative
    genomics.

    Parameters
    ----------
    chunk_sim : ChunkSimilarityMatrix
        Full chunk-level similarity matrix.
    similarity_threshold : float
        Minimum similarity for a ribbon to be drawn.
    orientation : str
        ``"horizontal"`` (doc1 top, doc2 bottom) or ``"vertical"``
        (doc1 left, doc2 right).
    chunk_labels : bool
        If ``True``, label each chunk segment with its index.
    title : str, optional
        Plot title.  Auto-generated if ``None``.
    save_path : str or Path, optional
        Save figure to this path.
    figsize : tuple, optional
        Figure size.  Auto-scaled if ``None``.
    cmap : str
        Matplotlib colormap for chunk position colouring.

    Returns
    -------
    matplotlib.figure.Figure
    """
    matrix = chunk_sim.matrix
    n1, n2 = matrix.shape

    if title is None:
        title = f"Synteny: {chunk_sim.doc_id_1} vs {chunk_sim.doc_id_2}"

    colormap = plt.colormaps.get_cmap(cmap)

    if orientation == "horizontal":
        if figsize is None:
            figsize = (max(10, max(n1, n2) * 0.8), 6)
        fig, ax = plt.subplots(figsize=figsize)
        _draw_synteny_horizontal(ax, chunk_sim, matrix, n1, n2, colormap, similarity_threshold, chunk_labels)
    else:
        if figsize is None:
            figsize = (6, max(10, max(n1, n2) * 0.8))
        fig, ax = plt.subplots(figsize=figsize)
        _draw_synteny_vertical(ax, chunk_sim, matrix, n1, n2, colormap, similarity_threshold, chunk_labels)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def _draw_synteny_horizontal(ax, chunk_sim, matrix, n1, n2, colormap, threshold, chunk_labels):
    """Draw horizontal synteny: doc1 on top, doc2 on bottom."""
    bar_height = 0.08
    y_top = 1.0
    y_bottom = 0.0

    width1 = 1.0 / n1 if n1 > 0 else 1.0
    width2 = 1.0 / n2 if n2 > 0 else 1.0

    # Doc1 chunks (top)
    for i in range(n1):
        x = i * width1
        color = colormap(i / max(n1 - 1, 1))
        rect = Rectangle(
            (x, y_top - bar_height),
            width1,
            bar_height,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        if chunk_labels:
            ax.text(
                x + width1 / 2,
                y_top - bar_height / 2,
                str(chunk_sim.chunk_indices_1[i]),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    # Doc2 chunks (bottom)
    for j in range(n2):
        x = j * width2
        color = colormap(j / max(n2 - 1, 1))
        rect = Rectangle(
            (x, y_bottom),
            width2,
            bar_height,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        if chunk_labels:
            ax.text(
                x + width2 / 2,
                y_bottom + bar_height / 2,
                str(chunk_sim.chunk_indices_2[j]),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    # Ribbons
    ribbon_top = y_top - bar_height
    ribbon_bottom = y_bottom + bar_height

    for i in range(n1):
        for j in range(n2):
            sim = float(matrix[i, j])
            if sim < threshold:
                continue
            alpha = _alpha_from_similarity(sim, threshold)
            color = colormap(i / max(n1 - 1, 1))
            path = _make_ribbon_path_horizontal(
                i * width1,
                (i + 1) * width1,
                ribbon_top,
                j * width2,
                (j + 1) * width2,
                ribbon_bottom,
            )
            ax.add_patch(PathPatch(path, facecolor=(*color[:3], alpha), edgecolor="none"))

    # Labels
    ax.text(0.5, y_top + 0.02, chunk_sim.doc_id_1, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.5, y_bottom - 0.02, chunk_sim.doc_id_2, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.12)


def _draw_synteny_vertical(ax, chunk_sim, matrix, n1, n2, colormap, threshold, chunk_labels):
    """Draw vertical synteny: doc1 on left, doc2 on right."""
    bar_width = 0.08
    x_left = 0.0
    x_right = 1.0

    height1 = 1.0 / n1 if n1 > 0 else 1.0
    height2 = 1.0 / n2 if n2 > 0 else 1.0

    # Doc1 chunks (left bar, top-to-bottom order)
    for i in range(n1):
        y = 1.0 - (i + 1) * height1
        color = colormap(i / max(n1 - 1, 1))
        rect = Rectangle(
            (x_left, y),
            bar_width,
            height1,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        if chunk_labels:
            ax.text(
                x_left + bar_width / 2,
                y + height1 / 2,
                str(chunk_sim.chunk_indices_1[i]),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    # Doc2 chunks (right bar)
    for j in range(n2):
        y = 1.0 - (j + 1) * height2
        color = colormap(j / max(n2 - 1, 1))
        rect = Rectangle(
            (x_right - bar_width, y),
            bar_width,
            height2,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        if chunk_labels:
            ax.text(
                x_right - bar_width / 2,
                y + height2 / 2,
                str(chunk_sim.chunk_indices_2[j]),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    # Ribbons
    ribbon_left = x_left + bar_width
    ribbon_right = x_right - bar_width

    for i in range(n1):
        for j in range(n2):
            sim = float(matrix[i, j])
            if sim < threshold:
                continue
            alpha = _alpha_from_similarity(sim, threshold)
            color = colormap(i / max(n1 - 1, 1))
            y1_top = 1.0 - i * height1
            y1_bottom = 1.0 - (i + 1) * height1
            y2_top = 1.0 - j * height2
            y2_bottom = 1.0 - (j + 1) * height2
            path = _make_ribbon_path_vertical(y1_top, y1_bottom, ribbon_left, y2_top, y2_bottom, ribbon_right)
            ax.add_patch(PathPatch(path, facecolor=(*color[:3], alpha), edgecolor="none"))

    # Labels
    ax.text(
        x_left - 0.02,
        0.5,
        chunk_sim.doc_id_1,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        x_right + 0.02,
        0.5,
        chunk_sim.doc_id_2,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        rotation=270,
    )
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)


# ------------------------------------------------------------------ #
# Chord diagram                                                        #
# ------------------------------------------------------------------ #


def plot_chord(
    chunk_sim: ChunkSimilarityMatrix,
    similarity_threshold: float = 0.7,
    min_chunk_distance: int = 3,
    chunk_labels: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 10),
    cmap: str = "viridis",
    **kwargs,
) -> Figure:
    """Chord (Circos-style) diagram for chunk self-similarity.

    Chunks are arranged as arcs around a circle with interior ribbons
    connecting self-similar regions, analogous to Circos plots in genomics.

    Parameters
    ----------
    chunk_sim : ChunkSimilarityMatrix
        Chunk self-similarity matrix (``doc_id_1 == doc_id_2``).
    similarity_threshold : float
        Minimum similarity for a chord to be drawn.
    min_chunk_distance : int
        Minimum index distance between chunks for a chord to be drawn.
        Filters out trivially similar adjacent chunks.
    chunk_labels : bool
        If ``True``, label each arc segment with its index.
    title : str, optional
        Plot title.  Auto-generated if ``None``.
    save_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size.
    cmap : str
        Matplotlib colormap for chunk position colouring.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if chunk_sim.doc_id_1 != chunk_sim.doc_id_2:
        raise ValueError(
            "Chord diagrams require self-comparison "
            f"(got doc_id_1={chunk_sim.doc_id_1!r}, doc_id_2={chunk_sim.doc_id_2!r}). "
            "Use plot_synteny for cross-document comparison."
        )

    matrix = chunk_sim.matrix
    n = matrix.shape[0]

    if title is None:
        title = f"Self-similarity: {chunk_sim.doc_id_1}"

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    colormap = plt.colormaps.get_cmap(cmap)

    if n == 0:
        ax.set_title(title)
        ax.axis("off")
        return fig

    # Arc geometry
    outer_r = 1.0
    arc_width = 0.08
    inner_r = outer_r - arc_width
    gap_angle = 2.0 * np.pi * 0.01  # 1% of circle per gap
    chunk_angle = (2.0 * np.pi - n * gap_angle) / n

    # Compute arc start/end angles (chunk 0 at 12-o'clock, going clockwise)
    arc_starts = np.empty(n)
    arc_ends = np.empty(n)
    theta = np.pi / 2  # start at 12 o'clock
    for i in range(n):
        arc_starts[i] = theta
        arc_ends[i] = theta - chunk_angle  # clockwise
        theta -= chunk_angle + gap_angle

    # Draw arc segments
    for i in range(n):
        color = colormap(i / max(n - 1, 1))
        theta_range = np.linspace(arc_starts[i], arc_ends[i], 50)
        x_outer = outer_r * np.cos(theta_range)
        y_outer = outer_r * np.sin(theta_range)
        x_inner = inner_r * np.cos(theta_range[::-1])
        y_inner = inner_r * np.sin(theta_range[::-1])
        ax.fill(
            np.concatenate([x_outer, x_inner]),
            np.concatenate([y_outer, y_inner]),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if chunk_labels:
            mid_theta = (arc_starts[i] + arc_ends[i]) / 2
            label_r = outer_r + 0.06
            ax.text(
                label_r * np.cos(mid_theta),
                label_r * np.sin(mid_theta),
                str(chunk_sim.chunk_indices_1[i]),
                ha="center",
                va="center",
                fontsize=6,
            )

    # Draw chords
    ribbon_half = chunk_angle * 0.35
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) < min_chunk_distance:
                continue
            sim = float(matrix[i, j])
            if sim < similarity_threshold:
                continue

            alpha = _alpha_from_similarity(sim, similarity_threshold)
            color = colormap(i / max(n - 1, 1))
            mid_i = (arc_starts[i] + arc_ends[i]) / 2
            mid_j = (arc_starts[j] + arc_ends[j]) / 2

            path = _make_chord_path(
                inner_r,
                mid_i + ribbon_half,
                mid_i - ribbon_half,
                mid_j + ribbon_half,
                mid_j - ribbon_half,
            )
            ax.add_patch(PathPatch(path, facecolor=(*color[:3], alpha), edgecolor="none"))

    margin = outer_r * 0.25
    ax.set_xlim(-outer_r - margin, outer_r + margin)
    ax.set_ylim(-outer_r - margin, outer_r + margin)
    ax.set_title(title, pad=20)
    ax.axis("off")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
