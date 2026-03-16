# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/types.py
"""Dataclasses for the visualization module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class EmbeddingProjection:
    """Result of dimensionality reduction.

    Attributes
    ----------
    coordinates : np.ndarray
        (N, n_components) projected coordinates.
    method : str
        Reduction method used (``"pca"`` or ``"tsne"``).
    doc_ids : list of str
        Document IDs corresponding to each row.
    transformer : Any
        Fitted transformer object (PCA instance or dict with params).
        Used to project new points into the same space.
    n_components : int
        Number of output dimensions.
    explained_variance : Optional[np.ndarray]
        Explained variance ratio (PCA only).
    """

    coordinates: np.ndarray
    method: str
    doc_ids: List[str]
    transformer: Any = None
    n_components: int = 2
    explained_variance: Optional[np.ndarray] = None


@dataclass
class ClusterResult:
    """Result of clustering.

    Attributes
    ----------
    labels : np.ndarray
        (N,) cluster label for each point.
    n_clusters : int
        Number of clusters.
    centroids : Optional[np.ndarray]
        (K, D) cluster centroids, if available.
    inertia : Optional[float]
        Sum of squared distances to nearest centroid.
    """

    labels: np.ndarray
    n_clusters: int
    centroids: Optional[np.ndarray] = None
    inertia: Optional[float] = None


@dataclass
class QueryOverlay:
    """Overlay for rendering query points on an embedding map.

    Attributes
    ----------
    query_text : str
        The query string (used for legend/labels).
    query_embedding : np.ndarray
        (D,) embedding vector of the query.
    scores : np.ndarray
        (N,) similarity score per document; used for dot sizing.
    """

    query_text: str
    query_embedding: np.ndarray
    scores: np.ndarray
