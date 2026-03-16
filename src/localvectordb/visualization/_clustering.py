# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/visualization/_clustering.py
"""Clustering utilities for document embeddings."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from localvectordb.visualization.types import ClusterResult


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "kmeans",
    **kwargs,
) -> ClusterResult:
    """Cluster embeddings using k-means.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) embeddings.
    n_clusters : int, optional
        Number of clusters. If ``None``, determined automatically via
        :func:`find_optimal_clusters`.
    method : str
        Clustering method (currently only ``"kmeans"``).
    **kwargs
        Forwarded to ``KMeans``.

    Returns
    -------
    ClusterResult
    """
    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method '{method}'. Use 'kmeans'.")

    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return ClusterResult(labels=np.array([], dtype=int), n_clusters=0)

    if n_clusters is None:
        n_clusters = find_optimal_clusters(embeddings)

    # Ensure n_clusters doesn't exceed n_samples
    n_clusters = min(n_clusters, n_samples)

    if n_clusters <= 1:
        return ClusterResult(
            labels=np.zeros(n_samples, dtype=int),
            n_clusters=1,
            centroids=np.mean(embeddings, axis=0, keepdims=True),
            inertia=0.0,
        )

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42, **kwargs)
    labels = kmeans.fit_predict(embeddings)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centroids=kmeans.cluster_centers_,
        inertia=float(kmeans.inertia_),
    )


def find_optimal_clusters(
    embeddings: np.ndarray,
    max_k: Optional[int] = None,
) -> int:
    """Determine the optimal number of clusters via silhouette analysis.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) embeddings.
    max_k : int, optional
        Maximum number of clusters to try. Defaults to ``min(10, N - 1)``.

    Returns
    -------
    int
        Optimal cluster count (>= 2, or 1 if too few samples).
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return 1

    if max_k is None:
        max_k = min(10, n_samples - 1)
    max_k = max(2, min(max_k, n_samples - 1))

    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(embeddings, labels))
        if score > best_score:
            best_score = score
            best_k = k

    return best_k
