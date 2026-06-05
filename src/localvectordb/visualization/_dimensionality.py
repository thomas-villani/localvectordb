# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dimensionality reduction utilities (PCA, t-SNE)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from localvectordb.visualization.types import EmbeddingProjection


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    doc_ids: Optional[List[str]] = None,
    **kwargs,
) -> EmbeddingProjection:
    """Project high-dimensional embeddings into a lower-dimensional space.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) array of embeddings.
    method : str
        ``"pca"`` or ``"tsne"``.
    n_components : int
        Number of output dimensions (2 or 3).
    doc_ids : list of str, optional
        Document IDs matching rows.  Defaults to string indices.
    **kwargs
        Forwarded to the underlying sklearn estimator.

    Returns
    -------
    EmbeddingProjection
    """
    if doc_ids is None:
        doc_ids = [str(i) for i in range(len(embeddings))]

    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return EmbeddingProjection(
            coordinates=np.array([]).reshape(0, n_components),
            method=method,
            doc_ids=doc_ids,
            n_components=n_components,
        )

    if method == "pca":
        effective_components = min(n_components, n_samples, embeddings.shape[1])
        pca = PCA(n_components=effective_components, **kwargs)
        coords = pca.fit_transform(embeddings)
        return EmbeddingProjection(
            coordinates=coords,
            method="pca",
            doc_ids=doc_ids,
            transformer=pca,
            n_components=effective_components,
            explained_variance=pca.explained_variance_ratio_,
        )

    if method == "tsne":
        # t-SNE needs perplexity < n_samples
        perplexity = kwargs.pop("perplexity", min(30.0, max(1.0, n_samples - 1)))
        if n_samples <= n_components:
            # Too few samples for t-SNE; fall back to PCA
            return reduce_dimensions(embeddings, method="pca", n_components=n_components, doc_ids=doc_ids, **kwargs)

        tsne = TSNE(n_components=n_components, perplexity=perplexity, **kwargs)
        coords = tsne.fit_transform(embeddings)

        # Store a PCA fallback transformer for projecting new query points
        # (t-SNE cannot natively transform unseen data)
        effective_pca_components = min(n_components, n_samples, embeddings.shape[1])
        pca_fallback = PCA(n_components=effective_pca_components)
        pca_fallback.fit(embeddings)

        transformer = {
            "tsne": tsne,
            "pca_fallback": pca_fallback,
            "original_embeddings": embeddings,
            "coordinates": coords,
        }
        return EmbeddingProjection(
            coordinates=coords,
            method="tsne",
            doc_ids=doc_ids,
            transformer=transformer,
            n_components=n_components,
        )

    raise ValueError(f"Unknown method '{method}'. Use 'pca' or 'tsne'.")


def project_new_points(projection: EmbeddingProjection, new_embeddings: np.ndarray) -> np.ndarray:
    """Project new points into an existing reduced space.

    For PCA this uses the fitted transform directly.  For t-SNE the stored
    PCA fallback is used (t-SNE cannot transform unseen data).

    Parameters
    ----------
    projection : EmbeddingProjection
        Previously fitted projection.
    new_embeddings : np.ndarray
        (M, D) new embeddings to project.

    Returns
    -------
    np.ndarray
        (M, n_components) projected coordinates.
    """
    if projection.method == "pca":
        result: np.ndarray = projection.transformer.transform(new_embeddings)
        return result
    if projection.method == "tsne":
        pca_fallback = projection.transformer["pca_fallback"]
        result_tsne: np.ndarray = pca_fallback.transform(new_embeddings)
        return result_tsne
    raise ValueError(f"Cannot project new points for method '{projection.method}'")
