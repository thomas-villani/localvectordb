"""
Document comparison and similarity analysis.

Provides document-level and chunk-level comparison methods, nearest-neighbor
search, and pairwise similarity matrices.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np

from localvectordb.core import (
    ChunkAlignment,
    ChunkSimilarityMatrix,
    DocumentComparisonResult,
    DocumentSimilarityMatrix,
    QueryResult,
)
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import DatabaseError

if TYPE_CHECKING:
    from faiss import Index

    from localvectordb._pools import AsyncConnectionPool, ConnectionPool, ReadWriteLock
    from localvectordb.section_detection import SectionDetector

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, returned in [0, 1] via (1 + cos) / 2."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos = float(np.dot(a, b) / (norm_a * norm_b))
    return (cos + 1.0) / 2.0


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix between rows of a and b, in [0, 1]."""
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = a / np.maximum(norms_a, 1e-8)
    b_norm = b / np.maximum(norms_b, 1e-8)
    cos_matrix = a_norm @ b_norm.T
    result: np.ndarray = (cos_matrix + 1.0) / 2.0
    return result


class ComparisonMixin(LocalVectorDBBase, ABC):
    """Mixin providing document comparison and nearest-neighbor methods."""

    # Redeclare attributes from LocalVectorDBBase and composed class.
    _read_write_lock: "ReadWriteLock"
    connection_pool: "ConnectionPool"
    async_connection_pool: Optional["AsyncConnectionPool"]
    index: Optional["Index"]

    _hierarchical_embeddings: bool
    _faiss_lock: "ReadWriteLock"
    section_index: Optional["Index"]
    document_index: Optional["Index"]
    _section_detector: Optional["SectionDetector"]

    # Forward-declared methods implemented by sibling mixins (SearchMixin) and
    # LocalVectorDBCore. Declared under TYPE_CHECKING only -- real bodies here would
    # shadow the concrete implementations, since ComparisonMixin precedes
    # LocalVectorDBCore in LocalVectorDB's MRO.
    if TYPE_CHECKING:

        def _reconstruct_embeddings_batch(self, faiss_ids: List[int]) -> np.ndarray: ...

        def _get_documents_metadata_batch(self, conn: Any, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]: ...

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_document_embedding(self, doc_id: str) -> np.ndarray:
        """Return the document-level embedding for *doc_id*.

        In hierarchical mode the embedding is stored in ``document_index``
        and looked up via the ``doc_faiss_id`` column.  Otherwise it is
        reconstructed as the mean of chunk embeddings.
        """
        if self._hierarchical_embeddings and self.document_index is not None:
            with self.connection_pool.get_connection() as conn:
                row = conn.execute("SELECT doc_faiss_id FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row is None:
                raise ValueError(f"Document '{doc_id}' not found")
            doc_faiss_id = row["doc_faiss_id"]
            if doc_faiss_id is None:
                raise ValueError(f"Document '{doc_id}' has no document-level embedding")
            with self._faiss_lock.read_lock():
                # faiss's type stub annotates reconstruct() as returning torch.Tensor,
                # but at runtime it returns an np.ndarray; cast to keep mypy honest
                # across environments where torch is / isn't installed.
                embedding = cast(np.ndarray, self.document_index.reconstruct(int(doc_faiss_id)))
            return embedding

        # Non-hierarchical: average chunk embeddings
        chunk_embs, _ = self._get_chunk_embeddings_for_doc(doc_id)
        if chunk_embs.shape[0] == 0:
            raise ValueError(f"Document '{doc_id}' has no chunk embeddings")
        result: np.ndarray = np.mean(chunk_embs, axis=0).astype(np.float32)
        return result

    def _get_document_embeddings_batch(self, doc_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Return embeddings for multiple documents.

        Returns
        -------
        embeddings : np.ndarray
            (N, D) array of document embeddings.
        valid_doc_ids : List[str]
            Document IDs for which embeddings were successfully obtained.
        """
        embeddings = []
        valid_ids: List[str] = []
        for doc_id in doc_ids:
            try:
                emb = self._get_document_embedding(doc_id)
                embeddings.append(emb)
                valid_ids.append(doc_id)
            except ValueError:
                logger.warning("Skipping document '%s': no embedding available", doc_id)
        if not embeddings:
            return np.array([]).reshape(0, self.embedding_dimension), []
        return np.array(embeddings, dtype=np.float32), valid_ids

    def _get_chunk_embeddings_for_doc(self, doc_id: str) -> Tuple[np.ndarray, List[int]]:
        """Return chunk embeddings and their indices for a document.

        Returns
        -------
        chunk_embeddings : np.ndarray
            (C, D) array of chunk embeddings.
        chunk_indices : List[int]
            Matching chunk indices within the document.
        """
        with self.connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT chunk_index, faiss_id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        if not rows:
            return np.array([]).reshape(0, self.embedding_dimension), []

        faiss_ids = [r["faiss_id"] for r in rows if r["faiss_id"] is not None]
        chunk_indices = [r["chunk_index"] for r in rows if r["faiss_id"] is not None]
        if not faiss_ids:
            return np.array([]).reshape(0, self.embedding_dimension), []

        embs = self._reconstruct_embeddings_batch(faiss_ids)
        return embs, chunk_indices

    # ------------------------------------------------------------------ #
    # Public API – document-level comparison                               #
    # ------------------------------------------------------------------ #

    def compare_documents(self, doc_id_1: str, doc_id_2: str) -> float:
        """Return cosine similarity [0, 1] between two documents (centroid-based).

        Parameters
        ----------
        doc_id_1 : str
            First document ID.
        doc_id_2 : str
            Second document ID.

        Returns
        -------
        float
            Cosine similarity normalised to [0, 1].
        """
        emb1 = self._get_document_embedding(doc_id_1)
        emb2 = self._get_document_embedding(doc_id_2)
        return _cosine_similarity(emb1, emb2)

    def nearest_neighbors(
        self,
        doc_id: str,
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Return the *k* most similar documents to *doc_id*.

        Parameters
        ----------
        doc_id : str
            Reference document.
        k : int
            Maximum number of neighbours to return.
        score_threshold : float
            Minimum similarity score to include.
        filters : dict, optional
            Metadata filter dict applied to candidates. Filter fields must be
            declared in the metadata schema; unknown fields or unsupported
            operators raise ``DatabaseError``.

        Returns
        -------
        List[QueryResult]
            Sorted by score descending; the reference document is excluded.
        """
        from localvectordb._filters import FilterQueryBuilder, matches_metadata_filter, validate_filter_spec

        if filters:
            validate_filter_spec(filters, self.metadata_schema)

        ref_emb = self._get_document_embedding(doc_id)

        # Gather candidate document IDs. Push a SQL-expressible metadata filter
        # into this query so only matching documents are reconstructed and scored
        # (T1.3); the Python matcher below stays the authority for any residual
        # (e.g. dot-notation) filter. This does not starve results either way,
        # since every matching document is scored before the top-k truncation.
        candidate_sql = "SELECT id FROM documents"
        candidate_params: tuple = ()
        if filters:
            try:
                where_clause, where_params = FilterQueryBuilder(self.metadata_schema).build_where_clause(filters)
            except DatabaseError:
                where_clause = ""
            if where_clause:
                candidate_sql = f"SELECT id FROM documents WHERE {where_clause}"
                candidate_params = tuple(where_params)
        with self.connection_pool.get_connection() as conn:
            rows = conn.execute(candidate_sql, candidate_params).fetchall()
        all_ids = [r["id"] for r in rows if r["id"] != doc_id]

        if not all_ids:
            return []

        embs, valid_ids = self._get_document_embeddings_batch(all_ids)
        if embs.shape[0] == 0:
            return []

        # Compute similarities
        ref_norm = ref_emb / max(float(np.linalg.norm(ref_emb)), 1e-8)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / np.maximum(norms, 1e-8)
        scores = ((embs_norm @ ref_norm) + 1.0) / 2.0

        # Fetch content + metadata for filtering
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(valid_ids))
            cursor = conn.execute(
                f"SELECT id, content FROM documents WHERE id IN ({placeholders})",
                valid_ids,
            )
            doc_content = {r["id"]: r["content"] for r in cursor.fetchall()}
            doc_metadata_batch = self._get_documents_metadata_batch(conn, valid_ids)

        results: List[QueryResult] = []
        for idx, vid in enumerate(valid_ids):
            score = float(scores[idx])
            if score < score_threshold:
                continue
            meta = doc_metadata_batch.get(vid, {})
            if filters and not matches_metadata_filter(meta, filters):
                continue
            results.append(
                QueryResult(
                    id=vid,
                    score=score,
                    type="document",
                    content=doc_content.get(vid, ""),
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def pairwise_similarity_matrix(self, doc_ids: Optional[List[str]] = None) -> DocumentSimilarityMatrix:
        """Compute an NxN similarity matrix for all (or selected) documents.

        Parameters
        ----------
        doc_ids : list of str, optional
            Specific document IDs. If ``None``, all documents are used.

        Returns
        -------
        DocumentSimilarityMatrix
        """
        if doc_ids is None:
            with self.connection_pool.get_connection() as conn:
                rows = conn.execute("SELECT id FROM documents").fetchall()
            doc_ids = [r["id"] for r in rows]

        embs, valid_ids = self._get_document_embeddings_batch(doc_ids)
        if embs.shape[0] == 0:
            return DocumentSimilarityMatrix(
                matrix=np.array([]).reshape(0, 0),
                doc_ids=[],
                embeddings=embs,
            )

        matrix = _cosine_similarity_matrix(embs, embs)
        return DocumentSimilarityMatrix(
            matrix=matrix,
            doc_ids=valid_ids,
            embeddings=embs,
        )

    # ------------------------------------------------------------------ #
    # Public API – chunk-level comparison                                  #
    # ------------------------------------------------------------------ #

    def compare_documents_detailed(
        self,
        doc_id_1: str,
        doc_id_2: str,
        chunk_threshold: float = 0.7,
    ) -> DocumentComparisonResult:
        """Rich chunk-level comparison between two documents.

        For each chunk in *doc_id_1*, finds the best-matching chunk in *doc_id_2*
        (and vice-versa). A chunk counts as "matched" if its best-match similarity
        meets *chunk_threshold*.

        Parameters
        ----------
        doc_id_1 : str
            First document ID.
        doc_id_2 : str
            Second document ID.
        chunk_threshold : float
            Minimum similarity for a chunk pair to count as "matched".

        Returns
        -------
        DocumentComparisonResult
        """
        overall = self.compare_documents(doc_id_1, doc_id_2)

        embs1, idx1 = self._get_chunk_embeddings_for_doc(doc_id_1)
        embs2, idx2 = self._get_chunk_embeddings_for_doc(doc_id_2)

        if embs1.shape[0] == 0 or embs2.shape[0] == 0:
            return DocumentComparisonResult(
                doc_id_1=doc_id_1,
                doc_id_2=doc_id_2,
                overall_similarity=overall,
                chunk_alignments=[],
                matched_ratio_1=0.0,
                matched_ratio_2=0.0,
                unmatched_chunks_1=idx1,
                unmatched_chunks_2=idx2,
            )

        # (C1, C2) cosine similarity matrix
        sim_matrix = _cosine_similarity_matrix(embs1, embs2)

        # Best match for each chunk in doc_1 → doc_2
        alignments: List[ChunkAlignment] = []
        matched_1 = set()
        for i, ci in enumerate(idx1):
            best_j = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_j])
            alignments.append(
                ChunkAlignment(
                    chunk_index_1=ci,
                    chunk_index_2=idx2[best_j],
                    similarity=best_sim,
                )
            )
            if best_sim >= chunk_threshold:
                matched_1.add(ci)
        alignments.sort(key=lambda a: a.similarity, reverse=True)

        # Best match for each chunk in doc_2 → doc_1
        matched_2 = set()
        for j, cj in enumerate(idx2):
            best_sim_rev = float(np.max(sim_matrix[:, j]))
            if best_sim_rev >= chunk_threshold:
                matched_2.add(cj)

        unmatched_1 = [ci for ci in idx1 if ci not in matched_1]
        unmatched_2 = [cj for cj in idx2 if cj not in matched_2]

        return DocumentComparisonResult(
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            overall_similarity=overall,
            chunk_alignments=alignments,
            matched_ratio_1=len(matched_1) / len(idx1) if idx1 else 0.0,
            matched_ratio_2=len(matched_2) / len(idx2) if idx2 else 0.0,
            unmatched_chunks_1=unmatched_1,
            unmatched_chunks_2=unmatched_2,
        )

    # ------------------------------------------------------------------ #
    # Public API – async twins                                             #
    #                                                                      #
    # Comparison is CPU/SQLite-bound sync work; the async variants run it  #
    # off the event-loop thread so ``await db.<op>_async(...)`` works on a #
    # LocalVectorDB the same way it does on RemoteVectorDB.                #
    # ------------------------------------------------------------------ #

    async def compare_documents_async(self, doc_id_1: str, doc_id_2: str) -> float:
        """Async twin of :meth:`compare_documents`."""
        return await asyncio.to_thread(self.compare_documents, doc_id_1, doc_id_2)

    async def compare_documents_detailed_async(
        self,
        doc_id_1: str,
        doc_id_2: str,
        chunk_threshold: float = 0.7,
    ) -> DocumentComparisonResult:
        """Async twin of :meth:`compare_documents_detailed`."""
        return await asyncio.to_thread(self.compare_documents_detailed, doc_id_1, doc_id_2, chunk_threshold)

    async def nearest_neighbors_async(
        self,
        doc_id: str,
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Async twin of :meth:`nearest_neighbors`."""
        return await asyncio.to_thread(
            lambda: self.nearest_neighbors(doc_id, k=k, score_threshold=score_threshold, filters=filters)
        )

    async def pairwise_similarity_matrix_async(self, doc_ids: Optional[List[str]] = None) -> DocumentSimilarityMatrix:
        """Async twin of :meth:`pairwise_similarity_matrix`."""
        return await asyncio.to_thread(self.pairwise_similarity_matrix, doc_ids)

    # ------------------------------------------------------------------ #
    # Public API – chunk similarity matrix                                 #
    # ------------------------------------------------------------------ #

    def chunk_similarity_matrix(
        self,
        doc_id_1: str,
        doc_id_2: Optional[str] = None,
    ) -> ChunkSimilarityMatrix:
        """Compute the full chunk-level pairwise similarity matrix.

        When *doc_id_2* is ``None``, computes self-similarity within
        *doc_id_1* (useful for chord diagrams).

        Parameters
        ----------
        doc_id_1 : str
            First document ID.
        doc_id_2 : str, optional
            Second document ID.  Defaults to *doc_id_1*.

        Returns
        -------
        ChunkSimilarityMatrix
        """
        if doc_id_2 is None:
            doc_id_2 = doc_id_1

        embs1, idx1 = self._get_chunk_embeddings_for_doc(doc_id_1)
        if embs1.shape[0] == 0:
            raise ValueError(f"Document '{doc_id_1}' has no chunk embeddings")

        if doc_id_1 == doc_id_2:
            embs2, idx2 = embs1, idx1
        else:
            embs2, idx2 = self._get_chunk_embeddings_for_doc(doc_id_2)
            if embs2.shape[0] == 0:
                raise ValueError(f"Document '{doc_id_2}' has no chunk embeddings")

        matrix = _cosine_similarity_matrix(embs1, embs2)
        return ChunkSimilarityMatrix(
            matrix=matrix,
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            chunk_indices_1=idx1,
            chunk_indices_2=idx2,
        )

    # ------------------------------------------------------------------ #
    # Convenience visualisation wrappers                                   #
    # ------------------------------------------------------------------ #

    def visualize_documents(
        self,
        doc_ids: Optional[List[str]] = None,
        method: str = "tsne",
        color_by: Optional[str] = None,
        n_clusters: Optional[int] = None,
        interactive: bool = False,
        **kwargs,
    ):
        """Project document embeddings to 2-D and plot.

        Parameters
        ----------
        doc_ids : list of str, optional
            Documents to include.  All documents if ``None``.
        method : str
            ``"tsne"`` or ``"pca"``.
        color_by : str, optional
            Metadata field name used for point colouring.
        n_clusters : int, optional
            If set, cluster embeddings and colour by cluster.
        interactive : bool
            Use plotly for interactive plots instead of matplotlib.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        from localvectordb.visualization import (
            cluster_embeddings,
            plot_clusters,
            plot_embedding_map,
            reduce_dimensions,
        )

        if doc_ids is None:
            with self.connection_pool.get_connection() as conn:
                rows = conn.execute("SELECT id FROM documents").fetchall()
            doc_ids = [r["id"] for r in rows]

        embs, valid_ids = self._get_document_embeddings_batch(doc_ids)
        if embs.shape[0] == 0:
            raise ValueError("No document embeddings available to visualise")

        # Metadata for colour-by
        labels = None
        if color_by:
            with self.connection_pool.get_connection() as conn:
                doc_meta = self._get_documents_metadata_batch(conn, valid_ids)
            labels = [str(doc_meta.get(did, {}).get(color_by, "")) for did in valid_ids]

        projection = reduce_dimensions(embs, method=method, doc_ids=valid_ids, **kwargs)

        if n_clusters is not None:
            clusters = cluster_embeddings(embs, n_clusters=n_clusters)
            if interactive:
                from localvectordb.visualization import plot_clusters_interactive

                return plot_clusters_interactive(projection, clusters, **kwargs)
            return plot_clusters(projection, clusters, **kwargs)

        if interactive:
            from localvectordb.visualization import plot_embedding_map_interactive

            return plot_embedding_map_interactive(projection, color_by=labels, **kwargs)
        return plot_embedding_map(projection, color_by=labels, **kwargs)

    def visualize_queries(
        self,
        queries: List[str],
        doc_ids: Optional[List[str]] = None,
        method: str = "tsne",
        interactive: bool = False,
        **kwargs,
    ):
        """Visualise how queries relate to the document embedding space.

        Parameters
        ----------
        queries : list of str
            Query strings to overlay on the map.
        doc_ids : list of str, optional
            Documents to include.  All documents if ``None``.
        method : str
            Dimensionality reduction method.
        interactive : bool
            Use plotly instead of matplotlib.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        from localvectordb.visualization import (
            plot_embedding_map,
            reduce_dimensions,
        )
        from localvectordb.visualization.types import QueryOverlay

        if doc_ids is None:
            with self.connection_pool.get_connection() as conn:
                rows = conn.execute("SELECT id FROM documents").fetchall()
            doc_ids = [r["id"] for r in rows]

        embs, valid_ids = self._get_document_embeddings_batch(doc_ids)
        if embs.shape[0] == 0:
            raise ValueError("No document embeddings available to visualise")

        # Embed queries
        query_embeddings = self.embedding_provider.embed_sync(queries)
        query_embs = np.array(query_embeddings, dtype=np.float32)

        # Compute similarity scores for each query against all docs
        overlays: List[QueryOverlay] = []
        for i, q_text in enumerate(queries):
            q_emb = query_embs[i]
            q_norm = q_emb / max(float(np.linalg.norm(q_emb)), 1e-8)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            doc_norm = embs / np.maximum(norms, 1e-8)
            scores = ((doc_norm @ q_norm) + 1.0) / 2.0
            overlays.append(
                QueryOverlay(
                    query_text=q_text,
                    query_embedding=q_emb,
                    scores=scores,
                )
            )

        projection = reduce_dimensions(embs, method=method, doc_ids=valid_ids, **kwargs)

        if interactive:
            from localvectordb.visualization import plot_embedding_map_interactive

            return plot_embedding_map_interactive(projection, queries=overlays, **kwargs)
        return plot_embedding_map(projection, queries=overlays, **kwargs)

    def visualize_synteny(
        self,
        doc_id_1: str,
        doc_id_2: str,
        similarity_threshold: float = 0.7,
        orientation: str = "horizontal",
        chunk_labels: bool = False,
        interactive: bool = False,
        **kwargs,
    ):
        """Synteny ribbon diagram comparing chunks of two documents.

        Parameters
        ----------
        doc_id_1 : str
            First document ID.
        doc_id_2 : str
            Second document ID.
        similarity_threshold : float
            Minimum similarity for a ribbon to be drawn.
        orientation : str
            ``"horizontal"`` or ``"vertical"``.
        chunk_labels : bool
            Label each chunk segment with its index.
        interactive : bool
            Use plotly instead of matplotlib.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        from localvectordb.visualization import plot_synteny

        chunk_sim = self.chunk_similarity_matrix(doc_id_1, doc_id_2)
        if interactive:
            from localvectordb.visualization import plot_synteny_interactive

            return plot_synteny_interactive(
                chunk_sim,
                similarity_threshold=similarity_threshold,
                orientation=orientation,
                chunk_labels=chunk_labels,
                **kwargs,
            )
        return plot_synteny(
            chunk_sim,
            similarity_threshold=similarity_threshold,
            orientation=orientation,
            chunk_labels=chunk_labels,
            **kwargs,
        )

    def visualize_chord(
        self,
        doc_id: str,
        similarity_threshold: float = 0.7,
        min_chunk_distance: int = 3,
        chunk_labels: bool = False,
        interactive: bool = False,
        **kwargs,
    ):
        """Chord (Circos-style) diagram for chunk self-similarity.

        Parameters
        ----------
        doc_id : str
            Document ID.
        similarity_threshold : float
            Minimum similarity for a chord to be drawn.
        min_chunk_distance : int
            Minimum index distance between chunks for a chord.
        chunk_labels : bool
            Label each arc segment with its index.
        interactive : bool
            Use plotly instead of matplotlib.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        from localvectordb.visualization import plot_chord

        chunk_sim = self.chunk_similarity_matrix(doc_id)
        if interactive:
            from localvectordb.visualization import plot_chord_interactive

            return plot_chord_interactive(
                chunk_sim,
                similarity_threshold=similarity_threshold,
                min_chunk_distance=min_chunk_distance,
                chunk_labels=chunk_labels,
                **kwargs,
            )
        return plot_chord(
            chunk_sim,
            similarity_threshold=similarity_threshold,
            min_chunk_distance=min_chunk_distance,
            chunk_labels=chunk_labels,
            **kwargs,
        )
