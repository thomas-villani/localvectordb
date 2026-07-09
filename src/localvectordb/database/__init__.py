from __future__ import annotations

from localvectordb.database._comparison import ComparisonMixin
from localvectordb.database._core import LocalVectorDBCore
from localvectordb.database._crud import CrudMixin
from localvectordb.database._ingest import PipelineMixin
from localvectordb.database._metadata import MetadataMixin
from localvectordb.database._repair import RepairMixin, RepairReport
from localvectordb.database._search import SearchMixin
from localvectordb.database._tuning import LocalTuningMixin, TuningMixin
from localvectordb.database.base import BaseVectorDB


class LocalVectorDB(
    LocalTuningMixin,
    PipelineMixin,
    SearchMixin,
    MetadataMixin,
    CrudMixin,
    ComparisonMixin,
    RepairMixin,
    LocalVectorDBCore,
):
    """
    Document-first vector database with SQLite + FAISS + embeddings

    This is the main interface for LocalVectorDB v1.0, designed around documents
    rather than chunks. All chunking is handled internally.

    Parameters
    ----------
    name : str
        Database name (used for file naming)
    base_path : str, optional
        Directory to store database files, by default ".lvdb"
    metadata_schema : str | Dict[str, MetadataField], optional
        Schema definition for metadata fields
    doc_id_pattern : str, optional
        Pattern for auto-generating document IDs, by default "doc_{idx}"
    embedding_provider : str, optional
        Embedding provider name, by default "ollama"
    embedding_model : str, optional
        Embedding model name, by default "nomic-embed-text"
    embedding_config : Dict[str, Any], optional
        Configuration for embedding provider
    chunking_method : str, optional
        Chunking method, by default "sentences"
    chunk_size : int, optional
        Maximum tokens per chunk, by default 500
    chunk_overlap : int, optional
        Overlap between consecutive chunks, by default 1. Measured in the unit of
        ``chunking_method`` (sentences for "sentences", tokens for "tokens", words
        for "words", lines for "lines"/"code-blocks", characters for "characters",
        paragraphs for "paragraphs"), NOT tokens — only "tokens" shares its unit
        with ``chunk_size``. Keep it small (e.g. 1-3); a value larger than the
        number of units a chunk holds produces highly redundant chunks.
    enable_gpu : bool, optional
        Whether to use GPU for FAISS, by default False
    enable_fts : bool, optional
        Whether to enable full-text search, by default True
    create_if_not_exists: bool, default = True
        If False, raises DatabaseNotFoundError if the database doesn't exist.
    """

    pass


__all__ = ["LocalVectorDB", "BaseVectorDB", "TuningMixin", "RepairReport"]
