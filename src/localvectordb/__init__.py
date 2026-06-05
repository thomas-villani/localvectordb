# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
LocalVectorDB - Document-First Vector Database with SQLite + FAISS

A Python library providing a document-focused vector database built on SQLite,
FAISS, and pluggable embedding providers. Offers both local and remote
(client-server) usage with a unified API.

Copyright (c) 2025 Tom Villani, Ph.D. / New Jersey Innovation Institute (NJII)
Licensed under the PolyForm Noncommercial License 1.0.0.
"""

__version__ = "0.1.0"

from localvectordb.backup import BackupManager, IncrementalBackupManager, PointInTimeRecoveryManager
from localvectordb.chunking import ChunkerFactory
from localvectordb.client import RemoteVectorDB
from localvectordb.core import (
    ChunkAlignment,
    DocumentComparisonResult,
    DocumentSimilarityMatrix,
    MetadataField,
    Section,
    SectionBoundary,
)
from localvectordb.cursor import QueryCursor
from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import EmbeddingRegistry
from localvectordb.extractors import ExtractorRegistry, get_extractor_registry
from localvectordb.factory import VectorDB
from localvectordb.migration import Migration, MigrationEngine
from localvectordb.query_builder import QueryBuilder
from localvectordb.reranking import RerankerRegistry
from localvectordb.sqlite_tuning import (
    SqliteProfile,
    get_profile_description,
    get_sqlite_pragma_profile,
    is_valid_sqlite_pragma_profile,
)
from localvectordb.validation import ClaimResult, FactChecker, FactCheckResult, Polarity
from localvectordb.versioning import VersionManager

__all__ = [
    "__version__",
    "LocalVectorDB",
    "ChunkerFactory",
    "EmbeddingRegistry",
    "RerankerRegistry",
    "RemoteVectorDB",
    "VectorDB",
    "MetadataField",
    "Section",
    "SectionBoundary",
    "ChunkAlignment",
    "DocumentComparisonResult",
    "DocumentSimilarityMatrix",
    "BackupManager",
    "IncrementalBackupManager",
    "PointInTimeRecoveryManager",
    "Migration",
    "MigrationEngine",
    "VersionManager",
    "QueryCursor",
    "QueryBuilder",
    "ExtractorRegistry",
    "get_extractor_registry",
    "SqliteProfile",
    "get_profile_description",
    "is_valid_sqlite_pragma_profile",
    "get_sqlite_pragma_profile",
    "get_common_metadata_schemas",
    "FactChecker",
    "FactCheckResult",
    "ClaimResult",
    "Polarity",
]


def __getattr__(name):
    if name == "get_common_metadata_schemas":
        from localvectordb._schema import get_common_metadata_schemas

        return get_common_metadata_schemas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
