# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
LocalVectorDB MCP Server (stdio-based)

Provides Model Context Protocol server for LocalVectorDB, enabling LLMs
to interact with vector databases through a unified tool interface.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastmcp import FastMCP

from localvectordb import VectorDB
from localvectordb.embeddings import EmbeddingRegistry
from localvectordb.exceptions import (
    DatabaseNotFoundError,
    DocumentNotFoundError,
)
from localvectordb.utils import get_system_version
from localvectordb_server.mcp.config import MCPConfig
from localvectordb_server.utils.schema import parse_metadata_schema

logger = logging.getLogger(__name__)

# Global manager instance
mcp_manager: Optional["MCPManager"] = None


class MCPManager:
    """Simple manager for MCP database operations using VectorDB factory"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.databases: Dict[str, Any] = {}  # Cache for database instances
        self._lock = asyncio.Lock()

    async def get_database(self, name: str):
        """Get database instance using factory pattern - auto-detects local/remote"""
        async with self._lock:
            if name not in self.databases:
                # Get path or URL for this database
                db_path = self.config.get_database_path(name)

                # Prepare kwargs - merge defaults with any remote settings
                kwargs = self.config.db_defaults.copy()

                # If it's a URL, add remote defaults
                if db_path.startswith(("http://", "https://")):
                    kwargs.update(self.config.remote_defaults)

                # Create database using factory
                try:
                    self.databases[name] = VectorDB(
                        name=name,
                        base_path=db_path,
                        create_if_not_exists=False,  # Don't auto-create in read mode
                        **kwargs,
                    )
                except Exception as e:
                    logger.error(f"Failed to connect to database '{name}': {e}")
                    raise DatabaseNotFoundError(f"Database '{name}' not found at {db_path}") from e

            return self.databases[name]

    async def list_databases(self) -> List[str]:
        """List available databases"""
        databases: List[str] = []

        # Add explicitly mapped databases
        databases.extend(self.config.databases_map.keys())

        # Add databases from root directory if it exists
        root_path = Path(self.config.databases_root)
        if root_path.exists() and root_path.is_dir():
            # Look for SQLite database files
            for db_file in root_path.glob("*.sqlite"):
                db_name = db_file.stem
                if db_name not in databases:
                    databases.append(db_name)

        return sorted(databases)

    async def create_database(self, name: str, metadata_schema: Optional[Dict[str, Any]] = None, **kwargs):
        """Create a new database (write mode only)"""
        self.config.check_write_permission("create_database")

        # Get path for new database
        db_path = self.config.get_database_path(name)

        # Parse metadata schema if provided
        parsed_schema = None
        if metadata_schema:
            parsed_schema = parse_metadata_schema(metadata_schema)

        # Merge defaults with provided kwargs
        db_kwargs = self.config.db_defaults.copy()
        db_kwargs.update(kwargs)

        # Create database
        db = VectorDB(
            name=name, base_path=db_path, metadata_schema=parsed_schema, create_if_not_exists=True, **db_kwargs
        )

        # Cache it
        async with self._lock:
            self.databases[name] = db

        return db

    async def delete_database(self, name: str):
        """Delete a database (write mode only)"""
        self.config.check_write_permission("delete_database")

        # Remove from cache
        async with self._lock:
            if name in self.databases:
                # Close if it's a local database
                db = self.databases[name]
                if hasattr(db, "close"):
                    db.close()
                del self.databases[name]

        # Delete files if it's a local database
        db_path = self.config.get_database_path(name)
        if not db_path.startswith(("http://", "https://")):
            # Local database - delete files
            db_file = Path(db_path) / f"{name}.sqlite"
            if db_file.exists():
                db_file.unlink()

            # Delete FAISS index
            faiss_file = Path(db_path) / f"{name}.faiss"
            if faiss_file.exists():
                faiss_file.unlink()

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        async with self._lock:
            for db in self.databases.values():
                if hasattr(db, "close"):
                    db.close()
            self.databases.clear()


# Create lifespan context manager for initialization
from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(mcp):
    """Lifespan context manager for MCP server initialization and cleanup"""
    global mcp_manager

    # Configure logging to stderr
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config = MCPConfig.load()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, config.log_level))

    # Initialize manager
    mcp_manager = MCPManager(config)

    # Dynamically register tools based on configuration
    enabled_tools = config.get_enabled_tools()
    registered_count = 0

    for tool_name, tool_info in TOOL_REGISTRY.items():
        # Check if tool should be enabled
        if tool_name in enabled_tools:
            # Check if it's a write tool and we're in read-only mode
            if not tool_info["read_only"] and config.mode == "read-only":
                logger.debug(f"Skipping write tool '{tool_name}' in read-only mode")
                continue

            # Register the tool with MCP using the decorator
            if not tool_info["registered"]:
                mcp.tool()(tool_info["function"])
                # The tool is now registered with MCP
                tool_info["registered"] = True
                registered_count += 1
                logger.debug(f"Registered tool: {tool_name}")
        else:
            logger.debug(f"Tool '{tool_name}' not in enabled tools list")

    logger.info(f"LocalVectorDB MCP Server started in {config.mode} mode")
    logger.info(f"Database root: {config.databases_root}")
    logger.info(f"Registered {registered_count} tools out of {len(TOOL_REGISTRY)} available")

    yield  # Server runs here

    # Cleanup on shutdown
    if mcp_manager:
        await mcp_manager.cleanup()
    logger.info("LocalVectorDB MCP Server shut down")


# Initialize FastMCP server with lifespan
mcp = FastMCP("LocalVectorDB MCP Server", lifespan=lifespan)

# Store all tool functions for dynamic registration
TOOL_REGISTRY = {}


def register_tool(name: str, read_only: bool = True):
    """Decorator to register tools in the registry"""

    def decorator(func):
        TOOL_REGISTRY[name] = {"function": func, "read_only": read_only, "registered": False}
        return func

    return decorator


def _get_manager() -> "MCPManager":
    """Get the MCP manager instance, raising if not initialized."""
    if mcp_manager is None:
        raise RuntimeError("MCP manager not initialized")
    return mcp_manager


def register_mcp_tool(func):
    """Helper to register a function as an MCP tool with proper metadata"""
    return mcp.tool()(func)


# ============= READ-ONLY TOOLS =============


@register_tool("list_databases", read_only=True)
async def list_databases() -> Dict[str, Any]:
    """
    List all available vector databases

    Returns:
        Dictionary with database names and count
    """
    try:
        manager = _get_manager()
        databases = await manager.list_databases()
        return {"databases": databases, "count": len(databases), "mode": manager.config.mode}
    except Exception as e:
        logger.error(f"Error listing databases: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("get_database_info", read_only=True)
async def get_database_info(database_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific database

    Args:
        database_name: Name of the database

    Returns:
        Database statistics and configuration
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        # Get stats
        stats = db.get_stats()

        # Get configuration info
        info = {
            "name": db.name,
            "stats": stats,
            "config": {
                "embedding_provider": (
                    db.embedding_provider.provider_name if hasattr(db, "embedding_provider") else "unknown"
                ),
                "embedding_model": db.embedding_provider.model if hasattr(db, "embedding_provider") else "unknown",
                "embedding_dimension": db.embedding_dimension if hasattr(db, "embedding_dimension") else None,
                "chunking_method": db.chunking_method if hasattr(db, "chunking_method") else "unknown",
                "chunk_size": db.chunk_size if hasattr(db, "chunk_size") else None,
                "chunk_overlap": db.chunk_overlap if hasattr(db, "chunk_overlap") else None,
                "fts_enabled": db.fts_enabled if hasattr(db, "fts_enabled") else False,
            },
        }

        # Add metadata schema if available
        if hasattr(db, "metadata_schema") and db.metadata_schema:
            info["metadata_schema"] = {
                field_name: {"type": field.type.value, "indexed": field.indexed, "required": field.required}
                for field_name, field in db.metadata_schema.items()
            }

        return info

    except DatabaseNotFoundError as e:
        return {"error": str(e), "error_code": "DATABASE_NOT_FOUND"}
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("query_database", read_only=True)
async def query_database(
    database_name: str,
    query: str,
    search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
    return_type: Literal["documents", "chunks", "context"] = "documents",
    k: int = 10,
    score_threshold: float = 0.0,
    filters: Optional[Dict] = None,
    vector_weight: float = 0.7,
    context_window: int = 2,
    semantic_dedup_threshold: Optional[float] = None,
    document_scoring_method: str = "frequency_boost",
) -> Dict[str, Any]:
    """
    Search a database using vector, keyword, or hybrid search

    Args:
        database_name: Name of the database to search
        query: Search query text
        search_type: Type of search (vector, keyword, hybrid)
        return_type: Return documents, chunks, or context
        k: Number of results to return
        score_threshold: Minimum score threshold
        filters: Metadata filters (MongoDB-style)
        vector_weight: Weight for vector search in hybrid mode (0-1)
        context_window: Window size for context return type
        semantic_dedup_threshold: Threshold for semantic deduplication
        document_scoring_method: Method for scoring documents

    Returns:
        Search results with scores and metadata
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        # Use async query if available
        if hasattr(db, "query_async"):
            results = await db.query_async(
                query=query,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                context_window=context_window,
                semantic_dedup_threshold=semantic_dedup_threshold,
                document_scoring_method=document_scoring_method,
            )
        else:
            # Fallback to sync query
            results = db.query(
                query=query,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                context_window=context_window,
                semantic_dedup_threshold=semantic_dedup_threshold,
                document_scoring_method=document_scoring_method,
            )

        # Serialize results
        serialized_results = []
        for result in results:
            data = {
                "id": result.id,
                "score": float(result.score),
                "type": result.type,
                "content": result.content,
                "metadata": result.metadata,
            }
            if hasattr(result, "document_id") and result.document_id:
                data["document_id"] = result.document_id
            if hasattr(result, "position") and result.position:
                data["position"] = {
                    "index": result.position.index,
                    "total": result.position.total,
                    "start_char": result.position.start_char,
                    "end_char": result.position.end_char,
                }
            serialized_results.append(data)

        return {
            "results": serialized_results,
            "search_type": search_type,
            "return_type": return_type,
            "total_results": len(serialized_results),
        }

    except DatabaseNotFoundError as e:
        return {"error": str(e), "error_code": "DATABASE_NOT_FOUND"}
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("filter_documents", read_only=True)
async def filter_documents(
    database_name: str, filters: Dict[str, Any], limit: int = 100, offset: int = 0
) -> Dict[str, Any]:
    """
    Filter documents by metadata

    Args:
        database_name: Name of the database
        filters: Metadata filters (MongoDB-style)
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Filtered documents
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        # Use filter method
        if hasattr(db, "filter_async"):
            documents = await db.filter_async(where=filters, limit=limit, offset=offset)
        else:
            documents = db.filter(where=filters, limit=limit, offset=offset)

        # Serialize documents
        serialized_docs = []
        for doc in documents:
            serialized_docs.append(
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                }
            )

        return {"documents": serialized_docs, "count": len(serialized_docs), "filters": filters}

    except Exception as e:
        logger.error(f"Error filtering documents: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("get_document", read_only=True)
async def get_document(database_name: str, document_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific document by ID

    Args:
        database_name: Name of the database
        document_id: ID of the document

    Returns:
        Document content and metadata
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        # Get document
        if hasattr(db, "get_async"):
            doc = await db.get_async(document_id)
        else:
            doc = db.get(document_id)

        if doc is None:
            return {"error": f"Document '{document_id}' not found", "error_code": "DOCUMENT_NOT_FOUND"}

        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            "content_hash": doc.content_hash if hasattr(doc, "content_hash") else None,
        }

    except Exception as e:
        logger.error(f"Error getting document: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("check_documents_exist", read_only=True)
async def check_documents_exist(database_name: str, document_ids: List[str]) -> Dict[str, Any]:
    """
    Check if documents exist in the database

    Args:
        database_name: Name of the database
        document_ids: List of document IDs to check

    Returns:
        Dictionary mapping document IDs to existence status
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        # Check existence
        if hasattr(db, "exists_async"):
            exists_map = await db.exists_async(document_ids)
        else:
            exists_map = db.exists(document_ids)

        return {"exists": exists_map, "total_checked": len(document_ids), "total_found": sum(exists_map.values())}

    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("get_metadata_schema", read_only=True)
async def get_metadata_schema(database_name: str) -> Dict[str, Any]:
    """
    Get the metadata schema for a database

    Args:
        database_name: Name of the database

    Returns:
        Metadata schema definition
    """
    try:
        manager = _get_manager()
        db = await manager.get_database(database_name)

        if hasattr(db, "metadata_schema") and db.metadata_schema:
            schema = {}
            for field_name, field in db.metadata_schema.items():
                schema[field_name] = {"type": field.type.value, "indexed": field.indexed, "required": field.required}
            return {"schema": schema}
        else:
            return {"schema": {}, "message": "No metadata schema defined"}

    except Exception as e:
        logger.error(f"Error getting metadata schema: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("get_system_info", read_only=True)
async def get_system_info() -> Dict[str, Any]:
    """
    Get system information and configuration

    Returns:
        System version, configuration, and status
    """
    try:
        if mcp_manager is not None:
            manager = mcp_manager
            return {
                "version": get_system_version(),
                "mode": manager.config.mode,
                "database_root": manager.config.databases_root,
                "available_providers": EmbeddingRegistry.list(),
                "databases_count": len(await manager.list_databases()),
                "enabled_tools": manager.config.get_enabled_tools(),
            }
        else:
            return {
                "version": get_system_version(),
                "mode": "not_initialized",
                "database_root": None,
                "available_providers": EmbeddingRegistry.list(),
                "databases_count": 0,
                "enabled_tools": [],
            }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


# ============= WRITE TOOLS (only available in read-write mode) =============


@register_tool("create_database", read_only=False)
async def create_database(
    name: str,
    metadata_schema: Optional[Dict[str, Any]] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunking_method: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a new vector database

    Args:
        name: Database name
        metadata_schema: Schema for document metadata (field_name -> type or config dict)
        embedding_provider: Provider for embeddings (e.g., "ollama", "openai")
        embedding_model: Model name for embeddings
        chunking_method: Method for chunking documents
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Database configuration and status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("create_database")

        # Build kwargs from provided parameters
        kwargs: Dict[str, Any] = {}
        if embedding_provider:
            kwargs["embedding_provider"] = embedding_provider
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        if chunking_method:
            kwargs["chunking_method"] = chunking_method
        if chunk_size:
            kwargs["chunk_size"] = chunk_size
        if chunk_overlap:
            kwargs["chunk_overlap"] = chunk_overlap

        # Create database
        db = await manager.create_database(name=name, metadata_schema=metadata_schema, **kwargs)

        return {
            "message": f"Successfully created database '{name}'",
            "status": "success",
            "config": {
                "name": db.name,
                "embedding_provider": kwargs.get(
                    "embedding_provider", manager.config.db_defaults["embedding_provider"]
                ),
                "embedding_model": kwargs.get("embedding_model", manager.config.db_defaults["embedding_model"]),
                "chunking_method": kwargs.get("chunking_method", manager.config.db_defaults["chunking_method"]),
                "chunk_size": kwargs.get("chunk_size", manager.config.db_defaults["chunk_size"]),
                "chunk_overlap": kwargs.get("chunk_overlap", manager.config.db_defaults["chunk_overlap"]),
            },
        }

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("delete_database", read_only=False)
async def delete_database(name: str) -> Dict[str, Any]:
    """
    Delete a vector database

    Args:
        name: Database name to delete

    Returns:
        Deletion status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("delete_database")

        await manager.delete_database(name)

        return {"message": f"Successfully deleted database '{name}'", "status": "success"}

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        logger.error(f"Error deleting database: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("upsert_documents", read_only=False)
async def upsert_documents(
    database_name: str,
    documents: Union[str, List[str]],
    metadata: Optional[Union[Dict, List[Dict]]] = None,
    ids: Optional[Union[str, List[str]]] = None,
    batch_size: int = 100,
    similarity_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Insert or update documents in the database

    Args:
        database_name: Name of the database
        documents: Document(s) to upsert
        metadata: Metadata for documents
        ids: Optional document IDs
        batch_size: Batch size for processing
        similarity_threshold: Threshold for similarity detection

    Returns:
        Document IDs and operation status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("upsert_documents")

        # Normalize inputs
        if isinstance(documents, str):
            documents = [documents]
        if metadata and isinstance(metadata, dict):
            metadata = [metadata]
        if ids and isinstance(ids, str):
            ids = [ids]

        # Get database
        db = await manager.get_database(database_name)

        # Upsert documents
        if hasattr(db, "upsert_async"):
            result_ids = await db.upsert_async(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
            )
        else:
            result_ids = db.upsert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
            )

        return {"message": f"Successfully processed {len(documents)} documents", "ids": result_ids, "status": "success"}

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        logger.error(f"Error upserting documents: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("update_document", read_only=False)
async def update_document(
    database_name: str, document_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update a document's content and/or metadata

    Args:
        database_name: Name of the database
        document_id: ID of the document to update
        content: New content (optional)
        metadata: New or updated metadata (optional)

    Returns:
        Update status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("update_document")

        db = await manager.get_database(database_name)

        # Update document
        if hasattr(db, "update_async"):
            await db.update_async(doc_id=document_id, content=content, metadata=metadata)
        else:
            db.update(doc_id=document_id, content=content, metadata=metadata)

        return {"message": f"Successfully updated document '{document_id}'", "status": "success"}

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except DocumentNotFoundError as e:
        return {"error": str(e), "error_code": "DOCUMENT_NOT_FOUND"}
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("delete_document", read_only=False)
async def delete_document(database_name: str, document_id: str) -> Dict[str, Any]:
    """
    Delete a document from the database

    Args:
        database_name: Name of the database
        document_id: ID of the document to delete

    Returns:
        Deletion status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("delete_document")

        db = await manager.get_database(database_name)

        # Delete document
        if hasattr(db, "delete_async"):
            deleted_count = await db.delete_async(document_id)
        else:
            deleted_count = db.delete(document_id)

        if deleted_count == 0:
            return {"error": f"Document '{document_id}' not found", "error_code": "DOCUMENT_NOT_FOUND"}

        return {
            "message": f"Successfully deleted document '{document_id}'",
            "status": "success",
            "deleted_count": deleted_count,
        }

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@register_tool("update_metadata_schema", read_only=False)
async def update_metadata_schema(database_name: str, metadata_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the metadata schema for a database

    Args:
        database_name: Name of the database
        metadata_schema: New metadata schema definition

    Returns:
        Update status
    """
    try:
        manager = _get_manager()
        manager.config.check_write_permission("update_metadata_schema")

        db = await manager.get_database(database_name)

        # Parse metadata schema
        parsed_schema = parse_metadata_schema(metadata_schema)

        # Update schema
        if hasattr(db, "update_metadata_schema_async"):
            await db.update_metadata_schema_async(parsed_schema)
        elif hasattr(db, "update_metadata_schema"):
            db.update_metadata_schema(parsed_schema)
        else:
            return {"error": "Database does not support metadata schema updates", "error_code": "NOT_SUPPORTED"}

        return {"message": f"Successfully updated metadata schema for database '{database_name}'", "status": "success"}

    except PermissionError as e:
        return {"error": str(e), "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        logger.error(f"Error updating metadata schema: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


# Initialize all tool definitions to populate the registry
def _initialize_tools():
    """Initialize tool definitions to populate TOOL_REGISTRY"""
    # This ensures all @register_tool decorators are executed
    pass


# Call it to populate the registry
_initialize_tools()


# Entry point for CLI
async def run_mcp_server(mode: str = "read-only"):
    """Run the stdio MCP server"""
    # Set mode in environment for initialization
    os.environ["LVDB_MCP_MODE"] = mode

    # Run FastMCP server with stdio
    await mcp.run_stdio_async()


# Main entry point for direct execution
if __name__ == "__main__":
    mode = os.getenv("LVDB_MCP_MODE", "read-only")
    asyncio.run(run_mcp_server(mode))
