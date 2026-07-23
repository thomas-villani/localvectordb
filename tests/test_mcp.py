"""
Tests for localvectordb_server.mcp module.

Tests the MCP (Model Context Protocol) server implementation including:
- MCPConfig configuration handling
- MCPManager database operations
- Tool registration via TOOL_REGISTRY
- Individual tool functions (business logic)
- Error handling for invalid inputs and missing databases
"""

import asyncio
import textwrap
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The MCP server depends on the optional `mcp` extra (fastmcp). Skip the whole
# module when it is not installed so the suite stays green without that extra.
pytest.importorskip("fastmcp")

from localvectordb.exceptions import DatabaseNotFoundError, DocumentNotFoundError  # noqa: E402
from localvectordb_server.mcp.config import MCPConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_config(**overrides):
    """Create an MCPConfig with sensible test defaults."""
    kwargs = {
        "mode": "read-write",
        "databases_root": "/tmp/test_dbs",
    }
    kwargs.update(overrides)
    return MCPConfig(**kwargs)


def _make_document(doc_id="doc1", content="hello world", metadata=None):
    """Create a lightweight mock document."""
    doc = SimpleNamespace(
        id=doc_id,
        content=content,
        metadata=metadata or {},
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        updated_at=datetime(2025, 6, 1, 12, 0, 0),
        content_hash="abc123",
    )
    return doc


def _make_query_result(
    result_id="r1",
    score=0.95,
    content="result text",
    result_type="document",
    document_id=None,
    position=None,
    metadata=None,
):
    """Create a lightweight mock query result."""
    r = SimpleNamespace(
        id=result_id,
        score=score,
        type=result_type,
        content=content,
        metadata=metadata or {},
        document_id=document_id,
        position=position,
    )
    return r


# ---------------------------------------------------------------------------
# MCPConfig tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMCPConfig:
    """Tests for MCPConfig dataclass and its class methods."""

    def test_default_values(self):
        config = MCPConfig()
        assert config.mode == "read-only"
        assert config.databases_root == "./databases"
        assert config.databases_map == {}
        assert "embedding_provider" in config.db_defaults
        assert config.log_level == "INFO"

    def test_check_write_permission_readonly(self):
        config = MCPConfig(mode="read-only")
        with pytest.raises(PermissionError, match="not allowed in read-only mode"):
            config.check_write_permission("create_database")

    def test_check_write_permission_readwrite(self):
        config = MCPConfig(mode="read-write")
        # Should not raise
        config.check_write_permission("create_database")

    def test_get_database_path_explicit_mapping(self):
        config = MCPConfig(databases_map={"mydb": "/custom/path"})
        assert config.get_database_path("mydb") == "/custom/path"

    def test_get_database_path_default_root(self):
        config = MCPConfig(databases_root="/default/root")
        path = config.get_database_path("unknown_db")
        # Should resolve to the absolute form of databases_root
        assert "default" in path and "root" in path

    def test_get_enabled_tools_readonly(self):
        config = MCPConfig(mode="read-only")
        tools = config.get_enabled_tools()
        assert "list_databases" in tools
        assert "query_database" in tools
        # Write tools should NOT be present
        assert "create_database" not in tools
        assert "upsert_documents" not in tools

    def test_get_enabled_tools_readwrite(self):
        config = MCPConfig(mode="read-write")
        tools = config.get_enabled_tools()
        assert "list_databases" in tools
        assert "create_database" in tools
        assert "upsert_documents" in tools
        assert "delete_document" in tools

    def test_from_env_mode(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_MODE", "read-write")
        config = MCPConfig.from_env()
        assert config.mode == "read-write"

    def test_from_env_invalid_mode_ignored(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_MODE", "invalid-mode")
        config = MCPConfig.from_env()
        assert config.mode == "read-only"  # default

    def test_from_env_databases_root(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_DATABASES_ROOT", "/my/dbs")
        config = MCPConfig.from_env()
        assert config.databases_root == "/my/dbs"

    def test_from_env_databases_map(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_DATABASES_MAP", "db1=/path/one,db2=http://remote:8000")
        config = MCPConfig.from_env()
        assert config.databases_map["db1"] == "/path/one"
        assert config.databases_map["db2"] == "http://remote:8000"

    def test_from_env_embedding_provider(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_EMBEDDING_PROVIDER", "openai")
        config = MCPConfig.from_env()
        assert config.db_defaults["embedding_provider"] == "openai"

    def test_from_env_embedding_model(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_EMBEDDING_MODEL", "text-embedding-3-small")
        config = MCPConfig.from_env()
        assert config.db_defaults["embedding_model"] == "text-embedding-3-small"

    def test_from_env_chunk_size(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_CHUNK_SIZE", "1000")
        config = MCPConfig.from_env()
        assert config.db_defaults["chunk_size"] == 1000

    def test_from_env_chunk_size_invalid(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_CHUNK_SIZE", "not_a_number")
        config = MCPConfig.from_env()
        # Should fall back to default
        assert config.db_defaults["chunk_size"] == 500

    def test_from_env_log_level(self, monkeypatch):
        monkeypatch.setenv("LVDB_MCP_LOG_LEVEL", "debug")
        config = MCPConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_from_file(self, tmp_path):
        toml_content = textwrap.dedent("""\
            [mcp]
            mode = "read-write"
            log_level = "DEBUG"
            max_concurrent_operations = 5

            [databases]
            root = "/data/dbs"
            map = {mydb = "/data/mydb"}

            [defaults]
            embedding_provider = "openai"
            embedding_model = "text-embedding-ada-002"
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        config = MCPConfig.from_file(str(config_file))
        assert config.mode == "read-write"
        assert config.log_level == "DEBUG"
        assert config.max_concurrent_operations == 5
        assert config.databases_root == "/data/dbs"
        assert config.databases_map["mydb"] == "/data/mydb"
        assert config.db_defaults["embedding_provider"] == "openai"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MCPConfig.from_file("/nonexistent/config.toml")

    def test_load_defaults(self, monkeypatch):
        """load() with no file falls back to from_env()."""
        monkeypatch.delenv("LVDB_MCP_CONFIG", raising=False)
        monkeypatch.delenv("LVDB_MCP_MODE", raising=False)
        config = MCPConfig.load()
        assert config.mode == "read-only"

    def test_load_with_env_config_file(self, monkeypatch, tmp_path):
        toml_content = textwrap.dedent("""\
            [mcp]
            mode = "read-write"
        """)
        config_file = tmp_path / "env_config.toml"
        config_file.write_text(toml_content)
        monkeypatch.setenv("LVDB_MCP_CONFIG", str(config_file))
        config = MCPConfig.load()
        assert config.mode == "read-write"


# ---------------------------------------------------------------------------
# TOOL_REGISTRY tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolRegistry:
    """Verify the TOOL_REGISTRY is properly populated."""

    def test_registry_populated(self):
        from localvectordb_server.mcp.server import TOOL_REGISTRY

        assert len(TOOL_REGISTRY) > 0

    def test_read_only_tools_registered(self):
        from localvectordb_server.mcp.server import TOOL_REGISTRY

        expected_read_tools = [
            "list_databases",
            "get_database_info",
            "query_database",
            "find_related_documents",
            "filter_documents",
            "get_document",
            "check_documents_exist",
            "grep_documents",
            "list_prefixes",
            "get_metadata_schema",
            "get_system_info",
        ]
        for tool_name in expected_read_tools:
            assert tool_name in TOOL_REGISTRY, f"Missing read tool: {tool_name}"
            assert TOOL_REGISTRY[tool_name]["read_only"] is True

    def test_write_tools_registered(self):
        from localvectordb_server.mcp.server import TOOL_REGISTRY

        expected_write_tools = [
            "create_database",
            "delete_database",
            "upsert_documents",
            "update_document",
            "patch_document",
            "delete_document",
            "update_metadata_schema",
        ]
        for tool_name in expected_write_tools:
            assert tool_name in TOOL_REGISTRY, f"Missing write tool: {tool_name}"
            assert TOOL_REGISTRY[tool_name]["read_only"] is False

    def test_registry_entries_have_function(self):
        from localvectordb_server.mcp.server import TOOL_REGISTRY

        for name, info in TOOL_REGISTRY.items():
            assert callable(info["function"]), f"Tool {name} function not callable"
            assert "read_only" in info
            assert "registered" in info


# ---------------------------------------------------------------------------
# MCPManager tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMCPManager:
    """Tests for the MCPManager class."""

    def test_init(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config()
        manager = MCPManager(config)
        assert manager.config is config
        assert manager.databases == {}

    def test_list_databases_explicit_map(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config(
            databases_map={"alpha": "/a", "beta": "/b"},
            databases_root="/nonexistent",
        )
        manager = MCPManager(config)
        result = _run(manager.list_databases())
        assert "alpha" in result
        assert "beta" in result

    def test_list_databases_from_root(self, tmp_path):
        from localvectordb_server.mcp.server import MCPManager

        # Create fake .sqlite files
        (tmp_path / "docs.sqlite").touch()
        (tmp_path / "notes.sqlite").touch()

        config = _make_config(databases_root=str(tmp_path))
        manager = MCPManager(config)
        result = _run(manager.list_databases())
        assert "docs" in result
        assert "notes" in result

    def test_get_database_caches_instance(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config()
        manager = MCPManager(config)

        mock_db = MagicMock()
        with patch("localvectordb_server.mcp.server.VectorDB", return_value=mock_db) as mock_vectordb:
            db1 = _run(manager.get_database("testdb"))
            db2 = _run(manager.get_database("testdb"))
            assert db1 is db2
            # Caching means the underlying VectorDB is constructed exactly once.
            assert mock_vectordb.call_count == 1

    def test_get_database_not_found_raises(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config()
        manager = MCPManager(config)

        with patch(
            "localvectordb_server.mcp.server.VectorDB",
            side_effect=Exception("no such db"),
        ):
            with pytest.raises(DatabaseNotFoundError):
                _run(manager.get_database("missing"))

    def test_delete_database_removes_from_cache(self, tmp_path):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config(mode="read-write", databases_root=str(tmp_path))
        manager = MCPManager(config)

        mock_db = MagicMock()
        mock_db.close = MagicMock()
        manager.databases["mydb"] = mock_db

        _run(manager.delete_database("mydb"))
        assert "mydb" not in manager.databases
        mock_db.close.assert_called_once()

    def test_create_database_read_only_denied(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config(mode="read-only")
        manager = MCPManager(config)

        with pytest.raises(PermissionError):
            _run(manager.create_database("newdb"))

    def test_create_database_success(self, tmp_path):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config(mode="read-write", databases_root=str(tmp_path))
        manager = MCPManager(config)

        mock_db = MagicMock()
        with patch("localvectordb_server.mcp.server.VectorDB", return_value=mock_db):
            result = _run(manager.create_database("newdb"))
            assert result is mock_db
            assert "newdb" in manager.databases

    def test_cleanup(self):
        from localvectordb_server.mcp.server import MCPManager

        config = _make_config()
        manager = MCPManager(config)

        mock_db1 = MagicMock()
        mock_db2 = MagicMock()
        manager.databases = {"a": mock_db1, "b": mock_db2}

        _run(manager.cleanup())
        mock_db1.close.assert_called_once()
        mock_db2.close.assert_called_once()
        assert manager.databases == {}


@pytest.mark.unit
class TestConfigExampleSync:
    """``lvdb mcp config-example`` output must stay in sync with ``MCPConfig`` defaults."""

    def test_example_roundtrips_to_defaults(self, tmp_path):
        import tomllib

        from localvectordb_server.cli._mcp import _render_example_config

        text = _render_example_config()

        # It must be valid TOML...
        tomllib.loads(text)

        # ...and round-trip through MCPConfig.from_file back to the canonical
        # defaults, so a change to MCPConfig can never silently drift from the
        # emitted example the way a hardcoded literal did.
        path = tmp_path / "mcp-config.toml"
        path.write_text(text, encoding="utf-8")
        loaded = MCPConfig.from_file(str(path))
        defaults = MCPConfig()

        assert loaded.db_defaults == defaults.db_defaults
        assert loaded.remote_defaults == defaults.remote_defaults
        assert loaded.mode == defaults.mode
        assert loaded.log_level == defaults.log_level
        assert loaded.max_concurrent_operations == defaults.max_concurrent_operations
        assert loaded.operation_timeout == defaults.operation_timeout

    def test_example_lists_all_tools(self):
        from localvectordb_server.cli._mcp import _render_example_config

        text = _render_example_config()
        defaults = MCPConfig()
        for tool in defaults.read_only_tools + defaults.write_tools:
            assert f'"{tool}"' in text, f"config-example missing tool: {tool}"

    def test_sentences_overlap_is_sane(self):
        # Guards the specific bug this class was added for: overlap for the
        # "sentences" method is counted in *sentences*, so it must stay small
        # (a 500-token chunk holds far fewer than 50 sentences).
        defaults = MCPConfig()
        if defaults.db_defaults["chunking_method"] == "sentences":
            assert defaults.db_defaults["chunk_overlap"] <= 5


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_manager_fixture():
    """Set up a mock MCPManager as the global mcp_manager for tool tests."""
    import localvectordb_server.mcp.server as srv

    config = _make_config(mode="read-write")
    manager = MagicMock()
    manager.config = config
    manager.databases = {}

    original = srv.mcp_manager
    srv.mcp_manager = manager
    yield manager
    srv.mcp_manager = original


@pytest.mark.unit
class TestListDatabasesTool:
    def test_returns_databases(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import list_databases

        mcp_manager_fixture.list_databases = AsyncMock(return_value=["db1", "db2"])
        result = _run(list_databases())
        assert result["databases"] == ["db1", "db2"]
        assert result["count"] == 2
        assert result["mode"] == "read-write"

    def test_handles_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import list_databases

        mcp_manager_fixture.list_databases = AsyncMock(side_effect=RuntimeError("boom"))
        result = _run(list_databases())
        assert "error" in result
        assert "boom" in result["error"]


@pytest.mark.unit
class TestGetDatabaseInfoTool:
    def test_returns_info(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_database_info

        mock_db = MagicMock()
        mock_db.name = "testdb"
        mock_db.get_stats.return_value = {"documents": 10, "chunks": 50}
        mock_db.embedding_provider.provider_name = "ollama"
        mock_db.embedding_provider.model = "nomic-embed-text"
        mock_db.embedding_dimension = 768
        mock_db.chunking_method = "sentences"
        mock_db.chunk_size = 500
        mock_db.chunk_overlap = 50
        mock_db.fts_enabled = True
        mock_db.metadata_schema = None

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_database_info("testdb"))

        assert result["name"] == "testdb"
        assert result["stats"]["documents"] == 10
        assert result["config"]["embedding_provider"] == "ollama"
        assert result["config"]["embedding_dimension"] == 768

    def test_database_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_database_info

        mcp_manager_fixture.get_database = AsyncMock(side_effect=DatabaseNotFoundError("not found"))
        result = _run(get_database_info("missing"))
        assert result["error_code"] == "DATABASE_NOT_FOUND"


@pytest.mark.unit
class TestQueryDatabaseTool:
    def test_basic_query(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import query_database

        mock_result = _make_query_result(result_id="r1", score=0.9, content="match")
        mock_db = MagicMock()
        mock_db.query.return_value = [mock_result]
        # Ensure it uses sync path
        del mock_db.query_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        result = _run(query_database("testdb", "search text"))
        assert result["total_results"] == 1
        assert result["results"][0]["id"] == "r1"
        assert result["results"][0]["score"] == 0.9
        assert result["search_type"] == "hybrid"

    def test_query_with_position(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import query_database

        position = SimpleNamespace(index=0, total=3, start_char=0, end_char=100)
        mock_result = _make_query_result(document_id="doc1", position=position)
        mock_db = MagicMock()
        mock_db.query.return_value = [mock_result]
        del mock_db.query_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(query_database("testdb", "query"))
        assert result["results"][0]["document_id"] == "doc1"
        assert result["results"][0]["position"]["index"] == 0
        assert result["results"][0]["position"]["total"] == 3

    def test_query_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import query_database

        mock_result = _make_query_result()
        mock_db = MagicMock()
        mock_db.query_async = AsyncMock(return_value=[mock_result])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(query_database("testdb", "query", search_type="vector"))
        assert result["search_type"] == "vector"
        assert result["total_results"] == 1

    def test_query_search_level_forwarded(self, mcp_manager_fixture):
        """search_level should be forwarded to the database query."""
        from localvectordb_server.mcp.server import query_database

        mock_result = _make_query_result()
        mock_db = MagicMock()
        mock_db.query.return_value = [mock_result]
        del mock_db.query_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        _run(query_database("testdb", "query", search_level="sections", return_type="sections"))
        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["search_level"] == "sections"
        assert call_kwargs["return_type"] == "sections"

    def test_query_search_level_async_forwarded(self, mcp_manager_fixture):
        """search_level should be forwarded on the async query path."""
        from localvectordb_server.mcp.server import query_database

        mock_db = MagicMock()
        mock_db.query_async = AsyncMock(return_value=[_make_query_result()])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        _run(query_database("testdb", "query", search_level="documents"))
        assert mock_db.query_async.call_args[1]["search_level"] == "documents"

    def test_query_database_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import query_database

        mcp_manager_fixture.get_database = AsyncMock(side_effect=DatabaseNotFoundError("no db"))
        result = _run(query_database("missing", "query"))
        assert result["error_code"] == "DATABASE_NOT_FOUND"

    def test_query_unexpected_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import query_database

        mcp_manager_fixture.get_database = AsyncMock(side_effect=RuntimeError("unexpected"))
        result = _run(query_database("testdb", "query"))
        assert "error" in result
        assert result["error_type"] == "RuntimeError"


@pytest.mark.unit
class TestFilterDocumentsTool:
    def test_basic_filter(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import filter_documents

        mock_doc = _make_document(doc_id="d1", content="filtered")
        mock_db = MagicMock()
        mock_db.filter.return_value = [mock_doc]
        del mock_db.filter_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        result = _run(filter_documents("testdb", {"category": "test"}))
        assert result["count"] == 1
        assert result["documents"][0]["id"] == "d1"
        assert result["filters"] == {"category": "test"}

    def test_filter_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import filter_documents

        mock_doc = _make_document()
        mock_db = MagicMock()
        mock_db.filter_async = AsyncMock(return_value=[mock_doc])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(filter_documents("testdb", {"x": 1}, limit=50, offset=10))
        assert result["count"] == 1
        mock_db.filter_async.assert_called_once_with(where={"x": 1}, limit=50, offset=10)

    def test_filter_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import filter_documents

        mcp_manager_fixture.get_database = AsyncMock(side_effect=RuntimeError("fail"))
        result = _run(filter_documents("testdb", {}))
        assert "error" in result


@pytest.mark.unit
class TestGetDocumentTool:
    def test_get_existing_document(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_doc = _make_document(doc_id="doc1", content="hello")
        mock_db = MagicMock()
        mock_db.get.return_value = mock_doc
        del mock_db.get_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1"))
        assert result["id"] == "doc1"
        assert result["content"] == "hello"
        assert result["content_hash"] == "abc123"

    def test_get_nonexistent_document(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = MagicMock()
        mock_db.get.return_value = None
        del mock_db.get_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "missing_id"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"

    def test_get_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_doc = _make_document()
        mock_db = MagicMock()
        mock_db.get_async = AsyncMock(return_value=mock_doc)

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1"))
        assert result["id"] == "doc1"


@pytest.mark.unit
class TestGetDocumentPortions:
    """Sub-document retrieval via the ``get_document`` tool's portion args."""

    MARKDOWN = "# Introduction\nalpha beta gamma.\n\n## Installation\ninstall me now.\n"

    def _db_with_doc(self, content):
        doc = _make_document(doc_id="doc1", content=content)
        mock_db = MagicMock()
        mock_db.get.return_value = doc
        return mock_db

    def test_section(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", section="Installation"))
        assert result["mode"] == "section"
        assert "install me now" in result["content"]

    def test_outline(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", outline=True))
        assert result["mode"] == "outline"
        headings = [item["heading"] for item in result["outline"]]
        assert "Installation" in headings

    def test_char_range(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", char_range="0:14"))
        assert result["mode"] == "range"
        assert result["content"] == "# Introduction"

    def test_chunk(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        position = SimpleNamespace(to_dict=lambda: {"start": 0, "end": 10})
        chunk = SimpleNamespace(index=0, content="chunk zero", position=position)
        mock_db.get_chunks.return_value = [chunk]

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", chunk="0"))
        assert result["mode"] == "chunk"
        assert result["chunks"][0]["index"] == 0
        assert result["chunks"][0]["content"] == "chunk zero"

    def test_bad_range_is_value_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", char_range="not-a-range"))
        assert result["error_type"] == "ValueError"

    def test_mutually_exclusive_modes(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_document

        mock_db = self._db_with_doc(self.MARKDOWN)
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_document("testdb", "doc1", char_range="0:5", line_range="1:1"))
        assert "error" in result


@pytest.mark.unit
class TestFindRelatedDocumentsTool:
    def test_returns_neighbors(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import find_related_documents

        mock_db = MagicMock()
        del mock_db.nearest_neighbors_async
        mock_db.nearest_neighbors.return_value = [_make_query_result(result_id="n1", score=0.9)]

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(find_related_documents("testdb", "doc1"))
        assert result["document_id"] == "doc1"
        assert result["results"][0]["id"] == "n1"
        assert result["total_results"] == 1

    def test_forwards_options(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import find_related_documents

        mock_db = MagicMock()
        del mock_db.nearest_neighbors_async
        mock_db.nearest_neighbors.return_value = []

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        _run(find_related_documents("testdb", "doc1", k=3, score_threshold=0.5, filters={"a": 1}))
        call = mock_db.nearest_neighbors.call_args
        assert call[0][0] == "doc1"
        assert call[1]["k"] == 3
        assert call[1]["score_threshold"] == 0.5
        assert call[1]["filters"] == {"a": 1}

    def test_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import find_related_documents

        mock_db = MagicMock()
        mock_db.nearest_neighbors_async = AsyncMock(return_value=[_make_query_result(result_id="n2")])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(find_related_documents("testdb", "doc1"))
        assert result["results"][0]["id"] == "n2"

    def test_document_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import find_related_documents

        mock_db = MagicMock()
        del mock_db.nearest_neighbors_async
        mock_db.nearest_neighbors.side_effect = DocumentNotFoundError("missing")

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(find_related_documents("testdb", "missing"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"


@pytest.mark.unit
class TestCheckDocumentsExistTool:
    def test_check_existing_documents(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import check_documents_exist

        mock_db = MagicMock()
        # exists() returns a List[bool] aligned to the input ids, not a mapping.
        mock_db.exists.return_value = [True, False]
        del mock_db.exists_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(check_documents_exist("testdb", ["d1", "d2"]))
        assert result["exists"]["d1"] is True
        assert result["exists"]["d2"] is False
        assert result["total_checked"] == 2
        assert result["total_found"] == 1

    def test_check_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import check_documents_exist

        mock_db = MagicMock()
        mock_db.exists_async = AsyncMock(return_value=[True])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(check_documents_exist("testdb", ["d1"]))
        assert result["exists"]["d1"] is True
        assert result["total_found"] == 1

    def test_check_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import check_documents_exist

        mcp_manager_fixture.get_database = AsyncMock(side_effect=RuntimeError("fail"))
        result = _run(check_documents_exist("testdb", ["d1"]))
        assert "error" in result


@pytest.mark.unit
class TestGrepDocumentsTool:
    def test_basic_grep(self, mcp_manager_fixture):
        from localvectordb.core import GrepMatch
        from localvectordb_server.mcp.server import grep_documents

        mock_db = MagicMock()
        mock_db.grep.return_value = [
            GrepMatch(doc_id="docs/a", line_number=2, line="TODO: fix", start=0, end=4, match="TODO"),
        ]
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        result = _run(grep_documents("testdb", "TODO"))
        assert result["total_matches"] == 1
        assert result["matches"][0]["doc_id"] == "docs/a"
        assert result["matches"][0]["line_number"] == 2
        assert result["matches"][0]["match"] == "TODO"
        assert result["pattern"] == "TODO"
        assert result["truncated"] is False

    def test_forwards_options(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import grep_documents

        mock_db = MagicMock()
        mock_db.grep.return_value = []
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        _run(
            grep_documents(
                "testdb",
                "foo",
                regex=True,
                ignore_case=True,
                whole_word=True,
                context=2,
                prefix="docs/",
                filters={"id": {"$startswith": "docs/"}},
                max_count=3,
                limit=10,
            )
        )
        call = mock_db.grep.call_args
        assert call[0][0] == "foo"
        assert call[1]["regex"] is True
        assert call[1]["ignore_case"] is True
        assert call[1]["whole_word"] is True
        assert call[1]["context"] == 2
        assert call[1]["prefix"] == "docs/"
        # The tool exposes the filter as `filters` but forwards it as `where`.
        assert call[1]["where"] == {"id": {"$startswith": "docs/"}}
        assert call[1]["max_count"] == 3
        assert call[1]["limit"] == 10

    def test_truncated_flag_when_limit_hit(self, mcp_manager_fixture):
        from localvectordb.core import GrepMatch
        from localvectordb_server.mcp.server import grep_documents

        mock_db = MagicMock()
        mock_db.grep.return_value = [
            GrepMatch(doc_id="d", line_number=i, line="x", start=0, end=1, match="x") for i in range(1, 3)
        ]
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(grep_documents("testdb", "x", limit=2))
        assert result["truncated"] is True

    def test_invalid_regex_is_value_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import grep_documents

        mock_db = MagicMock()
        mock_db.grep.side_effect = ValueError("bad regex")
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(grep_documents("testdb", "(unclosed", regex=True))
        assert result["error_type"] == "ValueError"

    def test_not_supported_for_remote(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import grep_documents

        # A remote database has no grep() method -> reported as NOT_SUPPORTED.
        mock_db = MagicMock(spec=[])
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(grep_documents("testdb", "x"))
        assert result["error_code"] == "NOT_SUPPORTED"

    def test_database_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import grep_documents

        mcp_manager_fixture.get_database = AsyncMock(side_effect=DatabaseNotFoundError("no db"))
        result = _run(grep_documents("missing", "x"))
        assert result["error_code"] == "DATABASE_NOT_FOUND"


@pytest.mark.unit
class TestListPrefixesTool:
    def test_basic_listing(self, mcp_manager_fixture):
        from localvectordb.core import PrefixEntry, PrefixListing
        from localvectordb_server.mcp.server import list_prefixes

        listing = PrefixListing(
            prefix="docs/",
            delimiter="/",
            prefixes=[PrefixEntry(name="reports/", path="docs/reports/", is_prefix=True, count=3)],
            documents=[PrefixEntry(name="readme", path="docs/readme", is_prefix=False, count=1)],
        )
        mock_db = MagicMock()
        mock_db.list_prefixes.return_value = listing
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        result = _run(list_prefixes("testdb", prefix="docs/"))
        assert result["prefix"] == "docs/"
        assert result["prefix_count"] == 1
        assert result["document_count"] == 1
        assert result["prefixes"][0]["path"] == "docs/reports/"
        assert result["documents"][0]["name"] == "readme"

    def test_forwards_delimiter(self, mcp_manager_fixture):
        from localvectordb.core import PrefixListing
        from localvectordb_server.mcp.server import list_prefixes

        mock_db = MagicMock()
        mock_db.list_prefixes.return_value = PrefixListing(prefix="", delimiter="::")
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        _run(list_prefixes("testdb", prefix="", delimiter="::"))
        mock_db.list_prefixes.assert_called_once_with(prefix="", delimiter="::")

    def test_empty_delimiter_is_value_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import list_prefixes

        mock_db = MagicMock()
        mock_db.list_prefixes.side_effect = ValueError("delimiter must not be empty")
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(list_prefixes("testdb", delimiter=""))
        assert result["error_type"] == "ValueError"

    def test_not_supported_for_remote(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import list_prefixes

        mock_db = MagicMock(spec=[])
        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(list_prefixes("testdb"))
        assert result["error_code"] == "NOT_SUPPORTED"

    def test_database_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import list_prefixes

        mcp_manager_fixture.get_database = AsyncMock(side_effect=DatabaseNotFoundError("no db"))
        result = _run(list_prefixes("missing"))
        assert result["error_code"] == "DATABASE_NOT_FOUND"


@pytest.mark.unit
class TestGetMetadataSchemaTool:
    def test_schema_present(self, mcp_manager_fixture):
        from localvectordb.core import MetadataField, MetadataFieldType
        from localvectordb_server.mcp.server import get_metadata_schema

        field = MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=False)
        mock_db = MagicMock()
        mock_db.metadata_schema = {"title": field}

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_metadata_schema("testdb"))
        assert "title" in result["schema"]
        assert result["schema"]["title"]["type"] == "text"
        assert result["schema"]["title"]["indexed"] is True

    def test_schema_absent(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_metadata_schema

        mock_db = MagicMock()
        mock_db.metadata_schema = None

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(get_metadata_schema("testdb"))
        assert result["schema"] == {}
        assert "message" in result


@pytest.mark.unit
class TestGetSystemInfoTool:
    def test_returns_system_info(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_system_info

        mcp_manager_fixture.list_databases = AsyncMock(return_value=["a", "b"])

        with (
            patch(
                "localvectordb_server.mcp.server.get_system_version",
                return_value="1.0.0",
            ),
            patch("localvectordb_server.mcp.server.EmbeddingRegistry") as mock_registry,
        ):
            mock_registry.list.return_value = ["ollama", "openai"]
            result = _run(get_system_info())

        assert result["version"] == "1.0.0"
        assert result["mode"] == "read-write"
        assert result["databases_count"] == 2
        assert result["available_providers"] == ["ollama", "openai"]

    def test_handles_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import get_system_info

        mcp_manager_fixture.list_databases = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "localvectordb_server.mcp.server.get_system_version",
                return_value="1.0.0",
            ),
            patch("localvectordb_server.mcp.server.EmbeddingRegistry") as mock_registry,
        ):
            mock_registry.list.return_value = []
            result = _run(get_system_info())

        assert "error" in result


# ---------------------------------------------------------------------------
# Write tool tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDatabaseTool:
    def test_create_success(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import create_database

        mock_db = MagicMock()
        mock_db.name = "newdb"
        mcp_manager_fixture.create_database = AsyncMock(return_value=mock_db)

        result = _run(create_database("newdb"))
        assert result["status"] == "success"
        assert "newdb" in result["message"]

    def test_create_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import create_database

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(create_database("newdb"))
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_create_with_params(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import create_database

        mock_db = MagicMock()
        mock_db.name = "custom"
        mcp_manager_fixture.create_database = AsyncMock(return_value=mock_db)

        result = _run(
            create_database(
                "custom",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                chunk_size=1000,
            )
        )
        assert result["status"] == "success"
        assert result["config"]["embedding_provider"] == "openai"
        assert result["config"]["embedding_model"] == "text-embedding-3-small"
        assert result["config"]["chunk_size"] == 1000

    def test_create_error(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import create_database

        mcp_manager_fixture.create_database = AsyncMock(side_effect=RuntimeError("disk full"))
        result = _run(create_database("newdb"))
        assert "error" in result
        assert "disk full" in result["error"]


@pytest.mark.unit
class TestDeleteDatabaseTool:
    def test_delete_success(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_database

        mcp_manager_fixture.delete_database = AsyncMock()
        result = _run(delete_database("olddb"))
        assert result["status"] == "success"

    def test_delete_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_database

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(delete_database("olddb"))
        assert result["error_code"] == "PERMISSION_DENIED"


@pytest.mark.unit
class TestUpsertDocumentsTool:
    def test_upsert_single_document(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import upsert_documents

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["id1"]
        del mock_db.upsert_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(upsert_documents("testdb", "single doc"))
        assert result["status"] == "success"
        assert result["ids"] == ["id1"]
        # Verify the string was wrapped in a list
        mock_db.upsert.assert_called_once()
        call_kwargs = mock_db.upsert.call_args
        assert call_kwargs.kwargs["documents"] == ["single doc"]

    def test_upsert_multiple_documents(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import upsert_documents

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["id1", "id2"]
        del mock_db.upsert_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(
            upsert_documents(
                "testdb",
                ["doc one", "doc two"],
                metadata=[{"a": 1}, {"a": 2}],
                ids=["id1", "id2"],
            )
        )
        assert result["status"] == "success"
        assert len(result["ids"]) == 2

    def test_upsert_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import upsert_documents

        mock_db = MagicMock()
        mock_db.upsert_async = AsyncMock(return_value=["id1"])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(upsert_documents("testdb", "doc"))
        assert result["status"] == "success"

    def test_upsert_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import upsert_documents

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(upsert_documents("testdb", "doc"))
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_upsert_normalizes_single_metadata(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import upsert_documents

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["id1"]
        del mock_db.upsert_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(upsert_documents("testdb", "doc", metadata={"key": "val"}, ids="single_id"))
        assert result["status"] == "success"
        call_kwargs = mock_db.upsert.call_args.kwargs
        assert call_kwargs["metadata"] == [{"key": "val"}]
        assert call_kwargs["ids"] == ["single_id"]


@pytest.mark.unit
class TestUpdateDocumentTool:
    def test_update_success(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_document

        mock_db = MagicMock()
        del mock_db.update_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(update_document("testdb", "doc1", content="new text"))
        assert result["status"] == "success"
        mock_db.update.assert_called_once_with(doc_id="doc1", content="new text", metadata=None)

    def test_update_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_document

        mock_db = MagicMock()
        mock_db.update_async = AsyncMock()

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(update_document("testdb", "doc1", metadata={"x": 1}))
        assert result["status"] == "success"

    def test_update_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_document

        mock_db = MagicMock()
        mock_db.update.side_effect = DocumentNotFoundError("not found", missing_ids=["doc1"])
        del mock_db.update_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(update_document("testdb", "doc1", content="new"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"

    def test_update_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_document

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(update_document("testdb", "doc1", content="x"))
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_update_noop_is_reported_as_not_updated(self, mcp_manager_fixture):
        # update() returns False when the document already matches. The tool used to
        # discard the bool and always claim success, so an agent could not tell
        # "my edit landed" from "nothing changed".
        from localvectordb_server.mcp.server import update_document

        mock_db = MagicMock()
        mock_db.update.return_value = False
        del mock_db.update_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(update_document("testdb", "doc1", content="identical text"))
        assert result["updated"] is False
        assert "error" not in result

    def test_update_real_change_is_reported_as_updated(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_document

        mock_db = MagicMock()
        mock_db.update.return_value = True
        del mock_db.update_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(update_document("testdb", "doc1", content="new text"))
        assert result["updated"] is True


@pytest.mark.unit
class TestPatchDocumentTool:
    def _result(self, updated=True):
        from localvectordb.patching import PatchResult

        return PatchResult(updated=updated, new_hash="abc123", ops_applied=1)

    def test_patch_success_builds_replace_op(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        del mock_db.patch_async
        mock_db.patch.return_value = self._result()

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="brown", new_string="red"))
        assert result["status"] == "success"
        assert result["updated"] is True
        assert result["new_hash"] == "abc123"
        assert result["ops_applied"] == 1
        mock_db.patch.assert_called_once_with(
            doc_id="doc1",
            ops=[{"op": "replace", "find": "brown", "replace": "red", "count": 1}],
            expect_hash=None,
        )

    def test_patch_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        mock_db.patch_async = AsyncMock(return_value=self._result())

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="a", new_string="b", count=2))
        assert result["status"] == "success"
        mock_db.patch_async.assert_awaited_once_with(
            doc_id="doc1",
            ops=[{"op": "replace", "find": "a", "replace": "b", "count": 2}],
            expect_hash=None,
        )

    def test_patch_noop_reported(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        del mock_db.patch_async
        mock_db.patch.return_value = self._result(updated=False)

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="x", new_string="x"))
        assert result["updated"] is False
        assert "error" not in result

    def test_patch_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        del mock_db.patch_async
        mock_db.patch.side_effect = DocumentNotFoundError("not found", missing_ids=["doc1"])

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="a", new_string="b"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"

    def test_patch_conflict(self, mcp_manager_fixture):
        from localvectordb.exceptions import PatchConflictError
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        del mock_db.patch_async
        mock_db.patch.side_effect = PatchConflictError("stale")

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="a", new_string="b", expect_hash="x"))
        assert result["error_code"] == "HASH_CONFLICT"

    def test_patch_failed(self, mcp_manager_fixture):
        from localvectordb.exceptions import PatchError
        from localvectordb_server.mcp.server import patch_document

        mock_db = MagicMock()
        del mock_db.patch_async
        mock_db.patch.side_effect = PatchError("unmatched find")

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(patch_document("testdb", "doc1", old_string="zzz", new_string="b"))
        assert result["error_code"] == "PATCH_FAILED"

    def test_patch_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import patch_document

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(patch_document("testdb", "doc1", old_string="a", new_string="b"))
        assert result["error_code"] == "PERMISSION_DENIED"


@pytest.mark.unit
class TestDeleteDocumentTool:
    def test_delete_success(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_document

        mock_db = MagicMock()
        mock_db.delete.return_value = 1
        del mock_db.delete_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(delete_document("testdb", "doc1"))
        assert result["status"] == "success"
        assert result["deleted_count"] == 1

    def test_delete_not_found(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_document

        mock_db = MagicMock()
        mock_db.delete.return_value = 0
        del mock_db.delete_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(delete_document("testdb", "missing"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"

    def test_delete_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_document

        mock_db = MagicMock()
        mock_db.delete_async = AsyncMock(return_value=1)

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)
        result = _run(delete_document("testdb", "doc1"))
        assert result["status"] == "success"

    def test_delete_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import delete_document

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(delete_document("testdb", "doc1"))
        assert result["error_code"] == "PERMISSION_DENIED"


@pytest.mark.unit
class TestUpdateMetadataSchemaTool:
    def test_update_success(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_metadata_schema

        mock_db = MagicMock()
        mock_db.update_metadata_schema = MagicMock()
        del mock_db.update_metadata_schema_async

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        with patch(
            "localvectordb_server.mcp.server.parse_metadata_schema",
            return_value={"title": MagicMock()},
        ):
            result = _run(update_metadata_schema("testdb", {"title": "text"}))
        assert result["status"] == "success"

    def test_update_async_path(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_metadata_schema

        mock_db = MagicMock()
        mock_db.update_metadata_schema_async = AsyncMock()

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        with patch(
            "localvectordb_server.mcp.server.parse_metadata_schema",
            return_value={"title": MagicMock()},
        ):
            result = _run(update_metadata_schema("testdb", {"title": "text"}))
        assert result["status"] == "success"

    def test_update_not_supported(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_metadata_schema

        mock_db = MagicMock(spec=[])  # empty spec -> no attributes

        mcp_manager_fixture.get_database = AsyncMock(return_value=mock_db)

        with patch(
            "localvectordb_server.mcp.server.parse_metadata_schema",
            return_value={},
        ):
            result = _run(update_metadata_schema("testdb", {}))
        assert result["error_code"] == "NOT_SUPPORTED"

    def test_update_readonly_denied(self, mcp_manager_fixture):
        from localvectordb_server.mcp.server import update_metadata_schema

        mcp_manager_fixture.config.mode = "read-only"
        result = _run(update_metadata_schema("testdb", {"x": "text"}))
        assert result["error_code"] == "PERMISSION_DENIED"
