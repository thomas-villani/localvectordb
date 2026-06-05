"""
Tests for localvectordb_server._dbmanager module.

Covers DatabaseManager lifecycle: creating, opening, listing, deleting databases,
error handling for invalid/non-existent databases, name validation,
connection management, cleanup, and health monitoring.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from localvectordb_server._dbmanager import (
    DatabaseManager,
    DatabaseRegistry,
    DatabaseRegistryError,
)
from localvectordb_server._error_handlers import APIError
from localvectordb_server.config import Config, DatabaseSettings, EmbeddingSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_app(tmp_path):
    """Build a real Config pointing at a temp dir.

    DatabaseManager now takes a Config object directly (the FastAPI refactor dropped
    the Flask app wrapper), so this returns a Config with the database root and an
    in-process SimpleCache registry suitable for unit tests.
    """
    config_obj = Config()
    config_obj.database.root_dir = str(tmp_path)
    config_obj.server.db_registry_type = "SimpleCache"
    config_obj.server.db_registry_settings = {}
    config_obj.server.cache_type = "SimpleCache"
    config_obj.server.cache_settings = None
    config_obj.server.use_single_cache = False
    return config_obj


def _make_mock_db(name="testdb", closed=False, doc_count=0, chunk_count=0):
    """Return a mock that looks like a LocalVectorDB instance."""
    db = MagicMock()
    db.name = name
    db.closed = closed
    db.embedding_dimension = 384
    db.get_stats.return_value = {
        "documents": doc_count,
        "chunks": chunk_count,
        "embedding_model": "mock-model",
        "embedding_provider": "mock",
        "embedding_dimension": 384,
        "chunk_size": 500,
        "chunking_method": "lines",
        "chunk_overlap": 1,
    }
    return db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_dir(tmp_path):
    """Provide a temporary directory for databases."""
    return tmp_path


@pytest.fixture()
def mock_app(tmp_db_dir):
    return _make_mock_app(tmp_db_dir)


@pytest.fixture()
def manager(mock_app):
    """Create a DatabaseManager and ensure it is torn down after the test."""
    with patch.object(DatabaseManager, "_sync_registry_from_filesystem"):
        mgr = DatabaseManager(mock_app)
        yield mgr
        mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseRegistry (unit-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatabaseRegistry:
    """Tests for the DatabaseRegistry helper that wraps cachelib."""

    def test_register_and_list(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        registry.register_database("db1", {"name": "db1"})
        registry.register_database("db2", {"name": "db2"})

        assert registry.list_databases() == ["db1", "db2"]

    def test_database_exists(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        registry.register_database("mydb", {"name": "mydb"})

        assert registry.database_exists("mydb") is True
        assert registry.database_exists("nope") is False

    def test_unregister_database(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        registry.register_database("db1", {"name": "db1"})
        registry.unregister_database("db1")

        assert registry.database_exists("db1") is False
        assert registry.list_databases() == []

    def test_get_database_metadata(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        meta = {"name": "db1", "extra": 42}
        registry.register_database("db1", meta)

        retrieved = registry.get_database_metadata("db1")
        assert retrieved["name"] == "db1"
        assert retrieved["extra"] == 42

    def test_get_metadata_nonexistent_returns_none(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        assert registry.get_database_metadata("ghost") is None

    def test_update_database_metadata(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        registry.register_database("db1", {"version": 1})
        registry.update_database_metadata("db1", {"version": 2})

        assert registry.get_database_metadata("db1")["version"] == 2

    def test_update_metadata_nonexistent_raises(self):
        from cachelib import SimpleCache

        registry = DatabaseRegistry(SimpleCache())
        with pytest.raises(DatabaseRegistryError):
            registry.update_database_metadata("ghost", {"x": 1})


# ---------------------------------------------------------------------------
# DatabaseManager – name validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatabaseNameValidation:
    """Tests for _validate_database_name."""

    def test_valid_names(self, manager):
        assert manager._validate_database_name("mydb") is True
        assert manager._validate_database_name("my-db") is True
        assert manager._validate_database_name("my_db") is True
        assert manager._validate_database_name("DB123") is True
        assert manager._validate_database_name("a") is True

    def test_empty_name_rejected(self, manager):
        assert manager._validate_database_name("") is False

    def test_none_rejected(self, manager):
        assert manager._validate_database_name(None) is False

    def test_path_traversal_rejected(self, manager):
        assert manager._validate_database_name("..") is False
        assert manager._validate_database_name("../etc") is False
        assert manager._validate_database_name("foo/../bar") is False

    def test_hidden_file_rejected(self, manager):
        assert manager._validate_database_name(".hidden") is False

    def test_invalid_characters_rejected(self, manager):
        for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
            assert manager._validate_database_name(f"db{char}name") is False

    def test_null_byte_rejected(self, manager):
        assert manager._validate_database_name("db\x00name") is False

    def test_control_characters_rejected(self, manager):
        assert manager._validate_database_name("db\x01name") is False

    def test_windows_reserved_names_rejected(self, manager):
        for name in ["con", "prn", "aux", "nul", "com1", "lpt1"]:
            assert manager._validate_database_name(name) is False

    def test_too_long_name_rejected(self, manager):
        assert manager._validate_database_name("a" * 65) is False

    def test_max_length_accepted(self, manager):
        assert manager._validate_database_name("a" * 64) is True

    def test_name_starting_with_digit(self, manager):
        assert manager._validate_database_name("1abc") is True

    def test_unicode_path_separators_rejected(self, manager):
        assert manager._validate_database_name("db\u2215name") is False  # division slash
        assert manager._validate_database_name("db\uff0fname") is False  # fullwidth solidus


# ---------------------------------------------------------------------------
# DatabaseManager – create_db
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDatabase:
    """Tests for DatabaseManager.create_db."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    @patch("localvectordb.database.LocalVectorDB")
    def test_create_db_success(self, MockLocalVectorDB, _mock_sync, mock_app, tmp_db_dir):
        mock_db = _make_mock_db("newdb")
        MockLocalVectorDB.return_value = mock_db

        mgr = DatabaseManager(mock_app)
        try:
            db_config = DatabaseSettings()
            emb_config = EmbeddingSettings()

            result = mgr.create_db("newdb", None, db_config, emb_config)

            assert result is mock_db
            assert "newdb" in mgr.databases
            assert mgr.registry.database_exists("newdb")
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_create_db_invalid_name_raises(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            with pytest.raises(APIError) as exc_info:
                mgr.create_db("../bad", None, DatabaseSettings(), EmbeddingSettings())
            assert exc_info.value.error_code == "INVALID_DATABASE_NAME"
            assert exc_info.value.status_code == 400
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    @patch("localvectordb.database.LocalVectorDB")
    def test_create_db_duplicate_raises(self, MockLocalVectorDB, _mock_sync, mock_app):
        mock_db = _make_mock_db("dup")
        MockLocalVectorDB.return_value = mock_db

        mgr = DatabaseManager(mock_app)
        try:
            db_config = DatabaseSettings()
            emb_config = EmbeddingSettings()

            mgr.create_db("dup", None, db_config, emb_config)

            with pytest.raises(APIError) as exc_info:
                mgr.create_db("dup", None, db_config, emb_config)
            assert exc_info.value.error_code == "DATABASE_ALREADY_EXISTS"
            assert exc_info.value.status_code == 409
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    @patch("localvectordb.database.LocalVectorDB")
    def test_create_db_failure_cleans_up(self, MockLocalVectorDB, _mock_sync, mock_app, tmp_db_dir):
        MockLocalVectorDB.side_effect = RuntimeError("boom")

        mgr = DatabaseManager(mock_app)
        try:
            with pytest.raises(APIError) as exc_info:
                mgr.create_db("faildb", None, DatabaseSettings(), EmbeddingSettings())
            assert exc_info.value.error_code == "DATABASE_CREATION_FAILED"
            # Should not be in the cache or registry
            assert "faildb" not in mgr.databases
            assert not mgr.registry.database_exists("faildb")
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – get_db
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetDatabase:
    """Tests for DatabaseManager.get_db."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_get_db_from_cache(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            mock_db = _make_mock_db("cached")
            mgr.databases["cached"] = (mock_db, datetime.now(UTC))
            mgr.registry.register_database("cached", {"name": "cached"})

            result = mgr.get_db("cached")
            assert result is mock_db
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_get_db_not_found_raises(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            with pytest.raises(APIError) as exc_info:
                mgr.get_db("nonexistent")
            assert exc_info.value.error_code == "DATABASE_NOT_FOUND"
            assert exc_info.value.status_code == 404
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_get_db_unhealthy_reloads(self, _mock_sync, mock_app):
        """If the cached db fails health check, it should be evicted."""
        mgr = DatabaseManager(mock_app)
        try:
            unhealthy_db = _make_mock_db("sick", closed=True)
            mgr.databases["sick"] = (unhealthy_db, datetime.now(UTC))
            # Not in registry, so after eviction it will not be found
            with pytest.raises(APIError) as exc_info:
                mgr.get_db("sick")
            assert exc_info.value.error_code == "DATABASE_NOT_FOUND"
            # The unhealthy db should have been removed from cache
            assert "sick" not in mgr.databases
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    @patch("localvectordb.database.LocalVectorDB")
    def test_get_db_loads_from_disk(self, MockLocalVectorDB, _mock_sync, mock_app, tmp_db_dir):
        """When db is in registry but not cached, it should load from disk."""
        mock_db = _make_mock_db("ondisk")
        MockLocalVectorDB.return_value = mock_db

        mgr = DatabaseManager(mock_app)
        try:
            mgr.registry.register_database("ondisk", {"name": "ondisk"})

            result = mgr.get_db("ondisk")
            assert result is mock_db
            assert "ondisk" in mgr.databases
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – list_databases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListDatabases:
    """Tests for DatabaseManager.list_databases."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_list_empty(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            assert mgr.list_databases() == []
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_list_returns_registered_dbs(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            mgr.registry.register_database("alpha", {"name": "alpha"})
            mgr.registry.register_database("beta", {"name": "beta"})

            result = mgr.list_databases()
            assert result == ["alpha", "beta"]
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_list_sorted(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            mgr.registry.register_database("zeta", {"name": "zeta"})
            mgr.registry.register_database("alpha", {"name": "alpha"})

            result = mgr.list_databases()
            assert result == ["alpha", "zeta"]
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – delete_database
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteDatabase:
    """Tests for DatabaseManager.delete_database."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_delete_nonexistent_returns_false(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            result = mgr.delete_database("ghost")
            assert result is False
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_delete_removes_from_registry_and_cache(self, _mock_sync, mock_app, tmp_db_dir):
        mgr = DatabaseManager(mock_app)
        try:
            mock_db = _make_mock_db("delme")
            mgr.databases["delme"] = (mock_db, datetime.now(UTC))
            mgr.registry.register_database("delme", {"name": "delme"})

            result = mgr.delete_database("delme")

            assert result is True
            assert "delme" not in mgr.databases
            assert not mgr.registry.database_exists("delme")
            mock_db.close.assert_called_once()
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_delete_removes_files(self, _mock_sync, mock_app, tmp_db_dir):
        mgr = DatabaseManager(mock_app)
        try:
            # Create fake files on disk
            sqlite_file = tmp_db_dir / "filedb.sqlite"
            faiss_file = tmp_db_dir / "filedb.faiss"
            sqlite_file.touch()
            faiss_file.touch()

            mgr.registry.register_database("filedb", {"name": "filedb"})

            result = mgr.delete_database("filedb")

            assert result is True
            assert not sqlite_file.exists()
            assert not faiss_file.exists()
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – close_all / cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseAll:
    """Tests for DatabaseManager.close_all and connection cleanup."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_close_all_closes_databases(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        db1 = _make_mock_db("db1")
        db2 = _make_mock_db("db2")
        mgr.databases["db1"] = (db1, datetime.now(UTC))
        mgr.databases["db2"] = (db2, datetime.now(UTC))

        mgr.close_all()

        db1.close.assert_called_once()
        db2.close.assert_called_once()
        assert len(mgr.databases) == 0

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_close_all_is_idempotent(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        mgr.close_all()
        # Second call should be a no-op, not raise
        mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_close_all_signals_shutdown_event(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        assert not mgr._shutdown_event.is_set()
        mgr.close_all()
        assert mgr._shutdown_event.is_set()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_close_all_handles_close_errors(self, _mock_sync, mock_app):
        """Closing a database that errors should not prevent other databases from closing."""
        mgr = DatabaseManager(mock_app)
        bad_db = _make_mock_db("bad")
        bad_db.close.side_effect = RuntimeError("close failed")
        good_db = _make_mock_db("good")

        mgr.databases["bad"] = (bad_db, datetime.now(UTC))
        mgr.databases["good"] = (good_db, datetime.now(UTC))

        # Should not raise
        mgr.close_all()
        good_db.close.assert_called_once()
        assert len(mgr.databases) == 0


# ---------------------------------------------------------------------------
# DatabaseManager – cleanup inactive connections
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCleanupInactive:
    """Tests for _cleanup_inactive."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_cleanup_removes_old_connections(self, _mock_sync, mock_app):
        mock_app.server.db_timeout = 60  # 60 seconds

        mgr = DatabaseManager(mock_app)
        try:
            old_db = _make_mock_db("old")
            recent_db = _make_mock_db("recent")

            # old_db was accessed 2 minutes ago
            mgr.databases["old"] = (old_db, datetime.now(UTC) - timedelta(minutes=2))
            # recent_db was accessed just now
            mgr.databases["recent"] = (recent_db, datetime.now(UTC))

            mgr._cleanup_inactive()

            assert "old" not in mgr.databases
            assert "recent" in mgr.databases
            old_db.close.assert_called_once()
            recent_db.close.assert_not_called()
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_cleanup_no_effect_when_all_active(self, _mock_sync, mock_app):
        mock_app.server.db_timeout = 3600

        mgr = DatabaseManager(mock_app)
        try:
            db = _make_mock_db("active")
            mgr.databases["active"] = (db, datetime.now(UTC))

            mgr._cleanup_inactive()

            assert "active" in mgr.databases
            db.close.assert_not_called()
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – health checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthChecks:
    """Tests for _check_database_health and _perform_health_checks."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_healthy_db_returns_true(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            db = _make_mock_db("healthy")
            assert mgr._check_database_health(db) is True
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_closed_db_returns_false(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            db = _make_mock_db("closed", closed=True)
            assert mgr._check_database_health(db) is False
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_db_raising_on_stats_returns_false(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            db = _make_mock_db("broken")
            db.get_stats.side_effect = RuntimeError("db error")
            assert mgr._check_database_health(db) is False
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_perform_health_checks_evicts_unhealthy(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            bad_db = _make_mock_db("bad", closed=True)
            good_db = _make_mock_db("good")

            mgr.databases["bad"] = (bad_db, datetime.now(UTC))
            mgr.databases["good"] = (good_db, datetime.now(UTC))

            # Force the health check interval to have passed
            mgr._last_health_check = datetime.now(UTC) - timedelta(minutes=10)

            mgr._perform_health_checks()

            assert "bad" not in mgr.databases
            assert "good" in mgr.databases
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – error tracking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorTracking:
    """Tests for _record_error."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_record_error_tracks_counts(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            mgr._record_error("db1", RuntimeError("err1"))
            mgr._record_error("db1", RuntimeError("err2"))
            mgr._record_error("db2", ValueError("err3"))

            assert mgr._error_counts["db1"] == 2
            assert mgr._error_counts["db2"] == 1
            assert mgr._last_errors["db1"]["type"] == "RuntimeError"
            assert mgr._last_errors["db2"]["type"] == "ValueError"
        finally:
            mgr.close_all()


# ---------------------------------------------------------------------------
# DatabaseManager – get_manager_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestManagerStats:
    """Tests for get_manager_stats."""

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_stats_basic_shape(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            stats = mgr.get_manager_stats()

            assert "active_databases" in stats
            assert "total_databases" in stats
            assert "uptime_seconds" in stats
            assert "worker_id" in stats
            assert "databases" in stats
            assert isinstance(stats["uptime_seconds"], float)
        finally:
            mgr.close_all()

    @patch("localvectordb_server._dbmanager.DatabaseManager._sync_registry_from_filesystem")
    def test_stats_reflects_active_dbs(self, _mock_sync, mock_app):
        mgr = DatabaseManager(mock_app)
        try:
            db = _make_mock_db("active")
            mgr.databases["active"] = (db, datetime.now(UTC))

            stats = mgr.get_manager_stats()
            assert stats["active_databases"] == 1
        finally:
            mgr.close_all()
