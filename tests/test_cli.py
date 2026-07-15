"""
Tests for localvectordb_server CLI module.

Tests CLI argument parsing, output formatting, and error handling using
Click's CliRunner. Database operations and external services are mocked.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from localvectordb_server.cli import cli

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_db_folder(tmp_path):
    """Temporary folder that acts as the database root directory."""
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def fake_config(tmp_db_folder):
    """Return a minimal mock Config object that satisfies the CLI."""

    @dataclass
    class _Security:
        require_api_key: bool = False
        key_database_path: Optional[str] = None
        cors_enabled: bool = False
        cors_allowed_origins: str = "*"

    @dataclass
    class _Server:
        host: str = "127.0.0.1"
        port: int = 5000
        security: _Security = field(default_factory=_Security)
        cache_enabled: bool = False
        cache_type: str = "SimpleCache"
        enable_rate_limiting: bool = False
        rate_limit: str = "100 per minute"
        file_upload_enabled: bool = False
        db_registry_type: str = ""
        db_registry_settings: dict = field(default_factory=dict)
        cache_settings: dict = field(default_factory=dict)
        max_request_size: int = 10 * 1024 * 1024

    @dataclass
    class _Embedding:
        provider: str = "ollama"
        model: str = "nomic-embed-text"
        base_url: Optional[str] = None
        api_key: Optional[str] = None
        batch_size: int = 64
        timeout: int = 30
        max_retries: int = 3
        config: dict = field(default_factory=dict)

    @dataclass
    class _Database:
        root_dir: str = ""
        chunk_size: int = 500
        chunk_overlap: int = 1
        chunking_method: str = "sentences"
        default_metadata_schema: dict = field(default_factory=dict)

    @dataclass
    class _Backup:
        enabled: bool = True
        default_location: str = "./backups"

    @dataclass
    class _Migration:
        enabled: bool = True
        migration_dir: str = "./migrations"

    @dataclass
    class _Config:
        database: _Database = field(default_factory=_Database)
        embedding: _Embedding = field(default_factory=_Embedding)
        server: _Server = field(default_factory=_Server)
        backup: _Backup = field(default_factory=_Backup)
        migration: _Migration = field(default_factory=_Migration)

        def generate_toml(self):
            return (
                "[database]\n"
                f'root_dir = "{self.database.root_dir}"\n'
                f"chunk_size = {self.database.chunk_size}\n\n"
                "[embedding]\n"
                f'provider = "{self.embedding.provider}"\n'
                f'model = "{self.embedding.model}"\n\n'
                "[server]\n"
                f'host = "{self.server.host}"\n'
                f"port = {self.server.port}\n"
            )

        def validate(self):
            return True

    cfg = _Config()
    cfg.database.root_dir = tmp_db_folder
    return cfg


@pytest.fixture
def config_file(tmp_path, fake_config):
    """Write a minimal TOML config file and return its path."""
    cfg_path = tmp_path / ".lvdb-config.toml"
    cfg_path.write_text(fake_config.generate_toml(), encoding="utf-8")
    return str(cfg_path)


def _patch_cli_init(fake_config, config_file, tmp_db_folder):
    """Return a combined patch context-manager that bypasses the cli group
    callback's config resolution so that subcommands can be tested in isolation.

    The cli callback imports find_config_file and load_config locally, so we
    patch at the source modules."""
    from contextlib import ExitStack

    class _Ctx:
        def __enter__(self_):
            self_._stack = ExitStack()
            self_._stack.__enter__()
            self_._stack.enter_context(
                patch("localvectordb_server.cli._utils.find_config_file", return_value=config_file)
            )
            self_._stack.enter_context(patch("localvectordb_server.config.load_config", return_value=fake_config))
            return self_

        def __exit__(self_, *exc):
            return self_._stack.__exit__(*exc)

    return _Ctx()


# ============================================================================
# Priority 1: lvdb list
# ============================================================================


@pytest.mark.unit
class TestListDatabases:

    def test_list_empty_folder(self, runner, fake_config, config_file, tmp_db_folder):
        """Listing an empty db folder should succeed with no output."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_list_shows_databases(self, runner, fake_config, config_file, tmp_db_folder):
        """Listing should print database names derived from .sqlite files."""
        # Create dummy .sqlite files
        Path(tmp_db_folder, "alpha.sqlite").touch()
        Path(tmp_db_folder, "beta.sqlite").touch()

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_list_ignores_non_sqlite(self, runner, fake_config, config_file, tmp_db_folder):
        """Non-.sqlite files should not appear in the listing."""
        Path(tmp_db_folder, "alpha.sqlite").touch()
        Path(tmp_db_folder, "notes.txt").touch()

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "notes" not in result.output

    def test_list_details_flag(self, runner, fake_config, config_file, tmp_db_folder):
        """--details flag should print column headers."""
        Path(tmp_db_folder, "mydb.sqlite").touch()

        mock_db = MagicMock()
        mock_db.get_stats.return_value = {
            "documents": 10,
            "chunks": 50,
            "embedding_model": "nomic-embed-text",
            "chunking_method": "sentences",
        }

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb.database.LocalVectorDB", return_value=mock_db),
        ):
            result = runner.invoke(cli, ["list", "--details"])
        assert result.exit_code == 0
        assert "Name" in result.output
        assert "Documents" in result.output


# ============================================================================
# Priority 1: lvdb create
# ============================================================================


@pytest.mark.unit
class TestCreateDatabase:

    def test_create_success(self, runner, fake_config, config_file, tmp_db_folder):
        """Creating a database should print a success message."""
        mock_db = MagicMock()
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb.database.LocalVectorDB", return_value=mock_db),
        ):
            result = runner.invoke(cli, ["create", "newdb"])
        assert result.exit_code == 0
        assert "Created database" in result.output
        assert "newdb" in result.output

    def test_create_already_exists(self, runner, fake_config, config_file, tmp_db_folder):
        """If LocalVectorDB raises an error, the CLI should report it."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb.database.LocalVectorDB",
                side_effect=Exception("Database already exists"),
            ),
        ):
            result = runner.invoke(cli, ["create", "existing"])
        assert result.exit_code != 0
        assert "Error creating database" in result.stderr

    def test_create_with_options(self, runner, fake_config, config_file, tmp_db_folder):
        """Ensure CLI options are forwarded to LocalVectorDB constructor."""
        mock_db = MagicMock()
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb.database.LocalVectorDB",
                return_value=mock_db,
            ) as mock_cls,
        ):
            result = runner.invoke(
                cli,
                [
                    "create",
                    "newdb",
                    "--embedding-model",
                    "all-minilm",
                    "--chunk-size",
                    "200",
                    "--chunking-method",
                    "tokens",
                ],
            )
        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["embedding_model"] == "all-minilm"
        assert call_kwargs["chunk_size"] == 200
        assert call_kwargs["chunking_method"] == "tokens"

    def test_create_shows_settings(self, runner, fake_config, config_file, tmp_db_folder):
        """Success output should display the configuration used."""
        mock_db = MagicMock()
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb.database.LocalVectorDB", return_value=mock_db),
        ):
            result = runner.invoke(cli, ["create", "mydb"])
        assert "embedding_model" in result.output
        assert "chunk_size" in result.output


# ============================================================================
# Priority 1: lvdb delete
# ============================================================================


@pytest.mark.unit
class TestDeleteDatabase:

    def test_delete_success_with_confirm(self, runner, fake_config, config_file, tmp_db_folder):
        """Deleting with --confirm should remove files without prompting."""
        sqlite_path = Path(tmp_db_folder) / "mydb.sqlite"
        faiss_path = Path(tmp_db_folder) / "mydb.faiss"
        sqlite_path.write_text("fake", encoding="utf-8")
        faiss_path.write_text("fake", encoding="utf-8")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["delete", "mydb", "--confirm"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()
        assert not sqlite_path.exists()
        assert not faiss_path.exists()

    def test_delete_not_found(self, runner, fake_config, config_file, tmp_db_folder):
        """Deleting a non-existent database should error to stderr and exit nonzero."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["delete", "nonexistent", "--confirm"])
        assert result.exit_code != 0
        assert "not found" in result.stderr.lower()

    def test_delete_aborted_by_user(self, runner, fake_config, config_file, tmp_db_folder):
        """If the user types something other than 'confirm', deletion is aborted."""
        Path(tmp_db_folder, "mydb.sqlite").write_text("fake", encoding="utf-8")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["delete", "mydb"], input="no\n")
        assert result.exit_code == 0
        assert "aborted" in result.output.lower()
        # File should still exist
        assert Path(tmp_db_folder, "mydb.sqlite").exists()


# ============================================================================
# lvdb rename
# ============================================================================


@pytest.mark.unit
class TestRenameDatabase:

    def test_rename_moves_all_files(self, runner, fake_config, config_file, tmp_db_folder):
        """Rename should move the sqlite, faiss, and hierarchical sidecar files."""
        for suffix in (".sqlite", ".faiss", "_sections.faiss", "_documents.faiss"):
            Path(tmp_db_folder, f"old{suffix}").write_text("x", encoding="utf-8")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["rename", "old", "new"])

        assert result.exit_code == 0
        for suffix in (".sqlite", ".faiss", "_sections.faiss", "_documents.faiss"):
            assert not Path(tmp_db_folder, f"old{suffix}").exists()
            assert Path(tmp_db_folder, f"new{suffix}").exists()

    def test_rename_does_not_touch_other_databases(self, runner, fake_config, config_file, tmp_db_folder):
        """A similarly-named sibling database must not be renamed."""
        Path(tmp_db_folder, "old.sqlite").write_text("x", encoding="utf-8")
        Path(tmp_db_folder, "older.sqlite").write_text("x", encoding="utf-8")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["rename", "old", "new"])

        assert result.exit_code == 0
        assert Path(tmp_db_folder, "new.sqlite").exists()
        assert Path(tmp_db_folder, "older.sqlite").exists()  # untouched

    def test_rename_source_not_found(self, runner, fake_config, config_file, tmp_db_folder):
        """Renaming a missing database should fail with a non-zero exit code."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["rename", "missing", "new"])
        assert result.exit_code != 0

    def test_rename_target_exists(self, runner, fake_config, config_file, tmp_db_folder):
        """Renaming onto an existing database name should fail without moving files."""
        Path(tmp_db_folder, "old.sqlite").write_text("x", encoding="utf-8")
        Path(tmp_db_folder, "new.sqlite").write_text("y", encoding="utf-8")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["rename", "old", "new"])
        assert result.exit_code != 0
        assert Path(tmp_db_folder, "old.sqlite").exists()  # not moved


# ============================================================================
# lvdb version
# ============================================================================


@pytest.mark.unit
class TestVersion:

    def test_version_prints_package_version(self, runner):
        """`lvdb version` prints the installed version and exits 0 (no config needed)."""
        from importlib.metadata import version as pkg_version

        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert pkg_version("localvectordb") in result.output


# ============================================================================
# Priority 1: lvdb config show
# ============================================================================


@pytest.mark.unit
class TestConfigShow:

    def test_config_show(self, runner, fake_config, config_file, tmp_db_folder):
        """config show should output the TOML configuration."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "[database]" in result.output
        assert "[embedding]" in result.output

    def test_config_show_json(self, runner, fake_config, config_file, tmp_db_folder):
        """config show --json should output valid JSON."""
        # We need asdict to work, so we mock it
        mock_dict = {
            "database": {"root_dir": tmp_db_folder, "chunk_size": 500},
            "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
            "server": {"host": "127.0.0.1", "port": 5000},
            "backup": {},
            "migration": {},
        }
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.cli._config.asdict", return_value=mock_dict),
        ):
            result = runner.invoke(cli, ["config", "show", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output.split("\n", 2)[-1])  # skip header lines
        assert "database" in parsed

    def test_config_show_section(self, runner, fake_config, config_file, tmp_db_folder):
        """config show --section database should only show the database section."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["config", "show", "--section", "database"])
        assert result.exit_code == 0
        assert "[database]" in result.output


# ============================================================================
# Priority 1: lvdb config set
# ============================================================================


@pytest.mark.unit
class TestConfigSet:

    def test_config_set_dry_run(self, runner, fake_config, config_file, tmp_db_folder):
        """config set --dry-run should not write changes."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                return_value="127.0.0.1",
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "config",
                    "set",
                    "server.host",
                    "0.0.0.0",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_config_set_force(self, runner, fake_config, config_file, tmp_db_folder):
        """config set --force should skip the confirmation prompt."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                return_value="127.0.0.1",
            ),
            patch("localvectordb_server.cli._config.set_nested_value") as mock_set,
        ):
            result = runner.invoke(
                cli,
                [
                    "config",
                    "set",
                    "server.host",
                    "0.0.0.0",
                    "--force",
                ],
            )
        assert result.exit_code == 0
        assert "updated" in result.output.lower() or "Configuration" in result.output
        mock_set.assert_called_once()

    def test_config_set_invalid_key(self, runner, fake_config, config_file, tmp_db_folder):
        """Setting an invalid key should fail gracefully."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                side_effect=ValueError("Invalid key path"),
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "config",
                    "set",
                    "invalid.key",
                    "value",
                    "--force",
                ],
            )
        assert result.exit_code != 0


# ============================================================================
# Priority 1: lvdb config get
# ============================================================================


@pytest.mark.unit
class TestConfigGet:

    def test_config_get_value(self, runner, fake_config, config_file, tmp_db_folder):
        """config get should retrieve and display a config value."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                return_value="127.0.0.1",
            ),
        ):
            result = runner.invoke(cli, ["config", "get", "server.host"])
        assert result.exit_code == 0
        assert "127.0.0.1" in result.output

    def test_config_get_raw_format(self, runner, fake_config, config_file, tmp_db_folder):
        """config get --format raw should print just the value."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                return_value=5000,
            ),
        ):
            result = runner.invoke(cli, ["config", "get", "server.port", "--format", "raw"])
        assert result.exit_code == 0
        assert result.output.strip() == "5000"

    def test_config_get_json_format(self, runner, fake_config, config_file, tmp_db_folder):
        """config get --format json should print JSON."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                return_value=True,
            ),
        ):
            result = runner.invoke(cli, ["config", "get", "server.security.require_api_key", "--format", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output.strip()) is True

    def test_config_get_invalid_key(self, runner, fake_config, config_file, tmp_db_folder):
        """Getting a nonexistent key should fail."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.cli._config.get_nested_value",
                side_effect=ValueError("not found"),
            ),
        ):
            result = runner.invoke(cli, ["config", "get", "bad.key"])
        assert result.exit_code != 0


# ============================================================================
# Priority 1: lvdb auth create-key
# ============================================================================


@pytest.mark.unit
class TestAuthCreateKey:

    @staticmethod
    def _make_key_record():
        """Build a mock KeyRecord for testing."""
        from localvectordb_server.keymanager import PermissionLevel

        rec = MagicMock()
        rec.id = "key-abc123"
        rec.plain_key = "lvdb_test_key_abc123"
        rec.description = "Test key"
        rec.permission_level = PermissionLevel.READ_WRITE
        rec.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rec.expires_at = None
        rec.days_until_expiry = None
        rec.to_dict.return_value = {
            "id": "key-abc123",
            "description": "Test key",
            "permission_level": "read_write",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        return rec

    def test_create_key_table_output(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key should display the new key in table format."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "create-key", "--description", "Test key"])
        assert result.exit_code == 0
        assert "lvdb_test_key_abc123" in result.output
        assert "key-abc123" in result.output

    def test_create_key_json_output(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key --format json should emit valid JSON."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "create-key", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "plain_key" in parsed

    def test_create_key_key_only_output(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key --format key-only should emit only the key string."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "create-key", "--format", "key-only"])
        assert result.exit_code == 0
        assert result.output.strip() == "lvdb_test_key_abc123"

    def test_create_key_with_expiry(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key with --expires-days should pass the value through."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(
                cli,
                [
                    "auth",
                    "create-key",
                    "--expires-days",
                    "30",
                    "--format",
                    "key-only",
                ],
            )
        assert result.exit_code == 0
        mock_km.create_key.assert_called_once()
        assert mock_km.create_key.call_args[1]["expires_days"] == 30

    def test_create_key_error(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key should handle errors gracefully."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.keymanager.get_key_manager",
                side_effect=Exception("DB error"),
            ),
        ):
            result = runner.invoke(cli, ["auth", "create-key"])
        assert result.exit_code != 0
        assert "Error" in result.stderr


# ============================================================================
# Priority 1: lvdb auth list-keys
# ============================================================================


@pytest.mark.unit
class TestAuthListKeys:

    @staticmethod
    def _make_key_records():
        from localvectordb_server.keymanager import PermissionLevel

        key1 = MagicMock()
        key1.id = "key-001"
        key1.description = "Admin key"
        key1.permission_level = PermissionLevel.READ_WRITE
        key1.active = True
        key1.is_expired = False
        key1.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        key1.expires_at = None
        key1.days_until_expiry = None
        key1.last_used = None
        key1.to_dict.return_value = {"id": "key-001", "description": "Admin key"}

        key2 = MagicMock()
        key2.id = "key-002"
        key2.description = "Read-only key"
        key2.permission_level = PermissionLevel.READ_ONLY
        key2.active = True
        key2.is_expired = False
        key2.created_at = datetime(2025, 2, 1, tzinfo=timezone.utc)
        key2.expires_at = None
        key2.days_until_expiry = None
        key2.last_used = None
        key2.to_dict.return_value = {"id": "key-002", "description": "Read-only key"}

        return [key1, key2]

    def test_list_keys_table(self, runner, fake_config, config_file, tmp_db_folder):
        """list-keys should display keys in table format."""
        mock_km = MagicMock()
        mock_km.list_keys.return_value = self._make_key_records()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "list-keys"])
        assert result.exit_code == 0
        assert "key-001" in result.output
        assert "key-002" in result.output

    def test_list_keys_json(self, runner, fake_config, config_file, tmp_db_folder):
        """list-keys --format json should emit valid JSON."""
        mock_km = MagicMock()
        mock_km.list_keys.return_value = self._make_key_records()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "list-keys", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_list_keys_empty(self, runner, fake_config, config_file, tmp_db_folder):
        """list-keys with no keys should print a 'no keys' message."""
        mock_km = MagicMock()
        mock_km.list_keys.return_value = []

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "list-keys"])
        assert result.exit_code == 0
        assert "No API keys found" in result.output

    def test_list_keys_with_stats(self, runner, fake_config, config_file, tmp_db_folder):
        """list-keys --show-stats should display statistics."""
        mock_km = MagicMock()
        mock_km.list_keys.return_value = self._make_key_records()
        mock_km.get_stats.return_value = {
            "total_keys": 2,
            "active_keys": 2,
            "expired_keys": 0,
            "expiring_soon": 0,
            "recently_used": 1,
        }

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "list-keys", "--show-stats"])
        assert result.exit_code == 0
        assert "Total keys" in result.output

    def test_list_keys_error(self, runner, fake_config, config_file, tmp_db_folder):
        """list-keys should handle errors gracefully."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.keymanager.get_key_manager",
                side_effect=Exception("Connection failed"),
            ),
        ):
            result = runner.invoke(cli, ["auth", "list-keys"])
        assert result.exit_code != 0
        assert "Error" in result.stderr


# ============================================================================
# Priority 1: lvdb auth revoke-key
# ============================================================================


@pytest.mark.unit
class TestAuthRevokeKey:

    def test_revoke_key_success(self, runner, fake_config, config_file, tmp_db_folder):
        """Revoking an active key with --confirm should succeed."""
        mock_km = MagicMock()
        key_rec = MagicMock()
        key_rec.id = "key-001"
        key_rec.active = True
        key_rec.description = "Test"
        key_rec.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_km.get_key.return_value = key_rec
        mock_km.revoke_key.return_value = True

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "revoke-key", "key-001", "--confirm"])
        assert result.exit_code == 0
        assert "revoked" in result.output.lower()

    def test_revoke_key_not_found(self, runner, fake_config, config_file, tmp_db_folder):
        """Revoking a nonexistent key should fail."""
        mock_km = MagicMock()
        mock_km.get_key.return_value = None

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "revoke-key", "bad-key", "--confirm"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ============================================================================
# Priority 2: lvdb db <name> add
# ============================================================================


@pytest.mark.unit
class TestDbAdd:

    def _make_db_ctx(self, fake_config, config_file, tmp_db_folder, mock_db):
        """Patch both the CLI init and the db_group callback so that
        ctx.obj['db'] is set to our mock database."""
        api_key_path = os.path.join(tmp_db_folder, "api_keys.db")
        obj = {
            "config": fake_config,
            "config_path": config_file,
            "api_key_db_path": api_key_path,
            "db_folder": tmp_db_folder,
            "db_name": "testdb",
            "db": mock_db,
        }

        @click.pass_context
        def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
            ctx.ensure_object(dict)
            ctx.obj = obj

        @click.pass_context
        def _patched_db_group_callback(ctx, name):
            ctx.obj.update({"db_name": name, "db": mock_db})

        from localvectordb_server.cli._db import db_group

        return (
            patch.object(cli, "callback", _patched_cli_callback),
            patch.object(db_group, "callback", _patched_db_group_callback),
        )

    def test_add_text_directly(self, runner, fake_config, config_file, tmp_db_folder):
        """Adding text directly should call db.upsert."""
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_001"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", "Hello world"])
        assert result.exit_code == 0
        assert "doc_001" in result.output
        mock_db.upsert.assert_called_once()

    def test_add_from_file(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        """Adding a file should read it and pass content to db.upsert."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("Document content here", encoding="utf-8")

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_002"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", str(test_file)])
        assert result.exit_code == 0
        assert "doc_002" in result.output
        # Verify the file content was passed
        call_args = mock_db.upsert.call_args
        assert "Document content here" in call_args[1]["documents"]

    def test_add_html_file_is_extracted(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        """Adding an HTML file should extract it to Markdown before upserting."""
        html_file = tmp_path / "page.html"
        html_file.write_bytes(b"<html><body><h1>Hi</h1><p>World</p></body></html>")

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_html"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", str(html_file)])
        assert result.exit_code == 0
        call_args = mock_db.upsert.call_args
        # Content is Markdown, not raw HTML.
        assert call_args[1]["documents"] == ["# Hi\n\nWorld"]
        # Auto metadata records the extracted source format.
        assert call_args[1]["metadata"][0]["source_format"] == "html"

    def test_add_no_input(self, runner, fake_config, config_file, tmp_db_folder):
        """Calling add with no arguments should produce an error."""
        mock_db = MagicMock()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add"])
        assert result.exit_code != 0

    def test_add_with_metadata(self, runner, fake_config, config_file, tmp_db_folder):
        """Adding with --metadata should pass metadata to upsert."""
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_003"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                [
                    "db",
                    "testdb",
                    "add",
                    "some text",
                    "--metadata",
                    '{"author": "Test"}',
                ],
            )
        assert result.exit_code == 0
        call_args = mock_db.upsert.call_args
        assert call_args[1]["metadata"] == [{"author": "Test"}]

    def test_add_with_id(self, runner, fake_config, config_file, tmp_db_folder):
        """Adding with --id should pass the id to upsert."""
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["custom-id"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                [
                    "db",
                    "testdb",
                    "add",
                    "some text",
                    "--id",
                    "custom-id",
                ],
            )
        assert result.exit_code == 0
        call_args = mock_db.upsert.call_args
        assert call_args[1]["ids"] == ["custom-id"]


# ============================================================================
# lvdb db <name> patch
# ============================================================================


@pytest.mark.unit
class TestDbPatch:
    def _make_db_ctx(self, fake_config, config_file, tmp_db_folder, mock_db):
        api_key_path = os.path.join(tmp_db_folder, "api_keys.db")
        obj = {
            "config": fake_config,
            "config_path": config_file,
            "api_key_db_path": api_key_path,
            "db_folder": tmp_db_folder,
            "db_name": "testdb",
            "db": mock_db,
        }

        @click.pass_context
        def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
            ctx.ensure_object(dict)
            ctx.obj = obj

        @click.pass_context
        def _patched_db_group_callback(ctx, name):
            ctx.obj.update({"db_name": name, "db": mock_db})

        from localvectordb_server.cli._db import db_group

        return (
            patch.object(cli, "callback", _patched_cli_callback),
            patch.object(db_group, "callback", _patched_db_group_callback),
        )

    def _mock_db(self, updated=True):
        from localvectordb.patching import PatchResult

        mock_db = MagicMock()
        mock_db.patch.return_value = PatchResult(updated=updated, new_hash="abc123def456", ops_applied=1)
        return mock_db

    def test_patch_find_replace(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "patch", "doc_1", "--find", "brown", "--replace", "red"])
        assert result.exit_code == 0
        assert "Successfully patched" in result.output
        args, kwargs = mock_db.patch.call_args
        assert args[0] == "doc_1"
        assert args[1] == [{"op": "replace", "find": "brown", "replace": "red", "count": 1}]
        assert kwargs["expect_hash"] is None

    def test_patch_append(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "patch", "doc_1", "--append", "!"])
        assert result.exit_code == 0
        assert mock_db.patch.call_args[0][1] == [{"op": "append", "text": "!"}]

    def test_patch_expect_hash_passed_through(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli, ["db", "testdb", "patch", "doc_1", "--append", "!", "--expect-hash", "cafef00d"]
            )
        assert result.exit_code == 0
        assert mock_db.patch.call_args[1]["expect_hash"] == "cafef00d"

    def test_patch_noop_message(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db(updated=False)
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "patch", "doc_1", "--find", "x", "--replace", "x"])
        assert result.exit_code == 0
        assert "already up to date" in result.output

    def test_patch_find_without_replace_errors(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "patch", "doc_1", "--find", "brown"])
        assert result.exit_code != 0
        mock_db.patch.assert_not_called()

    def test_patch_no_ops_errors(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = self._mock_db()
        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "patch", "doc_1"])
        assert result.exit_code != 0
        mock_db.patch.assert_not_called()


# ============================================================================
# Priority 2: lvdb db <name> search
# ============================================================================


@pytest.mark.unit
class TestDbSearch:

    def _make_db_ctx(self, fake_config, config_file, tmp_db_folder, mock_db):
        api_key_path = os.path.join(tmp_db_folder, "api_keys.db")
        obj = {
            "config": fake_config,
            "config_path": config_file,
            "api_key_db_path": api_key_path,
            "db_folder": tmp_db_folder,
            "db_name": "testdb",
            "db": mock_db,
        }

        @click.pass_context
        def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
            ctx.ensure_object(dict)
            ctx.obj = obj

        @click.pass_context
        def _patched_db_group_callback(ctx, name):
            ctx.obj.update({"db_name": name, "db": mock_db})

        from localvectordb_server.cli._db import db_group

        return (
            patch.object(cli, "callback", _patched_cli_callback),
            patch.object(db_group, "callback", _patched_db_group_callback),
        )

    def _make_search_result(self, doc_id="doc_1", content="Result text", score=0.95):
        r = MagicMock()
        r.id = doc_id
        r.content = content
        r.score = score
        r.type = "document"
        r.metadata = {"author": "Test"}
        return r

    def test_search_basic(self, runner, fake_config, config_file, tmp_db_folder):
        """Basic search should show results."""
        mock_db = MagicMock()
        mock_db.query.return_value = [self._make_search_result()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "test query"])
        assert result.exit_code == 0
        assert "Result text" in result.output

    def test_search_no_results(self, runner, fake_config, config_file, tmp_db_folder):
        """Search with no results should show a message."""
        mock_db = MagicMock()
        mock_db.query.return_value = []

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "no match"])
        assert result.exit_code == 0
        assert "No results" in result.stderr

    def test_search_json_output(self, runner, fake_config, config_file, tmp_db_folder):
        """Search with --json should output valid JSON."""
        mock_db = MagicMock()
        mock_db.query.return_value = [self._make_search_result()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query", "--format", "json"])
        assert result.exit_code == 0
        # The status line goes to stderr; extract the JSON portion from output
        json_start = result.output.index("[")
        parsed = json.loads(result.output[json_start:])
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == "doc_1"

    def test_search_with_options(self, runner, fake_config, config_file, tmp_db_folder):
        """Search options should be forwarded to db.query."""
        mock_db = MagicMock()
        mock_db.query.return_value = [self._make_search_result()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                [
                    "db",
                    "testdb",
                    "search",
                    "query",
                    "--limit",
                    "3",
                    "--search-type",
                    "hybrid",
                ],
            )
        assert result.exit_code == 0
        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["k"] == 3
        assert call_kwargs["search_type"] == "hybrid"

    def test_search_level_forwarded(self, runner, fake_config, config_file, tmp_db_folder):
        """--search-level and --return-type sections should be forwarded to db.query."""
        mock_db = MagicMock()
        section_result = self._make_search_result()
        section_result.type = "section"
        mock_db.query.return_value = [section_result]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                [
                    "db",
                    "testdb",
                    "search",
                    "query",
                    "--search-level",
                    "sections",
                    "--return-type",
                    "sections",
                ],
            )
        assert result.exit_code == 0
        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["search_level"] == "sections"
        assert call_kwargs["return_type"] == "sections"

    def test_search_level_defaults_to_chunks(self, runner, fake_config, config_file, tmp_db_folder):
        """search_level should default to 'chunks' when the flag is omitted."""
        mock_db = MagicMock()
        mock_db.query.return_value = [self._make_search_result()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query"])
        assert result.exit_code == 0
        assert mock_db.query.call_args[1]["search_level"] == "chunks"

    def test_search_error(self, runner, fake_config, config_file, tmp_db_folder):
        """Search errors should be reported cleanly."""
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("Embedding failure")

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query"])
        assert result.exit_code != 0
        assert "Search error" in result.stderr


@pytest.mark.unit
class TestDbRelated:
    """Tests for ``lvdb db <name> related`` (nearest-neighbour retrieval)."""

    def _make_db_ctx(self, fake_config, config_file, tmp_db_folder, mock_db):
        obj = {
            "config": fake_config,
            "config_path": config_file,
            "api_key_db_path": os.path.join(tmp_db_folder, "api_keys.db"),
            "db_folder": tmp_db_folder,
            "db_name": "testdb",
            "db": mock_db,
        }

        @click.pass_context
        def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
            ctx.ensure_object(dict)
            ctx.obj = obj

        @click.pass_context
        def _patched_db_group_callback(ctx, name):
            ctx.obj.update({"db_name": name, "db": mock_db})

        from localvectordb_server.cli._db import db_group

        return (
            patch.object(cli, "callback", _patched_cli_callback),
            patch.object(db_group, "callback", _patched_db_group_callback),
        )

    def _make_neighbor(self, doc_id="doc_2", content="Neighbour text", score=0.88):
        r = MagicMock()
        r.id = doc_id
        r.content = content
        r.score = score
        r.type = "document"
        r.metadata = {"author": "Test"}
        return r

    def test_related_basic(self, runner, fake_config, config_file, tmp_db_folder):
        """related should list the neighbouring documents."""
        mock_db = MagicMock()
        mock_db.nearest_neighbors.return_value = [self._make_neighbor()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "related", "doc_1"])
        assert result.exit_code == 0
        assert "Neighbour text" in result.output
        # The reference id is forwarded positionally.
        assert mock_db.nearest_neighbors.call_args[0][0] == "doc_1"

    def test_related_json_output(self, runner, fake_config, config_file, tmp_db_folder):
        """related --json should emit a JSON list."""
        mock_db = MagicMock()
        mock_db.nearest_neighbors.return_value = [self._make_neighbor()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "related", "doc_1", "--format", "json"])
        assert result.exit_code == 0
        json_start = result.output.index("[")
        parsed = json.loads(result.output[json_start:])
        assert parsed[0]["id"] == "doc_2"

    def test_related_forwards_options(self, runner, fake_config, config_file, tmp_db_folder):
        """--limit/--score-threshold/--metadata-filter should reach nearest_neighbors."""
        mock_db = MagicMock()
        mock_db.nearest_neighbors.return_value = [self._make_neighbor()]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                [
                    "db",
                    "testdb",
                    "related",
                    "doc_1",
                    "--limit",
                    "3",
                    "--score-threshold",
                    "0.2",
                    "--metadata-filter",
                    '{"author": "Smith"}',
                ],
            )
        assert result.exit_code == 0
        call_kwargs = mock_db.nearest_neighbors.call_args[1]
        assert call_kwargs["k"] == 3
        assert call_kwargs["score_threshold"] == 0.2
        assert call_kwargs["filters"] == {"author": "Smith"}

    def test_related_no_results(self, runner, fake_config, config_file, tmp_db_folder):
        """No neighbours should print a message rather than error."""
        mock_db = MagicMock()
        mock_db.nearest_neighbors.return_value = []

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "related", "doc_1"])
        assert result.exit_code == 0
        assert "No related documents" in result.stderr

    def test_related_document_not_found(self, runner, fake_config, config_file, tmp_db_folder):
        """A missing reference document should exit with an error."""
        from localvectordb.exceptions import DocumentNotFoundError

        mock_db = MagicMock()
        mock_db.name = "testdb"
        mock_db.nearest_neighbors.side_effect = DocumentNotFoundError("missing")

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "related", "ghost"])
        assert result.exit_code != 0
        assert "was not found" in result.output


# ============================================================================
# Priority 2: lvdb serve
# ============================================================================


@pytest.mark.unit
class TestServe:

    def test_serve_creates_app(self, runner, fake_config, config_file, tmp_db_folder):
        """serve should build the app and launch it via uvicorn."""
        mock_app = MagicMock()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.app.create_app",
                return_value=mock_app,
            ) as mock_create,
            patch("uvicorn.run") as mock_run,
        ):
            result = runner.invoke(cli, ["serve", "--disable-ollama-check"])
        assert result.exit_code == 0
        mock_create.assert_called_once()
        mock_run.assert_called_once()

    def test_serve_config_error(self, runner, fake_config, config_file, tmp_db_folder):
        """serve should report configuration errors."""
        from localvectordb.exceptions import ConfigurationError

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.app.create_app",
                side_effect=ConfigurationError("bad config"),
            ),
        ):
            result = runner.invoke(cli, ["serve", "--disable-ollama-check"])
        assert result.exit_code != 0
        assert "Configuration error" in result.stderr


# ============================================================================
# Priority 2: lvdb config init
# ============================================================================


@pytest.mark.unit
class TestConfigInit:

    def test_config_init_creates_file(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        """config init should create a new config file."""
        output_path = str(tmp_path / "new-config.toml")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["config", "init", "--output", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

    def test_config_init_json_format(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        """config init --format json should create a JSON config."""
        output_path = str(tmp_path / "new-config.json")

        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(
                cli,
                [
                    "config",
                    "init",
                    "--format",
                    "json",
                    "--output",
                    output_path,
                ],
            )
        assert result.exit_code == 0
        assert os.path.exists(output_path)
        # Verify it's valid JSON
        with open(output_path) as f:
            data = json.load(f)
        assert "database" in data


# ============================================================================
# Priority 2: lvdb auth status
# ============================================================================


@pytest.mark.unit
class TestAuthStatus:

    def test_auth_status_table(self, runner, fake_config, config_file, tmp_db_folder):
        """auth status should display auth configuration."""
        mock_km = MagicMock()
        mock_km.get_stats.return_value = {
            "total_keys": 5,
            "active_keys": 3,
            "expired_keys": 2,
            "expiring_soon": 0,
            "recently_used": 1,
        }

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "Authentication Status" in result.output

    def test_auth_status_json(self, runner, fake_config, config_file, tmp_db_folder):
        """auth status --format json should emit valid JSON."""
        mock_km = MagicMock()
        mock_km.get_stats.return_value = {
            "total_keys": 1,
            "active_keys": 1,
            "expired_keys": 0,
            "expiring_soon": 0,
            "recently_used": 0,
        }

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "status", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "auth_enabled" in parsed


# ============================================================================
# Edge cases / miscellaneous
# ============================================================================


@pytest.mark.unit
class TestCLIHelpMessages:

    def test_top_level_help(self, runner):
        """The top-level --help should list all subcommands."""
        with patch(
            "localvectordb_server.cli._utils.find_config_file",
            return_value=None,
        ):
            result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.output
        assert "create" in result.output
        assert "list" in result.output
        assert "config" in result.output
        assert "auth" in result.output

    def test_create_help(self, runner, fake_config, config_file, tmp_db_folder):
        """lvdb create --help should show create options."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["create", "--help"])
        assert result.exit_code == 0
        assert "--embedding-model" in result.output
        assert "--chunk-size" in result.output

    def test_config_help(self, runner):
        """lvdb config --help should show config subcommands."""
        with patch(
            "localvectordb_server.cli._utils.find_config_file",
            return_value=None,
        ):
            result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "set" in result.output
        assert "init" in result.output

    def test_auth_help(self, runner, fake_config, config_file, tmp_db_folder):
        """lvdb auth --help should show auth subcommands."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["auth", "--help"])
        assert result.exit_code == 0
        assert "create-key" in result.output
        assert "list-keys" in result.output
        assert "revoke-key" in result.output


# ============================================================================
# Consistency fixes: subcommand --help without a DB, config errors, add ids
# ============================================================================


@pytest.mark.unit
class TestDbSubcommandHelpWithoutDb:
    """`lvdb db <name> <cmd> --help` must print help without requiring the
    database (or even the DB folder) to exist."""

    def test_search_help_with_missing_db(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["db", "no_such_db", "search", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output
        assert "--search-type" in result.output

    def test_nested_schema_update_help_with_missing_db(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["db", "no_such_db", "schema", "update", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_short_help_flag_with_missing_db(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["db", "no_such_db", "add", "-h"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_missing_db_still_errors_without_help(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["db", "no_such_db", "info"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


@pytest.mark.unit
class TestConfigLoadErrors:
    """A broken config file should produce a friendly error, not a traceback."""

    def test_invalid_toml_reports_friendly_error(self, runner, tmp_path):
        bad = tmp_path / ".lvdb-config.toml"
        bad.write_text("[database\nroot_dir = broken", encoding="utf-8")
        result = runner.invoke(cli, ["--config", str(bad), "list"])
        assert result.exit_code == 5  # EXIT_CODE_CONFIGURATION_ERROR (5, not Click's usage-error 2)
        assert "Error loading configuration" in result.output
        assert "Traceback" not in result.output

    def test_invalid_config_values_report_friendly_error(self, runner, tmp_path):
        bad = tmp_path / ".lvdb-config.toml"
        bad.write_text('[server]\nport = -5\n[database]\nroot_dir = "dbs"\n', encoding="utf-8")
        result = runner.invoke(cli, ["--config", str(bad), "list"])
        assert result.exit_code == 5  # EXIT_CODE_CONFIGURATION_ERROR
        assert "Error loading configuration" in result.output
        assert "Traceback" not in result.output


@pytest.mark.unit
class TestDbAddDefaultIds:
    """CLI `add <file>` derives ids from filename stems, matching the library's
    upsert_from_file; text/stdin inputs keep auto-generated ids."""

    def _make_db_ctx(self, fake_config, config_file, tmp_db_folder, mock_db):
        api_key_path = os.path.join(tmp_db_folder, "api_keys.db")
        obj = {
            "config": fake_config,
            "config_path": config_file,
            "api_key_db_path": api_key_path,
            "db_folder": tmp_db_folder,
            "db_name": "testdb",
            "db": mock_db,
        }

        @click.pass_context
        def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
            ctx.ensure_object(dict)
            ctx.obj = obj

        @click.pass_context
        def _patched_db_group_callback(ctx, name):
            ctx.obj.update({"db_name": name, "db": mock_db})

        from localvectordb_server.cli._db import db_group

        return (
            patch.object(cli, "callback", _patched_cli_callback),
            patch.object(db_group, "callback", _patched_db_group_callback),
        )

    def test_add_file_uses_filename_stem_as_id(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        test_file = tmp_path / "my_report.txt"
        test_file.write_text("Report content", encoding="utf-8")

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["my_report"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", str(test_file)])
        assert result.exit_code == 0
        assert mock_db.upsert.call_args[1]["ids"] == ["my_report"]

    def test_add_text_keeps_generated_id(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_1"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", "just some text"])
        assert result.exit_code == 0
        assert mock_db.upsert.call_args[1]["ids"] == [None]

    def test_add_explicit_id_wins_over_stem(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        test_file = tmp_path / "my_report.txt"
        test_file.write_text("Report content", encoding="utf-8")

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["custom"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", str(test_file), "--id", "custom"])
        assert result.exit_code == 0
        assert mock_db.upsert.call_args[1]["ids"] == ["custom"]

    def test_add_duplicate_stems_fall_back_to_generated_ids(
        self, runner, fake_config, config_file, tmp_db_folder, tmp_path
    ):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "readme.txt").write_text("A", encoding="utf-8")
        (tmp_path / "b" / "readme.txt").write_text("B", encoding="utf-8")

        mock_db = MagicMock()
        mock_db.upsert.return_value = ["readme", "doc_2"]

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(
                cli,
                ["db", "testdb", "add", str(tmp_path / "a" / "readme.txt"), str(tmp_path / "b" / "readme.txt")],
            )
        assert result.exit_code == 0
        ids = mock_db.upsert.call_args[1]["ids"]
        assert ids[0] == "readme"
        assert ids[1] is None  # duplicate stem falls back to a generated id
        assert "Duplicate document id" in result.output


# ============================================================================
# Pre-release fixes: nonzero exit codes, --help without config, flag renames
# ============================================================================


def _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db, db_name="testdb"):
    """Patch the cli + db group callbacks so ctx.obj['db'] is a mock database."""
    obj = {
        "config": fake_config,
        "config_path": config_file,
        "api_key_db_path": os.path.join(tmp_db_folder, "api_keys.db"),
        "db_folder": tmp_db_folder,
        "db_name": db_name,
        "db": mock_db,
    }

    @click.pass_context
    def _patched_cli_callback(ctx, config, db_folder, verbose=False, quiet=False):
        ctx.ensure_object(dict)
        ctx.obj = obj

    @click.pass_context
    def _patched_db_group_callback(ctx, name):
        ctx.obj.update({"db_name": name, "db": mock_db})

    from localvectordb_server.cli._db import db_group

    return (
        patch.object(cli, "callback", _patched_cli_callback),
        patch.object(db_group, "callback", _patched_db_group_callback),
    )


@pytest.mark.unit
class TestHelpWithoutConfig:
    """`--help` for every command must render (exit 0) with no config file."""

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["list", "--help"],
            ["create", "--help"],
            ["delete", "--help"],
            ["db", "mydb", "--help"],
            ["db", "mydb", "search", "--help"],
            ["auth", "--help"],
            ["auth", "create-key", "--help"],
        ],
    )
    def test_help_renders_without_config(self, runner, args):
        with patch("localvectordb_server.cli._utils.find_config_file", return_value=None):
            result = runner.invoke(cli, args)
        assert result.exit_code == 0
        assert "Usage" in result.output


@pytest.mark.unit
class TestRequireConfigErrors:
    """Real (non-help) invocations of config-needing commands exit 1 cleanly."""

    @pytest.mark.parametrize("args", [["list"], ["create", "x"], ["delete", "x"], ["rename", "a", "b"]])
    def test_missing_config_exits_one(self, runner, args):
        with patch("localvectordb_server.cli._utils.find_config_file", return_value=None):
            result = runner.invoke(cli, args)
        assert result.exit_code != 0
        assert "No configuration file found" in result.stderr

    def test_delete_missing_database_exits_one(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["delete", "ghost", "--confirm"])
        assert result.exit_code != 0
        assert "not found" in result.stderr.lower()


@pytest.mark.unit
class TestTuningExitCodes:
    def test_tuning_get_unknown_db_exits_one(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["tuning", "get", "nosuchdb"])
        assert result.exit_code != 0
        assert "not found" in result.stderr.lower()

    def test_maintenance_optimize_unknown_db_exits_one(self, runner, fake_config, config_file, tmp_db_folder):
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["maintenance", "optimize", "nosuchdb"])
        assert result.exit_code != 0
        assert "not found" in result.stderr.lower()


def _write_backup_tar(path, backup_id="bkp-123", database_name="mydb", backup_type="full"):
    """Write a minimal *.lvdb-backup tar containing a manifest.json."""
    import io
    import tarfile

    manifest = {
        "backup_id": backup_id,
        "database_name": database_name,
        "backup_type": backup_type,
        "created_at": "2025-01-01T00:00:00+00:00",
    }
    data = json.dumps(manifest).encode("utf-8")
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


@pytest.mark.unit
class TestBackupExitCodes:
    def test_verify_missing_backup_exits_one(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        loc = tmp_path / "backups"
        loc.mkdir()
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["backup", "verify", "nope", "--location", str(loc)])
        assert result.exit_code != 0

    def test_verify_invalid_backup_exits_one(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        loc = tmp_path / "backups"
        loc.mkdir()
        _write_backup_tar(loc / "bkp-123.lvdb-backup", backup_id="bkp-123")

        mock_bm = MagicMock()
        mock_bm.verify_backup.return_value = False
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.cli._backup.BackupManager", return_value=mock_bm),
        ):
            result = runner.invoke(cli, ["backup", "verify", "bkp-123", "--location", str(loc)])
        assert result.exit_code != 0
        assert "verification failed" in result.stderr.lower()

    def test_pitr_no_backups_exits_one(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        loc = tmp_path / "backups"
        loc.mkdir()
        dest = tmp_path / "restored"
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(
                cli, ["backup", "pitr", "2024-01-15 14:30:00", "--to-location", str(dest), "--location", str(loc)]
            )
        assert result.exit_code != 0


@pytest.mark.unit
class TestMigrateExitCodes:
    def test_apply_failure_exits_one(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        Path(tmp_db_folder, "mydb.sqlite").write_text("x", encoding="utf-8")
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()

        mock_engine = MagicMock()
        mock_engine.migrate.return_value = {"success": False, "error": "boom"}
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.cli._migration.MigrationEngine", return_value=mock_engine),
        ):
            result = runner.invoke(cli, ["migrate", "apply", "mydb", "--no-backup", "--migrations-dir", str(mig_dir)])
        assert result.exit_code != 0
        assert "Migration failed" in result.stderr

    def test_rollback_failure_exits_one(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        Path(tmp_db_folder, "mydb.sqlite").write_text("x", encoding="utf-8")
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()

        mock_engine = MagicMock()
        mock_engine.rollback.return_value = {"success": False, "error": "boom"}
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.cli._migration.MigrationEngine", return_value=mock_engine),
        ):
            result = runner.invoke(
                cli, ["migrate", "rollback", "mydb", "1.0.0", "--no-backup", "--migrations-dir", str(mig_dir)]
            )
        assert result.exit_code != 0
        assert "Rollback failed" in result.stderr


@pytest.mark.unit
class TestDbAddPathContract:
    def test_missing_pathlike_arg_errors(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", "nosuchfile.txt"])
        assert result.exit_code != 0
        assert "looks like a file path" in result.stderr
        mock_db.upsert.assert_not_called()

    def test_text_flag_forces_literal(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_1"]
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", "nosuchfile.txt", "--text"])
        assert result.exit_code == 0
        assert mock_db.upsert.call_args[1]["documents"] == ["nosuchfile.txt"]

    def test_plain_sentence_still_text(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.upsert.return_value = ["doc_1"]
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "add", "just a plain sentence"])
        assert result.exit_code == 0
        assert mock_db.upsert.call_args[1]["documents"] == ["just a plain sentence"]


@pytest.mark.unit
class TestDbSearchJsonEmpty:
    def test_search_empty_json_prints_empty_array(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.query.return_value = []
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "no match", "--format", "json"])
        assert result.exit_code == 0
        # Status lines go to stderr; the stdout JSON payload is an empty array.
        assert result.output[result.output.index("[") :].strip() == "[]"

    def test_related_empty_json_prints_empty_array(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.nearest_neighbors.return_value = []
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "related", "doc_1", "--format", "json"])
        assert result.exit_code == 0
        assert result.output[result.output.index("[") :].strip() == "[]"


@pytest.mark.unit
class TestFormatFlagStandardization:
    """The `-j` short is a convenience alias for `--format json`; `--json` is gone."""

    def test_search_dash_j_is_json_alias(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        r = MagicMock()
        r.id, r.content, r.score, r.type, r.metadata = "doc_1", "Result text", 0.9, "document", {}
        mock_db.query.return_value = [r]
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query", "-j"])
        assert result.exit_code == 0
        parsed = json.loads(result.output[result.output.index("[") :])
        assert parsed[0]["id"] == "doc_1"

    def test_old_json_flag_is_rejected(self, runner, fake_config, config_file, tmp_db_folder):
        mock_db = MagicMock()
        mock_db.query.return_value = []
        p1, p2 = _db_ctx_patches(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query", "--json"])
        # --json is no longer a valid option (usage error, Click exit 2).
        assert result.exit_code == 2

    def test_config_set_short_force_is_y(self, runner, fake_config, config_file, tmp_db_folder):
        """`-y` (not `-f`) is the short for `config set --force`; `-f` is reserved for --format."""
        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.cli._config.get_nested_value", return_value="127.0.0.1"),
            patch("localvectordb_server.cli._config.set_nested_value"),
        ):
            result = runner.invoke(cli, ["config", "set", "server.host", "0.0.0.0", "-y"])
        assert result.exit_code == 0


@pytest.mark.unit
class TestConfigInitCorsAndForce:
    def test_cors_origins_persist(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        out = tmp_path / "cfg.toml"
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(
                cli, ["config", "init", "--cors-origins", "http://localhost:3000", "--output", str(out)]
            )
        assert result.exit_code == 0
        text = out.read_text(encoding="utf-8")
        assert "http://localhost:3000" in text
        # The default "*" must not be what got written for the origins list.
        assert 'cors_allowed_origins = "*"' not in text

    def test_force_overwrites_existing_noninteractive(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        out = tmp_path / "cfg.toml"
        out.write_text("old", encoding="utf-8")
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            # Closed stdin (input="") would hang the old getchar()-based prompt.
            result = runner.invoke(cli, ["config", "init", "--output", str(out), "--force"], input="")
        assert result.exit_code == 0
        assert out.read_text(encoding="utf-8") != "old"

    def test_existing_file_without_force_aborts(self, runner, fake_config, config_file, tmp_db_folder, tmp_path):
        out = tmp_path / "cfg.toml"
        out.write_text("old", encoding="utf-8")
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            runner.invoke(cli, ["config", "init", "--output", str(out)], input="")
        # Does not hang; file left untouched.
        assert out.read_text(encoding="utf-8") == "old"
