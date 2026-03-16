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
        """Deleting a non-existent database should print a 'not found' message."""
        with _patch_cli_init(fake_config, config_file, tmp_db_folder):
            result = runner.invoke(cli, ["delete", "nonexistent", "--confirm"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

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
            result = runner.invoke(cli, ["config", "show", "--json"])
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
        """create-key --output json should emit valid JSON."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "create-key", "--output", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "plain_key" in parsed

    def test_create_key_key_only_output(self, runner, fake_config, config_file, tmp_db_folder):
        """create-key --output key-only should emit only the key string."""
        mock_km = MagicMock()
        mock_km.create_key.return_value = self._make_key_record()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "create-key", "--output", "key-only"])
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
                    "--output",
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
        assert "Error" in result.output


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
        """list-keys --output json should emit valid JSON."""
        mock_km = MagicMock()
        mock_km.list_keys.return_value = self._make_key_records()

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch("localvectordb_server.keymanager.get_key_manager", return_value=mock_km),
        ):
            result = runner.invoke(cli, ["auth", "list-keys", "--output", "json"])
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
        assert "Error" in result.output


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
        def _patched_cli_callback(ctx, config, db_folder):
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
        def _patched_cli_callback(ctx, config, db_folder):
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
            result = runner.invoke(cli, ["db", "testdb", "search", "query", "--json"])
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

    def test_search_error(self, runner, fake_config, config_file, tmp_db_folder):
        """Search errors should be reported cleanly."""
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("Embedding failure")

        p1, p2 = self._make_db_ctx(fake_config, config_file, tmp_db_folder, mock_db)
        with p1, p2:
            result = runner.invoke(cli, ["db", "testdb", "search", "query"])
        assert result.exit_code != 0
        assert "Search error" in result.stderr


# ============================================================================
# Priority 2: lvdb serve
# ============================================================================


@pytest.mark.unit
class TestServe:

    def test_serve_creates_app(self, runner, fake_config, config_file, tmp_db_folder):
        """serve should call create_app and app.run."""
        mock_app = MagicMock()
        mock_app.config_obj = fake_config

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.create_app",
                return_value=mock_app,
            ),
        ):
            result = runner.invoke(cli, ["serve", "--disable-ollama-check"])
        assert result.exit_code == 0
        mock_app.run.assert_called_once()

    def test_serve_config_error(self, runner, fake_config, config_file, tmp_db_folder):
        """serve should report configuration errors."""
        from localvectordb.exceptions import ConfigurationError

        with (
            _patch_cli_init(fake_config, config_file, tmp_db_folder),
            patch(
                "localvectordb_server.create_app",
                side_effect=ConfigurationError("bad config"),
            ),
        ):
            result = runner.invoke(cli, ["serve", "--disable-ollama-check"])
        assert result.exit_code != 0
        assert "Configuration error" in result.output


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
        """auth status --output json should emit valid JSON."""
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
            result = runner.invoke(cli, ["auth", "status", "--output", "json"])
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
