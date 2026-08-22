"""End-to-end backward-compatibility tests for on-disk database upgrades.

Unlike ``test_migration.py`` (which exercises the metadata-schema
``MigrationEngine`` in isolation), these tests open a database that has been
downgraded on disk to a genuinely *older* layout and assert that simply
reopening it auto-upgrades the file, and that the stored data stays queryable.

Two version registers are involved, and they are independent:

* ``config.schema_version`` -- localvectordb's own TABLE LAYOUT, an integer
  advanced by the numbered ``SCHEMA_MIGRATIONS`` in ``_schema.py``. This is what
  the upgrade-on-open path maintains.
* ``PRAGMA user_version`` / ``config.db_version`` -- the user's
  metadata-migration lineage (semver, ``1.0.0`` baseline), owned by the
  ``MigrationEngine``. Opening a database must start it on a new file and must
  never touch it on an existing one.

This guards the property every release relies on -- on-disk changes have been
purely additive, so databases written by older versions keep opening cleanly.
"""

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from localvectordb._pools import ReadWriteLock
from localvectordb._schema import SCHEMA_MIGRATIONS, SCHEMA_VERSION, DatabaseSchema
from localvectordb.database import LocalVectorDB
from localvectordb.migration import MigrationEngine


def _make_db(tmpdir: str, name: str = "legacydb") -> LocalVectorDB:
    """Construct a MockEmbeddings-backed on-disk database."""
    return LocalVectorDB(
        name=name,
        base_path=tmpdir,
        embedding_provider="mock",
        embedding_model="mock",
        enable_fts=True,
    )


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _config(conn: sqlite3.Connection, key: str):
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _downgrade_to_legacy(sqlite_path: Path) -> None:
    """Rewrite an on-disk DB to look like a pre-hierarchical, pre-versioning DB.

    Strips the columns/tables that later versions add automatically on open so
    that reopening must re-create them, and removes both version registers.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        # section_id is indexed; SQLite refuses DROP COLUMN while an index
        # references it, so drop the index first (an old DB never had either).
        conn.execute("DROP INDEX IF EXISTS idx_chunks_section_id")
        conn.execute("ALTER TABLE chunks DROP COLUMN section_id")
        conn.execute("DROP INDEX IF EXISTS idx_chunk_sections_section_id")
        conn.execute("DROP TABLE IF EXISTS chunk_sections")
        conn.execute("ALTER TABLE documents DROP COLUMN doc_faiss_id")
        conn.execute("ALTER TABLE metadata_schema DROP COLUMN embedding_enabled")
        conn.execute("ALTER TABLE metadata_schema DROP COLUMN fts_enabled")
        # Drop all version tracking to mimic a database that predates it.
        conn.execute("DROP TABLE IF EXISTS migration_log")
        conn.execute("PRAGMA user_version = 0")
        conn.execute(
            "DELETE FROM config WHERE key IN ('db_version', 'version_updated_at', "
            "'schema_version', 'schema_version_updated_at', 'created_by_version')"
        )
        conn.commit()
    finally:
        conn.close()


def test_schema_migrations_are_append_only_and_numbered():
    """The registry is the contract: contiguous from 1, and SCHEMA_VERSION is its tail."""
    versions = [m.version for m in SCHEMA_MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert SCHEMA_VERSION == versions[-1]
    for m in SCHEMA_MIGRATIONS:
        assert m.description
        assert m.ops, f"migration {m.version} has no ops"


@pytest.mark.integration
@pytest.mark.database
def test_legacy_ondisk_db_auto_upgrades_on_open():
    """An old-layout on-disk DB is upgraded and stays queryable when reopened."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Build a current-format DB with some data.
        db = _make_db(tmpdir)
        db.upsert(
            [
                "The quick brown fox jumps over the lazy dog.",
                "Vector databases store high dimensional embeddings.",
            ],
            ids=["d1", "d2"],
        )
        db.save()
        db.close()

        sqlite_path = Path(tmpdir) / "legacydb.sqlite"

        # 2. Downgrade the file on disk to a legacy shape.
        _downgrade_to_legacy(sqlite_path)
        with closing(sqlite3.connect(sqlite_path)) as pre:
            assert "section_id" not in _columns(pre, "chunks")
            assert "doc_faiss_id" not in _columns(pre, "documents")
            assert _config(pre, "schema_version") is None
            assert pre.execute("PRAGMA user_version").fetchone()[0] == 0

        # 3. Reopen with current code -> synchronous constructor auto-migrates.
        db2 = _make_db(tmpdir)
        try:
            with closing(sqlite3.connect(sqlite_path)) as post:
                # Numbered migrations re-created every missing column/table.
                assert "section_id" in _columns(post, "chunks")
                assert "doc_faiss_id" in _columns(post, "documents")
                assert (
                    post.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_sections'"
                    ).fetchone()
                    is not None
                )
                assert "embedding_enabled" in _columns(post, "metadata_schema")
                assert "fts_enabled" in _columns(post, "metadata_schema")
                # ... and stamped the table-layout version the code expects.
                assert int(_config(post, "schema_version")) == SCHEMA_VERSION
                assert _config(post, "created_by_version")
                # The metadata-migration lineage was (re)started at its baseline.
                assert post.execute("PRAGMA user_version").fetchone()[0] > 0
                assert _config(post, "db_version") == "1.0.0"
                migration_log = post.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_log'"
                ).fetchone()
                assert migration_log is not None

            # 4. Stored data survived the upgrade and is still searchable.
            results = db2.query("brown fox", k=2, search_type="hybrid")
            assert len(results) > 0
            # 5. Both registers are visible to an operator.
            stats = db2.get_stats()
            assert stats["schema_version"] == SCHEMA_VERSION
            assert stats["created_by_version"]
            report = db2.diagnose(sample=10)
            assert report.schema_version == SCHEMA_VERSION
            assert f"version {SCHEMA_VERSION} of {SCHEMA_VERSION}" in report.summary
        finally:
            db2.close()


@pytest.mark.integration
@pytest.mark.database
def test_reopen_is_a_noop_and_created_by_is_never_overwritten():
    """A current database reopens without re-stamping; created_by_version is write-once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        db.upsert(["hello world"], ids=["d1"])
        db.save()
        db.close()
        sqlite_path = Path(tmpdir) / "legacydb.sqlite"
        with closing(sqlite3.connect(sqlite_path)) as conn:
            first_stamp = _config(conn, "schema_version_updated_at")
            conn.execute("UPDATE config SET value = '0.0.0-test' WHERE key = 'created_by_version'")
            conn.commit()

        db2 = _make_db(tmpdir)
        db2.close()
        with closing(sqlite3.connect(sqlite_path)) as conn:
            assert _config(conn, "schema_version_updated_at") == first_stamp
            assert _config(conn, "created_by_version") == "0.0.0-test"


@pytest.mark.integration
@pytest.mark.database
def test_newer_schema_version_is_not_downgraded():
    """A file written by a newer release keeps its stamp; we warn rather than rewrite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        db.close()
        sqlite_path = Path(tmpdir) / "legacydb.sqlite"
        with closing(sqlite3.connect(sqlite_path)) as conn:
            conn.execute("UPDATE config SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 7),))
            conn.commit()

        db2 = _make_db(tmpdir)
        try:
            assert db2.get_stats()["schema_version"] == SCHEMA_VERSION + 7
            report = db2.diagnose(sample=10)
            assert f"version {SCHEMA_VERSION + 7} of {SCHEMA_VERSION}" in report.summary
        finally:
            db2.close()


@pytest.mark.integration
@pytest.mark.database
def test_metadata_migration_lineage_and_schema_version_are_independent(tmp_path):
    """A user metadata migration moves db_version/user_version and nothing else."""
    db = _make_db(str(tmp_path))
    db.close()
    sqlite_path = tmp_path / "legacydb.sqlite"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    engine = MigrationEngine(sqlite_path, migrations_dir, auto_backup=False)
    template = engine.create_migration_template(version="1.1.0", description="add category")
    template.write_text(
        """
from typing import Any, Dict
from localvectordb.migration import Migration
from localvectordb.core import MetadataField, MetadataFieldType

class AddCategory(Migration):
    version = "1.1.0"
    description = "add category"
    dependencies = []

    def get_schema_changes(self) -> Dict[str, Any]:
        return {"new_schema": {"category": MetadataField(type=MetadataFieldType.TEXT, indexed=True)}}

    def get_rollback_changes(self) -> Dict[str, Any]:
        return {"new_schema": {}, "drop_columns": True}
""",
        encoding="utf-8",
    )
    result = engine.migrate()
    assert result["success"] is True

    with closing(sqlite3.connect(sqlite_path)) as conn:
        assert _config(conn, "db_version") == "1.1.0"
        # The table-layout register did not move.
        assert int(_config(conn, "schema_version")) == SCHEMA_VERSION


@pytest.mark.integration
@pytest.mark.database
async def test_async_initialize_stamps_both_registers():
    """initialize_async stamps the layout version and the lineage baseline like the sync path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "async_init.sqlite"
        schema = DatabaseSchema(db_path, ReadWriteLock())

        # Initialize a fresh database purely through the async path.
        await schema.initialize_async()

        with closing(sqlite3.connect(db_path)) as conn:
            assert int(_config(conn, "schema_version")) == SCHEMA_VERSION
            assert _config(conn, "created_by_version")
            # Metadata-migration lineage baseline, same as the sync path.
            assert conn.execute("PRAGMA user_version").fetchone()[0] > 0
            assert _config(conn, "db_version") == "1.0.0"
            assert conn.execute("SELECT COUNT(*) FROM migration_log").fetchone()[0] >= 1
