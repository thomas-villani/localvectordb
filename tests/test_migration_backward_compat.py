"""End-to-end backward-compatibility tests for on-disk database upgrades.

Unlike ``test_migration.py`` (which exercises the metadata-schema
``MigrationEngine`` in isolation), these tests open a database that has been
downgraded on disk to a genuinely *older* schema shape and assert that simply
reopening it auto-upgrades the file: the idempotent ``ADD COLUMN`` guards and
version-tracking backfill in :mod:`localvectordb._schema` run on open, and the
stored data remains queryable.

This guards the property the RC relies on -- every on-disk format change to date
has been purely additive, so databases written by older versions must keep
opening cleanly.
"""

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from localvectordb._pools import ReadWriteLock
from localvectordb._schema import DatabaseSchema
from localvectordb.database import LocalVectorDB


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


def _downgrade_to_legacy(sqlite_path: Path) -> None:
    """Rewrite an on-disk DB to look like a pre-hierarchical, pre-versioning DB.

    Strips the columns/tables that later versions add automatically on open so
    that reopening must re-create them.
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        # section_id is indexed; SQLite refuses DROP COLUMN while an index
        # references it, so drop the index first (an old DB never had either).
        conn.execute("DROP INDEX IF EXISTS idx_chunks_section_id")
        conn.execute("ALTER TABLE chunks DROP COLUMN section_id")
        conn.execute("ALTER TABLE documents DROP COLUMN doc_faiss_id")
        conn.execute("ALTER TABLE metadata_schema DROP COLUMN embedding_enabled")
        conn.execute("ALTER TABLE metadata_schema DROP COLUMN fts_enabled")
        # Drop all version tracking to mimic a database that predates it.
        conn.execute("DROP TABLE IF EXISTS migration_log")
        conn.execute("PRAGMA user_version = 0")
        conn.execute("DELETE FROM config WHERE key IN ('db_version', 'version_updated_at')")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
@pytest.mark.database
def test_legacy_ondisk_db_auto_upgrades_on_open():
    """An old-format on-disk DB is upgraded and stays queryable when reopened."""
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
            assert pre.execute("PRAGMA user_version").fetchone()[0] == 0

        # 3. Reopen with current code -> synchronous constructor auto-migrates.
        db2 = _make_db(tmpdir)
        try:
            with closing(sqlite3.connect(sqlite_path)) as post:
                # Additive column guards re-created the missing columns.
                assert "section_id" in _columns(post, "chunks")
                assert "doc_faiss_id" in _columns(post, "documents")
                assert "embedding_enabled" in _columns(post, "metadata_schema")
                assert "fts_enabled" in _columns(post, "metadata_schema")
                # Version tracking was re-initialized.
                assert post.execute("PRAGMA user_version").fetchone()[0] > 0
                assert post.execute("SELECT value FROM config WHERE key = 'db_version'").fetchone() is not None
                migration_log = post.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_log'"
                ).fetchone()
                assert migration_log is not None

            # 4. Stored data survived the upgrade and is still searchable.
            results = db2.query("brown fox", k=2, search_type="hybrid")
            assert len(results) > 0
        finally:
            db2.close()


@pytest.mark.integration
@pytest.mark.database
async def test_async_initialize_stamps_version_tracking():
    """initialize_async stamps version metadata like the sync path (async-safe)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "async_init.sqlite"
        schema = DatabaseSchema(db_path, ReadWriteLock())

        # Initialize a fresh database purely through the async path.
        await schema.initialize_async()

        with closing(sqlite3.connect(db_path)) as conn:
            # user_version encodes the current schema version (non-zero).
            assert conn.execute("PRAGMA user_version").fetchone()[0] > 0
            # db_version config row and a migration_log entry were written.
            assert conn.execute("SELECT value FROM config WHERE key = 'db_version'").fetchone() is not None
            assert conn.execute("SELECT COUNT(*) FROM migration_log").fetchone()[0] >= 1
