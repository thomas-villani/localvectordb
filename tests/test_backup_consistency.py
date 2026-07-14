"""
Backup consistency across the SQLite <-> FAISS pair.

A backup copies two stores separately: SQLite through its online backup API, and the
FAISS index as a plain file copy. Nothing coordinated them, so a write landing between
the two produced a mutually inconsistent backup.

The dangerous skew is one-directional. Per the dual-store governing rule:

    prefer orphan vectors, never dangling rows

* **Orphan vectors** (index has ids SQLite doesn't reference) are harmless -- search
  JOINs them away and ``repair`` sweeps them.
* **Dangling rows** (SQLite rows whose ``faiss_id`` is absent from the index) are
  poisonous: the vectors have to be re-embedded to recover.

An upsert commits SQLite per batch and flushes FAISS once at the end, so a backup taken
mid-upsert could capture exactly the poisonous case. Passing the live database to
``BackupManager(..., db=db)`` holds its write lock and flushes the index for the
duration of the snapshot, which closes that window.
"""

import threading
import time

import faiss
import pytest

from localvectordb.backup import BackupConfig, BackupManager, BackupType
from localvectordb.database import LocalVectorDB

pytestmark = pytest.mark.integration


def make_db(tmp_path, name="backup_test"):
    return LocalVectorDB(
        name=name,
        base_path=str(tmp_path / "db"),
        embedding_provider="mock",
        embedding_model="mock-model",
    )


def dangling_rows(sqlite_path, faiss_path):
    """faiss_ids SQLite references that are missing from the index -- must always be empty."""
    import sqlite3

    with sqlite3.connect(sqlite_path) as conn:
        referenced = {r[0] for r in conn.execute("SELECT faiss_id FROM chunks WHERE faiss_id IS NOT NULL")}
        try:
            referenced |= {r[0] for r in conn.execute("SELECT faiss_id FROM column_embeddings")}
        except sqlite3.OperationalError:  # pragma: no cover
            pass

    if not faiss_path.exists():
        return referenced

    index = faiss.read_index(str(faiss_path))
    present = set(faiss.vector_to_array(index.id_map).tolist())
    return referenced - present


class TestLiveBackupConsistency:
    def test_backup_of_live_db_under_concurrent_writes_has_no_dangling_rows(self, tmp_path):
        """The real bug: back up while a writer hammers the database."""
        db = make_db(tmp_path)
        db.upsert(documents=[f"seed document number {i}" for i in range(5)], ids=[f"seed{i}" for i in range(5)])

        config = BackupConfig(backup_location=tmp_path / "backups")
        manager = BackupManager(db.db_path, db=db)
        manager.config = config

        # Bounded on purpose: an unthrottled writer rewrites the whole index on every
        # upsert, which is enough to exhaust a small machine before it proves anything.
        # A handful of interleaved writes is all it takes to hit the copy-vs-replace race.
        stop = threading.Event()
        errors = []
        WRITES = 25

        def writer():
            for n in range(WRITES):
                if stop.is_set():
                    return
                try:
                    db.upsert(documents=[f"concurrent document {n}"], ids=[f"w{n}"])
                    time.sleep(0.01)
                except Exception as e:  # pragma: no cover - surfaced below
                    errors.append(e)
                    return

        t = threading.Thread(target=writer, name="BackupWriter", daemon=True)
        t.start()
        try:
            backup_ids = [manager.create_backup(BackupType.FULL) for _ in range(2)]
        finally:
            stop.set()
            t.join(timeout=60)

        assert not errors, f"writer thread failed: {errors[0]!r}"

        for backup_id in backup_ids:
            restore_dir = tmp_path / f"restore_{backup_id[:8]}"
            manager.restore_backup(backup_id, restore_dir, overwrite_existing=True)
            sqlite_path = restore_dir / f"{db.name}.sqlite"
            faiss_path = restore_dir / f"{db.name}.faiss"
            assert sqlite_path.exists(), f"restored SQLite missing for {backup_id}"

            missing = dangling_rows(sqlite_path, faiss_path)
            assert not missing, f"backup {backup_id} has {len(missing)} dangling row(s): {sorted(missing)[:10]}"

        db.close()

    def test_quiesce_flushes_a_dirty_index_before_copying(self, tmp_path):
        """
        A live database with unflushed vectors must still back up consistently.

        The index is flushed under the write lock, so the copied .faiss carries every
        vector SQLite's copy references.
        """
        db = make_db(tmp_path)
        db.upsert(documents=["alpha document"], ids=["a"])

        # Force the in-RAM index to diverge from disk the way an interrupted flush would.
        db._index_dirty = True

        manager = BackupManager(db.db_path, db=db)
        manager.config = BackupConfig(backup_location=tmp_path / "backups")
        backup_id = manager.create_backup(BackupType.FULL)

        # The quiesce flushed it, so the database is clean again.
        assert db._index_dirty is False

        restore_dir = tmp_path / "restore_dirty"
        manager.restore_backup(backup_id, restore_dir, overwrite_existing=True)
        assert not dangling_rows(restore_dir / f"{db.name}.sqlite", restore_dir / f"{db.name}.faiss")
        db.close()


class TestPathOnlyBackupStillWorks:
    def test_backup_of_a_closed_database_by_path(self, tmp_path):
        """The original path-only API is unchanged and needs no live database."""
        db = make_db(tmp_path)
        db.upsert(documents=["only document"], ids=["one"])
        db_path, name = db.db_path, db.name
        db.close()

        manager = BackupManager(db_path)
        manager.config = BackupConfig(backup_location=tmp_path / "backups")
        assert manager._db is None

        backup_id = manager.create_backup(BackupType.FULL)
        restore_dir = tmp_path / "restore_closed"
        manager.restore_backup(backup_id, restore_dir, overwrite_existing=True)

        assert not dangling_rows(restore_dir / f"{name}.sqlite", restore_dir / f"{name}.faiss")
