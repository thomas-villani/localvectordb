"""
Index-persistence dirty tracking and read-only memory-mapped databases.

Two coupled behaviors are exercised here:

1. **Dirty tracking.** ``save()`` (and therefore ``close()``) must rewrite the
   ``.faiss`` file only when the in-RAM index has actually diverged from disk. A
   database that merely served reads is *clean*, so closing it -- or having the
   server evict it on idle -- must not rewrite the file. That matters for
   multi-worker read fan-out: N workers independently rewriting the same index
   file on eviction is a write race (and, on Windows, the ``os.replace`` flake).

2. **Read-only mmap.** ``mmap_index=True`` opens the FAISS index with
   ``IO_FLAG_MMAP`` so many workers share one page-cached copy. An mmap'd index
   *silently* copies its storage out of the mapping when grown -- so a naive
   write would succeed in RAM, diverge from disk, and vanish on eviction. Every
   mutating path must therefore refuse, loudly and synchronously, on such a
   database.

Real FAISS + real SQLite + deterministic ``"mock"`` embeddings (seeded from a
hash of the text, so the same string embeds identically every time -- which is
what lets the mmap-reader query assert an exact-match ranking).
"""

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import UnsupportedIndexOperationError

pytestmark = pytest.mark.integration


def make_db(tmp_path, name="persist_test", **kwargs):
    kwargs.setdefault("embedding_provider", "mock")
    kwargs.setdefault("embedding_model", "mock-model")
    return LocalVectorDB(name=name, base_path=str(tmp_path), **kwargs)


def index_path(tmp_path, name="persist_test"):
    return Path(tmp_path) / f"{name}.faiss"


def build_populated_db(tmp_path):
    """Create a two-document database on disk and close it cleanly."""
    db = make_db(tmp_path)
    db.upsert(documents=["the quick brown fox jumps over the lazy dog"], ids=["fox"])
    db.upsert(documents=["a completely unrelated sentence about deep ocean currents"], ids=["ocean"])
    db.close()
    assert index_path(tmp_path).exists()


class TestDirtyTracking:
    def test_fresh_index_is_dirty_and_first_save_persists(self, tmp_path):
        db = make_db(tmp_path)
        # A brand-new, still-empty index has no file yet, so it must be dirty or the
        # first save would short-circuit and never write it.
        assert db._index_dirty is True
        try:
            db.save()
            assert index_path(tmp_path).exists()
            assert db._index_dirty is False
        finally:
            db.close()

    def test_read_only_session_leaves_index_clean(self, tmp_path):
        build_populated_db(tmp_path)
        db = make_db(tmp_path, create_if_not_exists=False)
        try:
            results = db.query("quick brown fox", k=2)
            assert {r.id for r in results} == {"fox", "ocean"}
            # Reads never dirty the index.
            assert db._index_dirty is False
        finally:
            db.close()

    def test_clean_close_does_not_rewrite_the_file(self, tmp_path):
        build_populated_db(tmp_path)
        db = make_db(tmp_path, create_if_not_exists=False)
        db.query("quick brown fox", k=1)
        # Spy on the actual write primitive: a clean close must never reach it.
        with mock.patch.object(db, "_save_internal", wraps=db._save_internal) as spy:
            db.close()
        spy.assert_not_called()

    def test_mutation_marks_dirty_and_close_rewrites(self, tmp_path):
        build_populated_db(tmp_path)
        db = make_db(tmp_path, create_if_not_exists=False)
        db.upsert(documents=["a third, newly added document"], ids=["third"])
        assert db._index_dirty is True
        with mock.patch.object(db, "_save_internal", wraps=db._save_internal) as spy:
            db.close()
        spy.assert_called()

    def test_delete_marks_dirty(self, tmp_path):
        build_populated_db(tmp_path)
        db = make_db(tmp_path, create_if_not_exists=False)
        try:
            db.delete("fox")
            assert db._index_dirty is True
        finally:
            db.close()

    def test_rewrite_round_trips_new_content(self, tmp_path):
        """A dirty close must actually persist -- the reopened DB sees the new doc."""
        build_populated_db(tmp_path)
        db = make_db(tmp_path, create_if_not_exists=False)
        db.upsert(documents=["persisted across reopen via a dirty save"], ids=["third"])
        db.close()

        reopened = make_db(tmp_path, create_if_not_exists=False)
        try:
            assert reopened.get("third") is not None
        finally:
            reopened.close()


class TestMmapReadOnly:
    def test_reader_serves_correct_rankings(self, tmp_path):
        build_populated_db(tmp_path)
        reader = make_db(tmp_path, create_if_not_exists=False, mmap_index=True)
        try:
            top = reader.query("quick brown fox", k=1)
            assert top[0].id == "fox"
        finally:
            reader.close()

    @pytest.mark.parametrize(
        "operation",
        [
            lambda db: db.upsert(documents=["nope"], ids=["z"]),
            lambda db: db.insert(documents=["nope"], ids=["z2"]),
            lambda db: db.update("fox", content="changed"),
            lambda db: db.delete("fox"),
        ],
        ids=["upsert", "insert", "update", "delete"],
    )
    def test_reader_refuses_writes(self, tmp_path, operation):
        build_populated_db(tmp_path)
        reader = make_db(tmp_path, create_if_not_exists=False, mmap_index=True)
        try:
            with pytest.raises(UnsupportedIndexOperationError):
                operation(reader)
        finally:
            reader.close()

    def test_reader_refuses_writes_without_touching_the_index(self, tmp_path):
        """A refused write must be synchronous and leave the vector count unchanged."""
        build_populated_db(tmp_path)
        reader = make_db(tmp_path, create_if_not_exists=False, mmap_index=True)
        try:
            before = reader.index.ntotal
            with pytest.raises(UnsupportedIndexOperationError):
                reader.upsert(documents=["nope"], ids=["z"])
            assert reader.index.ntotal == before
        finally:
            reader.close()

    def test_reader_close_does_not_rewrite(self, tmp_path):
        build_populated_db(tmp_path)
        reader = make_db(tmp_path, create_if_not_exists=False, mmap_index=True)
        reader.query("quick brown fox", k=1)
        with mock.patch.object(reader, "_save_internal", wraps=reader._save_internal) as spy:
            reader.close()
        spy.assert_not_called()

    def test_async_reader_refuses_writes(self, tmp_path):
        build_populated_db(tmp_path)
        reader = make_db(tmp_path, create_if_not_exists=False, mmap_index=True)

        async def _run():
            with pytest.raises(UnsupportedIndexOperationError):
                await reader.upsert_async(documents=["nope"], ids=["z"])
            with pytest.raises(UnsupportedIndexOperationError):
                await reader.delete_async("fox")

        try:
            asyncio.run(_run())
        finally:
            reader.close()
