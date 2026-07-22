"""H12: ingest must not silently swallow per-document failures.

Before the fix, a document that failed to embed/write was dropped from the
returned ID list with no error (async returned ``None``; the sync worker re-raised
in a worker thread the caller never saw). Now every ingest path is best-effort
internally -- each document commits in its own transaction -- and the ``errors``
contract decides what the caller sees: ``errors="raise"`` (default) raises
``IngestError`` naming the failed IDs after the succeeded docs are committed;
``errors="ignore"`` returns only the IDs that landed.
"""

from unittest import mock

import pytest

from localvectordb.core import Chunk, ChunkPosition
from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import IngestError


@pytest.fixture
def db(tmp_path):
    database = LocalVectorDB(
        name="ingest_errs",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="test-model",
        chunk_size=100,
        chunk_overlap=0,
        enable_fts=True,
    )
    yield database
    database.close()


def _fail_chunk_insert_for(db, bad_id):
    """Patch the sync bulk chunk insert so only ``bad_id`` raises."""
    orig = db._insert_chunks_bulk

    def side_effect(conn, chunks_data):
        if chunks_data and chunks_data[0][0] == bad_id:
            raise RuntimeError("boom")
        return orig(conn, chunks_data)

    return mock.patch.object(db, "_insert_chunks_bulk", side_effect=side_effect)


# ---------------------------------------------------------------------------
# The IngestError value object
# ---------------------------------------------------------------------------


class TestIngestErrorType:
    def test_carries_failures_and_succeeded(self):
        err = IngestError("msg", failures={"b": "boom"}, succeeded=["a"])
        assert err.failures == {"b": "boom"}
        assert err.succeeded == ["a"]

    def test_defaults_are_empty(self):
        err = IngestError("msg")
        assert err.failures == {}
        assert err.succeeded == []


# ---------------------------------------------------------------------------
# Sync upsert (main pipeline -> SAVEPOINT-isolated batch)
# ---------------------------------------------------------------------------


class TestSyncUpsertErrors:
    def test_raise_by_default(self, db):
        with _fail_chunk_insert_for(db, "bad"):
            with pytest.raises(IngestError) as ei:
                db.upsert(
                    ["good one text", "the bad document", "good two text"],
                    ids=["g1", "bad", "g2"],
                )
        assert "bad" in ei.value.failures
        assert set(ei.value.succeeded) == {"g1", "g2"}

    def test_succeeded_docs_stay_committed_after_raise(self, db):
        with _fail_chunk_insert_for(db, "bad"):
            with pytest.raises(IngestError):
                db.upsert(
                    ["good one text", "the bad document", "good two text"],
                    ids=["g1", "bad", "g2"],
                )
        # The batch-mates of the failed doc committed; the failed doc did not.
        assert db.exists(["g1", "g2", "bad"]) == [True, True, False]

    def test_ignore_returns_landed_ids(self, db):
        with _fail_chunk_insert_for(db, "bad"):
            landed = db.upsert(
                ["good one text", "the bad document", "good two text"],
                ids=["g1", "bad", "g2"],
                errors="ignore",
            )
        assert set(landed) == {"g1", "g2"}
        assert "bad" not in landed
        assert db.exists(["bad"]) == [False]

    def test_no_failure_no_raise(self, db):
        ids = db.upsert(["a doc", "b doc"], ids=["a", "b"])
        assert set(ids) == {"a", "b"}

    def test_partial_failure_leaves_no_dangling_vectors(self, db):
        # A failed doc's SAVEPOINT rollback must discard the vectors it added, or
        # the index would carry orphans / the SQLite rows would dangle.
        with _fail_chunk_insert_for(db, "bad"):
            with pytest.raises(IngestError):
                db.upsert(["good one", "bad one", "good two"], ids=["g1", "bad", "g2"])
        # A query must still succeed and only surface the committed docs.
        results = db.query("good", search_type="vector", k=10, return_type="documents")
        returned = {r.id for r in results}
        assert "bad" not in returned
        assert returned <= {"g1", "g2"}


# ---------------------------------------------------------------------------
# Sync upsert_from_chunks (per-doc transaction pipeline)
# ---------------------------------------------------------------------------


class TestSyncFromChunksErrors:
    def _chunks(self, text):
        return [Chunk(content=text, position=ChunkPosition(0, len(text), 1, 1, 1, len(text) + 1), tokens=2, index=0)]

    def test_raise_by_default(self, db):
        with _fail_chunk_insert_for(db, "bad"):
            with pytest.raises(IngestError) as ei:
                db.upsert_from_chunks(
                    {"g1": self._chunks("good one"), "bad": self._chunks("bad"), "g2": self._chunks("good two")}
                )
        assert "bad" in ei.value.failures
        assert set(ei.value.succeeded) == {"g1", "g2"}
        assert db.exists(["g1", "g2", "bad"]) == [True, True, False]

    def test_ignore_returns_landed(self, db):
        with _fail_chunk_insert_for(db, "bad"):
            landed = db.upsert_from_chunks(
                {"g1": self._chunks("good one"), "bad": self._chunks("bad"), "g2": self._chunks("good two")},
                errors="ignore",
            )
        assert set(landed) == {"g1", "g2"}


# ---------------------------------------------------------------------------
# Async upsert
# ---------------------------------------------------------------------------


class TestAsyncUpsertErrors:
    def _fail_async_chunk_insert_for(self, db, bad_id):
        orig = db._insert_chunks_bulk_async

        async def side_effect(conn, chunks_data):
            if chunks_data and chunks_data[0][0] == bad_id:
                raise RuntimeError("boom")
            return await orig(conn, chunks_data)

        return mock.patch.object(db, "_insert_chunks_bulk_async", side_effect=side_effect)

    async def test_raise_by_default(self, db):
        with self._fail_async_chunk_insert_for(db, "bad"):
            with pytest.raises(IngestError) as ei:
                await db.upsert_async(
                    ["good one text", "the bad document", "good two text"],
                    ids=["g1", "bad", "g2"],
                )
        assert "bad" in ei.value.failures
        assert set(ei.value.succeeded) == {"g1", "g2"}
        assert db.exists(["g1", "g2", "bad"]) == [True, True, False]

    async def test_ignore_returns_landed(self, db):
        with self._fail_async_chunk_insert_for(db, "bad"):
            landed = await db.upsert_async(
                ["good one text", "the bad document", "good two text"],
                ids=["g1", "bad", "g2"],
                errors="ignore",
            )
        assert set(landed) == {"g1", "g2"}
        assert db.exists(["bad"]) == [False]

    async def test_no_failure_no_raise(self, db):
        ids = await db.upsert_async(["a doc", "b doc"], ids=["a", "b"])
        assert set(ids) == {"a", "b"}
