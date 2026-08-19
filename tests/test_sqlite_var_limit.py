"""Regression tests for SQLite's bound-parameter limit ("too many SQL variables").

SQLite caps bound parameters per statement at compile time: 999 before 3.32,
32,766 since. Any ``IN (?,?,...)`` list expanded from a caller-sized id list
must therefore batch through :func:`localvectordb.utils.iter_sql_id_batches`.
These tests drive each reachable path with an id list comfortably over the
modern limit; before the batching fix every one of them died with
``sqlite3.OperationalError`` (first seen at the 50k scale of the tier-2 insert
benchmark, where ``upsert`` pre-fetches existing chunks for all ids at once).

The lists are made of mostly-missing ids on a small database, so each test
exercises the real >32k-variable shape in well under a second.
"""

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import DocumentNotFoundError
from localvectordb.utils import SQLITE_MAX_VARS, iter_sql_id_batches

# Comfortably above SQLITE_MAX_VARIABLE_NUMBER on every modern build (32,766).
N_IDS = 40_000

pytestmark = [pytest.mark.integration, pytest.mark.database]


def _make_db(tmp_path) -> LocalVectorDB:
    return LocalVectorDB(
        name="varlimitdb",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="mock",
        enable_fts=False,
    )


def _many_missing_ids() -> list:
    return [f"missing-{i}" for i in range(N_IDS)]


@pytest.fixture
def db(tmp_path):
    database = _make_db(tmp_path)
    database.upsert(
        ["alpha document", "beta document", "gamma document"],
        ids=["d1", "d2", "d3"],
    )
    yield database
    database.close()


def test_iter_sql_id_batches_shapes():
    items = list(range(2 * SQLITE_MAX_VARS + 5))
    batches = list(iter_sql_id_batches(items))
    assert [len(b) for b in batches] == [SQLITE_MAX_VARS, SQLITE_MAX_VARS, 5]
    assert [x for b in batches for x in b] == items
    assert list(iter_sql_id_batches([])) == []


def test_exists_with_many_ids(db):
    ids = ["d1", *_many_missing_ids(), "d3"]
    result = db.exists(ids)
    assert len(result) == len(ids)
    assert result[0] is True
    assert result[-1] is True
    assert not any(result[1:-1])


def test_get_with_many_ids_reports_missing_not_sql_error(db):
    with pytest.raises(DocumentNotFoundError):
        db.get(["d1", *_many_missing_ids()])


def test_delete_with_many_ids(db):
    deleted = db.delete([*_many_missing_ids(), "d1", "d2"])
    assert deleted == 2
    assert db.exists(["d1", "d2", "d3"]) == [False, False, True]


def test_fetch_existing_chunks_with_many_ids(db):
    existing = db._fetch_existing_chunks_batch([*_many_missing_ids(), "d1", "d2", "d3"])
    assert set(existing) == {"d1", "d2", "d3"}


async def test_exists_async_with_many_ids(db):
    ids = ["d2", *_many_missing_ids()]
    result = await db.exists_async(ids)
    assert len(result) == len(ids)
    assert result[0] is True
    assert not any(result[1:])


async def test_get_async_with_many_ids_reports_missing_not_sql_error(db):
    with pytest.raises(DocumentNotFoundError):
        await db.get_async(["d1", *_many_missing_ids()])


async def test_delete_async_with_many_ids(db):
    deleted = await db.delete_async([*_many_missing_ids(), "d3"])
    assert deleted == 1


async def test_check_existing_ids_async_with_many_ids(db):
    db._ensure_async_pool()
    found = await db._check_existing_ids_async([*_many_missing_ids(), "d1"])
    assert found == {"d1"}


async def test_fetch_existing_chunks_async_with_many_ids(db):
    db._ensure_async_pool()
    existing = await db._fetch_existing_chunks_batch_async([*_many_missing_ids(), "d2"])
    assert set(existing) == {"d2"}
