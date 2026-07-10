"""T1.3 -- metadata filters are pushed into search, not applied to a starved pool.

Before this fix, ``query()``/``query_async()`` (and the cursor and hierarchical
paths) fetched a *fixed* candidate pool -- ``initial_k = k*2..k*4`` for FAISS,
``LIMIT initial_k`` for FTS -- and then discarded non-matching rows in Python. A
filter matching a small fraction of the corpus therefore returned far fewer than
``k`` results even though matches existed further down the ranking, silently.

The fix pushes a SQL-expressible filter into the index (a FAISS ``IDSelector``
over the matching id set) or into the FTS query (a subquery so ``LIMIT`` applies
after filtering), so a selective filter returns its matches. ``matches_metadata_
filter`` stays the authority, so results never change for the no-filter path or
for filters where SQL and Python agree.

These tests use a **real** FAISS index (mock embeddings, FAISS unpatched): a mock
index returns random ids, so it cannot exercise the selector at all.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from localvectordb._filters import matches_metadata_filter
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB

# The rare category matches only these documents out of ``CORPUS_SIZE``. With
# ``k`` small the pre-fix fixed pool (k*2..k*4) almost never contained all of
# them, so filtered search returned fewer than exist.
CORPUS_SIZE = 80
RARE_DOC_IDS = {"doc7", "doc31", "doc58"}
SHARED_TERMS = "alpha beta gamma delta topics documents things"


def _make_db(tmp: str, *, index_type: str = "IndexFlatL2", hierarchical: bool = False) -> LocalVectorDB:
    db = LocalVectorDB(
        name="t13",
        base_path=tmp,
        embedding_provider="mock",
        embedding_model="mock-model",
        faiss_index_type=index_type,
        hierarchical_embeddings=hierarchical,
        metadata_schema={
            "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "priority": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
        },
    )
    contents, metas, ids = [], [], []
    for i in range(CORPUS_SIZE):
        doc_id = f"doc{i}"
        rare = doc_id in RARE_DOC_IDS
        contents.append(f"{SHARED_TERMS} document number {i} paragraph content body text")
        metas.append({"category": "rare" if rare else "common", "priority": 1 if rare else 5})
        ids.append(doc_id)
    db.upsert(contents, metadata=metas, ids=ids)
    return db


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    database = _make_db(tmp)
    yield database
    database.close()
    shutil.rmtree(tmp, ignore_errors=True)


class TestSupportsIdSelector:
    """The selector is supported by every index type except IndexLSH."""

    @pytest.mark.parametrize(
        "index_type,expected",
        [
            ("IndexFlatL2", True),
            ("IndexFlatIP", True),
            ("IndexHNSWFlat", True),
            ("IndexLSH", False),
        ],
    )
    def test_capability_detection(self, index_type, expected):
        tmp = tempfile.mkdtemp()
        try:
            database = _make_db(tmp, index_type=index_type)
            assert database.supports_id_selector is expected
            database.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSelectiveFilterNotStarved:
    """A filter matching 3/80 docs returns all 3, not whatever survived the pool."""

    @pytest.mark.parametrize("search_type", ["vector", "keyword", "hybrid"])
    def test_all_matches_returned_chunks(self, db, search_type):
        results = db.query(
            SHARED_TERMS,
            search_type=search_type,
            k=5,
            filters={"category": "rare"},
            return_type="chunks",
        )
        found = {r.document_id for r in results}
        assert found == RARE_DOC_IDS
        assert all(r.metadata.get("category") == "rare" for r in results)

    @pytest.mark.parametrize("search_type", ["vector", "keyword", "hybrid"])
    def test_all_matches_returned_documents(self, db, search_type):
        results = db.query(
            SHARED_TERMS,
            search_type=search_type,
            k=5,
            filters={"category": "rare"},
            return_type="documents",
        )
        assert {r.id for r in results} == RARE_DOC_IDS

    @pytest.mark.parametrize("search_type", ["vector", "keyword", "hybrid"])
    def test_operator_filter_pushed_down(self, db, search_type):
        # priority == 1 only for the rare docs; exercises a non-equality operator.
        results = db.query(
            SHARED_TERMS,
            search_type=search_type,
            k=5,
            filters={"priority": {"$lt": 2}},
            return_type="chunks",
        )
        assert {r.document_id for r in results} == RARE_DOC_IDS

    async def test_async_all_matches_returned(self, db):
        for search_type in ("vector", "keyword", "hybrid"):
            results = await db.query_async(
                SHARED_TERMS,
                search_type=search_type,
                k=5,
                filters={"category": "rare"},
                return_type="chunks",
            )
            found = {r.document_id for r in results}
            assert found == RARE_DOC_IDS, search_type

    def test_cursor_streams_all_matches(self, db):
        cursor = db.query_cursor(
            SHARED_TERMS,
            search_type="vector",
            k=5,
            filters={"category": "rare"},
            return_type="chunks",
            batch_size=10,
        )
        found = {r.document_id for r in cursor.fetch_all()}
        assert found == RARE_DOC_IDS

    def test_unfiltered_unchanged(self, db):
        # The no-filter path must be untouched: k results, no starvation logic.
        results = db.query(SHARED_TERMS, search_type="vector", k=10, return_type="documents")
        assert len(results) == 10

    def test_filter_matching_everything_equals_unfiltered(self, db):
        # A filter that matches all docs must return the same top-k as no filter,
        # proving the pushdown does not perturb ranking when it selects everyone.
        plain = db.query(SHARED_TERMS, search_type="vector", k=10, return_type="documents")
        filtered = db.query(
            SHARED_TERMS,
            search_type="vector",
            k=10,
            filters={"priority": {"$gte": 1}},
            return_type="documents",
        )
        assert [r.id for r in filtered] == [r.id for r in plain]


class TestHierarchicalFilterPushdown:
    """Section- and document-level searches also honour a selective filter."""

    @pytest.mark.parametrize("search_level", ["sections", "documents"])
    def test_hierarchical_filtered(self, search_level):
        tmp = tempfile.mkdtemp()
        try:
            database = _make_db(tmp, hierarchical=True)
            results = database.query(
                SHARED_TERMS,
                search_type="vector",
                search_level=search_level,
                k=5,
                filters={"category": "rare"},
                return_type="chunks" if search_level == "sections" else "documents",
            )
            doc_ids = {r.document_id or r.id for r in results}
            assert doc_ids <= RARE_DOC_IDS
            assert doc_ids  # non-empty: the matches were found, not starved
            database.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestNonPushdownFallback:
    """LSH (no selector) and dot-notation filters fall back without corrupting results."""

    def test_lsh_falls_back_and_warns(self, caplog):
        tmp = tempfile.mkdtemp()
        try:
            database = _make_db(tmp, index_type="IndexLSH")
            assert database.supports_id_selector is False
            with caplog.at_level("WARNING", logger="localvectordb.database._search"):
                results = database.query(
                    SHARED_TERMS,
                    search_type="vector",
                    k=5,
                    filters={"category": "rare"},
                    return_type="chunks",
                )
            # Whatever survives is correctly filtered (never a wrong-category leak)...
            assert all(r.metadata.get("category") == "rare" for r in results)
            # ...and the starvation is surfaced, not silent.
            assert any("could not be pushed into the index" in m for m in caplog.messages)
            database.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dot_notation_filter_does_not_crash(self):
        tmp = tempfile.mkdtemp()
        try:
            database = LocalVectorDB(
                name="json",
                base_path=tmp,
                embedding_provider="mock",
                embedding_model="mock-model",
                metadata_schema={"info": MetadataField(type=MetadataFieldType.JSON)},
            )
            database.upsert(
                [f"{SHARED_TERMS} doc {i}" for i in range(20)],
                metadata=[{"info": {"tier": "gold" if i == 3 else "silver"}} for i in range(20)],
                ids=[f"d{i}" for i in range(20)],
            )
            # Not SQL-expressible -> pushdown declined, Python matcher still applied.
            results = database.query(
                SHARED_TERMS,
                search_type="vector",
                k=5,
                filters={"info.tier": "gold"},
                return_type="chunks",
            )
            assert all(r.metadata.get("info", {}).get("tier") == "gold" for r in results)
            database.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSqlPythonFilterParity:
    """The SQL pushdown must select exactly the documents the Python matcher keeps.

    If the two diverged, pushing the filter into the query would silently change
    which documents a filtered search can return. This asserts equivalence over a
    range of operators on a real dataset.
    """

    FILTERS = [
        {"category": "rare"},
        {"category": {"$ne": "rare"}},
        {"priority": {"$lt": 2}},
        {"priority": {"$gte": 5}},
        {"priority": {"$in": [1, 5]}},
        {"category": {"$in": ["rare", "common"]}},
        {"$and": [{"category": "rare"}, {"priority": 1}]},
        {"$or": [{"category": "rare"}, {"priority": {"$gt": 100}}]},
        {"$not": {"category": "rare"}},
        {"category": {"$startswith": "ra"}},
    ]

    @pytest.mark.parametrize("filters", FILTERS)
    def test_pushdown_matches_python(self, db, filters):
        # Documents the Python matcher accepts.
        with db.connection_pool.get_connection() as conn:
            rows = conn.execute("SELECT id, category, priority FROM documents").fetchall()
        python_ids = {
            row["id"]
            for row in rows
            if matches_metadata_filter({"category": row["category"], "priority": row["priority"]}, filters)
        }

        # Documents the SQL pushdown selects (via the shared WHERE builder).
        built = db._build_filter_where(filters)
        if built is None:
            # Not SQL-expressible (e.g. $in against a typed column): the Python
            # matcher is the sole authority, so there is nothing to reconcile.
            pytest.skip("filter is not pushed down; Python matcher is authoritative")
        where_clause, params = built
        with db.connection_pool.get_connection() as conn:
            sql_rows = conn.execute(f"SELECT id FROM documents WHERE {where_clause}", params).fetchall()
        sql_ids = {row["id"] for row in sql_rows}

        assert sql_ids == python_ids, filters
