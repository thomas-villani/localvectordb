"""The persisted default reranker: resolution, persistence, laziness, semantics.

A database can carry a default reranker in its own config (like the embedding
provider), applied by every ``query()`` unless overridden per call. These tests
pin the four contracts that make that safe:

* resolution -- explicit constructor arg wins with a warning, saved value
  applies when the caller says nothing, key absence means "no default";
* persistence -- one JSON config key, surviving reopen, deletable;
* laziness -- opening a database never constructs a reranker, the first query
  that needs it does, and the instance is cached per database;
* semantics -- ``reranker=False`` disables for one call, cursors ignore the
  default entirely, multi-column applies it once to the merged pool, and a
  builder rerank step suppresses it (no double-rerank).
"""

import json
import sqlite3

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.database._core import LocalVectorDBCore
from localvectordb.reranking import MockReranker, RerankerRegistry

MOCK_CFG = {"provider": "mock", "model": "mock-reranker"}
DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "lazy dogs sleep all day in the warm sun",
    "foxes are quick and clever animals in the wild",
    "the sun rises early over the quiet farm",
]


class _CountingReranker(MockReranker):
    """MockReranker that counts rerank invocations, class-wide."""

    rerank_calls = 0

    @property
    def provider_name(self) -> str:
        return "counting"

    def rerank(self, query, results, top_k=None):
        type(self).rerank_calls += 1
        return super().rerank(query, results, top_k=top_k)


@pytest.fixture
def counting_provider():
    RerankerRegistry.register("counting", _CountingReranker)
    _CountingReranker.rerank_calls = 0
    yield _CountingReranker
    RerankerRegistry._providers.pop("counting", None)


@pytest.fixture
def construction_counter(monkeypatch):
    """Counts RerankerRegistry.create_reranker calls without changing behavior."""
    calls = {"n": 0}
    orig = RerankerRegistry.create_reranker.__func__

    def counting(cls, provider, model=None, **kw):
        calls["n"] += 1
        return orig(cls, provider, model, **kw)

    monkeypatch.setattr(RerankerRegistry, "create_reranker", classmethod(counting))
    return calls


def _db(tmp_path, **kwargs) -> LocalVectorDB:
    kwargs.setdefault("embedding_provider", "mock")
    return LocalVectorDB(name="rrdb", base_path=str(tmp_path), **kwargs)


def _seed(db: LocalVectorDB) -> None:
    for text in DOCS:
        db.upsert(text)


def _raw_config_value(db: LocalVectorDB, key: str = "default_reranker"):
    with db.connection_pool.get_connection() as conn:
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _delete_config_key(db_path, key: str = "default_reranker") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM config WHERE key = ?", (key,))
    conn.commit()
    conn.close()


@pytest.mark.unit
class TestSavedRerankerResolution:
    """Unit-level: the resolver, independent of a real database."""

    def test_absent_key_returns_request(self):
        assert LocalVectorDBCore._resolve_saved_reranker(MOCK_CFG, {}) == MOCK_CFG
        assert LocalVectorDBCore._resolve_saved_reranker(None, {"embedding_model": "m"}) is None

    def test_saved_wins_when_caller_silent(self):
        loaded = {"default_reranker": json.dumps(MOCK_CFG)}
        assert LocalVectorDBCore._resolve_saved_reranker(None, loaded) == MOCK_CFG

    def test_divergent_request_warns_and_wins(self, caplog):
        loaded = {"default_reranker": json.dumps(MOCK_CFG)}
        requested = {"provider": "mock", "model": "other"}
        with caplog.at_level("WARNING"):
            resolved = LocalVectorDBCore._resolve_saved_reranker(requested, loaded)
        assert resolved == requested
        assert any("overrides the persisted default reranker" in r.message for r in caplog.records)

    def test_restated_request_is_silent(self, caplog):
        loaded = {"default_reranker": json.dumps(MOCK_CFG)}
        with caplog.at_level("WARNING"):
            resolved = LocalVectorDBCore._resolve_saved_reranker(dict(MOCK_CFG), loaded)
        assert resolved == MOCK_CFG
        assert not caplog.records

    def test_corrupt_json_warns_and_falls_back(self, caplog):
        loaded = {"default_reranker": "{not json"}
        with caplog.at_level("WARNING"):
            assert LocalVectorDBCore._resolve_saved_reranker(None, loaded) is None
            assert LocalVectorDBCore._resolve_saved_reranker(MOCK_CFG, loaded) == MOCK_CFG
        assert any("unreadable default_reranker" in r.message for r in caplog.records)


@pytest.mark.integration
class TestPersistence:
    def test_round_trips_as_json_and_survives_reopen(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        raw = _raw_config_value(db)
        assert json.loads(raw) == MOCK_CFG  # real JSON, not str(dict)
        db.close()

        db2 = _db(tmp_path)  # no reranker_config: inherits the saved default
        assert db2.get_default_reranker() == MOCK_CFG
        db2.close()

    def test_pre_feature_database_has_no_default(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        db_path = db.db_path
        db.close()
        _delete_config_key(db_path)

        db2 = _db(tmp_path)
        assert db2.get_default_reranker() is None
        db2.close()

    def test_override_on_reopen_warns_and_repersists(self, tmp_path, caplog):
        _db(tmp_path, reranker_config=MOCK_CFG).close()
        override = {"provider": "mock", "model": "override-model"}
        with caplog.at_level("WARNING"):
            db2 = _db(tmp_path, reranker_config=override)
        assert db2.get_default_reranker() == override
        assert any("overrides the persisted default reranker" in r.message for r in caplog.records)
        db2.close()

        db3 = _db(tmp_path)  # the override is now the persisted default
        assert db3.get_default_reranker() == override
        db3.close()

    def test_unknown_provider_at_creation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            _db(tmp_path, reranker_config={"provider": "no-such-provider"})


@pytest.mark.integration
class TestMutators:
    def test_set_persists_and_none_deletes(self, tmp_path):
        db = _db(tmp_path)
        db.set_default_reranker(MOCK_CFG)
        assert json.loads(_raw_config_value(db)) == MOCK_CFG
        db.close()

        db2 = _db(tmp_path)
        assert db2.get_default_reranker() == MOCK_CFG
        db2.set_default_reranker(None)
        assert _raw_config_value(db2) is None
        db2.close()

        db3 = _db(tmp_path)
        assert db3.get_default_reranker() is None
        db3.close()

    def test_persist_false_does_not_survive_reopen(self, tmp_path):
        _db(tmp_path, reranker_config=MOCK_CFG).close()
        db = _db(tmp_path)
        db.set_default_reranker(None, persist=False)
        assert db.get_default_reranker() is None
        db.close()

        db2 = _db(tmp_path)
        assert db2.get_default_reranker() == MOCK_CFG
        db2.close()

    def test_unknown_provider_raises_and_leaves_default(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            db.set_default_reranker({"provider": "no-such-provider"})
        assert db.get_default_reranker() == MOCK_CFG
        db.close()

    def test_missing_provider_key_raises(self, tmp_path):
        db = _db(tmp_path)
        with pytest.raises(ValueError, match="non-empty 'provider'"):
            db.set_default_reranker({"model": "x"})
        db.close()

    def test_set_invalidates_cached_instance(self, tmp_path, construction_counter):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        db.query("quick fox", k=2)
        assert construction_counter["n"] == 1
        db.set_default_reranker({"provider": "mock", "model": "second"})
        db.query("quick fox", k=2)
        assert construction_counter["n"] == 2
        db.close()

    def test_raw_api_key_warns_env_ref_does_not(self, tmp_path, caplog):
        db = _db(tmp_path)
        with caplog.at_level("WARNING"):
            db.set_default_reranker({"provider": "mock", "api_key": "sk-raw-secret"})
        assert any("raw api_key" in r.message for r in caplog.records)
        caplog.clear()
        with caplog.at_level("WARNING"):
            db.set_default_reranker({"provider": "mock", "api_key": "$MY_KEY_VAR"})
        assert not any("raw api_key" in r.message for r in caplog.records)
        db.close()

    def test_get_returns_a_copy(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        db.get_default_reranker()["model"] = "mutated"
        assert db.get_default_reranker() == MOCK_CFG
        db.close()


@pytest.mark.integration
class TestLaziness:
    def test_open_and_ingest_construct_nothing(self, tmp_path, construction_counter):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        assert construction_counter["n"] == 0
        db.close()

        db2 = _db(tmp_path)  # reopen with a persisted default: still nothing
        assert construction_counter["n"] == 0
        db2.close()

    def test_instance_is_cached_across_queries(self, tmp_path, construction_counter):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        for _ in range(3):
            db.query("quick fox", k=2)
        assert construction_counter["n"] == 1
        db.close()


@pytest.mark.integration
class TestQuerySemantics:
    def test_default_applies_and_false_disables(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        reranked = db.query("quick fox", k=2)
        assert all("original_score" in (r.metadata or {}) for r in reranked)
        plain = db.query("quick fox", k=2, reranker=False)
        assert all("original_score" not in (r.metadata or {}) for r in plain)
        db.close()

    def test_per_call_overrides_beat_default(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        # per-call config: the default (counting) must not run
        db.query("quick fox", k=2, reranker_config=MOCK_CFG)
        assert counting_provider.rerank_calls == 0
        # per-call instance: same
        db.query("quick fox", k=2, reranker=MockReranker())
        assert counting_provider.rerank_calls == 0
        # no per-call args: the default runs
        db.query("quick fox", k=2)
        assert counting_provider.rerank_calls == 1
        db.close()

    def test_false_with_config_raises(self, tmp_path):
        db = _db(tmp_path)
        _seed(db)
        with pytest.raises(ValueError, match="cannot be\\s+combined"):
            db.query("quick fox", k=2, reranker=False, reranker_config=MOCK_CFG)
        db.close()

    async def test_async_applies_default_once(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        results = await db.query_async("quick fox", k=2)
        assert counting_provider.rerank_calls == 1
        assert all("original_score" in (r.metadata or {}) for r in results)
        db.close()

    async def test_async_delegated_level_applies_default_once(self, tmp_path, counting_provider):
        """The fused/hierarchical delegation must not double-apply the default."""
        db = LocalVectorDB(
            name="rrdb_hier",
            base_path=str(tmp_path),
            embedding_provider="mock",
            hierarchical_embeddings=True,
            reranker_config={"provider": "counting"},
        )
        db.upsert("# Foxes\n\nthe quick brown fox jumps\n\n# Dogs\n\nlazy dogs sleep all day")
        db.upsert("# Farm\n\nthe sun rises early over the quiet farm")
        await db.query_async("quick fox", k=2, search_level="sections")
        assert counting_provider.rerank_calls == 1
        db.close()

    def test_default_widens_fetch_k(self, tmp_path):
        pools = []

        class _RecordingDefault(MockReranker):
            @property
            def provider_name(self):
                return "recording"

            def rerank(self, query, results, top_k=None):
                pools.append(len(results))
                return super().rerank(query, results, top_k=top_k)

        RerankerRegistry.register("recording", _RecordingDefault)
        try:
            db = _db(tmp_path, reranker_config={"provider": "recording"})
            for i in range(12):
                db.upsert(f"quick fox document number {i} with some words")
            db.query("quick fox", k=2)
            # 5*k = 10 candidates fetched for the reranker, not k = 2.
            assert pools == [10]
        finally:
            RerankerRegistry._providers.pop("recording", None)
            db.close()


@pytest.mark.integration
class TestCursorsIgnoreDefault:
    def test_cursor_neither_raises_nor_reranks(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        results = list(db.query_cursor("quick fox").stream_individual())
        assert results
        assert all("original_score" not in (r.metadata or {}) for r in results)
        db.close()

    def test_cursor_accepts_explicit_false(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        assert list(db.query_cursor("quick fox", reranker=False).stream_individual())
        db.close()

    def test_cursor_still_rejects_explicit_reranker(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        with pytest.raises(ValueError):
            db.query_cursor("quick fox", reranker=MockReranker())
        with pytest.raises(ValueError):
            db.query_cursor("quick fox", reranker_config=MOCK_CFG)
        db.close()

    async def test_async_cursor_matches(self, tmp_path):
        db = _db(tmp_path, reranker_config=MOCK_CFG)
        _seed(db)
        cursor = await db.query_cursor_async("quick fox")
        batch = await cursor.fetch_all_async()
        assert batch
        with pytest.raises(ValueError):
            await db.query_cursor_async("quick fox", reranker_config=MOCK_CFG)
        db.close()


@pytest.mark.integration
class TestMultiColumnAppliesOnce:
    def test_merged_pool_reranked_exactly_once(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        results = db.query_multi_column("quick fox", k=2, return_type="chunks")
        assert results
        # One rerank of the merged pool; the content leg (reranker=False) adds none.
        assert counting_provider.rerank_calls == 1
        db.close()

    def test_false_disables_for_multi_column(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        db.query_multi_column("quick fox", k=2, return_type="chunks", reranker=False)
        assert counting_provider.rerank_calls == 0
        db.close()

    async def test_async_matches(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        await db.query_multi_column_async("quick fox", k=2, return_type="chunks")
        assert counting_provider.rerank_calls == 1
        db.close()


@pytest.mark.integration
class TestBuilderInteraction:
    def test_builder_without_rerank_inherits_default(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        results = db.query_builder().search("quick fox").limit(2).execute()
        assert results
        assert counting_provider.rerank_calls == 1
        db.close()

    def test_builder_rerank_suppresses_default(self, tmp_path, counting_provider):
        db = _db(tmp_path, reranker_config={"provider": "counting"})
        _seed(db)
        results = db.query_builder().search("quick fox").limit(2).rerank_by_model("mock").execute()
        assert results
        # The builder's own cross_encoder ran instead of the persisted default.
        assert counting_provider.rerank_calls == 0
        db.close()
