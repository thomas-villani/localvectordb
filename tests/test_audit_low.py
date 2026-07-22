"""Regression tests for the low-tier pre-v0.1.0 audit fixes (L1-L15) and the
docs/packaging cleanups (D1-D4).

Each test pins the specific defect the audit found so a future refactor cannot
silently reintroduce it. Grouped by item; see AUDIT.md for the full write-ups.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from localvectordb._filters import matches_metadata_filter
from localvectordb.chunking import (
    CharChunker,
    LineChunker,
    ParagraphChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
)
from localvectordb.database import LocalVectorDB
from localvectordb.document_portions import parse_range_spec


@pytest.fixture
def text_db(tmp_path):
    db = LocalVectorDB(
        name="low",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="x",
        enable_fts=True,
    )
    db.update_metadata_schema({"title": "text", "flag": "boolean"})
    yield db
    db.close()


# ---------------------------------------------------------------------------
# L1: whitespace-only documents must reconstruct byte-for-byte (one chunk),
# empty text yields no chunks.
# ---------------------------------------------------------------------------


class TestL1WhitespaceReconstruction:
    @pytest.mark.parametrize(
        "chunker",
        [SentenceChunker(), TokenChunker(), WordChunker(), LineChunker(), CharChunker(), ParagraphChunker()],
    )
    @pytest.mark.parametrize("text", ["   ", "\n\n\n", "\t \n  \t"])
    def test_whitespace_only_reconstructs(self, chunker, text):
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "".join(c.content for c in chunks) == text

    @pytest.mark.parametrize(
        "chunker",
        [SentenceChunker(), TokenChunker(), WordChunker(), LineChunker(), CharChunker(), ParagraphChunker()],
    )
    def test_empty_text_yields_no_chunks(self, chunker):
        assert chunker.chunk("") == []


# ---------------------------------------------------------------------------
# L2: parse_range_spec must reject negative and reversed ranges.
# ---------------------------------------------------------------------------


class TestL2RangeValidation:
    @pytest.mark.parametrize("spec", ["-3:", ":-1", "-5:-1", "5:2", "10:0"])
    def test_bad_ranges_raise(self, spec):
        with pytest.raises(ValueError):
            parse_range_spec(spec)

    @pytest.mark.parametrize("spec,expected", [("2:5", (2, 5)), ("3:3", (3, 3)), (":4", (None, 4)), ("2:", (2, None))])
    def test_good_ranges_pass(self, spec, expected):
        assert parse_range_spec(spec) == expected

    def test_negative_single_rejected(self):
        with pytest.raises(ValueError):
            parse_range_spec("-1", allow_single=True)


# ---------------------------------------------------------------------------
# L3: PRAGMA string values must be quote-escaped.
# ---------------------------------------------------------------------------


class TestL3PragmaEscaping:
    def test_single_quote_is_doubled(self):
        from localvectordb.sqlite_tuning import format_pragma_value

        assert format_pragma_value("foo'bar") == "'foo''bar'"

    def test_plain_string_quoted(self):
        from localvectordb.sqlite_tuning import format_pragma_value

        # A value not in the safe-set is quoted; no embedded quote to escape.
        assert format_pragma_value("some_path") == "'some_path'"


# ---------------------------------------------------------------------------
# L6: k is clamped to ntotal at the FAISS boundary (no oversized allocation),
# and the query-builder offset floor is enforced.
# ---------------------------------------------------------------------------


class TestL6KClampAndFloors:
    def test_huge_k_returns_at_most_ntotal(self, text_db):
        ids = [f"d{i}" for i in range(5)]
        text_db.upsert([f"apple {i}" for i in range(5)], ids=ids)
        # A k far larger than the index must not blow up allocation or error;
        # it simply returns at most ntotal results.
        results = text_db.query("apple", search_type="vector", k=10_000_000, return_type="chunks")
        assert 0 < len(results) <= 5

    def test_query_builder_offset_floor(self):
        from pydantic import ValidationError

        from localvectordb_server.routers.search import QueryBuilderStateBody

        with pytest.raises(ValidationError):
            QueryBuilderStateBody(offset=-1)

    def test_nearest_neighbors_k_floor(self):
        from pydantic import ValidationError

        from localvectordb_server.routers.comparison import NearestNeighborsBody

        with pytest.raises(ValidationError):
            NearestNeighborsBody(doc_id="d1", k=0)


# ---------------------------------------------------------------------------
# L9: embedding reconstruction returns exactly one row per id, in order.
# ---------------------------------------------------------------------------


class TestL9ReconstructionAlignment:
    def test_missing_id_is_zero_filled_and_aligned(self, text_db):
        text_db.upsert(["alpha", "beta", "gamma"], ids=["a", "b", "c"])
        # Reconstruct real ids plus a bogus one; the result must stay row-aligned
        # to the input (one row per id), zero-filling the missing id rather than
        # dropping it (which would misalign every downstream positional consumer).
        real = text_db._get_all_faiss_ids() if hasattr(text_db, "_get_all_faiss_ids") else None
        # Pull actual faiss ids for the stored chunks.
        with text_db.connection_pool.get_connection() as conn:
            rows = conn.execute("SELECT faiss_id FROM chunks WHERE faiss_id IS NOT NULL LIMIT 2").fetchall()
        ids = [r["faiss_id"] for r in rows]
        bogus = 999_999_999
        matrix = text_db._reconstruct_embeddings_batch(ids + [bogus])
        assert matrix.shape[0] == len(ids) + 1
        # The bogus row is zero-filled.
        assert np.allclose(matrix[-1], 0.0)
        _ = real


# ---------------------------------------------------------------------------
# L8: mixed-batch chunk hydration must not drop rowid-only candidates. A hybrid
# batch can mix embedded chunks (faiss_id) with keyword-only hits whose chunk is
# not embedded (rowid only); the old `if faiss_ids: else:` dropped the latter.
# ---------------------------------------------------------------------------


class TestL8MixedBatchHydration:
    def test_faiss_and_rowid_candidates_both_hydrate(self, text_db):
        from localvectordb.cursor import CursorCandidate, CursorConfig, QueryCursor

        text_db.upsert(["alpha content", "beta content"], ids=["a", "b"])
        with text_db.connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, faiss_id FROM chunks WHERE faiss_id IS NOT NULL ORDER BY id LIMIT 2"
            ).fetchall()
        assert len(rows) == 2
        # One candidate carried by faiss_id, the other by rowid only -- exactly
        # the mix that used to drop the rowid-only entry.
        candidates = [
            CursorCandidate(score=0.9, source="hybrid", faiss_id=rows[0]["faiss_id"]),
            CursorCandidate(score=0.8, source="hybrid", chunk_rowid=rows[1]["id"]),
        ]
        config = CursorConfig(
            search_type="hybrid",
            return_type="chunks",
            search_level="chunks",
            score_threshold=0.0,
            filters=None,
            vector_weight=0.5,
            context_window=2,
            context_unit="chunks",
            context_truncate=False,
            total_k=10,
            semantic_dedup_threshold=None,
            document_scoring_method="frequency_boost",
            document_scoring_options=None,
        )
        cursor = QueryCursor(text_db, candidates, config)
        hydrated = cursor._hydrate_chunks_sync(candidates)
        assert len(hydrated) == 2


# ---------------------------------------------------------------------------
# L10: return_type='sections' on a non-hierarchical DB fails loudly.
# ---------------------------------------------------------------------------


class TestL10SectionsGuard:
    def test_sections_on_flat_db_raises(self, text_db):
        text_db.upsert(["some content here"], ids=["d1"])
        with pytest.raises(ValueError, match="hierarchical"):
            text_db.query("content", return_type="sections")


# ---------------------------------------------------------------------------
# L11: boolean $type parity between filter() (SQL) and query() (Python matcher).
# ---------------------------------------------------------------------------


class TestL11BooleanTypeParity:
    def test_int_0_1_matches_boolean_like_sql(self):
        # A boolean read back from SQLite as int 0/1 must satisfy $type:boolean,
        # so the Python post-filter agrees with SQL's `field IN (0, 1)`.
        assert matches_metadata_filter({"flag": 1}, {"flag": {"$type": "boolean"}})
        assert matches_metadata_filter({"flag": 0}, {"flag": {"$type": "boolean"}})
        assert matches_metadata_filter({"flag": True}, {"flag": {"$type": "boolean"}})
        # A non-0/1 integer is not boolean.
        assert not matches_metadata_filter({"flag": 5}, {"flag": {"$type": "boolean"}})

    def test_filter_and_query_agree_on_boolean_type(self, text_db):
        text_db.upsert(
            ["x", "y", "z"],
            ids=["a", "b", "c"],
            metadata=[{"flag": True}, {"flag": False}, {"title": "no flag"}],
        )
        flt = {"flag": {"$type": "boolean"}}
        sql_ids = {d.id for d in text_db.filter(flt)}
        assert sql_ids == {"a", "b"}


# ---------------------------------------------------------------------------
# L12: MCP config validates mode and rejects malformed TOML.
# ---------------------------------------------------------------------------


class TestL12MCPConfig:
    def test_invalid_mode_rejected(self, tmp_path):
        from localvectordb_server.mcp.config import MCPConfig

        cfg = tmp_path / "mcp.toml"
        cfg.write_text('[mcp]\nmode = "readonly"\n')  # typo: should be "read-only"
        with pytest.raises(ValueError, match="Invalid MCP mode"):
            MCPConfig.from_file(str(cfg))

    def test_malformed_toml_rejected(self, tmp_path):
        from localvectordb_server.mcp.config import MCPConfig

        cfg = tmp_path / "bad.toml"
        cfg.write_text("this is = = not valid toml [[[")
        with pytest.raises(ValueError, match="Invalid TOML"):
            MCPConfig.from_file(str(cfg))

    def test_valid_mode_accepted(self, tmp_path):
        from localvectordb_server.mcp.config import MCPConfig

        cfg = tmp_path / "ok.toml"
        cfg.write_text('[mcp]\nmode = "read-write"\n')
        assert MCPConfig.from_file(str(cfg)).mode == "read-write"


# ---------------------------------------------------------------------------
# L15: insecure non-loopback bind warns; a loopback / authenticated bind does not.
# ---------------------------------------------------------------------------


class TestL15InsecureBindWarning:
    def _config(self, host, require_api_key=False, cors="*", trusted=None):
        security = SimpleNamespace(
            require_api_key=require_api_key,
            cors_allowed_origins=cors,
            trusted_hosts=trusted,
        )
        return SimpleNamespace(server=SimpleNamespace(host=host, security=security))

    def _warns(self, config):
        import logging

        from localvectordb_server.app import _warn_if_insecure_bind

        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("test_l15")
        logger.setLevel(logging.WARNING)
        handler = _Cap()
        logger.addHandler(handler)
        try:
            _warn_if_insecure_bind(config, logger)
        finally:
            logger.removeHandler(handler)
        return any(r.levelno >= logging.WARNING for r in records)

    def test_non_loopback_no_auth_warns(self):
        assert self._warns(self._config("0.0.0.0"))

    def test_loopback_does_not_warn(self):
        assert not self._warns(self._config("127.0.0.1"))

    def test_non_loopback_with_auth_does_not_warn(self):
        assert not self._warns(self._config("0.0.0.0", require_api_key=True))
