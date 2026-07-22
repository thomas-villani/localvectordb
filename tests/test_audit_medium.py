"""Regression tests for the medium-tier pre-v0.1.0 audit fixes (M1-M14).

Each test pins the specific defect the audit found so a future refactor cannot
silently reintroduce it. Grouped by item; see AUDIT.md for the full write-ups.
"""

from unittest import mock

import numpy as np
import pytest

from localvectordb._filters import matches_metadata_filter
from localvectordb.core import QueryResult
from localvectordb.database import LocalVectorDB
from localvectordb.reranking import SentenceTransformersReranker
from localvectordb.section_detection import SectionDetector
from localvectordb_server._serializers import serialize_query_result


@pytest.fixture
def text_db(tmp_path):
    db = LocalVectorDB(
        name="med",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="x",
        enable_fts=True,
    )
    db.update_metadata_schema({"title": "text"})
    yield db
    db.close()


# ---------------------------------------------------------------------------
# M1: document_id must serialize for every result type, not just chunks.
# ---------------------------------------------------------------------------


class TestM1DocumentIdSerialization:
    @pytest.mark.parametrize("rtype", ["chunk", "section", "context", "enriched"])
    def test_document_id_emitted_for_all_types(self, rtype):
        result = QueryResult(id="r1", score=0.9, type=rtype, content="c", document_id="parent")
        assert serialize_query_result(result)["document_id"] == "parent"

    def test_document_id_absent_when_none(self):
        result = QueryResult(id="d1", score=0.9, type="document", content="c", document_id=None)
        assert "document_id" not in serialize_query_result(result)


# ---------------------------------------------------------------------------
# M4: SentenceTransformersReranker must apply the sigmoid exactly once.
# ---------------------------------------------------------------------------


class TestM4SingleSigmoid:
    def test_no_double_sigmoid(self):
        logits = np.array([2.0, 0.0, -2.0])
        expected = 1.0 / (1.0 + np.exp(-logits))  # one sigmoid
        double = 1.0 / (1.0 + np.exp(-expected))  # the buggy compression

        reranker = SentenceTransformersReranker(model="dummy")
        fake_ce = mock.MagicMock()
        fake_ce.predict.return_value = logits
        with mock.patch.object(reranker, "_load_model", return_value=fake_ce):
            results = [QueryResult(id=f"r{i}", score=0.0, type="chunk", content=f"c{i}") for i in range(3)]
            ranked = reranker.rerank("q", results)

        by_id = {r.id: r.score for r in ranked}
        assert by_id["r0"] == pytest.approx(expected[0], abs=1e-6)
        assert by_id["r1"] == pytest.approx(expected[1], abs=1e-6)
        # The double-sigmoid bug would have compressed r0 to ~0.707, not ~0.88.
        assert abs(by_id["r0"] - double[0]) > 0.1


# ---------------------------------------------------------------------------
# M5: large-k / rerank over-fetch must not be truncated by the 100 cap.
# ---------------------------------------------------------------------------


class TestM5NoHiddenCap:
    def test_hybrid_k_above_100_returns_more_than_100(self, text_db):
        n = 150
        ids = [f"d{i}" for i in range(n)]
        text_db.upsert([f"apple document number {i}" for i in range(n)], ids=ids)
        results = text_db.query("apple", search_type="hybrid", k=130, return_type="documents")
        assert len(results) > 100


# ---------------------------------------------------------------------------
# M6: multi-column similarity scores must stay in [0, 1] on non-unit vectors.
# ---------------------------------------------------------------------------


class TestM6BoundedColumnScores:
    def test_scores_bounded_for_unnormalized_vectors(self, text_db):
        # Deliberately large-magnitude (non-unit) vectors: a raw dot product would
        # blow past [0, 1]; cosine normalization keeps the mapped score bounded.
        rng = np.random.default_rng(0)
        query = rng.normal(size=64) * 10.0
        fields = rng.normal(size=(20, 64)) * 10.0
        scores = text_db._calculate_embedding_similarities(query, fields)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


# ---------------------------------------------------------------------------
# M7: SQL pushdown (filter) must match the Python matcher for string ops.
# ---------------------------------------------------------------------------


class TestM7FilterStringParity:
    def _seed(self, db):
        docs = {
            "d1": {"title": "Apple Pie"},
            "d2": {"title": "apple pie"},
            "d3": {"title": "pineapple"},
            "d4": {"title": "grape%fruit"},
            "d5": {"title": "Banana"},
            "d6": {},
        }
        db.upsert([f"content {i}" for i in docs], ids=list(docs), metadata=list(docs.values()))
        return docs

    @pytest.mark.parametrize(
        "flt",
        [
            {"title": {"$contains": "apple"}},
            {"title": {"$contains": "Apple"}},
            {"title": {"$ilike": "apple"}},
            {"title": {"$like": "apple"}},
            {"title": {"$startswith": "apple"}},
            {"title": {"$startswith": "Apple"}},
            {"title": {"$endswith": "pie"}},
            {"title": {"$endswith": "Pie"}},
            {"title": {"$contains": "%"}},
            {"title": {"$like": "grape%fruit"}},
            {"title": {"$not_contains": "apple"}},
            {"title": {"$endswith": ""}},
        ],
    )
    def test_sql_matches_python(self, text_db, flt):
        docs = self._seed(text_db)
        sql_ids = {d.id for d in text_db.filter(flt)}
        py_ids = {i for i, m in docs.items() if matches_metadata_filter(m, flt)}
        assert sql_ids == py_ids


# ---------------------------------------------------------------------------
# M11: section end_line must not overshoot into the next header's line.
# ---------------------------------------------------------------------------


class TestM11SectionEndLine:
    def test_end_line_stops_at_section_content(self):
        text = "# A\nbody line\n\n# B\nlast\n"
        # lines: 1:# A  2:body line  3:(blank)  4:# B  5:last
        secs = SectionDetector().detect_sections(text)
        by_heading = {s.heading: s for s in secs}
        assert (by_heading["A"].start_line, by_heading["A"].end_line) == (1, 2)
        assert (by_heading["B"].start_line, by_heading["B"].end_line) == (4, 5)

    def test_preamble_end_line(self):
        text = "intro\n\n# H\ntext\n"
        secs = SectionDetector().detect_sections(text)
        preamble = next(s for s in secs if s.heading is None)
        assert preamble.start_line == 1
        assert preamble.end_line == 1


# ---------------------------------------------------------------------------
# M2: the server must not cap query params local query() accepts.
# ---------------------------------------------------------------------------


class TestM2NoServerCaps:
    def test_query_body_accepts_large_values(self):
        from localvectordb_server.routers._models import QueryBody

        # All of these 422'd before the fix while succeeding locally.
        body = QueryBody(query="q", k=5000, rerank_k=800, context_window=100, context_unit="chunks")
        assert body.k == 5000
        assert body.rerank_k == 800
        assert body.context_window == 100

    def test_query_body_still_rejects_nonsense_floors(self):
        from pydantic import ValidationError

        from localvectordb_server.routers._models import QueryBody

        with pytest.raises(ValidationError):
            QueryBody(query="q", k=0)


# ---------------------------------------------------------------------------
# M12: server.max_request_size must be enforced (413 on oversized body).
# ---------------------------------------------------------------------------


class TestM12RequestSizeLimit:
    def _client(self, max_bytes):
        from types import SimpleNamespace

        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from localvectordb_server.app import RequestSizeLimitMiddleware

        app = FastAPI()
        config = SimpleNamespace(server=SimpleNamespace(max_request_size=max_bytes))
        app.add_middleware(RequestSizeLimitMiddleware, config=config)

        @app.post("/echo")
        def echo(payload: dict):
            return {"ok": True}

        return TestClient(app)

    def test_body_within_limit_passes(self):
        client = self._client(10_000)
        assert client.post("/echo", json={"a": "x"}).status_code == 200

    def test_oversized_body_rejected_413(self):
        client = self._client(100)
        resp = client.post("/echo", json={"a": "x" * 500})
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "REQUEST_TOO_LARGE"
