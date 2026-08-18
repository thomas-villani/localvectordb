"""Tests for db.diagnose() and the post-ingest truncation warning.

The measurement rules under test come straight from the retrieval study:

* Coverage is sum(min(t, cap)) / sum(t) -- never a mean-based estimate, which
  said "barely truncated" while 44.9% of text was being discarded.
* Token counts are exact only when they come from the encoder's own tokenizer;
  everything else is labelled estimated.
* The ingest warning fires on substantial loss only (46% of chunks truncated
  measured harmless), and once per instance.
"""

import tempfile
import warnings

import pytest

from localvectordb.database import DiagnoseReport, LocalVectorDB
from localvectordb.database._diagnose import (
    _COVERAGE_WARN_THRESHOLD,
    _coverage_from_counts,
    _median,
    _resolve_token_counter,
    _strided_sample,
)
from localvectordb.embeddings import MockEmbeddings

pytestmark = pytest.mark.unit


MARKDOWN_DOC = "\n\n".join(
    f"# Heading {i}\n\nThis is the body text of section number {i}. "
    "It contains several sentences about a distinct topic. "
    "Each sentence adds a little more content to the section."
    for i in range(6)
)


@pytest.fixture
def hier_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = LocalVectorDB(
            name="diagnose_test",
            base_path=tmpdir,
            embedding_provider="mock",
            embedding_model="mock",
            hierarchical_embeddings=True,
            chunk_size=30,
            chunk_overlap=0,
            enable_fts=True,
        )
        yield db
        db.close()


class TestPureHelpers:
    def test_coverage_all_within_cap(self):
        coverage, truncated = _coverage_from_counts([10, 20, 30], cap=50)
        assert coverage == 1.0
        assert truncated == 0.0

    def test_coverage_counts_text_share_not_chunk_share(self):
        # One chunk of 100 tokens against a cap of 50: half the TEXT is lost
        # even though only a third of the CHUNKS are truncated alongside two
        # short ones. The mean-based estimate this replaces gets this wrong.
        coverage, truncated = _coverage_from_counts([100, 25, 25], cap=50)
        assert coverage == (50 + 25 + 25) / 150
        assert truncated == pytest.approx(1 / 3)

    def test_coverage_empty(self):
        assert _coverage_from_counts([], cap=10) == (1.0, 0.0)

    def test_median(self):
        assert _median([]) is None
        assert _median([5]) == 5
        assert _median([1, 9, 5]) == 5

    def test_strided_sample_deterministic_and_capped(self):
        ids = list(range(100))
        first = _strided_sample(ids, 10)
        assert first == _strided_sample(ids, 10)
        assert len(first) <= 11
        assert first[0] == 0
        assert _strided_sample(ids, 200) == ids

    def test_mock_provider_counts_are_estimated(self):
        counter, exact, source = _resolve_token_counter(MockEmbeddings("mock", dimension=8))
        assert exact is False
        assert "estimated" in source
        assert counter("some text here") > 0


class TestDiagnose:
    def test_report_shape_on_hierarchical_db(self, hier_db):
        hier_db.upsert(documents=[MARKDOWN_DOC], ids=["doc1"])
        report = hier_db.diagnose()

        assert report.documents == 1
        assert report.chunks > 0
        assert report.sections > 0
        # MockEmbeddings reports no context, so coverage is honestly unmeasurable
        # rather than silently wrong.
        assert report.context_tokens is None
        assert report.chunk_coverage is None
        assert report.tokens_exact is False
        assert report.median_chunk_tokens is not None
        assert report.mean_chunks_per_document is not None
        assert report.chunkless_section_share is not None
        assert 0.0 <= report.chunkless_section_share <= 1.0
        assert report.fts_status.get("chunks_fts") == "ok"
        assert report.fts_status.get("documents_fts") == "ok"
        assert report.fts_status.get("sections_fts") == "ok"

        summary = report.summary
        assert "diagnose_test" in summary
        assert "Encoder coverage" in summary
        assert "Fanout" in summary

    def test_coverage_matches_stored_token_aggregate(self, hier_db, monkeypatch):
        # A known context cap turns the coverage line on. The estimated path
        # reads the stored per-chunk counts, so the report must agree with the
        # same aggregate computed by hand.
        monkeypatch.setattr(MockEmbeddings, "max_input_tokens", 15, raising=False)
        hier_db.upsert(documents=[MARKDOWN_DOC], ids=["doc1"])
        report = hier_db.diagnose()

        assert report.context_tokens == 15
        with hier_db.connection_pool.get_connection() as conn:
            rows = conn.execute("SELECT tokens FROM chunks").fetchall()
        counts = [int(r["tokens"]) for r in rows]
        expected_coverage, expected_truncated = _coverage_from_counts(counts, 15)
        assert report.chunk_coverage == pytest.approx(expected_coverage)
        assert report.truncated_chunk_share == pytest.approx(expected_truncated)
        assert report.chunks_measured == len(counts)

        if report.chunk_coverage < _COVERAGE_WARN_THRESHOLD:
            assert any("never enters any vector" in w for w in report.warnings)

    def test_sections_measured_against_window(self, hier_db, monkeypatch):
        monkeypatch.setattr(MockEmbeddings, "max_input_tokens", 15, raising=False)
        hier_db.upsert(documents=[MARKDOWN_DOC], ids=["doc1"])
        report = hier_db.diagnose()

        assert report.sections_measured == report.sections
        assert report.median_section_tokens is not None
        assert report.sections_over_context_share is not None
        # Every section body above is far longer than 15 tokens.
        assert report.sections_over_context_share > 0.5

    def test_diagnose_async_delegates(self, hier_db):
        import asyncio

        hier_db.upsert(documents=[MARKDOWN_DOC], ids=["doc1"])
        report = asyncio.run(hier_db.diagnose_async())
        assert isinstance(report, DiagnoseReport)
        assert report.chunks > 0


class TestTruncationWarning:
    def _long_doc(self, sentences=120):
        return " ".join(
            f"Sentence number {i} carries a reasonable amount of ordinary prose content." for i in range(sentences)
        )

    def test_warns_once_on_substantial_loss(self, monkeypatch):
        # Cap of 5 tokens against ~15-token chunks: coverage far below the
        # threshold, and enough chunks to clear the minimum-corpus guard.
        monkeypatch.setattr(MockEmbeddings, "max_input_tokens", 5, raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="trunc_warn",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                chunk_size=15,
                chunk_overlap=0,
            )
            try:
                with pytest.warns(UserWarning, match="never enters any vector"):
                    db.upsert(documents=[self._long_doc()], ids=["doc1"])

                # Once per instance: a second ingest stays quiet.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    db.upsert(documents=[self._long_doc()], ids=["doc2"])
                assert not [w for w in caught if "never enters any vector" in str(w.message)]
            finally:
                db.close()

    def test_no_warning_without_context(self):
        # MockEmbeddings reports no context by default: nothing to measure
        # against, so the check must stay silent rather than guess.
        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="trunc_none",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                chunk_size=15,
                chunk_overlap=0,
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    db.upsert(documents=[self._long_doc()], ids=["doc1"])
                assert not [w for w in caught if "never enters any vector" in str(w.message)]
            finally:
                db.close()

    def test_no_warning_below_minimum_corpus(self, monkeypatch):
        monkeypatch.setattr(MockEmbeddings, "max_input_tokens", 5, raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="trunc_small",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                chunk_size=15,
                chunk_overlap=0,
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    db.upsert(documents=[self._long_doc(sentences=3)], ids=["doc1"])
                assert not [w for w in caught if "never enters any vector" in str(w.message)]
            finally:
                db.close()
