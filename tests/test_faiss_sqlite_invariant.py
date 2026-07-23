"""
Dual-store integrity tests for the SQLite <-> FAISS pair.

LocalVectorDB splits every document across two stores: rows in SQLite and
vectors in FAISS, joined on ``chunks.faiss_id``. The invariant that makes that
join sound is:

    index.ntotal == COUNT(chunks WHERE faiss_id IS NOT NULL)
                  + COUNT(column_embeddings)

...and, across the union of those two tables, every ``faiss_id`` is unique.

Nothing asserted that before, which is how a duplicate-id bug shipped: FAISS ids
were allocated from ``index.ntotal``, but ``remove_ids`` *decrements* ``ntotal``,
so any delete or replacing-upsert re-issued ids that were still live. A
duplicated id makes the hydration join (``WHERE c.faiss_id IN (...)``) return two
chunk rows for one vector, and both documents get scored from that single
vector's distance.

These tests must run against a **real** FAISS index. The ``mock_faiss_index``
fixture in ``conftest.py`` returns ``np.random`` distances and ignores the query
vector entirely, so it cannot observe any of this. The ``"mock"`` embedding
provider is fine and is used deliberately: it is deterministic (it seeds
``np.random`` from a hash of the text), so embedding the same string twice yields
the same vector, which is what lets us assert exact-match retrieval.
"""

import contextlib
import sqlite3
from unittest import mock

import numpy as np
import pytest

from localvectordb.database import LocalVectorDB

pytestmark = pytest.mark.integration


# Index types whose base index supports remove_ids. Verified against faiss
# 1.14.2 through the IndexIDMap2 wrapper. IndexHNSWFlat is deliberately absent:
# its remove_ids raises, so ntotal never decrements and the duplicate-id bug is
# unreachable there. It has its own failure mode, covered separately below.
DELETING_INDEX_TYPES = ["IndexFlatL2", "IndexFlatIP", "IndexLSH"]

# Index types that rank correctly, and so can carry the retrieval assertions.
# IndexFlatIP is included as of T1.4: its metric is now detected correctly and its
# vectors are normalized at the boundary (see TestInnerProductScoring), so it no
# longer inverts the ranking.
RANKING_INDEX_TYPES = ["IndexFlatL2", "IndexFlatIP", "IndexLSH"]


def make_db(tmp_path, name="invariant_test", **kwargs):
    """Real FAISS, real SQLite, deterministic embeddings, temp dir."""
    kwargs.setdefault("embedding_provider", "mock")
    kwargs.setdefault("embedding_model", "mock-model")
    return LocalVectorDB(name=name, base_path=str(tmp_path), **kwargs)


def sqlite_faiss_ids(db):
    """Every faiss_id SQLite believes is live, across both tables sharing the id space."""
    with db.connection_pool.get_connection() as conn:
        chunk_ids = [r[0] for r in conn.execute("SELECT faiss_id FROM chunks WHERE faiss_id IS NOT NULL").fetchall()]
        try:
            col_ids = [r[0] for r in conn.execute("SELECT faiss_id FROM column_embeddings").fetchall()]
        except sqlite3.OperationalError:  # pragma: no cover - table always exists in current schema
            col_ids = []
    return chunk_ids + col_ids


def assert_invariant(db, context=""):
    """The dual-store invariant, asserted after a mutating op."""
    ids = sqlite_faiss_ids(db)

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate faiss_id(s) {duplicates} after {context}; rows={sorted(ids)}"

    assert db.index.ntotal == len(
        ids
    ), f"ntotal ({db.index.ntotal}) != rows referencing a vector ({len(ids)}) after {context}"


def live_faiss_ids(db):
    """External ids actually present in the FAISS index."""
    import faiss

    if db.index.ntotal == 0:
        return []
    return sorted(int(i) for i in faiss.vector_to_array(db.index.id_map))


class TestDualStoreInvariant:
    """The invariant must survive an arbitrary interleaving of mutating operations."""

    @pytest.mark.parametrize("index_type", DELETING_INDEX_TYPES)
    def test_invariant_holds_across_mutation_interleaving(self, tmp_path, index_type):
        db = make_db(tmp_path, faiss_index_type=index_type)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."], ids=["docA", "docB", "docC"]
            )
            assert_invariant(db, "initial upsert")

            # A replacing upsert: this is the operation that re-issued ids.
            db.upsert(["alpha alpha alpha updated."], ids=["docA"])
            assert_invariant(db, "replacing upsert of docA")

            db.delete("docC")
            assert_invariant(db, "delete of docC")

            db.upsert(["delta delta delta."], ids=["docD"])
            assert_invariant(db, "insert of docD after a delete")

            db.upsert(["bravo bravo bravo revised."], ids=["docB"])
            assert_invariant(db, "replacing upsert of docB after a delete")

            db.delete("docA")
            db.upsert(["echo echo echo."], ids=["docE"])
            assert_invariant(db, "delete then insert")
        finally:
            db.close()

    def test_metadata_update_rollback_leaves_no_dangling_vectors(self, tmp_path):
        """H4: a mid-transaction failure while re-embedding a metadata field must not
        have already removed the old field vector. The rollback restores the old
        rows, so their vector must still be in the index -- otherwise it dangles
        (recoverable only by re-embedding). The removal is deferred to post-commit.
        """
        from localvectordb.core import MetadataField, MetadataFieldType

        schema = {"title": MetadataField(type=MetadataFieldType.TEXT, embedding_enabled=True)}
        db = make_db(tmp_path, metadata_schema=schema)
        try:
            db.upsert(["body text here"], [{"title": "original title"}], ids=["docA"])
            assert_invariant(db, "initial upsert")
            old_ids = set(sqlite_faiss_ids(db))
            assert old_ids, "expected the seed doc to have chunk + metadata vectors"

            # Fail after the (deferred) removal of the old metadata vector but
            # before the new one is generated/committed, forcing the rollback path.
            with mock.patch.object(db, "_generate_metadata_embeddings", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    db.update("docA", metadata={"title": "a completely new title"})

            # No dangling rows: every faiss_id SQLite references is present in the index.
            dangling = set(sqlite_faiss_ids(db)) - set(live_faiss_ids(db))
            assert not dangling, f"rollback left dangling rows: {sorted(dangling)}"
            assert_invariant(db, "rolled-back metadata update")
            # The original metadata (and its vector) survived intact.
            assert db.get("docA").metadata["title"] == "original title"
            assert set(sqlite_faiss_ids(db)) == old_ids
        finally:
            db.close()

    def test_no_duplicate_ids_after_replacing_upsert(self, tmp_path):
        """The minimal repro: replacing upsert on a 3-doc database, no explicit delete."""
        db = make_db(tmp_path)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."], ids=["docA", "docB", "docC"]
            )
            before = sorted(sqlite_faiss_ids(db))
            assert len(set(before)) == len(before), f"already broken before the upsert: {before}"

            db.upsert(["alpha alpha alpha updated."], ids=["docA"])

            after = sorted(sqlite_faiss_ids(db))
            duplicates = sorted({i for i in after if after.count(i) > 1})
            assert not duplicates, f"replacing upsert re-issued faiss_id(s) {duplicates}; ids={after}"
        finally:
            db.close()

    def test_sqlite_ids_match_faiss_ids(self, tmp_path):
        """Every id SQLite references exists in FAISS, and vice versa: no orphans, no dangling rows."""
        db = make_db(tmp_path)
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            db.upsert(["alpha alpha alpha updated."], ids=["docA"])
            db.delete("docB")

            assert sorted(set(sqlite_faiss_ids(db))) == live_faiss_ids(
                db
            ), "SQLite and FAISS disagree about which vectors are live"
        finally:
            db.close()


class TestRetrievalCorrectnessAfterMutation:
    """
    The user-visible consequence of a duplicated id.

    Replacing docA frees its old id and re-issues docC's still-live id. Both rows
    then hydrate from one vector, and because the hydration map is keyed by
    faiss_id (``faiss_id_to_row[row["faiss_id"]] = row``), the later row wins:

        query "charlie charlie charlie." -> [("docA", 1.0), ("docB", 0.37)]

    docA -- whose text is unrelated -- is returned with a perfect score, and docC,
    which actually contains the queried text, does not appear at all.
    """

    @pytest.mark.parametrize("index_type", RANKING_INDEX_TYPES)
    def test_document_is_retrievable_by_its_own_text_after_a_sibling_is_replaced(self, tmp_path, index_type):
        db = make_db(tmp_path, faiss_index_type=index_type)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."],
                ids=["docA", "docB", "docC"],
            )
            db.upsert(["alpha alpha alpha updated."], ids=["docA"])

            results = db.query("charlie charlie charlie.", search_type="vector", k=3)
            ranked = [(r.id, round(r.score, 4)) for r in results]

            assert results, "query returned nothing"
            assert results[0].id == "docC", f"querying docC's own text ranked '{results[0].id}' first; got {ranked}"
        finally:
            db.close()

    @pytest.mark.parametrize("index_type", RANKING_INDEX_TYPES)
    def test_unrelated_document_is_not_scored_from_another_documents_vector(self, tmp_path, index_type):
        """docA's text shares nothing with the query; it must not be returned at all, let alone at 1.0."""
        db = make_db(tmp_path, faiss_index_type=index_type)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."],
                ids=["docA", "docB", "docC"],
            )
            db.upsert(["alpha alpha alpha updated."], ids=["docA"])

            results = db.query("charlie charlie charlie.", search_type="vector", k=3)
            ranked = [(r.id, round(r.score, 4)) for r in results]
            top = {r.id: r.score for r in results}

            assert (
                top.get("docA", 0.0) < 0.99
            ), f"docA was scored from docC's vector and returned at {top.get('docA')}; got {ranked}"
        finally:
            db.close()

    def test_untouched_document_still_ranks_first_for_its_own_text(self, tmp_path):
        """The sibling that keeps its own vector must be unaffected."""
        db = make_db(tmp_path)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."],
                ids=["docA", "docB", "docC"],
            )
            db.upsert(["alpha alpha alpha updated."], ids=["docA"])

            results = db.query("bravo bravo bravo.", search_type="vector", k=3)
            assert results and results[0].id == "docB", f"got {[(r.id, round(r.score, 4)) for r in results]}"
        finally:
            db.close()

    def test_deleted_document_never_resurfaces(self, tmp_path):
        """A deleted document's vector must not be reachable through another document's id."""
        db = make_db(tmp_path)
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."], ids=["docA", "docB", "docC"]
            )
            db.delete("docB")
            db.upsert(["delta delta delta."], ids=["docD"])

            for r in db.query("bravo bravo bravo.", search_type="vector", k=5):
                assert r.id != "docB", "deleted document resurfaced in search results"
        finally:
            db.close()


class TestInnerProductScoring:
    """
    T1.4 regression: an ``IndexFlatIP`` index must rank correctly.

    Two defects in the same scoring boundary conspired here, and neither was an
    id-integrity problem (they reproduce on a pristine database with ids [0, 1, 2]
    and no mutation at all):

    1. **Metric misdetection.** ``_detect_faiss_metric_type`` unwrapped the
       ``IndexIDMap2`` via ``.index``, which returns a downcast-less generic
       ``faiss.Index`` whose class name is just ``"Index"``. That matched neither
       "IP" nor "L2", so it fell through to the L2 default and scored the inner
       products with ``1/(1+ip)`` -- which *inverts* the ranking, handing the top
       score to the least similar document. Fixed by reading ``metric_type``.

    2. **No normalization.** ``(ip + 1) / 2`` assumes unit vectors, but nothing
       called ``faiss.normalize_L2`` and providers disagree on their ``normalize``
       default. Fixed by normalizing at the write and query boundary for IP
       indices. (``MockEmbeddings`` already returns unit vectors, so the direct
       proof of this half is ``test_normalize_for_index_*`` below, which feeds a
       genuinely unnormalized vector through the helper.)
    """

    def test_ip_metric_type_is_detected(self, tmp_path):
        ip = make_db(tmp_path, name="ip_metric", faiss_index_type="IndexFlatIP")
        l2 = make_db(tmp_path, name="l2_metric", faiss_index_type="IndexFlatL2")
        try:
            assert ip._get_faiss_metric_type() == "IP"
            assert l2._get_faiss_metric_type() == "L2"
        finally:
            ip.close()
            l2.close()

    def test_ip_index_ranks_the_matching_document_first(self, tmp_path):
        db = make_db(tmp_path, name="ip_ranking", faiss_index_type="IndexFlatIP")
        try:
            db.upsert(
                ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."],
                ids=["docA", "docB", "docC"],
            )
            assert_invariant(db, "pristine upsert")

            results = db.query("charlie charlie charlie.", search_type="vector", k=3)
            assert results[0].id == "docC", f"got {[(r.id, round(r.score, 4)) for r in results]}"
        finally:
            db.close()

    def test_normalize_for_index_makes_ip_vectors_unit_norm(self, tmp_path):
        # Direct test of the boundary helper against a genuinely unnormalized
        # vector (mock embeddings are already unit-norm, so this is the only way
        # to prove the normalization half actually does something).
        db = make_db(tmp_path, name="ip_normhelper", faiss_index_type="IndexFlatIP")
        try:
            raw = np.array([[3.0, 4.0] + [0.0] * (db.embedding_dimension - 2)], dtype=np.float32)
            out = db._normalize_for_index(raw, db.index)
            assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-6)
            assert raw[0, 0] == pytest.approx(3.0), "helper must not mutate the caller's buffer"
        finally:
            db.close()

    def test_normalize_for_index_is_a_noop_on_l2(self, tmp_path):
        # The IP-gated normalization must not touch L2, or L2 rankings (and the
        # retrieval baseline, which runs on IndexFlatL2) would silently shift.
        db = make_db(tmp_path, name="l2_normhelper", faiss_index_type="IndexFlatL2")
        try:
            raw = np.array([[3.0, 4.0] + [0.0] * (db.embedding_dimension - 2)], dtype=np.float32)
            out = db._normalize_for_index(raw, db.index)
            assert np.linalg.norm(out[0]) == pytest.approx(5.0, abs=1e-6)
        finally:
            db.close()


class TestEmptiedIndexIsPersisted:
    """Deleting every document must not leave a stale index file on disk."""

    def test_index_file_reflects_empty_database(self, tmp_path):
        db = make_db(tmp_path, name="emptied")
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            db.save()
            index_path = db.index_path

            db.delete("docA")
            db.delete("docB")
            db.save()
        finally:
            db.close()

        reopened = LocalVectorDB(
            name="emptied", base_path=str(tmp_path), embedding_provider="mock", embedding_model="mock-model"
        )
        try:
            assert reopened.index.ntotal == 0, (
                f"reopened database has {reopened.index.ntotal} vectors but no documents; "
                f"stale index at {index_path}"
            )
            assert_invariant(reopened, "reopen after deleting every document")
        finally:
            reopened.close()


class TestCounterSurvivesReopen:
    """Ids must never be re-issued across a close/reopen cycle."""

    def test_ids_are_not_reused_after_reopen(self, tmp_path):
        db = make_db(tmp_path, name="reopened")
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            first_ids = set(sqlite_faiss_ids(db))
            db.delete("docA")
            db.save()
        finally:
            db.close()

        db2 = LocalVectorDB(
            name="reopened", base_path=str(tmp_path), embedding_provider="mock", embedding_model="mock-model"
        )
        try:
            db2.upsert(["charlie charlie charlie."], ids=["docC"])
            assert_invariant(db2, "insert after reopen")

            after = sqlite_faiss_ids(db2)
            fresh = [i for i in after if i not in first_ids]
            assert fresh, "expected docC to receive a freshly allocated id"
            assert max(fresh) >= max(first_ids), (
                f"reopen re-issued a previously used id: allocated {fresh}, "
                f"but {sorted(first_ids)} were already used before close"
            )
        finally:
            db2.close()


class TestAtomicIndexPersistence:
    """A crash between serializing the index and renaming it must not destroy the old one."""

    def test_failed_save_leaves_previous_index_intact(self, tmp_path):
        db = make_db(tmp_path, name="atomic")
        try:
            # upsert() persists on its own, so the file on disk already holds these two.
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            good_bytes = db.index_path.read_bytes()

            # Diverge the in-RAM index from disk without going through upsert, so the
            # only thing that could write the change is the save() we are about to break.
            new_id = db._allocate_faiss_ids("main", 1)
            db.index.add_with_ids(np.random.rand(1, db.embedding_dimension).astype(np.float32), new_id)
            assert db.index.ntotal == 3

            with mock.patch("os.replace", side_effect=OSError("simulated crash before rename")):
                with pytest.raises(OSError):
                    db.save()

            assert db.index_path.read_bytes() == good_bytes, "a failed save corrupted the on-disk index"
            assert not list(tmp_path.glob("*.tmp")), "a failed save left a temp file behind"

            # Drop the RAM-only vector: close() saves, and would otherwise persist it.
            db.index.remove_ids(new_id)
        finally:
            with contextlib.suppress(Exception):
                db.close()

        reopened = LocalVectorDB(
            name="atomic",
            base_path=str(tmp_path),
            embedding_provider="mock",
            embedding_model="mock-model",
        )
        try:
            assert reopened.index.ntotal == 2, "the surviving index did not load"
        finally:
            reopened.close()

    def test_no_temp_file_is_left_behind_on_success(self, tmp_path):
        db = make_db(tmp_path, name="tidy")
        try:
            db.upsert(["alpha alpha alpha."], ids=["docA"])
            db.save()
            leftovers = list(tmp_path.glob("*.tmp"))
            assert not leftovers, f"temp files left behind: {leftovers}"
        finally:
            db.close()


class TestRepair:
    """Rebuild from SQLite, reassigning every id."""

    @staticmethod
    def _corrupt(db, victim_doc, stolen_id, freed_id):
        """Forge the exact damage the old ntotal-based allocator produced."""
        with db.connection_pool.get_connection() as conn:
            conn.execute("UPDATE chunks SET faiss_id = ? WHERE document_id = ?", (stolen_id, victim_doc))
            conn.commit()
        db.index.remove_ids(np.array([freed_id], dtype=np.int64))
        db.save()

    def test_corrupted_database_refuses_to_open_and_repair_restores_it(self, tmp_path):
        from localvectordb.database._repair import open_for_repair
        from localvectordb.exceptions import IndexIntegrityError

        db = make_db(tmp_path, name="corrupt")
        db.upsert(
            ["alpha alpha alpha.", "bravo bravo bravo.", "charlie charlie charlie."], ids=["docA", "docB", "docC"]
        )
        self._corrupt(db, victim_doc="docA", stolen_id=2, freed_id=0)
        db.close()

        with pytest.raises(IndexIntegrityError, match="duplicate FAISS id"):
            make_db(tmp_path, name="corrupt")

        repairer = open_for_repair("corrupt", str(tmp_path))
        try:
            dry = repairer.repair(dry_run=True)
            assert dry.duplicate_ids == [2]
            assert dry.dry_run

            report = repairer.repair()
            # docB's id is unique, so its vector is recovered from the index. docA and
            # docC share id 2 -- reconstruct() cannot say which vector is whose, so both
            # are re-embedded.
            assert report.reconstructed == 1, report.summary
            assert report.reembedded == 2, report.summary
        finally:
            repairer.close()

        db2 = make_db(tmp_path, name="corrupt")
        try:
            assert_invariant(db2, "repair")
            assert db2.query("charlie charlie charlie.", search_type="vector", k=1)[0].id == "docC"
            assert db2.query("alpha alpha alpha.", search_type="vector", k=1)[0].id == "docA"
        finally:
            db2.close()

    @pytest.mark.parametrize("index_type", DELETING_INDEX_TYPES + ["IndexHNSWFlat"])
    def test_healthy_repair_never_calls_the_embedding_provider(self, tmp_path, index_type):
        """A clean database compacts for free: every vector is reconstructed in place."""
        db = make_db(tmp_path, name="clean", faiss_index_type=index_type)
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])

            with mock.patch.object(
                type(db.embedding_provider), "embed_sync", side_effect=AssertionError("provider was called")
            ):
                report = db.repair()

            assert report.reembedded == 0
            assert report.reconstructed == 2
            assert report.base_index_type == index_type, "repair changed the index type"
            assert_invariant(db, "repair of a healthy database")
        finally:
            db.close()

    def test_repair_drops_orphan_vectors(self, tmp_path):
        db = make_db(tmp_path, name="orphans")
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            # A vector nobody owns: the residue of a crash after SQLite committed a delete.
            db.index.add_with_ids(
                np.random.rand(1, db.embedding_dimension).astype(np.float32), np.array([999], dtype=np.int64)
            )
            assert db.index.ntotal == 3

            report = db.repair()
            assert report.orphan_vectors == [999]
            assert report.dropped == 1
            assert_invariant(db, "repair with an orphan vector")
        finally:
            db.close()


class TestHNSWCannotSilentlyDrop:
    """
    IndexHNSWFlat cannot remove vectors: faiss raises. Today that exception is
    swallowed, so delete() reports success while the vector survives -- a
    different bug from the duplicate-id one, on a disjoint index type.

    A delete must not silently succeed.
    """

    def test_delete_on_hnsw_does_not_silently_orphan(self, tmp_path):
        db = make_db(tmp_path, name="hnsw", faiss_index_type="IndexHNSWFlat")
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            ntotal_before = db.index.ntotal

            try:
                db.delete("docA")
            except Exception:
                # Raising is the acceptable outcome: the caller learns the delete did not happen.
                return

            # If it did not raise, the vector must actually be gone.
            assert db.index.ntotal == ntotal_before - 1, (
                "delete() on an HNSW index reported success but left the vector in place "
                f"(ntotal {ntotal_before} -> {db.index.ntotal})"
            )
        finally:
            db.close()

    def test_append_only_still_works_on_hnsw(self, tmp_path):
        """Construction and pure-append use must keep working; benchmarks/tier1_ann.py sweeps HNSW."""
        db = make_db(tmp_path, name="hnsw_append", faiss_index_type="IndexHNSWFlat")
        try:
            db.upsert(["alpha alpha alpha.", "bravo bravo bravo."], ids=["docA", "docB"])
            assert db.index.ntotal == len(sqlite_faiss_ids(db))
            assert db.query("alpha alpha alpha.", search_type="vector", k=1)
        finally:
            db.close()
