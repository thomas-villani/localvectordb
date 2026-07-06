"""End-to-end test: local library flow with real embeddings.

Covers: database creation with a typed metadata schema, upsert/get/exists/
update/delete, vector/keyword/hybrid search with metadata filters, the
query builder, chunk retrieval and position tracking, document comparison and
nearest neighbours, backup/restore, and hierarchical (section-level) search.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_local.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, make_parser, temp_workdir
from make_fixtures import COOKING_MD, FINANCE_DOCX, ML_TEXT, SPACE_MD

GARDEN_TEXT = (
    "Raised garden beds warm up earlier in spring and drain better than open "
    "ground. Fill them with a mix of topsoil and compost, mulch generously to "
    "retain moisture, and rotate crops each season so soil-borne pests never "
    "settle in. Tomatoes, courgettes, and salad greens all thrive in beds "
    "between twenty and thirty centimetres deep."
)

DOCS = {
    "space": (
        SPACE_MD,
        {
            "title": "A Brief History of Space Exploration",
            "topic": "space",
            "rating": 4.8,
            "published": "2023-05-01",
            "tags": ["history", "space"],
        },
    ),
    "cooking": (
        COOKING_MD,
        {
            "title": "The Fundamentals of French Cooking",
            "topic": "food",
            "rating": 4.2,
            "published": "2022-11-15",
            "tags": ["cooking", "french"],
        },
    ),
    "ml": (
        ML_TEXT,
        {
            "title": "An Introduction to Machine Learning",
            "topic": "ai",
            "rating": 4.9,
            "published": "2024-02-20",
            "tags": ["ml", "ai"],
        },
    ),
    "finance": (
        FINANCE_DOCX,
        {
            "title": "Meridian Robotics Quarterly Report",
            "topic": "finance",
            "rating": 3.7,
            "published": "2024-04-30",
            "tags": ["finance"],
        },
    ),
    "garden": (
        GARDEN_TEXT,
        {
            "title": "Raised Bed Gardening",
            "topic": "garden",
            "rating": 2.9,
            "published": "2021-07-04",
            "tags": ["garden"],
        },
    ),
}


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)

    from localvectordb import VectorDB
    from localvectordb.backup import BackupConfig, BackupManager, BackupType
    from localvectordb.core import MetadataField, MetadataFieldType
    from localvectordb.exceptions import DocumentNotFoundError, DuplicateDocumentIDError

    c = Checker(f"e2e_local ({provider}/{model})")

    schema = {
        "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "topic": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "rating": MetadataField(type=MetadataFieldType.REAL, indexed=True),
        "published": MetadataField(type=MetadataFieldType.DATE, indexed=True),
        "tags": MetadataField(type=MetadataFieldType.JSON),
    }

    with temp_workdir("lvdb-e2e-local-") as workdir:
        c.section("create + ingest")
        db = VectorDB(
            "e2e_local",
            workdir / "data",
            metadata_schema=schema,
            embedding_provider=provider,
            embedding_model=model,
            # Small chunk budget so the ~400-token fixture docs split into
            # several chunks and the position-tracking checks are meaningful.
            chunk_size=120,
        )
        try:
            ids = db.upsert(
                documents=[content for content, _ in DOCS.values()],
                metadata=[meta for _, meta in DOCS.values()],
                ids=list(DOCS.keys()),
            )
            c.check("upsert returns all ids", sorted(ids) == sorted(DOCS.keys()), f"got {ids}")
            c.check("count matches", db.count() == len(DOCS), f"count={db.count()}")
            c.check("exists true for known id", db.exists("space") is True)
            c.check("exists batch", db.exists(["space", "nope"]) == [True, False])
            doc = db.get("cooking")
            c.check("get returns full content", doc.content == COOKING_MD)
            c.check("get returns metadata", doc.metadata.get("topic") == "food")

            c.section("search: vector / keyword / hybrid")
            r = db.query("astronauts landing on the moon", search_type="vector", k=3)
            c.check("vector: moon landing -> space doc", bool(r) and r[0].id == "space", f"top={[x.id for x in r]}")
            r = db.query("how do I make a classic butter and egg-yolk sauce", search_type="vector", k=3)
            c.check(
                "vector: sauce question -> cooking doc", bool(r) and r[0].id == "cooking", f"top={[x.id for x in r]}"
            )
            r = db.query("backpropagation", search_type="keyword", k=3)
            c.check("keyword: exact term -> ml doc", bool(r) and r[0].id == "ml", f"top={[x.id for x in r]}")
            r = db.query("growth and profits", search_type="hybrid", k=5, filters={"topic": "finance"})
            c.check(
                "hybrid + filter returns only finance",
                bool(r) and {x.id for x in r} == {"finance"},
                f"ids={[x.id for x in r]}",
            )
            r = db.query("moon landing", search_type="vector", k=5, score_threshold=0.99)
            c.check("score_threshold=0.99 filters everything", r == [], f"got {[x.id for x in r]}")

            c.section("metadata filtering")
            docs = db.filter(where={"rating": {"$gte": 4.0}}, order_by="rating DESC")
            c.check(
                "filter $gte + order_by",
                [d.id for d in docs] == ["ml", "space", "cooking"],
                f"got {[d.id for d in docs]}",
            )
            docs = db.filter(where={"title": {"$like": "%French%"}})
            c.check("filter $like", [d.id for d in docs] == ["cooking"], f"got {[d.id for d in docs]}")
            docs = db.filter(where={"$and": [{"rating": {"$lt": 4.0}}, {"topic": {"$ne": "garden"}}]})
            c.check("filter $and/$lt/$ne", [d.id for d in docs] == ["finance"], f"got {[d.id for d in docs]}")
            docs = db.filter(where={"tags": {"$contains": "french"}})
            c.check("filter JSON $contains", [d.id for d in docs] == ["cooking"], f"got {[d.id for d in docs]}")
            c.check("count with filters", db.count(filters={"topic": "space"}) == 1)

            c.section("query builder")
            results = db.query_builder().hybrid("neural networks and embeddings").filter(topic="ai").limit(3).execute()
            c.check(
                "query_builder hybrid+filter", bool(results) and results[0].id == "ml", f"got {[x.id for x in results]}"
            )
            n = db.query_builder().search("anything").filter(rating={"$gte": 4.0}).count()
            c.check("query_builder count", n == 3, f"got {n}")

            c.section("update")
            db.update("finance", metadata={"rating": 4.1})
            c.check("update metadata persists", db.get("finance").metadata["rating"] == 4.1)
            db.update("garden", content=GARDEN_TEXT + " Water deeply but infrequently.")
            c.check("update content persists", db.get("garden").content.endswith("infrequently."))

            c.section("chunks + positions")
            chunks = db.get_chunks("space")
            c.check("document was chunked", len(chunks) > 1, f"got {len(chunks)} chunks")
            c.check("chunk indices ordered", [ch.index for ch in chunks] == list(range(len(chunks))))
            positions_ok = all(SPACE_MD[ch.position.start : ch.position.end] == ch.content for ch in chunks)
            c.check("chunk positions reconstruct source text exactly", positions_ok)

            c.section("comparison + nearest neighbours")
            sim_self = db.compare_documents("space", "space")
            sim_cross = db.compare_documents("space", "cooking")
            c.check("self similarity ~1", sim_self > 0.99, f"got {sim_self:.4f}")
            c.check("cross-topic similarity < self", sim_cross < sim_self, f"got {sim_cross:.4f}")
            nn = db.nearest_neighbors("ml", k=3)
            c.check(
                "nearest_neighbors excludes reference doc", all(x.id != "ml" for x in nn), f"got {[x.id for x in nn]}"
            )
            c.check("nearest_neighbors returns results", len(nn) > 0)

            c.section("delete + duplicate handling")
            try:
                db.insert("another garden text", ids="garden")
                c.check("insert duplicate id raises", False, "no exception raised")
            except DuplicateDocumentIDError:
                c.check("insert duplicate id raises", True)
            deleted = db.delete("garden")
            c.check("delete returns count", deleted == 1, f"got {deleted}")
            c.check("deleted doc gone", db.exists("garden") is False)
            try:
                db.get("garden")
                c.check("get deleted raises DocumentNotFoundError", False, "no exception raised")
            except DocumentNotFoundError:
                c.check("get deleted raises DocumentNotFoundError", True)
            c.check("count after delete", db.count() == len(DOCS) - 1)

            c.section("backup + restore")
            db.save()
            db_files = list((workdir / "data").rglob("*.sqlite"))
            c.check("sqlite file exists on disk", len(db_files) == 1, f"found {db_files}")
        finally:
            db.close()

        if db_files:
            mgr = BackupManager(db_files[0], config=BackupConfig(backup_location=workdir / "backups"))
            backup_id = mgr.create_backup(BackupType.FULL)
            c.check("backup created", bool(backup_id), f"id={backup_id}")
            c.check("backup verifies", mgr.verify_backup(backup_id) is True)
            restore_dir = workdir / "restored"
            mgr.restore_backup(backup_id, restore_location=restore_dir, overwrite_existing=True)
            restored = VectorDB(
                "e2e_local",
                restore_dir,
                embedding_provider=provider,
                embedding_model=model,
                create_if_not_exists=False,
            )
            try:
                c.check("restored db has same docs", restored.count() == len(DOCS) - 1, f"count={restored.count()}")
                r = restored.query("lunar missions", search_type="vector", k=1)
                c.check("restored db still searchable", bool(r) and r[0].id == "space", f"top={[x.id for x in r]}")
            finally:
                restored.close()

        c.section("hierarchical (section-level) retrieval")
        hdb = VectorDB(
            "e2e_hier",
            workdir / "hier",
            embedding_provider=provider,
            embedding_model=model,
            hierarchical_embeddings=True,
        )
        try:
            hdb.upsert([SPACE_MD], ids=["guide"])
            sections = hdb.query(
                "reusable rockets and commercial spaceflight companies",
                search_type="vector",
                search_level="sections",
                k=5,
            )
            c.check("section-level search returns results", len(sections) > 0)
            headings = [s.metadata.get("section_heading") for s in sections]
            c.check("section results carry section_heading metadata", any(headings), f"got {headings}")
            c.check(
                "commercial-era section ranks in top results",
                any("Commercial" in (h or "") for h in headings),
                f"got {headings}",
            )
            docs_level = hdb.query("history of space travel", search_level="documents", k=1)
            c.check(
                "document-level search works",
                bool(docs_level) and docs_level[0].id == "guide",
                f"got {[x.id for x in docs_level]}",
            )
        finally:
            hdb.close()

        c.section("reranking (cross-encoder)")
        try:
            from localvectordb.reranking import SentenceTransformersReranker

            rdb = VectorDB("e2e_local", workdir / "data", embedding_provider=provider, embedding_model=model)
            try:
                reranker = SentenceTransformersReranker()
                r = rdb.query(
                    "training neural networks with gradient descent", search_type="vector", k=4, reranker=reranker
                )
                c.check("reranked query returns results", bool(r), "no results")
                c.check(
                    "reranker stored original_score",
                    bool(r) and "original_score" in r[0].metadata,
                    f"metadata keys={list(r[0].metadata) if r else []}",
                )
            finally:
                rdb.close()
        except Exception as exc:  # model download may be unavailable offline
            c.skip("reranking", f"{type(exc).__name__}: {exc}")

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
