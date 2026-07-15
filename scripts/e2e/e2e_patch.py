"""End-to-end test: document patch API with real embeddings.

Covers the ``patch()`` surface end-to-end through the *real* re-embedding path
that mocks cannot exercise: ``patch()`` -> ``apply_ops`` -> ``update()`` ->
``upsert()`` re-embeds the changed chunks through the real async provider pool.
Exercises replace/splice/append/prepend, ``expect_hash`` optimistic concurrency,
the no-op contract, the error contracts (missing doc / unmatched find /
overlapping ops), metadata merge, and that a patched document stays retrievable
by both its new vector content and its new keyword tokens.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_patch.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, make_parser, temp_workdir

DOC = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models require careful tuning of hyperparameters. "
    "Neural networks learn representations from data through backpropagation."
)


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType
    from localvectordb.exceptions import DocumentNotFoundError, PatchConflictError, PatchError

    c = Checker(f"e2e_patch ({provider}/{model})")
    schema = {"reviewed": MetadataField(type=MetadataFieldType.BOOLEAN, indexed=True)}

    with temp_workdir("lvdb-e2e-patch-") as workdir:
        db = VectorDB(
            "e2e_patch",
            workdir / "data",
            metadata_schema=schema,
            embedding_provider=provider,
            embedding_model=model,
            # Small budget so the doc splits into several chunks and the
            # chunk-reuse-on-patch path is actually exercised.
            chunk_size=60,
        )
        try:
            c.section("setup")
            db.upsert(documents=[DOC], ids=["doc1"], metadata=[{}])
            orig = db.get("doc1")
            c.check("doc ingested", orig.content == DOC)
            orig_hash = orig.content_hash
            c.check("has content_hash", bool(orig_hash))

            c.section("replace op (real re-embed)")
            res = db.patch("doc1", [{"op": "replace", "find": "brown", "replace": "red"}])
            c.check("replace updated=True", res.updated is True, f"res={res.to_dict()}")
            c.check("replace ops_applied=1", res.ops_applied == 1)
            c.check("new_hash differs", res.new_hash != orig_hash)
            after = db.get("doc1")
            c.check("content spliced", "red fox" in after.content and "brown" not in after.content)
            c.check("rest of doc intact", "backpropagation" in after.content)
            c.check("get hash matches result", after.content_hash == res.new_hash)

            c.section("search works after re-embed")
            r = db.query("a fast red-colored fox", search_type="vector", k=1)
            c.check("vector search returns patched doc", bool(r) and r[0].id == "doc1", f"top={[x.id for x in r]}")
            r = db.query("red", search_type="keyword", k=1)
            c.check("keyword finds new token 'red'", bool(r) and r[0].id == "doc1", f"top={[x.id for x in r]}")

            c.section("expect_hash optimistic concurrency")
            cur = db.get("doc1").content_hash
            try:
                db.patch("doc1", [{"op": "replace", "find": "dog", "replace": "cat"}], expect_hash="deadbeef")
                c.check("stale expect_hash raises", False, "no exception")
            except PatchConflictError:
                c.check("stale expect_hash -> PatchConflictError", True)
            res = db.patch("doc1", [{"op": "replace", "find": "dog", "replace": "cat"}], expect_hash=cur)
            c.check("matching expect_hash succeeds", res.updated is True)
            c.check("cat splice landed", "lazy cat" in db.get("doc1").content)

            c.section("splice / append / prepend")
            db.patch("doc1", [{"op": "append", "text": " THE END."}])
            c.check("append lands", db.get("doc1").content.endswith(" THE END."))
            db.patch("doc1", [{"op": "prepend", "text": "TITLE: "}])
            c.check("prepend lands", db.get("doc1").content.startswith("TITLE: "))

            c.section("no-op contract (updated=False, not not-found)")
            cur = db.get("doc1").content_hash
            res = db.patch("doc1", [{"op": "replace", "find": "TITLE: ", "replace": "TITLE: "}])
            c.check("identical replace -> updated=False", res.updated is False, f"res={res.to_dict()}")
            c.check("no-op keeps hash", res.new_hash == cur)

            c.section("error contracts")
            try:
                db.patch("missing", [{"op": "append", "text": "x"}])
                c.check("missing doc raises", False, "no exception")
            except DocumentNotFoundError:
                c.check("missing doc -> DocumentNotFoundError", True)
            try:
                db.patch("doc1", [{"op": "replace", "find": "zzz-not-present", "replace": "x"}])
                c.check("unmatched find raises", False, "no exception")
            except PatchError:
                c.check("unmatched find -> PatchError", True)
            try:
                db.patch(
                    "doc1",
                    [
                        {"op": "splice", "start": 0, "end": 5, "text": "A"},
                        {"op": "splice", "start": 3, "end": 8, "text": "B"},
                    ],
                )
                c.check("overlapping splices raise", False, "no exception")
            except PatchError:
                c.check("overlapping splices -> PatchError", True)

            c.section("metadata merge")
            db.patch("doc1", [{"op": "append", "text": "!"}], metadata={"reviewed": True})
            # BOOLEAN metadata round-trips as SQLite 1/0 (same as update()); check truthy.
            c.check("metadata merged on patch", bool(db.get("doc1").metadata.get("reviewed")))
        finally:
            db.close()

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
