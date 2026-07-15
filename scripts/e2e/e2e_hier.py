"""End-to-end test: hierarchical + fused retrieval with real embeddings.

Covers the recently shipped section-level retrieval work:

* **raw-span section vectors** — with ``hierarchical_embeddings=True`` a section
  is embedded from its actual span text (the new default ``rawspan`` strategy),
  not from a centroid of its chunk vectors.
* **``search_level="fused"``** — blends chunk retrieval with section (raw-span)
  retrieval, weighted by ``section_weight``, and returns either fused documents
  or fused sections.

These ranking assertions are only meaningful with a *real* backend: with
``MockEmbeddings`` the "right" section/document cannot actually rank first.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_hier.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, make_parser, temp_workdir
from make_fixtures import COOKING_MD, SPACE_MD


def _headings(results) -> list:
    return [r.metadata.get("section_heading") for r in results]


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)

    from localvectordb import VectorDB

    c = Checker(f"e2e_hier ({provider}/{model})")

    with temp_workdir("lvdb-e2e-hier-") as workdir:
        c.section("hierarchical DB defaults to raw-span section vectors")
        hdb = VectorDB(
            "e2e_hier",
            workdir / "data",
            embedding_provider=provider,
            embedding_model=model,
            hierarchical_embeddings=True,
            # Small budget so sections span several chunks (chunk vs section
            # representations then genuinely differ).
            chunk_size=90,
        )
        try:
            c.check("hierarchical_embeddings enabled", hdb.hierarchical_embeddings is True)
            c.check(
                "section vectors default to rawspan",
                hdb.section_vector_strategy == "rawspan",
                f"got {hdb.section_vector_strategy!r}",
            )

            hdb.upsert([SPACE_MD, COOKING_MD], ids=["space", "cooking"])

            c.section("direct section-level (raw-span) retrieval")
            sections = hdb.query(
                "the first humans to walk on the Moon",
                search_type="vector",
                search_level="sections",
                k=5,
            )
            c.check("section search returns results", len(sections) > 0)
            c.check("section results carry section_heading", any(_headings(sections)), f"got {_headings(sections)}")
            c.check(
                "Apollo section ranks top for a Moon-landing query",
                "Apollo" in (sections[0].metadata.get("section_heading") or ""),
                f"top heading={sections[0].metadata.get('section_heading')!r}",
            )
            c.check(
                "section content is the raw span, not a chunk fragment",
                "Tranquility Base" in (sections[0].content or ""),
                f"content[:60]={(sections[0].content or '')[:60]!r}",
            )

            c.section("fused retrieval -> documents")
            fd = hdb.query(
                "reusable boosters and commercial spaceflight companies",
                search_type="vector",
                search_level="fused",
                return_type="documents",
                k=2,
            )
            c.check("fused/documents returns results", len(fd) > 0)
            c.check("fused/documents result type is document", fd[0].type == "document", f"type={fd[0].type!r}")
            c.check("fused/documents ranks space top", fd[0].id == "space", f"top={[x.id for x in fd]}")
            fd2 = hdb.query(
                "a hollandaise emulsion of egg yolk and clarified butter",
                search_type="vector",
                search_level="fused",
                return_type="documents",
                k=2,
            )
            c.check(
                "fused/documents ranks cooking top for a sauce query",
                fd2[0].id == "cooking",
                f"top={[x.id for x in fd2]}",
            )

            c.section("fused retrieval -> sections")
            fs = hdb.query(
                "Neil Armstrong stepping onto the lunar surface",
                search_type="vector",
                search_level="fused",
                return_type="sections",
                k=5,
            )
            c.check("fused/sections returns results", len(fs) > 0)
            c.check("fused/sections result type is section", fs[0].type == "section", f"type={fs[0].type!r}")
            c.check(
                "fused/sections surfaces the Apollo section",
                any("Apollo" in (h or "") for h in _headings(fs)),
                f"got {_headings(fs)}",
            )

            c.section("section_weight sweep (chunk-only .. section-only)")
            chunk_only = hdb.query(
                "the first Moon landing",
                search_type="vector",
                search_level="fused",
                return_type="documents",
                section_weight=0.0,
                k=2,
            )
            c.check("section_weight=0.0 (chunk-only) returns results", len(chunk_only) > 0)
            c.check(
                "section_weight=0.0 ranks space top", chunk_only[0].id == "space", f"top={[x.id for x in chunk_only]}"
            )
            section_only = hdb.query(
                "the first Moon landing",
                search_type="vector",
                search_level="fused",
                return_type="documents",
                section_weight=1.0,
                k=2,
            )
            c.check("section_weight=1.0 (section-only) returns results", len(section_only) > 0)
            c.check(
                "section_weight=1.0 ranks space top",
                section_only[0].id == "space",
                f"top={[x.id for x in section_only]}",
            )

            c.section("guards")
            try:
                hdb.query_cursor("moon landing", search_level="fused")
                c.check("fused rejected for cursor/streaming", False, "no exception")
            except ValueError:
                c.check("fused rejected for cursor/streaming -> ValueError", True)

            c.section("fused via async (delegates to sync)")
            import asyncio

            async def _afused():
                return await hdb.query_async(
                    "reusable boosters and commercial spaceflight",
                    search_type="vector",
                    search_level="fused",
                    return_type="documents",
                    k=2,
                )

            afd = asyncio.run(_afused())
            c.check("query_async fused returns results", len(afd) > 0)
            c.check("query_async fused ranks space top", afd[0].id == "space", f"top={[x.id for x in afd]}")
        finally:
            hdb.close()

        c.section("fused requires hierarchical_embeddings")
        flat = VectorDB(
            "e2e_flat",
            workdir / "flat",
            embedding_provider=provider,
            embedding_model=model,
        )
        try:
            flat.upsert([SPACE_MD], ids=["space"])
            try:
                flat.query("moon landing", search_level="fused")
                c.check("fused on non-hierarchical DB raises", False, "no exception")
            except ValueError:
                c.check("fused on non-hierarchical DB -> ValueError", True)
        finally:
            flat.close()

        c.section("legacy centroid strategy still works")
        cdb = VectorDB(
            "e2e_centroid",
            workdir / "centroid",
            embedding_provider=provider,
            embedding_model=model,
            hierarchical_embeddings=True,
            section_vector_strategy="centroid",
            chunk_size=90,
        )
        try:
            c.check(
                "centroid strategy respected",
                cdb.section_vector_strategy == "centroid",
                f"got {cdb.section_vector_strategy!r}",
            )
            cdb.upsert([SPACE_MD, COOKING_MD], ids=["space", "cooking"])
            cf = cdb.query(
                "reusable boosters and commercial spaceflight",
                search_type="vector",
                search_level="fused",
                return_type="documents",
                k=2,
            )
            c.check("centroid fused returns results", len(cf) > 0)
            c.check("centroid fused ranks space top", cf[0].id == "space", f"top={[x.id for x in cf]}")
        finally:
            cdb.close()

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
