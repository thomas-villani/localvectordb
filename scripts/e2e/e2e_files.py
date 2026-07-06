"""End-to-end test: real file ingestion across formats.

Ingests genuine PDF, DOCX, XLSX, HTML, Markdown, and Python files (generated
by make_fixtures.py) through the extraction pipeline with real embeddings,
then verifies that cross-format semantic and keyword queries retrieve the
right document.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_files.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, ensure_fixtures, make_parser, temp_workdir

FIXTURE_FILES = [
    "space_exploration.md",
    "french_cooking.md",
    "machine_learning.pdf",
    "financial_report.docx",
    "warehouse_inventory.xlsx",
    "climate_currents.html",
    "task_scheduler.py",
]

# Query -> expected top document (filename stem), per search type.
SEMANTIC_QUERIES = {
    "training deep neural networks and avoiding overfitting": "machine_learning",
    "the Apollo missions and the history of human spaceflight": "space_exploration",
    "classic French sauces and knife technique": "french_cooking",
    "quarterly revenue, gross margin, and full-year guidance": "financial_report",
    "how ocean currents move heat around the planet": "climate_currents",
}

KEYWORD_QUERIES = {
    "Tranquility Base": "space_exploration",
    "brunoise": "french_cooking",
    "thermohaline": "climate_currents",
}


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)
    fixtures = ensure_fixtures()

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    c = Checker(f"e2e_files ({provider}/{model})")

    with temp_workdir("lvdb-e2e-files-") as workdir:
        db = VectorDB(
            "e2e_files",
            workdir,
            metadata_schema={
                "source_format": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            },
            embedding_provider=provider,
            embedding_model=model,
            chunk_size=150,
        )
        try:
            c.section("ingest real files (pdf/docx/xlsx/html/md/py)")
            paths = [fixtures / name for name in FIXTURE_FILES]
            metadata = [{"source_format": p.suffix.lstrip(".")} for p in paths]
            ids = db.upsert_from_file(paths, metadata=metadata)
            c.check("all files ingested", len(ids) == len(FIXTURE_FILES), f"got {ids}")
            expected_ids = {p.stem for p in paths}
            c.check("ids default to filename stems", set(ids) == expected_ids, f"got {sorted(ids)}")
            c.check("count matches file count", db.count() == len(FIXTURE_FILES))

            c.section("extracted content sanity")
            pdf_doc = db.get("machine_learning")
            c.check("pdf text extracted", "backpropagation" in pdf_doc.content.lower(), f"len={len(pdf_doc.content)}")
            docx_doc = db.get("financial_report")
            c.check("docx headings preserved as markdown", "## Executive Summary" in docx_doc.content)
            xlsx_doc = db.get("warehouse_inventory")
            c.check(
                "xlsx extracted as markdown table",
                "| SKU |" in xlsx_doc.content and "Cordless Drill" in xlsx_doc.content,
            )
            html_doc = db.get("climate_currents")
            c.check("html converted to markdown headings", "# Understanding Ocean Currents" in html_doc.content)
            py_doc = db.get("task_scheduler")
            c.check("source code fenced as python", "```python" in py_doc.content)

            c.section("cross-format semantic retrieval (vector search)")
            for query, expected in SEMANTIC_QUERIES.items():
                r = db.query(query, search_type="vector", k=3)
                c.check(
                    f"vector: {query[:48]!r} -> {expected}", bool(r) and r[0].id == expected, f"top={[x.id for x in r]}"
                )

            c.section("keyword retrieval (FTS5)")
            for query, expected in KEYWORD_QUERIES.items():
                r = db.query(query, search_type="keyword", k=3)
                c.check(
                    f"keyword: {query!r} -> {expected}", bool(r) and r[0].id == expected, f"top={[x.id for x in r]}"
                )

            c.section("hybrid + metadata filter on file docs")
            r = db.query("machine intelligence", search_type="hybrid", k=5, filters={"source_format": "pdf"})
            c.check(
                "hybrid restricted to pdf docs",
                bool(r) and {x.id for x in r} == {"machine_learning"},
                f"got {[x.id for x in r]}",
            )
            docs = db.filter(where={"source_format": {"$in": ["md", "html"]}})
            c.check(
                "filter $in on source_format",
                {d.id for d in docs} == {"space_exploration", "french_cooking", "climate_currents"},
                f"got {sorted(d.id for d in docs)}",
            )

            c.section("re-ingest is idempotent (upsert)")
            ids2 = db.upsert_from_file(fixtures / "french_cooking.md", metadata={"source_format": "md"})
            c.check(
                "re-upsert same file keeps one doc", db.count() == len(FIXTURE_FILES), f"count={db.count()}, ids={ids2}"
            )
        finally:
            db.close()

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
