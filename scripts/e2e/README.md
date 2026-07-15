# End-to-End Test Suite

Release-qualification scripts that exercise localvectordb against **real
embedding backends and real documents** — no mocks. They complement the unit
and integration test suite (`pytest`), which mocks embedding providers and
therefore cannot catch provider-contract or async-context bugs.

These scripts are excluded from the published package (the whole `/scripts`
tree is excluded from the sdist).

## Prerequisites

A real embedding backend, auto-detected in this order:

1. **Ollama** (preferred): `ollama serve` running with `nomic-embed-text`
   pulled (`ollama pull nomic-embed-text`).
2. **sentence-transformers** (fallback, fully local): installed by
   `uv sync --dev`; uses `all-MiniLM-L6-v2` (downloaded from HF on first use).

Force a backend with `--provider ollama|sentence_transformers` on any script.

## Running

```bash
# Everything (fixtures + 4 suites)
./.venv/Scripts/python.exe scripts/e2e/run_all.py

# Individual suites
./.venv/Scripts/python.exe scripts/e2e/e2e_local.py    # library: CRUD/search/filters/backup/hierarchical/rerank
./.venv/Scripts/python.exe scripts/e2e/e2e_files.py    # real PDF/DOCX/XLSX/HTML/MD/py ingestion + retrieval
./.venv/Scripts/python.exe scripts/e2e/e2e_hier.py     # raw-span section vectors + search_level="fused" retrieval
./.venv/Scripts/python.exe scripts/e2e/e2e_patch.py    # document patch API (find/replace + splice, expect_hash)
./.venv/Scripts/python.exe scripts/e2e/e2e_server.py   # lvdb serve + RemoteVectorDB + REST + auth + upload + SSE
./.venv/Scripts/python.exe scripts/e2e/e2e_cli.py      # lvdb CLI workflow
```

Each script prints `[PASS]/[FAIL]` per check and exits non-zero on failure.
All state lives in temp directories that are cleaned up afterwards.

## Files

| File | Purpose |
|------|---------|
| `make_fixtures.py` | Generates real fixture documents (PDF via PyMuPDF, DOCX via python-docx, XLSX via openpyxl, plus MD/HTML/py) into `fixtures/`. Five distinct topics so semantic-retrieval assertions are meaningful. |
| `_common.py` | Provider auto-detection, check runner, temp dirs. |
| `e2e_local.py` | Local `LocalVectorDB` flow with a typed metadata schema: upsert/get/exists/update/delete, vector/keyword/hybrid search with filters, query builder, chunk position tracking, comparison + nearest neighbours, backup/restore, hierarchical (section-level) retrieval, cross-encoder reranking. |
| `e2e_files.py` | `upsert_from_file` across six real formats, extraction sanity, cross-format semantic + keyword retrieval, filters, idempotent re-ingest. |
| `e2e_hier.py` | Hierarchical retrieval: raw-span section vectors (the default strategy) vs legacy centroids, direct section-level search, `search_level="fused"` blending chunk + section legs into fused documents and fused sections, the `section_weight` sweep (chunk-only .. section-only), fused rejection for streaming/cursor, and fused via `query_async`. |
| `e2e_patch.py` | Document patch API through the real re-embedding path: replace/splice/append/prepend, `expect_hash` optimistic concurrency, the no-op vs not-found vs conflict contract, error contracts (unmatched find / overlapping ops), metadata merge, and that a patched doc stays retrievable by its new vector + keyword content. |
| `e2e_server.py` | Boots a real `lvdb serve` subprocess with API-key auth + upload enabled; exercises RemoteVectorDB parity, raw REST (list, multipart upload), SSE streaming, and read-only-key permission enforcement. |
| `e2e_cli.py` | Full `lvdb` CLI workflow: create/list, add (files + inline text), search (hybrid/keyword, JSON output), get/related/stats/info, delete document + database. |
| `run_all.py` | Runs everything and prints a summary table. |

## Bugs this suite caught before v0.1.0

Kept here as a reminder of why mock-free e2e runs matter:

- `$contains`/`$not_contains` filters bound 1 SQL param for 2 placeholders
  (crashed every JSON-field filter).
- JSON metadata fields returned as raw strings from `get()`/`filter()`, which
  also broke partial `update()` on any document with a JSON field.
- Server config/CLI/API rejected every embedding provider except
  ollama/openai.
- Server DB creation passed `api_key=None` into providers that don't accept
  it (broke `ollama` creation via HTTP).
- `/documents/count` called `db.count(where=...)` (kwarg doesn't exist).
- All server search endpoints called sync `db.query()` on the event loop —
  `embed_sync()` guard made vector/hybrid search fail with every real
  provider (mocks passed).
- SSE streaming endpoint iterated the `query_cursor_async` coroutine instead
  of awaiting it and using `stream_individual_async()`.
- `/health` called Ollama with a 60 s timeout × 3 retries inline; also the
  `localhost` default stalled ~2.5 s/request on Windows (IPv6 first). Now
  127.0.0.1 + 2 s single attempt.
