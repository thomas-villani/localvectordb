# LocalVectorDB MCP bundle (`.mcpb`)

This directory packages LocalVectorDB's MCP server as an **MCP bundle** — a
single `.mcpb` file you drag onto Claude Desktop's *Extensions* pane to install
the server in one click, no JSON editing.

The bundle carries no LocalVectorDB code: `manifest.json` + `pyproject.toml`
declare a dependency on the published `localvectordb[mcp]` package, and
`src/server.py` is a thin launcher. At load time Claude Desktop's `uv` runtime
resolves and runs the LocalVectorDB stdio MCP server from PyPI.

## Requirements

- A recent Claude Desktop (the one with the *Extensions* / MCP-bundle installer).
- [`uv`](https://docs.astral.sh/uv/) on PATH — the bundle's declared runtime.
- Python ≥ 3.12 (provisioned by `uv` if needed).

## Configuration

Claude Desktop prompts for two optional settings when you install the bundle;
they map to the server's environment variables:

- **Databases directory** (`LVDB_MCP_DATABASES_ROOT`) — folder holding your
  LocalVectorDB databases. Defaults to `~/localvectordb`.
- **Server mode** (`LVDB_MCP_MODE`) — `read-only` (default) exposes search/read
  tools; `read-write` also allows creating and modifying databases.

## Install (end users)

1. Download `localvectordb.mcpb` (a release asset, or build it below).
2. Open Claude Desktop → **Settings → Extensions** and drop the file in (or
   double-click it).
3. Set the databases directory (and mode, if you want writes), and the
   `list_databases` / `query_database` / `filter_documents` / … tools appear.

Prefer to wire it up by hand? `lvdb mcp serve` runs the same stdio server — see
the [MCP docs](https://thomas-villani.github.io/localvectordb/).

## Build / repack

```bash
npm install -g @anthropic-ai/mcpb   # one-time
mcpb validate manifest.json
mcpb pack . localvectordb.mcpb      # from this dir (bare `mcpb pack` names it mcpb.mcpb)
```

The bundle version is kept in lock-step with the package by `bump-my-version`
(see `[tool.bumpversion]` in the root `pyproject.toml`): a release bumps
`manifest.json`, this `pyproject.toml`, and its `localvectordb[mcp]>=` pin
together.
