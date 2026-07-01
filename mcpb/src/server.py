"""MCPB launcher for the LocalVectorDB MCP server (uv runtime).

Thin wrapper over LocalVectorDB's stdio MCP server
(``localvectordb_server.mcp.server.run_mcp_server``). It exists only to give the
MCPB ``uv`` runtime a stable entry file; all the real logic lives in the
installed ``localvectordb`` package.

Configuration is passed by the MCPB host as environment variables (see the
``env`` block of ``manifest.json``): ``LVDB_MCP_MODE`` selects read-only vs
read-write, and ``LVDB_MCP_DATABASES_ROOT`` points at the folder of databases.
The server reads these when its lifespan starts. We forward the mode explicitly
because ``run_mcp_server`` writes ``LVDB_MCP_MODE`` from its argument, so passing
the env value keeps the host's choice from being overwritten by the default.
"""

import asyncio
import os


def main() -> int:
    from localvectordb_server.mcp.server import run_mcp_server

    mode = os.getenv("LVDB_MCP_MODE", "read-only")
    asyncio.run(run_mcp_server(mode=mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
