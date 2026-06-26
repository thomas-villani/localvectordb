"""
LocalVectorDB MCP Server Module

Provides Model Context Protocol (MCP) server implementation for LocalVectorDB,
enabling LLMs and tools like Claude Desktop to interact with vector databases.
"""

__all__ = ["run_mcp_server"]


def __getattr__(name: str):
    # Lazily import the server (which pulls in the optional `fastmcp` dependency)
    # so that importing lightweight submodules like `mcp.config` — and, by
    # extension, the `lvdb` CLI — does not require the `mcp` extra to be installed.
    if name == "run_mcp_server":
        from localvectordb_server.mcp.server import run_mcp_server

        return run_mcp_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
