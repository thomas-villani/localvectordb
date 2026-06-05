# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
LocalVectorDB MCP Server Module

Provides Model Context Protocol (MCP) server implementation for LocalVectorDB,
enabling LLMs and tools like Claude Desktop to interact with vector databases.
"""

from localvectordb_server.mcp.server import run_mcp_server

__all__ = ["run_mcp_server"]
