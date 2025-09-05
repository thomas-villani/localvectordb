# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/mcp/__init__.py
"""
LocalVectorDB MCP Server Module

Provides Model Context Protocol (MCP) server implementation for LocalVectorDB,
enabling LLMs and tools like Claude Desktop to interact with vector databases.
"""

from localvectordb_server.mcp.server import run_mcp_server

__all__ = ["run_mcp_server"]