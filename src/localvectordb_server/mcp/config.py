"""
MCP Server Configuration

Simple configuration management for LocalVectorDB MCP server that maps
directly to LocalVectorDB constructor parameters.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

# The only valid security modes. A typo here must fail closed: an unvalidated
# ``mode`` other than "read-only" makes check_write_permission() fail OPEN and
# silently permit writes, so from_file() validates against this set.
_VALID_MODES = ("read-only", "read-write")


@dataclass
class MCPConfig:
    """Configuration for MCP server operation"""

    # Security mode
    mode: Literal["read-only", "read-write"] = "read-only"

    # Database paths/URLs
    databases_root: str = "./databases"  # Default root for local databases
    databases_map: Dict[str, str] = field(default_factory=dict)  # Name -> path/URL mapping

    # Default database creation parameters
    db_defaults: Dict[str, Any] = field(
        default_factory=lambda: {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "chunk_size": 500,
            # Overlap is in *sentences* for the "sentences" method (not tokens),
            # so keep it small — matches the core LocalVectorDB default.
            "chunk_overlap": 1,
            "chunking_method": "sentences",
            "enable_fts": True,
            "enable_gpu": False,
        }
    )

    # Connection settings for remote databases
    remote_defaults: Dict[str, Any] = field(
        default_factory=lambda: {"timeout": 30, "max_retries": 3, "retry_delay": 1.0}
    )

    # Operational settings
    max_concurrent_operations: int = 10
    operation_timeout: int = 300  # seconds
    log_operations: bool = True
    log_level: str = "INFO"

    # Tools to expose
    read_only_tools: List[str] = field(
        default_factory=lambda: [
            "list_databases",
            "get_database_info",
            "query_database",
            "find_related_documents",
            "filter_documents",
            "get_document",
            "check_documents_exist",
            "get_metadata_schema",
            "get_system_info",
        ]
    )

    write_tools: List[str] = field(
        default_factory=lambda: [
            "create_database",
            "delete_database",
            "upsert_documents",
            "update_document",
            "patch_document",
            "delete_document",
            "update_metadata_schema",
        ]
    )

    def check_write_permission(self, operation: str):
        """Check if write operations are allowed"""
        if self.mode == "read-only":
            raise PermissionError(f"Operation '{operation}' not allowed in read-only mode")

    def get_database_path(self, name: str) -> str:
        """Get the path or URL for a database"""
        # Check if there's an explicit mapping
        if name in self.databases_map:
            return self.databases_map[name]

        # Otherwise use the default root directory
        return str(Path(self.databases_root).resolve())

    def get_enabled_tools(self) -> List[str]:
        """Get list of tools enabled for current mode"""
        tools = self.read_only_tools.copy()
        if self.mode == "read-write":
            tools.extend(self.write_tools)
        return tools

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables"""
        config = cls()

        # Mode
        if mode := os.getenv("LVDB_MCP_MODE"):
            if mode in ("read-only", "read-write"):
                config.mode = cast(Literal["read-only", "read-write"], mode)

        # Database root
        if root := os.getenv("LVDB_MCP_DATABASES_ROOT"):
            config.databases_root = root

        # Database mappings (format: name1=path1,name2=path2)
        if mappings := os.getenv("LVDB_MCP_DATABASES_MAP"):
            for mapping in mappings.split(","):
                if "=" in mapping:
                    name, path = mapping.split("=", 1)
                    config.databases_map[name.strip()] = path.strip()

        # Default embedding provider
        if provider := os.getenv("LVDB_MCP_EMBEDDING_PROVIDER"):
            config.db_defaults["embedding_provider"] = provider

        # Default embedding model
        if model := os.getenv("LVDB_MCP_EMBEDDING_MODEL"):
            config.db_defaults["embedding_model"] = model

        # Chunk size
        if chunk_size := os.getenv("LVDB_MCP_CHUNK_SIZE"):
            try:
                config.db_defaults["chunk_size"] = int(chunk_size)
            except ValueError:
                # Non-integer LVDB_MCP_CHUNK_SIZE; keep the default chunk size.
                pass

        # Log level
        if log_level := os.getenv("LVDB_MCP_LOG_LEVEL"):
            config.log_level = log_level.upper()

        return config

    @classmethod
    def from_file(cls, path: str) -> "MCPConfig":
        """Load configuration from TOML file"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                # Surface a clear, actionable error instead of a raw parser message.
                raise ValueError(f"Invalid TOML in MCP config '{path}': {e}") from e

        config = cls()

        # Load MCP section
        if mcp := data.get("mcp"):
            if "mode" in mcp:
                mode = mcp["mode"]
                if mode not in _VALID_MODES:
                    # Reject rather than fail open: an unrecognized mode would make
                    # check_write_permission() permit writes (it only blocks
                    # "read-only"), so a typo like "readonly" must not slip through.
                    raise ValueError(f"Invalid MCP mode {mode!r}; must be one of {_VALID_MODES}.")
                config.mode = mode
            if "log_level" in mcp:
                config.log_level = mcp["log_level"]
            if "log_operations" in mcp:
                config.log_operations = mcp["log_operations"]
            if "max_concurrent_operations" in mcp:
                config.max_concurrent_operations = mcp["max_concurrent_operations"]
            if "operation_timeout" in mcp:
                config.operation_timeout = mcp["operation_timeout"]
            if "read_only_tools" in mcp:
                config.read_only_tools = mcp["read_only_tools"]
            if "write_tools" in mcp:
                config.write_tools = mcp["write_tools"]
        # Load databases section
        if databases := data.get("databases"):
            if "root" in databases:
                config.databases_root = databases["root"]
            if "map" in databases:
                config.databases_map = databases["map"]

        # Load defaults section
        if defaults := data.get("defaults"):
            config.db_defaults.update(defaults)

        # Load remote section
        if remote := data.get("remote"):
            config.remote_defaults.update(remote)

        return config

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "MCPConfig":
        """Load configuration from file or environment"""
        # Start with environment
        config = cls.from_env()

        # Override with file if provided
        if config_path:
            file_config = cls.from_file(config_path)
            # File takes precedence
            return file_config

        # Check for config file in environment
        if config_file := os.getenv("LVDB_MCP_CONFIG"):
            return cls.from_file(config_file)

        return config
