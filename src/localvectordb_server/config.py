"""
LocalVectorDB Server Configuration

This module provides the configuration model and helpers used by the
LocalVectorDB server (v1.0). Configuration objects are represented as
dataclasses (DatabaseSettings, EmbeddingSettings, ServerSettings,
BackupSettings, MigrationSettings, SecuritySettings) and composed by the
Config container class. The module supports loading configuration from
multiple sources and formats, merging overrides intelligently, and
validating the resulting configuration.

Features:
- Typed dataclass settings with validate() methods
- Load from TOML, JSON
- Load and override from environment variables (``LVDB_`` prefixed)
- Generate TOML output and export Flask-compatible config
- Intelligent merging that preserves non-default base values

Supported environment variables (prefix ``LVDB_``):
- Sectioned variables like LVDB_DATABASE_* LVDB_SERVER_* LVDB_EMBEDDING_*
- Nested server security variables like LVDB_SERVER_SECURITY_*

See class docstrings for detailed field descriptions and validation rules.
"""

import copy
import json
import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin, get_type_hints

import click
import tomli_w

from localvectordb._schema import get_common_metadata_schemas
from localvectordb.core import MetadataField
from localvectordb.exceptions import ConfigurationError


class BaseSettings(ABC):

    @abstractmethod
    def validate(self) -> bool:
        pass

    def copy(self):
        return copy.deepcopy(self)

    def update_from_dict(self, update_dict, raise_errors: bool = False):
        """Updates the attributes from a given dict."""
        for k, v in update_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif raise_errors:
                raise AttributeError(f"Type {type(self)} does not have '{k}' attribute.")


@dataclass
class EmbeddingSettings(BaseSettings):
    """Settings for embedding providers."""

    provider: str = "ollama"  # any provider registered in localvectordb.embeddings.EmbeddingRegistry
    model: str = "nomic-embed-text"
    base_url: Optional[str] = None  # Provider-specific base URL
    api_key: Optional[str] = None  # API key for providers that need it
    batch_size: int = 64
    timeout: int = 30  # Request timeout in seconds
    max_retries: int = 3

    # Provider-specific configurations
    config: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        # Deferred import: keeps config importable without triggering embedding
        # provider plugin discovery until validation actually runs.
        from localvectordb.embeddings import EmbeddingRegistry

        available = EmbeddingRegistry.list()
        if not isinstance(self.provider, str) or self.provider.lower() not in available:
            raise ConfigurationError(f"provider must be one of: {', '.join(sorted(available))} (got {self.provider!r})")

        if not isinstance(self.model, str) or not self.model:
            raise ConfigurationError("model must be a non-empty string")

        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be a positive integer")

        if self.timeout <= 0:
            raise ConfigurationError("timeout must be a positive integer")

        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be a non-negative integer")

        # Provider-specific validation
        if self.provider.lower() == "openai" and not self.api_key and not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("OpenAI provider requires api_key or OPENAI_API_KEY environment variable")

        return True


@dataclass
class DatabaseSettings(BaseSettings):
    """Settings related to database operations and connections"""

    root_dir: str = "./.lvdb"
    timeout: int = 300  # seconds
    connection_pool_size: int = 10
    enable_gpu: bool = False
    enable_fts: bool = True

    faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2"
    faiss_index_hnsw_flat_neighbors: Optional[int] = None  # Only used for IndexHNSWFlat
    faiss_index_lsh_bits: Optional[int] = None  # Only used for IndexLSH, number of bits for the index.

    # Default database parameters when creating new ones
    chunk_size: int = 500  # Renamed from chunk_tokens for v1.0
    chunk_overlap: int = 1
    chunking_method: str = "lines"

    # Default metadata schema for new databases
    default_metadata_schema: Dict[str, MetadataField] = field(default_factory=dict)

    def validate(self):
        # Validate root_dir
        if not isinstance(self.root_dir, str) or not self.root_dir:
            raise ConfigurationError("root_dir must be a non-empty string")

        # Validate numeric settings
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ConfigurationError("timeout must be a positive integer")

        if not isinstance(self.connection_pool_size, int) or self.connection_pool_size <= 0:
            raise ConfigurationError("connection_pool_size must be a positive integer")

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be a non-negative integer")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")

        if self.faiss_index_type and self.faiss_index_type not in (
            "IndexFlatL2",
            "IndexFlatIP",
            "IndexHNSWFlat",
            "IndexLSH",
        ):
            raise ConfigurationError(
                "faiss_index_type must be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH"
            )

        # Validate metadata schema
        for field_name, _field_config in self.default_metadata_schema.items():
            if not isinstance(field_name, str) or not field_name:
                raise ConfigurationError("Metadata field names must be non-empty strings")

        return True


@dataclass
class BackupSettings(BaseSettings):
    """Settings for database backup operations."""

    enabled: bool = True
    default_location: str = "./backups"
    retention_days: int = 30  # Keep backups for 30 days
    max_backups: int = 50  # Maximum number of backups to keep per database
    compression_type: Literal["gzip", "lzma", "none"] = "gzip"
    compression_level: int = 6  # 1-9 for gzip, 0-9 for lzma

    # Auto-backup settings
    auto_backup_enabled: bool = False
    auto_backup_interval_hours: int = 24  # Daily backups
    auto_backup_type: Literal["full", "incremental"] = "incremental"

    # Performance settings
    backup_chunk_size: int = 1024 * 1024  # 1MB chunks for streaming
    verify_backups: bool = True  # Verify backup integrity after creation

    def validate(self):
        if self.retention_days < 0:
            raise ConfigurationError("retention_days must be non-negative")

        if self.max_backups <= 0:
            raise ConfigurationError("max_backups must be positive")

        if self.compression_level < 0 or self.compression_level > 9:
            raise ConfigurationError("compression_level must be between 0 and 9")

        if self.auto_backup_interval_hours <= 0:
            raise ConfigurationError("auto_backup_interval_hours must be positive")

        if self.backup_chunk_size <= 0:
            raise ConfigurationError("backup_chunk_size must be positive")

        # Validate paths
        if not isinstance(self.default_location, str) or not self.default_location:
            raise ConfigurationError("default_location must be a non-empty string")

        return True


@dataclass
class MigrationSettings(BaseSettings):
    """Settings for database migration operations."""

    enabled: bool = True
    migration_dir: str = "./migrations"
    auto_migrate: bool = False  # Automatically apply pending migrations on startup
    backup_before_migration: bool = True  # Create backup before applying migrations

    # Safety settings
    require_confirmation: bool = True  # Require confirmation for destructive operations
    allow_destructive_migrations: bool = False  # Allow migrations that could lose data
    max_rollback_steps: int = 10  # Maximum number of migration steps to rollback

    # Template settings
    migration_template_author: Optional[str] = None  # Default author for new migrations
    migration_template_format: Literal["python", "sql"] = "python"

    def validate(self):
        if self.max_rollback_steps < 0:
            raise ConfigurationError("max_rollback_steps must be non-negative")

        # Validate paths
        if not isinstance(self.migration_dir, str) or not self.migration_dir:
            raise ConfigurationError("migration_dir must be a non-empty string")

        return True


@dataclass
class ExtractionSettings(BaseSettings):
    """Settings for file-content extraction (powered by all2md).

    These map onto the security-relevant options of the all2md-backed extractor.
    Defaults are hardened for untrusted uploads; relax them only for trusted
    content. The values are forwarded to the extractor as keyword arguments by
    :meth:`extractor_kwargs`.
    """

    # Allow the converter to fetch remote assets referenced by a document
    # (images, stylesheets, etc.). Off by default to avoid SSRF on uploads.
    allow_remote_fetch: bool = False
    # Host allowlist applied when allow_remote_fetch is True (None = all hosts).
    allowed_hosts: Optional[List[str]] = None
    # HTML only: strip scripts / event handlers / other dangerous elements.
    strip_dangerous_elements: bool = True
    # How embedded attachments/assets are handled: "skip" (default), "save", etc.
    attachment_mode: str = "skip"

    def validate(self):
        if not isinstance(self.attachment_mode, str) or not self.attachment_mode:
            raise ConfigurationError("attachment_mode must be a non-empty string")

        if self.allowed_hosts is not None:
            if not isinstance(self.allowed_hosts, list):
                raise ConfigurationError("allowed_hosts must be a list of strings or None")
            for host in self.allowed_hosts:
                if not isinstance(host, str) or not host:
                    raise ConfigurationError("Each entry in allowed_hosts must be a non-empty string")

        return True

    def extractor_kwargs(self) -> Dict[str, Any]:
        """Return the keyword arguments to forward to the extractor."""
        return {
            "allow_remote_fetch": self.allow_remote_fetch,
            "allowed_hosts": self.allowed_hosts,
            "strip_dangerous_elements": self.strip_dangerous_elements,
            "attachment_mode": self.attachment_mode,
        }


@dataclass
class SecuritySettings(BaseSettings):
    """Security-related settings for the server."""

    # API Key authentication
    require_api_key: bool = False
    api_key_header: str = "Authorization"  # Header name for API key
    trusted_hosts: Optional[List[str]] = None  # Host header validation patterns (e.g., ["localhost", "*.example.com"])

    # Key management
    key_database_path: Optional[str] = None  # None = auto-determined from db_root_dir
    default_key_expiry_days: Optional[int] = None  # None = no default expiration
    auto_prune_expired_keys: bool = False  # Automatically remove expired keys
    key_audit_logging: bool = True  # Log key usage for audit trails
    auth_log_level: str = "INFO"
    warn_expiring_days: int = 7  # Days before expiry to warn about

    # CORS settings
    cors_enabled: bool = True
    cors_allowed_origins: Union[str, List[str]] = "*"
    cors_allowed_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )
    cors_allowed_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization"])
    cors_max_age: int = 86400  # 24 hours

    # CSP/HTTP header settings (Talisman)
    security_headers_enabled: bool = True
    force_https: bool = False
    strict_transport_security: bool = True
    strict_transport_security_max_age: int = 31536000
    content_security_policy: dict = field(
        default_factory=lambda: {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline'",
            "style-src": "'self' 'unsafe-inline'",
            "connect-src": "'self'",
            "img-src": "'self' data:",
            "frame-ancestors": "none",
        }
    )
    content_type_nosniff: bool = True
    x_frame_options: str = "DENY"
    x_xss_protection: bool = True
    referrer_policy: str = "strict-origin-when-cross-origin"

    def validate(self):
        # Validate API key settings if required
        if self.require_api_key:
            if self.default_key_expiry_days is not None and self.default_key_expiry_days <= 0:
                raise ConfigurationError("default_key_expiry_days must be a positive integer or None")

            if self.warn_expiring_days <= 0:
                raise ConfigurationError("warn_expiring_days must be a positive integer")

        # Validate key database path if specified
        if self.key_database_path:
            valid_key_db_path = True
            reason = ""
            try:
                key_path = Path(self.key_database_path)
            except ValueError:
                valid_key_db_path = False
                reason = "Must use valid key database path"
            else:
                if not os.path.exists(key_path.parent):
                    valid_key_db_path = False
                    reason = f"`key_database_path` parent directory does not exist: {str(key_path.parent)}"
                elif key_path.exists() and not os.access(key_path, os.W_OK):
                    valid_key_db_path = False
                    reason = f"Cannot write to `key_database_path`: {str(key_path)}"
                elif not os.access(key_path.parent, os.W_OK):
                    valid_key_db_path = False
                    reason = "key_database_path cannot access"

            if not valid_key_db_path:
                raise ConfigurationError(f"Invalid key_database_path: {reason}")

            # Validate CORS allowed origins
        if isinstance(self.cors_allowed_origins, str):
            if self.cors_allowed_origins != "*" and not self.cors_allowed_origins:
                raise ConfigurationError("cors_allowed_origins string must be '*' or a non-empty string")
        elif isinstance(self.cors_allowed_origins, list):
            if len(self.cors_allowed_origins) == 0:
                raise ConfigurationError("cors_allowed_origins list cannot be empty")
            for origin in self.cors_allowed_origins:
                if not isinstance(origin, str) or not origin:
                    raise ConfigurationError("Each origin in cors_allowed_origins must be a non-empty string")
        else:
            raise ConfigurationError("cors_allowed_origins must be either a string or a list of strings")

        # Validate trusted_hosts patterns if specified
        if self.trusted_hosts is not None:
            if not isinstance(self.trusted_hosts, list):
                raise ConfigurationError("trusted_hosts must be a list of strings")

            if len(self.trusted_hosts) == 0:
                raise ConfigurationError("trusted_hosts list cannot be empty (use None to disable)")

            # Import here to avoid circular imports
            try:
                from localvectordb_server.utils.hostmatch import validate_trusted_host_patterns

                validation_errors = validate_trusted_host_patterns(self.trusted_hosts)
                if validation_errors:
                    error_msg = "Invalid trusted_hosts patterns:\n" + "\n".join(
                        f"  - {error}" for error in validation_errors
                    )
                    raise ConfigurationError(error_msg)
            except ImportError:
                # Fallback validation if hostmatch module not available
                for i, pattern in enumerate(self.trusted_hosts):
                    if not isinstance(pattern, str) or not pattern.strip():
                        raise ConfigurationError(f"trusted_hosts[{i}] must be a non-empty string") from None

        return True


@dataclass
class ServerSettings(BaseSettings):
    """Settings related to the flask API server"""

    debug: bool = False
    environment: str = "development"

    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Turn on to accept various file formats with upload route
    file_upload_enabled: bool = False
    max_request_size: int = 10 * 1024 * 1024  # 10MB default

    # Feature flags
    # enable_async_processing: bool = True   - not yet implemented
    enable_structured_logging: Optional[bool] = None  # Computed in __post_init__ based on debug
    enable_performance_logging: bool = False
    enable_rate_limiting: bool = False
    # Settings for Flask-Limiter
    rate_limit: str = "100 per minute"
    # Can also provide a redis url
    rate_limit_storage_uri: str = "memory://"

    # Cache settings
    cache_enabled: bool = False
    cache_ignore_errors: bool = True
    cache_timeout: int = 300  # 5 min
    cache_key_prefix: str = "lvdb_cache_"
    # Which cachelib cache to use: https://cachelib.readthedocs.io/en/stable/
    cache_type: Literal[
        "SimpleCache",
        "RedisCache",
        "FileSystemCache",
        "MemcachedCache",
        "UWSGICache",
        "DynamoDbCache",
        "MongoDbCache",
        "NullCache",
    ] = "SimpleCache"
    # Contains the keyword-arguments passed to the cache constructor. See cachelib docs for details.
    # DEPRECATED: Use specific cache backend configurations below instead
    cache_settings: Optional[dict] = None

    # Redis cache specific settings (used when cache_type = "RedisCache")
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_url: Optional[str] = None  # If set, overrides individual redis_* settings
    redis_socket_timeout: Optional[float] = None
    redis_socket_connect_timeout: Optional[float] = None

    # FileSystem cache specific settings (used when cache_type = "FileSystemCache")
    filesystem_cache_dir: Optional[str] = None
    filesystem_cache_threshold: int = 500

    # Memcached cache specific settings (used when cache_type = "MemcachedCache")
    memcached_servers: List[str] = field(default_factory=lambda: ["127.0.0.1:11211"])
    memcached_username: Optional[str] = None
    memcached_password: Optional[str] = None

    # Database registry settings for multi-worker coordination
    db_registry_type: Literal[
        "SimpleCache",
        "RedisCache",
        "FileSystemCache",
        "MemcachedCache",
        "UWSGICache",
        "DynamoDbCache",
        "MongoDbCache",
        "NullCache",
    ] = "SimpleCache"

    # Will try to use the cache_settings if not set and cache_types match.
    db_registry_settings: Optional[dict] = None

    # Set to True to use the same cache for db_registry as general cache.
    use_single_cache: bool = False

    proxy_enabled: bool = False
    # Forwarded-header handling for deployments behind a reverse proxy. When
    # proxy_enabled is True, HostValidationMiddleware (see app.py) honors the
    # X-Forwarded-Host header from trusted_proxies. Keys: x_for, x_proto,
    # x_host, x_port, x_prefix.
    proxy_settings: Optional[dict] = None

    # List of trusted proxy IP addresses/CIDR blocks that are allowed to set forwarded headers
    # Required when proxy_enabled=True for security
    trusted_proxies: List[str] = field(default_factory=list)

    # Security settings
    security: SecuritySettings = field(default_factory=SecuritySettings)

    def __post_init__(self):
        """Compute fields that depend on other instance fields."""
        # Set enable_structured_logging based on debug mode if not explicitly set
        if self.enable_structured_logging is None:
            self.enable_structured_logging = not self.debug

    def validate(self):
        # Validate host
        if not isinstance(self.host, str) or not self.host:
            raise ConfigurationError("host must be a non-empty string")

        # Validate port is in allowed range
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ConfigurationError("port must be an integer between 1 and 65535")

        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")

        # Validate log format
        if not isinstance(self.log_format, str) or not self.log_format:
            raise ConfigurationError("log_format must be a non-empty string")

        # Validate performance settings
        if self.max_request_size <= 0:
            raise ConfigurationError("max_request_size must be a positive integer")

        if self.proxy_enabled:
            if not isinstance(self.proxy_settings, dict):
                raise ConfigurationError(
                    "If `proxy_enabled` is True, `proxy_settings` must be a dict containing "
                    "one or more of the following keys: x_for, x_proto, x_host, x_prefix"
                )

            # Require trusted_proxies when proxy is enabled for security
            if not self.trusted_proxies:
                raise ConfigurationError(
                    "When `proxy_enabled` is True, `trusted_proxies` must be configured with "
                    "a list of trusted proxy IP addresses or CIDR blocks for security"
                )

        # Validate trusted_proxies format
        if self.trusted_proxies:
            import ipaddress

            for proxy in self.trusted_proxies:
                try:
                    # Validate IP address or CIDR block format
                    ipaddress.ip_network(proxy, strict=False)
                except ValueError as e:
                    raise ConfigurationError(f"Invalid IP address or CIDR block in trusted_proxies: {proxy}") from e

        if self.cache_enabled:
            valid_cache_types = (
                "SimpleCache",
                "RedisCache",
                "FileSystemCache",
                "MemcachedCache",
                "UWSGICache",
                "DynamoDbCache",
                "MongoDbCache",
                "NullCache",
            )

            # Validate cache type
            if self.cache_type not in valid_cache_types:
                raise ConfigurationError(f"cache_type must be one of: {valid_cache_types}")

            # Backend-specific validation
            if self.cache_type == "RedisCache":
                if self.redis_url and (
                    self.redis_host != "localhost" or self.redis_port != 6379 or self.redis_password
                ):
                    raise ConfigurationError(
                        "When redis_url is set, do not set individual redis_host/port/password settings"
                    )
                if not self.redis_url and not self.redis_host:
                    raise ConfigurationError("Redis cache requires either redis_url or redis_host to be configured")

            elif self.cache_type == "FileSystemCache":
                if not self.filesystem_cache_dir:
                    raise ConfigurationError("FileSystem cache requires filesystem_cache_dir to be configured")
                if self.filesystem_cache_threshold <= 0:
                    raise ConfigurationError("filesystem_cache_threshold must be a positive integer")

            elif self.cache_type == "MemcachedCache":
                if not self.memcached_servers:
                    raise ConfigurationError("Memcached cache requires memcached_servers to be configured")
                for server in self.memcached_servers:
                    if not isinstance(server, str) or ":" not in server:
                        raise ConfigurationError(f"Invalid memcached server format: {server}. Use 'host:port' format")

        # Validate security settings
        self.security.validate()

        return True


@dataclass
class Config:
    """Main configuration container with v1.0 enhancements."""

    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    backup: BackupSettings = field(default_factory=BackupSettings)
    migration: MigrationSettings = field(default_factory=MigrationSettings)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)

    def validate(self):
        return (
            self.database.validate()
            and self.embedding.validate()
            and self.server.validate()
            and self.backup.validate()
            and self.migration.validate()
            and self.extraction.validate()
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from file with v1.0 enhancements."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        if file_path.suffix.lower() == ".toml":
            return cls._from_toml(file_path)
        elif file_path.suffix.lower() == ".json":
            return cls._from_json(file_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary with v1.0 support."""
        if not isinstance(data, dict):
            raise TypeError("Configuration `data` must be a dictionary containing configuration data.")
        if not data:
            raise ConfigurationError("Configuration `data` is empty!")

        # Create config with default values
        config = cls()

        # Process database settings
        if "database" in data and isinstance(data["database"], dict):
            db_data = data["database"]
            for key, value in db_data.items():
                if key == "default_metadata_schema" and isinstance(value, dict):
                    # Parse metadata schema
                    schema = {}
                    for field_name, field_config in value.items():
                        if isinstance(field_config, dict):
                            schema[field_name] = MetadataField(**field_config)
                        else:
                            schema[field_name] = MetadataField(type=str(field_config))
                    setattr(config.database, key, schema)
                elif hasattr(config.database, key):
                    setattr(config.database, key, value)

        # Process embedding settings
        if "embedding" in data and isinstance(data["embedding"], dict):
            for key, value in data["embedding"].items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)

        # Process server settings
        if "server" in data and isinstance(data["server"], dict):
            server_data = data["server"]
            for key, value in server_data.items():
                if key == "security" and isinstance(value, dict):
                    # Handle nested security settings
                    for sec_key, sec_value in value.items():
                        if hasattr(config.server.security, sec_key):
                            setattr(config.server.security, sec_key, sec_value)
                elif hasattr(config.server, key):
                    setattr(config.server, key, value)

        # Process backup settings
        if "backup" in data and isinstance(data["backup"], dict):
            for key, value in data["backup"].items():
                if hasattr(config.backup, key):
                    setattr(config.backup, key, value)

        # Process migration settings
        if "migration" in data and isinstance(data["migration"], dict):
            for key, value in data["migration"].items():
                if hasattr(config.migration, key):
                    setattr(config.migration, key, value)

        # Process extraction settings
        if "extraction" in data and isinstance(data["extraction"], dict):
            for key, value in data["extraction"].items():
                if hasattr(config.extraction, key):
                    setattr(config.extraction, key, value)

        return config

    @classmethod
    def _from_toml(cls, path: Path) -> "Config":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data: Dict[str, Any] = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def _from_json(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, base=None, prefix: str = "LVDB_") -> "Config":
        """Load configuration from environment variables with v1.0 support."""
        config = copy.deepcopy(base) or cls()

        # Enhanced environment variable processing
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            name = env_name[len(prefix) :].lower()
            parts = name.split("_", 2)  # Allow for deeper nesting

            if len(parts) >= 2 and parts[0] in ["database", "embedding", "server", "backup", "migration", "extraction"]:
                section_name = parts[0]
                key = "_".join(parts[1:])
                section_obj = getattr(config, section_name)

                # Handle nested server.security settings
                if section_name == "server" and len(parts) >= 3 and parts[1] == "security":
                    security_key = "_".join(parts[2:])
                    security_obj = section_obj.security
                    if hasattr(security_obj, security_key):
                        value = cls._convert_env_value(env_value, security_obj, security_key)
                        setattr(security_obj, security_key, value)
                    continue

                # Convert environment variable value to the appropriate type
                value = cls._convert_env_value(env_value, section_obj, key)

                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

        return config

    @staticmethod
    def _convert_env_value(value: str, obj: Any, key: str) -> Any:
        """Convert environment variable string to appropriate type based on type hints.

        Parameters
        ----------
        value : str
            The environment variable value as a string.
        obj : Any
            The object containing the attribute to set.
        key : str
            The attribute name.

        Returns
        -------
        Any
            The converted value in the appropriate type.
        """
        if not hasattr(obj, key):
            return value

        # Get the expected type for this attribute
        hints = get_type_hints(obj.__class__)
        if key not in hints:
            return value

        target_type = hints[key]
        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle basic types first (these have no origin)
        if target_type is bool:
            return value.lower() in ["true", "yes", "1", "on"]
        elif target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is str:
            return value

        # Handle Union types (including Optional which is Union[T, None])
        elif origin is Union:
            # Filter out NoneType to get actual types
            non_none_types = [arg for arg in args if arg is not type(None)]

            # Special case: Union[str, List[str]] for cors_allowed_origins
            # Use get_origin() to handle typing system changes in Python 3.11+
            is_str_list_union = False
            if len(non_none_types) == 2:
                has_str = str in non_none_types
                has_list_str = any(get_origin(arg) is list for arg in non_none_types)
                is_str_list_union = has_str and has_list_str

            if is_str_list_union:
                # Try to parse as JSON array first
                if value.startswith("[") and value.endswith("]"):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Fallback to comma-separated list
                        items = value[1:-1].split(",")
                        return [item.strip(" \"'") for item in items if item.strip()]
                # If it contains commas or semicolons, treat as list
                elif "," in value or ";" in value:
                    # Support both comma and semicolon delimiters
                    delimiter = ";" if ";" in value else ","
                    return [item.strip(" \"'") for item in value.split(delimiter) if item.strip()]
                # Otherwise, return as string
                else:
                    return value

            # Handle Optional[T] (Union[T, None])
            elif len(non_none_types) == 1:
                inner_type = non_none_types[0]
                # Check if the value represents None
                if value.lower() in ["none", "null", ""]:
                    return None

                # Convert using the inner type
                inner_origin = get_origin(inner_type)

                if inner_type is bool:
                    return value.lower() in ["true", "yes", "1", "on"]
                elif inner_type is int:
                    return int(value)
                elif inner_type is float:
                    return float(value)
                elif inner_type is str:
                    return value
                elif inner_origin is list:
                    # Handle Optional[List[str]]
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            items = value[1:-1].split(",")
                            return [item.strip(" \"'") for item in items if item.strip()]
                    else:
                        return [item.strip(" \"'") for item in value.split(",") if item.strip()]
                elif inner_origin is dict:
                    # Handle Optional[dict]
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return {}
                else:
                    # For other types, just return the value
                    return value

            # Other Union types - not expected in our config
            else:
                # Default to string
                return value

        # Handle List types
        elif origin is list:
            if value.startswith("[") and value.endswith("]"):
                # Handle JSON array format
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Fallback to comma-separated
                    items = value[1:-1].split(",")
                    return [item.strip(" \"'") for item in items if item.strip()]
            else:
                # Handle comma-separated format
                return [item.strip(" \"'") for item in value.split(",") if item.strip()]

        # Handle Dict types
        elif origin is dict or target_type is dict:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Return empty dict if parsing fails
                return {}

        # Handle Literal types (e.g., Literal["gzip", "lzma", "none"])
        elif origin is Literal:
            # Literal values are strings in our config, just return the value
            # Validation will happen elsewhere
            return value

        # Default: return as-is
        return value

    @classmethod
    def update_from_dict(cls, config: "Config", update_map: dict) -> "Config":
        new_cfg = cls()

        for key, values in update_map.items():
            if not isinstance(values, dict):
                raise ValueError("Expected dict of dicts for `update_from_dict`.")

            cfg_obj: BaseSettings
            if key == "database":
                cfg_obj = new_cfg.database
            elif key == "embedding":
                cfg_obj = new_cfg.embedding
            elif key == "server":
                cfg_obj = new_cfg.server
            elif key == "backup":
                cfg_obj = new_cfg.backup
            elif key == "migration":
                cfg_obj = new_cfg.migration
            elif key == "extraction":
                cfg_obj = new_cfg.extraction
            else:
                raise KeyError(
                    "Expected keys: 'database', 'embedding', 'server', 'backup', 'migration', "
                    f"'extraction', found: {key}"
                )

            for cfg_key, cfg_value in values.items():
                if hasattr(cfg_obj, cfg_key):
                    setattr(cfg_obj, cfg_key, cfg_value)
                else:
                    raise KeyError(f"Configuration setting `{cfg_key}` does not exist.")

        return config.merge(new_cfg)

    def to_flask_config(self) -> Dict[str, Any]:
        """Convert to format expected by Flask app.config with v1.0 support."""
        result = {}

        # Database settings with DB_ prefix (maintaining backward compatibility)
        for key, value in asdict(self.database).items():
            if key == "default_metadata_schema":
                continue
            else:
                result[f"DB_{key.upper()}"] = value

        # Embedding settings with EMBEDDING_ prefix
        for key, value in asdict(self.embedding).items():
            result[f"EMBEDDING_{key.upper()}"] = value

        # Backup settings with BACKUP_ prefix
        for key, value in asdict(self.backup).items():
            result[f"BACKUP_{key.upper()}"] = value

        # Migration settings with MIGRATION_ prefix
        for key, value in asdict(self.migration).items():
            result[f"MIGRATION_{key.upper()}"] = value

        # Generate Flask-Caching compatible configuration
        cache_config = self._generate_cache_config()

        # Server settings (maintain existing names for backward compatibility)
        result.update(
            {
                "DEBUG": self.server.debug,
                "ENVIRONMENT": self.server.environment,
                "DB_ROOT_DIR": self.database.root_dir,
                "LOG_LEVEL": self.server.log_level,
                "LOG_FORMAT": self.server.log_format,
                "REQUIRE_API_KEY": self.server.security.require_api_key,
                "CORS_ENABLED": self.server.security.cors_enabled,
                "CORS_ALLOWED_ORIGINS": self.server.security.cors_allowed_origins,
                "MAX_CONTENT_LENGTH": self.server.max_request_size,
                "AUTH_LOG_LEVEL": self.server.security.auth_log_level,
                "TRUSTED_HOSTS": self.server.security.trusted_hosts,
                "TRUSTED_PROXIES": self.server.trusted_proxies,
                "CACHE_IGNORE_ERRORS": self.server.cache_ignore_errors,
                "CACHE_NO_NULL_WARNING": not self.server.cache_enabled,
                "CACHE_KEY_PREFIX": self.server.cache_key_prefix,
            }
        )

        # Add cache configuration
        result.update(cache_config)

        return result

    def _generate_cache_config(self) -> Dict[str, Any]:
        """Generate Flask-Caching compatible configuration based on cache type and settings"""
        cache_config: Dict[str, Any] = {
            "CACHE_TYPE": "NullCache" if not self.server.cache_enabled else self.server.cache_type,
            "CACHE_DEFAULT_TIMEOUT": self.server.cache_timeout,
        }

        if not self.server.cache_enabled:
            return cache_config

        # Generate backend-specific configuration
        if self.server.cache_type == "RedisCache":
            if self.server.redis_url:
                cache_config["CACHE_REDIS_URL"] = self.server.redis_url
            else:
                cache_config["CACHE_REDIS_HOST"] = self.server.redis_host
                cache_config["CACHE_REDIS_PORT"] = self.server.redis_port
                if self.server.redis_password:
                    cache_config["CACHE_REDIS_PASSWORD"] = self.server.redis_password
                cache_config["CACHE_REDIS_DB"] = self.server.redis_db

            # Add socket timeout options if configured
            if self.server.redis_socket_timeout is not None:
                cache_config["CACHE_OPTIONS"] = cache_config.get("CACHE_OPTIONS", {})
                cache_config["CACHE_OPTIONS"]["socket_timeout"] = self.server.redis_socket_timeout
            if self.server.redis_socket_connect_timeout is not None:
                cache_config["CACHE_OPTIONS"] = cache_config.get("CACHE_OPTIONS", {})
                cache_config["CACHE_OPTIONS"]["socket_connect_timeout"] = self.server.redis_socket_connect_timeout

        elif self.server.cache_type == "FileSystemCache":
            if self.server.filesystem_cache_dir:
                cache_config["CACHE_DIR"] = self.server.filesystem_cache_dir
            cache_config["CACHE_THRESHOLD"] = self.server.filesystem_cache_threshold

        elif self.server.cache_type == "MemcachedCache":
            cache_config["CACHE_MEMCACHED_SERVERS"] = self.server.memcached_servers
            if self.server.memcached_username:
                cache_config["CACHE_MEMCACHED_USERNAME"] = self.server.memcached_username
            if self.server.memcached_password:
                cache_config["CACHE_MEMCACHED_PASSWORD"] = self.server.memcached_password

        return cache_config

    def generate_toml(self) -> str:
        """Generate enhanced TOML configuration for v1.0 using tomli-w."""

        def clean_none_values(obj):
            """Recursively remove None values and empty containers from nested dicts/lists."""
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if value is not None:
                        cleaned_value = clean_none_values(value)
                        # Only include non-empty containers and non-None values
                        if cleaned_value is not None and (not isinstance(cleaned_value, (dict, list)) or cleaned_value):
                            cleaned[key] = cleaned_value
                return cleaned
            elif isinstance(obj, list):
                cleaned = []
                for item in obj:
                    cleaned_item = clean_none_values(item)
                    if cleaned_item is not None:
                        cleaned.append(cleaned_item)
                return cleaned
            else:
                return obj

        # Convert config to dict
        config_dict = clean_none_values(self.to_dict())

        # Process metadata schema for proper TOML serialization
        if config_dict.get("database", {}).get("default_metadata_schema"):
            metadata_schema = config_dict["database"]["default_metadata_schema"]
            processed_schema = {}
            for field_name, field_config in metadata_schema.items():
                if hasattr(field_config, "__dict__"):
                    # Convert MetadataField object to dict
                    field_dict = clean_none_values(asdict(field_config))
                    # Convert enum to string if needed
                    if hasattr(field_dict.get("type"), "value"):
                        field_dict["type"] = field_dict["type"].value
                    elif hasattr(field_dict.get("type"), "__str__"):
                        field_dict["type"] = str(field_dict["type"])
                    processed_schema[field_name] = field_dict
                elif isinstance(field_config, dict):
                    # Handle already converted dict
                    processed_field = clean_none_values(field_config.copy())
                    if hasattr(processed_field.get("type"), "value"):
                        processed_field["type"] = processed_field["type"].value
                    elif hasattr(processed_field.get("type"), "__str__"):
                        processed_field["type"] = str(processed_field["type"])
                    processed_schema[field_name] = processed_field

            # Update the config dict
            config_dict["database"]["default_metadata_schema"] = processed_schema

        # Handle security nested structure - flatten for TOML
        if "server" in config_dict and "security" in config_dict["server"]:
            # Create server.security section
            security_config = config_dict["server"].pop("security")
            config_dict.setdefault("server", {})["security"] = security_config

        # Generate TOML with header comment
        output = BytesIO()
        output.write("# LocalVectorDB Server Configuration v1.0\n\n".encode("utf-8"))

        # Use tomli_w to dump the config
        tomli_w.dump(config_dict, output)

        return output.getvalue().decode("utf-8")

    def to_dict(self):
        return {
            "database": asdict(self.database),
            "embedding": asdict(self.embedding),
            "server": asdict(self.server),
            "backup": asdict(self.backup),
            "migration": asdict(self.migration),
            "extraction": asdict(self.extraction),
        }

    def apply_common_schema(self, schema_name: str):
        """Apply a predefined metadata schema."""
        common_schemas = get_common_metadata_schemas()
        if schema_name in common_schemas:
            schema_value = common_schemas[schema_name]
            if isinstance(schema_value, dict):
                # When called without args, values are dict[str, MetadataField]
                parsed_schema: Dict[str, MetadataField] = {}
                for k, v in schema_value.items():
                    if isinstance(v, MetadataField):
                        parsed_schema[k] = v
                self.database.default_metadata_schema = parsed_schema
            else:
                raise ConfigurationError(f"Unexpected schema type for '{schema_name}'")
        else:
            available = ", ".join(common_schemas.keys())
            raise ConfigurationError(f"Unknown schema '{schema_name}'. Available: {available}")

    def merge(self, other: "Config") -> "Config":
        """Enhanced merge that respects non-default values in base config.

        Only applies override values if they are explicitly different from defaults,
        preserving custom base config values when override contains defaults.
        """
        result = Config()

        # Helper function to merge dataclass fields intelligently
        def merge_dataclass(base: Any, override: Any, result_obj: Any) -> None:
            # Get default instance for comparison
            base_type = type(base)
            default_instance = base_type()

            for key in asdict(override).keys():
                base_value = getattr(base, key)
                override_value = getattr(override, key)

                try:
                    default_value = getattr(default_instance, key)
                except AttributeError:
                    # Fallback if field doesn't exist in default
                    setattr(result_obj, key, override_value)
                    continue

                # Special handling for metadata schema
                if key == "default_metadata_schema" and isinstance(override_value, dict):
                    if not override_value:
                        # Use base value if override is empty
                        setattr(result_obj, key, base_value)
                    else:
                        # Merge schemas - base takes precedence for conflicting keys
                        merged_schema = base_value.copy()
                        merged_schema.update(override_value)
                        setattr(result_obj, key, merged_schema)
                # Handle nested dataclasses recursively
                elif (
                    is_dataclass(base_value)
                    and is_dataclass(override_value)
                    and type(base_value) is type(override_value)
                ):
                    # Create a new instance of the dataclass and recursively merge
                    nested_cls: Any = type(base_value)
                    nested_result = nested_cls()
                    merge_dataclass(base_value, override_value, nested_result)
                    setattr(result_obj, key, nested_result)
                elif isinstance(override_value, (list, dict)):
                    # For container types, check if they're empty (default)
                    if not override_value:
                        # Use the base value if override is empty
                        setattr(result_obj, key, base_value)
                    else:
                        setattr(result_obj, key, override_value)
                else:
                    # Smart merge logic:
                    # - If override is default but base is not default: keep base value
                    # - Otherwise: use override value (either it's non-default, or both are default)
                    try:
                        if override_value == default_value and base_value != default_value:
                            setattr(result_obj, key, base_value)
                        else:
                            setattr(result_obj, key, override_value)
                    except (TypeError, ValueError):
                        # If comparison fails (e.g., unhashable types), use override
                        setattr(result_obj, key, override_value)

        # Merge all sections
        merge_dataclass(self.database, other.database, result.database)
        merge_dataclass(self.embedding, other.embedding, result.embedding)
        merge_dataclass(self.server, other.server, result.server)
        merge_dataclass(self.backup, other.backup, result.backup)
        merge_dataclass(self.migration, other.migration, result.migration)
        merge_dataclass(self.extraction, other.extraction, result.extraction)

        return result


def load_config(
    configuration: Union[str, Config, None] = None,
    validate: bool = True,
    verbose: bool = False,
    apply_schema: Optional[str] = None,
) -> Config:
    """Load and construct a Config object from various sources.

    This function provides a flexible interface for creating a Config instance
    using one or more of the supported input mechanisms (explicit Config
    instance, dict, configuration file path, environment variables, and
    predefined metadata schemas). The resulting Config will start from the
    library defaults, optionally have a common metadata schema applied, be
    merged with the provided configuration source, have environment variables
    applied, and can be validated before being returned.

    Parameters
    ----------
    configuration : Union[str, Config, None], optional
        Source of configuration to load. May be:

        - a path to a configuration file (TOML, JSON, or INI), in which case the
          file will be loaded if it exists;
        - a Config instance, which will be merged into defaults;
        - a dict mapping section names ("database", "embedding", "server",
          "backup", "migration") to dicts of values for those sections;
        - None, in which case defaults are used and the LVDB_SERVER_CONFIG
          environment variable is consulted for a file path.
    validate : bool, optional
        If True (default), call Config.validate() on the resulting configuration
        and raise ConfigurationError on validation failures.
    verbose : bool, optional
        If True, print progress/debug messages to stderr using click.secho.
        Defaults to False.
    apply_schema : Optional[str], optional
        If provided, the named common metadata schema (from
        get_common_metadata_schemas()) will be applied to
        database.default_metadata_schema before other configuration sources
        are merged.

    Returns
    -------
    Config
        The constructed (and optionally validated) configuration object.

    Raises
    ------
    ConfigurationError
        If loading or validation fails.
    FileNotFoundError
        If attempting to load from a configuration file path that does not
        exist.
    ValueError
        If a provided configuration file has an unsupported suffix/format.
    TypeError
        If `configuration` is not a supported type.
    """

    # Start with default config
    config = Config()

    # Apply common schema if requested
    if apply_schema:
        config.apply_common_schema(apply_schema)

    if isinstance(configuration, Config):
        if verbose:
            click.secho("Loading from `Config` obj in args", fg="blue", err=True)
        config = config.merge(configuration)
    elif isinstance(configuration, dict):
        if verbose:
            click.secho("Loading from `dict` in args", fg="blue", err=True)
        config = Config.update_from_dict(config, configuration)
    elif isinstance(configuration, str) or configuration is None:
        # Try to load config file
        if not configuration:
            # Check environment variable
            configuration = os.getenv("LVDB_SERVER_CONFIG")

        if configuration and os.path.exists(configuration):
            if verbose:
                click.secho(f"Loading configuration from {os.path.abspath(configuration)}", fg="blue", err=True)
            try:
                file_config = Config.from_file(configuration)
                config = config.merge(file_config)
            except Exception as e:
                click.secho(f"Error loading config file: {e}", fg="bright_red", err=True)
                raise ConfigurationError(f"Failed to load configuration file: {str(repr(e))}") from e
        else:
            if verbose:
                click.secho("No config file provided, using default configuration", fg="blue", err=True)
    else:
        if verbose:
            click.secho(f"Loading from `{type(configuration)}` obj in args", fg="blue", err=True)
        obj_config = dict(configuration.__dict__)
        Config.update_from_dict(config, obj_config)

    # Apply environment variables
    config = Config.from_env(config)

    if validate:
        config.validate()

    return config
