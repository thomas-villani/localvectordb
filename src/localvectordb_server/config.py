# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/config.py
"""
LocalVectorDB Server Configuration Management v1.0

Enhanced configuration module supporting LocalVectorDB v1.0 features including:
- Document-first architecture configuration
- Structured metadata schema defaults
- Embedding provider configurations
- Migration settings for v1.x to v1.0
- Performance tuning options
- Connection pooling settings

Main Components:
    - DatabaseSettings: Enhanced database configuration with v1.0 features
    - EmbeddingSettings: Dedicated embedding provider configuration
    - ServerSettings: Server-specific configuration including security settings
    - Config: Main configuration container with methods for loading and saving

Environment Variables:
    - LVDB_DATABASE_DEFAULT_METADATA_SCHEMA
    - LVDB_DATABASE_CONNECTION_POOL_SIZE
    - LVDB_DATABASE_ENABLE_GPU
    - LVDB_DATABASE_ENABLE_FTS
    - LVDB_EMBEDDING_PROVIDER
    - LVDB_EMBEDDING_BASE_URL
    - LVDB_EMBEDDING_API_KEY
    - LVDB_MIGRATION_AUTO_DETECT
    - LVDB_MIGRATION_BACKUP_ON_MIGRATE

"""
import copy
import json
import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, get_type_hints, Literal

import click

from localvectordb.core import MetadataField, get_common_metadata_schemas
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
    provider: str = "ollama"  # ollama, openai
    model: str = "nomic-embed-text"
    base_url: Optional[str] = None  # Provider-specific base URL
    api_key: Optional[str] = None  # API key for providers that need it
    batch_size: int = 64
    timeout: int = 30  # Request timeout in seconds - TODO: implement this
    max_retries: int = 3  # TODO: implement this.

    # Provider-specific configurations
    config: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.provider, str) or self.provider.lower() not in ("ollama", "openai"):
            raise ConfigurationError("provider must be 'ollama' or 'openai'")

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
    faiss_index_hnsw_flat_neighbors: int = None  # Only used for IndexHNSWFlat
    faiss_index_lsh_bits: int = None  # Only used for IndexLSH, number of bits for the index.

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

        if self.faiss_index_type and self.faiss_index_type not in ("IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"):
            raise ConfigurationError("faiss_index_type must be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH")

        # Validate metadata schema
        for field_name, field_config in self.default_metadata_schema.items():
            if not isinstance(field_name, str) or not field_name:
                raise ConfigurationError("Metadata field names must be non-empty strings")

        return True


@dataclass
class ServerSettings(BaseSettings):
    """Settings related to the flask API server"""
    debug: bool = False
    environment: str = "development"

    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Turn on to accept various file formats with upload route
    file_upload_enabled: bool = False
    max_request_size: int = 100 * 1024 * 1024  # 100MB default

    # Feature flags
    # enable_async_processing: bool = True   - not yet implemented
    enable_structured_logging: bool = not debug
    enable_performance_logging: bool = False
    enable_rate_limiting: bool = False
    # Settings for Flask-Limiter
    rate_limit: str = "100 per minute"
    # Can also provide a redis url
    rate_limit_storage_uri: str = "memory://"

    # Cache settings
    cache_enabled: bool = False
    cache_ignore_errors: bool = True
    cache_timeout: int = 300   # 5 min
    cache_key_prefix: str = "lvdb_cache_"
    # Which cachelib cache to use: https://cachelib.readthedocs.io/en/stable/
    cache_type: Literal["SimpleCache", "RedisCache", "FileSystemCache",
                        "MemcachedCache", "UWSGICache", "DynamoDbCache",
                        "MongoDbCache", "NullCache"] = "SimpleCache"
    # Contains the keyword-arguments passed to the cache constructor. See cachelib docs for details.
    cache_settings: dict = None

    # Database registry settings for multi-worker coordination
    db_registry_type: Literal["SimpleCache", "RedisCache", "FileSystemCache",
                        "MemcachedCache", "UWSGICache", "DynamoDbCache",
                        "MongoDbCache", "NullCache"] = "SimpleCache"

    # Will try to use the cache_settings if not set and cache_types match.
    db_registry_settings: dict = None

    # Set to True to use the same cache for db_registry as general cache.
    use_single_cache: bool = False


    proxy_enabled: bool = False
    # These proxy settings are passed to the werkzeug ProxyFix middleware. Keys are: x_for, x_proto, x_host, x_port, x_prefix
    # read more: https://werkzeug.palletsprojects.com/en/stable/middleware/proxy_fix/
    proxy_settings: dict = None

    # Security settings
    require_api_key: bool = False
    api_key_header: str = "Authorization"  # Header name for API key

    trusted_hosts: List[str] = None

    key_database_path: Optional[str] = None  # None = auto-determined from db_root_dir
    default_key_expiry_days: Optional[int] = None  # None = no default expiration
    auto_prune_expired_keys: bool = False  # Automatically remove expired keys
    key_audit_logging: bool = True  # Log key usage for audit trails
    auth_log_level: str = "INFO"
    warn_expiring_days: int = 7  # Days before expiry to warn about

    # CORS settings
    cors_enabled: bool = True
    cors_allowed_origins: Union[str, List[str]] = "*"
    cors_allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_allowed_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization"])
    cors_max_age: int = 86400  # 24 hours


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

        # Validate API key settings if required
        if self.require_api_key:
            if self.default_key_expiry_days is not None and self.default_key_expiry_days <= 0:
                raise ConfigurationError("default_key_expiry_days must be a positive integer or None")

            if self.warn_expiring_days <= 0:
                raise ConfigurationError("warn_expiring_days must be a positive integer")

            # Validate key database path if specified
        if self.key_database_path:
            try:
                Path(self.key_database_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(f"Invalid key_database_path: {e}")

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

        if self.proxy_enabled:
            if not isinstance(self.proxy_settings, dict):
                raise ConfigurationError("If `proxy_enabled` is True, `proxy_settings` must be a dict containing "
                                         "one or more of the following keys: x_for, x_proto, x_host, x_prefix")

        if self.cache_enabled:
            if self.cache_type not in ("SimpleCache", "RedisCache", "FileSystemCache","MemcachedCache",
                                       "UWSGICache", "DynamoDbCache", "MongoDbCache"):
                raise ConfigurationError("cache_type must be ")


        return True


@dataclass
class Config:
    """Main configuration container with v1.0 enhancements."""
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    server: ServerSettings = field(default_factory=ServerSettings)

    def validate(self):
        return (
                self.database.validate() and
                self.embedding.validate() and
                self.server.validate()
        )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from file with v1.0 enhancements."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        if path.suffix.lower() == '.toml':
            return cls._from_toml(path)
        elif path.suffix.lower() == '.json':
            return cls._from_json(path)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary with v1.0 support."""
        if not isinstance(data, dict):
            raise TypeError("Configuration `data` must be a dictionary containing configuration data.")
        if not data:
            raise ConfigurationError("Configuration `data` is empty!")

        # Create config with default values
        config = cls()

        # Process database settings
        if 'database' in data and isinstance(data['database'], dict):
            db_data = data['database']
            for key, value in db_data.items():
                if key == 'default_metadata_schema' and isinstance(value, dict):
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
        if 'embedding' in data and isinstance(data['embedding'], dict):
            for key, value in data['embedding'].items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)

        # Process server settings
        if 'server' in data and isinstance(data['server'], dict):
            server_data = data['server']
            for key, value in server_data.items():
                if key == 'security' and isinstance(value, dict):
                    # Handle nested security settings
                    for sec_key, sec_value in value.items():
                        if hasattr(config.server, sec_key):
                            setattr(config.server, sec_key, sec_value)
                elif hasattr(config.server, key):
                    setattr(config.server, key, value)

        return config

    @classmethod
    def _from_toml(cls, path: Path) -> "Config":
        """Load configuration from TOML file."""
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls.from_dict(data)


    @classmethod
    def _from_json(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def _from_ini(cls, path: Path) -> "Config":
        """Load configuration from INI file with v1.0 support."""
        import configparser
        parser = configparser.ConfigParser()
        parser.read(path)

        config = cls()

        # Helper function to convert string values to appropriate types
        def convert_value(value: str, target_type):
            if target_type == bool:
                return value.lower() in ['true', 'yes', '1', 'on']
            elif target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
            elif target_type == list:
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                return value

        # Process all sections
        sections = {
            'database': config.database,
            'embedding': config.embedding,
            'server': config.server,
        }

        for section_name, section_obj in sections.items():
            if section_name in parser:
                for key, value in parser[section_name].items():
                    if hasattr(section_obj, key):
                        attr_type = type(getattr(section_obj, key))
                        parsed_value = convert_value(value, attr_type)
                        setattr(section_obj, key, parsed_value)

        # Handle nested sections like server.security
        if 'server.security' in parser:
            for key, value in parser['server.security'].items():
                if hasattr(config.server, key):
                    attr_type = type(getattr(config.server, key))
                    parsed_value = convert_value(value, attr_type)
                    setattr(config.server, key, parsed_value)

        return config


    @classmethod
    def from_env(cls, base=None, prefix: str = "LVDB_") -> "Config":
        """Load configuration from environment variables with v1.0 support."""
        config = copy.deepcopy(base) or cls()

        # Enhanced environment variable processing
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            name = env_name[len(prefix):].lower()
            parts = name.split('_', 2)  # Allow for deeper nesting

            if len(parts) >= 2 and parts[0] in ['database', 'embedding', 'server']:
                section_name = parts[0]
                key = '_'.join(parts[1:])
                section_obj = getattr(config, section_name)

                # Convert environment variable value to the appropriate type
                value = cls._convert_env_value(env_value, section_obj, key)

                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

            # Handle legacy environment variables
            elif name in ['host', 'port', 'log_level', 'root_dir']:
                legacy_mapping = {
                    'host': ('server', 'host'),
                    'port': ('server', 'port'),
                    'log_level': ('server', 'log_level'),
                    'root_dir': ('database', 'root_dir')
                }

                if name in legacy_mapping:
                    section_name, attr = legacy_mapping[name]
                    section_obj = getattr(config, section_name)
                    value = cls._convert_env_value(env_value, section_obj, attr)
                    setattr(section_obj, attr, value)

        return config

    @staticmethod
    def _convert_env_value(value: str, obj: Any, key: str) -> Any:
        """Enhanced environment variable conversion with v1.0 support."""
        if not hasattr(obj, key):
            return value

        # Get the expected type for this attribute
        hints = get_type_hints(obj.__class__)
        if key not in hints:
            return value

        target_type = hints[key]

        # Handle simple types
        if target_type == bool:
            return value.lower() in ['true', 'yes', '1', 'on']
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return value
        # Handle Optional types
        elif hasattr(target_type, '__origin__') and target_type.__origin__ is Union:
            # This is likely Optional[SomeType]
            non_none_types = [arg for arg in target_type.__args__ if arg != type(None)]
            if non_none_types:
                # Use the first non-None type for conversion
                first_type = non_none_types[0]
                second_type = non_none_types[1] if len(non_none_types) > 1 else None

                if first_type == bool:
                    return value.lower() in ['true', 'yes', '1', 'on']
                elif first_type == int:
                    return int(value)
                elif first_type == float:
                    return float(value)
                elif first_type == str and second_type == List[str]:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        if "," in value:
                            value = list(map(lambda s: s.strip(' "\''), value.split(",")))
                        return value
                elif hasattr(first_type, '__origin__') and first_type.__origin__ == List[str]:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                else:
                    return value
        # Handle List[str]
        elif hasattr(target_type, '__origin__') and target_type.__origin__ == list:
            if value.startswith('[') and value.endswith(']'):
                # Handle JSON-like format
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Fallback to comma-separated
                    items = value[1:-1].split(',')
                    return [item.strip(' "\'') for item in items if item.strip()]
            else:
                # Handle comma-separated format
                return [item.strip() for item in value.split(',') if item.strip()]
        # Handle Dict types
        elif hasattr(target_type, '__origin__') and target_type.__origin__ == dict:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}

        return value

    @classmethod
    def update_from_dict(cls, config, update_map: dict):
        new_cfg = cls()

        for key, values in update_map.items():
            if not isinstance(values, dict):
                raise ValueError("Expected dict of dicts for `update_from_dict`.")

            if key == "database":
                cfg_obj = new_cfg.database
            elif key == "embedding":
                cfg_obj = new_cfg.embedding
            elif key == "server":
                cfg_obj = new_cfg.server
            else:
                raise KeyError(f"Expected keys: 'database', 'embedding', 'server', found: {key}")

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
            if key == 'default_metadata_schema':
                continue
            else:
                result[f"DB_{key.upper()}"] = value

        # Embedding settings with EMBEDDING_ prefix
        for key, value in asdict(self.embedding).items():
            result[f"EMBEDDING_{key.upper()}"] = value


        cache_options = None
        if self.server.cache_settings:
            cache_options = {}
            for k, v in self.server.cache_settings:
                # Handle signal to load from environment variable (e.g. for redis password)
                if v[0] == "$":
                    cache_options[k] = os.getenv(v[1:])
                else:
                    cache_options[k] = v

        # Server settings (maintain existing names for backward compatibility)
        result.update({
            "DEBUG": self.server.debug,
            "ENVIRONMENT": self.server.environment,
            "DB_ROOT_DIR": self.database.root_dir,
            "LOG_LEVEL": self.server.log_level,
            "LOG_FORMAT": self.server.log_format,
            "REQUIRE_API_KEY": self.server.require_api_key,
            "CORS_ENABLED": self.server.cors_enabled,
            "CORS_ALLOWED_ORIGINS": self.server.cors_allowed_origins,
            "MAX_CONTENT_LENGTH": self.server.max_request_size,
            "AUTH_LOG_LEVEL": self.server.auth_log_level,
            "TRUSTED_HOSTS": self.server.trusted_hosts,
            "CACHE_TYPE": "NullCache" if not self.server.cache_enabled else self.server.cache_type,
            "CACHE_DEFAULT_TIMEOUT": self.server.cache_timeout,
            "CACHE_OPTIONS": cache_options,
            "CACHE_IGNORE_ERRORS": self.server.cache_ignore_errors,
            "CACHE_NO_NULL_WARNING": not self.server.cache_enabled,
            "CACHE_KEY_PREFIX": self.server.cache_key_prefix
        })

        return result

    def generate_toml(self) -> str:
        """Generate enhanced TOML configuration for v1.0."""
        lines = ["# LocalVectorDB Server Configuration v1.0\n"]

        # Database section
        lines.append("[database]\n")
        db_dict = asdict(self.database)

        # Handle metadata schema separately
        metadata_schema = db_dict.pop('default_metadata_schema', {})

        for key, value in db_dict.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"\n')
            elif isinstance(value, bool):
                lines.append(f'{key} = {str(value).lower()}\n')
            elif value is not None:
                lines.append(f'{key} = {value}\n')

        # Add metadata schema subsection
        if metadata_schema:
            lines.append("\n[database.metadata_schema]\n")
            for field_name, field_config in metadata_schema.items():
                if isinstance(field_config, dict):
                    # Format as inline table
                    props = []
                    for prop_key, prop_value in field_config.items():
                        if prop_key == 'type' and hasattr(prop_value, 'value'):
                            prop_value = prop_value.value

                        if isinstance(prop_value, str):
                            props.append(f'{prop_key} = "{prop_value}"')
                        elif isinstance(prop_value, bool):
                            props.append(f'{prop_key} = {str(prop_value).lower()}')
                        elif prop_value is not None:
                            props.append(f'{prop_key} = {prop_value}')
                    lines.append(f'{field_name} = {{ {", ".join(props)} }}\n')

        lines.append("\n")

        # Embedding section
        lines.append("[embedding]\n")
        for key, value in asdict(self.embedding).items():
            if value is None:
                continue
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"\n')
            elif isinstance(value, bool):
                lines.append(f'{key} = {str(value).lower()}\n')
            elif isinstance(value, dict):
                # Handle config dict
                if value:  # Only include if not empty
                    lines.append(f'{key} = {value}\n')
            else:
                lines.append(f'{key} = {value}\n')
        lines.append("\n")

        # Server section
        lines.append("[server]\n")
        server_dict = asdict(self.server)

        # Security fields to handle separately
        security_fields = {
            'require_api_key', 'authorized_api_keys', 'api_key_header',
            'cors_enabled', 'cors_allowed_origins', 'cors_allowed_methods',
            'cors_allowed_headers', 'cors_max_age', 'key_database_path',
            'default_key_expiry_days', 'auto_prune_expired_keys',
            'key_audit_logging', 'auth_log_level', 'warn_expiring_days', 'trusted_hosts'
        }

        # TODO: handle the new cache_settings and proxy_settings

        security_config = {}
        for key, value in list(server_dict.items()):
            if key in security_fields:
                security_config[key] = server_dict.pop(key)

        # Regular server settings
        for key, value in server_dict.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"\n')
            elif isinstance(value, bool):
                lines.append(f'{key} = {str(value).lower()}\n')
            elif isinstance(value, list):
                if value:  # Only include if not empty
                    items_str = ", ".join([f'"{item}"' if isinstance(item, str) else str(item) for item in value])
                    lines.append(f'{key} = [{items_str}]\n')
            elif value is not None:
                lines.append(f'{key} = {value}\n')

        lines.append("\n")

        # Security subsection
        if security_config:
            lines.append("[server.security]\n")
            for key, value in security_config.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    lines.append(f'{key} = {str(value).lower()}\n')
                elif isinstance(value, list):
                    if value:
                        items_str = ", ".join([f'"{item}"' if isinstance(item, str) else str(item) for item in value])
                        lines.append(f'{key} = [{items_str}]\n')
                    else:
                        lines.append(f'{key} = []\n')
                elif value is not None:
                    lines.append(f'{key} = {value}\n')
            lines.append("\n")

        return "".join(lines)

    def to_dict(self):
        return {
            "database": asdict(self.database),
            "embedding": asdict(self.embedding),
            "server": asdict(self.server)
        }

    def apply_common_schema(self, schema_name: str):
        """Apply a predefined metadata schema."""
        common_schemas = get_common_metadata_schemas()
        if schema_name in common_schemas:
            self.database.default_metadata_schema = common_schemas[schema_name]
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
        def merge_dataclass(base: Any, override: Any, result: Any):
            # Get default instance for comparison
            default_instance = type(base)()

            for key in asdict(override).keys():
                base_value = getattr(base, key)
                override_value = getattr(override, key)

                try:
                    default_value = getattr(default_instance, key)
                except AttributeError:
                    # Fallback if field doesn't exist in default
                    setattr(result, key, override_value)
                    continue

                # Special handling for metadata schema
                if key == 'default_metadata_schema' and isinstance(override_value, dict):
                    if not override_value:
                        # Use base value if override is empty
                        setattr(result, key, base_value)
                    else:
                        # Merge schemas - base takes precedence for conflicting keys
                        merged_schema = base_value.copy()
                        merged_schema.update(override_value)
                        setattr(result, key, merged_schema)
                elif isinstance(override_value, (list, dict)):
                    # For container types, check if they're empty (default)
                    if not override_value:
                        # Use the base value if override is empty
                        setattr(result, key, base_value)
                    else:
                        setattr(result, key, override_value)
                else:
                    # Smart merge logic:
                    # - If override is default but base is not default: keep base value
                    # - Otherwise: use override value (either it's non-default, or both are default)
                    try:
                        if override_value == default_value and base_value != default_value:
                            setattr(result, key, base_value)
                        else:
                            setattr(result, key, override_value)
                    except (TypeError, ValueError):
                        # If comparison fails (e.g., unhashable types), use override
                        setattr(result, key, override_value)

        # Merge all sections
        merge_dataclass(self.database, other.database, result.database)
        merge_dataclass(self.embedding, other.embedding, result.embedding)
        merge_dataclass(self.server, other.server, result.server)

        return result


def load_config(
        configuration: Union[str, Config, None] = None,
        validate: bool = True,
        verbose: bool = False,
        apply_schema: Optional[str] = None
) -> Config:
    """Smart

    Parameters
    ----------
    configuration
    validate
    verbose
    apply_schema

    Returns
    -------

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
                raise ConfigurationError(f"Failed to load configuration file: {str(repr(e))}")
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