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
    - MigrationSettings: Settings for handling v1.x to v1.0 migrations
    - Config: Main configuration container with methods for loading and saving

New v1.0 Features:
    - Default metadata schemas for common use cases
    - Embedding provider plugin configuration
    - Connection pooling settings
    - Performance optimization flags
    - Automatic migration detection and handling

Environment Variables:
    All v1.x environment variables are supported, plus new v1.0 variables:
    - LVDB_DATABASE_DEFAULT_METADATA_SCHEMA
    - LVDB_DATABASE_CONNECTION_POOL_SIZE
    - LVDB_DATABASE_ENABLE_GPU
    - LVDB_DATABASE_ENABLE_FTS
    - LVDB_EMBEDDING_PROVIDER
    - LVDB_EMBEDDING_BASE_URL
    - LVDB_EMBEDDING_API_KEY
    - LVDB_MIGRATION_AUTO_DETECT
    - LVDB_MIGRATION_BACKUP_ON_MIGRATE

Example v1.0 Configuration::

.. code-block: toml

    [database]
    root_dir = "./.lvdb"
    timeout = 300
    connection_pool_size = 10
    enable_gpu = false
    enable_fts = true

    # Default settings for new databases
    chunk_size = 500
    chunk_overlap = 1
    chunking_method = "sentences"

    [database.metadata_schema]
    title = {type = "text", indexed = true}
    author = {type = "text", indexed = true}
    date = {type = "date", indexed = true}
    tags = {type = "json"}

    [embedding]
    provider = "ollama"
    model = "nomic-embed-text"
    base_url = "http://localhost:11434"
    batch_size = 64
    timeout = 30

    [server]
    host = "127.0.0.1"
    port = 5000
    log_level = "INFO"

    [server.security]
    require_api_key = false
    authorized_api_keys = []
    cors_enabled = true
    cors_allowed_origins = "*"

    [migration]
    auto_detect = true
    backup_on_migrate = true
    backup_dir = "./backups"

"""
import copy
#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, get_type_hints

import click

from localvectordb.core import MetadataField, get_common_metadata_schemas
from localvectordb.embeddings import EmbeddingRegistry
from localvectordb.exceptions import ConfigurationError
from localvectordb.chunking import ChunkerFactory


@dataclass
class EmbeddingSettings:
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
class DatabaseSettings:
    """Enhanced settings for database operations with v1.0 support."""
    root_dir: str = "./.lvdb"
    timeout: int = 300  # seconds
    connection_pool_size: int = 10
    enable_gpu: bool = False
    enable_fts: bool = True

    # Performance settings
    embeddings_batch_size: int = 64  # Deprecated: use EmbeddingSettings.batch_size
    auto_save_interval: int = 300  # Auto-save interval in seconds (0 = disabled)

    # Default database parameters when creating new ones
    chunk_size: int = 500  # Renamed from chunk_tokens for v1.0
    chunk_overlap: int = 1
    embedding_model: str = "nomic-embed-text"  # Deprecated: use EmbeddingSettings
    provider: str = "ollama"  # Deprecated: use EmbeddingSettings
    chunking_method: str = "sentences"  # Renamed from chunk_method for v1.0

    # Default metadata schema for new databases
    default_metadata_schema: Dict[str, MetadataField] = field(default_factory=dict)

    # Migration settings
    migration_auto_detect: bool = True
    migration_backup_on_migrate: bool = True
    migration_backup_dir: str = "./backups"

    def validate(self):
        # Validate root_dir
        if not isinstance(self.root_dir, str) or not self.root_dir:
            raise ConfigurationError("root_dir must be a non-empty string")

        # Validate numeric settings
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ConfigurationError("timeout must be a positive integer")

        if not isinstance(self.connection_pool_size, int) or self.connection_pool_size <= 0:
            raise ConfigurationError("connection_pool_size must be a positive integer")

        if not isinstance(self.embeddings_batch_size, int) or self.embeddings_batch_size <= 0:
            raise ConfigurationError("embeddings_batch_size must be a positive integer")

        if not isinstance(self.auto_save_interval, int) or self.auto_save_interval < 0:
            raise ConfigurationError("auto_save_interval must be a non-negative integer")

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be a non-negative integer")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")

        if not isinstance(self.chunking_method, str) or self.chunking_method not in ChunkerFactory.list_methods():
            raise ConfigurationError(f"chunking_method must be one of {ChunkerFactory.list_methods()}")

        if not isinstance(self.provider, str) or self.provider.lower() not in EmbeddingRegistry.list():
            raise ConfigurationError(f"provider must be one of: {EmbeddingRegistry.list()}")

        if not isinstance(self.embedding_model, str) or not self.embedding_model:
            raise ConfigurationError("embedding_model must be a non-empty string")

        # Validate metadata schema
        for field_name, field_config in self.default_metadata_schema.items():
            if not isinstance(field_name, str) or not field_name:
                raise ConfigurationError("Metadata field names must be non-empty strings")

        return True

    @staticmethod
    def get_common_metadata_schemas() -> Dict[str, Dict[str, MetadataField]]:
        """Get predefined metadata schemas for common use cases"""
        return get_common_metadata_schemas()


@dataclass
class ServerSettings:
    """Enhanced server-specific settings."""
    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Performance settings
    max_request_size: int = 100 * 1024 * 1024  # 100MB default
    request_timeout: int = 300  # 5 minutes default

    # Feature flags (not yet implemented)
    # enable_async_processing: bool = True
    # enable_request_logging: bool = True
    # enable_performance_metrics: bool = False

    # Security settings
    require_api_key: bool = False
    api_key_header: str = "Authorization"  # Header name for API key

    # NEW: Key Management Settings
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

        if self.request_timeout <= 0:
            raise ConfigurationError("request_timeout must be a positive integer")

        # if self.worker_count is not None and self.worker_count <= 0:
        #     raise ConfigurationError("worker_count must be a positive integer or None")

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
        elif path.suffix.lower() in ['.yaml', '.yml']:
            return cls._from_yaml(path)
        elif path.suffix.lower() == '.json':
            return cls._from_json(path)
        elif path.suffix.lower() in ['.ini', '.cfg']:
            return cls._from_ini(path)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary with v1.0 support."""
        if not data or not isinstance(data, dict):
            raise TypeError("Configuration `data` must be a dictionary")

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
    def _from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration. Install with: pip install pyyaml")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def _from_json(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        import json
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
            # 'migration': config.migration
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

            if env_name == f"{prefix}SERVER_CONFIG":
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
                if first_type == bool:
                    return value.lower() in ['true', 'yes', '1', 'on']
                elif first_type == int:
                    return int(value)
                elif first_type == float:
                    return float(value)
                else:
                    return value
        # Handle List[str]
        elif hasattr(target_type, '__origin__') and target_type.__origin__ == list:
            if value.startswith('[') and value.endswith(']'):
                # Handle JSON-like format
                import json
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
            import json
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
                # Convert to serializable format
                schema_dict = {}
                for field_name, field_obj in value.items():
                    if isinstance(field_obj, MetadataField):
                        field_dict = asdict(field_obj)
                        if hasattr(field_dict['type'], 'value'):
                            field_dict['type'] = field_dict['type'].value
                        schema_dict[field_name] = field_dict
                    else:
                        schema_dict[field_name] = field_obj
                result[f"DB_{key.upper()}"] = schema_dict
            else:
                result[f"DB_{key.upper()}"] = value

        # Embedding settings with EMBEDDING_ prefix
        for key, value in asdict(self.embedding).items():
            result[f"EMBEDDING_{key.upper()}"] = value

        # Server settings (maintain existing names for backward compatibility)
        result.update({
            "LOG_LEVEL": self.server.log_level,
            "LOG_FORMAT": self.server.log_format,
            "REQUIRE_API_KEY": self.server.require_api_key,
            "CORS_ENABLED": self.server.cors_enabled,
            "CORS_ALLOWED_ORIGINS": self.server.cors_allowed_origins,
            "MAX_REQUEST_SIZE": self.server.max_request_size,
            "REQUEST_TIMEOUT": self.server.request_timeout,
            "AUTH_LOG_LEVEL": self.server.auth_log_level,
            "API_KEY_DB_PATH": self.server.key_database_path or os.path.join(self.database.root_dir, "api_keys.db"),
            "API_KEY_AUDIT_LOGGING": self.server.key_audit_logging,
            "API_KEY_HEADER": self.server.api_key_header,
            "API_KEY_PRUNE_EXPIRED": self.server.auto_prune_expired_keys
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
            else:
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
            'key_audit_logging', 'auth_log_level', 'warn_expiring_days'
        }

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

        # Migration section
        # lines.append("[migration]\n")
        # for key, value in asdict(self.migration).items():
        #     if isinstance(value, str):
        #         lines.append(f'{key} = "{value}"\n')
        #     elif isinstance(value, bool):
        #         lines.append(f'{key} = {str(value).lower()}\n')
        #     else:
        #         lines.append(f'{key} = {value}\n')

        return "".join(lines)

    def apply_common_schema(self, schema_name: str):
        """Apply a predefined metadata schema."""
        common_schemas = self.database.get_common_metadata_schemas()
        if schema_name in common_schemas:
            self.database.default_metadata_schema = common_schemas[schema_name]
        else:
            available = ", ".join(common_schemas.keys())
            raise ConfigurationError(f"Unknown schema '{schema_name}'. Available: {available}")

    def merge(self, other: "Config") -> "Config":
        """Enhanced merge."""
        result = Config()

        # Helper function to merge dataclass fields
        def merge_dataclass(base: Any, override: Any, result: Any):
            for key, value in asdict(override).items():
                override_value = getattr(override, key)
                # Special handling for metadata schema
                if key == 'default_metadata_schema' and isinstance(override_value, dict):
                    if not override_value:
                        # Use base value if override is empty
                        setattr(result, key, getattr(base, key))
                    else:
                        # Merge schemas
                        base_schema = getattr(base, key)
                        merged_schema = base_schema.copy()
                        merged_schema.update(override_value)
                        setattr(result, key, merged_schema)
                elif isinstance(override_value, (list, dict)):
                    # For container types, check if they're empty (default)
                    if not override_value:
                        # Use the base value if override is empty
                        setattr(result, key, getattr(base, key))
                    else:
                        setattr(result, key, override_value)
                else:
                    # Use the override value
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
    """Enhanced config loading with v1.0 support."""
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