# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_config.py
# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# tests/test_config.py
"""
Tests for localvectordb_server.config module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import ConfigurationError
from localvectordb_server.config import (
    BaseSettings, EmbeddingSettings, DatabaseSettings, ServerSettings,
    Config, load_config
)


class TestEmbeddingSettings:
    """Test EmbeddingSettings configuration."""

    def test_default_values(self):
        """Test default embedding settings."""
        settings = EmbeddingSettings()

        assert settings.provider == "ollama"
        assert settings.model == "nomic-embed-text"
        assert settings.base_url is None
        assert settings.api_key is None
        assert settings.batch_size == 64
        assert settings.timeout == 30
        assert settings.max_retries == 3
        assert settings.config == {}

    def test_validation_success(self):
        """Test successful validation."""
        settings = EmbeddingSettings()
        assert settings.validate() is True

    def test_validation_invalid_provider(self):
        """Test validation with invalid provider."""
        settings = EmbeddingSettings(provider="invalid")

        with pytest.raises(ConfigurationError, match="provider must be 'ollama' or 'openai'"):
            settings.validate()

    def test_validation_empty_model(self):
        """Test validation with empty model."""
        settings = EmbeddingSettings(model="")

        with pytest.raises(ConfigurationError, match="model must be a non-empty string"):
            settings.validate()

    def test_validation_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        settings = EmbeddingSettings(batch_size=0)

        with pytest.raises(ConfigurationError, match="batch_size must be a positive integer"):
            settings.validate()

    def test_validation_invalid_timeout(self):
        """Test validation with invalid timeout."""
        settings = EmbeddingSettings(timeout=-1)

        with pytest.raises(ConfigurationError, match="timeout must be a positive integer"):
            settings.validate()

    def test_validation_invalid_max_retries(self):
        """Test validation with invalid max_retries."""
        settings = EmbeddingSettings(max_retries=-1)

        with pytest.raises(ConfigurationError, match="max_retries must be a non-negative integer"):
            settings.validate()

    @patch.dict(os.environ, {}, clear=True)
    def test_validation_openai_no_api_key(self):
        """Test validation for OpenAI without API key."""
        settings = EmbeddingSettings(provider="openai")

        with pytest.raises(ConfigurationError, match="OpenAI provider requires api_key"):
            settings.validate()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_validation_openai_with_env_key(self):
        """Test validation for OpenAI with environment API key."""
        settings = EmbeddingSettings(provider="openai")
        assert settings.validate() is True

    def test_validation_openai_with_api_key(self):
        """Test validation for OpenAI with explicit API key."""
        settings = EmbeddingSettings(provider="openai", api_key="test-key")
        assert settings.validate() is True

    def test_copy_method(self):
        """Test copy method."""
        settings = EmbeddingSettings(provider="openai", model="text-embedding-ada-002")
        copied = settings.copy()

        assert copied.provider == "openai"
        assert copied.model == "text-embedding-ada-002"
        assert copied is not settings

    def test_update_from_dict(self):
        """Test update_from_dict method."""
        settings = EmbeddingSettings()
        update_dict = {"provider": "openai", "model": "test-model", "batch_size": 32}

        settings.update_from_dict(update_dict)

        assert settings.provider == "openai"
        assert settings.model == "test-model"
        assert settings.batch_size == 32

    def test_update_from_dict_with_invalid_attribute(self):
        """Test update_from_dict with invalid attribute and raise_errors=True."""
        settings = EmbeddingSettings()
        update_dict = {"invalid_attr": "value"}

        with pytest.raises(AttributeError):
            settings.update_from_dict(update_dict, raise_errors=True)

    def test_update_from_dict_with_invalid_attribute_ignore(self):
        """Test update_from_dict with invalid attribute and raise_errors=False."""
        settings = EmbeddingSettings()
        update_dict = {"invalid_attr": "value", "provider": "openai"}

        # Should not raise error
        settings.update_from_dict(update_dict, raise_errors=False)
        assert settings.provider == "openai"


class TestDatabaseSettings:
    """Test DatabaseSettings configuration."""

    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()

        assert settings.root_dir == "./.lvdb"
        assert settings.timeout == 300
        assert settings.connection_pool_size == 10
        assert settings.enable_gpu is False
        assert settings.enable_fts is True
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 1
        assert settings.chunking_method == "lines"
        assert settings.default_metadata_schema == {}

    def test_validation_success(self):
        """Test successful validation."""
        settings = DatabaseSettings()
        assert settings.validate() is True

    def test_validation_empty_root_dir(self):
        """Test validation with empty root_dir."""
        settings = DatabaseSettings(root_dir="")

        with pytest.raises(ConfigurationError, match="root_dir must be a non-empty string"):
            settings.validate()

    def test_validation_invalid_timeout(self):
        """Test validation with invalid timeout."""
        settings = DatabaseSettings(timeout=0)

        with pytest.raises(ConfigurationError, match="timeout must be a positive integer"):
            settings.validate()

    def test_validation_invalid_connection_pool_size(self):
        """Test validation with invalid connection pool size."""
        settings = DatabaseSettings(connection_pool_size=0)

        with pytest.raises(ConfigurationError, match="connection_pool_size must be a positive integer"):
            settings.validate()

    def test_validation_invalid_chunk_size(self):
        """Test validation with invalid chunk size."""
        settings = DatabaseSettings(chunk_size=0)

        with pytest.raises(ConfigurationError, match="chunk_size must be a positive integer"):
            settings.validate()

    def test_validation_invalid_chunk_overlap(self):
        """Test validation with invalid chunk overlap."""
        settings = DatabaseSettings(chunk_overlap=-1)

        with pytest.raises(ConfigurationError, match="chunk_overlap must be a non-negative integer"):
            settings.validate()

    def test_validation_chunk_overlap_too_large(self):
        """Test validation with chunk overlap >= chunk size."""
        settings = DatabaseSettings(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ConfigurationError, match="chunk_overlap must be less than chunk_size"):
            settings.validate()



class TestServerSettings:
    """Test ServerSettings configuration."""

    def test_default_values(self):
        """Test default server settings."""
        settings = ServerSettings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 5000
        assert settings.log_level == "INFO"
        assert settings.require_api_key is False
        assert settings.cors_enabled is True
        assert settings.cors_allowed_origins == "*"

    def test_validation_success(self):
        """Test successful validation."""
        settings = ServerSettings()
        assert settings.validate() is True

    def test_validation_empty_host(self):
        """Test validation with empty host."""
        settings = ServerSettings(host="")

        with pytest.raises(ConfigurationError, match="host must be a non-empty string"):
            settings.validate()

    def test_validation_invalid_port_low(self):
        """Test validation with port too low."""
        settings = ServerSettings(port=0)

        with pytest.raises(ConfigurationError, match="port must be an integer between 1 and 65535"):
            settings.validate()

    def test_validation_invalid_port_high(self):
        """Test validation with port too high."""
        settings = ServerSettings(port=65536)

        with pytest.raises(ConfigurationError, match="port must be an integer between 1 and 65535"):
            settings.validate()

    def test_validation_invalid_log_level(self):
        """Test validation with invalid log level."""
        settings = ServerSettings(log_level="INVALID")

        with pytest.raises(ConfigurationError, match="log_level must be one of"):
            settings.validate()

    def test_validation_empty_log_format(self):
        """Test validation with empty log format."""
        settings = ServerSettings(log_format="")

        with pytest.raises(ConfigurationError, match="log_format must be a non-empty string"):
            settings.validate()

    def test_validation_invalid_max_request_size(self):
        """Test validation with invalid max request size."""
        settings = ServerSettings(max_request_size=0)

        with pytest.raises(ConfigurationError, match="max_request_size must be a positive integer"):
            settings.validate()

    def test_validation_cors_allowed_origins_list(self):
        """Test validation with CORS origins as list."""
        settings = ServerSettings(cors_allowed_origins=["http://localhost", "https://example.com"])
        assert settings.validate() is True

    def test_validation_cors_allowed_origins_empty_list(self):
        """Test validation with empty CORS origins list."""
        settings = ServerSettings(cors_allowed_origins=[])

        with pytest.raises(ConfigurationError, match="cors_allowed_origins list cannot be empty"):
            settings.validate()

    def test_validation_cors_allowed_origins_invalid_type(self):
        """Test validation with invalid CORS origins type."""
        settings = ServerSettings(cors_allowed_origins=123)

        with pytest.raises(ConfigurationError, match="cors_allowed_origins must be either a string or a list"):
            settings.validate()


class TestConfig:
    """Test Config main configuration container."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()

        assert isinstance(config.database, DatabaseSettings)
        assert isinstance(config.embedding, EmbeddingSettings)
        assert isinstance(config.server, ServerSettings)

    def test_validation_success(self):
        """Test successful configuration validation."""
        config = Config()
        assert config.validate() is True

    def test_validation_failure(self):
        """Test configuration validation failure."""
        config = Config()
        config.database.chunk_size = 0  # Invalid value

        with pytest.raises(ConfigurationError):
            config.validate()

    def test_from_dict_empty(self):
        """Test creating config from empty dict."""
        with pytest.raises(ConfigurationError):
            config = Config.from_dict({})

    def test_from_dict_with_data(self):
        """Test creating config from dict with data."""
        data = {
            "database": {
                "root_dir": "/custom/path",
                "chunk_size": 1000
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-ada-002"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            }
        }

        config = Config.from_dict(data)

        assert config.database.root_dir == "/custom/path"
        assert config.database.chunk_size == 1000
        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-ada-002"
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8080

    def test_from_dict_with_metadata_schema(self):
        """Test creating config with metadata schema."""
        data = {
            "database": {
                "default_metadata_schema": {
                    "title": {"type": "text", "indexed": True},
                    "author": {"type": "text", "required": True}
                }
            }
        }

        config = Config.from_dict(data)

        assert "title" in config.database.default_metadata_schema
        assert "author" in config.database.default_metadata_schema
        assert isinstance(config.database.default_metadata_schema["title"], MetadataField)
        assert config.database.default_metadata_schema["title"].indexed is True
        assert config.database.default_metadata_schema["author"].required is True

    def test_from_dict_invalid_data(self):
        """Test creating config from invalid data."""
        with pytest.raises(TypeError, match="Configuration `data` must be a dictionary"):
            Config.from_dict(None)

        with pytest.raises(TypeError, match="Configuration `data` must be a dictionary"):
            Config.from_dict("invalid")

    def test_from_file_toml(self, temp_dir):
        """Test loading config from TOML file."""
        toml_content = """
[database]
root_dir = "/tmp/test"
chunk_size = 800

[embedding]
provider = "openai"
model = "test-model"

[server]
host = "0.0.0.0"
port = 9000
"""

        toml_file = temp_dir / "config.toml"
        toml_file.write_text(toml_content)

        config = Config.from_file(str(toml_file))

        assert config.database.root_dir == "/tmp/test"
        assert config.database.chunk_size == 800
        assert config.embedding.provider == "openai"
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9000

    def test_from_file_json(self, temp_dir):
        """Test loading config from JSON file."""
        json_data = {
            "database": {"root_dir": "/json/test", "chunk_size": 600},
            "embedding": {"provider": "ollama", "model": "json-model"},
            "server": {"host": "127.0.0.1", "port": 7000}
        }

        json_file = temp_dir / "config.json"
        json_file.write_text(json.dumps(json_data))

        config = Config.from_file(str(json_file))

        assert config.database.root_dir == "/json/test"
        assert config.database.chunk_size == 600
        assert config.embedding.model == "json-model"
        assert config.server.port == 7000

    def test_from_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_file("/path/that/does/not/exist.toml")

    def test_from_file_unsupported_format(self, temp_dir):
        """Test loading config from unsupported file format."""
        unsupported_file = temp_dir / "config.xml"
        unsupported_file.write_text("<config></config>")

        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            Config.from_file(str(unsupported_file))

    @patch.dict(os.environ, {
        "LVDB_DATABASE_ROOT_DIR": "/env/test",
        "LVDB_DATABASE_CHUNK_SIZE": "777",
        "LVDB_EMBEDDING_PROVIDER": "openai",
        "LVDB_SERVER_PORT": "6000"
    }, clear=True)
    def test_from_env(self):
        """Test loading config from environment variables."""
        config = Config.from_env()

        assert config.database.root_dir == "/env/test"
        assert config.database.chunk_size == 777
        assert config.embedding.provider == "openai"
        assert config.server.port == 6000

    @patch.dict(os.environ, {"LVDB_DATABASE_ENABLE_GPU": "true"}, clear=True)
    def test_from_env_boolean_conversion(self):
        """Test environment variable boolean conversion."""
        config = Config.from_env()
        assert config.database.enable_gpu is True

    @patch.dict(os.environ, {"LVDB_SERVER_CORS_ALLOWED_ORIGINS": '["http://localhost", "https://example.com"]'},
                clear=True)
    def test_from_env_list_conversion(self):
        """Test environment variable list conversion."""
        config = Config.from_env()
        assert config.server.cors_allowed_origins == ["http://localhost", "https://example.com"]

    def test_to_flask_config(self):
        """Test conversion to Flask configuration format."""
        config = Config()
        config.database.root_dir = "/test/path"
        config.embedding.provider = "openai"
        config.server.require_api_key = True

        flask_config = config.to_flask_config()

        assert flask_config["DB_ROOT_DIR"] == "/test/path"
        assert flask_config["EMBEDDING_PROVIDER"] == "openai"
        assert flask_config["REQUIRE_API_KEY"] is True
        # assert "API_KEY_DB_PATH" in flask_config

    def test_generate_toml(self):
        """Test TOML generation."""
        config = Config()
        config.database.root_dir = "/test/toml"
        config.embedding.provider = "openai"
        config.server.port = 8080

        toml_str = config.generate_toml()

        assert 'root_dir = "/test/toml"' in toml_str
        assert 'provider = "openai"' in toml_str
        assert 'port = 8080' in toml_str
        assert '[database]' in toml_str
        assert '[embedding]' in toml_str
        assert '[server]' in toml_str

    def test_apply_common_schema(self):
        """Test applying common metadata schema."""
        config = Config()
        config.apply_common_schema("documents")

        schema = config.database.default_metadata_schema
        assert "title" in schema
        assert "author" in schema
        assert isinstance(schema["title"], MetadataField)

    def test_apply_common_schema_invalid(self):
        """Test applying invalid common schema."""
        config = Config()

        with pytest.raises(ConfigurationError, match="Unknown schema"):
            config.apply_common_schema("invalid_schema")

    def test_merge_configs(self):
        """Test merging two configurations."""
        base_config = Config()
        base_config.database.root_dir = "/base/path"
        base_config.embedding.provider = "ollama"

        override_config = Config()
        override_config.database.chunk_size = 800
        override_config.server.port = 9000

        merged = base_config.merge(override_config)

        # Base values should be preserved where not overridden
        assert merged.database.root_dir == "/base/path"
        assert merged.embedding.provider == "ollama"

        # Override values should be applied
        assert merged.database.chunk_size == 800
        assert merged.server.port == 9000

    def test_merge_metadata_schema(self):
        """Test merging configurations with metadata schemas."""
        base_config = Config()
        base_config.database.default_metadata_schema = {
            "title": MetadataField(type=MetadataFieldType.TEXT)
        }

        override_config = Config()
        override_config.database.default_metadata_schema = {
            "author": MetadataField(type=MetadataFieldType.TEXT)
        }

        merged = base_config.merge(override_config)

        # Both schemas should be present
        assert "title" in merged.database.default_metadata_schema
        assert "author" in merged.database.default_metadata_schema

    def test_update_from_dict_method(self):
        """Test update_from_dict class method."""
        base_config = Config()

        update_map = {
            "database": {"root_dir": "/updated/path", "chunk_size": 999},
            "server": {"port": 7777}
        }

        updated_config = Config.update_from_dict(base_config, update_map)

        assert updated_config.database.root_dir == "/updated/path"
        assert updated_config.database.chunk_size == 999
        assert updated_config.server.port == 7777

    def test_update_from_dict_invalid_section(self):
        """Test update_from_dict with invalid section."""
        base_config = Config()

        update_map = {"invalid_section": {"key": "value"}}

        with pytest.raises(KeyError, match="Expected keys:"):
            Config.update_from_dict(base_config, update_map)

    def test_update_from_dict_invalid_key(self):
        """Test update_from_dict with invalid configuration key."""
        base_config = Config()

        update_map = {"database": {"invalid_key": "value"}}

        with pytest.raises(KeyError, match="Configuration setting `invalid_key` does not exist"):
            Config.update_from_dict(base_config, update_map)


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_with_config_object(self):
        """Test loading config with Config object."""
        input_config = Config()
        input_config.database.root_dir = "/test/config"

        result = load_config(input_config, verbose=False)

        assert isinstance(result, Config)
        assert result.database.root_dir == "/test/config"

    def test_load_config_with_dict(self):
        """Test loading config with dictionary."""
        input_dict = {
            "database": {"root_dir": "/dict/test"},
            "server": {"port": 8888}
        }

        result = load_config(input_dict, verbose=False)

        assert isinstance(result, Config)
        assert result.database.root_dir == "/dict/test"
        assert result.server.port == 8888

    def test_load_config_with_file_path(self, temp_dir):
        """Test loading config from file path."""
        toml_content = """
[database]
root_dir = "/file/test"

[server]
port = 5555
"""

        config_file = temp_dir / "test_config.toml"
        config_file.write_text(toml_content)

        result = load_config(str(config_file), verbose=False)

        assert result.database.root_dir == "/file/test"
        assert result.server.port == 5555

    def test_load_config_file_not_found(self):
        """Test loading config from non-existent file."""
        result = load_config("/non/existent/file.toml", verbose=False)

        # Should return default config
        assert isinstance(result, Config)
        assert result.database.root_dir == "./.lvdb"

    @patch.dict(os.environ, {"LVDB_SERVER_CONFIG": ""}, clear=True)
    def test_load_config_no_config(self):
        """Test loading config with no configuration provided."""
        result = load_config(None, verbose=False)

        # Should return default config
        assert isinstance(result, Config)
        assert result.database.root_dir == "./.lvdb"

    @patch.dict(os.environ, {"LVDB_DATABASE_ROOT_DIR": "/env/override"}, clear=True)
    def test_load_config_with_env_override(self):
        """Test that environment variables override config file."""
        input_config = Config()
        input_config.database.root_dir = "/original/path"

        result = load_config(input_config, verbose=False)

        # Environment should override the input config
        assert result.database.root_dir == "/env/override"

    def test_load_config_with_apply_schema(self):
        """Test loading config with schema application."""
        result = load_config(None, apply_schema="documents", verbose=False)

        assert "title" in result.database.default_metadata_schema
        assert "author" in result.database.default_metadata_schema

    def test_load_config_validation_disabled(self):
        """Test loading config with validation disabled."""
        input_config = Config()
        input_config.database.chunk_size = 0  # Invalid value

        # Should not raise an error with validate=False
        result = load_config(input_config, validate=False, verbose=False)
        assert result.database.chunk_size == 0

    def test_load_config_validation_enabled_invalid(self):
        """Test loading config with validation enabled and invalid config."""
        input_config = Config()
        input_config.database.chunk_size = 0  # Invalid value

        with pytest.raises(ConfigurationError):
            load_config(input_config, validate=True, verbose=False)

    @patch('click.secho')
    def test_load_config_verbose_mode(self, mock_secho):
        """Test load_config in verbose mode."""
        input_config = Config()

        load_config(input_config, verbose=True)

        # Should have called click.secho for verbose output
        mock_secho.assert_called()

    def test_load_config_with_custom_object(self):
        """Test loading config with custom object having __dict__."""

        class CustomConfig:
            def __init__(self):
                self.database = {"root_dir": "/custom/test"}
                self.server = {"port": 3333}

        custom_obj = CustomConfig()
        result = load_config(custom_obj, verbose=False)

        assert isinstance(result, Config)


class TestConfigErrorHandling:
    """Test error handling in configuration."""

    def test_config_file_invalid_toml(self, temp_dir):
        """Test handling of invalid TOML file."""
        invalid_toml = temp_dir / "invalid.toml"
        invalid_toml.write_text("[invalid toml content")

        with pytest.raises(ConfigurationError):
            load_config(str(invalid_toml), verbose=False)

    def test_env_conversion_invalid_int(self):
        """Test environment variable conversion with invalid integer."""
        with patch.dict(os.environ, {"LVDB_SERVER_PORT": "not_an_integer"}, clear=True):
            with pytest.raises(ValueError):
                Config.from_env()

    def test_env_conversion_invalid_json(self):
        """Test environment variable conversion with invalid JSON."""
        with patch.dict(os.environ, {"LVDB_DATABASE_DEFAULT_METADATA_SCHEMA": "invalid json"}, clear=True):
            # Should not raise error, but return empty dict for invalid JSON
            config = Config.from_env()
            # The conversion should handle the error gracefully
