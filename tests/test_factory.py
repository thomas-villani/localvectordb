"""
Tests for localvectordb.factory module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from localvectordb.factory import VectorDB
from localvectordb.core import MetadataField, MetadataFieldType


class TestVectorDBFactory:
    """Test VectorDB factory function."""

    def test_local_database_creation(self, temp_dir, sample_metadata_schema):
        """Test creating local database via factory."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_instance = Mock()
            mock_local.return_value = mock_instance

            db = VectorDB(
                name="test_db",
                base_path=temp_dir,
                metadata_schema=sample_metadata_schema,
                embedding_model="test-model",
                chunk_size=500,
                enable_gpu=True
            )

            assert db == mock_instance
            mock_local.assert_called_once_with(
                name="test_db",
                base_path=temp_dir,
                metadata_schema=sample_metadata_schema,
                embedding_model="test-model",
                chunk_size=500,
                enable_gpu=True
            )

    def test_local_database_string_path(self, sample_metadata_schema):
        """Test creating local database with string path."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_instance = Mock()
            mock_local.return_value = mock_instance

            db = VectorDB(
                name="test_db",
                base_path="./local_storage",
                metadata_schema=sample_metadata_schema
            )

            assert db == mock_instance
            mock_local.assert_called_once_with(
                name="test_db",
                base_path="./local_storage",
                metadata_schema=sample_metadata_schema
            )

    def test_remote_database_http_url(self, sample_metadata_schema):
        """Test creating remote database with http URL."""
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_instance = Mock()
            mock_remote.return_value = mock_instance

            db = VectorDB(
                name="test_db",
                base_path="http://localhost:5000",
                api_key="test-key",
                metadata_schema=sample_metadata_schema,
                request_timeout=30
            )

            assert db == mock_instance
            mock_remote.assert_called_once_with(
                name="test_db",
                base_url="http://localhost:5000",
                api_key="test-key",
                metadata_schema=sample_metadata_schema,
                request_timeout=30
            )

    def test_remote_database_https_url(self):
        """Test creating remote database with https URL."""
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_instance = Mock()
            mock_remote.return_value = mock_instance

            db = VectorDB(
                name="test_db",
                base_path="https://api.example.com",
                api_key="secret-key"
            )

            assert db == mock_instance
            mock_remote.assert_called_once_with(
                name="test_db",
                base_url="https://api.example.com",
                api_key="secret-key"
            )

    def test_parameter_filtering_for_local(self):
        """Test that remote-only parameters are filtered for local databases."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_instance = Mock()
            mock_local.return_value = mock_instance

            VectorDB(
                name="test_db",
                base_path="./local",
                api_key="should-be-filtered",  # Remote-only
                request_timeout=30,  # Remote-only
                enable_gpu=True,  # Local-only
                connection_pool_size=5  # Local-only
            )

            # Check that remote-only params were filtered out
            call_kwargs = mock_local.call_args[1]
            assert "api_key" not in call_kwargs
            assert "request_timeout" not in call_kwargs
            assert call_kwargs["enable_gpu"] is True
            assert call_kwargs["connection_pool_size"] == 5

    def test_parameter_filtering_for_remote(self):
        """Test that local-only parameters are filtered for remote databases."""
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_instance = Mock()
            mock_remote.return_value = mock_instance

            VectorDB(
                name="test_db",
                base_path="http://localhost:5000",
                api_key="test-key",  # Remote-only
                request_timeout=30,  # Remote-only
                connection_pool_size=10  # Local-only - should be filtered
            )

            # Check that local-only params were filtered out
            call_kwargs = mock_remote.call_args[1]
            assert "connection_pool_size" not in call_kwargs
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["request_timeout"] == 30

    def test_shared_parameters_passed_through(self):
        """Test that shared parameters are passed to both local and remote."""
        shared_params = {
            "metadata_schema": {"author": MetadataField(type=MetadataFieldType.TEXT)},
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_config": {"api_key": "test"},
            "chunking_method": "sentences",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "enable_gpu": False,
            "enable_fts": True,
            "create_if_not_exists": False
        }

        # Test local database
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            VectorDB(
                name="test_db",
                base_path="./local",
                **shared_params
            )

            call_kwargs = mock_local.call_args[1]
            for key, value in shared_params.items():
                assert call_kwargs[key] == value

        # Test remote database
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            VectorDB(
                name="test_db",
                base_path="http://localhost:5000",
                **shared_params
            )

            call_kwargs = mock_remote.call_args[1]
            for key, value in shared_params.items():
                assert call_kwargs[key] == value

    def test_path_object_detection(self, temp_dir):
        """Test that Path objects are properly detected as local."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            VectorDB("test_db", temp_dir)

            mock_local.assert_called_once()
            assert mock_local.call_args[1]["base_path"] == temp_dir

    def test_url_edge_cases(self):
        """Test URL detection edge cases."""
        test_cases = [
            ("http://", True),
            ("https://", True),
            ("HTTP://EXAMPLE.COM", True),
            ("HTTPS://EXAMPLE.COM", True),
            ("ftp://example.com", False),
            ("file://path", False),
            ("http", False),
            ("https", False),
            ("/http://not-a-url", False),
        ]

        for url, should_be_remote in test_cases:
            with patch('localvectordb.factory.LocalVectorDB') as mock_local, \
                    patch('localvectordb.factory.RemoteVectorDB') as mock_remote:

                mock_local.return_value = Mock()
                mock_remote.return_value = Mock()

                VectorDB("test_db", url)

                if should_be_remote:
                    mock_remote.assert_called_once()
                    mock_local.assert_not_called()
                else:
                    mock_local.assert_called_once()
                    mock_remote.assert_not_called()

    def test_comprehensive_local_example(self, temp_dir):
        """Test comprehensive local database example from docstring."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
        }

        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            db = VectorDB(
                "my_docs",
                temp_dir,
                metadata_schema=metadata_schema,
                embedding_model="nomic-embed-text",
                chunk_size=500
            )

            mock_local.assert_called_once_with(
                name="my_docs",
                base_path=temp_dir,
                metadata_schema=metadata_schema,
                embedding_model="nomic-embed-text",
                chunk_size=500
            )

    def test_comprehensive_remote_example(self):
        """Test comprehensive remote database example from docstring."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
        }

        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            db = VectorDB(
                "my_docs",
                "http://localhost:5000",
                api_key="your_api_key",
                metadata_schema=metadata_schema
            )

            mock_remote.assert_called_once_with(
                name="my_docs",
                base_url="http://localhost:5000",
                api_key="your_api_key",
                metadata_schema=metadata_schema
            )

    def test_seamless_switching_example(self):
        """Test seamless switching example from docstring."""

        def create_database(use_remote=False):
            if use_remote:
                base_path = "http://localhost:5000"
                extra_kwargs = {"api_key": "your_api_key"}
            else:
                base_path = "./local_storage"
                extra_kwargs = {"enable_gpu": True}

            return VectorDB(
                "my_database",
                base_path,
                embedding_model="nomic-embed-text",
                chunk_size=500,
                **extra_kwargs
            )

        # Test local creation
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            local_db = create_database(use_remote=False)

            mock_local.assert_called_once_with(
                name="my_database",
                base_path="./local_storage",
                embedding_model="nomic-embed-text",
                chunk_size=500,
                enable_gpu=True
            )

        # Test remote creation
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            remote_db = create_database(use_remote=True)

            mock_remote.assert_called_once_with(
                name="my_database",
                base_url="http://localhost:5000",
                embedding_model="nomic-embed-text",
                chunk_size=500,
                api_key="your_api_key"
            )

    def test_empty_kwargs(self):
        """Test factory with minimal parameters."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            db = VectorDB("test_db", "./local")

            mock_local.assert_called_once_with(
                name="test_db",
                base_path="./local"
            )

    def test_all_parameters_local(self, temp_dir):
        """Test factory with all possible local parameters."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'rating': MetadataField(type=MetadataFieldType.REAL),
            'tags': MetadataField(type=MetadataFieldType.JSON)
        }

        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            db = VectorDB(
                name="comprehensive_db",
                base_path=temp_dir,
                metadata_schema=metadata_schema,
                doc_id_pattern="custom_{idx}",
                chunk_id_pattern="{doc_id}:chunk_{chunk_idx}",
                embedding_provider="openai",
                embedding_model="text-embedding-3-large",
                embedding_config={"api_key": "test-key"},
                chunking_method="paragraphs",
                chunk_size=1000,
                chunk_overlap=200,
                enable_gpu=True,
                enable_fts=False,
                connection_pool_size=20,
                create_if_not_exists=False
            )

            # Verify all parameters were passed
            call_kwargs = mock_local.call_args[1]
            assert call_kwargs["metadata_schema"] == metadata_schema
            assert call_kwargs["doc_id_pattern"] == "custom_{idx}"
            assert call_kwargs["chunk_id_pattern"] == "{doc_id}:chunk_{chunk_idx}"
            assert call_kwargs["embedding_provider"] == "openai"
            assert call_kwargs["embedding_model"] == "text-embedding-3-large"
            assert call_kwargs["embedding_config"] == {"api_key": "test-key"}
            assert call_kwargs["chunking_method"] == "paragraphs"
            assert call_kwargs["chunk_size"] == 1000
            assert call_kwargs["chunk_overlap"] == 200
            assert call_kwargs["enable_gpu"] is True
            assert call_kwargs["enable_fts"] is False
            assert call_kwargs["connection_pool_size"] == 20
            assert call_kwargs["create_if_not_exists"] is False

    def test_all_parameters_remote(self):
        """Test factory with all possible remote parameters."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'category': MetadataField(type=MetadataFieldType.TEXT),
            'priority': MetadataField(type=MetadataFieldType.INTEGER)
        }

        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            db = VectorDB(
                name="comprehensive_remote_db",
                base_path="https://api.vectordb.com",
                api_key="secret-api-key",
                create_if_not_exists=True,
                metadata_schema=metadata_schema,
                embedding_provider="ollama",
                embedding_model="nomic-embed-text",
                embedding_config={"base_url": "http://localhost:11434"},
                chunking_method="tokens",
                chunk_size=800,
                chunk_overlap=80,
                enable_gpu=False,
                enable_fts=True,
                request_timeout=120
            )

            # Verify all parameters were passed
            call_kwargs = mock_remote.call_args[1]
            assert call_kwargs["api_key"] == "secret-api-key"
            assert call_kwargs["create_if_not_exists"] is True
            assert call_kwargs["metadata_schema"] == metadata_schema
            assert call_kwargs["embedding_provider"] == "ollama"
            assert call_kwargs["embedding_model"] == "nomic-embed-text"
            assert call_kwargs["embedding_config"] == {"base_url": "http://localhost:11434"}
            assert call_kwargs["chunking_method"] == "tokens"
            assert call_kwargs["chunk_size"] == 800
            assert call_kwargs["chunk_overlap"] == 80
            assert call_kwargs["enable_gpu"] is False
            assert call_kwargs["enable_fts"] is True
            assert call_kwargs["request_timeout"] == 120

            # Verify local-only parameters were filtered out
            assert "connection_pool_size" not in call_kwargs


class TestFactoryDocstringExamples:
    """Test that all examples from the factory docstring work correctly."""

    def test_local_database_example(self, temp_dir):
        """Test the local database example from docstring."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            # Example from docstring
            db = VectorDB(
                "my_docs",
                temp_dir,
                metadata_schema={
                    'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
                },
                embedding_model="nomic-embed-text",
                chunk_size=500
            )

            assert mock_local.called
            call_kwargs = mock_local.call_args[1]
            assert call_kwargs["embedding_model"] == "nomic-embed-text"
            assert call_kwargs["chunk_size"] == 500

    def test_remote_database_example(self):
        """Test the remote database example from docstring."""
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            # Example from docstring
            db = VectorDB(
                "my_docs",
                "http://localhost:5000",
                api_key="your_api_key",
                metadata_schema={
                    'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
                }
            )

            assert mock_remote.called
            call_kwargs = mock_remote.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:5000"
            assert call_kwargs["api_key"] == "your_api_key"

    def test_conditional_creation_example(self):
        """Test the conditional database creation example."""

        def create_database(use_remote=False):
            if use_remote:
                base_path = "http://localhost:5000"
                extra_kwargs = {"api_key": "your_api_key"}
            else:
                base_path = "./local_storage"
                extra_kwargs = {"enable_gpu": True}

            return VectorDB(
                "my_database",
                base_path,
                embedding_model="nomic-embed-text",
                chunk_size=500,
                **extra_kwargs
            )

        # Test local creation
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()
            local_db = create_database(use_remote=False)
            assert mock_local.called

        # Test remote creation
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()
            remote_db = create_database(use_remote=True)
            assert mock_remote.called


class TestFactoryErrorCases:
    """Test error cases and edge conditions for the factory."""

    def test_invalid_local_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            # Simulate LocalVectorDB raising ValueError for invalid params
            mock_local.side_effect = ValueError("Invalid chunk_size")

            with pytest.raises(ValueError, match="Invalid chunk_size"):
                VectorDB("test_db", "./local", chunk_size=-1)

    def test_invalid_remote_parameters(self):
        """Test that invalid remote parameters raise appropriate errors."""
        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            # Simulate RemoteVectorDB raising ValueError for invalid params
            mock_remote.side_effect = ValueError("Invalid API key")

            with pytest.raises(ValueError, match="Invalid API key"):
                VectorDB("test_db", "http://localhost:5000", api_key="")

    def test_base_path_type_conversion(self):
        """Test that base_path is properly converted to string for URL checking."""

        class CustomPath:
            def __str__(self):
                return "http://localhost:5000"

        with patch('localvectordb.factory.RemoteVectorDB') as mock_remote:
            mock_remote.return_value = Mock()

            # Should be detected as remote due to string conversion
            VectorDB("test_db", CustomPath())

            mock_remote.assert_called_once()

    def test_none_base_path(self):
        """Test handling of None base_path."""
        with patch('localvectordb.factory.LocalVectorDB') as mock_local:
            mock_local.return_value = Mock()

            # None should be converted to "None" string
            VectorDB("test_db", None)

            mock_local.assert_called_once()
            call_kwargs = mock_local.call_args[1]
            assert call_kwargs["base_path"] is None