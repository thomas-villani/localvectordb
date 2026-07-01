"""
Tests for localvectordb.exceptions module.
"""

import pytest

from localvectordb.exceptions import (
    BaseLocalVectorDBException,
    ConfigurationError,
    DatabaseNotFoundError,
    DuplicateDocumentIDError,
    EmbeddingError,
    OllamaNotFoundError,
)


@pytest.mark.unit
class TestBaseLocalVectorDBException:
    """Test base exception class."""

    def test_create_base_exception(self):
        """Test creating base exception."""
        exc = BaseLocalVectorDBException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_inheritance_chain(self):
        """Test that base exception inherits from Exception."""
        exc = BaseLocalVectorDBException("Test")
        assert isinstance(exc, Exception)

    def test_empty_message(self):
        """Test creating exception with empty message."""
        exc = BaseLocalVectorDBException()
        assert str(exc) == ""

    def test_non_string_message(self):
        """Test creating exception with non-string message."""
        exc = BaseLocalVectorDBException(123)
        assert str(exc) == "123"


@pytest.mark.unit
class TestDatabaseNotFoundError:
    """Test DatabaseNotFoundError exception."""

    def test_create_database_not_found_error(self):
        """Test creating database not found error."""
        exc = DatabaseNotFoundError("Database 'test' not found")
        # Note: for some reason the string chars are escaped?
        assert str(exc) == "\"Database 'test' not found\""
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, KeyError)

    def test_inheritance_chain(self):
        """Test inheritance chain for DatabaseNotFoundError."""
        exc = DatabaseNotFoundError("Test")

        # Should inherit from multiple exception types
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, KeyError)
        assert isinstance(exc, Exception)

    def test_catch_as_key_error(self):
        """Test that DatabaseNotFoundError can be caught as KeyError."""
        with pytest.raises(KeyError):
            raise DatabaseNotFoundError("Database not found")

    def test_catch_as_base_exception(self):
        """Test that DatabaseNotFoundError can be caught as base exception."""
        with pytest.raises(BaseLocalVectorDBException):
            raise DatabaseNotFoundError("Database not found")

    def test_with_database_name(self):
        """Test creating error with specific database name."""
        db_name = "my_vector_db"
        exc = DatabaseNotFoundError(f"Database '{db_name}' not found in './data' directory")

        assert db_name in str(exc)
        assert "not found" in str(exc)


@pytest.mark.unit
class TestDuplicateDocumentIDError:
    """Test DuplicateDocumentIDError exception."""

    def test_create_duplicate_id_error(self):
        """Test creating duplicate document ID error."""
        exc = DuplicateDocumentIDError("Document with ID 'doc_123' already exists")
        assert str(exc) == "Document with ID 'doc_123' already exists"
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, ValueError)

    def test_inheritance_chain(self):
        """Test inheritance chain for DuplicateDocumentIDError."""
        exc = DuplicateDocumentIDError("Test")

        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, ValueError)
        assert isinstance(exc, Exception)

    def test_catch_as_value_error(self):
        """Test that DuplicateDocumentIDError can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise DuplicateDocumentIDError("Duplicate ID")

    def test_catch_as_base_exception(self):
        """Test that DuplicateDocumentIDError can be caught as base exception."""
        with pytest.raises(BaseLocalVectorDBException):
            raise DuplicateDocumentIDError("Duplicate ID")

    def test_with_document_id(self):
        """Test creating error with specific document ID."""
        doc_id = "document_12345"
        exc = DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")

        assert doc_id in str(exc)
        assert "already exists" in str(exc)


@pytest.mark.unit
class TestOllamaNotFoundError:
    """Test OllamaNotFoundError exception."""

    def test_create_ollama_not_found_error(self):
        """Test creating Ollama not found error."""
        exc = OllamaNotFoundError("Ollama is not installed or not running")
        assert str(exc) == "Ollama is not installed or not running"
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)

    def test_inheritance_chain(self):
        """Test inheritance chain for OllamaNotFoundError."""
        exc = OllamaNotFoundError("Test")

        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)
        assert isinstance(exc, Exception)

    def test_catch_as_runtime_error(self):
        """Test that OllamaNotFoundError can be caught as RuntimeError."""
        with pytest.raises(RuntimeError):
            raise OllamaNotFoundError("Ollama not found")

    def test_catch_as_base_exception(self):
        """Test that OllamaNotFoundError can be caught as base exception."""
        with pytest.raises(BaseLocalVectorDBException):
            raise OllamaNotFoundError("Ollama not found")

    def test_typical_messages(self):
        """Test typical error messages for OllamaNotFoundError."""
        messages = [
            "Ollama is not installed or not running.",
            "Could not connect to Ollama at http://localhost:11434",
            "Ollama service is not available",
        ]

        for message in messages:
            exc = OllamaNotFoundError(message)
            assert str(exc) == message
            assert isinstance(exc, OllamaNotFoundError)


@pytest.mark.unit
class TestEmbeddingError:
    """Test EmbeddingError exception."""

    def test_create_embedding_error(self):
        """Test creating embedding error."""
        exc = EmbeddingError("Failed to generate embeddings")
        assert str(exc) == "Failed to generate embeddings"
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)

    def test_inheritance_chain(self):
        """Test inheritance chain for EmbeddingError."""
        exc = EmbeddingError("Test")

        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)
        assert isinstance(exc, Exception)

    def test_catch_as_runtime_error(self):
        """Test that EmbeddingError can be caught as RuntimeError."""
        with pytest.raises(RuntimeError):
            raise EmbeddingError("Embedding failed")

    def test_catch_as_base_exception(self):
        """Test that EmbeddingError can be caught as base exception."""
        with pytest.raises(BaseLocalVectorDBException):
            raise EmbeddingError("Embedding failed")

    def test_embedding_specific_messages(self):
        """Test typical embedding error messages."""
        messages = [
            "Failed to generate embeddings for text",
            "Embedding model 'unknown-model' not found",
            "Embedding dimension mismatch: expected 384, got 512",
            "OpenAI API key is invalid or expired",
        ]

        for message in messages:
            exc = EmbeddingError(message)
            assert str(exc) == message
            assert isinstance(exc, EmbeddingError)


@pytest.mark.unit
class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_create_configuration_error(self):
        """Test creating configuration error."""
        exc = ConfigurationError("Invalid configuration parameter")
        assert str(exc) == "Invalid configuration parameter"
        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)

    def test_inheritance_chain(self):
        """Test inheritance chain for ConfigurationError."""
        exc = ConfigurationError("Test")

        assert isinstance(exc, BaseLocalVectorDBException)
        assert isinstance(exc, RuntimeError)
        assert isinstance(exc, Exception)

    def test_catch_as_runtime_error(self):
        """Test that ConfigurationError can be caught as RuntimeError."""
        with pytest.raises(RuntimeError):
            raise ConfigurationError("Bad config")

    def test_catch_as_base_exception(self):
        """Test that ConfigurationError can be caught as base exception."""
        with pytest.raises(BaseLocalVectorDBException):
            raise ConfigurationError("Bad config")

    def test_configuration_specific_messages(self):
        """Test typical configuration error messages."""
        messages = [
            "Invalid chunk_size: must be positive integer",
            "Unsupported embedding provider: 'unknown'",
            "Invalid metadata field type: 'custom'",
            "Conflicting configuration: enable_gpu=True but FAISS GPU not available",
        ]

        for message in messages:
            exc = ConfigurationError(message)
            assert str(exc) == message
            assert isinstance(exc, ConfigurationError)


@pytest.mark.unit
class TestExceptionUsagePractices:
    """Test common usage patterns and best practices for exceptions."""

    def test_exception_chaining(self):
        """Test exception chaining with 'raise from'."""
        original_error = ValueError("Original error")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise DatabaseNotFoundError("Database error") from e
        except DatabaseNotFoundError as exc:
            assert exc.__cause__ == original_error
            assert "Database error" in str(exc)

    def test_multiple_exception_catching(self):
        """Test catching multiple related exceptions."""
        exceptions_to_test = [
            DatabaseNotFoundError("DB not found"),
            DuplicateDocumentIDError("Duplicate ID"),
            EmbeddingError("Embedding failed"),
            ConfigurationError("Bad config"),
            OllamaNotFoundError("Ollama not found"),
        ]

        for exc in exceptions_to_test:
            # All should be catchable as base exception
            with pytest.raises(BaseLocalVectorDBException):
                raise exc

    def test_specific_exception_handling(self):
        """Test handling specific exceptions differently."""

        def handle_database_operation():
            """Simulate database operation that can fail in different ways."""
            import random

            error_type = random.choice(["not_found", "duplicate", "embedding", "config", "ollama"])

            if error_type == "not_found":
                raise DatabaseNotFoundError("Database not found")
            elif error_type == "duplicate":
                raise DuplicateDocumentIDError("Duplicate document")
            elif error_type == "embedding":
                raise EmbeddingError("Embedding failed")
            elif error_type == "config":
                raise ConfigurationError("Invalid config")
            elif error_type == "ollama":
                raise OllamaNotFoundError("Ollama not available")

        # Test that we can catch and handle each type specifically
        with pytest.raises(BaseLocalVectorDBException):
            handle_database_operation()

    def test_error_message_formatting(self):
        """Test consistent error message formatting."""
        # Test that error messages can include context
        db_name = "test_database"
        doc_id = "doc_12345"
        model_name = "text-embedding-ada-002"

        exc1 = DatabaseNotFoundError(f"Database '{db_name}' not found in specified path")
        exc2 = DuplicateDocumentIDError(f"Document '{doc_id}' already exists in database '{db_name}'")
        exc3 = EmbeddingError(f"Failed to load embedding model '{model_name}'")

        assert db_name in str(exc1)
        assert db_name in str(exc2) and doc_id in str(exc2)
        assert model_name in str(exc3)

    def test_exception_without_message(self):
        """Test that exceptions can be created without messages."""
        exceptions = [
            BaseLocalVectorDBException(),
            DatabaseNotFoundError(),
            DuplicateDocumentIDError(),
            EmbeddingError(),
            ConfigurationError(),
            OllamaNotFoundError(),
        ]

        for exc in exceptions:
            # Should not raise any errors
            assert isinstance(exc, BaseLocalVectorDBException)
            # String representation should work
            str(exc)

    def test_exception_with_additional_context(self):
        """Test exceptions with additional context information."""
        # Create exceptions with detailed context
        context = {"database_name": "my_db", "operation": "upsert", "document_count": 5, "error_code": "DB001"}

        message = (
            f"Operation '{context['operation']}' failed on database '{context['database_name']}' "
            f"while processing {context['document_count']} documents. Error code: {context['error_code']}"
        )

        exc = DatabaseNotFoundError(message)

        # Verify context is preserved in error message
        for _key, value in context.items():
            assert str(value) in str(exc)

    def test_nested_exception_handling(self):
        """Test nested exception handling scenarios."""

        def operation_level_1():
            try:
                operation_level_2()
            except EmbeddingError as e:
                raise ConfigurationError("Configuration issue during embedding") from e

        def operation_level_2():
            try:
                operation_level_3()
            except OllamaNotFoundError as e:
                raise EmbeddingError("Embedding service unavailable") from e

        def operation_level_3():
            raise OllamaNotFoundError("Ollama service not running")

        with pytest.raises(ConfigurationError) as exc_info:
            operation_level_1()

        # Check exception chain
        exc = exc_info.value
        assert isinstance(exc, ConfigurationError)
        assert isinstance(exc.__cause__, EmbeddingError)
        assert isinstance(exc.__cause__.__cause__, OllamaNotFoundError)

        # Original error should be accessible
        assert "Ollama service not running" in str(exc.__cause__.__cause__)


@pytest.mark.unit
class TestExceptionDocumentation:
    """Test that exceptions have proper docstrings and are well-documented."""

    @staticmethod
    def _custom_exceptions():
        """Every exception class actually defined in localvectordb.exceptions."""
        import inspect

        import localvectordb.exceptions as exc_mod

        return [
            obj
            for _, obj in inspect.getmembers(exc_mod, inspect.isclass)
            if issubclass(obj, BaseLocalVectorDBException) and obj.__module__ == exc_mod.__name__
        ]

    def test_all_exceptions_have_docstrings(self):
        """Every custom exception carries a non-empty docstring."""
        excs = self._custom_exceptions()
        assert excs, "no custom exceptions discovered"
        missing = [e.__name__ for e in excs if not (e.__doc__ and e.__doc__.strip())]
        assert not missing, f"exceptions missing docstrings: {missing}"

    def test_all_exceptions_defined(self):
        """Test that all expected exceptions are defined and importable."""
        from localvectordb.exceptions import (
            BaseLocalVectorDBException,
            ConfigurationError,
            DatabaseNotFoundError,
            DuplicateDocumentIDError,
            EmbeddingError,
            OllamaNotFoundError,
        )

        # All should be classes
        exceptions = [
            BaseLocalVectorDBException,
            DatabaseNotFoundError,
            DuplicateDocumentIDError,
            OllamaNotFoundError,
            EmbeddingError,
            ConfigurationError,
        ]

        for exc_class in exceptions:
            assert isinstance(exc_class, type)
            assert issubclass(exc_class, Exception)

    def test_exception_naming_convention(self):
        """Every custom exception name is PascalCase and ends in Error/Exception."""
        for exc_class in self._custom_exceptions():
            name = exc_class.__name__
            assert name.endswith("Error") or name.endswith("Exception"), name
            assert name[0].isupper(), name
            assert "_" not in name, name
