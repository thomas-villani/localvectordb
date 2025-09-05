# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/exceptions.py
from typing import Union, List


class BaseLocalVectorDBException(Exception):
    pass

class DatabaseError(BaseLocalVectorDBException):
    pass

class DatabaseNotFoundError(DatabaseError, KeyError):
    """Raised if the Database cannot be found"""
    pass

class MetadataFilterError(DatabaseError, ValueError):
    """Raised when there's an error in metadata filter specification or processing"""
    pass

class DuplicateDocumentIDError(DatabaseError, ValueError):
    """Raised when inserting document(s) and the id(s) already exist"""
    pass

class DocumentNotFoundError(DatabaseError, KeyError):
    """Raised when one or more requested documents cannot be found"""

    def __init__(self, message: str, missing_ids: Union[str, List[str], None] = None):
        super().__init__(message)
        self.missing_ids = missing_ids if isinstance(missing_ids, list) else [missing_ids] if missing_ids else []

class EmbeddingError(BaseLocalVectorDBException, RuntimeError):
    pass

class OllamaNotFoundError(EmbeddingError):
    """Raised when Ollama is not installed or not running."""
    pass

class ConfigurationError(BaseLocalVectorDBException, RuntimeError):
    pass

class ConnectionPoolError(BaseLocalVectorDBException):
    pass
