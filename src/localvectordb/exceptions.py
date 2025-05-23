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

class BaseLocalVectorDBException(Exception):
    pass

class DatabaseNotFoundError(BaseLocalVectorDBException, KeyError, FileNotFoundError):
    pass

class DuplicateDocumentIDError(BaseLocalVectorDBException, ValueError):
    pass

class OllamaNotFoundError(BaseLocalVectorDBException, RuntimeError):
    """Raised when Ollama is not installed or not running."""
    pass

class EmbeddingError(BaseLocalVectorDBException, RuntimeError):
    pass

class ConfigurationError(BaseLocalVectorDBException, RuntimeError):
    pass
