# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/utils/__init__.py
"""
Shared utilities for LocalVectorDB server.
"""

from .checkdeps import check_ollama_installation, check_ollama_service
from .schema import parse_metadata_schema

__all__ = ['parse_metadata_schema', "check_ollama_service", "check_ollama_installation"]
