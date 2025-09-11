# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database/__init__.py
from __future__ import annotations

from localvectordb.database.base import BaseVectorDB
from localvectordb.database._core import LocalVectorDBCore
from localvectordb.database._ingest import PipelineMixin
from localvectordb.database._search import SearchMixin
from localvectordb.database._metadata import MetadataMixin
from localvectordb.database._crud import CrudMixin


class LocalVectorDB(PipelineMixin, SearchMixin, MetadataMixin, CrudMixin, LocalVectorDBCore):
    """Composition of domain-specific mixins with the base implementation.
    
    This preserves the public API: `from localvectordb.database import LocalVectorDB`.
    """
    pass

__all__ = ["LocalVectorDB", "BaseVectorDB"]

