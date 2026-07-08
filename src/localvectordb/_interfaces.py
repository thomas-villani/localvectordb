"""Shared typing interfaces for LocalVectorDB.

This module holds the structural (``Protocol``) contracts that are referenced by
*both* the database backends (``localvectordb.database.base``) and the concrete
query builders (``localvectordb.query_builder`` / ``localvectordb.client``). It
deliberately imports only from :mod:`localvectordb.core` (plus ``typing``) so it
sits at the bottom of the import graph: keeping it dependency-light lets
``base.py`` reference :class:`QueryBuilderInterface` without creating an
``base`` <-> ``query_builder`` import cycle.

``QueryBuilderInterface`` is re-exported from :mod:`localvectordb.query_builder`
for backwards compatibility, so existing ``from localvectordb.query_builder
import QueryBuilderInterface`` imports keep working.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Protocol, Union, runtime_checkable

from localvectordb.core import QueryResult


@runtime_checkable
class QueryBuilderInterface(Protocol):
    """Portable fluent contract shared by the local and remote query builders.

    This :class:`typing.Protocol` declares the subset of the fluent API that
    **both** :class:`~localvectordb.query_builder.QueryBuilder` (local, SQLite +
    FAISS) and ``RemoteQueryBuilder`` (HTTP, server-side execution) can honour
    identically. A chain written against this surface -- e.g.
    ``db.query_builder().search(...).filter(...).order_by(...).limit(...).execute()``
    -- behaves the same whether ``db`` is a ``LocalVectorDB`` or a
    ``RemoteVectorDB``. ``VectorDB()`` therefore returns this type so callers can
    migrate local-to-remote without touching their query code.

    A ``Protocol`` (structural typing) is used deliberately instead of an ABC to
    avoid MRO/method-shadowing issues in the local mixin hierarchy; conformance
    is asserted statically in ``query_builder.py`` and in ``client.py``.

    Deliberately excluded (not portable, hence not part of the shared contract):

    - ``search_level`` / ``semantic_dedup`` -- accepted locally but the server's
      QueryBuilder-execute endpoint has no state field for them, so the remote
      backend cannot honour them (it raises ``NotImplementedError``).
    - ``having`` / ``having_count``, ``rerank*``, ``explain``, ``validate``,
      ``debug_info``, ``cursor`` / ``stream`` / ``get_execution_plan*`` -- local
      only; the remote backend has no server-side equivalent.

    Note also that granular arguments that the server does not read
    (``context`` window size, ``documents`` scoring options) are accepted by the
    remote builder for signature parity but currently governed by server
    defaults; see the corresponding ``RemoteQueryBuilder`` methods.
    """

    # Search
    def search(
        self,
        query: str,
        search_type: Optional[str] = None,
        vector_weight: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> "QueryBuilderInterface": ...

    def search_field(self, field: str, query: str) -> "QueryBuilderInterface": ...

    def filter(self, field: Optional[str] = None, value: Any = None, **kwargs: Any) -> "QueryBuilderInterface": ...

    def semantic_filter(
        self, field: str, concept: str, threshold: float = ..., metric: Any = ...
    ) -> "QueryBuilderInterface": ...

    def vector(self, query: str, score_threshold: Optional[float] = None) -> "QueryBuilderInterface": ...

    def keyword(self, query: str, score_threshold: Optional[float] = None) -> "QueryBuilderInterface": ...

    def hybrid(
        self, query: str, vector_weight: float = ..., score_threshold: Optional[float] = None
    ) -> "QueryBuilderInterface": ...

    # Return-type configuration
    def documents(
        self, scoring_method: Any = ..., scoring_options: Optional[dict] = None
    ) -> "QueryBuilderInterface": ...

    def chunks(self) -> "QueryBuilderInterface": ...

    def sections(self) -> "QueryBuilderInterface": ...

    def context(self, window_size: int = ...) -> "QueryBuilderInterface": ...

    # Ordering
    def order_by(self, field: str, direction: str = ...) -> "QueryBuilderInterface": ...

    def order_by_score(self, direction: str = ...) -> "QueryBuilderInterface": ...

    def clear_ordering(self) -> "QueryBuilderInterface": ...

    # Pagination
    def limit(self, n: int) -> "QueryBuilderInterface": ...

    def offset(self, n: int) -> "QueryBuilderInterface": ...

    # Grouping / aggregation
    def group_by(self, *fields: str) -> "QueryBuilderInterface": ...

    def aggregate(self, field: str, function: Any, alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    def count_by(self, field: str = ..., alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    def sum_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    def avg_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    def min_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    def max_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilderInterface": ...

    # Execution
    #
    # ``execute`` narrows to ``List[QueryResult]`` for the common (non-streaming)
    # call both backends share; the local overloads still satisfy it. ``execute_async``
    # is declared with the local backend's wider union return so its
    # (non-overloaded) signature conforms -- the remote backend returns the
    # ``List`` branch, which is a subtype.
    def execute(self) -> List[QueryResult]: ...

    async def execute_async(self) -> Union[List[QueryResult], Iterator[List[QueryResult]]]: ...

    def count(self) -> int: ...

    async def count_async(self) -> int: ...
