"""API-surface parity guards.

These tests lock in two invariants that the v0.1.0 audit found broken:

1. Sync/async parity — every ``x`` / ``x_async`` query method must share the
   same parameter names, defaults, and kinds. (Regression guard for the bug
   where ``query()`` defaulted ``search_type='vector'`` while ``query_async()``
   defaulted ``'hybrid'`` — identical code returned different results.)

2. Local/remote parity — ``RemoteVectorDB`` must mirror ``LocalVectorDB``'s
   unified-query signatures, so the ``VectorDB()`` factory's "swap local for
   remote, same API" promise actually holds.

They compare (default, kind) per parameter, deliberately ignoring annotations
(the local impl and the HTTP client legitimately differ on ``Literal`` option
lists). They run purely via ``inspect`` — no database or network needed.
"""

import inspect

import pytest

from localvectordb.client import RemoteQueryBuilder, RemoteVectorDB
from localvectordb.database import LocalVectorDB
from localvectordb.query_builder import QueryBuilder, QueryBuilderInterface

pytestmark = pytest.mark.unit

# The portable fluent contract both query builders must expose (mirrors
# ``QueryBuilderInterface``). A chain restricted to these methods is guaranteed
# to build and execute identically on a LocalVectorDB or a RemoteVectorDB.
QUERY_BUILDER_INTERFACE_METHODS = [
    "search",
    "search_field",
    "filter",
    "semantic_filter",
    "vector",
    "keyword",
    "hybrid",
    "documents",
    "chunks",
    "sections",
    "context",
    "order_by",
    "order_by_score",
    "clear_ordering",
    "limit",
    "offset",
    "group_by",
    "aggregate",
    "count_by",
    "sum_by",
    "avg_by",
    "min_by",
    "max_by",
    "execute",
    "execute_async",
    "count",
    "count_async",
]

# Sync method -> its async counterpart.
SYNC_ASYNC_PAIRS = [
    ("query", "query_async"),
    ("query_cursor", "query_cursor_async"),
    ("query_stream", "query_stream_async"),
    ("query_multi_column", "query_multi_column_async"),
    ("patch", "patch_async"),
]

# Unified-query methods that must be call-compatible between local and remote.
LOCAL_REMOTE_METHODS = ["query", "query_async", "patch", "patch_async"]


def _param_spec(cls, name):
    """Return {param_name: (default, kind)} for a method, excluding ``self``.

    Annotations are intentionally excluded — we care about call compatibility
    (names, defaults, positional/keyword kind), not type hints.
    """
    method = getattr(cls, name, None)
    assert method is not None, f"{cls.__name__} is missing expected method {name!r}"
    params = inspect.signature(method).parameters
    return {n: (p.default, p.kind) for n, p in params.items() if n != "self"}


def _describe_diff(a, b):
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in set(a) | set(b) if a.get(k) != b.get(k)}


@pytest.mark.parametrize("sync_name, async_name", SYNC_ASYNC_PAIRS)
def test_sync_async_signature_parity(sync_name, async_name):
    """A query method and its _async twin must have identical call signatures."""
    sync_spec = _param_spec(LocalVectorDB, sync_name)
    async_spec = _param_spec(LocalVectorDB, async_name)
    assert (
        sync_spec == async_spec
    ), f"{sync_name}() and {async_name}() signatures diverge: {_describe_diff(sync_spec, async_spec)}"


@pytest.mark.parametrize("name", LOCAL_REMOTE_METHODS)
def test_local_remote_signature_parity(name):
    """RemoteVectorDB must mirror LocalVectorDB's unified-query signatures."""
    local_spec = _param_spec(LocalVectorDB, name)
    remote_spec = _param_spec(RemoteVectorDB, name)
    assert local_spec == remote_spec, (
        f"LocalVectorDB.{name}() and RemoteVectorDB.{name}() signatures diverge: "
        f"{_describe_diff(local_spec, remote_spec)}"
    )


def test_query_default_search_type_is_hybrid():
    """The unified query default is 'hybrid' everywhere (explicit value guard)."""
    for cls in (LocalVectorDB, RemoteVectorDB):
        for name in ("query", "query_async"):
            default = _param_spec(cls, name)["search_type"][0]
            assert default == "hybrid", f"{cls.__name__}.{name}() default search_type is {default!r}, expected 'hybrid'"


# --------------------------------------------------------------------------- #
# QueryBuilder local/remote parity (finding R2).
#
# A fluent ``db.query_builder()....`` chain must be portable between backends.
# These guards lock the shared surface (``QueryBuilderInterface``), the aligned
# signatures/defaults, and the wire state the remote builder serializes.
# --------------------------------------------------------------------------- #


def _stub_remote_builder() -> RemoteQueryBuilder:
    """A RemoteQueryBuilder whose ``db`` is never touched (state building only)."""
    return RemoteQueryBuilder(object())  # type: ignore[arg-type]


def test_both_query_builders_expose_shared_interface():
    """Local and remote builders expose every portable interface method."""
    local = QueryBuilder(object())  # type: ignore[arg-type]
    remote = _stub_remote_builder()
    for name in QUERY_BUILDER_INTERFACE_METHODS:
        assert hasattr(local, name), f"QueryBuilder missing shared method {name!r}"
        assert hasattr(remote, name), f"RemoteQueryBuilder missing shared method {name!r}"
    # runtime_checkable structural conformance
    assert isinstance(local, QueryBuilderInterface)
    assert isinstance(remote, QueryBuilderInterface)


def test_remote_query_builder_chain_builds_expected_state():
    """A representative portable chain serializes to the state the server reads."""
    builder = (
        _stub_remote_builder()
        .search("machine learning", search_type="vector")
        .filter("year", gte_=2020)
        .semantic_filter("methodology", "neural networks", threshold=0.8)
        .order_by("year")
        .limit(5)
        .offset(2)
    )
    state = builder.to_dict()

    assert state["search_clauses"] == [
        {"text": "machine learning", "columns": None, "search_type": "vector", "score_threshold": None}
    ]
    assert state["exact_filters"] == [{"field": "year", "conditions": {"gte_": 2020}}]
    assert state["semantic_filters"] == [
        {"field": "methodology", "concept": "neural networks", "threshold": 0.8, "metric": "cosine"}
    ]
    # order_by default direction is 'desc' (aligned with the local builder).
    assert state["order_by"] == [{"field": "year", "direction": "desc"}]
    assert state["limit"] == 5
    assert state["offset"] == 2
    # to_dict keys the server reads must remain stable (backward-compatible wire).
    assert set(state) == {
        "search_clauses",
        "exact_filters",
        "semantic_filters",
        "search_type",
        "vector_weight",
        "return_type",
        "order_by",
        "limit",
        "offset",
        "group_by",
        "aggregations",
    }


def test_remote_search_first_param_is_query_and_positional():
    """Remote ``search`` first param aligns with local (name ``query``, positional)."""
    local_p = _param_spec(QueryBuilder, "search")
    remote_p = _param_spec(RemoteQueryBuilder, "search")
    assert "query" in local_p and "query" in remote_p
    assert "text" not in remote_p  # old ``text`` first param is gone


def test_remote_order_by_default_direction_is_desc():
    """Remote ``order_by`` default direction matches the local builder ('desc')."""
    assert _param_spec(RemoteQueryBuilder, "order_by")["direction"][0] == "desc"
    assert _param_spec(QueryBuilder, "order_by")["direction"][0] == "desc"


def test_remote_search_field_serializes_like_local_ilike():
    """``search_field`` maps to an $ilike-style exact filter the server can rebuild."""
    state = _stub_remote_builder().search_field("title", "vector db").to_dict()
    assert state["exact_filters"] == [{"field": "title", "conditions": {"ilike_": "vector db"}}]


def test_remote_convenience_search_types():
    """vector/keyword/hybrid set the clause search_type and fold vector_weight."""
    assert _stub_remote_builder().vector("q").to_dict()["search_clauses"][0]["search_type"] == "vector"
    assert _stub_remote_builder().keyword("q").to_dict()["search_clauses"][0]["search_type"] == "keyword"
    hybrid_state = _stub_remote_builder().hybrid("q", vector_weight=0.3).to_dict()
    assert hybrid_state["search_clauses"][0]["search_type"] == "hybrid"
    assert hybrid_state["vector_weight"] == 0.3


@pytest.mark.parametrize("method, args", [("search_level", ("chunks",)), ("semantic_dedup", (0.9,))])
def test_remote_unsupported_methods_raise(method, args):
    """Methods with no server-side state field are gated with NotImplementedError."""
    with pytest.raises(NotImplementedError):
        getattr(_stub_remote_builder(), method)(*args)
