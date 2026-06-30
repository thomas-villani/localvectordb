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

from localvectordb.client import RemoteVectorDB
from localvectordb.database import LocalVectorDB

pytestmark = pytest.mark.unit

# Sync method -> its async counterpart.
SYNC_ASYNC_PAIRS = [
    ("query", "query_async"),
    ("query_cursor", "query_cursor_async"),
    ("query_stream", "query_stream_async"),
    ("query_multi_column", "query_multi_column_async"),
]

# Unified-query methods that must be call-compatible between local and remote.
LOCAL_REMOTE_METHODS = ["query", "query_async"]


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
