"""Helpers for SQLite URI database paths (e.g. shared-cache in-memory dbs).

A ``db_path`` such as ``file:memdb_x?mode=memory&cache=shared`` is a **SQLite URI**,
not a filesystem path. It must be handled carefully:

* It must be passed to ``sqlite3.connect`` / ``aiosqlite.connect`` with ``uri=True``,
  otherwise SQLite treats the whole string as a filename.
* It must **not** be wrapped in :class:`pathlib.Path`, which mangles the ``?query``
  portion. On Windows, a Path-wrapped URI passed to ``connect`` without ``uri=True``
  makes SQLite create a stray 0-byte ``file`` via NTFS alternate-data-stream
  semantics (``file:memdb_x`` -> base file ``file`` + stream ``memdb_x?...``), and
  silently degrades the "in-memory" database into an on-disk one.

Real filesystem paths are unaffected (``uri`` stays ``False``; they are wrapped in
``Path`` as before), so file-backed databases behave exactly as they did.
"""

from pathlib import Path
from typing import Union


def is_sqlite_uri(db_path: object) -> bool:
    """Return ``True`` if ``db_path`` is a SQLite URI (a ``str`` starting with ``file:``)."""
    return isinstance(db_path, str) and db_path.startswith("file:")


def normalize_db_path(db_path: Union[str, Path]) -> Union[str, Path]:
    """Keep SQLite URIs as raw strings; wrap real filesystem paths in :class:`Path`."""
    return db_path if is_sqlite_uri(db_path) else Path(db_path)
