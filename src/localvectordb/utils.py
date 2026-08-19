import importlib.metadata
import os
import re
from datetime import datetime
from typing import Iterator, Optional, Sequence, TypeVar, Union

_T = TypeVar("_T")

# SQLite caps the number of bound parameters per statement at compile time:
# 999 before 3.32, 32,766 since. Any `IN (?,?,...)` list expanded from
# caller-supplied ids must be split into batches below the OLDER bound, or an
# upsert/get/delete of a large corpus dies with "too many SQL variables"
# (first seen at the 50k scale of the tier-2 insert benchmark). 900 leaves
# headroom for the handful of fixed parameters some statements add.
SQLITE_MAX_VARS = 900


def iter_sql_id_batches(items: Sequence[_T], batch_size: int = SQLITE_MAX_VARS) -> Iterator[Sequence[_T]]:
    """Yield slices of ``items`` sized to fit one SQL statement's variable limit."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def describe_exception(exc: BaseException) -> str:
    """Render an exception as ``Type: message``, never as an empty string.

    ``httpx.ReadTimeout`` -- the single most common real failure in a bulk ingest
    or a cross-encoder rerank -- carries no message at all, so the idiomatic
    ``f"...: {e}"`` logs its prefix and nothing else, and an operator watching a
    stalled job sees a blank line followed by a retry. The type name is the
    diagnostic here; the message is the optional part.
    """
    name = type(exc).__name__
    message = str(exc).strip()
    return f"{name}: {message}" if message else name


def resolve_env_ref(value: Optional[str], *, what: str = "value") -> Optional[str]:
    """Resolve a ``$ENV_VAR`` reference to its environment value.

    Credentials may be passed as a literal string or as a ``$NAME`` reference
    (all-uppercase) to be read from the environment. If a reference is given but
    the variable is unset, raise a clear error naming the variable rather than
    silently returning ``None`` (which surfaces later as a confusing
    "key required" failure). Non-reference values are returned unchanged.
    """
    if value is not None and value.startswith("$") and value[1:].isupper():
        env_name = value[1:]
        resolved = os.getenv(env_name)
        if resolved is None:
            raise ValueError(f"Environment variable {env_name!r} referenced by {what} is not set")
        return resolved
    return value


def get_system_version() -> str:
    try:
        system_version = importlib.metadata.version("localvectordb")
    except importlib.metadata.PackageNotFoundError:
        system_version = "dev"
    return system_version


def make_filename_safe(name: str, max_length: int = 255) -> str:
    # Define invalid characters based on the operating system
    if os.name == "nt":  # Windows
        invalid_chars = r'[<>:"/\\|?*]'
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
    else:  # POSIX-compliant systems (Linux, macOS)
        invalid_chars = r"[/:]"
        reserved_names = set()  # No reserved names typically on POSIX

    # Replace invalid characters with an underscore
    safe_name = re.sub(invalid_chars, "_", name)

    # Strip leading/trailing spaces and periods (applies to Windows)
    if os.name == "nt":
        safe_name = safe_name.strip(" .")
    else:
        safe_name = safe_name.strip()

    # Ensure the name is not a reserved name (Windows only)
    if os.name == "nt" and safe_name.upper() in reserved_names:
        safe_name += "_safe"

    # Truncate to maximum length (ensure allowance for file extensions)
    safe_name = safe_name[:max_length]

    # Return a fallback name if the result is empty
    return safe_name


def parse_iso8601(s: Union[str, datetime]) -> datetime:
    """
    Parse an ISO 8601 datetime string with automatic Z suffix handling.

    This function centralizes datetime parsing logic to handle the common
    case where ISO 8601 strings end with 'Z' (UTC timezone), which
    datetime.fromisoformat() cannot parse directly.

    Parameters
    ----------
    s : Union[str, datetime]
        ISO 8601 datetime string or datetime object. If already a datetime,
        returns it unchanged.

    Returns
    -------
    datetime
        Parsed datetime object with timezone information preserved.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a valid datetime.

    Examples
    --------
    >>> parse_iso8601("2023-12-01T10:30:00Z")
    datetime.datetime(2023, 12, 1, 10, 30, tzinfo=datetime.timezone.utc)

    >>> parse_iso8601("2023-12-01T10:30:00+00:00")
    datetime.datetime(2023, 12, 1, 10, 30, tzinfo=datetime.timezone.utc)

    >>> parse_iso8601("2023-12-01T10:30:00")
    datetime.datetime(2023, 12, 1, 10, 30)
    """
    if isinstance(s, datetime):
        return s

    if not isinstance(s, str):
        raise ValueError(f"Expected str or datetime, got {type(s)}")

    # Handle the common case where ISO 8601 strings end with 'Z' (UTC)
    # datetime.fromisoformat() can't parse 'Z', but can parse '+00:00'
    normalized_string = s.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(normalized_string)
    except ValueError as e:
        raise ValueError(f"Unable to parse datetime string '{s}': {e}") from e
