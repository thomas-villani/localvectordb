# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import importlib.metadata
import os
import re
from datetime import datetime
from typing import Optional, Union


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
