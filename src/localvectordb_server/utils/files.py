"""Filesystem-safety helpers.

Provides a dependency-free ``secure_filename`` so the server does not pull in
Werkzeug (a Flask-stack dependency left over from the pre-FastAPI server). The
implementation mirrors ``werkzeug.utils.secure_filename`` semantics: strip
directory components, transliterate to ASCII, keep only a safe character set,
and guard against Windows reserved device names.
"""

import os
import re
import unicodedata

__all__ = ["secure_filename"]

_FILENAME_STRIP_RE = re.compile(r"[^A-Za-z0-9_.-]")
_WINDOWS_DEVICE_FILES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(10)),
    *(f"LPT{i}" for i in range(10)),
}


def secure_filename(filename: str) -> str:
    """Return a filename safe to use on the local filesystem.

    Path separators are removed, the name is transliterated to ASCII, only
    ``[A-Za-z0-9_.-]`` characters are kept, and Windows reserved device names
    are prefixed with an underscore. Returns an empty string if nothing usable
    remains (callers should supply their own fallback name in that case).
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in (os.sep, os.path.altsep):
        if sep:
            filename = filename.replace(sep, " ")

    filename = _FILENAME_STRIP_RE.sub("", "_".join(filename.split())).strip("._")

    if os.name == "nt" and filename and filename.split(".")[0].upper() in _WINDOWS_DEVICE_FILES:
        filename = f"_{filename}"

    return filename
