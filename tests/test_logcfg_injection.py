"""
Tests for log-injection defenses in localvectordb_server._logcfg.

A crafted exception message (or any user-derived value) containing a newline
must not be able to emit a physical log line that mimics a genuine top-level
record. The plain-text formatter indents continuation lines; the structured
formatter escapes them as part of JSON encoding.
"""

import json
import logging
import re
import sys

import pytest

from localvectordb_server._logcfg import (
    SafePlainFormatter,
    StructuredFormatter,
    sanitize_log_value,
)

# A payload that, if rendered verbatim on its own line, forges a CRITICAL record.
_FORGED = "AUDIT: admin key rotated"
_EVIL = f"boom\n2026-07-09 12:00:00 [CRITICAL] localvectordb.security: {_FORGED}"

# Matches a genuine top-level record start (timestamp at column 0).
_RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} .*\[CRITICAL\]", re.MULTILINE)


def _record_with_exc(message: str) -> logging.LogRecord:
    try:
        raise ValueError(message)
    except ValueError:
        rec = logging.LogRecord(
            name="localvectordb_server.routers.streaming",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="Streaming query failed",
            args=(),
            exc_info=sys.exc_info(),
        )
        return rec


@pytest.mark.unit
def test_sanitize_log_value_strips_control_chars():
    assert "\n" not in sanitize_log_value(_EVIL)
    assert "\r" not in sanitize_log_value("a\r\nb")
    assert sanitize_log_value("x" * 500).endswith("…")


@pytest.mark.unit
def test_plain_formatter_neutralizes_forged_record_via_exc_info():
    """exc_info bypasses call-site sanitization; the formatter must still be safe."""
    out = SafePlainFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s").format(_record_with_exc(_EVIL))
    # The forged content still appears (we don't drop data)...
    assert _FORGED in out
    # ...but never as a top-level record: no CRITICAL line starts at column 0.
    assert not _RECORD_START.search(out)
    # Continuation lines are indented with the guard prefix.
    assert "  | " in out


@pytest.mark.unit
def test_plain_formatter_preserves_traceback_readability():
    out = SafePlainFormatter("%(message)s").format(_record_with_exc("plain error"))
    assert "Traceback (most recent call last):" in out
    assert "ValueError: plain error" in out


@pytest.mark.unit
def test_structured_formatter_escapes_newlines():
    """JSON encoding keeps the whole record on one line regardless of payload."""
    out = StructuredFormatter().format(_record_with_exc(_EVIL))
    # Exactly one physical line.
    assert "\n" not in out
    # Valid JSON, and the payload lives inside the exception field as escaped text.
    parsed = json.loads(out)
    assert _FORGED in parsed["exception"]["message"]
