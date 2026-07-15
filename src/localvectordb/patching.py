"""Pure document patch operations (find/replace + span splice).

This module has **no database dependency**: it turns a document's current content
plus a list of ops into the new content string. Every higher-level surface
(:meth:`LocalVectorDB.patch`, the HTTP ``PATCH`` route, the ``patch_document`` MCP
tool, the CLI ``patch`` command) compiles down to :func:`apply_ops`.

Op shapes (``op`` selects the kind):

- ``{"op": "splice", "start": int, "end": int, "text": str}`` -- the primitive.
  ``start``/``end`` are **character** offsets into the original content
  (``0 <= start <= end <= len(content)``); ``content[start:end]`` is replaced by
  ``text``.
- ``{"op": "replace", "find": str, "replace": str, "count": int = 1}`` -- ``find``
  must occur **exactly** ``count`` times (non-overlapping) or the whole patch
  fails. Each occurrence becomes a splice.
- ``{"op": "append", "text": str}`` / ``{"op": "prepend", "text": str}`` -- the
  common no-anchor cases; map to zero-width splices at the ends.

All ops in one call are resolved against the **original** content (not applied
sequentially against each other's output), validated non-overlapping, and applied
atomically. Any unmatched/ambiguous ``find``, out-of-range or overlapping splice,
or malformed op raises :class:`PatchError` and produces no output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .exceptions import PatchError

__all__ = ["PatchResult", "apply_ops"]


@dataclass
class PatchResult:
    """Outcome of a patch.

    ``updated`` is ``False`` only when the ops produced content byte-for-byte
    identical to what was stored (and no metadata changed). ``new_hash`` is the
    SHA-256 of the resulting content; ``ops_applied`` is the number of ops in the
    call.
    """

    updated: bool
    new_hash: str
    ops_applied: int

    def to_dict(self) -> Dict[str, Any]:
        return {"updated": self.updated, "new_hash": self.new_hash, "ops_applied": self.ops_applied}


# A normalized splice: (start, end, replacement_text).
_Splice = Tuple[int, int, str]


def _fail(message: str) -> None:
    raise PatchError(message)


def _require_int(value: Any, label: str) -> int:
    # bool is an int subclass, but not a valid offset/count.
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(label)
    return int(value)


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        _fail(label)
    return str(value)


def _splice_from_op(op: Dict[str, Any], content: str, index: int) -> List[_Splice]:
    """Resolve a single op against the *original* ``content`` into splices."""
    if not isinstance(op, dict):
        _fail(f"op {index} must be an object, got {type(op).__name__}")
    kind = op.get("op")
    if not isinstance(kind, str):
        _fail(f"op {index} is missing a string 'op' field")

    if kind == "splice":
        start = _require_int(op.get("start"), f"op {index} (splice): 'start' must be an int")
        end = _require_int(op.get("end"), f"op {index} (splice): 'end' must be an int")
        text = _require_str(op.get("text"), f"op {index} (splice): 'text' must be a string")
        if not (0 <= start <= end <= len(content)):
            _fail(
                f"op {index} (splice): start={start}, end={end} out of range " f"for content of length {len(content)}"
            )
        return [(start, end, text)]

    if kind == "replace":
        find = _require_str(op.get("find"), f"op {index} (replace): 'find' must be a non-empty string")
        if find == "":
            _fail(f"op {index} (replace): 'find' must be a non-empty string")
        replace = _require_str(op.get("replace"), f"op {index} (replace): 'replace' must be a string")
        count = _require_int(op.get("count", 1), f"op {index} (replace): 'count' must be a positive int")
        if count < 1:
            _fail(f"op {index} (replace): 'count' must be a positive int")
        matches: List[int] = []
        cursor = 0
        while True:
            j = content.find(find, cursor)
            if j < 0:
                break
            matches.append(j)
            cursor = j + len(find)
        if len(matches) != count:
            raise PatchError(
                f"op {index} (replace): expected {count} occurrence(s) of {find!r} but found {len(matches)}; "
                f"refine the 'find' text or adjust 'count' so the match is unambiguous"
            )
        n = len(find)
        return [(j, j + n, replace) for j in matches]

    if kind == "append":
        text = _require_str(op.get("text"), f"op {index} (append): 'text' must be a string")
        return [(len(content), len(content), text)]

    if kind == "prepend":
        text = _require_str(op.get("text"), f"op {index} (prepend): 'text' must be a string")
        return [(0, 0, text)]

    raise PatchError(f"op {index}: unknown op {kind!r} (expected one of: splice, replace, append, prepend)")


def apply_ops(content: str, ops: List[Dict[str, Any]]) -> str:
    """Apply ``ops`` to ``content`` and return the new content.

    Ops resolve against the original ``content``. The resulting splices are sorted
    and validated non-overlapping, then applied in a single left-to-right pass.
    Raises :class:`PatchError` on any invalid, unmatched, ambiguous, or overlapping
    op; nothing is applied partially.
    """
    if not isinstance(ops, list) or len(ops) == 0:
        _fail("ops must be a non-empty list")

    splices: List[_Splice] = []
    for index, op in enumerate(ops):
        splices.extend(_splice_from_op(op, content, index))

    # Stable sort by (start, end): keeps op order among equal keys (e.g. two
    # inserts at the same offset apply in the order given).
    splices.sort(key=lambda s: (s[0], s[1]))

    # Reject overlaps: each splice must end at or before the next one's start.
    for (a_start, a_end, _), (b_start, _, _) in zip(splices, splices[1:], strict=False):
        if a_end > b_start:
            raise PatchError(
                f"overlapping ops: a change over [{a_start}, {a_end}) collides with one starting at {b_start}; "
                f"ops in a single patch must touch disjoint spans"
            )

    parts: List[str] = []
    cursor = 0
    for start, end, text in splices:
        parts.append(content[cursor:start])
        parts.append(text)
        cursor = end
    parts.append(content[cursor:])
    return "".join(parts)
