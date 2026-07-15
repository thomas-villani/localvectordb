"""Unit tests for the pure patch-ops module (``localvectordb.patching``).

No database involvement: these exercise :func:`apply_ops` directly, which is the
verifiable core every patch surface compiles down to.
"""

import pytest

from localvectordb.exceptions import PatchError
from localvectordb.patching import PatchResult, apply_ops


class TestReplace:
    def test_single_replace(self):
        assert apply_ops("the quick brown fox", [{"op": "replace", "find": "brown", "replace": "red"}]) == (
            "the quick red fox"
        )

    def test_replace_count_matches_all(self):
        out = apply_ops("a a a", [{"op": "replace", "find": "a", "replace": "b", "count": 3}])
        assert out == "b b b"

    def test_unmatched_find_raises(self):
        with pytest.raises(PatchError, match="expected 1 occurrence"):
            apply_ops("hello", [{"op": "replace", "find": "zzz", "replace": "x"}])

    def test_ambiguous_find_raises(self):
        # 'o' occurs twice but count defaults to 1 -> ambiguous.
        with pytest.raises(PatchError, match="found 2"):
            apply_ops("the quick brown fox", [{"op": "replace", "find": "o", "replace": "0"}])

    def test_wrong_count_raises(self):
        with pytest.raises(PatchError):
            apply_ops("a a a", [{"op": "replace", "find": "a", "replace": "b", "count": 2}])

    def test_empty_find_raises(self):
        with pytest.raises(PatchError, match="non-empty"):
            apply_ops("hello", [{"op": "replace", "find": "", "replace": "x"}])

    def test_non_positive_count_raises(self):
        with pytest.raises(PatchError):
            apply_ops("aa", [{"op": "replace", "find": "a", "replace": "b", "count": 0}])

    def test_replace_matches_are_non_overlapping(self):
        # 'aa' in 'aaaa' matches at 0 and 2 (non-overlapping), so count must be 2.
        assert apply_ops("aaaa", [{"op": "replace", "find": "aa", "replace": "b", "count": 2}]) == "bb"


class TestSplice:
    def test_basic_splice(self):
        assert apply_ops("the quick brown fox", [{"op": "splice", "start": 4, "end": 9, "text": "slow"}]) == (
            "the slow brown fox"
        )

    def test_zero_width_insert(self):
        assert apply_ops("abc", [{"op": "splice", "start": 1, "end": 1, "text": "X"}]) == "aXbc"

    def test_delete_span(self):
        assert apply_ops("abcdef", [{"op": "splice", "start": 2, "end": 4, "text": ""}]) == "abef"

    @pytest.mark.parametrize("start,end", [(-1, 2), (0, 100), (3, 2)])
    def test_out_of_range_raises(self, start, end):
        with pytest.raises(PatchError, match="out of range"):
            apply_ops("abcdef", [{"op": "splice", "start": start, "end": end, "text": "x"}])

    def test_non_int_offset_raises(self):
        with pytest.raises(PatchError):
            apply_ops("abcdef", [{"op": "splice", "start": "0", "end": 1, "text": "x"}])

    def test_bool_offset_rejected(self):
        # bool is an int subclass but is not a valid offset.
        with pytest.raises(PatchError):
            apply_ops("abcdef", [{"op": "splice", "start": True, "end": 2, "text": "x"}])


class TestAppendPrepend:
    def test_append(self):
        assert apply_ops("hello", [{"op": "append", "text": "!"}]) == "hello!"

    def test_prepend(self):
        assert apply_ops("hello", [{"op": "prepend", "text": ">> "}]) == ">> hello"

    def test_append_and_prepend_together(self):
        out = apply_ops("hello", [{"op": "append", "text": "!"}, {"op": "prepend", "text": ">> "}])
        assert out == ">> hello!"


class TestMultiOp:
    def test_ops_resolve_against_original_offsets(self):
        # Both splices reference offsets in the ORIGINAL string, not each other's output.
        out = apply_ops(
            "abcdef",
            [
                {"op": "splice", "start": 0, "end": 1, "text": "X"},
                {"op": "splice", "start": 5, "end": 6, "text": "Y"},
            ],
        )
        assert out == "XbcdeY"

    def test_length_changing_prefix_edit_keeps_later_offsets_valid(self):
        # Inserting a long string at the front must not corrupt the later splice,
        # because offsets are resolved against the original.
        out = apply_ops(
            "0123456789",
            [
                {"op": "splice", "start": 0, "end": 0, "text": "INSERTED"},
                {"op": "splice", "start": 9, "end": 10, "text": "!"},
            ],
        )
        assert out == "INSERTED012345678!"

    def test_overlapping_ops_raise(self):
        with pytest.raises(PatchError, match="overlapping"):
            apply_ops(
                "abcdef",
                [
                    {"op": "splice", "start": 0, "end": 3, "text": "x"},
                    {"op": "splice", "start": 2, "end": 4, "text": "y"},
                ],
            )

    def test_adjacent_spans_are_allowed(self):
        out = apply_ops(
            "abcdef",
            [
                {"op": "splice", "start": 0, "end": 3, "text": "X"},
                {"op": "splice", "start": 3, "end": 6, "text": "Y"},
            ],
        )
        assert out == "XY"

    def test_replace_and_append_compose(self):
        out = apply_ops(
            "the quick brown fox",
            [
                {"op": "replace", "find": "brown", "replace": "red"},
                {"op": "append", "text": " runs"},
            ],
        )
        assert out == "the quick red fox runs"


class TestValidation:
    def test_empty_ops_raises(self):
        with pytest.raises(PatchError, match="non-empty"):
            apply_ops("hello", [])

    def test_unknown_op_raises(self):
        with pytest.raises(PatchError, match="unknown op"):
            apply_ops("hello", [{"op": "frobnicate", "text": "x"}])

    def test_missing_op_field_raises(self):
        with pytest.raises(PatchError):
            apply_ops("hello", [{"text": "x"}])

    def test_op_not_a_dict_raises(self):
        with pytest.raises(PatchError):
            apply_ops("hello", ["not-a-dict"])

    def test_atomic_no_partial_on_failure(self):
        # The first op is valid, the second is unmatched. Nothing should apply
        # (apply_ops returns a value or raises; there is no in-place mutation).
        with pytest.raises(PatchError):
            apply_ops(
                "the quick brown fox",
                [
                    {"op": "replace", "find": "brown", "replace": "red"},
                    {"op": "replace", "find": "zzz", "replace": "x"},
                ],
            )


class TestUnicode:
    def test_offsets_are_characters_not_bytes(self):
        # 'café' is 4 characters; the accented char is multi-byte in UTF-8. A
        # splice at char offset 4 must land after 'é', not mid-byte.
        text = "café au lait"
        out = apply_ops(text, [{"op": "splice", "start": 4, "end": 4, "text": "!"}])
        assert out == "café! au lait"

    def test_replace_across_multibyte(self):
        assert apply_ops("héllo wörld", [{"op": "replace", "find": "wörld", "replace": "planet"}]) == "héllo planet"


class TestPatchResult:
    def test_to_dict(self):
        r = PatchResult(updated=True, new_hash="abc", ops_applied=2)
        assert r.to_dict() == {"updated": True, "new_hash": "abc", "ops_applied": 2}
