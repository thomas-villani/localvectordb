"""Tests for lexical (grep-style) content search (``LocalVectorDB.grep``)."""

import pytest

from localvectordb import GrepMatch, LocalVectorDB

pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_path):
    database = LocalVectorDB(
        name="grep_test",
        base_path=str(tmp_path / "db"),
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    database.upsert(
        [
            "line one\nTODO: fix this\nline three\nanother TODO here\nend",
            "no match here\njust text",
            "def foo():\n    return 42\nclass Bar:\n    pass",
        ],
        ids=["docs/a", "docs/b", "code/c"],
    )
    return database


def test_literal_match_reports_position(db):
    matches = db.grep("TODO")
    assert all(isinstance(m, GrepMatch) for m in matches)
    assert [(m.doc_id, m.line_number, m.start, m.match) for m in matches] == [
        ("docs/a", 2, 0, "TODO"),
        ("docs/a", 4, 8, "TODO"),
    ]


def test_context_lines(db):
    first, second = db.grep("TODO", context=1)
    assert first.before == ["line one"]
    assert first.after == ["line three"]
    assert second.before == ["line three"]
    assert second.after == ["end"]


def test_before_after_override_context(db):
    m, _ = db.grep("TODO", context=5, before_context=0, after_context=1)
    assert m.before == []
    assert m.after == ["line three"]


def test_ignore_case(db):
    assert [m.line_number for m in db.grep("todo", ignore_case=True)] == [2, 4]
    assert db.grep("todo") == []


def test_regex(db):
    matches = db.grep(r"^(def|class)\s", regex=True)
    assert [(m.doc_id, m.line_number, m.match) for m in matches] == [
        ("code/c", 1, "def "),
        ("code/c", 3, "class "),
    ]


def test_invalid_regex_raises(db):
    with pytest.raises(ValueError):
        db.grep("(unclosed", regex=True)


def test_literal_metacharacters_are_not_regex(db):
    # A literal search for "foo()" must match the text, not be parsed as a regex.
    matches = db.grep("foo()")
    assert [(m.doc_id, m.line_number) for m in matches] == [("code/c", 1)]


def test_whole_word(db):
    # "other" appears only as a substring of "another"; whole-word must exclude it.
    assert db.grep("other", whole_word=True) == []
    assert [m.line_number for m in db.grep("other", whole_word=False)] == [4]


def test_prefix_scopes_corpus(db):
    assert {m.doc_id for m in db.grep("TODO", prefix="docs/")} == {"docs/a"}
    assert db.grep("def", prefix="docs/") == []


def test_where_filter(db):
    # `$startswith` is the id-prefix operator (this codebase's `$like` is literal
    # substring containment, not SQL LIKE).
    assert {m.doc_id for m in db.grep("return", where={"id": {"$startswith": "code/"}})} == {"code/c"}


def test_limit_caps_total(db):
    assert len(db.grep("TODO", limit=1)) == 1
    assert db.grep("TODO", limit=0) == []


def test_max_count_per_document(db):
    matches = db.grep("TODO", max_count=1)
    assert [(m.doc_id, m.line_number) for m in matches] == [("docs/a", 2)]


def test_empty_pattern_rejected(db):
    with pytest.raises(ValueError):
        db.grep("")


def test_to_dict_roundtrip(db):
    m = db.grep("TODO", context=1)[0]
    restored = GrepMatch.from_dict(m.to_dict())
    assert restored == m
