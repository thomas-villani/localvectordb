"""Tests for the SQL filter builder (``localvectordb/_filters.py``).

This module is the SQL-injection defense surface for metadata filtering, yet it
previously ran only in production (QueryBuilder tests mock ``db.filter``). These
tests exercise identifier validation, parameterization, logical operators, and
FTS sanitization directly — including executing generated clauses against a real
SQLite database to prove malicious values cannot inject.
"""

import sqlite3

import pytest

from localvectordb._filters import (
    FilterQueryBuilder,
    FTSQuerySanitization,
    _validate_and_quote_identifier,
    validate_filter_spec,
)
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DatabaseError, MetadataFilterError


def _schema():
    return {
        "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "rating": MetadataField(type=MetadataFieldType.REAL, indexed=True),
        "tags": MetadataField(type=MetadataFieldType.JSON),
    }


class TestIdentifierValidation:
    def test_valid_identifier_is_quoted(self):
        assert _validate_and_quote_identifier("author") == '"author"'
        assert _validate_and_quote_identifier("_field2") == '"_field2"'

    @pytest.mark.parametrize(
        "bad",
        [
            "author; DROP TABLE documents",
            "field OR 1=1",
            "1field",  # cannot start with a digit
            "with space",
            'quote"inside',
            "",
            "author--",
        ],
    )
    def test_injection_identifiers_are_rejected(self, bad):
        with pytest.raises(DatabaseError):
            _validate_and_quote_identifier(bad)


class TestFilterQueryBuilder:
    def test_simple_equality_is_parameterized(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"author": "Jane"})
        assert "?" in clause
        assert "Jane" not in clause  # value must not be inlined
        assert params == ["Jane"]

    def test_unknown_field_is_rejected(self):
        builder = FilterQueryBuilder(_schema())
        with pytest.raises(DatabaseError):
            builder.build_where_clause({"nonexistent": "x"})

    def test_injection_in_field_name_is_rejected(self):
        builder = FilterQueryBuilder(_schema())
        with pytest.raises(DatabaseError):
            builder.build_where_clause({"author OR 1=1": "x"})

    def test_logical_operators_structure_and_params(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"$or": [{"author": "A"}, {"author": "B"}]})
        assert " OR " in clause
        assert params == ["A", "B"]

    def test_and_not_operators(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"$and": [{"author": "A"}, {"$not": {"rating": {"$lt": 3}}}]})
        assert " AND " in clause and "NOT" in clause
        assert params == ["A", 3.0]  # rating coerced to REAL

    def test_in_operator_lists_params(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"author": {"$in": ["A", "B", "C"]}})
        assert "IN" in clause
        assert params == ["A", "B", "C"]

    def test_order_by_valid_and_rejects_injection(self):
        builder = FilterQueryBuilder(_schema())
        assert builder.build_order_by_clause("rating DESC") == 'ORDER BY "rating" DESC'
        for bad in ["rating; DROP TABLE documents", "evil_field ASC", "rating SIDEWAYS"]:
            with pytest.raises(DatabaseError):
                builder.build_order_by_clause(bad)


class TestValidateFilterSpec:
    """validate_filter_spec keeps the in-memory filter path (query(filters=...))
    consistent with the SQL path (filter(where=...)): unknown fields and
    operators raise instead of silently matching nothing."""

    def test_valid_specs_pass(self):
        schema = _schema()
        validate_filter_spec({}, schema)
        validate_filter_spec({"author": "Jane"}, schema)
        validate_filter_spec({"rating": {"$gte": 3, "$lte": 5}}, schema)
        validate_filter_spec({"tags": {"$contains": "python"}}, schema)
        validate_filter_spec(
            {"$and": [{"author": "A"}, {"$or": [{"rating": {"$gt": 1}}, {"$not": {"author": "B"}}]}]},
            schema,
        )

    def test_reserved_columns_allowed(self):
        validate_filter_spec({"id": "doc_1", "created_at": {"$exists": True}}, _schema())

    def test_dot_notation_validates_first_segment(self):
        validate_filter_spec({"tags.nested": "x"}, _schema())
        with pytest.raises(DatabaseError, match="not found in metadata schema"):
            validate_filter_spec({"unknown.nested": "x"}, _schema())

    def test_unknown_field_raises(self):
        with pytest.raises(DatabaseError, match="'nonexistent' not found in metadata schema"):
            validate_filter_spec({"nonexistent": "x"}, _schema())

    def test_raises_metadata_filter_error_specifically(self):
        """The specific type matters: MetadataFilterError is also a ValueError and
        maps to HTTP 400 (INVALID_FILTER) on the server, unlike a bare DatabaseError."""
        with pytest.raises(MetadataFilterError):
            validate_filter_spec({"nonexistent": "x"}, _schema())
        with pytest.raises(MetadataFilterError):
            validate_filter_spec({"rating": {"$between": [1, 5]}}, _schema())

    def test_unknown_field_inside_logical_operator_raises(self):
        with pytest.raises(DatabaseError, match="not found in metadata schema"):
            validate_filter_spec({"$and": [{"author": "A"}, {"nonexistent": 1}]}, _schema())
        with pytest.raises(DatabaseError, match="not found in metadata schema"):
            validate_filter_spec({"$not": {"nonexistent": 1}}, _schema())

    def test_unknown_operator_raises(self):
        with pytest.raises(DatabaseError, match="Unsupported operator"):
            validate_filter_spec({"rating": {"$between": [1, 5]}}, _schema())
        with pytest.raises(DatabaseError, match="Unsupported operator"):
            validate_filter_spec({"$xor": [{"author": "A"}]}, _schema())

    def test_malformed_logical_operators_raise(self):
        with pytest.raises(DatabaseError):
            validate_filter_spec({"$and": {"author": "A"}}, _schema())  # must be a list
        with pytest.raises(DatabaseError):
            validate_filter_spec({"$not": [{"author": "A"}]}, _schema())  # must be a dict


class TestNoInjectionAgainstRealSqlite:
    """Execute generated clauses against a real DB to prove values can't inject."""

    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, author TEXT, rating REAL)")
        conn.execute("INSERT INTO docs (author, rating) VALUES ('Jane', 4.5)")
        conn.execute("INSERT INTO docs (author, rating) VALUES ('Bob', 3.0)")
        conn.commit()
        return conn

    def test_malicious_value_is_treated_as_literal(self):
        builder = FilterQueryBuilder(_schema())
        payload = "'; DROP TABLE docs;--"
        clause, params = builder.build_where_clause({"author": payload})
        assert params == [payload]  # the injection string is bound, not inlined

        conn = self._db()
        # executescript would run multiple statements; parameterized execute cannot.
        rows = conn.execute(f"SELECT id FROM docs WHERE {clause}", params).fetchall()
        assert rows == []  # no author literally equals the payload

        # The table must still exist and be intact — no DROP occurred.
        count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        assert count == 2
        conn.close()

    def test_legitimate_filter_matches_expected_rows(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"rating": {"$gte": 4.0}})
        conn = self._db()
        rows = conn.execute(f"SELECT author FROM docs WHERE {clause}", params).fetchall()
        assert [r[0] for r in rows] == ["Jane"]
        conn.close()


class TestJsonContainsAgainstRealSqlite:
    """Regression: $contains/$not_contains on JSON fields reused one bound
    parameter for two SQL placeholders, so executing the clause raised
    sqlite3.ProgrammingError (binding count mismatch)."""

    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, author TEXT, rating REAL, tags TEXT)")
        conn.execute("INSERT INTO docs (author, rating, tags) VALUES ('Jane', 4.5, '[\"python\", \"tutorial\"]')")
        conn.execute("INSERT INTO docs (author, rating, tags) VALUES ('Bob', 3.0, '[\"golang\"]')")
        conn.commit()
        return conn

    def test_contains_binds_one_param_per_placeholder(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"tags": {"$contains": "python"}})
        assert clause.count("?") == len(params)

        conn = self._db()
        rows = conn.execute(f"SELECT author FROM docs WHERE {clause}", params).fetchall()
        assert [r[0] for r in rows] == ["Jane"]
        conn.close()

    def test_not_contains_binds_one_param_per_placeholder(self):
        builder = FilterQueryBuilder(_schema())
        clause, params = builder.build_where_clause({"tags": {"$not_contains": "python"}})
        assert clause.count("?") == len(params)

        conn = self._db()
        rows = conn.execute(f"SELECT author FROM docs WHERE {clause}", params).fetchall()
        assert [r[0] for r in rows] == ["Bob"]
        conn.close()


class TestFTSSanitization:
    def test_multi_term_becomes_or_of_quoted_terms(self):
        # OR, not AND: BM25 ranks partial matches, and requiring every term (stopwords
        # included) made real queries match nothing. See tests/test_keyword_search_semantics.py.
        assert FTSQuerySanitization.sanitize_fts_query("hello world") == '"hello" OR "world"'

    def test_single_term_quoted(self):
        assert FTSQuerySanitization.sanitize_fts_query("python") == '"python"'

    def test_special_characters_are_stripped(self):
        # FTS5 special characters must be removed from bare terms.
        out = FTSQuerySanitization.sanitize_fts_query("foo(bar)")
        assert out == '"foobar"'
        assert "(" not in out and ")" not in out

    def test_empty_query_returns_empty(self):
        assert FTSQuerySanitization.sanitize_fts_query("") == ""
        assert FTSQuerySanitization.sanitize_fts_query("   ") == ""

    def test_clean_term_removes_fts_operators(self):
        assert FTSQuerySanitization.clean_term("term*^:") == "term"
