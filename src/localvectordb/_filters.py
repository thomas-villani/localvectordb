"""Enhanced filtering system for LocalVectorDB with SQL query generation.

This module provides utilities for filtering vector database results based on metadata
criteria. It supports complex queries with nested fields, comparison operators, and
logical combinations while generating safe parameterized SQL queries.

The filtering system supports:
    - Nested field access using dot notation (limited in SQL)
    - Comparison operators ($eq, $ne, $gt, $lt, $gte, $lte, $in, $nin)
    - String operations ($like, $ilike, $contains, $startswith, $endswith)
    - Array operations ($contains, $not_contains for JSON fields)
    - Existence checks ($exists, $not_exists)
    - Logical operators ($and, $or, $not)
    - Type checking ($type)
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DatabaseError, MetadataFilterError

# Pattern for valid SQL identifiers (alphanumeric and underscores, starting with letter or underscore)
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_and_quote_identifier(name: str) -> str:
    """Validate and quote a SQL identifier for safe use in queries.

    This function validates the identifier against a strict pattern and
    wraps it in double quotes with proper escaping for defense in depth.

    Parameters
    ----------
    name : str
        The identifier to validate and quote

    Returns
    -------
    str
        The safely quoted identifier (e.g., '"field_name"')

    Raises
    ------
    DatabaseError
        If the identifier contains unsafe characters
    """
    if not _SAFE_IDENTIFIER_PATTERN.match(name):
        raise DatabaseError(
            f"Invalid SQL identifier '{name}': must contain only alphanumeric "
            "characters and underscores, and start with a letter or underscore"
        )
    # Escape any embedded double quotes (shouldn't happen due to validation, but defense in depth)
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


FILTER_OPERATORS = (
    "$eq",
    "$ne",
    "$gt",
    "$lt",
    "$gte",
    "$lte",
    "$like",
    "$ilike",
    "$in",
    "$nin",
    "$contains",
    "$not_contains",
    "$exists",
    "$not_exists",
    "$startswith",
    "$endswith",
    "$type",
)


class FilterQueryBuilder:
    """Builds safe SQL WHERE clauses from filter specifications."""

    # SQL operators mapping
    OPERATORS = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$lt": "<",
        "$gte": ">=",
        "$lte": "<=",
        "$like": "LIKE",
        "$ilike": "LIKE",  # Case-insensitive, handled with LOWER()
        "$in": "IN",
        "$nin": "NOT IN",
    }

    # Reserved column names that cannot be used as metadata fields
    RESERVED_COLUMNS = {"id", "content", "content_hash", "created_at", "updated_at", "rowid"}

    def __init__(self, metadata_schema: Dict[str, MetadataField]):
        """Initialize the query builder with metadata schema.

        Parameters
        ----------
        metadata_schema : Dict[str, MetadataField]
            The database metadata schema for validation
        """
        self.metadata_schema = metadata_schema
        self.params: list[Any] = []
        self.param_counter = 0

    def _get_next_param_name(self) -> str:
        """Generate next parameter placeholder name."""
        self.param_counter += 1
        return f"param_{self.param_counter}"

    def _add_param(self, value: Any) -> str:
        """Add a parameter and return its placeholder.

        Parameters
        ----------
        value : Any
            Value to add as parameter

        Returns
        -------
        str
            Parameter placeholder (e.g., "?")
        """
        self.params.append(value)
        return "?"

    def _validate_field_name(self, field: str) -> None:
        """Validate that field name is safe and exists in schema.

        Parameters
        ----------
        field : str
            Field name to validate

        Raises
        ------
        DatabaseError
            If field name is invalid or doesn't exist
        """
        # Check for SQL injection patterns
        if not _SAFE_IDENTIFIER_PATTERN.match(field):
            raise DatabaseError(f"Invalid field name: {field}")

        # Check if it's a reserved column
        if field.lower() in self.RESERVED_COLUMNS:
            return  # Allow reserved columns

        # Check if it exists in metadata schema
        if field not in self.metadata_schema:
            raise DatabaseError(f"Field '{field}' not found in metadata schema")

    def _get_field_type(self, field: str) -> Optional[MetadataFieldType]:
        """Get the type of a metadata field.

        Parameters
        ----------
        field : str
            Field name

        Returns
        -------
        Optional[MetadataFieldType]
            Field type or None if not in schema
        """
        if field.lower() in self.RESERVED_COLUMNS:
            # Handle reserved columns
            if field in ["created_at", "updated_at"]:
                return MetadataFieldType.DATE
            elif field in ["content_hash", "id"]:
                return MetadataFieldType.TEXT
            else:
                return MetadataFieldType.TEXT

        field_def = self.metadata_schema.get(field)
        return field_def.type if field_def else None  # type: ignore[return-value]

    def _convert_value_for_type(self, value: Any, field_type: MetadataFieldType) -> Any:
        """Convert value to appropriate type for SQL parameter.

        Parameters
        ----------
        value : Any
            Value to convert
        field_type : MetadataFieldType
            Target field type

        Returns
        -------
        Any
            Converted value
        """
        if value is None:
            return None

        if field_type == MetadataFieldType.JSON:
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return value
        elif field_type == MetadataFieldType.DATE:
            if isinstance(value, datetime):
                return value.isoformat()
            return value
        elif field_type == MetadataFieldType.BOOLEAN:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif field_type == MetadataFieldType.INTEGER:
            return int(value) if value is not None else None
        elif field_type == MetadataFieldType.REAL:
            return float(value) if value is not None else None
        else:
            return str(value) if value is not None else None

    def _build_simple_condition(self, field: str, operator: str, value: Any) -> str:
        """Build a simple field operator value condition.

        Parameters
        ----------
        field : str
            Field name
        operator : str
            SQL operator
        value : Any
            Comparison value

        Returns
        -------
        str
            SQL condition string
        """
        self._validate_field_name(field)
        field_type = self._get_field_type(field)

        # Convert value to appropriate type
        converted_value = self._convert_value_for_type(value, field_type) if field_type else value

        # Handle special cases
        if operator in ("IN", "NOT IN"):
            if not isinstance(value, (list, tuple)):
                raise DatabaseError(f"Operator {operator} requires a list/array value")
            if not value:
                # Empty list - handle specially
                return "1=0" if operator == "IN" else "1=1"

            # Convert all values
            converted_values = [self._convert_value_for_type(v, field_type) if field_type else v for v in value]
            placeholders = [self._add_param(v) for v in converted_values]
            placeholder_str = f"({', '.join(placeholders)})"
            return f"{field} {operator} {placeholder_str}"
        else:
            param_placeholder = self._add_param(converted_value)
            return f"{field} {operator} {param_placeholder}"

    def _build_json_condition(self, field: str, json_op: str, value: Any) -> str:
        """Build conditions for JSON field operations.

        Parameters
        ----------
        field : str
            JSON field name
        json_op : str
            JSON operation ($contains, $not_contains, etc.)
        value : Any
            Value to check

        Returns
        -------
        str
            SQL condition string
        """
        self._validate_field_name(field)
        field_type = self._get_field_type(field)

        if field_type != MetadataFieldType.JSON:
            raise DatabaseError(f"JSON operations only supported on JSON fields, {field} is {field_type}")

        if json_op == "$contains":
            # Check if JSON array contains value. Placeholders are positional
            # ("?"), so the value must be bound once per occurrence in the SQL.
            json_value = json.dumps(value)
            placeholder_1 = self._add_param(json_value)
            placeholder_2 = self._add_param(json_value)
            # Use JSON_EXTRACT or fallback to string search
            return (
                f"(json_extract({field}, '$') LIKE '%' || {placeholder_1} "
                f"|| '%' OR {field} LIKE '%' || {placeholder_2} || '%')"
            )
        elif json_op == "$not_contains":
            # Check if JSON array does not contain value
            json_value = json.dumps(value)
            placeholder_1 = self._add_param(json_value)
            placeholder_2 = self._add_param(json_value)
            return (
                f"NOT (json_extract({field}, '$') LIKE '%' || {placeholder_1} "
                f"|| '%' OR {field} LIKE '%' || {placeholder_2} || '%')"
            )
        else:
            raise DatabaseError(f"Unsupported JSON operation: {json_op}")

    def _build_string_condition(self, field: str, str_op: str, value: Any) -> str:
        """Build conditions for string operations.

        Parameters
        ----------
        field : str
            Field name
        str_op : str
            String operation ($startswith, $endswith, $contains)
        value : Any
            String value

        Returns
        -------
        str
            SQL condition string
        """
        self._validate_field_name(field)

        if not isinstance(value, str):
            value = str(value)

        if str_op == "$startswith":
            pattern = f"{value}%"
        elif str_op == "$endswith":
            pattern = f"%{value}"
        elif str_op == "$contains":
            pattern = f"%{value}%"
        else:
            raise DatabaseError(f"Unsupported string operation: {str_op}")

        param_placeholder = self._add_param(pattern)
        return f"{field} LIKE {param_placeholder}"

    def _build_existence_condition(self, field: str, exists: bool) -> str:
        """Build existence check condition.

        Parameters
        ----------
        field : str
            Field name
        exists : bool
            Whether field should exist

        Returns
        -------
        str
            SQL condition string
        """
        self._validate_field_name(field)

        if exists:
            return f"{field} IS NOT NULL"
        else:
            return f"{field} IS NULL"

    def _build_type_condition(self, field: str, expected_type: str) -> str:
        """Build type check condition.

        Parameters
        ----------
        field : str
            Field name
        expected_type : str
            Expected type name

        Returns
        -------
        str
            SQL condition string
        """
        self._validate_field_name(field)

        # Map type names to SQL checks
        type_checks = {
            "null": f"{field} IS NULL",
            "string": f"{field} IS NOT NULL AND typeof({field}) = 'text'",
            "number": f"{field} IS NOT NULL AND typeof({field}) IN ('integer', 'real')",
            "integer": f"{field} IS NOT NULL AND typeof({field}) = 'integer'",
            "real": f"{field} IS NOT NULL AND typeof({field}) = 'real'",
            "boolean": f"{field} IS NOT NULL AND {field} IN (0, 1)",
            "array": f"{field} IS NOT NULL AND substr({field}, 1, 1) = '['",
            "object": f"{field} IS NOT NULL AND substr({field}, 1, 1) = '{{'",
        }

        if expected_type not in type_checks:
            raise DatabaseError(f"Unsupported type check: {expected_type}")

        return type_checks[expected_type]

    def build_condition(self, field: str, condition: Any) -> str:
        """Build SQL condition from field and condition specification.

        Parameters
        ----------
        field : str
            Field name
        condition : Any
            Condition specification

        Returns
        -------
        str
            SQL WHERE condition
        """
        # Simple equality
        if not isinstance(condition, dict):
            return self._build_simple_condition(field, "=", condition)

        # Complex condition with operators
        conditions = []
        for op, value in condition.items():
            if op in self.OPERATORS:
                sql_op = self.OPERATORS[op]
                if op == "$ilike":
                    # Special handling for case-insensitive LIKE
                    self._validate_field_name(field)
                    if not isinstance(value, str):
                        value = str(value)
                    # Add wildcards if not present
                    if "%" not in value:
                        value = f"%{value}%"
                    param_placeholder = self._add_param(value.lower())
                    conditions.append(f"LOWER({field}) LIKE {param_placeholder}")
                else:
                    conditions.append(self._build_simple_condition(field, sql_op, value))
            elif op in ["$contains", "$not_contains"]:
                field_type = self._get_field_type(field)
                if field_type == MetadataFieldType.JSON:
                    conditions.append(self._build_json_condition(field, op, value))
                else:
                    # Treat as string contains
                    if op == "$contains":
                        conditions.append(self._build_string_condition(field, "$contains", value))
                    else:
                        conditions.append(f"NOT ({self._build_string_condition(field, '$contains', value)})")
            elif op in ["$startswith", "$endswith"]:
                conditions.append(self._build_string_condition(field, op, value))
            elif op == "$exists":
                conditions.append(self._build_existence_condition(field, value))
            elif op == "$not_exists":
                conditions.append(self._build_existence_condition(field, not value))
            elif op == "$type":
                conditions.append(self._build_type_condition(field, value))
            else:
                raise DatabaseError(f"Unsupported operator: {op}")

        if len(conditions) == 1:
            return conditions[0]
        else:
            return f"({' AND '.join(conditions)})"

    def build_order_by_clause(self, order_by: str, valid_columns: Optional[set] = None) -> str:
        """Build secure ORDER BY clause with proper identifier quoting.

        Parameters
        ----------
        order_by : str
            ORDER BY specification (e.g., "field_name DESC", "created_at")
        valid_columns : Optional[set]
            Set of valid column names. If None, uses schema + reserved columns.

        Returns
        -------
        str
            Safe SQL ORDER BY clause

        Raises
        ------
        DatabaseError
            If order_by format is invalid or contains unsafe field names
        """
        if not order_by or not order_by.strip():
            raise DatabaseError("ORDER BY clause cannot be empty")

        # Parse the order_by string
        order_parts = order_by.strip().split()
        if len(order_parts) == 0 or len(order_parts) > 2:
            raise DatabaseError("Invalid ORDER BY format. Use 'field_name' or 'field_name ASC/DESC'")

        field_name = order_parts[0].strip()
        direction = order_parts[1].upper() if len(order_parts) == 2 else "ASC"

        # Validate direction
        if direction not in ("ASC", "DESC"):
            raise DatabaseError("ORDER BY direction must be ASC or DESC")

        # Determine valid columns
        if valid_columns is None:
            valid_columns = set(self.RESERVED_COLUMNS)
            valid_columns.update(self.metadata_schema.keys())

        # Validate field exists in schema
        if field_name not in valid_columns:
            raise DatabaseError(f"Field '{field_name}' not found in schema. Valid fields: {sorted(valid_columns)}")

        # Validate and quote the field name for SQL safety (includes format validation)
        quoted_field = _validate_and_quote_identifier(field_name)
        return f"ORDER BY {quoted_field} {direction}"

    def build_where_clause(self, filter_spec: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build complete WHERE clause from filter specification.

        Parameters
        ----------
        filter_spec : Dict[str, Any]
            Filter specification

        Returns
        -------
        Tuple[str, List[Any]]
            SQL WHERE clause and parameters list
        """
        if not filter_spec:
            return "", []

        self.params = []
        self.param_counter = 0

        where_clause = self._build_where_recursive(filter_spec)
        return where_clause, self.params

    def _build_where_recursive(self, filter_spec: Dict[str, Any]) -> str:
        """Recursively build WHERE clause handling logical operators.

        Parameters
        ----------
        filter_spec : Dict[str, Any]
            Filter specification

        Returns
        -------
        str
            SQL WHERE clause
        """
        conditions = []

        for key, value in filter_spec.items():
            if key == "$and":
                if not isinstance(value, list):
                    raise DatabaseError("$and operator requires a list of conditions")
                and_conditions = [self._build_where_recursive(cond) for cond in value]
                if and_conditions:
                    conditions.append(f"({' AND '.join(and_conditions)})")
            elif key == "$or":
                if not isinstance(value, list):
                    raise DatabaseError("$or operator requires a list of conditions")
                or_conditions = [self._build_where_recursive(cond) for cond in value]
                if or_conditions:
                    conditions.append(f"({' OR '.join(or_conditions)})")
            elif key == "$not":
                if not isinstance(value, dict):
                    raise DatabaseError("$not operator requires a condition object")
                not_condition = self._build_where_recursive(value)
                conditions.append(f"NOT ({not_condition})")
            else:
                # Regular field condition
                conditions.append(self.build_condition(key, value))

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return f"({' AND '.join(conditions)})"
        else:
            return "1=1"  # No conditions


def validate_filter_spec(filter_spec: Dict[str, Any], metadata_schema: Dict[str, MetadataField]) -> None:
    """Validate a metadata filter specification against the database schema.

    Ensures that in-memory filtering (``matches_metadata_filter``) rejects the
    same inputs the SQL path (``FilterQueryBuilder``) rejects, instead of
    silently matching nothing: unknown field names and unsupported operators
    raise ``MetadataFilterError`` up front.

    Dot-notation fields (e.g. ``"author.name"``) are validated on their first
    segment only, since nested access into JSON metadata is supported by the
    in-memory matcher.

    Parameters
    ----------
    filter_spec : Dict[str, Any]
        Filter specification (same format accepted by ``filter(where=...)``
        and ``query(filters=...)``)
    metadata_schema : Dict[str, MetadataField]
        The database metadata schema to validate field names against

    Raises
    ------
    MetadataFilterError
        If a field is not in the schema or an operator is not supported.
        Subclasses ``DatabaseError`` and ``ValueError``; the HTTP server maps
        it to a 400 client error.
    """
    if not filter_spec:
        return

    if not isinstance(filter_spec, dict):
        raise MetadataFilterError(f"Filter specification must be a dict, got {type(filter_spec).__name__}")

    for key, value in filter_spec.items():
        if key == "$and" or key == "$or":
            if not isinstance(value, list):
                raise MetadataFilterError(f"{key} operator requires a list of conditions")
            for cond in value:
                validate_filter_spec(cond, metadata_schema)
        elif key == "$not":
            if not isinstance(value, dict):
                raise MetadataFilterError("$not operator requires a condition object")
            validate_filter_spec(value, metadata_schema)
        elif key.startswith("$"):
            raise MetadataFilterError(f"Unsupported operator: {key}")
        else:
            # Field condition: validate the field name (first dot segment)
            root = key.split(".", 1)[0]
            if root.lower() not in FilterQueryBuilder.RESERVED_COLUMNS and root not in metadata_schema:
                raise MetadataFilterError(f"Field '{root}' not found in metadata schema")
            # Operator-style condition: validate operator names
            if isinstance(value, dict):
                for op in value:
                    if op not in FILTER_OPERATORS:
                        raise MetadataFilterError(f"Unsupported operator: {op}")


# Legacy compatibility functions
def get_nested_value(data: dict, path: str) -> Any:
    """Get value from nested dictionary using dot notation path.

    Note: This function is kept for backward compatibility but is not used
    in SQL generation since SQLite doesn't support nested JSON queries easily.

    Parameters
    ----------
    data : dict
        Dictionary to search in
    path : str
        Path in dot notation (e.g., "author.name")

    Returns
    -------
    Any
        Value at the specified path, or None if not found
    """
    if not data:
        return None

    keys = path.split(".")
    value = data

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def check_metadata_condition(metadata: dict, field: str, condition: Union[dict, Any]) -> bool:
    """Check if metadata matches a single condition (in-memory filtering).

    This function is kept for backward compatibility and in-memory filtering
    when SQL generation is not needed.

    Parameters
    ----------
    metadata : dict
        Document metadata dictionary
    field : str
        Field to check (can use dot notation)
    condition : Union[dict, Any]
        Condition to check

    Returns
    -------
    bool
        Whether the condition is met
    """
    value = get_nested_value(metadata, field)

    # Direct equality comparison
    if not isinstance(condition, dict):
        return bool(value == condition)

    # Operator-based comparison
    for op, target in condition.items():
        if op == "$eq":
            return bool(value == target)
        elif op == "$ne":
            return bool(value != target)
        elif op == "$gt":
            return bool(value > target) if value is not None else False
        elif op == "$lt":
            return bool(value < target) if value is not None else False
        elif op == "$gte":
            return bool(value >= target) if value is not None else False
        elif op == "$lte":
            return bool(value <= target) if value is not None else False
        elif op == "$ilike":
            return str(target).lower() in str(value).lower() if value is not None else False
        elif op == "$like":
            return str(target) in str(value) if value is not None else False
        elif op == "$contains":
            if isinstance(value, list):
                return any(t in value for t in ([target] if not isinstance(target, list) else target))
            return target in str(value) if value is not None else False
        elif op == "$not_contains":
            if isinstance(value, list):
                return not all(t in value for t in ([target] if not isinstance(target, list) else target))
            return target not in str(value) if value is not None else True
        elif op == "$exists":
            return bool((value is not None) == target)
        elif op == "$not_exists":
            return bool((value is None) == target)
        elif op == "$in":
            return value in target if isinstance(target, (list, tuple)) and value is not None else False
        elif op == "$nin":
            return value not in target if isinstance(target, (list, tuple)) and value is not None else True
        elif op == "$startswith":
            return str(value).startswith(str(target)) if value is not None else False
        elif op == "$endswith":
            return str(value).endswith(str(target)) if value is not None else False
        elif op == "$type":
            if target == "null":
                return value is None
            elif target == "string":
                return isinstance(value, str)
            elif target == "number":
                return isinstance(value, (int, float))
            elif target == "integer":
                return isinstance(value, int)
            elif target == "real":
                return isinstance(value, float)
            elif target == "boolean":
                return isinstance(value, bool)
            elif target == "array":
                return isinstance(value, list)
            elif target == "object":
                return isinstance(value, dict)
            else:
                return False
        else:
            raise DatabaseError(f"Unsupported operator: {op}")

    return True


def matches_metadata_filter(doc_or_metadata, metadata_filter: dict) -> bool:
    """Check if a document matches metadata filter criteria (in-memory).

    This function is kept for backward compatibility and in-memory filtering.

    Parameters
    ----------
    doc_or_metadata : dict | object
        Document metadata or document object
    metadata_filter : dict
        Filter specification

    Returns
    -------
    bool
        Whether document matches all criteria
    """
    if not metadata_filter:
        return True

    if isinstance(doc_or_metadata, dict):
        metadata = doc_or_metadata
    else:
        # Try to get metadata from object
        if hasattr(doc_or_metadata, "metadata"):
            metadata = doc_or_metadata.metadata
        else:
            return False

    # Handle logical operators
    if "$and" in metadata_filter:
        return all(matches_metadata_filter(metadata, cond) for cond in metadata_filter["$and"])

    if "$or" in metadata_filter:
        return any(matches_metadata_filter(metadata, cond) for cond in metadata_filter["$or"])

    if "$not" in metadata_filter:
        return not matches_metadata_filter(metadata, metadata_filter["$not"])

    # Handle field conditions
    return all(
        check_metadata_condition(metadata, field, condition)
        for field, condition in metadata_filter.items()
        if not field.startswith("$")
    )


# FTS5 only recognises AND/OR/NOT as operators when they are written in uppercase and
# stand alone between two operands. Detection therefore runs against the *original* query,
# never against an uppercased copy: uppercasing is what turns an ordinary English "and"
# into an operator.
_FTS_BOOLEAN_OPERATOR = re.compile(r"\s(?:AND|OR|NOT)\s")
_FTS_BOOLEAN_SPLIT = re.compile(r"\s+(AND|OR|NOT)\s+")

# A cleaned term must retain at least one alphanumeric character to be a real FTS5 token.
# `clean_term` preserves hyphens and apostrophes, so "(+)-" survives cleaning as a bare "-",
# which tokenizes to nothing and only adds noise to the MATCH expression.
_HAS_ALPHANUMERIC = re.compile(r"[^\W_]", re.UNICODE)


class FTSQuerySanitization:

    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """
        Sanitize a user query into a safe FTS5 MATCH expression.

        Bare natural-language text is treated as a description of what the user wants:
        its terms are OR-joined so BM25 can rank partial matches, which is what BM25 is
        for. Requiring every term (including stopwords) to appear makes almost any
        real-world sentence match nothing at all.

        Explicit search syntax is treated as a precise instruction and given FTS5's own
        semantics:

        - ``"exact phrase"`` matches that phrase.
        - Uppercase ``AND`` / ``OR`` / ``NOT`` between operands are boolean operators,
          exactly as FTS5 defines them. Lowercase ``and`` / ``or`` / ``not`` are ordinary
          words, again exactly as FTS5 defines them.

        Every term is cleaned of FTS5 metacharacters and quoted, so no part of the user's
        input can be interpreted as syntax.

        Examples
        --------
        >>> FTSQuerySanitization.sanitize_fts_query("aspirin does not reduce risk")
        '"aspirin" OR "does" OR "not" OR "reduce" OR "risk"'
        >>> FTSQuerySanitization.sanitize_fts_query("aspirin AND risk")
        '"aspirin" AND "risk"'
        """
        if not query or not query.strip():
            return ""

        query = query.strip()

        # If the entire query is already quoted, treat as exact phrase
        if query.startswith('"') and query.endswith('"') and query.count('"') == 2:
            # Validate the phrase doesn't contain FTS5 special chars that could break things
            inner_query = query[1:-1]
            if FTSQuerySanitization.is_safe_phrase(inner_query):
                return query
            else:
                # Fall back to safe handling
                return f'"{FTSQuerySanitization.clean_term(inner_query)}"'

        # Check if query contains quotes for phrase matching
        if '"' in query:
            return FTSQuerySanitization.handle_phrase_query(query)

        # Boolean operators, per FTS5's rule: uppercase, and standing between operands.
        if _FTS_BOOLEAN_OPERATOR.search(query):
            return FTSQuerySanitization.handle_boolean_query(query)

        # Plain text: OR the terms and let BM25 rank the partial matches.
        return " OR ".join(FTSQuerySanitization.quote_terms(query))

    @staticmethod
    def quote_terms(text: str) -> List[str]:
        """Clean, filter and quote the whitespace-separated terms of ``text``.

        Quoting each term as a single-token phrase is what makes injection impossible:
        a cleaned term can never re-enter the expression as FTS5 syntax.
        """
        quoted = []
        for word in text.split():
            term = FTSQuerySanitization.clean_term(word)
            if term and _HAS_ALPHANUMERIC.search(term):
                quoted.append(f'"{term}"')
        return quoted

    @staticmethod
    def is_safe_phrase(phrase: str) -> bool:
        """Check if a phrase is safe to use in FTS5 without additional escaping"""
        # Avoid phrases with FTS5 special characters that could cause issues
        dangerous_chars = ["*", ":", "^", "(", ")", "[", "]", "{", "}"]
        return not any(char in phrase for char in dangerous_chars)

    @staticmethod
    def clean_term(term: str) -> str:
        """Clean a single term for safe FTS5 usage"""
        # Remove FTS5 special characters but preserve basic word characters
        # Keep unicode word characters, numbers, hyphens, apostrophes
        clean_term = re.sub(r"[^\w\s\'-]", "", term, flags=re.UNICODE).strip()
        return clean_term

    @staticmethod
    def handle_phrase_query(query: str) -> str:
        """Handle queries that contain quoted phrases"""
        # Split on quotes to separate phrases from individual terms
        parts = []
        in_quote = False
        current_part = ""

        i = 0
        while i < len(query):
            char = query[i]
            if char == '"':
                if in_quote:
                    # End of phrase
                    if current_part.strip():
                        clean_phrase = FTSQuerySanitization.clean_term(current_part)
                        if clean_phrase:
                            parts.append(f'"{clean_phrase}"')
                    current_part = ""
                    in_quote = False
                else:
                    # Start of phrase - first process any pending non-quoted content
                    if current_part.strip():
                        # Split into terms and add as AND
                        parts.extend(FTSQuerySanitization.quote_terms(current_part))
                    current_part = ""
                    in_quote = True
            else:
                current_part += char
            i += 1

        # Handle any remaining content
        if current_part.strip():
            if in_quote:
                # Unclosed quote - treat as phrase anyway
                clean_phrase = FTSQuerySanitization.clean_term(current_part)
                if clean_phrase:
                    parts.append(f'"{clean_phrase}"')
            else:
                # Regular terms
                parts.extend(FTSQuerySanitization.quote_terms(current_part))

        return " AND ".join(parts) if parts else ""

    @staticmethod
    def handle_boolean_query(query: str) -> str:
        """Handle queries with explicit uppercase AND/OR/NOT operators.

        The query is split on the operators *without* changing its case. Each operand
        keeps its own words as separate terms rather than being glued into a phrase,
        and a multi-word operand is parenthesized so FTS5's precedence (NOT, then AND,
        then OR) cannot silently regroup it.

        ``vitamin D AND calcium supplementation`` becomes
        ``("vitamin" AND "D") AND ("calcium" AND "supplementation")`` --- not the phrase
        pair ``"vitamin D" AND "calcium supplementation"``.
        """
        # Odd indices are the captured operators; even indices are the operands.
        parts = _FTS_BOOLEAN_SPLIT.split(query.strip())

        tokens: List[str] = []
        for index, part in enumerate(parts):
            if index % 2 == 1:
                tokens.append(part)
                continue
            terms = FTSQuerySanitization.quote_terms(part)
            if not terms:
                # An operand cleaned away to nothing, so the boolean structure is broken.
                # Drop the operators and fall back to ranking the surviving terms.
                return " OR ".join(FTSQuerySanitization.quote_terms(" ".join(parts[::2])))
            tokens.append(terms[0] if len(terms) == 1 else "(" + " AND ".join(terms) + ")")

        if FTSQuerySanitization.is_valid_boolean_structure(tokens):
            return " ".join(tokens)
        return " OR ".join(FTSQuerySanitization.quote_terms(" ".join(parts[::2])))

    @staticmethod
    def is_valid_boolean_structure(tokens: List[str]) -> bool:
        """Check if boolean query structure is valid"""
        if not tokens:
            return False

        # Should start and end with terms, not operators
        if tokens[0] in ["AND", "OR", "NOT"] or tokens[-1] in ["AND", "OR"]:
            return False

        # Operators and terms should alternate (roughly)
        operator_count = sum(1 for token in tokens if token in ["AND", "OR", "NOT"])
        term_count = len(tokens) - operator_count

        # Should have roughly one fewer operator than terms
        return operator_count <= term_count
