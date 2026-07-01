"""Regression tests for metadata type validation (`_validate_metadata_batch`).

Covers the falsy-value validation bypass fixed in ``database/_metadata.py``: prior
to the fix, a ``if value and ...`` guard meant falsy-but-wrong-type values (``0``,
``0.0``, ``""``, ``False``) skipped type validation entirely, so e.g. a TEXT field
would silently accept the integer ``0``.
"""

import pytest

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB


@pytest.fixture(scope="function")
def typed_db(tmp_path):
    """Real LocalVectorDB with one field of each scalar type."""
    db = LocalVectorDB(
        name="meta_validation",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="test-model",
        embedding_config={"dimension": 32},
        metadata_schema={
            "title": MetadataField(type=MetadataFieldType.TEXT),
            "count": MetadataField(type=MetadataFieldType.INTEGER),
            "score": MetadataField(type=MetadataFieldType.REAL),
            "active": MetadataField(type=MetadataFieldType.BOOLEAN),
            "tags": MetadataField(type=MetadataFieldType.JSON),
        },
        chunk_size=100,
        chunk_overlap=0,
    )
    yield db
    db.close()


@pytest.mark.database
class TestFalsyValueValidation:
    """Falsy values must still be type-checked (the fixed bug)."""

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("title", 0),  # int for TEXT
            ("title", 0.0),  # float for TEXT
            ("title", False),  # bool for TEXT
            ("count", ""),  # empty str for INTEGER
            ("count", []),  # empty list for INTEGER
            ("score", ""),  # empty str for REAL
            ("tags", 0),  # int for JSON
            ("tags", ""),  # empty str for JSON
        ],
    )
    def test_falsy_wrong_type_is_rejected(self, typed_db, field, bad_value):
        with pytest.raises(ValueError, match=f"Metadata field '{field}'"):
            typed_db._validate_metadata_batch([{field: bad_value}])

    @pytest.mark.parametrize(
        "field, good_value",
        [
            ("count", 0),  # int is valid INTEGER
            ("score", 0.0),  # float is valid REAL
            ("score", 0),  # int is accepted for REAL
            ("active", False),  # bool is valid BOOLEAN
            ("active", 0),  # int is accepted for BOOLEAN
            ("title", ""),  # empty str is a valid TEXT
            ("tags", []),  # empty list is valid JSON
            ("tags", {}),  # empty dict is valid JSON
        ],
    )
    def test_valid_falsy_values_pass(self, typed_db, field, good_value):
        # Should not raise.
        typed_db._validate_metadata_batch([{field: good_value}])

    def test_none_is_allowed(self, typed_db):
        # Explicit None stays nullable and skips type validation.
        typed_db._validate_metadata_batch([{"title": None, "count": None}])

    def test_bypass_is_caught_through_upsert(self, typed_db):
        """End-to-end: an out-of-type falsy value is rejected at upsert time."""
        with pytest.raises(ValueError, match="Metadata field 'title'"):
            typed_db.upsert(documents=["hello world"], metadata=[{"title": 0}])

    def test_correct_types_upsert_cleanly(self, typed_db):
        ids = typed_db.upsert(
            documents=["a valid document"],
            metadata=[{"title": "", "count": 0, "score": 0.0, "active": False, "tags": []}],
        )
        assert len(ids) == 1
