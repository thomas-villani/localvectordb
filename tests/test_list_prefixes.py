"""Tests for S3-style document-id prefix listing (``LocalVectorDB.list_prefixes``)."""

import pytest

from localvectordb import LocalVectorDB, PrefixListing

pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_path):
    database = LocalVectorDB(
        name="prefix_test",
        base_path=str(tmp_path / "db"),
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    ids = [
        "docs/reports/q1",
        "docs/reports/q2",
        "docs/guide/intro",
        "docs/guide/adv",
        "docs/readme",
        "notes/todo",
        "docs",  # a leaf whose id equals a folder prefix
    ]
    database.upsert([f"content-{i}" for i in range(len(ids))], ids=ids)
    return database


def test_top_level_lists_folders_and_leaves(db):
    listing = db.list_prefixes("")
    assert isinstance(listing, PrefixListing)

    folders = {e.path: e.count for e in listing.prefixes}
    assert folders == {"docs/": 5, "notes/": 1}
    assert all(e.is_prefix for e in listing.prefixes)

    docs = {e.path for e in listing.documents}
    assert docs == {"docs"}
    assert all(not e.is_prefix and e.count == 1 for e in listing.documents)


def test_descend_into_folder(db):
    listing = db.list_prefixes("docs/")
    folders = {e.path: e.count for e in listing.prefixes}
    assert folders == {"docs/reports/": 2, "docs/guide/": 2}
    assert {e.path for e in listing.documents} == {"docs/readme"}


def test_leaf_level(db):
    listing = db.list_prefixes("docs/guide/")
    assert listing.prefixes == []
    assert {e.path for e in listing.documents} == {"docs/guide/intro", "docs/guide/adv"}


def test_names_are_relative_to_prefix(db):
    listing = db.list_prefixes("docs/")
    names = {e.name for e in listing.prefixes} | {e.name for e in listing.documents}
    assert names == {"reports/", "guide/", "readme"}


def test_document_equal_to_prefix_is_not_its_own_child(db):
    # "docs" exists as a document, but listing "docs/" must not surface it.
    listing = db.list_prefixes("docs/")
    assert "docs" not in {e.path for e in listing.documents}


def test_unknown_prefix_is_empty(db):
    listing = db.list_prefixes("nope/")
    assert listing.prefixes == []
    assert listing.documents == []


def test_custom_delimiter(tmp_path):
    database = LocalVectorDB(
        name="delim_test",
        base_path=str(tmp_path / "db"),
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    database.upsert(["a", "b", "c"], ids=["a::x::1", "a::x::2", "a::y"])
    listing = database.list_prefixes("a::", delimiter="::")
    assert {e.path: e.count for e in listing.prefixes} == {"a::x::": 2}
    assert {e.path for e in listing.documents} == {"a::y"}


def test_empty_delimiter_rejected(db):
    with pytest.raises(ValueError):
        db.list_prefixes("docs/", delimiter="")


def test_glob_metacharacters_in_ids_are_literal(tmp_path):
    # Ids containing GLOB metacharacters must be matched literally, not as patterns.
    database = LocalVectorDB(
        name="glob_test",
        base_path=str(tmp_path / "db"),
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    database.upsert(["a", "b"], ids=["a[x]/one", "ay/two"])
    listing = database.list_prefixes("a[x]/")
    # Only the literal "a[x]/" child, not "ay/two" (which a raw glob class would match).
    assert {e.path for e in listing.documents} == {"a[x]/one"}


def test_to_dict_roundtrip(db):
    listing = db.list_prefixes("docs/")
    restored = PrefixListing.from_dict(listing.to_dict())
    assert restored.prefix == listing.prefix
    assert {e.path for e in restored.prefixes} == {e.path for e in listing.prefixes}
    assert {e.path for e in restored.documents} == {e.path for e in listing.documents}
