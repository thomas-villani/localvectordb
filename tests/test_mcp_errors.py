"""Error text and folder listings that reach an agent through the MCP tools (issue #79)."""

import asyncio

import pytest

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DatabaseNotFoundError, DocumentNotFoundError

pytest.importorskip("fastmcp")

from localvectordb_server.mcp.config import MCPConfig  # noqa: E402


@pytest.fixture()
def manager(tmp_path):
    import localvectordb_server.mcp.server as srv

    mgr = srv.MCPManager(MCPConfig(mode="read-write", databases_root=str(tmp_path)))
    original = srv.mcp_manager
    srv.mcp_manager = mgr
    asyncio.run(mgr.create_database("notes", embedding_provider="mock", embedding_model="test-model"))
    yield mgr
    srv.mcp_manager = original
    asyncio.run(mgr.cleanup())


@pytest.mark.parametrize(
    "error",
    [DatabaseNotFoundError("Database 'x' not found"), DocumentNotFoundError("Document not found: x", "x")],
)
def test_not_found_errors_stringify_to_their_message_and_stay_key_errors(error):
    assert str(error) == error.args[0]
    assert isinstance(error, KeyError)


@pytest.mark.parametrize("bad_type", ["string", "str", "TEXT"])
def test_invalid_field_type_lists_the_valid_types(bad_type):
    with pytest.raises(ValueError, match="Valid types: text, integer, real, boolean, date, json"):
        MetadataField(type=bad_type)


def test_valid_field_types_still_resolve():
    assert MetadataFieldType("text") is MetadataFieldType.TEXT


@pytest.mark.integration
class TestToolErrors:
    def test_missing_database_error_has_no_path_or_stray_quotes(self, manager, tmp_path):
        from localvectordb_server.mcp.server import get_database_info

        result = asyncio.run(get_database_info("missing"))
        assert result == {"error": "Database 'missing' not found", "error_code": "DATABASE_NOT_FOUND"}
        assert str(tmp_path) not in str(result)

    def test_missing_document_error_has_no_stray_quotes(self, manager):
        from localvectordb_server.mcp.server import update_document

        result = asyncio.run(update_document("notes", "nope", content="x"))
        assert result["error_code"] == "DOCUMENT_NOT_FOUND"
        assert not result["error"].startswith(("'", '"'))


@pytest.mark.integration
class TestListPrefixesFolderName:
    @staticmethod
    def _seed(ids):
        from localvectordb_server.mcp.server import upsert_documents

        asyncio.run(upsert_documents("notes", documents=[f"Doc {i}." for i in ids], ids=ids))

    @staticmethod
    def _list(prefix):
        from localvectordb_server.mcp.server import list_prefixes

        return asyncio.run(list_prefixes("notes", prefix=prefix))

    def test_folder_name_without_trailing_delimiter_lists_its_children(self, manager):
        self._seed(["people/ann", "people/bob", "people/team/cy"])
        assert self._list("people") == self._list("people/")
        listing = self._list("people")
        assert listing["prefix"] == "people/"
        assert [entry["name"] for entry in listing["documents"]] == ["ann", "bob"]
        assert [entry["path"] for entry in listing["prefixes"]] == ["people/team/"]

    def test_partial_prefix_matching_several_names_is_not_rewritten(self, manager):
        self._seed(["people/ann", "people2/bob"])
        listing = self._list("people")
        assert listing["prefix"] == "people"
        assert {entry["path"] for entry in listing["prefixes"]} == {"people/", "people2/"}

    def test_partial_prefix_of_one_folder_still_rolls_up_to_it(self, manager):
        self._seed(["people/ann"])
        listing = self._list("peo")
        assert [entry["path"] for entry in listing["prefixes"]] == ["people/"]
