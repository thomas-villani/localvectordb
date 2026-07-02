"""Tests for sub-document retrieval on ``lvdb db <name> get``.

Covers the pure ``parse_range_spec`` helper (unit) and the ``--chunk`` /
``--range`` / ``--lines`` / ``--section`` / ``--outline`` selection flags driven
end-to-end against a real ``LocalVectorDB`` (integration).
"""

import json
import tempfile
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from localvectordb.database import LocalVectorDB
from localvectordb_server.cli import cli
from localvectordb_server.cli._utils import parse_range_spec

MARKDOWN_DOC = (
    "# Introduction\n"
    "alpha beta gamma delta epsilon zeta eta theta.\n\n"
    "## Installation\n"
    "install one two three four five six seven eight.\n\n"
    "## Usage\n"
    "usage details nine ten eleven twelve thirteen fourteen.\n"
)


# ---------------------------------------------------------------------------
# parse_range_spec (pure unit)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseRangeSpec:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("2:5", (2, 5)),
            ("2:", (2, None)),
            (":5", (None, 5)),
            (":", (None, None)),
            ("  3 : 7 ", (3, 7)),
        ],
    )
    def test_ranges(self, spec, expected):
        assert parse_range_spec(spec) == expected

    def test_single_allowed(self):
        assert parse_range_spec("3", allow_single=True) == (3, 3)

    def test_single_rejected_by_default(self):
        with pytest.raises(ValueError):
            parse_range_spec("3")

    @pytest.mark.parametrize("spec", ["a:b", "", "   ", "1:2:3", "x", ":y"])
    def test_malformed(self, spec):
        with pytest.raises(ValueError):
            parse_range_spec(spec, allow_single=True)


# ---------------------------------------------------------------------------
# Integration: real DB + CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def get_db():
    """A real DB holding one multi-section Markdown document, tiny chunks."""
    tmp = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="getparts",
        base_path=tmp,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_config={"dimension": 32},
        chunk_size=5,  # small so the doc spans several chunks
        chunk_overlap=0,
    )
    ids = db.upsert(documents=[MARKDOWN_DOC], metadata=[{}])
    yield db, ids[0]
    db.close()


def _patches(real_db):
    """Inject ``real_db`` as ctx.obj['db'], bypassing config/db-folder lookup."""

    @click.pass_context
    def _cli_cb(ctx, *args, **kwargs):
        ctx.ensure_object(dict)
        ctx.obj = {"db": real_db, "db_folder": "unused"}

    @click.pass_context
    def _db_cb(ctx, name):
        ctx.obj.update({"db_name": name, "db": real_db})

    from localvectordb_server.cli._db import db_group

    return patch.object(cli, "callback", _cli_cb), patch.object(db_group, "callback", _db_cb)


@pytest.mark.integration
class TestGetParts:
    def _invoke(self, runner, real_db, args):
        p1, p2 = _patches(real_db)
        with p1, p2:
            return runner.invoke(cli, ["db", "getparts", "get", *args])

    def test_whole_document(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id])
        assert result.exit_code == 0
        assert "# Introduction" in result.output

    def test_char_range(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--range", "0:14"])
        assert result.exit_code == 0
        assert result.output.strip() == "# Introduction"

    def test_line_range(self, runner, get_db):
        db, doc_id = get_db
        # Lines are 1-based inclusive; line 1 is the top heading.
        result = self._invoke(runner, db, [doc_id, "--lines", "1:1"])
        assert result.exit_code == 0
        assert result.output.strip() == "# Introduction"

    def test_section_by_name(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--section", "Installation"])
        assert result.exit_code == 0
        assert result.output.startswith("## Installation")
        assert "install one two three" in result.output

    def test_section_case_insensitive(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--section", "installation"])
        assert result.exit_code == 0
        assert "## Installation" in result.output

    def test_section_not_found_lists_available(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--section", "Nope"])
        assert result.exit_code != 0
        assert "Installation" in result.output  # available headings listed

    def test_outline_text(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--outline"])
        assert result.exit_code == 0
        for heading in ("Introduction", "Installation", "Usage"):
            assert heading in result.output

    def test_outline_json(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--outline", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        headings = [item["heading"] for item in payload["outline"]]
        assert "Introduction" in headings
        assert "Installation" in headings

    def test_chunk_single(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--chunk", "0"])
        assert result.exit_code == 0
        assert result.output.strip()  # first chunk has content

    def test_chunk_json_has_index_and_position(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--chunk", "0", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["chunks"][0]["index"] == 0
        assert "start" in payload["chunks"][0]["position"]

    def test_chunk_out_of_range(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--chunk", "9999"])
        assert result.exit_code != 0

    def test_mutually_exclusive_flags(self, runner, get_db):
        db, doc_id = get_db
        result = self._invoke(runner, db, [doc_id, "--range", "0:5", "--lines", "1:1"])
        assert result.exit_code != 0
        assert "--range" in result.output and "--lines" in result.output
