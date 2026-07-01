"""Tests for the shared file-ingest helper and the ``lvdb chunk`` command.

Covers ``load_file_for_ingest`` (extractor routing / plaintext / binary
fallback) and the standalone ``lvdb chunk`` JSONL command. Neither needs a
database, so these avoid the CLI/db mocking used elsewhere.
"""

import json

import pytest
from click.testing import CliRunner

from localvectordb_server.cli import cli
from localvectordb_server.cli._utils import load_file_for_ingest


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# load_file_for_ingest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadFileForIngest:
    def test_plaintext_read_verbatim(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Document content here", encoding="utf-8")
        result = load_file_for_ingest(str(f))
        assert result.error is None
        assert result.text == "Document content here"
        # Plain text is not extracted, so no source_format is attached.
        assert "source_format" not in result.metadata
        assert result.metadata["filename"] == "doc.txt"

    def test_html_extracted_to_markdown(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_bytes(b"<html><body><h1>Hi</h1><p>World</p></body></html>")
        result = load_file_for_ingest(str(f))
        assert result.error is None
        assert result.text == "# Hi\n\nWorld"
        assert result.metadata["source_format"] == "html"
        assert "extraction_method" in result.metadata

    def test_csv_extracted_to_markdown_table(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n1,2\n")
        result = load_file_for_ingest(str(f))
        assert result.error is None
        assert "| a | b |" in result.text
        assert result.metadata["source_format"] == "csv"

    def test_binary_without_extractor_reports_error(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        result = load_file_for_ingest(str(f))
        assert result.text is None
        assert result.error  # a human-readable reason

    def test_force_extract_on_text(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Document content here", encoding="utf-8")
        result = load_file_for_ingest(str(f), force_extract=True)
        assert result.error is None
        # Forced through the extractor -> plaintext format recorded.
        assert result.metadata["source_format"] == "plaintext"


# ---------------------------------------------------------------------------
# lvdb chunk
# ---------------------------------------------------------------------------


def _jsonl(text):
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]


@pytest.mark.unit
class TestChunkCommand:
    def test_chunk_direct_text(self, runner):
        result = runner.invoke(cli, ["chunk", "One sentence here. Second sentence now.", "-M", "sentences", "-s", "5"])
        assert result.exit_code == 0
        records = _jsonl(result.stdout)
        assert len(records) >= 1
        first = records[0]
        assert set(first) >= {"content", "index", "tokens", "position"}
        assert set(first["position"]) == {"start", "end", "line", "column", "end_line", "end_column"}
        # Single input: no "source" key.
        assert "source" not in first

    def test_chunk_stdin(self, runner):
        result = runner.invoke(cli, ["chunk", "-", "-M", "words", "-s", "3"], input="alpha beta gamma delta epsilon")
        assert result.exit_code == 0
        assert len(_jsonl(result.stdout)) >= 1

    def test_chunk_file_and_multiple_sources(self, runner, tmp_path):
        a = tmp_path / "a.txt"
        a.write_text("Plain content one.", encoding="utf-8")
        b = tmp_path / "b.txt"
        b.write_text("Plain content two.", encoding="utf-8")
        result = runner.invoke(cli, ["chunk", str(a), str(b), "-M", "lines"])
        assert result.exit_code == 0
        records = _jsonl(result.stdout)
        # Multiple inputs -> every record carries its source path.
        assert records and all("source" in r for r in records)

    def test_chunk_html_file_is_extracted(self, runner, tmp_path):
        f = tmp_path / "page.html"
        f.write_bytes(b"<html><body><h1>Heading</h1><p>Body text.</p></body></html>")
        result = runner.invoke(cli, ["chunk", str(f), "-M", "lines"])
        assert result.exit_code == 0
        records = _jsonl(result.stdout)
        joined = "".join(r["content"] for r in records)
        assert "# Heading" in joined  # markdown, not raw HTML

    def test_chunk_output_file(self, runner, tmp_path):
        out = tmp_path / "out.jsonl"
        result = runner.invoke(cli, ["chunk", "some text to chunk", "-O", str(out)])
        assert result.exit_code == 0
        assert len(_jsonl(out.read_text(encoding="utf-8"))) >= 1

    def test_chunk_invalid_method(self, runner):
        result = runner.invoke(cli, ["chunk", "x", "-M", "bogus"])
        assert result.exit_code != 0

    def test_chunk_no_args(self, runner):
        result = runner.invoke(cli, ["chunk"])
        assert result.exit_code != 0
