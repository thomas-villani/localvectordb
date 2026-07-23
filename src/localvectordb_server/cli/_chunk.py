"""``lvdb chunk`` — run the chunkers standalone and emit JSONL.

This command exposes LocalVectorDB's position-aware chunkers without any
database, embedding, or config. It reads text from files, globs, stdin, or
direct arguments, chunks each input, and writes one JSON object per chunk to
stdout (or ``--output``). Rich file formats (PDF, DOCX, HTML, ...) are extracted
to Markdown first, exactly as ``lvdb db <name> add`` does.
"""

import glob
import json
import os
import sys

import click

from localvectordb.chunking import ChunkerFactory
from localvectordb_server.cli._utils import (
    EXIT_CODE_ERROR,
    get_stdin_input,
    load_file_for_ingest,
)


@click.command("chunk")
@click.argument("files_or_text", nargs=-1)
@click.option(
    "--method",
    "-M",
    type=click.Choice(ChunkerFactory.list_methods()),
    default="sentences",
    show_default=True,
    help="Chunking strategy to use.",
)
@click.option(
    "--max-tokens",
    "--chunk-size",
    "-s",
    "max_tokens",
    type=int,
    default=500,
    show_default=True,
    help="Maximum tokens per chunk.",
)
@click.option(
    "--overlap",
    type=int,
    default=0,
    show_default=True,
    help="Token overlap between consecutive chunks (ignored by some strategies).",
)
@click.option(
    "--delimiter",
    default=None,
    type=str,
    help="Delimiter for --method delimiter (default: a blank line). Escapes "
    r"\n, \t, \r are interpreted; e.g. --delimiter '\n---\n'.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write JSONL to this file instead of stdout.",
)
@click.option(
    "--extract/--no-extract",
    default=None,
    help="Force (or disable) text extraction for file inputs. Default: auto.",
)
def chunk_command(files_or_text, method, max_tokens, overlap, delimiter, output, extract):
    """
    Chunk text and emit JSONL — one JSON object per chunk, no embedding.

    Accepts files, globs, direct text, or ``-`` for stdin. Each output record has
    ``content``, ``index`` (within its source), ``tokens``, ``position`` (start,
    end, line, column, end_line, end_column) and, for multi-input runs, ``source``.

    \b
    Examples:
        \b
        lvdb chunk notes.md
        lvdb chunk report.pdf --method sentences --max-tokens 300
        lvdb chunk "docs/*.md" --method paragraphs -o chunks.jsonl
        echo "some long text..." | lvdb chunk - --method words --overlap 20
    """
    if len(files_or_text) == 0:
        click.secho(
            "Error: FILES_OR_TEXT is required. Provide file path(s), a glob, text, " "or '-' to read from stdin.",
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    force_extract = bool(extract)

    # Collect (source_label, text) pairs. source_label is None for the single
    # stdin/text case so we can omit the noisy "source" key when it adds nothing.
    inputs: list[tuple[str | None, str]] = []

    if len(files_or_text) == 1 and files_or_text[0] == "-":
        inputs.append((None, get_stdin_input(True, "No input provided to stdin")))
    else:
        for item in files_or_text:
            item = item.strip("'").strip('"')
            if os.path.isfile(item):
                click.secho(f"Reading {item}...", fg="blue", err=True)
                result = load_file_for_ingest(item, force_extract=force_extract)
                if result.text is None:
                    click.secho(f"Skipping `{item}`: {result.error}", fg="bright_red", err=True)
                    continue
                inputs.append((item, result.text))
            elif os.path.isdir(os.path.dirname(item) or ".") and any(c in os.path.basename(item) for c in "*?[]"):
                for file in glob.glob(item, recursive=True):
                    if not os.path.isfile(file):
                        continue
                    click.secho(f"Reading {file}...", fg="blue", err=True)
                    result = load_file_for_ingest(file, force_extract=force_extract)
                    if result.text is None:
                        click.secho(f"Skipping `{file}`: {result.error}", fg="bright_red", err=True)
                        continue
                    inputs.append((file, result.text))
            else:
                # Treat as literal text. Label it only when there are several inputs.
                inputs.append(("<text>", item))

    if not inputs:
        click.secho("Error: no readable input to chunk.", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    # `delimiter` is only meaningful for the delimiter strategy; forward it there
    # (other chunkers do not accept it). Interpret common escapes so a newline
    # delimiter survives a shell that does not, e.g. --delimiter '\n---\n'.
    chunker_kwargs = {}
    if method == "delimiter" and delimiter is not None:
        chunker_kwargs["delimiter"] = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    try:
        chunker = ChunkerFactory.create_chunker(method, max_tokens=max_tokens, overlap=overlap, **chunker_kwargs)
    except (ValueError, TypeError) as e:
        click.secho(f"Error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    multiple = len(inputs) > 1
    out = open(output, "w", encoding="utf-8") if output else sys.stdout
    total = 0
    try:
        for source, text in inputs:
            chunks = chunker.chunk(text)
            for c in chunks:
                record = {
                    "content": c.content,
                    "index": c.index,
                    "tokens": c.tokens,
                    "position": c.position.to_dict(),
                }
                if multiple and source is not None:
                    record["source"] = source
                out.write(json.dumps(record, ensure_ascii=False))
                out.write("\n")
                total += 1
    finally:
        if output:
            out.close()

    click.secho(
        f"Wrote {total} chunk(s) from {len(inputs)} input(s)" + (f" to {output}" if output else ""),
        fg="green",
        err=True,
    )


__all__ = ["chunk_command"]
