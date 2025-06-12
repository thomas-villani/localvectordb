# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/cli/_db.py
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import click

from localvectordb_server.cli._utils import find_config_file, EXIT_CODE_ERROR, print_db_stats, get_stdin_input


@click.group('db')
@click.argument("name")
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
@click.pass_context
def db_group(ctx, name, config, db_folder):
    """
    Commands related to a specific database NAME.

    Provides subcommands for interacting with a specific database, such as adding, searching,
    updating, and deleting documents.

    \b
    Examples:
        \b
        lvdb db mydb info
        lvdb db mydb add document.txt
        lvdb db mydb search "query text"
        lvdb db mydb shell

    """
    if not db_folder:
        config_path = find_config_file(config)
        from localvectordb_server.config import load_config
        cfg = load_config(config_path)
        db_folder = cfg.database.root_dir

    if not db_folder or not os.path.exists(db_folder):
        click.secho(
            f"DB_FOLDER {'not specified and not found in configuration' if not db_folder else 'does not exist'}.",
            fg="bright_red", err=True
            )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    from localvectordb.exceptions import DatabaseNotFoundError
    try:
        from localvectordb.database import LocalVectorDB
        db = LocalVectorDB(name=name, base_path=db_folder, create_if_not_exists=False)
    except DatabaseNotFoundError as e:
        click.secho(f"Database '{name}' was not found in {os.path.abspath(db_folder)}!",
                    fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    ctx.obj = {"db_name": name, "db_folder": db_folder, "db": db}


@db_group.command('info')
@click.pass_context
def show_db_info(ctx):
    """
    Show the configuration info for a database

    Displays configuration and statistics for the specified database, including embedding model,
    provider, chunking, and metadata schema.

    \b
    Example:
        \b
        lvdb db mydb info
    """
    db = ctx.obj["db"]

    try:
        stats = db.get_stats()
        click.echo("Database Info\n"
                   "-------------")
        click.echo(f"  Database: {db.name}")
        click.echo(f"  Path: {os.path.abspath(ctx.obj['db_folder'])}")
        click.echo(f"  Embedding model: {stats['embedding_model']}")
        click.echo(f"  Embedding provider: {stats['embedding_provider']}")
        click.echo(f"  Chunk size: {stats['chunk_size']}")
        click.echo(f"  Chunking method: {stats['chunking_method']}")
        click.echo(f"  Chunk overlap: {stats['chunk_overlap']}")
        click.echo(f"  FTS search available: {stats['fts_enabled']}")
        click.echo(f"  Total Documents: {stats['documents']}")
        click.echo(f"  Total Chunks: {stats['chunks']}")

        # Show metadata schema if available
        if hasattr(db, 'metadata_schema') and db.metadata_schema:
            click.echo(f"  Metadata fields: {len(db.metadata_schema)}")
            for field_name in db.metadata_schema:
                click.echo(f"    - {field_name} {db.metadata_schema[field_name].type.upper()}")

    except Exception as e:
        click.secho(f"Error reading database info: {str(repr(e))}", fg='bright_red', err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@db_group.command('stats')
@click.pass_context
def show_db_stats(ctx):
    """
    Show database statistics

    Displays detailed statistics for the database, such as document and chunk counts, embedding
    model, and configuration.

    \b
    Example:
        \b
        lvdb db mydb stats
    """
    db = ctx.obj["db"]
    print_db_stats(db)


@db_group.command('list')
@click.option('--limit', '-n', type=int, default=None, help="Limit number of ids returned")
@click.option('--offset', '-s', type=int, default=0, help="Offset of ids returned")
@click.option('--output', '-o', type=click.Path(exists=False, file_okay=True), default=None, help="Output to file")
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False, help="Output in json format")
@click.pass_context
def list_document_ids(ctx, limit, offset, output, output_as_json):
    """
    List document IDs in database

    Lists the IDs of documents stored in the database. Supports pagination, output to file, and
    JSON formatting.

    \b
    Examples:
        \b
        lvdb db mydb list
        lvdb db mydb list --limit 10 --offset 20 --json
    """
    db = ctx.obj["db"]

    # Get all documents and apply pagination
    all_docs = db.filter(limit=limit, offset=offset)
    ids = [doc.id for doc in all_docs]

    if output_as_json:
        output_str = json.dumps(ids)
    else:
        output_str = '\n'.join(ids)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(output_str)
        click.secho(f"Results written to `{output}`", fg="blue", err=True)
    else:
        click.secho(f"Document IDs in {db.name}", fg="cyan", err=True)
        click.echo(output_str)


@db_group.command('search')
@click.argument('query')
@click.option('--limit', '-k', '-n', default=5, help='Maximum number of results')
@click.option('--search-type', '-t', default='vector',
              type=click.Choice(['vector', 'keyword', 'hybrid']),
              help='Type of search to perform')
@click.option('--return-type', '-r', default='documents',
              type=click.Choice(['documents', 'chunks']),
              help='Whether to return documents or chunks')
@click.option('--score-threshold', default=0.0, type=float, help='Minimum score threshold')
@click.option('--vector-weight', default=0.7, type=float, help='Weight for vector search in hybrid mode')
@click.option('--metadata-filter', help='Metadata filter in JSON format')
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False)
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False), help='Output file for results')
@click.option('--metadata/--no-metadata', '-m', default=False, help='Include metadata in output')
@click.option('--pretty', '-p', default=False, is_flag=True)
@click.pass_context
def search(
        ctx, query, limit, search_type, return_type, score_threshold, vector_weight,
        metadata_filter, output_as_json, output, metadata, pretty
        ):
    """
    Search a vector database using the unified query interface.

    Performs a search on the database using vector, keyword, or hybrid methods. Supports metadata
    filtering, result formatting, and output to file.

    \b
    Examples:
    \b
        lvdb db mydb search "search text" --limit 5 --search-type hybrid
        lvdb db mydb search "search text" --metadata-filter '{"author":"Smith"}' --json

    """
    # Parse metadata filter if provided
    filter_dict = None
    if metadata_filter:
        try:
            filter_dict = json.loads(metadata_filter)
        except json.JSONDecodeError:
            click.secho("Error: Metadata filter must be valid JSON", fg='red', err=True)
            raise click.Abort()

    db = ctx.obj["db"]

    # Read from stdin
    if query == "-":
        query = get_stdin_input(True, "Error: No query provided!")

    click.secho(f"Performing {search_type} search for `{query[:100]}`...", fg="blue", err=True)

    try:
        results = db.query(
            query=query,
            search_type=search_type,
            return_type=return_type,
            k=limit,
            score_threshold=score_threshold,
            filters=filter_dict,
            vector_weight=vector_weight
        )
    except Exception as e:
        click.secho(f"Search error: {str(e)}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    if not results:
        click.secho("No results found.", fg="red", err=True)
        return

    # Format and display results
    if not output_as_json:
        output_str = ""
        if pretty:
            if len(query) > 100:
                query = query[:100] + "..."
            query = query.strip().replace("\n", " \\ ")
            title = f"{search_type.title()} Search Results for `{query}`: {len(results)} Results"
            header = title + "\n" + ("=" * len(title)) + "\n"
            if not output:
                header = click.style(header, fg="magenta")
            output_str += header

        for i, result in enumerate(results, 1):
            if pretty:
                doc_header = f"\n{i}. Document: {result.id} (Score: {result.score:.4f})\n"
                doc_header += ("-" * 40) + "\n"
                if not output:
                    doc_header = click.style(doc_header, fg="cyan")
                output_str += doc_header

                if not output:
                    output_str += click.style(result.content, fg="bright_white") + "\n"
                else:
                    output_str += result.content + "\n"
            else:
                output_str += f"Document: {result.id}\n"
                output_str += result.content + "\n"

            if metadata:
                json_str = json.dumps(result.metadata, indent=2 if pretty else None)
                if pretty and not output:
                    output_str += click.style("\n~~~~~\n\n", fg="yellow")
                    output_str += click.style("Metadata: ", fg="yellow")
                    json_str = click.style(json_str, fg="yellow")
                else:
                    output_str += "\n~~~~~\n\n"
                    output_str += "Metadata: "
                output_str += json_str + "\n"

            if i < len(results):
                output_str += click.style(f"\n{'-' * 40}\n\n", fg="cyan") if (pretty and not output) else "\n-----\n\n"
    else:
        result_data = [{
            'id': result.id,
            'type': result.type,
            'content': result.content,
            'score': result.score,
            'metadata': result.metadata
        } for result in results]

        if not metadata:
            for d in result_data:
                d.pop("metadata", None)
        output_str = json.dumps(result_data, indent=2 if pretty else None)

    if output:
        with open(output, 'w') as f:
            f.write(output_str)
        click.echo(f"Results saved to {output}", err=True)
    else:
        click.echo(output_str)


@db_group.command('add')
@click.argument('files_or_text', nargs=-1)
@click.option('--metadata', '-m', default=None,
              help='Metadata for the document in JSON format or path to .json file.')
@click.option('--id', '-i', default=None, help='Set the id(s) for the document, separated by ",".')
@click.pass_context
def add_to_database(ctx, files_or_text, metadata, id):
    """
    Add document(s) to the database.

    Adds one or more documents to the database from files, globs, stdin, or direct text. Supports
    attaching metadata and specifying document IDs.

    \b
    Examples:
        \b
        lvdb db mydb add file.txt
        lvdb db mydb add "docs/*.md"
        cat file.txt | lvdb db mydb add -
        lvdb db mydb add file1.txt file2.txt --metadata '[{"author":"A"},{"author":"B"}]' --id "id1,id2"
    """
    db = ctx.obj['db']

    all_inputs = []
    auto_metadata = []

    if len(files_or_text) == 0:
        click.secho(
            f"Error: FILES_OR_TEXT is required. Must be file path, glob, str to add, or '-' "
            "to read from stdin\n"
            "Usage:\n"
            "   $ lvdb db <DB_NAME> add path/to/the/file.txt [OPTIONS]\n"
            "   $ lvdb db <DB_NAME> add path/to/the/*.glob [OPTIONS]\n"
            "   $ echo 'text to add' | lvdb db <DB_NAME> add - [OPTIONS]",
            fg='bright_red', err=True
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    if len(files_or_text) == 1 and files_or_text[0] == '-':
        input_data = get_stdin_input(True, "No input provided to stdin")
        all_inputs.append(input_data)
        auto_metadata.append({"source": "stdin"})
    else:
        for file_or_text_input in files_or_text:
            file_or_text_input = file_or_text_input.strip("'").strip('"')

            if os.path.isfile(file_or_text_input):
                click.secho(f"Reading {file_or_text_input}...", fg="blue", err=True)
                with open(file_or_text_input, "r", encoding="utf-8") as f:
                    data = f.read()
                all_inputs.append(data)
                auto_metadata.append({
                    "filename": os.path.basename(file_or_text_input),
                    "path": os.path.abspath(file_or_text_input),
                    "ext": os.path.splitext(file_or_text_input)[1],
                    "bytes": len(data.encode("utf-8"))
                })
            elif os.path.isdir(os.path.dirname(file_or_text_input)):
                glob_pattern = os.path.basename(file_or_text_input)
                if any(c in glob_pattern for c in '*?[]'):
                    matching_files = glob.glob(file_or_text_input, recursive=True)
                    for file in matching_files:
                        click.echo(f"Reading {file}...", err=True)
                        try:
                            with open(file, "r", encoding="utf-8") as f:
                                data = f.read()
                        except UnicodeDecodeError:
                            click.secho(f"Unicode Decoding error, file `{file}` is probably binary, skipping!",
                                        fg="bright_red", err=True)
                            continue
                        all_inputs.append(data)
                        auto_metadata.append({
                            "filename": os.path.basename(file),
                            "path": os.path.abspath(file),
                            "ext": os.path.splitext(file)[1],
                            "bytes": len(data.encode("utf-8"))
                        })
                else:
                    click.secho(f"Error: invalid pattern: {file_or_text_input}", fg="bright_red", err=True)
            else:
                all_inputs.append(file_or_text_input)
                auto_metadata.append({"source": "cli"})

    # Handle metadata
    if metadata:
        # if metadata == "auto":
        #     metadata = auto_metadata
        if os.path.isfile(metadata):
            with open(metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                click.secho("Error: if `--metadata` is provided, must be valid JSON", fg='bright_red', err=True)
                raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

        if isinstance(metadata, dict):
            metadata = [metadata]
        if len(metadata) != len(all_inputs):
            click.secho("Error: if providing `--metadata`, length must match number of documents. "
                        f"Found: {len(metadata)}, expected: {len(all_inputs)}.",
                        fg='bright_red', err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

    # Handle IDs
    if id is not None:
        if os.path.isfile(id):
            with open(id, "r", encoding="utf-8") as f:
                data = f.read()
            if id.lower().endswith(".json"):
                id = json.loads(data)
            else:
                id = [line.strip() for line in data.split("\n") if line.strip()]
        else:
            id = [i.strip() for i in id.split(",")]

        if len(id) != len(all_inputs):
            click.secho(
                "Error: if providing `--id`, length must match number of documents. "
                f"Found: {len(id)}, expected: {len(all_inputs)}.",
                fg='bright_red', err=True
            )
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

    try:
        click.secho(f"Adding {len(all_inputs)} document(s)...", fg="blue", err=True)

        new_ids = db.upsert(
            documents=all_inputs,
            metadata=metadata,
            ids=id
        )

        click.echo(f"Successfully added {len(all_inputs)} document(s)!\nCreated ids:", err=True)
        click.echo(','.join(new_ids))

    except Exception as e:
        click.secho(f"Error: Unexpected error while adding documents: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('get')
@click.argument('doc_id')
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False)
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False), help='Output file for results')
@click.option('--metadata/--no-metadata', '-m', default=False, help='Enable/Disable retrieving document metadata')
@click.option('--pretty', '-p', is_flag=True, default=False, help='Output results with title and formatting')
@click.pass_context
def get_document(ctx, doc_id, output_as_json, output, metadata, pretty):
    """
    Retrieve document DOC_ID from database

    Fetches the content and (optionally) metadata of a document by its ID. Supports output as
    JSON, pretty formatting, and writing to a file.

    \b
    Examples:
        \b
        lvdb db mydb get doc_1
        lvdb db mydb get doc_1 --json --metadata
    """
    db = ctx.obj['db']

    try:
        doc = db.get(doc_id)
        if doc is None:
            click.echo(f"Document {doc_id} was not found in '{db.name}'")
            return

        content = doc.content
        meta = doc.metadata

        if output_as_json:
            output_dict = {
                'id': doc_id,
                'content': content
            }
            if metadata:
                output_dict['metadata'] = meta

            output_str = json.dumps(output_dict)
        else:
            output_str = ""
            if pretty:
                title = f"Document: {doc_id}"
                if not output:
                    output_str += click.style(title + "\n", fg="cyan")
                    output_str += click.style("=" * len(title), fg="cyan") + "\n"
                    output_str += click.style(content, fg="bright_white") + "\n"
                else:
                    output_str += title + "\n"
                    output_str += "=" * len(title) + "\n"
                    output_str += content + "\n"
            else:
                output_str += content + "\n"

            if metadata:
                if pretty and not output:
                    output_str += click.style("\n~~~~~\n\n", fg="yellow")
                    output_str += click.style("Metadata: ", fg="cyan")
                else:
                    output_str += "\n~~~~~\n\n"
                    output_str += "Metadata: "
                output_str += json.dumps(meta, indent=2 if pretty else None) + "\n"

        if output:
            with open(output, 'w', encoding="utf-8") as f:
                f.write(output_str)
            click.echo(f"Results saved to {output}", err=True)
        else:
            click.echo(output_str)

    except Exception as e:
        click.secho(f"Error retrieving document: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@db_group.command('update')
@click.argument('doc_id')
@click.argument('file_or_text')
@click.option('--metadata', '-m', default=None, help='Metadata for the document in JSON format')
@click.pass_context
def update_document(ctx, doc_id, file_or_text, metadata):
    """
    Update document DOC_ID with new content and/or metadata

    Updates the content and/or metadata of a document in the database. Content can be provided
    as a file, text, or via stdin.

    \b
    Examples:
        \b
        lvdb db mydb update doc_1 new_content.txt
        echo "new content" | lvdb db mydb update doc_1 -
    """
    db = ctx.obj['db']

    if file_or_text == "-":
        file_or_text = get_stdin_input(True, "Error: No data found in stdin")
    elif os.path.isfile(file_or_text):
        with open(file_or_text, "r", encoding="utf-8") as f:
            file_or_text = f.read()

    # Parse metadata if provided
    metadata_dict = None
    if metadata:
        if os.path.isfile(metadata):
            with open(metadata, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
        else:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError as e:
                click.secho("Error: if `--metadata` is provided, must be valid JSON", fg='bright_red', err=True)
                raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    try:
        updated = db.update(doc_id, content=file_or_text, metadata=metadata_dict)
        if updated:
            click.echo(f"Successfully updated document: {doc_id}")
        else:
            click.echo(f"Document {doc_id} not found")

    except Exception as e:
        click.secho(f"Error: Unexpected error while updating document: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('delete')
@click.argument('doc_id')
@click.pass_context
def delete_document(ctx, doc_id):
    """
    Delete document DOC_ID from database

    Deletes a document from the database by its ID.

    \b
    Example:
        \b
        lvdb db mydb delete doc_1

    """
    db = ctx.obj['db']

    try:
        if not db.exists(doc_id):
            click.echo(f"Document {doc_id} not found")
            return

        deleted_count = db.delete(doc_id)
        if deleted_count > 0:
            click.echo(f"Successfully deleted document: {doc_id}")
        else:
            click.echo(f"No documents were deleted")

    except Exception as e:
        click.secho(f"Error: Unexpected error while deleting document: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('shell')
@click.pass_context
def shell(ctx):
    """
    Start an interactive shell for database operations.

    Launches an interactive shell for performing database operations such as search, add, get,
    delete, list, stats, and more. Type 'help' for available commands.

    """
    import glob

    db = ctx.obj['db']

    try:
        click.echo(click.style(f"Connected to database: ", fg="green")
                   + click.style(db.name, fg="green", underline=True))

        stats = db.get_stats()
        click.secho(f"Documents: {stats['documents']}, Chunks: {stats['chunks']}", fg="blue")
        click.echo(f"Type 'help' for available commands, 'exit' to quit")

        # Simple REPL
        while True:
            try:
                command = click.prompt(f"{db.name}> ", type=str)

                if command.lower() in ('exit', 'quit', 'q'):
                    break

                if command.lower() in ('help', '?'):
                    click.echo("Available commands:")
                    click.echo("  search \"<query>\" [limit] [type] - Search for documents")
                    click.echo("    Types: vector (default), keyword, hybrid")
                    click.echo("  get <id>                       - Get document by ID")
                    click.echo("  add <file or glob>             - Add file(s) to database")
                    click.echo("  delete <id>                    - Delete document by ID")
                    click.echo("  list [limit] [offset]          - List document IDs")
                    click.echo("  count                          - Show document count")
                    click.echo("  stats                          - Show database statistics")
                    click.echo("  info                           - Show database information")
                    click.echo("  clear                          - Clear the console")
                    click.echo("  exit/quit                      - Exit shell")
                    continue

                if command.lower().startswith('search'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: search <query> [limit] [type]", fg="magenta")
                        continue

                    args = parts[1]
                    limit = 5
                    search_type = "vector"

                    # Parse query in quotes
                    if args.count('"') >= 2:
                        start_quote = args.index('"')
                        end_quote = args.index('"', start_quote + 1)
                        query_str = args[start_quote + 1:end_quote]
                        leftover = args[end_quote + 1:].strip()

                        # Parse remaining args
                        remaining_parts = leftover.split()
                        if len(remaining_parts) >= 1 and remaining_parts[0].isdigit():
                            limit = int(remaining_parts[0])
                        if len(remaining_parts) >= 2 and remaining_parts[1] in ['vector', 'keyword', 'hybrid']:
                            search_type = remaining_parts[1]
                    else:
                        query_str = args
                        arg_split = args.rsplit(" ", 2)
                        if len(arg_split) >= 2 and arg_split[-1] in ['vector', 'keyword', 'hybrid']:
                            search_type = arg_split[-1]
                            query_str = " ".join(arg_split[:-1])
                        if len(arg_split) >= 2 and arg_split[-2].isdigit():
                            limit = int(arg_split[-2])
                            query_str = " ".join(arg_split[:-2])

                    click.secho(f"{search_type.title()} search for `{query_str[:100]}`...", fg="blue")

                    try:
                        results = db.query(
                            query=query_str,
                            search_type=search_type,
                            k=limit
                        )

                        click.echo("Results:\n========\n")
                        if not results:
                            click.secho("No results found.", fg="yellow")
                        else:
                            for i, result in enumerate(results, 1):
                                click.echo(f"{i}. {result.id} (Score: {result.score:.4f}):")
                                content_preview = result.content[:200]
                                click.echo(f"   {content_preview}")
                                if len(result.content) > 200:
                                    click.echo("   ...")
                                click.secho("\n-----\n", fg="cyan")
                    except Exception as e:
                        click.secho(f"Search error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('get'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: get <id>", fg="magenta")
                        continue
                    doc_id = parts[1].strip()

                    try:
                        doc = db.get(doc_id)
                        if doc:
                            click.secho(f"Document: {doc_id}\n------------------", fg="cyan")
                            click.echo(doc.content)
                            if doc.metadata:
                                click.secho("\nMetadata:", fg="cyan")
                                click.echo(json.dumps(doc.metadata, indent=2))
                        else:
                            click.secho(f"Document `{doc_id}` not found.", fg="bright_red")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('delete'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: delete <id>", fg="magenta")
                        continue
                    doc_id = parts[1].strip()

                    try:
                        if db.exists(doc_id):
                            confirm = click.confirm(f"Are you sure you want to delete document '{doc_id}'?")
                            if confirm:
                                db.delete(doc_id)
                                click.secho(f"Document '{doc_id}' deleted.", fg="green")
                            else:
                                click.secho("Deletion canceled.", fg="yellow")
                        else:
                            click.secho(f"Document '{doc_id}' does not exist.", fg="bright_red")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('list'):
                    parts = command.split()
                    limit = 10
                    offset = 0

                    if len(parts) > 1 and parts[1].isdigit():
                        limit = int(parts[1])
                    if len(parts) > 2 and parts[2].isdigit():
                        offset = int(parts[2])

                    try:
                        docs = db.filter(limit=limit, offset=offset)
                        total = len(db.filter())  # Get total count

                        if not docs:
                            click.secho("No documents found.", fg="yellow")
                        else:
                            click.secho(f"Document IDs (showing {len(docs)} of {total}):", fg="blue")
                            for i, doc in enumerate(docs, offset + 1):
                                click.echo(f"{i}. {doc.id}")

                            if offset + limit < total:
                                click.secho(f"\nUse 'list {limit} {offset + limit}' to see the next page", fg="yellow")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'count':
                    try:
                        stats = db.get_stats()
                        click.secho(f"Document count: {stats['documents']}, Chunk count: {stats['chunks']}", fg="blue")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'stats':
                    print_db_stats(db)
                    continue

                if command.lower() == 'info':
                    try:
                        stats = db.get_stats()
                        click.secho("Database Information:", fg="blue")
                        click.echo(f"  Name: {db.name}")
                        click.echo(f"  Embedding model: {stats['embedding_model']}")
                        click.echo(f"  Embedding provider: {stats['embedding_provider']}")
                        click.echo(f"  Vector dimension: {stats['embedding_dimension']}")
                        click.echo(f"  Chunking method: {stats['chunking_method']}")
                        click.echo(f"  Chunk size: {stats['chunk_size']}")
                        click.echo(f"  Chunk overlap: {stats['chunk_overlap']}")
                        click.echo(f"  FTS search: {'enabled' if stats['fts_enabled'] else 'disabled'}")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'clear':
                    click.clear()
                    continue

                if command.lower().startswith('add '):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: add <file or glob>", fg="magenta")
                        continue

                    file_pattern = parts[1].strip()
                    matching_files = glob.glob(file_pattern, recursive=True)

                    if not matching_files:
                        click.secho(f"No files found matching '{file_pattern}'", fg="bright_red")
                        continue

                    click.secho(f"Found {len(matching_files)} files. Adding to database...", fg="blue")

                    documents = []
                    metadata = []

                    for file_path in matching_files:
                        try:
                            path = Path(file_path)
                            if not path.is_file():
                                click.secho(f"Skipping {file_path} (not a file)", fg="yellow")
                                continue

                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            except UnicodeError:
                                click.secho(f"Cannot decode {file_path} as unicode, skipping!", fg="yellow")
                                continue

                            documents.append(content)
                            metadata.append({
                                "source": file_path,
                                "filename": path.name,
                                "extension": path.suffix,
                                "added_at": datetime.now().isoformat()
                            })

                        except Exception as e:
                            click.secho(f"Error processing {file_path}: {str(e)}", fg="bright_red")

                    if documents:
                        try:
                            doc_ids = db.upsert(documents=documents, metadata=metadata)
                            click.secho(f"Successfully added {len(documents)} documents", fg="green")
                            click.echo(f"Created IDs: {', '.join(doc_ids)}")
                        except Exception as e:
                            click.secho(f"Error adding documents: {str(e)}", fg="bright_red")
                    continue

                # Unknown command
                click.secho(f"Unknown command: {command}", fg="bright_red")
                click.echo("Type 'help' for available commands")

            except click.exceptions.Abort:
                click.secho("\nCtrl+C detected, Exiting!", fg="red")
                break
            except Exception as e:
                click.secho(f"Error: {str(e)}", fg="bright_red")
                continue

        click.secho("Database connection closed.", fg="green")

    except Exception as e:
        click.secho(f"Fatal error: {str(e)}", fg="bright_red")
        raise click.Abort()
    finally:
        db.close()


