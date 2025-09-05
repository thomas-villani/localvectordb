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

import click

from localvectordb_server.cli._utils import find_config_file, EXIT_CODE_ERROR, print_db_stats, get_stdin_input, \
    format_table


@click.group('db')
@click.argument("name")
@click.pass_context
def db_group(ctx, name):
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
    db_folder = ctx.obj["db_folder"]

    # if not db_folder:
    #     cfg = ctx.obj["config"]
    #     db_folder = cfg.database.root_dir

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

    ctx.obj = {"db_name": name, "db": db}


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


@db_group.group('schema')
@click.pass_context
def schema_group(ctx):
    """
    Manage database metadata schema.

    Commands for viewing and updating the metadata schema of the database, including
    support for column remapping to rename existing columns while preserving data.

    \b
    Examples:
        \b
        lvdb db mydb schema show
        lvdb db mydb schema update --schema schema.json
        lvdb db mydb schema update --schema '{"title": "text", "author": "text"}' --mapping '{"old_author": "author"}'
    """
    pass


@schema_group.command('show')
@click.option('--format', '-f', type=click.Choice(['pretty', 'json', 'table']), default='pretty',
              help='Output format')
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False),
              help='Output to file instead of stdout')
@click.pass_context
def show_schema(ctx, format, output):
    """
    Display current metadata schema.

    Shows the current metadata schema with field definitions, types, and properties.
    Supports multiple output formats for different use cases.

    \b
    Examples:
        \b
        lvdb db mydb schema show
        lvdb db mydb schema show --format json
        lvdb db mydb schema show --format table --output schema.json
    """
    db = ctx.obj['db']

    try:
        schema_info = db.get_metadata_schema_info()
        schema_fields = schema_info.get('fields', {})

        if not schema_fields:
            click.echo("No metadata schema defined for this database.")
            return

        output_str = ""

        if format == 'json':
            # JSON format - suitable for programmatic use
            schema_data = {}
            for field_name, field_def in schema_fields.items():
                schema_data[field_name] = {
                    'type': field_def.type.value,
                    'indexed': field_def.indexed,
                    'required': field_def.required,
                    'default_value': field_def.default_value
                }
            output_str = json.dumps(schema_data, indent=2)

        elif format == 'table':
            # Table format - good for overview
            headers = ['Field Name', 'Type', 'Indexed', 'Required', 'Default Value']
            rows = []
            for field_name, field_def in schema_fields.items():
                default_val = str(field_def.default_value) if field_def.default_value is not None else 'None'
                if len(default_val) > 30:
                    default_val = default_val[:27] + "..."
                rows.append([
                    field_name,
                    field_def.type.value.upper(),
                    "✓" if field_def.indexed else "✗",
                    "✓" if field_def.required else "✗",
                    default_val
                ])
            output_str = format_table(headers, rows)

        else:  # pretty format
            # Pretty format - human readable
            output_str += click.style("Database Metadata Schema", fg="cyan", bold=True) + "\n"
            output_str += "=" * 25 + "\n\n"

            for field_name, field_def in schema_fields.items():
                output_str += click.style(f"Field: {field_name}", fg="green", bold=True) + "\n"
                output_str += f"  Type: {field_def.type.value.upper()}\n"
                output_str += f"  Indexed: {'Yes' if field_def.indexed else 'No'}\n"
                output_str += f"  Required: {'Yes' if field_def.required else 'No'}\n"
                if field_def.default_value is not None:
                    if isinstance(field_def.default_value, (dict, list)):
                        default_display = json.dumps(field_def.default_value, indent=4)
                        output_str += f"  Default Value:\n    {default_display.replace(chr(10), chr(10) + '    ')}\n"
                    else:
                        output_str += f"  Default Value: {field_def.default_value}\n"
                else:
                    output_str += f"  Default Value: None\n"
                output_str += "\n"

        if output:
            with open(output, 'w') as f:
                f.write(output_str)
            click.echo(f"Schema information saved to {output}")
        else:
            click.echo(output_str)

    except Exception as e:
        click.secho(f"Error retrieving schema: {str(e)}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@schema_group.command('update')
@click.option('--schema', '-s', type=str,
              help='Path to JSON file or JSON string containing new schema definition')
@click.option('--mapping', '-m', type=str,
              help='Column mapping as JSON string or or path to JSON file (old_name: new_name)')
@click.option('--drop-columns', '--drop', is_flag=True, default=False,
              help='Actually drop removed columns (WARNING: data loss)')
@click.option('--dry-run', '--dry', is_flag=True, default=False,
              help='Show what would be changed without making changes')
@click.option('--force', '-f', is_flag=True, default=False,
              help='Skip confirmation prompts')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Show detailed output')
@click.pass_context
def update_schema(ctx, schema, mapping, drop_columns, dry_run, force, verbose):
    """
    Update database metadata schema with optional column remapping.

    Updates the metadata schema and optionally renames existing columns by transferring
    their data. Supports both file-based and command-line input for schema and mappings.

    \b
    The new schema can be provided as:
    - JSON file or JSON string with --schema

    \b
    Column mappings can be provided as:
    - JSON file or JSON string with --mapping

    \b
    Schema Format:
        {
            "field_name": {
                "type": "text|integer|real|boolean|date|json",
                "indexed": true|false,
                "required": true|false,
                "default_value": value
            }
        }

    \b
    Mapping Format:
        {
            "old_column_name": "new_column_name",
            "another_old": "another_new"
        }

    \b
    Examples:
        \b
        # Update schema from file
        lvdb db mydb schema update --schema new_schema.json

        # Update with column remapping
        lvdb db mydb schema update --schema new_schema.json --mapping '{"old_author": "author"}'

        # Dry run to see changes
        lvdb db mydb schema update --schema new_schema.json --dry-run

        # Update with file-based mapping
        lvdb db mydb schema update --schema new_schema.json --mapping mappings.json

        # Shorthand schema with string input
        lvdb db mydb schema update --schema '{"title": "text", "author": "text"}' --mapping '{"old_author": "author"}'
    """
    db = ctx.obj['db']

    # Validate input combinations
    if not schema:
        click.secho("Error: --schema must be provided", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    try:
        # Parse new schema
        if os.path.exists(schema):

            if verbose:
                click.echo(f"Loading schema from file: {schema}")
            with open(schema, 'r') as f:
                schema_data = json.load(f)
        else:
            if verbose:
                click.echo("Parsing schema from command line")
            schema_data = json.loads(schema)

        # Parse column mapping if provided
        column_mapping = None
        if mapping and os.path.exists(mapping):
            if verbose:
                click.echo(f"Loading column mapping from file: {mapping}")
            with open(mapping, 'r') as f:
                column_mapping = json.load(f)
        elif mapping:
            if verbose:
                click.echo("Parsing column mapping from command line")
            column_mapping = json.loads(mapping)

        # Convert schema data to MetadataField objects
        from localvectordb.core import MetadataField, MetadataFieldType

        new_schema = {}
        for field_name, field_config in schema_data.items():
            if isinstance(field_config, str):
                # Simple string type
                new_schema[field_name] = MetadataField(type=MetadataFieldType(field_config))
            elif isinstance(field_config, dict):
                # Full configuration
                field_type = MetadataFieldType(field_config['type'])
                indexed = field_config.get('indexed', False)
                required = field_config.get('required', False)
                default_value = field_config.get('default_value', None)

                new_schema[field_name] = MetadataField(
                    type=field_type,
                    indexed=indexed,
                    required=required,
                    default_value=default_value
                )
            else:
                click.secho(f"Error: Invalid field configuration for '{field_name}'", fg='bright_red')
                raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Show current schema for comparison
        if verbose:
            click.echo("\n" + click.style("Current Schema:", fg="yellow"))
            current_schema = db.metadata_schema
            if current_schema:
                for field_name, field_def in current_schema.items():
                    click.echo(f"  {field_name}: {field_def.type.value}")
            else:
                click.echo("  (No schema defined)")

        # Show planned changes
        click.echo("\n" + click.style("Planned Changes:", fg="cyan", bold=True))

        current_schema = db.metadata_schema

        # Show new fields
        new_fields = [name for name in new_schema.keys() if name not in current_schema]
        if new_fields:
            click.echo(f"  {click.style('New fields:', fg='green')} {', '.join(new_fields)}")

        # Show removed fields (accounting for remapping)
        removed_fields = []
        for name in current_schema.keys():
            if name not in new_schema:
                # Check if it's being remapped
                if not column_mapping or name not in column_mapping:
                    removed_fields.append(name)
        if removed_fields:
            click.echo(f"  {click.style('Removed fields:', fg='red')} {', '.join(removed_fields)}")

        # Show column remapping
        if column_mapping:
            click.echo(f"  {click.style('Column remapping:', fg='blue')}")
            for old_col, new_col in column_mapping.items():
                click.echo(f"    {old_col} → {new_col}")

        # Show warnings
        if drop_columns and removed_fields:
            click.echo(f"  {click.style('WARNING:', fg='bright_red', bold=True)} "
                       f"--drop-columns specified. Data in removed columns will be permanently lost!")

        if not new_fields and not removed_fields and not column_mapping:
            click.echo("  No changes detected.")
            return

        # Confirm changes unless forced or dry-run
        if not dry_run and not force:
            click.echo()
            if not click.confirm(click.style("Proceed with schema update?", fg="yellow")):
                click.echo("Schema update cancelled.")
                return

        if dry_run:
            click.echo(f"\n{click.style('DRY RUN:', fg='blue', bold=True)} No changes were made.")
            return

        # Apply the schema update
        click.echo(f"\n{click.style('Applying schema update...', fg='blue')}")

        changes = db.update_metadata_schema(
            new_schema=new_schema,
            column_mapping=column_mapping,
            drop_columns=drop_columns
        )

        # Report results
        click.echo(f"\n{click.style('Schema Update Complete!', fg='green', bold=True)}")

        if changes['added_fields']:
            click.echo(f"  {click.style('Added fields:', fg='green')} {', '.join(changes['added_fields'])}")

        if changes['removed_fields']:
            click.echo(f"  {click.style('Removed fields:', fg='red')} {', '.join(changes['removed_fields'])}")

        if changes['modified_fields']:
            modified_names = [f['field_name'] for f in changes['modified_fields']]
            click.echo(f"  {click.style('Modified fields:', fg='blue')} {', '.join(modified_names)}")

        if changes['remapped_columns']:
            click.echo(f"  {click.style('Remapped columns:', fg='cyan')}")
            for remap in changes['remapped_columns']:
                click.echo(f"    {remap['old_column']} → {remap['new_column']} "
                           f"({remap['rows_transferred']} rows transferred)")

        if changes['populated_defaults']:
            click.echo(f"  {click.style('Populated defaults:', fg='yellow')}")
            for default_info in changes['populated_defaults']:
                click.echo(f"    {default_info['field_name']}: {default_info['rows_updated']} rows updated")

        # Show warnings and errors
        if changes['warnings']:
            click.echo(f"\n{click.style('Warnings:', fg='yellow')}")
            for warning in changes['warnings']:
                click.echo(f"  ⚠ {warning}")

        if changes['errors']:
            click.echo(f"\n{click.style('Errors:', fg='red')}")
            for error in changes['errors']:
                click.echo(f"  ✗ {error}")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        if verbose:
            click.echo(f"\n{click.style('Updated Schema:', fg='green')}")
            updated_schema = db.metadata_schema
            for field_name, field_def in updated_schema.items():
                indexed_str = " (indexed)" if field_def.indexed else ""
                required_str = " (required)" if field_def.required else ""
                default_str = f" (default: {field_def.default_value})" if field_def.default_value is not None else ""
                click.echo(f"  {field_name}: {field_def.type.value}{indexed_str}{required_str}{default_str}")

    except json.JSONDecodeError as e:
        click.secho(f"Error: Invalid JSON format: {str(e)}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    except KeyError as e:
        click.secho(f"Error: Missing required field in schema: {str(e)}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    except ValueError as e:
        click.secho(f"Error: {str(e)}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    except Exception as e:
        click.secho(f"Error: Unexpected error during schema update: {str(e)}", fg='bright_red')
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@schema_group.command('export')
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'toml']), default='json',
              help='Output format')
@click.option('--include-data', '--with-data', is_flag=True, default=False,
              help='Include sample data for each field type')
@click.pass_context
def export_schema(ctx, output, format, include_data):
    """
    Export current schema to a file.

    Exports the current metadata schema to a JSON or TOML file that can be used
    with the 'update' command or modified for schema changes.

    \b
    Examples:
        \b
        lvdb db mydb schema export --output current_schema.json
        lvdb db mydb schema export --output schema.toml --format toml
        lvdb db mydb schema export --output schema_with_samples.json --include-data
    """
    db = ctx.obj['db']

    try:
        schema_info = db.get_metadata_schema_info()
        schema_fields = schema_info.get('fields', {})

        if not schema_fields:
            click.echo("No metadata schema to export.")
            return

        # Convert to exportable format
        export_data = {}
        for field_name, field_def in schema_fields.items():
            field_data = {
                'type': field_def.type.value,
                'indexed': field_def.indexed,
                'required': field_def.required
            }

            if field_def.default_value is not None:
                field_data['default_value'] = field_def.default_value

            # Add sample data if requested
            if include_data:
                sample_values = {
                    'text': 'Sample text value',
                    'integer': 42,
                    'real': 3.14159,
                    'boolean': True,
                    'date': '2024-01-01',
                    'json': {'key': 'value', 'array': [1, 2, 3]}
                }
                field_data['_sample_value'] = sample_values.get(field_def.type.value, 'Sample value')

            export_data[field_name] = field_data

        # Write to file
        if format == 'toml':
            try:
                import toml
                with open(output, 'w') as f:
                    toml.dump({'metadata_schema': export_data}, f)
            except ImportError:
                click.secho("Error: TOML format requires the 'toml' package. Install with: pip install toml",
                            fg='bright_red')
                click.secho("Falling back to JSON format...", fg='yellow')
                format = 'json'  # Fall back to JSON

        if format == 'json':  # Handle both explicit JSON and fallback case
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2)

        click.echo(f"Schema exported to {output}")
        click.echo(f"Fields exported: {len(export_data)}")

    except Exception as e:
        click.secho(f"Error exporting schema: {str(e)}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


# Helper function to add to the existing CLI utilities section
def _validate_schema_format(schema_data):
    """Validate that schema data has the correct format"""
    if not isinstance(schema_data, dict):
        raise ValueError("Schema must be a dictionary")

    valid_types = {'text', 'integer', 'real', 'boolean', 'date', 'json'}

    for field_name, field_config in schema_data.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"Field name must be a non-empty string: '{field_name}'")

        if isinstance(field_config, str):
            # Simple string type
            if field_config not in valid_types:
                raise ValueError(f"Invalid field type '{field_config}' for field '{field_name}'. "
                                 f"Valid types: {', '.join(valid_types)}")
        elif isinstance(field_config, dict):
            # Full configuration
            if 'type' not in field_config:
                raise ValueError(f"Field '{field_name}' missing required 'type' property")

            if field_config['type'] not in valid_types:
                raise ValueError(f"Invalid field type '{field_config['type']}' for field '{field_name}'. "
                                 f"Valid types: {', '.join(valid_types)}")

            # Validate boolean properties
            for bool_prop in ['indexed', 'required']:
                if bool_prop in field_config and not isinstance(field_config[bool_prop], bool):
                    raise ValueError(f"Property '{bool_prop}' for field '{field_name}' must be boolean")
        else:
            raise ValueError(f"Invalid field configuration for '{field_name}'. "
                             f"Must be string type or configuration object")


def _validate_mapping_format(mapping_data):
    """Validate that mapping data has the correct format"""
    if not isinstance(mapping_data, dict):
        raise ValueError("Column mapping must be a dictionary")

    for old_col, new_col in mapping_data.items():
        if not isinstance(old_col, str) or not old_col.strip():
            raise ValueError(f"Old column name must be a non-empty string: '{old_col}'")
        if not isinstance(new_col, str) or not new_col.strip():
            raise ValueError(f"New column name must be a non-empty string: '{new_col}'")


# Add the shell command!
from localvectordb_server.cli._shell import shell
db_group.add_command(shell)
