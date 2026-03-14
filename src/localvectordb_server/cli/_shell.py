# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_shell.py
from datetime import datetime

import click

from localvectordb_server.cli._utils import format_table, print_db_stats


@click.command('shell')
@click.pass_context
def shell(ctx):
    """
    Start an interactive shell for database operations.

    Launches an interactive shell for performing database operations such as search, add, get,
    delete, list, stats, schema management, and more. Type 'help' for available commands.

    """
    import glob
    import json
    import shlex
    from pathlib import Path

    db = ctx.obj['db']

    def parse_command(command_line):
        """Parse command line into command and arguments"""
        try:
            parts = shlex.split(command_line)
            if not parts:
                return None, []
            return parts[0], parts[1:]
        except ValueError as e:
            click.secho(f"Error parsing command: {e}", fg="bright_red")
            return None, []

    def show_help():
        """Display help for all available commands"""
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
        click.echo("")
        click.echo("Schema Management:")
        click.echo("  schema show [format]           - Show current schema (pretty|json|table)")
        click.echo("  schema update <file>           - Update schema from JSON file")
        click.echo("  schema update-str <json>       - Update schema from JSON string")
        click.echo("  schema export <file>           - Export current schema to file")
        click.echo("  schema map <old> <new>         - Add column mapping for next update")
        click.echo("  schema map-clear               - Clear column mappings")
        click.echo("  schema map-show                - Show current column mappings")
        click.echo("")
        click.echo("General:")
        click.echo("  clear                          - Clear the console")
        click.echo("  help                           - Show this help")
        click.echo("  exit/quit                      - Exit shell")

    def format_schema_table(schema_fields):
        """Format schema as a table"""
        if not schema_fields:
            return "No schema defined"

        headers = ['Field Name', 'Type', 'Indexed', 'Required', 'Default Value']
        rows = []
        for field_name, field_def in schema_fields.items():
            default_val = str(field_def.default_value) if field_def.default_value is not None else 'None'
            if len(default_val) > 25:
                default_val = default_val[:22] + "..."
            rows.append([
                field_name,
                field_def.type.value.upper(),
                "✓" if field_def.indexed else "✗",
                "✓" if field_def.required else "✗",
                default_val
            ])

        return format_table(headers, rows)

    def format_schema_pretty(schema_fields):
        """Format schema in pretty format"""
        if not schema_fields:
            return "No schema defined"

        output = click.style("Current Metadata Schema", fg="cyan", bold=True) + "\n"
        output += "=" * 24 + "\n\n"

        for field_name, field_def in schema_fields.items():
            output += click.style(f"Field: {field_name}", fg="green", bold=True) + "\n"
            output += f"  Type: {field_def.type.value.upper()}\n"
            output += f"  Indexed: {'Yes' if field_def.indexed else 'No'}\n"
            output += f"  Required: {'Yes' if field_def.required else 'No'}\n"
            if field_def.default_value is not None:
                if isinstance(field_def.default_value, (dict, list)):
                    default_display = json.dumps(field_def.default_value, indent=2)
                    output += f"  Default Value:\n    {default_display.replace(chr(10), chr(10) + '    ')}\n"
                else:
                    output += f"  Default Value: {field_def.default_value}\n"
            else:
                output += "  Default Value: None\n"
            output += "\n"

        return output

    def handle_schema_command(args):
        """Handle schema-related commands"""
        if not args:
            click.secho("Error: schema command requires a subcommand", fg="bright_red")
            click.echo("Try: schema show, schema update <file>, schema export <file>")
            return

        subcmd = args[0]

        if subcmd == "show":
            # Show current schema
            format_type = args[1] if len(args) > 1 else "pretty"

            try:
                schema_info = db.get_metadata_schema_info()
                schema_fields = schema_info.get('fields', {})

                if format_type == "json":
                    schema_data = {}
                    for field_name, field_def in schema_fields.items():
                        schema_data[field_name] = {
                            'type': field_def.type.value,
                            'indexed': field_def.indexed,
                            'required': field_def.required,
                            'default_value': field_def.default_value
                        }
                    click.echo(json.dumps(schema_data, indent=2))
                elif format_type == "table":
                    click.echo(format_schema_table(schema_fields))
                else:  # pretty
                    click.echo(format_schema_pretty(schema_fields))

            except Exception as e:
                click.secho(f"Error retrieving schema: {str(e)}", fg='bright_red')

        elif subcmd == "update":
            # Update schema from file
            if len(args) < 2:
                click.secho("Error: schema update requires a file path", fg="bright_red")
                return

            file_path = args[1]
            if not Path(file_path).exists():
                click.secho(f"Error: File '{file_path}' not found", fg="bright_red")
                return

            try:
                with open(file_path, 'r') as f:
                    schema_data = json.load(f)

                # Convert to MetadataField objects
                from localvectordb.core import MetadataField, MetadataFieldType
                new_schema = {}
                for field_name, field_config in schema_data.items():
                    if isinstance(field_config, str):
                        new_schema[field_name] = MetadataField(type=MetadataFieldType(field_config))
                    elif isinstance(field_config, dict):
                        field_type = MetadataFieldType(field_config['type'])
                        new_schema[field_name] = MetadataField(
                            type=field_type,
                            indexed=field_config.get('indexed', False),
                            required=field_config.get('required', False),
                            default_value=field_config.get('default_value', None)
                        )

                # Check if we have column mappings
                column_mapping = getattr(handle_schema_command, '_column_mapping', None)

                # Show planned changes
                click.echo(f"\n{click.style('Planned Changes:', fg='cyan', bold=True)}")
                current_schema = db.metadata_schema

                new_fields = [name for name in new_schema.keys() if name not in current_schema]
                if new_fields:
                    click.echo(f"  {click.style('New fields:', fg='green')} {', '.join(new_fields)}")

                removed_fields = []
                for name in current_schema.keys():
                    if name not in new_schema:
                        if not column_mapping or name not in column_mapping:
                            removed_fields.append(name)
                if removed_fields:
                    click.echo(f"  {click.style('Removed fields:', fg='red')} {', '.join(removed_fields)}")

                if column_mapping:
                    click.echo(f"  {click.style('Column remapping:', fg='blue')}")
                    for old_col, new_col in column_mapping.items():
                        click.echo(f"    {old_col} → {new_col}")

                if not new_fields and not removed_fields and not column_mapping:
                    click.echo("  No changes detected.")
                    return

                # Confirm changes
                if not click.confirm(f"\n{click.style('Proceed with schema update?', fg='yellow')}"):
                    click.echo("Schema update cancelled.")
                    return

                # Apply update
                click.echo(f"\n{click.style('Applying schema update...', fg='blue')}")
                changes = db.update_metadata_schema(
                    new_schema=new_schema,
                    column_mapping=column_mapping
                )

                # Clear column mapping after use
                if hasattr(handle_schema_command, '_column_mapping'):
                    delattr(handle_schema_command, '_column_mapping')

                # Report results
                click.echo(f"\n{click.style('Schema Update Complete!', fg='green', bold=True)}")

                if changes['added_fields']:
                    click.echo(f"  {click.style('Added fields:', fg='green')} {', '.join(changes['added_fields'])}")

                if changes['removed_fields']:
                    click.echo(f"  {click.style('Removed fields:', fg='red')} {', '.join(changes['removed_fields'])}")

                if changes['remapped_columns']:
                    click.echo(f"  {click.style('Remapped columns:', fg='cyan')}")
                    for remap in changes['remapped_columns']:
                        click.echo(f"    {remap['old_column']} → {remap['new_column']} "
                                   f"({remap['rows_transferred']} rows transferred)")

                if changes['populated_defaults']:
                    click.echo(f"  {click.style('Populated defaults:', fg='yellow')}")
                    for default_info in changes['populated_defaults']:
                        click.echo(f"    {default_info['field_name']}: {default_info['rows_updated']} rows updated")

                if changes['warnings']:
                    click.echo(f"\n{click.style('Warnings:', fg='yellow')}")
                    for warning in changes['warnings']:
                        click.echo(f"  ⚠ {warning}")

                if changes['errors']:
                    click.echo(f"\n{click.style('Errors:', fg='red')}")
                    for error in changes['errors']:
                        click.echo(f"  ✗ {error}")

            except json.JSONDecodeError as e:
                click.secho(f"Error: Invalid JSON in file: {str(e)}", fg='bright_red')
            except Exception as e:
                click.secho(f"Error updating schema: {str(e)}", fg='bright_red')

        elif subcmd == "update-str":
            # Update schema from JSON string
            if len(args) < 2:
                click.secho("Error: schema update-str requires a JSON string", fg="bright_red")
                return

            json_str = " ".join(args[1:])  # Rejoin the JSON string

            try:
                schema_data = json.loads(json_str)

                # Convert to MetadataField objects
                from localvectordb.core import MetadataField, MetadataFieldType
                new_schema = {}
                for field_name, field_config in schema_data.items():
                    if isinstance(field_config, str):
                        new_schema[field_name] = MetadataField(type=MetadataFieldType(field_config))
                    elif isinstance(field_config, dict):
                        field_type = MetadataFieldType(field_config['type'])
                        new_schema[field_name] = MetadataField(
                            type=field_type,
                            indexed=field_config.get('indexed', False),
                            required=field_config.get('required', False),
                            default_value=field_config.get('default_value', None)
                        )

                # Use same logic as file-based update
                column_mapping = getattr(handle_schema_command, '_column_mapping', None)

                # Apply update with confirmation
                if click.confirm(f"{click.style('Apply schema update?', fg='yellow')}"):
                    changes = db.update_metadata_schema(
                        new_schema=new_schema,
                        column_mapping=column_mapping
                    )

                    if hasattr(handle_schema_command, '_column_mapping'):
                        delattr(handle_schema_command, '_column_mapping')

                    click.echo(f"{click.style('Schema updated successfully!', fg='green')}")
                    if changes['remapped_columns']:
                        for remap in changes['remapped_columns']:
                            click.echo(f"  Remapped: {remap['old_column']} → {remap['new_column']}")

            except json.JSONDecodeError as e:
                click.secho(f"Error: Invalid JSON string: {str(e)}", fg='bright_red')
            except Exception as e:
                click.secho(f"Error updating schema: {str(e)}", fg='bright_red')

        elif subcmd == "export":
            # Export current schema
            if len(args) < 2:
                click.secho("Error: schema export requires a file path", fg="bright_red")
                return

            file_path = args[1]

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

                    export_data[field_name] = field_data

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                click.echo(f"Schema exported to {file_path}")
                click.echo(f"Fields exported: {len(export_data)}")

            except Exception as e:
                click.secho(f"Error exporting schema: {str(e)}", fg='bright_red')

        elif subcmd == "map":
            # Add column mapping
            if len(args) < 3:
                click.secho("Error: schema map requires <old_column> <new_column>", fg="bright_red")
                return

            old_col, new_col = args[1], args[2]

            # Store mapping for next update
            if not hasattr(handle_schema_command, '_column_mapping'):
                handle_schema_command._column_mapping = {}

            handle_schema_command._column_mapping[old_col] = new_col
            click.echo(f"Added column mapping: {old_col} → {new_col}")
            click.echo("Use 'schema map-show' to see all mappings, 'schema update' to apply")

        elif subcmd == "map-clear":
            # Clear column mappings
            if hasattr(handle_schema_command, '_column_mapping'):
                delattr(handle_schema_command, '_column_mapping')
            click.echo("Column mappings cleared")

        elif subcmd == "map-show":
            # Show current column mappings
            column_mapping = getattr(handle_schema_command, '_column_mapping', None)
            if column_mapping:
                click.echo("Current column mappings:")
                for old_col, new_col in column_mapping.items():
                    click.echo(f"  {old_col} → {new_col}")
            else:
                click.echo("No column mappings defined")

        else:
            click.secho(f"Unknown schema command: {subcmd}", fg="bright_red")
            click.echo("Available schema commands: show, update, update-str, export, map, map-clear, map-show")

    try:
        click.echo(click.style("Connected to database: ", fg="green")
                   + click.style(db.name, fg="green", underline=True))

        stats = db.get_stats()
        click.secho(f"Documents: {stats['documents']}, Chunks: {stats['chunks']}", fg="blue")
        click.echo("Type 'help' for available commands, 'exit' to quit")

        # Simple REPL with enhanced schema support
        while True:
            try:
                command_line = click.prompt(f"{db.name}> ", type=str)
                command, args = parse_command(command_line)

                if not command:
                    continue

                if command.lower() in ('exit', 'quit', 'q'):
                    break

                elif command.lower() in ('help', '?'):
                    show_help()
                    continue

                elif command.lower() == 'schema':
                    handle_schema_command(args)
                    continue

                elif command.lower() == 'clear':
                    # Clear the console
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                elif command.lower() == 'count':
                    stats = db.get_stats()
                    click.echo(f"Total documents: {stats['documents']}")
                    continue

                elif command.lower() == 'stats':
                    print_db_stats(db)
                    continue

                elif command.lower() == 'info':
                    stats = db.get_stats()
                    click.echo("Database Info")
                    click.echo("-------------")
                    click.echo(f"  Database: {db.name}")
                    click.echo(f"  Embedding model: {stats['embedding_model']}")
                    click.echo(f"  Total Documents: {stats['documents']}")
                    click.echo(f"  Total Chunks: {stats['chunks']}")
                    click.echo(f"  Schema fields: {len(db.metadata_schema)}")
                    continue

                elif command.lower() == 'list':
                    limit = int(args[0]) if len(args) > 0 and args[0].isdigit() else 20
                    offset = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0

                    doc_ids = db.list_document_ids(limit=limit, offset=offset)
                    if doc_ids:
                        for doc_id in doc_ids:
                            click.echo(doc_id)
                    else:
                        click.echo("No documents found")
                    continue

                elif command.lower() == 'get':
                    if not args:
                        click.secho("Error: get command requires a document ID", fg="bright_red")
                        continue

                    doc_id = args[0]
                    try:
                        doc = db.get(doc_id)
                        if doc:
                            click.echo(f"Document: {doc.id}")
                            click.echo("-" * 40)
                            click.echo(doc.content)
                            if doc.metadata:
                                click.echo("\nMetadata:")
                                click.echo(json.dumps(doc.metadata, indent=2))
                        else:
                            click.echo(f"Document {doc_id} not found")
                    except Exception as e:
                        click.secho(f"Error retrieving document: {str(e)}", fg="bright_red")
                    continue

                elif command.lower() == 'delete':
                    if not args:
                        click.secho("Error: delete command requires a document ID", fg="bright_red")
                        continue

                    doc_id = args[0]
                    try:
                        if db.exists(doc_id):
                            if click.confirm(f"Delete document '{doc_id}'?"):
                                deleted_count = db.delete(doc_id)
                                if deleted_count > 0:
                                    click.echo(f"Successfully deleted document: {doc_id}")
                                else:
                                    click.echo("No documents were deleted")
                        else:
                            click.echo(f"Document {doc_id} not found")
                    except Exception as e:
                        click.secho(f"Error deleting document: {str(e)}", fg="bright_red")
                    continue

                elif command.lower() == 'search':
                    if not args:
                        click.secho("Error: search command requires a query", fg="bright_red")
                        continue

                    query = args[0]
                    limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
                    search_type = args[2] if len(args) > 2 and args[2] in ['vector', 'keyword', 'hybrid'] else 'vector'

                    try:
                        results = db.search(query, limit=limit, search_type=search_type)

                        if results:
                            click.echo(f"Found {len(results)} results:")
                            click.echo("=" * 40)
                            for i, result in enumerate(results, 1):
                                click.echo(f"\n{i}. Document: {result.id} (Score: {result.score:.4f})")
                                click.echo("-" * 40)
                                # Truncate content for display
                                content = result.content
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                click.echo(content)
                        else:
                            click.echo("No results found")
                    except Exception as e:
                        click.secho(f"Error searching: {str(e)}", fg="bright_red")
                    continue

                elif command.lower() == 'add':
                    if not args:
                        click.secho("Error: add command requires file path or glob pattern", fg="bright_red")
                        continue

                    pattern = args[0]
                    matching_files = glob.glob(pattern, recursive=True)

                    if not matching_files:
                        click.secho(f"No files found matching pattern: {pattern}", fg="yellow")
                        continue

                    click.secho(f"Found {len(matching_files)} file(s). Adding to database...", fg="blue")

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
        raise click.Abort() from e
    finally:
        db.close()
