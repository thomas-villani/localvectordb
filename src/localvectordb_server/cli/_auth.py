# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_auth.py

from datetime import UTC

import click

from localvectordb_server.cli._utils import EXIT_CODE_ERROR
from localvectordb_server.keymanager import PermissionLevel


@click.group()
@click.pass_context
def auth(ctx):
    """
    Manage API authentication settings.

    Provides subcommands to create, list, revoke, rotate, and manage API keys for authentication.

    \b
    Examples:
        \b
        lvdb auth create-key --description "For testing"
        lvdb auth list-keys --active-only
        lvdb auth revoke-key <key_id>
        lvdb auth status
    """
    pass


@auth.command('create-key')
@click.option('--description', '-d', help='Description of the key purpose')
@click.option('--expires-days', type=int, help='Number of days until key expires')
@click.option('--created-by', help='Identifier of who is creating the key')
@click.option('--permission-level', '-p', type=click.Choice(['read_only', 'read_write']),
              default='read_write', help='Permission level for the key (default: read_write)')
@click.option('--output', '-o', type=click.Choice(['table', 'json', 'key-only']),
              default='table', help='Output format')
@click.pass_context
def create_api_key(ctx, description, expires_days, created_by, permission_level, output):
    """
    Create a new API key.

    Generates a new API key for authenticating to the server. You can specify a description,
    expiration, permission level, and creator. Output can be shown as a table, JSON, or key only.

    \b
    Examples:
        \b
        lvdb auth create-key --description "For admin" --permission-level read_write
        lvdb auth create-key --description "For monitoring" --permission-level read_only
        lvdb auth create-key --expires-days 30 --output json
    """
    try:
        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        # Convert string to enum
        perm_enum = PermissionLevel(permission_level)

        # Create the key
        key_record = key_manager.create_key(
            description=description,
            expires_days=expires_days,
            created_by=created_by,
            permission_level=perm_enum
        )

        if output == 'key-only':
            # Just output the key for scripting
            click.echo(key_record.plain_key)
        elif output == 'json':
            # JSON output
            import json
            output_data = key_record.to_dict()
            output_data['plain_key'] = key_record.plain_key
            click.echo(json.dumps(output_data, indent=2))
        else:
            # Table format (default)
            click.secho("✓ API Key Created Successfully", fg="green", bold=True)
            click.echo()
            click.secho("Key Details:", fg="cyan")
            click.echo(f"  Key ID: {key_record.id}")
            click.echo(f"  Description: {key_record.description or 'None'}")
            click.echo(f"  Permission Level: {key_record.permission_level.value}")
            click.echo(f"  Created: {key_record.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            if key_record.expires_at:
                click.echo(f"  Expires: {key_record.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                click.echo(f"  Days until expiry: {key_record.days_until_expiry}")
            else:
                click.echo("  Expires: Never")

            click.echo()
            click.secho("API Key (save this now - it won't be shown again):", fg="yellow", bold=True)
            click.secho(f"  {key_record.plain_key}", fg="green", bold=True)
            click.echo()
            click.secho("⚠️  Store this key securely - it cannot be retrieved again!", fg="red")

    except Exception as e:
        click.secho(f"Error creating API key: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('list-keys')
@click.option('--active-only', '-a', is_flag=True, help='Show only active keys')
@click.option('--include-expired/--no-expired', default=True, help='Include expired keys')
@click.option('--output', '-o', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.option('--show-stats', '-s', is_flag=True, help='Show key management statistics')
@click.pass_context
def list_api_keys(ctx, active_only, include_expired, output, show_stats):
    """
    List API keys.

    Lists all API keys in the key database. You can filter for active/expired keys and show
    management statistics. Output can be table or JSON.

    \b
    Examples:
        \b
        lvdb auth list-keys
        lvdb auth list-keys --active-only --output json
    """
    try:
        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        # Get keys
        keys = key_manager.list_keys(
            active_only=active_only,
            include_expired=include_expired
        )

        if output == 'json':
            import json
            keys_data = [key.to_dict() for key in keys]

            if show_stats:
                stats = key_manager.get_stats()
                output_data = {
                    'keys': keys_data,
                    'stats': stats
                }
            else:
                output_data = keys_data

            click.echo(json.dumps(output_data, indent=2))

        else:
            # Table format
            if show_stats:
                stats = key_manager.get_stats()
                click.secho("Key Management Statistics:", fg="blue", bold=True)
                click.echo(f"  Total keys: {stats['total_keys']}")
                click.echo(f"  Active keys: {stats['active_keys']}")
                click.echo(f"  Expired keys: {stats['expired_keys']}")
                click.echo(f"  Expiring soon (7 days): {stats['expiring_soon']}")
                click.echo(f"  Recently used (24h): {stats['recently_used']}")
                click.echo()

            if not keys:
                click.secho("No API keys found.", fg="yellow")
                return

            click.secho("API Keys:", fg="blue", bold=True)
            click.echo()

            # Table header
            click.secho(
                f"{'ID':<20} {'Description':<25} {'Permission':<12} {'Status':<10} {'Created':<12} {'Expires':<12} {'Last Used':<12}",
                fg="cyan")
            click.secho("-" * 135, fg="cyan")

            for key in keys:
                # Status
                if not key.active:
                    status = click.style("REVOKED", fg="red")
                elif key.is_expired:
                    status = click.style("EXPIRED", fg="yellow")
                else:
                    status = click.style("ACTIVE", fg="green")

                # Dates
                created = key.created_at.strftime('%Y-%m-%d') if key.created_at else 'Unknown'

                if key.expires_at:
                    expires = key.expires_at.strftime('%Y-%m-%d')
                    if key.days_until_expiry is not None and key.days_until_expiry <= 7 and not key.is_expired:
                        expires = click.style(expires, fg="yellow")
                else:
                    expires = "Never"

                last_used = key.last_used.strftime('%Y-%m-%d') if key.last_used else "Never"

                # Description truncation
                desc = (key.description or "")[:23]
                if len(key.description or "") > 23:
                    desc += ".."

                # Permission level with styling
                perm_display = key.permission_level.value
                if key.permission_level == PermissionLevel.READ_ONLY:
                    perm_display = click.style(perm_display, fg="blue")
                else:
                    perm_display = click.style(perm_display, fg="green")

                click.echo(
                    f"{key.id:<20} {desc:<25} {perm_display:<22} {status:<20} {created:<12} {expires:<22} {last_used:<12}")

    except Exception as e:
        click.secho(f"Error listing API keys: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('revoke-key')
@click.argument('key_id')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def revoke_api_key(ctx, key_id, confirm):
    """
    Revoke (deactivate) an API key.

    Deactivates the specified API key, preventing further use. Prompts for confirmation unless
    the --confirm flag is used.

    \b
    Examples:
        \b
        lvdb auth revoke-key <key_id>
        lvdb auth revoke-key <key_id> --confirm
    """
    try:
        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        # Get key details for confirmation
        key_record = key_manager.get_key(key_id)
        if not key_record:
            click.secho(f"Key '{key_id}' not found.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        if not key_record.active:
            click.secho(f"Key '{key_id}' is already revoked.", fg="yellow")
            return

        # Confirmation
        if not confirm:
            click.echo("Key Details:")
            click.echo(f"  ID: {key_record.id}")
            click.echo(f"  Description: {key_record.description or 'None'}")
            click.echo(f"  Created: {key_record.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            click.echo()

            if not click.confirm(
                    click.style(f"Are you sure you want to revoke key '{key_id}'?", fg="red")
            ):
                click.echo("Revocation cancelled.")
                return

        # Revoke the key
        success = key_manager.revoke_key(key_id)
        if success:
            click.secho(f"✓ Key '{key_id}' has been revoked.", fg="green")
        else:
            click.secho(f"Failed to revoke key '{key_id}'.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

    except Exception as e:
        click.secho(f"Error revoking API key: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('rotate-key')
@click.argument('key_id')
@click.option('--output', '-o', type=click.Choice(['table', 'json', 'key-only']),
              default='table', help='Output format')
@click.pass_context
def rotate_api_key(ctx, key_id, output):
    """
    Rotate an API key (create new, deactivate old).

    Creates a new API key to replace an existing one, deactivating the old key. Outputs the new key
    in table, JSON, or key-only format.

    \b
    Examples:
        \b
        lvdb auth rotate-key <key_id>
        lvdb auth rotate-key <key_id> --output key-only
    """
    try:
        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        # Get original key details
        original_key = key_manager.get_key(key_id)
        if not original_key:
            click.secho(f"Key '{key_id}' not found.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        if not original_key.active:
            click.secho(f"Cannot rotate inactive key '{key_id}'.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Rotate the key
        new_key = key_manager.rotate_key(key_id)
        if not new_key:
            click.secho(f"Failed to rotate key '{key_id}'.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        if output == 'key-only':
            click.echo(new_key.plain_key)
        elif output == 'json':
            import json
            output_data = {
                'original_key_id': key_id,
                'new_key': new_key.to_dict()
            }
            output_data['new_key']['plain_key'] = new_key.plain_key
            click.echo(json.dumps(output_data, indent=2))
        else:
            click.secho("✓ API Key Rotated Successfully", fg="green", bold=True)
            click.echo()
            click.secho("Original Key:", fg="cyan")
            click.echo(f"  ID: {key_id} (now revoked)")
            click.echo()
            click.secho("New Key Details:", fg="cyan")
            click.echo(f"  Key ID: {new_key.id}")
            click.echo(f"  Description: {new_key.description}")

            # Permission level with styling
            perm_display = new_key.permission_level.value
            if new_key.permission_level == PermissionLevel.READ_ONLY:
                perm_display = click.style(perm_display, fg="blue")
            else:
                perm_display = click.style(perm_display, fg="green")
            click.echo(f"  Permission Level: {perm_display}")

            click.echo(f"  Created: {new_key.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            if new_key.expires_at:
                click.echo(f"  Expires: {new_key.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                click.echo("  Expires: Never")

            click.echo()
            click.secho("New API Key (save this now):", fg="yellow", bold=True)
            click.secho(f"  {new_key.plain_key}", fg="green", bold=True)
            click.echo()
            click.secho("⚠️  Update your applications with the new key!", fg="red")

    except Exception as e:
        click.secho(f"Error rotating API key: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('prune-expired')
@click.option('--soft-delete/--hard-delete', default=True,
              help='Soft delete (deactivate) vs hard delete (remove from database)')
@click.option('--dry-run', '-n', is_flag=True, help='Show what would be pruned without actually doing it')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def prune_expired_keys(ctx, soft_delete, dry_run, confirm):
    """
    Remove or deactivate expired API keys.

    Finds expired API keys and either deactivates (soft delete) or permanently deletes (hard delete)
    them. Supports dry-run and confirmation.

    \b
    Examples:
        \b
        lvdb auth prune-expired
        lvdb auth prune-expired --hard-delete --dry-run

    """
    try:
        from datetime import datetime

        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        # Find expired keys
        all_keys = key_manager.list_keys(active_only=False, include_expired=True)
        expired_keys = [key for key in all_keys if key.is_expired and key.active]

        if not expired_keys:
            click.secho("No expired keys found.", fg="green")
            return

        # Show what will be pruned
        click.secho(f"Found {len(expired_keys)} expired key(s):", fg="yellow")
        click.echo()

        for key in expired_keys:
            expired_days = (datetime.now(UTC) - key.expires_at).days
            click.echo(f"  {key.id}: {key.description or 'No description'} "
                       f"(expired {expired_days} days ago)")

        click.echo()

        if dry_run:
            action = "deactivated" if soft_delete else "deleted"
            click.secho(f"DRY RUN: {len(expired_keys)} keys would be {action}", fg="blue")
            return

        # Confirmation
        action = "deactivate" if soft_delete else "permanently delete"
        if not confirm:
            if not click.confirm(
                    click.style(f"Are you sure you want to {action} these {len(expired_keys)} expired keys?",
                                fg="red")
            ):
                click.echo("Operation cancelled.")
                return

        # Prune the keys
        count = key_manager.prune_expired(soft_delete=soft_delete)

        action_past = "deactivated" if soft_delete else "deleted"
        click.secho(f"✓ {count} expired key(s) {action_past}.", fg="green")

    except Exception as e:
        click.secho(f"Error pruning expired keys: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('key-info')
@click.argument('key_id')
@click.option('--output', '-o', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def show_key_info(ctx, key_id, output):
    """
    Show detailed information about an API key.

    Displays information about a specific API key, including status, creation, expiration, and usage.

    \b
    Examples:
        \b
        lvdb auth key-info <key_id>
        lvdb auth key-info <key_id> --output json
    """
    try:
        from localvectordb_server.keymanager import get_key_manager

        api_key_db_path = ctx.obj.get('api_key_db_path')
        key_manager = get_key_manager(api_key_db_path)

        key_record = key_manager.get_key(key_id)
        if not key_record:
            click.secho(f"Key '{key_id}' not found.", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        if output == 'json':
            import json
            click.echo(json.dumps(key_record.to_dict(), indent=2))
        else:
            click.secho(f"API Key Information: {key_id}", fg="blue", bold=True)
            click.echo()

            # Basic info
            click.secho("Basic Information:", fg="cyan")
            click.echo(f"  ID: {key_record.id}")
            click.echo(f"  Description: {key_record.description or 'None'}")

            # Permission level with styling
            perm_display = key_record.permission_level.value
            if key_record.permission_level == PermissionLevel.READ_ONLY:
                perm_display = click.style(perm_display, fg="blue")
            else:
                perm_display = click.style(perm_display, fg="green")
            click.echo(f"  Permission Level: {perm_display}")

            click.echo(f"  Created by: {key_record.created_by or 'Unknown'}")
            click.echo()

            # Status
            click.secho("Status:", fg="cyan")
            if not key_record.active:
                status = click.style("REVOKED", fg="red")
            elif key_record.is_expired:
                status = click.style("EXPIRED", fg="yellow")
            else:
                status = click.style("ACTIVE", fg="green")
            click.echo(f"  Status: {status}")
            click.echo()

            # Dates
            click.secho("Dates:", fg="cyan")
            click.echo(f"  Created: {key_record.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            if key_record.expires_at:
                click.echo(f"  Expires: {key_record.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                if key_record.days_until_expiry is not None:
                    if key_record.is_expired:
                        days_desc = f"{abs(key_record.days_until_expiry)} days ago"
                        color = "red"
                    elif key_record.days_until_expiry <= 7:
                        days_desc = f"{key_record.days_until_expiry} days"
                        color = "yellow"
                    else:
                        days_desc = f"{key_record.days_until_expiry} days"
                        color = "green"

                    click.echo("  Days until expiry: " +
                               click.style(days_desc, fg=color))
            else:
                click.echo("  Expires: Never")

            if key_record.last_used:
                click.echo(f"  Last used: {key_record.last_used.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                click.echo("  Last used: Never")

    except Exception as e:
        click.secho(f"Error getting key info: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@auth.command('status')
@click.option('--output', '-o', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def auth_status(ctx, output):
    """
    Show the current authentication status.

    Displays the authentication status of the server, including whether API authentication is enabled,
    and statistics about the key database.

    \b
    Examples:
        \b
        lvdb auth status
        lvdb auth status --output json
    """
    try:
        cfg = ctx.obj['config']
        config_path = ctx.obj['config_path']
        api_key_db_path = ctx.obj.get('api_key_db_path')

        # Get basic auth status
        auth_enabled = cfg.server.security.require_api_key
        # config_api_keys = cfg.server.authorized_api_keys

        # Get database key status
        db_status = {"available": False, "stats": {}}
        try:
            from localvectordb_server.keymanager import get_key_manager
            key_manager = get_key_manager(api_key_db_path)
            db_status["available"] = True
            db_status["stats"] = key_manager.get_stats()
        except Exception as e:
            db_status["error"] = str(e)

        if output == 'json':
            import json
            status_data = {
                "config_file": config_path,
                "auth_enabled": auth_enabled,
                "database_keys": db_status
            }
            click.echo(json.dumps(status_data, indent=2))
        else:
            click.secho("Authentication Status", fg="blue", bold=True)
            click.echo("Configuration file: " + click.style(f"{config_path}", fg="blue"))
            click.echo("API Authentication: " +
                       click.style(f"{'Enabled' if auth_enabled else 'Disabled'}",
                                   fg="green" if auth_enabled else "red"))
            click.echo()

            click.secho("Database-managed Keys:", fg="cyan")
            if db_status["available"]:
                stats = db_status["stats"]
                click.echo("  Status: " + click.style("Available", fg="green"))
                click.echo(f"  Total keys: {stats.get('total_keys', 0)}")
                click.echo(f"  Active keys: {stats.get('active_keys', 0)}")
                click.echo(f"  Expired keys: {stats.get('expired_keys', 0)}")
                click.echo(f"  Expiring soon (7 days): {stats.get('expiring_soon', 0)}")
                click.echo(f"  Recently used (24h): {stats.get('recently_used', 0)}")

                if stats.get('expiring_soon', 0) > 0:
                    click.echo()
                    click.secho(f"  ⚠️  {stats['expiring_soon']} key(s) expiring within 7 days", fg="yellow")
            else:
                click.echo("  Status: " + click.style("Not Available", fg="red"))
                if "error" in db_status:
                    click.echo(f"  Error: {db_status['error']}")

    except Exception as e:
        click.secho(f"Error reading auth status: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
