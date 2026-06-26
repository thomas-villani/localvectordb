"""
CLI commands for SQLite performance tuning management.

This module provides command-line interface for managing SQLite tuning profiles,
pragma settings, and auto-tuning functionality for LocalVectorDB databases.
"""

import json
from pathlib import Path

import click

from localvectordb.database import LocalVectorDB
from localvectordb.sqlite_tuning import AutoTuner, list_profiles


@click.group(name="tuning")
def tuning_group():
    """SQLite performance tuning commands."""
    pass


@tuning_group.command("list")
def list_tuning_profiles():
    """List available SQLite tuning profiles with descriptions."""

    profiles = list_profiles()

    click.echo("\nAvailable SQLite Tuning Profiles:\n")

    for name, description in profiles:
        click.echo(f"  • {click.style(name, fg='cyan', bold=True)}")
        click.echo(f"    {description}\n")

    click.echo(f"Total: {len(profiles)} profiles available")


@tuning_group.command("get")
@click.argument("database")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def get_tuning_config(ctx, database, format):
    """Show current SQLite tuning configuration for a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        # Open database and get tuning config
        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        config = db.get_sqlite_tuning()
        db.close()

        if format == "json":
            click.echo(json.dumps(config, indent=2))
        else:
            # Table format
            click.echo(f"\nSQLite Tuning Configuration for '{database}':\n")

            click.echo(f"Profile: {click.style(config['profile'], fg='green', bold=True)}")

            if config["overrides"]:
                click.echo("\nProfile Overrides:")
                for key, value in config["overrides"].items():
                    click.echo(f"  {key}: {value}")

            click.echo("\nActive Pragma Settings:")
            for key, value in config["pragmas"].items():
                click.echo(f"  {key}: {value}")

    except Exception as e:
        click.echo(f"Error getting tuning configuration: {e}", err=True)


@tuning_group.command("set")
@click.argument("database")
@click.argument("profile")
@click.option(
    "--override", "-o", multiple=True, metavar="KEY=VALUE", help="Pragma override (can be used multiple times)"
)
@click.option("--no-persist", is_flag=True, help="Don't persist settings to database")
@click.option("--dry-run", is_flag=True, help="Show what would be applied without applying")
@click.pass_context
def set_tuning_profile(ctx, database, profile, override, no_persist, dry_run):
    """Apply SQLite tuning profile to a database."""

    try:
        # Parse overrides
        overrides = {}
        for override_str in override:
            if "=" not in override_str:
                click.echo(f"Error: Invalid override format '{override_str}'. Use KEY=VALUE", err=True)
                return

            key, value = override_str.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse as number, boolean, or keep as string
            if value.lower() in ("true", "false"):
                overrides[key] = value.lower() == "true"
            elif value.isdigit():
                overrides[key] = int(value)
            elif value.lstrip("-").isdigit():
                overrides[key] = int(value)
            else:
                overrides[key] = value

        if dry_run:
            click.echo(f"\nDry Run - Would apply to '{database}':")
            click.echo(f"Profile: {profile}")
            if overrides:
                click.echo("Overrides:")
                for key, value in overrides.items():
                    click.echo(f"  {key}: {value}")
            click.echo(f"Persist: {not no_persist}")
            return

        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        # Apply tuning
        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        db.set_sqlite_tuning(profile, overrides, persist=not no_persist)

        # Show updated config
        _new_config = db.get_sqlite_tuning()
        db.close()

        click.echo(f"\nApplied SQLite tuning profile '{click.style(profile, fg='green')}' to '{database}'")

        if overrides:
            click.echo("\nApplied overrides:")
            for key, value in overrides.items():
                click.echo(f"  {key}: {value}")

        if not no_persist:
            click.echo("\nSettings persisted to database")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        click.echo(f"Error applying tuning profile: {e}", err=True)


@tuning_group.command("set-pragma")
@click.argument("database")
@click.argument("pragma_key")
@click.argument("pragma_value")
@click.option("--no-persist", is_flag=True, help="Don't persist settings to database")
@click.pass_context
def set_pragma_override(ctx, database, pragma_key, pragma_value, no_persist):
    """Set a specific pragma override for a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        # Parse pragma value
        if pragma_value.lower() in ("true", "false"):
            parsed_value = pragma_value.lower() == "true"
        elif pragma_value.isdigit():
            parsed_value = int(pragma_value)
        elif pragma_value.lstrip("-").isdigit():
            parsed_value = int(pragma_value)
        else:
            parsed_value = pragma_value

        # Get current config and update
        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        current_config = db.get_sqlite_tuning()

        overrides = dict(current_config["overrides"])
        overrides[pragma_key] = parsed_value

        db.set_sqlite_tuning(current_config["profile"], overrides, persist=not no_persist)
        db.close()

        click.echo(
            f" Set pragma '{click.style(pragma_key, fg='cyan')}'"
            f" = '{click.style(str(parsed_value), fg='yellow')}' for '{database}'"
        )

        if not no_persist:
            click.echo(" Setting persisted to database")

    except Exception as e:
        click.echo(f"Error setting pragma: {e}", err=True)


@tuning_group.command("auto")
@click.argument("database")
@click.option("--interactive", "-i", is_flag=True, help="Run interactive workload interview")
@click.option(
    "--workload-type",
    type=click.Choice(["read_heavy", "write_heavy", "balanced", "batch_ingest", "real_time"]),
    help="Workload type (skips interview)",
)
@click.option("--memory-constraint", type=click.Choice(["generous", "moderate", "limited"]), help="Memory availability")
@click.option(
    "--durability", type=click.Choice(["critical", "high", "normal", "low"]), help="Data durability importance"
)
@click.option("--apply", is_flag=True, help="Apply recommended settings immediately")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def auto_tune_database(ctx, database, interactive, workload_type, memory_constraint, durability, apply, format):
    """Get auto-tuning recommendations for a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        # Prepare workload info
        workload = None
        if not interactive and (workload_type or memory_constraint or durability):
            workload = {
                "workload_type": workload_type or "balanced",
                "memory_constraint": memory_constraint or "moderate",
                "durability_level": durability or "normal",
                "concurrent_users": 5,
                "document_size": "medium",
            }

        # Get recommendations
        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        recommendation = db.auto_tune(workload=workload, interactive=interactive, apply=apply)
        db.close()

        if format == "json":
            click.echo(json.dumps(recommendation, indent=2))
        else:
            # Table format
            click.echo(f"\n Auto-Tuning Recommendation for '{database}':\n")

            click.echo(f"Recommended Profile: {click.style(recommendation['profile_name'], fg='green', bold=True)}")

            if recommendation["pragma_overrides"]:
                click.echo("\nRecommended Overrides:")
                for key, value in recommendation["pragma_overrides"].items():
                    click.echo(f"  {key}: {value}")

            click.echo(f"\nEstimated Memory Usage: {recommendation['estimated_memory_mb']} MB")

            click.echo("\nReasoning:")
            for reason in recommendation["reasoning"]:
                click.echo(f"  • {reason}")

            if recommendation["applied"]:
                click.echo(f"\n {click.style('Settings have been applied!', fg='green', bold=True)}")
            else:
                click.echo("\n Run with --apply to apply these settings")

    except Exception as e:
        click.echo(f"Error running auto-tuner: {e}", err=True)


@click.group(name="maintenance")
def maintenance_group():
    """Database maintenance commands."""
    pass


@maintenance_group.command("checkpoint")
@click.argument("database")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["PASSIVE", "FULL", "RESTART", "TRUNCATE"]),
    default="PASSIVE",
    help="Checkpoint mode",
)
@click.pass_context
def checkpoint_database(ctx, database, mode):
    """Run SQLite WAL checkpoint on a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        db.sqlite_checkpoint(mode)
        db.close()

        click.echo(f" SQLite WAL checkpoint completed for '{database}' with mode '{mode}'")

    except Exception as e:
        click.echo(f"Error running checkpoint: {e}", err=True)


@maintenance_group.command("optimize")
@click.argument("database")
@click.pass_context
def optimize_database(ctx, database):
    """Run SQLite PRAGMA optimize on a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)
        db.sqlite_optimize()
        db.close()

        click.echo(f" SQLite PRAGMA optimize completed for '{database}'")

    except Exception as e:
        click.echo(f"Error running optimize: {e}", err=True)


@maintenance_group.command("vacuum")
@click.argument("database")
@click.option("--incremental", "-i", is_flag=True, help="Run incremental vacuum instead")
@click.option("--pages", type=int, default=2000, help="Pages to reclaim (incremental only)")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def vacuum_database(ctx, database, incremental, pages, confirm):
    """Run SQLite VACUUM on a database."""

    try:
        db_folder = ctx.obj.get("db_folder")
        if not db_folder:
            click.echo("Error: Database folder not configured", err=True)
            return

        db_path = Path(db_folder) / f"{database}.sqlite"
        if not db_path.exists():
            click.echo(f"Error: Database '{database}' not found", err=True)
            return

        if not incremental and not confirm:
            click.echo("  WARNING: VACUUM requires exclusive database access and may take significant time.")
            if not click.confirm(f"Proceed with VACUUM on '{database}'?"):
                click.echo("Operation cancelled")
                return

        db = LocalVectorDB(name=database, base_path=db_folder, create_if_not_exists=False)

        if incremental:
            db.sqlite_incremental_vacuum(pages)
            click.echo(f" SQLite incremental vacuum completed for '{database}' ({pages} pages)")
        else:
            db.sqlite_vacuum()
            click.echo(f" SQLite VACUUM completed for '{database}'")

        db.close()

    except Exception as e:
        click.echo(f"Error running vacuum: {e}", err=True)


@maintenance_group.command("analyze-system")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
def analyze_system_resources(format):
    """Analyze system resources for tuning recommendations."""

    try:
        system_info = AutoTuner.analyze_system()

        if format == "json":
            resources = {
                "total_ram_mb": system_info.total_ram_mb,
                "available_ram_mb": system_info.available_ram_mb,
                "cpu_cores": system_info.cpu_cores,
                "disk_type": system_info.disk_type,
                "disk_free_gb": system_info.disk_free_gb,
                "os_type": system_info.os_type,
            }
            click.echo(json.dumps(resources, indent=2))
        else:
            click.echo("\n  System Resource Analysis:\n")

            click.echo(f"Total RAM: {click.style(f'{system_info.total_ram_mb:,} MB', fg='cyan')}")
            click.echo(f"Available RAM: {click.style(f'{system_info.available_ram_mb:,} MB', fg='green')}")
            click.echo(f"CPU Cores: {click.style(str(system_info.cpu_cores), fg='cyan')}")
            click.echo(f"Disk Type: {click.style(system_info.disk_type, fg='yellow')}")
            click.echo(f"Free Disk Space: {click.style(f'{system_info.disk_free_gb:.1f} GB', fg='cyan')}")
            click.echo(f"Operating System: {click.style(system_info.os_type, fg='cyan')}")

            # Provide basic recommendations
            click.echo("\n Quick Recommendations:")

            if system_info.available_ram_mb >= 8192:
                click.echo("  • System has abundant RAM - consider 'read_optimized' profile")
            elif system_info.available_ram_mb <= 2048:
                click.echo("  • Limited RAM detected - consider 'memory_saver' profile")

            if system_info.disk_type == "SSD":
                click.echo("  • SSD detected - can use larger mmap_size and cache_size")
            elif system_info.disk_type == "HDD":
                click.echo("  • HDD detected - reduce WAL checkpoints and mmap usage")

    except Exception as e:
        click.echo(f"Error analyzing system: {e}", err=True)


# Export command groups
__all__ = ["tuning_group", "maintenance_group"]
