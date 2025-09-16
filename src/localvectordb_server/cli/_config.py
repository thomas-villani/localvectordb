# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_config.py
import json
import os
from dataclasses import asdict
from pathlib import Path

import click

from localvectordb_server.cli._utils import (
    DEFAULT_CONFIG_FILE,
    _format_value_for_display,
    get_nested_value,
    set_nested_value,
)


@click.group('config', invoke_without_command=True)
@click.pass_context
def config_group(ctx):
    """
    View or modify the server configuration.

    Provides subcommands to show, get, set, or initialize the server configuration file.
    Supports TOML and JSON formats and allows dot notation for accessing nested config values.

    \b
    Examples:
        \b
        lvdb config show
        lvdb config get server.host
        lvdb config set database.chunk_size 1000
        lvdb config init --output ./.lvdb-config.toml

    """

    # If no subcommand was invoked, display current config
    if ctx.invoked_subcommand is None:
        ctx.invoke(show_config)


@config_group.command('show')
@click.option('--format', '-f', type=click.Choice(['toml', 'json', ]), default=None,
              help='Output format (defaults to format of config file)')
@click.option('--toml', 'format', flag_value='toml', help="Output in `toml` format")
@click.option('--json', 'format', flag_value='json', help="Output in `json` format")
@click.option('--section', '-s', type=click.Choice(['database', 'embedding', 'server']), default=None,
              help='Only show specific section')
@click.pass_context
def show_config(ctx, format, section):
    """
    Display current configuration.

    Shows the current server configuration in TOML or JSON format. You can filter by section
    (database, embedding, server) and choose the output format.

    \b
    Examples:
        \b
        lvdb config show
        lvdb config show --format json
        lvdb config show --section database
    """
    cfg = ctx.obj['config']
    config_path = ctx.obj['config_path']

    # Determine output format based on file extension if not specified
    if not format:
        suffix = Path(config_path).suffix.lower()
        if suffix == '.toml':
            format = 'toml'
        elif suffix == '.json':
            format = 'json'
        else:
            format = 'toml'  # Default to TOML for unknown formats

    # Generate configuration string
    if format == 'toml':
        config_str = cfg.generate_toml()
    else:
        # For other formats, convert to dict and handle appropriately
        config_dict = {
            'database': asdict(cfg.database),
            'embedding': asdict(cfg.embedding),
            'server': asdict(cfg.server),
        }

        if format == 'json':
            config_str = json.dumps(config_dict, indent=2)
        else:
            config_str = cfg.generate_toml()  # Fallback to TOML

    # Filter by section if requested
    if section:
        if format == 'toml':
            section_header = f"[{section}]"
            lines = config_str.split('\n')
            section_start = -1
            section_end = len(lines)

            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    section_start = i
                elif section_start >= 0 and line.strip().startswith('[') and i > section_start:
                    section_end = i
                    break

            if section_start >= 0:
                config_str = '\n'.join(lines[section_start:section_end])
            else:
                click.secho(f"Section '{section}' not found in configuration", fg="bright_red")
                return
        elif format == "json":
            json_obj = json.loads(config_str)
            if section in json_obj:
                json_obj = {section: json_obj[section]}
                config_str = json.dumps(json_obj, indent=2)
            else:
                click.secho(f"Section '{section}' not found in configuration", fg="bright_red")
                return

    # Display the configuration
    title = f"Configuration from: {config_path}"
    click.secho(title, fg="cyan")
    click.secho("=" * len(title), fg="cyan")
    click.echo(config_str)



@config_group.command('get')
@click.argument('key')
@click.option('--format', '-f', type=click.Choice(['raw', 'json', 'pretty']),
              default='pretty', help='Output format')
@click.pass_context
def get_config_value(ctx, key, format):
    """
    Get a configuration value for KEY.

    KEY is a variable in dot notation including the section: section.key

    \b
    Examples:
        \b
        lvdb config get server.host
        lvdb config get server.security.require_api_key
        lvdb config get database.chunk_size
        lvdb config get embedding.model
    """
    try:
        cfg = ctx.obj['config']
        value = get_nested_value(cfg, key)

        if format == 'raw':
            # Raw value as string
            if isinstance(value, bool):
                click.echo("true" if value else "false")
            elif value is None:
                click.echo("null")
            else:
                click.echo(str(value))
        elif format == 'json':
            # JSON format
            from localvectordb import MetadataField
            if isinstance(value, MetadataField):
                json_value = {
                    'type': value.type.value if hasattr(value.type, 'value') else str(value.type),
                    'indexed': value.indexed,
                    'required': value.required,
                    'default_value': value.default_value
                }
            else:
                json_value = value
            click.echo(json.dumps(json_value))
        else:
            # Pretty format (default)
            click.secho(f"Configuration: {key}", fg="cyan")
            click.secho("=" * (len(key) + 15), fg="cyan")
            click.echo(_format_value_for_display(value))

    except ValueError as e:
        click.secho(f"Error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(1)


@config_group.command('set')
@click.argument('key')
@click.argument('value')
@click.option('--dry-run', '-n', is_flag=True,
              help='Show what would be changed without saving')
@click.option('--force', '-f', is_flag=True,
              help='Skip confirmation prompt')
@click.pass_context
def set_config_value(ctx, key, value, dry_run, force):
    """
    Set a configuration value using dot notation.

    Allows updating a configuration value in the config file. Supports dry-run and confirmation prompts.

    \b
    Examples:
        \b
        lvdb config set server.port 8080
        lvdb config set server.security.require_api_key true
        lvdb config set database.chunk_size 1000
        lvdb config set embedding.model "all-minilm-l6-v2"
        lvdb config set server.security.cors_allowed_origins '["http://localhost:3000"]'

    """
    try:
        cfg = ctx.obj['config']
        config_path = ctx.obj['config_path']

        # Get current value for comparison
        try:
            old_value = get_nested_value(cfg, key)
        except ValueError:
            old_value = "<not set>"

        # Show what will change
        click.secho("Configuration Change:", fg="cyan")
        click.secho("=" * 21, fg="cyan")
        click.echo(f"Key: {key}")
        click.echo(f"Old value: {_format_value_for_display(old_value)}")
        click.echo(f"New value: {value}")

        if dry_run:
            click.secho("\n[DRY RUN] No changes made.", fg="yellow")
            return

        # Confirmation unless forced
        if not force:
            click.echo()
            if not click.confirm("Apply this change?"):
                click.secho("Cancelled.", fg="yellow")
                return

        # Apply the change
        set_nested_value(cfg, key, value)

        # Validate and save
        cfg.validate()
        config_text = cfg.generate_toml()
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_text)

        click.secho(f"\n✓ Configuration updated and saved to {config_path}", fg="green")

        # Show the actual applied value (in case of type conversion)
        try:
            actual_value = get_nested_value(cfg, key)
            if str(actual_value) != value:
                click.secho(f"Applied value: {_format_value_for_display(actual_value)}", fg="blue")
        except Exception:
            pass

    except ValueError as e:
        click.secho(f"Error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(1)


@config_group.command('init')
@click.option('--format', '-f', type=click.Choice(['toml', 'json']), default='toml',
              help='Configuration file format (default: toml)')
@click.option('--output', type=click.Path(resolve_path=True), help='Path to create config file')
@click.option('--schema',
              type=click.Choice(['files', 'documents', 'research_papers', 'code_repository', 'customer_support']),
              help='Apply a predefined metadata schema')
# Multi-worker options
@click.option('--multi-worker', is_flag=True, help='Configure for multi-worker deployment with file-based registry')
@click.option('--redis-registry', help='Configure Redis-based registry (provide Redis URL)')
# Cache options
@click.option('--enable-cache', is_flag=True, help='Enable response caching')
@click.option('--cache-type', type=click.Choice(['file', 'redis', 'memory', 'memcached']),
              help='Cache backend type')
@click.option('--cache-redis-url', help='Redis URL for cache (e.g. redis://localhost:6379/0)')
# Rate limiting options
@click.option('--enable-rate-limiting', is_flag=True, help='Enable API rate limiting')
@click.option('--rate-limit', help='Rate limit (e.g. "100 per minute", "1000 per hour")')
# CORS options
@click.option('--enable-cors', is_flag=True, help='Enable CORS for web applications')
@click.option('--cors-origins', help='Allowed CORS origins (comma-separated or "all" for "*")')
# Security options
@click.option('--enable-auth', is_flag=True, help='Enable API key authentication')
# Server options
@click.option('--host', help='Server host (default: 127.0.0.1)')
@click.option('--port', type=int, help='Server port (default: 5000)')
@click.option('--enable-file-upload', is_flag=True, help='Enable file upload routes on the server')
@click.option('--max-request-size-mb', type=int, help='Set the maximum request size in MB', default=100)
# Interactive mode
@click.option('--interactive', '-I', is_flag=True, help='Interactive guided configuration')
@click.pass_context
def init_config(
        ctx, format, output, schema, multi_worker, redis_registry,
        enable_cache, cache_type, cache_redis_url, enable_rate_limiting,
        rate_limit, enable_cors, cors_origins, enable_auth, host, port, enable_file_upload, max_request_size_mb,
        interactive
        ):
    """
    Initialize a new configuration file with default settings.

    Creates a new server configuration file with options for multi-worker deployment,
    caching, rate limiting, CORS, and authentication. Use --interactive for a guided setup.

    \b
    Examples:
        \b
        # Basic configuration
        lvdb config init --output ./my-config.toml

        # Production setup with Redis
        lvdb config init --redis-registry redis://localhost:6379/1 \\
                        --enable-cache --cache-type redis --cache-redis-url redis://localhost:6379/0 \\
                        --enable-rate-limiting --rate-limit "1000 per hour" \\
                        --enable-cors --cors-origins "https://myapp.com" \\
                        --enable-auth

        # Interactive guided setup
        lvdb config init --interactive

        # Quick multi-worker setup
        lvdb config init --multi-worker --enable-cache --cache-type file
    """

    if interactive:
        return _interactive_config_init(format, output)

    if not output:
        output = f"./{DEFAULT_CONFIG_FILE}.{format}"

    if os.path.exists(output):
        click.echo(f"Configuration file `{output}` exists! Overwrite (Y/n)?")
        char = click.getchar()
        if char.lower() != "y":
            return 0

    from localvectordb_server.config import Config

    # Create default configuration
    config = Config()

    # Apply common schema if requested
    if schema:
        config.apply_common_schema(schema)
        click.secho(f"Applied metadata schema: {schema}", fg="green")

    # Configure server basics
    if host:
        config.server.host = host
    if port:
        config.server.port = port

    # Configure multi-worker deployment
    if multi_worker and redis_registry:
        click.secho("Warning: Both --multi-worker and --redis-registry specified. Using Redis registry.", fg="yellow")

    if redis_registry:
        _configure_redis_registry(config, redis_registry)
    elif multi_worker:
        _configure_file_registry(config)

    # Configure caching
    if enable_cache:
        _configure_cache(config, cache_type, cache_redis_url, redis_registry)

    # Configure rate limiting
    if enable_rate_limiting:
        _configure_rate_limiting(config, rate_limit)

    # Configure CORS
    if enable_cors:
        _configure_cors(config, cors_origins)

    # Configure authentication
    if enable_auth:
        _configure_auth(config)

    if enable_file_upload:
        config.server.file_upload_enabled = True
        if max_request_size_mb <= 0:
            click.secho("Error: --max-request-size-mb must be a positive integer", err=True, fg="bright_red")
            raise click.Abort()

        config.server.max_request_size = max_request_size_mb * 1024 * 1024

    # Generate and save configuration
    _save_config(config, output, format)

    # Print summary
    _print_config_summary(config, output, multi_worker or redis_registry)


def _interactive_config_init(format, output):
    """Interactive guided configuration setup"""

    click.secho("LocalVectorDB Server Configuration Wizard", fg="blue", bold=True)
    click.echo("This wizard will guide you through setting up your LocalVectorDB server configuration.\n")

    from localvectordb_server.config import Config
    config = Config()

    # Basic settings
    click.secho("1. Basic Server Settings", fg="cyan", bold=True)

    if not output:
        default_output = f"./{DEFAULT_CONFIG_FILE}.{format}"
        output = click.prompt("Configuration file path", default=default_output)

    if os.path.exists(output):
        if not click.confirm(f"Configuration file '{output}' exists. Overwrite?"):
            click.echo("Configuration cancelled.")
            return

    # Server host and port
    host = click.prompt("Server host", default="127.0.0.1")
    port = click.prompt("Server port", type=int, default=5000)
    config.server.host = host
    config.server.port = port

    # Database settings
    click.echo()
    click.secho("2. Database Settings", fg="cyan", bold=True)

    db_path = click.prompt("Database directory", default="./.lvdb")
    config.database.root_dir = db_path

    # Metadata schema
    if click.confirm("Apply a predefined metadata schema?", default=False):
        schema_choices = ['files', 'documents', 'research_papers', 'code_repository', 'customer_support']
        click.echo("Available schemas:")
        for i, schema in enumerate(schema_choices, 1):
            click.echo(f"  {i}. {schema}")

        while True:
            try:
                choice = click.prompt("Select schema (1-5)", type=int)
                if 1 <= choice <= 5:
                    schema = schema_choices[choice - 1]
                    config.apply_common_schema(schema)
                    click.secho(f"Applied schema: {schema}", fg="green")
                    break
                else:
                    click.echo("Please enter a number between 1 and 5")
            except click.Abort:
                break

    # Multi-worker deployment
    click.echo()
    click.secho("3. Multi-Worker Deployment", fg="cyan", bold=True)

    deployment_type = click.prompt(
        "Deployment type",
        type=click.Choice(['single', 'multi-file', 'multi-redis'], case_sensitive=False),
        default='single',
        help_text="single=single worker, multi-file=multiple workers on one machine, multi-redis=distributed workers"
    )

    if deployment_type == 'multi-file':
        _configure_file_registry(config)
    elif deployment_type == 'multi-redis':
        redis_url = click.prompt("Redis URL for registry", default="redis://localhost:6379/1")
        _configure_redis_registry(config, redis_url)

    # Caching
    click.echo()
    click.secho("4. Response Caching", fg="cyan", bold=True)

    if click.confirm("Enable response caching?", default=False):
        if deployment_type == 'multi-redis':
            # Suggest Redis cache for consistency
            if click.confirm("Use Redis for caching (recommended for distributed setup)?", default=True):
                cache_redis_url = click.prompt("Redis URL for cache", default="redis://localhost:6379/0")
                config.server.cache_enabled = True
                config.server.cache_type = "RedisCache"
                config.server.cache_settings = _parse_redis_url(cache_redis_url)
            else:
                cache_type = click.prompt("Cache type", type=click.Choice(['file', 'memcached']), default='file')
                _configure_cache(config, cache_type, None, None)
        else:
            cache_type = click.prompt(
                "Cache type",
                type=click.Choice(['file', 'redis', 'memcached']),
                default='file'
            )

            if cache_type == 'redis':
                cache_redis_url = click.prompt("Redis URL for cache", default="redis://localhost:6379/0")
                _configure_cache(config, cache_type, cache_redis_url, None)
            else:
                _configure_cache(config, cache_type, None, None)

    # Rate limiting
    click.echo()
    click.secho("5. Rate Limiting", fg="cyan", bold=True)

    if click.confirm("Enable API rate limiting?", default=False):
        rate_limit = click.prompt("Rate limit", default="100 per minute")
        _configure_rate_limiting(config, rate_limit)

    # CORS
    click.echo()
    click.secho("6. CORS (Cross-Origin Resource Sharing)", fg="cyan", bold=True)

    if click.confirm("Enable CORS for web applications?", default=True):
        cors_type = click.prompt(
            "CORS configuration",
            type=click.Choice(['all', 'localhost', 'custom'], case_sensitive=False),
            default='localhost',
            help_text="all=allow all origins (*), localhost=allow localhost only, custom=specify origins"
        )

        if cors_type == 'all':
            _configure_cors(config, "all")
        elif cors_type == 'localhost':
            _configure_cors(config, "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000")
        else:  # custom
            origins = click.prompt("Allowed origins (comma-separated)")
            _configure_cors(config, origins)

    # Authentication
    click.echo()
    click.secho("7. API Authentication", fg="cyan", bold=True)

    if click.confirm("Enable API key authentication?", default=False):
        _configure_auth(config)

    # Embedding settings
    click.echo()
    click.secho("8. Default Embedding Provider", fg="cyan", bold=True)

    # TODO: this should be dynamic from the registry
    provider = click.prompt(
        "Embedding provider",
        type=click.Choice(['ollama', 'openai'], case_sensitive=False),
        default='ollama'
    )
    config.embedding.provider = provider

    if provider == 'ollama':
        model = click.prompt("Ollama model", default="nomic-embed-text")
        config.embedding.model = model
    elif provider == 'openai':
        model = click.prompt("OpenAI model", default="text-embedding-3-small")
        config.embedding.model = model

        if click.confirm("Set OpenAI API key now?", default=False):
            api_key = click.prompt("OpenAI API key", hide_input=True)
            config.embedding.api_key = api_key
        else:
            click.echo("Note: Set OPENAI_API_KEY environment variable before starting the server")

    click.echo()
    click.secho("9. File upload", fg="cyan", bold=True)

    enable_upload = click.confirm("Enable file upload?")
    if enable_upload:
        config.server.file_upload_enabled = True
        max_file_size_mb = click.prompt("Maximum file size (in MB)", type=int, default=100)
        config.server.max_request_size = max_file_size_mb * 1024 * 1024

    # Generate and save configuration
    click.echo()
    click.secho("10. Generating Configuration", fg="cyan", bold=True)

    _save_config(config, output, format)

    # Print summary
    click.echo()
    _print_config_summary(config, output, deployment_type != 'single')

    # Post-setup recommendations
    _print_setup_recommendations(config, deployment_type, output)


def _configure_redis_registry(config, redis_url):
    """Configure Redis-based database registry"""
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(redis_url)
        if parsed.scheme != 'redis':
            raise ValueError("Redis URL must start with 'redis://'")

        config.server.db_registry_type = "RedisCache"
        config.server.db_registry_settings = _parse_redis_url(redis_url)
        click.secho(f"Configured Redis registry: {redis_url}", fg="green")

    except Exception as e:
        click.secho(f"Invalid Redis URL: {e}", fg="red")
        raise click.Abort()


def _configure_file_registry(config):
    """Configure file-based database registry"""
    config.server.db_registry_type = "FileSystemCache"
    config.server.db_registry_settings = {
        "cache_dir": "./.lvdb/registry_cache"
    }
    click.secho("Configured file-based registry for multi-worker deployment", fg="green")


def _configure_cache(config, cache_type, cache_redis_url, redis_registry_url):
    """Configure response caching"""
    config.server.cache_enabled = True

    if cache_type == 'redis':
        config.server.cache_type = "RedisCache"
        if cache_redis_url:
            config.server.cache_settings = _parse_redis_url(cache_redis_url)
        elif redis_registry_url:
            # Use same Redis but different DB
            registry_settings = _parse_redis_url(redis_registry_url)
            cache_settings = registry_settings.copy()
            cache_settings['db'] = cache_settings.get('db', 1) - 1  # Use DB 0 for cache
            config.server.cache_settings = cache_settings
        else:
            config.server.cache_settings = {"host": "localhost", "port": 6379, "db": 0}
        click.secho("Configured Redis cache", fg="green")
    elif cache_type == 'memory':
        config.server.cache_type = "SimpleCache"
        click.secho("Configured memory-based cache - NOT RECOMMENDED FOR MULTIPLE WORKERS", fg="green")
    elif cache_type == 'file':
        config.server.cache_type = "FileSystemCache"
        config.server.cache_settings = {"cache_dir": "./.lvdb/cache"}
        click.secho("Configured file-based cache", fg="green")

    elif cache_type == 'memcached':
        config.server.cache_type = "MemcachedCache"
        config.server.cache_settings = {"servers": ["127.0.0.1:11211"]}
        click.secho("Configured Memcached cache", fg="green")
    else:
        # Default to file cache
        config.server.cache_type = "FileSystemCache"
        config.server.cache_settings = {"cache_dir": "./.lvdb/cache"}
        click.secho("Configured file-based cache", fg="green")


def _configure_rate_limiting(config, rate_limit):
    """Configure API rate limiting"""
    config.server.enable_rate_limiting = True
    if rate_limit:
        config.server.rate_limit = rate_limit
    click.secho(f"Configured rate limiting: {config.server.rate_limit}", fg="green")


def _configure_cors(config, cors_origins):
    """Configure CORS settings"""
    config.server.cors_enabled = True

    if cors_origins:
        if cors_origins.lower() in ['all', '*']:
            config.server.cors_allowed_origins = "*"
            click.secho("Configured CORS: Allow all origins (*)", fg="yellow")
            click.secho("Warning: Allowing all origins may be insecure in production", fg="red")
        else:
            origins = [origin.strip() for origin in cors_origins.split(',')]
            config.server.cors_allowed_origins = origins
            click.secho(f"Configured CORS origins: {origins}", fg="green")
    else:
        # Default to localhost for development
        config.server.cors_allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
        click.secho("Configured CORS: localhost development origins: "
                    f"{', '.join(config.server.cors_allowed_origins)}", fg="green")


def _configure_auth(config):
    """Configure API key authentication"""
    config.server.require_api_key = True
    click.secho("Configured API key authentication", fg="green")
    click.echo("Note: Use 'lvdb auth create-key' to generate API keys after starting the server")


def _parse_redis_url(redis_url):
    """Parse Redis URL into connection settings"""
    import urllib.parse

    parsed = urllib.parse.urlparse(redis_url)
    settings = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
    }

    if parsed.path and len(parsed.path) > 1:
        try:
            settings["db"] = int(parsed.path[1:])  # Remove leading '/'
        except ValueError:
            pass

    if parsed.password:
        settings["password"] = parsed.password

    if parsed.username:
        settings["username"] = parsed.username

    return settings


def _save_config(config, output, format):
    """Save configuration to file"""
    if format == "toml":
        config_text = config.generate_toml()
    elif format == "json":
        config_text = json.dumps({
            "database": asdict(config.database),
            "server": asdict(config.server),
            "embedding": asdict(config.embedding)
        }, indent=2)

    with open(output, "w", encoding="utf-8") as f:
        f.write(config_text)

    click.secho(f"Configuration file `{output}` created!", fg="green", bold=True)


def _print_config_summary(config, output, is_multi_worker):
    """Print configuration summary"""
    click.echo()
    click.secho("Configuration Summary:", fg="blue", bold=True)
    click.echo(f"Configuration file: {output}")
    click.echo(f"Server: {config.server.host}:{config.server.port}")
    click.echo(f"Database path: {config.database.root_dir}")
    click.echo(f"Embedding: {config.embedding.provider}/{config.embedding.model}")

    if is_multi_worker:
        click.echo(f"Registry: {config.server.db_registry_type}")

    if config.server.cache_enabled:
        click.echo(f"Cache: {config.server.cache_type}")

    if config.server.enable_rate_limiting:
        click.echo(f"Rate limit: {config.server.rate_limit}")

    if config.server.cors_enabled:
        origins = config.server.cors_allowed_origins
        if origins == "*":
            click.echo("CORS: All origins")
        else:
            click.echo(f"CORS: {len(origins) if isinstance(origins, list) else 1} origins")

    if config.server.require_api_key:
        click.echo("Authentication: API keys required")

    click.echo(f"File upload: {'Enabled' if config.server.file_upload_enabled else 'Disabled'}")


def _print_setup_recommendations(config, deployment_type, output):
    """Print post-setup recommendations"""
    click.echo()
    click.secho("Next Steps:", fg="green", bold=True)

    click.echo("1. Start the server:")
    click.echo(f"   $ lvdb serve --config {output}")

    if deployment_type != 'single':
        click.echo()
        click.echo("2. Multi-worker deployment notes:")
        if deployment_type == 'multi-file':
            click.echo("   • Ensure all workers can access the same database directory")
            click.echo("   • File-based registry requires shared filesystem")
        elif deployment_type == 'multi-redis':
            click.echo("   • Ensure Redis server is running and accessible")
            click.echo("   • Start multiple worker processes with the same config")

    if config.server.cache_enabled and config.server.cache_type == "RedisCache":
        click.echo("   • Ensure Redis server is running for caching")

    if config.server.require_api_key:
        click.echo()
        click.echo("3. Create API keys:")
        click.echo("   $ lvdb auth create-key --description 'My App'")

    if config.embedding.provider == "openai" and not config.embedding.api_key:
        click.echo()
        click.echo("4. Set OpenAI API key:")
        click.echo("   $ export OPENAI_API_KEY=your_api_key_here")

    click.echo()
    click.echo("For more configuration options:")
    click.echo("   $ lvdb config show")
    click.echo("   $ lvdb config set <key> <value>")
