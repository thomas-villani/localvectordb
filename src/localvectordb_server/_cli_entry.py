"""
Console entry point for ``lvdb``.

The CLI (and its dependencies: click, fastapi, uvicorn, ...) ships in the
optional ``server`` extra, but the ``lvdb`` script is installed unconditionally
with the base package. This shim exists so a base install greets the user with
an actionable install hint instead of a raw ModuleNotFoundError traceback.
"""

import sys


def main() -> None:
    try:
        from localvectordb_server.cli import cli
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        print(
            f"lvdb requires the server extras (missing dependency: {missing or exc}).\n"
            'Install them with:  pip install "localvectordb[server]"',
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
