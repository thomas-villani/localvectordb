"""
Console entry point for ``lvdb``.

The ``lvdb`` script is installed unconditionally with the base package, but the
CLI's dependencies (click, tomli-w, bcrypt) ship in the optional ``cli`` extra —
the lighter install that runs everything except ``lvdb serve``. The full HTTP
server (fastapi, uvicorn, ...) is the ``server`` extra, which includes ``cli``.
This shim exists so a base install greets the user with an actionable install
hint instead of a raw ModuleNotFoundError traceback.
"""

import sys


def main() -> None:
    try:
        from localvectordb_server.cli import cli
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        print(
            f"lvdb requires the CLI extra (missing dependency: {missing or exc}).\n"
            'Install it with:      pip install "localvectordb[cli]"\n'
            'For the HTTP server:  pip install "localvectordb[server]"',
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
