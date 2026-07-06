"""End-to-end test: the `lvdb` command-line interface.

Drives the real CLI executable through a full workflow: create a database,
add documents from files and inline text, search, fetch, find related
documents, and delete — all against a real embedding backend.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_cli.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, ensure_fixtures, make_parser, temp_workdir

DB = "e2ecli"


def lvdb_exe() -> str:
    scripts_dir = Path(sys.executable).parent
    exe = scripts_dir / ("lvdb.exe" if sys.platform == "win32" else "lvdb")
    return str(exe) if exe.exists() else "lvdb"


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)
    fixtures = ensure_fixtures()

    c = Checker(f"e2e_cli ({provider}/{model})")

    with temp_workdir("lvdb-e2e-cli-") as workdir:
        config_path = workdir / "config.toml"
        config_path.write_text(
            f"""
[database]
root_dir = '{(workdir / "dbs").as_posix()}'
chunking_method = "sentences"
chunk_size = 200

[embedding]
provider = "{provider}"
model = "{model}"
""",
            encoding="utf-8",
        )

        def lvdb(*cli_args: str, timeout: float = 300) -> subprocess.CompletedProcess:
            return subprocess.run(
                [lvdb_exe(), "--config", str(config_path), *cli_args],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )

        c.section("create + list databases")
        r = lvdb("create", DB, "--embedding-provider", provider, "--embedding-model", model)
        c.check("lvdb create succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])
        c.check("create reports database name", DB in r.stdout, r.stdout[-300:])
        r = lvdb("list")
        c.check("lvdb list shows database", r.returncode == 0 and DB in r.stdout, r.stdout[-300:])

        c.section("add documents (file + inline text)")
        r = lvdb("db", DB, "add", str(fixtures / "space_exploration.md"), "--id", "space_exploration")
        c.check("add file succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])
        r = lvdb("db", DB, "add", str(fixtures / "french_cooking.md"), "--id", "french_cooking")
        c.check("add second file succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])
        r = lvdb(
            "db",
            DB,
            "add",
            "A proper espresso extracts 36 grams of coffee in thirty seconds at nine bars of pressure.",
            "--id",
            "espresso",
        )
        c.check("add inline text with id succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])

        r = lvdb("db", DB, "list")
        expected_ids = ["space_exploration", "french_cooking", "espresso"]
        c.check("doc list shows all ids", all(s in r.stdout for s in expected_ids), r.stdout[-400:])

        c.section("search")
        r = lvdb("db", DB, "search", "the Apollo moon landings", "--search-type", "hybrid", "--limit", "3", "--json")
        c.check("hybrid search succeeds", r.returncode == 0, r.stderr[-300:])
        c.check("search finds space doc", "space_exploration" in r.stdout, r.stdout[-400:])
        try:
            payload = json.loads(r.stdout)
            c.check("search --json is valid JSON", True)
            first = payload[0] if isinstance(payload, list) else payload.get("results", [{}])[0]
            c.check("top search hit is space doc", "space_exploration" in str(first), str(first)[:200])
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            c.check("search --json is valid JSON", False, f"{exc}: {r.stdout[:200]}")

        r = lvdb("db", DB, "search", "brunoise", "--search-type", "keyword", "--limit", "2")
        c.check("keyword search finds cooking doc", r.returncode == 0 and "french_cooking" in r.stdout, r.stdout[-300:])

        c.section("get + related")
        r = lvdb("db", DB, "get", "space_exploration", "--json", "--metadata")
        c.check("get returns document", r.returncode == 0 and "Apollo" in r.stdout, r.stdout[-300:])
        r = lvdb("db", DB, "related", "french_cooking", "--limit", "2")
        c.check("related succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])

        c.section("stats + info")
        r = lvdb("db", DB, "stats")
        c.check("stats succeeds", r.returncode == 0, r.stderr[-300:])
        r = lvdb("db", DB, "info")
        c.check("info shows embedding model", r.returncode == 0 and model in r.stdout, r.stdout[-300:])

        c.section("delete document + database")
        r = lvdb("db", DB, "delete", "espresso")
        c.check("doc delete succeeds", r.returncode == 0 and "deleted" in r.stdout.lower(), r.stdout[-300:])
        r = lvdb("db", DB, "list")
        c.check("deleted doc gone from list", "espresso" not in r.stdout, r.stdout[-300:])
        r = lvdb("delete", DB, "--confirm")
        c.check("database delete succeeds", r.returncode == 0, r.stderr[-300:] or r.stdout[-300:])
        r = lvdb("list")
        c.check("database gone from list", DB not in r.stdout, r.stdout[-300:])

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
