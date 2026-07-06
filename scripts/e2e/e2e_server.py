"""End-to-end test: HTTP server + RemoteVectorDB client + raw REST API.

Boots a real `lvdb serve` subprocess with API-key auth and file upload
enabled, then exercises the RemoteVectorDB client (create/upsert/query/
filter/update/compare/stream/delete), the raw REST endpoints (multipart file
upload, database listing), and permission enforcement (read-only key must not
write, missing key must be rejected).

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/e2e_server.py [--provider ollama|sentence_transformers]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import Checker, detect_provider, ensure_fixtures, free_port, make_parser, temp_workdir

DB_NAME = "e2e_remote"

DOCS = {
    "volcano": (
        "Shield volcanoes like Mauna Loa erupt fluid basaltic lava that travels "
        "long distances, building broad gentle slopes over millennia.",
        {"topic": "geology"},
    ),
    "espresso": (
        "A proper espresso shot extracts about 36 grams of coffee from 18 grams "
        "of finely ground beans in roughly thirty seconds at nine bars of pressure.",
        {"topic": "coffee"},
    ),
    "sailing": (
        "Tacking moves a sailboat upwind by zigzagging across the wind, while "
        "jibing turns the stern through the wind when sailing downwind.",
        {"topic": "sailing"},
    ),
}


def lvdb_exe() -> str:
    scripts_dir = Path(sys.executable).parent
    exe = scripts_dir / ("lvdb.exe" if sys.platform == "win32" else "lvdb")
    return str(exe) if exe.exists() else "lvdb"


def create_key(config_path: Path, permission: str) -> str:
    out = subprocess.run(
        [
            lvdb_exe(),
            "--config",
            str(config_path),
            "auth",
            "create-key",
            "-d",
            f"e2e {permission}",
            "-p",
            permission,
            "--output",
            "key-only",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    key = out.stdout.strip().splitlines()[-1].strip()
    if not key:
        raise RuntimeError(f"no key in output: {out.stdout!r} / {out.stderr!r}")
    return key


def wait_for_health(base_url: str, proc: subprocess.Popen, log_path: Path, timeout: float = 90.0) -> None:
    import httpx

    last_error: Exception | None = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{base_url}/api/v1/health", timeout=2).status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    print(f"last health-check error: {last_error!r}")
    print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
    raise TimeoutError("server did not become healthy in time")


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()
    provider, model = detect_provider(args.provider)
    fixtures = ensure_fixtures()

    import httpx

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType
    from localvectordb.exceptions import MetadataFilterError

    c = Checker(f"e2e_server ({provider}/{model})")

    with temp_workdir("lvdb-e2e-server-") as workdir:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        root_dir = workdir / "databases"
        root_dir.mkdir()
        config_path = workdir / "config.toml"
        config_path.write_text(
            f"""
[database]
root_dir = '{root_dir.as_posix()}'

[embedding]
provider = "{provider}"
model = "{model}"

[server]
host = "127.0.0.1"
port = {port}
file_upload_enabled = true

[server.security]
require_api_key = true
key_database_path = '{(workdir / "api_keys.db").as_posix()}'
""",
            encoding="utf-8",
        )

        c.section("boot: keys + server process")
        rw_key = create_key(config_path, "read_write")
        ro_key = create_key(config_path, "read_only")
        c.check("api keys created", bool(rw_key) and bool(ro_key) and rw_key != ro_key)

        log_path = workdir / "server.log"
        server_log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [lvdb_exe(), "--config", str(config_path), "serve"],
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_health(base_url, proc, log_path)
            c.check("server healthy", True)

            c.section("authentication enforcement")
            r = httpx.get(f"{base_url}/api/v1/databases", timeout=10)
            c.check("missing key rejected", r.status_code in (401, 403), f"got {r.status_code}")
            r = httpx.get(
                f"{base_url}/api/v1/databases", headers={"Authorization": "Bearer not-a-real-key"}, timeout=10
            )
            c.check("bogus key rejected", r.status_code in (401, 403), f"got {r.status_code}")

            c.section("RemoteVectorDB client flow")
            db = VectorDB(
                DB_NAME,
                base_url,
                api_key=rw_key,
                metadata_schema={"topic": MetadataField(type=MetadataFieldType.TEXT, indexed=True)},
                embedding_provider=provider,
                embedding_model=model,
            )
            ids = db.upsert(
                documents=[content for content, _ in DOCS.values()],
                metadata=[meta for _, meta in DOCS.values()],
                ids=list(DOCS.keys()),
            )
            c.check("remote upsert returns ids", sorted(ids) == sorted(DOCS.keys()), f"got {ids}")
            c.check("remote count", db.count() == len(DOCS), f"count={db.count()}")
            c.check("remote exists batch", db.exists(["volcano", "nope"]) == [True, False])
            c.check("remote get", db.get("espresso").metadata.get("topic") == "coffee")

            r = db.query("how do boats sail against the wind", search_type="vector", k=2)
            c.check("remote vector search", bool(r) and r[0].id == "sailing", f"top={[x.id for x in r]}")
            r = db.query("basaltic", search_type="keyword", k=2)
            c.check("remote keyword search", bool(r) and r[0].id == "volcano", f"top={[x.id for x in r]}")
            r = db.query("brewing strong coffee under pressure", search_type="hybrid", k=3, filters={"topic": "coffee"})
            c.check("remote hybrid + filter", bool(r) and {x.id for x in r} == {"espresso"}, f"got {[x.id for x in r]}")

            try:
                db.query("anything", k=1, filters={"not_a_real_field": "x"})
                c.check("remote query rejects unknown filter field", False, "no error raised")
            except MetadataFilterError:
                c.check("remote query rejects unknown filter field", True)

            docs = db.filter(where={"topic": {"$in": ["geology", "sailing"]}})
            c.check(
                "remote filter $in", {d.id for d in docs} == {"volcano", "sailing"}, f"got {sorted(d.id for d in docs)}"
            )

            c.check("remote update", db.update("volcano", metadata={"topic": "volcanology"}) is True)
            c.check("remote update visible", db.get("volcano").metadata["topic"] == "volcanology")

            sim = db.compare_documents("volcano", "espresso")
            c.check("remote compare_documents in [0,1]", 0.0 <= sim <= 1.0, f"got {sim}")
            nn = db.nearest_neighbors("sailing", k=2)
            c.check(
                "remote nearest_neighbors excludes self",
                all(x.id != "sailing" for x in nn),
                f"got {[x.id for x in nn]}",
            )

            streamed = list(db.query_stream("volcanic eruptions", search_type="vector", k=2))
            c.check("remote query_stream yields results", len(streamed) > 0, f"got {len(streamed)}")

            c.section("raw REST: listing + multipart upload")
            headers = {"Authorization": f"Bearer {rw_key}"}

            r = httpx.post(
                f"{base_url}/api/v1/{DB_NAME}/query",
                headers=headers,
                json={"query": "x", "search_type": "keyword", "filters": {"not_a_real_field": 1}},
                timeout=30,
            )
            c.check(
                "REST bad filter -> 400 INVALID_FILTER",
                r.status_code == 400 and r.json().get("error", {}).get("code") == "INVALID_FILTER",
                f"got {r.status_code}: {r.text[:200]}",
            )

            r = httpx.get(f"{base_url}/api/v1/databases", headers=headers, timeout=10)
            names = [
                d["name"] if isinstance(d, dict) else d
                for d in r.json().get("databases", r.json() if isinstance(r.json(), list) else [])
            ]
            c.check("database listed via REST", DB_NAME in str(names), f"got {names}")

            with (
                (fixtures / "machine_learning.pdf").open("rb") as f1,
                (fixtures / "financial_report.docx").open("rb") as f2,
            ):
                r = httpx.post(
                    f"{base_url}/api/v1/{DB_NAME}/upload",
                    headers=headers,
                    files=[
                        ("files", ("machine_learning.pdf", f1, "application/pdf")),
                        (
                            "files",
                            (
                                "financial_report.docx",
                                f2,
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            ),
                        ),
                    ],
                    data={"use_filename_as_id": "true", "metadata": '{"topic": "upload"}'},
                    timeout=180,
                )
            c.check("upload returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")
            if r.status_code == 200:
                body = r.json()
                c.check("upload processed both files", body.get("files_processed") == 2, f"got {body}")
            qr = db.query("operating margin and revenue guidance", search_type="vector", k=3)
            c.check(
                "uploaded docx retrievable semantically",
                any(x.id == "financial_report.docx" for x in qr),
                f"got {[x.id for x in qr]}",
            )

            r = httpx.post(
                f"{base_url}/api/v1/search",
                headers=headers,
                json={"query": "volcanic eruptions and lava", "search_type": "vector", "k": 2},
                timeout=60,
            )
            c.check(
                "global cross-database search",
                r.status_code == 200 and DB_NAME in r.text,
                f"got {r.status_code}: {r.text[:200]}",
            )

            c.section("read-only key enforcement")
            ro_headers = {"Authorization": f"Bearer {ro_key}"}
            r = httpx.post(
                f"{base_url}/api/v1/{DB_NAME}/query",
                headers=ro_headers,
                json={"query": "volcano", "search_type": "keyword", "k": 2},
                timeout=30,
            )
            c.check("read-only key can query", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
            r = httpx.post(
                f"{base_url}/api/v1/{DB_NAME}/documents",
                headers=ro_headers,
                json={"documents": ["sneaky write"], "ids": ["sneaky"]},
                timeout=30,
            )
            c.check("read-only key cannot write", r.status_code == 403, f"got {r.status_code}: {r.text[:200]}")

            c.section("delete + teardown")
            c.check("remote delete", db.delete("volcano") == 1)
            c.check("remote deleted gone", db.exists("volcano") is False)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            server_log.close()

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
