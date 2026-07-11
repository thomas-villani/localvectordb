"""
Rebuild a database's FAISS indices from its SQLite rows.

Databases written before FAISS ids became monotonic can contain duplicate
``faiss_id`` values: ids were allocated from ``index.ntotal``, which ``remove_ids``
decrements, so any delete or replacing upsert re-issued ids that were still live.
A duplicated id makes one vector hydrate two chunk rows, so queries return the
wrong document. ``_verify_integrity`` refuses to open such a database and points
here.

Repair reassigns every id from a fresh monotonic counter and rewrites the SQLite
columns that reference them.

Recovering the vectors themselves is decided **per id**, not globally:

* An id that appears exactly once can be read straight back out of the index with
  ``reconstruct()`` -- no embedding provider, no network, no cost.
* A **duplicated** id cannot. ``reconstruct(id)`` returns whichever single vector
  the id map points at, so one of the two colliding chunks would silently receive
  the other's vector -- swapping a visible bug for an invisible one. Both are
  re-embedded from ``chunks.content``.
* An id live in the index with no owning row is an orphan and is dropped.

So a clean database compacts for free, and a corrupted one only pays to re-embed
the chunks that actually collided.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from localvectordb.database.base import LocalVectorDBBase

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    """What repair found, and what it did (or would do, under ``dry_run``)."""

    dry_run: bool = False
    duplicate_ids: List[int] = field(default_factory=list)
    orphan_vectors: List[int] = field(default_factory=list)
    dangling_rows: List[int] = field(default_factory=list)
    reconstructed: int = 0
    reembedded: int = 0
    dropped: int = 0
    base_index_type: str = ""
    sections_rebuilt: int = 0
    documents_rebuilt: int = 0

    @property
    def healthy(self) -> bool:
        return not (self.duplicate_ids or self.orphan_vectors or self.dangling_rows)

    def summary(self) -> str:
        findings = (
            f"{len(self.duplicate_ids)} duplicate id(s), "
            f"{len(self.orphan_vectors)} orphan vector(s), "
            f"{len(self.dangling_rows)} dangling row(s)"
        )
        if self.dry_run:
            if self.healthy:
                return "Database is healthy; nothing to repair."
            return (
                f"Found {findings}. Re-run without --dry-run to rebuild. "
                f"Chunks sharing a duplicated id must be re-embedded; the rest are reconstructed in place."
            )
        if self.healthy:
            return f"Database was healthy; index compacted ({self.reconstructed} vectors)."
        return (
            f"Found {findings}. Rebuilt: {self.reconstructed} reconstructed, "
            f"{self.reembedded} re-embedded, {self.dropped} dropped."
        )


class RepairMixin(LocalVectorDBBase, ABC):
    """Mixed into LocalVectorDB. Requires the core FAISS/SQLite attributes."""

    def _empty_clone_of(self, index) -> Any:
        """
        An empty index of exactly the same type, metric and trained state.

        ``clone_index`` then ``reset`` avoids reconstructing constructor arguments,
        which cannot be recovered faithfully (an IndexHNSWFlat reports 2*M neighbours
        at level 0, and IndexLSH carries a trained rotation).
        """
        base = self._unwrap_base_index(index)
        clone = faiss.clone_index(base)
        clone.reset()
        return faiss.IndexIDMap2(clone)

    def _collect_main_id_owners(self, conn) -> Dict[int, List[Tuple[str, Any]]]:
        """faiss_id -> [(table, rowid), ...] over the tables sharing the main id space."""
        owners: Dict[int, List[Tuple[str, Any]]] = {}
        for row in conn.execute("SELECT rowid AS rid, faiss_id AS fid FROM chunks WHERE faiss_id IS NOT NULL"):
            owners.setdefault(int(row["fid"]), []).append(("chunks", row["rid"]))
        for row in conn.execute("SELECT rowid AS rid, faiss_id AS fid FROM column_embeddings"):
            owners.setdefault(int(row["fid"]), []).append(("column_embeddings", row["rid"]))
        return owners

    def _diagnose(self, conn) -> RepairReport:
        report = RepairReport(base_index_type=self._base_index_type_name(self.index))
        owners = self._collect_main_id_owners(conn)
        live = {int(i) for i in self._live_faiss_ids(self.index)}

        report.duplicate_ids = sorted(fid for fid, rows in owners.items() if len(rows) > 1)
        report.orphan_vectors = sorted(live - set(owners))
        report.dangling_rows = sorted(set(owners) - live)
        return report

    def _reembed_chunk_contents(self, conn, rowids: List[Any]) -> Dict[Any, np.ndarray]:
        """Re-embed chunk rows whose vectors cannot be trusted."""
        if not rowids:
            return {}
        placeholders = ",".join("?" * len(rowids))
        rows = conn.execute(
            f"SELECT rowid AS rid, content FROM chunks WHERE rowid IN ({placeholders})", rowids
        ).fetchall()
        texts = [r["content"] for r in rows]
        embeddings = self.embedding_provider.embed_sync(texts)
        arr = np.asarray(embeddings, dtype=np.float32)
        return {r["rid"]: arr[i] for i, r in enumerate(rows)}

    def repair(self, dry_run: bool = False) -> RepairReport:
        """
        Rebuild the FAISS indices from SQLite, reassigning every id.

        Parameters
        ----------
        dry_run
            Report what is wrong without modifying anything.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")

        with self.connection_pool.get_connection() as conn:
            report = self._diagnose(conn)

            if dry_run:
                report.dry_run = True
                return report

            owners = self._collect_main_id_owners(conn)
            live = {int(i) for i in self._live_faiss_ids(self.index)}
            duplicates = set(report.duplicate_ids)

            # reconstruct() only returns a trustworthy vector for an id that maps to
            # exactly one row and is actually present in the index.
            can_reconstruct = isinstance(self.index, faiss.IndexIDMap2)
            if not can_reconstruct:
                logger.warning("Index is not an IndexIDMap2; vectors cannot be reconstructed and will be re-embedded.")

            reembed_rowids: List[Any] = []
            reconstruct_plan: List[Tuple[Any, int]] = []  # (chunk rowid, old faiss id)
            column_plan: List[Tuple[Any, int]] = []  # (column_embeddings rowid, old faiss id)

            for fid, rows in owners.items():
                trustworthy = can_reconstruct and fid in live and fid not in duplicates
                for table, rowid in rows:
                    if table == "chunks":
                        if trustworthy:
                            reconstruct_plan.append((rowid, fid))
                        else:
                            reembed_rowids.append(rowid)
                    else:
                        if trustworthy:
                            column_plan.append((rowid, fid))
                        else:
                            # A metadata-field vector has no recoverable source text here
                            # (the field text lives in documents.metadata and may have
                            # changed); drop it and let the next upsert recreate it.
                            report.dropped += 1

            reembedded = self._reembed_chunk_contents(conn, reembed_rowids)

            new_index = self._empty_clone_of(self.index)
            with self._faiss_id_lock:
                self._faiss_id_counters["main"] = 0

            updates_chunks: List[Tuple[int, Any]] = []
            updates_columns: List[Tuple[int, Any]] = []
            vectors: List[np.ndarray] = []
            new_ids: List[int] = []

            def _stage(vec: Any, rowid: Any, table: str) -> None:
                new_id = int(self._allocate_faiss_ids("main", 1)[0])
                vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
                new_ids.append(new_id)
                (updates_chunks if table == "chunks" else updates_columns).append((new_id, rowid))

            for rowid, old_id in reconstruct_plan:
                _stage(self.index.reconstruct(old_id), rowid, "chunks")
                report.reconstructed += 1
            for rowid, old_id in column_plan:
                _stage(self.index.reconstruct(old_id), rowid, "column_embeddings")
                report.reconstructed += 1
            for rowid, vec in reembedded.items():
                _stage(vec, rowid, "chunks")
                report.reembedded += 1

            report.dropped += len(report.orphan_vectors)

            if vectors:
                stacked = self._normalize_for_index(np.vstack(vectors), new_index)
                new_index.add_with_ids(stacked, np.array(new_ids, dtype=np.int64))

            conn.execute("BEGIN")
            try:
                if updates_chunks:
                    conn.executemany("UPDATE chunks SET faiss_id = ? WHERE rowid = ?", updates_chunks)
                if updates_columns:
                    conn.executemany("UPDATE column_embeddings SET faiss_id = ? WHERE rowid = ?", updates_columns)
                # Rows whose vector could not be recovered at all lose their reference.
                dropped_column_rowids = [
                    rowid
                    for fid, rows in owners.items()
                    for table, rowid in rows
                    if table == "column_embeddings" and not (can_reconstruct and fid in live and fid not in duplicates)
                ]
                if dropped_column_rowids:
                    placeholders = ",".join("?" * len(dropped_column_rowids))
                    conn.execute(
                        f"DELETE FROM column_embeddings WHERE rowid IN ({placeholders})", dropped_column_rowids
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            with self._faiss_lock.write_lock():
                self.index = new_index

            self._rebuild_hierarchical_indices(conn, report)

        self._save_internal()
        self._save_faiss_counters()
        logger.info(f"Repair complete: {report.summary()}")
        return report

    def _rebuild_hierarchical_indices(self, conn, report: RepairReport) -> None:
        """Reassign section/document ids from their own fresh counters."""
        if not self._hierarchical_embeddings:
            return

        for name, index_attr, table, id_column, row_key in (
            ("section", "section_index", "sections", "faiss_id", "id"),
            ("document", "document_index", "documents", "doc_faiss_id", "id"),
        ):
            index = getattr(self, index_attr)
            if index is None:
                continue

            live = {int(i) for i in self._live_faiss_ids(index)}
            rows = conn.execute(
                f"SELECT {row_key} AS key, {id_column} AS fid FROM {table} WHERE {id_column} IS NOT NULL"
            ).fetchall()

            new_index = self._empty_clone_of(index)
            with self._faiss_id_lock:
                self._faiss_id_counters[name] = 0

            vectors: List[np.ndarray] = []
            new_ids: List[int] = []
            updates: List[Tuple[int, Any]] = []
            orphaned_keys: List[Any] = []

            for row in rows:
                old_id = int(row["fid"])
                if old_id not in live or not isinstance(index, faiss.IndexIDMap2):
                    orphaned_keys.append(row["key"])
                    continue
                new_id = int(self._allocate_faiss_ids(name, 1)[0])
                vectors.append(np.asarray(index.reconstruct(old_id), dtype=np.float32).reshape(-1))
                new_ids.append(new_id)
                updates.append((new_id, row["key"]))

            if vectors:
                stacked = self._normalize_for_index(np.vstack(vectors), new_index)
                new_index.add_with_ids(stacked, np.array(new_ids, dtype=np.int64))

            conn.execute("BEGIN")
            try:
                if updates:
                    conn.executemany(f"UPDATE {table} SET {id_column} = ? WHERE {row_key} = ?", updates)
                if orphaned_keys:
                    placeholders = ",".join("?" * len(orphaned_keys))
                    conn.execute(
                        f"UPDATE {table} SET {id_column} = NULL WHERE {row_key} IN ({placeholders})", orphaned_keys
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            with self._faiss_lock.write_lock():
                setattr(self, index_attr, new_index)

            if name == "section":
                report.sections_rebuilt = len(updates)
            else:
                report.documents_rebuilt = len(updates)


def _saved_embedding_config(db_path: Any) -> Dict[str, str]:
    """Read the embedding provider/model a database was built with, straight from SQLite."""
    import sqlite3

    saved: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT key, value FROM config WHERE key IN ('embedding_provider', 'embedding_model')"
            ).fetchall()
        finally:
            conn.close()
        saved = {k: v for k, v in rows}
    except sqlite3.Error as e:  # pragma: no cover - unreadable/missing db handled by the caller
        logger.debug(f"Could not read saved embedding config: {e}")
    return saved


def open_for_repair(name: str, base_path: str, **kwargs: Any) -> Any:
    """
    Open a database bypassing the on-open integrity check.

    ``_verify_integrity`` raises on precisely the databases repair exists to fix, so
    the normal constructor (and ``get_ctx_db``) cannot reach them.

    The constructor validates its *default* embedding provider before it loads the
    database's saved config, which would make repairing a database require whichever
    provider happens to be the default to be reachable. Read the saved provider from
    SQLite first so repair works offline whenever it does not need to re-embed.
    """
    from pathlib import Path

    from localvectordb.database import LocalVectorDB

    kwargs.setdefault("create_if_not_exists", False)

    db_path = Path(base_path) / f"{name}.sqlite"
    saved = _saved_embedding_config(db_path)
    if "embedding_provider" in saved:
        kwargs.setdefault("embedding_provider", saved["embedding_provider"])
    if "embedding_model" in saved:
        kwargs.setdefault("embedding_model", saved["embedding_model"])

    return LocalVectorDB(name=name, base_path=base_path, _skip_integrity_check=True, **kwargs)
