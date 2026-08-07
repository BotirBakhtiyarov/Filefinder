"""SQLite persistence and vector-store abstraction (spec section 4.3).

Provides the ``files`` / ``files_fts`` (FTS5) / ``file_chunks`` /
``chat_history`` / ``kv`` tables in ``files.db`` (WAL mode, single connection
guarded by an :class:`threading.RLock`), the :class:`VectorStore` abstract base
with a ChromaDB desktop implementation and a sqlite-vec Android skeleton, and
the :class:`Database` facade used by the rest of the core.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

from filefinder.core.config import get_app_data_dir, is_android

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    size INTEGER,
    modified_time REAL,
    content_hash TEXT,
    text_preview TEXT,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS file_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT,
    embedding_id TEXT,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filename,
    content,
    path UNINDEXED
);
"""

_FTS_FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS files_fts (
    path TEXT UNIQUE,
    filename TEXT,
    content TEXT
);
"""


# ----------------------------------------------------------------------
# Vector store abstraction
# ----------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Add or replace vectors with their ids and metadata."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by id."""

    @abstractmethod
    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, float]]:
        """Return the ``top_k`` nearest ``(id, score)`` pairs (higher is better)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors."""


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store for desktop platforms.

    Uses a lazy ``PersistentClient`` rooted at ``<app data>/chroma`` with a
    cosine-distance collection. Cosine distances are converted to similarity
    scores via ``score = 1 - distance``.
    """

    _COLLECTION_NAME = "filefinder"

    def __init__(
        self,
        persist_dir: Optional[str | os.PathLike[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._persist_dir = str(persist_dir or (get_app_data_dir() / "chroma"))
        self._collection_name = collection_name or self._COLLECTION_NAME
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.RLock()

    def _ensure_collection(self) -> Any:
        """Lazily create the Chroma client and collection."""
        with self._lock:
            if self._collection is not None:
                return self._collection
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "chromadb is required for the desktop vector store"
                ) from exc
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert vectors into the collection."""
        if not ids:
            return
        collection = self._ensure_collection()
        collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=[dict(m) for m in metadatas],
        )

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by id; missing ids are ignored."""
        if not ids:
            return
        try:
            self._ensure_collection().delete(ids=[str(i) for i in ids])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector delete failed: %s", exc)

    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, float]]:
        """Return ``(id, similarity)`` pairs, similarity = 1 - cosine distance."""
        collection = self._ensure_collection()
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=max(1, min(top_k, collection.count())),
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [(str(i), 1.0 - float(d)) for i, d in zip(ids, distances)]

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return int(self._ensure_collection().count())

    def clear(self) -> None:
        """Drop and recreate the collection."""
        with self._lock:
            if self._client is None:
                self._ensure_collection()
            try:
                self._client.delete_collection(self._collection_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Collection delete during clear: %s", exc)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )


class SqliteVecStore(VectorStore):
    """sqlite-vec vector store skeleton for Android (Phase 2).

    The ``sqlite-vec`` extension is imported lazily; all operations raise
    :class:`NotImplementedError` until the Phase 2 implementation lands.
    """

    _PHASE2_MSG = (
        "SqliteVecStore is an Android (Phase 2) backend and is not implemented "
        "yet. Install the 'sqlite-vec' package and provide the vec0 table "
        "implementation before enabling Android builds."
    )

    def __init__(self, db_path: Optional[str | os.PathLike[str]] = None) -> None:
        self._db_path = str(db_path or (get_app_data_dir() / "vectors.db"))
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            logger.info("sqlite-vec not installed; Android vector store unavailable.")

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        raise NotImplementedError(self._PHASE2_MSG)

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError(self._PHASE2_MSG)

    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, float]]:
        raise NotImplementedError(self._PHASE2_MSG)

    def count(self) -> int:
        raise NotImplementedError(self._PHASE2_MSG)

    def clear(self) -> None:
        raise NotImplementedError(self._PHASE2_MSG)


_store_lock = threading.RLock()
_vector_store: Optional[VectorStore] = None
_image_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the platform-appropriate :class:`VectorStore` singleton."""
    global _vector_store
    if _vector_store is None:
        with _store_lock:
            if _vector_store is None:
                _vector_store = SqliteVecStore() if is_android() else ChromaStore()
    return _vector_store


def get_image_vector_store() -> VectorStore:
    """Return the vector store singleton for CLIP image embeddings.

    Uses a dedicated ``images`` collection in the same Chroma persistence
    directory on desktop, and the Android sqlite-vec skeleton (with its own
    database file) on Android.
    """
    global _image_vector_store
    if _image_vector_store is None:
        with _store_lock:
            if _image_vector_store is None:
                _image_vector_store = (
                    SqliteVecStore(db_path=get_app_data_dir() / "image_vectors.db")
                    if is_android()
                    else ChromaStore(collection_name="images")
                )
    return _image_vector_store


# ----------------------------------------------------------------------
# Database facade
# ----------------------------------------------------------------------


class Database:
    """Facade over ``files.db`` plus the platform vector store.

    A single SQLite connection is shared across threads
    (``check_same_thread=False``) and guarded by an :class:`threading.RLock`.
    """

    def __init__(self, db_path: Optional[str | os.PathLike[str]] = None) -> None:
        self._db_path = str(db_path or (get_app_data_dir() / "files.db"))
        self._lock = threading.RLock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._init_schema()

    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables, enable WAL, and record the schema version."""
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self._fts_enabled = True
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "FTS5 unavailable (%s); falling back to LIKE search.", exc
                )
                self._conn.executescript(_FTS_FALLBACK_SCHEMA)
            if self.get_kv("schema_version") is None:
                self.set_kv("schema_version", SCHEMA_VERSION)
            self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying SQLite connection (use under external locking care)."""
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    # --------------------------------------------------------------
    # kv store
    # --------------------------------------------------------------

    def set_kv(self, key: str, value: str) -> None:
        """Set a key/value pair in the ``kv`` table."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_kv(self, key: str) -> Optional[str]:
        """Get a value from the ``kv`` table, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    # --------------------------------------------------------------
    # Files
    # --------------------------------------------------------------

    def upsert_file(
        self,
        path: str,
        filename: str,
        extension: str,
        size: int,
        modified_time: float,
        content_hash: str,
        text_preview: str,
    ) -> int:
        """Insert or replace a file record and refresh its FTS row.

        Returns:
            The file's row id.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO files(path, filename, extension, size, modified_time,
                                  content_hash, text_preview, indexed_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,
                    extension=excluded.extension,
                    size=excluded.size,
                    modified_time=excluded.modified_time,
                    content_hash=excluded.content_hash,
                    text_preview=excluded.text_preview,
                    indexed_at=excluded.indexed_at
                """,
                (path, filename, extension, size, modified_time,
                 content_hash, text_preview, time.time()),
            )
            row = self._conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()
            file_id = int(row["id"])
            self._fts_delete(path)
            self._conn.execute(
                "INSERT INTO files_fts(path, filename, content) VALUES(?, ?, ?)",
                (path, filename, text_preview or ""),
            )
            self._conn.commit()
        return file_id

    def _fts_delete(self, path: str) -> None:
        """Remove a path from the FTS table (caller holds the lock)."""
        self._conn.execute("DELETE FROM files_fts WHERE path = ?", (path,))

    def remove_file(self, path: str) -> bool:
        """Remove a file, its chunks and its vectors. Returns True if found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return False
            file_id = int(row["id"])
            embedding_ids = [
                str(r["embedding_id"])
                for r in self._conn.execute(
                    "SELECT embedding_id FROM file_chunks WHERE file_id = ?"
                    " AND embedding_id IS NOT NULL",
                    (file_id,),
                )
            ]
            self._fts_delete(path)
            self._conn.execute(
                "DELETE FROM file_chunks WHERE file_id = ?", (file_id,)
            )
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._conn.commit()
        try:
            get_vector_store().delete(embedding_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector cleanup failed for %s: %s", path, exc)
        try:
            get_image_vector_store().delete([str(file_id)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image vector cleanup failed for %s: %s", path, exc)
        return True

    def get_file_by_path(self, path: str) -> Optional[dict[str, Any]]:
        """Return the file row for ``path`` as a dict, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM files WHERE path = ?", (path,)
            ).fetchone()
        return dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> Optional[dict[str, Any]]:
        """Return the file row for ``file_id`` as a dict, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_files_by_extensions(self, extensions: list[str]) -> list[dict[str, Any]]:
        """Return all file rows whose extension is in ``extensions``.

        Args:
            extensions: lowercase extensions including the dot (e.g. ``.png``).

        Returns:
            List of file rows as dicts (empty when ``extensions`` is empty).
        """
        if not extensions:
            return []
        placeholders = ",".join("?" for _ in extensions)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM files WHERE extension IN ({placeholders})",
                tuple(extensions),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_files(self) -> list[dict[str, Any]]:
        """Return every indexed file row as a dict."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM files").fetchall()
        return [dict(row) for row in rows]

    def keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Full-text keyword search over filename and content.

        Returns:
            Joined file rows with an extra ``rank`` key (raw bm25 score, lower
            is better) ordered by relevance. Falls back to LIKE matching when
            FTS5 is unavailable.
        """
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            if self._fts_enabled:
                try:
                    rows = self._conn.execute(
                        """
                        SELECT f.*, bm25(files_fts) AS rank
                        FROM files_fts
                        JOIN files f ON f.path = files_fts.path
                        WHERE files_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (self._fts_query(query), limit),
                    ).fetchall()
                    return [dict(r) for r in rows]
                except sqlite3.OperationalError as exc:
                    logger.warning("FTS query failed (%s); using LIKE.", exc)
            like = f"%{query}%"
            rows = self._conn.execute(
                """
                SELECT f.*, 0.0 AS rank FROM files_fts
                JOIN files f ON f.path = files_fts.path
                WHERE files_fts.filename LIKE ? OR files_fts.content LIKE ?
                LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _fts_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression (OR of quoted terms)."""
        terms = [t for t in query.replace('"', " ").split() if t]
        if not terms:
            return '""'
        return " OR ".join(f'"{term}"' for term in terms)

    # --------------------------------------------------------------
    # Chunks
    # --------------------------------------------------------------

    def add_chunks(
        self,
        file_id: int,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Store text chunks and their embeddings for a file.

        Embedding ids have the form ``"<file_id>:<chunk_index>"``.

        Returns:
            The embedding ids that were stored.
        """
        embedding_ids = [f"{file_id}:{i}" for i in range(len(chunks))]
        with self._lock:
            self._conn.execute(
                "DELETE FROM file_chunks WHERE file_id = ?", (file_id,)
            )
            self._conn.executemany(
                "INSERT INTO file_chunks(file_id, chunk_index, content,"
                " embedding_id) VALUES(?, ?, ?, ?)",
                [
                    (file_id, i, content, embedding_ids[i])
                    for i, content in enumerate(chunks)
                ],
            )
            self._conn.commit()
        metadatas = [{"file_id": file_id, "chunk_index": i} for i in range(len(chunks))]
        get_vector_store().add(embedding_ids, embeddings, metadatas)
        return embedding_ids

    def get_chunk_content(self, file_id: int, chunk_index: int) -> Optional[str]:
        """Return a chunk's text, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM file_chunks WHERE file_id = ?"
                " AND chunk_index = ?",
                (file_id, chunk_index),
            ).fetchone()
        return str(row["content"]) if row else None

    # --------------------------------------------------------------
    # Stats / maintenance
    # --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return index statistics.

        Returns:
            Dict with ``total_files``, ``total_chunks``, ``vector_count``,
            ``db_size_bytes`` and ``last_indexed`` (epoch float or None).
        """
        with self._lock:
            total_files = self._conn.execute(
                "SELECT COUNT(*) AS c FROM files"
            ).fetchone()["c"]
            total_chunks = self._conn.execute(
                "SELECT COUNT(*) AS c FROM file_chunks"
            ).fetchone()["c"]
        try:
            vector_count = get_vector_store().count()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vector count unavailable: %s", exc)
            vector_count = 0
        try:
            db_size = os.path.getsize(self._db_path)
        except OSError:
            db_size = 0
        last_indexed_raw = self.get_kv("last_indexed")
        try:
            last_indexed = float(last_indexed_raw) if last_indexed_raw else None
        except ValueError:
            last_indexed = None
        return {
            "total_files": int(total_files),
            "total_chunks": int(total_chunks),
            "vector_count": vector_count,
            "db_size_bytes": db_size,
            "last_indexed": last_indexed,
        }

    def clear_index(self) -> None:
        """Delete all files, chunks, FTS rows and vectors."""
        with self._lock:
            self._conn.execute("DELETE FROM file_chunks")
            self._conn.execute("DELETE FROM files")
            self._conn.execute("DELETE FROM files_fts")
            self._conn.commit()
        try:
            get_vector_store().clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector store clear failed: %s", exc)
        try:
            get_image_vector_store().clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image vector store clear failed: %s", exc)

    # --------------------------------------------------------------
    # Chat history
    # --------------------------------------------------------------

    def add_chat_message(self, conversation_id: str, role: str, content: str) -> int:
        """Append a message to a conversation. Returns the row id."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO chat_history(conversation_id, role, content, timestamp)"
                " VALUES(?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )
            self._conn.commit()
        return int(cursor.lastrowid)

    def get_chat_history(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Return messages of a conversation in chronological order."""
        with self._lock:
            if limit is None:
                rows = self._conn.execute(
                    "SELECT * FROM chat_history WHERE conversation_id = ?"
                    " ORDER BY timestamp, id",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM chat_history WHERE conversation_id = ?
                        ORDER BY timestamp DESC, id DESC LIMIT ?
                    ) ORDER BY timestamp, id
                    """,
                    (conversation_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return conversations, most recently active first.

        Each row is a dict with ``conversation_id`` and ``timestamp``
        (last activity) keys.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT conversation_id, MAX(timestamp) AS last_ts"
                " FROM chat_history GROUP BY conversation_id ORDER BY last_ts DESC"
            ).fetchall()
        return [
            {"conversation_id": str(r["conversation_id"]), "timestamp": r["last_ts"]}
            for r in rows
        ]

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete all messages of a conversation."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_history WHERE conversation_id = ?",
                (conversation_id,),
            )
            self._conn.commit()


_db_lock = threading.RLock()
_db: Optional[Database] = None


def get_db() -> Database:
    """Return the module-level :class:`Database` singleton."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database()
    return _db
