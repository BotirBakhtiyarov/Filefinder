"""Hybrid semantic + keyword search.

Semantic scores come from the vector store (cosine similarity of BGE
embeddings), keyword scores from SQLite FTS5 bm25 (normalized to 0..1).
The hybrid score is ``SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT *
keyword``. The engine degrades gracefully: without an embedding model it
falls back to keyword-only, and without FTS hits to semantic-only.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from filefinder.core import constants
from filefinder.core.db import get_db, get_image_vector_store, get_vector_store
from filefinder.core.embeddings import ModelNotAvailableError, get_embedder

logger = logging.getLogger(__name__)

# Oversampling factor for the vector store query before max-per-file merge.
_SEMANTIC_OVERSAMPLE = 3


@dataclass
class SearchResult:
    """A single merged search hit."""

    path: str
    filename: str
    extension: str
    snippet: str
    score: float
    modified_time: float
    size: int


@dataclass
class SearchFilters:
    """Optional post-hoc filters applied to candidate results."""

    file_types: Optional[list[str]] = None
    date_from: Optional[float] = None
    date_to: Optional[float] = None
    directory: Optional[str] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None

    def matches(self, row: dict[str, Any]) -> bool:
        """Return True if a file row passes all configured filters."""
        if self.file_types:
            types = {e.lower() if e.startswith(".") else f".{e.lower()}"
                     for e in self.file_types}
            if str(row.get("extension", "")).lower() not in types:
                return False
        modified = float(row.get("modified_time") or 0.0)
        if self.date_from is not None and modified < self.date_from:
            return False
        if self.date_to is not None and modified > self.date_to:
            return False
        if self.directory:
            directory = self.directory.replace("\\", "/").rstrip("/") + "/"
            path = str(row.get("path", "")).replace("\\", "/")
            if not path.startswith(directory) and path != directory.rstrip("/"):
                return False
        size = int(row.get("size") or 0)
        if self.min_size is not None and size < self.min_size:
            return False
        if self.max_size is not None and size > self.max_size:
            return False
        return True


# Spec/legacy alias: the design spec calls the filters dataclass ``Filters``.
Filters = SearchFilters


# ----------------------------------------------------------------------
# Defensive helpers over the Database facade / vector store
# ----------------------------------------------------------------------

def _row_to_dict(row: Any) -> Optional[dict[str, Any]]:
    """Best-effort conversion of a db row (dict, sqlite3.Row) to a dict."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except (TypeError, ValueError):
        return None


def _get_connection(db: Any) -> Any:
    """Probe the Database object for a raw sqlite3 connection."""
    for name in ("_get_conn", "get_conn", "connect", "connection"):
        attr = getattr(db, name, None)
        if callable(attr):
            try:
                return attr()
            except TypeError:
                continue
    for name in ("conn", "_conn"):
        attr = getattr(db, name, None)
        if attr is not None and not callable(attr):
            return attr
    return None


def _get_file_by_id(db: Any, file_id: Any) -> Optional[dict[str, Any]]:
    """Fetch a file row by primary key, via facade or raw SQL fallback."""
    getter = getattr(db, "get_file_by_id", None)
    if callable(getter):
        return _row_to_dict(getter(file_id))
    conn = _get_connection(db)
    if conn is None:
        return None
    try:
        cursor = conn.execute(
            "SELECT id, path, filename, extension, size, modified_time,"
            " content_hash, text_preview, indexed_at FROM files WHERE id = ?",
            (file_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [d[0] for d in cursor.description]
        return dict(zip(columns, row))
    except Exception as exc:  # noqa: BLE001
        logger.warning("file lookup by id failed: %s", exc)
        return None


def _get_chunk_content(db: Any, file_id: Any, chunk_index: int) -> Optional[str]:
    """Fetch a chunk's text, via facade or raw SQL fallback."""
    for name in ("get_chunk_content", "get_chunk"):
        getter = getattr(db, name, None)
        if callable(getter):
            try:
                content = getter(file_id, chunk_index)
            except TypeError:
                continue
            if isinstance(content, str):
                return content
            row = _row_to_dict(content)
            if row and "content" in row:
                return str(row["content"])
    conn = _get_connection(db)
    if conn is None:
        return None
    try:
        cursor = conn.execute(
            "SELECT content FROM file_chunks WHERE file_id = ? AND chunk_index = ?",
            (file_id, chunk_index),
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("chunk lookup failed: %s", exc)
        return None


def _parse_embedding_id(embedding_id: Any) -> Optional[tuple[Any, int]]:
    """Parse an embedding id of the form ``'<file_id>:<chunk_index>'``."""
    match = re.match(r"^(?P<file>.+):(?P<chunk>\d+)$", str(embedding_id))
    if not match:
        return None
    file_part: Any = match.group("file")
    try:
        file_part = int(file_part)
    except ValueError:
        pass
    return file_part, int(match.group("chunk"))


# ----------------------------------------------------------------------
# Search engine
# ----------------------------------------------------------------------

class SearchEngine:
    """Hybrid semantic + keyword search over the indexed corpus."""

    # --------------------------------------------------------------
    # Semantic branch
    # --------------------------------------------------------------

    def _semantic_scores(
        self, query: str, top_k: int
    ) -> Optional[dict[str, tuple[float, str]]]:
        """Return ``{path: (score, snippet)}`` or None if unavailable."""
        try:
            embedding = get_embedder().embed_text(query)
        except ModelNotAvailableError as exc:
            logger.warning("Semantic search unavailable (no model): %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - degrade to keyword-only
            logger.warning("Query embedding failed: %s", exc)
            return None

        try:
            store = get_vector_store()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector store unavailable: %s", exc)
            return None

        try:
            hits = store.query(embedding, top_k * _SEMANTIC_OVERSAMPLE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector store query failed: %s", exc)
            return None

        db = get_db()
        best: dict[str, tuple[float, str]] = {}
        for embedding_id, score in hits or []:
            parsed = _parse_embedding_id(embedding_id)
            if parsed is None:
                continue
            file_id, chunk_index = parsed
            file_row = _get_file_by_id(db, file_id)
            if not file_row:
                continue
            path = str(file_row.get("path", ""))
            score = float(score)
            if path in best and best[path][0] >= score:
                continue
            snippet = _get_chunk_content(db, file_id, chunk_index)
            if not snippet:
                snippet = str(file_row.get("text_preview") or "")
            best[path] = (score, snippet)
        return best

    # --------------------------------------------------------------
    # Image branch (CLIP)
    # --------------------------------------------------------------

    def _image_scores(
        self, query: str, top_k: int
    ) -> Optional[dict[str, tuple[float, str, dict[str, Any]]]]:
        """Return ``{path: (score, snippet, row)}`` for image hits, or None.

        CLIP cosine similarities are used as-is (they are already roughly
        0..1). Any failure is logged and yields None so image search can
        never break text search.
        """
        try:
            from filefinder.core.image_embeddings import get_image_embedder

            embedder = get_image_embedder()
            if not embedder.is_available():
                return None
            embedding = embedder.embed_query(query)
        except ModelNotAvailableError as exc:
            logger.warning("Image search unavailable (no CLIP model): %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - degrade to text-only
            logger.warning("Image query embedding failed: %s", exc)
            return None

        try:
            hits = get_image_vector_store().query(embedding, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image vector store query failed: %s", exc)
            return None

        db = get_db()
        results: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for vector_id, score in hits or []:
            try:
                file_id: Any = int(vector_id)
            except (TypeError, ValueError):
                file_id = str(vector_id)
            file_row = _get_file_by_id(db, file_id)
            if not file_row:
                continue
            path = str(file_row.get("path", ""))
            snippet = str(file_row.get("text_preview") or "") or str(
                file_row.get("filename") or ""
            )
            results[path] = (float(score), snippet, file_row)

        # Relevance gate: CLIP cosine similarities sit in a narrow band, so
        # unrelated images still score ~0.2. Drop anything below an absolute
        # floor and anything far below the best hit — otherwise "red car"
        # surfaces every image in the index.
        if results:
            best = max(score for score, _, _ in results.values())
            cutoff = max(
                constants.IMAGE_MIN_SCORE, best * constants.IMAGE_RELATIVE_RATIO
            )
            results = {
                path: (min(score * constants.IMAGE_SCORE_BOOST, 1.0), snippet, row)
                for path, (score, snippet, row) in results.items()
                if score >= cutoff
            }
        return results

    # --------------------------------------------------------------
    # Keyword branch
    # --------------------------------------------------------------

    def _keyword_scores(
        self, query: str, top_k: int
    ) -> dict[str, tuple[float, str, dict[str, Any]]]:
        """Return ``{path: (normalized_score, snippet, row)}`` from FTS5."""
        db = get_db()
        try:
            rows = db.keyword_search(query, top_k * _SEMANTIC_OVERSAMPLE)
        except Exception as exc:  # noqa: BLE001 - degrade to semantic-only
            logger.warning("Keyword search failed: %s", exc)
            return {}

        parsed: list[tuple[dict[str, Any], float]] = []
        for row in rows or []:
            row_dict = _row_to_dict(row)
            if not row_dict:
                continue
            # FTS5 bm25() returns negative values where lower is better.
            raw = row_dict.get("rank", row_dict.get("score", row_dict.get("bm25", 0.0)))
            try:
                relevance = -float(raw)
            except (TypeError, ValueError):
                relevance = 0.0
            parsed.append((row_dict, relevance))

        if not parsed:
            return {}

        relevances = [r for _, r in parsed]
        low, high = min(relevances), max(relevances)
        span = high - low

        results: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for row_dict, relevance in parsed:
            normalized = (relevance - low) / span if span > 0 else 1.0
            path = str(row_dict.get("path", ""))
            snippet = str(row_dict.get("text_preview") or "")
            results[path] = (normalized, snippet, row_dict)
        return results

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = constants.TOP_K,
    ) -> list[SearchResult]:
        """Run a hybrid search and return merged, filtered results.

        Args:
            query: free-text query.
            filters: optional :class:`SearchFilters` applied post-hoc.
            top_k: maximum number of results.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
        """
        query = (query or "").strip()
        if not query:
            return []

        semantic = self._semantic_scores(query, top_k)
        keyword = self._keyword_scores(query, top_k)

        db = get_db()
        combined: dict[str, tuple[float, str]] = {}
        paths = set(keyword) | set(semantic or {})
        for path in paths:
            sem = semantic.get(path) if semantic else None
            kw = keyword.get(path)
            if semantic is None:
                # No model / embedding failure: keyword-only.
                score = kw[0] if kw else 0.0
            elif not keyword:
                # No FTS hits: semantic-only.
                score = sem[0] if sem else 0.0
            else:
                score = (constants.SEMANTIC_WEIGHT * (sem[0] if sem else 0.0)
                         + constants.KEYWORD_WEIGHT * (kw[0] if kw else 0.0))
            snippet = (sem[1] if sem else "") or (kw[1] if kw else "")
            combined[path] = (score, snippet)

        # Image-content (CLIP) branch: merged with document results using
        # the raw cosine score (kept as-is). Fully guarded: image search
        # must never break text search.
        try:
            images = self._image_scores(query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image search failed: %s", exc)
            images = None
        if images:
            for path, (img_score, img_snippet, _row) in images.items():
                if path in combined:
                    existing_score, existing_snippet = combined[path]
                    combined[path] = (
                        max(existing_score, img_score),
                        existing_snippet or img_snippet,
                    )
                else:
                    combined[path] = (img_score, img_snippet)

        results: list[SearchResult] = []
        for path, (score, snippet) in combined.items():
            row = keyword.get(path, (0.0, "", None))[2]
            if row is None:
                row = _row_to_dict(db.get_file_by_path(path)) or {"path": path}
            if filters is not None and not filters.matches(row):
                continue
            results.append(
                SearchResult(
                    path=str(row.get("path", path)),
                    filename=str(row.get("filename", "")),
                    extension=str(row.get("extension", "")),
                    snippet=snippet,
                    score=float(score),
                    modified_time=float(row.get("modified_time") or 0.0),
                    size=int(row.get("size") or 0),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


_engine: Optional[SearchEngine] = None
_engine_lock = threading.Lock()


def get_search_engine() -> SearchEngine:
    """Return the module-level :class:`SearchEngine` singleton."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SearchEngine()
    return _engine
