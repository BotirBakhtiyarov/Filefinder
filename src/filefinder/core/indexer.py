"""Filesystem scanning, chunking and index pipeline.

Walks monitored directories, extracts text via
:mod:`filefinder.core.file_parser`, chunks it, embeds it via
:mod:`filefinder.core.embeddings` and persists everything through
:mod:`filefinder.core.db`.

The embedding model is optional: when it is missing
(:class:`ModelNotAvailableError`) files are still indexed (metadata +
FTS content) and only vector chunks are skipped, with a warning logged.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from filefinder.core import constants
from filefinder.core.config import get_config
from filefinder.core.db import get_db, get_image_vector_store
from filefinder.core.embeddings import ModelNotAvailableError, get_embedder
from filefinder.core.file_parser import UnsupportedFileError, extract_text
from filefinder.core.image_embeddings import get_image_embedder

logger = logging.getLogger(__name__)

# Read size for streaming content hashes.
_HASH_BLOCK_SIZE = 65536

# Number of characters of a file's extracted text stored as preview.
TEXT_PREVIEW_LENGTH = 500

ProgressCallback = Callable[[int, int, str], None]


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------

def _is_hidden(name: str) -> bool:
    """Return True if a file/dir name should be treated as hidden."""
    return name.startswith(".")


def scan_directory(
    path: str,
    extensions: set[str] | None = None,
) -> list[str]:
    """Recursively collect supported files under ``path``.

    Hidden directories and files are skipped and symlinks are never
    followed (``followlinks=False``), which also prevents symlink loops.

    Args:
        path: root directory to scan.
        extensions: allowed lowercase extensions (with dot). Defaults to
            :data:`constants.SUPPORTED_EXTENSIONS`.

    Returns:
        Sorted list of absolute file paths.
    """
    if extensions is None:
        extensions = constants.SUPPORTED_EXTENSIONS
    root = Path(path)
    if not root.is_dir():
        logger.warning("scan_directory: not a directory: %s", path)
        return []

    found: list[str] = []
    excluded = {d.lower() for d in constants.EXCLUDED_DIR_NAMES}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune hidden and excluded (caches, VCS, system, app-junk)
        # directories in-place so os.walk does not descend.
        dirnames[:] = [
            d for d in dirnames if not _is_hidden(d) and d.lower() not in excluded
        ]
        for filename in filenames:
            if _is_hidden(filename):
                continue
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in extensions:
                found.append(str(file_path))
    found.sort()
    return found


# ----------------------------------------------------------------------
# Hashing & chunking
# ----------------------------------------------------------------------

def _content_hash(path: str) -> str:
    """Compute the MD5 hex digest of a file, reading it in blocks.

    Args:
        path: file to hash.

    Returns:
        Lowercase hex digest string.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = constants.CHUNK_SIZE,
    overlap: int = constants.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks of roughly ``chunk_size`` characters.

    Chunk boundaries prefer paragraph breaks (blank lines); paragraphs
    longer than ``chunk_size`` are hard-split. Consecutive chunks share
    up to ``overlap`` trailing/leading characters for context continuity.

    Args:
        text: the full extracted document text.
        chunk_size: target maximum characters per chunk.
        overlap: characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    # Split on paragraph boundaries (one or more blank lines).
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    # Further split oversized paragraphs with a sliding window.
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            pieces.append(paragraph[start:end])
            if end >= len(paragraph):
                break
            start = end - overlap

    # Greedily pack pieces into chunks without exceeding chunk_size.
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        chunks.append(current)

    # Apply character overlap between consecutive chunks when the greedy
    # packing produced no natural overlap (paragraph-packed chunks).
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for previous, chunk in itertools.pairwise(chunks):
            prefix = previous[-overlap:]
            merged = prefix + chunk
            overlapped.append(merged if len(merged) <= chunk_size + overlap else chunk)
        chunks = overlapped

    return [c for c in chunks if c.strip()]


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------

def process_file(path: str) -> dict:
    """Extract text and collect metadata for a single file.

    Args:
        path: file to process.

    Returns:
        Dict with ``path``, ``filename``, ``extension``, ``size``,
        ``modified_time``, ``content_hash`` and ``text`` keys.
    """
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size": stat.st_size,
        "modified_time": stat.st_mtime,
        "content_hash": _content_hash(str(file_path)),
        "text": extract_text(str(file_path)),
    }


def _index_image(path: str, file_id: int, filename: str) -> None:
    """Embed an image file with CLIP and upsert it into the image store.

    Never raises: every failure is logged and swallowed so image indexing
    can never break the regular text indexing of a file.

    Args:
        path: image file path.
        file_id: row id of the file in the ``files`` table.
        filename: display name stored in the vector metadata.
    """
    try:
        embedder = get_image_embedder()
        if not embedder.is_available():
            return
        vector = embedder.embed_image(path)
        if vector is None:
            return
        get_image_vector_store().add(
            [str(file_id)],
            [vector],
            [{"path": path, "filename": filename}],
        )
        logger.debug("Image-embedded %s (file_id=%s)", path, file_id)
    except Exception as exc:  # noqa: BLE001 - never fail the file
        logger.warning("Image embedding failed for %s: %s", path, exc)


def index_file(path: str) -> bool:
    """Index (or re-index) a single file.

    Unchanged files (matching content hash) are skipped. Changed files
    are deleted and re-added (delete-then-add chunks). When the embedding
    model is unavailable only metadata + FTS content are stored.

    Args:
        path: file to index.

    Returns:
        True if the file was (re-)indexed, False if it was skipped
        (unsupported, unchanged, missing or empty content).

    Raises:
        Exception: unexpected indexing errors are logged and re-raised so
            batch callers can count them as failures.
    """
    db = get_db()
    file_path = Path(path)

    extension = file_path.suffix.lower()
    if extension not in constants.SUPPORTED_EXTENSIONS:
        logger.debug("Skipping unsupported file: %s", path)
        return False
    if not file_path.is_file():
        logger.warning("Skipping missing file: %s", path)
        return False

    # Fast path: skip files whose size + mtime are unchanged without
    # re-reading the whole file for the content hash (keeps automatic
    # startup scans cheap).
    existing_meta = db.get_file_by_path(str(file_path))
    if existing_meta:
        try:
            st = file_path.stat()
            if existing_meta.get("size") == st.st_size and abs(
                float(existing_meta.get("modified_time") or 0) - st.st_mtime
            ) < 1e-6:
                logger.debug("Skipping unmodified file: %s", path)
                return False
        except (OSError, TypeError, ValueError):
            pass

    try:
        info = process_file(str(file_path))
    except UnsupportedFileError:
        logger.debug("Skipping unsupported file: %s", path)
        return False
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        raise

    existing = db.get_file_by_path(info["path"])
    if existing and existing.get("content_hash") == info["content_hash"]:
        logger.debug("Skipping unchanged file: %s", path)
        return False

    text = (info["text"] or "").strip()
    is_fallback = False
    if not text:
        # No extractable text (e.g. image without OCR, or a scanned PDF
        # while Tesseract is missing): still index the file by its name so
        # it stays findable.
        text = file_path.stem.replace("_", " ").replace("-", " ").strip()
        if not text:
            logger.info("Skipping file with no extractable text: %s", path)
            return False
        is_fallback = True
        logger.info("No extractable text for %s; indexing by filename", path)

    # Delete-then-add: drop old row, chunks and embeddings first.
    if existing:
        db.remove_file(info["path"])

    file_id = db.upsert_file(
        path=info["path"],
        filename=info["filename"],
        extension=info["extension"],
        size=info["size"],
        modified_time=info["modified_time"],
        content_hash=info["content_hash"],
        text_preview=text[:TEXT_PREVIEW_LENGTH],
    )

    chunks = chunk_text(text)
    # CLIP image embedding is independent of the text (BGE) model — do it
    # first so images are content-searchable even without the text model.
    if extension in constants.IMAGE_EXTENSIONS:
        _index_image(info["path"], file_id, info["filename"])
    if is_fallback:
        # Filename-only content produces degenerate semantic embeddings
        # (single meaningless tokens match everything). Such files stay
        # keyword-searchable via FTS (filename is indexed) but get no
        # semantic vectors.
        logger.debug("Skipping semantic vectors for filename-only file: %s", path)
        return True
    if len(text) < constants.MIN_SEMANTIC_TEXT_LENGTH:
        # Very short texts (OCR noise from icons/stickers) also produce
        # junk semantic matches — keyword search covers them.
        logger.debug("Skipping semantic vectors for very short text: %s", path)
        return True
    try:
        embeddings = get_embedder().embed_batch(chunks)
    except ModelNotAvailableError as exc:
        logger.warning(
            "Embedding model unavailable (%s); indexed metadata only for %s",
            exc, path,
        )
        return True
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", path, exc)
        raise

    db.add_chunks(file_id, chunks, embeddings)
    logger.debug("Indexed %s (%d chunks)", path, len(chunks))
    return True


def index_directories(
    dirs: list[str],
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Index every supported file under the given directories.

    Designed to run in a worker thread; cancellation is cooperative via
    ``cancel_event``. Per-file failures are logged, skipped and counted.

    Args:
        dirs: directories to scan and index.
        progress_cb: optional ``(current, total, filename)`` callback.
        cancel_event: optional :class:`threading.Event` to abort early.

    Returns:
        Stats dict with ``total``, ``indexed``, ``skipped``, ``failed``
        and ``cancelled`` keys.
    """
    files: list[str] = []
    for directory in dirs:
        files.extend(scan_directory(directory))

    stats = {
        "total": len(files),
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "cancelled": False,
    }

    for current, path in enumerate(files, start=1):
        if cancel_event is not None and cancel_event.is_set():
            stats["cancelled"] = True
            logger.info("Indexing cancelled at %d/%d files", current - 1, len(files))
            break
        if progress_cb is not None:
            try:
                progress_cb(current, stats["total"], Path(path).name)
            except Exception as exc:  # noqa: BLE001 - never break on UI cb
                logger.warning("progress callback failed: %s", exc)
        try:
            if index_file(path):
                stats["indexed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:  # noqa: BLE001 - per-file failure
            logger.error("Failed to index %s: %s", path, exc)
            stats["failed"] += 1

    logger.info("Indexing finished: %s", stats)
    return stats


def remove_file(path: str) -> bool:
    """Remove a file (its row, chunks and embeddings) from the index.

    Args:
        path: indexed file path.

    Returns:
        True if a record existed and was removed.
    """
    db = get_db()
    existing = db.get_file_by_path(path)
    if not existing:
        return False
    try:
        get_image_vector_store().delete([str(existing["id"])])
    except Exception as exc:  # noqa: BLE001 - never fail the removal
        logger.warning("Image vector cleanup failed for %s: %s", path, exc)
    db.remove_file(path)
    logger.debug("Removed %s from index", path)
    return True


def reindex_all(
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Clear the whole index and re-index all monitored directories.

    Args:
        progress_cb: optional ``(current, total, filename)`` callback.
        cancel_event: optional :class:`threading.Event` to abort early.

    Returns:
        Stats dict as returned by :func:`index_directories`.
    """
    db = get_db()
    logger.info("Reindex requested: clearing index")
    db.clear_index()
    dirs = list(get_config().monitored_dirs)
    return index_directories(dirs, progress_cb, cancel_event)
