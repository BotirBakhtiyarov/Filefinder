"""Streamed, resumable download helper for update packages.

Downloads to a ``<dest>.part`` file (resumable via HTTP Range requests),
optionally verifies a SHA-256 digest, and atomically renames to the final
destination on success.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

import requests

from filefinder.core.config import t

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 256 * 1024  # 256 KB
_TIMEOUT = 15

ProgressCallback = Callable[[int, int], None]


class DownloadError(Exception):
    """Raised when a download fails due to a network or server error."""


class ChecksumError(Exception):
    """Raised when the downloaded file does not match the expected SHA-256."""


def _sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    dest: str | Path,
    progress_cb: ProgressCallback | None = None,
    sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``dest`` with resume support and optional checksum.

    A partial download is kept in ``<dest>.part``; a subsequent call resumes
    it with an HTTP ``Range`` header. On success the part file is atomically
    renamed to ``dest``.

    Args:
        url: URL to download.
        dest: Final destination path.
        progress_cb: Optional callback invoked as
            ``progress_cb(downloaded_bytes, total_bytes)`` after each chunk.
            ``total_bytes`` may be ``0`` when the server does not send a
            Content-Length.
        sha256: Optional expected SHA-256 hex digest.

    Returns:
        Path to the downloaded file (``dest``).

    Raises:
        DownloadError: On network or HTTP errors.
        ChecksumError: When ``sha256`` is given and the digest mismatches.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    downloaded = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    if downloaded:
        logger.info("Resuming download of %s at %d bytes", url, downloaded)

    try:
        with requests.get(
            url, stream=True, timeout=_TIMEOUT, headers=headers
        ) as resp:
            if downloaded and resp.status_code == 200:
                # Server ignored the Range header; start over.
                logger.info("Server does not support resume; restarting download")
                downloaded = 0
            try:
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise DownloadError(t("updater.error.download_failed")) from exc

            remaining = int(resp.headers.get("Content-Length", 0) or 0)
            total = downloaded + remaining
            mode = "ab" if downloaded else "wb"
            with open(part, mode) as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total)
    except DownloadError:
        raise
    except requests.RequestException as exc:
        logger.warning("Download failed: %s", exc)
        raise DownloadError(t("updater.error.download_failed")) from exc
    except OSError as exc:
        logger.warning("Download I/O failed: %s", exc)
        raise DownloadError(t("updater.error.download_failed")) from exc

    if sha256:
        actual = _sha256_of(part)
        if actual.lower() != sha256.lower():
            logger.error("Checksum mismatch for %s: %s != %s", url, actual, sha256)
            # Drop the poisoned part file so a retry starts clean.
            part.unlink(missing_ok=True)
            raise ChecksumError(t("updater.error.checksum_mismatch"))

    os.replace(part, dest)
    logger.info("Downloaded %s -> %s (%d bytes)", url, dest, downloaded)
    return dest
