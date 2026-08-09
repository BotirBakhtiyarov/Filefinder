"""Check GitHub Releases for a newer FileFinder version.

Queries the latest-release endpoint of the repository configured in
``filefinder.core.constants.GITHUB_REPO``, compares semantic versions and
selects the platform-appropriate ``.zip`` asset plus its optional sibling
``.sha256`` asset.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import requests

from filefinder.core import constants
from filefinder.core.config import t

logger = logging.getLogger(__name__)

_API_URL = "https://api.github.com/repos/{repo}/releases/latest"
_TIMEOUT = 15
_ACCEPT_HEADER = {"Accept": "application/vnd.github+json"}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


class UpdateCheckError(Exception):
    """Raised when checking for updates fails (network error or bad response)."""


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a semver-ish string into a comparable tuple of ints.

    Args:
        version: Version string such as ``"v3.2.1"`` or ``"3.2"``.

    Returns:
        Tuple of integer components, e.g. ``(3, 2, 1)``. Non-numeric
        components count as ``0``.
    """
    text = version.strip().lstrip("vV")
    parts: list[int] = []
    for piece in text.split("."):
        match = re.match(r"\d+", piece)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def _platform_suffix() -> str:
    """Return the asset platform suffix for the current platform.

    Returns:
        ``"windows"``, ``"android"`` or ``"linux"``.
    """
    if sys.platform.startswith("win"):
        return "windows"
    if hasattr(sys, "getandroidapiname") or "ANDROID_PRIVATE" in os.environ:
        return "android"
    return "linux"


def _fetch_sha256(asset: dict[str, Any]) -> str | None:
    """Download a ``.sha256`` asset and extract the hex digest.

    Args:
        asset: GitHub release asset dict.

    Returns:
        The 64-char hex digest, or ``None`` if it cannot be fetched/parsed.
    """
    url = asset.get("browser_download_url")
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch sha256 asset %s: %s", url, exc)
        return None
    match = _SHA256_RE.search(resp.text)
    if not match:
        logger.warning("No sha256 digest found in %s", url)
        return None
    return match.group().lower()


def _select_zip_asset(
    assets: list[dict[str, Any]], version: str, suffix: str
) -> dict[str, Any] | None:
    """Pick the right zip asset for this installation type.

    Frozen (flet build) installs use ``FileFinder-v{ver}-{platform}.zip``
    (the full exe bundle). Source/zip (python) installs must instead take
    the SAC-safe payload ``FileFinder-{ver}.zip`` — extracting the exe
    bundle over a python install would corrupt it.
    """
    from filefinder.updater.installer import is_frozen_install

    if not is_frozen_install():
        exact = f"filefinder-{version}.zip".lower()
        match = next(
            (a for a in assets if a.get("name", "").lower() == exact), None
        )
        if match is not None:
            return match
    return next(
        (
            a
            for a in assets
            if f"-{suffix}" in a.get("name", "") and a.get("name", "").endswith(".zip")
        ),
        None,
    )


def check_update(current_version: str) -> dict[str, Any] | None:
    """Check GitHub for a newer release of FileFinder.

    Args:
        current_version: Currently installed version, e.g. ``"3.0.0"``
            (a leading ``v`` is tolerated).

    Returns:
        A dict with keys ``version``, ``url``, ``notes``, ``published_at``
        and ``sha256`` (``None`` when no hash asset exists) if a newer
        release with a matching platform asset is available; ``None`` when
        the current version is up to date or no matching asset exists.

    Raises:
        UpdateCheckError: On network failures or unexpected API responses.
    """
    url = _API_URL.format(repo=constants.GITHUB_REPO)
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers=_ACCEPT_HEADER)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Update check failed: %s", exc)
        raise UpdateCheckError(t("updater.error.check_failed")) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise UpdateCheckError(t("updater.error.check_failed")) from exc

    tag = str(data.get("tag_name", "")).strip()
    if not tag:
        raise UpdateCheckError(t("updater.error.check_failed"))

    remote_version = _parse_version(tag)
    local_version = _parse_version(current_version)
    # Pad to equal length so (3, 0) and (3, 0, 0) compare equal.
    width = max(len(remote_version), len(local_version))
    remote_version += (0,) * (width - len(remote_version))
    local_version += (0,) * (width - len(local_version))
    if remote_version <= local_version:
        logger.info("Up to date: local=%s remote=%s", current_version, tag)
        return None

    suffix = _platform_suffix()
    assets: list[dict[str, Any]] = data.get("assets") or []
    zip_asset = _select_zip_asset(assets, tag.lstrip("vV"), suffix)
    if zip_asset is None:
        logger.info("Release %s has no asset for platform '%s'", tag, suffix)
        return None

    sha256: str | None = None
    zip_name = zip_asset.get("name", "")
    hash_asset = next(
        (
            a
            for a in assets
            if a.get("name", "") == f"{zip_name}.sha256"
            or (f"-{suffix}" in a.get("name", "") and a.get("name", "").endswith(".sha256"))
        ),
        None,
    )
    if hash_asset is not None:
        sha256 = _fetch_sha256(hash_asset)

    logger.info("Update available: %s -> %s (%s)", current_version, tag, zip_name)
    return {
        "version": tag.lstrip("vV"),
        "url": zip_asset.get("browser_download_url"),
        "notes": data.get("body") or "",
        "published_at": data.get("published_at") or "",
        "sha256": sha256,
    }
