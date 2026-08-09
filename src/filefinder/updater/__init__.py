"""Auto-update package: check, download, install and restart."""

from __future__ import annotations

from filefinder.updater.downloader import ChecksumError, DownloadError, download
from filefinder.updater.github_checker import UpdateCheckError, check_update
from filefinder.updater.installer import InstallError, install, restart

__all__ = [
    "ChecksumError",
    "DownloadError",
    "InstallError",
    "UpdateCheckError",
    "check_update",
    "download",
    "install",
    "restart",
]
