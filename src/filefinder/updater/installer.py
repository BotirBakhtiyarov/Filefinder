"""Install a downloaded update package and restart the application.

Phase 1 targets Windows (frozen builds); running from source uses the
project root as the "app dir" so the flow stays testable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from filefinder.core.config import t

logger = logging.getLogger(__name__)

_BACKUP_DIR = "_backup"
# Never backed up, overwritten or deleted during install/restore.
_EXCLUDED_NAMES = {_BACKUP_DIR, "models", "venv", ".venv", ".git"}


class InstallError(Exception):
    """Raised when installing an update package fails."""


def _app_dir() -> Path:
    """Return the application installation directory.

    Returns:
        Parent directory of ``sys.executable`` when running frozen
        (PyInstaller/Flet build), otherwise the project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # <root>/src/filefinder/updater/installer.py -> <root>
    return Path(__file__).resolve().parents[3]


def _backup_current(app_dir: Path, backup_dir: Path) -> None:
    """Copy the current app files into ``backup_dir``.

    Args:
        app_dir: Application directory.
        backup_dir: Destination backup directory (created if missing).
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in app_dir.iterdir():
        if item.name in _EXCLUDED_NAMES:
            continue
        target = backup_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _restore_backup(app_dir: Path, backup_dir: Path) -> None:
    """Restore the app dir from a backup after a failed install.

    Args:
        app_dir: Application directory.
        backup_dir: Backup directory created by :func:`_backup_current`.
    """
    for item in app_dir.iterdir():
        if item.name in _EXCLUDED_NAMES:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    for item in backup_dir.iterdir():
        target = app_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def install(zip_path: str | Path) -> Path:
    """Install an update zip over the application directory.

    Backs up the current installation to ``<app>/_backup``, extracts the zip
    over the app directory and, on any failure, restores the backup.

    Args:
        zip_path: Path to the downloaded update ``.zip``.

    Returns:
        The application directory that was updated.

    Raises:
        InstallError: When the zip cannot be extracted (the previous
            installation is restored first).
    """
    zip_path = Path(zip_path)
    app_dir = _app_dir()
    backup_dir = app_dir / _BACKUP_DIR

    logger.info("Installing update %s into %s", zip_path, app_dir)
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        _backup_current(app_dir, backup_dir)
    except OSError as exc:
        logger.error("Failed to create backup: %s", exc)
        raise InstallError(t("updater.error.install_failed")) from exc

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(app_dir)
    except Exception as exc:
        logger.error("Update extraction failed, restoring backup: %s", exc)
        try:
            _restore_backup(app_dir, backup_dir)
        except OSError:
            logger.exception("Backup restore failed; installation may be broken")
        raise InstallError(t("updater.error.install_failed")) from exc

    logger.info("Update installed successfully")
    return app_dir


def restart() -> None:
    """Relaunch the application and terminate the current process.

    Raises:
        NotImplementedError: On non-Windows platforms (manual restart
            required in Phase 1).
    """
    if not sys.platform.startswith("win"):
        raise NotImplementedError("manual restart")
    exe = sys.executable
    logger.info("Restarting application: %s", exe)
    subprocess.Popen([exe])
    os._exit(0)
