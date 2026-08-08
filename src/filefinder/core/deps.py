"""Automatic installation of external dependencies (Tesseract OCR).

FileFinder v3 is fully self-contained: when the Tesseract OCR engine is
missing on Windows, :func:`ensure_tesseract` installs it automatically —
first via ``winget``, falling back to downloading the official UB-Mannheim
installer and running it silently into the app data directory. Language
data (``eng``, ``chi_sim``) is fetched afterwards via
:func:`ensure_language_data`.

Every public function degrades gracefully: failures are logged and
``False`` is returned instead of raising, so the UI never crashes.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from filefinder.core import ocr
from filefinder.core.config import get_app_data_dir, is_windows

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]

# winget / silent-installer timeouts are generous: installs can be slow.
_WINGET_TIMEOUT_S = 900
_INSTALLER_TIMEOUT_S = 900

_WINGET_PACKAGE_ID = "UB-Mannheim.TesseractOCR"

# Official UB-Mannheim Windows installers, tried in order.
_INSTALLER_URLS: tuple[str, ...] = (
    "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.5.0.20241111.exe",
    "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.4.0.20240606.exe",
)

# Fast (integerized) traineddata files from the official mirror.
_TESSDATA_URL_TEMPLATE = (
    "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
)

DEFAULT_LANGS: tuple[str, ...] = ("eng", "chi_sim")


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------

def is_tesseract_installed() -> bool:
    """Return True when a usable Tesseract binary is available.

    Delegates to :func:`filefinder.core.ocr.is_ocr_available` (which checks
    both ``pytesseract`` and the tesseract executable).
    """
    try:
        return ocr.is_ocr_available()
    except Exception as exc:  # noqa: BLE001 - never raise to the UI
        logger.warning("OCR availability check failed: %s", exc)
        return False


def _emit_status(status_cb: StatusCallback | None, msg_key: str) -> None:
    """Invoke the UI status callback with an i18n key, best-effort."""
    if status_cb is None:
        return
    try:
        status_cb(msg_key)
    except Exception as exc:  # noqa: BLE001 - status display must not break
        logger.debug("status callback failed for %r: %s", msg_key, exc)


# ----------------------------------------------------------------------
# Install strategies
# ----------------------------------------------------------------------

def _install_via_winget() -> bool:
    """Try installing Tesseract via winget (may trigger a UAC prompt).

    Returns:
        True when winget ran and reported success (exit code 0).
    """
    if not shutil.which("winget"):
        logger.info("winget not available; skipping winget install.")
        return False
    cmd = [
        "winget", "install",
        "--id", _WINGET_PACKAGE_ID,
        "-e",
        "--silent",
        "--accept-source-agreements",
        "--accept-package-agreements",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_WINGET_TIMEOUT_S,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - timeout, OS errors, etc.
        logger.warning("winget install failed to run: %s", exc)
        return False
    logger.info(
        "winget install exited %d\nstdout: %s\nstderr: %s",
        result.returncode, result.stdout, result.stderr,
    )
    return result.returncode == 0


def _download_installer(progress_cb: ProgressCallback | None) -> Path | None:
    """Download the official Tesseract installer, trying mirrors in order.

    Args:
        progress_cb: optional ``(downloaded_bytes, total_bytes)`` callback.

    Returns:
        Path to the downloaded installer, or ``None`` when all URLs failed.
    """
    from filefinder.updater import downloader  # lazy import

    dest = get_app_data_dir() / "downloads" / "tesseract-setup.exe"
    for url in _INSTALLER_URLS:
        try:
            return downloader.download(url, dest, progress_cb=progress_cb)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            logger.warning("Installer download failed (%s): %s", url, exc)
    return None


def _run_installer(installer: Path) -> bool:
    """Run the NSIS installer silently into ``<appdata>/Tesseract``.

    ``/D`` must be the LAST argument, unquoted, with backslashes.

    Args:
        installer: path to the downloaded setup executable.

    Returns:
        True when the install completed and ``tesseract.exe`` exists.
    """
    target = get_app_data_dir() / "Tesseract"
    cmd = [str(installer), "/S", f"/D={target}"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_INSTALLER_TIMEOUT_S,
            check=False,
        )
        logger.info(
            "Tesseract installer exited %d\nstdout: %s\nstderr: %s",
            result.returncode, result.stdout, result.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract installer failed to run: %s", exc)
        return False
    exe = target / "tesseract.exe"
    if not exe.is_file():
        logger.warning("Installer finished but %s is missing.", exe)
        return False
    return True


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def ensure_tesseract(
    progress_cb: ProgressCallback | None = None,
    status_cb: StatusCallback | None = None,
) -> bool:
    """Ensure Tesseract OCR is installed (Windows only).

    Steps: try ``winget`` first, then fall back to downloading the official
    installer and running it silently into the app data directory. On
    success the OCR language data is fetched too.

    Args:
        progress_cb: optional ``(downloaded_bytes, total_bytes)`` callback
            for download progress.
        status_cb: optional callback receiving i18n keys for UI status
            lines (``deps.status_*``).

    Returns:
        True when a usable Tesseract is available afterwards. Never raises.
    """
    try:
        if is_tesseract_installed():
            return True
        if not is_windows():
            logger.info("Automatic Tesseract install is Windows-only.")
            return False

        _emit_status(status_cb, "deps.status_winget")
        if _install_via_winget():
            ocr.reset_resolution()
            if is_tesseract_installed():
                _emit_status(status_cb, "deps.status_langdata")
                ensure_language_data(progress_cb=progress_cb)
                return True
            logger.warning("winget reported success but tesseract not found.")

        _emit_status(status_cb, "deps.status_download")
        installer = _download_installer(progress_cb)
        if installer is not None:
            _emit_status(status_cb, "deps.status_install")
            if _run_installer(installer):
                ocr.reset_resolution()

        if not is_tesseract_installed():
            logger.error("Tesseract installation failed (all strategies).")
            return False
        _emit_status(status_cb, "deps.status_langdata")
        ensure_language_data(progress_cb=progress_cb)
        return True
    except Exception:
        logger.exception("ensure_tesseract failed")
        return False


def get_user_tessdata_dir() -> Path:
    """Return the user-writable tessdata directory.

    Deliberately placed in the plain home directory (``~/.filefinder``)
    rather than the app data dir: Python installed from the Microsoft Store
    virtualizes writes to ``%APPDATA%``, so files stored there are
    invisible to the native ``tesseract.exe`` process. One-time migration
    copies any previously downloaded packs from the old location.
    """
    new_dir = Path.home() / ".filefinder" / "tessdata"
    try:
        from filefinder.core.config import get_app_data_dir

        old_dir = Path(get_app_data_dir()) / "tessdata"
        if old_dir != new_dir and old_dir.is_dir():
            import shutil

            new_dir.mkdir(parents=True, exist_ok=True)
            for trained in old_dir.glob("*.traineddata"):
                target = new_dir / trained.name
                if not target.exists():
                    shutil.copy2(trained, target)
    except OSError as exc:
        logger.debug("tessdata migration skipped: %s", exc)
    return new_dir


def _dir_writable(path: Path) -> bool:
    """Return True if ``path`` can be created and written to."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True
    except OSError:
        return False


def _writable_tessdata_dir(primary: Path) -> Path | None:
    """Pick a writable tessdata dir, falling back to the app data dir.

    When the tesseract installation lives in a protected location (e.g.
    ``C:\\Program Files``) its tessdata dir is not writable for regular
    users. In that case the fallback ``<appdata>/tessdata`` is used and any
    existing ``*.traineddata`` files from the primary dir are copied over
    so no language is lost. The OCR layer picks the same directory via
    :func:`filefinder.core.ocr.get_tessdata_dir`.
    """
    if _dir_writable(primary):
        return primary
    fallback = get_user_tessdata_dir()
    if not _dir_writable(fallback):
        logger.warning("No writable tessdata dir (tried %s, %s)", primary, fallback)
        return None
    logger.info("Tessdata dir %s not writable; using %s", primary, fallback)
    try:
        import shutil

        for trained in primary.glob("*.traineddata"):
            target = fallback / trained.name
            if not target.exists():
                shutil.copy2(trained, target)
    except OSError as exc:
        logger.warning("Could not copy existing traineddata files: %s", exc)
    return fallback


def ensure_language_data(
    langs: Sequence[str] = DEFAULT_LANGS,
    progress_cb: ProgressCallback | None = None,
) -> bool:
    """Ensure the given ``.traineddata`` files exist in the tessdata dir.

    The tessdata directory is resolved from the detected tesseract binary
    (``<tesseract_dir>/tessdata``) and created when missing. Missing
    languages are downloaded from the official ``tessdata_fast`` mirror.

    Args:
        langs: language codes to ensure (default ``eng`` + ``chi_sim``).
        progress_cb: optional ``(downloaded_bytes, total_bytes)`` callback.

    Returns:
        True when every requested language file is present afterwards.
        Never raises.
    """
    try:
        cmd = ocr.get_tesseract_cmd()
        if cmd is None:
            logger.warning("No tesseract binary; cannot fetch language data.")
            return False
        tessdata = _writable_tessdata_dir(Path(cmd).resolve().parent / "tessdata")
        if tessdata is None:
            return False

        from filefinder.updater import downloader  # lazy import

        for lang in langs:
            target = tessdata / f"{lang}.traineddata"
            if target.is_file():
                continue
            url = _TESSDATA_URL_TEMPLATE.format(lang=lang)
            try:
                downloader.download(url, target, progress_cb=progress_cb)
            except Exception as exc:  # noqa: BLE001 - report at the end
                logger.warning("Failed to download %s: %s", url, exc)

        missing = [
            lang for lang in langs
            if not (tessdata / f"{lang}.traineddata").is_file()
        ]
        if missing:
            logger.warning("Missing language data after install: %s", missing)
            return False
        return True
    except Exception:
        logger.exception("ensure_language_data failed")
        return False


# ----------------------------------------------------------------------
# Poppler (PDF rendering for pdf2image)
# ----------------------------------------------------------------------

_POPPLER_RELEASE_API = (
    "https://api.github.com/repos/oschwartz10612/poppler-windows/releases/latest"
)
_POPPLER_TIMEOUT_S = 60


def get_poppler_install_dir() -> Path:
    """Return the managed poppler install dir (``~/.filefinder/poppler``).

    Deliberately placed in the plain home directory rather than the app data
    dir: Python installed from the Microsoft Store virtualizes ``%APPDATA%``,
    so native executables cannot see files stored there.
    """
    return Path.home() / ".filefinder" / "poppler"


def get_poppler_bin_dir() -> Path | None:
    """Return the directory containing ``pdfinfo.exe``/``pdftoppm.exe``.

    Checks the managed install dir first (``Library/bin`` layout and the
    plain ``bin`` zip-layout variation), then ``PATH``. Never raises.

    Returns:
        The directory path, or ``None`` when poppler is not found.
    """
    try:
        base = get_poppler_install_dir()
        for candidate in (base / "Library" / "bin", base / "bin"):
            if (candidate / "pdfinfo.exe").is_file() and (
                candidate / "pdftoppm.exe"
            ).is_file():
                return candidate
        which_cmd = shutil.which("pdfinfo")
        if which_cmd:
            return Path(which_cmd).resolve().parent
    except Exception as exc:  # noqa: BLE001 - detection must never break
        logger.debug("poppler detection failed: %s", exc)
    return None


def _download_poppler(progress_cb: ProgressCallback | None) -> Path | None:
    """Download the latest poppler-windows release zip from GitHub.

    Args:
        progress_cb: optional ``(downloaded_bytes, total_bytes)`` callback.

    Returns:
        Path to the downloaded zip, or ``None`` on failure.
    """
    try:
        import requests  # lazy import

        resp = requests.get(_POPPLER_RELEASE_API, timeout=_POPPLER_TIMEOUT_S)
        resp.raise_for_status()
        assets = resp.json().get("assets", [])
        url = next(
            (a["browser_download_url"] for a in assets
             if a.get("browser_download_url", "").endswith(".zip")),
            None,
        )
        if url is None:
            logger.warning("No .zip asset found in poppler release.")
            return None
    except Exception as exc:  # noqa: BLE001 - network/JSON errors
        logger.warning("Could not query poppler release: %s", exc)
        return None

    from filefinder.updater import downloader  # lazy import

    dest = get_poppler_install_dir() / "downloads" / "poppler.zip"
    try:
        return downloader.download(url, dest, progress_cb=progress_cb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Poppler download failed (%s): %s", url, exc)
        return None


def _extract_poppler(zip_path: Path) -> Path | None:
    """Extract the poppler zip into the managed install dir.

    The zip contains a top-level ``poppler-XX/`` directory with
    ``Library/bin`` inside; its contents are flattened into
    ``~/.filefinder/poppler/`` so the binaries land in one of the
    well-known locations checked by :func:`get_poppler_bin_dir`.

    Returns:
        The resolved bin dir, or ``None`` when ``pdfinfo.exe`` is missing.
    """
    import zipfile

    install_dir = get_poppler_install_dir()
    staging = install_dir / "_staging"
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)
        matches = list(staging.rglob("pdfinfo.exe"))
        if not matches:
            logger.warning("Poppler zip did not contain pdfinfo.exe.")
            return None
        # Move the contents of the top-level extracted dir into install_dir.
        top = matches[0].parent
        while top.parent != staging:
            top = top.parent
        for child in top.iterdir():
            target = install_dir / child.name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(child), str(target))
        return get_poppler_bin_dir()
    except Exception as exc:  # noqa: BLE001 - corrupt zip, OS errors, etc.
        logger.warning("Poppler extraction failed: %s", exc)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_poppler(bin_dir: Path) -> bool:
    """Run ``pdfinfo -v`` to confirm the extracted binaries work."""
    try:
        result = subprocess.run(
            [str(bin_dir / "pdfinfo.exe"), "-v"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        logger.info(
            "pdfinfo -v exited %d\nstdout: %s\nstderr: %s",
            result.returncode, result.stdout, result.stderr,
        )
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfinfo verification failed to run: %s", exc)
        return False


def ensure_poppler(
    progress_cb: ProgressCallback | None = None,
    status_cb: StatusCallback | None = None,
) -> bool:
    """Ensure poppler (pdfinfo/pdftoppm) is installed (Windows only).

    Downloads the latest ``oschwartz10612/poppler-windows`` release zip and
    extracts it into ``~/.filefinder/poppler/``.

    Args:
        progress_cb: optional ``(downloaded_bytes, total_bytes)`` callback
            for download progress.
        status_cb: optional callback receiving i18n keys for UI status
            lines (``deps.status_*``).

    Returns:
        True when a usable poppler is available afterwards. Never raises.
    """
    try:
        if get_poppler_bin_dir() is not None:
            return True
        if not is_windows():
            logger.info("Automatic poppler install is Windows-only.")
            return False

        _emit_status(status_cb, "deps.status_poppler")
        zip_path = _download_poppler(progress_cb)
        if zip_path is None:
            return False
        _emit_status(status_cb, "deps.status_poppler_extract")
        bin_dir = _extract_poppler(zip_path)
        if bin_dir is None:
            return False
        if not _verify_poppler(bin_dir):
            logger.error("Poppler verification failed after install.")
            return False
        return True
    except Exception:
        logger.exception("ensure_poppler failed")
        return False


def ensure_pdf_ocr(
    progress_cb: ProgressCallback | None = None,
    status_cb: StatusCallback | None = None,
) -> bool:
    """Ensure everything needed for PDF OCR: Tesseract + poppler.

    Returns:
        True only when both components are available afterwards.
    """
    ok_tesseract = ensure_tesseract(progress_cb=progress_cb, status_cb=status_cb)
    ok_poppler = ensure_poppler(progress_cb=progress_cb, status_cb=status_cb)
    return ok_tesseract and ok_poppler
