"""OCR support via Tesseract (pytesseract).

All heavy/optional dependencies (``pytesseract``, ``Pillow``, ``pdf2image``,
``cv2``) are imported lazily so this module imports cleanly without them.
Every public function degrades gracefully: failures are logged and an empty
string (or ``False``) is returned instead of raising.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OCR languages: Simplified Chinese + English, with automatic fallback.
_LANG_PREFERRED = "chi_sim+eng"
_LANG_FALLBACK = "eng"

# Common Tesseract install location on Windows.
_WINDOWS_TESSERACT = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Cached resolution result (only successful resolutions are cached, so a
# just-installed tesseract is picked up on the next lookup even without an
# explicit :func:`reset_resolution`).
_resolved_cmd: str | None = None


# ----------------------------------------------------------------------
# Tesseract binary / data discovery
# ----------------------------------------------------------------------

def _resolve_tesseract_cmd() -> str | None:
    """Locate the tesseract executable.

    Resolution order: ``TESSERACT_CMD`` environment variable, ``PATH``
    lookup, the common Windows install path, then the app-local install
    (``<app_data_dir>/Tesseract/tesseract.exe``, used by the automatic
    installer in :mod:`filefinder.core.deps`).

    Returns:
        Path to the executable, or ``None`` if not found.
    """
    global _resolved_cmd
    if _resolved_cmd is not None:
        return _resolved_cmd
    cmd: str | None = None
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and Path(env_cmd).is_file():
        cmd = env_cmd
    else:
        which_cmd = shutil.which("tesseract")
        if which_cmd:
            cmd = which_cmd
        elif Path(_WINDOWS_TESSERACT).is_file():
            cmd = _WINDOWS_TESSERACT
        else:
            from filefinder.core.config import get_app_data_dir  # lazy

            appdata_cmd = get_app_data_dir() / "Tesseract" / "tesseract.exe"
            if appdata_cmd.is_file():
                cmd = str(appdata_cmd)
    if cmd is not None:
        _resolved_cmd = cmd
    return cmd


def reset_resolution() -> None:
    """Clear the cached tesseract path.

    Call after a fresh install so the new binary is picked up without an
    app restart. (Failed lookups are never cached, so this is mostly a
    safety net for future caching changes.)
    """
    global _resolved_cmd
    _resolved_cmd = None


def get_tesseract_cmd() -> str | None:
    """Return the resolved tesseract executable path, or ``None``."""
    return _resolve_tesseract_cmd()


def _get_pytesseract() -> Any:
    """Import pytesseract and point it at the resolved binary.

    Returns:
        The pytesseract module, or ``None`` if unavailable.
    """
    try:
        import pytesseract  # lazy import
    except ImportError:
        logger.info("pytesseract not installed; OCR disabled.")
        return None
    cmd = _resolve_tesseract_cmd()
    if cmd is None:
        logger.info("Tesseract executable not found; OCR disabled.")
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def _tessdata_dir(tesseract_cmd: str) -> Path:
    """Return the tessdata directory for a given tesseract binary."""
    return Path(tesseract_cmd).resolve().parent / "tessdata"


def check_tesseract_data() -> dict[str, bool]:
    """Report which OCR language packs exist in the tessdata directory.

    Returns:
        Mapping of language code (``"chi_sim"``, ``"eng"``) to availability.
        Both are ``False`` if tesseract itself is not found.
    """
    result = {"chi_sim": False, "eng": False}
    cmd = _resolve_tesseract_cmd()
    if cmd is None:
        return result
    tessdata = _tessdata_dir(cmd)
    for lang in result:
        result[lang] = (tessdata / f"{lang}.traineddata").is_file()
    # Also consider the user-level tessdata dir (downloaded language packs
    # land there when the system dir is not writable).
    user_dir = get_tessdata_dir()
    if user_dir is not None:
        for lang, present in result.items():
            if not present:
                result[lang] = (user_dir / f"{lang}.traineddata").is_file()
    return result


def _select_languages() -> str:
    """Pick the OCR language string, falling back to eng without chi_sim."""
    data = check_tesseract_data()
    if data.get("chi_sim") and data.get("eng"):
        return _LANG_PREFERRED
    if data.get("eng"):
        logger.info("chi_sim traineddata missing; falling back to eng OCR.")
        return _LANG_FALLBACK
    return _LANG_PREFERRED  # let tesseract report the error downstream


def is_ocr_available() -> bool:
    """Return True if pytesseract and a tesseract binary are both present."""
    return _get_pytesseract() is not None


# ----------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------

def _deskew(image: Any) -> Any:
    """Deskew an image with OpenCV if available; otherwise return as-is."""
    try:
        import cv2  # optional dependency
        import numpy as np
    except ImportError:
        return image
    try:
        gray = np.array(image)
        inverted = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inverted > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45.0:
            angle = -(90.0 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.1:
            return image
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        from PIL import Image

        return Image.fromarray(rotated)
    except Exception as exc:  # noqa: BLE001 - deskew is best-effort
        logger.debug("Deskew failed (ignored): %s", exc)
        return image


def _preprocess(image: Any) -> Any:
    """Grayscale + autocontrast (+ optional deskew) a PIL image for OCR."""
    from PIL import ImageOps

    if image.mode not in ("L", "RGB"):
        # Palette images with transparency must go through RGBA first,
        # otherwise PIL warns and the alpha channel produces dark artifacts.
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        if image.mode == "RGBA":
            # Flatten onto a white background before grayscale.
            from PIL import Image

            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = _deskew(gray)
    return gray


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def get_tessdata_dir() -> Path | None:
    """Return the user-level tessdata dir when it holds traineddata files.

    Language packs downloaded by :mod:`filefinder.core.deps` land in
    ``<appdata>/tessdata`` when the system tessdata dir (e.g. under
    ``C:\\Program Files``) is not writable. Missing languages that exist in
    the system tessdata dir are backfilled (copied) so the user dir is
    self-contained when Tesseract is pointed at it.
    """
    try:
        from filefinder.core.deps import get_user_tessdata_dir

        user_dir = get_user_tessdata_dir()
        if not (user_dir.is_dir() and any(user_dir.glob("*.traineddata"))):
            return None
        # Backfill languages available in the system tessdata dir.
        try:
            import shutil

            cmd = _resolve_tesseract_cmd()
            if cmd is not None:
                system_dir = _tessdata_dir(cmd)
                if system_dir != user_dir:
                    for trained in system_dir.glob("*.traineddata"):
                        target = user_dir / trained.name
                        if not target.exists():
                            shutil.copy2(trained, target)
        except OSError as exc:
            logger.debug("Could not backfill traineddata: %s", exc)
        return user_dir
    except Exception as exc:  # noqa: BLE001
        logger.debug("User tessdata dir unavailable: %s", exc)
    return None


def _prepare_tesseract_env() -> None:
    """Point Tesseract at the user-level tessdata dir when one is in use.

    Uses the ``TESSDATA_PREFIX`` environment variable (pytesseract passes
    ``--tessdata-dir`` with literal quotes on Windows, which breaks
    Tesseract's argument parsing).
    """
    user_dir = get_tessdata_dir()
    if user_dir is not None:
        os.environ["TESSDATA_PREFIX"] = str(user_dir)


def extract_text_from_image(path: str) -> str:
    """OCR a single image file.

    Args:
        path: filesystem path of the image.

    Returns:
        Recognized text, or ``""`` if OCR is unavailable or fails.
    """
    pytesseract = _get_pytesseract()
    if pytesseract is None:
        return ""
    try:
        from PIL import Image

        _prepare_tesseract_env()
        with Image.open(path) as image:
            processed = _preprocess(image)
            text = pytesseract.image_to_string(processed, lang=_select_languages())
        return (text or "").strip()
    except Exception as exc:  # noqa: BLE001 - never crash on bad image
        logger.warning("OCR failed for image %s: %s", path, exc)
        return ""


def extract_text_from_pdf(path: str) -> str:
    """OCR every page of a (scanned) PDF via pdf2image + Tesseract.

    Args:
        path: filesystem path of the PDF.

    Returns:
        Recognized text of all pages joined by newlines, or ``""`` if
        pdf2image/poppler or Tesseract is unavailable or fails.
    """
    pytesseract = _get_pytesseract()
    if pytesseract is None:
        return ""
    try:
        from pdf2image import convert_from_path  # lazy import (needs poppler)
    except ImportError:
        logger.info("pdf2image not installed; PDF OCR disabled.")
        return ""
    try:
        poppler_path: str | None = None
        try:
            from filefinder.core import deps  # lazy import

            bin_dir = deps.get_poppler_bin_dir()
            if bin_dir is not None:
                poppler_path = str(bin_dir)
        except Exception as exc:  # noqa: BLE001 - detection is best-effort
            logger.debug("poppler detection failed: %s", exc)
        if poppler_path is not None:
            images = convert_from_path(path, poppler_path=poppler_path)
        else:
            images = convert_from_path(path)
    except Exception as exc:  # noqa: BLE001 - poppler missing/broken, etc.
        logger.warning("pdf2image failed for %s (poppler available?): %s", path, exc)
        return ""

    lang = _select_languages()
    _prepare_tesseract_env()
    parts: list[str] = []
    for index, image in enumerate(images):
        try:
            processed = _preprocess(image)
            text = pytesseract.image_to_string(processed, lang=lang)
            if text and text.strip():
                parts.append(text.strip())
        except Exception as exc:  # noqa: BLE001 - skip broken page
            logger.warning("OCR failed for page %d of %s: %s", index, path, exc)
    return "\n".join(parts)
