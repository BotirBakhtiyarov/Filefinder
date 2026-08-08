"""Configuration, i18n, encryption and platform helpers (spec section 4.2).

Provides platform detection, the per-user app data directory, the :class:`Config`
dataclass persisted to ``config.json`` (with the LLM API key encrypted at rest via
Fernet, keyed from the machine id), the ``t()`` i18n lookup, model directory
helpers and path-safety checks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from filefinder.core.constants import APP_ID

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "zh"
FALLBACK_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("zh", "uz", "en")

_MODEL_NAME_DESKTOP = "bge-m3-onnx"
_MODEL_NAME_ANDROID = "bge-small-zh-v1.5-onnx"

_PBKDF2_ITERATIONS = 100_000


# ----------------------------------------------------------------------
# Platform detection
# ----------------------------------------------------------------------


def is_android() -> bool:
    """Return True when running on Android (e.g. via Flet mobile)."""
    if os.environ.get("FLET_PLATFORM", "").lower() == "android":
        return True
    return hasattr(sys, "getandroidapilevel")


def is_windows() -> bool:
    """Return True when running on Windows."""
    return sys.platform.startswith("win")


def is_linux() -> bool:
    """Return True when running on Linux (excluding Android)."""
    return sys.platform.startswith("linux") and not is_android()


def get_app_data_dir() -> Path:
    """Return the per-user application data directory, creating it if needed.

    The ``FILEFINDER_DATA_DIR`` environment variable takes precedence over
    all platform defaults (used by the test suite and portable installs).

    Returns:
        ``$FILEFINDER_DATA_DIR`` when set, otherwise ``%APPDATA%/FileFinder``
        on Windows, ``~/.config/filefinder`` on Linux, and the Flet client
        storage path on Android.
    """
    override = os.environ.get("FILEFINDER_DATA_DIR", "").strip()
    if override:
        root = Path(override)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create app data dir %s: %s", root, exc)
        return root
    if is_android():
        # Flet exposes client storage via FLET_APP_STORAGE_DATA; fall back to home.
        base = os.environ.get("FLET_APP_STORAGE_DATA")
        root = Path(base) if base else Path.home() / ".filefinder"
    elif is_windows():
        base = os.environ.get("APPDATA")
        root = (Path(base) if base else Path.home() / "AppData" / "Roaming") / "FileFinder"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_ID
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create app data dir %s: %s", root, exc)
    return root


def get_model_dir() -> Path:
    """Return the directory holding downloaded embedding models."""
    path = get_app_data_dir() / "models"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create model dir %s: %s", path, exc)
    return path


def get_model_name() -> str:
    """Return the embedding model directory name for the current platform."""
    return _MODEL_NAME_ANDROID if is_android() else _MODEL_NAME_DESKTOP


# ----------------------------------------------------------------------
# API-key encryption (Fernet, machine-bound)
# ----------------------------------------------------------------------


def _machine_id() -> bytes:
    """Build a stable per-machine identifier from the MAC and node name."""
    raw = f"{uuid.getnode()}:{platform.node()}".encode()
    return raw


def _get_fernet() -> Any:
    """Return a Fernet instance, deriving/loading the key from ``secret.key``.

    The salt and the derived Fernet key are stored together in ``secret.key``
    (JSON) inside the app data dir. The key is derived from the machine id via
    PBKDF2-HMAC-SHA256.

    Raises:
        RuntimeError: if the ``cryptography`` package is unavailable.
    """
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise RuntimeError(
            "cryptography package is required for API-key encryption"
        ) from exc

    secret_path = get_app_data_dir() / "secret.key"
    if secret_path.exists():
        try:
            data = json.loads(secret_path.read_text(encoding="utf-8"))
            return Fernet(data["key"].encode("ascii"))
        except Exception as exc:  # noqa: BLE001 - regenerate on corruption
            logger.warning("Invalid secret.key, regenerating: %s", exc)

    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    key = base64.urlsafe_b64encode(kdf.derive(_machine_id()))
    try:
        secret_path.write_text(
            json.dumps({"salt": salt.hex(), "key": key.decode("ascii")}),
            encoding="utf-8",
        )
        try:
            os.chmod(secret_path, 0o600)
        except OSError:
            pass  # best-effort on non-POSIX filesystems
    except OSError as exc:
        logger.warning("Could not persist secret.key: %s", exc)
    return Fernet(key)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a string for at-rest storage; returns "" on failure."""
    if not plaintext:
        return ""
    try:
        return _get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.error("Encryption failed: %s", exc)
        return ""


def decrypt_secret(token: str) -> str:
    """Decrypt a string produced by :func:`encrypt_secret`; "" on failure."""
    if not token:
        return ""
    try:
        return _get_fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decryption failed (key changed?): %s", exc)
        return ""


def _machine_key_hash() -> str:
    """Short hash of the machine id, useful for diagnostics."""
    return hashlib.sha256(_machine_id()).hexdigest()[:12]


# ----------------------------------------------------------------------
# Config dataclass + persistence
# ----------------------------------------------------------------------


@dataclass
class Config:
    """User configuration persisted to ``config.json`` in the app data dir."""

    monitored_dirs: list[str] = field(default_factory=list)
    theme: str = "dark"
    language: str = DEFAULT_LANGUAGE
    llm_provider: str = "deepseek"  # "deepseek" | "ollama" | "custom" | "off"
    llm_api_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""  # decrypted in memory; encrypted at rest
    llm_model: str = "deepseek-chat"
    embedding_model: str = ""
    auto_update_check: bool = True

    def __post_init__(self) -> None:
        if not self.embedding_model:
            self.embedding_model = get_model_name()


def _config_path() -> Path:
    return get_app_data_dir() / "config.json"


def _get_fixed_drive_roots() -> list[str]:
    """Return roots of local fixed drives other than the system drive.

    Windows-only (e.g. ``D:\\``); returns an empty list elsewhere or on
    any detection failure. Removable/CD/network drives are excluded.
    """
    if not is_windows():
        return []
    roots: list[str] = []
    try:
        import ctypes
        import string

        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
        system_drive = (os.environ.get("SystemDrive") or "C:").upper()
        drive_fixed = 3
        for letter in string.ascii_uppercase:
            root = f"{letter}:{os.sep}"
            if root.upper().startswith(system_drive):
                continue
            if get_drive_type(root) == drive_fixed:
                roots.append(root)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fixed drive detection failed: %s", exc)
    return roots


def get_default_monitored_dirs() -> list[str]:
    """Return standard user folders to index automatically on first run.

    Includes the user's standard folders plus the roots of any additional
    fixed drives (e.g. ``D:\\``) so documents stored outside the system
    drive are covered too. Only existing directories are returned. Android
    returns an empty list (folder access goes through SAF and must be
    user-granted).
    """
    if is_android():
        return []
    home = Path.home()
    names = ("Documents", "Desktop", "Downloads", "Pictures", "Music", "Videos")
    dirs = [str(home / name) for name in names if (home / name).is_dir()]
    dirs.extend(_get_fixed_drive_roots())
    return dirs


def load_config() -> Config:
    """Load the config from disk, returning defaults when missing/corrupt.

    On first run (no ``config.json`` yet) the monitored folders are seeded
    with :func:`get_default_monitored_dirs` so indexing works out of the box.
    """
    path = _config_path()
    if not path.exists():
        return Config(monitored_dirs=get_default_monitored_dirs())
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse config.json, using defaults: %s", exc)
        return Config()
    known = set(Config.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = Config(**filtered)
    cfg.llm_api_key = decrypt_secret(str(data.get("llm_api_key", "")))
    return cfg


def save_config(cfg: Config) -> None:
    """Persist the config to disk, encrypting the API key at rest."""
    data = asdict(cfg)
    data["llm_api_key"] = encrypt_secret(cfg.llm_api_key)
    try:
        _config_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.error("Could not save config.json: %s", exc)


_config_lock = threading.RLock()
_config: Config | None = None


def get_config() -> Config:
    """Return the module-level :class:`Config` singleton (loaded lazily)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


# ----------------------------------------------------------------------
# i18n
# ----------------------------------------------------------------------


def get_assets_dir() -> Path:
    """Return the assets directory, handling frozen (PyInstaller/flet build) apps."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass) / "assets"
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir / "assets", exe_dir / "data" / "assets"):
            if candidate.is_dir():
                return candidate
        return exe_dir / "assets"
    # src/filefinder/core/config.py -> package dir is parents[1]
    return Path(__file__).resolve().parents[1] / "assets"


_i18n_lock = threading.RLock()
_i18n_cache: dict[str, dict[str, str]] = {}
_current_language: str | None = None


def load_i18n(lang: str) -> dict[str, str]:
    """Load the translation dict for ``lang`` from ``assets/i18n/<lang>.json``.

    Returns an empty dict when the file is missing or invalid.
    """
    with _i18n_lock:
        if lang in _i18n_cache:
            return _i18n_cache[lang]
    path = get_assets_dir() / "i18n" / f"{lang}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        strings = {str(k): str(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        if lang in SUPPORTED_LANGUAGES:
            logger.warning("Could not load i18n file %s: %s", path, exc)
        strings = {}
    with _i18n_lock:
        _i18n_cache[lang] = strings
    return strings


def set_language(lang: str) -> None:
    """Set the active UI language (does not persist; save via :func:`save_config`)."""
    global _current_language
    with _i18n_lock:
        _current_language = lang
        get_config().language = lang
    load_i18n(lang)


def _active_language() -> str:
    with _i18n_lock:
        if _current_language:
            return _current_language
    return get_config().language or DEFAULT_LANGUAGE


def t(key: str, **kwargs: Any) -> str:
    """Translate ``key`` into the active language.

    Fallback order: active language → zh (default) → en → the key itself.
    ``kwargs`` are interpolated with :meth:`str.format` when provided.
    """
    lang = _active_language()
    text: str | None = load_i18n(lang).get(key)
    if text is None and lang != DEFAULT_LANGUAGE:
        text = load_i18n(DEFAULT_LANGUAGE).get(key)
    if text is None and lang != FALLBACK_LANGUAGE:
        text = load_i18n(FALLBACK_LANGUAGE).get(key)
    if text is None:
        text = key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("i18n format failed for key %r: %s", key, exc)
    return text


# ----------------------------------------------------------------------
# Path safety
# ----------------------------------------------------------------------


def is_safe_path(p: str | os.PathLike[str], roots: list[str] | None = None) -> bool:
    """Check that ``p`` resolves inside one of the monitored root directories.

    Rejects ``..`` traversal escaping the monitored roots.

    Args:
        p: path to validate.
        roots: allowed roots; defaults to the configured ``monitored_dirs``.

    Returns:
        True when the resolved path is inside at least one monitored root.
    """
    if roots is None:
        roots = get_config().monitored_dirs
    if not roots:
        return False
    try:
        resolved = Path(p).resolve()
    except (OSError, ValueError) as exc:
        logger.warning("Unresolvable path %r: %s", p, exc)
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    return False
