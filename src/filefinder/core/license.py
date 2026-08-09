"""Freemium license key validation, device binding and tier limits.

License keys have the form ``FF3-G1-G2-G3`` where each group is 4 chars
from :data:`ALPHABET` (32 chars, no ambiguous ``0/1/I/O``). ``G3`` is an
HMAC-SHA256 checksum over ``'FF3' + G1 + G2`` keyed with a secret from
:mod:`filefinder.core._license_secret` (gitignored; a built-in dev secret
with a logged warning is used as fallback so tests/CI keep working).

Subscription expiry encoding (offline, no server check): the FIRST 2 chars
of ``G1`` encode the expiry month as a month-index in base-32 over
:data:`ALPHABET`, where ``month_index = (year - 2026) * 12 + (month - 1)``,
``char0 = ALPHABET[month_index // 32]`` and ``char1 = ALPHABET[month_index
% 32]`` (see :func:`_encode_month` / :func:`_decode_month`). A key is valid
THROUGH THE END of the encoded month (local date). Because the checksum
covers ``G1``, the expiry is tamper-protected. NOTE (accepted limitation):
offline expiry relies on the system clock; clock-rollback is an accepted
risk.

Activation state is persisted to ``license.json`` in the app data dir and
bound to a device id (CPU + disk serial hash on Windows, ``/etc/machine-id``
on Linux, a persistent UUID elsewhere / as fallback). Tier limits are
enforced by the core modules via :func:`get_limit` / :func:`check_feature`
and the kv-backed usage counters.
"""

from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import logging
import re
import secrets
import subprocess
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from filefinder.core import config, constants

logger = logging.getLogger(__name__)

# Key alphabet: 32 chars, excludes ambiguous 0/1/I/O.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# HMAC secret for the G3 checksum. The real secret lives in the gitignored
# ``_license_secret.py``; without it a built-in dev secret is used so tests
# and CI keep working (keys generated with one secret fail with the other).
try:
    from filefinder.core._license_secret import SECRET
except ImportError:
    SECRET = b"ff3-development-only-secret-not-for-production"
    logger.warning(
        "_license_secret.py not found; using the built-in dev secret. "
        "Keys generated now will not validate in production builds."
    )

KEY_PATTERN = re.compile(r"^FF3-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$")

LICENSE_FILENAME = "license.json"
MACHINE_ID_FILENAME = "machine_id"

# Epoch year for the month-index expiry encoding (see module docstring).
EPOCH_YEAR = 2026

# Tier limits. ``None`` means unlimited.
LIMITS: dict[str, dict[str, Any]] = {
    "free": {
        "max_files": None,
        "max_results": None,
        "ocr_monthly": None,
        "max_watch_dirs": constants.FREE_WATCH_DIRS,
        "chat_queries": constants.FREE_CHAT_QUERIES,
        "rag": False,
        "export": False,
    },
    "premium": {
        "max_files": None,
        "max_results": None,
        "ocr_monthly": None,
        "max_watch_dirs": None,
        "chat_queries": None,
        "rag": True,
        "export": True,
    },
}

# kv key prefixes for usage counters.
_OCR_COUNTER_PREFIX = "ocr_"
_CHAT_COUNTER_KEY = "chat_queries"


# ----------------------------------------------------------------------
# Key format / checksum
# ----------------------------------------------------------------------

def is_valid_format(key: str) -> bool:
    """Return True if ``key`` matches the ``FF3-XXXX-XXXX-XXXX`` format."""
    return bool(KEY_PATTERN.match((key or "").strip()))


def _checksum_group(g1: str, g2: str) -> str:
    """Return the 4-char HMAC-SHA256 checksum group for ``'FF3'+G1+G2``.

    Char ``i`` of ``G3`` is ``ALPHABET[digest[i] % 32]`` where ``digest``
    is ``hmac.new(SECRET, ('FF3'+G1+G2).encode(), hashlib.sha256)``.
    """
    digest = hmac.new(
        SECRET, ("FF3" + g1 + g2).encode("utf-8"), hashlib.sha256
    ).digest()
    return "".join(ALPHABET[digest[i] % 32] for i in range(4))


def validate_checksum(key: str) -> bool:
    """Return True if ``key`` has a valid format AND HMAC checksum group.

    The whole ``G3`` group must equal :func:`_checksum_group` of ``G1``/``G2``.
    """
    key = (key or "").strip().upper()
    if not is_valid_format(key):
        return False
    _prefix, g1, g2, g3 = key.split("-")
    return hmac.compare_digest(g3, _checksum_group(g1, g2))


def generate_key(expiry: date | None = None) -> str:
    """Generate a new random license key with a valid checksum.

    Args:
        expiry: expiry month (any day within it); the key stays valid
            through the last day of that month. Defaults to the end of
            NEXT month (comfortable default for tests/admin flows).
    """
    if expiry is None:
        today = date.today()
        month_index = (today.year - EPOCH_YEAR) * 12 + today.month  # +1 month
        expiry = date(
            EPOCH_YEAR + month_index // 12,
            month_index % 12 + 1,
            1,
        )
    g1 = _encode_month(expiry) + "".join(secrets.choice(ALPHABET) for _ in range(2))
    g2 = "".join(secrets.choice(ALPHABET) for _ in range(4))
    return f"FF3-{g1}-{g2}-{_checksum_group(g1, g2)}"


# ----------------------------------------------------------------------
# Expiry encoding (first 2 chars of G1, base-32 month index)
# ----------------------------------------------------------------------

def _encode_month(d: date) -> str:
    """Encode the month of ``d`` as 2 :data:`ALPHABET` chars.

    ``month_index = (year - EPOCH_YEAR) * 12 + (month - 1)`` is written as
    ``ALPHABET[month_index // 32] + ALPHABET[month_index % 32]``.
    """
    month_index = (d.year - EPOCH_YEAR) * 12 + (d.month - 1)
    return ALPHABET[month_index // 32] + ALPHABET[month_index % 32]


def _decode_month(g1: str) -> date | None:
    """Decode the first 2 chars of ``g1`` into the last day of that month.

    Returns None when ``g1`` is malformed.
    """
    if len(g1) < 2 or g1[0] not in ALPHABET or g1[1] not in ALPHABET:
        return None
    month_index = ALPHABET.index(g1[0]) * 32 + ALPHABET.index(g1[1])
    year = EPOCH_YEAR + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def get_key_expiry(key: str) -> date | None:
    """Return the expiry date encoded in ``key`` (last day of the month).

    The key is valid THROUGH THE END of that month (local date). Returns
    None when the key is malformed. The checksum is NOT verified here.
    """
    key = (key or "").strip().upper()
    if not is_valid_format(key):
        return None
    return _decode_month(key.split("-")[1])


# ----------------------------------------------------------------------
# Blacklist
# ----------------------------------------------------------------------

def load_blacklist() -> list[str]:
    """Return the revoked key list from ``assets/blacklist.json``.

    A missing or unreadable file yields an empty list.
    """
    path = config.get_assets_dir() / "blacklist.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        revoked = data.get("revoked", [])
        return [str(k).strip().upper() for k in revoked]
    except Exception as exc:  # noqa: BLE001 - missing file is fine
        logger.debug("Blacklist unavailable (%s); treating as empty.", exc)
        return []


def is_blacklisted(key: str) -> bool:
    """Return True if ``key`` has been revoked."""
    return (key or "").strip().upper() in load_blacklist()


# ----------------------------------------------------------------------
# Device id
# ----------------------------------------------------------------------

def _run_powershell(command: str) -> str | None:
    """Run a PowerShell command and return its trimmed stdout, or None."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 - no powershell, etc.
        logger.debug("PowerShell failed (%s): %s", command, exc)
        return None
    if result.returncode != 0:
        logger.debug("PowerShell returned %s for %s", result.returncode, command)
        return None
    output = (result.stdout or "").strip()
    return output or None


def _windows_device_id() -> str | None:
    """Hash of CPU ProcessorId + disk SerialNumber, or None on failure."""
    cpu = _run_powershell("(Get-CimInstance Win32_Processor).ProcessorId")
    disk = _run_powershell("(Get-CimInstance Win32_DiskDrive).SerialNumber")
    if not cpu and not disk:
        return None
    raw = f"{cpu or ''}|{disk or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _uuid_device_id() -> str:
    """Return a persistent random UUID stored in the app data dir."""
    path = config.get_app_data_dir() / MACHINE_ID_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    new_id = uuid.uuid4().hex
    try:
        path.write_text(new_id, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist machine id: %s", exc)
    return new_id


def get_device_id() -> str:
    """Return a stable device identifier for license binding.

    Windows: sha256(CPU ProcessorId + disk serial) hex[:24], falling back
    to a persistent UUID on any failure. Linux: ``/etc/machine-id``
    (UUID fallback). Android: persistent UUID.
    """
    if config.is_windows():
        device_id = _windows_device_id()
        if device_id:
            return device_id
        logger.info("Falling back to UUID device id.")
        return _uuid_device_id()
    if config.is_android():
        return _uuid_device_id()
    try:
        machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        if machine_id:
            return machine_id
    except OSError as exc:
        logger.debug("/etc/machine-id unavailable: %s", exc)
    return _uuid_device_id()


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def _license_path() -> Path:
    """Return the path of the persisted license file."""
    return config.get_app_data_dir() / LICENSE_FILENAME


def _load_license() -> dict[str, Any] | None:
    """Load the persisted license record, or None when missing/corrupt."""
    path = _license_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - corrupt file
        logger.warning("Could not parse %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _save_license(record: dict[str, Any]) -> None:
    """Persist the license record to disk."""
    try:
        _license_path().write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.error("Could not save license: %s", exc)


# ----------------------------------------------------------------------
# Public activation API
# ----------------------------------------------------------------------

def activate(key: str) -> tuple[bool, str]:
    """Activate a license key on this device.

    Args:
        key: the license key (whitespace/case tolerant).

    Returns:
        ``(success, code)`` where code is one of ``'ok'``,
        ``'invalid_format'``, ``'invalid_checksum'``, ``'blacklisted'``,
        ``'expired'`` or ``'device_mismatch'``. The first activation binds
        the key to the current device id; re-activating the same key on
        the same device is idempotent, on a different device it fails with
        ``'device_mismatch'``. A new valid key replaces the old one.
        ``'expired'`` means the key's encoded expiry month has passed.
    """
    key = (key or "").strip().upper()
    if not is_valid_format(key):
        logger.info("Activation rejected: invalid format")
        return False, "invalid_format"
    if not validate_checksum(key):
        logger.info("Activation rejected: invalid checksum")
        return False, "invalid_checksum"
    if is_blacklisted(key):
        logger.info("Activation rejected: key is revoked")
        return False, "blacklisted"
    expiry = get_key_expiry(key)
    if expiry is None or expiry < date.today():
        logger.info("Activation rejected: key expired")
        return False, "expired"

    device_id = get_device_id()
    existing = _load_license()
    if existing and existing.get("key") == key:
        if existing.get("device_id") != device_id:
            logger.warning("Activation rejected: key bound to another device")
            return False, "device_mismatch"
        logger.info("License key already active on this device")
        return True, "ok"

    _save_license(
        {
            "key": key,
            "activated_at": datetime.now().isoformat(timespec="seconds"),
            "device_id": device_id,
            "tier": "premium",
            "checksum_valid": True,
            "expires_at": expiry.isoformat(),
        }
    )
    logger.info("License activated (tier=premium, expires_at=%s)", expiry)
    return True, "ok"


def deactivate() -> None:
    """Remove the persisted license, reverting to the free tier."""
    try:
        _license_path().unlink(missing_ok=True)
        logger.info("License deactivated")
    except OSError as exc:
        logger.warning("Could not remove license file: %s", exc)


def is_pro() -> bool:
    """Return True when a valid, unexpired premium license is active.

    The checksum AND the encoded expiry month are recomputed from the
    stored key on every call. An expired stored license is treated as
    free; the license file itself is left untouched.
    """
    record = _load_license()
    if not record:
        return False
    if record.get("tier") != "premium":
        return False
    key = str(record.get("key", ""))
    if not validate_checksum(key):
        return False
    expiry = get_key_expiry(key)
    return expiry is not None and expiry >= date.today()


def get_tier() -> str:
    """Return the current tier: ``'premium'`` or ``'free'``."""
    return "premium" if is_pro() else "free"


def _mask_key(key: str) -> str:
    """Mask a key for display, keeping the prefix and first group."""
    parts = (key or "").split("-")
    if len(parts) != 4:
        return ""
    return f"{parts[0]}-{parts[1]}-****-****"


def get_status() -> dict[str, Any]:
    """Return the license status for display.

    Returns:
        Dict with ``tier``, ``key_masked``, ``activated_at``,
        ``device_id``, ``expires_at`` (ISO date) and ``days_remaining``
        keys. On the free tier the display fields are None, except
        ``expires_at``/``days_remaining`` which are still derived from a
        stored (now expired) key when one exists — a negative
        ``days_remaining`` signals an expired subscription to the UI.
    """
    record = _load_license()
    expiry = get_key_expiry(str(record.get("key", ""))) if record else None
    expires_at = expiry.isoformat() if expiry else None
    days_remaining = (expiry - date.today()).days if expiry else None
    if not record or not is_pro():
        return {
            "tier": "free",
            "key_masked": None,
            "activated_at": None,
            "device_id": None,
            "expires_at": expires_at,
            "days_remaining": days_remaining,
        }
    return {
        "tier": "premium",
        "key_masked": _mask_key(str(record.get("key", ""))),
        "activated_at": record.get("activated_at"),
        "device_id": record.get("device_id"),
        "expires_at": expires_at,
        "days_remaining": days_remaining,
    }


# ----------------------------------------------------------------------
# Tier limits & usage counters
# ----------------------------------------------------------------------

def get_limit(name: str) -> int | None:
    """Return the numeric limit ``name`` for the current tier.

    Returns:
        The limit, or None when unlimited (premium) or unknown.
    """
    value = LIMITS.get(get_tier(), LIMITS["free"]).get(name)
    return value if isinstance(value, int) else None


def check_feature(name: str) -> bool:
    """Return True if the boolean feature ``name`` (rag/export) is allowed."""
    return bool(LIMITS.get(get_tier(), LIMITS["free"]).get(name, False))


def _counter_kv_key(name: str) -> str:
    """Map a logical counter name to its kv key.

    ``'ocr'`` / ``'ocr_monthly'`` map to a per-month key ``ocr_YYYY-MM``;
    every other name (e.g. ``'chat_queries'``) is used as-is.
    """
    if name in ("ocr", "ocr_monthly"):
        return f"{_OCR_COUNTER_PREFIX}{datetime.now():%Y-%m}"
    return name


def get_counter(name: str) -> int:
    """Return the current value of usage counter ``name`` (0 on failure)."""
    try:
        from filefinder.core.db import get_db  # lazy: avoids import cycle

        raw = get_db().get_kv(_counter_kv_key(name))
        return int(raw) if raw else 0
    except Exception as exc:  # noqa: BLE001 - counters must never break flows
        logger.warning("Could not read counter %s: %s", name, exc)
        return 0


def increment_counter(name: str) -> int:
    """Increment usage counter ``name`` and return the new value.

    Returns:
        The new counter value (best-effort; 0 when persistence failed).
    """
    new_value = get_counter(name) + 1
    try:
        from filefinder.core.db import get_db  # lazy: avoids import cycle

        get_db().set_kv(_counter_kv_key(name), str(new_value))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist counter %s: %s", name, exc)
    return new_value
