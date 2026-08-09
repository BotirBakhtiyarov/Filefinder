"""Tests for the freemium license/tier system (core/license.py) and its
enforcement hooks in search and watcher.

All tests run fully offline against an isolated ``FILEFINDER_DATA_DIR``.
"""

from __future__ import annotations

import calendar
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from filefinder.core import license as license_mod
from filefinder.core.config import t

KEY_RE = re.compile(r"^FF3-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$")


def _last_day(year: int, month: int) -> date:
    """Return the last day of the given month."""
    return date(year, month, calendar.monthrange(year, month)[1])


def _shift_months(base: date, months: int) -> date:
    """Return the 1st of ``base``'s month shifted by ``months`` months."""
    month_index = base.year * 12 + (base.month - 1) + months
    return date(month_index // 12, month_index % 12 + 1, 1)


# ----------------------------------------------------------------------
# Format & checksum
# ----------------------------------------------------------------------

def test_generate_key_format_and_checksum() -> None:
    """Generated keys match the format and carry a valid checksum."""
    for _ in range(50):
        key = license_mod.generate_key()
        assert KEY_RE.match(key), key
        assert license_mod.is_valid_format(key)
        assert license_mod.validate_checksum(key)
        # No ambiguous characters ever appear.
        assert not set(key) & set("01IO")


def test_validate_checksum_rejects_broken_keys() -> None:
    """Bad formats and bad checksum characters are rejected."""
    assert not license_mod.validate_checksum("")
    assert not license_mod.validate_checksum("FF3-AAAA-AAAA")  # too short
    assert not license_mod.validate_checksum("XX3-AAAA-AAAA-AAAA")  # prefix
    assert not license_mod.validate_checksum("FF3-AAAA-AAAA-AAAAA")  # too long
    assert not license_mod.validate_checksum("ff3-aaaa-aaaa-aaaa")  # lowercase

    # Valid format but wrong checksum char: flip G3[0] of a real key.
    key = license_mod.generate_key()
    g = key.split("-")
    replacement = "A" if g[3][0] != "A" else "B"
    broken = f"{g[0]}-{g[1]}-{g[2]}-{replacement}{g[3][1:]}"
    assert license_mod.is_valid_format(broken)
    assert not license_mod.validate_checksum(broken)


def test_generate_validate_roundtrip() -> None:
    """generate_key -> validate_checksum roundtrip."""
    key = license_mod.generate_key()
    assert license_mod.validate_checksum(key)


def test_checksum_is_hmac_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tampering with ANY G3 char invalidates the key; a wrong secret fails."""
    key = license_mod.generate_key()
    g = key.split("-")
    for i in range(4):
        replacement = "A" if g[3][i] != "A" else "B"
        tampered_g3 = g[3][:i] + replacement + g[3][i + 1:]
        broken = f"{g[0]}-{g[1]}-{g[2]}-{tampered_g3}"
        assert license_mod.is_valid_format(broken)
        assert not license_mod.validate_checksum(broken)

    # A key generated under a different secret must not validate.
    monkeypatch.setattr(license_mod, "SECRET", b"some-other-secret")
    assert not license_mod.validate_checksum(key)
    foreign = license_mod.generate_key()
    monkeypatch.undo()
    assert license_mod.validate_checksum(key)
    assert not license_mod.validate_checksum(foreign)


# ----------------------------------------------------------------------
# Expiry encoding
# ----------------------------------------------------------------------

def test_expiry_encode_decode_roundtrip() -> None:
    """_encode_month/_decode_month roundtrip for a range of months."""
    for year, month in [
        (2026, 1), (2026, 8), (2026, 12), (2027, 1), (2027, 6), (2030, 11),
    ]:
        encoded = license_mod._encode_month(date(year, month, 15))
        assert len(encoded) == 2
        assert license_mod._decode_month(encoded + "ZZ") == _last_day(year, month)


def test_get_key_expiry() -> None:
    """get_key_expiry returns the encoded month-end, None when malformed."""
    key = license_mod.generate_key(expiry=date(2027, 3, 15))
    assert license_mod.get_key_expiry(key) == date(2027, 3, 31)
    assert license_mod.validate_checksum(key)  # checksum covers the expiry

    assert license_mod.get_key_expiry("") is None
    assert license_mod.get_key_expiry("nope") is None
    assert license_mod.get_key_expiry("FF3-AAAA-AAAA") is None


def test_generate_key_default_expiry_is_next_month() -> None:
    """generate_key() defaults to the end of next month."""
    key = license_mod.generate_key()
    first_of_next = _shift_months(date.today(), 1)
    assert license_mod.get_key_expiry(key) == _last_day(
        first_of_next.year, first_of_next.month
    )


def test_generate_keys_cli_valid_months_and_expires(tmp_path: Path) -> None:
    """The admin CLI encodes --valid-months / --expires correctly."""
    script = Path(__file__).resolve().parents[1] / "tools" / "generate_keys.py"

    def run(*extra: str) -> list[str]:
        out = tmp_path / "keys.csv"
        result = subprocess.run(
            [sys.executable, str(script), "--count", "2", "--out", str(out),
             *extra],
            check=True, capture_output=True, text=True,
        )
        keys = [line.split()[0] for line in result.stdout.splitlines()]
        assert len(keys) == 2
        for key in keys:
            assert license_mod.validate_checksum(key)
        return keys

    # --valid-months 1: expires at the end of THIS month.
    today = date.today()
    for key in run("--valid-months", "1"):
        assert license_mod.get_key_expiry(key) == _last_day(today.year, today.month)

    # --valid-months 3: expires at the end of (this month + 2).
    first_plus2 = _shift_months(today, 2)
    for key in run("--valid-months", "3"):
        assert license_mod.get_key_expiry(key) == _last_day(
            first_plus2.year, first_plus2.month
        )

    # --expires overrides --valid-months.
    for key in run("--expires", "2026-12", "--valid-months", "9"):
        assert license_mod.get_key_expiry(key) == date(2026, 12, 31)


# ----------------------------------------------------------------------
# Blacklist
# ----------------------------------------------------------------------

def test_blacklist_blocks_revoked_key(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A revoked key fails activation with 'blacklisted'."""
    from filefinder.core import config

    key = license_mod.generate_key()
    fake_assets = tmp_path / "assets"
    fake_assets.mkdir()
    (fake_assets / "blacklist.json").write_text(
        json.dumps({"revoked": [key]}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "get_assets_dir", lambda: fake_assets)
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")

    assert license_mod.is_blacklisted(key)
    assert license_mod.activate(key) == (False, "blacklisted")


def test_missing_blacklist_file_is_empty(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing blacklist.json means no revoked keys."""
    from filefinder.core import config

    monkeypatch.setattr(config, "get_assets_dir", lambda: tmp_path / "nope")
    assert license_mod.load_blacklist() == []
    assert not license_mod.is_blacklisted(license_mod.generate_key())


# ----------------------------------------------------------------------
# Activation flow
# ----------------------------------------------------------------------

def test_activate_ok_and_idempotent(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activation succeeds, persists, and re-activation is idempotent."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    key = license_mod.generate_key()

    assert license_mod.activate(key) == (True, "ok")
    assert license_mod.is_pro()
    assert license_mod.get_tier() == "premium"

    # Idempotent re-activation with the same key on the same device.
    assert license_mod.activate(key) == (True, "ok")

    record = json.loads(
        (app_data_dir / "license.json").read_text(encoding="utf-8")
    )
    assert record["key"] == key
    assert record["device_id"] == "dev-1"
    assert record["tier"] == "premium"
    assert record["checksum_valid"] is True
    assert record["activated_at"]
    assert record["expires_at"] == license_mod.get_key_expiry(key).isoformat()

    status = license_mod.get_status()
    assert status["tier"] == "premium"
    assert status["key_masked"] == key.rsplit("-", 2)[0] + "-****-****"
    assert status["device_id"] == "dev-1"
    assert status["expires_at"] == license_mod.get_key_expiry(key).isoformat()
    assert status["days_remaining"] == (
        license_mod.get_key_expiry(key) - date.today()
    ).days


def test_activate_bad_keys(app_data_dir: Path) -> None:
    """Malformed and bad-checksum keys are rejected with proper codes."""
    assert license_mod.activate("") == (False, "invalid_format")
    assert license_mod.activate("hello") == (False, "invalid_format")

    key = license_mod.generate_key()
    g = key.split("-")
    replacement = "A" if g[3][0] != "A" else "B"
    bad_checksum = f"{g[0]}-{g[1]}-{g[2]}-{replacement}{g[3][1:]}"
    assert license_mod.activate(bad_checksum) == (False, "invalid_checksum")
    assert not license_mod.is_pro()
    assert license_mod.get_tier() == "free"


def test_activate_device_mismatch(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same key on a different device is rejected."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    key = license_mod.generate_key()
    assert license_mod.activate(key) == (True, "ok")

    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-2")
    assert license_mod.activate(key) == (False, "device_mismatch")


def test_activate_expired_key(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key whose encoded expiry month has passed fails with 'expired'."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    last_month = _shift_months(date.today(), -1)
    key = license_mod.generate_key(
        expiry=_last_day(last_month.year, last_month.month)
    )
    assert license_mod.validate_checksum(key)  # well-formed, just expired
    assert license_mod.activate(key) == (False, "expired")
    assert not license_mod.is_pro()
    assert not (app_data_dir / "license.json").exists()


def test_is_pro_false_after_expiry(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored license past its expiry month is treated as free."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    last_month = _shift_months(date.today(), -1)
    expired = _last_day(last_month.year, last_month.month)
    key = license_mod.generate_key(expiry=expired)
    # Simulate a license that was activated while still valid.
    (app_data_dir / "license.json").write_text(
        json.dumps(
            {
                "key": key,
                "activated_at": "2026-01-01T10:00:00",
                "device_id": "dev-1",
                "tier": "premium",
                "checksum_valid": True,
                "expires_at": expired.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    assert not license_mod.is_pro()
    assert license_mod.get_tier() == "free"
    # The file is kept (not deleted) and still reports the expiry.
    assert (app_data_dir / "license.json").exists()
    status = license_mod.get_status()
    assert status["tier"] == "free"
    assert status["expires_at"] == expired.isoformat()
    assert status["days_remaining"] < 0


def test_activate_new_key_replaces_old(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activating a new valid key replaces the previous license."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    key1 = license_mod.generate_key()
    key2 = license_mod.generate_key()
    assert license_mod.activate(key1) == (True, "ok")
    assert license_mod.activate(key2) == (True, "ok")
    record = json.loads(
        (app_data_dir / "license.json").read_text(encoding="utf-8")
    )
    assert record["key"] == key2


def test_deactivate(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deactivation removes the license and reverts to the free tier."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    assert license_mod.activate(license_mod.generate_key()) == (True, "ok")
    assert license_mod.is_pro()

    license_mod.deactivate()
    assert not license_mod.is_pro()
    assert license_mod.get_tier() == "free"
    assert not (app_data_dir / "license.json").exists()
    status = license_mod.get_status()
    assert status == {
        "tier": "free",
        "key_masked": None,
        "activated_at": None,
        "device_id": None,
        "expires_at": None,
        "days_remaining": None,
    }


# ----------------------------------------------------------------------
# Free-tier limits
# ----------------------------------------------------------------------

def test_free_tier_limits() -> None:
    """Only chat/RAG/export/watch-dirs are limited on the free tier."""
    free = license_mod.LIMITS["free"]
    assert free["max_files"] is None
    assert free["max_results"] is None
    assert free["ocr_monthly"] is None
    assert free["max_watch_dirs"] == 1
    assert free["chat_queries"] == 20
    assert free["rag"] is False
    assert free["export"] is False


# ----------------------------------------------------------------------
# Counters
# ----------------------------------------------------------------------

def test_usage_counters(app_data_dir: Path) -> None:
    """Counters increment and persist via the db kv store."""
    assert license_mod.get_counter("chat_queries") == 0
    assert license_mod.increment_counter("chat_queries") == 1
    assert license_mod.increment_counter("chat_queries") == 2
    assert license_mod.get_counter("chat_queries") == 2

    assert license_mod.get_counter("ocr_monthly") == 0
    assert license_mod.increment_counter("ocr_monthly") == 1
    assert license_mod.get_counter("ocr_monthly") == 1


# ----------------------------------------------------------------------
# Search: no free-tier result cap
# ----------------------------------------------------------------------

def _build_search_env(monkeypatch: pytest.MonkeyPatch, doc_count: int = 60):
    """Wire a SearchEngine to a fake embedder/store with ``doc_count`` docs."""
    from tests.fakes import (
        FakeEmbedder,
        FakeVectorStore,
        get_shared_db,
        make_search_engine,
    )

    store = FakeVectorStore()
    db = get_shared_db(monkeypatch, store)
    for i in range(doc_count):
        db.upsert_file(
            path=f"F:/docs/file{i}.txt",
            filename=f"file{i}.txt",
            extension=".txt",
            size=10,
            modified_time=1.0,
            content_hash=f"hash{i}",
            text_preview="common searchterm content",
        )
    engine = make_search_engine(monkeypatch, db, FakeEmbedder(), store)
    return engine


def test_search_top_k_not_capped_for_free(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free tier keeps the requested top_k (no result cap)."""
    engine = _build_search_env(monkeypatch, doc_count=60)
    results = engine.search("searchterm", top_k=100)
    assert len(results) == 60


def test_search_top_k_uncapped_for_premium(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premium tier keeps the requested top_k."""
    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    assert license_mod.activate(license_mod.generate_key()) == (True, "ok")
    engine = _build_search_env(monkeypatch, doc_count=60)
    results = engine.search("searchterm", top_k=100)
    assert len(results) == 60


# ----------------------------------------------------------------------
# Enforcement: watcher free tier watches a single directory
# ----------------------------------------------------------------------

class _RecordingWatcher:
    """FileWatcher stand-in that records the dirs it was started with."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def start(self, dirs: list, on_change_cb) -> bool:  # noqa: ANN001
        self.calls.append(list(dirs))
        return True

    def stop(self) -> None:
        pass


def test_watcher_free_single_dir(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free tier only watches the first of the requested directories."""
    from filefinder.core import watcher

    recorder = _RecordingWatcher()
    monkeypatch.setattr(watcher, "FileWatcher", lambda: recorder)
    monkeypatch.setattr(watcher, "_watcher", None)

    assert watcher.start(["/a", "/b", "/c"], lambda kind, path: None) is True
    assert recorder.calls == [["/a"]]


def test_watcher_premium_all_dirs(
    app_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premium tier watches every requested directory."""
    from filefinder.core import watcher

    monkeypatch.setattr(license_mod, "get_device_id", lambda: "dev-1")
    assert license_mod.activate(license_mod.generate_key()) == (True, "ok")

    recorder = _RecordingWatcher()
    monkeypatch.setattr(watcher, "FileWatcher", lambda: recorder)
    monkeypatch.setattr(watcher, "_watcher", None)

    assert watcher.start(["/a", "/b"], lambda kind, path: None) is True
    assert recorder.calls == [["/a", "/b"]]


# ----------------------------------------------------------------------
# Enforcement: chat free tier
# ----------------------------------------------------------------------

def test_chat_rag_is_pro_only(app_data_dir: Path) -> None:
    """Free tier with use_rag=True gets the pro-only message, no LLM call."""
    from filefinder.core.chat import ChatEngine

    engine = ChatEngine()
    reply = engine.send_message("hello", conversation_id="c1", use_rag=True)
    assert reply == t("license.pro_only")
    # RAG rejection must not consume a chat query.
    assert license_mod.get_counter("chat_queries") == 0


def test_chat_free_query_limit(app_data_dir: Path) -> None:
    """Free tier chat queries are counted and blocked past the cap."""
    from filefinder.core.chat import ChatEngine

    engine = ChatEngine()
    # Simulate a user already at the cap: the next query must be rejected.
    for _ in range(license_mod.get_limit("chat_queries") or 20):
        license_mod.increment_counter("chat_queries")
    reply = engine.send_message("hello", conversation_id="c1", use_rag=False)
    assert reply == t("license.chat_limit")


def test_chat_stream_rag_is_pro_only(app_data_dir: Path) -> None:
    """The streaming API yields the pro-only message for free-tier RAG."""
    from filefinder.core.chat import ChatEngine

    engine = ChatEngine()
    chunks = list(engine.send_message_stream("hello", conversation_id="c1"))
    assert chunks == [("answer", t("license.pro_only"))]
