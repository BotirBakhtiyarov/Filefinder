"""Unit tests for filefinder.core.deps (Tesseract auto-install).

All subprocess calls and downloads are mocked: no network, no real installs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from filefinder.core import deps, ocr


@pytest.fixture(autouse=True)
def _reset_ocr_resolution(monkeypatch: pytest.MonkeyPatch):
    """Ensure ocr resolution cache is clean before/after every test."""
    ocr.reset_resolution()
    yield
    ocr.reset_resolution()


def test_is_tesseract_installed_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_tesseract_installed delegates to ocr.is_ocr_available."""
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: True)
    assert deps.is_tesseract_installed() is True
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: False)
    assert deps.is_tesseract_installed() is False


def test_ensure_tesseract_short_circuits_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No subprocess or download runs when tesseract is already present."""
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: True)

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("subprocess/download must not be invoked")

    monkeypatch.setattr(deps.subprocess, "run", _boom)
    monkeypatch.setattr(deps.shutil, "which", _boom)

    assert deps.ensure_tesseract() is True


def test_ensure_tesseract_non_windows_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-install is Windows-only; elsewhere it gives up gracefully."""
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: False)
    assert deps.ensure_tesseract() is False


def test_ensure_language_data_skips_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No download when all requested traineddata files already exist."""
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    for lang in ("eng", "chi_sim"):
        (tessdata / f"{lang}.traineddata").write_bytes(b"x")
    fake_cmd = tmp_path / "tesseract.exe"
    fake_cmd.write_bytes(b"")
    monkeypatch.setattr(ocr, "get_tesseract_cmd", lambda: str(fake_cmd))

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("download must not be invoked")

    monkeypatch.setattr(
        "filefinder.updater.downloader.download", _boom, raising=False
    )

    assert deps.ensure_language_data() is True


def test_ensure_language_data_downloads_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Missing languages are downloaded into the resolved tessdata dir."""
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"x")  # chi_sim missing
    fake_cmd = tmp_path / "tesseract.exe"
    fake_cmd.write_bytes(b"")
    monkeypatch.setattr(ocr, "get_tesseract_cmd", lambda: str(fake_cmd))

    downloaded: list[str] = []

    def fake_download(url, dest, progress_cb=None, sha256=None):
        downloaded.append(str(url))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake")
        if progress_cb is not None:
            progress_cb(4, 4)
        return Path(dest)

    monkeypatch.setattr(
        "filefinder.updater.downloader.download", fake_download, raising=False
    )

    assert deps.ensure_language_data() is True
    assert downloaded == [deps._TESSDATA_URL_TEMPLATE.format(lang="chi_sim")]
    assert (tessdata / "chi_sim.traineddata").is_file()


def test_ensure_language_data_reports_missing_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Returns False (never raises) when a language download fails."""
    fake_cmd = tmp_path / "tesseract.exe"
    fake_cmd.write_bytes(b"")
    monkeypatch.setattr(ocr, "get_tesseract_cmd", lambda: str(fake_cmd))

    def failing_download(url, dest, progress_cb=None, sha256=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "filefinder.updater.downloader.download", failing_download,
        raising=False,
    )

    assert deps.ensure_language_data() is False


def test_ensure_language_data_no_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns False (never raises) when no tesseract binary is found."""
    monkeypatch.setattr(ocr, "get_tesseract_cmd", lambda: None)
    assert deps.ensure_language_data() is False


# ----------------------------------------------------------------------
# Poppler helpers
# ----------------------------------------------------------------------


def test_get_poppler_bin_dir_managed_library_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Detects the managed install under ~/.filefinder/poppler/Library/bin."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    bin_dir = tmp_path / ".filefinder" / "poppler" / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "pdfinfo.exe").write_bytes(b"x")
    (bin_dir / "pdftoppm.exe").write_bytes(b"x")
    assert deps.get_poppler_bin_dir() == bin_dir


def test_get_poppler_bin_dir_managed_plain_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Detects the zip-layout variation ~/.filefinder/poppler/bin."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    bin_dir = tmp_path / ".filefinder" / "poppler" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "pdfinfo.exe").write_bytes(b"x")
    (bin_dir / "pdftoppm.exe").write_bytes(b"x")
    assert deps.get_poppler_bin_dir() == bin_dir


def test_get_poppler_bin_dir_from_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Falls back to pdfinfo found on PATH."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no managed install
    fake = tmp_path / "elsewhere" / "pdfinfo.exe"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    monkeypatch.setattr(deps.shutil, "which", lambda name: str(fake))
    assert deps.get_poppler_bin_dir() == fake.parent.resolve()


def test_get_poppler_bin_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Returns None when poppler is nowhere to be found."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    assert deps.get_poppler_bin_dir() is None


def test_ensure_poppler_short_circuits_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No download or subprocess runs when poppler is already present."""
    monkeypatch.setattr(deps, "get_poppler_bin_dir", lambda: tmp_path)

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("subprocess/download must not be invoked")

    monkeypatch.setattr(deps.subprocess, "run", _boom)
    assert deps.ensure_poppler() is True


def test_ensure_poppler_non_windows_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-install is Windows-only; elsewhere it gives up gracefully."""
    monkeypatch.setattr(deps, "get_poppler_bin_dir", lambda: None)
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    assert deps.ensure_poppler() is False


def test_ensure_poppler_returns_false_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns False (never raises) when the release download fails."""
    monkeypatch.setattr(deps, "get_poppler_bin_dir", lambda: None)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "_download_poppler", lambda progress_cb: None)
    statuses: list[str] = []
    assert deps.ensure_poppler(status_cb=statuses.append) is False
    assert statuses == ["deps.status_poppler"]


def test_ensure_pdf_ocr_combines_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_pdf_ocr is True only when both components succeed."""
    monkeypatch.setattr(deps, "ensure_tesseract", lambda **kw: True)
    monkeypatch.setattr(deps, "ensure_poppler", lambda **kw: True)
    assert deps.ensure_pdf_ocr() is True
    monkeypatch.setattr(deps, "ensure_poppler", lambda **kw: False)
    assert deps.ensure_pdf_ocr() is False
    monkeypatch.setattr(deps, "ensure_tesseract", lambda **kw: False)
    monkeypatch.setattr(deps, "ensure_poppler", lambda **kw: True)
    assert deps.ensure_pdf_ocr() is False
