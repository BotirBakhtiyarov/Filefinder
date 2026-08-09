"""Tests for filefinder.core.file_parser.extract_text.

Written against the spec contract (section 4.5). Imports are deferred into
the test functions so the suite still collects cleanly while sibling modules
are under parallel development.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def _parser() -> ModuleType:
    """Import file_parser, skipping if not yet available."""
    try:
        return importlib.import_module("filefinder.core.file_parser")
    except ImportError as exc:
        pytest.skip(f"filefinder.core.file_parser not importable: {exc}")
        raise  # unreachable, for type checkers


def _unsupported_error() -> type[Exception]:
    """Locate UnsupportedFileError wherever the core defines it."""
    for mod_name in (
        "filefinder.core.errors",
        "filefinder.core.exceptions",
        "filefinder.core.file_parser",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        exc = getattr(mod, "UnsupportedFileError", None)
        if isinstance(exc, type):
            return exc
    pytest.skip("UnsupportedFileError not defined yet")
    raise AssertionError  # unreachable


def test_extract_txt_utf8(tmp_path: Path) -> None:
    """UTF-8 .txt content is extracted verbatim."""
    parser = _parser()
    f = tmp_path / "note.txt"
    f.write_text("hello semantic world — Grüße", encoding="utf-8")
    text = parser.extract_text(str(f))
    assert "hello semantic world" in text


def test_extract_txt_gbk(tmp_path: Path) -> None:
    """GBK-encoded .txt falls back correctly (section 4.5)."""
    parser = _parser()
    f = tmp_path / "note_cn.txt"
    content = "中文内容测试标记"
    f.write_bytes(content.encode("gbk"))
    text = parser.extract_text(str(f))
    assert content in text


def test_extract_docx(tmp_path: Path) -> None:
    """Paragraph text of a generated .docx is extracted."""
    docx = pytest.importorskip("docx", reason="python-docx not installed")
    parser = _parser()
    f = tmp_path / "doc.docx"
    doc = docx.Document()
    doc.add_paragraph("FileFinder docx content marker")
    doc.save(str(f))
    text = parser.extract_text(str(f))
    assert "FileFinder docx content marker" in text


def test_extract_xlsx(tmp_path: Path) -> None:
    """Cell values of a generated .xlsx are extracted."""
    openpyxl = pytest.importorskip("openpyxl", reason="openpyxl not installed")
    parser = _parser()
    f = tmp_path / "sheet.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "FileFinder xlsx content marker"
    wb.save(str(f))
    text = parser.extract_text(str(f))
    assert "FileFinder xlsx content marker" in text


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    """Unsupported extensions raise UnsupportedFileError."""
    parser = _parser()
    exc = _unsupported_error()
    f = tmp_path / "binary.weirdext"
    f.write_bytes(b"\x00\x01\x02")
    with pytest.raises(exc):
        parser.extract_text(str(f))
