"""Tests for filefinder.core.search hybrid ranking, filters and snippets.

Uses a fake vector store + fake embedder against a real (tmp) SQLite db, per
the spec contract in sections 4.3 and 4.8.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests import fakes

CONTENT_A = "apple banana smoothie recipe with fresh apple slices"
CONTENT_B = "cherry pie dessert with sweet cherry filling"


def _upsert_file(
    db: Any,
    path: Path,
    content: str,
    size: int,
    mtime: float,
) -> Any:
    """Insert a file row + one chunk into the db, tolerating signature drift.

    Args:
        db: Database facade instance.
        path: Fake file path.
        content: File text content.
        size: Fake file size in bytes.
        mtime: Fake modification timestamp.

    Returns:
        The new file's id.
    """
    kwargs = dict(
        path=str(path),
        filename=path.name,
        extension=path.suffix,
        size=size,
        modified_time=mtime,
        content_hash=hashlib.md5(content.encode("utf-8")).hexdigest(),
        text_preview=content[:200],
    )
    try:
        result = db.upsert_file(**kwargs)
    except TypeError:
        result = db.upsert_file(
            path=str(path),
            filename=path.name,
            extension=path.suffix,
            size=size,
            modified_time=mtime,
        )
    if isinstance(result, int):
        return result
    row = db.get_file_by_path(str(path))
    if row is None:
        pytest.fail("upsert_file did not persist the file row")
    if isinstance(row, dict):
        return row["id"]
    try:
        return row["id"]
    except (TypeError, KeyError):
        return getattr(row, "id")


def _add_chunk(db: Any, embedder: fakes.FakeEmbedder, file_id: Any, content: str) -> None:
    """Add a single chunk with its embedding for ``file_id``."""
    embedding = embedder.embed_text(content)
    try:
        db.add_chunks(file_id, [content], [embedding])
    except TypeError:
        db.add_chunks(file_id=file_id, chunks=[content], embeddings=[embedding])


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path):
    """Set up db + search engine backed by fakes, with two indexed files.

    Returns:
        Tuple of (engine, path_a, path_b). File A matches the query "apple",
        is a .txt of size 100 modified at t=1000; file B is a .docx of size
        5000 modified at t=2000 and does not match "apple".
    """
    store = fakes.FakeVectorStore()
    embedder = fakes.FakeEmbedder()
    db = fakes.get_shared_db(monkeypatch, store)
    engine = fakes.make_search_engine(monkeypatch, db, embedder, store)

    path_a = tmp_path / "alpha.txt"
    path_b = tmp_path / "beta.docx"
    id_a = _upsert_file(db, path_a, CONTENT_A, size=100, mtime=1000.0)
    id_b = _upsert_file(db, path_b, CONTENT_B, size=5000, mtime=2000.0)
    _add_chunk(db, embedder, id_a, CONTENT_A)
    _add_chunk(db, embedder, id_b, CONTENT_B)
    return engine, path_a, path_b


def _paths(results: list[Any]) -> list[str]:
    return [str(getattr(r, "path")) for r in results]


def test_hybrid_scoring_order(wired) -> None:
    """The file matching semantically and by keyword ranks first."""
    engine, path_a, path_b = wired
    results = fakes.run_search(engine, "apple")
    assert results, "expected at least one result for 'apple'"
    assert _paths(results)[0].endswith("alpha.txt")
    scores = {str(getattr(r, "path")): float(getattr(r, "score")) for r in results}
    if str(path_b) in scores:
        assert scores[str(path_a)] > scores[str(path_b)]


def test_snippet_selection(wired) -> None:
    """The snippet of the top result comes from the matching chunk."""
    engine, path_a, _ = wired
    results = fakes.run_search(engine, "apple")
    top = results[0]
    assert str(getattr(top, "path")).endswith("alpha.txt")
    assert "apple" in str(getattr(top, "snippet")).lower()


def test_filter_by_extension(wired) -> None:
    """file_types filter excludes non-matching extensions."""
    engine, path_a, _ = wired
    filters = fakes.make_filters(file_types=[".docx"])
    if filters is None:
        pytest.skip("Filters dataclass not available yet")
    results = fakes.run_search(engine, "apple", filters=filters)
    assert str(path_a) not in _paths(results)


def test_filter_by_date(wired) -> None:
    """date_from filter excludes older files."""
    engine, path_a, _ = wired
    filters = fakes.make_filters(date_from=1500.0)
    if filters is None:
        pytest.skip("Filters dataclass not available yet")
    results = fakes.run_search(engine, "apple", filters=filters)
    assert str(path_a) not in _paths(results)


def test_filter_by_size(wired) -> None:
    """min_size filter excludes small files."""
    engine, path_a, _ = wired
    filters = fakes.make_filters(min_size=1000)
    if filters is None:
        pytest.skip("Filters dataclass not available yet")
    results = fakes.run_search(engine, "apple", filters=filters)
    assert str(path_a) not in _paths(results)
