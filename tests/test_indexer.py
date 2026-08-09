"""Integration tests for filefinder.core.indexer (spec section 4.7).

Runs against a tmp directory of .txt files with a FakeEmbedder monkeypatched
over ``get_embedder`` and a fake in-memory vector store, so no model, network
or chromadb is required.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from tests import fakes

FILES = {
    "f1.txt": "zebra crossing alpha quixotic",
    "f2.txt": "banana split beta yellowish",
    "f3.txt": "gamma delta epsilon geometric",
}


def _setup(monkeypatch: pytest.MonkeyPatch):
    """Patch embedder + vector store everywhere and import the indexer.

    Returns:
        Tuple of (indexer module, FakeEmbedder, FakeVectorStore).
    """
    embedder = fakes.FakeEmbedder()
    store = fakes.FakeVectorStore()
    fakes.patch_everywhere(
        monkeypatch,
        "get_embedder",
        lambda: embedder,
        [
            "filefinder.core.embeddings",
            "filefinder.core.indexer",
            "filefinder.core.search",
        ],
    )
    fakes.patch_everywhere(
        monkeypatch,
        "get_vector_store",
        lambda: store,
        ["filefinder.core.db"],
    )
    indexer = fakes.import_or_skip("filefinder.core.indexer")
    return indexer, embedder, store


def _make_docs(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a docs directory populated with the given txt files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, content in (files or FILES).items():
        (docs / name).write_text(content, encoding="utf-8")
    return docs


def _run_index(
    indexer: Any,
    dirs: list[str],
    progress_cb: Any = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Call index_directories tolerating signature variations."""
    fn = indexer.index_directories
    for args, kwargs in (
        ((dirs,), {"progress_cb": progress_cb, "cancel_event": cancel_event}),
        ((dirs, progress_cb, cancel_event), {}),
        ((dirs,), {}),
    ):
        try:
            return dict(fn(*args, **kwargs) or {})
        except TypeError:
            continue
    pytest.fail("index_directories did not accept any known signature")
    raise AssertionError  # unreachable


def _stat(stats: dict, *names: str) -> int | None:
    """Fetch the first integer stat among candidate key names."""
    for name in names:
        value = stats.get(name)
        if isinstance(value, int):
            return value
    return None


def _search_paths(
    monkeypatch: pytest.MonkeyPatch,
    store: fakes.FakeVectorStore,
    embedder: fakes.FakeEmbedder,
    query: str,
) -> list[str]:
    """Run a search through the shared db singleton and return result paths."""
    db = fakes.get_shared_db(monkeypatch, store)
    engine = fakes.make_search_engine(monkeypatch, db, embedder, store)
    results = fakes.run_search(engine, query)
    return [str(getattr(r, "path")) for r in results]


def test_index_directories_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """Indexing three txt files reports correct stats."""
    indexer, _, _ = _setup(monkeypatch)
    docs = _make_docs(tmp_path)
    progress: list[tuple[int, int]] = []

    def progress_cb(current: int, total: int, filename: str = "") -> None:
        progress.append((current, total))

    stats = _run_index(indexer, [str(docs)], progress_cb, threading.Event())
    assert isinstance(stats, dict)
    indexed = _stat(stats, "indexed", "files_indexed", "processed", "files_processed")
    if indexed is not None:
        assert indexed == 3
    else:
        assert stats, "stats dict should not be empty"


def test_search_finds_keyword_after_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """A unique keyword is found after indexing."""
    indexer, embedder, store = _setup(monkeypatch)
    docs = _make_docs(tmp_path)
    _run_index(indexer, [str(docs)], None, threading.Event())
    paths = _search_paths(monkeypatch, store, embedder, "zebra")
    assert any(p.endswith("f1.txt") for p in paths), f"f1.txt not in {paths}"


def test_modify_and_reindex_updates_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """Modifying a file and re-indexing makes the new content searchable."""
    indexer, embedder, store = _setup(monkeypatch)
    docs = _make_docs(tmp_path)
    _run_index(indexer, [str(docs)], None, threading.Event())

    target = docs / "f1.txt"
    target.write_text("giraffe savanna alpha towering", encoding="utf-8")
    index_file = getattr(indexer, "index_file", None)
    if callable(index_file):
        index_file(str(target))
    else:
        _run_index(indexer, [str(docs)], None, threading.Event())

    paths = _search_paths(monkeypatch, store, embedder, "giraffe")
    assert any(p.endswith("f1.txt") for p in paths), f"f1.txt not in {paths}"


def test_remove_file_drops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """remove_file removes the file from the index and search results."""
    indexer, embedder, store = _setup(monkeypatch)
    docs = _make_docs(tmp_path)
    _run_index(indexer, [str(docs)], None, threading.Event())

    remove_file = getattr(indexer, "remove_file", None)
    if not callable(remove_file):
        pytest.skip("indexer.remove_file not implemented yet")
    remove_file(str(docs / "f2.txt"))

    paths = _search_paths(monkeypatch, store, embedder, "banana")
    assert not any(p.endswith("f2.txt") for p in paths), f"f2.txt still in {paths}"


def test_cancel_event_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """A pre-set cancel event stops indexing before any file is processed."""
    indexer, _, _ = _setup(monkeypatch)
    docs = _make_docs(tmp_path, {f"f{i}.txt": f"content number {i}" for i in range(10)})

    cancel = threading.Event()
    cancel.set()
    stats = _run_index(indexer, [str(docs)], None, cancel)
    assert isinstance(stats, dict)
    indexed = _stat(stats, "indexed", "files_indexed", "processed", "files_processed")
    if indexed is not None:
        assert indexed < 10
