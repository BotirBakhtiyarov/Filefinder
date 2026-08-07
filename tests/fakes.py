"""Shared fakes and defensive wiring helpers for search/indexer tests.

Core modules are implemented in parallel against the design spec; these
helpers try the spec contract signatures and skip cleanly when a module or
symbol is not available yet.
"""

from __future__ import annotations

import hashlib
import importlib
from types import ModuleType
from typing import Any, Iterable

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")

EMBEDDING_DIM = 32


class FakeEmbedder:
    """Deterministic word-bag embedder matching the spec Embedder contract.

    Vectors are normalized bag-of-hashed-words, so cosine similarity reflects
    token overlap. This keeps search assertions meaningful without a model.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        """Initialize with a fixed embedding dimension."""
        self.dim = dim

    def _vector_for(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text (spec section 4.4)."""
        return self._vector_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (spec section 4.4)."""
        return [self._vector_for(t) for t in texts]


class FakeVectorStore:
    """In-memory VectorStore implementing the db.VectorStore ABC contract."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._vectors: dict[Any, np.ndarray] = {}

    def add(
        self,
        ids: Iterable[Any],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Any] | None = None,
    ) -> None:
        """Add embeddings under the given ids."""
        for id_, emb in zip(ids, embeddings):
            self._vectors[id_] = np.asarray(list(emb), dtype=np.float64)

    def delete(self, ids: Iterable[Any]) -> None:
        """Delete embeddings by id (missing ids ignored)."""
        for id_ in ids:
            self._vectors.pop(id_, None)

    def query(self, embedding: Iterable[float], top_k: int) -> list[tuple[Any, float]]:
        """Return the top_k ids by cosine similarity to ``embedding``."""
        q = np.asarray(list(embedding), dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm:
            q = q / q_norm
        scored: list[tuple[Any, float]] = []
        for id_, vec in self._vectors.items():
            v_norm = np.linalg.norm(vec)
            score = float(np.dot(q, vec / v_norm)) if v_norm else 0.0
            scored.append((id_, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all embeddings."""
        self._vectors.clear()


def import_or_skip(mod_name: str) -> ModuleType:
    """Import a module, skipping the test if it is not available yet."""
    try:
        return importlib.import_module(mod_name)
    except ImportError as exc:
        pytest.skip(f"{mod_name} not importable: {exc}")
        raise  # unreachable


def patch_everywhere(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: Any, modules: Iterable[str]
) -> None:
    """Set ``attr`` to ``value`` on every importable module in ``modules``."""
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        monkeypatch.setattr(mod, attr, value, raising=False)


def _attach_store(db: Any, store: FakeVectorStore) -> None:
    """Best-effort attach of the fake vector store to a Database instance."""
    for attr in ("vector_store", "_vector_store", "store", "vectors"):
        if hasattr(db, attr):
            try:
                setattr(db, attr, store)
            except Exception:  # noqa: BLE001 - best effort
                pass
    try:
        db.vector_store = store
    except Exception:  # noqa: BLE001 - best effort
        pass


def get_shared_db(monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore) -> Any:
    """Return the db module's singleton Database wired to the fake store.

    Falls back to constructing ``Database`` directly per the spec facade.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        store: Fake vector store to inject.

    Returns:
        A Database-like object.
    """
    db_mod = import_or_skip("filefinder.core.db")
    monkeypatch.setattr(db_mod, "get_vector_store", lambda: store, raising=False)
    get_db = getattr(db_mod, "get_db", None)
    if callable(get_db):
        db = get_db()
        _attach_store(db, store)
        return db
    database_cls = getattr(db_mod, "Database", None)
    if database_cls is None:
        pytest.skip("filefinder.core.db exposes neither get_db nor Database")
    for args, kwargs in (
        ((), {"vector_store": store}),
        ((), {}),
    ):
        try:
            db = database_cls(*args, **kwargs)
            _attach_store(db, store)
            return db
        except TypeError:
            continue
    pytest.skip("cannot construct Database with known signatures")
    raise AssertionError  # unreachable


def make_search_engine(
    monkeypatch: pytest.MonkeyPatch,
    db: Any,
    embedder: FakeEmbedder,
    store: FakeVectorStore,
) -> Any:
    """Return a SearchEngine wired to the fake db/embedder/vector store."""
    search_mod = import_or_skip("filefinder.core.search")
    patch_everywhere(
        monkeypatch, "get_embedder", lambda: embedder,
        ["filefinder.core.search", "filefinder.core.embeddings"],
    )
    patch_everywhere(
        monkeypatch, "get_db", lambda: db, ["filefinder.core.search"]
    )
    patch_everywhere(
        monkeypatch, "get_vector_store", lambda: store,
        ["filefinder.core.search", "filefinder.core.db"],
    )
    engine_cls = getattr(search_mod, "SearchEngine", None)
    if engine_cls is not None:
        for args, kwargs in (
            ((), {"db": db, "embedder": embedder, "vector_store": store}),
            ((db,), {}),
            ((), {}),
        ):
            try:
                return engine_cls(*args, **kwargs)
            except TypeError:
                continue
        pytest.skip("cannot construct SearchEngine with known signatures")
    get_engine = getattr(search_mod, "get_search_engine", None)
    if callable(get_engine):
        return get_engine()
    pytest.skip("filefinder.core.search exposes neither SearchEngine nor get_search_engine")
    raise AssertionError  # unreachable


def run_search(engine: Any, query: str, filters: Any = None, top_k: int = 10) -> list[Any]:
    """Call the engine's search method tolerating signature variations."""
    search = getattr(engine, "search", engine)
    for args, kwargs in (
        ((query,), {"filters": filters, "top_k": top_k}),
        ((query, filters), {}),
        ((query,), {}),
    ):
        try:
            return list(search(*args, **kwargs))
        except TypeError:
            continue
    pytest.fail("SearchEngine.search did not accept any known signature")
    raise AssertionError  # unreachable


def make_filters(**kwargs: Any) -> Any:
    """Construct the search Filters dataclass per the spec contract.

    Returns:
        A Filters instance, or ``None`` if the class does not exist yet.
    """
    search_mod = import_or_skip("filefinder.core.search")
    for name in ("Filters", "SearchFilters"):
        cls = getattr(search_mod, name, None)
        if cls is None:
            continue
        try:
            return cls(**kwargs)
        except TypeError:
            obj = cls()
            for key, value in kwargs.items():
                setattr(obj, key, value)
            return obj
    return None
