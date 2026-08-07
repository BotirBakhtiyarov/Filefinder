"""Shared fixtures for the FileFinder v3 test suite.

All tests run fully offline: no model downloads, no network, no OCR.
Sibling core modules are implemented in parallel, so fixtures are defensive
about resetting their module-level singletons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# src-layout: make the package importable without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Module-level singleton attribute names to clear between tests, per module.
_SINGLETON_ATTRS: dict[str, tuple[str, ...]] = {
    "filefinder.core.config": (
        "_config", "_CONFIG", "_i18n", "_translations", "_language",
        "_current_language",
    ),
    "filefinder.core.db": (
        "_db", "_DB", "_vector_store", "_VECTOR_STORE", "_image_vector_store",
    ),
    "filefinder.core.embeddings": ("_embedder", "_EMBEDDER"),
    "filefinder.core.image_embeddings": ("_image_embedder", "_IMAGE_EMBEDDER"),
    "filefinder.core.search": ("_search_engine", "_engine", "_SEARCH_ENGINE"),
    "filefinder.core.indexer": ("_indexer", "_INDEXER"),
}


def reset_singletons() -> None:
    """Best-effort reset of known module-level singletons (already imported)."""
    for mod_name, attrs in _SINGLETON_ATTRS.items():
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, None)
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass


@pytest.fixture()
def app_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the app data directory for one test.

    Contract note: ``filefinder.core.config.get_app_data_dir()`` MUST honor
    the ``FILEFINDER_DATA_DIR`` environment variable before falling back to
    platform defaults. This fixture relies on that behavior.

    Args:
        tmp_path: pytest per-test tmp dir.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Path to the isolated app data directory.
    """
    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FILEFINDER_DATA_DIR", str(data_dir))
    reset_singletons()
    # Defensive: if config does not (yet) honor FILEFINDER_DATA_DIR, patch
    # get_app_data_dir directly so tests never touch the real app data dir.
    try:
        import filefinder.core.config as config_mod
    except ImportError:
        config_mod = None
    if config_mod is not None:
        try:
            honored = Path(config_mod.get_app_data_dir()) == data_dir
        except Exception:  # noqa: BLE001 - treat probe failure as not honored
            honored = False
        if not honored:
            monkeypatch.setattr(
                config_mod, "get_app_data_dir", lambda: data_dir, raising=False
            )
    yield data_dir
    reset_singletons()
