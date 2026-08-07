"""Tests for filefinder.core.embeddings math helpers and model gating.

Written against the spec contract (section 4.4): tokenize -> ONNX session ->
mean-pool -> L2 normalize. Helpers may live at module level or as
``Embedder`` staticmethods, so lookups try both and skip when nothing is
available yet.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")


def _embeddings() -> ModuleType:
    """Import embeddings, skipping if not yet available."""
    try:
        return importlib.import_module("filefinder.core.embeddings")
    except ImportError as exc:
        pytest.skip(f"filefinder.core.embeddings not importable: {exc}")
        raise  # unreachable


def _helper(mod: ModuleType, *names: str) -> Callable[..., Any]:
    """Return the first callable among candidate helper names, else skip.

    Looks at module level first, then on the ``Embedder`` class (where the
    reference implementation keeps ``_mean_pool``/``_normalize`` as
    staticmethods).
    """
    containers: list[Any] = [mod]
    embedder_cls = getattr(mod, "Embedder", None)
    if embedder_cls is not None:
        containers.append(embedder_cls)
    for name in names:
        for container in containers:
            fn = getattr(container, name, None)
            if callable(fn):
                return fn
    pytest.skip(f"no pooling/normalize helper found among {names}")
    raise AssertionError  # unreachable


def _model_not_available_error(mod: ModuleType) -> type[Exception]:
    """Locate ModelNotAvailableError wherever the core defines it."""
    for mod_name, attr in (
        ("filefinder.core.embeddings", "ModelNotAvailableError"),
        ("filefinder.core.errors", "ModelNotAvailableError"),
        ("filefinder.core.exceptions", "ModelNotAvailableError"),
    ):
        try:
            candidate_mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        exc = getattr(candidate_mod, attr, None)
        if isinstance(exc, type):
            return exc
    pytest.skip("ModelNotAvailableError not defined yet")
    raise AssertionError  # unreachable


def test_mean_pool() -> None:
    """Mean-pooling averages token embeddings weighted by the attention mask."""
    mod = _embeddings()
    mean_pool = _helper(mod, "_mean_pool", "mean_pool", "_mean_pooling")
    token_embeddings = np.array(
        [
            [[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]],  # last token masked out
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
        ],
        dtype=np.float32,
    )
    attention_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    pooled = np.asarray(mean_pool(token_embeddings, attention_mask), dtype=np.float64)
    expected = np.array(
        [
            [(1.0 + 3.0) / 2.0, (0.0 + 2.0) / 2.0],
            [(2.0 + 4.0 + 6.0) / 3.0, (2.0 + 4.0 + 6.0) / 3.0],
        ]
    )
    assert pooled.shape == (2, 2)
    assert np.allclose(pooled, expected, atol=1e-6)


def test_l2_normalize() -> None:
    """L2 normalization yields unit norm and survives zero vectors."""
    mod = _embeddings()
    normalize = _helper(mod, "_l2_normalize", "l2_normalize", "_normalize", "normalize")
    try:
        # Per-vector form (spec: normalize each pooled embedding).
        out = np.asarray(normalize([3.0, 4.0]), dtype=np.float64)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)
        zero = np.asarray(normalize([0.0, 0.0]), dtype=np.float64)
        assert np.all(np.isfinite(zero)), "zero vector must not produce NaN"
    except TypeError:
        # Batch (2D) form.
        arr = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        out = np.asarray(normalize(arr), dtype=np.float64)
        assert out.shape == arr.shape
        norms = np.linalg.norm(out, axis=-1)
        assert np.isclose(norms[0], 1.0, atol=1e-6)
        assert np.isclose(norms[2], 1.0, atol=1e-6)
        assert np.all(np.isfinite(out)), "zero vector must not produce NaN"


def test_embed_text_without_model_raises(
    monkeypatch: pytest.MonkeyPatch, app_data_dir: Path
) -> None:
    """Embedding without a local model raises ModelNotAvailableError."""
    mod = _embeddings()
    exc = _model_not_available_error(mod)

    embedder_cls = getattr(mod, "Embedder", None)
    if embedder_cls is not None:
        embedder = embedder_cls()
        # Force the "model missing" path regardless of what exists on disk.
        monkeypatch.setattr(embedder, "_find_model_onnx", lambda: None, raising=False)
        monkeypatch.setattr(embedder, "_find_tokenizer", lambda: None, raising=False)
        monkeypatch.setattr(
            embedder, "is_model_available", lambda: False, raising=False
        )
        with pytest.raises(exc):
            embedder.embed_text("hello world")
        return

    embed_text = getattr(mod, "embed_text", None)
    if not callable(embed_text):
        pytest.skip("neither Embedder.embed_text nor module embed_text available")
    monkeypatch.setattr(mod, "is_model_available", lambda: False, raising=False)
    with pytest.raises(exc):
        embed_text("hello world")
