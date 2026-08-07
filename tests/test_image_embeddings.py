"""Tests for filefinder.core.image_embeddings pure helpers and model gating.

These tests run fully offline: no CLIP model download is required. Helpers
(``_mean_pool`` / ``_normalize`` / ``_preprocess_image``) are exercised
directly on the :class:`ImageEmbedder` class, and model-dependent paths are
only checked for their "model missing" behavior unless the CLIP model files
happen to be present.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")


def _module() -> ModuleType:
    """Import image_embeddings (must always import cleanly, no heavy deps)."""
    return importlib.import_module("filefinder.core.image_embeddings")


def test_mean_pool() -> None:
    """Mean-pooling averages token embeddings weighted by the attention mask."""
    mod = _module()
    token_embeddings = np.array(
        [
            [[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]],  # last token masked out
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
        ],
        dtype=np.float32,
    )
    attention_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    pooled = np.asarray(
        mod.ImageEmbedder._mean_pool(token_embeddings, attention_mask),
        dtype=np.float64,
    )
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
    mod = _module()
    normalize = mod.ImageEmbedder._normalize
    out = np.asarray(normalize([3.0, 4.0]), dtype=np.float64)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)
    zero = np.asarray(normalize([0.0, 0.0]), dtype=np.float64)
    assert np.all(np.isfinite(zero)), "zero vector must not produce NaN"


def test_preprocess_image_chw_shape() -> None:
    """Preprocessing yields a normalized CHW float32 tensor of 3x224x224."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    mod = _module()

    # Landscape and portrait images must both end up 3x224x224.
    for size in ((400, 200), (200, 400), (224, 224)):
        img = Image.new("RGB", size, color=(128, 64, 32))
        arr = np.asarray(mod.ImageEmbedder._preprocess_image(img))
        assert arr.shape == (3, 224, 224)
        assert arr.dtype == np.float32
        # CLIP normalization keeps values within a few standard deviations.
        assert np.all(np.isfinite(arr))
        assert float(np.abs(arr).max()) < 10.0


def test_preprocess_image_normalization_values() -> None:
    """A mid-gray image maps to roughly -mean/std per channel."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    mod = _module()

    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    arr = np.asarray(mod.ImageEmbedder._preprocess_image(img), dtype=np.float64)
    means = arr.mean(axis=(1, 2))
    expected = [(128 / 255.0 - m) / s for m, s in zip(mod._CLIP_MEAN, mod._CLIP_STD)]
    assert np.allclose(means, expected, atol=1e-2)


def test_is_available_false_without_model(app_data_dir: Path) -> None:
    """Without downloaded files the embedder reports not available."""
    mod = _module()
    embedder = mod.ImageEmbedder()
    assert embedder.is_available() is False
    assert embedder._find_text_model() is None
    assert embedder._find_vision_model() is None
    assert embedder._find_tokenizer() is None


def test_embed_query_without_model_raises(app_data_dir: Path) -> None:
    """Embedding a query without the CLIP model raises ModelNotAvailableError."""
    mod = _module()
    from filefinder.core.embeddings import ModelNotAvailableError

    embedder = mod.ImageEmbedder()
    with pytest.raises(ModelNotAvailableError):
        embedder.embed_query("qizil mashina")


def test_embed_images_without_model_raises(app_data_dir: Path) -> None:
    """Embedding images without the CLIP model raises ModelNotAvailableError."""
    mod = _module()
    from filefinder.core.embeddings import ModelNotAvailableError

    embedder = mod.ImageEmbedder()
    with pytest.raises(ModelNotAvailableError):
        embedder.embed_images(["some.png"])


def test_embed_images_empty_batch(app_data_dir: Path) -> None:
    """An empty batch needs no model and returns an empty list."""
    mod = _module()
    embedder = mod.ImageEmbedder()
    assert embedder.embed_images([]) == []


def test_model_dir_is_clip_sibling(app_data_dir: Path) -> None:
    """The CLIP model dir is a 'clip' sibling of the BGE model dir."""
    mod = _module()
    from filefinder.core.config import get_model_dir

    clip_dir = mod.get_clip_model_dir()
    assert clip_dir.name == "clip"
    assert clip_dir.parent == Path(get_model_dir()).parent


def test_model_dependent_embedding(app_data_dir: Path) -> None:
    """Smoke: embed a real image when the CLIP model is present (else skip)."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    mod = _module()
    embedder = mod.ImageEmbedder()
    if not embedder.is_available():
        pytest.skip("CLIP model files not present")

    img_path = Path(app_data_dir) / "sample.png"
    Image.new("RGB", (300, 200), color=(200, 30, 30)).save(img_path)
    vector = embedder.embed_image(str(img_path))
    assert vector is not None
    assert len(vector) == 512
    assert np.isclose(np.linalg.norm(np.asarray(vector)), 1.0, atol=1e-4)

    query = embedder.embed_query("red car")
    assert len(query) == 512
