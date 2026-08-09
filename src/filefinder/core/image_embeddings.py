"""Image-content embeddings via CLIP (ONNX runtime).

Provides a lazily-initialized :class:`ImageEmbedder` singleton that loads a
quantized multilingual CLIP text tower and a quantized CLIP vision tower and
produces L2-normalized embeddings in a shared 512-dim space, so images can be
found by describing their content (e.g. ``"qizil mashina"`` / ``"red car"``).

Heavy dependencies (``onnxruntime``, ``tokenizers``, ``PIL``,
``huggingface_hub``) are imported lazily so this module always imports
cleanly. :class:`ModelNotAvailableError` is reused from
:mod:`filefinder.core.embeddings`.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filefinder.core.config import get_model_dir
from filefinder.core.constants import (
    CLIP_TEXT_REPO,
    CLIP_VISION_REPO,
    IMAGE_EMBEDDING_DIM,
)
from filefinder.core.embeddings import ModelNotAvailableError

logger = logging.getLogger(__name__)

# CLIP tokenizer limit (77 BPE tokens for ViT-B/32).
_MAX_LENGTH = 77

# CLIP ViT-B/32 image preprocessing parameters.
_IMAGE_SIZE = 224
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# File filter for the text-tower snapshot download.
_TEXT_ALLOW_PATTERNS = [
    "onnx/model_quint8_avx2.onnx",
    "onnx/model.onnx",
    "2_Dense/model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece*",
    "special_tokens_map.json",
    "config.json",
    "vocab*",
]

# File filter for the vision-tower snapshot download.
_VISION_ALLOW_PATTERNS = [
    "onnx/vision_model_quantized.onnx",
    "onnx/vision_model.onnx",
]

# Canonical filenames inside the CLIP model directory.
_TEXT_ONNX_NAME = "text_model.onnx"
_VISION_ONNX_NAME = "vision_model.onnx"
_TOKENIZER_NAME = "tokenizer.json"
# Pinned model revisions (supply-chain safety; bandit B615).
_TEXT_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"
_VISION_REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"

# 768→512 linear projection applied after mean-pooling the text tower
# (the sentence-transformers ONNX export only contains the transformer).
_DENSE_NAME = "dense.safetensors"


def get_clip_model_dir() -> Path:
    """Return the directory holding the downloaded CLIP model files."""
    return Path(get_model_dir()).parent / "clip"


class ImageEmbedder:
    """Lazy CLIP ONNX model wrapper (text + vision towers).

    The model directory (:func:`get_clip_model_dir`) is expected to contain
    ``text_model.onnx``, ``vision_model.onnx`` and ``tokenizer.json`` as
    produced by :meth:`download_model`.
    """

    def __init__(self) -> None:
        self._text_session: Any = None
        self._vision_session: Any = None
        self._tokenizer: Any = None
        self._dense_weight: Any = None  # 512x768 projection matrix
        self._lock = threading.Lock()
        self._load_failed = False

    # ------------------------------------------------------------------
    # Paths / availability
    # ------------------------------------------------------------------

    def _model_dir(self) -> Path:
        return get_clip_model_dir()

    def _find_text_model(self) -> Path | None:
        """Locate the CLIP text-tower ONNX file, or None."""
        path = self._model_dir() / _TEXT_ONNX_NAME
        return path if path.is_file() else None

    def _find_vision_model(self) -> Path | None:
        """Locate the CLIP vision-tower ONNX file, or None."""
        path = self._model_dir() / _VISION_ONNX_NAME
        return path if path.is_file() else None

    def _find_tokenizer(self) -> Path | None:
        """Locate the CLIP tokenizer.json, or None."""
        path = self._model_dir() / _TOKENIZER_NAME
        return path if path.is_file() else None

    def _find_dense(self) -> Path | None:
        """Locate the text projection (Dense) weights file, or None."""
        path = self._model_dir() / _DENSE_NAME
        return path if path.is_file() else None

    def is_available(self) -> bool:
        """Return True if all CLIP model files exist on disk."""
        return (
            self._find_text_model() is not None
            and self._find_vision_model() is not None
            and self._find_tokenizer() is not None
            and self._find_dense() is not None
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load both ONNX sessions and the tokenizer (thread-safe).

        Raises:
            ModelNotAvailableError: if model files are missing or loading
                fails.
        """
        if (
            self._text_session is not None
            and self._vision_session is not None
            and self._tokenizer is not None
        ):
            return
        with self._lock:
            if (
                self._text_session is not None
                and self._vision_session is not None
                and self._tokenizer is not None
            ):
                return
            if self._load_failed:
                raise ModelNotAvailableError(
                    "CLIP model failed to load previously."
                )
            text_path = self._find_text_model()
            vision_path = self._find_vision_model()
            tokenizer_path = self._find_tokenizer()
            dense_path = self._find_dense()
            if (
                text_path is None
                or vision_path is None
                or tokenizer_path is None
                or dense_path is None
            ):
                raise ModelNotAvailableError(
                    f"CLIP model not found in {self._model_dir()}. "
                    "Call download_model() first."
                )
            try:
                import onnxruntime as ort  # lazy heavy import
                from safetensors import safe_open  # lazy import
                from tokenizers import Tokenizer  # lazy heavy import

                self._text_session = ort.InferenceSession(
                    str(text_path),
                    providers=["CPUExecutionProvider"],
                )
                self._vision_session = ort.InferenceSession(
                    str(vision_path),
                    providers=["CPUExecutionProvider"],
                )
                with safe_open(str(dense_path), framework="numpy") as f:
                    self._dense_weight = f.get_tensor("linear.weight")
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                tokenizer.enable_truncation(max_length=_MAX_LENGTH)
                tokenizer.enable_padding()
                self._tokenizer = tokenizer
                logger.info("CLIP model loaded from %s", self._model_dir())
            except Exception as exc:
                self._load_failed = True
                logger.error("Failed to load CLIP model: %s", exc)
                raise ModelNotAvailableError(
                    f"Failed to load CLIP model: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Embedding math (pure helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> list[list[float]]:
        """Mean-pool token embeddings over the attention mask.

        Args:
            last_hidden_state: Array-like of shape (batch, seq_len, hidden).
            attention_mask: Array-like of shape (batch, seq_len) with 1/0.

        Returns:
            List of pooled embedding vectors, one per sequence.
        """
        pooled: list[list[float]] = []
        batch = last_hidden_state.tolist()
        masks = attention_mask.tolist()
        for tokens, mask in zip(batch, masks):
            hidden = len(tokens[0]) if tokens else 0
            acc = [0.0] * hidden
            count = 0
            for token_vec, m in zip(tokens, mask):
                if m:
                    for i, v in enumerate(token_vec):
                        acc[i] += float(v)
                    count += 1
            if count == 0:
                count = 1
            pooled.append([v / count for v in acc])
        return pooled

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2-normalize a vector; returns the input unchanged if zero norm."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _preprocess_image(image: Any) -> Any:
        """Resize, center-crop and normalize a PIL image for CLIP.

        Resizes so the shorter side is ``_IMAGE_SIZE`` (BICUBIC), center-crops
        to 224x224, scales to float32/255 and normalizes with the CLIP
        mean/std, returning a CHW array.

        Args:
            image: a PIL image (any mode; converted to RGB).

        Returns:
            ``numpy.float32`` array of shape ``(3, 224, 224)``.
        """
        import numpy as np  # local import; numpy is a hard dep of onnxruntime
        from PIL import Image  # lazy heavy import

        if image.mode == "P" and "transparency" in image.info:
            # Avoids PIL's "Palette images with Transparency" warning.
            image = image.convert("RGBA")
        if image.mode == "RGBA":
            # Flatten onto white so transparent areas don't become black.
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        img = image.convert("RGB")
        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError("image has zero-sized dimension")
        if width < height:
            new_width = _IMAGE_SIZE
            new_height = max(1, round(height * _IMAGE_SIZE / width))
        else:
            new_height = _IMAGE_SIZE
            new_width = max(1, round(width * _IMAGE_SIZE / height))
        img = img.resize((new_width, new_height), getattr(Image, 'Resampling', Image).BICUBIC)

        left = (img.size[0] - _IMAGE_SIZE) // 2
        top = (img.size[1] - _IMAGE_SIZE) // 2
        img = img.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))

        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        mean = np.asarray(_CLIP_MEAN, dtype=np.float32)
        std = np.asarray(_CLIP_STD, dtype=np.float32)
        arr = (arr - mean) / std
        return np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the CLIP text tower.

        Args:
            text: free-text query (multilingual).

        Returns:
            L2-normalized 512-dim embedding.

        Raises:
            ModelNotAvailableError: if the model is not available.
        """
        self._load()
        import numpy as np  # local import; numpy is a hard dep of onnxruntime

        encoding = self._tokenizer.encode(text or " ")
        input_names = {inp.name for inp in self._text_session.get_inputs()}
        feed: dict[str, Any] = {}
        if "input_ids" in input_names:
            feed["input_ids"] = np.asarray([encoding.ids], dtype=np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.asarray(
                [encoding.attention_mask], dtype=np.int64
            )

        output_names = [out.name for out in self._text_session.get_outputs()]
        outputs = self._text_session.run(None, feed)
        named = dict(zip(output_names, outputs))

        if "sentence_embedding" in named:
            vector = named["sentence_embedding"][0].tolist()
        else:
            hidden = named.get("last_hidden_state", named.get("token_embeddings"))
            if hidden is None:
                hidden = outputs[0]
            attention = feed.get(
                "attention_mask",
                np.asarray([encoding.attention_mask], dtype=np.int64),
            )
            vector = self._mean_pool(hidden, attention)[0]
        # Apply the 768→512 sentence projection when the ONNX export only
        # contains the raw transformer (sentence-transformers exports).
        if len(vector) != IMAGE_EMBEDDING_DIM and self._dense_weight is not None:
            vector = (
                np.asarray(vector, dtype=np.float32) @ self._dense_weight.T
            ).tolist()
        return self._normalize([float(v) for v in vector])

    def embed_image(self, path: str) -> list[float] | None:
        """Embed a single image file with the CLIP vision tower.

        Args:
            path: image file path.

        Returns:
            L2-normalized 512-dim embedding, or None when the image cannot
            be read or embedded (the error is logged).
        """
        return self.embed_images([path])[0]

    def embed_images(self, paths: list[str]) -> list[list[float] | None]:
        """Embed a batch of image files with the CLIP vision tower.

        Per-image errors are logged and yield None at that position instead
        of failing the whole batch.

        Args:
            paths: image file paths.

        Returns:
            List of L2-normalized embeddings (or None per failed image),
            aligned with ``paths``.

        Raises:
            ModelNotAvailableError: if the model is not available.
        """
        if not paths:
            return []
        self._load()
        import numpy as np  # local import; numpy is a hard dep of onnxruntime
        from PIL import Image  # lazy heavy import

        results: list[list[float] | None] = [None] * len(paths)
        tensors: list[Any] = []
        positions: list[int] = []
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as img:
                    tensors.append(self._preprocess_image(img))
                positions.append(index)
            except Exception as exc:  # noqa: BLE001 - skip unreadable image
                logger.warning("Failed to preprocess image %s: %s", path, exc)
        if not tensors:
            return results

        batch = np.stack(tensors).astype(np.float32)
        input_name = self._vision_session.get_inputs()[0].name
        output_names = [out.name for out in self._vision_session.get_outputs()]
        outputs = self._vision_session.run(None, {input_name: batch})
        named = dict(zip(output_names, outputs))
        embeds = named.get("image_embeds", outputs[0])

        for row, position in zip(embeds.tolist(), positions):
            results[position] = self._normalize([float(v) for v in row])
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_model(
        self, progress_cb: Callable[[int, int], None] | None = None
    ) -> None:
        """Download the CLIP text and vision towers from HuggingFace.

        Uses ``huggingface_hub.snapshot_download`` filtered to the quantized
        ONNX graphs and tokenizer files, then copies them into
        :func:`get_clip_model_dir` under canonical names
        (``text_model.onnx`` / ``vision_model.onnx`` / ``tokenizer.json``).

        Args:
            progress_cb: optional callback ``(downloaded_bytes, total_bytes)``.
                Called with best-effort values; ``total_bytes`` may be 0 when
                unknown.

        Raises:
            ModelNotAvailableError: if the download fails.
        """
        import shutil

        target_dir = self._model_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download  # lazy heavy import

            logger.info("Downloading CLIP text tower %s ...", CLIP_TEXT_REPO)
            if progress_cb is not None:
                progress_cb(0, 0)
            text_snapshot = Path(
                snapshot_download(
                    repo_id=CLIP_TEXT_REPO,
                    revision=_TEXT_REVISION,
                    allow_patterns=_TEXT_ALLOW_PATTERNS,
                )
            )
            if progress_cb is not None:
                progress_cb(1, 2)
            logger.info("Downloading CLIP vision tower %s ...", CLIP_VISION_REPO)
            vision_snapshot = Path(
                snapshot_download(
                    repo_id=CLIP_VISION_REPO,
                    revision=_VISION_REVISION,
                    allow_patterns=_VISION_ALLOW_PATTERNS,
                )
            )
        except Exception as exc:
            logger.error("CLIP model download failed: %s", exc)
            raise ModelNotAvailableError(
                f"CLIP model download failed: {exc}"
            ) from exc

        try:
            text_onnx = self._pick_onnx(
                text_snapshot,
                ["onnx/model_quint8_avx2.onnx", "onnx/model.onnx"],
            )
            if text_onnx is None:
                raise ModelNotAvailableError(
                    "CLIP text tower ONNX not found in downloaded snapshot."
                )
            self._copy_with_data(text_onnx, target_dir / _TEXT_ONNX_NAME)

            vision_onnx = self._pick_onnx(
                vision_snapshot,
                ["onnx/vision_model_quantized.onnx", "onnx/vision_model.onnx"],
            )
            if vision_onnx is None:
                raise ModelNotAvailableError(
                    "CLIP vision tower ONNX not found in downloaded snapshot."
                )
            self._copy_with_data(vision_onnx, target_dir / _VISION_ONNX_NAME)

            tokenizer_src = text_snapshot / "tokenizer.json"
            if tokenizer_src.is_file():
                shutil.copy2(tokenizer_src, target_dir / _TOKENIZER_NAME)

            # 768→512 sentence projection weights (required — the exported
            # ONNX graph only contains the raw transformer).
            dense_src = text_snapshot / "2_Dense" / "model.safetensors"
            if dense_src.is_file():
                shutil.copy2(dense_src, target_dir / _DENSE_NAME)

            # Best-effort tokenizer extras — not all are required.
            for extra in (
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
                "vocab.txt",
                "vocab.json",
                "sentencepiece.bpe.model",
            ):
                src = text_snapshot / extra
                if src.is_file():
                    try:
                        shutil.copy2(src, target_dir / extra)
                    except OSError:
                        pass

            if progress_cb is not None:
                progress_cb(2, 2)
            logger.info("CLIP model files placed in %s", target_dir)
        except ModelNotAvailableError:
            raise
        except Exception as exc:
            logger.error("Failed to stage CLIP model files: %s", exc)
            raise ModelNotAvailableError(
                f"Failed to stage CLIP model files: {exc}"
            ) from exc

        if not self.is_available():
            raise ModelNotAvailableError(
                "Download completed but CLIP model files are missing."
            )

    @staticmethod
    def _pick_onnx(snapshot_dir: Path, candidates: list[str]) -> Path | None:
        """Return the first existing ONNX path among relative candidates."""
        for relative in candidates:
            path = snapshot_dir / relative
            if path.is_file():
                return path
        return None

    @staticmethod
    def _copy_with_data(src: Path, dest: Path) -> None:
        """Copy an ONNX file to ``dest`` plus any ``*_data`` siblings.

        External-data sidecars keep their original names (ONNX graphs store
        the data path relative to the graph file); the quantized CLIP graphs
        used here are self-contained, so this is best-effort.
        """
        import shutil

        shutil.copy2(src, dest)
        for sibling in src.parent.iterdir():
            if (
                sibling.is_file()
                and sibling.name.startswith(src.name)
                and sibling.suffix != ".onnx"
            ):
                try:
                    shutil.copy2(sibling, dest.parent / sibling.name)
                except OSError:
                    pass


_image_embedder: ImageEmbedder | None = None
_image_embedder_lock = threading.Lock()


def get_image_embedder() -> ImageEmbedder:
    """Return the module-level :class:`ImageEmbedder` singleton."""
    global _image_embedder
    if _image_embedder is None:
        with _image_embedder_lock:
            if _image_embedder is None:
                _image_embedder = ImageEmbedder()
    return _image_embedder
