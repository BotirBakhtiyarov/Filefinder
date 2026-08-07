"""Text embedding via ONNX runtime (BGE models).

Provides a lazily-initialized :class:`Embedder` singleton that loads a
quantized ONNX embedding model plus its HuggingFace tokenizer and produces
L2-normalized, mean-pooled sentence embeddings.

Heavy dependencies (``onnxruntime``, ``tokenizers``, ``huggingface_hub``)
are imported lazily so this module always imports cleanly.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from filefinder.core.config import get_model_dir, is_android

logger = logging.getLogger(__name__)

# HuggingFace repos per platform.
_HF_REPO_DESKTOP = "BAAI/bge-m3"
_HF_REPO_ANDROID = "BAAI/bge-small-zh-v1.5"

# Best-effort file filter for snapshot_download: tokenizer/config json files,
# vocab/merges text files, and everything under onnx/ (where the quantized
# model.onnx usually lives for bge-m3).
_ALLOW_PATTERNS = ["*.json", "*.txt", "onnx/*", "*.onnx"]

# Tokenizer limits.
_MAX_LENGTH = 512
_BATCH_CHUNK = 16


class ModelNotAvailableError(Exception):
    """Raised when the embedding model is not present and cannot be used."""


class Embedder:
    """Lazy ONNX embedding model wrapper.

    The model directory is obtained from :func:`config.get_model_dir` and is
    expected to contain ``model.onnx`` and ``tokenizer.json`` (possibly under
    an ``onnx/`` subdirectory for ``model.onnx``).
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._load_failed = False

    # ------------------------------------------------------------------
    # Paths / availability
    # ------------------------------------------------------------------

    def _model_dir(self) -> Path:
        return Path(get_model_dir())

    def _find_model_onnx(self) -> Optional[Path]:
        """Locate the ONNX model file in the model directory.

        Accepts any ``*.onnx`` placed in the model dir (or its ``onnx/``
        subdirectory), preferring quantized variants. The original filename
        is preserved by :meth:`download_model` so that external-data
        references (e.g. ``model.onnx_data``) keep resolving.
        """
        model_dir = self._model_dir()
        candidates = list(model_dir.glob("*.onnx"))
        onnx_sub = model_dir / "onnx"
        if onnx_sub.is_dir():
            candidates.extend(onnx_sub.glob("*.onnx"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: ("quantized" not in p.name.lower(), len(p.name)))
        return candidates[0]

    def _find_tokenizer(self) -> Optional[Path]:
        tokenizer_path = self._model_dir() / "tokenizer.json"
        return tokenizer_path if tokenizer_path.is_file() else None

    def is_model_available(self) -> bool:
        """Return True if the ONNX model and tokenizer files exist on disk.

        Models whose weights live in external data files (e.g. bge-m3's
        ``model.onnx`` referencing ``model.onnx_data``) are only considered
        available when at least one ``*.onnx_data`` file sits next to the
        graph file — otherwise onnxruntime fails at load time.
        """
        model_path = self._find_model_onnx()
        if model_path is None or self._find_tokenizer() is None:
            return False
        try:
            size = model_path.stat().st_size
            if size < 100 * 1024 * 1024:  # external-data graphs are small
                blob = model_path.read_bytes()
                if b".onnx_data" in blob and not any(
                    model_path.parent.glob("*.onnx_data")
                ):
                    return False
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the ONNX session and tokenizer (thread-safe, idempotent).

        Raises:
            ModelNotAvailableError: if model files are missing or loading fails.
        """
        if self._session is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return
            if self._load_failed:
                raise ModelNotAvailableError(
                    "Embedding model failed to load previously."
                )
            model_path = self._find_model_onnx()
            tokenizer_path = self._find_tokenizer()
            if model_path is None or tokenizer_path is None:
                raise ModelNotAvailableError(
                    f"Embedding model not found in {self._model_dir()}. "
                    "Call download_model() first."
                )
            try:
                import onnxruntime as ort  # lazy heavy import
                from tokenizers import Tokenizer  # lazy heavy import

                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                tokenizer.enable_truncation(max_length=_MAX_LENGTH)
                tokenizer.enable_padding()
                self._tokenizer = tokenizer
                logger.info("Embedding model loaded from %s", model_path)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._load_failed = True
                logger.error("Failed to load embedding model: %s", exc)
                raise ModelNotAvailableError(
                    f"Failed to load embedding model: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Embedding math
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

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        """Embed a single chunk of texts (internal, assumes model loaded)."""
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = [e.ids for e in encodings]
        attention_mask = [e.attention_mask for e in encodings]

        input_names = {inp.name for inp in self._session.get_inputs()}
        feed: dict[str, Any] = {}

        import numpy as np  # local import; numpy is a hard dep of onnxruntime

        if "input_ids" in input_names:
            feed["input_ids"] = np.asarray(input_ids, dtype=np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.asarray(attention_mask, dtype=np.int64)
        if "token_type_ids" in input_names:
            token_type_ids = [getattr(e, "type_ids", [0] * len(e.ids)) for e in encodings]
            feed["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)

        outputs = self._session.run(None, feed)
        last_hidden_state = outputs[0]
        attention = feed.get("attention_mask")
        if attention is None:
            attention = np.asarray(attention_mask, dtype=np.int64)
        pooled = self._mean_pool(last_hidden_state, attention)
        return [self._normalize(vec) for vec in pooled]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text.

        Raises:
            ModelNotAvailableError: if the model is not available.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in chunks of 16.

        Raises:
            ModelNotAvailableError: if the model is not available.
        """
        if not texts:
            return []
        self._load()
        results: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_CHUNK):
            chunk = texts[start : start + _BATCH_CHUNK]
            chunk = [c if c else " " for c in chunk]
            results.extend(self._embed_chunk(chunk))
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_model(
        self, progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download the embedding model from HuggingFace.

        Uses ``huggingface_hub.snapshot_download`` filtered to tokenizer and
        ONNX files. After download, files are copied into ``get_model_dir()``
        (picking a single ``model.onnx`` if several variants exist).

        Args:
            progress_cb: optional callback ``(downloaded_bytes, total_bytes)``.
                Called with best-effort values; ``total_bytes`` may be 0 when
                unknown.

        Raises:
            ModelNotAvailableError: if the download fails.
        """
        repo = _HF_REPO_ANDROID if is_android() else _HF_REPO_DESKTOP
        target_dir = self._model_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download  # lazy heavy import

            logger.info("Downloading embedding model %s ...", repo)
            if progress_cb is not None:
                progress_cb(0, 0)
            snapshot_path = Path(
                snapshot_download(
                    repo_id=repo,
                    allow_patterns=_ALLOW_PATTERNS,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Model download failed: %s", exc)
            raise ModelNotAvailableError(f"Model download failed: {exc}") from exc

        # Copy the needed files into the model dir, preferring a quantized or
        # plain model.onnx if multiple ONNX variants exist.
        try:
            import shutil

            tokenizer_src = snapshot_path / "tokenizer.json"
            if tokenizer_src.is_file():
                shutil.copy2(tokenizer_src, target_dir / "tokenizer.json")

            onnx_candidates = sorted(
                snapshot_path.rglob("*.onnx"),
                key=lambda p: ("quantized" not in p.name.lower(), len(str(p))),
            )
            if onnx_candidates:
                chosen = onnx_candidates[0]
                # Remove stale ONNX files from previous (possibly partial)
                # downloads so _find_model_onnx never picks a broken one.
                for stale in list(target_dir.glob("*.onnx")) + list(
                    target_dir.glob("*.onnx_data")
                ):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                # Keep the ORIGINAL filename: ONNX external-data references
                # (e.g. "model.onnx_data") are stored relative to it.
                shutil.copy2(chosen, target_dir / chosen.name)
                # Copy external data files belonging to the chosen model
                # (e.g. model.onnx_data), which hold the actual weights.
                for sibling in chosen.parent.iterdir():
                    if (
                        sibling.is_file()
                        and sibling.name.startswith(chosen.name)
                        and sibling.suffix != ".onnx"
                    ):
                        shutil.copy2(sibling, target_dir / sibling.name)
                logger.info(
                    "Staged ONNX model %s (+%d external data file(s))",
                    chosen.name,
                    len(list(target_dir.glob(chosen.name + "*"))) - 1,
                )

            # Best-effort extras (vocab, merges, config) — not required.
            for extra in ("vocab.txt", "merges.txt", "config.json",
                          "special_tokens_map.json", "tokenizer_config.json"):
                src = snapshot_path / extra
                if src.is_file():
                    try:
                        shutil.copy2(src, target_dir / extra)
                    except OSError:
                        pass

            if progress_cb is not None:
                progress_cb(1, 1)
            logger.info("Model files placed in %s", target_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to stage model files: %s", exc)
            raise ModelNotAvailableError(f"Failed to stage model files: {exc}") from exc

        if not self.is_model_available():
            raise ModelNotAvailableError(
                "Download completed but model.onnx/tokenizer.json are missing."
            )


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Return the module-level :class:`Embedder` singleton."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder


def get_embedding_dim() -> int:
    """Return the embedding dimension for the current platform."""
    from filefinder.core.constants import (
        EMBEDDING_DIM_ANDROID,
        EMBEDDING_DIM_DESKTOP,
    )

    return EMBEDDING_DIM_ANDROID if is_android() else EMBEDDING_DIM_DESKTOP
