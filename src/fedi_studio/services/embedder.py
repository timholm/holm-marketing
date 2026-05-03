"""Static embeddings via Model2Vec.

Model: minishlab/potion-base-32M
- 30MB on disk
- 256-dim float32 vectors
- ~100k sentences/sec on CPU
- 93% of MiniLM quality at 70x the speed

This is the foundation of the new scorer. The same model is used to embed
posts on ingest and to project Tim's user_centroid for cosine similarity.

NOTE: First call downloads ~30MB to ~/.cache/huggingface. Cache it in the
container image to avoid runtime download.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "minishlab/potion-base-32M"
EMBEDDING_DIM = 512  # potion-base-32M outputs 512-dim vectors by default


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the model. Cached for the process lifetime."""
    from model2vec import StaticModel

    log.info("Loading Model2Vec %s...", MODEL_NAME)
    model = StaticModel.from_pretrained(MODEL_NAME)
    log.info("Loaded. Embedding dim: %d", model.dim)
    return model


def embed(text: str) -> np.ndarray:
    """Embed a single string. Returns a 256-dim float32 numpy array."""
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    model = _get_model()
    vec = model.encode([text], show_progress_bar=False)
    return vec[0].astype(np.float32)


def embed_batch(texts: Iterable[str]) -> np.ndarray:
    """Embed many strings at once. Returns (N, 256) float32 array."""
    text_list = [t if t and t.strip() else " " for t in texts]
    if not text_list:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    return model.encode(text_list, show_progress_bar=False).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns -1 to 1."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
