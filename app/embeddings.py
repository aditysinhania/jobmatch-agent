import threading

import numpy as np

from .config import settings

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                _model = TextEmbedding(model_name=settings.embed_model, cache_dir=settings.embed_cache)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with all-MiniLM-L6-v2 (384-d, ONNX via fastembed), L2-normalised."""
    if not texts:
        return []
    vecs = np.array(list(_get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
