"""FAISS retrieval only.

This module never calls an embedding provider - it searches a FAISS index
that is already built (by build_index.py) against a query vector that the
caller already has. Loading and searching are separate, controlled
operations: a missing index raises RAGIndexNotFoundError instead of an
opaque traceback from faiss/numpy.
"""
import os
from typing import Any, Dict, List, Tuple


class RAGIndexNotFoundError(Exception):
    """Raised when the FAISS index or its metadata file is missing on disk."""


def index_exists(index_path: str, metadata_path: str) -> bool:
    return os.path.exists(index_path) and os.path.exists(metadata_path)


def load_index(index_path: str, metadata_path: str) -> Tuple[Any, Any]:
    if not index_exists(index_path, metadata_path):
        raise RAGIndexNotFoundError(
            f"FAISS index not found at {index_path!r} / {metadata_path!r}. "
            "Run backend/build_index.py first to build it."
        )

    import faiss
    import numpy as np

    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    return index, metadata


def search(index: Any, metadata: Any, query_vector: List[float], k: int = 4) -> List[Dict[str, Any]]:
    """Search an already-loaded FAISS index with an already-computed query vector."""
    import numpy as np

    query_arr = np.array([query_vector], dtype="float32")
    distances, indices = index.search(query_arr, k)

    results: List[Dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["score"] = float(distance)
        results.append(item)
    return results
