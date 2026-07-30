"""A tiny, dependency-light vector store (numpy cosine similarity).

Deliberately simple and transparent — it holds the embeddings in a numpy matrix
and ranks by cosine similarity. For a larger corpus this can be swapped for a
dedicated vector database (Chroma, FAISS, pgvector) without touching the rest
of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = None   # (n, d) float32 matrix
        self.docs: list[dict] = []  # metadata aligned row-for-row with `vectors`

    def add(self, vectors, docs: list[dict]) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.docs.extend(docs)

    def search(self, query_vector, k: int = 4) -> list[tuple[dict, float]]:
        """Return the top-k (doc, similarity) pairs by cosine similarity."""
        query = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
        unit = self.vectors / norms
        q_unit = query / (np.linalg.norm(query) + 1e-8)
        sims = unit @ q_unit
        top = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in top]

    def save(self, index_file: str, docs_file: str) -> None:
        Path(index_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_file, vectors=self.vectors)
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(self.docs, f)

    @classmethod
    def load(cls, index_file: str, docs_file: str) -> VectorStore:
        store = cls()
        store.vectors = np.load(index_file)["vectors"]
        with open(docs_file, "r", encoding="utf-8") as f:
            store.docs = json.load(f)
        return store
