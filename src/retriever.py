"""
Retriever: loads the pickle index and finds the most relevant chunks
for a given query using cosine similarity.
"""

import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import INDEX_PATH, EMBEDDING_MODEL, TOP_K, OPENAI_API_KEY
from ingest import Chunk

client = OpenAI(api_key=OPENAI_API_KEY)


class SearchResult(NamedTuple):
    chunk: Chunk
    score: float    # cosine similarity [0, 1]


def _load_index() -> tuple[list[Chunk], np.ndarray]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run `python src/ingest.py` first."
        )
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    chunks: list[Chunk] = data["chunks"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    # нормируем один раз при загрузке, чтобы косинусное сходство = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return chunks, embeddings


# Загружаем индекс один раз при импорте модуля (кешируется в памяти)
_chunks, _embeddings = _load_index()


def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec /= max(np.linalg.norm(vec), 1e-10)
    return vec


def search(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Returns top_k most relevant chunks for the query, sorted by score desc."""
    query_vec = embed_query(query)
    scores = _embeddings @ query_vec   # dot product = cosine sim (after normalization)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(chunk=_chunks[i], score=float(scores[i]))
        for i in top_indices
    ]
